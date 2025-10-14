#!/usr/bin/env python3
"""
MongoDB to Kafka Producer (Docker-Ready)
Fetches OHLCV data from local MongoDB and streams it to Kafka in Docker
"""

import json
import logging
import time
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from confluent_kafka import Producer, KafkaError
from confluent_kafka.admin import AdminClient, NewTopic
from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration from environment variables with sensible defaults"""
    
    # MongoDB Configuration (on host machine)
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://host.docker.internal:27017/")
    MONGO_DATABASE = os.getenv("MONGO_DATABASE", "ohlcv_data")
    MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "combined_rows")
    MONGO_CONNECTION_TIMEOUT_MS = int(os.getenv("MONGO_TIMEOUT_MS", "10000"))
    MONGO_MAX_RETRIES = int(os.getenv("MONGO_MAX_RETRIES", "5"))
    MONGO_RETRY_DELAY_SEC = int(os.getenv("MONGO_RETRY_DELAY_SEC", "5"))
    
    # Kafka Configuration (in Docker)
    KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "ohlcv-stream")
    KAFKA_NUM_PARTITIONS = int(os.getenv("KAFKA_NUM_PARTITIONS", "3"))
    KAFKA_REPLICATION_FACTOR = int(os.getenv("KAFKA_REPLICATION_FACTOR", "1"))
    KAFKA_MAX_RETRIES = int(os.getenv("KAFKA_MAX_RETRIES", "5"))
    KAFKA_RETRY_DELAY_SEC = int(os.getenv("KAFKA_RETRY_DELAY_SEC", "5"))
    
    # Kafka Producer Settings (optimized for reliability)
    KAFKA_CONFIG = {
        'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
        'acks': 'all',  # Wait for all replicas
        'retries': 10,  # Retry failed sends
        'retry.backoff.ms': 500,  # Wait between retries
        'max.in.flight.requests.per.connection': 1,  # Ensure ordering
        'compression.type': 'snappy',
        'linger.ms': 10,
        'request.timeout.ms': 30000,  # 30 second timeout
        'delivery.timeout.ms': 120000,  # 2 minute total timeout
    }
    
    # Processing Configuration
    DELAY_BETWEEN_RECORDS = float(os.getenv("DELAY_BETWEEN_RECORDS", "1.0"))
    PREFETCH_ALL_DATA = os.getenv("PREFETCH_ALL_DATA", "true").lower() == "true"
    
    # Logging Configuration
    LOG_DIR = os.getenv("LOG_DIR", "/app/logs/kafka")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(config: Config) -> logging.Logger:
    """
    Set up comprehensive logging to file and console
    
    Args:
        config: Configuration object
        
    Returns:
        Configured logger instance
    """
    # Create logs directory
    log_dir = Path(config.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"kafka_producer_{timestamp}.log"
    
    # Get log level
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger("KafkaProducer")
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


# ============================================================================
# Kafka Helper with Retry Logic
# ============================================================================

class KafkaProducerHelper:
    """Helper class for Kafka producer operations with retry logic"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        """Initialize Kafka producer"""
        self.config = config
        self.logger = logger
        self.producer: Optional[Producer] = None
        self.delivered_count = 0
        self.failed_count = 0
    
    def wait_for_kafka(self) -> bool:
        """
        Wait for Kafka to be ready with retry logic
        
        Returns:
            True if Kafka is ready, False otherwise
        """
        self.logger.info("Waiting for Kafka to be ready...")
        
        for attempt in range(1, self.config.KAFKA_MAX_RETRIES + 1):
            try:
                self.logger.info(f"Attempt {attempt}/{self.config.KAFKA_MAX_RETRIES}: Connecting to Kafka...")
                
                # Try to create admin client
                admin_client = AdminClient({
                    'bootstrap.servers': self.config.KAFKA_BOOTSTRAP_SERVERS,
                    'socket.timeout.ms': 10000
                })
                
                # List topics to verify connection
                metadata = admin_client.list_topics(timeout=10)
                
                self.logger.info(f"✓ Kafka is ready! Found {len(metadata.topics)} topics")
                return True
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt} failed: {e}")
                
                if attempt < self.config.KAFKA_MAX_RETRIES:
                    self.logger.info(f"Retrying in {self.config.KAFKA_RETRY_DELAY_SEC} seconds...")
                    time.sleep(self.config.KAFKA_RETRY_DELAY_SEC)
                else:
                    self.logger.error("✗ Max retries reached. Kafka is not available.")
                    return False
        
        return False
    
    def create_topic_if_not_exists(self) -> bool:
        """
        Create Kafka topic if it doesn't exist
        
        Returns:
            True if topic exists or created successfully
        """
        try:
            admin_client = AdminClient({
                'bootstrap.servers': self.config.KAFKA_BOOTSTRAP_SERVERS
            })
            
            # Check if topic exists
            metadata = admin_client.list_topics(timeout=10)
            
            if self.config.KAFKA_TOPIC in metadata.topics:
                self.logger.info(f"✓ Topic '{self.config.KAFKA_TOPIC}' already exists")
                return True
            
            # Create topic
            self.logger.info(f"Creating topic '{self.config.KAFKA_TOPIC}'...")
            
            new_topic = NewTopic(
                self.config.KAFKA_TOPIC,
                num_partitions=self.config.KAFKA_NUM_PARTITIONS,
                replication_factor=self.config.KAFKA_REPLICATION_FACTOR
            )
            
            fs = admin_client.create_topics([new_topic])
            
            # Wait for operation to complete
            for topic, f in fs.items():
                try:
                    f.result()  # The result itself is None
                    self.logger.info(f"✓ Topic '{topic}' created successfully")
                except Exception as e:
                    self.logger.error(f"✗ Failed to create topic '{topic}': {e}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating topic: {e}")
            return False
    
    def connect(self) -> bool:
        """
        Connect to Kafka cluster with retry logic
        
        Returns:
            True if connection successful
        """
        try:
            # Wait for Kafka to be ready
            if not self.wait_for_kafka():
                return False
            
            # Create topic if needed
            if not self.create_topic_if_not_exists():
                return False
            
            self.logger.info("Initializing Kafka producer...")
            self.logger.info(f"Bootstrap servers: {self.config.KAFKA_BOOTSTRAP_SERVERS}")
            self.logger.info(f"Target topic: {self.config.KAFKA_TOPIC}")
            
            self.producer = Producer(self.config.KAFKA_CONFIG)
            
            self.logger.info("✓ Kafka producer initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Failed to initialize Kafka producer: {e}")
            return False
    
    def delivery_callback(self, err: Optional[KafkaError], msg):
        """Kafka delivery callback"""
        if err:
            self.failed_count += 1
            self.logger.error(
                f"✗ Message delivery FAILED: {err} | "
                f"Topic: {msg.topic()} | Partition: {msg.partition()}"
            )
        else:
            self.delivered_count += 1
            self.logger.debug(
                f"✓ Message delivered | "
                f"Topic: {msg.topic()} | Partition: {msg.partition()} | "
                f"Offset: {msg.offset()}"
            )
    
    def produce_message(self, key: str, value: Dict[str, Any]) -> bool:
        """
        Produce message to Kafka with retry logic
        
        Args:
            key: Message key
            value: Message value
            
        Returns:
            True if successful
        """
        max_retries = 3
        
        for attempt in range(1, max_retries + 1):
            try:
                # Serialize value
                value_json = json.dumps(value, default=str)
                
                # Produce message
                self.producer.produce(
                    topic=self.config.KAFKA_TOPIC,
                    key=key.encode('utf-8'),
                    value=value_json.encode('utf-8'),
                    callback=self.delivery_callback
                )
                
                # Poll for callbacks
                self.producer.poll(0)
                
                return True
                
            except BufferError:
                self.logger.warning("Buffer full, flushing...")
                self.producer.flush()
                
                if attempt < max_retries:
                    time.sleep(0.1)
                    continue
                else:
                    return False
                    
            except Exception as e:
                self.logger.error(f"Error producing message (attempt {attempt}): {e}")
                
                if attempt < max_retries:
                    time.sleep(0.5)
                    continue
                else:
                    return False
        
        return False
    
    def flush(self):
        """Flush all pending messages"""
        if self.producer:
            self.logger.info("Flushing remaining messages...")
            remaining = self.producer.flush(timeout=30)
            
            if remaining > 0:
                self.logger.warning(f"⚠ {remaining} messages were not delivered")
            else:
                self.logger.info("✓ All messages flushed successfully")
    
    def close(self):
        """Close producer"""
        if self.producer:
            self.flush()
            self.logger.info("Kafka producer closed")


# ============================================================================
# MongoDB Helper with Retry Logic
# ============================================================================

class MongoDBHelper:
    """Helper class for MongoDB operations with retry logic"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        """Initialize MongoDB helper"""
        self.config = config
        self.logger = logger
        self.client: Optional[MongoClient] = None
        self.collection = None
    
    def connect(self) -> bool:
        """
        Connect to MongoDB with retry logic
        
        Returns:
            True if connection successful
        """
        self.logger.info("Connecting to MongoDB on host machine...")
        self.logger.info(f"URI: {self.config.MONGO_URI}")
        self.logger.info(f"Database: {self.config.MONGO_DATABASE}")
        self.logger.info(f"Collection: {self.config.MONGO_COLLECTION}")
        
        for attempt in range(1, self.config.MONGO_MAX_RETRIES + 1):
            try:
                self.logger.info(f"Attempt {attempt}/{self.config.MONGO_MAX_RETRIES}...")
                
                # Create client
                self.client = MongoClient(
                    self.config.MONGO_URI,
                    serverSelectionTimeoutMS=self.config.MONGO_CONNECTION_TIMEOUT_MS,
                    connectTimeoutMS=self.config.MONGO_CONNECTION_TIMEOUT_MS
                )
                
                # Test connection
                self.client.server_info()
                
                # Get collection
                db = self.client[self.config.MONGO_DATABASE]
                self.collection = db[self.config.MONGO_COLLECTION]
                
                # Get document count
                doc_count = self.collection.count_documents({})
                
                self.logger.info(f"✓ Connected to MongoDB successfully")
                self.logger.info(f"Total documents in collection: {doc_count:,}")
                
                return True
                
            except PyMongoError as e:
                self.logger.warning(f"Attempt {attempt} failed: {e}")
                
                if attempt < self.config.MONGO_MAX_RETRIES:
                    self.logger.info(f"Retrying in {self.config.MONGO_RETRY_DELAY_SEC} seconds...")
                    time.sleep(self.config.MONGO_RETRY_DELAY_SEC)
                else:
                    self.logger.error("✗ Max retries reached. Cannot connect to MongoDB.")
                    return False
        
        return False
    
    def fetch_all_documents(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch all documents sorted by timestamp into memory
        
        Returns:
            List of documents or None on error
        """
        try:
            self.logger.info("Fetching all documents from MongoDB...")
            
            cursor = self.collection.find().sort("timestamp", ASCENDING)
            documents = list(cursor)
            
            self.logger.info(f"✓ Loaded {len(documents):,} documents into memory")
            return documents
            
        except PyMongoError as e:
            self.logger.error(f"Error fetching documents: {e}")
            return None
    
    def fetch_documents_cursor(self):
        """
        Get cursor for streaming documents
        
        Returns:
            Cursor or None on error
        """
        try:
            cursor = self.collection.find().sort("timestamp", ASCENDING)
            return cursor
            
        except PyMongoError as e:
            self.logger.error(f"Error creating cursor: {e}")
            return None
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self.logger.info("MongoDB connection closed")


# ============================================================================
# Main Producer Logic
# ============================================================================

class OHLCVStreamProducer:
    """Main class for streaming OHLCV data"""
    
    def __init__(self, config: Config):
        """Initialize producer"""
        self.config = config
        self.logger = setup_logging(config)
        self.mongo_helper = MongoDBHelper(config, self.logger)
        self.kafka_helper = KafkaProducerHelper(config, self.logger)
        self.processed_count = 0
        self.start_time = None
    
    def run(self):
        """Main execution"""
        self.logger.info("=" * 80)
        self.logger.info("OHLCV Data Streaming: MongoDB (Host) → Kafka (Docker)")
        self.logger.info("=" * 80)
        self.logger.info(f"Prefetch all data: {self.config.PREFETCH_ALL_DATA}")
        self.logger.info(f"Delay between records: {self.config.DELAY_BETWEEN_RECORDS}s")
        
        try:
            # Connect to MongoDB
            if not self.mongo_helper.connect():
                self.logger.error("Failed to connect to MongoDB. Exiting.")
                sys.exit(1)
            
            # Connect to Kafka
            if not self.kafka_helper.connect():
                self.logger.error("Failed to connect to Kafka. Exiting.")
                sys.exit(1)
            
            self.logger.info("-" * 80)
            self.logger.info("Starting data streaming...")
            self.logger.info("-" * 80)
            
            # Start processing
            self.start_time = time.time()
            
            if self.config.PREFETCH_ALL_DATA:
                self.process_prefetched()
            else:
                self.process_streaming()
            
            self.logger.info("✓ All documents processed successfully")
            
        except KeyboardInterrupt:
            self.logger.warning("\n⚠ Interrupted by user (Ctrl+C)")
            
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}", exc_info=True)
            
        finally:
            self.cleanup()
    
    def process_prefetched(self):
        """Process all documents after loading into memory"""
        # Fetch all documents
        documents = self.mongo_helper.fetch_all_documents()
        
        if documents is None:
            self.logger.error("Failed to fetch documents")
            return
        
        if len(documents) == 0:
            self.logger.warning("No documents to process")
            return
        
        # Close MongoDB connection (we have all data in memory)
        self.mongo_helper.close()
        
        # Process each document
        for doc in documents:
            try:
                self.process_single_document(doc)
                self.processed_count += 1
                
                # Log progress
                if self.processed_count % 10 == 0:
                    self.log_progress()
                    self.log_sample_doc_mid_progress(doc)
                
                # Wait before next record
                time.sleep(self.config.DELAY_BETWEEN_RECORDS)
                
            except Exception as e:
                self.logger.error(f"Error processing document: {e}")
                continue
    
    def process_streaming(self):
        """Process documents via streaming cursor"""
        cursor = self.mongo_helper.fetch_documents_cursor()
        
        if cursor is None:
            self.logger.error("Failed to get cursor")
            return
        
        for doc in cursor:
            try:
                self.process_single_document(doc)
                self.processed_count += 1
                
                # Log progress
                if self.processed_count % 10 == 0:
                    self.log_progress()
                
                # Wait before next record
                time.sleep(self.config.DELAY_BETWEEN_RECORDS)
                
            except Exception as e:
                self.logger.error(f"Error processing document: {e}")
                continue
    
    def process_single_document(self, doc: Dict[str, Any]):
        """Process single document"""
        # Create a copy to avoid modifying original data
        doc_copy = doc.copy()
        
        # Replace timestamp with current time to simulate live data
        current_timestamp = datetime.now()
        doc_copy['timestamp'] = current_timestamp
        
        # Use current timestamp for key
        key = current_timestamp.isoformat()
        
        # Convert ObjectId to string
        if '_id' in doc_copy:
            doc_copy['_id'] = str(doc_copy['_id'])
        
        # Send to Kafka with modified timestamp
        success = self.kafka_helper.produce_message(key=key, value=doc_copy)
        
        if not success:
            self.logger.warning(f"⚠ Failed to send document | Key: {key}")
    
    def log_progress(self):
        """Log processing progress"""
        elapsed = time.time() - self.start_time
        rate = self.processed_count / elapsed if elapsed > 0 else 0
        
        self.logger.info(
            f"Progress: {self.processed_count} documents | "
            f"Rate: {rate:.2f} docs/sec | "
            f"Delivered: {self.kafka_helper.delivered_count} | "
            f"Failed: {self.kafka_helper.failed_count}"
        )
    
    def log_sample_doc_mid_progress(self, doc):
        """Log a line sample document middle of the progress"""
                
        self.logger.info(
            f"Progress: {self.processed_count} documents | "
            f"Document Line Sample given below"
        )

        self.logger.info("Sample document:")
        self.logger.info("\n" + json.dumps(doc, indent=2, default=str))

    def cleanup(self):
        """Cleanup and print summary"""
        self.logger.info("-" * 80)
        self.logger.info("Cleaning up resources...")
        
        # Close connections
        self.kafka_helper.close()
        self.mongo_helper.close()
        
        # Print summary
        if self.start_time:
            elapsed = time.time() - self.start_time
            
            self.logger.info("=" * 80)
            self.logger.info("EXECUTION SUMMARY")
            self.logger.info("=" * 80)
            self.logger.info(f"Total documents processed: {self.processed_count}")
            self.logger.info(f"Messages delivered: {self.kafka_helper.delivered_count}")
            self.logger.info(f"Messages failed: {self.kafka_helper.failed_count}")
            self.logger.info(f"Total time: {elapsed:.2f} seconds")
            
            if elapsed > 0:
                self.logger.info(f"Average rate: {self.processed_count / elapsed:.2f} docs/sec")
            
            self.logger.info("=" * 80)
        
        self.logger.info("✓ Cleanup completed")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Main entry point"""
    config = Config()
    producer = OHLCVStreamProducer(config)
    producer.run()


if __name__ == "__main__":
    main()