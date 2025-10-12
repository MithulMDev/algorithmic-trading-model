#!/usr/bin/env python3
"""
Kafka to Spark Consumer (Docker-Ready)
Reads OHLCV data from Kafka and logs it in real-time using Structured Streaming
"""

import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_json, struct
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration from environment variables with sensible defaults"""
    
    # Kafka Configuration
    KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
    KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "ohlcv-stream")
    KAFKA_STARTING_OFFSETS = os.getenv("KAFKA_STARTING_OFFSETS", "earliest")
    KAFKA_MAX_OFFSETS_PER_TRIGGER = os.getenv("KAFKA_MAX_OFFSETS_PER_TRIGGER", "1000")
    
    # Spark Configuration
    APP_NAME = os.getenv("SPARK_APP_NAME", "OHLCV-Spark-Consumer")
    TRIGGER_INTERVAL = os.getenv("TRIGGER_INTERVAL", "5 seconds")
    CHECKPOINT_LOCATION = os.getenv("CHECKPOINT_LOCATION", "/app/checkpoints")
    
    # Logging Configuration
    LOG_DIR = os.getenv("LOG_DIR", "/app/logs")
    DATA_LOG_DIR = os.getenv("DATA_LOG_DIR", "/app/logs/data")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(config: Config) -> tuple:
    """
    Set up comprehensive logging to file and console
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (logger, data_log_file_path)
    """
    # Create logs directories
    log_dir = Path(config.LOG_DIR)
    data_log_dir = Path(config.DATA_LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    data_log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    app_log_file = log_dir / f"spark_consumer_{timestamp}.log"
    data_log_file = data_log_dir / f"spark_data_{timestamp}.jsonl"
    
    # Get log level
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger("SparkConsumer")
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler for application logs
    file_handler = logging.FileHandler(app_log_file)
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
    
    logger.info(f"Application logging initialized. Log file: {app_log_file}")
    logger.info(f"Data logging initialized. Data file: {data_log_file}")
    
    return logger, data_log_file


# ============================================================================
# Spark Session Setup
# ============================================================================

def create_spark_session(config: Config, logger: logging.Logger) -> Optional[SparkSession]:
    """
    Create and configure Spark session for Kafka streaming
    
    Args:
        config: Configuration object
        logger: Logger instance
        
    Returns:
        SparkSession or None on error
    """
    try:
        logger.info("Creating Spark session...")
        logger.info(f"App name: {config.APP_NAME}")
        logger.info(f"Kafka bootstrap servers: {config.KAFKA_BOOTSTRAP_SERVERS}")
        
        spark = SparkSession.builder \
            .appName(config.APP_NAME) \
            .config("spark.sql.streaming.checkpointLocation", config.CHECKPOINT_LOCATION) \
            .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true") \
            .config("spark.streaming.stopGracefullyOnShutdown", "true") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
        
        # Set log level for Spark
        spark.sparkContext.setLogLevel("WARN")
        
        logger.info("✓ Spark session created successfully")
        logger.info(f"Spark version: {spark.version}")
        
        return spark
        
    except Exception as e:
        logger.error(f"✗ Failed to create Spark session: {e}", exc_info=True)
        return None


# ============================================================================
# Schema Definition
# ============================================================================

def get_ohlcv_schema() -> StructType:
    """
    Define schema for OHLCV data
    Note: This is flexible and will handle additional fields
    
    Returns:
        StructType schema
    """
    return StructType([
        StructField("_id", StringType(), True),
        StructField("timestamp", TimestampType(), True),
        StructField("symbol", StringType(), True),
        StructField("open", DoubleType(), True),
        StructField("high", DoubleType(), True),
        StructField("low", DoubleType(), True),
        StructField("close", DoubleType(), True),
        StructField("volume", DoubleType(), True),
        # Add other fields as needed - schema is flexible
    ])


# ============================================================================
# Streaming Query Handler
# ============================================================================

class StreamingQueryHandler:
    """Handler for Spark Structured Streaming query"""
    
    def __init__(self, config: Config, logger: logging.Logger, data_log_file: Path):
        """Initialize handler"""
        self.config = config
        self.logger = logger
        self.data_log_file = data_log_file
        self.processed_count = 0
        self.batch_count = 0
        
        # Open data log file
        self.data_log_handle = open(data_log_file, 'a', encoding='utf-8')
        self.logger.info(f"Data log file opened: {data_log_file}")
    
    def process_batch(self, batch_df, batch_id):
        """
        Process each micro-batch of data
        
        Args:
            batch_df: DataFrame for current batch
            batch_id: Batch identifier
        """
        try:
            self.batch_count += 1
            batch_size = batch_df.count()
            
            if batch_size == 0:
                self.logger.debug(f"Batch {batch_id}: No records")
                return
            
            self.logger.info(f"Processing batch {batch_id} with {batch_size} records")
            
            # Convert to JSON and write to log file
            records = batch_df.collect()
            
            for record in records:
                # Convert Row to dictionary
                record_dict = record.asDict(recursive=True)
                
                # Write as JSON line
                json_line = json.dumps(record_dict, default=str)
                self.data_log_handle.write(json_line + '\n')
                
                self.processed_count += 1
            
            # Flush to ensure data is written
            self.data_log_handle.flush()
            
            # Log sample record
            if batch_size > 0:
                sample_record = records[0].asDict(recursive=True)
                self.logger.info(f"Sample record from batch {batch_id}:")
                self.logger.info(json.dumps(sample_record, indent=2, default=str))
            
            self.logger.info(
                f"✓ Batch {batch_id} processed | "
                f"Records in batch: {batch_size} | "
                f"Total processed: {self.processed_count}"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing batch {batch_id}: {e}", exc_info=True)
    
    def close(self):
        """Close data log file"""
        if self.data_log_handle:
            self.data_log_handle.close()
            self.logger.info("Data log file closed")
            self.logger.info(f"Total records logged: {self.processed_count}")
            self.logger.info(f"Total batches processed: {self.batch_count}")


# ============================================================================
# Main Consumer Logic
# ============================================================================

class SparkKafkaConsumer:
    """Main class for Spark Kafka streaming consumer"""
    
    def __init__(self, config: Config):
        """Initialize consumer"""
        self.config = config
        self.logger, self.data_log_file = setup_logging(config)
        self.spark: Optional[SparkSession] = None
        self.handler: Optional[StreamingQueryHandler] = None
        self.query = None
    
    def run(self):
        """Main execution"""
        self.logger.info("=" * 80)
        self.logger.info("OHLCV Data Streaming: Kafka → Spark Consumer")
        self.logger.info("=" * 80)
        self.logger.info(f"Kafka topic: {self.config.KAFKA_TOPIC}")
        self.logger.info(f"Starting offsets: {self.config.KAFKA_STARTING_OFFSETS}")
        self.logger.info(f"Trigger interval: {self.config.TRIGGER_INTERVAL}")
        self.logger.info(f"Checkpoint location: {self.config.CHECKPOINT_LOCATION}")
        
        try:
            # Create Spark session
            self.spark = create_spark_session(self.config, self.logger)
            if not self.spark:
                self.logger.error("Failed to create Spark session. Exiting.")
                sys.exit(1)
            
            # Create query handler
            self.handler = StreamingQueryHandler(
                self.config, 
                self.logger, 
                self.data_log_file
            )
            
            self.logger.info("-" * 80)
            self.logger.info("Reading from Kafka topic...")
            self.logger.info("-" * 80)
            
            # Read from Kafka
            kafka_df = self.spark \
                .readStream \
                .format("kafka") \
                .option("kafka.bootstrap.servers", self.config.KAFKA_BOOTSTRAP_SERVERS) \
                .option("subscribe", self.config.KAFKA_TOPIC) \
                .option("startingOffsets", self.config.KAFKA_STARTING_OFFSETS) \
                .option("maxOffsetsPerTrigger", self.config.KAFKA_MAX_OFFSETS_PER_TRIGGER) \
                .option("failOnDataLoss", "false") \
                .load()
            
            self.logger.info("✓ Kafka stream source created")
            
            # Parse JSON from Kafka value
            # Using a flexible approach that preserves all fields
            parsed_df = kafka_df.selectExpr(
                "CAST(key AS STRING) as kafka_key",
                "CAST(value AS STRING) as json_value",
                "topic",
                "partition",
                "offset",
                "timestamp as kafka_timestamp"
            )
            
            self.logger.info("✓ Data parsing configured")
            self.logger.info("-" * 80)
            self.logger.info("Starting streaming query...")
            self.logger.info("-" * 80)
            
            # Write stream using foreachBatch
            self.query = parsed_df \
                .writeStream \
                .foreachBatch(self.handler.process_batch) \
                .trigger(processingTime=self.config.TRIGGER_INTERVAL) \
                .option("checkpointLocation", self.config.CHECKPOINT_LOCATION) \
                .start()
            
            self.logger.info("✓ Streaming query started successfully")
            self.logger.info("Waiting for data... (Press Ctrl+C to stop)")
            
            # Wait for termination
            self.query.awaitTermination()
            
        except KeyboardInterrupt:
            self.logger.warning("\n⚠ Interrupted by user (Ctrl+C)")
            
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}", exc_info=True)
            
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup and print summary"""
        self.logger.info("-" * 80)
        self.logger.info("Cleaning up resources...")
        
        # Stop query
        if self.query and self.query.isActive:
            self.logger.info("Stopping streaming query...")
            self.query.stop()
            self.logger.info("✓ Streaming query stopped")
        
        # Close handler
        if self.handler:
            self.handler.close()
        
        # Stop Spark
        if self.spark:
            self.logger.info("Stopping Spark session...")
            self.spark.stop()
            self.logger.info("✓ Spark session stopped")
        
        # Print summary
        self.logger.info("=" * 80)
        self.logger.info("EXECUTION SUMMARY")
        self.logger.info("=" * 80)
        if self.handler:
            self.logger.info(f"Total batches processed: {self.handler.batch_count}")
            self.logger.info(f"Total records logged: {self.handler.processed_count}")
        self.logger.info(f"Data log file: {self.data_log_file}")
        self.logger.info("=" * 80)
        
        self.logger.info("✓ Cleanup completed")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Main entry point"""
    config = Config()
    consumer = SparkKafkaConsumer(config)
    consumer.run()


if __name__ == "__main__":
    main()