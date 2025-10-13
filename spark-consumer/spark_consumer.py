#!/usr/bin/env python3
"""
Kafka to Spark Consumer with Redis Integration (Docker-Ready)
Reads OHLCV data from Kafka, logs it, and stores latest records per symbol in Redis
"""

import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import redis
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
    
    # Redis Configuration
    REDIS_HOST = os.getenv("REDIS_HOST", "redis")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    REDIS_SOCKET_TIMEOUT = int(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))
    REDIS_KEY_PREFIX = "ohlcv:latest"  # Key pattern: ohlcv:latest:{symbol}
    
    # Logging Configuration
    LOG_DIR = os.getenv("LOG_DIR", "/app/logs")
    DATA_LOG_DIR = os.getenv("DATA_LOG_DIR", "/app/logs/data")
    REDIS_LOG_DIR = os.getenv("REDIS_LOG_DIR", "/app/logs/redis")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# ============================================================================
# Logging Setup - Clean and Minimal
# ============================================================================

def setup_logging(config: Config) -> tuple:
    """
    Set up clean, minimal logging to separate files
    
    Returns:
        Tuple of (main_logger, data_log_file, redis_logger)
    """
    # Create directories
    log_dir = Path(config.LOG_DIR)
    data_log_dir = Path(config.DATA_LOG_DIR)
    redis_log_dir = Path(config.REDIS_LOG_DIR)
    
    for dir_path in [log_dir, data_log_dir, redis_log_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create log files with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    app_log_file = log_dir / f"spark_{timestamp}.log"
    data_log_file = data_log_dir / f"data_{timestamp}.jsonl"
    redis_log_file = redis_log_dir / f"redis_{timestamp}.log"
    
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    
    # === Main Application Logger ===
    main_logger = logging.getLogger("App")
    main_logger.setLevel(log_level)
    main_logger.handlers = []
    
    # File handler for main app
    app_file_handler = logging.FileHandler(app_log_file)
    app_file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-5s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # Console handler for main app
    app_console_handler = logging.StreamHandler(sys.stdout)
    app_console_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-5s | %(message)s',
        datefmt='%H:%M:%S'
    ))
    
    main_logger.addHandler(app_file_handler)
    main_logger.addHandler(app_console_handler)
    
    # === Redis Logger (Separate) ===
    redis_logger = logging.getLogger("Redis")
    redis_logger.setLevel(log_level)
    redis_logger.handlers = []
    redis_logger.propagate = False  # Don't propagate to root
    
    # File handler for Redis
    redis_file_handler = logging.FileHandler(redis_log_file)
    redis_file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    redis_logger.addHandler(redis_file_handler)
    
    # Initialize logs
    main_logger.info("=" * 80)
    main_logger.info("SPARK CONSUMER STARTED")
    main_logger.info("=" * 80)
    main_logger.info(f"App Log:   {app_log_file}")
    main_logger.info(f"Data Log:  {data_log_file}")
    main_logger.info(f"Redis Log: {redis_log_file}")
    main_logger.info("=" * 80)
    
    return main_logger, data_log_file, redis_logger


# ============================================================================
# Redis Manager - Clean Logging
# ============================================================================

class RedisManager:
    """Manages Redis connections and operations"""
    
    def __init__(self, config: Config, main_logger: logging.Logger, redis_logger: logging.Logger):
        """Initialize Redis manager"""
        self.config = config
        self.logger = main_logger
        self.redis_logger = redis_logger
        self.redis_client: Optional[redis.Redis] = None
        self.write_count = 0
        self.error_count = 0
        self.symbols_updated = set()
        
        self._connect()
    
    def _connect(self):
        """Establish Redis connection"""
        try:
            self.logger.info(f"Connecting to Redis at {self.config.REDIS_HOST}:{self.config.REDIS_PORT}")
            
            self.redis_client = redis.Redis(
                host=self.config.REDIS_HOST,
                port=self.config.REDIS_PORT,
                db=self.config.REDIS_DB,
                socket_timeout=self.config.REDIS_SOCKET_TIMEOUT,
                decode_responses=True,
                socket_connect_timeout=5
            )
            
            # Test connection
            self.redis_client.ping()
            self.logger.info("Redis connected successfully")
            
            # Redis log header
            self.redis_logger.info("=" * 80)
            self.redis_logger.info("REDIS WRITE LOG")
            self.redis_logger.info("=" * 80)
            
        except redis.ConnectionError as e:
            self.logger.error(f"Redis connection failed: {e}")
            raise
    
    def store_latest_record(self, symbol: str, data: Dict[str, Any], 
                           record_timestamp: str, kafka_meta: Dict[str, Any], 
                           batch_id: int) -> bool:
        """
        Store latest OHLCV record for a symbol in Redis
        """
        try:
            redis_key = f"{self.config.REDIS_KEY_PREFIX}:{symbol}"
            
            # Prepare hash data
            hash_data = {
                'symbol': symbol,
                'open': str(data.get('open', '')),
                'high': str(data.get('high', '')),
                'low': str(data.get('low', '')),
                'close': str(data.get('close', '')),
                'volume': str(data.get('volume', '')),
                'date': str(data.get('Date', '')),
                'time': str(data.get('Time', '')),
                'record_timestamp': record_timestamp,
                'last_updated': datetime.now().isoformat(),
                'kafka_partition': str(kafka_meta.get('partition', '')),
                'kafka_offset': str(kafka_meta.get('offset', '')),
                'batch_id': str(batch_id),
            }
            
            # Write to Redis
            self.redis_client.hset(redis_key, mapping=hash_data)
            
            # Track statistics
            self.write_count += 1
            self.symbols_updated.add(symbol)
            
            # Clean Redis log
            close_price = float(data.get('close', 0))
            volume = float(data.get('volume', 0))
            self.redis_logger.info(
                f"{symbol:<15} | Date: {data.get('Date', 'N/A'):<8} Time: {data.get('Time', 'N/A'):<6} | "
                f"Close: ${close_price:>10,.2f} | Volume: {volume:>12,.0f} | "
                f"Offset: {kafka_meta.get('offset', 'N/A'):<6}"
            )
            
            return True
            
        except redis.RedisError as e:
            self.error_count += 1
            self.logger.error(f"Redis write failed for {symbol}: {e}")
            return False
        
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Unexpected Redis error for {symbol}: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Redis operation statistics"""
        return {
            'total_writes': self.write_count,
            'total_errors': self.error_count,
            'unique_symbols': len(self.symbols_updated),
            'symbols': sorted(list(self.symbols_updated))
        }
    
    def close(self):
        """Close Redis connection"""
        if self.redis_client:
            try:
                # Final summary to Redis log
                self.redis_logger.info("=" * 80)
                self.redis_logger.info("REDIS SESSION SUMMARY")
                self.redis_logger.info("=" * 80)
                self.redis_logger.info(f"Total Writes: {self.write_count}")
                self.redis_logger.info(f"Total Errors: {self.error_count}")
                self.redis_logger.info(f"Unique Symbols: {len(self.symbols_updated)}")
                self.redis_logger.info("=" * 80)
                
                self.redis_client.close()
                self.logger.info("Redis connection closed")
            except Exception as e:
                self.logger.error(f"Error closing Redis: {e}")


# ============================================================================
# Spark Session Setup
# ============================================================================

def create_spark_session(config: Config, logger: logging.Logger) -> Optional[SparkSession]:
    """Create and configure Spark session"""
    try:
        logger.info(f"Creating Spark session: {config.APP_NAME}")
        logger.info(f"Kafka servers: {config.KAFKA_BOOTSTRAP_SERVERS}")
        
        spark = SparkSession.builder \
            .appName(config.APP_NAME) \
            .config("spark.sql.streaming.checkpointLocation", config.CHECKPOINT_LOCATION) \
            .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true") \
            .config("spark.streaming.stopGracefullyOnShutdown", "true") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("WARN")
        
        logger.info(f"Spark session created (version {spark.version})")
        return spark
        
    except Exception as e:
        logger.error(f"Failed to create Spark session: {e}", exc_info=True)
        return None


# ============================================================================
# Schema Definition
# ============================================================================

def get_ohlcv_schema() -> StructType:
    """Define schema for OHLCV data"""
    return StructType([
        StructField("_id", StringType(), True),
        StructField("timestamp", TimestampType(), True),
        StructField("symbol", StringType(), True),
        StructField("open", DoubleType(), True),
        StructField("high", DoubleType(), True),
        StructField("low", DoubleType(), True),
        StructField("close", DoubleType(), True),
        StructField("volume", DoubleType(), True),
    ])


# ============================================================================
# Data Extraction Helper
# ============================================================================

def extract_symbols_from_record(ohlcv_data: Dict[str, Any]) -> list:
    """
    Extract individual symbol data from the nested structure
    
    Args:
        ohlcv_data: The parsed OHLCV data dictionary
        
    Returns:
        List of tuples (symbol_name, symbol_data_dict)
    """
    symbols = []
    record_timestamp = ohlcv_data.get('timestamp', '')
    
    # Iterate through all keys in the data
    for key, value in ohlcv_data.items():
        # Skip metadata fields
        if key in ['_id', 'timestamp']:
            continue
            
        # Check if this is a symbol data object (has 'Type' field)
        if isinstance(value, dict) and 'Type' in value:
            symbol_name = value.get('Type', 'UNKNOWN')
            symbols.append((symbol_name, value, record_timestamp))
    
    return symbols


# ============================================================================
# Streaming Query Handler
# ============================================================================

class StreamingQueryHandler:
    """Handler for Spark Structured Streaming query"""
    
    def __init__(self, config: Config, logger: logging.Logger, 
                 data_log_file: Path, redis_manager: RedisManager):
        """Initialize handler"""
        self.config = config
        self.logger = logger
        self.data_log_file = data_log_file
        self.redis_manager = redis_manager
        self.processed_count = 0
        self.batch_count = 0
        self.total_symbols_written = 0
        
        # Open data log file
        self.data_log_handle = open(data_log_file, 'a', encoding='utf-8')
        self.logger.info(f"Data log file opened: {data_log_file}")
    
    def process_batch(self, batch_df, batch_id):
        """Process each micro-batch of data"""
        try:
            self.batch_count += 1
            batch_size = batch_df.count()
            
            if batch_size == 0:
                return
            
            self.logger.info("-" * 80)
            self.logger.info(f"BATCH {batch_id} | Records: {batch_size}")
            self.logger.info("-" * 80)
            
            # Collect records
            records = batch_df.collect()
            redis_success_count = 0
            redis_fail_count = 0
            
            # Track first record for sample display
            first_record_shown = False
            
            for record in records:
                # Extract data
                json_value = record.json_value
                kafka_key = record.kafka_key
                partition = record.partition
                offset = record.offset
                kafka_timestamp = record.kafka_timestamp
                
                # Parse JSON
                try:
                    ohlcv_data = json.loads(json_value)
                except json.JSONDecodeError:
                    self.logger.error(f"Failed to parse JSON: {json_value[:100]}...")
                    continue
                
                # Write to data log file
                record_dict = {
                    'ohlcv_data': ohlcv_data,
                    'kafka_metadata': {
                        'key': kafka_key,
                        'partition': partition,
                        'offset': offset,
                        'timestamp': str(kafka_timestamp)
                    },
                    'batch_id': batch_id,
                    'processed_at': datetime.now().isoformat()
                }
                json_line = json.dumps(record_dict, default=str)
                self.data_log_handle.write(json_line + '\n')
                
                # Extract all symbols from this record
                symbols = extract_symbols_from_record(ohlcv_data)
                
                # Show sample of first record
                if not first_record_shown and symbols:
                    self.logger.info(f"Sample Record (Offset {offset}):")
                    self.logger.info(f"  Timestamp: {ohlcv_data.get('timestamp', 'N/A')}")
                    self.logger.info(f"  Symbols extracted: {len(symbols)}")
                    # Show first 3 symbols as sample
                    for i, (symbol, data, _) in enumerate(symbols[:3]):
                        self.logger.info(
                            f"    [{i+1}] {symbol}: "
                            f"Close=${data.get('close', 0):.2f} "
                            f"Volume={data.get('volume', 0):.0f}"
                        )
                    if len(symbols) > 3:
                        self.logger.info(f"    ... and {len(symbols) - 3} more symbols")
                    self.logger.info("-" * 80)
                    first_record_shown = True
                
                # Store each symbol in Redis
                kafka_meta = {
                    'partition': partition,
                    'offset': offset,
                    'timestamp': kafka_timestamp
                }
                
                for symbol_name, symbol_data, record_timestamp in symbols:
                    success = self.redis_manager.store_latest_record(
                        symbol=symbol_name,
                        data=symbol_data,
                        record_timestamp=record_timestamp,
                        kafka_meta=kafka_meta,
                        batch_id=batch_id
                    )
                    
                    if success:
                        redis_success_count += 1
                        self.total_symbols_written += 1
                    else:
                        redis_fail_count += 1
                
                self.processed_count += 1
            
            # Flush data log
            self.data_log_handle.flush()
            
            # Batch summary
            self.logger.info(
                f"Batch {batch_id} Complete | "
                f"Kafka Records: {batch_size} | "
                f"Symbols Written: {redis_success_count} | "
                f"Failed: {redis_fail_count} | "
                f"Total: {self.processed_count} records, {self.total_symbols_written} symbols"
            )
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"Error processing batch {batch_id}: {e}", exc_info=True)
    
    def close(self):
        """Close data log file"""
        if self.data_log_handle:
            self.data_log_handle.close()
            self.logger.info(
                f"Data log closed: {self.processed_count} records, "
                f"{self.total_symbols_written} symbols, {self.batch_count} batches"
            )


# ============================================================================
# Main Consumer Logic
# ============================================================================

class SparkKafkaConsumer:
    """Main class for Spark Kafka streaming consumer with Redis"""
    
    def __init__(self, config: Config):
        """Initialize consumer"""
        self.config = config
        self.logger, self.data_log_file, self.redis_logger = setup_logging(config)
        self.spark: Optional[SparkSession] = None
        self.redis_manager: Optional[RedisManager] = None
        self.handler: Optional[StreamingQueryHandler] = None
        self.query = None
    
    def run(self):
        """Main execution"""
        self.logger.info("Starting OHLCV Streaming: Kafka → Spark → Redis")
        self.logger.info(f"Topic: {self.config.KAFKA_TOPIC}")
        self.logger.info(f"Starting offsets: {self.config.KAFKA_STARTING_OFFSETS}")
        self.logger.info(f"Trigger interval: {self.config.TRIGGER_INTERVAL}")
        self.logger.info("-" * 80)
        
        try:
            # Create Spark session
            self.spark = create_spark_session(self.config, self.logger)
            if not self.spark:
                self.logger.error("Failed to create Spark session. Exiting.")
                sys.exit(1)
            
            # Initialize Redis
            self.redis_manager = RedisManager(self.config, self.logger, self.redis_logger)
            
            # Create query handler
            self.handler = StreamingQueryHandler(
                self.config, 
                self.logger, 
                self.data_log_file,
                self.redis_manager
            )
            
            self.logger.info("-" * 80)
            self.logger.info("Setting up Kafka stream...")
            
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
            
            self.logger.info("Kafka stream source created")
            
            # Parse data from Kafka
            parsed_df = kafka_df.selectExpr(
                "CAST(key AS STRING) as kafka_key",
                "CAST(value AS STRING) as json_value",
                "topic",
                "partition",
                "offset",
                "timestamp as kafka_timestamp"
            )
            
            self.logger.info("Data parsing configured")
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
            
            self.logger.info("Streaming query started successfully")
            self.logger.info("Waiting for data... (Press Ctrl+C to stop)")
            self.logger.info("=" * 80)
            
            # Wait for termination
            self.query.awaitTermination()
            
        except KeyboardInterrupt:
            self.logger.warning("Interrupted by user (Ctrl+C)")
            
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}", exc_info=True)
            
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup and print summary"""
        self.logger.info("=" * 80)
        self.logger.info("SHUTTING DOWN")
        self.logger.info("=" * 80)
        
        # Stop query
        if self.query and self.query.isActive:
            self.logger.info("Stopping streaming query...")
            self.query.stop()
            self.logger.info("Streaming query stopped")
        
        # Close handler
        if self.handler:
            self.handler.close()
        
        # Get Redis statistics
        if self.redis_manager:
            redis_stats = self.redis_manager.get_statistics()
            self.logger.info("-" * 80)
            self.logger.info("REDIS STATISTICS")
            self.logger.info("-" * 80)
            self.logger.info(f"Total Writes: {redis_stats['total_writes']}")
            self.logger.info(f"Total Errors: {redis_stats['total_errors']}")
            self.logger.info(f"Unique Symbols: {redis_stats['unique_symbols']}")
            if redis_stats['symbols']:
                symbols_str = ', '.join(redis_stats['symbols'][:10])
                if len(redis_stats['symbols']) > 10:
                    symbols_str += f" ... (+{len(redis_stats['symbols']) - 10} more)"
                self.logger.info(f"Symbols: {symbols_str}")
            self.logger.info("-" * 80)
            
            self.redis_manager.close()
        
        # Stop Spark
        if self.spark:
            self.logger.info("Stopping Spark session...")
            self.spark.stop()
            self.logger.info("Spark session stopped")
        
        # Print summary
        self.logger.info("=" * 80)
        self.logger.info("EXECUTION SUMMARY")
        self.logger.info("=" * 80)
        if self.handler:
            self.logger.info(f"Kafka Records Processed: {self.handler.processed_count}")
            self.logger.info(f"Symbols Written to Redis: {self.handler.total_symbols_written}")
            self.logger.info(f"Batches Processed: {self.handler.batch_count}")
        self.logger.info(f"Data Log: {self.data_log_file}")
        self.logger.info("=" * 80)
        self.logger.info("Cleanup completed")


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