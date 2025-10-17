#!/usr/bin/env python3
"""
Kafka to Spark Consumer with Redis Integration (Docker-Ready)
Reads OHLCV data from Kafka, explodes by symbol, enriches with indicators, 
and stores in Redis, InfluxDB, and ClickHouse
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
from pyspark.sql.functions import col, explode, udf
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    TimestampType, ArrayType, IntegerType
)

# import the db managers
from influx_manager import InfluxManager
from clickhouse_manager import ClickHouseManager

# import the indicator engine
from indicator_engine import IndicatorEngine

# ... [Keep all the Config, setup_logging, RedisManager, create_spark_session code exactly as is] ...

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
    TRIGGER_INTERVAL = os.getenv("TRIGGER_INTERVAL", "500 milliseconds")
    CHECKPOINT_LOCATION = os.getenv("CHECKPOINT_LOCATION", "/app/checkpoints")
    
    # Redis Configuration
    REDIS_HOST = os.getenv("REDIS_HOST", "redis")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    REDIS_SOCKET_TIMEOUT = int(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))
    REDIS_KEY_PREFIX = "ohlcv:latest"

    # InfluxDB Configuration
    INFLUX_URL = os.getenv("INFLUX_URL", "http://influxdb:8086")
    INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "my-super-secret-auth-token")
    INFLUX_ORG = os.getenv("INFLUX_ORG", "my-org")
    INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "ohlcv_data")

    # ClickHouse Configuration
    CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "clickhouse")
    CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT", "8123"))
    CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "default")
    CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "")
    CLICKHOUSE_DATABASE = os.getenv("CLICKHOUSE_DATABASE", "default")
    
    # Logging Configuration
    LOG_DIR = os.getenv("LOG_DIR", "/app/logs")
    DATA_LOG_DIR = os.getenv("DATA_LOG_DIR", "/app/logs/data")
    REDIS_LOG_DIR = os.getenv("REDIS_LOG_DIR", "/app/logs/redis")
    INFLUX_LOG_DIR = os.getenv("INFLUX_LOG_DIR", "/app/logs/influx")          
    CLICKHOUSE_LOG_DIR = os.getenv("CLICKHOUSE_LOG_DIR", "/app/logs/clickhouse")
    INDICATOR_ENG_LOG_DIR = os.getenv("INDICATOR_ENG_LOG_DIR", "/app/logs/aux_processors")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(config: Config) -> tuple:
    """Set up logging"""
    log_dir = Path(config.LOG_DIR)
    data_log_dir = Path(config.DATA_LOG_DIR)
    redis_log_dir = Path(config.REDIS_LOG_DIR)
    influx_log_dir = Path(config.INFLUX_LOG_DIR)
    clickhouse_log_dir = Path(config.CLICKHOUSE_LOG_DIR)
    indicator_eng_dir = Path(config.INDICATOR_ENG_LOG_DIR)
    
    for dir_path in [log_dir, data_log_dir, redis_log_dir, influx_log_dir, clickhouse_log_dir, indicator_eng_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create log files with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    app_log_file = log_dir / f"spark_{timestamp}.log"
    data_log_file = data_log_dir / f"data_{timestamp}.jsonl"
    redis_log_file = redis_log_dir / f"redis_{timestamp}.log"
    influx_log_file = influx_log_dir / f"influx_{timestamp}.log"
    clickhouse_log_file = clickhouse_log_dir / f"clickhouse_{timestamp}.log"
    indicatoreng_log_file = indicator_eng_dir / f"indicator_eng_{timestamp}.log"
    
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

    # === Indicator Engine Logger (Separate) ===
    indicator_eng_logger = logging.getLogger("IndicatorEngine")
    indicator_eng_logger.setLevel(log_level)
    indicator_eng_logger.handlers = []
    indicator_eng_logger.propagate = False
    
    # File handler for Redis
    indicator_eng_file_handler = logging.FileHandler(indicatoreng_log_file)
    indicator_eng_file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    indicator_eng_logger.addHandler(indicator_eng_file_handler)
    
    # === Redis Logger (Separate) ===
    redis_logger = logging.getLogger("Redis")
    redis_logger.setLevel(log_level)
    redis_logger.handlers = []
    redis_logger.propagate = False
    
    # File handler for Redis
    redis_file_handler = logging.FileHandler(redis_log_file)
    redis_file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    redis_logger.addHandler(redis_file_handler)

    # === InfluxDB Logger ===
    influx_logger = logging.getLogger("Influx")
    influx_logger.setLevel(log_level)
    influx_logger.handlers = []
    influx_logger.propagate = False
    
    influx_file_handler = logging.FileHandler(influx_log_file)
    influx_file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    influx_logger.addHandler(influx_file_handler)
    
    # === ClickHouse Logger ===
    clickhouse_logger = logging.getLogger("ClickHouse")
    clickhouse_logger.setLevel(log_level)
    clickhouse_logger.handlers = []
    clickhouse_logger.propagate = False
    
    clickhouse_file_handler = logging.FileHandler(clickhouse_log_file)
    clickhouse_file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    clickhouse_logger.addHandler(clickhouse_file_handler)
    
    # Initialize logs
    main_logger.info("=" * 80)
    main_logger.info("SPARK CONSUMER STARTED")
    main_logger.info("=" * 80)
    main_logger.info(f"App Log:   {app_log_file}")
    main_logger.info(f"Data Log:  {data_log_file}")
    main_logger.info(f"Redis Log: {redis_log_file}")
    main_logger.info(f"InfluxDB Log:   {influx_log_file}")
    main_logger.info(f"ClickHouse Log: {clickhouse_log_file}")
    main_logger.info(f"IndicatorEngine Log: {indicatoreng_log_file}")
    main_logger.info("=" * 80)
    
    return main_logger, data_log_file, redis_logger, influx_logger, clickhouse_logger, indicator_eng_logger



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
                'date': str(data.get('date', '')),
                'time': str(data.get('time', '')),
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
            
            # Safe logging with None handling
            close_price = float(data.get('close', 0)) if data.get('close') is not None else 0
            volume = float(data.get('volume', 0)) if data.get('volume') is not None else 0
            self.redis_logger.info(
                f"{symbol:<15} | Date: {data.get('date', 'N/A'):<8} Time: {data.get('time', 'N/A'):<6} | "
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
# UDF for Symbol Explosion - WITH DEBUGGING
# ============================================================================

def create_symbol_explosion_udf():
    """Create UDF to parse JSON and explode symbols"""
    
    symbol_schema = ArrayType(StructType([
        StructField("symbol", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("open", DoubleType(), True),
        StructField("high", DoubleType(), True),
        StructField("low", DoubleType(), True),
        StructField("close", DoubleType(), True),
        StructField("volume", DoubleType(), True),
        StructField("date", StringType(), True),
        StructField("time", StringType(), True),
    ]))
    
    @udf(returnType=symbol_schema)
    def parse_and_explode_udf(json_str):
        """Parse JSON and extract all symbols"""
        try:
            ohlcv_data = json.loads(json_str)
            symbols = []
            record_timestamp = ohlcv_data.get('timestamp', '')
            
            for key, value in ohlcv_data.items():
                if key in ['_id', 'timestamp']:
                    continue
                    
                if isinstance(value, dict) and 'Type' in value:
                    # Extract with explicit type conversion
                    symbol_dict = {
                        'symbol': str(value.get('Type', '')),
                        'timestamp': str(record_timestamp),
                        'open': float(value.get('open', 0)) if value.get('open') is not None else None,
                        'high': float(value.get('high', 0)) if value.get('high') is not None else None,
                        'low': float(value.get('low', 0)) if value.get('low') is not None else None,
                        'close': float(value.get('close', 0)) if value.get('close') is not None else None,
                        'volume': float(value.get('volume', 0)) if value.get('volume') is not None else None,
                        'date': str(value.get('Date', '')),
                        'time': str(value.get('Time', '')),
                    }
                    symbols.append(symbol_dict)
            
            return symbols
        except Exception as e:
            return []
    
    return parse_and_explode_udf


# ============================================================================
# Streaming Query Handler - WITH SAFE LOGGING
# ============================================================================

class StreamingQueryHandler:
    """Handler for Spark Structured Streaming query"""
    
    def __init__(self, config: Config, logger: logging.Logger, 
                 data_log_file: Path, redis_manager: RedisManager,
                 influx_manager: InfluxManager, clickhouse_manager: ClickHouseManager, 
                 indicator_engine: IndicatorEngine):
        self.config = config
        self.logger = logger
        self.data_log_file = data_log_file
        self.redis_manager = redis_manager
        self.influx_manager = influx_manager
        self.clickhouse_manager = clickhouse_manager
        self.indicator_engine = indicator_engine
        self.processed_count = 0
        self.batch_count = 0
        self.total_symbols_written = 0
        
        # Open data log file
        self.data_log_handle = open(data_log_file, 'a', encoding='utf-8')
        self.logger.info(f"Data log file opened: {data_log_file}")
    
    def process_batch(self, batch_df, batch_id):
        """Process each micro-batch with SAFE logging"""
        try:
            self.batch_count += 1
            batch_size = batch_df.count()
            
            if batch_size == 0:
                return
            
            self.logger.info("-" * 80)
            self.logger.info(f"BATCH {batch_id} | Raw Symbol Records: {batch_size}")
            self.logger.info("-" * 80)
            
            raw_records = batch_df.collect()
            
            # Convert Spark Rows to dicts with explicit type handling
            input_rows = []
            for record in raw_records:
                row_dict = {
                    'symbol': str(record.symbol) if record.symbol else '',
                    'timestamp': str(record.timestamp) if record.timestamp else '',
                    'open': float(record.open) if record.open is not None else None,
                    'high': float(record.high) if record.high is not None else None,
                    'low': float(record.low) if record.low is not None else None,
                    'close': float(record.close) if record.close is not None else None,
                    'volume': float(record.volume) if record.volume is not None else None,
                    'date': str(record.date) if record.date else '',
                    'time': str(record.time) if record.time else '',
                    'offset': int(record.offset) if record.offset is not None else 0,
                    'partition': int(record.partition) if record.partition is not None else 0,
                    'kafka_timestamp': record.kafka_timestamp if record.kafka_timestamp else datetime.now()
                }
                input_rows.append(row_dict)
            
            # DEBUG: Log first row to see what's coming from Spark
            if input_rows:
                first_row = input_rows[0]
                self.logger.info(f"DEBUG - First Spark Row: symbol={first_row['symbol']}, "
                               f"volume={first_row['volume']}, close={first_row['close']}")
            
            self.logger.info(f"Sending {len(input_rows)} rows to IndicatorEngine...")
            
            ready, enriched_rows = self.indicator_engine.process_batch(input_rows)
            
            if not ready:
                self.logger.info(f"IndicatorEngine warming up... (received {len(input_rows)} rows)")
                self.logger.info("Skipping writes to Redis/InfluxDB/ClickHouse")
                self.logger.info("=" * 80)
                return
            
            self.logger.info(f"IndicatorEngine ready! Received {len(enriched_rows)} enriched rows")
            self.logger.info("-" * 80)
            
            if len(enriched_rows) == 0:
                self.logger.warning("IndicatorEngine returned ready=True but no data")
                self.logger.info("=" * 80)
                return
            
            redis_success_count = 0
            redis_fail_count = 0
            
            # Track first record for sample display
            first_record_shown = False
            
            all_symbols_data = []
            latest_per_symbol = {}
            
            for enriched_row in enriched_rows:
                symbol = enriched_row['symbol']
                timestamp = enriched_row['timestamp']
                
                symbol_data = {
                    'open': enriched_row['open'],
                    'high': enriched_row['high'],
                    'low': enriched_row['low'],
                    'close': enriched_row['close'],
                    'volume': enriched_row['volume'],
                    'date': enriched_row['date'],
                    'time': enriched_row['time'],
                }
                
                base_fields = {'symbol', 'timestamp', 'open', 'high', 'low', 
                            'close', 'volume', 'date', 'time', 
                            'offset', 'partition', 'kafka_timestamp'}
                
                for field_name, field_value in enriched_row.items():
                    if field_name not in base_fields:
                        symbol_data[field_name] = field_value
                
                kafka_meta = {
                    'partition': enriched_row.get('partition'),
                    'offset': enriched_row.get('offset'),
                    'timestamp': enriched_row.get('kafka_timestamp')
                }
                
                record_dict = {
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'data': symbol_data,
                    'kafka_metadata': kafka_meta,
                    'batch_id': batch_id,
                    'processed_at': datetime.now().isoformat()
                }
                json_line = json.dumps(record_dict, default=str)
                self.data_log_handle.write(json_line + '\n')
                
                # SAFE LOGGING - Handle None values
                if not first_record_shown:
                    self.logger.info(f"Sample Enriched Record (Offset {kafka_meta['offset']}):")
                    self.logger.info(f"  Symbol: {symbol}")
                    self.logger.info(f"  Timestamp: {timestamp}")
                    
                    close_val = enriched_row.get('close')
                    volume_val = enriched_row.get('volume')
                    
                    if close_val is not None:
                        self.logger.info(f"  Close: ${close_val:.2f}")
                    else:
                        self.logger.info(f"  Close: None")
                    
                    if volume_val is not None:
                        self.logger.info(f"  Volume: {volume_val:.0f}")
                    else:
                        self.logger.info(f"  Volume: None (WARNING: Volume is null!)")
                    
                    indicator_fields = [k for k in enriched_row.keys() if k not in base_fields]
                    if indicator_fields:
                        self.logger.info(f"  Indicators: {', '.join(indicator_fields[:5])}")
                        for ind_field in indicator_fields[:3]:
                            val = enriched_row[ind_field]
                            if val is not None:
                                self.logger.info(f"    {ind_field}: {val}")
                    self.logger.info("-" * 80)
                    first_record_shown = True
                
                all_symbols_data.append((symbol, symbol_data, timestamp, kafka_meta))
                
                if symbol not in latest_per_symbol:
                    latest_per_symbol[symbol] = (symbol_data, timestamp, kafka_meta, len(latest_per_symbol))
                else:
                    existing_ts = latest_per_symbol[symbol][1]
                    if timestamp > existing_ts:
                        latest_per_symbol[symbol] = (symbol_data, timestamp, kafka_meta, len(latest_per_symbol))
                    elif timestamp == existing_ts:
                        latest_per_symbol[symbol] = (symbol_data, timestamp, kafka_meta, len(latest_per_symbol))
                
                self.processed_count += 1
            
            self.data_log_handle.flush()
            
            for symbol, (symbol_data, timestamp, kafka_meta, _) in latest_per_symbol.items():
                success = self.redis_manager.store_latest_record(
                    symbol=symbol,
                    data=symbol_data,
                    record_timestamp=timestamp,
                    kafka_meta=kafka_meta,
                    batch_id=batch_id
                )
                
                if success:
                    redis_success_count += 1
                    self.total_symbols_written += 1
                else:
                    redis_fail_count += 1
            
            influx_result = self.influx_manager.write_batch(all_symbols_data, batch_id)

            # Batch write to ClickHouse
            clickhouse_result = self.clickhouse_manager.write_batch(all_symbols_data, batch_id)
            
            self.logger.info(
                f"Batch {batch_id} Complete | "
                f"Enriched Rows: {len(enriched_rows)} | "
                f"Unique Symbols for Redis: {len(latest_per_symbol)} | "
                f"Redis: {redis_success_count}/{redis_fail_count} | "
                f"InfluxDB: {influx_result['success']}/{influx_result['failed']} | "
                f"ClickHouse: {clickhouse_result['success']}/{clickhouse_result['failed']} | "
                f"Total Processed: {self.processed_count}"
            )
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"Error processing batch {batch_id}: {e}", exc_info=True)
    
    def close(self):
        """Close data log file"""
        if self.data_log_handle:
            self.data_log_handle.close()
            self.logger.info(f"Data log closed: {self.processed_count} symbols, {self.batch_count} batches")



class SparkKafkaConsumer:
    """Main class for Spark Kafka streaming consumer with indicator enrichment"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger, self.data_log_file, self.redis_logger, self.influx_logger, self.clickhouse_logger, self.indicator_eng_logger = setup_logging(config)
        self.spark: Optional[SparkSession] = None
        self.redis_manager: Optional[RedisManager] = None
        self.influx_manager: Optional[InfluxManager] = None
        self.clickhouse_manager: Optional[ClickHouseManager] = None
        self.handler: Optional[StreamingQueryHandler] = None
        self.indicator_engine: Optional[IndicatorEngine] = None
        self.query = None
    
    def run(self):
        self.logger.info("Starting OHLCV Streaming: Kafka → Spark → Indicators → Redis/InfluxDB/ClickHouse")
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

            # Initialize InfluxDB
            self.influx_manager = InfluxManager(self.config, self.logger, self.influx_logger)
            self.clickhouse_manager = ClickHouseManager(self.config, self.logger, self.clickhouse_logger)
            self.indicator_engine = IndicatorEngine(self.indicator_eng_logger)
            self.logger.info("Indicator Engine initialized")
            
            self.handler = StreamingQueryHandler(
                self.config, 
                self.logger, 
                self.data_log_file,
                self.redis_manager,
                self.influx_manager,
                self.clickhouse_manager,
                self.indicator_engine
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
                "partition",
                "offset",
                "timestamp as kafka_timestamp"
            )
            
            self.logger.info("Data parsing configured")
            self.logger.info("-" * 80)
            self.logger.info("Setting up symbol explosion and indicator enrichment...")
            
            parse_and_explode = create_symbol_explosion_udf()
            
            symbols_df = parsed_df.select(
                explode(parse_and_explode(col("json_value"))).alias("symbol_data"),
                col("offset"),
                col("partition"),
                col("kafka_timestamp")
            ).select(
                "symbol_data.*",
                "offset",
                "partition",
                "kafka_timestamp"
            )
            
            self.logger.info("Symbol explosion configured")
            
            self.logger.info("-" * 80)
            self.logger.info("Starting streaming query...")
            self.logger.info("-" * 80)
            
            self.query = symbols_df \
                .writeStream \
                .foreachBatch(self.handler.process_batch) \
                .trigger(processingTime=self.config.TRIGGER_INTERVAL) \
                .option("checkpointLocation", self.config.CHECKPOINT_LOCATION) \
                .start()
            
            self.logger.info("Streaming query started successfully")
            self.logger.info("Processing: Kafka → Parse → Explode → IndicatorEngine → Store")
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
        
        if self.query and self.query.isActive:
            self.logger.info("Stopping streaming query...")
            self.query.stop()
            self.logger.info("Streaming query stopped")
        
        if self.handler:
            self.handler.close()
        
        if self.redis_manager:
            redis_stats = self.redis_manager.get_statistics()
            self.logger.info("-" * 80)
            self.logger.info("REDIS STATISTICS")
            self.logger.info("-" * 80)
            self.logger.info(f"Total Writes: {redis_stats['total_writes']}")
            self.logger.info(f"Total Errors: {redis_stats['total_errors']}")
            self.logger.info(f"Unique Symbols: {redis_stats['unique_symbols']}")
            self.logger.info("-" * 80)
            self.redis_manager.close()

        if self.influx_manager:
            influx_stats = self.influx_manager.get_statistics()
            self.logger.info("-" * 80)
            self.logger.info("INFLUXDB STATISTICS")
            self.logger.info("-" * 80)
            self.logger.info(f"Total Points: {influx_stats['total_writes']}")
            self.logger.info(f"Total Errors: {influx_stats['total_errors']}")
            self.logger.info(f"Unique Symbols: {influx_stats['unique_symbols']}")
            self.logger.info("-" * 80)
            self.influx_manager.close()

        if self.clickhouse_manager:
            clickhouse_stats = self.clickhouse_manager.get_statistics()
            self.logger.info("-" * 80)
            self.logger.info("CLICKHOUSE STATISTICS")
            self.logger.info("-" * 80)
            self.logger.info(f"Total Rows: {clickhouse_stats['total_writes']}")
            self.logger.info(f"Total Errors: {clickhouse_stats['total_errors']}")
            self.logger.info(f"Unique Symbols: {clickhouse_stats['unique_symbols']}")
            self.logger.info("-" * 80)
            self.clickhouse_manager.close()
        
        if self.spark:
            self.logger.info("Stopping Spark session...")
            self.spark.stop()
            self.logger.info("Spark session stopped")
        
        self.logger.info("=" * 80)
        self.logger.info("EXECUTION SUMMARY")
        self.logger.info("=" * 80)
        if self.handler:
            self.logger.info(f"Enriched Symbols Processed: {self.handler.processed_count}")
            self.logger.info(f"Symbols Written to Redis: {self.handler.total_symbols_written}")
            self.logger.info(f"Batches Processed: {self.handler.batch_count}")
        self.logger.info(f"Data Log: {self.data_log_file}")
        self.logger.info("=" * 80)
        self.logger.info("Cleanup completed")


def main():
    config = Config()
    consumer = SparkKafkaConsumer(config)
    consumer.run()


if __name__ == "__main__":
    main()