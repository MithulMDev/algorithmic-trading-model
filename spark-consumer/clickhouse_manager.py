"""
ClickHouse Manager for Columnar OHLCV Data Storage
Handles batch inserts to ClickHouse
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

import clickhouse_connect


class ClickHouseManager:
    """Manages ClickHouse connections and batch write operations"""
    
    def __init__(self, config, main_logger: logging.Logger, clickhouse_logger: logging.Logger):
        """Initialize ClickHouse manager"""
        self.config = config
        self.logger = main_logger
        self.clickhouse_logger = clickhouse_logger
        self.client = None
        self.write_count = 0
        self.error_count = 0
        self.symbols_written = set()
        
        self._connect()
    
    def _connect(self):
        """Establish ClickHouse connection"""
        try:
            self.logger.info(f"Connecting to ClickHouse at {self.config.CLICKHOUSE_HOST}:{self.config.CLICKHOUSE_PORT}")
            self.clickhouse_logger.info("=" * 80)
            self.clickhouse_logger.info("CLICKHOUSE CONNECTION INITIATED")
            self.clickhouse_logger.info("=" * 80)
            self.clickhouse_logger.info(f"Host: {self.config.CLICKHOUSE_HOST}")
            self.clickhouse_logger.info(f"Port: {self.config.CLICKHOUSE_PORT}")
            self.clickhouse_logger.info(f"Database: {self.config.CLICKHOUSE_DATABASE}")
            self.clickhouse_logger.info(f"User: {self.config.CLICKHOUSE_USER}")
            
            self.client = clickhouse_connect.get_client(
                host=self.config.CLICKHOUSE_HOST,
                port=self.config.CLICKHOUSE_PORT,
                username=self.config.CLICKHOUSE_USER,
                password=self.config.CLICKHOUSE_PASSWORD,
                database=self.config.CLICKHOUSE_DATABASE
            )
            
            # Test connection
            self.client.ping()
            self.logger.info("ClickHouse connected successfully")
            self.clickhouse_logger.info("Status: CONNECTED ✓")
            
            # Ensure table exists
            self.clickhouse_logger.info("-" * 80)
            self.clickhouse_logger.info("Verifying table 'ohlcv_data'...")
            self._create_table()
            
            # ClickHouse log header
            self.clickhouse_logger.info("=" * 80)
            self.clickhouse_logger.info("CLICKHOUSE WRITE LOG")
            self.clickhouse_logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"ClickHouse connection failed: {e}")
            self.clickhouse_logger.info("Status: FAILED ✗")
            self.clickhouse_logger.info(f"Error: {str(e)}")
            self.clickhouse_logger.info("=" * 80)
            raise
    
    def _create_table(self):
        """Create OHLCV table if it doesn't exist"""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS ohlcv_data (
            timestamp DateTime64(3),
            symbol String,
            open Float64,
            high Float64,
            low Float64,
            close Float64,
            volume Float64,
            date UInt32,
            time String,
            kafka_partition Int32,
            kafka_offset Int64,
            batch_id Int32,
            inserted_at DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY (symbol, timestamp)
        """
        
        try:
            self.client.command(create_table_query)
            self.logger.info("ClickHouse table 'ohlcv_data' verified/created")
            self.clickhouse_logger.info("Table 'ohlcv_data': READY ✓")
            self.clickhouse_logger.info("Engine: MergeTree | Order: (symbol, timestamp)")
        except Exception as e:
            self.logger.error(f"Failed to create ClickHouse table: {e}")
            self.clickhouse_logger.info(f"Table creation failed: {str(e)}")
            raise
    
    def write_batch(self, symbols_data: List[tuple], batch_id: int) -> Dict[str, int]:
        """
        Write batch of symbols to ClickHouse
        
        Args:
            symbols_data: List of tuples (symbol_name, symbol_data, record_timestamp, kafka_meta)
            batch_id: Batch identifier
            
        Returns:
            Dict with success and failure counts
        """
        success_count = 0
        fail_count = 0
        
        if not symbols_data:
            return {'success': 0, 'failed': 0}
        
        try:
            self.clickhouse_logger.info("-" * 80)
            self.clickhouse_logger.info(f"BATCH {batch_id} | Processing {len(symbols_data)} symbols")
            self.clickhouse_logger.info("-" * 80)
            
            rows = []
            
            # Track first symbol for detailed logging
            first_symbol_logged = False
            
            for symbol_name, symbol_data, record_timestamp, kafka_meta in symbols_data:
                try:
                    # Parse timestamp
                    timestamp = datetime.fromisoformat(record_timestamp.replace(' ', 'T'))
                    
                    # Prepare row
                    row = [
                        timestamp,
                        symbol_name,
                        float(symbol_data.get('open', 0)),
                        float(symbol_data.get('high', 0)),
                        float(symbol_data.get('low', 0)),
                        float(symbol_data.get('close', 0)),
                        float(symbol_data.get('volume', 0)),
                        int(symbol_data.get('Date', 0)),
                        str(symbol_data.get('Time', '')),
                        int(kafka_meta.get('partition', 0)),
                        int(kafka_meta.get('offset', 0)),
                        batch_id
                    ]
                    
                    rows.append(row)
                    self.symbols_written.add(symbol_name)
                    
                    # Log first symbol details
                    if not first_symbol_logged:
                        self.clickhouse_logger.info("Sample Row:")
                        self.clickhouse_logger.info(f"  Symbol: {symbol_name}")
                        self.clickhouse_logger.info(f"  Timestamp: {timestamp}")
                        self.clickhouse_logger.info(f"  OHLC: O=${symbol_data.get('open', 0):.2f} "
                                                   f"H=${symbol_data.get('high', 0):.2f} "
                                                   f"L=${symbol_data.get('low', 0):.2f} "
                                                   f"C=${symbol_data.get('close', 0):.2f}")
                        self.clickhouse_logger.info(f"  Volume: {symbol_data.get('volume', 0):,.0f}")
                        self.clickhouse_logger.info(f"  Date/Time: {symbol_data.get('Date', 'N/A')} {symbol_data.get('Time', 'N/A')}")
                        self.clickhouse_logger.info(f"  Kafka: Partition={kafka_meta.get('partition', 0)} Offset={kafka_meta.get('offset', 0)}")
                        self.clickhouse_logger.info("-" * 80)
                        first_symbol_logged = True
                    
                except Exception as e:
                    self.logger.error(f"Error preparing row for {symbol_name}: {e}")
                    self.clickhouse_logger.info(f"ERROR | Symbol: {symbol_name} | {str(e)[:50]}")
                    fail_count += 1
            
            # Batch insert
            if rows:
                self.clickhouse_logger.info(f"Inserting {len(rows)} rows into ClickHouse...")
                
                self.client.insert(
                    'ohlcv_data',
                    rows,
                    column_names=[
                        'timestamp', 'symbol', 'open', 'high', 'low', 'close',
                        'volume', 'date', 'time', 'kafka_partition',
                        'kafka_offset', 'batch_id'
                    ]
                )
                success_count = len(rows)
                self.write_count += success_count
                
                # Log success with symbols
                symbols_sample = sorted(set([row[1] for row in rows[:5]]))
                self.clickhouse_logger.info(f"SUCCESS | Rows Inserted: {success_count}")
                self.clickhouse_logger.info(f"Symbols: {', '.join(symbols_sample)}"
                                           f"{' ...' if len(rows) > 5 else ''}")
                self.clickhouse_logger.info(f"Total Rows So Far: {self.write_count:,}")
                self.clickhouse_logger.info(f"Unique Symbols: {len(self.symbols_written)}")
            
        except Exception as e:
            self.error_count += len(symbols_data)
            fail_count = len(symbols_data)
            self.logger.error(f"ClickHouse batch insert failed: {e}")
            self.clickhouse_logger.info("=" * 80)
            self.clickhouse_logger.info(f"BATCH INSERT FAILED")
            self.clickhouse_logger.info(f"Error: {str(e)}")
            self.clickhouse_logger.info(f"Failed Rows: {len(symbols_data)}")
            self.clickhouse_logger.info("=" * 80)
        
        return {'success': success_count, 'failed': fail_count}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get ClickHouse operation statistics"""
        return {
            'total_writes': self.write_count,
            'total_errors': self.error_count,
            'unique_symbols': len(self.symbols_written),
            'symbols': sorted(list(self.symbols_written))
        }
    
    def close(self):
        """Close ClickHouse connection"""
        if self.client:
            try:
                # Final summary to ClickHouse log
                self.clickhouse_logger.info("=" * 80)
                self.clickhouse_logger.info("CLICKHOUSE SESSION SUMMARY")
                self.clickhouse_logger.info("=" * 80)
                self.clickhouse_logger.info(f"Total Rows Inserted: {self.write_count:,}")
                self.clickhouse_logger.info(f"Total Errors: {self.error_count}")
                self.clickhouse_logger.info(f"Unique Symbols Processed: {len(self.symbols_written)}")
                
                if self.symbols_written:
                    symbols_list = sorted(list(self.symbols_written))
                    self.clickhouse_logger.info(f"Symbols ({len(symbols_list)}): {', '.join(symbols_list[:10])}"
                                               f"{' ...' if len(symbols_list) > 10 else ''}")
                
                self.clickhouse_logger.info(f"Success Rate: {(self.write_count / (self.write_count + self.error_count) * 100):.2f}%" 
                                           if (self.write_count + self.error_count) > 0 else "N/A")
                self.clickhouse_logger.info("=" * 80)
                self.clickhouse_logger.info("Connection: CLOSED ✓")
                self.clickhouse_logger.info("=" * 80)
                
                self.client.close()
                self.logger.info("ClickHouse connection closed")
            except Exception as e:
                self.logger.error(f"Error closing ClickHouse: {e}")
                self.clickhouse_logger.info(f"Error during close: {str(e)}")