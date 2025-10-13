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
            
            # Ensure table exists
            self._create_table()
            
            # ClickHouse log header
            self.clickhouse_logger.info("=" * 80)
            self.clickhouse_logger.info("CLICKHOUSE WRITE LOG")
            self.clickhouse_logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"ClickHouse connection failed: {e}")
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
        except Exception as e:
            self.logger.error(f"Failed to create ClickHouse table: {e}")
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
            rows = []
            
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
                    
                except Exception as e:
                    self.logger.error(f"Error preparing row for {symbol_name}: {e}")
                    fail_count += 1
            
            # Batch insert
            if rows:
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
                
                # Log to ClickHouse log file
                symbols_sample = sorted(set([row[1] for row in rows[:5]]))
                self.clickhouse_logger.info(
                    f"Batch {batch_id} | Rows: {success_count} | "
                    f"Symbols: {', '.join(symbols_sample)}"
                    f"{' ...' if len(rows) > 5 else ''}"
                )
            
        except Exception as e:
            self.error_count += len(symbols_data)
            fail_count = len(symbols_data)
            self.logger.error(f"ClickHouse batch insert failed: {e}")
        
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
                self.clickhouse_logger.info(f"Total Rows Inserted: {self.write_count}")
                self.clickhouse_logger.info(f"Total Errors: {self.error_count}")
                self.clickhouse_logger.info(f"Unique Symbols: {len(self.symbols_written)}")
                self.clickhouse_logger.info("=" * 80)
                
                self.client.close()
                self.logger.info("ClickHouse connection closed")
            except Exception as e:
                self.logger.error(f"Error closing ClickHouse: {e}")