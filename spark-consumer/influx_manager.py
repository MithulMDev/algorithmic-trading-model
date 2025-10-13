"""
InfluxDB Manager for Time-Series OHLCV Data
Handles batch writes to InfluxDB
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS


class InfluxManager:
    """Manages InfluxDB connections and batch write operations"""
    
    def __init__(self, config, main_logger: logging.Logger, influx_logger: logging.Logger):
        """Initialize InfluxDB manager"""
        self.config = config
        self.logger = main_logger
        self.influx_logger = influx_logger
        self.client: Optional[InfluxDBClient] = None
        self.write_api = None
        self.write_count = 0
        self.error_count = 0
        self.symbols_written = set()
        
        self._connect()
    
    def _connect(self):
        """Establish InfluxDB connection"""
        try:
            self.logger.info(f"Connecting to InfluxDB at {self.config.INFLUX_URL}")
            
            self.client = InfluxDBClient(
                url=self.config.INFLUX_URL,
                token=self.config.INFLUX_TOKEN,
                org=self.config.INFLUX_ORG,
                timeout=30_000
            )
            
            # Get write API (synchronous for simplicity)
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            
            # Test connection
            self.client.ping()
            self.logger.info("InfluxDB connected successfully")
            
            # InfluxDB log header
            self.influx_logger.info("=" * 80)
            self.influx_logger.info("INFLUXDB WRITE LOG")
            self.influx_logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"InfluxDB connection failed: {e}")
            raise
    
    def write_batch(self, symbols_data: List[tuple], batch_id: int) -> Dict[str, int]:
        """
        Write batch of symbols to InfluxDB
        
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
            points = []
            
            for symbol_name, symbol_data, record_timestamp, kafka_meta in symbols_data:
                try:
                    # Parse timestamp
                    timestamp = datetime.fromisoformat(record_timestamp.replace(' ', 'T'))
                    
                    # Create InfluxDB point
                    point = (
                        Point("stock_prices")
                        .tag("symbol", symbol_name)
                        .field("open", float(symbol_data.get('open', 0)))
                        .field("high", float(symbol_data.get('high', 0)))
                        .field("low", float(symbol_data.get('low', 0)))
                        .field("close", float(symbol_data.get('close', 0)))
                        .field("volume", float(symbol_data.get('volume', 0)))
                        .field("date", int(symbol_data.get('Date', 0)))
                        .field("time", str(symbol_data.get('Time', '')))
                        .time(timestamp)
                    )
                    
                    points.append(point)
                    self.symbols_written.add(symbol_name)
                    
                except Exception as e:
                    self.logger.error(f"Error creating point for {symbol_name}: {e}")
                    fail_count += 1
            
            # Batch write all points
            if points:
                self.write_api.write(
                    bucket=self.config.INFLUX_BUCKET,
                    org=self.config.INFLUX_ORG,
                    record=points
                )
                success_count = len(points)
                self.write_count += success_count
                
                # Log to InfluxDB log file
                symbols_sample = sorted(set([p._tags['symbol'] for p in points[:5]]))
                self.influx_logger.info(
                    f"Batch {batch_id} | Points: {success_count} | "
                    f"Symbols: {', '.join(symbols_sample)}"
                    f"{' ...' if len(points) > 5 else ''}"
                )
            
        except Exception as e:
            self.error_count += len(symbols_data)
            fail_count = len(symbols_data)
            self.logger.error(f"InfluxDB batch write failed: {e}")
        
        return {'success': success_count, 'failed': fail_count}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get InfluxDB operation statistics"""
        return {
            'total_writes': self.write_count,
            'total_errors': self.error_count,
            'unique_symbols': len(self.symbols_written),
            'symbols': sorted(list(self.symbols_written))
        }
    
    def close(self):
        """Close InfluxDB connection"""
        if self.client:
            try:
                # Final summary to InfluxDB log
                self.influx_logger.info("=" * 80)
                self.influx_logger.info("INFLUXDB SESSION SUMMARY")
                self.influx_logger.info("=" * 80)
                self.influx_logger.info(f"Total Points Written: {self.write_count}")
                self.influx_logger.info(f"Total Errors: {self.error_count}")
                self.influx_logger.info(f"Unique Symbols: {len(self.symbols_written)}")
                self.influx_logger.info("=" * 80)
                
                self.write_api.close()
                self.client.close()
                self.logger.info("InfluxDB connection closed")
            except Exception as e:
                self.logger.error(f"Error closing InfluxDB: {e}")