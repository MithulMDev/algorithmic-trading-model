"""
InfluxDB Manager for Time-Series OHLCV Data
Handles batch writes to InfluxDB with verification
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
        self.query_api = None
        self.write_count = 0
        self.error_count = 0
        self.symbols_written = set()
        
        self._connect()
    
    def _connect(self):
        """Establish InfluxDB connection"""
        try:
            self.logger.info(f"Connecting to InfluxDB at {self.config.INFLUX_URL}")
            self.influx_logger.info("=" * 80)
            self.influx_logger.info("INFLUXDB CONNECTION INITIATED")
            self.influx_logger.info("=" * 80)
            self.influx_logger.info(f"URL: {self.config.INFLUX_URL}")
            self.influx_logger.info(f"Organization: {self.config.INFLUX_ORG}")
            self.influx_logger.info(f"Bucket: {self.config.INFLUX_BUCKET}")
            
            self.client = InfluxDBClient(
                url=self.config.INFLUX_URL,
                token=self.config.INFLUX_TOKEN,
                org=self.config.INFLUX_ORG,
                timeout=30_000
            )
            
            # Get write API (synchronous for simplicity)
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            
            # Get query API for verification
            self.query_api = self.client.query_api()
            
            # Test connection
            self.client.ping()
            self.logger.info("InfluxDB connected successfully")
            
            # Verify bucket exists
            self._verify_bucket()
            
            # InfluxDB log header
            self.influx_logger.info("Status: CONNECTED ✓")
            self.influx_logger.info("=" * 80)
            self.influx_logger.info("INFLUXDB WRITE LOG")
            self.influx_logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"InfluxDB connection failed: {e}")
            self.influx_logger.info("Status: FAILED ✗")
            self.influx_logger.info(f"Error: {str(e)}")
            self.influx_logger.info("=" * 80)
            raise
    
    def _verify_bucket(self):
        """Verify bucket exists and is accessible"""
        try:
            buckets_api = self.client.buckets_api()
            bucket = buckets_api.find_bucket_by_name(self.config.INFLUX_BUCKET)
            
            if bucket:
                self.influx_logger.info(f"Bucket '{self.config.INFLUX_BUCKET}' found ✓")
                self.influx_logger.info(f"Bucket ID: {bucket.id}")
            else:
                self.influx_logger.info(f"WARNING: Bucket '{self.config.INFLUX_BUCKET}' not found!")
                self.logger.warning(f"Bucket '{self.config.INFLUX_BUCKET}' not found in InfluxDB")
        except Exception as e:
            self.influx_logger.info(f"Could not verify bucket: {str(e)}")
    
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
            self.influx_logger.info("-" * 80)
            self.influx_logger.info(f"BATCH {batch_id} | Processing {len(symbols_data)} symbols")
            self.influx_logger.info("-" * 80)
            
            points = []
            
            # Track first symbol for detailed logging
            first_symbol_logged = False
            first_timestamp = None
            
            for symbol_name, symbol_data, record_timestamp, kafka_meta in symbols_data:
                try:
                    # Parse timestamp
                    timestamp = datetime.fromisoformat(record_timestamp.replace(' ', 'T'))
                    
                    if first_timestamp is None:
                        first_timestamp = timestamp
                    
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
                    
                    # Log first symbol details
                    if not first_symbol_logged:
                        self.influx_logger.info("Sample Point Details:")
                        self.influx_logger.info(f"  Measurement: stock_prices")
                        self.influx_logger.info(f"  Symbol (tag): {symbol_name}")
                        self.influx_logger.info(f"  Timestamp: {timestamp}")
                        self.influx_logger.info(f"  Timestamp (ISO): {timestamp.isoformat()}")
                        self.influx_logger.info(f"  OHLC: O=${symbol_data.get('open', 0):.2f} "
                                               f"H=${symbol_data.get('high', 0):.2f} "
                                               f"L=${symbol_data.get('low', 0):.2f} "
                                               f"C=${symbol_data.get('close', 0):.2f}")
                        self.influx_logger.info(f"  Volume: {symbol_data.get('volume', 0):,.0f}")
                        self.influx_logger.info(f"  Date/Time: {symbol_data.get('Date', 'N/A')} {symbol_data.get('Time', 'N/A')}")
                        self.influx_logger.info("-" * 80)
                        first_symbol_logged = True
                    
                except Exception as e:
                    self.logger.error(f"Error creating point for {symbol_name}: {e}")
                    self.influx_logger.info(f"ERROR | Symbol: {symbol_name} | {str(e)[:50]}")
                    fail_count += 1
            
            # Batch write all points
            if points:
                self.influx_logger.info(f"Attempting to write {len(points)} points to InfluxDB...")
                self.influx_logger.info(f"Target: Bucket='{self.config.INFLUX_BUCKET}' Org='{self.config.INFLUX_ORG}'")
                
                try:
                    self.write_api.write(
                        bucket=self.config.INFLUX_BUCKET,
                        org=self.config.INFLUX_ORG,
                        record=points
                    )
                    success_count = len(points)
                    self.write_count += success_count
                    
                    # Log success with symbols
                    symbols_sample = sorted(set([p._tags['symbol'] for p in points[:5]]))
                    self.influx_logger.info(f"WRITE COMPLETED ✓")
                    self.influx_logger.info(f"Points Written: {success_count}")
                    self.influx_logger.info(f"Symbols: {', '.join(symbols_sample)}"
                                           f"{' ...' if len(points) > 5 else ''}")
                    
                    # VERIFY: Query to check if data actually exists
                    self.influx_logger.info("-" * 80)
                    self.influx_logger.info("VERIFYING DATA IN INFLUXDB...")
                    self._verify_data_written(first_timestamp, batch_id)
                    
                    self.influx_logger.info(f"Total Points So Far: {self.write_count}")
                    self.influx_logger.info(f"Unique Symbols: {len(self.symbols_written)}")
                    
                except Exception as write_error:
                    self.influx_logger.info(f"WRITE FAILED ✗")
                    self.influx_logger.info(f"Error: {str(write_error)}")
                    self.logger.error(f"InfluxDB write error: {write_error}", exc_info=True)
                    raise
            
        except Exception as e:
            self.error_count += len(symbols_data)
            fail_count = len(symbols_data)
            self.logger.error(f"InfluxDB batch write failed: {e}")
            self.influx_logger.info("=" * 80)
            self.influx_logger.info(f"BATCH WRITE FAILED")
            self.influx_logger.info(f"Error: {str(e)}")
            self.influx_logger.info(f"Failed Symbols: {len(symbols_data)}")
            self.influx_logger.info("=" * 80)
        
        return {'success': success_count, 'failed': fail_count}
    

    def write_signals_batch(self, signals_data: List[Dict]) -> Dict[str, int]:
        """
        Write batch of signal results to InfluxDB
        
        Args:
            signals_data: List of signal result dictionaries from SignalProcessor
            
        Returns:
            Dict with success and failure counts
        """
        success_count = 0
        fail_count = 0
        
        if not signals_data:
            return {'success': 0, 'failed': 0}
        
        try:
            self.influx_logger.info("-" * 80)
            self.influx_logger.info(f"SIGNALS BATCH | Processing {len(signals_data)} symbols")
            self.influx_logger.info("-" * 80)
            
            points = []
            
            # Track first signal for detailed logging
            first_signal_logged = False
            
            for signal_result in signals_data:
                try:
                    # Parse timestamp from last_updated field
                    timestamp = datetime.fromisoformat(signal_result['last_updated'])
                    
                    # Create InfluxDB point for signals_generated measurement
                    point = (
                        Point("signals_generated")
                        .tag("symbol", signal_result['symbol'])
                        .tag("regime", signal_result['regime'])
                        
                        # OHLCV fields
                        .field("open", float(signal_result['open']))
                        .field("high", float(signal_result['high']))
                        .field("low", float(signal_result['low']))
                        .field("close", float(signal_result['close']))
                        .field("volume", float(signal_result['volume']))
                        
                        # Indicator fields
                        .field("sma_10", float(signal_result['sma_10']))
                        .field("macd_line", float(signal_result['macd_line']))
                        .field("macd_signal", float(signal_result['macd_signal']))
                        .field("atr_14", float(signal_result['atr_14']))
                        .field("momentum", float(signal_result['momentum']))
                        .field("bb_upper", float(signal_result['bb_upper']))
                        .field("bb_lower", float(signal_result['bb_lower']))
                        .field("volume_ratio", float(signal_result['volume_ratio']))
                        .field("rsi_14", float(signal_result['rsi_14']))
                        .field("price_change", float(signal_result['price_change']))
                        .field("price_change_pct", float(signal_result['price_change_pct']))
                        
                        # Signal fields
                        .field("signal_1_momentum", float(signal_result['signal_1_momentum']))
                        .field("signal_2_macd", float(signal_result['signal_2_macd']))
                        .field("signal_3_rsi", float(signal_result['signal_3_rsi']))
                        .field("signal_4_bollinger", float(signal_result['signal_4_bollinger']))
                        .field("signal_5_volume", float(signal_result['signal_5_volume']))
                        .field("signal_6_trend_quality", float(signal_result['signal_6_trend_quality']))
                        
                        # Regime fields
                        .field("regime_confidence", float(signal_result['regime_confidence']))
                        
                        # Aggregated score fields
                        .field("marker_raw_score", float(signal_result['marker_raw_score']))
                        .field("marker_adjusted_score", float(signal_result['marker_adjusted_score']))
                        .field("marker_final_score", float(signal_result['marker_final_score']))
                        
                        # Metadata fields
                        .field("date", str(signal_result.get('date', '')))
                        .field("time", str(signal_result.get('time', '')))
                        
                        .time(timestamp)
                    )
                    
                    points.append(point)
                    
                    # Log first signal details
                    if not first_signal_logged:
                        self.influx_logger.info("Sample Signal Point Details:")
                        self.influx_logger.info(f"  Measurement: signals_generated")
                        self.influx_logger.info(f"  Symbol (tag): {signal_result['symbol']}")
                        self.influx_logger.info(f"  Regime (tag): {signal_result['regime']}")
                        self.influx_logger.info(f"  Timestamp: {timestamp.isoformat()}")
                        self.influx_logger.info(f"  OHLC: O=${signal_result['open']:.2f} "
                                            f"H=${signal_result['high']:.2f} "
                                            f"L=${signal_result['low']:.2f} "
                                            f"C=${signal_result['close']:.2f}")
                        self.influx_logger.info(f"  Final Score: {signal_result['marker_final_score']:.4f}")
                        self.influx_logger.info(f"  Signals: S1={signal_result['signal_1_momentum']:.2f} "
                                            f"S2={signal_result['signal_2_macd']:.2f} "
                                            f"S3={signal_result['signal_3_rsi']:.2f}")
                        self.influx_logger.info("-" * 80)
                        first_signal_logged = True
                    
                except Exception as e:
                    self.logger.error(f"Error creating point for {signal_result.get('symbol', 'UNKNOWN')}: {e}")
                    self.influx_logger.info(f"ERROR | Symbol: {signal_result.get('symbol', 'UNKNOWN')} | {str(e)[:50]}")
                    fail_count += 1
            
            # Batch write all points
            if points:
                self.influx_logger.info(f"Writing {len(points)} signal points to InfluxDB...")
                
                try:
                    self.write_api.write(
                        bucket=self.config.INFLUX_BUCKET,
                        org=self.config.INFLUX_ORG,
                        record=points
                    )
                    
                    success_count = len(points)
                    self.write_count += success_count
                    
                    self.influx_logger.info(f"SUCCESS ✓ | Written {success_count} signal points")
                    self.influx_logger.info(f"Total signals written: {self.write_count:,}")
                    
                except Exception as write_error:
                    fail_count = len(points)
                    self.error_count += fail_count
                    self.influx_logger.info(f"WRITE FAILED ✗ | Error: {str(write_error)[:100]}")
                    self.logger.error(f"InfluxDB signals write error: {write_error}", exc_info=True)
                    raise
            
        except Exception as e:
            self.error_count += len(signals_data)
            fail_count = len(signals_data)
            self.logger.error(f"InfluxDB signals batch write failed: {e}")
            self.influx_logger.info("=" * 80)
            self.influx_logger.info(f"SIGNALS BATCH WRITE FAILED")
            self.influx_logger.info(f"Error: {str(e)}")
            self.influx_logger.info(f"Failed Symbols: {len(signals_data)}")
            self.influx_logger.info("=" * 80)
        
        return {'success': success_count, 'failed': fail_count}
        
        
    def _verify_data_written(self, timestamp, batch_id):
        """Verify data was actually written to InfluxDB"""
        try:
            # Calculate time range around the data timestamp
            data_date = timestamp.date()
            start_time = f"{data_date}T00:00:00Z"
            end_time = f"{data_date}T23:59:59Z"
            
            self.influx_logger.info("Running verification query...")
            self.influx_logger.info(f"Checking for data on date: {data_date}")
            
            # Query using the actual data timestamp range
            query = f'''
            from(bucket: "{self.config.INFLUX_BUCKET}")
              |> range(start: {start_time}, stop: {end_time})
              |> filter(fn: (r) => r["_measurement"] == "stock_prices")
              |> count()
            '''
            
            result = self.query_api.query(org=self.config.INFLUX_ORG, query=query)
            
            total_records = 0
            for table in result:
                for record in table.records:
                    total_records += record.get_value()
            
            if total_records > 0:
                self.influx_logger.info(f"VERIFICATION SUCCESS ✓")
                self.influx_logger.info(f"Found {total_records} records in bucket")
            else:
                self.influx_logger.info(f"VERIFICATION WARNING ⚠")
                self.influx_logger.info(f"No records found in bucket for this timestamp")
            
            # Query for sample record with actual timestamp
            sample_query = f'''
            from(bucket: "{self.config.INFLUX_BUCKET}")
              |> range(start: {start_time}, stop: {end_time})
              |> filter(fn: (r) => r["_measurement"] == "stock_prices")
              |> limit(n: 1)
            '''
            
            sample_result = self.query_api.query(org=self.config.INFLUX_ORG, query=sample_query)
            
            if sample_result and len(sample_result) > 0:
                self.influx_logger.info(f"Measurement 'stock_prices' exists ✓")
                # Log first record details
                for table in sample_result:
                    if table.records:
                        first_record = table.records[0]
                        self.influx_logger.info(f"Sample Record from DB:")
                        self.influx_logger.info(f"  Time: {first_record.get_time()}")
                        self.influx_logger.info(f"  Symbol: {first_record.values.get('symbol', 'N/A')}")
                        self.influx_logger.info(f"  Field: {first_record.get_field()}")
                        self.influx_logger.info(f"  Value: {first_record.get_value()}")
                        break
            else:
                self.influx_logger.info(f"No data found for date {data_date}")
            
        except Exception as e:
            self.influx_logger.info(f"Verification query failed: {str(e)}")
            self.logger.error(f"Verification error: {e}", exc_info=True)
    
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
                # Final verification query
                self.influx_logger.info("=" * 80)
                self.influx_logger.info("FINAL DATA VERIFICATION")
                self.influx_logger.info("=" * 80)
                
                try:
                    # Count all records (no time limit for historical data)
                    count_query = f'''
                    from(bucket: "{self.config.INFLUX_BUCKET}")
                      |> range(start: 2017-01-01T00:00:00Z, stop: now())
                      |> filter(fn: (r) => r["_measurement"] == "stock_prices")
                      |> count()
                    '''
                    
                    result = self.query_api.query(org=self.config.INFLUX_ORG, query=count_query)
                    
                    total_in_db = 0
                    for table in result:
                        for record in table.records:
                            total_in_db += record.get_value()
                    
                    self.influx_logger.info(f"Total records in InfluxDB (all time): {total_in_db:,}")
                    
                    if total_in_db > 0:
                        # Get timestamp range
                        range_query = f'''
                        from(bucket: "{self.config.INFLUX_BUCKET}")
                          |> range(start: 2017-01-01T00:00:00Z, stop: now())
                          |> filter(fn: (r) => r["_measurement"] == "stock_prices")
                          |> group()
                          |> first()
                        '''
                        range_result = self.query_api.query(org=self.config.INFLUX_ORG, query=range_query)
                        
                        if range_result and len(range_result) > 0:
                            for table in range_result:
                                if table.records:
                                    first_time = table.records[0].get_time()
                                    self.influx_logger.info(f"Earliest record: {first_time}")
                                    break
                    else:
                        self.influx_logger.info("WARNING: No data found in InfluxDB!")
                        self.influx_logger.info("Possible issues:")
                        self.influx_logger.info("  1. Write operation failed silently")
                        self.influx_logger.info("  2. Writing to wrong bucket/org")
                        self.influx_logger.info("  3. Data not persisting")
                    
                except Exception as query_error:
                    self.influx_logger.info(f"Final verification failed: {str(query_error)}")
                
                # Final summary
                self.influx_logger.info("=" * 80)
                self.influx_logger.info("INFLUXDB SESSION SUMMARY")
                self.influx_logger.info("=" * 80)
                self.influx_logger.info(f"Total Points Written: {self.write_count:,}")
                self.influx_logger.info(f"Total Errors: {self.error_count}")
                self.influx_logger.info(f"Unique Symbols Processed: {len(self.symbols_written)}")
                
                if self.symbols_written:
                    symbols_list = sorted(list(self.symbols_written))
                    self.influx_logger.info(f"Symbols ({len(symbols_list)}): {', '.join(symbols_list[:10])}"
                                           f"{' ...' if len(symbols_list) > 10 else ''}")
                
                self.influx_logger.info(f"Success Rate: {(self.write_count / (self.write_count + self.error_count) * 100):.2f}%" 
                                       if (self.write_count + self.error_count) > 0 else "N/A")
                self.influx_logger.info("=" * 80)
                self.influx_logger.info("Connection: CLOSED ✓")
                self.influx_logger.info("=" * 80)
                
                self.write_api.close()
                self.client.close()
                self.logger.info("InfluxDB connection closed")
            except Exception as e:
                self.logger.error(f"Error closing InfluxDB: {e}")
                self.influx_logger.info(f"Error during close: {str(e)}") 