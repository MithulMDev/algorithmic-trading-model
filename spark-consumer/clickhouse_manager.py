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
    

    def _create_signals_table(self):
        """Create signals table if it doesn't exist"""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS signals_generated (
            timestamp DateTime64(3),
            symbol String,
            regime String,
            
            -- OHLCV data
            open Float64,
            high Float64,
            low Float64,
            close Float64,
            volume Float64,
            
            -- Indicators
            sma_10 Float64,
            macd_line Float64,
            macd_signal Float64,
            atr_14 Float64,
            momentum Float64,
            bb_upper Float64,
            bb_lower Float64,
            volume_ratio Float64,
            rsi_14 Float64,
            price_change Float64,
            price_change_pct Float64,
            
            -- Signals
            signal_1_momentum Float64,
            signal_2_macd Float64,
            signal_3_rsi Float64,
            signal_4_bollinger Float64,
            signal_5_volume Float64,
            signal_6_trend_quality Float64,
            
            -- Regime data
            regime_confidence Float64,
            
            -- Aggregated scores
            marker_raw_score Float64,
            marker_adjusted_score Float64,
            marker_final_score Float64,
            
            -- Metadata
            date String,
            time String,
            last_updated DateTime64(3),
            inserted_at DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY (symbol, timestamp)
        """
        
        try:
            self.client.command(create_table_query)
            self.logger.info("ClickHouse table 'signals_generated' verified/created")
            self.clickhouse_logger.info("Table 'signals_generated': READY ✓")
            self.clickhouse_logger.info("Engine: MergeTree | Order: (symbol, timestamp)")
        except Exception as e:
            self.logger.error(f"Failed to create ClickHouse signals table: {e}")
            self.clickhouse_logger.info(f"Signals table creation failed: {str(e)}")
            raise


    def write_signals_batch(self, signals_data: List[Dict]) -> Dict[str, int]:
        """
        Write batch of signal results to ClickHouse
        
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
            self.clickhouse_logger.info("-" * 80)
            self.clickhouse_logger.info(f"SIGNALS BATCH | Processing {len(signals_data)} symbols")
            self.clickhouse_logger.info("-" * 80)
            
            # Ensure signals table exists (first time only)
            if not hasattr(self, '_signals_table_created'):
                self._create_signals_table()
                self._signals_table_created = True
            
            rows = []
            
            # Track first signal for detailed logging
            first_signal_logged = False
            
            for signal_result in signals_data:
                try:
                    # Parse timestamps
                    timestamp = datetime.fromisoformat(signal_result.get('timestamp', signal_result['last_updated']))
                    last_updated = datetime.fromisoformat(signal_result['last_updated'])
                    
                    # Prepare row in the same order as table columns
                    row = [
                        timestamp,
                        str(signal_result['symbol']),
                        str(signal_result['regime']),
                        
                        # OHLCV
                        float(signal_result['open']),
                        float(signal_result['high']),
                        float(signal_result['low']),
                        float(signal_result['close']),
                        float(signal_result['volume']),
                        
                        # Indicators
                        float(signal_result['sma_10']),
                        float(signal_result['macd_line']),
                        float(signal_result['macd_signal']),
                        float(signal_result['atr_14']),
                        float(signal_result['momentum']),
                        float(signal_result['bb_upper']),
                        float(signal_result['bb_lower']),
                        float(signal_result['volume_ratio']),
                        float(signal_result['rsi_14']),
                        float(signal_result['price_change']),
                        float(signal_result['price_change_pct']),
                        
                        # Signals
                        float(signal_result['signal_1_momentum']),
                        float(signal_result['signal_2_macd']),
                        float(signal_result['signal_3_rsi']),
                        float(signal_result['signal_4_bollinger']),
                        float(signal_result['signal_5_volume']),
                        float(signal_result['signal_6_trend_quality']),
                        
                        # Regime
                        float(signal_result['regime_confidence']),
                        
                        # Scores
                        float(signal_result['marker_raw_score']),
                        float(signal_result['marker_adjusted_score']),
                        float(signal_result['marker_final_score']),
                        
                        # Metadata
                        str(signal_result.get('date', '')),
                        str(signal_result.get('time', '')),
                        last_updated
                    ]
                    
                    rows.append(row)
                    
                    # Log first signal details
                    if not first_signal_logged:
                        self.clickhouse_logger.info("Sample Signal Row:")
                        self.clickhouse_logger.info(f"  Symbol: {signal_result['symbol']}")
                        self.clickhouse_logger.info(f"  Regime: {signal_result['regime']}")
                        self.clickhouse_logger.info(f"  Timestamp: {timestamp}")
                        self.clickhouse_logger.info(f"  OHLC: O=${signal_result['open']:.2f} "
                                                f"H=${signal_result['high']:.2f} "
                                                f"L=${signal_result['low']:.2f} "
                                                f"C=${signal_result['close']:.2f}")
                        self.clickhouse_logger.info(f"  Final Score: {signal_result['marker_final_score']:.4f}")
                        self.clickhouse_logger.info(f"  Signals: S1={signal_result['signal_1_momentum']:.2f} "
                                                f"S2={signal_result['signal_2_macd']:.2f} "
                                                f"S3={signal_result['signal_3_rsi']:.2f}")
                        self.clickhouse_logger.info("-" * 80)
                        first_signal_logged = True
                    
                except Exception as e:
                    self.logger.error(f"Error preparing signal row for {signal_result.get('symbol', 'UNKNOWN')}: {e}")
                    self.clickhouse_logger.info(f"ERROR | Symbol: {signal_result.get('symbol', 'UNKNOWN')} | {str(e)[:50]}")
                    fail_count += 1
            
            # Batch insert
            if rows:
                self.clickhouse_logger.info(f"Inserting {len(rows)} signal rows into ClickHouse...")
                
                self.client.insert(
                    'signals_generated',
                    rows,
                    column_names=[
                        'timestamp', 'symbol', 'regime',
                        'open', 'high', 'low', 'close', 'volume',
                        'sma_10', 'macd_line', 'macd_signal', 'atr_14', 'momentum',
                        'bb_upper', 'bb_lower', 'volume_ratio', 'rsi_14',
                        'price_change', 'price_change_pct',
                        'signal_1_momentum', 'signal_2_macd', 'signal_3_rsi',
                        'signal_4_bollinger', 'signal_5_volume', 'signal_6_trend_quality',
                        'regime_confidence',
                        'marker_raw_score', 'marker_adjusted_score', 'marker_final_score',
                        'date', 'time', 'last_updated'
                    ]
                )
                success_count = len(rows)
                self.write_count += success_count
                
                # Log success with symbols
                symbols_sample = sorted(set([row[1] for row in rows[:5]]))
                self.clickhouse_logger.info(f"SUCCESS | Signal Rows Inserted: {success_count}")
                self.clickhouse_logger.info(f"Symbols: {', '.join(symbols_sample)}"
                                        f"{' ...' if len(rows) > 5 else ''}")
                self.clickhouse_logger.info(f"Total Signal Rows So Far: {self.write_count:,}")
            
        except Exception as e:
            self.error_count += len(signals_data)
            fail_count = len(signals_data)
            self.logger.error(f"ClickHouse signals batch insert failed: {e}")
            self.clickhouse_logger.info("=" * 80)
            self.clickhouse_logger.info(f"SIGNALS BATCH INSERT FAILED")
            self.clickhouse_logger.info(f"Error: {str(e)}")
            self.clickhouse_logger.info(f"Failed Rows: {len(signals_data)}")
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