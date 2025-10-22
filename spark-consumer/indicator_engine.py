#!/usr/bin/env python3
"""
Indicator Engine with Redis State Management
Calculates technical indicators on OHLCV data with warm-up period
"""

import json
import logging
import redis
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dateutil.parser import parse as parse_datetime



class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class IndicatorEngine:
    """
    Technical Indicator Engine with Redis-based state management
    Maintains per-symbol history and calculates indicators
    """
    
    def __init__(self, logger, redis_host: str = "redis", redis_port: int = 6379, 
                 redis_db: int = 1, warmup_rows: int = 60):
        """
        Initialize IndicatorEngine
        
        Args:
            redis_host: Redis host
            redis_port: Redis port
            redis_db: Redis database (use DB 1 to separate from main app)
            warmup_rows: Number of rows needed for warm-up per symbol
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.warmup_rows = warmup_rows
        self.min_return_index = 34  # 0-indexed, so 26th row
        
        # Connect to Redis
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=False,  # Handle encoding ourselves
            socket_timeout=5,
            socket_connect_timeout=5
        )
        
        # Test connection
        try:
            self.redis_client.ping()
        except redis.ConnectionError as e:
            raise ConnectionError(f"Cannot connect to Redis at {redis_host}:{redis_port}: {e}")
        
        # Track which symbols have completed initial warm-up
        self.warmed_up_symbols = set()
        
        # Logger
        self.logger = logger or logging.getLogger("IndicatorEngine")
        self.logger.info("=" * 80)
        self.logger.info("INDICATOR ENGINE INITIALIZED")
        self.logger.info("=" * 80)
        self.logger.info(f"Redis: {redis_host}:{redis_port}, DB: {redis_db}")
        self.logger.info(f"Warm-up threshold: {warmup_rows} rows per symbol")
        self.logger.info(f"First return starts from row: {self.min_return_index + 1}")
        self.logger.info("=" * 80)
    
    def _get_redis_key(self, symbol: str) -> str:
        """Get Redis key for symbol history"""
        return f"indicator:history:{symbol}"
    
    def _store_rows_in_redis(self, symbol: str, rows: List[Dict]) -> int:
        """
        Store rows in Redis for a symbol, maintaining sliding window
        Uses Redis pipeline for atomic operations to prevent race conditions
        
        Args:
            symbol: Symbol name
            rows: List of row dictionaries to store
        
        Returns:
            Total count of rows for this symbol after storage
        """
        redis_key = self._get_redis_key(symbol)
        
        # Use pipeline for atomic operations
        pipe = self.redis_client.pipeline()
        
        # Append each row using the PIPELINE (not redis_client directly)
        for row in rows:
            row['_enriched'] = False  # Mark as raw data
            row_json = json.dumps(row, cls=DateTimeEncoder)
            pipe.rpush(redis_key, row_json)  # ← Note: pipe.rpush, not self.redis_client.rpush
        
        # Add trim and count operations to the pipeline
        pipe.ltrim(redis_key, -self.warmup_rows, -1)

        # Get current count
        pipe.llen(redis_key)
        
        # Execute all commands atomically
        results = pipe.execute()
        
        # Return count from llen (last result)
        count = results[-1]
        return count
    
    def _get_history_from_redis(self, symbol: str) -> List[Dict]:
        """
        Retrieve full history for a symbol from Redis
        
        Args:
            symbol: Symbol name
        
        Returns:
            List of historical rows (max warmup_rows)
        """
        redis_key = self._get_redis_key(symbol)
        
        # Get all rows
        rows_bytes = self.redis_client.lrange(redis_key, 0, -1)
        
        # Decode JSON
        rows = []
        for row_bytes in rows_bytes:
            row = json.loads(row_bytes.decode('utf-8'))
            rows.append(row)
        
        return rows
    
    def _update_enriched_data_in_redis(self, symbol: str, enriched_rows: List[Dict], start_idx: int):
        """
        Update the stored Redis data with enriched indicator values
        
        Args:
            symbol: Symbol name
            enriched_rows: List of enriched row dictionaries to update in Redis
            start_idx: Starting index in the Redis list to update
        """
        redis_key = self._get_redis_key(symbol)
        
        # Use pipeline for atomic updates
        pipe = self.redis_client.pipeline()
        
        for i, enriched_row in enumerate(enriched_rows):
            # Calculate the Redis list index
            redis_index = start_idx + i
            
            # Convert enriched row to JSON
            row_json = json.dumps(enriched_row, cls=DateTimeEncoder)
            
            # Update the specific index in the Redis list
            pipe.lset(redis_key, redis_index, row_json)
        
        # Execute all updates atomically
        pipe.execute()
        
        self.logger.info(f"  {symbol}: Updated {len(enriched_rows)} rows with enriched data in Redis")
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators on a DataFrame using vectorized operations
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume
        
        Returns:
            DataFrame with added indicator columns
        """
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Extract price and volume series
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # ============================================================
        # Simple Moving Averages
        # ============================================================
        df['sma_5'] = close.rolling(window=5, min_periods=5).mean()
        df['sma_10'] = close.rolling(window=10, min_periods=10).mean()
        
        # ============================================================
        # Exponential Moving Averages
        # ============================================================
        df['ema_7'] = close.ewm(span=7, adjust=False, min_periods=7).mean()
        df['ema_11'] = close.ewm(span=11, adjust=False, min_periods=11).mean()
        
        # ============================================================
        # MACD (Moving Average Convergence Divergence)
        # Fast: 12, Slow: 26, Signal: 9
        # ============================================================
        ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
        df['macd_line'] = ema_12 - ema_26
        df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False, min_periods=9).mean()
        
        # ============================================================
        # ATR (Average True Range) - 14 period
        # ============================================================
        high_low = high - low
        high_close = np.abs(high - close.shift(1))
        low_close = np.abs(low - close.shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = true_range.rolling(window=14, min_periods=14).mean()
        
        # ============================================================
        # Momentum: (current - sma_10) / sma_10
        # ============================================================
        df['momentum'] = (close - df['sma_10']) / df['sma_10']
        
        # ============================================================
        # Bollinger Bands - 20 period, 2 std dev
        # ============================================================
        sma_20 = close.rolling(window=20, min_periods=20).mean()
        std_20 = close.rolling(window=20, min_periods=20).std()
        df['bb_upper'] = sma_20 + (2 * std_20)
        df['bb_lower'] = sma_20 - (2 * std_20)
        
        # ============================================================
        # Volume Moving Average - 10 period
        # ============================================================
        df['vma_10'] = volume.rolling(window=10, min_periods=10).mean()
        
        # ============================================================
        # Volume Ratio: current_volume / vma_10
        # ============================================================
        df['volume_ratio'] = volume / df['vma_10']
        
        # ============================================================
        # RSI (Relative Strength Index) - 14 period
        # ============================================================
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        
        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # ============================================================
        # Price Change (from previous close)
        # ============================================================
        df['price_change'] = close.diff()
        df['price_change_pct'] = close.pct_change() * 100
        
        return df
    
    def process_batch(self, rows: List[Dict]) -> Tuple[bool, List[Dict]]:
        """
        Process a batch of rows from Spark
        
        Workflow:
        1. During warm-up (< 60 rows per symbol):
           - Store rows in Redis
           - Return (False, [])
        
        2. First ready return (exactly at 60 rows):
           - Return rows 35-60 for each symbol (5 rows per symbol)
        
        3. Subsequent returns (> 60 rows):
           - Return enriched versions of all new input rows
        
        Args:
            rows: List of dictionaries with OHLCV data
                  Required fields: symbol, timestamp, open, high, low, close, volume,
                                  date, time, offset, partition, kafka_timestamp
        
        Returns:
            Tuple of (ready: bool, enriched_rows: List[Dict])
            - ready=False: Still warming up, enriched_rows is empty
            - ready=True: All symbols ready, enriched_rows contains indicator data
        """
        if not rows:
            return (False, [])
        
        # ============================================================
        # Group rows by symbol
        # ============================================================
        symbol_groups = {}
        for row in rows:
            symbol = row['symbol']
            if symbol not in symbol_groups:
                symbol_groups[symbol] = []
            symbol_groups[symbol].append(row)
        
        self.logger.info("-" * 80)
        self.logger.info(f"Received batch: {len(rows)} rows across {len(symbol_groups)} symbols")
        
        # ============================================================
        # Check state before storing
        # ============================================================
        symbol_states_before = {}
        for symbol in symbol_groups.keys():
            count = self.redis_client.llen(self._get_redis_key(symbol))
            symbol_states_before[symbol] = count
        
        # ============================================================
        # Store all new rows in Redis
        # ============================================================
        symbol_states_after = {}
        for symbol, symbol_rows in symbol_groups.items():
            total_count = self._store_rows_in_redis(symbol, symbol_rows)
            symbol_states_after[symbol] = total_count
        
        # ============================================================
        # Check if ALL symbols have reached warm-up threshold
        # ============================================================
        all_symbols_ready = True
        for symbol, count_after in symbol_states_after.items():
            if count_after < self.warmup_rows:
                all_symbols_ready = False
                self.logger.info(
                    f"  {symbol:<10} | {count_after:>2}/{self.warmup_rows} rows | "
                    f"Status: WARMING UP"
                )
            else:
                self.logger.info(
                    f"  {symbol:<10} | {count_after:>2}/{self.warmup_rows} rows | "
                    f"Status: READY"
                )
        
        # ============================================================
        # If not all symbols ready, return False
        # ============================================================
        if not all_symbols_ready:
            self.logger.info("=" * 80)
            self.logger.info("WARM-UP PHASE: Waiting for all symbols to reach threshold")
            self.logger.info("=" * 80)
            return (False, [])
        
        # ============================================================
        # ALL SYMBOLS READY - Calculate indicators
        # ============================================================
        self.logger.info("-" * 80)
        self.logger.info("ALL SYMBOLS READY - Calculating indicators...")
        self.logger.info("-" * 80)
        
        enriched_rows = []
        
        for symbol, symbol_rows in symbol_groups.items():
            # Get full history from Redis (exactly warmup_rows due to LTRIM)
            full_history = self._get_history_from_redis(symbol)
            
            if len(full_history) < self.warmup_rows:
                self.logger.warning(
                    f"  {symbol}: Insufficient history ({len(full_history)} rows) - SKIPPING"
                )
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame(full_history)
            
            # Calculate indicators on full history
            df_enriched = self._calculate_indicators(df)
            
            # ============================================================
            # Determine which rows to return
            # ============================================================
            is_first_ready = symbol not in self.warmed_up_symbols
            
            if is_first_ready:
                # First time ready: return rows from index 34 onwards (rows 35-60)
                start_idx = self.min_return_index
                self.warmed_up_symbols.add(symbol)
                num_returning = len(df_enriched) - start_idx
                self.logger.info(
                    f"  {symbol}: FIRST READY - Returning rows {start_idx+1} to {len(df_enriched)} "
                    f"({num_returning} rows)"
                )
            else:
                # Already warmed up: return only the new rows from this batch
                num_new_rows = len(symbol_rows)
                start_idx = len(df_enriched) - num_new_rows
                self.logger.info(
                    f"  {symbol}: Returning {num_new_rows} new row(s)"
                )
            
            # ============================================================
            # Extract and format rows to return
            # ============================================================
            symbol_enriched_rows = []  # Track enriched rows for this symbol
            for idx in range(start_idx, len(df_enriched)):
                enriched_row = df_enriched.iloc[idx]
                original_row = full_history[idx]

                # Validate timestamp
                try:
                    if isinstance(original_row['timestamp'], str):
                        parse_datetime(original_row['timestamp'])
                except:
                    self.logger.warning(f"Invalid timestamp: {original_row['timestamp']}")
                
                # Build output row with all fields
                output_row = {
                    # Original fields
                    'symbol': symbol,
                    'timestamp': original_row['timestamp'],
                    'open': original_row['open'],
                    'high': original_row['high'],
                    'low': original_row['low'],
                    'close': original_row['close'],
                    'volume': original_row['volume'],
                    'date': original_row['date'],
                    'time': original_row['time'],
                    'offset': original_row['offset'],
                    'partition': original_row['partition'],
                    'kafka_timestamp': original_row['kafka_timestamp'],
                    
                    # Technical Indicators
                    'sma_5': enriched_row.get('sma_5'),
                    'sma_10': enriched_row.get('sma_10'),
                    'ema_7': enriched_row.get('ema_7'),
                    'ema_11': enriched_row.get('ema_11'),
                    'macd_line': enriched_row.get('macd_line'),
                    'macd_signal': enriched_row.get('macd_signal'),
                    'atr_14': enriched_row.get('atr_14'),
                    'momentum': enriched_row.get('momentum'),
                    'bb_upper': enriched_row.get('bb_upper'),
                    'bb_lower': enriched_row.get('bb_lower'),
                    'vma_10': enriched_row.get('vma_10'),
                    'volume_ratio': enriched_row.get('volume_ratio'),
                    'rsi_14': enriched_row.get('rsi_14'),
                    'price_change': enriched_row.get('price_change'),
                    'price_change_pct': enriched_row.get('price_change_pct'),

                    # Enrichment marker
                    '_enriched': True,
                }
                
                # Convert NaN and Inf to None for JSON compatibility
                for key, value in output_row.items():
                    if isinstance(value, (float, np.floating)):
                        if np.isnan(value) or np.isinf(value):
                            output_row[key] = None

                symbol_enriched_rows.append(output_row)
                enriched_rows.append(output_row)
            
            # ============================================================
            # Update Redis with enriched data for this symbol
            # ============================================================
            if symbol_enriched_rows:
                self._update_enriched_data_in_redis(symbol, symbol_enriched_rows, start_idx)
        
        self.logger.info("=" * 80)
        self.logger.info(f"RETURNING {len(enriched_rows)} ENRICHED ROWS")
        self.logger.info("=" * 80)
        
        return (True, enriched_rows)
    
    def close(self):
        """Close Redis connection and cleanup"""
        try:
            if self.redis_client:
                self.redis_client.close()
                self.logger.info("IndicatorEngine closed - Redis connection terminated")
        except Exception as e:
            self.logger.error(f"Error closing IndicatorEngine: {e}")