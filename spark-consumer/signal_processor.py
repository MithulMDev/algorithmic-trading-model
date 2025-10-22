#!/usr/bin/env python3
"""
HFT Signal Generator - Production-Ready
Generates trading signals based on OHLCV + indicators data from Redis
Runs every 2 seconds, processes symbols in parallel
"""

import json
import logging
import time
import sys
import redis
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration with sensible defaults"""
    REDIS_HOST = "redis"
    REDIS_PORT = 6379
    REDIS_DB = 1  # Same as indicator engine
    
    # Processing
    PROCESS_INTERVAL = 2  # seconds
    PARALLEL_WORKERS = 2
    MIN_REQUIRED_ROWS = 60
    
    # Logging
    LOG_DIR = "/app/logs/signals"
    LOG_LEVEL = "INFO"
    
    # Safety Constants
    EPSILON_SMALL = 1e-8
    EPSILON_VOLUME = 1e-6
    EPSILON_PERCENT = 0.01
    MIN_VOLUME_THRESHOLD = 100


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(config: Config) -> logging.Logger:
    """Set up logging for the signal generator"""
    log_dir = Path(config.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"signals_{timestamp}.log"
    
    logger = logging.getLogger("SignalGenerator")
    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-5s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(message)s',
        datefmt='%H:%M:%S'
    ))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("=" * 100)
    logger.info("HFT SIGNAL GENERATOR STARTED")
    logger.info("=" * 100)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Redis: {config.REDIS_HOST}:{config.REDIS_PORT}, DB: {config.REDIS_DB}")
    logger.info(f"Processing interval: {config.PROCESS_INTERVAL}s with {config.PARALLEL_WORKERS} workers")
    logger.info("=" * 100)
    
    return logger


# ============================================================================
# Signal Calculation Engine
# ============================================================================

class SignalCalculator:
    """Calculates all 6 signals according to HFT specification"""
    
    def __init__(self, config: Config):
        self.config = config
        self.eps_small = config.EPSILON_SMALL
        self.eps_volume = config.EPSILON_VOLUME
        self.eps_percent = config.EPSILON_PERCENT
    
    def calculate_signal_1_momentum(self, df: pd.DataFrame) -> float:
        """Signal 1: Directional Momentum Strength"""
        if len(df) < 10:
            return 0.0
        
        try:
            # Get last 10 rows
            window = df.tail(10).copy()
            
            # Calculate momentum values
            momentum = (window['close'] - window['sma_10']) / (window['sma_10'] + self.eps_percent)
            
            # Split into early and recent
            momentum_early = momentum.iloc[:5].mean()
            momentum_recent = momentum.iloc[5:].mean()
            
            # Calculate acceleration
            momentum_delta = momentum_recent - momentum_early
            momentum_volatility = momentum.std() + self.eps_small
            
            # Z-score
            momentum_z = momentum_delta / momentum_volatility
            
            # Convert to [-1, +1] using tanh
            signal = np.tanh(momentum_z / 1.5)
            
            return float(np.clip(signal, -1, 1))
        except Exception:
            return 0.0
    
    def calculate_signal_2_macd(self, df: pd.DataFrame) -> float:
        """Signal 2: MACD Trend Alignment"""
        if len(df) < 15:
            return 0.0
        
        try:
            # Current and 10s ago
            macd_current = df['macd_line'].iloc[-1]
            macd_10s_ago = df['macd_line'].iloc[-10] if len(df) >= 10 else df['macd_line'].iloc[0]
            
            macd_delta = macd_current - macd_10s_ago
            
            # Normalize by recent range
            window_15 = df.tail(15)
            macd_range = window_15['macd_line'].max() - window_15['macd_line'].min()
            macd_range = max(macd_range, self.eps_small)
            
            macd_velocity = macd_delta / macd_range
            
            # MACD-Signal relationship
            macd_signal_diff = macd_current - df['macd_signal'].iloc[-1]
            macd_signal_range = abs(window_15['macd_line'] - window_15['macd_signal']).max()
            macd_position = macd_signal_diff / (macd_signal_range + self.eps_small)
            
            # Combine
            macd_combined = 0.6 * macd_velocity + 0.4 * macd_position
            
            return float(np.clip(macd_combined, -1, 1))
        except Exception:
            return 0.0
    
    def calculate_signal_3_rsi(self, df: pd.DataFrame) -> float:
        """Signal 3: RSI Mean Reversion with Momentum Filter"""
        if len(df) < 15:
            return 0.0
        
        try:
            rsi_current = df['rsi_14'].iloc[-1]
            rsi_5s_ago = df['rsi_14'].iloc[-5] if len(df) >= 5 else rsi_current
            rsi_10s_ago = df['rsi_14'].iloc[-10] if len(df) >= 10 else rsi_current
            
            # Position score
            if rsi_current < 20:
                rsi_extreme_score = 1.0
            elif rsi_current < 30:
                rsi_extreme_score = 0.7
            elif rsi_current < 40:
                rsi_extreme_score = 0.3
            elif rsi_current > 80:
                rsi_extreme_score = -1.0
            elif rsi_current > 70:
                rsi_extreme_score = -0.7
            elif rsi_current > 60:
                rsi_extreme_score = -0.3
            else:
                rsi_extreme_score = 0.0
            
            # Reversal detection
            rsi_change = rsi_current - rsi_5s_ago
            reversal_boost = 0.0
            
            if rsi_10s_ago < 30 and rsi_5s_ago < 35 and rsi_current > 35 and rsi_change > 0:
                reversal_boost = 0.5
            elif rsi_10s_ago > 70 and rsi_5s_ago > 65 and rsi_current < 65 and rsi_change < 0:
                reversal_boost = -0.5
            
            # Momentum filter
            momentum_current = df['momentum'].iloc[-1]
            momentum_filter = 0.0
            if abs(momentum_current) > 0.02:
                momentum_filter = np.sign(momentum_current) * 0.3
            
            # Combine
            rsi_raw = rsi_extreme_score + reversal_boost + momentum_filter
            
            return float(np.clip(rsi_raw, -1, 1))
        except Exception:
            return 0.0
    
    def calculate_signal_4_bollinger(self, df: pd.DataFrame) -> float:
        """Signal 4: Bollinger Band Dynamics"""
        if len(df) < 10:
            return 0.0
        
        try:
            close_current = df['close'].iloc[-1]
            bb_upper = df['bb_upper'].iloc[-1]
            bb_lower = df['bb_lower'].iloc[-1]
            
            bb_width = (bb_upper - bb_lower) + self.eps_small
            percentB = (close_current - bb_lower) / bb_width
            percentB = np.clip(percentB, -0.1, 1.1)
            
            # Velocity to bands (5s ago)
            if len(df) >= 5:
                close_5s_ago = df['close'].iloc[-5]
                bb_upper_5s_ago = df['bb_upper'].iloc[-5]
                bb_lower_5s_ago = df['bb_lower'].iloc[-5]
                bb_width_5s_ago = bb_upper_5s_ago - bb_lower_5s_ago
                
                distance_to_upper = bb_upper - close_current
                distance_to_lower = close_current - bb_lower
                distance_to_upper_5s = bb_upper_5s_ago - close_5s_ago
                distance_to_lower_5s = close_5s_ago - bb_lower_5s_ago
                
                velocity_to_upper = (distance_to_upper_5s - distance_to_upper) / (bb_width + self.eps_small)
                velocity_to_lower = (distance_to_lower_5s - distance_to_lower) / (bb_width + self.eps_small)
                
                # Width change
                width_change = (bb_width - bb_width_5s_ago) / (bb_width_5s_ago + self.eps_small)
            else:
                velocity_to_upper = 0
                velocity_to_lower = 0
                width_change = 0
            
            # Positional logic
            position_signal = 0.0
            if percentB > 0.95:
                if velocity_to_upper > 0.05:
                    position_signal = 0.8
                elif velocity_to_upper < -0.05:
                    position_signal = -0.7
            elif percentB < 0.05:
                if velocity_to_lower > 0.05:
                    position_signal = -0.8
                elif velocity_to_lower < -0.05:
                    position_signal = 0.7
            elif 0.45 <= percentB <= 0.55:
                position_signal = 0.0
            else:
                position_signal = (percentB - 0.5) * 0.5
            
            # Dynamic adjustment
            if width_change > 0.1:
                dynamic_adjustment = 1.2 if abs(position_signal) > 0.5 else 1.0
            elif width_change < -0.1:
                dynamic_adjustment = 0.7
            else:
                dynamic_adjustment = 1.0
            
            bb_raw = position_signal * dynamic_adjustment
            
            return float(np.clip(bb_raw, -1, 1))
        except Exception:
            return 0.0
    
    def calculate_signal_5_volume(self, df: pd.DataFrame) -> float:
        """Signal 5: Volume-Price Conviction"""
        if len(df) < 15:
            return 0.0
        
        try:
            window = df.tail(10).copy()
            
            # Price changes
            price_changes = window['price_change_pct'].values
            net_price_change = np.sum(price_changes)
            price_direction = np.sign(net_price_change)
            price_magnitude = abs(net_price_change)
            
            # Volume surge
            current_volume_ratio = df['volume_ratio'].iloc[-1]
            baseline_volume_ratio = df['volume_ratio'].tail(15).median()
            baseline_volume_ratio = max(baseline_volume_ratio, 0.5)
            
            volume_surge = (current_volume_ratio / baseline_volume_ratio) - 1.0
            
            # Volume consistency
            high_volume_count = (window['volume_ratio'] > 1.2).sum()
            volume_consistency = high_volume_count / 10.0
            
            # Price-volume alignment
            alignment_raw = price_magnitude * volume_surge * price_direction
            
            if volume_consistency > 0.6:
                alignment_adjusted = alignment_raw * 1.3
            else:
                alignment_adjusted = alignment_raw
            
            signal = np.tanh(alignment_adjusted * 2.0)
            
            return float(np.clip(signal, -1, 1))
        except Exception:
            return 0.0
    
    def calculate_signal_6_trend_quality(self, df: pd.DataFrame) -> float:
        """Signal 6: Trend Quality & Efficiency"""
        if len(df) < 10:
            return 0.0
        
        try:
            window = df.tail(10).copy()
            
            # Directional consistency
            directions = np.sign(window['price_change'].values)
            mode_direction = Counter(directions).most_common(1)[0][0]
            consistent_count = (directions == mode_direction).sum()
            directional_consistency = consistent_count / 10.0
            
            # Movement efficiency
            net_move = window['close'].iloc[-1] - window['close'].iloc[0]
            gross_move = window['price_change'].abs().sum()
            movement_efficiency = abs(net_move) / (gross_move + self.eps_small)
            
            # Magnitude consistency
            price_change_std = window['price_change'].abs().std()
            price_change_mean = window['price_change'].abs().mean()
            magnitude_consistency = 1.0 - (price_change_std / (price_change_mean + self.eps_small))
            magnitude_consistency = np.clip(magnitude_consistency, 0, 1)
            
            # Aggregate quality
            trend_quality = (directional_consistency + movement_efficiency + magnitude_consistency) / 3.0
            
            # Apply direction
            net_direction = np.sign(net_move)
            quality_raw = trend_quality * net_direction * 2.0 - 1.0
            
            return float(np.clip(quality_raw, -1, 1))
        except Exception:
            return 0.0


# ============================================================================
# Regime Detection
# ============================================================================

class RegimeDetector:
    """Detects market volatility regime"""
    
    def __init__(self, config: Config):
        self.config = config
        self.eps_small = config.EPSILON_SMALL
    
    def detect_regime(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Detect volatility regime
        
        Returns:
            Tuple of (regime_name, confidence)
        """
        if len(df) < 60:
            return ("NEUTRAL", 0.5)
        
        try:
            current_atr = df['atr_14'].iloc[-1]
            atr_window_60s = df['atr_14'].tail(60)
            
            # Calculate percentile
            atr_percentile = (atr_window_60s < current_atr).sum() / len(atr_window_60s) * 100
            
            # ATR rate of change
            atr_5s_ago = df['atr_14'].iloc[-5] if len(df) >= 5 else current_atr
            atr_change_rate = (current_atr - atr_5s_ago) / (atr_5s_ago + self.eps_small)
            
            # Determine regime
            if atr_percentile > 75 and atr_change_rate > 0.15:
                regime = "EXPLOSIVE"
                confidence = (atr_percentile - 75) / 25
            elif atr_percentile > 55:
                regime = "TRENDING"
                confidence = (atr_percentile - 55) / 45
            elif atr_percentile < 35:
                regime = "MEAN_REVERSION"
                confidence = (35 - atr_percentile) / 35
            else:
                regime = "NEUTRAL"
                confidence = 0.5
            
            return (regime, float(confidence))
        except Exception:
            return ("NEUTRAL", 0.5)


# ============================================================================
# Signal Aggregation
# ============================================================================

class SignalAggregator:
    """Aggregates signals with regime-specific weighting"""
    
    REGIME_WEIGHTS = {
        "EXPLOSIVE": {
            "signal_1": 0.30,
            "signal_6": 0.25,
            "signal_5": 0.20,
            "signal_2": 0.15,
            "signal_4": 0.07,
            "signal_3": 0.03,
        },
        "TRENDING": {
            "signal_1": 0.25,
            "signal_2": 0.20,
            "signal_6": 0.20,
            "signal_5": 0.17,
            "signal_4": 0.10,
            "signal_3": 0.08,
        },
        "NEUTRAL": {
            "signal_1": 0.1667,
            "signal_2": 0.1667,
            "signal_3": 0.1667,
            "signal_4": 0.1667,
            "signal_5": 0.1667,
            "signal_6": 0.1665,
        },
        "MEAN_REVERSION": {
            "signal_3": 0.32,
            "signal_4": 0.28,
            "signal_2": 0.15,
            "signal_5": 0.12,
            "signal_1": 0.08,
            "signal_6": 0.05,
        },
    }
    
    def __init__(self, config: Config):
        self.config = config
        self.eps_volume = config.EPSILON_VOLUME
    
    def aggregate_signals(self, signals: Dict[str, float], regime: str, 
                         df: pd.DataFrame) -> Tuple[float, float, float]:
        """
        Aggregate signals with regime weights and volume confidence
        
        Returns:
            Tuple of (raw_score, adjusted_score, final_score)
        """
        try:
            # Get weights for regime
            weights = self.REGIME_WEIGHTS.get(regime, self.REGIME_WEIGHTS["NEUTRAL"])
            
            # Calculate weighted raw score
            raw_score = sum(signals[key] * weights[key] for key in signals.keys())
            
            # Volume-based confidence adjustment
            current_volume_ratio = df['volume_ratio'].iloc[-1]
            volume_mean_60s = df['volume_ratio'].tail(60).mean()
            volume_std_60s = df['volume_ratio'].tail(60).std() + self.eps_volume
            
            volume_z = (current_volume_ratio - volume_mean_60s) / volume_std_60s
            
            # Determine confidence multiplier
            if volume_z > 2.0:
                confidence = 1.4
            elif volume_z > 1.0:
                confidence = 1.2
            elif volume_z > 0:
                confidence = 1.05
            elif volume_z < -2.0:
                confidence = 0.4
            elif volume_z < -1.0:
                confidence = 0.6
            else:
                confidence = 0.85
            
            # Apply confidence to magnitude (inverted exponent to fix amplification logic)
            # High confidence (>1) should amplify → use 1/confidence as exponent
            # Low confidence (<1) should dampen → use 1/confidence as exponent
            adjusted_score = np.sign(raw_score) * (abs(raw_score) ** (1.0 / confidence))
            
            # Convert to [0, 2] scale
            final_score = adjusted_score + 1.0
            
            return (float(raw_score), float(adjusted_score), float(final_score))
        except Exception:
            return (0.0, 0.0, 1.0)


# ============================================================================
# Main Signal Processor
# ============================================================================

class SignalProcessor:
    """Main processor for generating signals for a single symbol"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.signal_calc = SignalCalculator(config)
        self.regime_detector = RegimeDetector(config)
        self.aggregator = SignalAggregator(config)
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data quality"""
        required_cols = [
            'close', 'volume', 'sma_10', 'macd_line', 'macd_signal',
            'atr_14', 'momentum', 'bb_upper', 'bb_lower', 'volume_ratio',
            'rsi_14', 'price_change', 'price_change_pct'
        ]
        
        for col in required_cols:
            if col not in df.columns:
                self.logger.error(f"Missing column: {col}")
                return False
            
            if df[col].isna().all():
                self.logger.error(f"All NaN values in column: {col}")
                return False
        
        # Check volume threshold
        if df['volume'].iloc[-1] < self.config.MIN_VOLUME_THRESHOLD:
            self.logger.warning(f"Low volume: {df['volume'].iloc[-1]}")
        
        # Check ATR
        if df['atr_14'].iloc[-1] <= 0:
            self.logger.warning("Invalid ATR (<=0)")
        
        # Check Bollinger bands
        if df['bb_upper'].iloc[-1] <= df['bb_lower'].iloc[-1]:
            self.logger.warning("Invalid Bollinger bands")
        
        # Check RSI
        if not (0 <= df['rsi_14'].iloc[-1] <= 100):
            self.logger.warning(f"Invalid RSI: {df['rsi_14'].iloc[-1]}")
        
        return True
    
    def process_symbol(self, symbol: str, history: List[Dict]) -> Optional[Dict]:
        """Process a single symbol and generate signals"""
        try:
            if len(history) < self.config.MIN_REQUIRED_ROWS:
                self.logger.warning(
                    f"{symbol}: Insufficient data ({len(history)} rows, need {self.config.MIN_REQUIRED_ROWS})"
                )
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(history)
            
            # Validate data
            if not self.validate_data(df):
                self.logger.warning(f"{symbol}: Data validation failed")
                return None
            
            # Calculate all 6 signals
            signals = {
                'signal_1': self.signal_calc.calculate_signal_1_momentum(df),
                'signal_2': self.signal_calc.calculate_signal_2_macd(df),
                'signal_3': self.signal_calc.calculate_signal_3_rsi(df),
                'signal_4': self.signal_calc.calculate_signal_4_bollinger(df),
                'signal_5': self.signal_calc.calculate_signal_5_volume(df),
                'signal_6': self.signal_calc.calculate_signal_6_trend_quality(df),
            }
            
            # Detect regime
            regime, regime_confidence = self.regime_detector.detect_regime(df)
            
            # Aggregate signals
            raw_score, adjusted_score, final_score = self.aggregator.aggregate_signals(
                signals, regime, df
            )
            
            # Get latest row data
            latest = df.iloc[-1]
            
            # Build result
            result = {
                # Metadata
                'symbol': symbol,
                'timestamp': latest.get('timestamp', ''),
                'date': latest.get('date', ''),
                'time': latest.get('time', ''),
                'last_updated': datetime.now().isoformat(),
                
                # OHLCV
                'open': float(latest['open']),
                'high': float(latest['high']),
                'low': float(latest['low']),
                'close': float(latest['close']),
                'volume': float(latest['volume']),
                
                # Indicators
                'sma_10': float(latest['sma_10']),
                'macd_line': float(latest['macd_line']),
                'macd_signal': float(latest['macd_signal']),
                'atr_14': float(latest['atr_14']),
                'momentum': float(latest['momentum']),
                'bb_upper': float(latest['bb_upper']),
                'bb_lower': float(latest['bb_lower']),
                'volume_ratio': float(latest['volume_ratio']),
                'rsi_14': float(latest['rsi_14']),
                'price_change': float(latest['price_change']),
                'price_change_pct': float(latest['price_change_pct']),
                
                # Signals
                'signal_1_momentum': signals['signal_1'],
                'signal_2_macd': signals['signal_2'],
                'signal_3_rsi': signals['signal_3'],
                'signal_4_bollinger': signals['signal_4'],
                'signal_5_volume': signals['signal_5'],
                'signal_6_trend_quality': signals['signal_6'],
                
                # Regime
                'regime': regime,
                'regime_confidence': regime_confidence,
                
                # Aggregated scores
                'marker_raw_score': raw_score,
                'marker_adjusted_score': adjusted_score,
                'marker_final_score': final_score,
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"{symbol}: Error processing - {e}", exc_info=True)
            return None


# ============================================================================
# HFT Signal Generator
# ============================================================================

class HFTSignalGenerator:
    """Main HFT Signal Generator"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.redis_client = None
        self.processor = SignalProcessor(config, logger)
        self.iteration = 0
        
        self._connect_redis()
    
    def _connect_redis(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.REDIS_HOST,
                port=self.config.REDIS_PORT,
                db=self.config.REDIS_DB,
                decode_responses=False,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            self.logger.info("Redis connected successfully")
        except redis.ConnectionError as e:
            self.logger.error(f"Redis connection failed: {e}")
            raise
    
    def discover_symbols(self) -> List[str]:
        """Discover all symbols from Redis"""
        try:
            keys = self.redis_client.keys("indicator:history:*")
            symbols = [key.decode('utf-8').split(":")[-1] for key in keys]
            return sorted(symbols)
        except Exception as e:
            self.logger.error(f"Error discovering symbols: {e}")
            return []
    
    def fetch_symbol_history(self, symbol: str) -> Optional[List[Dict]]:
        """Fetch historical data for a symbol"""
        try:
            redis_key = f"indicator:history:{symbol}"
            rows_bytes = self.redis_client.lrange(redis_key, -60, -1)
            
            if not rows_bytes:
                return None
            
            rows = []
            for row_bytes in rows_bytes:
                row = json.loads(row_bytes.decode('utf-8'))
                rows.append(row)
            
            return rows
        except Exception as e:
            self.logger.error(f"{symbol}: Error fetching history - {e}")
            return None
    
    def write_signals_to_redis(self, symbol: str, result: Dict) -> bool:
        """Write signal results to Redis"""
        try:
            redis_key = f"signals:latest:{symbol}"
            
            # Convert all values to strings for Redis hash
            hash_data = {k: str(v) for k, v in result.items()}
            
            self.redis_client.hset(redis_key, mapping=hash_data)
            return True
        except Exception as e:
            self.logger.error(f"{symbol}: Error writing to Redis - {e}")
            return False
    
    def print_summary_table(self, results: List[Dict]):
        """Print summary table of all signals"""
        if not results:
            self.logger.info("No results to display")
            return
        
        self.logger.info("=" * 140)
        self.logger.info(f"SIGNAL SUMMARY - Iteration {self.iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 140)
        
        # Header
        header = (
            f"{'Symbol':<10} | {'Close':>10} | {'Volume':>12} | "
            f"{'Regime':<14} | {'Final Score':>11} | {'Signal':<12}"
        )
        self.logger.info(header)
        self.logger.info("-" * 140)
        
        # Sort by final score descending
        sorted_results = sorted(results, key=lambda x: x['marker_final_score'], reverse=True)
        
        for result in sorted_results:
            final_score = result['marker_final_score']
            
            # Determine signal label
            if final_score >= 1.5:
                signal = "STRONG BUY"
            elif final_score >= 1.1:
                signal = "WEAK BUY"
            elif final_score >= 0.9:
                signal = "HOLD"
            elif final_score >= 0.5:
                signal = "WEAK SELL"
            else:
                signal = "STRONG SELL"
            
            row = (
                f"{result['symbol']:<10} | "
                f"${result['close']:>9,.2f} | "
                f"{result['volume']:>12,.0f} | "
                f"{result['regime']:<14} | "
                f"{final_score:>11.4f} | "
                f"{signal:<12}"
            )
            self.logger.info(row)
        
        self.logger.info("=" * 140)
    
    def process_iteration(self):
        """Process one iteration of signal generation"""
        self.iteration += 1
        start_time = time.time()
        
        # Discover symbols
        symbols = self.discover_symbols()
        
        if not symbols:
            self.logger.warning("No symbols found in Redis")
            return
        
        self.logger.info(f"Processing {len(symbols)} symbols: {', '.join(symbols)}")
        
        # Process symbols in parallel (2 workers)
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.PARALLEL_WORKERS) as executor:
            # Submit tasks
            future_to_symbol = {}
            for symbol in symbols:
                history = self.fetch_symbol_history(symbol)
                if history:
                    future = executor.submit(self.processor.process_symbol, symbol, history)
                    future_to_symbol[future] = symbol
            
            # Collect results
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:
                        # Write to Redis
                        self.write_signals_to_redis(symbol, result)
                        results.append(result)
                except Exception as e:
                    self.logger.error(f"{symbol}: Error in parallel processing - {e}")
        
        # Print summary
        self.print_summary_table(results)
        
        elapsed = time.time() - start_time
        self.logger.info(f"Iteration {self.iteration} completed in {elapsed:.2f}s")
        self.logger.info("")
    
    def run(self):
        """Main run loop"""
        self.logger.info("Starting signal generation loop...")
        self.logger.info(f"Processing interval: {self.config.PROCESS_INTERVAL}s")
        self.logger.info("")
        
        try:
            while True:
                try:
                    self.process_iteration()
                except Exception as e:
                    self.logger.error(f"Error in iteration {self.iteration}: {e}", exc_info=True)
                
                # Sleep until next interval
                time.sleep(self.config.PROCESS_INTERVAL)
                
        except KeyboardInterrupt:
            self.logger.info("\nShutdown signal received (Ctrl+C)")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("=" * 100)
        self.logger.info("SHUTTING DOWN")
        self.logger.info("=" * 100)
        self.logger.info(f"Total iterations processed: {self.iteration}")
        
        if self.redis_client:
            self.redis_client.close()
            self.logger.info("Redis connection closed")
        
        self.logger.info("Cleanup completed")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    config = Config()
    logger = setup_logging(config)
    
    try:
        generator = HFTSignalGenerator(config, logger)
        generator.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()