#!/usr/bin/env python3
"""
Portfolio Management System - Redis Integrated
Processes market signals every 5 minutes and generates trade orders
Reads signals from Redis, writes portfolio state and trade orders to Redis
"""

import json
import logging
import time
import sys
import redis
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
from enum import Enum


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """System configuration"""
    
    # Redis
    REDIS_HOST = "redis"
    REDIS_PORT = 6379
    REDIS_DB = 1  # Same as signal generator
    
    # Processing
    PROCESS_INTERVAL = 300  # 5 minutes = 300 seconds
    
    # Logging
    LOG_DIR = "/app/logs/portfolio"
    LOG_LEVEL = "INFO"
    
    # Initial Capital
    INITIAL_CAPITAL = 10_000_000.0  # $10M starting capital
    
    # Position Sizing Limits
    MAX_POSITION = 0.20           # 20%
    MIN_POSITION = 0.05           # 5%
    TARGET_POSITIONS = 7          # Target 7 positions
    
    # Cash Management
    MIN_CASH_RESERVE = 0.10       # 10%
    TARGET_CASH_RESERVE = 0.15    # 15%
    MAX_CASH_RESERVE = 0.40       # 40%
    
    # Signal Classification
    STRONG_BUY_THRESHOLD = 1.5
    WEAK_BUY_THRESHOLD = 1.1
    NEUTRAL_THRESHOLD_LOW = 0.9
    WEAK_SELL_THRESHOLD = 0.7
    STRONG_SELL_THRESHOLD = 0.5
    
    # Trading Behavior
    SIGNAL_CHANGE_THRESHOLD = 0.2
    WEIGHT_CHANGE_THRESHOLD = 0.03
    ROTATION_ADVANTAGE = 1.20
    
    # Risk Limits
    MAX_DAILY_TRADES = 30
    MAX_SINGLE_LOSS = -0.10
    
    # Confidence & Volatility
    CONFIDENCE_LOOKBACK = 4
    MIN_CONFIDENCE = 0.3
    VOLATILITY_LOOKBACK = 20
    BASE_ALLOCATION = 0.15


class SignalClass(Enum):
    """Signal classification"""
    STRONG_BUY = "STRONG_BUY"
    WEAK_BUY = "WEAK_BUY"
    HOLD = "HOLD"
    WEAK_SELL = "WEAK_SELL"
    STRONG_SELL = "STRONG_SELL"


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Signal:
    """Market signal from Redis"""
    symbol: str
    timestamp: str
    close: float
    atr_14: float
    marker_final_score: float
    regime: str
    regime_confidence: float
    volume: float = 0.0
    
    @classmethod
    def from_redis_hash(cls, symbol: str, redis_data: Dict) -> 'Signal':
        """Create Signal from Redis hash"""
        return cls(
            symbol=symbol,
            timestamp=redis_data.get('timestamp', ''),
            close=float(redis_data.get('close', 0)),
            atr_14=float(redis_data.get('atr_14', 0)),
            marker_final_score=float(redis_data.get('marker_final_score', 1.0)),
            regime=redis_data.get('regime', 'NEUTRAL'),
            regime_confidence=float(redis_data.get('regime_confidence', 0.5)),
            volume=float(redis_data.get('volume', 0))
        )
    
    @property
    def classification(self) -> SignalClass:
        score = self.marker_final_score
        if score >= Config.STRONG_BUY_THRESHOLD:
            return SignalClass.STRONG_BUY
        elif score >= Config.WEAK_BUY_THRESHOLD:
            return SignalClass.WEAK_BUY
        elif score >= Config.NEUTRAL_THRESHOLD_LOW:
            return SignalClass.HOLD
        elif score >= Config.WEAK_SELL_THRESHOLD:
            return SignalClass.WEAK_SELL
        else:
            return SignalClass.STRONG_SELL
    
    @property
    def is_bullish(self) -> bool:
        return self.marker_final_score > Config.WEAK_BUY_THRESHOLD
    
    @property
    def is_bearish(self) -> bool:
        return self.marker_final_score < Config.NEUTRAL_THRESHOLD_LOW
    
    @property
    def is_neutral(self) -> bool:
        return not self.is_bullish and not self.is_bearish


@dataclass
class Position:
    """Portfolio position"""
    symbol: str
    shares: int
    avg_price: float
    current_price: float
    entry_signal: float = 0.0
    last_action_signal: float = 0.0
    last_action_cycle: int = 0
    cycles_held: int = 0
    
    @property
    def market_value(self) -> float:
        return self.shares * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.avg_price) * self.shares
    
    @property
    def unrealized_pnl_pct(self) -> float:
        return (self.current_price - self.avg_price) / self.avg_price if self.avg_price > 0 else 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Position':
        return cls(**data)


@dataclass
class Portfolio:
    """Portfolio state"""
    cash: float
    cycle: int = 0
    trades_today: int = 0
    last_updated: str = ""
    positions: Dict[str, Position] = field(default_factory=dict)
    signal_history: Dict[str, List[float]] = field(default_factory=dict)
    
    @property
    def total_value(self) -> float:
        return self.cash + sum(pos.market_value for pos in self.positions.values())
    
    @property
    def cash_percentage(self) -> float:
        return self.cash / self.total_value if self.total_value > 0 else 1.0
    
    @property
    def position_count(self) -> int:
        return len(self.positions)
    
    def get_position_weight(self, symbol: str) -> float:
        if symbol not in self.positions:
            return 0.0
        return self.positions[symbol].market_value / self.total_value
    
    def to_redis_dict(self) -> Dict:
        """Convert to dict for Redis storage"""
        return {
            'cash': str(self.cash),
            'total_value': str(self.total_value),
            'cycle': str(self.cycle),
            'trades_today': str(self.trades_today),
            'position_count': str(self.position_count),
            'cash_percentage': str(self.cash_percentage),
            'last_updated': self.last_updated,
            'positions': json.dumps({k: v.to_dict() for k, v in self.positions.items()}),
            'signal_history': json.dumps(self.signal_history)
        }
    
    @classmethod
    def from_redis_dict(cls, data: Dict) -> 'Portfolio':
        """Create Portfolio from Redis hash"""
        positions_json = data.get('positions', '{}')
        positions_dict = json.loads(positions_json) if positions_json else {}
        positions = {k: Position.from_dict(v) for k, v in positions_dict.items()}
        
        signal_history_json = data.get('signal_history', '{}')
        signal_history = json.loads(signal_history_json) if signal_history_json else {}
        
        return cls(
            cash=float(data.get('cash', Config.INITIAL_CAPITAL)),
            cycle=int(data.get('cycle', 0)),
            trades_today=int(data.get('trades_today', 0)),
            last_updated=data.get('last_updated', ''),
            positions=positions,
            signal_history=signal_history
        )


@dataclass
class TradeOrder:
    """Trade order for execution"""
    order_id: str
    symbol: str
    action: str  # "BUY" or "SELL"
    shares: int
    price: float
    dollar_amount: float
    reason: str
    priority: int
    target_weight: float
    current_weight: float
    signal_score: float
    signal_strength: float
    confidence: float
    timestamp: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(config: Config) -> logging.Logger:
    """Set up logging"""
    log_dir = Path(config.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"portfolio_{timestamp}.log"
    
    logger = logging.getLogger("PortfolioManager")
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
    logger.info("PORTFOLIO MANAGEMENT SYSTEM STARTED")
    logger.info("=" * 100)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Redis: {config.REDIS_HOST}:{config.REDIS_PORT}, DB: {config.REDIS_DB}")
    logger.info(f"Processing interval: {config.PROCESS_INTERVAL}s ({config.PROCESS_INTERVAL/60:.1f} minutes)")
    logger.info(f"Initial capital: ${config.INITIAL_CAPITAL:,.0f}")
    logger.info("=" * 100)
    
    return logger


# ============================================================================
# CORE CALCULATION MODULES
# ============================================================================

class SignalAnalyzer:
    """Analyzes and classifies signals"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def classify_signals(self, signals: List[Signal], portfolio: Portfolio) -> Dict[str, List[Signal]]:
        """Classify signals into buy/hold/sell"""
        classified = {'buy': [], 'hold': [], 'sell': []}
        
        for signal in signals:
            owns_stock = signal.symbol in portfolio.positions
            
            if signal.is_bullish:
                classified['buy'].append(signal)
            elif signal.is_neutral:
                if owns_stock:
                    classified['hold'].append(signal)
            else:
                if owns_stock:
                    classified['sell'].append(signal)
        
        self.logger.info(f"Classification: {len(classified['buy'])} BUY, "
                        f"{len(classified['hold'])} HOLD, {len(classified['sell'])} SELL")
        return classified
    
    def calculate_signal_strength(self, signal: Signal) -> float:
        """Convert score to strength [0, 1]"""
        score = signal.marker_final_score
        if score <= Config.WEAK_BUY_THRESHOLD:
            return 0.0
        strength = (score - Config.WEAK_BUY_THRESHOLD) / (2.0 - Config.WEAK_BUY_THRESHOLD)
        return min(max(strength, 0.0), 1.0)
    
    def calculate_confidence(self, signal: Signal, portfolio: Portfolio) -> float:
        """Calculate confidence from signal stability"""
        symbol = signal.symbol
        
        if symbol not in portfolio.signal_history or len(portfolio.signal_history[symbol]) < 2:
            return Config.MIN_CONFIDENCE
        
        history = portfolio.signal_history[symbol][-Config.CONFIDENCE_LOOKBACK:]
        
        mean_score = sum(history) / len(history)
        variance = sum((x - mean_score) ** 2 for x in history) / len(history)
        std_dev = math.sqrt(variance)
        
        cv = std_dev / mean_score if mean_score > 0 else 1.0
        confidence = 1.0 / (1.0 + cv)
        
        return max(confidence, Config.MIN_CONFIDENCE)


class PositionSizer:
    """Calculates position sizes"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def calculate_volatility_adjustment(self, signal: Signal, all_signals: List[Signal]) -> float:
        """Risk parity adjustment"""
        atrs = [s.atr_14 for s in all_signals if s.is_bullish and s.atr_14 > 0]
        if not atrs:
            return 1.0
        
        mean_atr = sum(atrs) / len(atrs)
        if signal.atr_14 <= 0:
            return 0.5
        
        adjustment = mean_atr / signal.atr_14
        return min(max(adjustment, 0.5), 2.0)
    
    def calculate_concentration_penalty(self, current_weight: float) -> float:
        """Soft cap with quadratic penalty"""
        ratio = current_weight / Config.MAX_POSITION
        penalty = 1.0 - (ratio ** 2)
        return max(penalty, 0.0)
    
    def calculate_position_size(self, signal: Signal, strength: float, confidence: float,
                                volatility_adj: float, current_weight: float) -> float:
        """Master position sizing formula"""
        base = Config.BASE_ALLOCATION
        conc_penalty = self.calculate_concentration_penalty(current_weight)
        
        raw_size = base * strength * confidence * volatility_adj * conc_penalty
        size = min(max(raw_size, Config.MIN_POSITION), Config.MAX_POSITION)
        
        if size < Config.MIN_POSITION:
            return 0.0
        return size


class TargetCalculator:
    """Calculates target portfolio weights"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.signal_analyzer = SignalAnalyzer(logger)
        self.position_sizer = PositionSizer(logger)
    
    def calculate_targets(self, signals: List[Signal], portfolio: Portfolio) -> Dict[str, float]:
        """Calculate target weights for each stock"""
        targets = {}
        classified = self.signal_analyzer.classify_signals(signals, portfolio)
        
        # SELL signals → 0%
        for signal in classified['sell']:
            targets[signal.symbol] = 0.0
        
        # HOLD signals → maintain
        for signal in classified['hold']:
            targets[signal.symbol] = portfolio.get_position_weight(signal.symbol)
        
        # BUY signals → calculate
        buy_signals = classified['buy']
        if not buy_signals:
            self.logger.info("No BUY signals")
            return targets
        
        capital_info = self._calculate_available_capital(signals, portfolio, classified)
        deployable = capital_info['deployable']
        
        if deployable <= 0:
            self.logger.warning("No deployable capital")
            return targets
        
        # Calculate allocations
        buy_allocations = {}
        for signal in buy_signals:
            strength = self.signal_analyzer.calculate_signal_strength(signal)
            confidence = self.signal_analyzer.calculate_confidence(signal, portfolio)
            volatility_adj = self.position_sizer.calculate_volatility_adjustment(signal, signals)
            current_weight = portfolio.get_position_weight(signal.symbol)
            
            size = self.position_sizer.calculate_position_size(
                signal, strength, confidence, volatility_adj, current_weight
            )
            
            if size > 0:
                buy_allocations[signal.symbol] = {
                    'target_weight': size,
                    'strength': strength,
                    'confidence': confidence
                }
        
        # Normalize to fit capital
        total_buy_weight = sum(alloc['target_weight'] for alloc in buy_allocations.values())
        max_deployable_weight = deployable / portfolio.total_value
        
        if total_buy_weight > max_deployable_weight:
            scale = max_deployable_weight / total_buy_weight
            self.logger.info(f"Scaling buy allocations by {scale:.2%}")
            for symbol in buy_allocations:
                buy_allocations[symbol]['target_weight'] *= scale
        
        for symbol, alloc in buy_allocations.items():
            targets[symbol] = alloc['target_weight']
        
        return targets
    
    def _calculate_available_capital(self, signals: List[Signal], portfolio: Portfolio,
                                    classified: Dict) -> Dict:
        """Calculate available capital"""
        total_value = portfolio.total_value
        current_cash = portfolio.cash
        
        capital_from_sells = sum(
            portfolio.positions[s.symbol].market_value
            for s in classified['sell']
            if s.symbol in portfolio.positions
        )
        
        total_available = current_cash + capital_from_sells
        min_reserve = Config.MIN_CASH_RESERVE * total_value
        deployable = max(total_available - min_reserve, 0.0)
        
        return {
            'current_cash': current_cash,
            'capital_from_sells': capital_from_sells,
            'total_available': total_available,
            'min_reserve': min_reserve,
            'deployable': deployable
        }


class TradeGenerator:
    """Generates trade orders"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def should_trade(self, symbol: str, target_weight: float, current_weight: float,
                    signal: Signal, portfolio: Portfolio) -> bool:
        """Determine if should trade"""
        if symbol not in portfolio.positions and target_weight > 0:
            return True
        if target_weight == 0 and symbol in portfolio.positions:
            return True
        
        weight_change = abs(target_weight - current_weight)
        if weight_change < Config.WEIGHT_CHANGE_THRESHOLD:
            return False
        
        position = portfolio.positions.get(symbol)
        if position:
            signal_change = abs(signal.marker_final_score - position.last_action_signal)
            if signal_change < Config.SIGNAL_CHANGE_THRESHOLD:
                return False
        
        return True
    
    def generate_orders(self, signals: List[Signal], targets: Dict[str, float],
                       portfolio: Portfolio) -> List[TradeOrder]:
        """Generate trade orders"""
        orders = []
        signal_map = {s.symbol: s for s in signals}
        order_counter = 0
        
        # SELL orders
        for symbol in list(portfolio.positions.keys()):
            target_weight = targets.get(symbol, 0.0)
            current_weight = portfolio.get_position_weight(symbol)
            
            if symbol not in signal_map:
                continue
            
            signal = signal_map[symbol]
            
            if target_weight < current_weight:
                if not self.should_trade(symbol, target_weight, current_weight, signal, portfolio):
                    continue
                
                position = portfolio.positions[symbol]
                if target_weight == 0:
                    shares_to_sell = position.shares
                    reason = f"EXIT: Signal {signal.marker_final_score:.2f} bearish"
                else:
                    target_value = target_weight * portfolio.total_value
                    current_value = position.market_value
                    dollars_to_sell = current_value - target_value
                    shares_to_sell = int(dollars_to_sell / position.current_price)
                    shares_to_sell = min(shares_to_sell, position.shares)
                    reason = f"TRIM: Rebalance to {target_weight:.1%}"
                
                if shares_to_sell > 0:
                    order_counter += 1
                    analyzer = SignalAnalyzer(self.logger)
                    strength = analyzer.calculate_signal_strength(signal)
                    confidence = analyzer.calculate_confidence(signal, portfolio)
                    
                    orders.append(TradeOrder(
                        order_id=f"ORD-{portfolio.cycle:04d}-{order_counter:03d}",
                        symbol=symbol,
                        action="SELL",
                        shares=shares_to_sell,
                        price=position.current_price,
                        dollar_amount=shares_to_sell * position.current_price,
                        reason=reason,
                        priority=1,
                        target_weight=target_weight,
                        current_weight=current_weight,
                        signal_score=signal.marker_final_score,
                        signal_strength=strength,
                        confidence=confidence,
                        timestamp=datetime.now().isoformat()
                    ))
        
        # BUY orders
        for symbol, target_weight in targets.items():
            if target_weight == 0:
                continue
            
            current_weight = portfolio.get_position_weight(symbol)
            
            if symbol not in signal_map:
                continue
            
            signal = signal_map[symbol]
            
            if target_weight > current_weight:
                if not self.should_trade(symbol, target_weight, current_weight, signal, portfolio):
                    continue
                
                target_value = target_weight * portfolio.total_value
                current_value = current_weight * portfolio.total_value
                dollars_to_buy = target_value - current_value
                shares_to_buy = int(dollars_to_buy / signal.close)
                
                if shares_to_buy > 0:
                    order_counter += 1
                    analyzer = SignalAnalyzer(self.logger)
                    strength = analyzer.calculate_signal_strength(signal)
                    confidence = analyzer.calculate_confidence(signal, portfolio)
                    
                    if symbol in portfolio.positions:
                        reason = f"ADD: Signal {signal.marker_final_score:.2f} strengthening"
                    else:
                        reason = f"ENTER: Signal {signal.marker_final_score:.2f} bullish"
                    
                    priority = 2 + int((2.0 - signal.marker_final_score) * 10)
                    
                    orders.append(TradeOrder(
                        order_id=f"ORD-{portfolio.cycle:04d}-{order_counter:03d}",
                        symbol=symbol,
                        action="BUY",
                        shares=shares_to_buy,
                        price=signal.close,
                        dollar_amount=shares_to_buy * signal.close,
                        reason=reason,
                        priority=priority,
                        target_weight=target_weight,
                        current_weight=current_weight,
                        signal_score=signal.marker_final_score,
                        signal_strength=strength,
                        confidence=confidence,
                        timestamp=datetime.now().isoformat()
                    ))
        
        orders.sort(key=lambda x: x.priority)
        return orders


# ============================================================================
# MAIN PORTFOLIO MANAGER
# ============================================================================

class PortfolioManager:
    """Main portfolio manager with Redis integration"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.redis_client = None
        self.target_calculator = TargetCalculator(logger)
        self.trade_generator = TradeGenerator(logger)
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
    
    def load_portfolio_state(self) -> Portfolio:
        """Load portfolio state from Redis"""
        try:
            redis_key = "portfolio:state"
            exists = self.redis_client.exists(redis_key)
            
            if not exists:
                self.logger.info("No existing portfolio state, creating new with initial capital")
                portfolio = Portfolio(
                    cash=self.config.INITIAL_CAPITAL,
                    last_updated=datetime.now().isoformat()
                )
                self.save_portfolio_state(portfolio)
                return portfolio
            
            data = self.redis_client.hgetall(redis_key)
            decoded_data = {k.decode('utf-8'): v.decode('utf-8') for k, v in data.items()}
            portfolio = Portfolio.from_redis_dict(decoded_data)
            
            self.logger.info(f"Loaded portfolio state: Cycle {portfolio.cycle}, "
                           f"${portfolio.total_value:,.0f} total")
            return portfolio
            
        except Exception as e:
            self.logger.error(f"Error loading portfolio state: {e}")
            return Portfolio(cash=self.config.INITIAL_CAPITAL)
    
    def save_portfolio_state(self, portfolio: Portfolio):
        """Save portfolio state to Redis"""
        try:
            redis_key = "portfolio:state"
            redis_data = portfolio.to_redis_dict()
            self.redis_client.hset(redis_key, mapping=redis_data)
            self.logger.debug("Portfolio state saved to Redis")
        except Exception as e:
            self.logger.error(f"Error saving portfolio state: {e}")
    
    def discover_signals(self) -> List[str]:
        """Discover all available signal symbols"""
        try:
            keys = self.redis_client.keys("signals:latest:*")
            symbols = [key.decode('utf-8').split(":")[-1] for key in keys]
            return sorted(symbols)
        except Exception as e:
            self.logger.error(f"Error discovering signals: {e}")
            return []
    
    def load_signals(self, symbols: List[str]) -> List[Signal]:
        """Load signals from Redis with freshness validation"""
        signals = []
        max_age_seconds = 60  # Don't use signals older than 1 minute
        current_time = datetime.now()
        
        for symbol in symbols:
            try:
                redis_key = f"signals:latest:{symbol}"
                data = self.redis_client.hgetall(redis_key)
                if not data:
                    self.logger.warning(f"{symbol}: No signal data found")
                    continue
                
                decoded_data = {k.decode('utf-8'): v.decode('utf-8') for k, v in data.items()}
                
                # ✅ CHECK SIGNAL FRESHNESS
                signal_timestamp = decoded_data.get('last_updated', '')
                if signal_timestamp:
                    try:
                        signal_dt = datetime.fromisoformat(signal_timestamp)
                        age_seconds = (current_time - signal_dt).total_seconds()
                        
                        if age_seconds > max_age_seconds:
                            self.logger.warning(
                                f"{symbol}: Signal is stale ({age_seconds:.0f}s old), skipping"
                            )
                            continue
                        
                        self.logger.debug(f"{symbol}: Signal age {age_seconds:.1f}s (fresh)")
                        
                    except ValueError as e:
                        self.logger.error(f"{symbol}: Invalid timestamp format: {signal_timestamp}")
                        continue
                else:
                    self.logger.warning(f"{symbol}: No timestamp in signal data, skipping")
                    continue
                
                signal = Signal.from_redis_hash(symbol, decoded_data)
                signals.append(signal)
                
            except Exception as e:
                self.logger.error(f"{symbol}: Error loading signal - {e}")
        
        self.logger.info(f"Loaded {len(signals)} fresh signals out of {len(symbols)} discovered")
        return signals
    
    def save_trade_orders(self, orders: List[TradeOrder]):
        """Save trade orders to Redis for executor"""
        try:
            redis_key = "portfolio:orders:pending"
            
            # Clear previous pending orders
            self.redis_client.delete(redis_key)
            
            # Add new orders
            if orders:
                orders_json = [json.dumps(order.to_dict()) for order in orders]
                self.redis_client.rpush(redis_key, *orders_json)
                self.logger.info(f"Saved {len(orders)} orders to Redis")
            else:
                self.logger.info("No orders to save")
                
        except Exception as e:
            self.logger.error(f"Error saving orders: {e}")
    
    def update_signal_history(self, signals: List[Signal], portfolio: Portfolio):
        """Update signal history for confidence calculation"""
        for signal in signals:
            if signal.symbol not in portfolio.signal_history:
                portfolio.signal_history[signal.symbol] = []
            
            portfolio.signal_history[signal.symbol].append(signal.marker_final_score)
            
            if len(portfolio.signal_history[signal.symbol]) > Config.VOLATILITY_LOOKBACK:
                portfolio.signal_history[signal.symbol] = \
                    portfolio.signal_history[signal.symbol][-Config.VOLATILITY_LOOKBACK:]
    
    def update_position_prices(self, signals: List[Signal], portfolio: Portfolio):
        """Update current prices in positions"""
        signal_map = {s.symbol: s for s in signals}
        for symbol, position in portfolio.positions.items():
            if symbol in signal_map:
                position.current_price = signal_map[symbol].close
    
    def validate_orders(self, orders: List[TradeOrder], portfolio: Portfolio) -> List[TradeOrder]:
        """Validate orders against cash constraints"""
        validated = []
        simulated_cash = portfolio.cash
        
        for order in orders:
            if order.action == "SELL":
                simulated_cash += order.dollar_amount
                validated.append(order)
            elif order.action == "BUY":
                if simulated_cash >= order.dollar_amount:
                    simulated_cash -= order.dollar_amount
                    validated.append(order)
                else:
                    self.logger.warning(f"{order.symbol}: Insufficient cash for BUY, skipping")
        
        if portfolio.trades_today + len(validated) > Config.MAX_DAILY_TRADES:
            excess = (portfolio.trades_today + len(validated)) - Config.MAX_DAILY_TRADES
            self.logger.warning(f"Would exceed daily limit, dropping {excess} orders")
            validated = validated[:-excess]
        
        return validated
    
    def log_portfolio_state(self, portfolio: Portfolio):
        """Log current portfolio state"""
        self.logger.info(f"Portfolio: ${portfolio.total_value:,.0f} total, "
                        f"${portfolio.cash:,.0f} cash ({portfolio.cash_percentage:.1%}), "
                        f"{portfolio.position_count} positions")
        
        if portfolio.positions:
            self.logger.info("Current Positions:")
            for symbol, pos in sorted(portfolio.positions.items()):
                weight = pos.market_value / portfolio.total_value
                pnl_pct = pos.unrealized_pnl_pct * 100
                self.logger.info(
                    f"  {symbol:<6}: {pos.shares:>6} shares @ ${pos.current_price:>8.2f} "
                    f"= ${pos.market_value:>12,.0f} ({weight:>5.1%}) [{pnl_pct:>+6.1f}%]"
                )
    
    def log_signals_summary(self, signals: List[Signal]):
        """Log signals summary table"""
        self.logger.info("")
        self.logger.info("=" * 110)
        self.logger.info("MARKET SIGNALS")
        self.logger.info("=" * 110)
        
        header = (f"{'Symbol':<8} | {'Close':>10} | {'ATR':>8} | {'Score':>8} | "
                 f"{'Class':<12} | {'Regime':<14} | {'Conf':>6}")
        self.logger.info(header)
        self.logger.info("-" * 110)
        
        for signal in sorted(signals, key=lambda s: s.marker_final_score, reverse=True):
            row = (f"{signal.symbol:<8} | ${signal.close:>9.2f} | "
                  f"{signal.atr_14:>8.2f} | {signal.marker_final_score:>8.4f} | "
                  f"{signal.classification.value:<12} | {signal.regime:<14} | "
                  f"{signal.regime_confidence:>6.2f}")
            self.logger.info(row)
        
        self.logger.info("=" * 110)
        self.logger.info("")
    
    def log_orders_comprehensive(self, orders: List[TradeOrder]):
        """Log comprehensive order details"""
        if not orders:
            self.logger.info("No trade orders generated this cycle")
            return
        
        self.logger.info("")
        self.logger.info("=" * 140)
        self.logger.info(f"TRADE ORDERS - {len(orders)} orders generated")
        self.logger.info("=" * 140)
        
        for order in orders:
            self.logger.info("")
            self.logger.info(f"Order ID: {order.order_id}")
            self.logger.info(f"  Symbol:         {order.symbol}")
            self.logger.info(f"  Action:         {order.action}")
            self.logger.info(f"  Shares:         {order.shares:,}")
            self.logger.info(f"  Price:          ${order.price:.2f}")
            self.logger.info(f"  Dollar Amount:  ${order.dollar_amount:,.2f}")
            self.logger.info(f"  Reason:         {order.reason}")
            self.logger.info(f"  Signal Score:   {order.signal_score:.4f}")
            self.logger.info(f"  Signal Strength:{order.signal_strength:.4f}")
            self.logger.info(f"  Confidence:     {order.confidence:.4f}")
            self.logger.info(f"  Current Weight: {order.current_weight:.2%}")
            self.logger.info(f"  Target Weight:  {order.target_weight:.2%}")
            self.logger.info(f"  Priority:       {order.priority}")
            self.logger.info(f"  Timestamp:      {order.timestamp}")
        
        self.logger.info("")
        self.logger.info("-" * 140)
        
        total_buy = sum(o.dollar_amount for o in orders if o.action == "BUY")
        total_sell = sum(o.dollar_amount for o in orders if o.action == "SELL")
        
        self.logger.info(f"Summary: {sum(1 for o in orders if o.action == 'BUY')} BUY orders = ${total_buy:,.0f}")
        self.logger.info(f"         {sum(1 for o in orders if o.action == 'SELL')} SELL orders = ${total_sell:,.0f}")
        self.logger.info(f"         Net flow: ${total_buy - total_sell:+,.0f}")
        self.logger.info("=" * 140)
        self.logger.info("")
    
    def process_cycle(self):
        """Process one portfolio management cycle"""
        self.iteration += 1
        start_time = time.time()
        
        self.logger.info("=" * 100)
        self.logger.info(f"PORTFOLIO CYCLE {self.iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 100)
        self.logger.info("")
        
        # Load portfolio state
        portfolio = self.load_portfolio_state()
        portfolio.cycle += 1
        portfolio.last_updated = datetime.now().isoformat()
        
        # Discover and load signals
        symbols = self.discover_signals()
        self.logger.info(f"Discovered {len(symbols)} symbols: {', '.join(symbols)}")
        
        if not symbols:
            self.logger.warning("No signals found, skipping cycle")
            return
        
        signals = self.load_signals(symbols)
        self.logger.info(f"Loaded {len(signals)} valid signals")
        
        if not signals:
            self.logger.warning("No valid signals, skipping cycle")
            return
        
        # Update state
        self.update_signal_history(signals, portfolio)
        self.update_position_prices(signals, portfolio)
        
        # Log current state
        self.log_portfolio_state(portfolio)
        self.log_signals_summary(signals)
        
        # Calculate targets
        targets = self.target_calculator.calculate_targets(signals, portfolio)
        
        # Generate orders
        orders = self.trade_generator.generate_orders(signals, targets, portfolio)
        
        # Validate orders
        validated_orders = self.validate_orders(orders, portfolio)
        
        # Log orders
        self.log_orders_comprehensive(validated_orders)
        
        # Save orders to Redis
        self.save_trade_orders(validated_orders)
        
        # Update portfolio state for next cycle
        portfolio.trades_today += len(validated_orders)
        self.save_portfolio_state(portfolio)
        
        elapsed = time.time() - start_time
        self.logger.info(f"Cycle {self.iteration} completed in {elapsed:.2f}s")
        self.logger.info("")
    
    def run(self):
        """Main run loop"""
        self.logger.info("Starting portfolio management loop...")
        self.logger.info(f"Processing every {self.config.PROCESS_INTERVAL}s")
        self.logger.info("")
        
        try:
            while True:
                try:
                    self.process_cycle()
                except Exception as e:
                    self.logger.error(f"Error in cycle {self.iteration}: {e}", exc_info=True)
                
                # Sleep until next cycle
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
        self.logger.info(f"Total cycles processed: {self.iteration}")
        
        if self.redis_client:
            self.redis_client.close()
            self.logger.info("Redis connection closed")
        
        self.logger.info("Cleanup completed")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    config = Config()
    logger = setup_logging(config)
    
    try:
        manager = PortfolioManager(config, logger)
        manager.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()