"""
Momentum Strategy
Trend following strategy with technical indicators
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import logging
from datetime import datetime, timedelta
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy using moving averages and RSI
    Follows trends with confirmation from multiple indicators
    """
    
    def __init__(self, data_handler, risk_manager, config):
        super().__init__(data_handler, risk_manager, config, "Momentum")
        
        # Get strategy-specific configuration
        self.strategy_config = config.get_strategy_config('momentum')
        
        # Strategy parameters
        self.fast_ma_period = self.strategy_config.get('parameters', {}).get('fast_ma_period', 12)
        self.slow_ma_period = self.strategy_config.get('parameters', {}).get('slow_ma_period', 26)
        self.signal_ma_period = self.strategy_config.get('parameters', {}).get('signal_ma_period', 9)
        self.rsi_period = self.strategy_config.get('parameters', {}).get('rsi_period', 14)
        self.rsi_oversold = self.strategy_config.get('parameters', {}).get('rsi_oversold', 30)
        self.rsi_overbought = self.strategy_config.get('parameters', {}).get('rsi_overbought', 70)
        self.volume_threshold = self.strategy_config.get('parameters', {}).get('volume_threshold', 1.5)
        
        # Risk management
        self.max_position_size = self.strategy_config.get('risk_management', {}).get('max_position_size', 0.08)
        self.stop_loss_pct = self.strategy_config.get('risk_management', {}).get('stop_loss_pct', 0.02)
        self.take_profit_pct = self.strategy_config.get('risk_management', {}).get('take_profit_pct', 0.06)
        
        # Universe of symbols to trade
        self.symbols = self.get_tradeable_symbols()
        
        # Technical indicators cache
        self.indicators_cache = {}
        
        logger.info(f"Momentum Strategy initialized with {len(self.symbols)} symbols")
    
    def get_tradeable_symbols(self) -> List[str]:
        """Get list of symbols to trade"""
        # Default momentum trading universe
        symbols = [
            'SPY', 'QQQ', 'IWM', 'DIA',  # ETFs
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',  # Tech
            'JPM', 'BAC', 'WFC', 'GS',  # Finance
            'XOM', 'CVX', 'COP',  # Energy
            'JNJ', 'PFE', 'UNH',  # Healthcare
        ]
        return symbols
    
    def calculate_technical_indicators(self, symbol: str, lookback_days: int = 100) -> Dict[str, Any]:
        """Calculate technical indicators for a symbol"""
        try:
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + 50)  # Extra buffer
            
            data = self.data_handler.get_historical_data(symbol, start_date, end_date)
            if data is None or len(data) < lookback_days:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            df = data.copy()
            
            # Calculate moving averages
            df['ema_fast'] = df['close'].ewm(span=self.fast_ma_period).mean()
            df['ema_slow'] = df['close'].ewm(span=self.slow_ma_period).mean()
            df['macd'] = df['ema_fast'] - df['ema_slow']
            df['macd_signal'] = df['macd'].ewm(span=self.signal_ma_period).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Calculate RSI
            df['rsi'] = self.calculate_rsi(df['close'], self.rsi_period)
            
            # Calculate Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Calculate volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Calculate price momentum
            df['price_momentum_5'] = df['close'].pct_change(5)
            df['price_momentum_10'] = df['close'].pct_change(10)
            df['price_momentum_20'] = df['close'].pct_change(20)
            
            # Get latest values
            latest = df.iloc[-1]
            
            indicators = {
                'symbol': symbol,
                'current_price': latest['close'],
                'ema_fast': latest['ema_fast'],
                'ema_slow': latest['ema_slow'],
                'macd': latest['macd'],
                'macd_signal': latest['macd_signal'],
                'macd_histogram': latest['macd_histogram'],
                'rsi': latest['rsi'],
                'bb_position': latest['bb_position'],
                'volume_ratio': latest['volume_ratio'],
                'price_momentum_5': latest['price_momentum_5'],
                'price_momentum_10': latest['price_momentum_10'],
                'price_momentum_20': latest['price_momentum_20'],
                'last_updated': datetime.now()
            }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            return None
    
    def calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self) -> List[Dict[str, Any]]:
        """Generate trading signals for all symbols"""
        signals = []
        
        try:
            # Update indicators for all symbols
            self.update_indicators()
            
            # Check existing positions for exit signals
            signals.extend(self.check_exit_signals())
            
            # Check for new entry signals
            signals.extend(self.check_entry_signals())
            
            # Validate all signals
            validated_signals = []
            for signal in signals:
                if self.validate_signal(signal):
                    self.add_signal_to_history(signal)
                    validated_signals.append(signal)
            
            return validated_signals
            
        except Exception as e:
            logger.error(f"Error generating momentum signals: {e}")
            return []
    
    def update_indicators(self):
        """Update technical indicators for all symbols"""
        for symbol in self.symbols:
            indicators = self.calculate_technical_indicators(symbol)
            if indicators:
                self.indicators_cache[symbol] = indicators
    
    def check_entry_signals(self) -> List[Dict[str, Any]]:
        """Check for new entry signals"""
        signals = []
        
        for symbol, indicators in self.indicators_cache.items():
            # Skip if already have position
            if symbol in self.positions and self.positions[symbol]['quantity'] != 0:
                continue
            
            signal = self.analyze_entry_signal(symbol, indicators)
            if signal:
                signals.append(signal)
        
        return signals
    
    def analyze_entry_signal(self, symbol: str, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze entry signal for a symbol"""
        try:
            # Extract indicators
            current_price = indicators['current_price']
            ema_fast = indicators['ema_fast']
            ema_slow = indicators['ema_slow']
            macd = indicators['macd']
            macd_signal = indicators['macd_signal']
            macd_histogram = indicators['macd_histogram']
            rsi = indicators['rsi']
            bb_position = indicators['bb_position']
            volume_ratio = indicators['volume_ratio']
            momentum_5 = indicators['price_momentum_5']
            momentum_10 = indicators['price_momentum_10']
            
            # Initialize signal strength
            signal_strength = 0
            action = None
            
            # MACD crossover signals
            if macd > macd_signal and macd_histogram > 0:
                signal_strength += 1  # Bullish MACD
            elif macd < macd_signal and macd_histogram < 0:
                signal_strength -= 1  # Bearish MACD
            
            # Moving average trend
            if ema_fast > ema_slow:
                signal_strength += 1  # Bullish trend
            else:
                signal_strength -= 1  # Bearish trend
            
            # RSI momentum (avoid extreme overbought/oversold)
            if 40 < rsi < 60:
                signal_strength += 0.5  # Neutral RSI is good for momentum
            elif rsi > self.rsi_overbought:
                signal_strength -= 1  # Overbought
            elif rsi < self.rsi_oversold:
                signal_strength -= 1  # Oversold
            
            # Price momentum
            if momentum_5 > 0.02 and momentum_10 > 0.05:
                signal_strength += 1  # Strong upward momentum
            elif momentum_5 < -0.02 and momentum_10 < -0.05:
                signal_strength -= 1  # Strong downward momentum
            
            # Volume confirmation
            if volume_ratio > self.volume_threshold:
                signal_strength += 0.5  # High volume confirmation
            
            # Bollinger Bands position
            if 0.2 < bb_position < 0.8:
                signal_strength += 0.5  # Not at extremes
            
            # Determine action based on signal strength
            if signal_strength >= 2.5:
                action = 'BUY'
                confidence = min(signal_strength / 4.0, 1.0)
            elif signal_strength <= -2.5:
                action = 'SELL'
                confidence = min(abs(signal_strength) / 4.0, 1.0)
            else:
                return None  # No clear signal
            
            # Calculate position size
            position_size = self.calculate_position_size(symbol, confidence)
            
            if position_size > 0:
                signal = {
                    'symbol': symbol,
                    'action': action,
                    'quantity': position_size,
                    'price': current_price,
                    'timestamp': datetime.now(),
                    'confidence': confidence,
                    'metadata': {
                        'signal_strength': signal_strength,
                        'macd': macd,
                        'macd_signal': macd_signal,
                        'rsi': rsi,
                        'momentum_5': momentum_5,
                        'momentum_10': momentum_10,
                        'volume_ratio': volume_ratio
                    }
                }
                
                logger.info(f"Entry signal for {symbol}: {action}, strength: {signal_strength:.2f}, confidence: {confidence:.2f}")
                return signal
            
        except Exception as e:
            logger.error(f"Error analyzing entry signal for {symbol}: {e}")
        
        return None
    
    def check_exit_signals(self) -> List[Dict[str, Any]]:
        """Check for exit signals on existing positions"""
        signals = []
        
        for symbol, position in self.positions.items():
            if position['quantity'] == 0:
                continue
            
            if symbol not in self.indicators_cache:
                continue
            
            indicators = self.indicators_cache[symbol]
            exit_signal = self.analyze_exit_signal(symbol, position, indicators)
            
            if exit_signal:
                signals.append(exit_signal)
        
        return signals
    
    def analyze_exit_signal(self, symbol: str, position: Dict, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze exit signal for a position"""
        try:
            current_price = indicators['current_price']
            avg_price = position['avg_price']
            quantity = position['quantity']
            
            # Calculate P&L
            if quantity > 0:  # Long position
                pnl_pct = (current_price - avg_price) / avg_price
            else:  # Short position
                pnl_pct = (avg_price - current_price) / avg_price
            
            should_exit = False
            exit_reason = ""
            
            # Stop loss
            if pnl_pct <= -self.stop_loss_pct:
                should_exit = True
                exit_reason = "stop_loss"
            
            # Take profit
            elif pnl_pct >= self.take_profit_pct:
                should_exit = True
                exit_reason = "take_profit"
            
            # Technical exit signals
            else:
                macd = indicators['macd']
                macd_signal = indicators['macd_signal']
                rsi = indicators['rsi']
                momentum_5 = indicators['price_momentum_5']
                
                # Exit long position
                if quantity > 0:
                    if (macd < macd_signal and rsi > self.rsi_overbought) or momentum_5 < -0.03:
                        should_exit = True
                        exit_reason = "technical_reversal"
                
                # Exit short position
                elif quantity < 0:
                    if (macd > macd_signal and rsi < self.rsi_oversold) or momentum_5 > 0.03:
                        should_exit = True
                        exit_reason = "technical_reversal"
            
            if should_exit:
                signal = {
                    'symbol': symbol,
                    'action': 'CLOSE',
                    'quantity': abs(quantity),
                    'price': current_price,
                    'timestamp': datetime.now(),
                    'confidence': 1.0,
                    'metadata': {
                        'exit_reason': exit_reason,
                        'pnl_pct': pnl_pct,
                        'entry_price': avg_price
                    }
                }
                
                logger.info(f"Exit signal for {symbol}: {exit_reason}, P&L: {pnl_pct:.2%}")
                return signal
            
        except Exception as e:
            logger.error(f"Error analyzing exit signal for {symbol}: {e}")
        
        return None
    
    def calculate_position_size(self, symbol: str, signal_strength: float) -> float:
        """Calculate position size based on risk management"""
        try:
            # Get current portfolio value
            portfolio_value = self.risk_manager.get_portfolio_value()
            
            # Calculate base position size
            base_position_value = portfolio_value * self.max_position_size
            
            # Adjust based on signal strength
            adjusted_position_value = base_position_value * signal_strength
            
            # Get current price
            current_price = self.data_handler.get_current_price(symbol)
            if not current_price:
                return 0
            
            # Calculate quantity
            quantity = adjusted_position_value / current_price
            
            # Apply risk management constraints
            max_quantity = self.risk_manager.get_max_position_size(symbol)
            quantity = min(quantity, max_quantity)
            
            return quantity
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get strategy summary"""
        active_positions = self.get_current_positions()
        
        summary = {
            'strategy_name': self.name,
            'total_symbols_monitored': len(self.symbols),
            'active_positions': len(active_positions),
            'positions_details': []
        }
        
        for symbol, position in active_positions.items():
            if symbol in self.indicators_cache:
                indicators = self.indicators_cache[symbol]
                current_price = indicators['current_price']
                pnl_pct = ((current_price - position['avg_price']) / position['avg_price']) * (1 if position['quantity'] > 0 else -1)
                
                position_detail = {
                    'symbol': symbol,
                    'quantity': position['quantity'],
                    'avg_price': position['avg_price'],
                    'current_price': current_price,
                    'pnl_pct': pnl_pct,
                    'rsi': indicators['rsi'],
                    'macd': indicators['macd']
                }
                summary['positions_details'].append(position_detail)
        
        return summary