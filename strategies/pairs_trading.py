"""
Pairs Trading Strategy
Statistical arbitrage strategy based on mean reversion of cointegrated pairs
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import logging
from datetime import datetime, timedelta
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class PairsTradingStrategy(BaseStrategy):
    """
    Pairs trading strategy using statistical arbitrage
    Identifies cointegrated pairs and trades mean reversion
    """
    
    def __init__(self, data_handler, risk_manager, config):
        super().__init__(data_handler, risk_manager, config, "Pairs Trading")
        
        # Get strategy-specific configuration
        self.strategy_config = config.get_strategy_config('pairs_trading')
        self.pairs_config = config.get_pairs_config()
        
        # Strategy parameters
        self.lookback_days = self.strategy_config.get('parameters', {}).get('lookback_period', 252)
        self.entry_threshold = self.strategy_config.get('parameters', {}).get('entry_zscore', 2.0)
        self.exit_threshold = self.strategy_config.get('parameters', {}).get('exit_zscore', 0.5)
        self.stop_loss_threshold = self.strategy_config.get('parameters', {}).get('stop_loss_zscore', 4.0)
        self.min_correlation = self.strategy_config.get('parameters', {}).get('min_correlation', 0.7)
        self.cointegration_pvalue = self.strategy_config.get('parameters', {}).get('cointegration_pvalue', 0.05)
        
        # Risk management
        self.max_position_size = self.strategy_config.get('risk_management', {}).get('max_position_size', 0.05)
        self.max_pairs_active = self.strategy_config.get('risk_management', {}).get('max_pairs_active', 10)
        
        # Active pairs and their statistics
        self.active_pairs = {}
        self.pair_statistics = {}
        
        logger.info(f"Pairs Trading Strategy initialized with {len(self.get_tradeable_pairs())} pairs")
    
    def get_tradeable_pairs(self) -> List[Tuple[str, str]]:
        """Get list of tradeable pairs from configuration"""
        pairs = []
        
        # Get predefined pairs from config
        predefined_pairs = self.pairs_config.get('predefined_pairs', [])
        for pair_config in predefined_pairs:
            pairs.append((pair_config['symbol1'], pair_config['symbol2']))
        
        return pairs
    
    def calculate_pair_statistics(self, symbol1: str, symbol2: str, lookback_days: int = None) -> Dict[str, Any]:
        """Calculate statistical measures for a pair"""
        if lookback_days is None:
            lookback_days = self.lookback_days
        
        try:
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + 30)  # Extra buffer
            
            data1 = self.data_handler.get_historical_data(symbol1, start_date, end_date)
            data2 = self.data_handler.get_historical_data(symbol2, start_date, end_date)
            
            if data1 is None or data2 is None or len(data1) < lookback_days or len(data2) < lookback_days:
                logger.warning(f"Insufficient data for pair {symbol1}/{symbol2}")
                return None
            
            # Align data by date
            df = pd.DataFrame({
                'price1': data1['close'],
                'price2': data2['close']
            }).dropna()
            
            if len(df) < lookback_days:
                logger.warning(f"Insufficient aligned data for pair {symbol1}/{symbol2}")
                return None
            
            # Take last lookback_days
            df = df.tail(lookback_days)
            
            # Calculate correlation
            correlation = df['price1'].corr(df['price2'])
            
            # Calculate cointegration
            from statsmodels.tsa.stattools import coint
            coint_score, coint_pvalue, _ = coint(df['price1'], df['price2'])
            
            # Calculate hedge ratio using linear regression
            from sklearn.linear_model import LinearRegression
            X = df['price1'].values.reshape(-1, 1)
            y = df['price2'].values
            reg = LinearRegression().fit(X, y)
            hedge_ratio = reg.coef_[0]
            
            # Calculate spread
            spread = df['price2'] - hedge_ratio * df['price1']
            
            # Calculate spread statistics
            spread_mean = spread.mean()
            spread_std = spread.std()
            current_spread = spread.iloc[-1]
            zscore = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0
            
            # Calculate half-life of mean reversion
            half_life = self.calculate_half_life(spread)
            
            statistics = {
                'symbol1': symbol1,
                'symbol2': symbol2,
                'correlation': correlation,
                'cointegration_score': coint_score,
                'cointegration_pvalue': coint_pvalue,
                'hedge_ratio': hedge_ratio,
                'spread_mean': spread_mean,
                'spread_std': spread_std,
                'current_spread': current_spread,
                'zscore': zscore,
                'half_life': half_life,
                'last_updated': datetime.now()
            }
            
            return statistics
            
        except Exception as e:
            logger.error(f"Error calculating statistics for pair {symbol1}/{symbol2}: {e}")
            return None
    
    def calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate half-life of mean reversion for spread"""
        try:
            from statsmodels.tsa.stattools import adfuller
            from statsmodels.regression.linear_model import OLS
            
            # Calculate lagged spread
            spread_lag = spread.shift(1).dropna()
            spread_diff = spread.diff().dropna()
            
            # Align series
            min_len = min(len(spread_lag), len(spread_diff))
            spread_lag = spread_lag.iloc[-min_len:]
            spread_diff = spread_diff.iloc[-min_len:]
            
            # Run regression: spread_diff = alpha + beta * spread_lag + error
            model = OLS(spread_diff, spread_lag).fit()
            beta = model.params.iloc[0]
            
            # Calculate half-life
            if beta < 0:
                half_life = -np.log(2) / beta
            else:
                half_life = np.inf
            
            return half_life
            
        except Exception as e:
            logger.warning(f"Error calculating half-life: {e}")
            return np.inf
    
    def is_pair_tradeable(self, statistics: Dict[str, Any]) -> bool:
        """Check if pair meets trading criteria"""
        if statistics is None:
            return False
        
        # Check correlation
        if abs(statistics['correlation']) < self.min_correlation:
            return False
        
        # Check cointegration
        if statistics['cointegration_pvalue'] > self.cointegration_pvalue:
            return False
        
        # Check half-life (should be reasonable for mean reversion)
        if statistics['half_life'] > 50 or statistics['half_life'] < 1:
            return False
        
        # Check spread standard deviation (avoid pairs with no volatility)
        if statistics['spread_std'] <= 0:
            return False
        
        return True
    
    def generate_signals(self) -> List[Dict[str, Any]]:
        """Generate trading signals for all pairs"""
        signals = []
        
        try:
            # Update pair statistics
            self.update_pair_statistics()
            
            # Check existing positions for exit signals
            signals.extend(self.check_exit_signals())
            
            # Check for new entry signals
            if len(self.active_pairs) < self.max_pairs_active:
                signals.extend(self.check_entry_signals())
            
            # Validate all signals
            validated_signals = []
            for signal in signals:
                if self.validate_signal(signal):
                    self.add_signal_to_history(signal)
                    validated_signals.append(signal)
            
            return validated_signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    def update_pair_statistics(self):
        """Update statistics for all tradeable pairs"""
        pairs = self.get_tradeable_pairs()
        
        for symbol1, symbol2 in pairs:
            pair_key = f"{symbol1}_{symbol2}"
            
            # Update statistics
            statistics = self.calculate_pair_statistics(symbol1, symbol2)
            if statistics:
                self.pair_statistics[pair_key] = statistics
    
    def check_entry_signals(self) -> List[Dict[str, Any]]:
        """Check for new entry signals"""
        signals = []
        
        for pair_key, stats in self.pair_statistics.items():
            if not self.is_pair_tradeable(stats):
                continue
            
            # Skip if pair is already active
            if pair_key in self.active_pairs:
                continue
            
            zscore = stats['zscore']
            
            # Check for entry signal
            if abs(zscore) >= self.entry_threshold:
                # Determine trade direction
                if zscore > self.entry_threshold:
                    # Spread is too high, short spread (sell symbol2, buy symbol1)
                    action1, action2 = 'BUY', 'SELL'
                elif zscore < -self.entry_threshold:
                    # Spread is too low, long spread (buy symbol2, sell symbol1)
                    action1, action2 = 'SELL', 'BUY'
                else:
                    continue
                
                # Calculate position sizes
                position_size1 = self.calculate_position_size(stats['symbol1'], abs(zscore))
                position_size2 = self.calculate_position_size(stats['symbol2'], abs(zscore))
                
                # Adjust position size based on hedge ratio
                position_size2 = position_size1 * stats['hedge_ratio']
                
                # Get current prices
                price1 = self.data_handler.get_current_price(stats['symbol1'])
                price2 = self.data_handler.get_current_price(stats['symbol2'])
                
                if price1 and price2:
                    # Create signals for both legs
                    signal1 = {
                        'symbol': stats['symbol1'],
                        'action': action1,
                        'quantity': position_size1,
                        'price': price1,
                        'timestamp': datetime.now(),
                        'confidence': min(abs(zscore) / self.entry_threshold, 1.0),
                        'metadata': {
                            'pair': pair_key,
                            'zscore': zscore,
                            'hedge_ratio': stats['hedge_ratio'],
                            'leg': 1
                        }
                    }
                    
                    signal2 = {
                        'symbol': stats['symbol2'],
                        'action': action2,
                        'quantity': position_size2,
                        'price': price2,
                        'timestamp': datetime.now(),
                        'confidence': min(abs(zscore) / self.entry_threshold, 1.0),
                        'metadata': {
                            'pair': pair_key,
                            'zscore': zscore,
                            'hedge_ratio': stats['hedge_ratio'],
                            'leg': 2
                        }
                    }
                    
                    signals.extend([signal1, signal2])
                    
                    # Mark pair as active
                    self.active_pairs[pair_key] = {
                        'entry_time': datetime.now(),
                        'entry_zscore': zscore,
                        'statistics': stats
                    }
                    
                    logger.info(f"Entry signal generated for pair {pair_key}, zscore: {zscore:.2f}")
        
        return signals
    
    def check_exit_signals(self) -> List[Dict[str, Any]]:
        """Check for exit signals on active pairs"""
        signals = []
        
        pairs_to_remove = []
        
        for pair_key, pair_info in self.active_pairs.items():
            if pair_key not in self.pair_statistics:
                continue
            
            stats = self.pair_statistics[pair_key]
            current_zscore = stats['zscore']
            entry_zscore = pair_info['entry_zscore']
            
            should_exit = False
            exit_reason = ""
            
            # Check for mean reversion (exit signal)
            if abs(current_zscore) <= self.exit_threshold:
                should_exit = True
                exit_reason = "mean_reversion"
            
            # Check for stop loss
            elif abs(current_zscore) >= self.stop_loss_threshold:
                should_exit = True
                exit_reason = "stop_loss"
            
            # Check for trend reversal (zscore changed sign significantly)
            elif np.sign(current_zscore) != np.sign(entry_zscore) and abs(current_zscore) > self.exit_threshold:
                should_exit = True
                exit_reason = "trend_reversal"
            
            if should_exit:
                # Generate exit signals for both legs
                symbol1, symbol2 = stats['symbol1'], stats['symbol2']
                
                # Get current positions
                pos1 = self.positions.get(symbol1, {}).get('quantity', 0)
                pos2 = self.positions.get(symbol2, {}).get('quantity', 0)
                
                if pos1 != 0 or pos2 != 0:
                    price1 = self.data_handler.get_current_price(symbol1)
                    price2 = self.data_handler.get_current_price(symbol2)
                    
                    if price1 and price2:
                        if pos1 != 0:
                            signal1 = {
                                'symbol': symbol1,
                                'action': 'CLOSE',
                                'quantity': abs(pos1),
                                'price': price1,
                                'timestamp': datetime.now(),
                                'confidence': 1.0,
                                'metadata': {
                                    'pair': pair_key,
                                    'zscore': current_zscore,
                                    'exit_reason': exit_reason,
                                    'leg': 1
                                }
                            }
                            signals.append(signal1)
                        
                        if pos2 != 0:
                            signal2 = {
                                'symbol': symbol2,
                                'action': 'CLOSE',
                                'quantity': abs(pos2),
                                'price': price2,
                                'timestamp': datetime.now(),
                                'confidence': 1.0,
                                'metadata': {
                                    'pair': pair_key,
                                    'zscore': current_zscore,
                                    'exit_reason': exit_reason,
                                    'leg': 2
                                }
                            }
                            signals.append(signal2)
                
                pairs_to_remove.append(pair_key)
                logger.info(f"Exit signal generated for pair {pair_key}, reason: {exit_reason}, zscore: {current_zscore:.2f}")
        
        # Remove closed pairs from active pairs
        for pair_key in pairs_to_remove:
            del self.active_pairs[pair_key]
        
        return signals
    
    def calculate_position_size(self, symbol: str, signal_strength: float) -> float:
        """Calculate position size based on risk management"""
        try:
            # Get current portfolio value
            portfolio_value = self.risk_manager.get_portfolio_value()
            
            # Calculate base position size
            base_position_value = portfolio_value * self.max_position_size
            
            # Adjust based on signal strength
            adjusted_position_value = base_position_value * min(signal_strength / self.entry_threshold, 1.0)
            
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
    
    def get_active_pairs_summary(self) -> Dict[str, Any]:
        """Get summary of active pairs"""
        summary = {
            'total_active_pairs': len(self.active_pairs),
            'pairs_details': []
        }
        
        for pair_key, pair_info in self.active_pairs.items():
            if pair_key in self.pair_statistics:
                stats = self.pair_statistics[pair_key]
                pair_detail = {
                    'pair': pair_key,
                    'entry_time': pair_info['entry_time'],
                    'entry_zscore': pair_info['entry_zscore'],
                    'current_zscore': stats['zscore'],
                    'correlation': stats['correlation'],
                    'half_life': stats['half_life']
                }
                summary['pairs_details'].append(pair_detail)
        
        return summary