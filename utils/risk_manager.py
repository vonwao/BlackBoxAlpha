"""
Risk Manager
Handles position sizing, risk controls, and portfolio management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Manages risk controls and position sizing for the trading bot
    """
    
    def __init__(self, config=None):
        self.config = config
        
        # Risk parameters
        self.max_portfolio_risk = config.get('trading.max_portfolio_risk', 0.02) if config else 0.02
        self.max_position_size = config.get('trading.max_position_size', 0.1) if config else 0.1
        self.max_correlation = 0.7  # Maximum correlation between positions
        self.max_sector_exposure = 0.3  # Maximum exposure to single sector
        
        # Portfolio tracking
        self.portfolio_value = 100000.0  # Starting portfolio value
        self.positions = {}
        self.daily_pnl_history = []
        self.risk_metrics = {}
        
        # Risk limits
        self.daily_loss_limit = 0.05  # 5% daily loss limit
        self.max_drawdown_limit = 0.15  # 15% maximum drawdown
        
        logger.info("Risk Manager initialized")
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        return self.portfolio_value
    
    def update_portfolio_value(self, new_value: float):
        """Update portfolio value"""
        old_value = self.portfolio_value
        self.portfolio_value = new_value
        
        # Calculate daily P&L
        daily_pnl = new_value - old_value
        self.daily_pnl_history.append({
            'date': datetime.now().date(),
            'pnl': daily_pnl,
            'portfolio_value': new_value
        })
        
        # Keep only last 252 days (1 year)
        if len(self.daily_pnl_history) > 252:
            self.daily_pnl_history = self.daily_pnl_history[-252:]
    
    def get_daily_pnl(self) -> float:
        """Get today's P&L"""
        if not self.daily_pnl_history:
            return 0.0
        
        today = datetime.now().date()
        for record in reversed(self.daily_pnl_history):
            if record['date'] == today:
                return record['pnl']
        
        return 0.0
    
    def calculate_position_size(self, symbol: str, signal_strength: float, 
                              current_price: float, volatility: float = None) -> float:
        """
        Calculate optimal position size based on risk management rules
        """
        try:
            # Base position size as percentage of portfolio
            base_position_value = self.portfolio_value * self.max_position_size
            
            # Adjust for signal strength
            adjusted_position_value = base_position_value * min(signal_strength, 1.0)
            
            # Volatility adjustment
            if volatility is not None and volatility > 0:
                # Reduce position size for high volatility assets
                vol_adjustment = min(1.0, 0.02 / volatility)  # Target 2% volatility
                adjusted_position_value *= vol_adjustment
            
            # Calculate quantity
            quantity = adjusted_position_value / current_price
            
            # Apply additional constraints
            max_quantity = self.get_max_position_size(symbol)
            quantity = min(quantity, max_quantity)
            
            # Check portfolio risk limits
            if not self.check_position_risk(symbol, quantity, current_price):
                quantity *= 0.5  # Reduce position size if risk is too high
            
            return max(0, quantity)
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0
    
    def get_max_position_size(self, symbol: str) -> float:
        """Get maximum allowed position size for a symbol"""
        # Maximum position value
        max_position_value = self.portfolio_value * self.max_position_size
        
        # Get current price (this would need to be passed or fetched)
        # For now, assume a reasonable price
        estimated_price = 100.0  # This should be actual current price
        
        return max_position_value / estimated_price
    
    def check_position_risk(self, symbol: str, quantity: float, price: float) -> bool:
        """Check if position meets risk requirements"""
        position_value = quantity * price
        
        # Check position size limit
        if position_value > self.portfolio_value * self.max_position_size:
            logger.warning(f"Position size too large for {symbol}")
            return False
        
        # Check portfolio concentration
        total_position_value = sum(pos.get('value', 0) for pos in self.positions.values())
        if (total_position_value + position_value) > self.portfolio_value * 0.8:
            logger.warning("Portfolio too concentrated")
            return False
        
        return True
    
    def check_portfolio_risk(self) -> Dict[str, bool]:
        """Check overall portfolio risk metrics"""
        risk_checks = {
            'daily_loss_limit': True,
            'max_drawdown_limit': True,
            'concentration_risk': True,
            'correlation_risk': True
        }
        
        try:
            # Check daily loss limit
            daily_pnl_pct = self.get_daily_pnl() / self.portfolio_value
            if daily_pnl_pct < -self.daily_loss_limit:
                risk_checks['daily_loss_limit'] = False
                logger.warning(f"Daily loss limit exceeded: {daily_pnl_pct:.2%}")
            
            # Check maximum drawdown
            max_drawdown = self.calculate_max_drawdown()
            if max_drawdown > self.max_drawdown_limit:
                risk_checks['max_drawdown_limit'] = False
                logger.warning(f"Maximum drawdown exceeded: {max_drawdown:.2%}")
            
            # Check concentration risk
            concentration = self.calculate_concentration_risk()
            if concentration > 0.5:  # More than 50% in single position
                risk_checks['concentration_risk'] = False
                logger.warning(f"High concentration risk: {concentration:.2%}")
            
            # Update risk metrics
            self.risk_metrics = {
                'daily_pnl_pct': daily_pnl_pct,
                'max_drawdown': max_drawdown,
                'concentration': concentration,
                'portfolio_value': self.portfolio_value,
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error checking portfolio risk: {e}")
        
        return risk_checks
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from daily_pnl_history (recorded portfolio values)"""
        if len(self.daily_pnl_history) < 2:
            return 0.0
        
        values = [record['portfolio_value'] for record in self.daily_pnl_history]
        peak = values[0]
        max_drawdown = 0.0
        
        for value in values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def calculate_concentration_risk(self) -> float:
        """Calculate portfolio concentration risk"""
        if not self.positions:
            return 0.0
        
        position_values = [pos.get('value', 0) for pos in self.positions.values()]
        total_value = sum(position_values)
        
        if total_value == 0:
            return 0.0
        
        # Calculate Herfindahl-Hirschman Index
        hhi = sum((value / total_value) ** 2 for value in position_values)
        
        # Convert to concentration percentage
        max_concentration = max(position_values) / total_value if total_value > 0 else 0
        
        return max_concentration
    
    def calculate_var(self, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk (VaR)"""
        if len(self.daily_pnl_history) < 30:
            return 0.0
        
        pnl_values = [record['pnl'] for record in self.daily_pnl_history[-30:]]
        var = np.percentile(pnl_values, confidence_level * 100)
        
        return abs(var)
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(self.daily_pnl_history) < 30:
            return 0.0
        
        returns = []
        for i in range(1, len(self.daily_pnl_history)):
            prev_value = self.daily_pnl_history[i-1]['portfolio_value']
            curr_value = self.daily_pnl_history[i]['portfolio_value']
            daily_return = (curr_value - prev_value) / prev_value
            returns.append(daily_return)
        
        if not returns:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize
        annual_return = mean_return * 252
        annual_std = std_return * np.sqrt(252)
        
        sharpe = (annual_return - risk_free_rate) / annual_std
        return sharpe
    
    def get_position_correlation(self, symbols: List[str], data_handler) -> pd.DataFrame:
        """Calculate correlation matrix for positions"""
        try:
            # Get historical data for all symbols
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)
            
            price_data = {}
            for symbol in symbols:
                data = data_handler.get_historical_data(symbol, start_date, end_date)
                if data is not None and not data.empty:
                    price_data[symbol] = data['close']
            
            if len(price_data) < 2:
                return pd.DataFrame()
            
            # Create DataFrame and calculate correlation
            df = pd.DataFrame(price_data).dropna()
            returns = df.pct_change().dropna()
            correlation_matrix = returns.corr()
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating position correlation: {e}")
            return pd.DataFrame()
    
    def check_correlation_risk(self, new_symbol: str, existing_symbols: List[str], 
                             data_handler) -> bool:
        """Check if adding new position would create correlation risk"""
        if not existing_symbols:
            return True
        
        all_symbols = existing_symbols + [new_symbol]
        correlation_matrix = self.get_position_correlation(all_symbols, data_handler)
        
        if correlation_matrix.empty:
            return True
        
        # Check correlations with new symbol
        if new_symbol in correlation_matrix.columns:
            correlations = correlation_matrix[new_symbol].drop(new_symbol)
            max_correlation = correlations.abs().max()
            
            if max_correlation > self.max_correlation:
                logger.warning(f"High correlation risk for {new_symbol}: {max_correlation:.2f}")
                return False
        
        return True
    
    def update_position(self, symbol: str, quantity: float, price: float, action: str):
        """Update position tracking"""
        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': 0,
                'avg_price': 0,
                'value': 0,
                'unrealized_pnl': 0,
                'realized_pnl': 0
            }
        
        position = self.positions[symbol]
        
        if action in ['BUY', 'SELL']:
            # Update position
            if action == 'BUY':
                total_cost = (position['quantity'] * position['avg_price']) + (quantity * price)
                total_quantity = position['quantity'] + quantity
            else:  # SELL
                total_cost = (position['quantity'] * position['avg_price']) - (quantity * price)
                total_quantity = position['quantity'] - quantity
            
            if total_quantity != 0:
                position['avg_price'] = abs(total_cost) / abs(total_quantity)
                position['quantity'] = total_quantity
            else:
                position['avg_price'] = 0
                position['quantity'] = 0
            
            position['value'] = abs(position['quantity']) * price
        
        elif action == 'CLOSE':
            # Close position
            if position['quantity'] != 0:
                realized_pnl = (price - position['avg_price']) * position['quantity']
                position['realized_pnl'] += realized_pnl
            
            position['quantity'] = 0
            position['avg_price'] = 0
            position['value'] = 0
    
    def get_risk_summary(self) -> Dict[str, float]:
        """Get comprehensive risk summary"""
        risk_checks = self.check_portfolio_risk()
        
        summary = {
            'portfolio_value': self.portfolio_value,
            'daily_pnl': self.get_daily_pnl(),
            'daily_pnl_pct': self.get_daily_pnl() / self.portfolio_value,
            'max_drawdown': self.calculate_max_drawdown(),
            'concentration_risk': self.calculate_concentration_risk(),
            'var_5pct': self.calculate_var(0.05),
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'total_positions': len([p for p in self.positions.values() if p['quantity'] != 0]),
            'risk_limits_ok': all(risk_checks.values())
        }
        
        return summary
    
    def should_halt_trading(self) -> Tuple[bool, str]:
        """Check if trading should be halted due to risk limits"""
        risk_checks = self.check_portfolio_risk()
        
        # Check critical risk limits
        if not risk_checks['daily_loss_limit']:
            return True, "Daily loss limit exceeded"
        
        if not risk_checks['max_drawdown_limit']:
            return True, "Maximum drawdown limit exceeded"
        
        # Check if portfolio value is too low
        if self.portfolio_value < 50000:  # Minimum portfolio value
            return True, "Portfolio value below minimum threshold"
        
        return False, ""
    
    def reset_for_backtest(self, initial_value: float = 100000.0):
        """Reset risk manager for backtesting"""
        self.portfolio_value = initial_value
        self.positions = {}
        self.daily_pnl_history = []
        self.risk_metrics = {}
        
        logger.info(f"Risk manager reset for backtesting with initial value: ${initial_value:,.2f}")