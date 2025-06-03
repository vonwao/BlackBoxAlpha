"""
Base Strategy Class
Abstract base class for all trading strategies
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, data_handler, risk_manager, config, name: str):
        self.data_handler = data_handler
        self.risk_manager = risk_manager
        self.config = config
        self.name = name
        self.positions = {}
        self.signals_history = []
        self.performance_metrics = {}
        
        logger.info(f"Initialized strategy: {self.name}")
    
    @abstractmethod
    def generate_signals(self) -> List[Dict[str, Any]]:
        """
        Generate trading signals
        Returns list of signal dictionaries with format:
        {
            'symbol': str,
            'action': 'BUY'|'SELL'|'CLOSE',
            'quantity': float,
            'price': float,
            'timestamp': datetime,
            'strategy': str,
            'confidence': float,
            'metadata': dict
        }
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, symbol: str, signal_strength: float) -> float:
        """Calculate position size based on risk management rules"""
        pass
    
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate signal before execution"""
        required_fields = ['symbol', 'action', 'quantity', 'price', 'timestamp']
        
        for field in required_fields:
            if field not in signal:
                logger.warning(f"Signal missing required field: {field}")
                return False
        
        # Check if action is valid
        if signal['action'] not in ['BUY', 'SELL', 'CLOSE']:
            logger.warning(f"Invalid action: {signal['action']}")
            return False
        
        # Check if quantity is positive
        if signal['quantity'] <= 0:
            logger.warning(f"Invalid quantity: {signal['quantity']}")
            return False
        
        return True
    
    def add_signal_to_history(self, signal: Dict[str, Any]):
        """Add signal to history for tracking"""
        signal['strategy'] = self.name
        self.signals_history.append(signal)
        
        # Keep only last 1000 signals to prevent memory issues
        if len(self.signals_history) > 1000:
            self.signals_history = self.signals_history[-1000:]
    
    def update_position(self, symbol: str, action: str, quantity: float, price: float):
        """Update position tracking"""
        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': 0,
                'avg_price': 0,
                'unrealized_pnl': 0,
                'realized_pnl': 0
            }
        
        position = self.positions[symbol]
        
        if action == 'BUY':
            # Calculate new average price
            total_cost = (position['quantity'] * position['avg_price']) + (quantity * price)
            total_quantity = position['quantity'] + quantity
            position['avg_price'] = total_cost / total_quantity if total_quantity > 0 else 0
            position['quantity'] = total_quantity
            
        elif action == 'SELL':
            if position['quantity'] >= quantity:
                # Calculate realized P&L
                realized_pnl = (price - position['avg_price']) * quantity
                position['realized_pnl'] += realized_pnl
                position['quantity'] -= quantity
                
                if position['quantity'] == 0:
                    position['avg_price'] = 0
            else:
                logger.warning(f"Insufficient position to sell {quantity} of {symbol}")
        
        elif action == 'CLOSE':
            if position['quantity'] > 0:
                # Close entire position
                realized_pnl = (price - position['avg_price']) * position['quantity']
                position['realized_pnl'] += realized_pnl
                position['quantity'] = 0
                position['avg_price'] = 0
    
    def get_current_positions(self) -> Dict[str, Dict]:
        """Get current positions"""
        return {symbol: pos for symbol, pos in self.positions.items() if pos['quantity'] != 0}
    
    def calculate_unrealized_pnl(self) -> float:
        """Calculate total unrealized P&L"""
        total_unrealized = 0
        
        for symbol, position in self.positions.items():
            if position['quantity'] != 0:
                try:
                    current_price = self.data_handler.get_current_price(symbol)
                    unrealized = (current_price - position['avg_price']) * position['quantity']
                    position['unrealized_pnl'] = unrealized
                    total_unrealized += unrealized
                except Exception as e:
                    logger.warning(f"Could not get current price for {symbol}: {e}")
        
        return total_unrealized
    
    def calculate_realized_pnl(self) -> float:
        """Calculate total realized P&L"""
        return sum(pos['realized_pnl'] for pos in self.positions.values())
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate and return performance metrics"""
        total_signals = len(self.signals_history)
        winning_signals = len([s for s in self.signals_history if s.get('pnl', 0) > 0])
        
        metrics = {
            'total_signals': total_signals,
            'winning_signals': winning_signals,
            'win_rate': winning_signals / total_signals if total_signals > 0 else 0,
            'total_realized_pnl': self.calculate_realized_pnl(),
            'total_unrealized_pnl': self.calculate_unrealized_pnl(),
            'active_positions': len(self.get_current_positions())
        }
        
        self.performance_metrics = metrics
        return metrics
    
    def reset_positions(self):
        """Reset all positions (for backtesting)"""
        self.positions = {}
        self.signals_history = []
        self.performance_metrics = {}
    
    def log_performance(self):
        """Log current performance metrics"""
        metrics = self.get_performance_metrics()
        logger.info(f"Strategy {self.name} Performance:")
        logger.info(f"  Total Signals: {metrics['total_signals']}")
        logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")
        logger.info(f"  Realized P&L: ${metrics['total_realized_pnl']:.2f}")
        logger.info(f"  Unrealized P&L: ${metrics['total_unrealized_pnl']:.2f}")
        logger.info(f"  Active Positions: {metrics['active_positions']}")
    
    def __str__(self) -> str:
        return f"{self.name} Strategy"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}')>"