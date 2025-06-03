"""
Unit tests for trading strategies
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from strategies.base_strategy import BaseStrategy
from config.config_manager import ConfigManager
from utils.data_handler import DataHandler
from utils.risk_manager import RiskManager

class MockDataHandler:
    """Mock data handler for testing"""
    
    def get_current_price(self, symbol):
        return 100.0
    
    def get_historical_data(self, symbol, start_date, end_date):
        # Generate mock historical data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(42)  # For reproducible tests
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.01)
        
        return pd.DataFrame({
            'close': prices,
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)

class MockRiskManager:
    """Mock risk manager for testing"""
    
    def get_portfolio_value(self):
        return 100000.0
    
    def get_max_position_size(self, symbol):
        return 1000.0
    
    def check_portfolio_risk(self):
        return {'all_checks': True}

class MockStrategy(BaseStrategy):
    """Mock strategy for testing base functionality"""
    
    def generate_signals(self):
        return [{
            'symbol': 'TEST',
            'action': 'BUY',
            'quantity': 100,
            'price': 100.0,
            'timestamp': datetime.now(),
            'confidence': 0.8
        }]
    
    def calculate_position_size(self, symbol, signal_strength):
        return 100.0

class TestBaseStrategy(unittest.TestCase):
    """Test cases for BaseStrategy"""
    
    def setUp(self):
        self.data_handler = MockDataHandler()
        self.risk_manager = MockRiskManager()
        self.config = {'test': True}
        self.strategy = MockStrategy(
            self.data_handler, 
            self.risk_manager, 
            self.config, 
            "Test Strategy"
        )
    
    def test_strategy_initialization(self):
        """Test strategy initialization"""
        self.assertEqual(self.strategy.name, "Test Strategy")
        self.assertEqual(self.strategy.positions, {})
        self.assertEqual(self.strategy.signals_history, [])
    
    def test_signal_validation(self):
        """Test signal validation"""
        valid_signal = {
            'symbol': 'TEST',
            'action': 'BUY',
            'quantity': 100,
            'price': 100.0,
            'timestamp': datetime.now()
        }
        self.assertTrue(self.strategy.validate_signal(valid_signal))
        
        # Test invalid signal
        invalid_signal = {
            'symbol': 'TEST',
            'action': 'INVALID',
            'quantity': -100,
            'price': 100.0
        }
        self.assertFalse(self.strategy.validate_signal(invalid_signal))
    
    def test_position_update(self):
        """Test position tracking"""
        # Test BUY
        self.strategy.update_position('TEST', 'BUY', 100, 100.0)
        self.assertEqual(self.strategy.positions['TEST']['quantity'], 100)
        self.assertEqual(self.strategy.positions['TEST']['avg_price'], 100.0)
        
        # Test additional BUY
        self.strategy.update_position('TEST', 'BUY', 100, 110.0)
        self.assertEqual(self.strategy.positions['TEST']['quantity'], 200)
        self.assertEqual(self.strategy.positions['TEST']['avg_price'], 105.0)
        
        # Test SELL
        self.strategy.update_position('TEST', 'SELL', 50, 120.0)
        self.assertEqual(self.strategy.positions['TEST']['quantity'], 150)
        self.assertGreater(self.strategy.positions['TEST']['realized_pnl'], 0)
    
    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        # Add some signals to history
        signals = [
            {'pnl': 100}, {'pnl': -50}, {'pnl': 200}, {'pnl': -25}
        ]
        self.strategy.signals_history = signals
        
        metrics = self.strategy.get_performance_metrics()
        self.assertEqual(metrics['total_signals'], 4)
        self.assertEqual(metrics['winning_signals'], 2)
        self.assertEqual(metrics['win_rate'], 0.5)

class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager"""
    
    def test_config_loading(self):
        """Test configuration loading"""
        try:
            config = ConfigManager()
            self.assertIsInstance(config.config, dict)
        except Exception as e:
            self.skipTest(f"Config files not available: {e}")
    
    def test_config_get(self):
        """Test configuration value retrieval"""
        config = ConfigManager()
        # Test with default value
        value = config.get('nonexistent.key', 'default')
        self.assertEqual(value, 'default')

if __name__ == '__main__':
    unittest.main()