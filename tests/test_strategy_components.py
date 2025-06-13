import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config.config_manager import ConfigManager
from strategies.pairs_trading import PairsTradingStrategy
from strategies.momentum_strategy import MomentumStrategy
from utils.risk_manager import RiskManager


class MockPairsDataHandler:
    """Mock data handler that returns predefined price data."""

    def __init__(self, df1, df2):
        self.df1 = df1
        self.df2 = df2

    def get_historical_data(self, symbol, start_date, end_date):
        if symbol == "AAA":
            return self.df1
        elif symbol == "BBB":
            return self.df2
        return None

    def get_current_price(self, symbol):
        return 100.0


class MockSimpleDataHandler:
    """Simpler mock for momentum strategy."""

    def get_current_price(self, symbol):
        return 100.0


class TestPairsTradingStatistics(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        dates = pd.date_range(end=datetime.now(), periods=60)
        price1 = np.cumsum(np.random.randn(60)) + 100
        price2 = price1 * 1.1 + np.random.normal(0, 0.5, 60)
        self.df1 = pd.DataFrame({"close": price1}, index=dates)
        self.df2 = pd.DataFrame({"close": price2}, index=dates)

        self.data_handler = MockPairsDataHandler(self.df1, self.df2)
        self.risk_manager = RiskManager()
        self.config = ConfigManager()
        self.strategy = PairsTradingStrategy(self.data_handler, self.risk_manager, self.config)

    def test_calculate_pair_statistics(self):
        stats = self.strategy.calculate_pair_statistics("AAA", "BBB", lookback_days=50)
        self.assertIsNotNone(stats)
        self.assertEqual(stats["symbol1"], "AAA")
        self.assertEqual(stats["symbol2"], "BBB")
        self.assertIn("correlation", stats)
        self.assertGreater(stats["correlation"], 0.8)
        self.assertIn("hedge_ratio", stats)
        self.assertGreaterEqual(stats["cointegration_pvalue"], 0.0)
        self.assertLessEqual(stats["cointegration_pvalue"], 1.0)


class TestMomentumStrategyEntrySignal(unittest.TestCase):
    def setUp(self):
        self.data_handler = MockSimpleDataHandler()
        self.risk_manager = RiskManager()
        self.config = ConfigManager()
        self.strategy = MomentumStrategy(self.data_handler, self.risk_manager, self.config)

    def _bullish_indicators(self):
        return {
            "current_price": 100.0,
            "ema_fast": 105.0,
            "ema_slow": 100.0,
            "macd": 1.5,
            "macd_signal": 1.0,
            "macd_histogram": 0.5,
            "rsi": 50.0,
            "bb_position": 0.5,
            "volume_ratio": 2.0,
            "price_momentum_5": 0.03,
            "price_momentum_10": 0.06,
            "price_momentum_20": 0.08,
            "last_updated": datetime.now(),
        }

    def _bearish_indicators(self):
        return {
            "current_price": 100.0,
            "ema_fast": 95.0,
            "ema_slow": 100.0,
            "macd": -1.5,
            "macd_signal": -1.0,
            "macd_histogram": -0.5,
            "rsi": 80.0,
            "bb_position": 0.1,
            "volume_ratio": 1.0,
            "price_momentum_5": -0.03,
            "price_momentum_10": -0.06,
            "price_momentum_20": -0.08,
            "last_updated": datetime.now(),
        }

    def _neutral_indicators(self):
        return {
            "current_price": 100.0,
            "ema_fast": 99.0,
            "ema_slow": 100.0,
            "macd": 0.5,
            "macd_signal": 0.4,
            "macd_histogram": 0.1,
            "rsi": 50.0,
            "bb_position": 0.5,
            "volume_ratio": 1.4,
            "price_momentum_5": 0.0,
            "price_momentum_10": 0.01,
            "price_momentum_20": 0.02,
            "last_updated": datetime.now(),
        }

    def test_bullish_entry_signal(self):
        indicators = self._bullish_indicators()
        with patch.object(self.strategy, "calculate_position_size", return_value=10.0):
            signal = self.strategy.analyze_entry_signal("AAA", indicators)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["action"], "BUY")

    def test_bearish_entry_signal(self):
        indicators = self._bearish_indicators()
        with patch.object(self.strategy, "calculate_position_size", return_value=10.0):
            signal = self.strategy.analyze_entry_signal("AAA", indicators)
        self.assertIsNotNone(signal)
        self.assertEqual(signal["action"], "SELL")

    def test_no_entry_signal(self):
        indicators = self._neutral_indicators()
        with patch.object(self.strategy, "calculate_position_size", return_value=10.0):
            signal = self.strategy.analyze_entry_signal("AAA", indicators)
        self.assertIsNone(signal)


class TestRiskManagerMetrics(unittest.TestCase):
    def setUp(self):
        self.risk_manager = RiskManager()

    def test_calculate_max_drawdown(self):
        self.risk_manager.daily_pnl_history = [
            {"date": datetime(2024, 1, 1).date(), "pnl": 0, "portfolio_value": 100000},
            {"date": datetime(2024, 1, 2).date(), "pnl": 10000, "portfolio_value": 110000},
            {"date": datetime(2024, 1, 3).date(), "pnl": -20000, "portfolio_value": 90000},
            {"date": datetime(2024, 1, 4).date(), "pnl": 5000, "portfolio_value": 95000},
        ]
        drawdown = self.risk_manager.calculate_max_drawdown()
        expected = (110000 - 90000) / 110000
        self.assertAlmostEqual(drawdown, expected, places=4)

    def test_check_portfolio_risk(self):
        self.risk_manager.daily_pnl_history = [
            {"date": datetime(2024, 1, 1).date(), "pnl": 0, "portfolio_value": 100000},
            {"date": datetime(2024, 1, 2).date(), "pnl": 10000, "portfolio_value": 110000},
            {"date": datetime.now().date(), "pnl": -7000, "portfolio_value": 90000},
        ]
        self.risk_manager.portfolio_value = 90000
        self.risk_manager.positions = {
            "A": {"value": 60000},
            "B": {"value": 30000},
            "C": {"value": 10000},
        }
        checks = self.risk_manager.check_portfolio_risk()
        self.assertFalse(checks["daily_loss_limit"])
        self.assertFalse(checks["max_drawdown_limit"])
        self.assertFalse(checks["concentration_risk"])
        self.assertTrue(checks["correlation_risk"])


if __name__ == "__main__":
    unittest.main()
