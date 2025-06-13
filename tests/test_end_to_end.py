import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config.config_manager import ConfigManager
from utils.risk_manager import RiskManager
from utils.backtester import Backtester
from strategies.momentum_strategy import MomentumStrategy


class MockDataHandler:
    """Provide deterministic price data for testing."""

    def get_current_price(self, symbol):
        # Return the last price from generated data
        return 110.0

    def get_historical_data(self, symbol, start_date, end_date, interval="1d"):
        # Ignore dates and return 200 days of synthetic data
        dates = pd.date_range(end=datetime.now(), periods=200, freq="D")
        prices = np.linspace(100, 120, len(dates))
        data = pd.DataFrame({
            "close": prices,
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "volume": np.random.randint(1000000, 5000000, len(dates)),
        }, index=dates)
        return data


class TestMomentumStrategy(MomentumStrategy):
    """Momentum strategy with a single test symbol."""

    def get_tradeable_symbols(self):
        return ["TEST"]


class EndToEndBacktestTest(unittest.TestCase):
    def test_backtest_runs(self):
        config = ConfigManager()
        data_handler = MockDataHandler()
        risk_manager = RiskManager(config)
        strategy = TestMomentumStrategy(data_handler, risk_manager, config)

        start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        backtester = Backtester(
            strategies=[strategy],
            data_handler=data_handler,
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000.0,
        )

        results = backtester.run()

        self.assertIn("final_value", results)
        self.assertIn("portfolio_history", results)
        self.assertGreater(results["final_value"], 0)


if __name__ == "__main__":
    unittest.main()
