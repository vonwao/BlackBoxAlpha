import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_handler import DataHandler

class TestDataHandler(unittest.TestCase):
    """Tests for DataHandler utility methods."""

    def setUp(self):
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        self.handler = DataHandler(cache_dir=self.temp_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_calculate_returns(self):
        """calculate_returns should create correct return columns."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        data = pd.DataFrame({
            'open': [10, 11, 12, 13, 14],
            'high': [11, 12, 13, 14, 15],
            'low': [9, 10, 11, 12, 13],
            'close': [10, 11, 12, 13, 14],
            'volume': [100]*5
        }, index=dates)

        result = self.handler.calculate_returns(data, periods=[1, 2])
        expected_1d = data['close'].pct_change(1)
        expected_2d = data['close'].pct_change(2)

        pd.testing.assert_series_equal(result['return_1d'], expected_1d, check_names=False)
        pd.testing.assert_series_equal(result['return_2d'], expected_2d, check_names=False)

    def test_calculate_volatility(self):
        """calculate_volatility should compute rolling volatility."""
        dates = pd.date_range(start="2023-01-01", periods=25, freq="D")
        data = pd.DataFrame({
            'open': np.arange(25),
            'high': np.arange(25) + 1,
            'low': np.arange(25) - 1,
            'close': np.arange(25),
            'volume': np.random.randint(100, 200, size=25)
        }, index=dates)

        result = self.handler.calculate_volatility(data, window=5)
        expected = data['close'].pct_change().rolling(window=5).std() * np.sqrt(252)

        pd.testing.assert_series_equal(result, expected, check_names=False)


if __name__ == '__main__':
    unittest.main()
