"""
Data Handler
Manages data fetching, caching, and processing for the trading bot
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import os
import pickle
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class DataHandler:
    """
    Handles all data operations including fetching, caching, and processing
    """
    
    def __init__(self, cache_dir: str = "data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.cache_dir / "historical").mkdir(exist_ok=True)
        (self.cache_dir / "cache").mkdir(exist_ok=True)
        
        # Data cache
        self.price_cache = {}
        self.historical_cache = {}
        
        # Cache settings
        self.cache_expiry_minutes = 5  # Real-time data cache expiry
        self.historical_cache_days = 1  # Historical data cache expiry
        
        logger.info("Data Handler initialized")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol with caching"""
        try:
            # Check cache first
            if symbol in self.price_cache:
                cached_data = self.price_cache[symbol]
                if self._is_cache_valid(cached_data['timestamp'], self.cache_expiry_minutes):
                    return cached_data['price']
            
            # Fetch from API
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Try different price fields
            price = None
            for field in ['regularMarketPrice', 'currentPrice', 'previousClose', 'ask', 'bid']:
                if field in info and info[field] is not None:
                    price = float(info[field])
                    break
            
            if price is None:
                # Fallback to recent historical data
                hist = ticker.history(period="1d", interval="1m")
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
            
            if price is not None:
                # Cache the result
                self.price_cache[symbol] = {
                    'price': price,
                    'timestamp': datetime.now()
                }
                return price
            else:
                logger.warning(f"Could not fetch current price for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                          interval: str = "1d") -> Optional[pd.DataFrame]:
        """Get historical data for a symbol with caching"""
        try:
            # Create cache key
            cache_key = f"{symbol}_{start_date.date()}_{end_date.date()}_{interval}"
            cache_file = self.cache_dir / "cache" / f"{cache_key}.pkl"
            
            # Check if cached data exists and is valid
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    
                    if self._is_cache_valid(cached_data['timestamp'], self.historical_cache_days * 24 * 60):
                        logger.debug(f"Using cached historical data for {symbol}")
                        return cached_data['data']
                except Exception as e:
                    logger.warning(f"Error loading cached data: {e}")
            
            # Fetch from API
            logger.info(f"Fetching historical data for {symbol} from {start_date.date()} to {end_date.date()}")
            ticker = yf.Ticker(symbol)
            
            # Download data
            data = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval=interval,
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                logger.warning(f"No historical data found for {symbol}")
                return None
            
            # Clean and process data
            data = self._clean_historical_data(data)
            
            # Cache the result
            try:
                cached_data = {
                    'data': data,
                    'timestamp': datetime.now()
                }
                with open(cache_file, 'wb') as f:
                    pickle.dump(cached_data, f)
            except Exception as e:
                logger.warning(f"Error caching data: {e}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    def get_multiple_symbols_data(self, symbols: List[str], start_date: datetime, 
                                end_date: datetime, interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """Get historical data for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            data = self.get_historical_data(symbol, start_date, end_date, interval)
            if data is not None:
                results[symbol] = data
            else:
                logger.warning(f"Failed to fetch data for {symbol}")
        
        return results
    
    def get_intraday_data(self, symbol: str, period: str = "1d", interval: str = "5m") -> Optional[pd.DataFrame]:
        """Get intraday data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No intraday data found for {symbol}")
                return None
            
            return self._clean_historical_data(data)
            
        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {e}")
            return None
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information and metadata"""
        try:
            # Check cache
            cache_key = f"info_{symbol}"
            if cache_key in self.historical_cache:
                cached_data = self.historical_cache[cache_key]
                if self._is_cache_valid(cached_data['timestamp'], 60):  # 1 hour cache
                    return cached_data['data']
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract relevant information
            symbol_info = {
                'symbol': symbol,
                'longName': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'marketCap': info.get('marketCap', 0),
                'volume': info.get('volume', 0),
                'averageVolume': info.get('averageVolume', 0),
                'beta': info.get('beta', 0),
                'trailingPE': info.get('trailingPE', 0),
                'dividendYield': info.get('dividendYield', 0),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', ''),
                'last_updated': datetime.now()
            }
            
            # Cache the result
            self.historical_cache[cache_key] = {
                'data': symbol_info,
                'timestamp': datetime.now()
            }
            
            return symbol_info
            
        except Exception as e:
            logger.error(f"Error fetching symbol info for {symbol}: {e}")
            return None
    
    def calculate_returns(self, data: pd.DataFrame, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """Calculate returns for different periods"""
        returns_data = data.copy()

        for period in periods:
            # Data columns are normalized to lowercase in ``_clean_historical_data``
            # so we access the ``close`` column accordingly.
            returns_data[f'return_{period}d'] = data['close'].pct_change(period)

        return returns_data
    
    def calculate_volatility(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate rolling volatility"""
        # Use the lower-case ``close`` column for consistency
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        return volatility
    
    def resample_data(self, data: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """Resample data to different frequency"""
        try:
            # All columns are stored in lowercase; resample using those names
            resampled = data.resample(frequency).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            return resampled
            
        except Exception as e:
            logger.error(f"Error resampling data: {e}")
            return data
    
    def align_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Align multiple dataframes by date"""
        if not data_dict:
            return {}
        
        # Find common date range
        all_indices = [df.index for df in data_dict.values()]
        common_start = max(idx.min() for idx in all_indices)
        common_end = min(idx.max() for idx in all_indices)
        
        # Align all dataframes
        aligned_data = {}
        for symbol, df in data_dict.items():
            aligned_df = df.loc[common_start:common_end]
            if not aligned_df.empty:
                aligned_data[symbol] = aligned_df
        
        return aligned_data
    
    def save_historical_data(self, symbol: str, data: pd.DataFrame):
        """Save historical data to file"""
        try:
            file_path = self.cache_dir / "historical" / f"{symbol}.csv"
            data.to_csv(file_path)
            logger.info(f"Saved historical data for {symbol} to {file_path}")
        except Exception as e:
            logger.error(f"Error saving historical data for {symbol}: {e}")
    
    def load_historical_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load historical data from file"""
        try:
            file_path = self.cache_dir / "historical" / f"{symbol}.csv"
            if file_path.exists():
                data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                return data
            return None
        except Exception as e:
            logger.error(f"Error loading historical data for {symbol}: {e}")
            return None
    
    def get_market_hours(self) -> Dict[str, datetime]:
        """Get market hours for current day"""
        now = datetime.now()
        
        # US market hours (Eastern Time)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return {
            'market_open': market_open,
            'market_close': market_close,
            'is_market_open': market_open <= now <= market_close
        }
    
    def _clean_historical_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize historical data"""
        # Standardize column names
        data.columns = [col.lower() for col in data.columns]
        
        # Remove any rows with all NaN values
        data = data.dropna(how='all')
        
        # Forward fill missing values
        data = data.fillna(method='ffill')
        
        # Remove any remaining NaN values
        data = data.dropna()
        
        return data
    
    def _is_cache_valid(self, timestamp: datetime, expiry_minutes: int) -> bool:
        """Check if cached data is still valid"""
        return (datetime.now() - timestamp).total_seconds() < (expiry_minutes * 60)
    
    def clear_cache(self):
        """Clear all cached data"""
        self.price_cache.clear()
        self.historical_cache.clear()
        
        # Clear file cache
        cache_dir = self.cache_dir / "cache"
        for file in cache_dir.glob("*.pkl"):
            try:
                file.unlink()
            except Exception as e:
                logger.warning(f"Error deleting cache file {file}: {e}")
        
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        cache_files = list((self.cache_dir / "cache").glob("*.pkl"))
        
        return {
            'price_cache_size': len(self.price_cache),
            'historical_cache_size': len(self.historical_cache),
            'file_cache_size': len(cache_files),
            'total_cache_files': len(cache_files)
        }