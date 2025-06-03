"""
Pairs Analysis Utility
Statistical analysis tools for pairs trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class PairsAnalyzer:
    """
    Advanced statistical analysis for pairs trading
    """
    
    def __init__(self, data_handler):
        self.data_handler = data_handler
        
    def find_cointegrated_pairs(self, symbols: List[str], lookback_days: int = 252,
                               min_correlation: float = 0.7, max_pvalue: float = 0.05) -> List[Dict]:
        """
        Find cointegrated pairs from a list of symbols
        """
        try:
            logger.info(f"Analyzing {len(symbols)} symbols for cointegrated pairs")
            
            # Get historical data for all symbols
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + 30)
            
            price_data = {}
            for symbol in symbols:
                data = self.data_handler.get_historical_data(symbol, start_date, end_date)
                if data is not None and len(data) >= lookback_days:
                    price_data[symbol] = data['close'].tail(lookback_days)
            
            if len(price_data) < 2:
                logger.warning("Insufficient data for pairs analysis")
                return []
            
            # Align all price series
            df = pd.DataFrame(price_data).dropna()
            
            cointegrated_pairs = []
            symbols_list = list(df.columns)
            
            # Test all possible pairs
            for i in range(len(symbols_list)):
                for j in range(i + 1, len(symbols_list)):
                    symbol1, symbol2 = symbols_list[i], symbols_list[j]
                    
                    try:
                        pair_analysis = self.analyze_pair(
                            df[symbol1], df[symbol2], symbol1, symbol2
                        )
                        
                        if (pair_analysis['correlation'] >= min_correlation and 
                            pair_analysis['cointegration_pvalue'] <= max_pvalue):
                            cointegrated_pairs.append(pair_analysis)
                            
                    except Exception as e:
                        logger.warning(f"Error analyzing pair {symbol1}/{symbol2}: {e}")
            
            # Sort by cointegration strength
            cointegrated_pairs.sort(key=lambda x: x['cointegration_pvalue'])
            
            logger.info(f"Found {len(cointegrated_pairs)} cointegrated pairs")
            return cointegrated_pairs
            
        except Exception as e:
            logger.error(f"Error finding cointegrated pairs: {e}")
            return []
    
    def analyze_pair(self, price1: pd.Series, price2: pd.Series, 
                    symbol1: str, symbol2: str) -> Dict:
        """
        Comprehensive analysis of a single pair
        """
        try:
            # Basic correlation
            correlation = price1.corr(price2)
            
            # Cointegration test
            coint_score, coint_pvalue, _ = coint(price1, price2)
            
            # Calculate hedge ratio using linear regression
            X = price1.values.reshape(-1, 1)
            y = price2.values
            reg = LinearRegression().fit(X, y)
            hedge_ratio = reg.coef_[0]
            r_squared = reg.score(X, y)
            
            # Calculate spread
            spread = price2 - hedge_ratio * price1
            
            # Spread statistics
            spread_mean = spread.mean()
            spread_std = spread.std()
            current_spread = spread.iloc[-1]
            zscore = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0
            
            # Stationarity test on spread
            adf_stat, adf_pvalue, _, _, _, _ = adfuller(spread.dropna())
            
            # Half-life of mean reversion
            half_life = self.calculate_half_life(spread)
            
            # Hurst exponent (mean reversion indicator)
            hurst_exponent = self.calculate_hurst_exponent(spread)
            
            # Spread volatility and stability
            spread_volatility = spread.std()
            spread_stability = self.calculate_spread_stability(spread)
            
            # Price ratio analysis
            price_ratio = price2 / price1
            ratio_mean = price_ratio.mean()
            ratio_std = price_ratio.std()
            ratio_zscore = (price_ratio.iloc[-1] - ratio_mean) / ratio_std if ratio_std > 0 else 0
            
            analysis = {
                'symbol1': symbol1,
                'symbol2': symbol2,
                'correlation': correlation,
                'cointegration_score': coint_score,
                'cointegration_pvalue': coint_pvalue,
                'hedge_ratio': hedge_ratio,
                'r_squared': r_squared,
                'spread_mean': spread_mean,
                'spread_std': spread_std,
                'current_spread': current_spread,
                'zscore': zscore,
                'adf_statistic': adf_stat,
                'adf_pvalue': adf_pvalue,
                'half_life': half_life,
                'hurst_exponent': hurst_exponent,
                'spread_volatility': spread_volatility,
                'spread_stability': spread_stability,
                'price_ratio_mean': ratio_mean,
                'price_ratio_std': ratio_std,
                'price_ratio_zscore': ratio_zscore,
                'analysis_date': datetime.now()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing pair {symbol1}/{symbol2}: {e}")
            return {}
    
    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate half-life of mean reversion using Ornstein-Uhlenbeck process
        """
        try:
            spread_clean = spread.dropna()
            if len(spread_clean) < 10:
                return np.inf
            
            # Lag the spread
            spread_lag = spread_clean.shift(1).dropna()
            spread_diff = spread_clean.diff().dropna()
            
            # Align series
            min_len = min(len(spread_lag), len(spread_diff))
            spread_lag = spread_lag.iloc[-min_len:]
            spread_diff = spread_diff.iloc[-min_len:]
            
            # Regression: spread_diff = alpha + beta * spread_lag + error
            X = spread_lag.values.reshape(-1, 1)
            y = spread_diff.values
            
            reg = LinearRegression().fit(X, y)
            beta = reg.coef_[0]
            
            # Calculate half-life
            if beta < 0:
                half_life = -np.log(2) / beta
                return max(0, half_life)  # Ensure positive
            else:
                return np.inf
                
        except Exception as e:
            logger.warning(f"Error calculating half-life: {e}")
            return np.inf
    
    def calculate_hurst_exponent(self, spread: pd.Series, max_lag: int = 20) -> float:
        """
        Calculate Hurst exponent to measure mean reversion tendency
        H < 0.5: Mean reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        """
        try:
            spread_clean = spread.dropna()
            if len(spread_clean) < max_lag * 2:
                return 0.5
            
            lags = range(2, max_lag + 1)
            tau = []
            
            for lag in lags:
                # Calculate variance of lagged differences
                diff = np.diff(spread_clean, n=lag)
                tau.append(np.var(diff))
            
            # Linear regression on log-log plot
            log_lags = np.log(lags)
            log_tau = np.log(tau)
            
            reg = LinearRegression().fit(log_lags.reshape(-1, 1), log_tau)
            hurst = reg.coef_[0] / 2.0
            
            return max(0, min(1, hurst))  # Clamp between 0 and 1
            
        except Exception as e:
            logger.warning(f"Error calculating Hurst exponent: {e}")
            return 0.5
    
    def calculate_spread_stability(self, spread: pd.Series, window: int = 30) -> float:
        """
        Calculate spread stability metric (lower is more stable)
        """
        try:
            if len(spread) < window * 2:
                return 1.0
            
            # Rolling standard deviation
            rolling_std = spread.rolling(window=window).std()
            
            # Coefficient of variation of rolling std
            stability = rolling_std.std() / rolling_std.mean() if rolling_std.mean() > 0 else 1.0
            
            return stability
            
        except Exception as e:
            logger.warning(f"Error calculating spread stability: {e}")
            return 1.0
    
    def backtest_pair(self, symbol1: str, symbol2: str, start_date: datetime, 
                     end_date: datetime, entry_threshold: float = 2.0, 
                     exit_threshold: float = 0.5, stop_loss: float = 4.0) -> Dict:
        """
        Backtest a specific pair with given parameters
        """
        try:
            # Get historical data
            data1 = self.data_handler.get_historical_data(symbol1, start_date, end_date)
            data2 = self.data_handler.get_historical_data(symbol2, start_date, end_date)
            
            if data1 is None or data2 is None:
                return {}
            
            # Align data
            df = pd.DataFrame({
                'price1': data1['close'],
                'price2': data2['close']
            }).dropna()
            
            if len(df) < 50:
                return {}
            
            # Calculate hedge ratio using first 60% of data
            split_point = int(len(df) * 0.6)
            formation_data = df.iloc[:split_point]
            trading_data = df.iloc[split_point:]
            
            # Calculate hedge ratio
            X = formation_data['price1'].values.reshape(-1, 1)
            y = formation_data['price2'].values
            reg = LinearRegression().fit(X, y)
            hedge_ratio = reg.coef_[0]
            
            # Calculate spread for formation period
            formation_spread = formation_data['price2'] - hedge_ratio * formation_data['price1']
            spread_mean = formation_spread.mean()
            spread_std = formation_spread.std()
            
            # Simulate trading on remaining data
            trades = []
            position = 0  # 0: no position, 1: long spread, -1: short spread
            entry_price1 = 0
            entry_price2 = 0
            
            for i, (date, row) in enumerate(trading_data.iterrows()):
                price1, price2 = row['price1'], row['price2']
                spread = price2 - hedge_ratio * price1
                zscore = (spread - spread_mean) / spread_std if spread_std > 0 else 0
                
                if position == 0:  # No position
                    if zscore > entry_threshold:
                        # Short spread (sell stock2, buy stock1)
                        position = -1
                        entry_price1 = price1
                        entry_price2 = price2
                        
                    elif zscore < -entry_threshold:
                        # Long spread (buy stock2, sell stock1)
                        position = 1
                        entry_price1 = price1
                        entry_price2 = price2
                
                else:  # Have position
                    should_exit = False
                    exit_reason = ""
                    
                    # Check exit conditions
                    if abs(zscore) <= exit_threshold:
                        should_exit = True
                        exit_reason = "mean_reversion"
                    elif abs(zscore) >= stop_loss:
                        should_exit = True
                        exit_reason = "stop_loss"
                    
                    if should_exit:
                        # Calculate P&L
                        if position == 1:  # Long spread
                            pnl = (price2 - entry_price2) - hedge_ratio * (price1 - entry_price1)
                        else:  # Short spread
                            pnl = (entry_price2 - price2) - hedge_ratio * (entry_price1 - price1)
                        
                        trades.append({
                            'entry_date': trading_data.index[max(0, i-1)],
                            'exit_date': date,
                            'entry_zscore': zscore,  # Approximate
                            'exit_zscore': zscore,
                            'pnl': pnl,
                            'exit_reason': exit_reason,
                            'position_type': 'long_spread' if position == 1 else 'short_spread'
                        })
                        
                        position = 0
            
            # Calculate performance metrics
            if trades:
                pnls = [trade['pnl'] for trade in trades]
                total_pnl = sum(pnls)
                win_rate = len([pnl for pnl in pnls if pnl > 0]) / len(pnls)
                avg_pnl = np.mean(pnls)
                sharpe = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0
                
                backtest_results = {
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'start_date': start_date,
                    'end_date': end_date,
                    'hedge_ratio': hedge_ratio,
                    'total_trades': len(trades),
                    'total_pnl': total_pnl,
                    'win_rate': win_rate,
                    'avg_pnl_per_trade': avg_pnl,
                    'sharpe_ratio': sharpe,
                    'best_trade': max(pnls),
                    'worst_trade': min(pnls),
                    'trades': trades
                }
            else:
                backtest_results = {
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'total_trades': 0,
                    'total_pnl': 0,
                    'message': 'No trades generated'
                }
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"Error backtesting pair {symbol1}/{symbol2}: {e}")
            return {}
    
    def plot_pair_analysis(self, symbol1: str, symbol2: str, lookback_days: int = 252):
        """
        Create comprehensive plots for pair analysis
        """
        try:
            # Get data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + 30)
            
            data1 = self.data_handler.get_historical_data(symbol1, start_date, end_date)
            data2 = self.data_handler.get_historical_data(symbol2, start_date, end_date)
            
            if data1 is None or data2 is None:
                logger.error("Could not fetch data for plotting")
                return
            
            # Align data
            df = pd.DataFrame({
                'price1': data1['close'],
                'price2': data2['close']
            }).dropna().tail(lookback_days)
            
            # Calculate analysis
            analysis = self.analyze_pair(df['price1'], df['price2'], symbol1, symbol2)
            hedge_ratio = analysis['hedge_ratio']
            spread = df['price2'] - hedge_ratio * df['price1']
            spread_mean = spread.mean()
            spread_std = spread.std()
            zscore = (spread - spread_mean) / spread_std
            
            # Create plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Price series
            ax1_twin = ax1.twinx()
            ax1.plot(df.index, df['price1'], label=symbol1, color='blue')
            ax1_twin.plot(df.index, df['price2'], label=symbol2, color='red')
            ax1.set_title(f'Price Series: {symbol1} vs {symbol2}')
            ax1.set_ylabel(f'{symbol1} Price', color='blue')
            ax1_twin.set_ylabel(f'{symbol2} Price', color='red')
            ax1.legend(loc='upper left')
            ax1_twin.legend(loc='upper right')
            
            # Scatter plot with regression line
            ax2.scatter(df['price1'], df['price2'], alpha=0.6)
            x_range = np.linspace(df['price1'].min(), df['price1'].max(), 100)
            y_pred = hedge_ratio * x_range
            ax2.plot(x_range, y_pred, 'r-', label=f'Hedge Ratio: {hedge_ratio:.3f}')
            ax2.set_xlabel(f'{symbol1} Price')
            ax2.set_ylabel(f'{symbol2} Price')
            ax2.set_title(f'Price Relationship (R²: {analysis["r_squared"]:.3f})')
            ax2.legend()
            
            # Spread and Z-score
            ax3.plot(df.index, spread, label='Spread', color='green')
            ax3.axhline(y=spread_mean, color='black', linestyle='--', alpha=0.7, label='Mean')
            ax3.axhline(y=spread_mean + 2*spread_std, color='red', linestyle='--', alpha=0.7, label='+2σ')
            ax3.axhline(y=spread_mean - 2*spread_std, color='red', linestyle='--', alpha=0.7, label='-2σ')
            ax3.set_title('Spread Over Time')
            ax3.set_ylabel('Spread')
            ax3.legend()
            
            # Z-score
            ax4.plot(df.index, zscore, label='Z-Score', color='purple')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.7)
            ax4.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='Entry Threshold')
            ax4.axhline(y=-2, color='red', linestyle='--', alpha=0.7)
            ax4.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Exit Threshold')
            ax4.axhline(y=-0.5, color='orange', linestyle='--', alpha=0.7)
            ax4.set_title('Z-Score Over Time')
            ax4.set_ylabel('Z-Score')
            ax4.legend()
            
            plt.tight_layout()
            
            # Add text box with statistics
            stats_text = f"""
Correlation: {analysis['correlation']:.3f}
Cointegration p-value: {analysis['cointegration_pvalue']:.4f}
Half-life: {analysis['half_life']:.1f} days
Hurst Exponent: {analysis['hurst_exponent']:.3f}
Current Z-Score: {analysis['zscore']:.2f}
"""
            fig.text(0.02, 0.02, stats_text, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            
            # Save plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_file = f"data/pairs/pair_analysis_{symbol1}_{symbol2}_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Pair analysis plot saved to {plot_file}")
            
        except Exception as e:
            logger.error(f"Error plotting pair analysis: {e}")
    
    def generate_pairs_report(self, pairs_list: List[Dict], output_file: str = None):
        """
        Generate comprehensive pairs analysis report
        """
        try:
            if not output_file:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f"data/pairs/pairs_report_{timestamp}.html"
            
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Pairs Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .good {{ background-color: #d4edda; }}
        .warning {{ background-color: #fff3cd; }}
        .bad {{ background-color: #f8d7da; }}
    </style>
</head>
<body>
    <h1>Pairs Trading Analysis Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>Total Pairs Analyzed: {len(pairs_list)}</p>
    
    <table>
        <tr>
            <th>Pair</th>
            <th>Correlation</th>
            <th>Cointegration p-value</th>
            <th>Hedge Ratio</th>
            <th>Current Z-Score</th>
            <th>Half-Life</th>
            <th>Hurst Exponent</th>
            <th>Quality</th>
        </tr>
"""
            
            for pair in pairs_list:
                # Determine quality
                quality = "Good"
                quality_class = "good"
                
                if (pair['cointegration_pvalue'] > 0.05 or 
                    abs(pair['correlation']) < 0.7 or 
                    pair['half_life'] > 50):
                    quality = "Poor"
                    quality_class = "bad"
                elif (pair['cointegration_pvalue'] > 0.01 or 
                      pair['half_life'] > 30):
                    quality = "Fair"
                    quality_class = "warning"
                
                html_content += f"""
        <tr class="{quality_class}">
            <td>{pair['symbol1']}/{pair['symbol2']}</td>
            <td>{pair['correlation']:.3f}</td>
            <td>{pair['cointegration_pvalue']:.4f}</td>
            <td>{pair['hedge_ratio']:.3f}</td>
            <td>{pair['zscore']:.2f}</td>
            <td>{pair['half_life']:.1f}</td>
            <td>{pair['hurst_exponent']:.3f}</td>
            <td>{quality}</td>
        </tr>
"""
            
            html_content += """
    </table>
</body>
</html>
"""
            
            # Ensure directory exists
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Pairs analysis report saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error generating pairs report: {e}")