"""
Backtesting Engine
Comprehensive backtesting framework for trading strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

class Backtester:
    """
    Backtesting engine for trading strategies
    """
    
    def __init__(self, strategies: List, data_handler, start_date: str, end_date: str,
                 initial_capital: float = 100000.0):
        self.strategies = strategies
        self.data_handler = data_handler
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.initial_capital = initial_capital
        
        # Backtesting results
        self.portfolio_history = []
        self.trades_history = []
        self.daily_returns = []
        self.results = {}
        
        # Create results directory
        self.results_dir = Path("data/backtest_results")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info(f"Backtester initialized for period {start_date} to {end_date}")
    
    def run(self) -> Dict:
        """Run the backtest"""
        try:
            logger.info("Starting backtest...")
            
            # Initialize portfolio
            portfolio_value = self.initial_capital
            positions = {}
            cash = self.initial_capital
            
            # Reset strategies for backtesting
            for strategy in self.strategies:
                strategy.reset_positions()
                if hasattr(strategy.risk_manager, 'reset_for_backtest'):
                    strategy.risk_manager.reset_for_backtest(self.initial_capital)
            
            # Generate date range for backtesting
            current_date = self.start_date
            
            while current_date <= self.end_date:
                # Skip weekends
                if current_date.weekday() >= 5:
                    current_date += timedelta(days=1)
                    continue
                
                # Process each strategy
                for strategy in self.strategies:
                    try:
                        # Generate signals for current date
                        signals = strategy.generate_signals()
                        
                        # Execute signals
                        for signal in signals:
                            trade_result = self.execute_backtest_trade(
                                signal, current_date, positions, cash
                            )
                            
                            if trade_result:
                                cash = trade_result['remaining_cash']
                                positions = trade_result['positions']
                                self.trades_history.append(trade_result['trade_record'])
                    
                    except Exception as e:
                        logger.error(f"Error processing strategy {strategy.name} on {current_date}: {e}")
                
                # Calculate portfolio value
                portfolio_value = self.calculate_portfolio_value(positions, cash, current_date)
                
                # Record daily portfolio state
                self.portfolio_history.append({
                    'date': current_date,
                    'portfolio_value': portfolio_value,
                    'cash': cash,
                    'positions_value': portfolio_value - cash,
                    'num_positions': len([p for p in positions.values() if p['quantity'] != 0])
                })
                
                # Calculate daily return
                if len(self.portfolio_history) > 1:
                    prev_value = self.portfolio_history[-2]['portfolio_value']
                    daily_return = (portfolio_value - prev_value) / prev_value
                    self.daily_returns.append(daily_return)
                
                current_date += timedelta(days=1)
            
            # Calculate final results
            self.results = self.calculate_performance_metrics()
            
            # Generate report
            self.generate_backtest_report()
            
            logger.info("Backtest completed successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {}
    
    def execute_backtest_trade(self, signal: Dict, date: datetime, positions: Dict, 
                             cash: float) -> Optional[Dict]:
        """Execute a trade in the backtest"""
        try:
            symbol = signal['symbol']
            action = signal['action']
            quantity = signal['quantity']
            price = signal['price']
            
            # Initialize position if doesn't exist
            if symbol not in positions:
                positions[symbol] = {'quantity': 0, 'avg_price': 0}
            
            position = positions[symbol]
            trade_cost = 0
            commission = 1.0  # $1 commission per trade
            
            if action == 'BUY':
                trade_value = quantity * price + commission
                if cash >= trade_value:
                    # Update position
                    total_cost = (position['quantity'] * position['avg_price']) + (quantity * price)
                    total_quantity = position['quantity'] + quantity
                    position['avg_price'] = total_cost / total_quantity
                    position['quantity'] = total_quantity
                    
                    cash -= trade_value
                    trade_cost = trade_value
                else:
                    logger.warning(f"Insufficient cash for BUY {symbol}: need ${trade_value:.2f}, have ${cash:.2f}")
                    return None
            
            elif action == 'SELL':
                if position['quantity'] >= quantity:
                    trade_value = quantity * price - commission
                    
                    # Calculate realized P&L
                    realized_pnl = (price - position['avg_price']) * quantity
                    
                    # Update position
                    position['quantity'] -= quantity
                    if position['quantity'] == 0:
                        position['avg_price'] = 0
                    
                    cash += trade_value
                    trade_cost = -trade_value
                else:
                    logger.warning(f"Insufficient position for SELL {symbol}: need {quantity}, have {position['quantity']}")
                    return None
            
            elif action == 'CLOSE':
                if position['quantity'] != 0:
                    quantity = abs(position['quantity'])
                    trade_value = quantity * price - commission
                    
                    # Calculate realized P&L
                    realized_pnl = (price - position['avg_price']) * position['quantity']
                    
                    # Close position
                    position['quantity'] = 0
                    position['avg_price'] = 0
                    
                    cash += trade_value
                    trade_cost = -trade_value
                else:
                    return None
            
            # Create trade record
            trade_record = {
                'date': date,
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'trade_value': abs(trade_cost),
                'commission': commission,
                'strategy': signal.get('strategy', 'Unknown'),
                'confidence': signal.get('confidence', 0),
                'metadata': signal.get('metadata', {})
            }
            
            return {
                'trade_record': trade_record,
                'positions': positions,
                'remaining_cash': cash
            }
            
        except Exception as e:
            logger.error(f"Error executing backtest trade: {e}")
            return None
    
    def calculate_portfolio_value(self, positions: Dict, cash: float, date: datetime) -> float:
        """Calculate total portfolio value"""
        positions_value = 0
        
        for symbol, position in positions.items():
            if position['quantity'] != 0:
                try:
                    # Get price for the date (simplified - using current price)
                    current_price = self.data_handler.get_current_price(symbol)
                    if current_price:
                        positions_value += abs(position['quantity']) * current_price
                except Exception as e:
                    logger.warning(f"Could not get price for {symbol}: {e}")
        
        return cash + positions_value
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.portfolio_history or not self.daily_returns:
            return {}
        
        # Portfolio values
        portfolio_values = [record['portfolio_value'] for record in self.portfolio_history]
        dates = [record['date'] for record in self.portfolio_history]
        
        # Basic metrics
        final_value = portfolio_values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Annualized return
        days = (self.end_date - self.start_date).days
        years = days / 365.25
        annualized_return = (final_value / self.initial_capital) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        daily_returns_array = np.array(self.daily_returns)
        volatility = np.std(daily_returns_array) * np.sqrt(252)  # Annualized
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        max_drawdown = self.calculate_max_drawdown(portfolio_values)
        
        # Win rate and trade statistics
        trade_stats = self.calculate_trade_statistics()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio
        downside_returns = daily_returns_array[daily_returns_array < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        results = {
            'start_date': self.start_date.strftime('%Y-%m-%d'),
            'end_date': self.end_date.strftime('%Y-%m-%d'),
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades_history),
            'win_rate': trade_stats.get('win_rate', 0),
            'avg_win': trade_stats.get('avg_win', 0),
            'avg_loss': trade_stats.get('avg_loss', 0),
            'profit_factor': trade_stats.get('profit_factor', 0),
            'best_trade': trade_stats.get('best_trade', 0),
            'worst_trade': trade_stats.get('worst_trade', 0),
            'avg_trade_duration': trade_stats.get('avg_duration', 0),
            'portfolio_history': self.portfolio_history,
            'trades_history': self.trades_history
        }
        
        return results
    
    def calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def calculate_trade_statistics(self) -> Dict:
        """Calculate trade-level statistics"""
        if not self.trades_history:
            return {}
        
        # Group trades by symbol to calculate P&L
        symbol_trades = {}
        for trade in self.trades_history:
            symbol = trade['symbol']
            if symbol not in symbol_trades:
                symbol_trades[symbol] = []
            symbol_trades[symbol].append(trade)
        
        # Calculate P&L for each completed trade pair
        completed_trades = []
        for symbol, trades in symbol_trades.items():
            position = 0
            entry_price = 0
            entry_date = None
            
            for trade in trades:
                if trade['action'] == 'BUY':
                    if position == 0:
                        entry_price = trade['price']
                        entry_date = trade['date']
                    position += trade['quantity']
                elif trade['action'] in ['SELL', 'CLOSE']:
                    if position > 0:
                        exit_price = trade['price']
                        exit_date = trade['date']
                        pnl = (exit_price - entry_price) * min(position, trade['quantity'])
                        duration = (exit_date - entry_date).days if entry_date else 0
                        
                        completed_trades.append({
                            'symbol': symbol,
                            'pnl': pnl,
                            'duration': duration,
                            'entry_price': entry_price,
                            'exit_price': exit_price
                        })
                        
                        position -= trade['quantity']
                        if position <= 0:
                            position = 0
                            entry_price = 0
                            entry_date = None
        
        if not completed_trades:
            return {}
        
        # Calculate statistics
        pnls = [trade['pnl'] for trade in completed_trades]
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]
        
        win_rate = len(winning_trades) / len(pnls) if pnls else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf')
        
        durations = [trade['duration'] for trade in completed_trades if trade['duration'] > 0]
        avg_duration = np.mean(durations) if durations else 0
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'best_trade': max(pnls) if pnls else 0,
            'worst_trade': min(pnls) if pnls else 0,
            'avg_duration': avg_duration
        }
    
    def generate_backtest_report(self):
        """Generate comprehensive backtest report"""
        try:
            # Create report filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = self.results_dir / f"backtest_report_{timestamp}.html"
            
            # Generate HTML report
            html_content = self.create_html_report()
            
            with open(report_file, 'w') as f:
                f.write(html_content)
            
            # Save results as JSON
            import json
            json_file = self.results_dir / f"backtest_results_{timestamp}.json"
            
            # Convert datetime objects to strings for JSON serialization
            results_copy = self.results.copy()
            if 'portfolio_history' in results_copy:
                for record in results_copy['portfolio_history']:
                    record['date'] = record['date'].isoformat()
            if 'trades_history' in results_copy:
                for trade in results_copy['trades_history']:
                    trade['date'] = trade['date'].isoformat()
            
            with open(json_file, 'w') as f:
                json.dump(results_copy, f, indent=2, default=str)
            
            logger.info(f"Backtest report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Error generating backtest report: {e}")
    
    def create_html_report(self) -> str:
        """Create HTML backtest report"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; text-align: center; }}
        .metric h3 {{ margin: 0; color: #333; }}
        .metric p {{ margin: 5px 0; font-size: 18px; font-weight: bold; }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Trading Bot Backtest Report</h1>
        <p>Period: {self.results.get('start_date', '')} to {self.results.get('end_date', '')}</p>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="metrics">
        <div class="metric">
            <h3>Total Return</h3>
            <p class="{'positive' if self.results.get('total_return', 0) >= 0 else 'negative'}">
                {self.results.get('total_return', 0):.2%}
            </p>
        </div>
        <div class="metric">
            <h3>Annualized Return</h3>
            <p class="{'positive' if self.results.get('annualized_return', 0) >= 0 else 'negative'}">
                {self.results.get('annualized_return', 0):.2%}
            </p>
        </div>
        <div class="metric">
            <h3>Sharpe Ratio</h3>
            <p>{self.results.get('sharpe_ratio', 0):.2f}</p>
        </div>
        <div class="metric">
            <h3>Max Drawdown</h3>
            <p class="negative">{self.results.get('max_drawdown', 0):.2%}</p>
        </div>
        <div class="metric">
            <h3>Win Rate</h3>
            <p>{self.results.get('win_rate', 0):.1%}</p>
        </div>
        <div class="metric">
            <h3>Total Trades</h3>
            <p>{self.results.get('total_trades', 0)}</p>
        </div>
    </div>
    
    <h2>Performance Summary</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Initial Capital</td><td>${self.results.get('initial_capital', 0):,.2f}</td></tr>
        <tr><td>Final Value</td><td>${self.results.get('final_value', 0):,.2f}</td></tr>
        <tr><td>Volatility</td><td>{self.results.get('volatility', 0):.2%}</td></tr>
        <tr><td>Sortino Ratio</td><td>{self.results.get('sortino_ratio', 0):.2f}</td></tr>
        <tr><td>Calmar Ratio</td><td>{self.results.get('calmar_ratio', 0):.2f}</td></tr>
        <tr><td>Profit Factor</td><td>{self.results.get('profit_factor', 0):.2f}</td></tr>
        <tr><td>Best Trade</td><td>${self.results.get('best_trade', 0):,.2f}</td></tr>
        <tr><td>Worst Trade</td><td>${self.results.get('worst_trade', 0):,.2f}</td></tr>
    </table>
    
</body>
</html>
"""
        return html
    
    def plot_results(self):
        """Plot backtest results"""
        try:
            if not self.portfolio_history:
                return
            
            # Extract data
            dates = [record['date'] for record in self.portfolio_history]
            values = [record['portfolio_value'] for record in self.portfolio_history]
            
            # Create plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Portfolio value over time
            ax1.plot(dates, values, linewidth=2)
            ax1.set_title('Portfolio Value Over Time')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.grid(True, alpha=0.3)
            
            # Daily returns distribution
            if self.daily_returns:
                ax2.hist(self.daily_returns, bins=50, alpha=0.7, edgecolor='black')
                ax2.set_title('Daily Returns Distribution')
                ax2.set_xlabel('Daily Return')
                ax2.set_ylabel('Frequency')
                ax2.grid(True, alpha=0.3)
            
            # Drawdown chart
            peak_values = np.maximum.accumulate(values)
            drawdowns = [(peak - val) / peak for peak, val in zip(peak_values, values)]
            ax3.fill_between(dates, drawdowns, 0, alpha=0.3, color='red')
            ax3.set_title('Drawdown Over Time')
            ax3.set_ylabel('Drawdown (%)')
            ax3.grid(True, alpha=0.3)
            
            # Monthly returns heatmap (if enough data)
            if len(self.daily_returns) > 30:
                monthly_returns = self.calculate_monthly_returns()
                if monthly_returns:
                    sns.heatmap(monthly_returns, annot=True, fmt='.1%', cmap='RdYlGn', 
                              center=0, ax=ax4)
                    ax4.set_title('Monthly Returns Heatmap')
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_file = self.results_dir / f"backtest_plots_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Backtest plots saved to {plot_file}")
            
        except Exception as e:
            logger.error(f"Error plotting results: {e}")
    
    def calculate_monthly_returns(self) -> Optional[pd.DataFrame]:
        """Calculate monthly returns for heatmap"""
        try:
            df = pd.DataFrame(self.portfolio_history)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Resample to monthly
            monthly = df['portfolio_value'].resample('M').last()
            monthly_returns = monthly.pct_change().dropna()
            
            # Create pivot table for heatmap
            monthly_returns.index = pd.to_datetime(monthly_returns.index)
            pivot_data = monthly_returns.groupby([
                monthly_returns.index.year,
                monthly_returns.index.month
            ]).first().unstack()
            
            return pivot_data
            
        except Exception as e:
            logger.error(f"Error calculating monthly returns: {e}")
            return None