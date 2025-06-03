#!/usr/bin/env python3
"""
Backtesting Script
Standalone script for running backtests on trading strategies
"""

import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.config_manager import ConfigManager
from strategies.pairs_trading import PairsTradingStrategy
from strategies.momentum_strategy import MomentumStrategy
from utils.data_handler import DataHandler
from utils.risk_manager import RiskManager
from utils.backtester import Backtester
from utils.notifications import NotificationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main backtesting function"""
    parser = argparse.ArgumentParser(description='Trading Strategy Backtester')
    parser.add_argument('--strategy', type=str, choices=['pairs', 'momentum', 'all'], 
                       default='all', help='Strategy to backtest')
    parser.add_argument('--start-date', type=str, required=True,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--initial-capital', type=float, default=100000.0,
                       help='Initial capital for backtesting')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots of results')
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed report')
    parser.add_argument('--config', type=str, default='config/config.json',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        # Validate dates
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        
        if start_date >= end_date:
            logger.error("Start date must be before end date")
            sys.exit(1)
        
        if end_date > datetime.now():
            logger.error("End date cannot be in the future")
            sys.exit(1)
        
        # Initialize components
        logger.info("Initializing backtesting components...")
        config = ConfigManager()
        data_handler = DataHandler()
        risk_manager = RiskManager(config)
        notification_manager = NotificationManager(config)
        
        # Initialize strategies
        strategies = []
        
        if args.strategy in ['pairs', 'all']:
            pairs_strategy = PairsTradingStrategy(
                data_handler=data_handler,
                risk_manager=risk_manager,
                config=config
            )
            strategies.append(pairs_strategy)
            logger.info("Pairs trading strategy added")
        
        if args.strategy in ['momentum', 'all']:
            momentum_strategy = MomentumStrategy(
                data_handler=data_handler,
                risk_manager=risk_manager,
                config=config
            )
            strategies.append(momentum_strategy)
            logger.info("Momentum strategy added")
        
        if not strategies:
            logger.error("No strategies selected for backtesting")
            sys.exit(1)
        
        # Run backtest
        logger.info(f"Starting backtest from {args.start_date} to {args.end_date}")
        logger.info(f"Initial capital: ${args.initial_capital:,.2f}")
        
        backtester = Backtester(
            strategies=strategies,
            data_handler=data_handler,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_capital=args.initial_capital
        )
        
        results = backtester.run()
        
        if not results:
            logger.error("Backtesting failed")
            sys.exit(1)
        
        # Display results
        print("\n" + "="*60)
        print("BACKTESTING RESULTS")
        print("="*60)
        print(f"Strategy: {args.strategy.upper()}")
        print(f"Period: {args.start_date} to {args.end_date}")
        print(f"Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"Final Value: ${results['final_value']:,.2f}")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Annualized Return: {results['annualized_return']:.2%}")
        print(f"Volatility: {results['volatility']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {results['sortino_ratio']:.2f}")
        print(f"Calmar Ratio: {results['calmar_ratio']:.2f}")
        print(f"Maximum Drawdown: {results['max_drawdown']:.2%}")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.1%}")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Best Trade: ${results['best_trade']:,.2f}")
        print(f"Worst Trade: ${results['worst_trade']:,.2f}")
        print("="*60)
        
        # Generate plots if requested
        if args.plot:
            logger.info("Generating plots...")
            backtester.plot_results()
        
        # Send results via notifications if configured
        try:
            notification_manager.send_backtest_results(results)
        except Exception as e:
            logger.warning(f"Could not send notification: {e}")
        
        # Performance analysis
        print("\nPERFORMANCE ANALYSIS:")
        print("-" * 30)
        
        if results['sharpe_ratio'] > 1.0:
            print("✅ Excellent risk-adjusted returns (Sharpe > 1.0)")
        elif results['sharpe_ratio'] > 0.5:
            print("✅ Good risk-adjusted returns (Sharpe > 0.5)")
        else:
            print("⚠️  Poor risk-adjusted returns (Sharpe < 0.5)")
        
        if results['max_drawdown'] < 0.1:
            print("✅ Low drawdown risk (< 10%)")
        elif results['max_drawdown'] < 0.2:
            print("⚠️  Moderate drawdown risk (10-20%)")
        else:
            print("❌ High drawdown risk (> 20%)")
        
        if results['win_rate'] > 0.6:
            print("✅ High win rate (> 60%)")
        elif results['win_rate'] > 0.4:
            print("✅ Reasonable win rate (40-60%)")
        else:
            print("⚠️  Low win rate (< 40%)")
        
        if results['total_trades'] > 50:
            print("✅ Sufficient trade sample size")
        else:
            print("⚠️  Small trade sample size - results may not be statistically significant")
        
        # Strategy-specific analysis
        print("\nSTRATEGY RECOMMENDATIONS:")
        print("-" * 30)
        
        if results['total_return'] > 0.1 and results['sharpe_ratio'] > 0.8:
            print("🚀 Strategy shows strong potential for live trading")
            print("   Consider starting with paper trading to validate")
        elif results['total_return'] > 0.05 and results['sharpe_ratio'] > 0.5:
            print("📈 Strategy shows moderate potential")
            print("   Consider parameter optimization and extended backtesting")
        else:
            print("📉 Strategy needs improvement")
            print("   Consider adjusting parameters or risk management rules")
        
        logger.info("Backtesting completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Backtesting interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Backtesting failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()