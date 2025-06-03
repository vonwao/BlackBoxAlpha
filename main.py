#!/usr/bin/env python3
"""
Main Trading Bot Runner
Handles multiple strategies with proper error handling and monitoring
"""

import sys
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.config_manager import ConfigManager
from strategies.pairs_trading import PairsTradingStrategy
from strategies.momentum_strategy import MomentumStrategy
from utils.data_handler import DataHandler
from utils.risk_manager import RiskManager
from utils.notifications import NotificationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, paper_trading=True):
        """Initialize the trading bot with all components"""
        self.config = ConfigManager()
        self.data_handler = DataHandler()
        self.risk_manager = RiskManager()
        self.notification_manager = NotificationManager()
        self.strategies = []
        self.paper_trading = paper_trading
        
        # Create logs directory if it doesn't exist
        Path('logs').mkdir(exist_ok=True)
        
        logger.info(f"Trading Bot initialized - Paper Trading: {self.paper_trading}")
        
    def initialize_strategies(self):
        """Initialize all active strategies"""
        try:
            if self.config.get('pairs_trading.enabled', False):
                pairs_strategy = PairsTradingStrategy(
                    data_handler=self.data_handler,
                    risk_manager=self.risk_manager,
                    config=self.config
                )
                self.strategies.append(pairs_strategy)
                logger.info("Pairs trading strategy initialized")
                
            if self.config.get('momentum.enabled', False):
                momentum_strategy = MomentumStrategy(
                    data_handler=self.data_handler,
                    risk_manager=self.risk_manager,
                    config=self.config
                )
                self.strategies.append(momentum_strategy)
                logger.info("Momentum strategy initialized")
                
            if not self.strategies:
                logger.warning("No strategies enabled in configuration")
                
        except Exception as e:
            logger.error(f"Error initializing strategies: {e}")
            raise
    
    def execute_trades(self, signals):
        """Execute trades based on strategy signals"""
        try:
            for signal in signals:
                if self.paper_trading:
                    logger.info(f"PAPER TRADE: {signal}")
                    self.notification_manager.send_trade_alert(signal, paper=True)
                else:
                    # Execute real trade
                    logger.info(f"EXECUTING TRADE: {signal}")
                    # Add actual trade execution logic here
                    self.notification_manager.send_trade_alert(signal, paper=False)
                    
        except Exception as e:
            logger.error(f"Error executing trades: {e}")
            self.notification_manager.send_error_alert(str(e))
    
    def update_monitoring(self):
        """Update monitoring and send periodic reports"""
        try:
            # Update portfolio metrics
            portfolio_value = self.risk_manager.get_portfolio_value()
            daily_pnl = self.risk_manager.get_daily_pnl()
            
            # Log performance metrics
            logger.info(f"Portfolio Value: ${portfolio_value:,.2f}")
            logger.info(f"Daily P&L: ${daily_pnl:,.2f}")
            
            # Send daily summary if configured
            if self.config.get('notifications.daily_summary', False):
                current_time = datetime.now()
                if current_time.hour == 16 and current_time.minute == 0:  # 4 PM market close
                    self.notification_manager.send_daily_summary(portfolio_value, daily_pnl)
                    
        except Exception as e:
            logger.error(f"Error updating monitoring: {e}")
    
    def run_trading_loop(self):
        """Main trading loop"""
        logger.info("Starting trading loop...")
        
        try:
            while True:
                loop_start = time.time()
                
                # Generate signals from all strategies
                for strategy in self.strategies:
                    try:
                        signals = strategy.generate_signals()
                        if signals:
                            self.execute_trades(signals)
                    except Exception as e:
                        logger.error(f"Error in strategy {strategy.__class__.__name__}: {e}")
                        
                # Check portfolio risk
                self.risk_manager.check_portfolio_risk()
                
                # Update monitoring
                self.update_monitoring()
                
                # Calculate sleep time to maintain loop interval
                loop_duration = time.time() - loop_start
                sleep_time = max(0, self.config.get('trading.loop_interval', 60) - loop_duration)
                
                logger.debug(f"Loop completed in {loop_duration:.2f}s, sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
        except Exception as e:
            logger.error(f"Critical error in trading loop: {e}")
            self.notification_manager.send_error_alert(f"Critical error: {str(e)}")
            raise
    
    def run_backtest(self, start_date, end_date, strategy_name=None):
        """Run backtesting for specified period"""
        from utils.backtester import Backtester
        
        backtester = Backtester(
            strategies=self.strategies if not strategy_name else [s for s in self.strategies if strategy_name in s.__class__.__name__.lower()],
            data_handler=self.data_handler,
            start_date=start_date,
            end_date=end_date
        )
        
        results = backtester.run()
        logger.info(f"Backtest completed: {results}")
        return results
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down trading bot...")
        
        # Close all positions if in paper trading mode
        if self.paper_trading:
            logger.info("Closing all paper positions...")
        
        # Send shutdown notification
        self.notification_manager.send_notification("Trading bot shutdown completed")
        logger.info("Trading bot shutdown completed")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Trading Bot')
    parser.add_argument('--paper-trading', action='store_true', default=True,
                       help='Run in paper trading mode (default: True)')
    parser.add_argument('--live-trading', action='store_true',
                       help='Run in live trading mode')
    parser.add_argument('--backtest', action='store_true',
                       help='Run backtesting mode')
    parser.add_argument('--start-date', type=str,
                       help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--strategy', type=str,
                       help='Specific strategy to run/backtest')
    
    args = parser.parse_args()
    
    # Determine trading mode
    paper_trading = not args.live_trading
    
    try:
        # Initialize bot
        bot = TradingBot(paper_trading=paper_trading)
        bot.initialize_strategies()
        
        if args.backtest:
            # Run backtest
            if not args.start_date or not args.end_date:
                logger.error("Backtest requires --start-date and --end-date")
                sys.exit(1)
            
            results = bot.run_backtest(args.start_date, args.end_date, args.strategy)
            print(f"Backtest Results: {results}")
        else:
            # Run live/paper trading
            bot.run_trading_loop()
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        if 'bot' in locals():
            bot.shutdown()

if __name__ == "__main__":
    main()