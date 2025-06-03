"""
Notifications Manager
Handles Telegram, email, and other notification systems
"""

import logging
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from typing import Dict, List, Optional
from datetime import datetime
import requests
import json

logger = logging.getLogger(__name__)

class NotificationManager:
    """
    Manages all notification systems for the trading bot
    """
    
    def __init__(self, config=None):
        self.config = config
        
        # Telegram settings
        self.telegram_enabled = config.get('notifications.telegram_enabled', False) if config else False
        self.telegram_bot_token = config.get_credentials('telegram_bot_token') if config else None
        self.telegram_chat_id = config.get_credentials('telegram_chat_id') if config else None
        
        # Email settings
        self.email_enabled = config.get('notifications.email_enabled', False) if config else False
        self.smtp_server = config.get_credentials('email_smtp_server') if config else None
        self.email_username = config.get_credentials('email_username') if config else None
        self.email_password = config.get_credentials('email_password') if config else None
        
        # Notification preferences
        self.trade_alerts = config.get('notifications.trade_alerts', True) if config else True
        self.error_alerts = config.get('notifications.error_alerts', True) if config else True
        self.daily_summary = config.get('notifications.daily_summary', True) if config else True
        
        logger.info("Notification Manager initialized")
    
    def send_trade_alert(self, signal: Dict, paper: bool = True):
        """Send trade execution alert"""
        if not self.trade_alerts:
            return
        
        try:
            trade_type = "PAPER" if paper else "LIVE"
            
            message = f"""
🔔 {trade_type} TRADE ALERT
━━━━━━━━━━━━━━━━━━━━
📊 Symbol: {signal['symbol']}
🎯 Action: {signal['action']}
📈 Quantity: {signal['quantity']:.2f}
💰 Price: ${signal['price']:.2f}
⏰ Time: {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
🎲 Confidence: {signal.get('confidence', 0):.1%}
📝 Strategy: {signal.get('strategy', 'Unknown')}
"""
            
            # Add metadata if available
            if 'metadata' in signal:
                metadata = signal['metadata']
                if 'zscore' in metadata:
                    message += f"📊 Z-Score: {metadata['zscore']:.2f}\n"
                if 'rsi' in metadata:
                    message += f"📈 RSI: {metadata['rsi']:.1f}\n"
                if 'exit_reason' in metadata:
                    message += f"🚪 Exit Reason: {metadata['exit_reason']}\n"
            
            self.send_notification(message)
            
        except Exception as e:
            logger.error(f"Error sending trade alert: {e}")
    
    def send_error_alert(self, error_message: str):
        """Send error alert"""
        if not self.error_alerts:
            return
        
        try:
            message = f"""
🚨 ERROR ALERT
━━━━━━━━━━━━━━━━━━━━
⚠️ Error: {error_message}
⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🤖 Bot: Trading Bot
"""
            
            self.send_notification(message, urgent=True)
            
        except Exception as e:
            logger.error(f"Error sending error alert: {e}")
    
    def send_daily_summary(self, portfolio_value: float, daily_pnl: float, 
                          positions: Dict = None, performance: Dict = None):
        """Send daily performance summary"""
        if not self.daily_summary:
            return
        
        try:
            pnl_emoji = "📈" if daily_pnl >= 0 else "📉"
            pnl_pct = (daily_pnl / portfolio_value) * 100
            
            message = f"""
📊 DAILY SUMMARY
━━━━━━━━━━━━━━━━━━━━
💼 Portfolio Value: ${portfolio_value:,.2f}
{pnl_emoji} Daily P&L: ${daily_pnl:,.2f} ({pnl_pct:+.2f}%)
📅 Date: {datetime.now().strftime('%Y-%m-%d')}
"""
            
            # Add position summary
            if positions:
                active_positions = len([p for p in positions.values() if p.get('quantity', 0) != 0])
                message += f"🎯 Active Positions: {active_positions}\n"
                
                # Top positions
                sorted_positions = sorted(
                    [(symbol, pos) for symbol, pos in positions.items() if pos.get('quantity', 0) != 0],
                    key=lambda x: abs(x[1].get('value', 0)),
                    reverse=True
                )[:5]
                
                if sorted_positions:
                    message += "\n📈 Top Positions:\n"
                    for symbol, pos in sorted_positions:
                        value = pos.get('value', 0)
                        pnl = pos.get('unrealized_pnl', 0)
                        message += f"• {symbol}: ${value:,.0f} (P&L: ${pnl:+,.0f})\n"
            
            # Add performance metrics
            if performance:
                message += f"\n📊 Performance Metrics:\n"
                if 'sharpe_ratio' in performance:
                    message += f"• Sharpe Ratio: {performance['sharpe_ratio']:.2f}\n"
                if 'max_drawdown' in performance:
                    message += f"• Max Drawdown: {performance['max_drawdown']:.1%}\n"
                if 'win_rate' in performance:
                    message += f"• Win Rate: {performance['win_rate']:.1%}\n"
            
            self.send_notification(message)
            
        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")
    
    def send_risk_alert(self, risk_type: str, details: str):
        """Send risk management alert"""
        try:
            message = f"""
⚠️ RISK ALERT
━━━━━━━━━━━━━━━━━━━━
🚨 Risk Type: {risk_type}
📋 Details: {details}
⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            self.send_notification(message, urgent=True)
            
        except Exception as e:
            logger.error(f"Error sending risk alert: {e}")
    
    def send_strategy_alert(self, strategy_name: str, message_text: str):
        """Send strategy-specific alert"""
        try:
            message = f"""
🎯 STRATEGY ALERT
━━━━━━━━━━━━━━━━━━━━
📊 Strategy: {strategy_name}
📝 Message: {message_text}
⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            self.send_notification(message)
            
        except Exception as e:
            logger.error(f"Error sending strategy alert: {e}")
    
    def send_notification(self, message: str, urgent: bool = False):
        """Send notification via all enabled channels"""
        try:
            # Send via Telegram
            if self.telegram_enabled:
                self.send_telegram_message(message)
            
            # Send via Email
            if self.email_enabled:
                subject = "🚨 URGENT: Trading Bot Alert" if urgent else "📊 Trading Bot Notification"
                self.send_email(subject, message)
            
            # Log the notification
            logger.info(f"Notification sent: {message[:100]}...")
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    def send_telegram_message(self, message: str):
        """Send message via Telegram"""
        if not self.telegram_enabled or not self.telegram_bot_token or not self.telegram_chat_id:
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.debug("Telegram message sent successfully")
            else:
                logger.error(f"Failed to send Telegram message: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
    
    def send_email(self, subject: str, message: str):
        """Send email notification"""
        if not self.email_enabled or not all([self.smtp_server, self.email_username, self.email_password]):
            return
        
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = self.email_username
            msg['To'] = self.email_username  # Send to self
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MimeText(message, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, 587)
            server.starttls()
            server.login(self.email_username, self.email_password)
            
            text = msg.as_string()
            server.sendmail(self.email_username, self.email_username, text)
            server.quit()
            
            logger.debug("Email sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
    
    def send_startup_notification(self):
        """Send bot startup notification"""
        try:
            message = f"""
🚀 TRADING BOT STARTED
━━━━━━━━━━━━━━━━━━━━
⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🤖 Status: Online
📊 Mode: {'Paper Trading' if self.config and self.config.is_paper_trading() else 'Live Trading'}
"""
            
            self.send_notification(message)
            
        except Exception as e:
            logger.error(f"Error sending startup notification: {e}")
    
    def send_shutdown_notification(self):
        """Send bot shutdown notification"""
        try:
            message = f"""
🛑 TRADING BOT STOPPED
━━━━━━━━━━━━━━━━━━━━
⏰ Stop Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🤖 Status: Offline
"""
            
            self.send_notification(message)
            
        except Exception as e:
            logger.error(f"Error sending shutdown notification: {e}")
    
    def send_backtest_results(self, results: Dict):
        """Send backtesting results"""
        try:
            message = f"""
📊 BACKTEST RESULTS
━━━━━━━━━━━━━━━━━━━━
📈 Total Return: {results.get('total_return', 0):.2%}
📉 Max Drawdown: {results.get('max_drawdown', 0):.2%}
🎯 Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}
📊 Win Rate: {results.get('win_rate', 0):.1%}
🔢 Total Trades: {results.get('total_trades', 0)}
⏰ Period: {results.get('start_date', '')} to {results.get('end_date', '')}
"""
            
            self.send_notification(message)
            
        except Exception as e:
            logger.error(f"Error sending backtest results: {e}")
    
    def test_notifications(self):
        """Test all notification channels"""
        test_message = f"""
🧪 NOTIFICATION TEST
━━━━━━━━━━━━━━━━━━━━
✅ This is a test message
⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🤖 All systems operational
"""
        
        try:
            self.send_notification(test_message)
            logger.info("Test notification sent successfully")
            return True
        except Exception as e:
            logger.error(f"Test notification failed: {e}")
            return False
    
    def get_notification_status(self) -> Dict[str, bool]:
        """Get status of notification channels"""
        return {
            'telegram_enabled': self.telegram_enabled and bool(self.telegram_bot_token and self.telegram_chat_id),
            'email_enabled': self.email_enabled and bool(self.smtp_server and self.email_username and self.email_password),
            'trade_alerts': self.trade_alerts,
            'error_alerts': self.error_alerts,
            'daily_summary': self.daily_summary
        }