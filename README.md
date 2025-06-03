# Python Trading Bot

A comprehensive algorithmic trading bot featuring pairs trading and momentum strategies with advanced risk management, backtesting, and deployment capabilities.

## 🚀 Features

### Trading Strategies
- **Pairs Trading**: Statistical arbitrage using cointegrated pairs
- **Momentum Strategy**: Trend following with technical indicators
- **Multi-strategy Support**: Run multiple strategies simultaneously

### Risk Management
- Position sizing based on portfolio risk
- Maximum drawdown controls
- Correlation risk monitoring
- Daily loss limits
- Stop-loss and take-profit mechanisms

### Data & Analytics
- Real-time and historical data fetching
- Advanced statistical analysis for pairs
- Comprehensive backtesting engine
- Performance metrics and reporting

### Notifications & Monitoring
- Telegram alerts for trades and errors
- Email notifications
- Daily performance summaries
- Risk alerts and monitoring

### Deployment Options
- Docker containerization
- Heroku deployment
- Railway deployment
- Local development setup

## 📁 Project Structure

```
trading-bot/
├── config/                     # Configuration files
│   ├── config.json            # Main configuration
│   ├── strategies.json        # Strategy parameters
│   └── pairs_config.json      # Pairs trading config
├── data/                      # Data storage
│   ├── historical/           # Historical price data
│   ├── pairs/               # Pairs analysis results
│   └── backtest_results/    # Backtest outputs
├── strategies/               # Trading strategies
│   ├── base_strategy.py     # Base strategy class
│   ├── pairs_trading.py     # Pairs trading strategy
│   └── momentum_strategy.py # Momentum strategy
├── utils/                   # Utility modules
│   ├── data_handler.py      # Data fetching/processing
│   ├── pairs_analysis.py    # Statistical pair analysis
│   ├── risk_manager.py      # Risk management
│   ├── backtester.py       # Backtesting engine
│   └── notifications.py    # Alert system
├── deploy/                  # Deployment configurations
│   ├── Dockerfile          # Docker configuration
│   ├── docker-compose.yml  # Multi-service setup
│   ├── heroku.yml          # Heroku deployment
│   └── railway.toml        # Railway deployment
├── main.py                 # Main bot runner
├── backtest.py            # Backtesting script
└── requirements.txt       # Python dependencies
```

## 🛠️ Installation & Setup

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd trading-bot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file or set environment variables:
```bash
export ALPACA_API_KEY="your_alpaca_api_key"
export ALPACA_SECRET_KEY="your_alpaca_secret_key"
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"  # Paper trading
export TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
export TELEGRAM_CHAT_ID="your_telegram_chat_id"
```

### 5. Configure Settings
Edit [`config/config.json`](config/config.json) to customize:
- Trading parameters
- Risk management settings
- Strategy configurations
- Notification preferences

## 🚦 Quick Start

### Paper Trading (Recommended)
```bash
python main.py --paper-trading
```

### Backtesting
```bash
# Backtest pairs strategy
python backtest.py --strategy pairs --start-date 2023-01-01 --end-date 2024-01-01

# Backtest all strategies
python backtest.py --strategy all --start-date 2023-01-01 --end-date 2024-01-01 --plot
```

### Live Trading (Use with caution)
```bash
python main.py --live-trading
```

## 📊 Strategy Details

### Pairs Trading Strategy
- **Methodology**: Statistical arbitrage using cointegrated stock pairs
- **Entry Signal**: Z-score > 2.0 (configurable)
- **Exit Signal**: Z-score < 0.5 (configurable)
- **Stop Loss**: Z-score > 4.0 (configurable)
- **Risk Management**: Maximum 5% position size per pair

**Key Features**:
- Automatic pair discovery and validation
- Cointegration testing
- Half-life calculation for mean reversion
- Hedge ratio optimization

### Momentum Strategy
- **Methodology**: Trend following with multiple technical indicators
- **Indicators**: MACD, RSI, Moving Averages, Bollinger Bands
- **Entry Conditions**: Multiple indicator confirmation
- **Exit Conditions**: Technical reversal or profit/loss targets

**Key Features**:
- Multi-timeframe analysis
- Volume confirmation
- Dynamic position sizing
- Trend strength measurement

## 🔧 Configuration

### Main Configuration ([`config/config.json`](config/config.json))
```json
{
  "trading": {
    "paper_trading": true,
    "max_position_size": 0.1,
    "max_portfolio_risk": 0.02,
    "loop_interval": 60
  },
  "pairs_trading": {
    "enabled": true,
    "min_correlation": 0.7,
    "entry_threshold": 2.0,
    "exit_threshold": 0.5,
    "stop_loss": 4.0
  },
  "momentum": {
    "enabled": true,
    "fast_ma": 12,
    "slow_ma": 26,
    "rsi_period": 14
  }
}
```

### Strategy Parameters ([`config/strategies.json`](config/strategies.json))
Detailed strategy-specific parameters including:
- Lookback periods
- Risk management rules
- Technical indicator settings
- Position sizing rules

### Pairs Configuration ([`config/pairs_config.json`](config/pairs_config.json))
- Predefined trading pairs
- Screening criteria
- Sector exposure limits
- Correlation thresholds

## 📈 Backtesting

The backtesting engine provides comprehensive analysis:

### Performance Metrics
- Total and annualized returns
- Sharpe, Sortino, and Calmar ratios
- Maximum drawdown
- Win rate and profit factor
- Trade-level statistics

### Usage Examples
```bash
# Basic backtest
python backtest.py --strategy pairs --start-date 2023-01-01 --end-date 2024-01-01

# Detailed analysis with plots
python backtest.py --strategy all --start-date 2023-01-01 --end-date 2024-01-01 --plot --report

# Custom initial capital
python backtest.py --strategy momentum --start-date 2023-06-01 --end-date 2024-01-01 --initial-capital 50000
```

## 🔔 Notifications

### Telegram Setup
1. Create a Telegram bot via [@BotFather](https://t.me/botfather)
2. Get your chat ID by messaging [@userinfobot](https://t.me/userinfobot)
3. Set environment variables:
   ```bash
   export TELEGRAM_BOT_TOKEN="your_bot_token"
   export TELEGRAM_CHAT_ID="your_chat_id"
   ```

### Email Setup
Configure SMTP settings in environment variables:
```bash
export EMAIL_SMTP_SERVER="smtp.gmail.com"
export EMAIL_USERNAME="your_email@gmail.com"
export EMAIL_PASSWORD="your_app_password"
```

## 🐳 Docker Deployment

### Local Docker
```bash
cd deploy
docker-compose up -d
```

### Production Deployment
```bash
# Build and run
docker-compose -f deploy/docker-compose.yml up -d

# View logs
docker-compose logs -f trading-bot

# Stop services
docker-compose down
```

## ☁️ Cloud Deployment

### Heroku
```bash
# Install Heroku CLI
heroku create your-trading-bot
heroku stack:set container
git push heroku main
```

### Railway
```bash
# Install Railway CLI
railway login
railway init
railway up
```

## 📊 Monitoring & Maintenance

### Daily Tasks
- ✅ Check Telegram notifications
- ✅ Review trading performance
- ✅ Monitor system health

### Weekly Tasks
- 📊 Analyze strategy performance
- 🔄 Review and update pair selections
- 📈 Check risk metrics

### Monthly Tasks
- 🧪 Full strategy backtesting
- 📋 Performance attribution analysis
- ⚙️ Strategy parameter optimization

## 🛡️ Safety Features

### Built-in Protections
- **Position Limits**: Maximum position size controls
- **Risk Controls**: Portfolio-level risk monitoring
- **Stop Losses**: Automatic loss limitation
- **Paper Trading**: Safe testing environment
- **Error Handling**: Comprehensive error management
- **Alerts**: Real-time notification system

### Risk Management
- Daily loss limits (5% default)
- Maximum drawdown limits (15% default)
- Position correlation monitoring
- Sector concentration limits
- Volatility-adjusted position sizing

## 🧪 Testing

### Run Unit Tests
```bash
pytest tests/ -v
```

### Strategy Testing
```bash
# Test pairs analysis
python -m utils.pairs_analysis

# Test notifications
python -c "from utils.notifications import NotificationManager; nm = NotificationManager(); nm.test_notifications()"
```

## 📚 API Documentation

### Key Classes

#### [`TradingBot`](main.py)
Main orchestrator class that coordinates all components.

#### [`PairsTradingStrategy`](strategies/pairs_trading.py)
Implements statistical arbitrage using cointegrated pairs.

#### [`MomentumStrategy`](strategies/momentum_strategy.py)
Trend-following strategy with technical indicators.

#### [`RiskManager`](utils/risk_manager.py)
Handles position sizing and portfolio risk controls.

#### [`DataHandler`](utils/data_handler.py)
Manages data fetching, caching, and processing.

#### [`Backtester`](utils/backtester.py)
Comprehensive backtesting engine with performance analytics.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ⚠️ Disclaimer

This trading bot is for educational and research purposes. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Always:

- Start with paper trading
- Understand the strategies before deploying capital
- Never risk more than you can afford to lose
- Consider consulting with a financial advisor
- Comply with all applicable regulations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📖 Documentation: Check this README and code comments
- 🐛 Issues: Open an issue on GitHub
- 💬 Discussions: Use GitHub Discussions for questions
- 📧 Contact: [Your contact information]

## 🙏 Acknowledgments

- [Alpaca Markets](https://alpaca.markets/) for trading API
- [yfinance](https://github.com/ranaroussi/yfinance) for market data
- [FreqTrade](https://www.freqtrade.io/) for inspiration
- Open source community for various libraries

---

**Happy Trading! 📈🚀**

*Remember: The best strategy is the one you understand and can stick with through different market conditions.*