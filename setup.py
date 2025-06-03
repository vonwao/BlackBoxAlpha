#!/usr/bin/env python3
"""
Setup script for the trading bot
"""

import os
import sys
import subprocess
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = [
        'data/historical',
        'data/pairs', 
        'data/backtest_results',
        'data/cache',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python version: {sys.version}")

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)

def create_env_file():
    """Create .env file from template if it doesn't exist"""
    if not Path('.env').exists():
        if Path('.env.example').exists():
            import shutil
            shutil.copy('.env.example', '.env')
            print("✅ Created .env file from template")
            print("⚠️  Please edit .env file with your API keys")
        else:
            print("⚠️  .env.example not found, please create .env manually")
    else:
        print("✅ .env file already exists")

def run_tests():
    """Run basic tests"""
    print("🧪 Running tests...")
    try:
        subprocess.check_call([sys.executable, "-m", "pytest", "tests/", "-v"])
        print("✅ All tests passed")
    except subprocess.CalledProcessError:
        print("⚠️  Some tests failed, but setup can continue")
    except FileNotFoundError:
        print("⚠️  pytest not found, skipping tests")

def main():
    """Main setup function"""
    print("🚀 Setting up Trading Bot...")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Create directories
    create_directories()
    
    # Install dependencies
    install_dependencies()
    
    # Create .env file
    create_env_file()
    
    # Run tests
    run_tests()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your API keys")
    print("2. Review config/config.json settings")
    print("3. Run a backtest: python backtest.py --strategy pairs --start-date 2023-01-01 --end-date 2024-01-01")
    print("4. Start paper trading: python main.py --paper-trading")
    print("\n📚 Read README.md for detailed instructions")

if __name__ == "__main__":
    main()