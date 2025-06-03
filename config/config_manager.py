"""
Configuration Manager
Handles loading and accessing configuration from JSON files
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config = {}
        self.strategies_config = {}
        self.pairs_config = {}
        
        self.load_all_configs()
    
    def load_all_configs(self):
        """Load all configuration files"""
        try:
            # Load main config
            main_config_path = self.config_dir / "config.json"
            if main_config_path.exists():
                with open(main_config_path, 'r') as f:
                    self.config = json.load(f)
            
            # Load strategies config
            strategies_config_path = self.config_dir / "strategies.json"
            if strategies_config_path.exists():
                with open(strategies_config_path, 'r') as f:
                    self.strategies_config = json.load(f)
            
            # Load pairs config
            pairs_config_path = self.config_dir / "pairs_config.json"
            if pairs_config_path.exists():
                with open(pairs_config_path, 'r') as f:
                    self.pairs_config = json.load(f)
                    
            # Load environment variables
            self.load_env_variables()
            
        except Exception as e:
            raise Exception(f"Error loading configuration: {e}")
    
    def load_env_variables(self):
        """Load sensitive configuration from environment variables"""
        env_vars = {
            'alpaca_api_key': os.getenv('ALPACA_API_KEY'),
            'alpaca_secret_key': os.getenv('ALPACA_SECRET_KEY'),
            'alpaca_base_url': os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
            'telegram_bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
            'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID'),
            'email_smtp_server': os.getenv('EMAIL_SMTP_SERVER'),
            'email_username': os.getenv('EMAIL_USERNAME'),
            'email_password': os.getenv('EMAIL_PASSWORD'),
        }
        
        # Add to config under 'credentials' section
        self.config['credentials'] = env_vars
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        Example: get('trading.max_position_size')
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_strategy_config(self, strategy_name: str) -> Dict:
        """Get configuration for a specific strategy"""
        return self.strategies_config.get(strategy_name, {})
    
    def get_pairs_config(self) -> Dict:
        """Get pairs trading configuration"""
        return self.pairs_config
    
    def get_credentials(self, key: str) -> Optional[str]:
        """Get credential from environment variables"""
        return self.config.get('credentials', {}).get(key)
    
    def is_paper_trading(self) -> bool:
        """Check if paper trading is enabled"""
        return self.get('trading.paper_trading', True)
    
    def get_max_position_size(self) -> float:
        """Get maximum position size"""
        return self.get('trading.max_position_size', 0.1)
    
    def get_max_portfolio_risk(self) -> float:
        """Get maximum portfolio risk"""
        return self.get('trading.max_portfolio_risk', 0.02)
    
    def get_loop_interval(self) -> int:
        """Get trading loop interval in seconds"""
        return self.get('trading.loop_interval', 60)
    
    def is_strategy_enabled(self, strategy_name: str) -> bool:
        """Check if a strategy is enabled"""
        return self.get(f'{strategy_name}.enabled', False)
    
    def get_notification_settings(self) -> Dict:
        """Get notification settings"""
        return self.get('notifications', {})
    
    def update_config(self, key: str, value: Any):
        """Update configuration value"""
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            main_config_path = self.config_dir / "config.json"
            
            # Remove credentials before saving
            config_to_save = self.config.copy()
            if 'credentials' in config_to_save:
                del config_to_save['credentials']
            
            with open(main_config_path, 'w') as f:
                json.dump(config_to_save, f, indent=2)
                
        except Exception as e:
            raise Exception(f"Error saving configuration: {e}")
    
    def validate_config(self) -> bool:
        """Validate configuration completeness"""
        required_keys = [
            'trading.max_position_size',
            'trading.max_portfolio_risk',
            'trading.loop_interval'
        ]
        
        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
        
        if missing_keys:
            raise Exception(f"Missing required configuration keys: {missing_keys}")
        
        return True
    
    def __str__(self) -> str:
        """String representation of configuration"""
        config_copy = self.config.copy()
        # Remove sensitive information
        if 'credentials' in config_copy:
            config_copy['credentials'] = {k: '***' if v else None for k, v in config_copy['credentials'].items()}
        
        return json.dumps(config_copy, indent=2)