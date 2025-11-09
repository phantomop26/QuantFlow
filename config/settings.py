"""
Configuration settings for the trading application
"""
import os
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ModelConfig:
    """Configuration for ML models"""
    LSTM_UNITS: int = 50
    DROPOUT_RATE: float = 0.2
    EPOCHS: int = 100
    BATCH_SIZE: int = 32
    SEQUENCE_LENGTH: int = 60
    VALIDATION_SPLIT: float = 0.2
    LEARNING_RATE: float = 0.001

@dataclass
class DataConfig:
    """Configuration for data handling"""
    DEFAULT_SYMBOL: str = "AAPL"
    DEFAULT_PERIOD: str = "5y"  # 5 years of data
    UPDATE_INTERVAL: int = 3600  # Update every hour (in seconds)
    FEATURE_COLUMNS: List[str] = None
    
    def __post_init__(self):
        if self.FEATURE_COLUMNS is None:
            self.FEATURE_COLUMNS = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'SMA_20', 'EMA_20', 'RSI', 'MACD', 'BB_upper', 'BB_lower'
            ]

@dataclass
class TradingConfig:
    """Configuration for trading logic"""
    INITIAL_CAPITAL: float = 10000.0
    MAX_POSITION_SIZE: float = 0.1  # 10% of portfolio
    STOP_LOSS_PCT: float = 0.05  # 5% stop loss
    TAKE_PROFIT_PCT: float = 0.15  # 15% take profit
    CONFIDENCE_THRESHOLD: float = 0.6  # Minimum confidence for trades

@dataclass
class APIConfig:
    """Configuration for external APIs"""
    ALPHA_VANTAGE_KEY: str = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    YAHOO_FINANCE_ENABLED: bool = True
    
# Global configuration instances
MODEL_CONFIG = ModelConfig()
DATA_CONFIG = DataConfig()
TRADING_CONFIG = TradingConfig()
API_CONFIG = APIConfig()

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)
