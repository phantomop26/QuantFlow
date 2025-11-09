"""
Data handling module for fetching, preprocessing, and managing financial data
"""
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import sys

# Add the config directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config.settings import DATA_CONFIG, RAW_DATA_DIR, PROCESSED_DATA_DIR

class DataHandler:
    """Handles all data operations including fetching, preprocessing, and storage"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.logger = self._setup_logger()
        self.db_path = os.path.join(PROCESSED_DATA_DIR, 'trading_data.db')
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for data operations"""
        logger = logging.getLogger('DataHandler')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Ensure logs directory exists
            logs_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            
            handler = logging.FileHandler(os.path.join(logs_dir, 'data_handler.log'))
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def fetch_data(self, symbol: str, period: str = "5y") -> pd.DataFrame:
        """
        Fetch historical data from Yahoo Finance
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            self.logger.info(f"Fetching data for {symbol} with period {period}")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
                
            # Remove timezone info to avoid issues
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            # Save raw data
            raw_file_path = os.path.join(RAW_DATA_DIR, f"{symbol}_{period}.csv")
            data.to_csv(raw_file_path)
            self.logger.info(f"Raw data saved to {raw_file_path}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataset
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators added
        """
        try:
            df = data.copy()
            
            # Simple Moving Averages
            df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            
            # Exponential Moving Average
            df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
            
            # Relative Strength Index
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['Close'], window=20)
            df['BB_upper'] = bollinger.bollinger_hband()
            df['BB_lower'] = bollinger.bollinger_lband()
            df['BB_middle'] = bollinger.bollinger_mavg()
            
            # Stochastic Oscillator
            df['Stoch_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
            df['Stoch_D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
            
            # Average True Range
            df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            
            # Price change indicators
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_Change_2'] = df['Close'].pct_change(periods=2)
            df['Price_Change_5'] = df['Close'].pct_change(periods=5)
            
            self.logger.info("Technical indicators added successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {str(e)}")
            raise
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data
        
        Args:
            data: Raw DataFrame with OHLCV and technical indicators
            
        Returns:
            Cleaned and preprocessed DataFrame
        """
        try:
            df = data.copy()
            
            # Remove any rows with all NaN values
            df = df.dropna(how='all')
            
            # Forward fill missing values for most columns
            df = df.ffill()
            
            # For small datasets, be more lenient with NaN handling
            if len(df) < 50:
                # Only drop rows where essential columns (OHLCV) have NaNs
                essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                available_essential = [col for col in essential_cols if col in df.columns]
                df = df.dropna(subset=available_essential)
                
                # Fill remaining NaN values with forward fill + backward fill
                df = df.ffill().bfill()
            else:
                # Drop remaining NaN values (usually at the beginning due to technical indicators)
                df = df.dropna()
            
            # Remove outliers using IQR method for volume (only for larger datasets)
            if 'Volume' in df.columns and len(df) > 50:  # Only remove outliers if we have enough data
                Q1 = df['Volume'].quantile(0.25)
                Q3 = df['Volume'].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:  # Only proceed if there's actual variance
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df = df[(df['Volume'] >= lower_bound) & (df['Volume'] <= upper_bound)]
            
            self.logger.info(f"Data preprocessed. Shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def create_sequences(self, data: pd.DataFrame, sequence_length: int = 60, 
                        target_column: str = 'Close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction
        
        Args:
            data: Preprocessed DataFrame
            sequence_length: Length of input sequences
            target_column: Column to predict
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        try:
            # Select feature columns
            feature_cols = DATA_CONFIG.FEATURE_COLUMNS
            available_cols = [col for col in feature_cols if col in data.columns]
            
            if not available_cols:
                raise ValueError("No feature columns available in data")
            
            # Prepare the data
            features = data[available_cols].values
            target = data[target_column].values
            
            # Scale the features
            features_scaled = self.scaler.fit_transform(features)
            
            # Save the scaler
            scaler_path = os.path.join(PROCESSED_DATA_DIR, 'feature_scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            
            # Create sequences
            X, y = [], []
            for i in range(sequence_length, len(features_scaled)):
                X.append(features_scaled[i-sequence_length:i])
                y.append(target[i])
            
            X, y = np.array(X), np.array(y)
            
            self.logger.info(f"Created sequences. X shape: {X.shape}, y shape: {y.shape}")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error creating sequences: {str(e)}")
            raise
    
    def save_to_database(self, data: pd.DataFrame, table_name: str):
        """
        Save data to SQLite database
        
        Args:
            data: DataFrame to save
            table_name: Name of the table
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                data.to_sql(table_name, conn, if_exists='replace', index=True)
            
            self.logger.info(f"Data saved to database table: {table_name}")
            
        except Exception as e:
            self.logger.error(f"Error saving to database: {str(e)}")
            raise
    
    def load_from_database(self, table_name: str) -> pd.DataFrame:
        """
        Load data from SQLite database
        
        Args:
            table_name: Name of the table
            
        Returns:
            DataFrame with loaded data
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                data = pd.read_sql_query(f"SELECT * FROM {table_name}", conn, index_col=0)
            
            self.logger.info(f"Data loaded from database table: {table_name}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading from database: {str(e)}")
            raise
    
    def get_latest_data(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """
        Get the latest data for real-time predictions
        
        Args:
            symbol: Stock symbol
            days: Number of recent days to get
            
        Returns:
            DataFrame with recent data
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if not data.empty:
                data = self.add_technical_indicators(data)
                data = self.preprocess_data(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting latest data: {str(e)}")
            raise
    
    def update_data_automatically(self, symbol: str):
        """
        Update data automatically for a given symbol
        
        Args:
            symbol: Stock symbol to update
        """
        try:
            # Fetch latest data
            latest_data = self.get_latest_data(symbol)
            
            if not latest_data.empty:
                # Save to database
                table_name = f"{symbol}_latest"
                self.save_to_database(latest_data, table_name)
                
                # Save processed data to CSV
                processed_file = os.path.join(PROCESSED_DATA_DIR, f"{symbol}_processed.csv")
                latest_data.to_csv(processed_file)
                
                self.logger.info(f"Data updated successfully for {symbol}")
            else:
                self.logger.warning(f"No latest data available for {symbol}")
                
        except Exception as e:
            self.logger.error(f"Error updating data automatically: {str(e)}")
            raise
