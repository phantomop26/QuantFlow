"""
Utility functions for the trading application
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import schedule
import time
import threading

# Add the config directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config.settings import LOGS_DIR, DATA_CONFIG

class Logger:
    """Centralized logging utility"""
    
    @staticmethod
    def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
        """Setup a logger with file and console handlers"""
        
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler
        log_path = os.path.join(LOGS_DIR, log_file)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_price_data(data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate price data for completeness and quality
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        
        # Check for null values
        null_counts = data[required_columns].isnull().sum()
        if null_counts.sum() > 0:
            issues.append(f"Null values found: {null_counts.to_dict()}")
        
        # Check for negative values
        for col in required_columns:
            if col in data.columns and (data[col] < 0).any():
                issues.append(f"Negative values found in {col}")
        
        # Check OHLC logic
        if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            # High should be >= Open, Close, Low
            if (data['High'] < data[['Open', 'Close', 'Low']].max(axis=1)).any():
                issues.append("High price is less than Open/Close/Low in some records")
            
            # Low should be <= Open, Close, High
            if (data['Low'] > data[['Open', 'Close', 'High']].min(axis=1)).any():
                issues.append("Low price is greater than Open/Close/High in some records")
        
        # Check data recency
        if not data.empty:
            last_date = data.index[-1]
            days_old = (datetime.now() - last_date).days
            if days_old > 7:
                issues.append(f"Data is {days_old} days old")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    @staticmethod
    def validate_model_input(X: np.ndarray, y: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Validate model input data
        
        Args:
            X: Input features array
            y: Target values array
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check shapes
        if X.shape[0] != y.shape[0]:
            issues.append(f"X and y have different sample counts: {X.shape[0]} vs {y.shape[0]}")
        
        # Check for NaN/inf values
        if np.isnan(X).any() or np.isinf(X).any():
            issues.append("X contains NaN or infinite values")
        
        if np.isnan(y).any() or np.isinf(y).any():
            issues.append("y contains NaN or infinite values")
        
        # Check minimum sample size (more lenient for small datasets)
        min_samples = 10  # Reduced from 100 for demo purposes
        if X.shape[0] < min_samples:
            issues.append(f"Insufficient training samples: {X.shape[0]} (minimum {min_samples} required)")
        
        # Check feature dimensions
        if len(X.shape) != 3:
            issues.append(f"X should have 3 dimensions, got {len(X.shape)}")
        
        is_valid = len(issues) == 0
        return is_valid, issues

class Scheduler:
    """Task scheduling utility"""
    
    def __init__(self):
        self.logger = Logger.setup_logger('Scheduler', 'scheduler.log')
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the scheduler in a separate thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_scheduler)
            self.thread.daemon = True
            self.thread.start()
            self.logger.info("Scheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self.thread:
            self.thread.join()
        self.logger.info("Scheduler stopped")
    
    def _run_scheduler(self):
        """Run the scheduler loop"""
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def schedule_data_update(self, symbols: List[str], interval_hours: int = 1):
        """
        Schedule automatic data updates
        
        Args:
            symbols: List of symbols to update
            interval_hours: Update interval in hours
        """
        from data.data_handler import DataHandler
        
        data_handler = DataHandler()
        
        def update_job():
            for symbol in symbols:
                try:
                    data_handler.update_data_automatically(symbol)
                    self.logger.info(f"Updated data for {symbol}")
                except Exception as e:
                    self.logger.error(f"Failed to update {symbol}: {str(e)}")
        
        schedule.every(interval_hours).hours.do(update_job)
        self.logger.info(f"Scheduled data updates every {interval_hours} hours for {symbols}")

class PerformanceAnalyzer:
    """Performance analysis utilities"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sharpe ratio
        """
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> float:
        """
        Calculate maximum drawdown
        
        Args:
            prices: Series of prices or portfolio values
            
        Returns:
            Maximum drawdown as percentage
        """
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    @staticmethod
    def calculate_win_rate(trades: List[Dict]) -> float:
        """
        Calculate win rate from trades
        
        Args:
            trades: List of trade dictionaries with 'realized_pnl' key
            
        Returns:
            Win rate as percentage
        """
        if not trades:
            return 0.0
        
        profitable_trades = sum(1 for trade in trades if trade.get('realized_pnl', 0) > 0)
        return (profitable_trades / len(trades)) * 100
    
    @staticmethod
    def analyze_trading_performance(trade_history: List[Dict], 
                                  portfolio_history: List[Dict]) -> Dict:
        """
        Comprehensive trading performance analysis
        
        Args:
            trade_history: List of executed trades
            portfolio_history: List of portfolio snapshots
            
        Returns:
            Dictionary with performance metrics
        """
        if not trade_history or not portfolio_history:
            return {}
        
        # Convert to DataFrames
        trades_df = pd.DataFrame(trade_history)
        portfolio_df = pd.DataFrame([
            {
                'timestamp': p['timestamp'],
                'total_value': p['portfolio_summary'].get('total_portfolio_value', 0)
            }
            for p in portfolio_history
        ])
        
        portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
        portfolio_df = portfolio_df.set_index('timestamp')
        
        # Calculate returns
        portfolio_df['returns'] = portfolio_df['total_value'].pct_change()
        
        # Performance metrics
        total_return = (portfolio_df['total_value'].iloc[-1] - portfolio_df['total_value'].iloc[0]) / portfolio_df['total_value'].iloc[0]
        
        metrics = {
            'total_return': total_return * 100,
            'total_trades': len(trades_df),
            'win_rate': PerformanceAnalyzer.calculate_win_rate(trade_history),
            'sharpe_ratio': PerformanceAnalyzer.calculate_sharpe_ratio(portfolio_df['returns'].dropna()),
            'max_drawdown': PerformanceAnalyzer.calculate_max_drawdown(portfolio_df['total_value']) * 100,
            'average_trade_pnl': trades_df['realized_pnl'].mean() if 'realized_pnl' in trades_df.columns else 0,
            'total_realized_pnl': trades_df['realized_pnl'].sum() if 'realized_pnl' in trades_df.columns else 0
        }
        
        return metrics

class RiskManager:
    """Risk management utilities"""
    
    @staticmethod
    def calculate_position_size(capital: float, entry_price: float, 
                              stop_loss_price: float, risk_pct: float = 0.02) -> float:
        """
        Calculate position size based on risk management
        
        Args:
            capital: Available capital
            entry_price: Entry price
            stop_loss_price: Stop loss price
            risk_pct: Percentage of capital to risk
            
        Returns:
            Position size (number of shares)
        """
        risk_amount = capital * risk_pct
        price_diff = abs(entry_price - stop_loss_price)
        
        if price_diff == 0:
            return 0
        
        position_size = risk_amount / price_diff
        return position_size
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (0.95 for 95% VaR)
            
        Returns:
            VaR value
        """
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def check_correlation_risk(symbols: List[str], data: Dict[str, pd.DataFrame], 
                             threshold: float = 0.7) -> Dict:
        """
        Check correlation risk between positions
        
        Args:
            symbols: List of symbols
            data: Dictionary of symbol -> price data
            threshold: Correlation threshold
            
        Returns:
            Dictionary with correlation analysis
        """
        if len(symbols) < 2:
            return {'high_correlation_pairs': [], 'avg_correlation': 0}
        
        # Create returns matrix
        returns_data = {}
        for symbol in symbols:
            if symbol in data and 'Close' in data[symbol].columns:
                returns_data[symbol] = data[symbol]['Close'].pct_change().dropna()
        
        if len(returns_data) < 2:
            return {'high_correlation_pairs': [], 'avg_correlation': 0}
        
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr()
        
        # Find high correlation pairs
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > threshold:
                    high_corr_pairs.append({
                        'pair': (correlation_matrix.columns[i], correlation_matrix.columns[j]),
                        'correlation': corr_value
                    })
        
        # Average correlation
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        avg_correlation = upper_triangle.stack().mean()
        
        return {
            'high_correlation_pairs': high_corr_pairs,
            'avg_correlation': avg_correlation,
            'correlation_matrix': correlation_matrix.to_dict()
        }

class Visualizer:
    """Visualization utilities"""
    
    @staticmethod
    def plot_model_performance(metrics: Dict[str, Dict], save_path: Optional[str] = None):
        """
        Plot model performance comparison
        
        Args:
            metrics: Dictionary of model_name -> metrics
            save_path: Path to save the plot
        """
        if not metrics:
            return
        
        # Extract metrics for plotting
        model_names = list(metrics.keys())
        val_rmse = [metrics[name].get('val_rmse', 0) for name in model_names]
        val_r2 = [metrics[name].get('val_r2', 0) for name in model_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # RMSE comparison
        ax1.bar(model_names, val_rmse, color='skyblue')
        ax1.set_title('Model Performance - Validation RMSE')
        ax1.set_ylabel('RMSE')
        ax1.tick_params(axis='x', rotation=45)
        
        # R² comparison
        ax2.bar(model_names, val_r2, color='lightgreen')
        ax2.set_title('Model Performance - Validation R²')
        ax2.set_ylabel('R² Score')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_portfolio_performance(portfolio_history: List[Dict], save_path: Optional[str] = None):
        """
        Plot portfolio performance over time
        
        Args:
            portfolio_history: List of portfolio snapshots
            save_path: Path to save the plot
        """
        if not portfolio_history:
            return
        
        # Extract data
        timestamps = [pd.to_datetime(p['timestamp']) for p in portfolio_history]
        portfolio_values = [p['portfolio_summary'].get('total_portfolio_value', 0) for p in portfolio_history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, portfolio_values, linewidth=2, color='blue')
        plt.title('Portfolio Performance Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Add annotations for key metrics
        if len(portfolio_values) > 1:
            initial_value = portfolio_values[0]
            final_value = portfolio_values[-1]
            total_return = ((final_value - initial_value) / initial_value) * 100
            
            plt.annotate(f'Total Return: {total_return:.2f}%',
                        xy=(timestamps[-1], final_value),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

# Global instances
logger = Logger.setup_logger('Utils', 'utils.log')
data_validator = DataValidator()
performance_analyzer = PerformanceAnalyzer()
risk_manager = RiskManager()
visualizer = Visualizer()
