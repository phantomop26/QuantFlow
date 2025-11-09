"""
Main application runner for the AI Trading System
"""
import sys
import os
import argparse
from datetime import datetime
import logging

# Add project root and src directory to the path
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.data.data_handler import DataHandler
from src.models.model_trainer import ModelTrainer
from src.models.trading_engine import TradingEngine
from src.utils.helpers import Logger, DataValidator, PerformanceAnalyzer, Scheduler
from config.settings import DATA_CONFIG, TRADING_CONFIG, MODEL_CONFIG

def setup_logging():
    """Setup main application logging"""
    return Logger.setup_logger('MainApp', 'main_app.log')

def train_models_pipeline(symbol: str = None, period: str = None):
    """
    Complete model training pipeline
    
    Args:
        symbol: Stock symbol to train on
        period: Time period for data
    """
    logger = setup_logging()
    symbol = symbol or DATA_CONFIG.DEFAULT_SYMBOL
    period = period or DATA_CONFIG.DEFAULT_PERIOD
    
    logger.info(f"Starting model training pipeline for {symbol}")
    
    try:
        # Set environment variables to avoid TensorFlow issues
        os.environ['DYLD_LIBRARY_PATH'] = '/Users/funda/homebrew/opt/libomp/lib'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        # Initialize components
        data_handler = DataHandler()
        model_trainer = ModelTrainer()
        
        # Fetch and preprocess data
        logger.info("Fetching and preprocessing data...")
        raw_data = data_handler.fetch_data(symbol, period)
        data_with_indicators = data_handler.add_technical_indicators(raw_data)
        processed_data = data_handler.preprocess_data(data_with_indicators)
        
        # Validate data
        is_valid, issues = DataValidator.validate_price_data(processed_data)
        if not is_valid:
            logger.warning(f"Data validation issues: {issues}")
        
        # Save processed data
        data_handler.save_to_database(processed_data, f"{symbol}_processed")
        
        # Create sequences for training (adapt sequence length based on data size)
        logger.info("Creating training sequences...")
        data_size = len(processed_data)
        if data_size < 15:
            logger.error(f"Insufficient data: {data_size} days. Need at least 15 days.")
            return False
        
        # Adapt sequence length based on available data
        sequence_length = min(MODEL_CONFIG.SEQUENCE_LENGTH, data_size // 3)  # Use max 1/3 of data for sequence
        sequence_length = max(sequence_length, 5)  # But at least 5 days
        
        logger.info(f"Using sequence length: {sequence_length} (data size: {data_size})")
        X, y = data_handler.create_sequences(processed_data, sequence_length)
        
        # Validate model input
        is_valid, issues = DataValidator.validate_model_input(X, y)
        if not is_valid:
            logger.error(f"Model input validation failed: {issues}")
            return False
        
        # Train models in order of reliability (skip TensorFlow models due to blocking issues)
        models_to_train = ['xgboost', 'random_forest']  # Only reliable models
        trained_models = {}
        
        for model_type in models_to_train:
            logger.info(f"Training {model_type} model...")
            
            try:
                model_name = f"{model_type}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                if model_type == 'xgboost':
                    model = model_trainer.train_xgboost_model(X, y, model_name)
                elif model_type == 'random_forest':
                    model = model_trainer.train_random_forest_model(X, y, model_name)
                elif model_type == 'lstm':
                    model = model_trainer.train_lstm_model(X, y, model_name)
                elif model_type == 'gru':
                    model = model_trainer.train_gru_model(X, y, model_name)
                
                trained_models[model_name] = model
                logger.info(f"Successfully trained {model_type} model: {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_type} model: {str(e)}")
                # Continue with other models even if one fails
                continue
        
        # Save model metrics
        model_trainer.save_metrics()
        
        # Display results
        performance_metrics = model_trainer.get_model_performance()
        logger.info("Training completed. Model performance:")
        
        for model_name, metrics in performance_metrics.items():
            logger.info(f"{model_name}:")
            logger.info(f"  Validation RMSE: {metrics.get('val_rmse', 0):.4f}")
            logger.info(f"  Validation R²: {metrics.get('val_r2', 0):.4f}")
            logger.info(f"  Validation MAE: {metrics.get('val_mae', 0):.4f}")
        
        return len(trained_models) > 0
        
    except Exception as e:
        logger.error(f"Model training pipeline failed: {str(e)}")
        return False

def run_backtest(symbol: str = None, model_name: str = None, start_date: str = None, end_date: str = None):
    """
    Run backtesting on historical data
    
    Args:
        symbol: Stock symbol
        model_name: Name of the trained model to use
        start_date: Start date for backtesting (YYYY-MM-DD)
        end_date: End date for backtesting (YYYY-MM-DD)
    """
    logger = setup_logging()
    symbol = symbol or DATA_CONFIG.DEFAULT_SYMBOL
    
    logger.info(f"Starting backtest for {symbol}")
    
    try:
        # Initialize components
        data_handler = DataHandler()
        model_trainer = ModelTrainer()
        trading_engine = TradingEngine()
        
        # Load data
        if start_date and end_date:
            # Fetch specific date range
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            data = data_handler.add_technical_indicators(data)
            data = data_handler.preprocess_data(data)
        else:
            # Load from database
            try:
                data = data_handler.load_from_database(f"{symbol}_processed")
            except:
                logger.info("No processed data found, fetching fresh data...")
                data = data_handler.fetch_data(symbol, "2y")
                data = data_handler.add_technical_indicators(data)
                data = data_handler.preprocess_data(data)
        
        # Load model
        if model_name:
            model_type = "keras" if "lstm" in model_name or "gru" in model_name else "sklearn"
            model = model_trainer.load_model(model_name, model_type)
        else:
            logger.error("Model name is required for backtesting")
            return False
        
        # Run backtest
        logger.info("Running backtest simulation...")
        
        # Load the model to check the expected feature count and derive sequence length
        import joblib
        model_path = os.path.join("models", f"{model_name}.pkl")
        temp_model = joblib.load(model_path)
        expected_features = temp_model.n_features_in_
        
        # Calculate the sequence length used during training (11 features per step)
        seq_length = expected_features // 11  # 11 features per timestep
        
        logger.info(f"Model expects {expected_features} features, using sequence length: {seq_length}")
        X, y = data_handler.create_sequences(data, seq_length)
        
        # Generate predictions
        try:
            predictions = model_trainer.predict(model_name, X)
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            logger.info(f"X shape: {X.shape}, expected features: trying with flattened data")
            # Try with flattened features for sklearn models
            X_flat = X.reshape(X.shape[0], -1)
            predictions = model.predict(X_flat)
        
        # Simulate trading
        prices = data['Close'].values[seq_length:]
        
        for i, (actual_price, predicted_price) in enumerate(zip(prices, predictions)):
            # Generate trading signal
            signal = trading_engine.generate_signals(
                predictions[max(0, i-5):i+1], 
                actual_price
            )
            
            # Execute trade
            trading_engine.execute_trade(symbol, signal, actual_price)
            
            # Update positions
            market_data = {symbol: actual_price}
            trading_engine.update_positions(market_data)
            
            # Check stop loss/take profit
            trading_engine.check_stop_loss_take_profit(market_data)
            
            # Save portfolio snapshot periodically
            if i % 10 == 0:
                trading_engine.save_portfolio_snapshot(market_data)
        
        # Analyze performance
        final_market_data = {symbol: prices[-1]}
        portfolio_summary = trading_engine.get_portfolio_summary(final_market_data)
        
        logger.info("Backtest Results:")
        logger.info(f"Initial Capital: ${trading_engine.initial_capital:,.2f}")
        logger.info(f"Final Portfolio Value: ${portfolio_summary.get('total_portfolio_value', 0):,.2f}")
        logger.info(f"Total Return: {portfolio_summary.get('total_return_pct', 0):.2f}%")
        logger.info(f"Number of Trades: {len(trading_engine.trade_history)}")
        
        # Save results
        trading_engine.save_trading_data(f"backtest_{symbol}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        return True
        
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        return False

def run_live_trading(symbol: str = None, model_name: str = None):
    """
    Run live trading (paper trading simulation)
    
    Args:
        symbol: Stock symbol
        model_name: Name of the trained model to use
    """
    logger = setup_logging()
    symbol = symbol or DATA_CONFIG.DEFAULT_SYMBOL
    
    logger.info(f"Starting live trading simulation for {symbol}")
    
    try:
        # Initialize components
        data_handler = DataHandler()
        model_trainer = ModelTrainer()
        trading_engine = TradingEngine()
        scheduler = Scheduler()
        
        # Load model
        if model_name:
            model_type = "keras" if "lstm" in model_name or "gru" in model_name else "sklearn"
            model = model_trainer.load_model(model_name, model_type)
        else:
            logger.error("Model name is required for live trading")
            return False
        
        # Schedule automatic data updates
        scheduler.schedule_data_update([symbol], interval_hours=1)
        scheduler.start()
        
        logger.info("Live trading simulation started. Press Ctrl+C to stop.")
        
        # Main trading loop
        import time
        while True:
            try:
                # Get latest data
                latest_data = data_handler.get_latest_data(symbol, days=100)
                
                if latest_data.empty:
                    logger.warning("No latest data available, skipping iteration")
                    time.sleep(300)  # Wait 5 minutes
                    continue
                
                # Create sequences and predict
                X, _ = data_handler.create_sequences(latest_data, MODEL_CONFIG.SEQUENCE_LENGTH)
                if len(X) > 0:
                    predictions = model_trainer.predict(model_name, X[-1:])  # Last sequence
                    
                    # Get current price
                    current_price = latest_data['Close'].iloc[-1]
                    
                    # Generate and execute trading signal
                    signal = trading_engine.generate_signals(predictions, current_price)
                    
                    if signal.value != "HOLD":
                        success = trading_engine.execute_trade(symbol, signal, current_price)
                        if success:
                            logger.info(f"Executed {signal.value} order for {symbol} at ${current_price:.2f}")
                    
                    # Update positions and check stop loss/take profit
                    market_data = {symbol: current_price}
                    trading_engine.update_positions(market_data)
                    executed_orders = trading_engine.check_stop_loss_take_profit(market_data)
                    
                    for order in executed_orders:
                        logger.info(f"Executed {order['reason']} order: {order}")
                    
                    # Save portfolio snapshot
                    trading_engine.save_portfolio_snapshot(market_data)
                
                # Wait before next iteration
                time.sleep(300)  # 5 minutes
                
            except KeyboardInterrupt:
                logger.info("Stopping live trading...")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(60)  # Wait 1 minute before retrying
        
        # Clean up
        scheduler.stop()
        
        # Save final results
        trading_engine.save_trading_data(f"live_trading_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        logger.info("Live trading simulation stopped")
        return True
        
    except Exception as e:
        logger.error(f"Live trading failed: {str(e)}")
        return False

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description='AI Trading Application')
    parser.add_argument('command', choices=['train', 'backtest', 'live', 'gui'], 
                       help='Command to run')
    parser.add_argument('--symbol', type=str, help='Stock symbol')
    parser.add_argument('--period', type=str, help='Time period for data')
    parser.add_argument('--model', type=str, help='Model name for trading/backtest')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        success = train_models_pipeline(args.symbol, args.period)
        sys.exit(0 if success else 1)
        
    elif args.command == 'backtest':
        if not args.model:
            print("Error: --model is required for backtesting")
            sys.exit(1)
        success = run_backtest(args.symbol, args.model, args.start_date, args.end_date)
        sys.exit(0 if success else 1)
        
    elif args.command == 'live':
        if not args.model:
            print("Error: --model is required for live trading")
            sys.exit(1)
        success = run_live_trading(args.symbol, args.model)
        sys.exit(0 if success else 1)
        
    elif args.command == 'gui':
        # Run Streamlit GUI
        import subprocess
        gui_path = os.path.join(os.path.dirname(__file__), 'src', 'gui', 'streamlit_app.py')
        subprocess.run(['streamlit', 'run', gui_path])

if __name__ == "__main__":
    main()
