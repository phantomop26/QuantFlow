"""
Simplified Machine Learning models for price prediction (focusing on working models)
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import logging
import os
from typing import Tuple, Dict, Any, Optional
import json

from config.settings import MODEL_CONFIG, MODELS_DIR

class ModelTrainer:
    """Handles training and evaluation of ML models (XGBoost and Random Forest focus)"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.models = {}
        self.model_metrics = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for model operations"""
        logger = logging.getLogger('ModelTrainer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            logs_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            
            handler = logging.FileHandler(os.path.join(logs_dir, 'model_trainer.log'))
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def train_xgboost_model(self, X: np.ndarray, y: np.ndarray, 
                           model_name: str = "xgboost_model") -> xgb.XGBRegressor:
        """
        Train XGBoost model
        
        Args:
            X: Input features (reshaped for XGBoost)
            y: Target values
            model_name: Name to save the model
            
        Returns:
            Trained XGBoost model
        """
        try:
            # Set environment for XGBoost
            os.environ['DYLD_LIBRARY_PATH'] = '/Users/funda/homebrew/opt/libomp/lib'
            
            # Reshape X for XGBoost (flatten sequences)
            X_reshaped = X.reshape(X.shape[0], -1)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_reshaped, y, test_size=MODEL_CONFIG.VALIDATION_SPLIT, random_state=42
            )
            
            # Build and train XGBoost model
            model = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                early_stopping_rounds=50,
                eval_metric='rmse'
            )
            
            self.logger.info(f"Training XGBoost model: {model_name}")
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Evaluate model
            train_predictions = model.predict(X_train)
            val_predictions = model.predict(X_val)
            
            metrics = self._calculate_metrics(y_train, train_predictions, y_val, val_predictions)
            self.model_metrics[model_name] = metrics
            
            # Save model
            model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
            joblib.dump(model, model_path)
            
            self.models[model_name] = model
            self.logger.info(f"XGBoost model trained and saved: {model_name}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error training XGBoost model: {str(e)}")
            raise
    
    def train_random_forest_model(self, X: np.ndarray, y: np.ndarray,
                                 model_name: str = "random_forest_model") -> RandomForestRegressor:
        """
        Train Random Forest model
        
        Args:
            X: Input features (reshaped for Random Forest)
            y: Target values
            model_name: Name to save the model
            
        Returns:
            Trained Random Forest model
        """
        try:
            # Reshape X for Random Forest (flatten sequences)
            X_reshaped = X.reshape(X.shape[0], -1)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_reshaped, y, test_size=MODEL_CONFIG.VALIDATION_SPLIT, random_state=42
            )
            
            # Build and train Random Forest model
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            self.logger.info(f"Training Random Forest model: {model_name}")
            model.fit(X_train, y_train)
            
            # Evaluate model
            train_predictions = model.predict(X_train)
            val_predictions = model.predict(X_val)
            
            metrics = self._calculate_metrics(y_train, train_predictions, y_val, val_predictions)
            self.model_metrics[model_name] = metrics
            
            # Save model
            model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
            joblib.dump(model, model_path)
            
            self.models[model_name] = model
            self.logger.info(f"Random Forest model trained and saved: {model_name}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error training Random Forest model: {str(e)}")
            raise
    
    def train_lstm_model(self, X: np.ndarray, y: np.ndarray, 
                        model_name: str = "lstm_model"):
        """
        Train LSTM model (placeholder - tries to import TensorFlow)
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=MODEL_CONFIG.VALIDATION_SPLIT, random_state=42
            )
            
            # Build model
            model = keras.Sequential([
                layers.LSTM(units=MODEL_CONFIG.LSTM_UNITS, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                layers.Dropout(MODEL_CONFIG.DROPOUT_RATE),
                layers.LSTM(units=MODEL_CONFIG.LSTM_UNITS // 2, return_sequences=False),
                layers.Dropout(MODEL_CONFIG.DROPOUT_RATE),
                layers.Dense(units=50, activation='relu'),
                layers.Dense(units=1)
            ])
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Train the model
            self.logger.info(f"Training LSTM model: {model_name}")
            model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=0)
            
            # Evaluate model
            train_predictions = model.predict(X_train).flatten()
            val_predictions = model.predict(X_val).flatten()
            
            metrics = self._calculate_metrics(y_train, train_predictions, y_val, val_predictions)
            self.model_metrics[model_name] = metrics
            
            # Save model
            model_path = os.path.join(MODELS_DIR, f"{model_name}.h5")
            model.save(model_path)
            
            self.models[model_name] = model
            self.logger.info(f"LSTM model trained and saved: {model_name}")
            
            return model
            
        except ImportError:
            self.logger.warning("TensorFlow not available, skipping LSTM model")
            raise ImportError("TensorFlow not available for LSTM training")
        except Exception as e:
            self.logger.error(f"Error training LSTM model: {str(e)}")
            raise
    
    def train_gru_model(self, X: np.ndarray, y: np.ndarray, 
                       model_name: str = "gru_model"):
        """
        Train GRU model (placeholder - tries to import TensorFlow)
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=MODEL_CONFIG.VALIDATION_SPLIT, random_state=42
            )
            
            # Build model
            model = keras.Sequential([
                layers.GRU(units=MODEL_CONFIG.LSTM_UNITS, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                layers.Dropout(MODEL_CONFIG.DROPOUT_RATE),
                layers.GRU(units=MODEL_CONFIG.LSTM_UNITS // 2, return_sequences=False),
                layers.Dropout(MODEL_CONFIG.DROPOUT_RATE),
                layers.Dense(units=50, activation='relu'),
                layers.Dense(units=1)
            ])
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Train the model
            self.logger.info(f"Training GRU model: {model_name}")
            model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=0)
            
            # Evaluate model
            train_predictions = model.predict(X_train).flatten()
            val_predictions = model.predict(X_val).flatten()
            
            metrics = self._calculate_metrics(y_train, train_predictions, y_val, val_predictions)
            self.model_metrics[model_name] = metrics
            
            # Save model
            model_path = os.path.join(MODELS_DIR, f"{model_name}.h5")
            model.save(model_path)
            
            self.models[model_name] = model
            self.logger.info(f"GRU model trained and saved: {model_name}")
            
            return model
            
        except ImportError:
            self.logger.warning("TensorFlow not available, skipping GRU model")
            raise ImportError("TensorFlow not available for GRU training")
        except Exception as e:
            self.logger.error(f"Error training GRU model: {str(e)}")
            raise
    
    def _calculate_metrics(self, y_train: np.ndarray, train_pred: np.ndarray,
                          y_val: np.ndarray, val_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        try:
            metrics = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'train_mae': mean_absolute_error(y_train, train_pred),
                'train_r2': r2_score(y_train, train_pred),
                'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                'val_mae': mean_absolute_error(y_val, val_pred),
                'val_r2': r2_score(y_val, val_pred)
            }
            
            self.logger.info(f"Model metrics calculated: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            raise
    
    def load_model(self, model_name: str, model_type: str = "sklearn") -> Any:
        """
        Load a saved model
        
        Args:
            model_name: Name of the model to load
            model_type: Type of model ('keras', 'sklearn')
            
        Returns:
            Loaded model
        """
        try:
            if model_type == "keras":
                import tensorflow as tf
                model_path = os.path.join(MODELS_DIR, f"{model_name}.h5")
                model = tf.keras.models.load_model(model_path)
            else:
                model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
                model = joblib.load(model_path)
            
            self.models[model_name] = model
            self.logger.info(f"Model loaded successfully: {model_name}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using a trained model
        
        Args:
            model_name: Name of the model to use
            X: Input data for prediction
            
        Returns:
            Predictions array
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
            
            model = self.models[model_name]
            
            # Handle different model types
            if isinstance(model, (xgb.XGBRegressor, RandomForestRegressor)):
                # Reshape for sklearn models
                X_reshaped = X.reshape(X.shape[0], -1)
                predictions = model.predict(X_reshaped)
            else:
                # Keras models
                predictions = model.predict(X)
                if len(predictions.shape) > 1:
                    predictions = predictions.flatten()
            
            self.logger.info(f"Predictions made using model: {model_name}")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all trained models"""
        return self.model_metrics.copy()
    
    def save_metrics(self):
        """Save model metrics to file"""
        try:
            metrics_path = os.path.join(MODELS_DIR, 'model_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(self.model_metrics, f, indent=2)
            
            self.logger.info("Model metrics saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")
            raise
