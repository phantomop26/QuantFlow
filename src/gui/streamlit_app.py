"""
Streamlit GUI for the trading application
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import logging
import os
import sys

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.data_handler import DataHandler
from models.model_trainer import ModelTrainer
from models.trading_engine import TradingEngine, TradeAction
from config.settings import DATA_CONFIG, TRADING_CONFIG, MODEL_CONFIG

# Configure Streamlit page
st.set_page_config(
    page_title="AI Trading Application",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-alert {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-alert {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

class TradingApp:
    """Main trading application class"""
    
    def __init__(self):
        self.data_handler = DataHandler()
        self.model_trainer = ModelTrainer()
        self.trading_engine = None
        
        # Initialize session state
        if 'trading_engine' not in st.session_state:
            st.session_state.trading_engine = TradingEngine()
        if 'trained_models' not in st.session_state:
            st.session_state.trained_models = {}
        if 'current_data' not in st.session_state:
            st.session_state.current_data = None
        if 'predictions' not in st.session_state:
            st.session_state.predictions = None
    
    def run(self):
        """Main application runner"""
        st.markdown('<h1 class="main-header">🤖 AI Trading Application</h1>', unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Main content area
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dashboard", 
            "📈 Data & Analysis", 
            "🧠 Model Training", 
            "💰 Trading", 
            "📋 Portfolio"
        ])
        
        with tab1:
            self.render_dashboard()
        
        with tab2:
            self.render_data_analysis()
        
        with tab3:
            self.render_model_training()
        
        with tab4:
            self.render_trading()
        
        with tab5:
            self.render_portfolio()
    
    def render_sidebar(self):
        """Render the sidebar with controls"""
        st.sidebar.title("🔧 Controls")
        
        # Symbol selection
        st.sidebar.subheader("Market Data")
        symbol = st.sidebar.text_input("Stock Symbol", value=DATA_CONFIG.DEFAULT_SYMBOL)
        period = st.sidebar.selectbox(
            "Time Period", 
            ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
            index=4  # Default to 2y
        )
        
        # Data refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            with st.spinner("Fetching latest data..."):
                try:
                    data = self.data_handler.fetch_data(symbol, period)
                    data_with_indicators = self.data_handler.add_technical_indicators(data)
                    processed_data = self.data_handler.preprocess_data(data_with_indicators)
                    st.session_state.current_data = processed_data
                    st.session_state.current_symbol = symbol
                    st.sidebar.success("Data refreshed successfully!")
                except Exception as e:
                    st.sidebar.error(f"Error fetching data: {str(e)}")
        
        # Model selection
        st.sidebar.subheader("Model Settings")
        model_type = st.sidebar.selectbox(
            "Model Type",
            ["LSTM", "GRU", "XGBoost", "Random Forest"]
        )
        st.session_state.selected_model_type = model_type
        
        # Trading settings
        st.sidebar.subheader("Trading Settings")
        initial_capital = st.sidebar.number_input(
            "Initial Capital ($)", 
            value=TRADING_CONFIG.INITIAL_CAPITAL,
            min_value=1000.0,
            step=1000.0
        )
        
        if st.sidebar.button("Reset Portfolio"):
            st.session_state.trading_engine = TradingEngine(initial_capital)
            st.sidebar.success("Portfolio reset!")
    
    def render_dashboard(self):
        """Render the main dashboard"""
        st.header("📊 Trading Dashboard")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Symbol", getattr(st.session_state, 'current_symbol', 'N/A'))
        
        with col2:
            if hasattr(st.session_state, 'trading_engine'):
                portfolio_value = st.session_state.trading_engine.current_capital
                st.metric("Portfolio Value", f"${portfolio_value:,.2f}")
        
        with col3:
            num_models = len(st.session_state.trained_models)
            st.metric("Trained Models", num_models)
        
        with col4:
            if hasattr(st.session_state.trading_engine, 'positions'):
                num_positions = len(st.session_state.trading_engine.positions)
                st.metric("Active Positions", num_positions)
        
        # Recent activity
        st.subheader("Recent Activity")
        if hasattr(st.session_state.trading_engine, 'trade_history') and st.session_state.trading_engine.trade_history:
            recent_trades = pd.DataFrame(st.session_state.trading_engine.trade_history[-10:])
            st.dataframe(recent_trades, use_container_width=True)
        else:
            st.info("No trading activity yet.")
    
    def render_data_analysis(self):
        """Render data analysis tab"""
        st.header("📈 Data Analysis")
        
        if st.session_state.current_data is not None:
            data = st.session_state.current_data
            
            # Price chart
            st.subheader("Price Chart with Technical Indicators")
            fig = self.create_price_chart(data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical indicators
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Technical Indicators")
                fig_tech = self.create_technical_indicators_chart(data)
                st.plotly_chart(fig_tech, use_container_width=True)
            
            with col2:
                st.subheader("Volume Analysis")
                fig_vol = self.create_volume_chart(data)
                st.plotly_chart(fig_vol, use_container_width=True)
            
            # Data statistics
            st.subheader("Data Statistics")
            st.dataframe(data.describe(), use_container_width=True)
            
        else:
            st.info("Please refresh data first using the sidebar controls.")
    
    def render_model_training(self):
        """Render model training tab"""
        st.header("🧠 Model Training")
        
        if st.session_state.current_data is not None:
            data = st.session_state.current_data
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Training Configuration")
                
                # Model parameters
                epochs = st.number_input("Epochs", value=MODEL_CONFIG.EPOCHS, min_value=10, max_value=1000)
                batch_size = st.number_input("Batch Size", value=MODEL_CONFIG.BATCH_SIZE, min_value=16, max_value=128)
                sequence_length = st.number_input("Sequence Length", value=MODEL_CONFIG.SEQUENCE_LENGTH, min_value=30, max_value=120)
                
                # Train model button
                if st.button("🚀 Train Model"):
                    self.train_selected_model(data, epochs, batch_size, sequence_length)
            
            with col2:
                st.subheader("Model Performance")
                
                if st.session_state.trained_models:
                    for model_name, metrics in st.session_state.trained_models.items():
                        st.write(f"**{model_name}**")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Validation RMSE", f"{metrics.get('val_rmse', 0):.4f}")
                        with col_b:
                            st.metric("Validation R²", f"{metrics.get('val_r2', 0):.4f}")
                        st.markdown("---")
                else:
                    st.info("No trained models yet.")
        
        else:
            st.info("Please refresh data first to train models.")
    
    def render_trading(self):
        """Render trading tab"""
        st.header("💰 Trading Interface")
        
        if st.session_state.current_data is not None and st.session_state.trained_models:
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Generate Predictions")
                
                # Model selection for prediction
                available_models = list(st.session_state.trained_models.keys())
                selected_model = st.selectbox("Select Model for Trading", available_models)
                
                if st.button("🔮 Generate Predictions"):
                    self.generate_predictions(selected_model)
                
                # Display predictions
                if st.session_state.predictions is not None:
                    predictions = st.session_state.predictions
                    current_price = st.session_state.current_data['Close'].iloc[-1]
                    predicted_price = predictions[-1]
                    
                    st.metric(
                        "Current Price", 
                        f"${current_price:.2f}"
                    )
                    st.metric(
                        "Predicted Price", 
                        f"${predicted_price:.2f}",
                        delta=f"{((predicted_price - current_price) / current_price) * 100:.2f}%"
                    )
            
            with col2:
                st.subheader("Manual Trading")
                
                # Trading controls
                trade_symbol = getattr(st.session_state, 'current_symbol', 'AAPL')
                current_price = st.session_state.current_data['Close'].iloc[-1]
                
                st.write(f"Trading: {trade_symbol} @ ${current_price:.2f}")
                
                col_buy, col_sell = st.columns(2)
                
                with col_buy:
                    if st.button("🟢 BUY", use_container_width=True):
                        success = st.session_state.trading_engine.execute_trade(
                            trade_symbol, TradeAction.BUY, current_price
                        )
                        if success:
                            st.success("Buy order executed!")
                        else:
                            st.error("Buy order failed!")
                
                with col_sell:
                    if st.button("🔴 SELL", use_container_width=True):
                        success = st.session_state.trading_engine.execute_trade(
                            trade_symbol, TradeAction.SELL, current_price
                        )
                        if success:
                            st.success("Sell order executed!")
                        else:
                            st.error("Sell order failed!")
            
            # Predictions chart
            if st.session_state.predictions is not None:
                st.subheader("Price Predictions")
                fig = self.create_predictions_chart()
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            if st.session_state.current_data is None:
                st.info("Please refresh data first.")
            if not st.session_state.trained_models:
                st.info("Please train a model first.")
    
    def render_portfolio(self):
        """Render portfolio tab"""
        st.header("📋 Portfolio Management")
        
        trading_engine = st.session_state.trading_engine
        
        # Portfolio summary
        if hasattr(st.session_state, 'current_symbol') and st.session_state.current_data is not None:
            current_price = st.session_state.current_data['Close'].iloc[-1]
            market_data = {st.session_state.current_symbol: current_price}
            
            portfolio_summary = trading_engine.get_portfolio_summary(market_data)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Portfolio Value", 
                    f"${portfolio_summary.get('total_portfolio_value', 0):,.2f}"
                )
            
            with col2:
                st.metric(
                    "Total Return", 
                    f"${portfolio_summary.get('total_return', 0):,.2f}",
                    delta=f"{portfolio_summary.get('total_return_pct', 0):.2f}%"
                )
            
            with col3:
                st.metric(
                    "Unrealized P&L", 
                    f"${portfolio_summary.get('unrealized_pnl', 0):,.2f}"
                )
            
            with col4:
                st.metric(
                    "Realized P&L", 
                    f"${portfolio_summary.get('realized_pnl', 0):,.2f}"
                )
        
        # Current positions
        st.subheader("Current Positions")
        positions = trading_engine.get_positions_summary()
        
        if positions:
            positions_df = pd.DataFrame(positions)
            st.dataframe(positions_df, use_container_width=True)
        else:
            st.info("No current positions.")
        
        # Trade history
        st.subheader("Trade History")
        if trading_engine.trade_history:
            trade_history_df = pd.DataFrame(trading_engine.trade_history)
            st.dataframe(trade_history_df, use_container_width=True)
            
            # Download trade history
            csv = trade_history_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Trade History",
                data=csv,
                file_name=f"trade_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No trade history yet.")
    
    def train_selected_model(self, data, epochs, batch_size, sequence_length):
        """Train the selected model"""
        try:
            with st.spinner("Training model... This may take a few minutes."):
                # Create sequences
                X, y = self.data_handler.create_sequences(data, sequence_length)
                
                model_type = st.session_state.selected_model_type.lower()
                
                # Update model config
                MODEL_CONFIG.EPOCHS = epochs
                MODEL_CONFIG.BATCH_SIZE = batch_size
                MODEL_CONFIG.SEQUENCE_LENGTH = sequence_length
                
                # Train model based on type
                if model_type == "lstm":
                    model = self.model_trainer.train_lstm_model(X, y, f"lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                elif model_type == "gru":
                    model = self.model_trainer.train_gru_model(X, y, f"gru_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                elif model_type == "xgboost":
                    model = self.model_trainer.train_xgboost_model(X, y, f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                elif model_type == "random forest":
                    model = self.model_trainer.train_random_forest_model(X, y, f"rf_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                
                # Store model metrics
                model_name = list(self.model_trainer.models.keys())[-1]
                st.session_state.trained_models[model_name] = self.model_trainer.model_metrics[model_name]
                
                st.success(f"Model {model_name} trained successfully!")
                
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
    
    def generate_predictions(self, model_name):
        """Generate predictions using selected model"""
        try:
            with st.spinner("Generating predictions..."):
                data = st.session_state.current_data
                sequence_length = MODEL_CONFIG.SEQUENCE_LENGTH
                
                # Create sequences for prediction
                X, _ = self.data_handler.create_sequences(data, sequence_length)
                
                # Load model if not already loaded
                if model_name not in self.model_trainer.models:
                    model_type = "keras" if "lstm" in model_name or "gru" in model_name else "sklearn"
                    self.model_trainer.load_model(model_name, model_type)
                
                # Generate predictions
                predictions = self.model_trainer.predict(model_name, X[-30:])  # Last 30 predictions
                st.session_state.predictions = predictions
                
                st.success("Predictions generated successfully!")
                
        except Exception as e:
            st.error(f"Error generating predictions: {str(e)}")
    
    def create_price_chart(self, data):
        """Create price chart with technical indicators"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price & Technical Indicators', 'Volume'),
            row_width=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Technical indicators
        if 'SMA_20' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(color='orange')),
                row=1, col=1
            )
        
        if 'EMA_20' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['EMA_20'], name='EMA 20', line=dict(color='red')),
                row=1, col=1
            )
        
        if 'BB_upper' in data.columns and 'BB_lower' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['BB_upper'], name='BB Upper', line=dict(color='gray', dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['BB_lower'], name='BB Lower', line=dict(color='gray', dash='dash')),
                row=1, col=1
            )
        
        # Volume
        fig.add_trace(
            go.Bar(x=data.index, y=data['Volume'], name='Volume'),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Stock Price Analysis",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_technical_indicators_chart(self, data):
        """Create technical indicators chart"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('RSI', 'MACD')
        )
        
        # RSI
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['RSI'], name='RSI'),
                row=1, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
        
        # MACD
        if 'MACD' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['MACD'], name='MACD'),
                row=2, col=1
            )
        
        if 'MACD_signal' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['MACD_signal'], name='MACD Signal'),
                row=2, col=1
            )
        
        fig.update_layout(height=400, title="Technical Indicators")
        return fig
    
    def create_volume_chart(self, data):
        """Create volume chart"""
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(x=data.index, y=data['Volume'], name='Volume')
        )
        
        if 'Volume_SMA' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['Volume_SMA'], name='Volume SMA', line=dict(color='red'))
            )
        
        fig.update_layout(
            title="Volume Analysis",
            xaxis_title="Date",
            yaxis_title="Volume",
            height=300
        )
        
        return fig
    
    def create_predictions_chart(self):
        """Create predictions chart"""
        data = st.session_state.current_data
        predictions = st.session_state.predictions
        
        # Get last 60 days of actual prices
        actual_prices = data['Close'].tail(60)
        
        # Create future dates for predictions
        last_date = actual_prices.index[-1]
        future_dates = pd.date_range(start=last_date, periods=len(predictions) + 1, freq='D')[1:]
        
        fig = go.Figure()
        
        # Actual prices
        fig.add_trace(
            go.Scatter(
                x=actual_prices.index,
                y=actual_prices.values,
                name='Actual Price',
                line=dict(color='blue')
            )
        )
        
        # Predictions
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=predictions,
                name='Predicted Price',
                line=dict(color='red', dash='dash')
            )
        )
        
        fig.update_layout(
            title="Price Predictions",
            xaxis_title="Date",
            yaxis_title="Price",
            height=400
        )
        
        return fig

# Run the application
if __name__ == "__main__":
    app = TradingApp()
    app.run()
