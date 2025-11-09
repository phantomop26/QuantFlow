"""
Trading logic and portfolio management
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from enum import Enum
import json
import os
import sys

# Add the config directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config.settings import TRADING_CONFIG, LOGS_DIR

class TradeAction(Enum):
    """Enumeration for trade actions"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class Position:
    """Represents a trading position"""
    def __init__(self, symbol: str, quantity: float, entry_price: float, 
                 entry_time: datetime, position_type: str = "LONG"):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.position_type = position_type
        self.current_price = entry_price
        self.unrealized_pnl = 0.0
        
    def update_price(self, current_price: float):
        """Update current price and calculate unrealized P&L"""
        self.current_price = current_price
        if self.position_type == "LONG":
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
    
    def get_value(self) -> float:
        """Get current value of the position"""
        return self.current_price * self.quantity

class TradingEngine:
    """Main trading engine for executing trades and managing portfolio"""
    
    def __init__(self, initial_capital: float = None):
        self.initial_capital = initial_capital or TRADING_CONFIG.INITIAL_CAPITAL
        self.current_capital = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
        self.portfolio_history: List[Dict] = []
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for trading operations"""
        logger = logging.getLogger('TradingEngine')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Ensure logs directory exists
            logs_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            
            handler = logging.FileHandler(os.path.join(logs_dir, 'trading_engine.log'))
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def generate_signals(self, predictions: np.ndarray, current_price: float,
                        confidence_scores: Optional[np.ndarray] = None) -> TradeAction:
        """
        Generate trading signals based on predictions
        
        Args:
            predictions: Model predictions
            current_price: Current market price
            confidence_scores: Confidence scores for predictions
            
        Returns:
            Trading action (BUY, SELL, HOLD)
        """
        try:
            if len(predictions) == 0:
                return TradeAction.HOLD
            
            # Get the latest prediction
            predicted_price = predictions[-1]
            price_change_pct = (predicted_price - current_price) / current_price
            
            # Check confidence if available
            if confidence_scores is not None and len(confidence_scores) > 0:
                confidence = confidence_scores[-1]
                if confidence < TRADING_CONFIG.CONFIDENCE_THRESHOLD:
                    self.logger.info(f"Low confidence ({confidence:.3f}), holding position")
                    return TradeAction.HOLD
            
            # Generate signals based on predicted price change
            if price_change_pct > 0.02:  # 2% upward movement predicted
                return TradeAction.BUY
            elif price_change_pct < -0.02:  # 2% downward movement predicted
                return TradeAction.SELL
            else:
                return TradeAction.HOLD
                
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return TradeAction.HOLD
    
    def execute_trade(self, symbol: str, action: TradeAction, current_price: float,
                     quantity: Optional[float] = None) -> bool:
        """
        Execute a trade
        
        Args:
            symbol: Trading symbol
            action: Trade action (BUY, SELL, HOLD)
            current_price: Current market price
            quantity: Quantity to trade (if None, calculated based on position sizing)
            
        Returns:
            True if trade executed successfully, False otherwise
        """
        try:
            if action == TradeAction.HOLD:
                return True
            
            # Calculate position size if not provided
            if quantity is None:
                max_position_value = self.current_capital * TRADING_CONFIG.MAX_POSITION_SIZE
                quantity = max_position_value / current_price
            
            trade_time = datetime.now()
            
            if action == TradeAction.BUY:
                return self._execute_buy(symbol, quantity, current_price, trade_time)
            elif action == TradeAction.SELL:
                return self._execute_sell(symbol, quantity, current_price, trade_time)
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            return False
    
    def _execute_buy(self, symbol: str, quantity: float, price: float, 
                    trade_time: datetime) -> bool:
        """Execute a buy order"""
        try:
            trade_value = quantity * price
            
            # Check if we have enough capital
            if trade_value > self.current_capital:
                self.logger.warning(f"Insufficient capital for buy order. Required: {trade_value}, Available: {self.current_capital}")
                return False
            
            # Execute the trade
            if symbol in self.positions:
                # Add to existing position
                existing_pos = self.positions[symbol]
                total_quantity = existing_pos.quantity + quantity
                weighted_avg_price = ((existing_pos.entry_price * existing_pos.quantity) + 
                                    (price * quantity)) / total_quantity
                
                existing_pos.quantity = total_quantity
                existing_pos.entry_price = weighted_avg_price
            else:
                # Create new position
                self.positions[symbol] = Position(symbol, quantity, price, trade_time, "LONG")
            
            # Update capital
            self.current_capital -= trade_value
            
            # Log the trade
            trade_record = {
                'symbol': symbol,
                'action': 'BUY',
                'quantity': quantity,
                'price': price,
                'value': trade_value,
                'timestamp': trade_time.isoformat(),
                'capital_after': self.current_capital
            }
            
            self.trade_history.append(trade_record)
            self.logger.info(f"BUY executed: {symbol} {quantity} @ {price}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing buy order: {str(e)}")
            return False
    
    def _execute_sell(self, symbol: str, quantity: float, price: float,
                     trade_time: datetime) -> bool:
        """Execute a sell order"""
        try:
            if symbol not in self.positions:
                self.logger.warning(f"No position found for {symbol} to sell")
                return False
            
            position = self.positions[symbol]
            
            # Check if we have enough quantity
            if quantity > position.quantity:
                self.logger.warning(f"Insufficient quantity to sell. Requested: {quantity}, Available: {position.quantity}")
                quantity = position.quantity  # Sell all available
            
            trade_value = quantity * price
            
            # Calculate realized P&L
            realized_pnl = (price - position.entry_price) * quantity
            
            # Update position
            position.quantity -= quantity
            
            # Remove position if fully sold
            if position.quantity <= 0:
                del self.positions[symbol]
            
            # Update capital
            self.current_capital += trade_value
            
            # Log the trade
            trade_record = {
                'symbol': symbol,
                'action': 'SELL',
                'quantity': quantity,
                'price': price,
                'value': trade_value,
                'realized_pnl': realized_pnl,
                'timestamp': trade_time.isoformat(),
                'capital_after': self.current_capital
            }
            
            self.trade_history.append(trade_record)
            self.logger.info(f"SELL executed: {symbol} {quantity} @ {price}, P&L: {realized_pnl:.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing sell order: {str(e)}")
            return False
    
    def update_positions(self, market_data: Dict[str, float]):
        """
        Update all positions with current market prices
        
        Args:
            market_data: Dictionary of symbol -> current_price
        """
        try:
            for symbol, position in self.positions.items():
                if symbol in market_data:
                    position.update_price(market_data[symbol])
                    
            self.logger.debug("Positions updated with current market prices")
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {str(e)}")
    
    def check_stop_loss_take_profit(self, market_data: Dict[str, float]) -> List[Dict]:
        """
        Check and execute stop loss and take profit orders
        
        Args:
            market_data: Dictionary of symbol -> current_price
            
        Returns:
            List of executed orders
        """
        executed_orders = []
        
        try:
            positions_to_close = []
            
            for symbol, position in self.positions.items():
                if symbol not in market_data:
                    continue
                    
                current_price = market_data[symbol]
                price_change_pct = (current_price - position.entry_price) / position.entry_price
                
                should_close = False
                reason = ""
                
                if position.position_type == "LONG":
                    if price_change_pct <= -TRADING_CONFIG.STOP_LOSS_PCT:
                        should_close = True
                        reason = "STOP_LOSS"
                    elif price_change_pct >= TRADING_CONFIG.TAKE_PROFIT_PCT:
                        should_close = True
                        reason = "TAKE_PROFIT"
                
                if should_close:
                    # Execute sell order
                    if self._execute_sell(symbol, position.quantity, current_price, datetime.now()):
                        executed_orders.append({
                            'symbol': symbol,
                            'action': 'SELL',
                            'reason': reason,
                            'quantity': position.quantity,
                            'price': current_price
                        })
                        positions_to_close.append(symbol)
            
            return executed_orders
            
        except Exception as e:
            self.logger.error(f"Error checking stop loss/take profit: {str(e)}")
            return []
    
    def get_portfolio_value(self, market_data: Dict[str, float]) -> float:
        """
        Calculate total portfolio value
        
        Args:
            market_data: Dictionary of symbol -> current_price
            
        Returns:
            Total portfolio value
        """
        try:
            total_value = self.current_capital
            
            for symbol, position in self.positions.items():
                if symbol in market_data:
                    position.update_price(market_data[symbol])
                    total_value += position.get_value()
            
            return total_value
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio value: {str(e)}")
            return self.current_capital
    
    def get_portfolio_summary(self, market_data: Dict[str, float]) -> Dict:
        """
        Get portfolio summary
        
        Args:
            market_data: Dictionary of symbol -> current_price
            
        Returns:
            Portfolio summary dictionary
        """
        try:
            self.update_positions(market_data)
            
            total_value = self.get_portfolio_value(market_data)
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_realized_pnl = sum(trade.get('realized_pnl', 0) for trade in self.trade_history)
            
            return {
                'initial_capital': self.initial_capital,
                'current_capital': self.current_capital,
                'total_portfolio_value': total_value,
                'total_return': total_value - self.initial_capital,
                'total_return_pct': ((total_value - self.initial_capital) / self.initial_capital) * 100,
                'unrealized_pnl': total_unrealized_pnl,
                'realized_pnl': total_realized_pnl,
                'number_of_positions': len(self.positions),
                'number_of_trades': len(self.trade_history)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {str(e)}")
            return {}
    
    def get_positions_summary(self) -> List[Dict]:
        """Get summary of all current positions"""
        try:
            positions_summary = []
            
            for symbol, position in self.positions.items():
                positions_summary.append({
                    'symbol': symbol,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'unrealized_pnl': position.unrealized_pnl,
                    'value': position.get_value(),
                    'entry_time': position.entry_time.isoformat()
                })
            
            return positions_summary
            
        except Exception as e:
            self.logger.error(f"Error getting positions summary: {str(e)}")
            return []
    
    def save_portfolio_snapshot(self, market_data: Dict[str, float]):
        """Save current portfolio state to history"""
        try:
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_summary': self.get_portfolio_summary(market_data),
                'positions': self.get_positions_summary()
            }
            
            self.portfolio_history.append(snapshot)
            
        except Exception as e:
            self.logger.error(f"Error saving portfolio snapshot: {str(e)}")
    
    def export_trading_data(self) -> Dict:
        """Export all trading data for analysis"""
        try:
            return {
                'initial_capital': self.initial_capital,
                'current_capital': self.current_capital,
                'trade_history': self.trade_history,
                'portfolio_history': self.portfolio_history,
                'current_positions': self.get_positions_summary()
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting trading data: {str(e)}")
            return {}
    
    def save_trading_data(self, filename: Optional[str] = None):
        """Save trading data to file"""
        try:
            if filename is None:
                filename = f"trading_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            filepath = os.path.join(LOGS_DIR, filename)
            trading_data = self.export_trading_data()
            
            with open(filepath, 'w') as f:
                json.dump(trading_data, f, indent=2)
            
            self.logger.info(f"Trading data saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving trading data: {str(e)}")
            raise
