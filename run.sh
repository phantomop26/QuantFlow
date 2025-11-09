#!/bin/bash

# AI Trading System - Easy Run Script
# This script sets up the environment and runs the trading application

# Set the correct library path for XGBoost
# Detect the homebrew path dynamically
HOMEBREW_PREFIX=$(brew --prefix)
export DYLD_LIBRARY_PATH=${HOMEBREW_PREFIX}/opt/libomp/lib:$DYLD_LIBRARY_PATH

# Activate virtual environment
source .venv/bin/activate

# Check if command line arguments are provided
if [ $# -eq 0 ]; then
    echo "🚀 AI Trading System"
    echo "===================="
    echo ""
    echo "Available commands:"
    echo "  ./run.sh train --symbol AAPL --period 6mo     # Train models"
    echo "  ./run.sh backtest --symbol AAPL --model MODEL  # Run backtest"
    echo "  ./run.sh live --symbol AAPL --model MODEL      # Live trading"
    echo "  ./run.sh gui                                   # Launch web interface"
    echo ""
    echo "Examples:"
    echo "  ./run.sh train --symbol AAPL --period 6mo"
    echo "  ./run.sh backtest --symbol AAPL --model xgboost_AAPL_20251108_203053"
    echo "  ./run.sh gui"
    echo ""
    exit 1
fi

# Run the Python application with all arguments
python main.py "$@"
