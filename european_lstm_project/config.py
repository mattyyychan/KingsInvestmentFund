"""
Configuration file for European LSTM Portfolio Project
Contains all hyperparameters, constraints, and global settings
"""

import os
from datetime import datetime

# ============================================================================
# PROJECT PATHS (Auto-detects Colab vs Local)
# ============================================================================
import sys

# Detect if running in Google Colab
IS_COLAB = 'google.colab' in sys.modules

if IS_COLAB:
    # Colab paths (assumes Google Drive is mounted)
    BASE_DIR = '/content/drive/MyDrive/european_lstm_project'
else:
    # Local paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MARKET_DATA_DIR = os.path.join(DATA_DIR, 'market_data')
PREDICTIONS_DIR = os.path.join(DATA_DIR, 'predictions')

# Source code directory
SRC_DIR = os.path.join(BASE_DIR, 'src')

# Notebooks directory
NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'notebooks')

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MARKET_DATA_DIR,
                  PREDICTIONS_DIR, SRC_DIR, NOTEBOOKS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# DATA COLLECTION PARAMETERS (Step 1: Universe Selection)
# ============================================================================

# Date ranges
DATA_START_DATE = '2018-01-01'
DATA_END_DATE = '2025-12-31'

# Training/Validation/Test splits
TRAIN_START = '2018-01-01'
TRAIN_END = '2022-12-31'

VALIDATION_START = '2023-01-01'
VALIDATION_END = '2023-12-31'

TEST_START = '2024-01-01'
TEST_END = '2025-12-31'

# Universe selection criteria
MIN_MARKET_CAP = 2_000_000_000  # €2 Billion
MIN_AVG_VOLUME = 1_000_000      # €1 Million daily volume
MIN_HISTORY_YEARS = 5            # Minimum 5 years of data
REQUIRE_POSITIVE_BOOK_EQUITY = True

# Sector balancing
GICS_SECTORS = [
    'Energy',
    'Materials',
    'Industrials',
    'Consumer Discretionary',
    'Consumer Staples',
    'Health Care',
    'Financials',
    'Information Technology',
    'Communication Services',
    'Utilities',
    'Real Estate'
]
STOCKS_PER_SECTOR = 4  # Target 4 stocks per sector = 44 total

# European market benchmark
BENCHMARK_TICKER = '^STOXX'  # STOXX Europe 600
BENCHMARK_NAME = 'STOXX600'

# ============================================================================
# FEATURE ENGINEERING PARAMETERS (Step 2)
# ============================================================================

# Systematic risk removal (Beta calculation)
BETA_WINDOW = 60  # 60-day rolling window for beta calculation

# Technical indicators
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
VOLUME_MA_PERIOD = 20
RETURNS_WINDOW = [5, 10, 20, 60]  # Multiple return windows
VOLATILITY_WINDOW = 20

# PCA configuration
PCA_VARIANCE_THRESHOLD = 0.95  # Keep components explaining 95% variance
PCA_FIT_ON_TRAIN_ONLY = True   # Critical: prevent data leakage

# Macro variables to download (yfinance tickers)
MACRO_VARIABLES = {
    'BUND_10Y': '^TNX',           # German 10-year Bund (proxy)
    'VSTOXX': '^V2TX',            # Euro Stoxx 50 Volatility
    'BRENT': 'BZ=F',              # Brent Crude Oil
    'GOLD': 'GC=F',               # Gold futures
    'EURUSD': 'EURUSD=X',         # EUR/USD exchange rate
}

# Macro feature engineering
MACRO_DIFFERENCE = True  # Calculate daily differences for stationarity
MACRO_LAG_DAYS = 1       # Lag macro variables by 1 day

# Fundamental features (from WRDS Compustat)
FUNDAMENTAL_FEATURES = [
    'pe_ratio',        # Price-to-Earnings
    'pb_ratio',        # Price-to-Book
    'debt_to_equity',  # Debt/Equity ratio
    'roe',             # Return on Equity
    'roa',             # Return on Assets
    'current_ratio',   # Current Ratio
    'quick_ratio',     # Quick Ratio
]

# ============================================================================
# LSTM MODEL PARAMETERS (Step 3)
# ============================================================================

# Model architecture
LOOKBACK_WINDOW = 60      # 60 days of historical data
LSTM_UNITS_1 = 64         # First LSTM layer units
LSTM_UNITS_2 = 32         # Second LSTM layer units
LSTM_DROPOUT = 0.2        # Dropout rate for regularization

# Training parameters
BATCH_SIZE = 32
EPOCHS = 100              # Maximum epochs
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.0    # Use explicit validation set instead

# Early stopping (critical for CPU efficiency)
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.0001
RESTORE_BEST_WEIGHTS = True

# Model saving
SAVE_BEST_MODEL = True
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Target generation (Binary classification)
TARGET_COLUMN = 'residual_return_positive'  # 1 if residual return > 0
RESIDUAL_RETURN_COLUMN = 'residual_return'

# ============================================================================
# PORTFOLIO OPTIMIZATION PARAMETERS (Step 4)
# ============================================================================

# Expected return calculation from LSTM probabilities
# E[r] = p * Avg_Up_Return + (1-p) * Avg_Down_Return
CALCULATE_EXPECTED_RETURNS_FROM_TRAIN = True  # Prevent leakage

# Covariance matrix
COVARIANCE_WINDOW_MONTHS = 12  # 12-month rolling window
USE_SHRINKAGE = True           # Ledoit-Wolf shrinkage for stability

# Risk-free rate (approximate Euro short-term rate)
RISK_FREE_RATE = 0.02  # 2% annualized

# Portfolio constraints
LONG_ONLY = True                    # w_i >= 0
FULL_INVESTMENT = True              # sum(w_i) = 1
MAX_POSITION_SIZE = 0.10            # 10% max per stock
MAX_SECTOR_WEIGHT = 0.15            # 15% max per sector

# Rebalancing
REBALANCE_FREQUENCY = 'monthly'  # Options: 'monthly', 'quarterly'

# Transaction costs
TRANSACTION_COST_BPS = 10  # 10 basis points (0.10%)

# CVXPY solver settings
SOLVER = 'ECOS'  # Options: 'ECOS', 'SCS', 'CVXOPT'
SOLVER_VERBOSE = False
MAX_ITERATIONS = 1000

# Fallback strategy if optimization fails
FALLBACK_TO_EQUAL_WEIGHT = True

# ============================================================================
# BACKTESTING & PERFORMANCE ANALYSIS (Step 5)
# ============================================================================

# Benchmark strategies for comparison
BENCHMARK_STRATEGIES = [
    'equal_weight',        # 1/N naive diversification
    'market_benchmark',    # STOXX 600
    'minimum_variance',    # Min variance (no ML views)
]

# Performance metrics
ANNUALIZATION_FACTOR = 252  # Trading days per year

# Rolling metrics windows
ROLLING_SHARPE_WINDOW = 126  # 6 months (in trading days)
ROLLING_VOLATILITY_WINDOW = 60

# Statistical tests
RUN_DIEBOLD_MARIANO_TEST = True  # Test forecast significance
DM_TEST_HORIZON = 1              # 1-step ahead forecast

# Visualization settings
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_DPI = 150
SAVE_FIGURES = True
FIGURE_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)

# ============================================================================
# COMPUTATIONAL SETTINGS
# ============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Parallel processing
USE_MULTIPROCESSING = False  # Set True for local multi-core CPU
N_JOBS = -1                  # Use all available cores

# GPU settings (auto-detected)
import tensorflow as tf
GPU_AVAILABLE = len(tf.config.list_physical_devices('GPU')) > 0
if GPU_AVAILABLE:
    print("✓ GPU detected - LSTM training will be accelerated")
else:
    print("○ No GPU detected - training will use CPU (slower)")

# Memory management for Colab
if IS_COLAB:
    # Limit TensorFlow GPU memory growth to prevent crashes
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU memory config error: {e}")

# ============================================================================
# WRDS CONNECTION SETTINGS
# ============================================================================

WRDS_USERNAME = None  # Will be prompted in notebook
WRDS_PASSWORD = None  # Will be prompted securely

# Compustat Global tables
COMPUSTAT_FUNDAMENTALS_TABLE = 'comp.g_funda'
COMPUSTAT_SECURITY_TABLE = 'comp.g_secd'
COMPUSTAT_COMPANY_TABLE = 'comp.g_company'

# European region codes for WRDS filtering
EUROPEAN_COUNTRY_CODES = [
    'AUT', 'BEL', 'DNK', 'FIN', 'FRA', 'DEU', 'GRC', 'IRL', 'ITA',
    'LUX', 'NLD', 'PRT', 'ESP', 'SWE', 'GBR', 'NOR', 'CHE', 'POL',
    'CZE', 'HUN'
]

# ============================================================================
# LOGGING & DEBUGGING
# ============================================================================

import logging

LOGGING_LEVEL = logging.INFO
LOG_FILE = os.path.join(BASE_DIR, 'lstm_portfolio.log')

# Configure logging
logging.basicConfig(
    level=LOGGING_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_config_summary():
    """Print a summary of key configuration settings"""
    print("=" * 80)
    print("EUROPEAN LSTM PORTFOLIO - CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"Environment: {'Google Colab' if IS_COLAB else 'Local'}")
    print(f"GPU Available: {GPU_AVAILABLE}")
    print(f"Base Directory: {BASE_DIR}")
    print()
    print("DATA PERIODS:")
    print(f"  Training:   {TRAIN_START} to {TRAIN_END}")
    print(f"  Validation: {VALIDATION_START} to {VALIDATION_END}")
    print(f"  Testing:    {TEST_START} to {TEST_END}")
    print()
    print("UNIVERSE:")
    print(f"  Target Stocks: {STOCKS_PER_SECTOR} per sector × {len(GICS_SECTORS)} sectors = {STOCKS_PER_SECTOR * len(GICS_SECTORS)}")
    print(f"  Min Market Cap: €{MIN_MARKET_CAP:,.0f}")
    print(f"  Min Daily Volume: €{MIN_AVG_VOLUME:,.0f}")
    print()
    print("LSTM MODEL:")
    print(f"  Lookback: {LOOKBACK_WINDOW} days")
    print(f"  Architecture: {LSTM_UNITS_1} → {LSTM_UNITS_2} → 1 (sigmoid)")
    print(f"  Early Stopping: {EARLY_STOPPING_PATIENCE} epochs patience")
    print()
    print("PORTFOLIO OPTIMIZATION:")
    print(f"  Objective: Maximize Sharpe Ratio")
    print(f"  Max Position: {MAX_POSITION_SIZE * 100}%")
    print(f"  Max Sector: {MAX_SECTOR_WEIGHT * 100}%")
    print(f"  Transaction Costs: {TRANSACTION_COST_BPS} bps")
    print(f"  Rebalance: {REBALANCE_FREQUENCY}")
    print("=" * 80)

if __name__ == "__main__":
    print_config_summary()
