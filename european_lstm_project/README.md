# European Long-Only LSTM Portfolio

An academic-grade long-only equity portfolio strategy for European stocks using LSTM neural networks to predict residual returns (alpha), combined with Maximum Sharpe Ratio optimization.

## 🎯 Project Overview

This project implements a sophisticated quantitative investment strategy:

1. **Universe**: 44-50 liquid European stocks (balanced across 11 GICS sectors)
2. **Signal Generation**: LSTM models predict probability of positive residual returns
3. **Risk Management**: Systematic market risk removed via rolling beta calculation
4. **Optimization**: Maximum Sharpe Ratio portfolio with position/sector constraints
5. **Execution**: Monthly rebalancing with transaction costs

## 📊 Key Features

### Advanced Techniques
- **Systematic Risk Removal**: Rolling 60-day beta calculation to isolate alpha
- **PCA Dimensionality Reduction**: Reduces 20+ technical indicators to 5-8 components
- **Macro Variable Integration**: Oil, rates, volatility, FX as additional features
- **Individual Stock Models**: Separate LSTM for each stock (44+ models)
- **Convex Optimization**: CVXPY-based Max Sharpe with constraints

### Academic Rigor
- ✅ **Zero Look-Ahead Bias**: Strict train/val/test splits
- ✅ **Proper Cross-Validation**: Scalers and PCA fit only on training data
- ✅ **Transaction Costs**: 10 bps per trade
- ✅ **Statistical Testing**: Diebold-Mariano forecast significance test
- ✅ **Benchmark Comparisons**: Equal-Weight, STOXX 600, Min Variance

## 🗂️ Repository Structure

```
european_lstm_project/
├── data/
│   ├── raw/                  # WRDS downloads
│   ├── processed/            # Feature-engineered datasets
│   ├── market_data/          # Benchmark & macro data
│   └── predictions/          # LSTM outputs
├── notebooks/
│   ├── 01_universe_selection.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_lstm_training.ipynb
│   ├── 04_portfolio_optimization.ipynb
│   └── 05_performance_analysis.ipynb
├── src/
│   ├── data_loader.py        # WRDS & yfinance utilities
│   ├── features.py           # PCA, residuals, technicals
│   ├── models.py             # LSTM architecture
│   └── optimization.py       # Max Sharpe CVXPY solver
├── config.py                 # All hyperparameters
└── requirements.txt
```

## 🚀 Quick Start

### Option 1: Google Colab Pro (Recommended - 15-30 min GPU training)

1. **Sign up for Colab Pro**: https://colab.research.google.com/
   - Cost: $10/month
   - GPU: Tesla T4 (much faster than CPU)

2. **Upload project to Google Drive**:
   ```
   european_lstm_project/ → Google Drive > MyDrive/
   ```

3. **Open Notebooks in Colab**:
   - Click on each notebook (01-05)
   - Select "Open with Google Colaboratory"

4. **Run Notebook 01**:
   - You'll be prompted for WRDS credentials
   - Data will download to your Google Drive

5. **Run Notebooks 02-05 sequentially**

### Option 2: Local Installation (2-4 hour CPU training)

```bash
# Clone repository
cd european_lstm_project/

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook notebooks/01_universe_selection.ipynb
```

## 📋 Prerequisites

### Required
- **WRDS Access**: Valid WRDS account with Compustat Global access
- **Python 3.8+**
- **10GB free disk space**

### Optional (for faster training)
- **Google Colab Pro** ($10/month) - Reduces training from 2+ hours to 15-30 minutes
- **Local GPU** (NVIDIA with CUDA support)

## 📝 Workflow

### Step 1: Universe Selection (Notebook 01)
- Connect to WRDS Compustat Global
- Filter for liquid European stocks (€2B+ market cap, €1M+ daily volume)
- Select 4 stocks per GICS sector (44 total)
- Save ticker list

**Output**: `data/processed/universe_tickers.csv`

### Step 2: Feature Engineering (Notebook 02)
- Calculate rolling beta vs. STOXX 600
- Compute residual returns (alpha signal)
- Download macro variables (Oil, VSTOXX, Bunds, EUR/USD)
- Generate 20+ technical indicators
- Apply PCA to reduce dimensionality (fit on train only)
- Create binary target: `1 if residual_return > 0`

**Output**: `data/processed/features_engineered.parquet`

### Step 3: LSTM Training (Notebook 03)
- Train individual LSTM model for each stock
- Architecture: LSTM(64) → LSTM(32) → Dense(1, sigmoid)
- Early stopping (patience=10 epochs)
- Generate probability predictions for test period

**Output**: `data/predictions/lstm_probabilities.csv`

### Step 4: Portfolio Optimization (Notebook 04)
- Convert probabilities to expected returns
- Calculate covariance matrix (12-month rolling, Ledoit-Wolf shrinkage)
- Solve Max Sharpe problem with CVXPY:
  - Objective: Maximize Sharpe Ratio
  - Constraints: Long-only, max 10% per stock, max 15% per sector
- Monthly rebalancing
- Apply 10 bps transaction costs

**Output**: `data/processed/portfolio_weights.csv`

### Step 5: Performance Analysis (Notebook 05)
- Backtest against benchmarks (Equal-Weight, STOXX 600, Min Variance)
- Calculate metrics: Sharpe, Max Drawdown, Information Ratio
- Diebold-Mariano statistical test
- Visualizations: Cumulative returns, sector allocation, efficient frontier

**Output**: Figures in `figures/`, performance report

## ⚙️ Configuration

All hyperparameters in `config.py`:

```python
# Key parameters
LOOKBACK_WINDOW = 60           # Days of history for LSTM
LSTM_UNITS_1 = 64              # First layer size
EARLY_STOPPING_PATIENCE = 10   # Epochs patience
MAX_POSITION_SIZE = 0.10       # 10% max per stock
MAX_SECTOR_WEIGHT = 0.15       # 15% max per sector
TRANSACTION_COST_BPS = 10      # 10 bps
REBALANCE_FREQUENCY = 'monthly'
```

## 🔒 Security Note (WRDS Credentials)

**Option 1: Colab/Jupyter (Secure Prompt)**
```python
from getpass import getpass
wrds_password = getpass("WRDS Password: ")  # Hidden input
```

**Option 2: Environment Variables (Local)**
```bash
export WRDS_USERNAME='your_username'
export WRDS_PASSWORD='your_password'
```

**Never commit credentials to Git!**

## 📊 Expected Results

### Training Time
- **Google Colab Pro (GPU)**: 15-30 minutes for 44 models
- **Local CPU**: 2-4 hours for 44 models

### Performance (Indicative)
Results depend on the specific period tested. Academic goal is to demonstrate:
- Positive alpha generation (residual returns)
- Sharpe Ratio improvement vs. benchmarks
- Statistical significance (Diebold-Mariano test)

## 🛠️ Troubleshooting

### WRDS Connection Issues
```python
# Test connection
import wrds
db = wrds.Connection(wrds_username='your_username')
db.list_libraries()  # Should show 'comp', 'crsp', etc.
```

### Out of Memory (Colab)
```python
# In config.py, reduce:
LOOKBACK_WINDOW = 30  # Instead of 60
BATCH_SIZE = 16       # Instead of 32
```

### Optimization Fails
- Check for NaN in expected returns or covariance
- Enable fallback: `FALLBACK_TO_EQUAL_WEIGHT = True`

## 📚 References

### Academic Papers
- Hochreiter & Schmidhuber (1997) - LSTM Architecture
- Markowitz (1952) - Portfolio Selection
- Ledoit & Wolf (2004) - Covariance Shrinkage

### Data Sources
- WRDS Compustat Global (institutional access required)
- Yahoo Finance (free macro/benchmark data)

## 📄 License

This project is for academic and educational purposes.

## 🤝 Contributing

This is an academic research project. For questions or improvements:
1. Open an issue
2. Submit a pull request with clear documentation

## 📧 Contact

For questions about WRDS setup, model architecture, or optimization:
- Check the inline comments in notebooks (extensive documentation)
- Review `config.py` for all tuneable parameters

---

**Note**: This is a research/educational project. Not financial advice. Past performance does not guarantee future results.
