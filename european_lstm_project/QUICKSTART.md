# Quick Start Guide - European LSTM Portfolio

## 🚀 How to Run Everything (Step-by-Step)

### **Current Status**: ✅ Phase 1 Complete

**What's Ready:**
- ✅ All Python utilities (`src/` folder)
- ✅ Configuration (`config.py`)
- ✅ 2 complete notebooks (Universe Selection, Feature Engineering)
- ✅ Complete documentation

**What's Needed:**
- ⏳ Notebooks 03-05 (LSTM Training, Optimization, Analysis)
- These will be created in next iteration

---

## 📁 Option 1: Run on Google Colab Pro (Recommended - FASTEST)

### **Setup (5 minutes, one-time)**

#### Step 1: Get Google Colab Pro
1. Go to: https://colab.research.google.com/signup
2. Subscribe ($10/month)
3. You get: Tesla T4 GPU (15-30 min training vs. 2-4 hours CPU)

#### Step 2: Upload Project to Google Drive
1. Download the entire `european_lstm_project/` folder from GitHub
2. Upload to: `Google Drive > MyDrive > european_lstm_project/`

**Your folder structure should look like:**
```
Google Drive/
└── MyDrive/
    └── european_lstm_project/
        ├── config.py
        ├── requirements.txt
        ├── data/
        ├── notebooks/
        │   ├── 01_universe_selection.ipynb
        │   └── 02_feature_engineering.ipynb
        └── src/
            ├── data_loader.py
            ├── features.py
            ├── models.py
            └── optimization.py
```

#### Step 3: Run Notebooks Sequentially

**Notebook 01: Universe Selection (10 min)**
1. In Google Drive, open: `european_lstm_project/notebooks/01_universe_selection.ipynb`
2. Click "Open with Google Colaboratory"
3. Click `Runtime` → `Run all`
4. When prompted:
   - Enter WRDS username
   - Enter WRDS password (hidden input)
5. Wait 5-10 minutes for data download
6. **Output**: `data/processed/universe_tickers.csv` (44 stocks)

**Notebook 02: Feature Engineering (45 min)**
1. Open `02_feature_engineering.ipynb` in Colab
2. Click `Runtime` → `Run all`
3. No prompts (uses data from Notebook 01)
4. Wait 30-45 minutes
5. **Output**: `data/processed/features_engineered.parquet`

**Notebook 03: LSTM Training** *(To be created)*
- Train 44 LSTM models
- GPU: 15-30 minutes
- CPU: 2-4 hours

**Notebook 04: Portfolio Optimization** *(To be created)*
- Max Sharpe optimization
- 30 minutes

**Notebook 05: Performance Analysis** *(To be created)*
- Backtesting & metrics
- 1 hour

---

## 💻 Option 2: Run Locally (Slower, but free)

### **Setup**

#### Step 1: Install Python (if needed)
- Download Python 3.8+: https://www.python.org/downloads/
- Verify: `python --version`

#### Step 2: Clone Repository
```bash
git clone https://github.com/mattyyychan/KingsInvestmentFund.git
cd KingsInvestmentFund/european_lstm_project
```

#### Step 3: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

#### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 5: Run Notebooks
```bash
jupyter notebook
```

Then open and run notebooks 01-05 in order.

---

## 📊 What Each Notebook Does

### **Notebook 01: Universe Selection**
**Input:** None (downloads from WRDS)
**Output:** 44 European stocks
**Time:** 10 minutes
**Academic:** Stratified sector sampling (Fama-French, 1993)

**What it does:**
1. Connects to WRDS Compustat Global
2. Filters for liquid stocks (€2B+ market cap, €1M+ volume)
3. Selects 4 stocks per sector (11 sectors = 44 total)
4. Downloads STOXX 600 benchmark
5. Saves ticker list

---

### **Notebook 02: Feature Engineering**
**Input:** Data from Notebook 01
**Output:** Engineered features for LSTM
**Time:** 30-45 minutes
**Academic:** CAPM (Sharpe, 1964), PCA (Connor & Korajczyk, 1988)

**What it does:**
1. **Calculates rolling beta** (60-day window)
   - β = Cov(Stock, Market) / Var(Market)
2. **Computes residual returns**:
   - Residual = Stock Return - (β × Market Return)
   - This isolates alpha (stock-picking skill)
3. **Downloads macro variables**:
   - Oil, interest rates, volatility, FX, gold
4. **Generates technical indicators**:
   - RSI, MACD, Bollinger Bands, momentum
5. **Applies PCA**:
   - 20+ indicators → 5-8 components
   - Fit ONLY on training data (no look-ahead!)
6. **Creates binary target**:
   - 1 if residual > 0, else 0

**Key Output:**
- ~15 final features per stock per day
- Binary target for LSTM training

---

### **Notebook 03: LSTM Training** *(To be created)*
**Input:** Features from Notebook 02
**Output:** 44 trained LSTM models
**Time:** 15-30 min (GPU) or 2-4 hours (CPU)
**Academic:** Fischer & Krauss (2018), Hochreiter (1997)

**What it will do:**
1. Create 60-day sequences
2. Train individual LSTM for each stock:
   - Architecture: LSTM(64) → LSTM(32) → Dense(1, sigmoid)
   - Early stopping (patience=10)
   - Dropout (20%)
3. Generate probability predictions
4. Validate on 2023 data
5. Test on 2024-2025

**Key Output:**
- 44 saved models (.keras files)
- Prediction probabilities CSV

---

### **Notebook 04: Portfolio Optimization** *(To be created)*
**Input:** LSTM predictions
**Output:** Optimized portfolio weights
**Time:** 30 minutes
**Academic:** Markowitz (1952), Sharpe (1966), Ledoit-Wolf (2004)

**What it will do:**
1. Convert probabilities to expected returns
2. Calculate covariance matrix (Ledoit-Wolf shrinkage)
3. Solve Max Sharpe optimization:
   - Maximize: (Return - Risk-Free) / Volatility
   - Constraints: Long-only, max 10%/stock, max 15%/sector
4. Monthly rebalancing
5. Apply transaction costs (10 bps)

**Key Output:**
- Monthly portfolio weights
- Turnover statistics

---

### **Notebook 05: Performance Analysis** *(To be created)*
**Input:** Portfolio weights
**Output:** Backtest results
**Time:** 1 hour
**Academic:** Diebold-Mariano (1995), Harvey et al. (2016)

**What it will do:**
1. Backtest 2024-2025
2. Calculate metrics:
   - Sharpe Ratio
   - Maximum Drawdown
   - Information Ratio
   - Turnover
3. Compare to benchmarks:
   - Equal-Weight (1/N)
   - STOXX 600
   - Minimum Variance
4. Statistical tests (Diebold-Mariano)
5. Visualizations:
   - Cumulative returns
   - Sector allocation
   - Efficient frontier

**Key Output:**
- Performance report
- Figures and charts

---

## 🎯 Expected Final Results

### **Performance Targets** (Academic Benchmarks)

Based on literature (Gu et al., 2020; Fischer & Krauss, 2018):

| Metric | STOXX 600 | Equal-Weight | Min Variance | **LSTM (Target)** |
|--------|-----------|--------------|--------------|-------------------|
| Sharpe Ratio | ~0.5 | ~0.6 | ~0.7 | **>1.0** |
| Max Drawdown | ~35% | ~30% | ~25% | **<25%** |
| Info Ratio | 0.0 | ~0.3 | ~0.5 | **>0.5** |
| Turnover | 0% | ~5% | ~10% | ~40% |

**Statistical Significance:**
- Diebold-Mariano test: p-value < 0.05
- Confirms LSTM adds value beyond random

---

## ⚠️ Important Notes

### **WRDS Access Required**
- You MUST have WRDS account
- Need Compustat Global access
- Free for university students/faculty
- Apply at: https://wrds-www.wharton.upenn.edu/

### **Data Privacy**
- WRDS credentials are entered securely (hidden input)
- Never committed to Git
- Stored only in your local session

### **Computational Requirements**
- **Minimum**: 8GB RAM, 10GB disk space
- **Recommended**: 16GB RAM, GPU for training
- **Google Colab Pro**: Handles everything automatically

### **Timeline**
- **Setup**: 5 minutes (one-time)
- **Notebook 01**: 10 minutes
- **Notebook 02**: 30-45 minutes
- **Notebooks 03-05**: ~2-3 hours (GPU) or 5-6 hours (CPU)
- **Total**: 4-5 hours spread over a few days

---

## 🐛 Troubleshooting

### **WRDS Connection Fails**
```python
# Test connection
import wrds
db = wrds.Connection(wrds_username='YOUR_USERNAME')
db.list_libraries()  # Should show 'comp', 'crsp', etc.
```

### **Out of Memory (Colab)**
```python
# In config.py, reduce:
LOOKBACK_WINDOW = 30  # Instead of 60
BATCH_SIZE = 16       # Instead of 32
```

### **Notebooks 03-05 Missing**
- Currently being created
- Will be added in next Git commit
- Check repository for updates

---

## 📖 Where to Get Help

### **Documentation Files:**
1. **START HERE**: `STRATEGY_EXPLAINED_SIMPLE.md` - Plain English explanation
2. `ACADEMIC_METHODOLOGY.md` - Full academic justification
3. `IMPLEMENTATION_GUIDE.md` - Detailed technical guide
4. `README.md` - Overview and setup

### **Academic Papers:**
All cited papers are listed in `ACADEMIC_METHODOLOGY.md`

### **Code Issues:**
- Check inline comments in notebooks (extensive documentation)
- Review `config.py` for all settings
- Open GitHub issue: https://github.com/mattyyychan/KingsInvestmentFund/issues

---

## ✅ Checklist Before Running

- [ ] WRDS account with Compustat Global access
- [ ] Google Colab Pro subscription **OR** local Python 3.8+
- [ ] 10GB free disk space
- [ ] 2-5 hours to let notebooks run
- [ ] Read `STRATEGY_EXPLAINED_SIMPLE.md`

---

## 🎯 Next Steps

### **Right Now (Phase 1 Complete):**
✅ All infrastructure built
✅ Notebooks 01-02 ready to run
✅ Complete documentation

### **Next Commit (Phase 2):**
⏳ Create Notebooks 03-05
⏳ Add example outputs
⏳ Final testing

### **After Running:**
📊 Analyze backtest results
📝 Write academic report
🎓 Present findings

---

**Questions?** Read `STRATEGY_EXPLAINED_SIMPLE.md` first, then ask!
