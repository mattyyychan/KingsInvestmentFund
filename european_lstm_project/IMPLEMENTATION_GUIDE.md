# Complete Implementation Guide

## European LSTM Portfolio - Step-by-Step Execution

---

## 🎯 What We've Built So Far

### ✅ **Phase 1: Foundation (COMPLETE)**

**Files Created:**
1. `config.py` - All hyperparameters and settings
2. `requirements.txt` - Dependencies
3. `src/data_loader.py` - WRDS and yfinance utilities
4. `src/features.py` - Feature engineering (PCA, residuals, technicals)
5. `src/models.py` - LSTM architecture
6. `src/optimization.py` - Max Sharpe solver
7. `README.md` - Setup instructions
8. `ACADEMIC_METHODOLOGY.md` - Complete academic justification
9. `notebooks/01_universe_selection.ipynb` - First notebook (ready to run)

---

## 📋 How to Run This Project

### **Option 1: Google Colab Pro (Recommended - 30 min total)**

#### Setup (One-Time):
1. **Sign up for Colab Pro**: https://colab.research.google.com/signup
   - Cost: $10/month
   - GPU: Tesla T4 (15-30 min training vs. 2+ hours CPU)

2. **Upload to Google Drive**:
   ```
   Upload entire "european_lstm_project" folder to:
   Google Drive > MyDrive/ > european_lstm_project/
   ```

3. **Get WRDS Credentials**:
   - Ensure you have WRDS account with Compustat Global access
   - Have username/password ready

#### Running Notebooks:

**Notebook 01: Universe Selection (10 min)**
1. Open `01_universe_selection.ipynb` in Colab
2. Click "Runtime" → "Run all"
3. When prompted, enter WRDS credentials
4. Wait for data download (~5-10 min)
5. Verify output: `data/processed/universe_tickers.csv` created

**Notebook 02: Feature Engineering** *(To be created)*
- Calculate rolling beta
- Download macro variables
- Apply PCA
- Create target variable

**Notebook 03: LSTM Training** *(To be created)*
- Train 44 individual LSTM models
- GPU accelerated (15-30 min on Colab Pro)
- Save predictions

**Notebook 04: Portfolio Optimization** *(To be created)*
- Convert probabilities to expected returns
- Solve Max Sharpe problem
- Monthly rebalancing

**Notebook 05: Performance Analysis** *(To be created)*
- Backtest vs. benchmarks
- Calculate Sharpe, Drawdown, Info Ratio
- Statistical tests

---

## 📚 Academic Strategy Explained

### **Layer 1: Signal Generation (LSTM)**

**What it does:**
- Predicts probability that a stock's **residual return** will be positive
- Residual return = Stock return - (Beta × Market return)
- Binary classification: Up (1) or Down (0)

**Why this matters (Academic):**

1. **CAPM Foundation (Sharpe, 1964)**:
   $$E[R_i] = R_f + \beta_i (E[R_m] - R_f) + \alpha_i$$

   - β captures systematic risk (market exposure)
   - α (alpha) is idiosyncratic performance (manager skill)
   - By predicting residuals, LSTM focuses on **alpha**, not beta

2. **LSTM Superiority (Fischer & Krauss, 2018)**:
   - LSTMs outperform traditional ML for stock prediction
   - Can capture non-linear patterns and long-term dependencies
   - 60-day lookback captures medium-term momentum

3. **Individual Stock Models (Feng et al., 2019)**:
   - Each stock has unique dynamics
   - 44 separate models vs. single pooled model
   - Captures stock-specific mean reversion, momentum

**Academic Sources:**
- Sharpe (1964) - CAPM
- Fischer & Krauss (2018) - "Deep learning with LSTM for financial markets"
- Hochreiter & Schmidhuber (1997) - Original LSTM paper

---

### **Layer 2: Risk Management (Residual Returns)**

**What it does:**
- Calculate 60-day rolling beta: $\beta_t = \frac{Cov(R_{stock}, R_{market})}{Var(R_{market})}$
- Compute residual: $r_{residual} = r_{stock} - \beta \times r_{market}$

**Why this matters (Academic):**

1. **Removes Market Risk**:
   - If STOXX 600 rises 2% and stock rises 3%:
     - Raw return: +3%
     - If β = 1.5, expected return = 1.5 × 2% = 3%
     - Residual return = 3% - 3% = **0%** (no alpha!)
   - Stock only outperformed because market rose (beta effect)

2. **Focuses on Stock Selection**:
   - Market timing vs. stock selection (Brinson et al., 1986)
   - Long-only portfolio can't profit from market prediction
   - Alpha is what matters for active management

3. **Time-Varying Beta (Ferson & Harvey, 1991)**:
   - Betas change over time (not constant as in classical CAPM)
   - 60-day rolling window balances responsiveness vs. noise

**Academic Sources:**
- Lintner (1965) - CAPM derivation
- Ferson & Harvey (1991) - "Variation of economic risk premiums"
- Brinson et al. (1986) - "Determinants of portfolio performance"

---

### **Layer 3: Portfolio Construction (Max Sharpe)**

**What it does:**
- Optimize portfolio weights to maximize:
  $$\text{Sharpe Ratio} = \frac{E[R_p] - R_f}{\sigma_p}$$

- Subject to constraints:
  - Long-only: $w_i \geq 0$
  - Fully invested: $\sum w_i = 1$
  - Max 10% per stock
  - Max 15% per sector

**Why this matters (Academic):**

1. **Mean-Variance Efficiency (Markowitz, 1952)**:
   - Sharpe Ratio identifies the portfolio with best risk-adjusted return
   - Lies on the efficient frontier
   - Optimal under quadratic utility

2. **Constraints Improve Performance (DeMiguel et al., 2009)**:
   - Unconstrained optimization suffers from estimation error
   - Position limits prevent extreme weights from noisy signals
   - "1/N" strategy often beats unconstrained MVO

3. **Ledoit-Wolf Shrinkage (2004)**:
   - Sample covariance matrix is noisy for N=44 stocks
   - Shrinkage towards structured target reduces error
   - Improves out-of-sample Sharpe Ratio

4. **Convex Formulation (Boyd, 2004)**:
   - Original Max Sharpe is non-convex (ratio of quadratics)
   - Reformulation using variable substitution makes it convex
   - CVXPY finds global optimum (not local)

**Academic Sources:**
- Markowitz (1952) - "Portfolio selection"
- Sharpe (1966) - "Mutual fund performance"
- Ledoit & Wolf (2004) - "Honey, I shrunk the sample covariance matrix"
- DeMiguel et al. (2009) - "Optimal versus naive diversification"

---

## 🔬 Key Technical Details

### **PCA (Principal Component Analysis)**

**Purpose:**
- Reduce 20+ technical indicators to 5-8 components
- Prevents overfitting (curse of dimensionality)
- Removes multicollinearity

**Critical Implementation:**
```python
# CORRECT (Prevents look-ahead bias)
pca = PCA(n_components=0.95)
pca.fit(X_train)  # Fit ONLY on training data
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)  # Apply to test

# WRONG (Look-ahead bias!)
pca.fit(X_all)  # Uses future information!
```

**Academic Justification:**
- Connor & Korajczyk (1988) - Factor models via PCA
- Lettau & Pelger (2020) - PCA factors in asset pricing

---

### **Macro Variables**

**Why include macroeconomic factors?**

1. **Chen, Roll, Ross (1986)**: Economic forces drive stock returns
   - Interest rates affect discount rates
   - Oil prices impact energy sector
   - Currency risk for exporters

2. **Variables we use:**
   - **German 10Y Bund**: Risk-free rate proxy
   - **VSTOXX**: Volatility index (fear gauge)
   - **Brent Oil**: Commodity risk
   - **EUR/USD**: Currency exposure
   - **Gold**: Safe haven indicator

3. **Stationarity:**
   - Take daily differences: $\Delta X_t = X_t - X_{t-1}$
   - Lag by 1 day to prevent look-ahead
   - Standard in time series econometrics

**Academic Sources:**
- Chen, Roll, Ross (1986) - "Economic forces and the stock market"
- Whaley (2000) - "The investor fear gauge" (VIX)

---

### **Transaction Costs**

**Why 10 basis points (0.10%)?**

1. **Institutional Execution Costs**:
   - Large-cap European stocks: 5-15 bps typical
   - Includes: bid-ask spread, market impact, fees
   - Lesmond et al. (2004): Ignoring costs overstates profitability

2. **Monthly Rebalancing**:
   - Balance: Performance vs. transaction costs
   - Carhart (1997): High turnover erodes alpha

**Formula:**
$$\text{Net Return}_t = \text{Gross Return}_t - (\text{Turnover}_t \times 0.001)$$

Where:
$$\text{Turnover}_t = \frac{1}{2} \sum_{i=1}^{N} |w_{i,t} - w_{i,t-1}|$$

**Academic Sources:**
- Lesmond et al. (2004) - "Illusory nature of momentum profits"
- Carhart (1997) - "Persistence in mutual fund performance"

---

## 🧪 Statistical Validation

### **Diebold-Mariano Test**

**Purpose:**
- Test if LSTM forecasts are significantly better than naive benchmark
- Prevents p-hacking and data mining

**Null Hypothesis:**
$$H_0: E[\text{Loss}_{\text{LSTM}}] = E[\text{Loss}_{\text{naive}}]$$

**Test Statistic:**
$$\text{DM} = \frac{\bar{d}}{\sqrt{\widehat{\text{Var}}(d)}}$$

Where $d_t = L(\text{LSTM}_t) - L(\text{naive}_t)$

**Interpretation:**
- Reject $H_0$ if p-value < 0.05
- Confirms LSTM adds value beyond randomness

**Academic Sources:**
- Diebold & Mariano (1995) - "Comparing predictive accuracy"
- Harvey et al. (2016) - "...and the cross-section of expected returns" (multiple testing)

---

## ⚠️ Common Pitfalls We Avoid

### 1. **Look-Ahead Bias**
**Problem:** Using future data to make past predictions
**Our Solution:**
- Strict train/val/test split (2018-2022 / 2023 / 2024-2025)
- Fit scalers, PCA only on training data
- Macro variables lagged by 1 day

**Academic Reference:** Bailey et al. (2014) - "Backtest overfitting"

### 2. **Survivorship Bias**
**Problem:** Only analyzing stocks that survived
**Our Solution:**
- WRDS Compustat includes delisted stocks
- No filtering based on end-of-period survival

**Academic Reference:** Brown et al. (1995) - "Survival"

### 3. **Transaction Costs**
**Problem:** Ignoring costs inflates returns
**Our Solution:**
- 10 bps per trade
- Monthly rebalancing (not daily)

**Academic Reference:** Lesmond et al. (2004)

### 4. **Overfitting**
**Problem:** Model fits noise, not signal
**Our Solution:**
- Early stopping (patience=10 epochs)
- Dropout (20%)
- PCA reduces dimensionality
- Position constraints

**Academic Reference:** Goodfellow et al. (2016) - "Deep Learning"

---

## 📊 Expected Outcomes

### **Performance Metrics** (Targets)

Based on academic literature, we expect:

1. **Sharpe Ratio**: 1.0 - 1.5 (annualized)
   - Benchmark (STOXX 600): ~0.5
   - Equal-weight: ~0.6
   - LSTM Max Sharpe: >1.0 (target)

2. **Information Ratio**: 0.5 - 1.0
   - Measures active return per unit of active risk
   - IR > 0.5 considered "good" (Grinold & Kahn, 2000)

3. **Maximum Drawdown**: <25%
   - Diversification + constraints limit downside
   - Compare to STOXX 600: ~30-40% (2008, 2020)

4. **Turnover**: 30-50% per month
   - Monthly rebalancing with constraints
   - Lower turnover → lower costs

### **Statistical Significance**

- Diebold-Mariano test: p-value < 0.05 (target)
- Confirms LSTM adds value beyond randomness

---

## 🚀 Next Steps

### **Immediate Actions:**

1. ✅ **Review PR**: https://github.com/mattyyychan/KingsInvestmentFund/pull/new/claude/sad-allen
   - Merge foundation code

2. **Run Notebook 01**:
   - Open in Google Colab
   - Connect WRDS, download data
   - Verify 44 stocks selected

3. **Request Notebooks 02-05**:
   - I can create these now if you're ready
   - Each notebook is ~500 lines with detailed explanations

### **Timeline Estimate:**

- **Today**: Merge PR, run Notebook 01 (30 min)
- **Tomorrow**: Feature engineering (Notebook 02) - 1 hour
- **Day 3**: LSTM training (Notebook 03) - 30 min on Colab Pro
- **Day 4**: Optimization (Notebook 04) - 30 min
- **Day 5**: Performance analysis (Notebook 05) - 1 hour

**Total time to full backtest: ~4-5 hours** (spread over 5 days)

---

## 📚 Master Reference List

### Core Theory
1. Markowitz (1952) - Portfolio selection
2. Sharpe (1964) - CAPM
3. Fama & French (1993) - Three-factor model

### Machine Learning
4. Hochreiter & Schmidhuber (1997) - LSTM
5. Gu, Kelly, & Xiu (2020) - ML in asset pricing
6. Fischer & Krauss (2018) - LSTM for stocks

### Risk & Optimization
7. Ledoit & Wolf (2004) - Covariance shrinkage
8. DeMiguel et al. (2009) - 1/N portfolio
9. Boyd (2004) - Convex optimization

### Empirics
10. Chen, Roll, Ross (1986) - Macro factors
11. Jegadeesh & Titman (1993) - Momentum
12. Amihud (2002) - Liquidity

### Methodology
13. Diebold & Mariano (1995) - Forecast testing
14. Harvey et al. (2016) - Multiple testing
15. Bailey et al. (2014) - Backtest overfitting

---

## ❓ FAQs

**Q: Why 44 stocks? Why not more?**
- Diversification benefits plateau after ~30-40 stocks (Evans & Archer, 1968)
- 44 = 4 per sector × 11 sectors (perfect balance)
- More stocks → more LSTMs to train → diminishing returns

**Q: Why individual LSTMs instead of one pooled model?**
- Captures stock-specific dynamics
- Easier to debug (isolate underperforming stocks)
- Academic precedent (Feng et al., 2019)

**Q: Why Max Sharpe instead of Max Return?**
- Risk-adjusted performance is what matters
- Leverage can scale returns but not Sharpe
- Industry standard (mutual funds, ETFs)

**Q: Why monthly rebalancing?**
- Balance: Responsiveness vs. transaction costs
- Standard in academic studies (DeMiguel et al., 2009)
- Aligns with monthly reporting cycles

**Q: Can I use this for real trading?**
- This is an **academic research project**
- Not financial advice
- Real implementation requires:
  - Regulatory compliance
  - Robust infrastructure
  - Risk management beyond backtest

---

**Ready to continue?** Let me know if you want me to create Notebooks 02-05 now, or if you want to run Notebook 01 first and verify the setup works!
