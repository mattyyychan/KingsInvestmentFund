# European LSTM Portfolio: Academic Methodology & Implementation Guide

## 📚 Theoretical Foundation

This document explains the academic rationale behind each component of the strategy, with citations to peer-reviewed research.

---

## 🎯 Overall Strategy Architecture

### Three-Layer Approach

```
Layer 1: Signal Generation (LSTM Neural Networks)
    ↓
Layer 2: Risk Management (Systematic Risk Removal)
    ↓
Layer 3: Portfolio Construction (Maximum Sharpe Optimization)
```

**Academic Precedent:**
- Gu, Kelly, & Xiu (2020). "Empirical Asset Pricing via Machine Learning." *Review of Financial Studies*, 33(5), 2223-2273.
  - Demonstrates superiority of machine learning models (including neural networks) in predicting stock returns
  - Shows that ensemble approaches combining ML with traditional factors improve performance

---

## 📊 STEP 1: Universe Selection & Data Collection

### Methodology: Stratified Sector Sampling

**What We Do:**
- Select 4-5 stocks per GICS sector (11 sectors × 4 = 44 stocks)
- Filter for liquidity: €2B+ market cap, €1M+ daily volume
- Ensure 5+ years continuous history

**Academic Justification:**

**1. Sector Diversification**
- Fama & French (1993). "Common risk factors in the returns on stocks and bonds." *Journal of Financial Economics*, 33(1), 3-56.
  - Industry effects are significant determinants of cross-sectional returns
  - Sector diversification reduces idiosyncratic risk

**2. Liquidity Screening**
- Amihud (2002). "Illiquidity and stock returns: cross-section and time-series effects." *Journal of Financial Markets*, 5(1), 31-56.
  - Illiquid stocks command a liquidity premium but are costly to trade
  - High-turnover strategies require liquid stocks to minimize implementation costs

**3. Market Capitalization Filter**
- Banz (1981). "The relationship between return and market value of common stocks." *Journal of Financial Economics*, 9(1), 3-18.
  - Size effect exists but very small/illiquid stocks have stale prices
  - €2B threshold ensures institutional-quality stocks

**Implementation in Code:**
```python
# From config.py
MIN_MARKET_CAP = 2_000_000_000  # €2 Billion (Amihud 2002)
MIN_AVG_VOLUME = 1_000_000      # €1 Million daily
STOCKS_PER_SECTOR = 4           # Fama-French sector diversification
```

---

## 🔬 STEP 2: Feature Engineering

### 2.1 Systematic Risk Removal (Residual Returns)

**What We Do:**
Calculate rolling 60-day beta (β) and compute:
$$r_{residual,t} = r_{stock,t} - \beta_t \times r_{market,t}$$

**Academic Justification:**

**1. Capital Asset Pricing Model (CAPM)**
- Sharpe (1964). "Capital asset prices: A theory of market equilibrium under conditions of risk." *Journal of Finance*, 19(3), 425-442.
- Lintner (1965). "The valuation of risk assets and the selection of risky investments in stock portfolios and capital budgets." *Review of Economics and Statistics*, 47(1), 13-37.

**CAPM Equation:**
$$E[R_i] = R_f + \beta_i (E[R_m] - R_f)$$

- Beta (β) measures systematic risk exposure
- Residual return isolates **alpha** (idiosyncratic performance)

**2. Rolling Beta Estimation**
- Ferson & Harvey (1991). "The variation of economic risk premiums." *Journal of Political Economy*, 99(2), 385-415.
  - Time-varying betas better capture changing market exposures
  - 60-day window balances responsiveness vs. estimation error

**Why This Matters:**
- LSTM predicts **alpha** (stock-specific returns), not market direction
- Removes confounding factor: a stock may rise simply because the market rises
- Focuses on manager skill: selecting stocks that outperform their beta-expected return

**Implementation:**
```python
# From src/features.py
class ResidualReturnCalculator:
    def calculate_rolling_beta(self, stock_returns, market_returns):
        rolling_cov = stock_returns.rolling(60).cov(market_returns)
        rolling_var = market_returns.rolling(60).var()
        beta = rolling_cov / rolling_var
        return beta

    def calculate_residual_returns(self, stock_returns, market_returns, beta):
        residuals = stock_returns - (beta * market_returns)
        return residuals
```

### 2.2 Technical Indicators

**What We Do:**
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Multi-period returns (5, 10, 20, 60 days)
- Rolling volatility

**Academic Justification:**

**1. Momentum & Reversal**
- Jegadeesh & Titman (1993). "Returns to buying winners and selling losers: Implications for stock market efficiency." *Journal of Finance*, 48(1), 65-91.
  - Momentum persists over 3-12 month horizons
  - Technical indicators capture this effect

**2. Volatility as a Predictive Signal**
- Ang et al. (2006). "The cross-section of volatility and expected returns." *Journal of Finance*, 61(1), 259-299.
  - Idiosyncratic volatility is inversely related to future returns
  - Rolling volatility provides predictive information

**3. Technical Analysis in Academic Literature**
- Brock, Lakonishok, & LeBaron (1992). "Simple technical trading rules and the stochastic properties of stock returns." *Journal of Finance*, 47(5), 1731-1764.
  - Simple technical indicators (moving averages) have predictive power
  - Contradicts weak-form EMH

### 2.3 Macroeconomic Variables

**What We Do:**
- German 10Y Bund Yield (risk-free rate proxy)
- VSTOXX (European equity volatility index)
- Brent Crude Oil (commodity risk)
- EUR/USD (currency risk)
- Gold (safe haven asset)

**Academic Justification:**

**1. Macro Factors and Stock Returns**
- Chen, Roll, & Ross (1986). "Economic forces and the stock market." *Journal of Business*, 59(3), 383-403.
  - Macroeconomic variables (interest rates, inflation, industrial production) explain stock returns
  - Five-factor model includes term spread and default spread

**2. Volatility Index (VSTOXX)**
- Whaley (2000). "The investor fear gauge." *Journal of Portfolio Management*, 26(3), 12-17.
  - VIX (and European equivalent VSTOXX) predicts future market returns
  - High volatility → risk aversion → lower expected returns

**3. Oil Prices**
- Jones & Kaul (1996). "Oil and the stock markets." *Journal of Finance*, 51(2), 463-491.
  - Oil price shocks affect stock returns, especially in energy and transport sectors

**4. Exchange Rates**
- Dumas & Solnik (1995). "The world price of foreign exchange risk." *Journal of Finance*, 50(2), 445-479.
  - Currency risk is priced in international equity returns

**Implementation:**
```python
# From config.py
MACRO_VARIABLES = {
    'BUND_10Y': '^TNX',     # Chen, Roll, Ross (1986)
    'VSTOXX': '^V2TX',      # Whaley (2000)
    'BRENT': 'BZ=F',        # Jones & Kaul (1996)
    'EURUSD': 'EURUSD=X',   # Dumas & Solnik (1995)
    'GOLD': 'GC=F'          # Safe haven
}
```

### 2.4 PCA (Principal Component Analysis)

**What We Do:**
- Apply PCA to 20+ technical indicators
- Keep components explaining 95% variance
- Fit PCA **only on training data** (prevent look-ahead bias)

**Academic Justification:**

**1. Dimensionality Reduction**
- Connor & Korajczyk (1988). "Risk and return in an equilibrium APT: Application of a new test methodology." *Journal of Financial Economics*, 21(2), 255-289.
  - APT (Arbitrage Pricing Theory) uses factor analysis to reduce dimensionality
  - PCA extracts common factors from multiple signals

**2. Curse of Dimensionality in ML**
- Hastie, Tibshirani, & Friedman (2009). *The Elements of Statistical Learning*. Springer.
  - High-dimensional feature spaces lead to overfitting
  - PCA mitigates multicollinearity

**3. Finance Applications**
- Lettau & Pelger (2020). "Factors that fit the time series and cross-section of stock returns." *Review of Financial Studies*, 33(5), 2274-2325.
  - PCA-based factors (similar to Fama-French) explain returns

**Why 95% Variance?**
- Standard threshold in applied statistics (Kaiser criterion)
- Balances information retention vs. noise reduction

**Critical: Fit on Training Data Only**
```python
# From src/features.py - CORRECT
if train_mask is not None:
    self.pca.fit(X_scaled[train_mask])  # Fit ONLY on train
    X_pca = self.pca.transform(X_scaled)  # Transform all
else:
    X_pca = self.pca.fit_transform(X_scaled)  # Wrong if not train-only!
```

**Why This Matters:**
- Bailey et al. (2014). "Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance." *Notices of the AMS*, 61(5), 458-471.
  - Look-ahead bias inflates backtest performance
  - Real-world implementation would not have access to future data

---

## 🤖 STEP 3: LSTM Neural Networks

### Architecture: Stacked LSTM

**What We Do:**
```
Input (60 timesteps, n_features)
    ↓
LSTM Layer 1 (64 units, return_sequences=True, dropout=0.2)
    ↓
LSTM Layer 2 (32 units, return_sequences=False, dropout=0.2)
    ↓
Dense Output (1 unit, sigmoid activation)
```

**Academic Justification:**

**1. LSTM for Time Series**
- Hochreiter & Schmidhuber (1997). "Long short-term memory." *Neural Computation*, 9(8), 1735-1780.
  - LSTMs solve vanishing gradient problem in RNNs
  - Can capture long-term dependencies in sequences

**2. Financial Applications**
- Fischer & Krauss (2018). "Deep learning with long short-term memory networks for financial market predictions." *European Journal of Operational Research*, 270(2), 654-669.
  - LSTMs outperform traditional ML (random forests, logistic regression) for stock prediction
  - 60-day lookback captures medium-term patterns

**3. Dropout for Regularization**
- Srivastava et al. (2014). "Dropout: a simple way to prevent neural networks from overfitting." *Journal of Machine Learning Research*, 15(1), 1929-1958.
  - 20% dropout prevents overfitting in deep networks
  - Especially critical for financial data (high noise-to-signal ratio)

**4. Binary Classification (Sigmoid Output)**
- Our target: `1 if residual_return > 0, else 0`
- Output: Probability that residual return will be positive
- Cross-entropy loss is appropriate for binary targets

**Why Not Regression?**
- Predicting exact returns is extremely difficult (low R²)
- Classification (up/down) is more robust
- Gu, Kelly, & Xiu (2020) show classification tasks often outperform regression in finance

**Individual Stock Models (Not Pooled):**
- We train **44 separate LSTMs** (one per stock)
- Alternative: Single model with stock embeddings (pooled approach)

**Justification:**
- Feng, Polson, & Xu (2019). "Deep learning in characteristics-sorted portfolios." *SSRN Working Paper*.
  - Stock-level heterogeneity matters
  - Individual models capture stock-specific dynamics

**Early Stopping:**
```python
# From src/models.py
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,  # Stop if no improvement for 10 epochs
    restore_best_weights=True
)
```

**Justification:**
- Prevents overfitting (Goodfellow et al., 2016, *Deep Learning*)
- Critical for CPU efficiency (saves time on doomed models)
- Academic standard in ML research

---

## 📈 STEP 4: Portfolio Optimization (Maximum Sharpe Ratio)

### Objective: Maximize Sharpe Ratio

**Mathematical Formulation:**

**Original Problem:**
$$\max_{w} \frac{w^T \mu - r_f}{\sqrt{w^T \Sigma w}}$$

Subject to:
- $\sum_{i=1}^{N} w_i = 1$ (full investment)
- $w_i \geq 0$ (long-only)
- $w_i \leq 0.10$ (max 10% per stock)
- $\sum_{i \in \text{sector}} w_i \leq 0.15$ (max 15% per sector)

**Problem:** Non-convex (ratio of quadratic forms)

**Convex Reformulation:**

Let $y = \frac{w}{\kappa}$ where $\kappa = w^T \mu$

Then:
$$\min_{y} y^T \Sigma y$$

Subject to:
- $y^T \mu = 1$ (normalization)
- $y \geq 0$ (long-only)

Finally: $w = \frac{y}{\sum y_i}$

**Academic Justification:**

**1. Sharpe Ratio as Optimality Criterion**
- Sharpe (1966). "Mutual fund performance." *Journal of Business*, 39(1), 119-138.
  - Sharpe Ratio measures risk-adjusted performance
  - Universally accepted in finance academia and practice

**2. Mean-Variance Optimization**
- Markowitz (1952). "Portfolio selection." *Journal of Finance*, 7(1), 77-91.
  - Modern Portfolio Theory foundation
  - Mean-variance efficiency is optimal under quadratic utility

**3. Convex Reformulation**
- Boyd & Vandenberghe (2004). *Convex Optimization*. Cambridge University Press.
  - Convex problems guarantee global optimum
  - CVXPY implements interior-point methods (polynomial time)

**4. Position Size Constraints**
- DeMiguel, Garlappi, & Uppal (2009). "Optimal versus naive diversification: How inefficient is the 1/N portfolio strategy?" *Review of Financial Studies*, 22(5), 1915-1953.
  - Unconstrained optimization suffers from estimation error
  - Constraints improve out-of-sample performance

**Why 10% Position Limit?**
- Standard in institutional asset management
- Prevents overconcentration from estimation error
- UCITS (EU fund regulation) mandates 10% max for single issuer

**Why 15% Sector Limit?**
- Industry practice (e.g., mutual fund prospectuses)
- Ensures sector diversification (Fama-French logic)

### Covariance Matrix Estimation: Ledoit-Wolf Shrinkage

**What We Do:**
$$\Sigma_{\text{shrunk}} = \delta F + (1 - \delta) S$$

Where:
- $S$ = sample covariance
- $F$ = shrinkage target (constant correlation)
- $\delta$ = shrinkage intensity (data-driven)

**Academic Justification:**

**1. Ledoit-Wolf Shrinkage**
- Ledoit & Wolf (2004). "Honey, I shrunk the sample covariance matrix." *Journal of Portfolio Management*, 30(4), 110-119.
  - Sample covariance is noisy for high-dimensional portfolios
  - Shrinkage improves out-of-sample performance
  - Optimal shrinkage intensity derived analytically

**2. Why Shrinkage Matters:**
- Jagannathan & Ma (2003). "Risk reduction in large portfolios: Why imposing the wrong constraints helps." *Journal of Finance*, 58(4), 1651-1683.
  - Estimation error in covariance dominates in medium-sized portfolios (N=50)
  - Shrinkage reduces extreme portfolio weights

**Implementation:**
```python
# From src/optimization.py
from sklearn.covariance import LedoitWolf

lw = LedoitWolf()
lw.fit(returns)  # Returns (T x N) matrix
covariance_shrunk = lw.covariance_
```

### Expected Returns from LSTM Probabilities

**What We Do:**
$$E[r_i] = p_i \times \text{Avg}_{\text{up}} + (1 - p_i) \times \text{Avg}_{\text{down}}$$

Where:
- $p_i$ = LSTM probability of positive residual return
- $\text{Avg}_{\text{up}}$ = Average return when residual > 0 (from training data)
- $\text{Avg}_{\text{down}}$ = Average return when residual < 0 (from training data)

**Academic Justification:**

**1. Expected Value Calculation**
- Standard probability theory (no citation needed)
- Avoids estimating returns directly (high error)

**2. Why Use Training Data for Avg Returns?**
- Prevents look-ahead bias
- Test-set returns are unknown in real-time implementation
- Consistent with academic backtest standards (Harvey, Liu, & Zhu, 2016)

**3. Alternative Approach (Not Used):**
- Black-Litterman (1992) model: Combines market equilibrium with investor views
- More complex; LSTM probabilities serve as "views"

---

## 📊 STEP 5: Backtesting & Performance Evaluation

### Benchmark Strategies

**1. Equal-Weight (1/N)**
- DeMiguel et al. (2009): "1/N" strategy often beats mean-variance optimization out-of-sample
- Simple diversification baseline

**2. STOXX Europe 600 (Market Benchmark)**
- Passive market-cap weighted index
- Represents "do nothing" alternative

**3. Minimum Variance**
- Haugen & Baker (1991). "The efficient market inefficiency of capitalization-weighted stock portfolios." *Journal of Portfolio Management*, 17(3), 35-40.
- Low-volatility anomaly: Min variance beats market empirically

### Performance Metrics

**1. Sharpe Ratio**
$$\text{Sharpe} = \frac{E[R_p - R_f]}{\sigma_p}$$

- Sharpe (1994). "The Sharpe ratio." *Journal of Portfolio Management*, 21(1), 49-58.
- Standard measure of risk-adjusted return

**2. Maximum Drawdown**
$$\text{MDD} = \max_{t \in [0,T]} \left( \max_{s \in [0,t]} P_s - P_t \right) / \max_{s \in [0,t]} P_s$$

- Magdon-Ismail & Atiya (2004). "Maximum drawdown." *Risk Magazine*, 17(10), 99-102.
- Captures worst-case loss (important for risk management)

**3. Information Ratio**
$$\text{IR} = \frac{E[R_p - R_b]}{\sigma(R_p - R_b)}$$

- Goodwin (1998). "The information ratio." *Financial Analysts Journal*, 54(4), 34-43.
- Measures active return per unit of active risk

**4. Portfolio Turnover**
$$\text{Turnover}_t = \frac{1}{2} \sum_{i=1}^{N} |w_{i,t} - w_{i,t-1}|$$

- Carhart (1997). "On persistence in mutual fund performance." *Journal of Finance*, 52(1), 57-82.
- High turnover → high transaction costs → lower net returns

### Statistical Significance Testing: Diebold-Mariano Test

**What We Do:**
Test if LSTM forecasts are significantly better than naive benchmark (e.g., random walk)

**Null Hypothesis:** $H_0: E[\text{Loss}_{\text{LSTM}}] = E[\text{Loss}_{\text{benchmark}}]$

**Test Statistic:**
$$\text{DM} = \frac{\bar{d}}{\sqrt{\text{Var}(d)/T}}$$

Where $d_t = \text{Loss}_{\text{LSTM},t} - \text{Loss}_{\text{benchmark},t}$

**Academic Justification:**

- Diebold & Mariano (1995). "Comparing predictive accuracy." *Journal of Business & Economic Statistics*, 13(3), 253-263.
  - Tests if two forecasting methods differ significantly
  - Appropriate for time series (accounts for autocorrelation)

**Why This Matters:**
- Harvey, Liu, & Zhu (2016). "...and the cross-section of expected returns." *Review of Financial Studies*, 29(1), 5-68.
  - Many published anomalies disappear when properly tested
  - Multiple testing adjustment critical (we use single test here)

**Implementation:**
```python
# From statsmodels
from statsmodels.tsa.stattools import acovf

def diebold_mariano_test(errors_lstm, errors_benchmark):
    d = errors_lstm - errors_benchmark
    dbar = np.mean(d)
    gamma_0 = np.var(d, ddof=1)
    T = len(d)
    dm_stat = dbar / np.sqrt(gamma_0 / T)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value
```

---

## 🔒 Preventing Common Pitfalls

### 1. Look-Ahead Bias (Data Leakage)

**Problem:**
- Using future information to make past predictions
- Overly optimistic backtest results

**Our Solution:**
```python
# CORRECT: Fit on training data only
TRAIN_START = '2018-01-01'
TRAIN_END = '2022-12-31'
TEST_START = '2024-01-01'

scaler = StandardScaler()
scaler.fit(X_train)  # Fit on train
X_test_scaled = scaler.transform(X_test)  # Apply to test
```

**Academic Reference:**
- Bailey et al. (2014). "Pseudo-Mathematics and Financial Charlatanism." *Notices of the AMS*.

### 2. Survivorship Bias

**Problem:**
- Only including stocks that survived entire period
- Ignores bankruptcies/delistings

**Our Solution:**
- Use WRDS Compustat (includes delisted stocks)
- Filter applied at start of period only

**Academic Reference:**
- Brown, Goetzmann, & Ross (1995). "Survival." *Journal of Finance*, 50(3), 853-873.

### 3. Transaction Costs

**Problem:**
- Ignoring costs overstates performance

**Our Solution:**
- 10 bps per trade (realistic for institutional execution)
- Monthly rebalancing (balance performance vs. costs)

**Academic Reference:**
- Lesmond, Schill, & Zhou (2004). "The illusory nature of momentum profits." *Journal of Financial Economics*, 71(2), 349-380.

---

## 📖 Complete Reference List

### Foundational Theory
1. **Markowitz (1952)** - Portfolio Selection (Mean-Variance Optimization)
2. **Sharpe (1964)** - CAPM (Beta, Systematic Risk)
3. **Fama & French (1993)** - Three-Factor Model (Size, Value, Market)
4. **Jegadeesh & Titman (1993)** - Momentum Effect

### Machine Learning in Finance
5. **Gu, Kelly, & Xiu (2020)** - Empirical Asset Pricing via Machine Learning
6. **Fischer & Krauss (2018)** - Deep Learning with LSTM for Finance
7. **Feng, Polson, & Xu (2019)** - Deep Learning in Portfolios

### Risk Management & Estimation
8. **Ledoit & Wolf (2004)** - Covariance Shrinkage
9. **DeMiguel, Garlappi, & Uppal (2009)** - 1/N Portfolio
10. **Jagannathan & Ma (2003)** - Constraints Reduce Estimation Error

### Macro Factors
11. **Chen, Roll, & Ross (1986)** - Economic Forces and Stock Market
12. **Jones & Kaul (1996)** - Oil and Stock Markets
13. **Whaley (2000)** - VIX (Volatility Index)

### Statistical Testing
14. **Diebold & Mariano (1995)** - Comparing Predictive Accuracy
15. **Harvey, Liu, & Zhu (2016)** - Multiple Testing in Finance

### Pitfalls & Biases
16. **Bailey et al. (2014)** - Backtest Overfitting
17. **Brown et al. (1995)** - Survivorship Bias
18. **Lesmond et al. (2004)** - Transaction Costs

---

## ✅ Implementation Checklist

Before running your backtest, verify:

- [ ] Train/Val/Test split is strict (no overlap)
- [ ] Scalers fit only on training data
- [ ] PCA fit only on training data
- [ ] Expected returns calculated from training statistics
- [ ] Transaction costs applied (10 bps)
- [ ] Covariance uses shrinkage
- [ ] Benchmarks include 1/N and market index
- [ ] Statistical significance tested (Diebold-Mariano)
- [ ] Results reported with standard errors
- [ ] Code is reproducible (random seed set)

---

**Next:** Proceed to Jupyter notebooks to implement this methodology step-by-step.
