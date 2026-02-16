# European LSTM Portfolio Strategy - Simple Explanation

## 🎯 What Are We Building?

**A European stock portfolio that uses AI to pick stocks and optimize weights.**

Think of it like a 3-step process:

```
Step 1: AI predicts which stocks will outperform
    ↓
Step 2: Remove market risk (focus on stock-picking skill)
    ↓
Step 3: Build optimal portfolio with those predictions
```

---

## 📈 The Complete Strategy (In Plain English)

### **STEP 1: Pick 44 European Stocks**

**What we do:**
- Start with ~600 European stocks
- Filter for big, liquid companies (€2B+ market cap)
- Select 4 stocks from each industry sector
- Result: 44 stocks, balanced across industries

**Why?**
- Big stocks = easy to trade
- Balanced sectors = don't bet everything on one industry
- 44 is enough for diversification

**Academic backing:** Fama & French (1993) - sector diversification reduces risk

---

### **STEP 2: Collect Data (7+ Years)**

**What we download:**

1. **Stock prices** (daily, 2018-2025)
2. **Financial ratios** (P/E, debt, profitability)
3. **Market benchmark** (STOXX 600 - like S&P 500 for Europe)
4. **Macro data**:
   - Interest rates (German bonds)
   - Oil prices
   - Volatility index (fear gauge)
   - Euro/Dollar exchange rate

**Why all this data?**
- AI needs lots of information to find patterns
- Macro data = economy-wide trends
- Stock data = company-specific performance

---

### **STEP 3: Remove Market Risk (The "Beta" Trick)**

**The Problem:**
- If the market rises 2%, most stocks rise too
- A stock going up 2.5% isn't necessarily "good picking"
- It might just be following the market

**Our Solution - Calculate "Residual Returns":**

```
Residual Return = Stock Return - (Beta × Market Return)
```

**Example:**
- Market (STOXX 600) goes up 2%
- Stock has Beta = 1.5 (50% more volatile than market)
- Stock goes up 3%

**Calculation:**
- Expected return (from market) = 1.5 × 2% = 3%
- Actual return = 3%
- Residual return = 3% - 3% = **0%**

**Interpretation:** Stock did NOT outperform - it just followed the market!

**Academic backing:**
- Sharpe (1964) - CAPM (Capital Asset Pricing Model)
- Beta measures systematic risk
- Residual = alpha (stock-picking skill)

---

### **STEP 4: Create Features for AI**

**We calculate 20+ indicators for each stock:**

**Technical Indicators** (price patterns):
- RSI (overbought/oversold)
- MACD (momentum)
- Bollinger Bands (volatility)
- Past returns (5, 10, 20, 60 days)

**Fundamental Indicators** (financial health):
- P/E ratio (valuation)
- Debt/Equity (leverage)
- ROE (profitability)

**Macro Indicators** (economy):
- Interest rate changes
- Oil price movements
- Volatility spikes

**Then we simplify with PCA:**
- 20+ indicators → 5-8 "principal components"
- Removes noise, keeps important patterns
- Prevents AI from overfitting

**Academic backing:**
- Connor & Korajczyk (1988) - PCA for factor models
- Hastie et al. (2009) - Curse of dimensionality

---

### **STEP 5: Train AI (LSTM Neural Network)**

**What it does:**
- Predicts: "Will this stock's residual return be positive?"
- Not: "Will the stock go up?"
- Instead: "Will it beat its expected return based on market movement?"

**How it works:**
- Looks at past 60 days of data
- Finds patterns humans can't see
- Outputs: Probability (0-100%) that stock will outperform

**Architecture:**
```
Input: 60 days of history (prices, indicators, macro)
    ↓
LSTM Layer 1 (64 neurons) - learns short-term patterns
    ↓
LSTM Layer 2 (32 neurons) - learns long-term patterns
    ↓
Output: Probability (0-1) that residual return > 0
```

**Why LSTM?**
- LSTM = "Long Short-Term Memory"
- Designed for time series (sequences)
- Better than regular neural networks for stock data

**We train 44 separate models:**
- One LSTM for each stock
- Each stock has unique patterns
- Training time: 15-30 min on GPU, 2-4 hours on CPU

**Academic backing:**
- Hochreiter (1997) - LSTM architecture
- Fischer & Krauss (2018) - LSTM beats traditional ML for stocks
- Gu, Kelly, Xiu (2020) - ML in asset pricing

---

### **STEP 6: Build Portfolio (Max Sharpe Optimization)**

**Now we have:**
- 44 stocks
- AI probability for each stock (e.g., 65% chance of outperforming)

**Goal:** Decide how much money to put in each stock

**Simple approach (what we DON'T do):**
- Equal weight: 1/44 = 2.27% in each stock
- Problem: Ignores predictions and risk!

**Our approach - Maximize Sharpe Ratio:**

**Sharpe Ratio Formula:**
```
Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Risk
```

**Interpretation:**
- Return per unit of risk
- Higher = better risk-adjusted performance
- Example: Sharpe = 1.5 means you get 1.5% extra return per 1% of risk

**Optimization Problem:**
```
Maximize: Sharpe Ratio
Subject to:
  - All weights sum to 100% (fully invested)
  - All weights ≥ 0 (long-only, no shorting)
  - No stock > 10% (don't concentrate too much)
  - No sector > 15% (diversify across industries)
```

**How we solve it:**
- Use CVXPY (convex optimization library)
- Finds mathematically optimal weights
- Considers both return predictions AND risk (covariance)

**Academic backing:**
- Markowitz (1952) - Mean-variance optimization
- Sharpe (1966) - Sharpe Ratio
- DeMiguel et al. (2009) - Constraints improve performance

---

### **STEP 7: Backtest & Evaluate**

**Monthly rebalancing:**
- Month 1: Use AI predictions to set weights
- Month 2: Recalculate predictions, adjust weights
- Repeat for entire test period (2024-2025)

**Transaction costs:**
- Every time we trade, we lose 0.10% (10 basis points)
- Realistic for institutional trading
- Prevents fake profits from ignoring costs

**Compare to benchmarks:**

| Strategy | Expected Sharpe | Description |
|----------|----------------|-------------|
| STOXX 600 | ~0.5 | Just buy the market |
| Equal Weight | ~0.6 | 1/44 in each stock |
| Min Variance | ~0.7 | Minimize risk only |
| **Our LSTM** | **>1.0** | AI + optimization |

**Statistical test (Diebold-Mariano):**
- Checks if our AI is significantly better than random guessing
- p-value < 0.05 → statistically significant

**Academic backing:**
- Carhart (1997) - Transaction costs matter
- Diebold & Mariano (1995) - Forecast testing
- Harvey et al. (2016) - Avoid p-hacking

---

## 🧠 Why This Strategy is Academically Sound

### **1. We Remove Market Risk (Beta)**
- **Problem:** Stocks go up when market goes up (not skill)
- **Solution:** Predict residual returns (alpha)
- **Academic:** CAPM (Sharpe, 1964)

### **2. We Use AI for Pattern Recognition**
- **Problem:** Humans miss complex, non-linear patterns
- **Solution:** LSTM neural network
- **Academic:** Fischer & Krauss (2018), Gu et al. (2020)

### **3. We Optimize Risk-Adjusted Returns**
- **Problem:** High returns don't matter if risk is too high
- **Solution:** Maximize Sharpe Ratio
- **Academic:** Markowitz (1952), Sharpe (1966)

### **4. We Avoid Common Pitfalls**
- **Look-ahead bias:** Train on 2018-2022, test on 2024-2025
- **Survivorship bias:** Include delisted stocks (WRDS)
- **Transaction costs:** 10 bps per trade
- **Overfitting:** Early stopping, dropout, PCA
- **Academic:** Bailey et al. (2014), Brown et al. (1995)

---

## 🔄 The Complete Workflow (Visual)

```
┌─────────────────────────────────────────┐
│ STEP 1: Universe Selection              │
│ - 600 stocks → 44 stocks                │
│ - Liquidity filters                     │
│ - Sector balance                        │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ STEP 2: Data Collection                 │
│ - Stock prices (2018-2025)              │
│ - Fundamentals (P/E, debt, etc.)        │
│ - Macro data (rates, oil, volatility)   │
│ - Benchmark (STOXX 600)                 │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ STEP 3: Feature Engineering             │
│ - Calculate rolling beta                │
│ - Compute residual returns              │
│ - Generate 20+ technical indicators     │
│ - Apply PCA (20→5-8 components)         │
│ - Create binary target (up/down)        │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ STEP 4: LSTM Training                   │
│ - Train 44 separate models              │
│ - Input: 60 days history                │
│ - Output: Probability (0-1)             │
│ - Validation: 2023                      │
│ - Test: 2024-2025                       │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ STEP 5: Portfolio Optimization          │
│ - Convert probabilities → expected returns │
│ - Calculate covariance matrix           │
│ - Maximize Sharpe Ratio (CVXPY)         │
│ - Apply constraints (10% max/stock)     │
│ - Monthly rebalancing                   │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ STEP 6: Performance Evaluation          │
│ - Calculate Sharpe Ratio                │
│ - Compare to benchmarks                 │
│ - Statistical significance test         │
│ - Report results                        │
└─────────────────────────────────────────┘
```

---

## 📊 Example: How It Works in Practice

### **Month 1 (January 2024):**

**AI Predictions:**
- Stock A (Tech): 75% probability of positive residual return
- Stock B (Energy): 45% probability
- Stock C (Finance): 60% probability
- ... (41 more stocks)

**Optimizer says:**
- Put 8% in Stock A (high confidence + low correlation)
- Put 2% in Stock B (low confidence)
- Put 5% in Stock C (medium confidence)
- ... total = 100%

**Actual results (end of month):**
- Stock A: +2.5% (beat its beta-expected return)
- Stock B: -1.0% (underperformed)
- Stock C: +1.2% (outperformed)
- **Portfolio:** +1.8% (after transaction costs)

### **Month 2 (February 2024):**
- Recalculate AI predictions
- Reoptimize portfolio
- Execute trades (pay transaction costs)
- Repeat...

---

## ✅ Current Status

### **What's Done:**
- ✅ All Python code (config, data loaders, LSTM, optimizer)
- ✅ Academic methodology documented
- ✅ Notebook 01 (universe selection)
- ✅ Implementation guide

### **What's Next:**
- ⏳ Notebooks 02-05 (I can create these now)
- ⏳ Run notebooks sequentially
- ⏳ Generate backtest results
- ⏳ Write academic report

### **Timeline:**
- **Today:** Create remaining notebooks (2 hours)
- **This Week:** Run all notebooks on Colab Pro (4-5 hours total)
- **Next Week:** Analyze results, write report

---

## ❓ Key Questions Answered

**Q: Is this a real trading strategy?**
- A: It's an **academic research project**
- Shows how to combine ML with portfolio theory
- Not financial advice

**Q: Will it beat the market?**
- A: Unknown until we backtest!
- Academic research shows ML can add value (Gu et al., 2020)
- But many strategies fail out-of-sample
- We'll know after running the notebooks

**Q: Why is this better than just buying STOXX 600?**
- A: Active management (stock selection)
- Risk optimization (Sharpe maximization)
- AI captures non-linear patterns
- **BUT:** Only works if AI predictions are good!

**Q: How do I know if it's working?**
- A: Statistical tests (Diebold-Mariano)
- Sharpe Ratio > benchmarks
- Positive Information Ratio
- Results must be significant (p < 0.05)

---

## 🚀 What You Need to Do Now

### **Option 1: Continue Building (Recommended)**
I can create Notebooks 02-05 right now:
- Notebook 02: Feature engineering (1 hour to run)
- Notebook 03: LSTM training (30 min on Colab Pro)
- Notebook 04: Portfolio optimization (30 min)
- Notebook 05: Performance analysis (1 hour)

**Say:** "Create the remaining 4 notebooks"

### **Option 2: Run What We Have**
Test Notebook 01 first:
1. Upload project to Google Drive
2. Open Notebook 01 in Colab
3. Run all cells
4. Verify data downloads correctly

**Say:** "I'll test Notebook 01 first"

### **Option 3: Ask Questions**
If anything is still unclear:

**Say:** "Explain [specific topic] in more detail"

---

## 📖 One-Sentence Summary

**We use LSTM neural networks to predict which European stocks will outperform (after removing market risk), then build an optimal portfolio by maximizing the Sharpe Ratio with constraints, validated against academic benchmarks.**

---

**What would you like to do next?**
