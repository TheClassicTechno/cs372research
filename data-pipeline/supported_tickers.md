
# Universe Design Principles

## Motivation

A multi-agent trading system can only be evaluated meaningfully if its opportunity set forces real macroeconomic and risk tradeoffs. If the universe is homogeneous (e.g., all mega-cap technology), portfolio performance becomes a momentum proxy rather than a test of structured reasoning.

Therefore, we design a **20-stock universe** with deliberate cross-factor dispersion to:

1. Induce sector rotation pressure
2. Create macro regime sensitivity
3. Force disagreement between agents (macro, value, risk, technical, sentiment)
4. Make Sharpe ratio sensitive to allocation quality

The goal is not alpha maximization but **reasoning evaluation under uncertainty**.

---

# Factor Exposure Framework

To formalize universe diversity, we characterize each stock along three interpretable axes:

### 1. Beta (Market Sensitivity)

Approximate equity beta relative to S&P 500.

* Low (< 0.9): Defensive
* Medium (0.9-1.2): Market-like
* High (> 1.2): Risk-on / volatile

### 2. Duration Proxy (Rate Sensitivity)

A qualitative proxy for interest-rate sensitivity:

* High Duration: Long-dated growth cash flows (AI, SaaS, biotech)
* Medium: Mixed exposure
* Low: Asset-heavy, commodity, or defensive cash-flow profiles

### 3. Cyclicality Score (1-5)

How sensitive the firm is to economic expansion/contraction:

* 1 = Highly defensive
* 3 = Moderately cyclical
* 5 = Highly cyclical

---

# Factor Exposure Table

| Ticker | Company                 | Sector                   | Beta          | Duration Proxy | Cyclicality (1-5) |
| ------ | ----------------------- | ------------------------ | ------------- | -------------- | ----------------- |
| NVDA   | NVIDIA Corporation      | Technology               | High (~1.6)   | High           | 4                 |
| MSFT   | Microsoft Corporation   | Technology               | Medium (~1.1) | High           | 3                 |
| AAPL   | Apple Inc.              | Technology               | Medium (~1.2) | Medium         | 3                 |
| GOOG   | Alphabet Inc.           | Communication Services   | Medium (~1.1) | High           | 3                 |
| AMZN   | Amazon.com Inc.         | Consumer Discretionary   | High (~1.2)   | High           | 4                 |
| META   | Meta Platforms Inc.     | Communication Services   | High (~1.3)   | High           | 4                 |
| TSLA   | Tesla Inc.              | Consumer Discretionary   | High (~1.8)   | High           | 5                 |
| NFLX   | Netflix Inc.            | Communication Services   | High (~1.3)   | High           | 3                 |
| LLY    | Eli Lilly and Company   | Healthcare               | Medium (~1.1) | High           | 2                 |
| UNH    | UnitedHealth Group      | Healthcare               | Low (~0.7)    | Low            | 1                 |
| JNJ    | Johnson & Johnson       | Healthcare               | Low (~0.6)    | Low            | 1                 |
| COST   | Costco Wholesale        | Consumer Defensive       | Medium (~1.0) | Low            | 2                 |
| WMT    | Walmart Inc.            | Consumer Defensive       | Low (~0.5)    | Low            | 1                 |
| AMT    | American Tower Corp.    | Real Estate              | Medium (~1.0) | High           | 2                 |
| CAT    | Caterpillar Inc.        | Industrials              | High (~1.3)   | Low            | 5                 |
| DAL    | Delta Air Lines         | Industrials              | High (~1.4)   | Low            | 5                 |
| XOM    | Exxon Mobil Corporation | Energy                   | Medium (~1.1) | Low            | 4                 |
| JPM    | JPMorgan Chase & Co.    | Financials               | Medium (~1.1) | Medium         | 4                 |
| GS     | Goldman Sachs Group     | Financials               | Medium (~1.2) | Medium         | 4                 |
| BAC    | Bank of America Corp.   | Financials               | Medium (~1.3) | Medium         | 4                 |

*(Betas are approximate long-run values; duration and cyclicality are qualitative research design labels.)*

---

# Structural Properties of the Universe

## 1. Beta Dispersion

* Low-beta defensive anchors: JNJ, UNH, WMT
* High-beta convex plays: NVDA, TSLA, DAL, CAT, META
* Medium stabilizers: MSFT, COST, XOM, NFLX

This dispersion makes Sharpe ratio sensitive to:

* Concentration risk
* Volatility clustering
* Risk manager intervention

---

## 2. Rate Sensitivity (Duration)

High-duration names:

* NVDA, MSFT, GOOG, AMZN, META, TSLA, NFLX
* LLY
* AMT

Low-duration names:

* XOM
* CAT, DAL
* JNJ, WMT, UNH, COST

This creates direct exposure to:

* Real yield shifts
* Fed policy surprises
* Inflation expectations

Agents must reason about rates explicitly.

---

## 3. Cyclicality Spectrum

Highly cyclical (5):

* CAT
* DAL
* TSLA

Moderately cyclical (4):

* Banks (JPM, GS, BAC)
* Energy (XOM)
* Tech platforms (AMZN, META)
* NVDA

Defensive (1-2):

* JNJ, UNH, WMT
* COST
* LLY, AMT

This enables:

* Recession vs soft landing narratives
* Consumer slowdown debates
* Oil shock regime shifts

---

## 4. Sector Coverage (9 sectors)

| Sector | Tickers | Count |
|---|---|---|
| Technology | AAPL, NVDA, MSFT | 3 |
| Communication Services | GOOG, META, NFLX | 3 |
| Consumer Discretionary | AMZN, TSLA | 2 |
| Financials | JPM, GS, BAC | 3 |
| Healthcare | UNH, LLY, JNJ | 3 |
| Consumer Defensive | COST, WMT | 2 |
| Industrials | CAT, DAL | 2 |
| Energy | XOM | 1 |
| Real Estate | AMT | 1 |

---

# Why This Matters for Sharpe Optimization

Sharpe ratio penalizes volatility.

A purely high-beta portfolio may outperform in raw return but fail in Sharpe.

The universe therefore:

* Rewards diversification reasoning
* Punishes correlated risk clustering
* Forces covariance consideration
* Makes risk manager role essential

In other words:

> Portfolio quality becomes a function of reasoning structure, not narrative enthusiasm.

---

# Experimental Implications

This universe enables controlled tests of:

### 1. Debate vs Single-Agent

Does debate reduce concentrated exposure to:

* AI-only
* Energy-only
* Credit-only portfolios?

### 2. Reasoning Quality vs Performance

Do higher CRIT / RAudit scores correlate with:

* Lower realized volatility?
* Better risk-adjusted returns?
* Fewer regime-inconsistent allocations?

### 3. Sycophancy & Regime Overcommitment

High-beta names (NVDA, TSLA, DAL) create temptation.
Defensive names (JNJ, UNH, WMT) create caution.
The universe tests whether agents:

* Overcommit to consensus
* Or maintain diversification discipline
