
# 3.X Universe Design Principles

## Motivation

A multi-agent trading system can only be evaluated meaningfully if its opportunity set forces real macroeconomic and risk tradeoffs. If the universe is homogeneous (e.g., all mega-cap technology), portfolio performance becomes a momentum proxy rather than a test of structured reasoning.

Therefore, we design a **16-stock universe** with deliberate cross-factor dispersion to:

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
* Medium (0.9–1.2): Market-like
* High (> 1.2): Risk-on / volatile

### 2. Duration Proxy (Rate Sensitivity)

A qualitative proxy for interest-rate sensitivity:

* High Duration: Long-dated growth cash flows (AI, SaaS, biotech)
* Medium: Mixed exposure
* Low: Asset-heavy, commodity, or defensive cash-flow profiles

### 3. Cyclicality Score (1–5)

How sensitive the firm is to economic expansion/contraction:

* 1 = Highly defensive
* 3 = Moderately cyclical
* 5 = Highly cyclical

---

# 📊 Factor Exposure Table

| Ticker | Company                 | Sector          | Beta          | Duration Proxy | Cyclicality (1–5) |
| ------ | ----------------------- | --------------- | ------------- | -------------- | ----------------- |
| NVDA   | NVIDIA Corporation      | Semiconductors  | High (~1.6)   | High           | 4                 |
| MSFT   | Microsoft Corporation   | Software/Cloud  | Medium (~1.1) | High           | 3                 |
| AAPL   | Apple Inc.              | Consumer Tech   | Medium (~1.2) | Medium         | 3                 |
| LLY    | Eli Lilly and Company   | Pharma          | Medium (~1.1) | High           | 2                 |
| UNH    | UnitedHealth Group      | Managed Care    | Low (~0.7)    | Low            | 1                 |
| KO     | Coca-Cola Company       | Staples         | Low (~0.6)    | Low            | 1                 |
| COST   | Costco Wholesale        | Retail          | Medium (~1.0) | Low            | 2                 |
| AMT    | American Tower Corp.    | REIT            | Medium (~1.0) | High           | 2                 |
| CAT    | Caterpillar Inc.        | Industrials     | High (~1.3)   | Low            | 5                 |
| DAL    | Delta Air Lines         | Airlines        | High (~1.4)   | Low            | 5                 |
| XOM    | Exxon Mobil Corporation | Energy          | Medium (~1.1) | Low            | 4                 |
| CVX    | Chevron Corporation     | Energy          | Medium (~1.0) | Low            | 4                 |
| JPM    | JPMorgan Chase & Co.    | Banking         | Medium (~1.1) | Medium         | 4                 |
| BAC    | Bank of America Corp.   | Banking         | Medium (~1.3) | Medium         | 4                 |
| AXP    | American Express Co.    | Payments/Credit | Medium (~1.2) | Medium         | 4                 |
| UBER   | Uber Technologies, Inc. | Platform        | High (~1.5)   | Medium         | 4                 |

*(Betas are approximate long-run values; duration and cyclicality are qualitative research design labels.)*

---

# Structural Properties of the Universe

## 1. Beta Dispersion

* Low-beta defensive anchors: KO, UNH
* High-beta convex plays: NVDA, DAL, UBER
* Medium stabilizers: MSFT, COST, XOM

This dispersion makes Sharpe ratio sensitive to:

* Concentration risk
* Volatility clustering
* Risk manager intervention

---

## 2. Rate Sensitivity (Duration)

High-duration names:

* NVDA
* MSFT
* LLY
* AMT

Low-duration names:

* XOM, CVX
* CAT
* DAL
* KO

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

Moderately cyclical (4):

* Banks
* Energy
* UBER
* NVDA

Defensive (1–2):

* KO
* UNH
* COST

This enables:

* Recession vs soft landing narratives
* Consumer slowdown debates
* Oil shock regime shifts

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

High-beta names (NVDA, DAL, UBER) create temptation.
Defensive names (KO, UNH) create caution.
The universe tests whether agents:

* Overcommit to consensus
* Or maintain diversification discipline

---

# Summary

This 16-name basket is constructed to maximize:

* Factor diversity
* Regime sensitivity
* Cross-sector disagreement
* Risk-adjusted evaluation integrity
