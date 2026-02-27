
# Sentiment Feature Derivation Methodology

This document describes **exactly how the sentiment features were constructed**, from raw news articles to final quarterly features used by trading agents.

The goal is:

* Reproducibility
* Statistical clarity
* Economic interpretability
* Auditability (for research use)

---

# 1. Raw Data Collection

## 1.1 Universe

* 8 tickers
* 4 quarters (2025Q1–2025Q4)

## 1.2 Article Selection

For each ticker and quarter:

1. Query news articles mentioning ticker symbol or company name.
2. Restrict publication dates to quarter boundaries.
3. Deduplicate identical headlines / URLs.
4. Remove non-financial articles (optional filtering).

Let:

* ( A_{i,t} ) = set of articles for ticker ( i ) in quarter ( t )
* ( N_{i,t} = |A_{i,t}| )

---

# 2. Article-Level Sentiment Scoring

Each article is assigned a scalar sentiment score:

[
s_{i,t,k}
]

Where:

* ( i ) = ticker
* ( t ) = quarter
* ( k ) = article index
* ( s \in [-1, 1] )

Sentiment model properties:

* FinBERT-style financial classifier (or equivalent)
* Positive → optimistic tone
* Negative → pessimistic tone
* Zero → neutral

---

# 3. Quarterly Aggregation

All features are derived from the article-level sentiment scores.

---

## 3.1 Article Count

[
\text{article_count}*{i,t} = N*{i,t}
]

This is a direct count.

Interpretation:

* Proxy for attention
* Higher count = more coverage

---

## 3.2 Mean Sentiment

genui{"math_block_widget_always_prefetched":{"content":"mean_sentiment_{i,t} = (1/N_{i,t}) * sum_{k=1}^{N_{i,t}} s_{i,t,k}"}}

This is the arithmetic average of article-level sentiment.

Properties:

* Bounded by [-1, 1]
* Captures aggregate tone

---

## 3.3 Sentiment Volatility

genui{"math_block_widget_always_prefetched":{"content":"sentiment_volatility_{i,t} = sqrt((1/N_{i,t}) * sum (s_{i,t,k} - mean_{i,t})^2)"}}

This is the cross-article dispersion within the quarter.

Interpretation:

* High → disagreement
* Low → consensus

---

## 3.4 Surprise Sentiment (Quarter-over-Quarter Change)

genui{"math_block_widget_always_prefetched":{"content":"surprise_{i,t} = mean_{i,t} - mean_{i,t-1}"}}

Notes:

* Undefined for first available quarter
* Represents narrative shift

Interpretation:

* Positive → improving sentiment
* Negative → deteriorating sentiment

---

## 3.5 Cross-Sectional Z-Score

Computed **within each quarter** across all tickers.

Step 1: Compute cross-sectional mean

[
\mu_t = \frac{1}{M} \sum_{i=1}^{M} mean_{i,t}
]

Step 2: Compute cross-sectional standard deviation

[
\sigma_t = \sqrt{
\frac{1}{M}
\sum_{i=1}^{M}
(mean_{i,t} - \mu_t)^2
}
]

Step 3: Compute standardized score

genui{"math_block_widget_always_prefetched":{"content":"z_{i,t} = (mean_{i,t} - mu_t) / sigma_t"}}

Properties:

* Mean zero within quarter
* Unit variance within quarter
* Enables relative ranking

---

# 4. Data Cleaning & Edge Handling

## 4.1 Missing Surprise (Q1)

For first quarter:

* No prior data exists
* Surprise is set to `null`
* NOT set to zero (zero would imply no change)

---

## 4.2 Small Sample Handling

If article_count is very small:

* Volatility may be unstable
* Optional minimum count threshold can be enforced

---
---

# 5. Feature Dependencies

| Feature              | Depends On                   |
| -------------------- | ---------------------------- |
| article_count        | raw article set              |
| mean_sentiment       | article-level scores         |
| sentiment_volatility | article-level scores         |
| surprise_sentiment   | mean_sentiment (lagged)      |
| cross_sectional_z    | mean_sentiment (all tickers) |

This ensures:

* No circular dependencies
* Transparent derivation chain

---

# 6. Statistical Properties

Given:

* 8 tickers
* 4 quarters
* 32 ticker-quarter observations

Cross-sectional z-scores:

* Mean = 0 (per quarter)
* Std ≈ 1 (per quarter)

Time-series features:

* Small sample
* Primarily exploratory
* More reliable with additional years

---

# 7. Economic Interpretation

The derived features capture:

1. Attention (article_count)
2. Tone (mean_sentiment)
3. Disagreement (sentiment_volatility)
4. Narrative shift (surprise_sentiment)
5. Relative strength (cross_sectional_z)

These are conceptually orthogonal dimensions of news flow.

---

# 8. Reproducibility Checklist

To fully reproduce:

* Raw news dataset
* Sentiment model version
* Quarter boundaries
* Deduplication logic
* Cross-sectional normalization procedure

All aggregation steps are deterministic given the article set.

---

# 9. Assumptions

1. Article sentiment reflects investor perception.
2. Quarterly aggregation is appropriate horizon.
3. Equal weighting of articles is acceptable.
4. Cross-sectional normalization removes time bias.

---

# 10. Limitations

* Small time sample (4 quarters)
* Model-based sentiment noise
* No price-based features included
* No sector-neutralization applied
* No decay weighting

---

# 11. Summary

Pipeline:

Raw Articles
→ Sentiment Scoring
→ Quarterly Aggregation
→ Cross-Sectional Standardization
→ Agent Features

All features are transparent transformations of article-level sentiment.

---

If you'd like, I can next:

* Add a “Statistical Validation Tests” section
* Add a “Leakage & Lookahead Bias” audit section
* Or produce a diagram (for your CS372 paper appendix) showing the full data flow visually


# 9. How Agents Should Reason About These Features

This section specifies how trading agents are expected to interpret and reason about the sentiment features during debate and portfolio construction.

The goal is not to enforce a single strategy, but to provide structured guidance so that agent reasoning remains economically coherent, statistically grounded, and internally consistent.

---

## 9.1 Core Principle

Agents should treat these features as **probabilistic signals**, not deterministic truths.

Each feature provides information about expected future returns, but none is sufficient on its own. Agents must:

* Combine signals
* Account for risk
* Compare across tickers
* Justify trade-offs

---

## 9.2 Interpreting Each Feature

### A. `cross_sectional_z` (Primary Ranking Signal)

This is the most important signal.

It represents how strong a ticker’s sentiment is relative to peers:

[
z_{i,t} =
\frac{
\text{mean_sentiment}_{i,t}
---------------------------

\mu_t
}{
\sigma_t
}
]

Interpretation rules:

* High positive z → sentiment stronger than peers
* High negative z → sentiment weaker than peers
* Near zero → neutral relative position

Agent reasoning guideline:

> “All else equal, I prefer tickers with higher cross_sectional_z.”

This supports long–short and ranking-based portfolio construction.

---

### B. `surprise_sentiment` (Momentum / Narrative Shift)

Defined as:

[
\text{surprise_sentiment}_{i,t}
===============================

## \text{mean_sentiment}_{i,t}

\text{mean_sentiment}_{i,t-1}
]

Interpretation:

* Positive → improving narrative
* Negative → deteriorating narrative

Agent reasoning guideline:

* Improving sentiment strengthens conviction.
* Deteriorating sentiment weakens conviction.
* Strong negative surprise may override positive level.

Example:

* High cross_sectional_z + positive surprise → strong long candidate
* High cross_sectional_z + large negative surprise → unstable long

---

### C. `mean_sentiment` (Absolute Level)

Absolute tone measure.

Interpretation:

* Strongly positive → broad optimism
* Strongly negative → broad pessimism

Agent reasoning guideline:

* Level matters less than relative strength.
* Level + surprise together form a regime signal.

---

### D. `sentiment_volatility` (Disagreement / Uncertainty)

Standard deviation of article sentiment:

[
\sigma_{i,t}
============

\sqrt{
\frac{1}{N}
\sum (s - \mu)^2
}
]

Interpretation:

* High → disagreement or mixed narratives
* Low → consensus tone

Agent reasoning guideline:

* High volatility increases risk.
* Strong signal + low volatility = high conviction.
* Strong signal + high volatility = unstable thesis.

Agents may reduce position size when volatility is high.

---

### E. `article_count` (Attention / Information Density)

Interpretation:

* High → market attention
* Low → neglected name

Agent reasoning guideline:

* Signals supported by high attention are more credible.
* Low attention may imply underreaction opportunities.
* Extremely high attention may imply crowded trade.

---

## 9.3 Signal Combination Logic

Agents should reason in combinations, not isolation.

Strong Long Setup:

* High cross_sectional_z
* Positive surprise_sentiment
* Moderate or low sentiment_volatility
* Sufficient article_count

Weak / Avoid Setup:

* Negative cross_sectional_z
* Negative surprise_sentiment
* High volatility
* Unstable narrative

Conflict Case Example:

* High cross_sectional_z
* Strong negative surprise_sentiment

Agent should explicitly debate:

* Is sentiment peaking?
* Is this a reversal signal?
* Is this short-term noise?

---

## 9.4 Portfolio Construction Reasoning

Agents should justify:

1. Ranking logic
2. Position sizing logic
3. Risk control logic

Example structured reasoning:

> “I rank tickers by cross_sectional_z.
> I overweight names with positive surprise_sentiment.
> I scale down exposure when sentiment_volatility is high.”

Agents should avoid:

* Using only one feature
* Ignoring cross-sectional comparisons
* Overweighting unstable signals

---

## 9.5 Risk Awareness

Sentiment is not price.

Agents must recognize:

* Sentiment may already be priced in.
* Reversals can occur after extreme readings.
* High volatility implies unstable narratives.

Agents should explicitly acknowledge uncertainty.

---

## 9.6 Debate-Specific Reasoning Expectations

During multi-agent debate:

Agents should:

* Challenge signal interpretation
* Question overreliance on a single metric
* Compare relative vs absolute strength
* Discuss stability vs momentum

Strong reasoning includes:

* Cross-sectional comparison
* Temporal comparison (surprise)
* Risk justification
* Allocation logic

Weak reasoning includes:

* “Sentiment is positive so buy.”
* Ignoring peer comparison.
* Ignoring volatility.

---

## 9.7 Minimal Coherent Trading Logic

At minimum, agents should be able to articulate:

* Why this ticker?
* Why this size?
* Why not alternatives?
* What risk exists?
* What signal supports this trade?

---

## 9.8 Summary for Agents

These features represent:

* Level (mean_sentiment)
* Momentum (surprise_sentiment)
* Relative strength (cross_sectional_z)
* Uncertainty (sentiment_volatility)
* Attention (article_count)

Agents should synthesize them into a coherent, risk-aware allocation decision.
