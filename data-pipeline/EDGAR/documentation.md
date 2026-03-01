
# EDGAR Filing Pipeline — Data Flow & Methodology

This document describes the full SEC EDGAR filing pipeline: from raw HTML on SEC servers to structured JSON summaries consumed by the quarterly snapshot builder and trading agents.

The goal is:

* Reproducibility
* Auditability
* Point-in-time safety
* Clear separation of narrative text from numeric fundamentals

---

# 1. Pipeline Overview

```
SEC EDGAR API
  -> get_sec_data.py  (download + text extraction)
  -> raw .txt files    (TICKER/YEAR/QUARTER/FORM_DATE.txt)
  -> filing_summarization_pipeline.py  (Claude API structured extraction)
  -> finished_summaries/TICKER/YEAR/QUARTER/FORM_summary.json
```

Two scripts, two stages, one strict rule: **no numeric extraction from filing text**.

---

# 2. Stage 1: Filing Text Extraction (`get_sec_data.py`)

## 2.1 Purpose

Download SEC filings and extract clean narrative text. Produces structured `.txt` files with section markers.

## 2.2 Data Sources

* SEC EDGAR Submissions API (`data.sec.gov/submissions/CIK{cik}.json`)
* SEC EDGAR Archives (`sec.gov/Archives/edgar/data/...`)
* Company ticker-to-CIK mapping (`sec.gov/files/company_tickers.json`)

## 2.3 Supported Filing Types

| Mode | Form Types | Selection Logic |
|------|-----------|-----------------|
| Manual (`--quarters Q1,Q2,Q3`) | 10-Q | Quarter specified directly |
| Manual (`--quarters ANNUAL`) | 10-K | Matches any quarter in year |
| Rolling (`--last-n N`) | 10-Q (Q1-Q3), 10-K (Q4) | Auto-maps Q4 to 10-K |
| Custom (`--forms 10-Q,8-K,4`) | Any form type | Explicit list |
| Bundle (`--bundle core`) | Institutional set | 10-K, 10-Q, 8-K, 4, SC 13D/G, etc. |

## 2.4 Target Resolution

CLI arguments are converted into **3-tuple targets**: `(year, quarter_or_None, form_type)`.

* `--years 2024 --quarters Q1,Q2` produces `[(2024, "Q1", "10-Q"), (2024, "Q2", "10-Q")]`
* `--years 2024 --quarters ANNUAL` produces `[(2024, None, "10-K")]` where None matches any quarter
* `--last-n 4` walks backward from the current quarter, auto-mapping Q4 to 10-K

`filter_filings()` receives these targets and is agnostic to how they were built.

## 2.5 Text Extraction Pipeline

For each matched filing:

1. **Download HTML** from SEC Archives via `download_html()`
2. **Clean HTML to text** via `clean_html_to_text()` using stdlib `HTMLParser`
   - Strips `<script>`, `<style>`, `<noscript>` tags
   - Converts block elements (`<p>`, `<div>`, `<br>`, headings) to line breaks
   - Normalizes whitespace, collapses triple+ newlines
   - Unescapes HTML entities
3. **Extract sections** via `extract_sections()`:
   - 10-K/10-Q: MD&A (Item 7 or Item 2) and Risk Factors (Item 1A)
   - 8-K: entire body (no item-level parsing)
4. **Format output** via `format_text_output()` with structured header:
   ```
   FORM: 10-Q
   FILING_DATE: 2024-08-01
   ACCESSION: 0001234567-24-001234

   ==== SECTION: MDA ====
   [Management's Discussion and Analysis text...]

   ==== SECTION: RISK_FACTORS ====
   [Risk Factors text...]
   ```
5. **Atomic write** to `TICKER/YEAR/QUARTER/FORM_DATE.txt` (tmp file -> rename)

## 2.6 What Is NOT Extracted

* Financial tables
* XBRL data
* Numeric values (revenue, EPS, etc.)
* Exhibits and attachments

All numeric fundamentals must come from structured APIs (e.g., SEC XBRL company-facts endpoint).

## 2.7 Caching and Idempotency

| Flag | Behavior |
|------|----------|
| Default | Skip filings already in `_index.json` with `text_saved=True` |
| `--force-refresh` | Re-extract all matching filings, update index |
| `--no-cache` | Ignore `_index.json` entirely, stateless run |

* `_submissions.json` caches SEC submissions data for 24 hours
* `_index.json` tracks per-accession extraction status
* Both use atomic writes (tmp -> rename)

## 2.8 Rate Limiting

* Global threading lock ensures max 5 requests/second to SEC
* `RATE_LIMIT_SECONDS = 0.2` between requests
* `MAX_WORKERS = 5` cap on parallel threads

## 2.9 Output Directory Structure

```
sec_filings/
  AAPL/
    _submissions.json        (cached SEC submissions)
    _index.json              (extraction tracking)
    2024/
      Q1/
        10-Q_2024-05-02.txt
      Q4/
        10-K_2024-10-31.txt
    2025/
      Q1/
        8-K_2025-01-15.txt
```

## 2.10 Amendment Handling

* Amendments (e.g., `10-K/A`) are excluded by default
* `--include-amendments` includes them
* For matching: the `/A` suffix is stripped, so `10-K/A` matches against `10-K` targets
* In the output, the original form name (with `/A`) is preserved
* Filesystem paths sanitize `/` to `-` (e.g., `10-K-A_2024-05-02.txt`)

---

# 3. Stage 2: Filing Summarization (`filing_summarization_pipeline.py`)

## 3.1 Purpose

Send narrative text to Claude for structured extraction. Produces validated JSON summaries.

## 3.2 Input Format

Reads `.txt` files produced by Stage 1. Parses the structured header (`FORM:`, `FILING_DATE:`, `ACCESSION:`) and section markers (`==== SECTION: NAME ====`).

## 3.3 Summary Schema

Every summary JSON conforms to this schema:

```json
{
  "form": "10-Q",
  "filing_date": "2024-08-01",
  "accession": "0001234567-24-001234",
  "mda": {
    "demand_signal": "<qualitative demand outlook>",
    "margin_trend": "<margin direction and drivers>",
    "inventory_signal": "<inventory buildup/drawdown>",
    "liquidity_commentary": "<cash position and liquidity>",
    "capex_trend": "<capital expenditure direction>",
    "tone_shift_score": 0.3
  },
  "risk_factors": {
    "new_risks_added": true,
    "refinancing_risk": "<refinancing or debt maturity risk>",
    "customer_concentration_change": "<concentration changes>"
  },
  "events": [
    {
      "type": "guidance",
      "direction": "positive",
      "severity": 0.7
    }
  ]
}
```

### Field Details

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `tone_shift_score` | float or null | -1.0 to +1.0 | Sentiment shift vs prior filing. 0.0 = neutral. null = section missing |
| `severity` | float or null | 0.0 to 1.0 | Event materiality. 0.0 = negligible, 1.0 = critical |
| `new_risks_added` | bool or null | true/false/null | Whether new risk factors appeared |
| `events[].type` | string | see below | Event category |
| `events[].direction` | string | positive/negative/neutral | Event impact direction |

Event type categories: `guidance`, `restructuring`, `acquisition`, `divestiture`, `legal`, `regulatory`, `leadership`, `earnings`, `offering`, `other`.

### 8-K Handling

For 8-K filings, `mda` fields are set to `"not applicable"` and `tone_shift_score` to `null`. The focus is on identifying material events in the `events` list.

## 3.4 Chunking Strategy

Large filings (>60,000 chars) are split on paragraph boundaries:

1. Split text on double newlines
2. Accumulate paragraphs until chunk exceeds `MAX_CHUNK_CHARS`
3. Each chunk is summarized independently
4. Chunk summaries are merged:
   - `mda`: first chunk with substantive content
   - `risk_factors`: first chunk with substantive content
   - `events`: concatenated and deduplicated by `(type, direction)`

## 3.5 Claude API Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Temperature | 0.0 | Deterministic output |
| Max tokens | 4096 | Sufficient for summary schema |
| Rate limit | 1.0s between calls | Conservative API politeness |

System prompt constraints:
* Return ONLY valid JSON
* NEVER hallucinate or invent financial numbers
* Use null or "not discussed" for missing information
* All string fields: 1-2 sentences max

## 3.6 Validation

`validate_summary_schema()` checks:
* All required top-level keys present
* Nested `mda` and `risk_factors` dicts have correct types
* Each event in `events` has `type`, `direction`, `severity`
* Type checking for strings, bools, floats, nulls

Validation warnings are logged but do not prevent saving.

## 3.7 Idempotency

* Existing summaries are skipped unless `--force` is passed
* Skip conditions: missing header, no extractable sections, output already exists

---

# 4. Example Commands

```bash
# Stage 1: Download and extract text for 2024 annual filings
python get_sec_data.py \
    --tickers AAPL,NVDA,MSFT \
    --years 2024 \
    --quarters ANNUAL \
    --output ./raw_filings

# Stage 1: Rolling window - last 4 quarters
python get_sec_data.py \
    --tickers AAPL,NVDA \
    --last-n 4 \
    --output ./raw_filings

# Stage 1: Full institutional bundle with parallel download
python get_sec_data.py \
    --tickers AAPL \
    --years 2024,2025 \
    --quarters Q1,Q2,Q3,Q4 \
    --bundle core \
    --include-amendments \
    --parallel --workers 3 \
    --output ./raw_filings

# Stage 2: Summarize all extracted filings
python filing_summarization_pipeline.py \
    --raw-dir ./raw_filings \
    --out-dir ./finished_summaries \
    --tickers AAPL,NVDA,MSFT

# Stage 2: Force re-summarize specific tickers
python filing_summarization_pipeline.py \
    --raw-dir ./raw_filings \
    --out-dir ./finished_summaries \
    --tickers AAPL \
    --force
```

---

# 5. Point-in-Time Safety

The EDGAR pipeline enforces point-in-time correctness:

1. **Filing date is the anchor.** Each filing's `filing_date` is the date it became publicly available on EDGAR.
2. **No forward-looking data.** The text extraction preserves the original filing date; the summarization pipeline passes it through unchanged.
3. **Downstream consumers** (quarterly snapshot builder) filter by `filing_date <= rebalance_date` before including any summary.
4. **No numeric extraction.** Financial numbers in filing text may reference future guidance or projections. By extracting only qualitative signals, we avoid accidentally leaking forward-looking numeric data.

---

# 6. Integration Points

## 6.1 Downstream: Quarterly Snapshot Builder

`quarterly_snapshot_builder.py` reads from `finished_summaries/` via `load_filing_summaries()`:
* Most recent 10-Q or 10-K with `filing_date <= rebalance_date`
* All 8-K filings with `filing_date` within 90 days of `rebalance_date`

## 6.2 Future: Structured Fundamentals

Numeric fundamentals (revenue, EPS, debt, ratios) should come from:
* SEC XBRL `company-facts` endpoint
* Financial data vendor APIs

These are separate from the narrative text pipeline and are stubbed in the snapshot builder.

---

# 7. Testing

```bash
# Run the full test suite (111 tests)
pytest tests/test_sec_downloader.py -v

# Test coverage includes:
# - Quarter mapping and rolling window computation
# - Amendment detection and filtering
# - HTML-to-text extraction (script/style stripping, block elements, whitespace)
# - Section extraction (MDA, Risk Factors, 8-K body)
# - Text output formatting
# - Filing download pipeline (mocked HTTP)
# - Index loading/saving
# - Schedule downloads with cache modes
# - Target building from CLI args
```
