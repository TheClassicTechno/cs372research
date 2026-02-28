#!/usr/bin/env python3
"""Compress case data into structured summaries for downstream trading agents.

Reads each case file in data/cases/{TICKER}/{YEAR}_Q{N}.json, separates
earnings and news items, then uses gpt-4o-mini to generate structured
summaries written to data/summaries/{TICKER}/{YEAR}_Q{N}/.

Usage:
    python scripts/compress_cases.py
    python scripts/compress_cases.py --dry-run       # show what would be processed
    python scripts/compress_cases.py --ticker AAPL    # process one ticker only
    python scripts/compress_cases.py --model gpt-4o   # use a different model
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

CASES_DIR = Path("data/cases")
SUMMARIES_DIR = Path("data/summaries")

# ─────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────

EARNINGS_SYSTEM_PROMPT = """\
You are a financial data extraction engine. You compress earnings call \
transcripts into structured JSON optimized for quantitative trading systems.

RULES:
- Do NOT hallucinate consensus values. If not explicitly stated → "not disclosed".
- Do NOT invent numbers. Only extract what is present in the text.
- Preserve all quantitative data verbatim.
- Preserve forward-looking statements and guidance.
- Preserve explicit causal reasoning.
- Remove greetings, operator chatter, repetition, safe-harbor language.
- Output ONLY valid JSON. No markdown, no explanation, no commentary."""

EARNINGS_USER_PROMPT = """\
Compress the following earnings transcript for {ticker} ({year} {quarter}) \
into the structured JSON schema below.

TARGET: 800–1200 tokens maximum output.

SCHEMA:
{{
  "metadata": {{
    "ticker": "",
    "year": "",
    "quarter": "",
    "original_char_length": <int>,
    "compressed_char_length": <int>
  }},
  "company_overview": {{
    "headline_result": "beat | miss | in-line | unclear",
    "management_tone": "improving | stable | deteriorating | mixed",
    "primary_theme": ""
  }},
  "financials": {{
    "revenue": {{
      "actual": "",
      "consensus": "",
      "delta_vs_consensus": "",
      "yoy_growth": ""
    }},
    "eps": {{
      "actual": "",
      "consensus": "",
      "delta_vs_consensus": "",
      "yoy_growth": ""
    }},
    "gross_margin": "",
    "operating_margin": "",
    "free_cash_flow": ""
  }},
  "guidance": {{
    "next_quarter": "",
    "full_year": "",
    "guidance_change_vs_prior": "raised | lowered | unchanged | unclear",
    "reason_for_change": ""
  }},
  "segment_highlights": [
    {{"segment": "", "performance": "", "driver": ""}}
  ],
  "explicit_risks": [
    {{"risk": "", "time_horizon": "short | medium | long | unspecified"}}
  ],
  "capital_allocation": {{
    "buybacks": "",
    "dividends": "",
    "debt_activity": "",
    "m_and_a": ""
  }},
  "causal_claims": [
    {{"claim": "", "type": "L1_association | L2_intervention | L3_counterfactual", "evidence_reference": ""}}
  ],
  "quantifiable_surprises": [
    {{"metric": "", "expected": "", "actual": "", "surprise": ""}}
  ],
  "high_signal_summary": [
    "Maximum 12 bullet points. No fluff."
  ]
}}

TRANSCRIPT ({original_chars:,} characters):

{transcript}"""

NEWS_SYSTEM_PROMPT = """\
You are a financial news aggregation engine. You compress collections of \
news items into structured JSON optimized for quantitative trading systems.

RULES:
- Categorize each distinct theme, not each individual headline.
- Determine sentiment from content, not from headline phrasing.
- Output ONLY valid JSON. No markdown, no explanation, no commentary."""

NEWS_USER_PROMPT = """\
Aggregate the following {count} news items for {ticker} ({year} {quarter}) \
into the structured JSON schema below.

SCHEMA:
{{
  "metadata": {{
    "ticker": "",
    "year": "",
    "quarter": "",
    "news_item_count": <int>
  }},
  "macro_news": [
    {{
      "headline_theme": "",
      "relevance_to_company": "",
      "sentiment": "positive | negative | neutral | mixed"
    }}
  ],
  "company_specific_news": [
    {{
      "headline_theme": "",
      "event_type": "earnings_preview | rating_change | merger | product | regulatory | macro_exposure | other",
      "sentiment": "positive | negative | neutral | mixed"
    }}
  ],
  "aggregate_news_signal": {{
    "overall_sentiment": "positive | negative | neutral | mixed",
    "dominant_theme": "",
    "conflicting_signals": true
  }}
}}

NEWS ITEMS:

{news_text}"""


# ─────────────────────────────────────────────────────────────────────
# LLM call with retry
# ─────────────────────────────────────────────────────────────────────

def call_llm(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    max_retries: int = 5,
) -> dict:
    """Call the LLM and parse JSON response, with exponential backoff on 429s."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content
            return json.loads(text)
        except RateLimitError as e:
            wait = 2 ** attempt * 5
            logger.warning("Rate limited (attempt %d/%d), waiting %ds: %s", attempt + 1, max_retries, wait, e)
            time.sleep(wait)
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON from LLM: %s", e)
            if attempt == max_retries - 1:
                raise
            time.sleep(2)

    raise RuntimeError(f"Failed after {max_retries} retries")


# ─────────────────────────────────────────────────────────────────────
# Case processing
# ─────────────────────────────────────────────────────────────────────

def process_case(
    case_path: Path,
    client: OpenAI,
    model: str,
    output_dir: Path,
) -> None:
    """Process a single case file: generate earnings + news summaries."""
    # Parse ticker / year / quarter from path
    ticker = case_path.parent.name
    stem = case_path.stem  # e.g. "2025_Q1"
    year, quarter = stem.split("_")

    logger.info("Processing %s/%s ...", ticker, stem)

    # Load case
    case = json.loads(case_path.read_text(encoding="utf-8"))
    items = case.get("case_data", {}).get("items", [])

    # Separate earnings and news (preserve original order)
    earnings_items = [i for i in items if i.get("kind") == "earnings"]
    news_items = [i for i in items if i.get("kind") != "earnings"]

    # Output directory
    out_dir = output_dir / ticker / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Earnings summary ──
    transcript = "\n\n".join(i["content"] for i in earnings_items)
    original_chars = len(transcript)

    if original_chars == 0:
        logger.warning("  No earnings content for %s/%s — writing empty summary.", ticker, stem)
        earnings_summary = {"metadata": {"ticker": ticker, "year": year, "quarter": quarter, "original_char_length": 0, "compressed_char_length": 0}, "note": "No earnings transcript found in case data."}
    else:
        # Truncate transcript to ~100k chars (~25k tokens) to stay within context window
        max_transcript_chars = 100_000
        if len(transcript) > max_transcript_chars:
            transcript = transcript[:max_transcript_chars] + "\n\n[...transcript truncated for processing...]"

        user_prompt = EARNINGS_USER_PROMPT.format(
            ticker=ticker,
            year=year,
            quarter=quarter,
            original_chars=original_chars,
            transcript=transcript,
        )
        logger.info("  Earnings: %d items, %s chars → calling %s...", len(earnings_items), f"{original_chars:,}", model)
        earnings_summary = call_llm(client, model, EARNINGS_SYSTEM_PROMPT, user_prompt)

        # Patch metadata with actual values
        if "metadata" in earnings_summary:
            earnings_summary["metadata"]["original_char_length"] = original_chars
            compressed = len(json.dumps(earnings_summary))
            earnings_summary["metadata"]["compressed_char_length"] = compressed

    (out_dir / "earnings_summary.json").write_text(
        json.dumps(earnings_summary, indent=2, ensure_ascii=False)
    )

    # ── News summary ──
    if not news_items:
        logger.warning("  No news items for %s/%s — writing empty summary.", ticker, stem)
        news_summary = {"metadata": {"ticker": ticker, "year": year, "quarter": quarter, "news_item_count": 0}, "note": "No news items found in case data."}
    else:
        news_text = "\n\n".join(
            f"[{i + 1}] ({item.get('kind', 'news')}) {item['content']}"
            for i, item in enumerate(news_items)
        )
        # Truncate if very long
        max_news_chars = 80_000
        if len(news_text) > max_news_chars:
            news_text = news_text[:max_news_chars] + "\n\n[...remaining items truncated...]"

        user_prompt = NEWS_USER_PROMPT.format(
            count=len(news_items),
            ticker=ticker,
            year=year,
            quarter=quarter,
            news_text=news_text,
        )
        logger.info("  News: %d items, %s chars → calling %s...", len(news_items), f"{len(news_text):,}", model)
        news_summary = call_llm(client, model, NEWS_SYSTEM_PROMPT, user_prompt)

        # Patch metadata
        if "metadata" in news_summary:
            news_summary["metadata"]["news_item_count"] = len(news_items)

    (out_dir / "news_summary.json").write_text(
        json.dumps(news_summary, indent=2, ensure_ascii=False)
    )

    logger.info("  Done: %s/%s", ticker, stem)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compress case data into structured summaries.")
    parser.add_argument("--dry-run", action="store_true", help="List files without processing.")
    parser.add_argument("--ticker", type=str, default=None, help="Process only this ticker.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model to use (default: gpt-4o-mini).")
    parser.add_argument("--cases-dir", type=str, default=str(CASES_DIR), help="Input cases directory.")
    parser.add_argument("--output-dir", type=str, default=str(SUMMARIES_DIR), help="Output summaries directory.")
    args = parser.parse_args()

    cases_dir = Path(args.cases_dir)
    output_dir = Path(args.output_dir)

    # Discover case files
    case_files = sorted(cases_dir.rglob("*.json"), key=lambda f: f.relative_to(cases_dir))

    if args.ticker:
        case_files = [f for f in case_files if f.parent.name == args.ticker]

    if not case_files:
        logger.error("No case files found in %s", cases_dir)
        sys.exit(1)

    logger.info("Found %d case file(s) to process.", len(case_files))

    if args.dry_run:
        for f in case_files:
            case = json.loads(f.read_text())
            items = case.get("case_data", {}).get("items", [])
            earnings = [i for i in items if i.get("kind") == "earnings"]
            news = [i for i in items if i.get("kind") != "earnings"]
            e_chars = sum(len(i["content"]) for i in earnings)
            n_chars = sum(len(i["content"]) for i in news)
            print(f"  {f.relative_to(cases_dir)}: {len(earnings)} earnings ({e_chars:,} chars), {len(news)} news ({n_chars:,} chars)")
        print(f"\nTotal: {len(case_files)} files. Use without --dry-run to process.")
        return

    # Initialize OpenAI client
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set. Export it or add to .env file.")
        sys.exit(1)

    client = OpenAI()

    # Process each case
    processed = 0
    failed = 0
    for case_path in case_files:
        try:
            process_case(case_path, client, args.model, output_dir)
            processed += 1
            # Small delay between cases to be kind to rate limits
            time.sleep(1)
        except Exception as exc:
            logger.error("FAILED %s: %s", case_path.relative_to(cases_dir), exc)
            failed += 1

    logger.info("Done. %d processed, %d failed.", processed, failed)


if __name__ == "__main__":
    main()
