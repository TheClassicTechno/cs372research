"""Tests for filing_summarization_pipeline.py — compressed state vector format.

All tests use mocked data. No external API calls.
"""

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "data-pipeline" / "EDGAR")
)
import filing_summarization_pipeline as fsp


# ==========================
# Fixtures — mock data
# ==========================

VALID_SUMMARY = {
    "ticker": "AAPL",
    "form": "10-Q",
    "filing_date": "2026-01-30",
    "fiscal_period": "2026-Q1",
    "period_type": "quarterly",
    "operating_state": (
        "Revenue growth was driven primarily by iPhone Pro models and "
        "Services, with most geographic segments contributing positively "
        "while Greater China declined. iPhone and iPad showed strong "
        "improvement while Mac and Wearables decreased on lower laptop "
        "and accessory sales. Services acceleration was led by "
        "advertising, App Store, and cloud services."
    ),
    "cost_structure": (
        "Products gross margin improved on favorable product mix, "
        "partially offset by new tariff costs on imported components. "
        "Services gross margin expanded on higher revenue and favorable "
        "service mix. R&D spending increased driven by infrastructure "
        "and headcount investments, while SG&A growth reflected higher "
        "variable selling costs."
    ),
    "material_events": (
        "Apple announced updated MacBook Pro, iPad Pro, and Apple "
        "Vision Pro during the quarter. The 2026 Annual Meeting of "
        "Shareholders was held with all directors re-elected, Ernst & "
        "Young ratified as auditor, and executive compensation approved. "
        "A shareholder China Entanglement Audit proposal was rejected."
    ),
    "macro_exposures": (
        "No material changes to the Company's risk factors since the "
        "prior 10-K filing. Tariff uncertainty remains the primary new "
        "macro exposure, with ultimate impact depending on additional "
        "measures and retaliatory responses from trading partners."
    ),
    "forward_outlook": (
        "Management intends to increase dividends annually subject to "
        "Board declaration. Cash balances and operations expected to "
        "satisfy requirements and capital return program over the next "
        "twelve months. Gross margins expected to face continued "
        "volatility and downward pressure."
    ),
    "uncertainty_profile": (
        "Ultimate impact of recently announced tariffs remains uncertain "
        "and depends on several factors including additional government "
        "measures. The Company is evaluating timing and method of "
        "adopting ASU 2025-06. Effective tax rate was impacted by "
        "foreign currency loss regulations."
    ),
    "word_count": 247,
}

ANNUAL_SUMMARY = {
    "ticker": "AAPL",
    "form": "10-K",
    "filing_date": "2025-10-31",
    "fiscal_period": "FY2025",
    "period_type": "annual",
    "operating_state": (
        "Total net sales increased year-over-year driven by iPhone Pro "
        "models, Services growth, and Mac recovery. Greater China was "
        "the only region with declining sales due to lower iPhone "
        "volumes. Wearables and Accessories continued to contract."
    ),
    "cost_structure": (
        "Products gross margin expanded on favorable costs and product "
        "mix but margin percentage declined due to tariff impacts. "
        "Services margin improved on revenue scale and favorable mix. "
        "Operating expenses grew driven by headcount and infrastructure."
    ),
    "material_events": (
        "New tariffs announced on imports from multiple countries "
        "including China and the EU. Company announced a new share "
        "repurchase program and raised the quarterly dividend. Google "
        "antitrust ruling and EU Digital Markets Act compliance "
        "required App Store and Safari changes."
    ),
    "macro_exposures": (
        "Tariff escalation represents the primary new macro risk with "
        "reciprocal tariffs threatened by multiple trading partners. "
        "Semiconductor import investigations add supply chain "
        "uncertainty. Manufacturing concentration in China mainland, "
        "India, and Vietnam creates geopolitical exposure."
    ),
    "forward_outlook": (
        "Management expects to continue annual dividend increases and "
        "share repurchases subject to Board authorization. Further "
        "business changes anticipated in response to regulatory "
        "developments. Gross margins expected to remain under "
        "volatility pressure from mix shifts and tariff costs."
    ),
    "uncertainty_profile": (
        "Tariff ultimate impact remains uncertain depending on "
        "additional measures and retaliatory responses. Outcomes of "
        "pending litigation including the Google antitrust remedies "
        "are inherently uncertain. Tax reserve adequacy cannot be "
        "guaranteed given evolving international frameworks."
    ),
    "word_count": 215,
}


# ==========================
# Test: count_words helper
# ==========================

class TestCountWords:
    def test_basic(self):
        assert fsp.count_words("hello world") == 2

    def test_empty(self):
        assert fsp.count_words("") == 0

    def test_multiword(self):
        assert fsp.count_words("one two three four five") == 5

    def test_extra_whitespace(self):
        assert fsp.count_words("  hello   world  ") == 2


# ==========================
# Test: compute_total_words
# ==========================

class TestComputeTotalWords:
    def test_sums_paragraphs(self):
        total = fsp.compute_total_words(VALID_SUMMARY)
        assert total == VALID_SUMMARY["word_count"]

    def test_ignores_non_paragraph_keys(self):
        d = {"ticker": "AAPL", "operating_state": "one two three"}
        assert fsp.compute_total_words(d) == 3

    def test_missing_keys_treated_as_zero(self):
        assert fsp.compute_total_words({}) == 0


# ==========================
# Test: normalize_whitespace
# ==========================

class TestNormalizeWhitespace:
    def test_collapse_spaces(self):
        assert fsp.normalize_whitespace("hello   world") == "hello world"

    def test_collapse_tabs(self):
        assert fsp.normalize_whitespace("hello\t\tworld") == "hello world"

    def test_collapse_newlines(self):
        assert fsp.normalize_whitespace("hello\n\n\nworld") == "hello\nworld"

    def test_strip(self):
        assert fsp.normalize_whitespace("  hello  ") == "hello"

    def test_mixed(self):
        result = fsp.normalize_whitespace("  a   b\n\n\nc  ")
        assert result == "a b\nc"


# ==========================
# Test: validate_summary
# ==========================

class TestValidateSummary:
    def test_valid_quarterly(self):
        errors = fsp.validate_summary(VALID_SUMMARY)
        assert errors == []

    def test_valid_annual(self):
        errors = fsp.validate_summary(ANNUAL_SUMMARY)
        assert errors == []

    def test_missing_required_keys(self):
        partial = {"ticker": "AAPL"}
        errors = fsp.validate_summary(partial)
        missing = [e for e in errors if e.startswith("missing key")]
        assert len(missing) >= len(fsp.REQUIRED_KEYS) - 1

    def test_wrong_type_ticker(self):
        bad = {**VALID_SUMMARY, "ticker": 123}
        errors = fsp.validate_summary(bad)
        assert any("ticker must be str" in e for e in errors)

    def test_wrong_type_word_count(self):
        bad = {**VALID_SUMMARY, "word_count": "two hundred"}
        errors = fsp.validate_summary(bad)
        assert any("word_count must be int" in e for e in errors)

    def test_paragraph_must_be_str(self):
        bad = {**VALID_SUMMARY, "operating_state": ["a", "b"]}
        errors = fsp.validate_summary(bad)
        assert any("operating_state must be str" in e for e in errors)

    def test_bullet_characters_detected(self):
        bad = {**VALID_SUMMARY, "operating_state": "- Revenue increased.\n- Margins improved."}
        errors = fsp.validate_summary(bad)
        assert any("bullet characters" in e for e in errors)

    def test_star_bullet_detected(self):
        bad = {**VALID_SUMMARY, "cost_structure": "* Costs rose.\n* Margins fell."}
        errors = fsp.validate_summary(bad)
        assert any("bullet characters" in e for e in errors)

    def test_word_count_over_limit(self):
        bad = {**VALID_SUMMARY, "word_count": 500}
        errors = fsp.validate_summary(bad)
        assert any("exceeds" in e for e in errors)

    def test_word_count_at_limit_ok(self):
        ok = {**VALID_SUMMARY, "word_count": 375}
        errors = fsp.validate_summary(ok)
        assert not any("exceeds" in e for e in errors)

    def test_extra_keys_flagged(self):
        bad = {**VALID_SUMMARY, "sentiment_score": 0.5}
        errors = fsp.validate_summary(bad)
        assert any("unexpected key: sentiment_score" in e for e in errors)

    def test_no_extra_keys_in_valid(self):
        errors = fsp.validate_summary(VALID_SUMMARY)
        assert not any("unexpected" in e for e in errors)


# ==========================
# Test: Structure — all required keys, no bullets
# ==========================

class TestSummaryStructure:
    @pytest.mark.parametrize("summary", [VALID_SUMMARY, ANNUAL_SUMMARY])
    def test_all_required_keys_present(self, summary):
        for key in fsp.REQUIRED_KEYS:
            assert key in summary, f"missing key: {key}"

    @pytest.mark.parametrize("summary", [VALID_SUMMARY, ANNUAL_SUMMARY])
    def test_paragraph_fields_are_strings(self, summary):
        for key in fsp.PARAGRAPH_KEYS:
            assert isinstance(summary[key], str), f"{key} not str"

    @pytest.mark.parametrize("summary", [VALID_SUMMARY, ANNUAL_SUMMARY])
    def test_no_bullet_characters_in_paragraphs(self, summary):
        for key in fsp.PARAGRAPH_KEYS:
            val = summary[key]
            assert not re.search(r"^\s*[-*•]\s", val, re.MULTILINE), (
                f"{key} contains bullet characters"
            )

    @pytest.mark.parametrize("summary", [VALID_SUMMARY, ANNUAL_SUMMARY])
    def test_no_list_values(self, summary):
        for key in fsp.PARAGRAPH_KEYS:
            assert not isinstance(summary[key], list), f"{key} is a list"

    @pytest.mark.parametrize("summary", [VALID_SUMMARY, ANNUAL_SUMMARY])
    def test_metadata_types(self, summary):
        assert isinstance(summary["ticker"], str)
        assert isinstance(summary["form"], str)
        assert isinstance(summary["filing_date"], str)
        assert isinstance(summary["fiscal_period"], str)
        assert isinstance(summary["period_type"], str)
        assert isinstance(summary["word_count"], int)


# ==========================
# Test: Word count — bounds
# ==========================

class TestWordCount:
    @pytest.mark.parametrize("summary", [VALID_SUMMARY, ANNUAL_SUMMARY])
    def test_under_max(self, summary):
        total = fsp.compute_total_words(summary)
        assert total <= fsp.MAX_SUMMARY_WORDS, (
            f"word count {total} exceeds {fsp.MAX_SUMMARY_WORDS}"
        )

    @pytest.mark.parametrize("summary", [VALID_SUMMARY, ANNUAL_SUMMARY])
    def test_above_minimum(self, summary):
        total = fsp.compute_total_words(summary)
        assert total >= fsp.MIN_SUMMARY_WORDS, (
            f"word count {total} below {fsp.MIN_SUMMARY_WORDS}"
        )

    @pytest.mark.parametrize("summary", [VALID_SUMMARY, ANNUAL_SUMMARY])
    def test_word_count_field_matches_computed(self, summary):
        computed = fsp.compute_total_words(summary)
        assert summary["word_count"] == computed


# ==========================
# Test: No repeated sentences across sections
# ==========================

def _extract_sentences(text: str) -> list:
    """Split text into sentences for dedup checking."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


class TestNoDuplicateSentences:
    @pytest.mark.parametrize("summary", [VALID_SUMMARY, ANNUAL_SUMMARY])
    def test_no_repeated_sentences_across_sections(self, summary):
        all_sentences = []
        for key in fsp.PARAGRAPH_KEYS:
            all_sentences.extend(_extract_sentences(summary[key]))

        seen = set()
        duplicates = []
        for s in all_sentences:
            normalized = s.lower().strip().rstrip(".")
            if normalized in seen:
                duplicates.append(s)
            seen.add(normalized)

        assert duplicates == [], f"Duplicate sentences found: {duplicates}"


# ==========================
# Test: Deterministic formatting — stable key order
# ==========================

class TestDeterministicFormatting:
    @pytest.mark.parametrize("summary", [VALID_SUMMARY, ANNUAL_SUMMARY])
    def test_key_order_matches_spec(self, summary):
        keys = list(summary.keys())
        expected_order = fsp.KEY_ORDER
        # All spec keys should appear in order
        filtered = [k for k in keys if k in expected_order]
        assert filtered == expected_order

    def test_save_produces_stable_order(self, tmp_path):
        out = tmp_path / "test.json"
        fsp.save_summary(out, VALID_SUMMARY)
        loaded = json.loads(out.read_text())
        keys = list(loaded.keys())
        expected = [k for k in fsp.KEY_ORDER if k in loaded]
        assert keys == expected

    def test_save_roundtrip_preserves_content(self, tmp_path):
        out = tmp_path / "test.json"
        fsp.save_summary(out, VALID_SUMMARY)
        loaded = json.loads(out.read_text())
        for key in fsp.REQUIRED_KEYS:
            assert loaded[key] == VALID_SUMMARY[key]


# ==========================
# Test: Compression — boilerplate dedup
# ==========================

BOILERPLATE_PHRASES = [
    "depends significantly on global economic conditions",
    "materially adversely affect",
    "subject to significant risks",
    "results of operations, financial condition and stock price",
]


class TestCompression:
    @pytest.mark.parametrize("summary", [VALID_SUMMARY, ANNUAL_SUMMARY])
    def test_boilerplate_appears_at_most_once(self, summary):
        full_text = " ".join(summary[k] for k in fsp.PARAGRAPH_KEYS)
        full_lower = full_text.lower()

        for phrase in BOILERPLATE_PHRASES:
            count = full_lower.count(phrase.lower())
            assert count <= 1, (
                f"Boilerplate phrase appears {count} times: '{phrase}'"
            )

    @pytest.mark.parametrize("summary", [VALID_SUMMARY, ANNUAL_SUMMARY])
    def test_no_bullet_dashes_in_text(self, summary):
        for key in fsp.PARAGRAPH_KEYS:
            assert "\n- " not in summary[key]
            assert "\n* " not in summary[key]

    @pytest.mark.parametrize("summary", [VALID_SUMMARY, ANNUAL_SUMMARY])
    def test_no_numbered_lists(self, summary):
        for key in fsp.PARAGRAPH_KEYS:
            assert not re.search(
                r"^\s*\d+[.)]\s", summary[key], re.MULTILINE
            ), f"{key} contains numbered list"


# ==========================
# Test: parse_filing_header
# ==========================

class TestParseFilingHeader:
    def test_standard_header(self):
        text = "FORM: 10-Q\nFILING_DATE: 2026-01-30\nACCESSION: 0000320193-26-000006\n\nBody"
        h = fsp.parse_filing_header(text)
        assert h["form"] == "10-Q"
        assert h["filing_date"] == "2026-01-30"
        assert h["accession"] == "0000320193-26-000006"

    def test_missing_fields(self):
        h = fsp.parse_filing_header("no headers here")
        assert h == {}


# ==========================
# Test: extract_body
# ==========================

class TestExtractBody:
    def test_standard(self):
        text = "FORM: 10-Q\nFILING_DATE: 2026-01-30\nACCESSION: 123\n\nThe body text."
        assert fsp.extract_body(text) == "The body text."

    def test_no_blank_line(self):
        text = "no blank line\ncontent here"
        assert fsp.extract_body(text) == "no blank line\ncontent here"


# ==========================
# Test: chunk_text
# ==========================

class TestChunkText:
    def test_under_limit_returns_single(self):
        assert fsp.chunk_text("short", 100) == ["short"]

    def test_splits_on_paragraphs(self):
        text = "aaa\n\nbbb\n\nccc"
        chunks = fsp.chunk_text(text, max_chars=8)
        assert len(chunks) >= 2

    def test_hard_split_oversized(self):
        text = "x" * 200
        chunks = fsp.chunk_text(text, max_chars=50)
        assert all(len(c) <= 50 for c in chunks)


# ==========================
# Test: output_path_for
# ==========================

class TestOutputPathFor:
    def test_standard(self):
        p = fsp.output_path_for(Path("/out"), "AAPL", 2026, "Q1")
        assert p == Path("/out/AAPL/2026/Q1.json")


# ==========================
# Test: _derive_metadata
# ==========================

class TestDeriveMetadata:
    def test_10q_quarterly(self, tmp_path):
        f = tmp_path / "filing.txt"
        f.write_text("FORM: 10-Q\nFILING_DATE: 2026-01-30\nACCESSION: 123\n\nBody")
        form, date, fp, pt = fsp._derive_metadata([f], 2026, "Q1")
        assert form == "10-Q"
        assert date == "2026-01-30"
        assert fp == "2026-Q1"
        assert pt == "quarterly"

    def test_10k_annual(self, tmp_path):
        f = tmp_path / "filing.txt"
        f.write_text("FORM: 10-K\nFILING_DATE: 2025-02-14\nACCESSION: 456\n\nBody")
        form, date, fp, pt = fsp._derive_metadata([f], 2024, "Q4")
        assert form == "10-K"
        assert fp == "FY2024"
        assert pt == "annual"

    def test_10k_takes_priority_over_8k(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("FORM: 8-K\nFILING_DATE: 2025-11-15\nACCESSION: 789\n\nBody")
        f2 = tmp_path / "b.txt"
        f2.write_text("FORM: 10-K\nFILING_DATE: 2025-02-14\nACCESSION: 456\n\nBody")
        form, date, fp, pt = fsp._derive_metadata([f1, f2], 2024, "Q4")
        assert form == "10-K"
        assert pt == "annual"

    def test_10q_takes_priority_over_8k(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("FORM: 8-K\nFILING_DATE: 2026-02-24\nACCESSION: 789\n\nBody")
        f2 = tmp_path / "b.txt"
        f2.write_text("FORM: 10-Q\nFILING_DATE: 2026-01-30\nACCESSION: 456\n\nBody")
        form, date, fp, pt = fsp._derive_metadata([f1, f2], 2026, "Q1")
        assert form == "10-Q"
        assert pt == "quarterly"


# ==========================
# Test: discover_quarterly_groups
# ==========================

class TestDiscoverQuarterlyGroups:
    def _setup_clean_text(self, tmp_path, files):
        """Create clean_text/TICKER/ with the given (filename, header) pairs."""
        clean = tmp_path / "clean_text" / "AAPL"
        clean.mkdir(parents=True)
        for name, header_text in files:
            (clean / name).write_text(header_text + "\n\nBody text here.")
        return tmp_path

    def test_groups_by_quarter(self, tmp_path):
        base = self._setup_clean_text(tmp_path, [
            ("AAPL_2026_Q1_10-Q_2026-01-30.txt", "FORM: 10-Q\nFILING_DATE: 2026-01-30\nACCESSION: 1"),
            ("AAPL_2026_Q1_8-K_2026-02-24.txt", "FORM: 8-K\nFILING_DATE: 2026-02-24\nACCESSION: 2"),
        ])
        groups = fsp.discover_quarterly_groups(base)
        assert ("AAPL", 2026, "Q1") in groups
        assert len(groups[("AAPL", 2026, "Q1")]) == 2

    def test_filters_by_ticker(self, tmp_path):
        base = self._setup_clean_text(tmp_path, [
            ("AAPL_2026_Q1_10-Q_2026-01-30.txt", "FORM: 10-Q\nFILING_DATE: 2026-01-30\nACCESSION: 1"),
        ])
        groups = fsp.discover_quarterly_groups(base, tickers=["NVDA"])
        assert len(groups) == 0

    def test_filters_by_year(self, tmp_path):
        base = self._setup_clean_text(tmp_path, [
            ("AAPL_2026_Q1_10-Q_2026-01-30.txt", "FORM: 10-Q\nFILING_DATE: 2026-01-30\nACCESSION: 1"),
        ])
        groups = fsp.discover_quarterly_groups(base, years=[2025])
        assert len(groups) == 0

    def test_excludes_non_target_forms(self, tmp_path):
        base = self._setup_clean_text(tmp_path, [
            ("AAPL_2026_Q1_4_2026-01-30.txt", "FORM: 4\nFILING_DATE: 2026-01-30\nACCESSION: 1"),
        ])
        groups = fsp.discover_quarterly_groups(base)
        assert len(groups) == 0


# ==========================
# Test: parse_json_response
# ==========================

class TestParseJsonResponse:
    def test_plain_json(self):
        assert fsp.parse_json_response('{"a": 1}') == {"a": 1}

    def test_code_fenced(self):
        raw = '```json\n{"a": 1}\n```'
        assert fsp.parse_json_response(raw) == {"a": 1}

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            fsp.parse_json_response("not json")
