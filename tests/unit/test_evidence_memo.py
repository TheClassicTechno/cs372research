"""Unit tests for memo evidence ID extraction."""

import pytest

from eval.evidence import extract_evidence_ids, extract_evidence_spans, parse_memo_evidence


# ── extract_evidence_ids ─────────────────────────────────────────────────────


class TestExtractEvidenceIds:
    def test_single_id(self):
        assert extract_evidence_ids("Based on [AAPL-RET60] data") == {"AAPL-RET60"}

    def test_multiple_ids(self):
        text = "Citing [L1-VIX] and [NVDA-F3] shows risk."
        assert extract_evidence_ids(text) == {"L1-VIX", "NVDA-F3"}

    def test_no_match(self):
        assert extract_evidence_ids("No evidence here.") == set()

    def test_l1_prefix(self):
        assert extract_evidence_ids("[L1-FF] factor model") == {"L1-FF"}

    def test_l1_cpi(self):
        assert extract_evidence_ids("Inflation [L1-CPI] rising") == {"L1-CPI"}

    def test_duplicates_collapsed(self):
        text = "[AAPL-RET60] and again [AAPL-RET60]"
        assert extract_evidence_ids(text) == {"AAPL-RET60"}

    def test_various_formats(self):
        text = "[MSFT-VOL20] [TSLA-BETA] [JPM-F1] [META-SHARPE]"
        expected = {"MSFT-VOL20", "TSLA-BETA", "JPM-F1", "META-SHARPE"}
        assert extract_evidence_ids(text) == expected

    def test_empty_string(self):
        assert extract_evidence_ids("") == set()

    def test_l0_narrative_ids(self):
        text = "Growth outlook [L0-GROWTH-RESILIENCE] supports risk-on [L0-AI-RALLY]."
        assert extract_evidence_ids(text) == {"L0-GROWTH-RESILIENCE", "L0-AI-RALLY"}

    def test_l0_inflation(self):
        assert extract_evidence_ids("[L0-INFLATION-PERSISTENCE] drives caution") == {
            "L0-INFLATION-PERSISTENCE"
        }

    def test_l2_earnings_ids(self):
        text = "EPS beat rate [L2-EPS-BEAT] and revision breadth [L2-REV-BREADTH]."
        assert extract_evidence_ids(text) == {"L2-EPS-BEAT", "L2-REV-BREADTH"}

    def test_l3_positioning_ids(self):
        text = "CTA flows [L3-CTA] and gamma exposure [L3-GAMMA]."
        assert extract_evidence_ids(text) == {"L3-CTA", "L3-GAMMA"}

    def test_mixed_levels(self):
        text = "[L0-MACRO-REGIME] [L1-VIX] [L2-DISPERSION] [L3-HF-LEV] [AAPL-RET60]"
        expected = {"L0-MACRO-REGIME", "L1-VIX", "L2-DISPERSION", "L3-HF-LEV", "AAPL-RET60"}
        assert extract_evidence_ids(text) == expected


# ── extract_evidence_spans (memo path) ───────────────────────────────────────


class TestExtractEvidenceSpansMemo:
    def test_ids_from_justification(self):
        decision = {
            "action_dict": {
                "justification": "Strong momentum [AAPL-RET60] in tech sector [L1-VIX].",
                "claims": [],
            },
        }
        result = extract_evidence_spans(decision)
        assert "AAPL-RET60" in result
        assert "L1-VIX" in result

    def test_ids_from_claim_text(self):
        decision = {
            "action_dict": {
                "justification": "",
                "claims": [
                    {"claim_text": "NVDA growth [NVDA-F1] is strong."},
                ],
            },
        }
        result = extract_evidence_spans(decision)
        assert "NVDA-F1" in result

    def test_ids_from_raw_response(self):
        decision = {
            "action_dict": {"justification": "", "claims": []},
            "raw_response": "Analysis shows [GOOG-ROE] declining.",
        }
        result = extract_evidence_spans(decision)
        assert "GOOG-ROE" in result

    def test_fallback_to_legacy_when_no_memo_ids(self):
        decision = {
            "action_dict": {
                "justification": "Tech is strong.",
                "claims": [
                    {"claim_text": "Revenue growing", "variables": ["nvda_revenue"]},
                ],
            },
        }
        result = extract_evidence_spans(decision)
        # Should fall back to legacy path — normalized variables
        assert len(result) > 0
        # The legacy path normalizes variables
        assert any("nvda" in s for s in result)

    def test_memo_ids_take_priority(self):
        """When memo IDs are found, legacy variables are NOT extracted."""
        decision = {
            "action_dict": {
                "justification": "Based on [AAPL-RET60] data.",
                "claims": [
                    {"claim_text": "Revenue up", "variables": ["legacy_var"]},
                ],
            },
        }
        result = extract_evidence_spans(decision)
        assert "AAPL-RET60" in result
        # Legacy variable should not be present (early return on memo IDs)
        assert "legacyvar" not in result

    def test_empty_decision_crashes(self):
        """Empty decision dict must crash — action_dict is required."""
        with pytest.raises(KeyError, match="action_dict"):
            extract_evidence_spans({})

    def test_missing_action_dict_crashes(self):
        """Decision without action_dict must crash."""
        with pytest.raises(KeyError, match="action_dict"):
            extract_evidence_spans({"raw_response": ""})

    def test_l0_ids_extracted_from_portfolio_rationale(self):
        """L0 narrative IDs in portfolio_rationale are extracted."""
        decision = {
            "action_dict": {
                "portfolio_rationale": "Growth [L0-GROWTH-RESILIENCE] and AI [L0-AI-RALLY] drive overweight.",
                "claims": [],
            },
        }
        result = extract_evidence_spans(decision)
        assert "L0-GROWTH-RESILIENCE" in result
        assert "L0-AI-RALLY" in result


# ── parse_memo_evidence (L0 lines) ───────────────────────────────────────────


class TestParseMemoEvidenceMultiLevel:
    def test_l0_lines_parsed(self):
        memo = (
            "[L0-GROWTH-RESILIENCE]  GDP growth remains above trend at 2.4%.\n"
            "[L0-AI-RALLY]  AI infrastructure spend accelerating.\n"
        )
        lookup = parse_memo_evidence(memo)
        assert "L0-GROWTH-RESILIENCE" in lookup
        assert "L0-AI-RALLY" in lookup
        assert "GDP growth" in lookup["L0-GROWTH-RESILIENCE"]

    def test_l2_lines_parsed(self):
        memo = "[L2-EPS-BEAT]  EPS beat rate 72% across S&P 500.\n"
        lookup = parse_memo_evidence(memo)
        assert "L2-EPS-BEAT" in lookup

    def test_mixed_levels(self):
        memo = (
            "[L0-MACRO-REGIME]  Higher for longer.\n"
            "[L1-VIX]  VIX: 17.35\n"
            "[L2-DISPERSION]  Return dispersion elevated.\n"
            "[AAPL-RET60]  60D Return: +11.9%\n"
        )
        lookup = parse_memo_evidence(memo)
        assert len(lookup) == 4
        assert "L0-MACRO-REGIME" in lookup
        assert "L1-VIX" in lookup
        assert "L2-DISPERSION" in lookup
        assert "AAPL-RET60" in lookup
