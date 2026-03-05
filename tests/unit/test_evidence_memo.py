"""Unit tests for memo evidence ID extraction."""

import pytest

from eval.evidence import extract_evidence_ids, extract_evidence_spans


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

    def test_empty_decision(self):
        result = extract_evidence_spans({})
        assert result == set()

    def test_missing_action_dict(self):
        result = extract_evidence_spans({"raw_response": ""})
        assert result == set()
