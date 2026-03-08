"""Unit tests for normalize_allocation() — the 8-step validation/normalization algorithm."""

import pytest

from multi_agent.graph import normalize_allocation


UNIVERSE_3 = ["AAPL", "MSFT", "GOOG"]
UNIVERSE_4 = ["AAPL", "MSFT", "GOOG", "TSLA"]


class TestNormalizeAllocation:
    """Core algorithm tests."""

    # ── basic normalization ──────────────────────────────────────────────

    def test_already_normalized(self):
        result = normalize_allocation(
            {"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2},
            UNIVERSE_3, max_weight=1.0, min_holdings=1,
        )
        assert result["AAPL"] == pytest.approx(0.5)
        assert result["MSFT"] == pytest.approx(0.3)
        assert result["GOOG"] == pytest.approx(0.2)

    def test_unnormalized_input(self):
        result = normalize_allocation(
            {"AAPL": 30, "MSFT": 70},
            ["AAPL", "MSFT"], max_weight=1.0, min_holdings=1,
        )
        assert result["AAPL"] == pytest.approx(0.3)
        assert result["MSFT"] == pytest.approx(0.7)

    # ── unknown tickers dropped ──────────────────────────────────────────

    def test_unknown_tickers_dropped(self):
        result = normalize_allocation(
            {"AAPL": 0.5, "XYZ": 0.5},
            UNIVERSE_3, max_weight=1.0, min_holdings=1,
        )
        assert "XYZ" not in result
        assert "AAPL" in result
        assert result["AAPL"] == pytest.approx(1.0)

    # ── missing universe tickers filled ──────────────────────────────────

    def test_missing_universe_tickers_get_zero(self):
        result = normalize_allocation(
            {"AAPL": 1.0},
            UNIVERSE_3, max_weight=1.0, min_holdings=1,
        )
        assert set(result.keys()) == set(UNIVERSE_3)
        assert result["MSFT"] == pytest.approx(0.0)
        assert result["GOOG"] == pytest.approx(0.0)

    # ── negative weights clamped ─────────────────────────────────────────

    def test_negative_weights_clamped(self):
        result = normalize_allocation(
            {"AAPL": -0.5, "MSFT": 1.0, "GOOG": 0.5},
            UNIVERSE_3, max_weight=1.0, min_holdings=1,
        )
        # AAPL should be 0 (clamped), MSFT and GOOG normalized from 1.5
        assert result["AAPL"] == pytest.approx(0.0)
        assert sum(result.values()) == pytest.approx(1.0)

    # ── all-zero fallback ────────────────────────────────────────────────

    def test_all_zero_fallback(self):
        result = normalize_allocation(
            {"AAPL": 0, "MSFT": 0, "GOOG": 0},
            UNIVERSE_3, max_weight=1.0, min_holdings=1,
        )
        # Equal-weight rounded to 2dp; one ticker absorbs rounding residual
        assert sum(result.values()) == pytest.approx(1.0)
        for t in UNIVERSE_3:
            assert result[t] == pytest.approx(1.0 / 3, abs=0.01)

    def test_empty_input_fallback(self):
        result = normalize_allocation(
            {},
            UNIVERSE_3, max_weight=1.0, min_holdings=1,
        )
        assert sum(result.values()) == pytest.approx(1.0)
        for t in UNIVERSE_3:
            assert result[t] == pytest.approx(1.0 / 3, abs=0.01)

    # ── max_weight cap ───────────────────────────────────────────────────

    def test_max_weight_cap_feasible(self):
        """With enough tickers, max_weight is enforced strictly."""
        result = normalize_allocation(
            {"AAPL": 0.9, "MSFT": 0.05, "GOOG": 0.05},
            UNIVERSE_3, max_weight=0.4, min_holdings=1,
        )
        assert result["AAPL"] <= 0.4 + 1e-6
        assert sum(result.values()) == pytest.approx(1.0)

    def test_max_weight_cap_infeasible(self):
        """With 2 tickers and max=0.4, constraint is infeasible — effective
        max becomes 1/n = 0.5 and result is equal weight."""
        result = normalize_allocation(
            {"AAPL": 0.9, "MSFT": 0.1},
            ["AAPL", "MSFT"], max_weight=0.4, min_holdings=1,
        )
        # Can't enforce 0.4 with 2 tickers; effective max is 0.5
        assert result["AAPL"] <= 0.5 + 1e-6
        assert sum(result.values()) == pytest.approx(1.0)

    def test_max_weight_iterative_convergence(self):
        """Multiple tickers over cap with enough room to redistribute."""
        result = normalize_allocation(
            {"AAPL": 0.8, "MSFT": 0.6, "GOOG": 0.1, "TSLA": 0.05},
            UNIVERSE_4, max_weight=0.33, min_holdings=1,
        )
        for t in UNIVERSE_4:
            assert result[t] <= 0.33 + 1e-6, f"{t} exceeds max: {result[t]}"
        assert sum(result.values()) == pytest.approx(1.0)

    # ── min_holdings enforcement ─────────────────────────────────────────

    def test_min_holdings_enforced(self):
        result = normalize_allocation(
            {"AAPL": 1.0, "MSFT": 0.0, "GOOG": 0.0, "TSLA": 0.0},
            UNIVERSE_4, max_weight=0.5, min_holdings=3,
        )
        non_zero = sum(1 for w in result.values() if w > 1e-8)
        assert non_zero >= 3

    def test_min_holdings_skipped_when_universe_too_small(self):
        """min_holdings=5 but only 3 tickers — should not crash."""
        result = normalize_allocation(
            {"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2},
            UNIVERSE_3, max_weight=1.0, min_holdings=5,
        )
        assert sum(result.values()) == pytest.approx(1.0)

    # ── tight constraints ────────────────────────────────────────────────

    def test_tight_constraints(self):
        """max_weight=0.34 with 3 tickers and min_holdings=3."""
        result = normalize_allocation(
            {"AAPL": 1.0, "MSFT": 0.0, "GOOG": 0.0},
            UNIVERSE_3, max_weight=0.34, min_holdings=3,
        )
        for t in UNIVERSE_3:
            assert result[t] <= 0.34 + 1e-6
            assert result[t] > 0  # all must be non-zero
        assert sum(result.values()) == pytest.approx(1.0)


# ── parametrized invariant tests ─────────────────────────────────────────────


_INVARIANT_CASES = [
    ({"AAPL": 0.5, "MSFT": 0.5}, ["AAPL", "MSFT"], 0.4, 1),
    ({"AAPL": 0.9, "MSFT": 0.1}, ["AAPL", "MSFT"], 0.6, 1),
    ({"A": 10, "B": 20, "C": 70}, ["A", "B", "C"], 0.4, 2),
    ({"X": 0.0}, ["X", "Y", "Z"], 1.0, 1),
    ({"X": -1.0, "Y": 2.0}, ["X", "Y"], 1.0, 1),
    ({}, ["A", "B", "C", "D"], 0.3, 3),
]


class TestNormalizeInvariants:
    @pytest.mark.parametrize("raw,universe,max_w,min_h", _INVARIANT_CASES)
    def test_sums_to_one(self, raw, universe, max_w, min_h):
        result = normalize_allocation(raw, universe, max_w, min_h)
        assert sum(result.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("raw,universe,max_w,min_h", _INVARIANT_CASES)
    def test_keys_match_universe(self, raw, universe, max_w, min_h):
        result = normalize_allocation(raw, universe, max_w, min_h)
        assert set(result.keys()) == set(universe)

    @pytest.mark.parametrize("raw,universe,max_w,min_h", _INVARIANT_CASES)
    def test_no_weight_exceeds_effective_max(self, raw, universe, max_w, min_h):
        """No weight exceeds effective max = max(max_weight, 1/n).
        When n * max_weight < 1.0, the constraint is infeasible and the
        effective cap is raised to 1/n."""
        result = normalize_allocation(raw, universe, max_w, min_h)
        effective_max = max(max_w, 1.0 / len(universe))
        for t, w in result.items():
            assert w <= effective_max + 1e-6, f"{t}={w} exceeds effective_max={effective_max}"

    @pytest.mark.parametrize("raw,universe,max_w,min_h", _INVARIANT_CASES)
    def test_no_negative_weights(self, raw, universe, max_w, min_h):
        result = normalize_allocation(raw, universe, max_w, min_h)
        for t, w in result.items():
            assert w >= -1e-8, f"{t}={w} is negative"
