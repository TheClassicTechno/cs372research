"""Unit tests for sector constraint enforcement functions."""

import pytest

from multi_agent.graph.sector_constraints import (
    build_sector_constraint_text,
    build_sector_map,
    enforce_sector_limits,
    filter_allocation_by_permissions,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

SECTORS = {
    "tech": ["AAPL", "NVDA"],
    "energy": ["XOM", "CVX"],
    "financials": ["JPM", "BAC"],
}

SECTOR_MAP = build_sector_map(SECTORS)

UNIVERSE = ["AAPL", "NVDA", "XOM", "CVX", "JPM", "BAC"]


# ── build_sector_map ─────────────────────────────────────────────────────


class TestBuildSectorMap:
    def test_basic(self):
        result = build_sector_map({"tech": ["AAPL", "NVDA"], "energy": ["XOM"]})
        assert result == {"AAPL": "tech", "NVDA": "tech", "XOM": "energy"}

    def test_single_ticker_sectors(self):
        result = build_sector_map({"a": ["X"], "b": ["Y"]})
        assert result == {"X": "a", "Y": "b"}

    def test_empty(self):
        assert build_sector_map({}) == {}


# ── enforce_sector_limits ────────────────────────────────────────────────


class TestEnforceSectorLimits:
    def test_no_limits_passthrough(self):
        alloc = {"AAPL": 0.5, "XOM": 0.5}
        result = enforce_sector_limits(alloc, SECTOR_MAP, {})
        assert result == alloc

    def test_none_limits_passthrough(self):
        alloc = {"AAPL": 0.5, "XOM": 0.5}
        result = enforce_sector_limits(alloc, SECTOR_MAP, None)
        assert result == alloc

    def test_over_max_capped(self):
        # tech = 70%, energy = 30% → cap tech at 40%
        alloc = {"AAPL": 0.4, "NVDA": 0.3, "XOM": 0.2, "CVX": 0.1}
        limits = {"tech": {"min": 0.0, "max": 0.40}}
        result = enforce_sector_limits(alloc, SECTOR_MAP, limits)
        tech_total = result["AAPL"] + result["NVDA"]
        assert tech_total <= 0.40 + 1e-6
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_under_min_boosted(self):
        # energy = 5%, needs min 20%
        alloc = {"AAPL": 0.6, "NVDA": 0.15, "XOM": 0.03, "CVX": 0.02, "JPM": 0.1, "BAC": 0.1}
        limits = {"energy": {"min": 0.20, "max": 1.0}}
        result = enforce_sector_limits(alloc, SECTOR_MAP, limits)
        energy_total = result["XOM"] + result["CVX"]
        assert energy_total >= 0.20 - 1e-6
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_both_over_and_under(self):
        # tech over max, energy under min
        alloc = {"AAPL": 0.45, "NVDA": 0.25, "XOM": 0.02, "CVX": 0.03, "JPM": 0.15, "BAC": 0.10}
        limits = {
            "tech": {"min": 0.0, "max": 0.40},
            "energy": {"min": 0.10, "max": 1.0},
        }
        result = enforce_sector_limits(alloc, SECTOR_MAP, limits)
        tech_total = result["AAPL"] + result["NVDA"]
        energy_total = result["XOM"] + result["CVX"]
        assert tech_total <= 0.40 + 1e-6
        assert energy_total >= 0.10 - 1e-6
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_preserves_sum_to_one(self):
        alloc = {t: 1.0 / len(UNIVERSE) for t in UNIVERSE}
        limits = {"tech": {"min": 0.10, "max": 0.35}, "energy": {"min": 0.10, "max": 0.35}}
        result = enforce_sector_limits(alloc, SECTOR_MAP, limits)
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_preserves_intra_sector_ratios(self):
        # AAPL:NVDA = 2:1 should be preserved after capping tech
        alloc = {"AAPL": 0.4, "NVDA": 0.2, "XOM": 0.2, "CVX": 0.2}
        limits = {"tech": {"min": 0.0, "max": 0.30}}
        result = enforce_sector_limits(alloc, SECTOR_MAP, limits)
        # Ratio should be preserved (2:1)
        if result["NVDA"] > 1e-8:
            ratio = result["AAPL"] / result["NVDA"]
            assert ratio == pytest.approx(2.0, rel=0.05)

    def test_single_ticker_sector(self):
        sm = build_sector_map({"solo": ["A"], "pair": ["B", "C"]})
        alloc = {"A": 0.8, "B": 0.1, "C": 0.1}
        limits = {"solo": {"min": 0.0, "max": 0.30}}
        result = enforce_sector_limits(alloc, sm, limits)
        assert result["A"] <= 0.30 + 1e-6
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_no_negative_weights(self):
        alloc = {"AAPL": 0.5, "NVDA": 0.3, "XOM": 0.1, "CVX": 0.1}
        limits = {
            "tech": {"min": 0.0, "max": 0.20},
            "energy": {"min": 0.30, "max": 1.0},
        }
        result = enforce_sector_limits(alloc, SECTOR_MAP, limits)
        assert all(w >= -1e-8 for w in result.values())


# ── filter_allocation_by_permissions ─────────────────────────────────────


class TestFilterPermissions:
    def test_wildcard_no_filtering(self):
        alloc = {"AAPL": 0.5, "XOM": 0.3, "JPM": 0.2}
        perms = {"macro": ["*"]}
        result = filter_allocation_by_permissions(alloc, "macro", SECTOR_MAP, perms)
        assert result == alloc

    def test_role_not_in_map_no_filtering(self):
        alloc = {"AAPL": 0.5, "XOM": 0.5}
        perms = {"value": ["tech"]}
        result = filter_allocation_by_permissions(alloc, "macro", SECTOR_MAP, perms)
        assert result == alloc

    def test_disallowed_zeroed(self):
        alloc = {"AAPL": 0.4, "NVDA": 0.2, "XOM": 0.2, "JPM": 0.2}
        perms = {"macro": ["energy", "financials"]}
        result = filter_allocation_by_permissions(alloc, "macro", SECTOR_MAP, perms)
        assert result["AAPL"] == 0.0
        assert result["NVDA"] == 0.0
        assert result["XOM"] == 0.2
        assert result["JPM"] == 0.2

    def test_all_disallowed(self):
        alloc = {"AAPL": 0.5, "NVDA": 0.5}
        perms = {"risk": ["energy"]}
        result = filter_allocation_by_permissions(alloc, "risk", SECTOR_MAP, perms)
        assert result["AAPL"] == 0.0
        assert result["NVDA"] == 0.0

    def test_empty_permissions_dict(self):
        alloc = {"AAPL": 0.5, "XOM": 0.5}
        result = filter_allocation_by_permissions(alloc, "macro", SECTOR_MAP, {})
        assert result == alloc

    def test_ticker_without_sector_mapping_kept(self):
        # If a ticker is not in the sector map, it should be kept
        alloc = {"AAPL": 0.3, "UNKNOWN": 0.3, "XOM": 0.4}
        perms = {"macro": ["energy"]}
        result = filter_allocation_by_permissions(alloc, "macro", SECTOR_MAP, perms)
        assert result["AAPL"] == 0.0  # tech, not in allowed
        assert result["UNKNOWN"] == 0.3  # no sector → kept
        assert result["XOM"] == 0.4  # energy, allowed


# ── build_sector_constraint_text ──────────────────────────────────────


class TestBuildSectorConstraintText:
    """Tests for the prompt constraint text builder."""

    def test_no_config_returns_empty(self):
        assert build_sector_constraint_text(None, "macro") == ""

    def test_empty_config_returns_empty(self):
        assert build_sector_constraint_text({}, "macro") == ""

    def test_permissions_only(self):
        cfg = {
            "sectors": SECTORS,
            "agent_sector_permissions": {
                "value": ["consumer", "energy", "financials"],
            },
        }
        text = build_sector_constraint_text(cfg, "value")
        assert "MANDATORY SECTOR CONSTRAINT" in text
        assert "VALUE" in text
        assert "consumer" in text
        assert "energy" in text
        assert "financials" in text
        assert "MUST assign 0.0" in text

    def test_permissions_wildcard_excluded(self):
        cfg = {
            "sectors": SECTORS,
            "agent_sector_permissions": {"macro": ["*"]},
        }
        text = build_sector_constraint_text(cfg, "macro")
        assert "MANDATORY SECTOR CONSTRAINT" not in text

    def test_permissions_role_not_in_map(self):
        cfg = {
            "sectors": SECTORS,
            "agent_sector_permissions": {"value": ["tech"]},
        }
        text = build_sector_constraint_text(cfg, "macro")
        assert "MANDATORY SECTOR CONSTRAINT" not in text

    def test_sector_limits(self):
        cfg = {
            "sectors": SECTORS,
            "sector_limits": {
                "tech": {"min": 0.10, "max": 0.40},
                "energy": {"max": 0.30},
            },
        }
        text = build_sector_constraint_text(cfg, "macro")
        assert "MANDATORY SECTOR LIMITS" in text
        assert "tech" in text
        assert "energy" in text
        assert "10%" in text
        assert "40%" in text
        assert "30%" in text

    def test_sector_limits_no_constraint_lines_skipped(self):
        """Sectors with default bounds (min=0, max=1) produce no lines."""
        cfg = {
            "sectors": SECTORS,
            "sector_limits": {
                "tech": {"min": 0.0, "max": 1.0},
            },
        }
        text = build_sector_constraint_text(cfg, "macro")
        assert "MANDATORY SECTOR LIMITS" not in text

    def test_max_sector_weight(self):
        cfg = {
            "sectors": SECTORS,
            "max_sector_weight": 0.35,
        }
        text = build_sector_constraint_text(cfg, "macro")
        assert "MANDATORY MAX SECTOR WEIGHT" in text
        assert "35%" in text

    def test_all_three_constraints(self):
        cfg = {
            "sectors": SECTORS,
            "agent_sector_permissions": {
                "value": ["energy", "financials"],
            },
            "sector_limits": {"tech": {"max": 0.40}},
            "max_sector_weight": 0.35,
        }
        text = build_sector_constraint_text(cfg, "value")
        assert "MANDATORY SECTOR CONSTRAINT" in text
        assert "MANDATORY SECTOR LIMITS" in text
        assert "MANDATORY MAX SECTOR WEIGHT" in text

    def test_judge_no_permissions(self):
        cfg = {
            "sectors": SECTORS,
            "agent_sector_permissions": {
                "value": ["energy", "financials"],
            },
            "sector_limits": {"tech": {"max": 0.40}},
        }
        text = build_sector_constraint_text(cfg, "judge", include_permissions=False)
        assert "MANDATORY SECTOR CONSTRAINT" not in text
        assert "MANDATORY SECTOR LIMITS" in text
