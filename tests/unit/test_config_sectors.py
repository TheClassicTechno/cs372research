"""Unit tests for sector configuration validation in SimulationConfig."""

import pytest

from models.config import SectorLimit, SimulationConfig


# ── Helpers ──────────────────────────────────────────────────────────────

def _base_config(**overrides) -> dict:
    """Minimal valid config dict for SimulationConfig."""
    base = {
        "dataset_path": "data-pipeline/final_snapshots",
        "tickers": ["AAPL", "NVDA", "XOM", "JPM"],
        "invest_quarter": "2025Q1",
        "agent": {
            "agent_system": "multi_agent_debate",
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
        },
        "allocation_constraints": {
            "max_weight": 0.50,
            "min_holdings": 2,
            "max_tickers": 10,
        },
    }
    base.update(overrides)
    return base


VALID_SECTORS = {
    "tech": ["AAPL", "NVDA"],
    "energy": ["XOM"],
    "financials": ["JPM"],
}


# ── SectorLimit model ───────────────────────────────────────────────────


class TestSectorLimit:
    def test_valid(self):
        sl = SectorLimit(min=0.10, max=0.40)
        assert sl.min == 0.10
        assert sl.max == 0.40

    def test_defaults(self):
        sl = SectorLimit()
        assert sl.min == 0.0
        assert sl.max == 1.0

    def test_min_gt_max_raises(self):
        with pytest.raises(ValueError, match="min .* > max"):
            SectorLimit(min=0.50, max=0.20)


# ── Backward compatibility ──────────────────────────────────────────────


class TestBackwardCompatibility:
    def test_no_sectors_loads_fine(self):
        config = SimulationConfig(**_base_config())
        assert config.sectors is None
        assert config.sector_limits is None
        assert config.agent_sector_permissions is None
        assert config.agent.sector_config is None

    def test_existing_config_file(self):
        """Agent YAML merged with a scenario produces a valid config with no sectors."""
        import yaml
        from run_simulation import _deep_merge

        with open("config/agents/debate_diverse_agents.yaml") as f:
            agent_raw = yaml.safe_load(f)
        # Agent configs don't include invest_quarter — scenarios supply it.
        scenario = {"invest_quarter": "2025Q1"}
        merged = _deep_merge(agent_raw, scenario)
        config = SimulationConfig(**merged)
        assert config.sectors is None
        assert config.agent.sector_config is None


# ── Sector validation ───────────────────────────────────────────────────


class TestSectorValidation:
    def test_valid_sector_config(self):
        config = SimulationConfig(**_base_config(sectors=VALID_SECTORS))
        assert config.sectors == VALID_SECTORS
        assert config.agent.sector_config is not None
        assert config.agent.sector_config["sectors"] == VALID_SECTORS

    def test_ticker_in_multiple_sectors_raises(self):
        bad_sectors = {
            "tech": ["AAPL", "NVDA"],
            "also_tech": ["AAPL"],  # duplicate
            "energy": ["XOM"],
            "financials": ["JPM"],
        }
        with pytest.raises(ValueError, match="appears in multiple sectors"):
            SimulationConfig(**_base_config(sectors=bad_sectors))

    def test_ticker_missing_from_sectors_raises(self):
        # JPM missing from sector map
        incomplete = {"tech": ["AAPL", "NVDA"], "energy": ["XOM"]}
        with pytest.raises(ValueError, match="missing from sector mapping"):
            SimulationConfig(**_base_config(sectors=incomplete))

    def test_sector_references_unknown_ticker_raises(self):
        bad_sectors = {
            "tech": ["AAPL", "NVDA", "FAKE"],
            "energy": ["XOM"],
            "financials": ["JPM"],
        }
        with pytest.raises(ValueError, match="unknown tickers"):
            SimulationConfig(**_base_config(sectors=bad_sectors))


# ── sector_limits validation ────────────────────────────────────────────


class TestSectorLimitsValidation:
    def test_sector_limits_without_sectors_raises(self):
        with pytest.raises(ValueError, match="requires 'sectors'"):
            SimulationConfig(**_base_config(
                sector_limits={"tech": {"min": 0.10, "max": 0.40}},
            ))

    def test_unknown_sector_in_limits_raises(self):
        with pytest.raises(ValueError, match="unknown sector"):
            SimulationConfig(**_base_config(
                sectors=VALID_SECTORS,
                sector_limits={"nonexistent": {"min": 0.0, "max": 0.50}},
            ))

    def test_sector_min_sum_exceeds_one_raises(self):
        with pytest.raises(ValueError, match="sum to .* > 1.0"):
            SimulationConfig(**_base_config(
                sectors=VALID_SECTORS,
                sector_limits={
                    "tech": {"min": 0.50, "max": 1.0},
                    "energy": {"min": 0.40, "max": 1.0},
                    "financials": {"min": 0.20, "max": 1.0},
                },
            ))

    def test_sector_max_sum_below_one_raises(self):
        with pytest.raises(ValueError, match="sum to .* < 1.0"):
            SimulationConfig(**_base_config(
                sectors=VALID_SECTORS,
                sector_limits={
                    "tech": {"min": 0.0, "max": 0.20},
                    "energy": {"min": 0.0, "max": 0.20},
                    "financials": {"min": 0.0, "max": 0.20},
                },
            ))

    def test_valid_sector_limits(self):
        config = SimulationConfig(**_base_config(
            sectors=VALID_SECTORS,
            sector_limits={
                "tech": {"min": 0.10, "max": 0.50},
                "energy": {"min": 0.05, "max": 0.30},
                "financials": {"min": 0.05, "max": 0.40},
            },
        ))
        assert config.agent.sector_config["sector_limits"] is not None


# ── agent_sector_permissions validation ─────────────────────────────────


class TestAgentSectorPermissions:
    def test_permissions_without_sectors_raises(self):
        with pytest.raises(ValueError, match="requires 'sectors'"):
            SimulationConfig(**_base_config(
                agent_sector_permissions={"macro": ["tech"]},
            ))

    def test_unknown_role_raises(self):
        with pytest.raises(ValueError, match="unknown role"):
            SimulationConfig(**_base_config(
                sectors=VALID_SECTORS,
                agent_sector_permissions={"nonexistent_role": ["tech"]},
            ))

    def test_unknown_sector_in_permissions_raises(self):
        with pytest.raises(ValueError, match="unknown sector"):
            SimulationConfig(**_base_config(
                sectors=VALID_SECTORS,
                agent_sector_permissions={"macro": ["nonexistent_sector"]},
            ))

    def test_wildcard_valid(self):
        config = SimulationConfig(**_base_config(
            sectors=VALID_SECTORS,
            agent_sector_permissions={"technical": ["*"]},
        ))
        perms = config.agent.sector_config["agent_sector_permissions"]
        assert perms["technical"] == ["*"]

    def test_valid_permissions(self):
        config = SimulationConfig(**_base_config(
            sectors=VALID_SECTORS,
            agent_sector_permissions={
                "macro": ["energy", "financials"],
                "value": ["tech", "financials"],
            },
        ))
        assert config.agent.sector_config["agent_sector_permissions"] is not None


# ── sector_config threading ─────────────────────────────────────────────


class TestSectorConfigThreading:
    def test_sector_config_packed_into_agent(self):
        config = SimulationConfig(**_base_config(
            sectors=VALID_SECTORS,
            sector_limits={
                "tech": {"min": 0.10, "max": 0.50},
                "energy": {"min": 0.05, "max": 0.30},
                "financials": {"min": 0.05, "max": 0.40},
            },
            agent_sector_permissions={"macro": ["tech"]},
        ))
        sc = config.agent.sector_config
        assert sc is not None
        assert sc["sectors"] == VALID_SECTORS
        assert sc["sector_limits"]["tech"] == {"min": 0.10, "max": 0.50}
        assert sc["agent_sector_permissions"]["macro"] == ["tech"]

    def test_sectors_only_packs_nulls(self):
        config = SimulationConfig(**_base_config(sectors=VALID_SECTORS))
        sc = config.agent.sector_config
        assert sc is not None
        assert sc["sectors"] == VALID_SECTORS
        assert sc["sector_limits"] is None
        assert sc["agent_sector_permissions"] is None


# ── max_sector_weight validation ──────────────────────────────────────


class TestMaxSectorWeight:
    def test_max_sector_weight_without_sectors_raises(self):
        with pytest.raises(ValueError, match="requires 'sectors'"):
            SimulationConfig(**_base_config(max_sector_weight=0.40))

    def test_max_sector_weight_valid(self):
        config = SimulationConfig(**_base_config(
            sectors=VALID_SECTORS,
            max_sector_weight=0.40,
        ))
        assert config.max_sector_weight == 0.40

    def test_max_sector_weight_packed_into_sector_config(self):
        config = SimulationConfig(**_base_config(
            sectors=VALID_SECTORS,
            max_sector_weight=0.35,
        ))
        sc = config.agent.sector_config
        assert sc is not None
        assert sc["max_sector_weight"] == 0.35
