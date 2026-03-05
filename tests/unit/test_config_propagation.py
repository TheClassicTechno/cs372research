"""Tests for config propagation and scenario merging.

Covers:
- _deep_merge() function (scenario overlay onto agent config)
- Allocation constraint feasibility validation
- All scenario YAML files load and validate after merging
- Config propagation: AgentConfig -> DebateConfig -> to_dict()
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from models.config import AgentConfig, AllocationConstraints, SimulationConfig


# ── Paths ────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_AGENTS_DIR = _PROJECT_ROOT / "config" / "agents"
_SCENARIOS_DIR = _PROJECT_ROOT / "config" / "scenarios"

# Use the diverse agents config as the base for scenario merge tests.
_BASE_AGENTS_YAML = _AGENTS_DIR / "debate_diverse_agents.yaml"


# ── helpers ──────────────────────────────────────────────────────────────────

def _base_agent(**overrides) -> dict:
    d = {
        "agent_system": "multi_agent_debate",
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "temperature": 0.3,
    }
    d.update(overrides)
    return d


def _sim_config(**overrides) -> SimulationConfig:
    d = {
        "dataset_path": "data/cases",
        "tickers": ["AAPL", "MSFT", "GOOG"],
        "invest_quarter": "2025Q1",
        "agent": _base_agent(),
    }
    d.update(overrides)
    return SimulationConfig(**d)


# Import _deep_merge from run_simulation
from run_simulation import _deep_merge


# ═════════════════════════════════════════════════════════════════════════════
# 1. _deep_merge() unit tests
# ═════════════════════════════════════════════════════════════════════════════


class TestDeepMerge:
    """Unit tests for the _deep_merge function that combines agent + scenario."""

    def test_override_replaces_scalar(self):
        base = {"invest_quarter": "2025Q1", "tickers": ["AAPL"]}
        override = {"invest_quarter": "2022Q2"}
        merged = _deep_merge(base, override)
        assert merged["invest_quarter"] == "2022Q2"
        assert merged["tickers"] == ["AAPL"]  # untouched

    def test_override_replaces_list(self):
        """Lists are replaced entirely, not concatenated."""
        base = {"tickers": ["AAPL", "MSFT"]}
        override = {"tickers": ["NVDA"]}
        merged = _deep_merge(base, override)
        assert merged["tickers"] == ["NVDA"]

    def test_nested_dict_merge(self):
        """Nested dicts are merged recursively (scenario overrides only matching keys)."""
        base = {
            "allocation_constraints": {
                "max_weight": 0.20,
                "min_holdings": 5,
                "fully_invested": True,
                "max_tickers": 20,
            }
        }
        override = {
            "allocation_constraints": {
                "max_weight": 0.30,
            }
        }
        merged = _deep_merge(base, override)
        assert merged["allocation_constraints"]["max_weight"] == 0.30
        assert merged["allocation_constraints"]["min_holdings"] == 5  # preserved
        assert merged["allocation_constraints"]["fully_invested"] is True
        assert merged["allocation_constraints"]["max_tickers"] == 20

    def test_override_adds_new_key(self):
        base = {"tickers": ["AAPL"]}
        override = {"sectors": {"tech": ["AAPL"]}}
        merged = _deep_merge(base, override)
        assert "sectors" in merged
        assert merged["sectors"]["tech"] == ["AAPL"]
        assert merged["tickers"] == ["AAPL"]

    def test_override_none_value(self):
        """None in override replaces base value."""
        base = {"max_sector_weight": 0.40}
        override = {"max_sector_weight": None}
        merged = _deep_merge(base, override)
        assert merged["max_sector_weight"] is None

    def test_empty_override(self):
        base = {"a": 1, "b": 2}
        merged = _deep_merge(base, {})
        assert merged == {"a": 1, "b": 2}

    def test_empty_base(self):
        override = {"a": 1}
        merged = _deep_merge({}, override)
        assert merged == {"a": 1}

    def test_does_not_mutate_base(self):
        base = {"agent": {"temperature": 0.3}}
        override = {"agent": {"temperature": 0.7}}
        _deep_merge(base, override)
        assert base["agent"]["temperature"] == 0.3  # unchanged

    def test_deeply_nested_merge(self):
        """Three levels of nesting."""
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"c": 99}}}
        merged = _deep_merge(base, override)
        assert merged["a"]["b"]["c"] == 99
        assert merged["a"]["b"]["d"] == 2

    def test_dict_replaces_non_dict(self):
        """If base has a scalar but override has a dict, override wins."""
        base = {"agent": "some_string"}
        override = {"agent": {"model": "gpt-4o"}}
        merged = _deep_merge(base, override)
        assert merged["agent"] == {"model": "gpt-4o"}

    def test_non_dict_replaces_dict(self):
        """If base has a dict but override has a scalar, override wins."""
        base = {"agent": {"model": "gpt-4o"}}
        override = {"agent": "disabled"}
        merged = _deep_merge(base, override)
        assert merged["agent"] == "disabled"


# ═════════════════════════════════════════════════════════════════════════════
# 2. Allocation constraint feasibility
# ═════════════════════════════════════════════════════════════════════════════


class TestAllocationFeasibility:
    """Tests for the max_weight * min_holdings >= 1.0 validation."""

    def test_infeasible_raises(self):
        """max_weight=0.30, min_holdings=3 -> 0.90 < 1.0 -> should fail."""
        with pytest.raises(ValueError, match="Impossible allocation constraints"):
            _sim_config(
                allocation_constraints={"max_weight": 0.30, "min_holdings": 3},
            )

    def test_barely_feasible_passes(self):
        """max_weight=0.34, min_holdings=3 -> 1.02 >= 1.0 -> should pass."""
        cfg = _sim_config(
            allocation_constraints={"max_weight": 0.34, "min_holdings": 3},
        )
        assert cfg.allocation_constraints.max_weight == 0.34

    def test_exact_boundary_passes(self):
        """max_weight=0.25, min_holdings=4 -> 1.00 == 1.0 -> should pass."""
        cfg = _sim_config(
            allocation_constraints={"max_weight": 0.25, "min_holdings": 4},
        )
        assert cfg.allocation_constraints.max_weight == 0.25

    def test_comfortable_margin_passes(self):
        """max_weight=0.20, min_holdings=5 -> 1.00 >= 1.0 -> should pass."""
        cfg = _sim_config(
            allocation_constraints={"max_weight": 0.20, "min_holdings": 5},
        )
        assert cfg.allocation_constraints.min_holdings == 5

    def test_infeasible_small_weight(self):
        """max_weight=0.10, min_holdings=5 -> 0.50 < 1.0 -> should fail."""
        with pytest.raises(ValueError, match="Impossible allocation constraints"):
            _sim_config(
                allocation_constraints={"max_weight": 0.10, "min_holdings": 5},
            )

    def test_empty_tickers_raises(self):
        with pytest.raises(ValueError, match="tickers must not be empty"):
            _sim_config(tickers=[])


# ═════════════════════════════════════════════════════════════════════════════
# 3. All scenario YAMLs load and validate after merging with agent config
# ═════════════════════════════════════════════════════════════════════════════


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _all_scenario_files() -> list[Path]:
    return sorted(_SCENARIOS_DIR.glob("*.yaml"))


def _all_scenario_ids() -> list[str]:
    return [f.stem for f in _all_scenario_files()]


@pytest.fixture(params=_all_scenario_files(), ids=_all_scenario_ids())
def scenario_path(request) -> Path:
    return request.param


class TestScenarioYAMLsLoad:
    """Verify every scenario YAML in config/scenarios/ produces a valid config."""

    def test_scenario_loads_standalone(self, scenario_path: Path):
        """Each scenario YAML is valid YAML with invest_quarter and tickers."""
        raw = _load_yaml(scenario_path)
        assert "invest_quarter" in raw, f"{scenario_path.name} missing invest_quarter"
        assert "tickers" in raw, f"{scenario_path.name} missing tickers"
        assert len(raw["tickers"]) > 0, f"{scenario_path.name} has empty tickers"

    def test_scenario_merges_with_agent_config(self, scenario_path: Path):
        """Merging scenario with agent config produces a valid SimulationConfig."""
        if not _BASE_AGENTS_YAML.exists():
            pytest.skip("debate_diverse_agents.yaml not found")

        base_raw = _load_yaml(_BASE_AGENTS_YAML)
        scenario_raw = _load_yaml(scenario_path)
        merged = _deep_merge(base_raw, scenario_raw)

        # Should not raise any validation errors
        config = SimulationConfig(**merged)
        assert config.invest_quarter == scenario_raw["invest_quarter"]
        assert set(config.tickers) == set(scenario_raw["tickers"])

    def test_scenario_ticker_count_within_max(self, scenario_path: Path):
        """Scenario's ticker count must not exceed its own max_tickers."""
        raw = _load_yaml(scenario_path)
        tickers = raw.get("tickers", [])
        max_tickers = raw.get("allocation_constraints", {}).get("max_tickers", 10)
        assert len(tickers) <= max_tickers, (
            f"{scenario_path.name}: {len(tickers)} tickers > max_tickers={max_tickers}"
        )

    def test_scenario_allocation_feasibility(self, scenario_path: Path):
        """Scenario's allocation constraints must be feasible."""
        raw = _load_yaml(scenario_path)
        ac = raw.get("allocation_constraints", {})
        max_weight = ac.get("max_weight", 0.40)
        min_holdings = ac.get("min_holdings", 3)
        product = max_weight * min_holdings
        assert product >= 1.0 - 1e-8, (
            f"{scenario_path.name}: max_weight({max_weight}) * min_holdings({min_holdings}) "
            f"= {product:.2f} < 1.0"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 4. Config propagation: AgentConfig -> DebateConfig -> to_dict()
# ═════════════════════════════════════════════════════════════════════════════


class TestConfigPropagation:
    """Verify settings flow from AgentConfig through DebateConfig to to_dict()."""

    def _build_debate_config(self, **agent_overrides) -> "DebateConfig":
        """Build a DebateConfig by going through the real adapter path."""
        from agents.multi_agent_debate import DebateAgentSystem

        agent_cfg = AgentConfig(**_base_agent(
            system_prompt_override="mock",
            **agent_overrides,
        ))
        system = DebateAgentSystem(agent_cfg)
        return system._debate_cfg

    def test_model_name_propagated(self):
        cfg = self._build_debate_config(llm_model="gpt-4o")
        assert cfg.model_name == "gpt-4o"

    def test_temperature_propagated(self):
        cfg = self._build_debate_config(temperature=0.9)
        assert cfg.temperature == 0.9

    def test_max_rounds_propagated(self):
        cfg = self._build_debate_config(max_rounds=5)
        assert cfg.max_rounds == 5

    def test_initial_beta_propagated(self):
        cfg = self._build_debate_config(pid_initial_beta=0.7)
        assert cfg.initial_beta == 0.7

    def test_parallel_agents_propagated(self):
        cfg = self._build_debate_config(parallel_agents=False)
        assert cfg.parallel_agents is False

    def test_llm_stagger_ms_propagated(self):
        cfg = self._build_debate_config(llm_stagger_ms=1000)
        assert cfg.llm_stagger_ms == 1000

    def test_max_concurrent_llm_propagated(self):
        cfg = self._build_debate_config(max_concurrent_llm=3)
        assert cfg.max_concurrent_llm == 3

    def test_logging_mode_propagated(self):
        cfg = self._build_debate_config(logging_mode="debug")
        assert cfg.logging_mode == "debug"

    def test_pid_enabled_propagated(self):
        cfg = self._build_debate_config(pid_enabled=True)
        assert cfg.pid_enabled is True
        assert cfg.pid_config is not None

    def test_pid_gains_propagated(self):
        cfg = self._build_debate_config(
            pid_enabled=True, pid_kp=0.10, pid_ki=0.005, pid_kd=0.02
        )
        assert cfg.pid_config.gains.Kp == 0.10
        assert cfg.pid_config.gains.Ki == 0.005
        assert cfg.pid_config.gains.Kd == 0.02

    def test_pid_rho_star_propagated(self):
        cfg = self._build_debate_config(pid_enabled=True, pid_rho_star=0.9)
        assert cfg.pid_config.rho_star == 0.9

    def test_prompt_profile_propagated(self):
        cfg = self._build_debate_config(prompt_profile="minimal")
        assert cfg.prompt_profile == "minimal"

    def test_prompt_file_overrides_propagated(self):
        overrides = {"role_macro": "roles/macro_diverse.txt"}
        cfg = self._build_debate_config(prompt_file_overrides=overrides)
        assert cfg.prompt_file_overrides["role_macro"] == "roles/macro_diverse.txt"

    def test_sector_config_propagated(self):
        sector_cfg = {
            "sectors": {"tech": ["AAPL"], "fin": ["JPM"]},
            "sector_limits": {"tech": {"min": 0.1, "max": 0.5}},
            "agent_sector_permissions": None,
            "max_sector_weight": 0.40,
        }
        cfg = self._build_debate_config(sector_config=sector_cfg)
        assert cfg.sector_config is not None
        assert cfg.sector_config["max_sector_weight"] == 0.40
        assert cfg.sector_config["sectors"]["tech"] == ["AAPL"]

    def test_log_rendered_prompts_propagated(self):
        cfg = self._build_debate_config(log_rendered_prompts=True)
        assert cfg.log_rendered_prompts is True

    def test_log_prompt_manifest_propagated(self):
        cfg = self._build_debate_config(log_prompt_manifest=True)
        assert cfg.log_prompt_manifest is True

    def test_console_display_propagated(self):
        cfg = self._build_debate_config(console_display=False)
        assert cfg.console_display is False


class TestDebateConfigToDict:
    """Verify DebateConfig.to_dict() includes all critical fields."""

    def _build_to_dict(self, **agent_overrides) -> dict:
        from agents.multi_agent_debate import DebateAgentSystem

        agent_cfg = AgentConfig(**_base_agent(
            system_prompt_override="mock",
            **agent_overrides,
        ))
        system = DebateAgentSystem(agent_cfg)
        return system._debate_cfg.to_dict()

    def test_sector_config_in_dict(self):
        sector_cfg = {
            "sectors": {"tech": ["AAPL"]},
            "sector_limits": None,
            "agent_sector_permissions": None,
            "max_sector_weight": 0.35,
        }
        d = self._build_to_dict(sector_config=sector_cfg)
        assert "sector_config" in d
        assert d["sector_config"]["max_sector_weight"] == 0.35

    def test_sector_config_none_in_dict(self):
        d = self._build_to_dict()
        assert "sector_config" in d
        assert d["sector_config"] is None

    def test_logging_mode_in_dict(self):
        d = self._build_to_dict(logging_mode="standard")
        assert d["logging_mode"] == "standard"

    def test_parallel_settings_in_dict(self):
        d = self._build_to_dict(
            parallel_agents=False,
            llm_stagger_ms=1000,
            max_concurrent_llm=2,
        )
        assert d["parallel_agents"] is False
        assert d["llm_stagger_ms"] == 1000
        assert d["max_concurrent_llm"] == 2

    def test_prompt_settings_in_dict(self):
        d = self._build_to_dict(
            prompt_profile="diverse_agents",
            prompt_file_overrides={"role_macro": "roles/macro_diverse.txt"},
        )
        assert d["prompt_profile"] == "diverse_agents"
        assert d["prompt_file_overrides"]["role_macro"] == "roles/macro_diverse.txt"

    def test_pid_enabled_in_dict(self):
        d = self._build_to_dict(pid_enabled=True)
        assert d["pid_enabled"] is True

    def test_roles_serialized_as_strings(self):
        d = self._build_to_dict()
        assert isinstance(d["roles"], list)
        assert all(isinstance(r, str) for r in d["roles"])

    def test_all_critical_keys_present(self):
        """Every key that nodes.py or runner.py reads must be in to_dict()."""
        d = self._build_to_dict()
        required_keys = [
            "roles", "max_rounds", "model_name", "temperature",
            "parallel_agents", "mock", "logging_mode",
            "sector_config", "log_rendered_prompts", "log_prompt_manifest",
            "pid_enabled", "console_display", "prompt_profile",
            "prompt_file_overrides", "use_system_causal_contract",
            "llm_stagger_ms", "max_concurrent_llm", "no_rate_limit",
        ]
        for key in required_keys:
            assert key in d, f"Missing key in to_dict(): {key}"


# ═════════════════════════════════════════════════════════════════════════════
# 5. End-to-end: scenario -> SimulationConfig -> sector_config packing
# ═════════════════════════════════════════════════════════════════════════════


class TestScenarioSectorPropagation:
    """Verify sector config from a scenario flows all the way to DebateConfig."""

    def test_sector_scenario_reaches_debate_config(self):
        """Load a sector scenario, merge, build SimulationConfig, check sector_config."""
        if not _BASE_AGENTS_YAML.exists():
            pytest.skip("debate_diverse_agents.yaml not found")

        sector_scenario = _SCENARIOS_DIR / "sector_constrained.yaml"
        if not sector_scenario.exists():
            pytest.skip("sector_constrained.yaml not found")

        base_raw = _load_yaml(_BASE_AGENTS_YAML)
        scenario_raw = _load_yaml(sector_scenario)
        merged = _deep_merge(base_raw, scenario_raw)
        config = SimulationConfig(**merged)

        # sector_config should be packed into agent
        sc = config.agent.sector_config
        assert sc is not None
        assert "sectors" in sc
        assert "sector_limits" in sc
        assert len(sc["sectors"]) > 0

        # Now push through the adapter
        from agents.multi_agent_debate import DebateAgentSystem

        system = DebateAgentSystem(config.agent)
        d = system._debate_cfg.to_dict()
        assert d["sector_config"] is not None
        assert d["sector_config"]["sectors"] == sc["sectors"]

    def test_non_sector_scenario_has_null_sector_config(self):
        """A scenario without sectors should have sector_config=None."""
        if not _BASE_AGENTS_YAML.exists():
            pytest.skip("debate_diverse_agents.yaml not found")

        simple_scenario = _SCENARIOS_DIR / "broad_14ticker.yaml"
        if not simple_scenario.exists():
            pytest.skip("broad_14ticker.yaml not found")

        base_raw = _load_yaml(_BASE_AGENTS_YAML)
        scenario_raw = _load_yaml(simple_scenario)
        merged = _deep_merge(base_raw, scenario_raw)
        config = SimulationConfig(**merged)

        assert config.agent.sector_config is None
