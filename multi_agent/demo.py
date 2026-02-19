#!/usr/bin/env python3
"""
Demo: Run the multi-agent debate system on sample observations.

Shows different configurations for ablation experiments:
  1. Default 4-agent debate (macro, value, risk, technical)
  2. With sentiment agent added
  3. With adversarial/devil's advocate
  4. High vs low agreeableness
  5. Multiple debate rounds
  6. No pipeline preprocessing

Usage:
    # Mock mode (no API key needed):
    python -m multi_agent.demo

    # Live mode (requires OPENAI_API_KEY):
    python -m multi_agent.demo --live

    # Run a single config in live mode:
    python -m multi_agent.demo --live --config 1

    # Run specific configs:
    python -m multi_agent.demo --live --config 1,6,8

    # Verbose mode (show full debate content):
    python -m multi_agent.demo --verbose
    python -m multi_agent.demo --live --verbose --config 1
"""

from __future__ import annotations

import sys
import time

from multi_agent.config import AgentRole, DebateConfig
from multi_agent.models import (
    Constraints,
    MarketState,
    Observation,
    PortfolioState,
)
from multi_agent.runner import MultiAgentRunner


# =============================================================================
# SAMPLE DATA GENERATOR
# =============================================================================


def generate_sample_observations() -> list[Observation]:
    """Generate 3 diverse observations for demo purposes."""
    return [
        # Observation 1: Bullish environment
        Observation(
            timestamp="2025-03-15T10:00:00Z",
            universe=["AAPL", "GOOGL", "MSFT"],
            market_state=MarketState(
                prices={"AAPL": 185.50, "GOOGL": 142.30, "MSFT": 390.00},
                returns={"AAPL": 0.025, "GOOGL": 0.012, "MSFT": 0.018},
                volatility={"AAPL": 0.22, "GOOGL": 0.25, "MSFT": 0.18},
            ),
            text_context=(
                "Fed signals potential rate cuts in Q2. AAPL earnings beat expectations "
                "by 12%. MSFT cloud revenue accelerating. Market breadth improving."
            ),
            portfolio_state=PortfolioState(
                cash=50000.0,
                positions={"AAPL": 100, "GOOGL": 0, "MSFT": 50},
            ),
            constraints=Constraints(max_leverage=2.0, max_position_size=500),
        ),
        # Observation 2: Mixed signals
        Observation(
            timestamp="2025-03-15T11:00:00Z",
            universe=["AAPL", "GOOGL", "MSFT"],
            market_state=MarketState(
                prices={"AAPL": 183.20, "GOOGL": 143.50, "MSFT": 387.00},
                returns={"AAPL": -0.012, "GOOGL": 0.008, "MSFT": -0.008},
                volatility={"AAPL": 0.28, "GOOGL": 0.24, "MSFT": 0.22},
            ),
            text_context=(
                "Inflation data comes in hotter than expected. Tech sector under pressure. "
                "However, GOOGL announces major AI partnership. Mixed signals."
            ),
            portfolio_state=PortfolioState(
                cash=45000.0,
                positions={"AAPL": 110, "GOOGL": 20, "MSFT": 50},
            ),
            constraints=Constraints(max_leverage=2.0, max_position_size=500),
        ),
        # Observation 3: Risk-off
        Observation(
            timestamp="2025-03-15T12:00:00Z",
            universe=["AAPL", "GOOGL", "MSFT"],
            market_state=MarketState(
                prices={"AAPL": 178.90, "GOOGL": 138.20, "MSFT": 380.50},
                returns={"AAPL": -0.023, "GOOGL": -0.037, "MSFT": -0.017},
                volatility={"AAPL": 0.35, "GOOGL": 0.38, "MSFT": 0.30},
            ),
            text_context=(
                "Geopolitical tensions escalate. VIX spikes to 28. Broad market selloff. "
                "Flight to safety underway. Treasury yields dropping."
            ),
            portfolio_state=PortfolioState(
                cash=40000.0,
                positions={"AAPL": 110, "GOOGL": 20, "MSFT": 50},
            ),
            constraints=Constraints(max_leverage=1.5, max_position_size=300),
        ),
    ]


# =============================================================================
# DEMO RUNNER
# =============================================================================


def run_demo_config(
    name: str,
    config: DebateConfig,
    observation: Observation,
) -> None:
    """Run a single demo configuration and print results."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  Roles: {[r.value for r in config.roles]}")
    print(f"  Rounds: {config.max_rounds} | Agreeableness: {config.agreeableness}")
    print(f"  Pipeline: news={config.enable_news_pipeline}, data={config.enable_data_pipeline}")
    print(f"  Adversarial: {config.enable_adversarial}")
    print(f"{'='*70}")

    runner = MultiAgentRunner(config)
    action, trace = runner.run(observation)

    print(f"\n  Decision: {trace.decision}")
    print(f"  Confidence: {action.confidence:.2f}")
    print(f"  Orders: {[f'{o.side} {o.size} {o.ticker}' for o in action.orders] or 'HOLD'}")
    print(f"  # Claims: {len(action.claims)}")
    for claim in action.claims[:3]:
        print(f"    [{claim.pearl_level.value}] {claim.claim_text[:80]}")
    if trace.strongest_objection:
        print(f"  Strongest objection: {trace.strongest_objection[:100]}")
    print(f"  Justification: {action.justification[:120]}...")


def _parse_config_arg() -> set[int] | None:
    """Parse --config N or --config 1,3,8 from sys.argv. Returns None = run all."""
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            return {int(x) for x in sys.argv[i + 1].split(",")}
    return None


def main():
    live = "--live" in sys.argv
    verbose = "--verbose" in sys.argv
    mock = not live
    selected = _parse_config_arg()

    if live and not __import__("os").environ.get("OPENAI_API_KEY"):
        print("ERROR: --live mode requires OPENAI_API_KEY environment variable")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  CS372 Multi-Agent Debate System — Demo")
    print(f"  Mode: {'LIVE (OpenAI API)' if live else 'MOCK (no API calls)'}")
    if verbose:
        print("  Verbose: ON (showing full debate content)")
    if selected:
        print(f"  Running configs: {sorted(selected)}")
    print("=" * 70)

    observations = generate_sample_observations()
    obs = observations[0]  # Use the bullish scenario for most configs
    obs_risk = observations[2]

    trace_dir = "/tmp/cs372_demo_traces"

    # All configs in order
    configs: list[tuple[int, str, DebateConfig, Observation]] = [
        (1, "Config 1: Default 4-Agent Debate", DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK, AgentRole.TECHNICAL],
            mock=mock, verbose=verbose, trace_dir=trace_dir,
        ), obs),
        (2, "Config 2: 5 Agents (+ Sentiment)", DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK, AgentRole.TECHNICAL, AgentRole.SENTIMENT],
            mock=mock, verbose=verbose, trace_dir=trace_dir,
        ), obs),
        (3, "Config 3: Adversarial Mode (Devil's Advocate)", DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
            enable_adversarial=True, mock=mock, verbose=verbose, trace_dir=trace_dir,
        ), obs),
        (4, "Config 4: High Agreeableness (0.9 — sycophantic)", DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
            agreeableness=0.9, mock=mock, verbose=verbose, trace_dir=trace_dir,
        ), obs),
        (5, "Config 5: Low Agreeableness (0.1 — confrontational)", DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
            agreeableness=0.1, mock=mock, verbose=verbose, trace_dir=trace_dir,
        ), obs),
        (6, "Config 6: 3 Debate Rounds", DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
            max_rounds=3, mock=mock, verbose=verbose, trace_dir=trace_dir,
        ), obs),
        (7, "Config 7: No Pipeline Preprocessing", DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK],
            enable_news_pipeline=False, enable_data_pipeline=False,
            mock=mock, verbose=verbose, trace_dir=trace_dir,
        ), obs),
        (8, "Config 8: Full System on Risk-Off Scenario", DebateConfig(
            roles=[AgentRole.MACRO, AgentRole.VALUE, AgentRole.RISK, AgentRole.TECHNICAL, AgentRole.SENTIMENT],
            enable_adversarial=True, max_rounds=2, agreeableness=0.3,
            mock=mock, verbose=verbose, trace_dir=trace_dir,
        ), obs_risk),
    ]

    # Filter to selected configs
    to_run = [(n, name, cfg, o) for n, name, cfg, o in configs if selected is None or n in selected]

    for i, (num, name, cfg, observation) in enumerate(to_run):
        if num == 8 and (selected is None or len(to_run) > 1):
            print("\n\n--- Switching to Risk-Off Scenario ---")

        run_demo_config(name, cfg, observation)

        # In live mode, wait 5s between configs to avoid rate limits
        if live and i < len(to_run) - 1:
            print("\n  [Waiting 5s before next config to avoid rate limits...]", flush=True)
            time.sleep(5)

    print(f"\n{'='*70}")
    print(f"  Demo complete! Traces saved to: {trace_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
