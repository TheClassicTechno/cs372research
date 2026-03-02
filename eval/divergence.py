"""
Portfolio divergence analyzer.
Calculates Jensen-Shannon divergence, Generalized Active Share, and Generalized Cosine Similarity 
between agent proposals to measure debate convergence.

Usage:
    python -m multi_agent.divergence /path/to/trace.json
"""

import math
import json
import sys
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class PortfolioState:
    agent_id: str
    orders: List[Dict[str, Any]]
    implied_positions: Dict[str, float]
    cash: float
    has_shorts: bool
    total_value: float
    distribution: Dict[str, float]

    def __str__(self):
        order_str = ", ".join([f"{x['side']} {x['size']} {x['ticker']}" for x in self.orders])
        return f"""Agent {self.agent_id}, total_value=${self.total_value:.2f}
    Orders: {order_str}
    Implied Positions: {self.implied_positions} | Cash: ${self.cash:.2f} | Has shorts: {self.has_shorts}
    Distribution: {", ".join([f"{k}: {v*100:.1f}%" for k, v in self.distribution.items() if abs(v) > 0.001])}"""

def calculate_weights(positions: Dict[str, float], cash: float, prices: Dict[str, float], total_value: float) -> Dict[str, float]:
    """
    Calculate portfolio weights relative to Net Total Value.
    Sum of weights will be approximately 1.0.
    Weights can be negative if shorts exist.
    """
    weights = {"CASH": cash / total_value}
    for ticker, qty in positions.items():
        if ticker not in prices:
            raise ValueError(f"Expected price for position: {ticker}")
        val = qty * float(prices[ticker])
        weights[ticker] = val / total_value
            
    return weights

def get_portfolio_state(
    agent_id: str,
    initial_positions: Dict[str, float], 
    initial_cash: float, 
    prices: Dict[str, float], 
    orders: List[Dict[str, Any]]
) -> PortfolioState:
    """
    Apply orders to initial portfolio.
    Returns (positions, cash, has_shorts, total_value).
    """
    positions = initial_positions.copy()
    cash = initial_cash

    for order in orders:
        ticker = order["ticker"]
        side = order["side"].lower()
        size = float(order["size"])
        
        if ticker not in prices:
            raise ValueError(f"Price missing for traded ticker: {ticker}")
        price = float(prices[ticker])
        if price <= 0:
            raise ValueError(f"Expected positive price: {ticker}")
        if size <= 0:
            raise ValueError(f"Expected positive order size: {ticker}")

        if side == "buy":
            cash -= size * price
            positions[ticker] = positions.get(ticker, 0.0) + size
        elif side == "sell":
            cash += size * price
            positions[ticker] = positions.get(ticker, 0.0) - size

    # Assert solvency (Cash must be non-negative)
    assert cash >= -1e-5, f"Cash cannot be negative: {cash:.2f}"
    
    # Calculate Total Value
    total_value = cash
    for ticker, qty in positions.items():
        if ticker not in prices:
            raise ValueError(f"Expected price for position: {ticker}")
        total_value += qty * float(prices[ticker])
        
    assert total_value > 0, f"Total portfolio value must be positive: {total_value:.2f}"

    # Check for short positions
    has_shorts = False
    for qty in positions.values():
        if qty < -1e-5:
            has_shorts = True
            break

    dist = calculate_weights(positions, cash, prices, total_value)
                
    return PortfolioState(agent_id=agent_id,
                          orders=orders, 
                          implied_positions=positions,
                          cash=cash, 
                          has_shorts=has_shorts, 
                          total_value=total_value, 
                          distribution=dist)

def get_mean_portfolio(portfolios: List[Dict[str, float]]) -> Dict[str, float]:
    # Note this is assumed to return the union of all keys across portfolios, even if some positions become zero.
    all_keys = set()
    for p in portfolios: all_keys.update(p.keys())
    mean = {k: 0.0 for k in all_keys}
    for p in portfolios:
        for k, v in p.items():
            mean[k] += v / len(portfolios)
    return mean

def kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Compute Kullback-Leibler divergence KL(P || Q)."""
    div = 0.0
    epsilon = 1e-10
    for k, p_val in p.items():
        if p_val > 0:
            q_val = q.get(k, 0.0)
            q_val = max(q_val, epsilon)
            div += p_val * math.log2(p_val / q_val)
    return div

def generalized_js_divergence(distributions: List[Dict[str, float]], consensus_portfolio: Dict[str, float] = None) -> float:
    """
    Compute Generalized Jensen-Shannon Divergence.
    Expects non-negative distributions.
    If consensus_portfolio is provided, it is used as the reference distribution M.
    """
    if not distributions: return 0.0
    n = len(distributions)
    # If consensus provided, we can compute divergence for even a single agent against it
    assert n > 1 or consensus_portfolio is not None, "At least 2 distributions or a consensus portfolio must be provided for JS Divergence."

    # Compute mixture M if not provided
    m = consensus_portfolio if consensus_portfolio else get_mean_portfolio(distributions)

    # Compute average KL divergence to M
    jsd = 0.0
    for dist in distributions:
        jsd += kl_divergence(dist, m) / n

    return jsd

def generalized_active_share(portfolios: List[Dict[str, float]], consensus_portfolio: Dict[str, float] = None) -> float:
    """
    Generalized Active Share (L1) = Average L1 distance from the Consensus Portfolio.
    Consensus = Mean of all portfolios, or provided explicitly.
    Returns average Active Share (0.0 to 1.0).
    """
    if not portfolios: return 0.0
    n = len(portfolios)
    assert n > 1 or consensus_portfolio is not None, "At least 2 portfolios or a consensus portfolio must be provided for Active Share Divergence."
    
    # 1. Calculate Consensus Portfolio if not provided
    consensus = consensus_portfolio if consensus_portfolio else get_mean_portfolio(portfolios)
            
    # 2. Calculate Active Share for each agent
    total_active_share = 0.0
    for p in portfolios:
        as_val = 0.0
        for k in consensus.keys():
            w_i = p.get(k, 0.0)
            c_i = consensus[k]
            as_val += abs(w_i - c_i)
        
        # Standard Active Share is 0.5 * L1 distance
        total_active_share += 0.5 * as_val
        
    return total_active_share / n

@dataclass
class DivergenceResult:
    """Structured result from divergence analysis."""

    js_divergence_by_round: Dict[int, float]
    active_share_by_round: Dict[int, float]
    judge_active_share: float | None = None
    judge_js_divergence: float | None = None


def analyze_divergence(data: dict) -> DivergenceResult:
    """Compute divergence metrics from an in-memory trace dict.

    This is the core analysis logic used by both the CLI entry point
    and the ``EvalPipeline``.
    """
    trace = data.get("trace")
    if trace is None:
        raise ValueError("JSON missing required 'trace' key")
    debate_turns = data.get("debate_turns")
    if debate_turns is None:
        raise ValueError("JSON missing required 'debate_turns' key")

    market_state = trace.get("initial_market_state")
    portfolio_state = trace.get("initial_portfolio_state")
    if market_state is None:
        raise ValueError("Trace missing 'initial_market_state'")
    if portfolio_state is None:
        raise ValueError("Trace missing 'initial_portfolio_state'")

    prices = market_state["prices"]
    initial_positions = portfolio_state["positions"]
    initial_cash = portfolio_state["cash"]

    proposals_by_round: Dict[int, List[Tuple[str, list]]] = defaultdict(list)
    for turn in debate_turns:
        turn_type = turn.get("type", "")
        agent_id = turn.get("agent_id", "")
        if agent_id == "judge" or turn_type == "judge_decision":
            continue
        if turn_type in ("proposal", "revision"):
            content = turn.get("content", {})
            orders = content.get("orders")
            if orders is None:
                continue
            round_num = turn.get("round", 0)
            proposals_by_round[round_num].append((agent_id, orders))

    js_by_round: Dict[int, float] = {}
    as_by_round: Dict[int, float] = {}

    for round_num in sorted(proposals_by_round.keys()):
        agent_orders = proposals_by_round[round_num]
        states = []
        any_shorts = False
        for agent_id, orders in agent_orders:
            try:
                state = get_portfolio_state(agent_id, initial_positions, initial_cash, prices, orders)
                states.append(state)
                if state.has_shorts:
                    any_shorts = True
            except (AssertionError, ValueError):
                continue

        if len(states) < 2:
            continue

        dists = [s.distribution for s in states]
        as_by_round[round_num] = generalized_active_share(dists)
        if not any_shorts:
            js_by_round[round_num] = generalized_js_divergence(dists)

    judge_as: float | None = None
    judge_js: float | None = None

    judge_turns = [t for t in debate_turns if t.get("type") == "judge_decision"]
    if len(judge_turns) == 1 and proposals_by_round:
        last_round = max(proposals_by_round.keys())
        agent_states = []
        for agent_id, orders in proposals_by_round[last_round]:
            try:
                agent_states.append(
                    get_portfolio_state(agent_id, initial_positions, initial_cash, prices, orders)
                )
            except (AssertionError, ValueError):
                pass

        if agent_states:
            mean_agent = get_mean_portfolio([s.distribution for s in agent_states])
            judge_content = judge_turns[0].get("content", {})
            judge_orders = judge_content.get("orders")
            if judge_orders is not None:
                try:
                    judge_state = get_portfolio_state("judge", initial_positions, initial_cash, prices, judge_orders)
                    judge_as = generalized_active_share([judge_state.distribution], consensus_portfolio=mean_agent)
                    if not judge_state.has_shorts and not any(w < -1e-5 for w in mean_agent.values()):
                        judge_js = generalized_js_divergence([judge_state.distribution], consensus_portfolio=mean_agent)
                except (AssertionError, ValueError):
                    pass

    return DivergenceResult(
        js_divergence_by_round=js_by_round,
        active_share_by_round=as_by_round,
        judge_active_share=judge_as,
        judge_js_divergence=judge_js,
    )


class DebateDivergenceAnalyzer:
    def __init__(self, trace_file: str):
        with open(trace_file, 'r') as f:
            self.data = json.load(f)

    def analyze(self):
        """Run divergence analysis and print results to stdout."""
        trace = self.data.get("trace", {})
        market_state = trace.get("initial_market_state", {})
        portfolio_state = trace.get("initial_portfolio_state", {})
        print(f"Initial portfolio: Positions={portfolio_state.get('positions', {})}, Cash=${portfolio_state.get('cash', 0):.2f}")
        print(f"Initial market prices: {market_state.get('prices', {})}")

        result = analyze_divergence(self.data)

        print(f"\n--- Divergence Analysis ---\n")
        for round_num in sorted(set(result.js_divergence_by_round) | set(result.active_share_by_round)):
            stage = "Proposals" if round_num == 0 else f"Revisions (Round {round_num})"
            print(f"==== Round {round_num} {stage} ====")

            if round_num in result.active_share_by_round:
                print(f"  Active Share Divergence (L1):  {result.active_share_by_round[round_num]:.4f}")
            if round_num in result.js_divergence_by_round:
                print(f"  JS Divergence:      {result.js_divergence_by_round[round_num]:.4f} bits")
            else:
                print("  [Shorts Detected -> Skipping JS Divergence]")
            print()

        if result.judge_active_share is not None:
            print("==== Judge vs. Final round mean ====")
            print(f"  Judge Active Share vs Mean: {result.judge_active_share:.4f}")
            if result.judge_js_divergence is not None:
                print(f"  Judge JS Divergence vs Mean: {result.judge_js_divergence:.4f} bits")
            else:
                print("  [Judge has shorts -> Skipping JS Divergence]")
        elif not result.js_divergence_by_round and not result.active_share_by_round:
            print("No judge decision found in debate turns.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m multi_agent.divergence <trace_file.json>")
        sys.exit(1)
        
    trace_file = sys.argv[1]
    
    try:
        analyzer = DebateDivergenceAnalyzer(trace_file)
        analyzer.analyze()
    except FileNotFoundError:
        print(f"File not found: {trace_file}")
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {trace_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
