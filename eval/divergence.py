"""
Portfolio divergence analyzer.
Calculates Jensen-Shannon divergence between agent proposals to measure debate convergence.

Usage:
    python -m multi_agent.divergence /path/to/trace.json
"""

import math
import json
import sys
from typing import Dict, List, Any

def get_portfolio_distribution(
    initial_positions: Dict[str, float], 
    initial_cash: float, 
    prices: Dict[str, float], 
    orders: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Apply orders to initial portfolio and return the resulting value distribution.
    Asserts that no shorting occurs (cash >= 0 and positions >= 0).
    """
    positions = initial_positions.copy()
    cash = initial_cash

    for order in orders:
        ticker = order.get("ticker")
        side = order.get("side", "").lower()
        size = float(order.get("size", 0.0))
        price = float(prices.get(ticker, 0.0))

        if not ticker or price <= 0 or size <= 0:
            continue

        if side == "buy":
            cash -= size * price
            positions[ticker] = positions.get(ticker, 0.0) + size
        elif side == "sell":
            cash += size * price
            positions[ticker] = positions.get(ticker, 0.0) - size

    assert cash >= -1e-5, f"Cash cannot be negative: {cash}" # -1e-5 for float precision
    for ticker, qty in positions.items():
        assert qty >= -1e-5, f"Position for {ticker} cannot be negative: {qty}"

    # Set precisely to 0 if very close due to float math
    cash = max(0.0, cash)
    for ticker in positions:
        positions[ticker] = max(0.0, positions[ticker])

    total_value = cash
    for ticker, qty in positions.items():
        total_value += qty * float(prices.get(ticker, 0.0))

    assert total_value > 0, "Total portfolio value must be positive."

    distribution = {}
    if cash > 0:
        distribution["CASH"] = cash / total_value
    for ticker, qty in positions.items():
        if qty > 0:
            distribution[ticker] = (qty * float(prices.get(ticker, 0.0))) / total_value

    return distribution

def kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Compute Kullback-Leibler divergence KL(P || Q)."""
    div = 0.0
    for k, p_val in p.items():
        if p_val > 0:
            q_val = q.get(k, 0.0)
            assert q_val > 0, f"Q must be > 0 where P > 0 for KL divergence. Missing key: {k}"
            div += p_val * math.log2(p_val / q_val)
    return div

def generalized_js_divergence(distributions: List[Dict[str, float]]) -> float:
    """
    Compute Generalized Jensen-Shannon Divergence for a list of distributions.
    Uniform weights (1/N) are used.
    """
    if not distributions:
        return 0.0
    n = len(distributions)
    if n == 1:
        return 0.0

    # Compute mixture M
    m = {}
    for dist in distributions:
        for k, v in dist.items():
            m[k] = m.get(k, 0.0) + v / n

    # Compute average KL divergence to M
    jsd = 0.0
    for dist in distributions:
        jsd += kl_divergence(dist, m) / n

    return jsd

class DebateDivergenceAnalyzer:
    def __init__(self, trace_file: str):
        with open(trace_file, 'r') as f:
            self.data = json.load(f)

        self.trace = self.data.get("trace", {})
        self.debate_turns = self.data.get("debate_turns", [])
        
        self.initial_market_state = self.trace.get("initial_market_state", {})
        self.initial_portfolio_state = self.trace.get("initial_portfolio_state", {})

    def analyze(self):
        if not self.initial_market_state or not self.initial_portfolio_state:
            print("Missing initial_market_state or initial_portfolio_state in trace.")
            return

        prices = self.initial_market_state.get("prices", {})
        initial_positions = self.initial_portfolio_state.get("positions", {})
        initial_cash = self.initial_portfolio_state.get("cash", 0.0)

        # Separate proposals by round
        # We assume round 0 is initial proposals, round 1+ are revisions
        proposals_by_round = {}

        for turn in self.debate_turns:
            turn_type = turn.get("type")
            round_num = turn.get("round", 0)
            agent_id = turn.get("agent_id", "")
            
            # Skip judge
            if agent_id == "judge" or turn_type == "judge_decision":
                continue

            if turn_type in ["proposal", "revision"]:
                content = turn.get("content", {})
                orders = content.get("orders", [])
                
                if round_num not in proposals_by_round:
                    proposals_by_round[round_num] = []
                    
                proposals_by_round[round_num].append((agent_id, orders))

        def get_distributions(agent_orders_list):
            dists = []
            for agent_id, orders in agent_orders_list:
                try:
                    dist = get_portfolio_distribution(initial_positions, initial_cash, prices, orders)
                    dists.append(dist)
                except AssertionError as e:
                    print(f"Warning: Skipped orders for {agent_id} due to assertion: {e}")
            return dists
        
        def fmt_order(orders):
            return [f"{x['side']} {x['size']} {x['ticker']}" for x in orders]

        print(f"--- Divergence Analysis ---")
        for round_num in sorted(proposals_by_round.keys()):
            agent_orders = proposals_by_round[round_num]
            dists = get_distributions(agent_orders)
            
            if len(dists) > 1:
                jsd = generalized_js_divergence(dists)
                stage = "Proposals" if round_num == 0 else f"Revisions (Round {round_num})"
                print(f"Round {round_num} {stage} - JS Divergence: {jsd:.4f} bits")
                
                for agent_order, dist in zip(agent_orders, dists):
                    agent_id, agent_order = agent_order
                    alloc_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in dist.items()])
                    print(f"  {agent_id}: {fmt_order(agent_order)} -> {alloc_str}")
                print()
            else:
                print(f"Round {round_num} - Not enough valid distributions to compare.")


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

if __name__ == "__main__":
    main()
