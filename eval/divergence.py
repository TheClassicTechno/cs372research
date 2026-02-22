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

class DebateDivergenceAnalyzer:
    def __init__(self, trace_file: str):
        with open(trace_file, 'r') as f:
            self.data = json.load(f)

        if "trace" not in self.data:
            raise ValueError("JSON missing required 'trace' key")
        self.trace = self.data["trace"]
        
        if "debate_turns" not in self.data:
            raise ValueError("JSON missing required 'debate_turns' key")
        self.debate_turns = self.data["debate_turns"]
        
        if "initial_market_state" not in self.trace:
            raise ValueError("Trace missing 'initial_market_state'")
        self.initial_market_state = self.trace["initial_market_state"]
        
        if "initial_portfolio_state" not in self.trace:
            raise ValueError("Trace missing 'initial_portfolio_state'")
        self.initial_portfolio_state = self.trace["initial_portfolio_state"]

    def analyze(self):
        prices = self.initial_market_state["prices"]
        initial_positions = self.initial_portfolio_state["positions"]
        initial_cash = self.initial_portfolio_state["cash"]

        print(f"Initial portfolio: Positions={initial_positions}, Cash=${initial_cash:.2f}")
        print(f"Initial market prices: {prices}")

        # Separate proposals by round
        proposals_by_round = defaultdict(list)

        for turn in self.debate_turns:
            turn_type = turn["type"]
            round_num = turn["round"]
            agent_id = turn["agent_id"]
            
            if agent_id == "judge" or turn_type == "judge_decision":
                continue

            if turn_type in ["proposal", "revision"]:
                content = turn["content"]
                if "orders" not in content:
                    print(f"Warning: 'orders' missing in {turn_type} turn for {agent_id}. Skipping.")
                    continue
                    
                orders = content["orders"]
                
                if round_num not in proposals_by_round:
                    proposals_by_round[round_num] = []
                    
                proposals_by_round[round_num].append((agent_id, orders))

        print(f"\n--- Divergence Analysis ---\n")
        for round_num in sorted(proposals_by_round.keys()):
            stage = "Proposals" if round_num == 0 else f"Revisions (Round {round_num})"
            print(f"==== Round {round_num} {stage} ====")
            agent_orders = proposals_by_round[round_num]
            if not agent_orders: continue

            # Calculate portfolio states for each agent 
            states = []
            any_shorts = False
            for agent_id, orders in agent_orders:
                try:
                    state = get_portfolio_state(agent_id, initial_positions, initial_cash, prices, orders)
                    print(state)
                    states.append(state)
                    if state.has_shorts: any_shorts = True
                except (AssertionError, ValueError) as e:
                    print(f"Skipping {agent_id} due to invalid state: {e}")
            
            if len(states) < 2:
                print(f"Round {round_num} - Not enough valid portfolios to compare.")
                continue

            # Extract distributions for metrics
            dists = [s.distribution for s in states]

            # Compute Metrics
            l1_score = generalized_active_share(dists)
            
            
            if any_shorts:
                print("  [Shorts Detected -> Skipping JS Divergence]")
            else:
                js_score = generalized_js_divergence(dists)
                print(f"  JS Divergence:      {js_score:.4f} bits")
            
            print(f"  Active Share Divergence (L1):  {l1_score:.4f}")
            
            print()

        # --- Judge Analysis ---
        judge_turns = [t for t in self.debate_turns if t["type"] == "judge_decision"]
        if not judge_turns:
            print("No judge decision found in debate turns.")
            return
        assert len(judge_turns) == 1, "Expected at most one judge decision turn."
        judge_turn = judge_turns[0]
        if judge_turn and proposals_by_round:
            print("==== Judge vs. Final round mean ====")
            # Get last round's agent portfolios
            last_round = max(proposals_by_round.keys())
            last_agent_orders = proposals_by_round[last_round]
            
            agent_states = []
            for agent_id, orders in last_agent_orders:
                try:
                    s = get_portfolio_state(agent_id, initial_positions, initial_cash, prices, orders)
                    agent_states.append(s)
                except: pass
            
            if not agent_states:
                print("No valid agent portfolios in final round.")
                return

            mean_agent_portfolio = get_mean_portfolio([s.distribution for s in agent_states])
            
            # Get judge portfolio
            judge_content = judge_turn["content"]
            if "orders" in judge_content:
                try:
                    judge_state = get_portfolio_state("judge", initial_positions, initial_cash, prices, judge_content["orders"])
                    print(judge_state)
                    
                    judge_dist = judge_state.distribution
                    
                    # Compare Judge to Mean Agent Portfolio
                    # We treat the Judge as a single "agent" comparing against the Consensus
                    
                    judge_l1 = generalized_active_share([judge_dist], consensus_portfolio=mean_agent_portfolio)
                    print(f"  Judge Active Share vs Mean: {judge_l1:.4f}")

                    if not judge_state.has_shorts and not any(w < -1e-5 for w in mean_agent_portfolio.values()):
                         judge_js = generalized_js_divergence([judge_dist], consensus_portfolio=mean_agent_portfolio)
                         print(f"  Judge JS Divergence vs Mean: {judge_js:.4f} bits")
                    else:
                        print("  [Judge has shorts -> Skipping JS Divergence]")

                except Exception as e:
                    print(f"Failed to process judge portfolio: {e}")
            else:
                print("Judge decision missing 'orders'.")


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
