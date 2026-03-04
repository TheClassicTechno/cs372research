import unittest
import unittest.mock
import math
import io
import json
from eval.divergence import (
    generalized_active_share, 
    generalized_js_divergence
)

class TestDivergenceMetrics(unittest.TestCase):

    def test_no_shorts_simple(self):
        """
        3 Agents, Long Only.
        P1: {A: 1.0}
        P2: {A: 0.5, B: 0.5}
        P3: {B: 1.0}
        """
        portfolios = [
            {"A": 1.0, "B": 0.0},
            {"A": 0.5, "B": 0.5},
            {"A": 0.0, "B": 1.0}
        ]
        
        # 1. Consensus (Mean)
        # A: (1 + 0.5 + 0) / 3 = 0.5
        # B: (0 + 0.5 + 1) / 3 = 0.5
        # Consensus: {A: 0.5, B: 0.5}
        
        # 2. Active Share Calculation
        # AS(P1): 0.5 * (|1-0.5| + |0-0.5|) = 0.5 * (0.5 + 0.5) = 0.5
        # AS(P2): 0.5 * (|0.5-0.5| + |0.5-0.5|) = 0.0
        # AS(P3): 0.5 * (|0-0.5| + |1-0.5|) = 0.5 * (0.5 + 0.5) = 0.5
        # Average AS: (0.5 + 0 + 0.5) / 3 = 1/3 = 0.3333...
        
        l1_score = generalized_active_share(portfolios)
        self.assertAlmostEqual(l1_score, 1.0/3.0, places=4)
        
        # 3. JS Divergence
        # Mixture M: {A: 0.5, B: 0.5}
        # KL(P1 || M): 1*log2(1/0.5) = 1
        # KL(P2 || M): 0.5*log2(1) + 0.5*log2(1) = 0
        # KL(P3 || M): 1*log2(1/0.5) = 1
        # Avg KL: (1 + 0 + 1) / 3 = 2/3
        js_score = generalized_js_divergence(portfolios)
        self.assertAlmostEqual(js_score, 2.0/3.0, places=4)

    def test_shorts_present(self):
        """
        3 Agents, Shorts Present.
        P1: {A: 0.0, B: 0.0} (Neutral)
        P2: {A: 0.5, B: -0.5} (Long/Short)
        P3: {A: -0.5, B: 0.5} (Short/Long)
        """
        portfolios = [
            {"A": 0.0, "B": 0.0},
            {"A": 0.5, "B": -0.5},
            {"A": -0.5, "B": 0.5}
        ]
        
        # 1. Consensus
        # A: 0
        # B: 0
        
        # 2. Active Share
        # AS(P1): 0
        # AS(P2): 0.5 * (|0.5-0| + |-0.5-0|) = 0.5 * 1 = 0.5
        # AS(P3): 0.5 * (|-0.5-0| + |0.5-0|) = 0.5
        # Avg AS: 1/3
        
        l1_score = generalized_active_share(portfolios)
        self.assertAlmostEqual(l1_score, 1.0/3.0, places=4)

    def test_identical_portfolios(self):
        """
        All portfolios identical. Divergence should be 0.
        """
        portfolios = [
            {"A": 0.5, "B": 0.5},
            {"A": 0.5, "B": 0.5},
            {"A": 0.5, "B": 0.5}
        ]
        
        l1_score = generalized_active_share(portfolios)
        self.assertAlmostEqual(l1_score, 0.0, places=4)
        
        js_score = generalized_js_divergence(portfolios)
        self.assertAlmostEqual(js_score, 0.0, places=4)

    def test_explicit_consensus(self):
        """
        Test providing an explicit consensus portfolio (e.g. Judge vs Mean).
        Judge: {A: 1.0}
        Mean: {A: 0.5, B: 0.5}
        """
        judge = [{"A": 1.0, "B": 0.0}]
        consensus = {"A": 0.5, "B": 0.5}
        
        # Active Share
        # AS(Judge): 0.5 * (|1-0.5| + |0-0.5|) = 0.5 * 1 = 0.5
        l1_score = generalized_active_share(judge, consensus_portfolio=consensus)
        self.assertAlmostEqual(l1_score, 0.5, places=4)
        
        # JS Divergence
        # KL(Judge || Consensus): 1*log2(1/0.5) = 1
        js_score = generalized_js_divergence(judge, consensus_portfolio=consensus)
        self.assertAlmostEqual(js_score, 1.0, places=4)

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    @unittest.mock.patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='{}')
    @unittest.mock.patch('json.load')
    def test_analyzer_integration(self, mock_json_load, mock_file, mock_stdout):
        from eval.divergence import DebateDivergenceAnalyzer
        
        # Mock trace data
        mock_data = {
            "trace": {
                "initial_market_state": {
                    "prices": {"A": 100.0, "B": 50.0}
                },
                "initial_portfolio_state": {
                    "cash": 10000.0,
                    "positions": {"A": 10.0}
                }
            },
            "debate_turns": [
                {
                    "type": "proposal",
                    "round": 0,
                    "agent_id": "agent1",
                    "content": {
                        "orders": [
                            {"ticker": "A", "side": "buy", "size": 10.0}
                        ]
                    }
                },
                {
                    "type": "proposal",
                    "round": 0,
                    "agent_id": "agent2",
                    "content": {
                        "orders": [
                            {"ticker": "B", "side": "buy", "size": 20.0}
                        ]
                    }
                },
                {
                    "type": "judge_decision",
                    "round": 0,
                    "agent_id": "judge",
                    "content": {
                        "orders": [
                            {"ticker": "A", "side": "buy", "size": 15.0} # Compromise
                        ]
                    }
                }
            ]
        }
        
        mock_json_load.return_value = mock_data
        
        analyzer = DebateDivergenceAnalyzer("dummy.json")
        analyzer.analyze()
        
        output = mock_stdout.getvalue()
        
        self.assertIn("--- Divergence Analysis ---", output)
        self.assertIn("Active Share Divergence (L1)", output)
        self.assertIn("Judge JS Divergence vs Mean", output)
        self.assertIn("Judge Active Share vs Mean", output)
