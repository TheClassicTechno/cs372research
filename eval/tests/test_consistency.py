import unittest
from unittest.mock import MagicMock, patch, mock_open
import json
from eval.consistency import ConsistencyJudge, ProposalConsistencyVerdict, DebateConsistencyVerdict, analyze_trace_file

class TestConsistencyJudge(unittest.TestCase):
    def setUp(self):
        # Mock the LLM and the chain
        self.mock_llm = MagicMock()
        self.mock_chain = MagicMock()
        self.mock_llm.with_structured_output.return_value = self.mock_chain
        
        # Initialize judge with mocked LLM
        with patch('eval.consistency.ChatOpenAI', return_value=self.mock_llm):
            self.judge = ConsistencyJudge()

    def get_chain_call_arg(self):
        # Helper to get the argument passed to the chain whether via invoke() or __call__()
        if self.mock_chain.invoke.call_count > 0:
            return self.mock_chain.invoke.call_args[0][0]
        elif self.mock_chain.call_count > 0:
            return self.mock_chain.call_args[0][0]
        else:
            self.fail("Chain was not called via invoke() or __call__()")

    def test_check_proposal(self):
        # Test data (abbreviated)
        turn = {
            "content": {
                "hypothesis": "Bullish on Tech",
                "justification": "Fed rates cut",
                "risks_or_falsifiers": "Inflation spike",
                "claims": [{"text": "Claim 1"}],
                "orders": [{"ticker": "AAPL", "side": "buy", "size": 10}],
                "confidence": 0.8
            },
            "input_params": {
                "context": "Market Observation..."
            }
        }

        # Mock chain response
        expected_result = MagicMock(
            verdict=ProposalConsistencyVerdict.CONSISTENT,
            explanation="Good reasoning",
            confidence=0.9
        )
        self.mock_chain.invoke.return_value = expected_result
        self.mock_chain.return_value = expected_result

        result = self.judge.check_proposal(turn)

        # Check return value
        self.assertEqual(result, expected_result)

        # Verify chain invocation inputs
        arg = self.get_chain_call_arg()
        arg_str = str(arg)
        self.assertIn("Bullish on Tech", arg_str)
        self.assertIn("Fed rates cut", arg_str)

    def test_check_critique(self):
        # Test data
        turn = {
            "content": {
                "critiques": [{"objection": "Assumption is wrong"}]
            },
            "input_params": {
                "context": "Context...",
                "all_proposals_for_critique": [{"proposal": "Buy AAPL"}],
                "my_proposal": "Buy GOOGL"
            }
        }

        expected_result = MagicMock(
            verdict=DebateConsistencyVerdict.CONSISTENT,
            explanation="Valid critique",
            confidence=0.9
        )
        self.mock_chain.invoke.return_value = expected_result
        self.mock_chain.return_value = expected_result

        result = self.judge.check_critique(turn)

        # Check return value
        self.assertEqual(result, expected_result)

        # Verify chain invocation inputs
        arg = self.get_chain_call_arg()
        arg_str = str(arg)
        self.assertIn("Buy AAPL", arg_str)
        self.assertIn("Buy GOOGL", arg_str)

    def test_check_revision(self):
        turn = {
            "content": {
                "revision_notes": "Reduced size",
                "justification": "Risk valid",
                "orders": [{"size": 50}],
                "confidence": 0.7
            },
            "input_params": {
                "context": "Context...",
                "my_proposal": {"orders": [{"size": 100}]},
                "critiques_received": [{"objection": "Too risky"}]
            }
        }

        expected_result = MagicMock(
            verdict=DebateConsistencyVerdict.CONSISTENT,
            explanation="Revision matches notes",
            confidence=0.95
        )
        self.mock_chain.invoke.return_value = expected_result
        self.mock_chain.return_value = expected_result

        result = self.judge.check_revision(turn)

        # Check return value
        self.assertEqual(result, expected_result)

        # Verify chain invocation inputs
        arg = self.get_chain_call_arg()
        arg_str = str(arg)
        self.assertIn('"size": 100', arg_str)
        self.assertIn("Reduced size", arg_str)

    def test_check_judge_decision(self):
        turn = {
            "content": {
                "audited_memo": "Balanced approach",
                "strongest_objection": "Risk concern",
                "orders": [{"ticker": "AAPL"}],
                "confidence": 0.85
            },
            "input_params": {
                "context": "Context...",
                "critiques_text": "Critiques...",
                "revisions_for_judge": [{"action": "Buy"}]
            }
        }

        expected_result = MagicMock(
            verdict=DebateConsistencyVerdict.CONSISTENT,
            explanation="Sound judgement",
            confidence=0.9
        )
        self.mock_chain.invoke.return_value = expected_result
        self.mock_chain.return_value = expected_result

        result = self.judge.check_judge_decision(turn)

        # Check return value
        self.assertEqual(result, expected_result)

        # Verify chain invocation inputs
        arg = self.get_chain_call_arg()
        arg_str = str(arg)
        self.assertIn("Balanced approach", arg_str)
        self.assertIn("Critiques...", arg_str)

    @patch('eval.consistency.ConsistencyJudge')
    @patch('builtins.open', new_callable=mock_open, read_data='{}')
    @patch('json.load')
    def test_analyze_trace_file_multiple_agents(self, mock_json_load, mock_file, MockJudgeClass):
        # Setup mock judge instance
        mock_judge_instance = MockJudgeClass.return_value
        
        # Setup mock trace data with 3 agents (MACRO, RISK, TECHNICAL)
        # Turn sequence: 
        # 0: Proposal (MACRO)
        # 1: Critique (MACRO critiques RISK and TECHNICAL) -> 2 critiques!
        
        mock_trace_data = {
            "debate_turns": [
                {
                    "type": "proposal",
                    "role": "macro",
                    "agent_id": "macro_1",
                    "content": {},
                    "input_params": {}
                },
                {
                    "type": "critique",
                    "role": "macro",
                    "agent_id": "macro_1",
                    "content": {
                        "critiques": [
                            {"target_role": "RISK", "objection": "Obj1"},
                            {"target_role": "TECHNICAL", "objection": "Obj2"}
                        ]
                    },
                    "input_params": {
                        "all_proposals_for_critique": [
                            {"role": "RISK", "proposal": "..."},
                            {"role": "TECHNICAL", "proposal": "..."}
                        ],
                        "context": "foo"
                    }
                }
            ]
        }
        mock_json_load.return_value = mock_trace_data
        
        analyze_trace_file("dummy_path.json")
        
        # Verify check_proposal called once
        mock_judge_instance.check_proposal.assert_called_once()
        
        # Verify check_critique called TWICE (once for RISK, once for TECHNICAL)
        self.assertEqual(mock_judge_instance.check_critique.call_count, 2)

if __name__ == '__main__':
    unittest.main()
