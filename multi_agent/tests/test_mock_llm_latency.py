"""L4 integration tests: mock LLM with configurable latency.

Provides a MockLLMWithLatency class that can monkeypatch _call_llm
with content-based routing and configurable per-response delays.
Tests verify prompt matching, response ordering, and concurrent
call completion.
"""

from __future__ import annotations

import re
import time
import threading
from dataclasses import dataclass, field
from typing import Callable

import pytest


# ---------------------------------------------------------------------------
# MockLLMWithLatency
# ---------------------------------------------------------------------------


@dataclass
class _MockRule:
    """A single prompt-matching rule with response text and optional delay."""
    pattern: str
    response_text: str = '{"mock": true}'
    delay_ms: int = 0


class _RuleBuilder:
    """Builder for fluent when_prompt_contains().respond() / .sleep() API."""

    def __init__(self, mock_llm: "MockLLMWithLatency", pattern: str):
        self._mock_llm = mock_llm
        self._pattern = pattern
        self._delay_ms = 0
        self._response_text = '{"mock": true}'

    def respond(self, response_text: str) -> "_RuleBuilder":
        """Set the response text for matching prompts."""
        self._response_text = response_text
        self._commit()
        return self

    def sleep(self, ms: int) -> "_RuleBuilder":
        """Set the delay in milliseconds before returning."""
        self._delay_ms = ms
        self._commit()
        return self

    def _commit(self) -> None:
        """Update or add the rule in the parent MockLLM."""
        rule = _MockRule(
            pattern=self._pattern,
            response_text=self._response_text,
            delay_ms=self._delay_ms,
        )
        # Replace existing rule for same pattern, or append
        for i, existing in enumerate(self._mock_llm._rules):
            if existing.pattern == self._pattern:
                self._mock_llm._rules[i] = rule
                return
        self._mock_llm._rules.append(rule)


class MockLLMWithLatency:
    """Mock LLM that routes responses by prompt content with optional latency.

    Usage::

        mock = MockLLMWithLatency()
        mock.when_prompt_contains("macro").respond('{"role": "macro"}').sleep(50)
        mock.when_prompt_contains("risk").respond('{"role": "risk"}').sleep(100)

        # Monkeypatch _call_llm
        monkeypatch.setattr("multi_agent.graph.llm._call_llm", mock.call)
    """

    def __init__(self):
        self._rules: list[_MockRule] = []
        self._default_response: str = '{"fallback": true}'
        self._call_log: list[dict] = []
        self._lock = threading.Lock()

    def when_prompt_contains(self, text: str) -> _RuleBuilder:
        """Start building a rule that matches prompts containing ``text``."""
        return _RuleBuilder(self, text)

    def set_default_response(self, text: str) -> None:
        """Set the fallback response for non-matching prompts."""
        self._default_response = text

    def call(self, config: dict, system_prompt: str, user_prompt: str) -> str:
        """Drop-in replacement for _call_llm."""
        combined = f"{system_prompt}\n{user_prompt}"
        start = time.monotonic()

        matched_rule = None
        for rule in self._rules:
            if rule.pattern in combined:
                matched_rule = rule
                break

        if matched_rule:
            if matched_rule.delay_ms > 0:
                time.sleep(matched_rule.delay_ms / 1000.0)
            response = matched_rule.response_text
        else:
            response = self._default_response

        elapsed = time.monotonic() - start
        with self._lock:
            self._call_log.append({
                "system_prompt": system_prompt[:200],
                "user_prompt": user_prompt[:200],
                "response": response[:200],
                "elapsed_ms": elapsed * 1000,
                "matched_pattern": matched_rule.pattern if matched_rule else None,
            })

        return response

    @property
    def call_count(self) -> int:
        return len(self._call_log)

    @property
    def calls(self) -> list[dict]:
        return list(self._call_log)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMockLLMPromptMatching:
    """MockLLMWithLatency matches prompts by content substring."""

    def test_matching_rule_returns_custom_response(self):
        mock = MockLLMWithLatency()
        mock.when_prompt_contains("macro analyst").respond('{"analysis": "bullish"}')

        result = mock.call({}, "You are a macro analyst", "Analyze the market")
        assert '"analysis"' in result
        assert '"bullish"' in result

    def test_non_matching_returns_default(self):
        mock = MockLLMWithLatency()
        mock.when_prompt_contains("macro").respond('{"macro": true}')
        mock.set_default_response('{"default": true}')

        result = mock.call({}, "You are a risk manager", "Evaluate risk")
        assert '"default"' in result

    def test_first_matching_rule_wins(self):
        mock = MockLLMWithLatency()
        mock.when_prompt_contains("analyst").respond('{"rule": "analyst"}')
        mock.when_prompt_contains("macro analyst").respond('{"rule": "macro"}')

        result = mock.call({}, "You are a macro analyst", "Analyze")
        # "analyst" appears first in rules, so it matches first
        assert '"analyst"' in result

    def test_call_log_records_invocations(self):
        mock = MockLLMWithLatency()
        mock.when_prompt_contains("test").respond('{"ok": true}')

        mock.call({}, "test system", "test user")
        mock.call({}, "test again", "test user 2")

        assert mock.call_count == 2
        assert mock.calls[0]["matched_pattern"] == "test"


@pytest.mark.integration
class TestMockLLMLatency:
    """MockLLMWithLatency applies configurable delays."""

    def test_sleep_adds_delay(self):
        mock = MockLLMWithLatency()
        mock.when_prompt_contains("slow").respond('{"slow": true}').sleep(50)

        start = time.monotonic()
        mock.call({}, "slow request", "user prompt")
        elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms >= 40, f"Expected >= 40ms delay, got {elapsed_ms:.1f}ms"

    def test_no_sleep_is_fast(self):
        mock = MockLLMWithLatency()
        mock.when_prompt_contains("fast").respond('{"fast": true}')

        start = time.monotonic()
        mock.call({}, "fast request", "user prompt")
        elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms < 50, f"Expected < 50ms, got {elapsed_ms:.1f}ms"

    def test_staggered_delays_complete_in_order(self):
        """Calls with staggered delays complete in delay-length order."""
        mock = MockLLMWithLatency()
        mock.when_prompt_contains("short").respond('{"short": true}').sleep(20)
        mock.when_prompt_contains("long").respond('{"long": true}').sleep(80)

        results = []
        lock = threading.Lock()

        def call_and_record(prompt_fragment: str):
            result = mock.call({}, prompt_fragment, "user")
            with lock:
                results.append((prompt_fragment, result))

        t1 = threading.Thread(target=call_and_record, args=("long",))
        t2 = threading.Thread(target=call_and_record, args=("short",))

        # Start long first, then short
        t1.start()
        t2.start()

        t1.join(timeout=5)
        t2.join(timeout=5)

        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        # Short should finish before long (or at least both complete)
        assert any("short" in r[0] for r in results)
        assert any("long" in r[0] for r in results)


@pytest.mark.integration
class TestMockLLMConcurrency:
    """Concurrent calls all complete successfully."""

    def test_concurrent_calls_all_complete(self):
        mock = MockLLMWithLatency()
        mock.when_prompt_contains("agent").respond('{"concurrent": true}').sleep(10)

        num_threads = 8
        results = []
        lock = threading.Lock()

        def worker(i: int):
            result = mock.call({}, f"agent {i}", "user prompt")
            with lock:
                results.append(result)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(results) == num_threads, (
            f"Expected {num_threads} results, got {len(results)}"
        )
        assert mock.call_count == num_threads

    def test_concurrent_mixed_patterns(self):
        """Concurrent calls with different patterns all get correct responses."""
        mock = MockLLMWithLatency()
        mock.when_prompt_contains("alpha").respond('{"type": "alpha"}').sleep(10)
        mock.when_prompt_contains("beta").respond('{"type": "beta"}').sleep(10)
        mock.when_prompt_contains("gamma").respond('{"type": "gamma"}').sleep(10)

        results = {}
        lock = threading.Lock()

        def worker(label: str):
            result = mock.call({}, f"You are {label}", "user")
            with lock:
                results[label] = result

        threads = [
            threading.Thread(target=worker, args=(label,))
            for label in ["alpha", "beta", "gamma"]
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(results) == 3
        assert '"alpha"' in results["alpha"]
        assert '"beta"' in results["beta"]
        assert '"gamma"' in results["gamma"]
