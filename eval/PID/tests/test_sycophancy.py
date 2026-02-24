"""Tests for eval.PID.sycophancy — entropy, JS divergence, overlap, signal."""

import math
import pytest
from eval.PID.sycophancy import (
    shannon_entropy,
    jensen_shannon_divergence,
    evidence_overlap,
    compute_sycophancy_signal,
)


class TestShannonEntropy:
    def test_fair_coin(self):
        """H(0.5) = 1 bit."""
        assert shannon_entropy(0.5) == pytest.approx(1.0)

    def test_certain(self):
        """H(0) = H(1) = 0."""
        assert shannon_entropy(0.0) == 0.0
        assert shannon_entropy(1.0) == 0.0

    def test_biased(self):
        """H(0.25) is between 0 and 1."""
        h = shannon_entropy(0.25)
        assert 0.0 < h < 1.0

    def test_symmetric(self):
        """H(p) == H(1-p)."""
        assert shannon_entropy(0.3) == pytest.approx(shannon_entropy(0.7))

    def test_negative_returns_zero(self):
        """Boundary convention for out-of-range values."""
        assert shannon_entropy(-0.1) == 0.0

    def test_above_one_returns_zero(self):
        assert shannon_entropy(1.5) == 0.0


class TestJensenShannonDivergence:
    def test_identical_scores(self):
        """All agents agree -> JS = 0."""
        assert jensen_shannon_divergence([0.8, 0.8, 0.8]) == pytest.approx(0.0)

    def test_maximally_divergent(self):
        """Extreme disagreement -> positive JS."""
        js = jensen_shannon_divergence([0.0, 1.0])
        assert js > 0.0

    def test_moderate_disagreement(self):
        js = jensen_shannon_divergence([0.3, 0.7])
        assert js > 0.0

    def test_single_agent_returns_zero(self):
        assert jensen_shannon_divergence([0.5]) == 0.0

    def test_empty_returns_zero(self):
        assert jensen_shannon_divergence([]) == 0.0

    def test_non_negative(self):
        """JS divergence should always be >= 0."""
        for scores in [[0.1, 0.9], [0.4, 0.6], [0.5, 0.5, 0.5]]:
            assert jensen_shannon_divergence(scores) >= 0.0

    def test_three_agents(self):
        js = jensen_shannon_divergence([0.2, 0.5, 0.8])
        assert js > 0.0


class TestEvidenceOverlap:
    def test_identical_sets(self):
        s = {"a", "b", "c"}
        assert evidence_overlap(s, s) == pytest.approx(1.0)

    def test_disjoint_sets(self):
        assert evidence_overlap({"a", "b"}, {"c", "d"}) == pytest.approx(0.0)

    def test_partial_overlap(self):
        ov = evidence_overlap({"a", "b", "c"}, {"b", "c", "d"})
        # intersection = {b,c} = 2, union = {a,b,c,d} = 4
        assert ov == pytest.approx(0.5)

    def test_empty_sets(self):
        """Both empty -> 0 by convention."""
        assert evidence_overlap(set(), set()) == pytest.approx(0.0)

    def test_one_empty(self):
        assert evidence_overlap({"a"}, set()) == pytest.approx(0.0)

    def test_subset(self):
        ov = evidence_overlap({"a", "b"}, {"a", "b", "c"})
        assert ov == pytest.approx(2.0 / 3.0)


class TestComputeSycophancySignal:
    def test_triggers(self):
        """JS drops sharply and overlap drops -> s_t = 1."""
        s = compute_sycophancy_signal(
            js_current=0.1, js_prev=0.3,
            ov_current=0.5, ov_prev=0.7,
            delta_s=0.05,
        )
        assert s == 1

    def test_no_trigger_js_increase(self):
        """JS increases -> no sycophancy."""
        s = compute_sycophancy_signal(
            js_current=0.4, js_prev=0.3,
            ov_current=0.5, ov_prev=0.7,
            delta_s=0.05,
        )
        assert s == 0

    def test_no_trigger_overlap_increase(self):
        """Overlap increases -> no sycophancy (even if JS drops)."""
        s = compute_sycophancy_signal(
            js_current=0.1, js_prev=0.3,
            ov_current=0.9, ov_prev=0.7,
            delta_s=0.05,
        )
        assert s == 0

    def test_no_trigger_small_js_drop(self):
        """JS drop smaller than delta_s -> no trigger."""
        s = compute_sycophancy_signal(
            js_current=0.28, js_prev=0.3,
            ov_current=0.5, ov_prev=0.7,
            delta_s=0.05,
        )
        assert s == 0

    def test_exact_threshold(self):
        """JS drop exactly equal to -delta_s is not strictly less -> no trigger."""
        s = compute_sycophancy_signal(
            js_current=0.25, js_prev=0.3,
            ov_current=0.5, ov_prev=0.7,
            delta_s=0.05,
        )
        assert s == 0

    def test_both_unchanged(self):
        s = compute_sycophancy_signal(
            js_current=0.3, js_prev=0.3,
            ov_current=0.7, ov_prev=0.7,
            delta_s=0.05,
        )
        assert s == 0
