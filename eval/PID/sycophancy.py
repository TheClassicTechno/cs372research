"""
Sycophancy detection signals for RAudit PID control.

=================================================================================
PURPOSE
=================================================================================

"Sycophancy" in a multi-agent debate means agents are converging on the same
answer — not because they reasoned well, but because they're copying each
other or caving to social pressure.  The debate LOOKS like it's converging
(JS divergence drops, agents agree more), but the quality of reasoning
hasn't actually improved.

This is dangerous because the PID controller might see low error (quality
looks on-target) and back off, when in fact the agents are just agreeing
for the wrong reasons.

This file detects sycophancy by watching for a specific suspicious pattern:

    1. Agent disagreement drops sharply (JS divergence decreases a lot), AND
    2. The evidence agents cite becomes LESS overlapping (not more).

If agents were genuinely converging through good reasoning, you'd expect
them to converge on similar evidence too.  When disagreement drops but
evidence overlap ALSO drops, that's a red flag: agents are agreeing on
conclusions without agreeing on the underlying reasoning.

=================================================================================
THE THREE SIGNALS
=================================================================================

    SIGNAL 1 — Jensen-Shannon Divergence (Eq 2):
        Measures how much the agents disagree on quality scores.
        High JS = lots of disagreement.  Low JS = agents are aligned.

        Input:  List of per-agent reasonableness scores [ρ_1, ρ_2, ...]
        Output: A single number ≥ 0.

    SIGNAL 2 — Evidence Overlap (Eq 3):
        Measures how much the agents' cited evidence overlaps (Jaccard index).
        High overlap = agents are looking at the same things.
        Low overlap  = agents are using different evidence.

        Input:  Two sets of evidence identifiers (one per agent).
        Output: A number in [0, 1].

    SIGNAL 3 — Sycophancy Indicator (Eq 4):
        Combines signals 1 and 2 over TWO consecutive rounds.
        Fires (s_t = 1) when BOTH of these happen at once:
            - JS dropped by more than delta_s  (agents suddenly agree more)
            - Overlap also dropped              (but on less shared evidence)

        Input:  JS and overlap from this round AND the previous round.
        Output: 0 or 1.

=================================================================================
NOTE ON JS DIVERGENCE IMPLEMENTATION
=================================================================================

The existing eval/divergence.py file contains a generalized_js_divergence()
function that operates on portfolio weight DICTIONARIES (multi-asset
distributions, using KL divergence between them).

The JS divergence here is DIFFERENT: it operates on SCALAR reasonableness
scores, treating each score as a Bernoulli probability parameter and using
binary Shannon entropy.  The two implementations serve different purposes
and are not interchangeable.

=================================================================================
WHERE THIS IS USED IN THE PIPELINE
=================================================================================

    External CRIT scorer produces per-agent scores and evidence sets
              │
              ▼
    jensen_shannon_divergence(scores)  →  js_current
    evidence_overlap(evidence_A, evidence_B)  →  ov_current
              │
              ▼
    PIDController.step(rho_bar, js_current, ov_current)
        │
        └── internally calls compute_sycophancy_signal(js_current, js_prev,
                                                       ov_current, ov_prev,
                                                       delta_s)
                │
                └── returns s_t (0 or 1)
                        │
                        ▼
                    compute_error() adds mu * s_t to the error signal
                    (if s_t=1, the controller pushes harder even though
                     the raw quality score might look fine)
"""

import math
from typing import List, Set


def shannon_entropy(p: float) -> float:
    """Binary Shannon entropy for a single probability value.

    Formula:  H(p) = -p * log2(p)  -  (1-p) * log2(1-p)

    This treats p as the probability of a Bernoulli random variable (like
    a biased coin flip).  The entropy measures uncertainty:
        H(0.5) = 1.0  (maximum uncertainty — fair coin)
        H(0.0) = 0.0  (no uncertainty — always tails)
        H(1.0) = 0.0  (no uncertainty — always heads)

    Used by jensen_shannon_divergence() below to compute the entropy of
    each agent's reasonableness score and the entropy of the mean score.

    Args:
        p: A probability value.  Values outside [0, 1] are treated as
           boundary cases returning 0.0 (no uncertainty).

    Returns:
        Entropy in bits.  Range [0, 1] for valid inputs.
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def jensen_shannon_divergence(scores: List[float]) -> float:
    """JS divergence over per-agent reasonableness scores. (RAudit Eq 2)

    Formula:  JS_t = H(mean of scores) - mean of H(each score)

    In plain English: "How much more uncertain are we about the AVERAGE
    agent's opinion than about each individual agent's opinion?"

    If all agents give the same score, JS = 0 (no divergence).
    If agents give very different scores, JS > 0 (high divergence).

    This is a symmetric, bounded measure of disagreement.  Unlike raw
    variance, it works well for probability-valued scores because it
    uses information-theoretic entropy.

    Note: This uses binary Shannon entropy on scalar scores, treating each
    score as a Bernoulli parameter.  This is different from the
    generalized_js_divergence in eval/divergence.py, which uses KL
    divergence over multi-asset portfolio weight dictionaries.

    Pipeline context:
        Input:  Per-agent reasonableness scores from CRIT scorer
                (e.g. [0.7, 0.8, 0.6] for three agents)
        Output: Single float fed to PIDController.step() as js_current,
                which stores it in history for sycophancy detection.

    Args:
        scores: List of per-agent reasonableness scores for one round.
                Each score should be in [0, 1].

    Returns:
        Non-negative JS divergence.  Returns 0.0 for empty or single-agent input.
    """
    if len(scores) < 2:
        return 0.0
    n = len(scores)
    rho_bar = sum(scores) / n
    h_mean = shannon_entropy(rho_bar)
    h_agents = sum(shannon_entropy(s) for s in scores) / n
    # JS divergence is non-negative by construction; clamp for float safety
    return max(0.0, h_mean - h_agents)


def evidence_overlap(set_a: Set, set_b: Set) -> float:
    """Evidence overlap via Jaccard index. (RAudit Eq 3)

    Formula:  Ov_t = |A ∩ B| / |A ∪ B|

    In plain English: "Of all the evidence cited by either agent, what
    fraction was cited by BOTH agents?"

    Examples:
        {"a", "b", "c"} vs {"b", "c", "d"}  →  2/4 = 0.5
        {"a", "b"} vs {"a", "b"}             →  2/2 = 1.0  (complete overlap)
        {"a"} vs {"b"}                       →  0/2 = 0.0  (no overlap)
        {} vs {}                             →  0.0         (empty convention)

    Pipeline context:
        Input:  Evidence sets from two agents (identifiers like claim IDs,
                source references, etc.) extracted by the CRIT scorer.
        Output: Single float fed to PIDController.step() as ov_current,
                stored in history for sycophancy detection.

    Args:
        set_a: Evidence identifiers cited by agent A.
        set_b: Evidence identifiers cited by agent B.

    Returns:
        Jaccard index in [0, 1].  Returns 0.0 if both sets are empty.
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def compute_sycophancy_signal(
    js_current: float,
    js_prev: float,
    ov_current: float,
    ov_prev: float,
    delta_s: float,
) -> int:
    """Detect whether agents are converging suspiciously. (RAudit Eq 4)

    Formula:  s_t = 1  if  (JS dropped by more than delta_s)
                          AND (evidence overlap also dropped)
              s_t = 0  otherwise

    Formally:  s_t = I[ ΔJS_t < -δ_s  ∧  ΔOv_t < 0 ]
    where ΔJS_t = JS_t - JS_{t-1}  and  ΔOv_t = Ov_t - Ov_{t-1}

    The intuition:
        - If agents are genuinely converging through good reasoning, you'd
          expect their disagreement (JS) to drop AND their shared evidence
          (overlap) to increase — they're finding common ground.
        - If agents are being sycophantic (just agreeing to agree), their
          disagreement drops BUT their evidence overlap ALSO drops — they're
          converging on conclusions without converging on reasons.

    When s_t = 1, the error signal in compute_error() gets a bonus penalty
    of +mu, which forces the PID controller to push β up (more exploration)
    to break the fake consensus.

    Pipeline context:
        Called internally by PIDController.step() using the JS and overlap
        history stored in PIDState.  Not called directly by external code.

        Inputs come from:
            js_current, js_prev   ← PIDState.js_history[-1] and [-2]
            ov_current, ov_prev   ← PIDState.ov_history[-1] and [-2]
            delta_s               ← PIDConfig.delta_s

        Output feeds into:
            compute_error() as the s_t parameter

    Args:
        js_current: JS divergence at the current round.
        js_prev:    JS divergence at the previous round.
        ov_current: Evidence overlap at the current round.
        ov_prev:    Evidence overlap at the previous round.
        delta_s:    Minimum JS drop to count as "sharp."  A drop of exactly
                    delta_s is NOT enough — it must be strictly more.

    Returns:
        1 if sycophancy detected, 0 otherwise.
    """
    delta_js = js_current - js_prev
    delta_ov = ov_current - ov_prev
    if delta_js < -delta_s and delta_ov < 0.0:
        return 1
    return 0
