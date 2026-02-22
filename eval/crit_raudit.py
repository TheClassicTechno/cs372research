"""RAudit-style CRIT scorer — blind process verification via four pillars.

Implements the reasonableness dial (ρ) from Chang & Geng (2026) "RAudit:
A Blind Auditing Protocol for LLM Reasoning".

=============================================================================
THEORETICAL BACKGROUND — CRIT and RAudit
=============================================================================

CRIT (Critical Reading Inquisitive Template) is a framework from Chang (2023)
"Prompting Large Language Models With the Socratic Method" for evaluating the
quality of arguments produced by LLMs. It borrows from Socratic pedagogy:
the idea that you can test whether someone *understands* their own argument
by cross-examining the reasoning steps, not by checking the final answer.

RAudit extends CRIT into a formal auditing protocol. The key insight is the
"blindness constraint": the auditor never sees the ground truth. This matters
because in real-world deployments (e.g., an LLM making trading decisions),
we often can't check correctness at evaluation time — we can only check
whether the *reasoning process* is sound. A correct answer reached by
flawed reasoning is still dangerous (it got lucky this time).

The RAudit "reasonableness dial" (ρ) operationalises this via four pillars,
each designed to catch a specific class of reasoning pathology:

  1. Logical validity   — catches non-sequiturs, circular reasoning, gaps
  2. Evidential support  — catches fabricated evidence, unsupported claims
  3. Alternative consideration — catches confirmation bias, premature certainty
  4. Causal alignment    — catches "rung collapse" (Pearl's causal ladder)

Rung collapse (Pillar 4) is particularly important for our multi-agent debate
system. Pearl's causal ladder has three levels:
  L1 (Association):    "X and Y are correlated" — observational data
  L2 (Intervention):   "If I do X, Y will change" — requires experiments
  L3 (Counterfactual): "If X hadn't happened, Y wouldn't have" — requires
                        structural causal models

Rung collapse occurs when an agent uses L1 evidence (e.g., "AAPL rose when
the Fed cut rates last time") to make L2 claims (e.g., "buying AAPL will
profit from the next rate cut"). The causal alignment pillar detects this.

=============================================================================
SCORING MECHANICS
=============================================================================

The evaluator LLM scores each pillar on a 1-10 integer scale. We normalise
to [0, 1] by dividing by 10. Then:

  gamma (γ) = arithmetic mean of the four normalised pillar scores.
              This is the composite reasonableness score, equivalent to
              ρ in the paper. Range: [0, 1].

  theta (θ) = structural confidence. This is a separate score (also 1-10,
              normalised) that captures how well-calibrated the argument is:
              does the agent's stated confidence match the evidence strength?
              Does it acknowledge uncertainty appropriately? Range: [0, 1].

  ρ (rho) = gamma. The paper defines ρ as the reasonableness dial output.

Thresholding (per §4.2 of the RAudit paper):
  ρ* = 0.8 is the default pass threshold. A reasoning trace with γ < 0.8
  is considered to have insufficient process quality, regardless of whether
  the final answer turns out to be correct.

  We also require theta >= 0.5 as a secondary gate — if structural
  confidence is very low, the trace fails even if pillar scores are decent.

Trace-output consistency is a boolean check: does the final answer stated
by the agent actually match what its own reasoning derives? This catches
the case where an LLM reasons toward conclusion A but then outputs B
(a known failure mode in chain-of-thought reasoning, sometimes called
"unfaithful reasoning").

=============================================================================
PIPELINE OVERVIEW
=============================================================================

For each debate turn, the scorer:
  1. Renders the reasoning trace into a structured prompt (Jinja templates)
  2. Sends it to an evaluator LLM (blind — no ground truth in the prompt)
  3. Parses the JSON response into per-pillar scores
  4. Computes gamma, theta, and trace-output consistency
  5. Returns a RAuditCRITResult

The artifact builder then aggregates across all turns in a debate to produce
a schema-conformant eval artifact (eval.schema.json v1.2.0).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import jinja2

from api_client.llm.factory import get_client
from api_client.llm.models import LLMRequest

logger = logging.getLogger(__name__)

# Path to the Jinja2 prompt templates used by the evaluator LLM.
# These live in eval/prompts/ alongside the scorer module.
_PROMPTS_DIR = Path(__file__).parent / "prompts"

# ---------------------------------------------------------------------------
# Threshold constants — from RAudit paper §4.2
# ---------------------------------------------------------------------------
# ρ* = 0.8: the minimum gamma (reasonableness) score for a trace to pass.
# This is deliberately high — the paper argues that in safety-critical
# applications, we should demand strong process quality. An agent that
# scores 0.7 may produce correct answers most of the time, but its
# reasoning process has enough gaps that we can't trust it reliably.
DEFAULT_GAMMA_THRESHOLD = 0.8

# theta threshold = 0.5: the minimum structural confidence score.
# This is lower than gamma because theta measures calibration quality,
# which is inherently harder to assess and more variable across traces.
DEFAULT_THETA_THRESHOLD = 0.5


# ===========================================================================
# Data classes — structured representation of evaluation results
# ===========================================================================

@dataclass
class PillarScore:
    """Score and justification for a single CRIT pillar.

    Each of the four pillars produces one of these. The score is normalised
    to [0, 1] (the LLM returns 1-10, we divide by 10). The justification
    is free text from the evaluator LLM citing specific evidence from the
    reasoning trace that supports the score.

    Attributes:
        score: Normalised pillar score in [0, 1]. 0.0 = worst, 1.0 = best.
               Scores map to qualitative bands:
                 0.1-0.3 = severe reasoning pathology detected
                 0.4-0.5 = significant issues
                 0.6-0.7 = adequate with minor concerns
                 0.8-1.0 = strong reasoning quality
        justification: The evaluator LLM's explanation for this score,
                       grounded in specific evidence from the trace. This
                       is critical for interpretability — a bare score is
                       meaningless without knowing *why* the evaluator
                       assigned it.
    """

    score: float = 0.0
    justification: str = ""


@dataclass
class PillarScores:
    """Per-pillar CRIT diagnostic scores from the RAudit reasonableness dial.

    Groups all four pillar scores together. Each pillar targets a different
    class of reasoning failure:

    Attributes:
        logical_validity: Pillar 1 — does the conclusion follow from the
            reasoning steps? Detects: non-sequiturs, circular reasoning,
            unsupported logical leaps, contradictions between steps.

        evidential_support: Pillar 2 — is every claim grounded in admitted
            evidence? Detects: fabricated citations, hallucinated data points,
            claims with no supporting evidence, evidence that doesn't actually
            support the claim it's attached to.

        alternative_consideration: Pillar 3 — have competing hypotheses been
            explored? Detects: confirmation bias (only seeking supporting
            evidence), premature certainty (jumping to conclusions), strawman
            treatment of counterarguments (acknowledging but not engaging).

        causal_alignment: Pillar 4 — does the reasoning match the required
            causal level? Uses Pearl's causal ladder (L1/L2/L3). Detects:
            rung collapse (using correlational L1 data to make interventional
            L2 claims), correlation-causation confusion, inappropriate
            counterfactual reasoning without structural causal models.
    """

    logical_validity: PillarScore = field(default_factory=PillarScore)
    evidential_support: PillarScore = field(default_factory=PillarScore)
    alternative_consideration: PillarScore = field(default_factory=PillarScore)
    causal_alignment: PillarScore = field(default_factory=PillarScore)

    def as_list(self) -> list[float]:
        """Return pillar scores as a flat list for aggregation.

        Order is fixed: [logical, evidential, alternative, causal].
        Used by the gamma computation (arithmetic mean of all four).
        """
        return [
            self.logical_validity.score,
            self.evidential_support.score,
            self.alternative_consideration.score,
            self.causal_alignment.score,
        ]


@dataclass
class RAuditCRITResult:
    """Result of a RAudit CRIT evaluation on a single reasoning trace.

    This is the output of scoring one debate turn. It contains both the
    aggregate scores (gamma, theta) and the full per-pillar diagnostic
    breakdown. The artifact builder reads gamma, theta, and notes to
    produce the eval schema artifact.

    Attributes:
        gamma: The composite reasonableness score ρ. Computed as the
            arithmetic mean of the four normalised pillar scores.
            Range: [0, 1]. This is the primary quality metric.
            A gamma of 0.8+ means the reasoning process is sound across
            all four dimensions. A gamma below 0.6 indicates serious
            reasoning deficiencies.

        theta: Structural confidence score. Measures how well-calibrated
            the agent's confidence is relative to its evidence strength.
            An agent that claims "95% confidence" with weak correlational
            evidence should get a low theta even if the pillar scores
            are reasonable. Range: [0, 1].
            Falls back to gamma if the evaluator LLM doesn't return it.

        pillars: The full per-pillar diagnostic breakdown. This is where
            the interpretability lives — you can see exactly which dimension
            of reasoning is strong or weak, and why.

        trace_output_consistent: Boolean flag for trace-output consistency.
            True if the final answer matches what the reasoning derives.
            False if the agent's reasoning leads to one conclusion but its
            stated answer is different (unfaithful chain-of-thought).
            This is a critical red flag — if False, the agent's stated
            reasoning cannot be trusted as an explanation of its behaviour.

        notes: Free-text blind assessment from the evaluator LLM. A
            holistic summary of reasoning quality, written without access
            to ground truth. Useful for human review of evaluation results.

        raw_response: The complete JSON dict returned by the evaluator LLM.
            Preserved for debugging and audit purposes. If the parser
            misinterprets something, you can inspect the raw response.
    """

    gamma: float = 0.0
    theta: float = 0.0
    pillars: PillarScores = field(default_factory=PillarScores)
    trace_output_consistent: bool = True
    notes: str = ""
    raw_response: dict = field(default_factory=dict)


# ===========================================================================
# RAuditCRITScorer — the main evaluation engine
# ===========================================================================

class RAuditCRITScorer:
    """Blind process verification scorer using RAudit four-pillar CRIT.

    This class orchestrates the evaluation of a single reasoning trace.
    It is "blind" in the RAudit sense: the evaluator LLM prompt contains
    only the agent's reasoning (claim, reasons, counterarguments, assumptions,
    decision) and NEVER the ground truth outcome. The evaluator assesses
    process validity, not outcome correctness.

    Architecture:
        1. Jinja2 templates render the reasoning trace into a structured
           prompt with the RAudit system instructions (four pillar rubric,
           blindness constraint, scoring guidance, output format).
        2. The prompt is sent to an evaluator LLM (e.g., gpt-4o-mini) with
           JSON response format to get structured scores.
        3. The JSON response is parsed into a RAuditCRITResult with
           normalised scores and per-pillar justifications.

    The evaluator LLM is a *different* model call from the debate agents —
    this is important for independence. You wouldn't want the same model
    that generated the reasoning to also evaluate it (though in practice,
    using a different model family is even better for reducing correlated
    biases).

    Parameters:
        provider: LLM provider name ("openai" or "anthropic"). Selects
            which API client to use for the evaluator calls.
        model: Model identifier for the evaluator LLM. Default is
            "gpt-4o-mini" — a good balance of cost, speed, and evaluation
            quality. For higher-stakes evaluations, consider "gpt-4o".
        temperature: Sampling temperature for the evaluator LLM. Default
            is 0.0 (deterministic) for reproducible evaluations. Higher
            values introduce variance in scores across runs.
        api_key: Optional API key override. If None, falls back to the
            OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.
        gamma_threshold: The ρ* threshold for pass/fail decisions.
            Default 0.8 per RAudit paper §4.2.
        theta_threshold: The minimum structural confidence for pass.
            Default 0.5.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        api_key: str | None = None,
        gamma_threshold: float = DEFAULT_GAMMA_THRESHOLD,
        theta_threshold: float = DEFAULT_THETA_THRESHOLD,
    ):
        # The LLM client handles retries, JSON parsing, and provider-specific
        # API differences. get_client() is a factory that returns an
        # OpenAI or Anthropic async client based on the provider string.
        self._client = get_client(provider, api_key=api_key)
        self._model = model

        # temperature=0.0 gives deterministic scoring: same trace always
        # gets the same scores. This is important for reproducibility in
        # research settings. In production, you might use temperature=0.1-0.3
        # and average over multiple evaluations for more robust scores.
        self._temperature = temperature

        self._gamma_threshold = gamma_threshold
        self._theta_threshold = theta_threshold

        # Jinja2 environment for loading prompt templates. StrictUndefined
        # means template rendering will fail loudly if a variable is missing,
        # rather than silently inserting empty strings.
        self._jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(_PROMPTS_DIR)),
            undefined=jinja2.StrictUndefined,
        )

    @property
    def gamma_threshold(self) -> float:
        """The ρ* threshold. Traces with gamma below this fail."""
        return self._gamma_threshold

    @property
    def theta_threshold(self) -> float:
        """Minimum structural confidence threshold."""
        return self._theta_threshold

    def score(
        self,
        claim: str,
        reasons: str,
        counterarguments: str = "None identified.",
        assumptions: str = "None stated.",
        final_decision: str = "",
        context: str = "",
    ) -> RAuditCRITResult:
        """Evaluate a single reasoning trace synchronously.

        Convenience wrapper around score_async() for non-async callers.
        Uses asyncio.run() internally, so don't call this from within an
        existing async event loop — use score_async() directly instead.
        """
        return asyncio.run(
            self.score_async(
                claim=claim,
                reasons=reasons,
                counterarguments=counterarguments,
                assumptions=assumptions,
                final_decision=final_decision,
                context=context,
            )
        )

    async def score_async(
        self,
        claim: str,
        reasons: str,
        counterarguments: str = "None identified.",
        assumptions: str = "None stated.",
        final_decision: str = "",
        context: str = "",
    ) -> RAuditCRITResult:
        """Evaluate a single reasoning trace asynchronously.

        This is the core evaluation method. It sends the reasoning trace
        to an evaluator LLM and parses the structured response.

        The evaluator is BLIND — it has no access to ground truth. It
        evaluates only whether the derivation steps support the conclusion.
        This is the fundamental RAudit constraint: process validity over
        outcome correctness.

        Parameters:
            claim: The conclusion Omega being evaluated. In our debate system,
                this is the agent's hypothesis (e.g., "bullish on tech sector
                due to strong earnings expectations"). This is what the four
                pillars evaluate — whether the reasoning *supports* this claim.

            reasons: Supporting reasons R, as a text block. These are the
                arguments the agent made in favour of its claim. May include
                justification text, individual claims with Pearl levels and
                confidence scores, and any other supporting material.

            counterarguments: Rival reasons R', as a text block. These are
                the objections, risks, and alternative hypotheses the agent
                considered. Strong reasoning engages seriously with R' —
                Pillar 3 (alternative consideration) specifically checks this.
                Defaults to "None identified." which will tank the Pillar 3
                score (rightly — an agent that considers no alternatives is
                exhibiting confirmation bias).

            assumptions: Key assumptions underlying the reasoning. These are
                extracted from the structured claims in the debate trace.
                Good reasoning makes assumptions explicit; hidden assumptions
                are a reasoning pathology that the evaluator should flag.

            final_decision: The concrete action taken (e.g., "BUY 150 AAPL;
                SELL 50 MSFT"). This is checked against the reasoning for
                trace-output consistency — does the decision match what the
                reasoning actually derives?

            context: Optional market/task context from the AgentTrace's
                what_i_saw field. This gives the evaluator background on the
                decision-making setting without revealing ground truth.

        Returns:
            RAuditCRITResult with gamma, theta, per-pillar scores, and notes.
        """
        # Load and render the prompt templates. The system template contains
        # the RAudit rubric (four pillar definitions, scoring bands, blindness
        # constraint, output format). The user template contains the specific
        # reasoning trace being evaluated.
        system_template = self._jinja_env.get_template("crit_raudit_system.jinja")
        user_template = self._jinja_env.get_template("crit_raudit_user.jinja")

        system_msg = system_template.render()
        user_msg = user_template.render(
            claim=claim,
            reasons=reasons,
            counterarguments=counterarguments,
            assumptions=assumptions,
            final_decision=final_decision,
            context=context,
        )

        # Build the LLM request. response_format="json" tells the API client
        # to request JSON output from the LLM and parse it automatically.
        # The client layer handles provider differences (OpenAI has native
        # JSON mode; Anthropic needs prompt-based JSON enforcement).
        request = LLMRequest(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            model=self._model,
            temperature=self._temperature,
            response_format="json",
        )

        # Make the async LLM call. The client handles retries on transient
        # errors (rate limits, timeouts, connection issues) with exponential
        # backoff. response.content is either a parsed dict (if JSON parsing
        # succeeded) or a raw string (if it didn't).
        response = await self._client.generate(request)
        return self._parse_response(response.content)

    @staticmethod
    def _parse_response(content: str | dict) -> RAuditCRITResult:
        """Parse the evaluator LLM's JSON response into a RAuditCRITResult.

        The evaluator returns a JSON object with this structure:
        {
            "pillars": {
                "logical_validity":        {"score": 1-10, "justification": "..."},
                "evidential_support":      {"score": 1-10, "justification": "..."},
                "alternative_consideration": {"score": 1-10, "justification": "..."},
                "causal_alignment":        {"score": 1-10, "justification": "..."}
            },
            "trace_output_consistent": true|false,
            "gamma": 1-10,    # LLM's self-computed gamma (we recompute anyway)
            "theta": 1-10,    # structural confidence
            "notes": "..."    # free-text blind assessment
        }

        We recompute gamma ourselves from the pillar scores rather than
        trusting the LLM's self-computed value, because LLMs sometimes
        get arithmetic wrong. theta is taken from the LLM since it's a
        holistic judgement that can't be derived from the four pillars.

        All 1-10 scores are normalised to [0, 1] via _normalize_score().
        """
        # Handle both string JSON and pre-parsed dict (the API client
        # may have already parsed it if JSON mode was used).
        if isinstance(content, str):
            data = json.loads(content)
        else:
            data = content

        # Parse the four pillar scores. Each pillar has a score (1-10) and
        # a justification string. We normalise scores to [0, 1] and preserve
        # justifications verbatim for interpretability.
        pillars_raw = data.get("pillars", {})

        pillars = PillarScores(
            # Pillar 1: Logical Validity
            # Does Omega follow from the reasoning steps?
            # Low scores indicate non-sequiturs, circular reasoning, or gaps.
            logical_validity=PillarScore(
                score=_normalize_score(
                    pillars_raw.get("logical_validity", {}).get("score", 0)
                ),
                justification=pillars_raw.get("logical_validity", {}).get(
                    "justification", ""
                ),
            ),
            # Pillar 2: Evidential Support
            # Is every claim grounded in admitted evidence?
            # Low scores indicate fabricated evidence or unsupported claims.
            evidential_support=PillarScore(
                score=_normalize_score(
                    pillars_raw.get("evidential_support", {}).get("score", 0)
                ),
                justification=pillars_raw.get("evidential_support", {}).get(
                    "justification", ""
                ),
            ),
            # Pillar 3: Alternative Consideration
            # Were competing hypotheses explored and addressed?
            # Low scores indicate confirmation bias or premature certainty.
            alternative_consideration=PillarScore(
                score=_normalize_score(
                    pillars_raw.get("alternative_consideration", {}).get("score", 0)
                ),
                justification=pillars_raw.get("alternative_consideration", {}).get(
                    "justification", ""
                ),
            ),
            # Pillar 4: Causal Alignment
            # Does the evidence level match the causal claim level?
            # Low scores indicate rung collapse (L1 evidence for L2/L3 claims)
            # or correlation-causation confusion.
            causal_alignment=PillarScore(
                score=_normalize_score(
                    pillars_raw.get("causal_alignment", {}).get("score", 0)
                ),
                justification=pillars_raw.get("causal_alignment", {}).get(
                    "justification", ""
                ),
            ),
        )

        # Compute gamma (ρ) as the arithmetic mean of the four pillar scores.
        # We recompute this ourselves rather than using the LLM's self-reported
        # gamma, because LLMs are unreliable at arithmetic. The LLM's gamma
        # field in the JSON is ignored in favour of this computation.
        pillar_scores = pillars.as_list()
        gamma = sum(pillar_scores) / 4.0 if pillar_scores else 0.0
        gamma = round(gamma, 4)  # Avoid floating-point noise in artifacts

        # Theta (structural confidence) is taken from the LLM's response
        # because it's a holistic judgement about calibration quality that
        # can't be derived from the four pillars alone. It captures whether
        # the agent's confidence matches its evidence strength.
        # If the LLM didn't return theta (or returned 0), fall back to gamma
        # as a reasonable proxy — well-reasoned arguments tend to be
        # well-calibrated.
        theta_raw = data.get("theta", 0)
        theta = _normalize_score(theta_raw) if theta_raw else gamma

        # Trace-output consistency: does the final answer match the reasoning?
        # Default to True (consistent) if not reported, since we don't want
        # to penalise traces where the evaluator didn't flag inconsistency.
        trace_output_consistent = data.get("trace_output_consistent", True)

        return RAuditCRITResult(
            gamma=gamma,
            theta=theta,
            pillars=pillars,
            trace_output_consistent=trace_output_consistent,
            notes=data.get("notes", ""),
            raw_response=data,
        )


# ===========================================================================
# Score normalisation
# ===========================================================================

def _normalize_score(value: int | float) -> float:
    """Normalise a 1-10 integer score to [0, 1] float.

    The evaluator LLM returns scores on a 1-10 integer scale (matching
    common rubric conventions and human intuition). We normalise to [0, 1]
    for internal computation and artifact storage.

    The mapping is linear: score/10. So:
      1  -> 0.1  (severe pathology)
      5  -> 0.5  (borderline)
      8  -> 0.8  (passes default threshold)
      10 -> 1.0  (perfect)

    Clamped to [0, 1] to handle edge cases (LLM returns 0, 11, etc.).
    Non-numeric values (None, empty string) return 0.0.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v / 10.0))


# ===========================================================================
# Eval artifact builder
# ===========================================================================

def build_raudit_eval_artifact(
    debate_id: str,
    run_id: str,
    turn_results: list[tuple[str, RAuditCRITResult]],
    evaluation_mode: str = "posthoc",
    gamma_threshold: float = DEFAULT_GAMMA_THRESHOLD,
    theta_threshold: float = DEFAULT_THETA_THRESHOLD,
    experiment_label: str | None = None,
    notes: str | None = None,
) -> dict:
    """Build a complete eval artifact from per-turn RAudit CRIT results.

    Aggregates individual turn evaluations into a debate-level summary
    conforming to eval.schema.json v1.2.0.

    Multi-turn aggregation logic:
        gamma_mean = arithmetic mean of all per-turn gamma scores.
        theta_mean = arithmetic mean of all per-turn theta scores.

        Per-turn pass: gamma >= gamma_threshold AND theta >= theta_threshold.
        threshold_pass (debate-level): True only if ALL turns pass.

        overall_verdict:
            "pass"  — every turn passes both gamma and theta thresholds
            "fail"  — no turn passes
            "mixed" — some turns pass, some fail (indicates inconsistent
                      reasoning quality across the debate)

    The "mixed" verdict is particularly informative — it suggests the agent
    reasons well in some contexts but not others, which may indicate
    sensitivity to topic complexity or prompt structure.

    Parameters:
        debate_id: Stable identifier for this debate (e.g., "debate_macro_value").
        run_id: Identifier for this specific run (e.g., "run_2026-02-19T20:26:08").
        turn_results: List of (turn_id, RAuditCRITResult) tuples, one per
            evaluated turn. Turn order matches the debate transcript order.
        evaluation_mode: "posthoc" (evaluate after debate completes) or
            "in_loop" (evaluate during debate, potentially influencing later
            turns). Currently only posthoc is implemented.
        gamma_threshold: The ρ* threshold for per-turn pass/fail. Default 0.8.
        theta_threshold: The structural confidence threshold. Default 0.5.
        experiment_label: Optional label for A/B experiment tracking (e.g.,
            "crit_in_loop_v1" vs "baseline_no_crit").
        notes: Optional free-text notes for the artifact.

    Returns:
        Dict conforming to eval.schema.json v1.2.0.
    """
    from datetime import datetime, timezone

    # Version strings for provenance tracking. These let us trace exactly
    # which version of the evaluator produced a given artifact, which is
    # critical for reproducibility when evaluator logic changes over time.
    SCHEMA_VERSION = "1.2.0"
    EVALUATOR_VERSION = "eval_raudit_crit_v1"
    CRIT_VERSION = "raudit_crit_v1"

    # Handle the empty case: no turns evaluated. This can happen if the
    # transcript parser found no evaluable turns (e.g., a trace with only
    # critiques and no proposals/revisions/judge decisions).
    if not turn_results:
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "debate_id": debate_id,
            "run_id": run_id,
            "evaluation_mode": evaluation_mode,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "eval_metadata": {
                "evaluator_version": EVALUATOR_VERSION,
                "crit_version": CRIT_VERSION,
                "rca_version": None,      # Root Cause Analysis — not yet implemented
                "t3_version": None,       # T3 evaluation — not yet implemented
                "pid_version": None,      # PID controller — not yet implemented
                "raudit_version": CRIT_VERSION,
                "notes": "No turns evaluated.",
            },
            "run_summary": {
                "overall_verdict": "fail",  # No turns = automatic fail
                "crit_summary": {
                    "gamma_mean": None,
                    "theta_mean": None,
                    "threshold_pass": False,
                    "notes": "No turns evaluated.",
                },
            },
            "turn_evaluations": None,
        }
        return artifact

    # ---------------------------------------------------------------------------
    # Per-turn evaluation loop: apply thresholds to each turn independently
    # ---------------------------------------------------------------------------
    turn_evaluations = []
    gammas = []       # Collect all gamma values for debate-level mean
    thetas = []       # Collect all theta values for debate-level mean
    turn_passes = []  # Track per-turn pass/fail for verdict computation

    for turn_id, result in turn_results:
        # A turn passes only if BOTH gamma and theta meet their thresholds.
        # This is a conjunction (AND), not a disjunction — we require both
        # process quality (gamma) and calibration quality (theta).
        turn_pass = (
            result.gamma >= gamma_threshold
            and result.theta >= theta_threshold
        )
        turn_passes.append(turn_pass)
        gammas.append(result.gamma)
        thetas.append(result.theta)

        # Build the per-turn evaluation record. The schema reserves slots
        # for other evaluation methods (rca, t3, pid, raudit) which are
        # set to None since only CRIT is implemented here.
        turn_evaluations.append({
            "turn_id": turn_id,
            "crit": {
                "gamma_mean": round(result.gamma, 4),
                "theta_mean": round(result.theta, 4),
                "threshold_pass": turn_pass,
                "notes": result.notes or None,
            },
            "rca": None,   # Root Cause Analysis — future eval method
            "t3": None,    # T3 evaluation — future eval method
            "pid": None,   # PID controller — future eval method
            "raudit": None,  # RAudit-specific extended fields — future
        })

    # ---------------------------------------------------------------------------
    # Debate-level aggregation
    # ---------------------------------------------------------------------------
    # gamma_mean: average reasoning quality across all turns. A high gamma_mean
    # with a "mixed" verdict means some turns are excellent but others are poor —
    # the agent is inconsistent.
    gamma_mean = sum(gammas) / len(gammas)
    theta_mean = sum(thetas) / len(thetas)

    # Verdict logic: strict (all must pass for "pass")
    all_pass = all(turn_passes)
    none_pass = not any(turn_passes)

    if all_pass:
        overall_verdict = "pass"    # Every turn meets both thresholds
    elif none_pass:
        overall_verdict = "fail"    # No turn meets thresholds
    else:
        overall_verdict = "mixed"   # Some turns pass, some fail

    # ---------------------------------------------------------------------------
    # Build the final artifact dict
    # ---------------------------------------------------------------------------
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "debate_id": debate_id,
        "run_id": run_id,
        "evaluation_mode": evaluation_mode,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "eval_metadata": {
            "evaluator_version": EVALUATOR_VERSION,
            "crit_version": CRIT_VERSION,
            "rca_version": None,
            "t3_version": None,
            "pid_version": None,
            "raudit_version": CRIT_VERSION,
            "notes": notes,
        },
        "run_summary": {
            "overall_verdict": overall_verdict,
            "crit_summary": {
                "gamma_mean": round(gamma_mean, 4),
                "theta_mean": round(theta_mean, 4),
                "threshold_pass": all_pass,
                "notes": notes,
            },
        },
        "turn_evaluations": turn_evaluations,
    }

    # Optional experiment config for A/B testing. When running controlled
    # experiments (e.g., comparing "CRIT-in-loop" vs "no CRIT" debate
    # conditions), this section records which experimental condition
    # produced this artifact.
    if experiment_label:
        artifact["experiment_config"] = {
            "label": experiment_label,
            "category": None,
            "interventions": {
                "crit_in_loop": evaluation_mode == "in_loop",
                "rca_in_loop": False,
            },
            "control": None,
            "extra_dimensions": None,
            "notes": None,
        }

    return artifact
