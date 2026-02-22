"""Transcript parser for multi_agent debate traces -> CRIT canonical traces.

=============================================================================
PURPOSE
=============================================================================

This module is the BRIDGE between the multi-agent debate system (which outputs
LangGraph trace JSON files) and the RAudit CRIT scorer (which evaluates
reasoning quality). It transforms raw debate transcripts into a normalised
"canonical trace" format that the scorer can consume.

The key challenge is that debate turns come in different shapes:
  - Proposals have a "hypothesis" field (the claim)
  - Revisions don't have "hypothesis" — the claim must be inferred
  - Judge decisions have an "audited_memo" instead of "justification"
  - Counterarguments appear in different fields across turn types

The parser handles all of these variants and produces a uniform CanonicalTrace
for each evaluable turn.

=============================================================================
WHAT GETS EVALUATED (AND WHAT DOESN'T)
=============================================================================

We evaluate THREE turn types that represent committed reasoning:
  1. Proposals (round 0) — agent's initial position with hypothesis + claims
  2. Revisions (round 1+) — agent's updated position after hearing critiques
  3. Judge decisions (final) — synthesised decision with audited memo

We SKIP critiques because they evaluate *others'* reasoning rather than
making their own claim->decision chain. A critique says "your argument has
flaw X" — it doesn't propose an alternative conclusion. Since CRIT evaluates
whether reasoning supports a conclusion, critiques aren't a good fit.

=============================================================================
CANONICAL TRACE FORMAT (CRIT's input)
=============================================================================

The canonical trace maps directly to the CRIT evaluation template:

  CLAIM (Omega)     = What the agent concludes. The four pillars all evaluate
                      whether the reasoning supports THIS specific claim.

  REASONS (R)       = Supporting arguments and evidence. These are what the
                      agent offers in favour of its claim. The evaluator checks
                      whether these reasons actually support the claim (Pillar 1:
                      logical validity) and whether they're grounded in evidence
                      (Pillar 2: evidential support).

  COUNTERARGUMENTS  = Rival reasons R'. What the agent considered against its
  (R')                own position. Pillar 3 (alternative consideration) checks
                      whether the agent seriously engaged with these, or just
                      paid lip service to them.

  ASSUMPTIONS       = Key assumptions. Good reasoning makes assumptions explicit.
                      Hidden assumptions are a reasoning pathology.

  FINAL DECISION    = The concrete action taken. Checked for trace-output
                      consistency — does the decision match the reasoning?

  CONTEXT           = Market/task setting (from AgentTrace.what_i_saw). Gives
                      background without revealing ground truth (maintains the
                      RAudit blindness constraint).

=============================================================================
MULTI-AGENT DEBATE TRACE FORMAT (what the parser reads)
=============================================================================

Input is a dict with three top-level keys:

  "trace": {
      "what_i_saw": "...",         # Market context — becomes CONTEXT
      "action": { "orders": [...], "claims": [...] },
      ...
  }

  "debate_turns": [                # Flat list of all debate interactions
      {
          "type": "proposal",      # Turn type (proposal/critique/revision/judge_decision)
          "round": 0,              # Round number (0 = proposals, 1+ = critique-revise cycles)
          "agent_id": "agent_macro",
          "content": {             # Parsed JSON from the agent's LLM response
              "hypothesis": "...", # The claim (proposals only)
              "justification": "...",
              "claims": [          # Structured claims with Pearl levels
                  {
                      "claim_text": "...",
                      "pearl_level": "L1",     # L1=association, L2=intervention, L3=counterfactual
                      "confidence": 0.7,
                      "assumptions": ["..."]
                  }
              ],
              "orders": [ {"ticker": "AAPL", "side": "buy", "size": 150} ],
              "risks_or_falsifiers": "..." or {...}
          }
      },
      ...
  ]

  "config": {                      # Debate configuration (not used by parser)
      "roles": ["macro", "value", "risk", "technical"],
      "max_rounds": 2,
      ...
  }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class CanonicalTrace:
    """CRIT-ready canonical representation of a single turn's reasoning.

    This is the normalised format that the RAudit CRIT scorer consumes.
    Every evaluable debate turn (proposal, revision, judge_decision) is
    transformed into one of these by the TranscriptParser.

    The string fields (reasons, counterarguments) are used by the RAudit
    scorer, which takes text blocks. The list fields (reasons_list,
    counterarguments_list) preserve individual items for potential future
    per-reason evaluation.

    Attributes:
        turn_id: Unique identifier for this turn within the debate.
            Format: "r{round}_{agent_id}_{type}"
            Example: "r0_agent_macro_proposal" (round 0, macro agent, proposal)
                     "r1_agent_value_revision" (round 1, value agent, revision)
                     "r2_judge_judge_decision" (round 2, judge, final decision)

        speaker_id: The agent that produced this turn (e.g., "agent_macro",
            "agent_value", "judge"). Used for display and tracking which
            agents produce stronger/weaker reasoning.

        claim: The conclusion Omega that the agent is defending. This is the
            focal point of CRIT evaluation — all four pillars assess whether
            the reasoning supports THIS claim.
            For proposals: extracted from the "hypothesis" field.
            For revisions: inferred from the first claim_text or first
                sentence of justification (revisions don't have "hypothesis").
            For judge decisions: first sentence of "audited_memo".

        reasons: Supporting reasons R as a single text block. Contains the
            agent's justification, structured claims (with Pearl levels and
            confidence), and any revision notes. This is the primary input
            to Pillars 1 (logical validity) and 2 (evidential support).

        counterarguments: Rival reasons R' as a single text block. Contains
            risks, falsifiers, and alternative hypotheses the agent considered.
            This is the primary input to Pillar 3 (alternative consideration).
            Defaults to "None identified." — agents that don't consider
            alternatives will receive low Pillar 3 scores (correctly).

        assumptions: Key assumptions extracted from the structured claims.
            Good reasoning makes assumptions explicit; hidden assumptions
            are a reasoning pathology that the evaluator should flag.
            Defaults to "None stated."

        final_decision: The concrete action taken, formatted as a readable
            string (e.g., "BUY 150 AAPL; SELL 50 MSFT (confidence: 0.7)").
            Checked for trace-output consistency — does the decision match
            what the reasoning actually derives?

        context: Market/task context from AgentTrace.what_i_saw. Provides
            the evaluator with background on the decision-making setting
            WITHOUT revealing ground truth (maintains blindness constraint).

        reasons_list: Individual supporting reasons as a list of strings.
            Each entry is one claim with Pearl level and confidence metadata.
            Preserved for potential per-reason evaluation (not currently used
            by the RAudit scorer, which takes the text block).

        counterarguments_list: Individual rival reasons as a list of strings.
            Each entry is one counterargument. Preserved for potential
            per-reason evaluation.
    """

    turn_id: str = ""
    speaker_id: str = ""
    claim: str = ""
    reasons: str = ""
    counterarguments: str = "None identified."
    assumptions: str = "None stated."
    final_decision: str = ""
    context: str = ""
    reasons_list: list[str] = field(default_factory=list)
    counterarguments_list: list[str] = field(default_factory=list)


class TranscriptParser:
    """Parses multi_agent debate trace JSON into canonical CRIT traces.

    This is the adapter between the LangGraph debate output format and the
    CRIT evaluation input format. It handles the structural differences
    between turn types (proposals have "hypothesis", revisions don't, etc.)
    and normalises everything into CanonicalTrace objects.

    Parameters:
        trace_data: Parsed JSON dict from a multi_agent debate trace file.
                    Expected keys: "trace", "debate_turns", "config".
    """

    def __init__(self, trace_data: dict):
        self._data = trace_data
        # "trace" contains the overall AgentTrace — we extract context from it
        self._trace = trace_data.get("trace", {})
        # "debate_turns" is the flat list of all debate interactions
        self._turns = trace_data.get("debate_turns", []) or []
        # "config" has debate settings (roles, max_rounds, etc.) — not currently used
        self._config = trace_data.get("config", {})
        # Build the context string once (it's the same for all turns in a debate)
        self._context = self._build_context()

    def _build_context(self) -> str:
        """Build context string from the top-level AgentTrace.what_i_saw.

        what_i_saw contains the enriched market observation that all agents
        in the debate received. It provides background for evaluation without
        revealing ground truth (maintaining the RAudit blindness constraint).
        """
        what_i_saw = self._trace.get("what_i_saw", "")
        if what_i_saw:
            return what_i_saw
        return ""

    def extract_traces(self) -> list[CanonicalTrace]:
        """Extract canonical traces from all evaluable turns.

        Iterates through debate_turns and creates one CanonicalTrace per
        evaluable turn. The three evaluable turn types are:

          - "proposal" (round 0): Agent's initial position. Has the clearest
            claim->reasons->decision structure.

          - "revision" (round 1+): Agent's updated position after hearing
            critiques. May incorporate feedback, change orders, or defend
            original position with additional arguments.

          - "judge_decision" (final round): The judge's synthesised decision.
            Aggregates the strongest arguments from all agents and makes
            the final call.

        Critiques ("critique" type) are SKIPPED because they evaluate others'
        reasoning rather than proposing their own claim->decision chain.
        CRIT needs a complete argument structure (claim + reasons + decision)
        to evaluate, and critiques don't have that.

        Returns:
            List of CanonicalTrace objects, in debate turn order.
        """
        traces = []

        for turn in self._turns:
            if not isinstance(turn, dict):
                continue

            turn_type = turn.get("type", "")

            # Dispatch to the appropriate parser based on turn type.
            # Each parser handles the structural differences of its turn type.
            if turn_type == "proposal":
                trace = self._parse_proposal(turn)
            elif turn_type == "revision":
                trace = self._parse_revision(turn)
            elif turn_type == "judge_decision":
                trace = self._parse_judge_decision(turn)
            else:
                # Skip critiques and any unrecognised turn types
                continue

            if trace is not None:
                traces.append(trace)

        return traces

    # -----------------------------------------------------------------------
    # Helper methods — shared formatting and extraction logic
    # -----------------------------------------------------------------------

    def _make_turn_id(self, turn: dict) -> str:
        """Build a turn_id from round, agent_id, and type.

        Format: "r{round}_{agent_id}_{type}"
        Example: "r0_agent_macro_proposal", "r1_agent_value_revision"
        """
        rnd = turn.get("round", 0)
        agent_id = turn.get("agent_id", "unknown")
        turn_type = turn.get("type", "unknown")
        return f"r{rnd}_{agent_id}_{turn_type}"

    def _format_orders(self, orders: list[dict] | None) -> str:
        """Format a list of order dicts into a readable decision string.

        Input:  [{"ticker": "AAPL", "side": "buy", "size": 150}, ...]
        Output: "BUY 150 AAPL; SELL 50 MSFT"

        This is the "FINAL DECISION" in the canonical trace, checked by
        the evaluator for trace-output consistency.
        """
        if not orders:
            return ""
        parts = []
        for o in orders:
            ticker = o.get("ticker", "?")
            side = o.get("side", "?")
            size = o.get("size", "?")
            parts.append(f"{side.upper()} {size} {ticker}")
        return "; ".join(parts)

    def _extract_assumptions(self, claims: list[dict] | None) -> str:
        """Extract all assumptions from a list of claim objects.

        Each claim in the debate trace can have an "assumptions" list —
        these are the conditions the agent assumes to be true for its
        claim to hold. Explicit assumptions are a sign of good reasoning;
        the evaluator can then check whether they're reasonable.

        Returns "None stated." if no assumptions are found, which is itself
        informative — an agent that states no assumptions may be overconfident.
        """
        if not claims:
            return "None stated."
        assumptions = []
        for c in claims:
            claim_assumptions = c.get("assumptions") or []
            assumptions.extend(claim_assumptions)
        if not assumptions:
            return "None stated."
        return "\n".join(f"- {a}" for a in assumptions)

    def _extract_claims_list(self, claims: list[dict] | None) -> list[str]:
        """Extract individual claim strings from structured claim objects.

        Each claim is formatted with its Pearl causal level and confidence
        so the evaluator can assess causal alignment (Pillar 4).

        Example output entry:
            "[L2] If the Fed cuts rates, tech stocks will rally (confidence: 0.7)"

        The Pearl level tag is critical for Pillar 4 evaluation:
          [L1] = association/observation — weakest causal claim
          [L2] = intervention — requires causal mechanism, not just correlation
          [L3] = counterfactual — requires structural causal model
        """
        if not claims:
            return []
        result = []
        for c in claims:
            text = c.get("claim_text", "").strip()
            if not text:
                continue
            pearl = c.get("pearl_level", "")
            confidence = c.get("confidence")
            # Prefix with Pearl level so the evaluator can check causal alignment
            entry = f"[{pearl}] {text}" if pearl else text
            if confidence is not None:
                entry += f" (confidence: {confidence})"
            result.append(entry)
        return result

    def _extract_counterarguments_list(self, risks: str | dict | None) -> list[str]:
        """Extract individual counterargument strings from risks_or_falsifiers.

        The risks_or_falsifiers field varies in format across agents:
          - String: free-text risk description (split on sentence boundaries)
          - Dict: structured with "evidence_change_mind", "confounder",
            "probability_wrong" fields

        Both formats are normalised into a flat list of individual
        counterargument strings for per-reason evaluation.
        """
        if risks is None:
            return []
        if isinstance(risks, str):
            text = risks.strip()
            if not text or text.lower() in ("none", "none.", "none identified.", "n/a"):
                return []
            # Split on sentence boundaries to get individual counterarguments.
            # This is a rough heuristic — it works for well-formed English text
            # but may oversplit on abbreviations (Dr., U.S., etc.).
            parts = [s.strip() for s in text.replace(". ", ".\n").split("\n") if s.strip()]
            # Filter out very short fragments (likely split artifacts)
            return [p for p in parts if len(p) > 5]
        if isinstance(risks, dict):
            # Structured format with named fields
            result = []
            ev = risks.get("evidence_or_event") or risks.get("evidence_change_mind", "")
            if ev and isinstance(ev, str) and ev.strip():
                result.append(f"Evidence that would change mind: {ev.strip()}")
            confounder = risks.get("confounder", "")
            if confounder and isinstance(confounder, str) and confounder.strip():
                result.append(f"Confounder: {confounder.strip()}")
            prob = risks.get("probability_wrong")
            if prob is not None:
                result.append(f"Probability wrong: {prob}")
            return result
        return []

    def _extract_claims_text(self, claims: list[dict] | None) -> str:
        """Extract claim_text entries as a formatted text block.

        Similar to _extract_claims_list but returns a single joined string
        for the "reasons" text block field (backward compat with the RAudit
        scorer, which takes text blocks rather than lists).
        """
        if not claims:
            return ""
        parts = []
        for c in claims:
            text = c.get("claim_text", "")
            confidence = c.get("confidence")
            pearl = c.get("pearl_level", "")
            if text:
                entry = f"[{pearl}] {text}"
                if confidence is not None:
                    entry += f" (confidence: {confidence})"
                parts.append(entry)
        return "\n".join(parts)

    def _format_risks(self, risks: str | dict | None) -> str:
        """Format risks_or_falsifiers into a readable counterarguments block.

        Handles both string and dict formats. Returns "None identified." if
        no risks are present — this is the default that signals to the
        evaluator that the agent didn't consider alternatives.
        """
        if risks is None:
            return "None identified."
        if isinstance(risks, str):
            return risks if risks.strip() else "None identified."
        if isinstance(risks, dict):
            parts = []
            if risks.get("evidence_or_event") or risks.get("evidence_change_mind"):
                ev = risks.get("evidence_or_event") or risks.get("evidence_change_mind", "")
                parts.append(f"Evidence that would change mind: {ev}")
            if risks.get("probability_wrong") is not None:
                parts.append(f"Probability wrong: {risks['probability_wrong']}")
            if risks.get("confounder"):
                parts.append(f"Confounder: {risks['confounder']}")
            return "\n".join(parts) if parts else "None identified."
        return "None identified."

    # -----------------------------------------------------------------------
    # Turn-type-specific parsers
    # -----------------------------------------------------------------------

    def _parse_proposal(self, turn: dict) -> CanonicalTrace | None:
        """Parse a proposal turn into a canonical trace.

        Proposals are the initial positions from round 0. They have the
        cleanest structure: an explicit "hypothesis" (the claim), a
        "justification" (the main argument), structured "claims" with
        Pearl levels, "orders" (the decision), and "risks_or_falsifiers".

        Mapping to canonical trace:
            hypothesis         -> claim (Omega)
            justification + claims -> reasons (R)
            risks_or_falsifiers   -> counterarguments (R')
            claims[].assumptions  -> assumptions
            orders + confidence   -> final_decision
        """
        content = turn.get("content", {})
        if not content:
            return None

        # Extract the raw fields from the proposal content
        hypothesis = content.get("hypothesis", "")        # The claim Omega
        justification = content.get("justification", "")  # Main argument text
        claims = content.get("claims") or []              # Structured claims with Pearl levels
        orders = content.get("orders") or []              # Trading orders (the decision)
        risks = content.get("risks_or_falsifiers")        # Counterarguments / risk factors
        confidence = content.get("confidence")            # Agent's stated confidence [0, 1]

        # Build the "reasons" text block from justification + structured claims.
        # The justification is the agent's narrative argument; the claims are
        # the specific propositions with Pearl causal levels and confidence.
        claims_text = self._extract_claims_text(claims)
        reasons_parts = []
        if justification:
            reasons_parts.append(justification)
        if claims_text:
            reasons_parts.append(f"Claims:\n{claims_text}")

        # Format the decision string from orders + confidence
        decision_str = self._format_orders(orders)
        if confidence is not None:
            decision_str += f" (confidence: {confidence})"

        # Build per-reason lists for potential per-reason evaluation.
        # Justification goes first (it's the main argument), followed by
        # individual claims with Pearl level annotations.
        reasons_list = self._extract_claims_list(claims)
        if justification and justification.strip():
            reasons_list.insert(0, justification.strip())
        counterarguments_list = self._extract_counterarguments_list(risks)

        return CanonicalTrace(
            turn_id=self._make_turn_id(turn),
            speaker_id=turn.get("agent_id", ""),
            claim=hypothesis or "No explicit claim identified.",
            reasons="\n\n".join(reasons_parts) if reasons_parts else "No explicit reasons stated.",
            counterarguments=self._format_risks(risks),
            assumptions=self._extract_assumptions(claims),
            final_decision=decision_str or "No explicit decision stated.",
            context=self._context,
            reasons_list=reasons_list,
            counterarguments_list=counterarguments_list,
        )

    def _parse_revision(self, turn: dict) -> CanonicalTrace | None:
        """Parse a revision turn into a canonical trace.

        Revisions are updated positions from round 1+. They differ from
        proposals in two important ways:

        1. No "hypothesis" field — the claim must be inferred from the first
           claim_text or the first sentence of the justification. This is
           because revisions respond to critiques rather than proposing a
           fresh hypothesis.

        2. A "revision_notes" field — explains what changed from the previous
           round and why. This is added to the reasons block so the evaluator
           can assess whether the revision was well-motivated.

        Mapping to canonical trace:
            first claim_text or first sentence of justification -> claim (Omega)
            justification + claims + revision_notes -> reasons (R)
            risks_or_falsifiers -> counterarguments (R')
            claims[].assumptions -> assumptions
            orders + confidence -> final_decision
        """
        content = turn.get("content", {})
        if not content:
            return None

        justification = content.get("justification", "")
        claims = content.get("claims") or []
        orders = content.get("orders") or []
        risks = content.get("risks_or_falsifiers")
        confidence = content.get("confidence")
        revision_notes = content.get("revision_notes", "")

        # Build reasons text block (justification + claims + revision notes)
        claims_text = self._extract_claims_text(claims)
        reasons_parts = []
        if justification:
            reasons_parts.append(justification)
        if claims_text:
            reasons_parts.append(f"Claims:\n{claims_text}")
        if revision_notes:
            reasons_parts.append(f"Revision notes: {revision_notes}")

        # Infer the claim since revisions don't have an explicit "hypothesis".
        # Priority: first claim_text > first sentence of justification > fallback.
        claim = ""
        if claims:
            claim = claims[0].get("claim_text", "")
        if not claim and justification:
            # Use first sentence of justification as the claim.
            # This is a heuristic — the first sentence usually states the
            # agent's main position before elaborating.
            first_dot = justification.find(".")
            if first_dot > 0:
                claim = justification[:first_dot + 1]
            else:
                claim = justification
        if not claim:
            claim = "No explicit claim identified."

        decision_str = self._format_orders(orders)
        if confidence is not None:
            decision_str += f" (confidence: {confidence})"

        # Build per-reason lists
        reasons_list = self._extract_claims_list(claims)
        if justification and justification.strip():
            reasons_list.insert(0, justification.strip())
        counterarguments_list = self._extract_counterarguments_list(risks)

        return CanonicalTrace(
            turn_id=self._make_turn_id(turn),
            speaker_id=turn.get("agent_id", ""),
            claim=claim,
            reasons="\n\n".join(reasons_parts) if reasons_parts else "No explicit reasons stated.",
            counterarguments=self._format_risks(risks),
            assumptions=self._extract_assumptions(claims),
            final_decision=decision_str or "No explicit decision stated.",
            context=self._context,
            reasons_list=reasons_list,
            counterarguments_list=counterarguments_list,
        )

    def _parse_judge_decision(self, turn: dict) -> CanonicalTrace | None:
        """Parse a judge_decision turn into a canonical trace.

        The judge decision is the final synthesis of the debate. It differs
        from proposals and revisions in several ways:

        1. "audited_memo" replaces "justification" — this is the judge's
           synthesised assessment of all agents' arguments, representing
           the strongest reasoning from the debate.

        2. "strongest_objection" is a dedicated field — the judge explicitly
           identifies the single most important counterargument. This is
           added to the counterarguments block.

        3. The claim is inferred from the first sentence of the audited_memo,
           since the judge doesn't state an explicit hypothesis.

        The judge decision is arguably the most important turn to evaluate,
        since it represents the final output of the debate system. If the
        judge's reasoning is flawed, the entire debate was for nothing.

        Mapping to canonical trace:
            first sentence of audited_memo -> claim (Omega)
            audited_memo + claims -> reasons (R)
            strongest_objection + risks_or_falsifiers -> counterarguments (R')
            claims[].assumptions -> assumptions
            orders + confidence -> final_decision
        """
        content = turn.get("content", {})
        if not content:
            return None

        audited_memo = content.get("audited_memo", "")       # Judge's synthesis
        strongest_objection = content.get("strongest_objection", "")  # Key counterargument
        claims = content.get("claims") or []
        orders = content.get("orders") or []
        risks = content.get("risks_or_falsifiers")
        confidence = content.get("confidence")

        # Build reasons from audited_memo + structured claims
        claims_text = self._extract_claims_text(claims)
        reasons_parts = []
        if audited_memo:
            reasons_parts.append(audited_memo)
        if claims_text:
            reasons_parts.append(f"Claims:\n{claims_text}")

        # Infer claim from first sentence of audited_memo
        claim = ""
        if audited_memo:
            first_dot = audited_memo.find(".")
            if first_dot > 0:
                claim = audited_memo[:first_dot + 1]
            else:
                claim = audited_memo
        if not claim:
            claim = "No explicit claim identified."

        # Counterarguments: combine the strongest_objection (judge's explicit
        # pick of the most important counterargument) with any additional
        # risks_or_falsifiers. This gives the evaluator a rich picture of
        # what alternatives were considered.
        counter_parts = []
        if strongest_objection:
            counter_parts.append(f"Strongest objection: {strongest_objection}")
        risks_str = self._format_risks(risks)
        if risks_str != "None identified.":
            counter_parts.append(risks_str)
        counterarguments = "\n".join(counter_parts) if counter_parts else "None identified."

        decision_str = self._format_orders(orders)
        if confidence is not None:
            decision_str += f" (confidence: {confidence})"

        # Build per-reason lists. For the judge, the audited_memo is the
        # primary reason, and the strongest_objection is the primary
        # counterargument.
        reasons_list = self._extract_claims_list(claims)
        if audited_memo and audited_memo.strip():
            reasons_list.insert(0, audited_memo.strip())
        counterarguments_list = self._extract_counterarguments_list(risks)
        if strongest_objection and strongest_objection.strip():
            counterarguments_list.insert(0, strongest_objection.strip())

        return CanonicalTrace(
            turn_id=self._make_turn_id(turn),
            speaker_id=turn.get("agent_id", ""),
            claim=claim,
            reasons="\n\n".join(reasons_parts) if reasons_parts else "No explicit reasons stated.",
            counterarguments=counterarguments,
            assumptions=self._extract_assumptions(claims),
            final_decision=decision_str or "No explicit decision stated.",
            context=self._context,
            reasons_list=reasons_list,
            counterarguments_list=counterarguments_list,
        )
