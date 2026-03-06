"""Node functions and factories for the LangGraph debate orchestrator.

Contains all node functions (sequential batch nodes), per-agent node
factories (for parallel execution), judge, aggregation, and trace builders.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from ..config import AgentRole, DebateConfig
from ..models import Observation
from ..prompts import (
    ROLE_SYSTEM_PROMPTS,
    SYSTEM_CAUSAL_CONTRACT,
    build_critique_prompt,
    build_judge_prompt,
    build_proposal_user_prompt,
    build_revision_prompt,
    resolve_prompt_profile,
)
from ..prompts.registry import resolve_beta, get_registry


def _has_agent_profiles(config: dict) -> bool:
    """Check if the new agent profile system is active."""
    return bool(config.get("agent_profiles"))


def _build_system_prompt_from_profile(
    config: dict, role: str, phase: str, user_prompt: str,
) -> "PromptBuildResult":
    """Build system prompt using new agent profile system."""
    profiles = config["agent_profiles"]
    if phase == "judge":
        profile = config.get("judge_profile", {})
    else:
        profile = profiles.get(role, {})
    registry = get_registry(config)
    return registry.build_from_profile(
        role=role,
        phase=phase,
        profile=profile,
        beta=resolve_beta(config, phase),
        user_prompt=user_prompt,
    )


def _get_user_sections_from_profile(config: dict, role: str, phase: str) -> list[str] | None:
    """Get user prompt sections from agent profile."""
    profiles = config.get("agent_profiles", {})
    if phase == "judge":
        profile = config.get("judge_profile", {})
    else:
        profile = profiles.get(role, {})
    usr = profile.get("user_prompts", {}).get(phase, {})
    return usr.get("sections")

from .allocation import normalize_allocation
from .sector_constraints import (
    build_sector_constraint_text,
    build_sector_map,
    enforce_sector_limits,
    filter_allocation_by_permissions,
)
from .display import (
    _print_allocation,
    _print_critique_summary,
    _verbose_critique,
    _verbose_judge,
    _verbose_proposal,
    _verbose_revision,
)
from .llm import _call_llm, _log_prompt, _parse_json
from .mocks import _mock_critique, _mock_judge, _mock_proposal, _mock_revision
from .state import DebateState, ParallelRoundState


def _call_llm_with_lifecycle(config: dict, system_prompt: str, user_prompt: str,
                              role: str, phase: str) -> str:
    """Wrap _call_llm with optional lifecycle callbacks for terminal display."""
    lifecycle = config.get("_llm_lifecycle")
    if not lifecycle:
        return _call_llm(config, system_prompt, user_prompt, role=role)

    start_fn, end_fn, _ = lifecycle
    call_id = f"{phase}_{role}"
    start_fn(call_id, role, phase)
    result = ""
    try:
        result = _call_llm(config, system_prompt, user_prompt, role=role)
    finally:
        end_fn(call_id, result)
    return result


def _apply_sector_permissions(
    raw_alloc: dict[str, float],
    role: str,
    config: dict,
) -> dict[str, float]:
    """Filter raw allocation by role's sector permissions (if configured).

    Returns the allocation unchanged if no sector config or permissions are set.
    """
    sector_cfg = config.get("sector_config")
    if not sector_cfg:
        return raw_alloc
    permissions = sector_cfg.get("agent_sector_permissions")
    if not permissions:
        return raw_alloc
    sector_map = build_sector_map(sector_cfg["sectors"])
    return filter_allocation_by_permissions(raw_alloc, role, sector_map, permissions)


def _apply_sector_limits(
    alloc: dict[str, float],
    config: dict,
) -> dict[str, float]:
    """Enforce sector exposure limits on a normalized allocation (if configured).

    Merges ``max_sector_weight`` (blanket cap) with per-sector ``sector_limits``
    before calling ``enforce_sector_limits``.  When both are set the stricter
    max wins for each sector.

    Returns the allocation unchanged if no sector config or limits are set.
    """
    sector_cfg = config.get("sector_config")
    if not sector_cfg:
        return alloc

    limits = sector_cfg.get("sector_limits") or {}
    max_sw = sector_cfg.get("max_sector_weight")
    sectors_def = sector_cfg.get("sectors")

    if not limits and max_sw is None:
        return alloc
    if not sectors_def:
        return alloc

    # Merge max_sector_weight into per-sector limits
    if max_sw is not None:
        merged: dict[str, dict] = {}
        for sector in sectors_def:
            existing = limits.get(sector, {})
            existing_max = existing.get("max", 1.0)
            merged[sector] = {
                "min": existing.get("min", 0.0),
                "max": min(existing_max, max_sw),
            }
        limits = merged

    if not limits:
        return alloc

    sector_map = build_sector_map(sectors_def)
    return enforce_sector_limits(alloc, sector_map, limits)


# =============================================================================
# SEQUENTIAL BATCH NODE FUNCTIONS
# =============================================================================


def build_context_node(state: DebateState) -> dict:
    """Build enriched context string from observation (memo/allocation mode)."""
    obs_model = Observation(**state["observation"])

    # Memo mode: the text_context IS the financial context.
    # Prepend only portfolio state and universe.
    header = (
        f"## Portfolio Allocation Task\n"
        f"- Cash to allocate: ${obs_model.portfolio_state.cash:,.2f}\n"
        f"- Allocation universe: {', '.join(obs_model.universe)}\n"
        f"- As-of: {obs_model.timestamp}\n"
    )
    memo_context = obs_model.text_context or ""
    enriched = header + "\n" + memo_context

    return {"enriched_context": enriched}


def propose_node(state: DebateState) -> dict:
    """All role agents generate their initial proposals.

    Idempotency guard: when the runner calls single_round_graph multiple
    times, propose is in the graph every round but should only execute
    once (round 1).  On subsequent rounds proposals already exist in
    state, so we return {} — a no-op that leaves all state fields
    untouched.  Uses len(...) > 0 for robustness (handles None, empty
    list).  This guard is harmless for the monolithic graph where propose
    runs exactly once.
    """
    if len(state.get("proposals") or []) > 0:
        return {}

    config = state["config"]
    context = state["enriched_context"]
    obs = state["observation"]
    roles = config.get("roles", ["macro", "value", "risk"])
    is_mock = config.get("mock", False)

    proposals = []
    turns = []

    for i, role in enumerate(roles):
        if not config.get("console_display"):
            print(f"  [Round 0 - Propose] {role.upper()} agent ({i+1}/{len(roles)})...", flush=True)

        use_profiles = _has_agent_profiles(config)
        if use_profiles:
            user_secs = _get_user_sections_from_profile(config, role, "propose")
        else:
            profile = resolve_prompt_profile(config, role, "propose")
            user_secs = profile.get("user_sections")

        sector_text = build_sector_constraint_text(
            config.get("sector_config"), role,
        )
        user_prompt = build_proposal_user_prompt(
            context,
            prompt_file_overrides=config.get("prompt_file_overrides"),
            user_sections=user_secs,
            sector_constraints=sector_text,
        )

        if use_profiles:
            build_result = _build_system_prompt_from_profile(config, role, "propose", user_prompt)
        else:
            system_blocks = profile.get("system_blocks")
            registry = get_registry(config)
            build_result = registry.build(
                role=role, phase="propose",
                beta=resolve_beta(config, "propose"),
                user_prompt=user_prompt,
                block_order=system_blocks,
                prompt_file_overrides=config.get("prompt_file_overrides"),
            )
        role_system = build_result.system_prompt

        _log_prompt(config, role, "propose", 0, role_system, user_prompt)
        _capture = config.get("_prompt_capture")
        if _capture:
            _capture("proposals", role, role_system, user_prompt)

        raw_text = None  # Raw LLM output for eval module
        if is_mock:
            result = _mock_proposal(role, obs, config)
            raw_text = json.dumps(result, indent=2)
        else:
            raw_text = _call_llm_with_lifecycle(config, role_system, user_prompt, role, "propose")
            result = _parse_json(raw_text)

        universe = obs.get("universe", [])
        raw_alloc = result.get("allocation", {})
        raw_alloc = _apply_sector_permissions(raw_alloc, role, config)

        # Pull constraints from config
        ac = config.get("allocation_constraints") or {}
        action_dict = {
            "allocation": normalize_allocation(
                raw_alloc, universe,
                max_weight=ac.get("max_weight", 1.0),
                min_holdings=ac.get("min_holdings", 1),
            ),
            "justification": result.get("justification", ""),
            "confidence": result.get("confidence", 0.5),
            "claims": result.get("claims", []),
        }

        if config.get("verbose"):
            _verbose_proposal(role, result)

        if not config.get("console_display"):
            _print_allocation(role, action_dict, "proposes")

        proposals.append({
            "role": role,
            "action_dict": action_dict,
            "raw_response": raw_text,
        })

        turns.append({
            "round": 0,
            "agent_id": f"agent_{role}",
            "role": role,
            "type": "proposal",
            "content": result,
            "raw_system_prompt": role_system,
            "raw_user_prompt": user_prompt,
            "raw_response": raw_text,
            "input_params": {
                "context": context,
            },
        })

    if not config.get("console_display"):
        print(f"  [Round 0 - Propose] All {len(roles)} proposals complete.", flush=True)
    return {
        "proposals": proposals,
        "debate_turns": turns,
        "current_round": 1,
    }


def critique_node(state: DebateState) -> dict:
    """All role agents critique each other's proposals (or prior revisions)."""
    config = state["config"]
    if config.get("propose_only"):
        return {}
    context = state["enriched_context"]
    current_round = state.get("current_round", 1)
    is_mock = config.get("mock", False)

    # After first round, critique the revisions; otherwise the proposals
    source = state.get("revisions") if state.get("revisions") else state["proposals"]

    all_proposals_for_critique = [
        {
            "role": p["role"],
            "proposal": json.dumps(p.get("action_dict", {})),
        }
        for p in source
    ]

    critiques = []
    turns = []

    for i, p in enumerate(source):
        role = p["role"]
        if not config.get("console_display"):
            print(f"  [Round {current_round} - Critique] {role.upper()} agent ({i+1}/{len(source)})...", flush=True)
        my_proposal = json.dumps(p.get("action_dict", {}))

        use_profiles = _has_agent_profiles(config)
        if use_profiles:
            user_secs = _get_user_sections_from_profile(config, role, "critique")
        else:
            profile = resolve_prompt_profile(config, role, "critique")
            user_secs = profile.get("user_sections")

        sector_text = build_sector_constraint_text(
            config.get("sector_config"), role,
        )
        prompt = build_critique_prompt(
            role, context, all_proposals_for_critique, my_proposal,
            prompt_file_overrides=config.get("prompt_file_overrides"),
            user_sections=user_secs,
            sector_constraints=sector_text,
        )

        if use_profiles:
            build_result = _build_system_prompt_from_profile(config, role, "critique", prompt)
        else:
            system_blocks = profile.get("system_blocks")
            registry = get_registry(config)
            build_result = registry.build(
                role=role, phase="critique",
                beta=resolve_beta(config, "critique"),
                user_prompt=prompt,
                block_order=system_blocks,
                prompt_file_overrides=config.get("prompt_file_overrides"),
            )
        system_msg = build_result.system_prompt

        _log_prompt(config, role, "critique", current_round, system_msg, prompt)
        _capture = config.get("_prompt_capture")
        if _capture:
            _capture("critiques", role, system_msg, prompt)

        raw_text = None  # Raw LLM output for eval module
        if is_mock:
            result = _mock_critique(role, source)
            raw_text = json.dumps(result, indent=2)
        else:
            raw_text = _call_llm_with_lifecycle(config, system_msg, prompt, role, "critique")
            result = _parse_json(raw_text)

        if config.get("verbose"):
            _verbose_critique(role, result)

        if not config.get("console_display"):
            _print_critique_summary(role, result)

        critiques.append({
            "role": role,
            "critiques": result.get("critiques", []),
            "self_critique": result.get("self_critique", ""),
        })

        turns.append({
            "round": current_round,
            "agent_id": f"agent_{role}",
            "role": role,
            "type": "critique",
            "content": result,
            "raw_system_prompt": system_msg,
            "raw_user_prompt": prompt,
            "raw_response": raw_text,
            "input_params": {
                "context": context,
                "all_proposals_for_critique": all_proposals_for_critique,
                "my_proposal": my_proposal,
            }
        })

    if not config.get("console_display"):
        print(f"  [Round {current_round} - Critique] All critiques complete.", flush=True)
    return {
        "critiques": critiques,
        "debate_turns": turns,
    }


def revise_node(state: DebateState) -> dict:
    """All role agents revise their proposals based on critiques received."""
    config = state["config"]
    if config.get("propose_only"):
        return {}
    context = state["enriched_context"]
    current_round = state.get("current_round", 1)
    is_mock = config.get("mock", False)
    obs = state["observation"]
    all_critiques = state.get("critiques", [])

    source = state.get("revisions") if state.get("revisions") else state["proposals"]

    revisions = []
    turns = []

    for i, p in enumerate(source):
        role = p["role"]
        if not config.get("console_display"):
            print(f"  [Round {current_round} - Revise] {role.upper()} agent ({i+1}/{len(source)})...", flush=True)
        my_proposal = json.dumps(p.get("action_dict", {}))

        # Collect critiques targeted at this role.
        # Normalize target_role: LLMs may return "MACRO", "Risk Agent", etc.
        critiques_received = []
        for c in all_critiques:
            for crit in c.get("critiques", []):
                target = crit.get("target_role", "").lower().split()[0]
                if target == role:
                    critiques_received.append({
                        "from_role": c["role"],
                        "objection": crit.get("objection", ""),
                        "falsifier": crit.get("falsifier"),
                    })

        use_profiles = _has_agent_profiles(config)
        if use_profiles:
            user_secs = _get_user_sections_from_profile(config, role, "revise")
        else:
            profile = resolve_prompt_profile(config, role, "revise")
            user_secs = profile.get("user_sections")

        sector_text = build_sector_constraint_text(
            config.get("sector_config"), role,
        )
        prompt = build_revision_prompt(
            role, context, my_proposal, critiques_received,
            prompt_file_overrides=config.get("prompt_file_overrides"),
            user_sections=user_secs,
            sector_constraints=sector_text,
        )

        if use_profiles:
            build_result = _build_system_prompt_from_profile(config, role, "revise", prompt)
        else:
            system_blocks = profile.get("system_blocks")
            registry = get_registry(config)
            build_result = registry.build(
                role=role, phase="revise",
                beta=resolve_beta(config, "revise"),
                user_prompt=prompt,
                block_order=system_blocks,
                prompt_file_overrides=config.get("prompt_file_overrides"),
            )
        system_msg = build_result.system_prompt

        _log_prompt(config, role, "revise", current_round, system_msg, prompt)
        _capture = config.get("_prompt_capture")
        if _capture:
            _capture("revisions", role, system_msg, prompt)

        if is_mock:
            result = _mock_revision(role, p.get("action_dict", {}), obs, config)
            raw_text = json.dumps(result, indent=2)
        else:
            raw_text = _call_llm_with_lifecycle(config, system_msg, prompt, role, "revise")
            result = _parse_json(raw_text)

        universe = obs.get("universe", [])
        raw_alloc = result.get("allocation", p.get("action_dict", {}).get("allocation", {}))
        raw_alloc = _apply_sector_permissions(raw_alloc, role, config)

        # Pull constraints from config
        ac = config.get("allocation_constraints") or {}
        action_dict = {
            "allocation": normalize_allocation(
                raw_alloc, universe,
                max_weight=ac.get("max_weight", 1.0),
                min_holdings=ac.get("min_holdings", 1),
            ),
            "justification": result.get("justification", ""),
            "confidence": result.get("confidence", 0.5),
            "claims": result.get("claims", []),
        }

        if config.get("verbose"):
            _verbose_revision(role, result)

        if not config.get("console_display"):
            _print_allocation(role, action_dict, "revised")

        revisions.append({
            "role": role,
            "action_dict": action_dict,
            "revision_notes": result.get("revision_notes", ""),
            "raw_response": raw_text,
        })

        turns.append({
            "round": current_round,
            "agent_id": f"agent_{role}",
            "role": role,
            "type": "revision",
            "content": result,
            "raw_system_prompt": system_msg,
            "raw_user_prompt": prompt,
            "raw_response": raw_text,
            "input_params": {
                "context": context,
                "my_proposal": my_proposal,
                "critiques_received": critiques_received,
            }
        })

    if not config.get("console_display"):
        print(f"  [Round {current_round} - Revise] All revisions complete.", flush=True)
    return {
        "revisions": revisions,
        "debate_turns": turns,
        "current_round": current_round + 1,
    }


# =============================================================================
# PER-AGENT NODE FACTORIES (for parallel single-round graph)
# =============================================================================
#
# Each factory returns a closure that handles ONE agent in a fan-out
# pattern.  The closure returns single-element lists so that
# ParallelRoundState's operator.add reducers can merge outputs from
# all parallel nodes at the sync barrier.
#
# Key differences from the batch node functions above:
#   - Per-agent nodes do NOT return current_round (avoids parallel
#     write conflict on a non-annotated int; the runner manages it).
#   - Return values are single-element lists, not full lists.
#   - Each closure captures `role` from the factory argument.
# =============================================================================


def _sync_noop(state: ParallelRoundState) -> dict:
    """No-op sync barrier node for fan-in.

    LangGraph merges all upstream outputs before running downstream
    nodes, so this node just passes through without modifying state.
    """
    return {}


def make_propose_node(role: str):
    """Factory: create a per-agent propose node for the given role.

    Extracts the loop body from propose_node.  Returns single-element
    lists for proposals and debate_turns.  Includes the same
    idempotency guard: if proposals already exist, returns {}.
    Does NOT return current_round.
    """

    def _propose(state: ParallelRoundState) -> dict:
        # Idempotency guard: skip if proposals already exist
        if len(state.get("proposals") or []) > 0:
            return {}

        config = state["config"]
        context = state["enriched_context"]
        obs = state["observation"]
        roles = config.get("roles", ["macro", "value", "risk"])
        is_mock = config.get("mock", False)

        i = roles.index(role) if role in roles else 0
        if not config.get("console_display"):
            print(f"  [Round 0 - Propose] {role.upper()} agent ({i+1}/{len(roles)})...", flush=True)

        use_profiles = _has_agent_profiles(config)
        if use_profiles:
            user_secs = _get_user_sections_from_profile(config, role, "propose")
        else:
            _profile = resolve_prompt_profile(config, role, "propose")
            user_secs = _profile.get("user_sections")

        sector_text = build_sector_constraint_text(
            config.get("sector_config"), role,
        )
        user_prompt = build_proposal_user_prompt(
            context,
            prompt_file_overrides=config.get("prompt_file_overrides"),
            user_sections=user_secs,
            sector_constraints=sector_text,
        )

        if use_profiles:
            build_result = _build_system_prompt_from_profile(config, role, "propose", user_prompt)
        else:
            system_blocks = _profile.get("system_blocks")
            registry = get_registry(config)
            build_result = registry.build(
                role=role, phase="propose",
                beta=resolve_beta(config, "propose"),
                user_prompt=user_prompt,
                block_order=system_blocks,
                prompt_file_overrides=config.get("prompt_file_overrides"),
            )
        role_system = build_result.system_prompt

        _log_prompt(config, role, "propose", 0, role_system, user_prompt)
        _capture = config.get("_prompt_capture")
        if _capture:
            _capture("proposals", role, role_system, user_prompt)

        raw_text = None
        if is_mock:
            result = _mock_proposal(role, obs, config)
            raw_text = json.dumps(result, indent=2)
        else:
            raw_text = _call_llm_with_lifecycle(config, role_system, user_prompt, role, "propose")
            result = _parse_json(raw_text)

        universe = obs.get("universe", [])
        raw_alloc = result.get("allocation", {})
        raw_alloc = _apply_sector_permissions(raw_alloc, role, config)

        # Pull constraints from config
        ac = config.get("allocation_constraints") or {}
        action_dict = {
            "allocation": normalize_allocation(
                raw_alloc, universe,
                max_weight=ac.get("max_weight", 1.0),
                min_holdings=ac.get("min_holdings", 1),
            ),
            "justification": result.get("justification", ""),
            "confidence": result.get("confidence", 0.5),
            "claims": result.get("claims", []),
        }

        if config.get("verbose"):
            _verbose_proposal(role, result)

        if not config.get("console_display"):
            _print_allocation(role, action_dict, "proposes")

        proposal = {
            "role": role,
            "action_dict": action_dict,
            "raw_response": raw_text,
        }

        turn = {
            "round": 0,
            "agent_id": f"agent_{role}",
            "role": role,
            "type": "proposal",
            "content": result,
            "raw_system_prompt": role_system,
            "raw_user_prompt": user_prompt,
            "raw_response": raw_text,
            "input_params": {
                "context": context,
            },
        }

        return {
            "proposals": [proposal],
            "debate_turns": [turn],
        }

    _propose.__name__ = f"propose_{role}"
    return _propose


def make_critique_node(role: str):
    """Factory: create a per-agent critique node for the given role.

    Extracts the loop body from critique_node.  Finds own entry in
    the source list by role field.  Reads ALL proposals for the
    critique prompt.  Returns single-element lists.
    """

    def _critique(state: ParallelRoundState) -> dict:
        config = state["config"]
        if config.get("propose_only"):
            return {}
        context = state["enriched_context"]
        current_round = state.get("current_round", 1)
        is_mock = config.get("mock", False)

        # After first round, critique the revisions; otherwise the proposals.
        # Sort by config role order for deterministic behavior (operator.add
        # merge order is non-deterministic).
        roles = config.get("roles", ["macro", "value", "risk"])
        role_order = {r: i for i, r in enumerate(roles)}
        raw_source = state.get("revisions") if state.get("revisions") else state["proposals"]
        source = sorted(raw_source, key=lambda e: role_order.get(e["role"], len(roles)))

        # Find own entry by role (safe lookup — missing role returns empty dict)
        p = next((entry for entry in source if entry["role"] == role), {})

        all_proposals_for_critique = [
            {
                "role": entry["role"],
                "proposal": json.dumps(entry.get("action_dict", {})),
            }
            for entry in source
        ]

        roles = config.get("roles", ["macro", "value", "risk"])
        i = roles.index(role) if role in roles else 0
        if not config.get("console_display"):
            print(f"  [Round {current_round} - Critique] {role.upper()} agent ({i+1}/{len(source)})...", flush=True)
        my_proposal = json.dumps(p.get("action_dict", {}))

        use_profiles = _has_agent_profiles(config)
        if use_profiles:
            user_secs = _get_user_sections_from_profile(config, role, "critique")
        else:
            _profile = resolve_prompt_profile(config, role, "critique")
            user_secs = _profile.get("user_sections")

        sector_text = build_sector_constraint_text(
            config.get("sector_config"), role,
        )
        prompt = build_critique_prompt(
            role, context, all_proposals_for_critique, my_proposal,
            prompt_file_overrides=config.get("prompt_file_overrides"),
            user_sections=user_secs,
            sector_constraints=sector_text,
        )

        if use_profiles:
            build_result = _build_system_prompt_from_profile(config, role, "critique", prompt)
        else:
            system_blocks = _profile.get("system_blocks")
            registry = get_registry(config)
            build_result = registry.build(
                role=role, phase="critique",
                beta=resolve_beta(config, "critique"),
                user_prompt=prompt,
                block_order=system_blocks,
                prompt_file_overrides=config.get("prompt_file_overrides"),
            )
        system_msg = build_result.system_prompt

        _log_prompt(config, role, "critique", current_round, system_msg, prompt)
        _capture = config.get("_prompt_capture")
        if _capture:
            _capture("critiques", role, system_msg, prompt)

        raw_text = None
        if is_mock:
            result = _mock_critique(role, source)
            raw_text = json.dumps(result, indent=2)
        else:
            raw_text = _call_llm_with_lifecycle(config, system_msg, prompt, role, "critique")
            result = _parse_json(raw_text)

        if config.get("verbose"):
            _verbose_critique(role, result)

        if not config.get("console_display"):
            _print_critique_summary(role, result)

        critique = {
            "role": role,
            "critiques": result.get("critiques", []),
            "self_critique": result.get("self_critique", ""),
        }

        turn = {
            "round": current_round,
            "agent_id": f"agent_{role}",
            "role": role,
            "type": "critique",
            "content": result,
            "raw_system_prompt": system_msg,
            "raw_user_prompt": prompt,
            "raw_response": raw_text,
            "input_params": {
                "context": context,
                "all_proposals_for_critique": all_proposals_for_critique,
                "my_proposal": my_proposal,
            },
        }

        return {
            "critiques": [critique],
            "debate_turns": [turn],
        }

    _critique.__name__ = f"critique_{role}"
    return _critique


def make_revise_node(role: str):
    """Factory: create a per-agent revise node for the given role.

    Extracts the loop body from revise_node.  Collects critiques
    targeted at this role.  Returns single-element lists.
    Does NOT return current_round.
    """

    def _revise(state: ParallelRoundState) -> dict:
        config = state["config"]
        if config.get("propose_only"):
            return {}
        context = state["enriched_context"]
        current_round = state.get("current_round", 1)
        is_mock = config.get("mock", False)
        obs = state["observation"]
        all_critiques = state.get("critiques", [])

        # Sort by config role order for deterministic behavior (operator.add
        # merge order is non-deterministic).
        roles = config.get("roles", ["macro", "value", "risk"])
        role_order = {r: i for i, r in enumerate(roles)}
        raw_source = state.get("revisions") if state.get("revisions") else state["proposals"]
        source = sorted(raw_source, key=lambda e: role_order.get(e["role"], len(roles)))

        # Find own entry by role (safe lookup — missing role returns empty dict)
        p = next((entry for entry in source if entry["role"] == role), {})

        roles = config.get("roles", ["macro", "value", "risk"])
        i = roles.index(role) if role in roles else 0
        if not config.get("console_display"):
            print(f"  [Round {current_round} - Revise] {role.upper()} agent ({i+1}/{len(source)})...", flush=True)
        my_proposal = json.dumps(p.get("action_dict", {}))

        # Collect critiques targeted at this role.
        # Normalize target_role: LLMs may return "MACRO", "Risk Agent", etc.
        critiques_received = []
        for c in all_critiques:
            for crit in c.get("critiques", []):
                target = crit.get("target_role", "").lower().split()[0]
                if target == role:
                    critiques_received.append({
                        "from_role": c["role"],
                        "objection": crit.get("objection", ""),
                        "falsifier": crit.get("falsifier"),
                    })

        use_profiles = _has_agent_profiles(config)
        if use_profiles:
            user_secs = _get_user_sections_from_profile(config, role, "revise")
        else:
            _profile = resolve_prompt_profile(config, role, "revise")
            user_secs = _profile.get("user_sections")

        sector_text = build_sector_constraint_text(
            config.get("sector_config"), role,
        )
        prompt = build_revision_prompt(
            role, context, my_proposal, critiques_received,
            prompt_file_overrides=config.get("prompt_file_overrides"),
            user_sections=user_secs,
            sector_constraints=sector_text,
        )

        if use_profiles:
            build_result = _build_system_prompt_from_profile(config, role, "revise", prompt)
        else:
            system_blocks = _profile.get("system_blocks")
            registry = get_registry(config)
            build_result = registry.build(
                role=role, phase="revise",
                beta=resolve_beta(config, "revise"),
                user_prompt=prompt,
                block_order=system_blocks,
                prompt_file_overrides=config.get("prompt_file_overrides"),
            )
        system_msg = build_result.system_prompt

        _log_prompt(config, role, "revise", current_round, system_msg, prompt)
        _capture = config.get("_prompt_capture")
        if _capture:
            _capture("revisions", role, system_msg, prompt)

        if is_mock:
            result = _mock_revision(role, p.get("action_dict", {}), obs, config)
            raw_text = json.dumps(result, indent=2)
        else:
            raw_text = _call_llm_with_lifecycle(config, system_msg, prompt, role, "revise")
            result = _parse_json(raw_text)

        universe = obs.get("universe", [])
        raw_alloc = result.get("allocation", p.get("action_dict", {}).get("allocation", {}))
        raw_alloc = _apply_sector_permissions(raw_alloc, role, config)

        # Pull constraints from config
        ac = config.get("allocation_constraints") or {}
        action_dict = {
            "allocation": normalize_allocation(
                raw_alloc, universe,
                max_weight=ac.get("max_weight", 1.0),
                min_holdings=ac.get("min_holdings", 1),
            ),
            "justification": result.get("justification", ""),
            "confidence": result.get("confidence", 0.5),
            "claims": result.get("claims", []),
        }

        if config.get("verbose"):
            _verbose_revision(role, result)

        if not config.get("console_display"):
            _print_allocation(role, action_dict, "revised")

        revision = {
            "role": role,
            "action_dict": action_dict,
            "revision_notes": result.get("revision_notes", ""),
            "raw_response": raw_text,
        }

        turn = {
            "round": current_round,
            "agent_id": f"agent_{role}",
            "role": role,
            "type": "revision",
            "content": result,
            "raw_system_prompt": system_msg,
            "raw_user_prompt": prompt,
            "raw_response": raw_text,
            "input_params": {
                "context": context,
                "my_proposal": my_proposal,
                "critiques_received": critiques_received,
            },
        }

        return {
            "revisions": [revision],
            "debate_turns": [turn],
        }

    _revise.__name__ = f"revise_{role}"
    return _revise


# =============================================================================
# CONDITIONAL EDGE + JUDGE + AGGREGATION + TRACE BUILDERS
# =============================================================================


def should_continue(state: DebateState) -> str:
    """Conditional edge: loop back to critique or proceed to judge."""
    current_round = state.get("current_round", 2)
    max_rounds = state.get("config", {}).get("max_rounds", 1)
    if current_round <= max_rounds:
        return "critique"
    return "judge"


def judge_node(state: DebateState) -> dict:
    """Judge synthesizes the debate into a single final trading decision."""
    config = state["config"]
    judge_type = config.get("judge_type", "llm")

    if not config.get("console_display"):
        print(f"  [Judge] Synthesizing final decision ({judge_type})...", flush=True)

    context = state["enriched_context"]
    revisions = state.get("revisions") or state.get("proposals", [])
    all_critiques = state.get("critiques", [])
    is_mock = config.get("mock", False)
    obs = state["observation"]
    universe = obs.get("universe", [])

    if judge_type == "average":
        # Compute unweighted average of all allocations
        total_alloc: dict[str, float] = {t: 0.0 for t in universe}
        total_conf = 0.0
        n = len(revisions)
        for r in revisions:
            alloc = r.get("action_dict", {}).get("allocation", {})
            for t, w in alloc.items():
                total_alloc[t] = total_alloc.get(t, 0.0) + w
            total_conf += r.get("action_dict", {}).get("confidence", 0.5)

        assert n > 0, f"No revisions found in state: {state}"
        avg_alloc = {t: w / n for t, w in total_alloc.items()}
        avg_conf = total_conf / n

        # Pull constraints from config
        ac = config.get("allocation_constraints") or {}
        final_action = {
            "allocation": normalize_allocation(
                avg_alloc, universe,
                max_weight=ac.get("max_weight", 1.0),
                min_holdings=ac.get("min_holdings", 1),
            ),
            "justification": f"Simple average of {n} agent allocations.",
            "confidence": avg_conf,
            "claims": [],
        }

        if not config.get("console_display"):
            _print_allocation("judge", final_action, "FINAL")

        return {
            "final_action": final_action,
            "strongest_objection": "",
            "audited_memo": "Simple average of agent allocations.",
            "debate_turns": [],  # No LLM call for average
        }

    # Default: LLM judge
    # Format critiques for the judge
    critiques_text = "\n".join(
        f"[{c['role']} -> {crit.get('target_role', '?')}]: {crit.get('objection', '')}"
        for c in all_critiques
        for crit in c.get("critiques", [])
    )

    revisions_for_judge = [
        {
            "role": r["role"],
            "action": json.dumps(r.get("action_dict", {})),
            "confidence": r.get("action_dict", {}).get("confidence", 0.5),
        }
        for r in revisions
    ]

    use_profiles = _has_agent_profiles(config)
    if use_profiles:
        user_secs = _get_user_sections_from_profile(config, "judge", "judge")
    else:
        profile = resolve_prompt_profile(config, "judge", "judge")
        user_secs = profile.get("user_sections")

    sector_text = build_sector_constraint_text(
        config.get("sector_config"), "judge", include_permissions=False,
    )
    prompt = build_judge_prompt(
        context, revisions_for_judge, critiques_text,
        prompt_file_overrides=config.get("prompt_file_overrides"),
        user_sections=user_secs,
        sector_constraints=sector_text,
    )

    if use_profiles:
        build_result = _build_system_prompt_from_profile(config, "judge", "judge", prompt)
    else:
        system_blocks = profile.get("system_blocks")
        registry = get_registry(config)
        build_result = registry.build(
            role="judge", phase="judge",
            beta=resolve_beta(config, "judge"),
            user_prompt=prompt,
            block_order=system_blocks,
            prompt_file_overrides=config.get("prompt_file_overrides"),
        )
    system_msg = build_result.system_prompt

    _log_prompt(config, "judge", "judge", 0, system_msg, prompt)
    _capture = config.get("_prompt_capture")
    if _capture:
        _capture("final", "judge", system_msg, prompt)

    raw_text = None  # Raw LLM output for eval module
    if is_mock:
        result = _mock_judge(revisions, config)
        raw_text = json.dumps(result, indent=2)
    else:
        raw_text = _call_llm_with_lifecycle(config, system_msg, prompt, "judge", "judge")
        result = _parse_json(raw_text)

    if config.get("verbose"):
        _verbose_judge(result)

    raw_alloc = result.get("allocation", {})

    # Pull constraints from config
    ac = config.get("allocation_constraints") or {}
    alloc = normalize_allocation(
        raw_alloc, universe,
        max_weight=ac.get("max_weight", 1.0),
        min_holdings=ac.get("min_holdings", 1),
    )
    alloc = _apply_sector_limits(alloc, config)
    final_action = {
        "allocation": alloc,
        "justification": result.get("audited_memo", result.get("justification", "")),
        "confidence": result.get("confidence", 0.5),
        "claims": result.get("claims", []),
    }
    if not config.get("console_display"):
        _print_allocation("judge", final_action, "FINAL")

    turns = [
        {
            "round": state.get("current_round", 2),
            "agent_id": "judge",
            "role": "judge",
            "type": "judge_decision",
            "content": result,
            "raw_system_prompt": system_msg,
            "raw_user_prompt": prompt,
            "raw_response": raw_text,
            "input_params": {
                "context": context,
                "revisions_for_judge": revisions_for_judge,
                "critiques_text": critiques_text
            }
        }
    ]

    return {
        "final_action": final_action,
        "strongest_objection": result.get("strongest_objection", ""),
        "audited_memo": result.get("audited_memo", ""),
        "debate_turns": turns,
    }


def _get_vote_direction(proposals: list, ticker: str) -> str:
    """Count buy/sell votes for a ticker across all proposals. Majority wins; ties = hold.

    Each agent gets exactly one vote per ticker (first matching order wins)
    to prevent multi-order double-counting.
    """
    buy_count = 0
    sell_count = 0
    for p in proposals:
        for o in p.get("action_dict", {}).get("orders", []):
            if o.get("ticker") == ticker:
                side = o.get("side")
                if side == "buy":
                    buy_count += 1
                elif side == "sell":
                    sell_count += 1
                break  # one vote per agent per ticker
    if buy_count > sell_count:
        return "buy"
    elif sell_count > buy_count:
        return "sell"
    return "hold"


def _get_median_size(proposals: list, ticker: str, side: str) -> float:
    """Median order size for a ticker+direction across proposals."""
    sizes = []
    for p in proposals:
        for o in p.get("action_dict", {}).get("orders", []):
            if o.get("ticker") == ticker and o.get("side") == side:
                sizes.append(o.get("size", 0))
    if not sizes:
        return 0.0
    sizes.sort()
    n = len(sizes)
    if n % 2 == 1:
        return float(sizes[n // 2])
    return (sizes[n // 2 - 1] + sizes[n // 2]) / 2.0


def aggregate_proposals_node(state: DebateState) -> dict:
    """LangGraph node: aggregate proposals by majority vote + median sizing."""
    proposals = state.get("proposals", [])
    obs = state["observation"]
    tickers = obs.get("universe", [])

    # Build aggregated orders
    orders = []
    for ticker in tickers:
        direction = _get_vote_direction(proposals, ticker)
        if direction in ("buy", "sell"):
            size = _get_median_size(proposals, ticker, direction)
            if size > 0:
                orders.append({"ticker": ticker, "side": direction, "size": size})

    # Detect disagreements
    disagreements = []
    for ticker in tickers:
        votes = []
        for p in proposals:
            for o in p.get("action_dict", {}).get("orders", []):
                if o.get("ticker") == ticker:
                    votes.append(f"{p['role']}:{o['side']}{o.get('size', 0)}")
        sides = set()
        for p in proposals:
            for o in p.get("action_dict", {}).get("orders", []):
                if o.get("ticker") == ticker:
                    sides.add(o.get("side"))
        if len(sides) > 1:
            disagreements.append(f"{ticker}: {' vs '.join(votes)}")

    # Find strongest objection: lowest-confidence agent who disagreed with consensus
    strongest_objection = ""
    min_conf = 1.0
    for ticker in tickers:
        direction = _get_vote_direction(proposals, ticker)
        if direction == "hold":
            continue  # no consensus to dissent from
        for p in proposals:
            for o in p.get("action_dict", {}).get("orders", []):
                if o.get("ticker") == ticker and o.get("side") != direction:
                    conf = p.get("action_dict", {}).get("confidence", 0.5)
                    if conf < min_conf:
                        min_conf = conf
                        justification = p.get("action_dict", {}).get("justification", "")
                        strongest_objection = (
                            f"[{p['role']}] (conf={conf:.2f}) dissented: {justification}"
                        )

    # Merge justifications
    merged_justification = " | ".join(
        f"[{p['role']}] {p.get('action_dict', {}).get('justification', '')}"
        for p in proposals
    )

    # Merge claims (deduplicated by claim_text)
    seen_claims = set()
    merged_claims = []
    for p in proposals:
        for c in p.get("action_dict", {}).get("claims", []):
            text = c.get("claim_text", "")
            if text and text not in seen_claims:
                seen_claims.add(text)
                merged_claims.append(c)

    # Average confidence
    confidences = [p.get("action_dict", {}).get("confidence", 0.5) for p in proposals]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

    final_action = {
        "orders": orders,
        "justification": merged_justification,
        "confidence": round(avg_confidence, 4),
        "claims": merged_claims,
    }

    return {
        "final_action": final_action,
        "strongest_objection": strongest_objection,
        "audited_memo": (
            f"Majority vote: {len(proposals)} agents. "
            + (f"Disagreements: {'; '.join(disagreements)}" if disagreements else "Unanimous.")
        ),
    }


def build_mv_trace_node(state: DebateState) -> dict:
    """LangGraph node: build AgentTrace for majority_vote architecture."""
    print("  [Trace] Building majority-vote trace...", flush=True)
    obs = state["observation"]
    final = state.get("final_action", {})
    config = state.get("config", {})

    orders_desc = (
        "; ".join(
            f"{o.get('side', '?')} {o.get('size', 0)} {o.get('ticker', '?')}"
            for o in final.get("orders", [])
        )
        or "Hold"
    )

    roles = config.get("roles", [])
    trace = {
        "observation_timestamp": obs.get("timestamp", ""),
        "architecture": "majority_vote",
        "what_i_saw": state.get("enriched_context", "")[:500] + "...",
        "hypothesis": (
            f"Majority vote: {len(roles)} agents ({', '.join(roles)}) "
            f"with vote-based aggregation and median sizing"
        ),
        "decision": orders_desc,
        "risks_or_falsifiers": state.get("audited_memo", "")[:500],
        "strongest_objection": state.get("strongest_objection", ""),
        "debate_turns": state.get("debate_turns", []),
        "action": final,
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }

    return {"trace": trace}


def build_trace_node(state: DebateState) -> dict:
    """Construct the final AgentTrace from the accumulated debate state."""
    print("  [Trace] Building auditable trace...", flush=True)
    obs = state["observation"]
    final = state.get("final_action", {})
    config = state.get("config", {})

    orders_desc = (
        "; ".join(
            f"{o.get('side', '?')} {o.get('size', 0)} {o.get('ticker', '?')}"
            for o in final.get("orders", [])
        )
        or "Hold"
    )

    roles = config.get("roles", [])
    trace = {
        "observation_timestamp": obs.get("timestamp", ""),
        "architecture": "debate",
        "what_i_saw": state.get("enriched_context", "")[:500] + "...",
        "hypothesis": (
            f"Multi-agent debate: {len(roles)} agents ({', '.join(roles)}), "
            f"rounds={config.get('max_rounds', 1)}"
        ),
        "decision": orders_desc,
        "risks_or_falsifiers": state.get("audited_memo", "")[:500],
        "strongest_objection": state.get("strongest_objection", ""),
        "debate_turns": state.get("debate_turns", []),
        "action": final,
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }

    return {"trace": trace}
