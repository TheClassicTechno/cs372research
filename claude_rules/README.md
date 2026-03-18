# Claude Code Guardrails

This directory defines the rules, constraints, and architectural guardrails
for LLM-generated code in this repository.

Claude is treated as an untrusted code generator.

All generated code must comply with:
- invariants.md
- anti_patterns.md
- architecture.md
- prompt_contract.md

Violations must result in rejection and correction.

This system exists to prevent:
- silent failures
- spaghetti code
- hidden state bugs
- non-deterministic behavior
