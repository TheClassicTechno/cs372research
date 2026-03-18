# Claude Prompt Contract

You MUST follow all rules in claude_rules/.

If a request violates any invariant:
- STOP
- Explain the violation
- Propose a safe alternative

You MUST NOT:
- introduce silent failures
- modify unrelated files
- create hidden state
- bypass validation

All outputs must:
- be deterministic
- be testable
- include validation
