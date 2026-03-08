# AI Development Workflow Rules

These rules define the **standard workflow for AI coding assistants (Claude, GPT, etc.)** when modifying this repository.

The goal is to ensure:

* safe modifications
* architectural compliance
* reliable test coverage
* minimal human interruption
* deterministic development behavior

AI assistants must follow this workflow **for every code change**.

---

# Core Development Loop

All development must follow the following loop:

```
1. Modify code
2. Run architecture checker
3. Run tests
4. Fix violations
5. Repeat until no violations remain
```

No code change is considered complete until:

* architecture rules pass
* all tests pass
* no violations remain

---

# Rule: Read All Rules First

Before modifying code, the AI assistant **must read all documentation inside**:

```
rules/
```

This directory contains the authoritative specifications for:

* architecture rules
* testing contracts
* dashboard constraints
* coding standards
* development workflow

AI assistants must **treat these documents as binding specifications**.

Do not rely on memory or assumptions — always consult the rules.

---

# Rule: Minimal Permission Requests

AI assistants must **minimize interruptions to the developer**.

Do **not repeatedly ask for permission** for routine development actions.

Instead:

* assume permission for normal development tasks
* request clarification only when necessary

Examples of actions that **do NOT require permission**:

* modifying code files
* running architecture checks
* running tests
* fixing lint or rule violations
* refactoring small functions
* reorganizing code within a file

Permission should only be requested for:

* large architectural changes
* modifying multiple modules
* introducing new dependencies
* altering public interfaces
* modifying directory structure

---

# Rule: Never Commit Code

AI assistants must **never create commits**.

The developer is responsible for committing code.

AI assistants may:

* modify files
* propose commit messages
* show diffs

But **must not run**:

```
git commit
git push
git merge
```

---

# Rule: Continuous Self-Correction

After modifying code, the AI assistant must **verify correctness**.

Mandatory checks:

1. Run architecture checker
2. Run test suite
3. Fix all violations
4. Repeat until clean

The assistant must **not stop after the first fix attempt**.

Continue iterating until:

* architecture checker passes
* tests pass
* no rule violations remain

---

# Rule: Architecture Checker Enforcement

After any modification, the assistant must run the architecture validation tool.

Example:

```
npm run check-architecture
```

or equivalent repository script.

Violations must be resolved **before continuing**.

---

# Rule: Tests Are Mandatory

Every feature must include **corresponding tests**.

The preferred testing hierarchy:

1. **Integration tests (Playwright)** for UI behavior
2. **Unit tests** for deterministic logic
3. **Snapshot tests** when useful

The assistant must ensure:

* new features are testable
* `data-testid` attributes exist where required
* tests do not rely on fragile selectors

---

# Rule: Prefer Integration Tests

Because the dashboard is a **UI debugging tool**, integration tests are preferred.

Example:

```
tests/dashboard/
```

Tests should verify:

* UI renders expected sections
* runs load correctly
* navigation works
* dashboard state updates correctly

---

# Rule: Do Not Perform Unrequested Refactors

Large-scale refactors must **never occur implicitly**.

If a modification requires changing **more than three modules**, the assistant must:

1. explain the refactor
2. list affected files
3. confirm behavior preservation

Do not perform repository-wide changes without explicit instruction.

---

# Rule: Preserve Existing Behavior

Unless explicitly instructed, all changes must be **behavior-preserving**.

The dashboard is a **diagnostic tool**, so stability is critical.

Avoid introducing breaking changes.

---

# Rule: Follow Architecture Document

All code must comply with the architecture rules defined in:

```
Debate Dashboard Architecture
```

This includes:

* layer boundaries
* component purity
* DOM contract
* event handling rules
* state management rules

Violations must be fixed immediately.

---

# Rule: Prefer Small Changes

AI assistants should prefer:

* incremental changes
* small patches
* localized edits

Avoid rewriting entire files unless absolutely necessary.

---

# Rule: Fail Fast

Do not introduce defensive defaults.

If required data is missing:

```
throw new Error(...)
```

Silent failures are prohibited.

---

# Rule: Single Source of Truth

Before implementing a new function, the assistant must check whether a similar function already exists.

If one exists:

* reuse it

If multiple variants exist:

* consolidate them

Avoid duplicated implementations.

---

# Rule: Deterministic Logging

All runtime outputs used by the dashboard must come from **structured files**.

Prefer:

```
logging/runs/<experiment>/<run_id>/
```

over in-memory state.

---

# Rule: Deterministic File Writes

Files written by the system must follow deterministic naming.

Example:

```
round_001.json
round_002.json
```

Avoid dynamic names such as:

```
temp.json
latest.json
```

---

# Rule: Explicit Errors Over Console Logs

Errors must be surfaced through exceptions.

Avoid patterns such as:

```
console.error(...)
return null
```

Instead:

```
throw new Error(...)
```

---

# Rule: Documentation Updates

If the assistant changes behavior, it must update:

* architecture documentation
* comments
* test expectations

Documentation must stay consistent with the code.

---

# Summary

AI assistants modifying this repository must follow this workflow:

```
READ RULES
MODIFY CODE
RUN ARCHITECTURE CHECKER
RUN TESTS
FIX VIOLATIONS
REPEAT
```

Key principles:

* minimize developer interruptions
* never commit code
* preserve architecture
* enforce tests
* fail fast on errors

This workflow ensures the repository remains **stable, deterministic, and safe for AI-assisted development**.
