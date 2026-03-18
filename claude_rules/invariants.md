# ============================================================
# Claude Guardrails — Semgrep Ruleset
# Enforces invariants from claude_rules/invariants.md
# ============================================================

rules:

# ============================================================
# 1. NO SILENT FAILURE
# ============================================================

- id: no-bare-except
  patterns:
    - pattern: |
        try:
          ...
        except:
          ...
  message: "Bare except is forbidden (Invariant: No silent failure)"
  severity: ERROR
  languages: [python]

- id: no-silent-except-pass
  patterns:
    - pattern: |
        try:
          ...
        except ...:
          pass
  message: "Silent exception handling (pass) is forbidden"
  severity: ERROR
  languages: [python]

- id: no-except-without-raise-or-log
  patterns:
    - pattern: |
        try:
          ...
        except $E:
          ...
    - pattern-not: |
        try:
          ...
        except $E:
          raise
    - pattern-not: |
        try:
          ...
        except $E:
          logging.$LOG(...)
    - pattern-not: |
        try:
          ...
        except $E:
          logger.$LOG(...)
  message: "Exceptions must be re-raised or logged"
  severity: WARNING
  languages: [python]


# ============================================================
# 2. DETERMINISM (NO UNSEEDED RANDOMNESS)
# ============================================================

- id: no-random-without-seed
  patterns:
    - pattern-either:
        - pattern: random.random(...)
        - pattern: np.random.$FUNC(...)
    - pattern-not-inside: |
        random.seed(...)
    - pattern-not-inside: |
        np.random.seed(...)
  message: "Randomness must be seeded for determinism"
  severity: ERROR
  languages: [python]


# ============================================================
# 3. NO IMPLICIT STATE MUTATION
# ============================================================

- id: no-global-state
  pattern: global $X
  message: "Global state usage is forbidden"
  severity: ERROR
  languages: [python]

- id: no-mutable-default-args
  patterns:
    - pattern: |
        def $FUNC(..., $ARG=[], ...):
            ...
    - pattern-either:
        - pattern: |
            def $FUNC(..., $ARG={}, ...):
                ...
  message: "Mutable default arguments are forbidden"
  severity: ERROR
  languages: [python]

- id: no-hidden-class-state-mutation
  patterns:
    - pattern: |
        self.$X = ...
  message: "Class state mutation detected — ensure explicit design"
  severity: WARNING
  languages: [python]


# ============================================================
# 4. EXPLICIT I/O (NO HIDDEN SIDE EFFECTS)
# ============================================================

- id: no-file-write-without-context
  pattern: open($FILE, "w")
  message: "File writes must use context manager and be logged"
  severity: WARNING
  languages: [python]

- id: no-print-debugging
  pattern: print(...)
  message: "Use structured logging instead of print"
  severity: WARNING
  languages: [python]


# ============================================================
# 5. TYPE SAFETY
# ============================================================

- id: missing-type-annotations
  patterns:
    - pattern: |
        def $FUNC(...):
            ...
    - pattern-not: |
        def $FUNC(...) -> ...:
            ...
  message: "Function must have return type annotation"
  severity: WARNING
  languages: [python]

- id: no-any-type
  pattern: Any
  message: "Avoid use of Any — enforce strict typing"
  severity: WARNING
  languages: [python]


# ============================================================
# 6. VALIDATION REQUIRED BEFORE RETURN
# ============================================================

- id: return-without-validation
  patterns:
    - pattern: |
        def $FUNC(...):
            ...
            return $X
    - pattern-not: |
        def $FUNC(...):
            ...
            assert ...
            return $X
    - pattern-not: |
        def $FUNC(...):
            ...
            validate($X)
            return $X
  message: "Return values must be validated (assert or validate())"
  severity: WARNING
  languages: [python]


# ============================================================
# 7. COMPLEXITY / SPAGHETTI GUARDRAILS
# ============================================================

- id: too-many-nested-blocks
  patterns:
    - pattern: |
        if ...:
            if ...:
                if ...:
                    if ...:
                        ...
  message: "Nesting depth > 3 detected (spaghetti risk)"
  severity: WARNING
  languages: [python]

- id: long-function-warning
  patterns:
    - pattern: |
        def $FUNC(...):
            ...
  message: "Check function length manually (enforce via AST tool)"
  severity: INFO
  languages: [python]


# ============================================================
# 8. MAGIC NUMBERS
# ============================================================

- id: magic-number-detected
  pattern: |
    $X = $NUM
  metavariable-pattern:
    metavariable: $NUM
    pattern: |
      [0-9]{2,}
  message: "Magic number detected — define named constant"
  severity: WARNING
  languages: [python]


# ============================================================
# 9. FORBIDDEN PATTERNS FROM anti_patterns.md
# ============================================================

- id: implicit-type-cast
  pattern: |
    int($X) + $Y
  message: "Implicit type casting detected — enforce explicit types"
  severity: WARNING
  languages: [python]

- id: side-effect-in-utility
  patterns:
    - pattern: |
        def $FUNC(...):
            ...
            open(...)
  message: "Utility function performing I/O side effects"
  severity: WARNING
  languages: [python]


# ============================================================
# END OF RULESET
# ============================================================