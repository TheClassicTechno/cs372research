def detect_state_mutations_outside_state_file(text: str, rel: str) -> None:
    """
    Enforce single mutation boundary.

    Only state.js may mutate shared state objects.

    Detects mutations to:
        appState.*
        runsViewState.*
        liveState.*
    """

    if rel.endswith("state.js"):
        return

    mutation_patterns = [
        r"\bappState\.[A-Za-z_$][\w$]*\s*=",
        r"\brunsViewState\.[A-Za-z_$][\w$]*\s*=",
        r"\bliveState\.[A-Za-z_$][\w$]*\s*=",

        r"\bappState\.[A-Za-z_$][\w$]*\s*\+=",
        r"\bappState\.[A-Za-z_$][\w$]*\s*\-=",

        r"\bappState\.[A-Za-z_$][\w$]*\.push\(",
        r"\bappState\.[A-Za-z_$][\w$]*\.splice\(",
        r"\bappState\.[A-Za-z_$][\w$]*\.add\(",
        r"\bappState\.[A-Za-z_$][\w$]*\.delete\(",
    ]

    for pattern in mutation_patterns:
        for match in re.finditer(pattern, text):

            line = text[:match.start()].count("\n") + 1

            violations.append(
                f"{rel}:{line}: shared state mutation detected outside state.js"
            )