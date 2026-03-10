#!/usr/bin/env python3
"""
Lightweight architecture validation for the Debate Dashboard.

Purpose
-------
This script enforces the architectural contract for the dashboard's frontend
code under tools/prompt_viewer/static/js.

It is intentionally heuristic-based rather than a full JavaScript parser.
The goal is to catch the most common architectural violations introduced by
AI assistants or careless refactors before they land in the codebase.

What it checks
--------------
- Layer boundary violations
- Forbidden imports
- Forbidden DOM access in lower layers
- Forbidden fetch/network calls in lower layers
- Inline event handlers
- window/global assignments
- HTML construction in views
- Missing file-level documentation comments
- Missing function documentation comments
- Oversized files and functions
- Missing viewToken guard patterns in async view code
- Missing data-testid usage in component markup
- App.js becoming a "god file"
- Duplicate function names across files
- Suspicious silent-failure patterns
- Dangerous fallback/default patterns
- Framework/build-tool introductions
- Non-deterministic or temp-like file naming strings
- Optional warning on console-heavy debugging

Additional style / discipline checks
------------------------------------
- Imperative accumulation loops where map/filter/reduce would be clearer
- Manual reductions
- Index-based loops over arrays
- Excessive mutation density
- Nested imperative loops
- Module-level mutable state
- Render purity checks
- Shared-state mutation auditing
- Side-effect detection inside components and render-like functions

Exit code
---------
0 on pass
1 on any failure
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict

# NOTE:
# This file lives at tools/dashboard/rules/.
# Frontend JS lives at tools/dashboard/static/js.
ROOT = pathlib.Path(__file__).resolve().parent.parent / "static" / "js"

failures: list[str] = []
warnings: list[str] = []

# Track function names across files to detect duplication.
function_index: dict[str, list[str]] = defaultdict(list)

# Thresholds
MAX_FILE_LINES = 400
MAX_APP_LINES = 150
MAX_FUNC_LINES = 30
MAX_CONSOLE_STATEMENTS_PER_FILE = 5
MAX_MUTATION_EVENTS_PER_FILE = 12
MAX_MUTATION_EVENTS_PER_FUNCTION = 5

# Allowed trivial scaffolding tags in views.
TRIVIAL_VIEW_TAGS = ("div", "span")

# View-specific action names app.js should not know about.
FORBIDDEN_APP_ACTION_NAMES = {
    "load-agents",
    "load-file",
    "open-run",
    "open-detail",
    "expand-round",
    "toggle-tree",
    "load-round",
}

# Words that suggest accidental framework introduction.
FORBIDDEN_FRAMEWORK_IMPORTS = (
    "react",
    "preact",
    "vue",
    "svelte",
    "solid-js",
    "lit",
    "next/",
    "nuxt",
)

# Suspicious temp-ish names in strings or code.
SUSPICIOUS_TEMP_NAMES = (
    "latest.json",
    "temp.json",
    "tmp.json",
    "debug.json",
    "output.json",
)

# Allowed state mutation zone
STATE_FILE = "state.js"

# Functional-style ops
FUNCTIONAL_ARRAY_OPS = (
    ".map(",
    ".filter(",
    ".reduce(",
    ".flatMap(",
    ".some(",
    ".every(",
    ".find(",
    ".findIndex(",
    ".sort(",
)

# Things that smell like side effects.
SIDE_EFFECT_TOKENS = (
    "fetch(",
    "localStorage.",
    "sessionStorage.",
    "document.cookie",
    "window.location",
    "history.pushState(",
    "history.replaceState(",
    "setTimeout(",
    "setInterval(",
    "requestAnimationFrame(",
    "addEventListener(",
    "removeEventListener(",
    "dispatchEvent(",
)

DOM_WRITE_TOKENS = (
    ".innerHTML",
    ".outerHTML",
    ".insertAdjacentHTML",
    ".appendChild(",
    ".prepend(",
    ".append(",
    ".replaceChildren(",
    ".remove(",
    ".removeChild(",
    ".classList.",
    ".style.",
    ".textContent =",
    ".innerText =",
    ".value =",
    ".checked =",
    ".disabled =",
    ".setAttribute(",
    ".removeAttribute(",
)

STATE_MUTATION_TOKENS = (
    "appState.",
    "state.",
    "runsViewState.",
    "liveState.",
)


def strip_js_comments(text: str) -> str:
    """Remove JS comments to reduce false positives in regex checks."""
    # Remove block comments first, then line comments.
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    return text


def count_lines(text: str) -> int:
    """Return line count for a text blob."""
    return text.count("\n") + 1


def relative_import_target(import_path: str) -> str:
    """Normalize import path for reporting."""
    return import_path.replace("\\", "/")


def find_import_paths(text: str) -> list[str]:
    """Extract JS import paths from ESM import statements."""
    return re.findall(r"""from\s+["']([^"']+)["']""", text)


def has_project_import(text: str) -> bool:
    """Detect whether a utils file imports from anywhere at all."""
    return bool(re.search(r"""\bimport\s+.*?\bfrom\s+["'][^"']+["']""", text, flags=re.S))


def contains_html_like_markup(text: str) -> bool:
    """Detect whether a file likely contains HTML strings/template literals."""
    return bool(re.search(r"""[`'"][^`'"]*<\s*[a-zA-Z]""", text))


def contains_nontrivial_html_markup(text: str) -> bool:
    """
    Detect likely non-trivial HTML generation in views.

    Allows trivial container scaffolding like:
      <div id="x"></div>
      <span class="x"></span>
    Warns on richer HTML.
    """
    matches = re.findall(r"""[`'"]([^`'"]*<\s*([a-zA-Z][\w-]*))""", text)
    for full_match, tag in matches:
        if tag.lower() not in TRIVIAL_VIEW_TAGS:
            return True
        # If trivial tags include nested markup or data rendering, still warn.
        if "${" in full_match:
            return True
    return False


def find_function_defs(text: str) -> list[tuple[str, int, int]]:
    """
    Find likely JS function definitions and estimate their line spans.

    Returns tuples of:
      (function_name, start_line, end_line)
    """
    results: list[tuple[str, int, int]] = []

    patterns = [
        # function foo(...) {
        re.compile(r"""function\s+([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{"""),
        # const foo = (...) => {
        re.compile(r"""(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>\s*\{"""),
        # const foo = async function(...) {
        re.compile(r"""(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*async\s+function\s*\([^)]*\)\s*\{"""),
        # export function foo(...) {
        re.compile(r"""export\s+function\s+([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{"""),
        # export const foo = (...) => {
        re.compile(r"""export\s+(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>\s*\{"""),
    ]

    lines = text.splitlines()
    for i, line in enumerate(lines):
        for pattern in patterns:
            m = pattern.search(line)
            if not m:
                continue
            name = m.group(1)
            start = i + 1

            # Estimate function end by brace counting from current line onward.
            brace_count = line.count("{") - line.count("}")
            end = start
            for j in range(i + 1, len(lines)):
                brace_count += lines[j].count("{")
                brace_count -= lines[j].count("}")
                end = j + 1
                if brace_count <= 0:
                    break
            results.append((name, start, end))
            break

    return results


def get_line_range(text: str, start_line: int, end_line: int) -> str:
    """Return source for the given 1-indexed line range."""
    lines = text.splitlines()
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line)
    return "\n".join(lines[start_idx:end_idx])


def has_file_doc_comment(raw_text: str) -> bool:
    """Require a top-of-file block comment or doc comment."""
    stripped = raw_text.lstrip()
    return stripped.startswith("/**") or stripped.startswith("/*")


def has_doc_comment_above_function(raw_text: str, start_line: int) -> bool:
    """
    Heuristically require a doc comment above a function.

    Looks at the preceding few non-empty lines for '/**'.
    """
    lines = raw_text.splitlines()
    idx = start_line - 2  # line above function
    lookback = 5
    snippet = []
    while idx >= 0 and lookback > 0:
        line = lines[idx].strip()
        if line:
            snippet.append(line)
            lookback -= 1
        idx -= 1
    joined = "\n".join(reversed(snippet))
    return "/**" in joined


def require_viewtoken_guard(text: str) -> bool:
    """
    Heuristic: if a view file contains async functions and DOM writes,
    it should reference appState.viewToken or a token guard.
    """
    has_async = bool(re.search(r"""\basync\b""", text))
    has_dom_write = any(token in text for token in DOM_WRITE_TOKENS)
    if not (has_async and has_dom_write):
        return True
    return "appState.viewToken" in text or "viewToken" in text


def is_probably_render_function(name: str, body: str) -> bool:
    """
    Heuristically detect render-like / pure-presentational functions.
    """
    lower_name = name.lower()
    if lower_name.startswith("render") or lower_name.startswith("build") or lower_name.startswith("template"):
        return True
    if contains_html_like_markup(body):
        return True
    if re.search(r"""\breturn\s+[`'"].*<\s*[a-zA-Z]""", body, flags=re.S):
        return True
    return False


def mutation_events(text: str) -> int:
    """Count rough mutation-like events in a text blob."""
    patterns = [
        r"""\.\s*(push|splice|shift|unshift|pop|sort|reverse|copyWithin|fill)\s*\(""",
        r"""\+\+|--""",
        r"""\b[A-Za-z_$][\w$.\]]*\s*[\+\-\*\/%]?=""",
        r"""\b(?:appState|state|runsViewState|liveState)\.[A-Za-z_$][\w$]*\s*=""",
    ]
    return sum(len(re.findall(p, text)) for p in patterns)


def has_side_effect_tokens(text: str) -> bool:
    """Return whether text contains obvious side-effect markers."""
    return any(token in text for token in SIDE_EFFECT_TOKENS) or any(token in text for token in DOM_WRITE_TOKENS)


def check_forbidden_defaults(text: str, rel: str) -> None:
    """Flag patterns that silently hide missing data."""
    patterns = [
        (r"""\?\?\s*(\[\]|\{\}|["']{0,2}|0|false|null)""", "nullish-coalescing default may hide missing data"),
        (r"""\|\|\s*(\[\]|\{\}|["']{0,2}|0|false|null)""", "logical-or default may hide missing data"),
        (r"""\.get\([^)]*,\s*[^)]+\)""", "dict-like default getter may hide missing data"),
        (r"""\?\.[A-Za-z_$][\w$]*""", "optional chaining may hide missing required fields"),
    ]
    for pattern, msg in patterns:
        if re.search(pattern, text):
            warnings.append(f"{rel}: {msg}")


def check_silent_failure_patterns(text: str, rel: str) -> None:
    """Detect common silent-failure or swallow patterns."""
    silent_patterns = [
        (r"""if\s*\(\s*!\s*[A-Za-z_$][\w$.]*\s*\)\s*return\b""", "early return on missing value may be a silent failure"),
        (r"""catch\s*\(\s*\w+\s*\)\s*\{\s*(?:console\.\w+\([^)]*\)\s*;?\s*)?\}""", "empty or effectively empty catch block"),
        (r"""catch\s*\(\s*\w+\s*\)\s*\{\s*return\s+(null|undefined|false|\[\]|\{\}|["']{0,2})\s*;?\s*\}""", "catch returns fallback value"),
    ]
    for pattern, msg in silent_patterns:
        if re.search(pattern, text, flags=re.S):
            warnings.append(f"{rel}: {msg}")


def check_framework_introduction(import_paths: list[str], rel: str) -> None:
    """Block framework or build-system creep."""
    joined = " ".join(import_paths).lower()
    for forbidden in FORBIDDEN_FRAMEWORK_IMPORTS:
        if forbidden in joined:
            failures.append(f"{rel}: forbidden framework/build import detected: {forbidden}")


def check_temp_filename_strings(text: str, rel: str) -> None:
    """Warn on suspicious temp-ish file naming."""
    lowered = text.lower()
    for name in SUSPICIOUS_TEMP_NAMES:
        if name in lowered:
            warnings.append(f"{rel}: suspicious non-deterministic/temp-like filename reference: {name}")


def check_testid_usage(text: str, rel: str) -> None:
    """
    Encourage stable selectors in files that likely render UI markup.
    Only warn.
    """
    if contains_html_like_markup(text):
        if "data-testid" not in text:
            warnings.append(f"{rel}: markup present but no data-testid found; interactive UI should expose stable test selectors")


def check_console_usage(text: str, rel: str) -> None:
    """Warn on excessive console usage."""
    matches = re.findall(r"""\bconsole\.(log|debug|info|warn|error)\s*\(""", text)
    if len(matches) > MAX_CONSOLE_STATEMENTS_PER_FILE:
        warnings.append(f"{rel}: excessive console usage ({len(matches)} statements)")


def check_file_size(text: str, rel: str) -> None:
    """Warn/fail on oversized files."""
    lines = count_lines(text)
    if rel == "app.js" and lines > MAX_APP_LINES:
        warnings.append(f"{rel}: app.js exceeds {MAX_APP_LINES} lines and may be becoming a god file ({lines} lines)")
    elif lines > MAX_FILE_LINES:
        warnings.append(f"{rel}: file exceeds {MAX_FILE_LINES} lines ({lines} lines); possible architectural drift")


def check_mutable_accumulator(text: str, rel: str) -> None:
    """
    Detect loops building arrays via mutation instead of map/filter.
    """
    pattern = r"""let\s+([A-Za-z_$][\w$]*)\s*=\s*\[\]\s*;.*?for\s*\(.*?\)\s*\{.*?\1\.push\("""
    if re.search(pattern, text, flags=re.S):
        warnings.append(f"{rel}: mutable accumulator pattern detected; prefer map/filter pipeline")


def check_manual_reduction(text: str, rel: str) -> None:
    """Detect sum / count accumulation loops that could be reduce()."""
    patterns = [
        r"""let\s+([A-Za-z_$][\w$]*)\s*=\s*0\s*;.*?for\s*\(.*?\)\s*\{.*?\1\s*\+=""",
        r"""let\s+([A-Za-z_$][\w$]*)\s*=\s*0\s*;.*?for\s*\(.*?\)\s*\{.*?\1\s*=\s*\1\s*\+""",
        r"""let\s+([A-Za-z_$][\w$]*)\s*=\s*['"]{0,2}\s*;.*?for\s*\(.*?\)\s*\{.*?\1\s*\+=""",
    ]
    for pattern in patterns:
        if re.search(pattern, text, flags=re.S):
            warnings.append(f"{rel}: manual reduction loop detected; consider Array.reduce()")
            break


def check_index_loop(text: str, rel: str) -> None:
    """Warn on index-based loops over arrays."""
    if re.search(r"""for\s*\(\s*let\s+\w+\s*=\s*0\s*;\s*\w+\s*<\s*[A-Za-z_$][\w$.\]]*\.length\s*;""", text):
        warnings.append(f"{rel}: index-based loop over array; prefer map/filter/reduce or for-of")


def check_excessive_mutation(text: str, rel: str) -> None:
    """Warn on heavy mutation density within a file."""
    events = mutation_events(text)
    if events > MAX_MUTATION_EVENTS_PER_FILE:
        warnings.append(f"{rel}: heavy mutation detected ({events} mutation-like events); consider functional transformations")


def check_nested_loops(text: str, rel: str) -> None:
    """Warn on imperative nested loops."""
    if re.search(r"""for\s*\(.*?\)\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*for\s*\(""", text, flags=re.S):
        warnings.append(f"{rel}: nested loops detected; consider map/reduce, indexing, or data restructuring")


def check_missing_functional_ops(text: str, rel: str) -> None:
    """Warn when loops are used but functional array operators never appear."""
    has_loop = bool(re.search(r"""\bfor\s*\(""", text))
    has_functional = any(op in text for op in FUNCTIONAL_ARRAY_OPS)
    if has_loop and not has_functional:
        warnings.append(f"{rel}: loops used but no functional array operators detected")


def check_mutable_globals(text: str, rel: str) -> None:
    """Detect likely mutable module-level state."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("import ") or stripped.startswith("export "):
            continue
        # Heuristic: stop once first function/class body likely begins.
        if re.search(r"""\b(function|class)\b""", stripped):
            break
        if re.match(r"""^(let|var)\s+[A-Za-z_$][\w$]*\s*=\s*(\{|\[|new\s+Map|new\s+Set|new\s+Date|\w+)""", stripped):
            warnings.append(f"{rel}:{i+1}: possible mutable module-level state")
            break


def check_render_purity(text: str, rel: str) -> None:
    """
    Detect side effects in render-like functions.

    Render/build/template functions should usually be pure:
      input -> markup/string/data structure
    They should not fetch, mutate shared state, attach listeners, or write DOM.
    """
    for name, start, end in find_function_defs(text):
        body = get_line_range(text, start, end)
        if not is_probably_render_function(name, body):
            continue

        if any(token in body for token in SIDE_EFFECT_TOKENS):
            warnings.append(f"{rel}:{start}-{end}: render-like function '{name}' appears to contain side effects")
        if any(token in body for token in DOM_WRITE_TOKENS):
            warnings.append(f"{rel}:{start}-{end}: render-like function '{name}' mutates DOM; render/build functions should be pure")
        if any(token in body for token in STATE_MUTATION_TOKENS):
            warnings.append(f"{rel}:{start}-{end}: render-like function '{name}' reaches into shared state")
        if re.search(r"""\b(addEventListener|removeEventListener|fetch|setTimeout|setInterval|requestAnimationFrame)\s*\(""", body):
            warnings.append(f"{rel}:{start}-{end}: render-like function '{name}' mixes rendering with lifecycle side effects")


def check_state_mutation_auditing(text: str, rel: str) -> None:
    """
    Detect suspicious state mutation outside the dedicated state file.

    This is intentionally broad: the goal is to catch surprise writes to shared
    mutable state and push code toward explicit state transitions.
    """
    if rel == STATE_FILE or rel.endswith(f"/{STATE_FILE}"):
        return

    patterns = [
        r"""\bappState\.[A-Za-z_$][\w$]*\s*=""",
        r"""\bstate\.[A-Za-z_$][\w$]*\s*=""",
        r"""\brunsViewState\.[A-Za-z_$][\w$]*\s*=""",
        r"""\bliveState\.[A-Za-z_$][\w$]*\s*=""",
        r"""\b(?:appState|state|runsViewState|liveState)\.[A-Za-z_$][\w$]*\.(?:push|splice|shift|unshift|pop)\s*\(""",
    ]
    for pattern in patterns:
        if re.search(pattern, text):
            warnings.append(f"{rel}: shared state mutation detected outside {STATE_FILE}; prefer explicit state helpers")
            break


def check_component_side_effects(text: str, rel: str) -> None:
    """
    Strengthen component purity expectations.

    Components should remain presentational. They should not reach outward into
    storage, timers, navigation, global state, or network access.
    """
    if not rel.startswith("components/"):
        return

    side_effect_patterns = [
        (r"""\bfetch\s*\(""", "components MUST NOT perform network requests"),
        (r"""\b(?:localStorage|sessionStorage)\.""", "components MUST NOT read/write browser storage"),
        (r"""\bdocument\.cookie\b""", "components MUST NOT touch cookies"),
        (r"""\b(?:setTimeout|setInterval|requestAnimationFrame)\s*\(""", "components MUST NOT own timers/animation scheduling"),
        (r"""\b(?:history\.pushState|history\.replaceState|window\.location)\b""", "components MUST NOT perform navigation side effects"),
        (r"""\baddEventListener\s*\(""", "components MUST NOT attach listeners directly"),
        (r"""\b(?:appState|state|runsViewState|liveState)\.[A-Za-z_$][\w$]*\s*=""", "components MUST NOT mutate shared state directly"),
    ]
    for pattern, message in side_effect_patterns:
        if re.search(pattern, text):
            failures.append(f"{rel}: {message}")


def check_function_side_effect_density(raw_text: str, rel: str) -> None:
    """
    Detect functions that combine too many responsibilities:
    mutation + DOM + network + timers + global state.
    """
    for name, start, end in find_function_defs(raw_text):
        body = get_line_range(raw_text, start, end)
        mut = mutation_events(body)
        dom_writes = sum(1 for token in DOM_WRITE_TOKENS if token in body)
        side_tokens = sum(1 for token in SIDE_EFFECT_TOKENS if token in body)
        shared_state_hits = sum(1 for token in STATE_MUTATION_TOKENS if token in body)

        if mut > MAX_MUTATION_EVENTS_PER_FUNCTION:
            warnings.append(f"{rel}:{start}-{end}: function '{name}' has heavy mutation density ({mut})")
        if dom_writes >= 2 and side_tokens >= 1:
            warnings.append(f"{rel}:{start}-{end}: function '{name}' mixes DOM writes with other side effects")
        if shared_state_hits and dom_writes:
            warnings.append(f"{rel}:{start}-{end}: function '{name}' mixes shared-state mutation with DOM mutation")


def check_functional_style(text: str, rel: str) -> None:
    """Bundle functional-style heuristics."""
    check_mutable_accumulator(text, rel)
    check_manual_reduction(text, rel)
    check_index_loop(text, rel)
    check_excessive_mutation(text, rel)
    check_nested_loops(text, rel)
    check_missing_functional_ops(text, rel)
    check_mutable_globals(text, rel)
    check_render_purity(text, rel)
    check_state_mutation_auditing(text, rel)
    check_component_side_effects(text, rel)
    check_function_side_effect_density(text, rel)


def check_function_quality(raw_text: str, stripped_text: str, rel: str) -> None:
    """Check function sizes, documentation, and duplication."""
    functions = find_function_defs(stripped_text)
    for name, start, end in functions:
        function_index[name].append(rel)
        length = max(1, end - start + 1)
        if length > MAX_FUNC_LINES:
            warnings.append(f"{rel}:{start}-{end}: function '{name}' exceeds {MAX_FUNC_LINES} lines ({length} lines)")
        if not has_doc_comment_above_function(raw_text, start):
            warnings.append(f"{rel}:{start}: function '{name}' is missing a documentation comment")


def check_import_direction(import_paths: list[str], rel: str) -> None:
    """Generic import direction checks by layer."""
    normalized = [relative_import_target(p) for p in import_paths]

    if rel.startswith("utils/"):
        if normalized:
            failures.append(f"{rel}: utils MUST NOT import project modules")

    if rel.startswith("components/"):
        for path in normalized:
            if "/api/" in path or path.startswith("../api/") or path.startswith("./api/") or "api/" in path:
                failures.append(f"{rel}: components MUST NOT import api/: {path}")
            if "state" in path:
                failures.append(f"{rel}: components MUST NOT import state: {path}")
            if "/views/" in path or "views/" in path:
                failures.append(f"{rel}: components MUST NOT import views/: {path}")

    if rel.startswith("api/") and rel != "api/client.js":
        for path in normalized:
            if "/views/" in path or "views/" in path:
                failures.append(f"{rel}: api MUST NOT import views/: {path}")
            if "/components/" in path or "components/" in path:
                failures.append(f"{rel}: api MUST NOT import components/: {path}")
            if "state" in path:
                failures.append(f"{rel}: api MUST NOT import state: {path}")
            if "/utils/" in path or "utils/" in path:
                failures.append(f"{rel}: api MUST NOT import utils/: {path}")

    if rel.startswith("views/"):
        for path in normalized:
            if path.endswith("/router.js") or path == "../router.js" or path == "./router.js":
                # Views importing router is usually wrong coupling.
                warnings.append(f"{rel}: view imports router directly ({path}); prefer router owning view lifecycle")
            if "/app.js" in path or path.endswith("app.js"):
                failures.append(f"{rel}: views MUST NOT import app.js: {path}")

    if rel == "router.js":
        for path in normalized:
            if "/api/" in path or "api/" in path:
                warnings.append(f"{rel}: router imports api ({path}); router should not become a data-loading layer")


def check_components_layer(text: str, rel: str) -> None:
    """Enforce purity rules for components."""
    if not rel.startswith("components/"):
        return
    if "fetch(" in text:
        failures.append(f"{rel}: components MUST NOT call fetch()")
    if "document." in text or "querySelector(" in text or "getElementById(" in text:
        failures.append(f"{rel}: components MUST NOT access DOM directly")
    if "addEventListener(" in text:
        failures.append(f"{rel}: components MUST NOT attach event listeners")
    if ".innerHTML" in text:
        failures.append(f"{rel}: components MUST NOT call innerHTML")
    if "state." in text or "appState." in text or "runsViewState." in text or "liveState." in text:
        warnings.append(f"{rel}: components may be reaching into shared state; components should be pure")


def check_utils_layer(text: str, rel: str) -> None:
    """Enforce utils purity."""
    if not rel.startswith("utils/"):
        return
    if "document." in text or "querySelector(" in text or "getElementById(" in text:
        failures.append(f"{rel}: utils MUST NOT access DOM")
    if "fetch(" in text:
        failures.append(f"{rel}: utils MUST NOT perform network requests")
    if "addEventListener(" in text:
        failures.append(f"{rel}: utils MUST NOT attach event listeners")


def check_api_layer(text: str, rel: str) -> None:
    """Enforce API layer constraints."""
    if not rel.startswith("api/"):
        return
    if "document." in text or "querySelector(" in text or "getElementById(" in text:
        failures.append(f"{rel}: api MUST NOT access DOM")
    if "addEventListener(" in text:
        failures.append(f"{rel}: api MUST NOT attach event listeners")


def check_views_layer(text: str, rel: str) -> None:
    """Warn on view violations that are hard to enforce perfectly."""
    if not rel.startswith("views/"):
        return
    if contains_nontrivial_html_markup(text):
        warnings.append(f"{rel}: views may contain non-trivial HTML construction; data-driven markup should live in components")
    if not require_viewtoken_guard(text):
        warnings.append(f"{rel}: async DOM-writing view code may be missing viewToken guard")
    if "fetch(" in text:
        warnings.append(f"{rel}: direct fetch() found in views; prefer going through api/* wrappers")
    if "addEventListener(" in text and "handleAction" not in text and "teardown" not in text:
        warnings.append(f"{rel}: direct event listener attachment found; verify lifecycle cleanup is correct")


def check_app_file(text: str, rel: str) -> None:
    """Keep app.js minimal and generic."""
    if rel != "app.js":
        return
    for action in FORBIDDEN_APP_ACTION_NAMES:
        if action in text:
            warnings.append(f"{rel}: contains view-specific action '{action}' (should dispatch generically to activeView.handleAction)")
    if "fetch(" in text:
        warnings.append(f"{rel}: app.js contains fetch(); app.js should remain thin")
    if "innerHTML" in text:
        warnings.append(f"{rel}: app.js mutates DOM directly; verify it is not taking over view responsibilities")


def check_router_file(text: str, rel: str) -> None:
    """Warn if router grows beyond routing/lifecycle role."""
    if rel != "router.js":
        return
    if "fetch(" in text:
        warnings.append(f"{rel}: router contains fetch(); router should not become a data-loading layer")
    if ".innerHTML" in text:
        warnings.append(f"{rel}: router writes DOM directly; verify router is not taking over view responsibilities")


def check_universal_patterns(text: str, rel: str) -> None:
    """Universal checks across all files."""
    if "onclick=" in text.lower():
        failures.append(f"{rel}: inline onclick found — use delegation with data-action")
    window_assigns = re.findall(r"""window\.\w+\s*=""", text)
    if window_assigns:
        warnings.append(f"{rel}: window global assignment found: {window_assigns}")
    if re.search(r"""\b(eval|new Function)\s*\(""", text):
        failures.append(f"{rel}: dynamic code execution detected")
    if re.search(r"""\bsetInterval\s*\(""", text) and rel != "views/liveView.js":
        warnings.append(f"{rel}: setInterval found outside expected polling area; verify teardown exists")
    check_console_usage(text, rel)
    check_temp_filename_strings(text, rel)
    check_forbidden_defaults(text, rel)
    check_silent_failure_patterns(text, rel)
    check_testid_usage(text, rel)
    check_functional_style(text, rel)


def check_file(raw_path: pathlib.Path, raw_text: str, rel: str) -> None:
    """Run all checks against a single JS file."""
    stripped_text = strip_js_comments(raw_text)
    import_paths = find_import_paths(stripped_text)

    if not has_file_doc_comment(raw_text):
        warnings.append(f"{rel}: missing top-of-file documentation comment")

    check_file_size(stripped_text, rel)
    check_import_direction(import_paths, rel)
    check_framework_introduction(import_paths, rel)
    check_components_layer(stripped_text, rel)
    check_utils_layer(stripped_text, rel)
    check_api_layer(stripped_text, rel)
    check_views_layer(stripped_text, rel)
    check_app_file(stripped_text, rel)
    check_router_file(stripped_text, rel)
    check_universal_patterns(stripped_text, rel)
    check_function_quality(raw_text, stripped_text, rel)


def check_duplicate_function_names() -> None:
    """Warn on duplicated function names across files."""
    for name, files in sorted(function_index.items()):
        uniq = sorted(set(files))
        if len(uniq) > 1 and name not in {"render", "teardown", "handleAction"}:
            warnings.append(
                f"duplicate function name '{name}' appears in multiple files: {', '.join(uniq)}"
            )


def print_report() -> None:
    """Print a human-readable report."""
    print("=" * 72)
    print("Debate Dashboard Architecture Check")
    print("=" * 72)

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for warning in warnings:
            print(f"  ⚠ {warning}")

    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for failure in failures:
            print(f"  ✗ {failure}")
        print(f"\nRESULT: FAIL ({len(failures)} failures, {len(warnings)} warnings)")
    else:
        print(f"\nRESULT: PASS ({len(warnings)} warnings)")


BASELINE_FILE = pathlib.Path(__file__).resolve().parent / "warning_baseline.json"


def _load_baseline() -> set[str]:
    """Load the set of known/accepted warnings from the baseline file."""
    if not BASELINE_FILE.exists():
        return set()
    return set(json.loads(BASELINE_FILE.read_text(encoding="utf-8")))


def _save_baseline(current_warnings: list[str]) -> None:
    """Snapshot current warnings as the new baseline."""
    BASELINE_FILE.write_text(
        json.dumps(sorted(current_warnings), indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Run the checker and exit with the correct status code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-baseline", action="store_true",
        help="Snapshot current warnings as the accepted baseline.",
    )
    args = parser.parse_args()

    if not ROOT.exists():
        print(f"ERROR: JS root does not exist: {ROOT}")
        sys.exit(1)

    js_files = sorted(ROOT.rglob("*.js"))
    if not js_files:
        print(f"ERROR: No JS files found under: {ROOT}")
        sys.exit(1)

    for path in js_files:
        try:
            raw_text = path.read_text(encoding="utf-8")
        except Exception as exc:
            failures.append(f"{path}: failed to read file: {exc}")
            continue

        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        check_file(path, raw_text, rel)

    check_duplicate_function_names()
    print_report()

    # --- Baseline enforcement ---
    if args.update_baseline:
        _save_baseline(warnings)
        print(f"\nBaseline updated: {len(warnings)} warnings saved to {BASELINE_FILE.name}")
        sys.exit(1 if failures else 0)

    baseline = _load_baseline()
    new_warnings = [w for w in warnings if w not in baseline]
    resolved = sorted(baseline - set(warnings))

    if resolved:
        print(f"\nRESOLVED ({len(resolved)} warnings fixed since baseline):")
        for w in resolved:
            print(f"  ✓ {w}")

    if new_warnings:
        print(f"\nNEW WARNINGS ({len(new_warnings)}) — these must be fixed or baseline updated:")
        for w in new_warnings:
            print(f"  ✗ {w}")
        sys.exit(1)

    if failures:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()