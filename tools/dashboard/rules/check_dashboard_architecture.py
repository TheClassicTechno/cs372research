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

Exit code
---------
0 on pass
1 on any failure
"""

from __future__ import annotations

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
    has_dom_write = any(
        token in text
        for token in (
            ".innerHTML",
            ".insertAdjacentHTML",
            ".appendChild(",
            ".replaceChildren(",
            ".textContent =",
            ".classList.",
        )
    )
    if not (has_async and has_dom_write):
        return True
    return "appState.viewToken" in text or "viewToken" in text


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


def main() -> None:
    """Run the checker and exit with the correct status code."""
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

    if failures:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()