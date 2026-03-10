#!/usr/bin/env python3
"""
Dashboard Silent Default / Fail-Fast Checker

Purpose
-------
This checker detects patterns where JavaScript code may silently hide errors
or missing data instead of failing fast. AI assistants (and humans) often
introduce defensive defaults that mask bugs and make debugging extremely
difficult.

The goal is to enforce a "fail fast" discipline in the dashboard codebase.

This script scans JS files under the dashboard static directory and flags
patterns where code may silently recover from missing data, swallow errors,
or provide default values instead of surfacing problems.

Examples of problematic patterns
--------------------------------

Silent defaults:
    value = obj.key || []
    value = obj.key ?? {}
    value = data || ""

Optional chaining masking missing data:
    obj?.field?.nested

Swallowed errors:
    catch (err) { }

Fallback returns:
    catch (err) { return null }

Silent early exits:
    if (!data) return;

Dictionary-style defaults:
    map.get(key, [])

These patterns make debugging extremely difficult because they hide the root
cause of failures.

Exit codes
----------
0  -> pass
1  -> violations detected
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent / "static" / "js"

violations: list[str] = []
warnings: list[str] = []


def strip_js_comments(text: str) -> str:
    """Remove JS comments to reduce false positives."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    return text


def detect_nullish_defaults(text: str, rel: str) -> None:
    """
    Detect nullish coalescing that hides missing data.

    Example:
        value = data ?? []
        config = options ?? {}
    """
    pattern = r"\?\?\s*(\[\]|\{\}|['\"]{0,2}|0|false|null)"
    for match in re.finditer(pattern, text):
        line = text[: match.start()].count("\n") + 1
        violations.append(
            f"{rel}:{line}: nullish-coalescing default may hide missing data"
        )


def detect_logical_or_defaults(text: str, rel: str) -> None:
    """
    Detect logical OR defaults masking missing values.

    Example:
        value = data || []
        name = user.name || ""
    """
    pattern = r"\|\|\s*(\[\]|\{\}|['\"]{0,2}|0|false|null)"
    for match in re.finditer(pattern, text):
        line = text[: match.start()].count("\n") + 1
        violations.append(
            f"{rel}:{line}: logical-OR default may hide missing data"
        )


def detect_optional_chaining(text: str, rel: str) -> None:
    """
    Detect optional chaining usage.

    Optional chaining often hides unexpected missing fields.
    """
    pattern = r"\?\.[A-Za-z_$]"
    for match in re.finditer(pattern, text):
        line = text[: match.start()].count("\n") + 1
        warnings.append(
            f"{rel}:{line}: optional chaining may hide missing required fields"
        )


def detect_empty_catch(text: str, rel: str) -> None:
    """
    Detect empty catch blocks.

    Example:
        try { ... }
        catch (err) { }
    """
    pattern = r"catch\s*\(\s*\w+\s*\)\s*\{\s*\}"
    for match in re.finditer(pattern, text, flags=re.S):
        line = text[: match.start()].count("\n") + 1
        violations.append(
            f"{rel}:{line}: empty catch block silently swallows errors"
        )


def detect_fallback_catch_returns(text: str, rel: str) -> None:
    """
    Detect catch blocks that return fallback values.

    Example:
        catch (err) { return null }
        catch (err) { return [] }
    """
    pattern = r"catch\s*\(\s*\w+\s*\)\s*\{\s*return\s+(null|undefined|false|\[\]|\{\}|['\"]{0,2})"
    for match in re.finditer(pattern, text, flags=re.S):
        line = text[: match.start()].count("\n") + 1
        violations.append(
            f"{rel}:{line}: catch block returns fallback value instead of failing"
        )


def detect_silent_early_return(text: str, rel: str) -> None:
    """
    Detect early returns when required values are missing.

    Example:
        if (!data) return;
    """
    pattern = r"if\s*\(\s*!\s*[A-Za-z_$][\w$.]*\s*\)\s*return\b"
    for match in re.finditer(pattern, text):
        line = text[: match.start()].count("\n") + 1
        warnings.append(
            f"{rel}:{line}: early return on missing value may hide bug"
        )


def detect_dictionary_get_default(text: str, rel: str) -> None:
    """
    Detect dictionary-style default values.

    Example:
        map.get(key, [])
    """
    pattern = r"\.get\([^)]*,\s*[^)]+\)"
    for match in re.finditer(pattern, text):
        line = text[: match.start()].count("\n") + 1
        warnings.append(
            f"{rel}:{line}: dictionary-style default getter may hide missing key"
        )
def detect_api_fallback_defaults(text: str, rel: str) -> None:
    """
    Detect API results defaulting to [] or {}.
    """

    pattern = r"await\s+[A-Za-z_$][\w$]*\([^)]*\)\s*\|\|\s*(\[\]|\{\})"

    for match in re.finditer(pattern, text):
        line = text[:match.start()].count("\n") + 1

        violations.append(
            f"{rel}:{line}: API result defaulted to []/{{}} instead of failing fast"
        )

def detect_unchecked_fetch_json(text: str, rel: str) -> None:
    """
    Detect fetch().json() usage without response.ok checks.
    """

    fetch_pattern = r"fetch\([^)]*\)"
    json_pattern = r"\.json\(\)"

    if re.search(fetch_pattern, text) and re.search(json_pattern, text):

        if "response.ok" not in text and "res.ok" not in text:
            warnings.append(
                f"{rel}: fetch().json() used without checking response.ok"
            )

def detect_unhandled_promises(text: str, rel: str) -> None:
    """
    Detect async calls that are neither awaited nor caught.
    """
    pattern = r"\b[A-Za-z_$][\w$]*\([^)]*\)\s*;"

    for match in re.finditer(pattern, text):
        call = match.group(0)

        # ignore obvious safe cases
        if "await " in call:
            continue
        if ".catch(" in call:
            continue
        if ".then(" in call:
            continue

        # heuristic: ignore console/logging
        if "console." in call:
            continue

        line = text[:match.start()].count("\n") + 1

        warnings.append(
            f"{rel}:{line}: possible unhandled Promise call (missing await or .catch)"
        )

def check_file(path: pathlib.Path) -> None:
    """Run all silent-default checks on a file."""
    raw = path.read_text(encoding="utf-8")
    text = strip_js_comments(raw)
    rel = str(path.relative_to(ROOT)).replace("\\", "/")

    detect_nullish_defaults(text, rel)
    detect_logical_or_defaults(text, rel)
    detect_optional_chaining(text, rel)
    detect_empty_catch(text, rel)
    detect_fallback_catch_returns(text, rel)
    detect_silent_early_return(text, rel)
    detect_dictionary_get_default(text, rel)


def print_report() -> None:
    """Print results."""
    print("=" * 70)
    print("Dashboard Silent Default Checker")
    print("=" * 70)

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠ {w}")

    if violations:
        print(f"\nVIOLATIONS ({len(violations)}):")
        for v in violations:
            print(f"  ✗ {v}")
        print(f"\nRESULT: FAIL ({len(violations)} violations)")
    else:
        print("\nRESULT: PASS")


def main() -> None:
    if not ROOT.exists():
        print(f"ERROR: JS root does not exist: {ROOT}")
        sys.exit(1)

    files = sorted(ROOT.rglob("*.js"))

    if not files:
        print("ERROR: no JS files found")
        sys.exit(1)

    for file in files:
        try:
            check_file(file)
        except Exception as e:
            violations.append(f"{file}: failed to analyze ({e})")

    print_report()

    if violations:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()