"""Integration test conftest — shared fixtures and pytest options."""

from __future__ import annotations

import pytest


def pytest_addoption(parser):
    """Add --update-snapshots option for golden prompt snapshot tests."""
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="Regenerate golden prompt snapshots instead of comparing against them.",
    )
