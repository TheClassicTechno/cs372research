#!/usr/bin/env python3
"""
Pipeline provenance tracking — shared module for run manifests and inline metadata.

Provides:
  - hash_file(path)           SHA-256 hex digest of a file
  - generate_run_id()         Unique run identifier (UTC timestamp + hex suffix)
  - get_run_id()              Read PIPELINE_RUN_ID from env (None if unset)
  - inline_provenance()       Dict with pipeline_run_id + generated_at_utc
  - git_info()                Dict with git_commit, git_branch, git_dirty
  - StageContext              Per-stage recorder for params, API calls, outputs
  - PipelineRun               Top-level orchestrator context
  - append_stage_to_manifest  For subprocess stages to write back to manifest

Usage from pipeline stages (subprocess-safe):

    from provenance import get_run_id, inline_provenance, StageContext, append_stage_to_manifest

    # Inline provenance — always safe, returns {} fields even without orchestrator
    prov = inline_provenance()
    output_dict.update(prov)

    # Stage reporting — only when run via orchestrator
    if get_run_id():
        ctx = StageContext("3_macro_data", 3, "macro/macro_quarter_builder.py")
        ctx.start()
        ctx.set_parameters({"back_years": 2, "fred_series": [...]})
        ctx.record_output("macro/data/macro_2025_Q1.json")
        ctx.complete("completed")
        append_stage_to_manifest(ctx)
"""

import datetime as dt
import hashlib
import json
import os
import platform
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------

def hash_file(path, chunk_size: int = 8192) -> Optional[str]:
    """Return SHA-256 hex digest of a file, or None if missing."""
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Run ID
# ---------------------------------------------------------------------------

def generate_run_id() -> str:
    """Generate a unique run ID: UTC timestamp + 5-char hex suffix."""
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(3)[:5]
    return f"{ts}_{suffix}"


def get_run_id() -> Optional[str]:
    """Read PIPELINE_RUN_ID from environment. Returns None if not set."""
    return os.environ.get("PIPELINE_RUN_ID")


def get_manifest_path() -> Optional[str]:
    """Read PIPELINE_MANIFEST_PATH from environment. Returns None if not set."""
    return os.environ.get("PIPELINE_MANIFEST_PATH")


# ---------------------------------------------------------------------------
# Inline provenance (added to individual output files)
# ---------------------------------------------------------------------------

def inline_provenance() -> Dict[str, str]:
    """Return dict with pipeline_run_id and generated_at_utc for embedding in output files."""
    result = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    run_id = get_run_id()
    if run_id:
        result["pipeline_run_id"] = run_id
    return result


# ---------------------------------------------------------------------------
# Git info
# ---------------------------------------------------------------------------

def git_info(repo_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Return git commit, branch, and dirty status."""
    cwd = str(repo_dir) if repo_dir else None
    info: Dict[str, Any] = {
        "git_commit": None,
        "git_branch": None,
        "git_dirty": None,
    }
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
        info["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
        info["git_dirty"] = len(status) > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return info


# ---------------------------------------------------------------------------
# StageContext — per-stage recorder
# ---------------------------------------------------------------------------

_DEFERRED = "__deferred__"


class StageContext:
    """Records metadata for a single pipeline stage."""

    def __init__(self, key: str, number: int, script: str):
        self.key = key
        self.number = number
        self.script = script
        self.status = "pending"
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.duration_seconds: Optional[float] = None
        self.parameters: Dict[str, Any] = {}
        self.api_calls: Dict[str, Any] = {}
        self.dependencies: List[Dict[str, Any]] = []
        self.outputs: List[Dict[str, Any]] = []

    def start(self) -> None:
        self.status = "running"
        self.started_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._start_time = dt.datetime.now(dt.timezone.utc)

    def complete(self, status: str = "completed") -> None:
        self.status = status
        self.completed_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if hasattr(self, "_start_time"):
            elapsed = (dt.datetime.now(dt.timezone.utc) - self._start_time).total_seconds()
            self.duration_seconds = round(elapsed, 1)

    def set_parameters(self, params: Dict[str, Any]) -> None:
        self.parameters = params

    def record_api_calls(self, name: str, metadata: Dict[str, Any]) -> None:
        self.api_calls[name] = metadata

    def record_output(self, output_path: str, input_paths: Optional[List[str]] = None) -> None:
        """Record an output file and its input dependencies. Hashes are deferred."""
        dep: Dict[str, Any] = {
            "output": output_path,
            "output_sha256": _DEFERRED,
        }
        if input_paths:
            dep["inputs"] = [{"path": p, "sha256": _DEFERRED} for p in input_paths]
        else:
            dep["inputs"] = []
        self.dependencies.append(dep)

    def record_simple_output(self, output_path: str) -> None:
        """Record a simple output file with no tracked inputs."""
        self.outputs.append({"path": output_path, "sha256": _DEFERRED})

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "script": self.script,
            "status": self.status,
        }
        if self.started_at:
            d["started_at_utc"] = self.started_at
        if self.completed_at:
            d["completed_at_utc"] = self.completed_at
        if self.duration_seconds is not None:
            d["duration_seconds"] = self.duration_seconds
        if self.parameters:
            d["parameters"] = self.parameters
        if self.api_calls:
            d["api_calls"] = self.api_calls
        if self.dependencies:
            d["dependencies"] = self.dependencies
        if self.outputs:
            d["outputs"] = self.outputs
        return d


# ---------------------------------------------------------------------------
# PipelineRun — top-level orchestrator context
# ---------------------------------------------------------------------------

class PipelineRun:
    """Top-level pipeline run context. Creates manifest, manages env vars."""

    def __init__(
        self,
        args: Any,
        tickers: List[str],
        quarters: List,
        pipeline_dir: Optional[Path] = None,
    ):
        self.run_id = generate_run_id()
        self.pipeline_dir = pipeline_dir or Path.cwd()
        self.provenance_dir = self.pipeline_dir / "provenance"
        self.manifest_path = self.provenance_dir / f"manifest_{self.run_id}.json"
        self.tickers = tickers
        self.quarters = quarters
        self.args = args
        self.stages: Dict[str, Dict[str, Any]] = {}
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None

    def start(self) -> None:
        """Set env vars and create initial manifest file."""
        self.started_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        os.environ["PIPELINE_RUN_ID"] = self.run_id
        os.environ["PIPELINE_MANIFEST_PATH"] = str(self.manifest_path)

        self.provenance_dir.mkdir(parents=True, exist_ok=True)

        # Write initial manifest skeleton
        manifest = self._build_manifest("running")
        self._write_manifest(manifest)

    def record_stage(self, key: str, stage_dict: Dict[str, Any]) -> None:
        """Record a stage's metadata (called by orchestrator after subprocess completes)."""
        self.stages[key] = stage_dict

    def finalize(self, status: str = "completed") -> Path:
        """Resolve deferred hashes, write final manifest, clean env."""
        self.completed_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Load manifest (subprocess stages may have appended to it)
        manifest = self._load_manifest()

        # Merge any orchestrator-recorded stages
        if "stages" not in manifest:
            manifest["stages"] = {}
        manifest["stages"].update(self.stages)

        # Resolve all deferred hashes
        self._resolve_hashes(manifest)

        # Update top-level fields
        manifest["status"] = status
        manifest["timing"]["completed_at_utc"] = self.completed_at
        if self.started_at:
            start = dt.datetime.strptime(self.started_at, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=dt.timezone.utc
            )
            end = dt.datetime.strptime(self.completed_at, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=dt.timezone.utc
            )
            manifest["timing"]["total_duration_seconds"] = round(
                (end - start).total_seconds(), 1
            )

        self._write_manifest(manifest)

        # Clean env
        os.environ.pop("PIPELINE_RUN_ID", None)
        os.environ.pop("PIPELINE_MANIFEST_PATH", None)

        return self.manifest_path

    def _build_manifest(self, status: str) -> Dict[str, Any]:
        git = git_info(self.pipeline_dir)

        # Build invocation args dict
        args_dict = {}
        if hasattr(self.args, "__dict__"):
            for k, v in vars(self.args).items():
                if v is not None:
                    args_dict[k] = v

        return {
            "manifest_version": "1.0",
            "pipeline_run_id": self.run_id,
            "status": status,
            "environment": {
                **git,
                "hostname": platform.node(),
                "python_version": platform.python_version(),
                "platform": platform.platform(),
            },
            "invocation": {
                "command": " ".join(sys.argv),
                "args": args_dict,
                "tickers_resolved": self.tickers,
                "quarters_resolved": [[y, q] for y, q in self.quarters],
            },
            "timing": {
                "started_at_utc": self.started_at,
                "completed_at_utc": None,
                "total_duration_seconds": None,
            },
            "stages": {},
        }

    def _write_manifest(self, manifest: Dict[str, Any]) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        tmp.rename(self.manifest_path)

    def _load_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            with open(self.manifest_path, "r") as f:
                return json.load(f)
        return self._build_manifest("running")

    def _resolve_hashes(self, manifest: Dict[str, Any]) -> None:
        """Walk the manifest and replace all __deferred__ hashes with real SHA-256."""
        for stage_data in manifest.get("stages", {}).values():
            for dep in stage_data.get("dependencies", []):
                if dep.get("output_sha256") == _DEFERRED:
                    dep["output_sha256"] = hash_file(
                        self.pipeline_dir / dep["output"]
                    )
                for inp in dep.get("inputs", []):
                    if inp.get("sha256") == _DEFERRED:
                        inp["sha256"] = hash_file(
                            self.pipeline_dir / inp["path"]
                        )
            for out in stage_data.get("outputs", []):
                if out.get("sha256") == _DEFERRED:
                    out["sha256"] = hash_file(
                        self.pipeline_dir / out["path"]
                    )


# ---------------------------------------------------------------------------
# Subprocess helper — stages call this to append their data to the manifest
# ---------------------------------------------------------------------------

def append_stage_to_manifest(ctx: StageContext) -> None:
    """Load manifest from PIPELINE_MANIFEST_PATH, merge this stage, write back.

    Safe for concurrent use: each stage writes under its own unique key.
    """
    manifest_path_str = get_manifest_path()
    if not manifest_path_str:
        return

    manifest_path = Path(manifest_path_str)
    if not manifest_path.exists():
        return

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    if "stages" not in manifest:
        manifest["stages"] = {}

    manifest["stages"][ctx.key] = ctx.to_dict()

    tmp = manifest_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    tmp.rename(manifest_path)
