"""S3SyncWorker — background thread that uploads closed WAL segments to S3.

Uploads only CLOSED segments (never the active segment).  Idempotent:
tracks uploaded segments in ``uploaded_segments.json`` and checks S3
existence before uploading.

S3 key layout:
    {prefix}/{github_username}/{run_id}/events/segment_NNNNNN.jsonl
    {prefix}/{github_username}/{run_id}/manifest.json

The manifest is uploaded last with ``status: "complete"`` only when
``finalize()`` is called (i.e., the run finished successfully).
Consumers should trust the manifest only when ``status == "complete"``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .event_logger import EventLogger

logger = logging.getLogger(__name__)


def _get_github_username() -> str:
    """Return the GitHub username from git config, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "config", "user.name"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            # Sanitize: replace spaces/special chars with underscores
            name = result.stdout.strip()
            return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return os.environ.get("USER", os.environ.get("USERNAME", "unknown"))


class S3SyncWorker:
    """Background thread that periodically uploads closed segments to S3.

    Args:
        event_logger: The EventLogger instance whose segments to upload.
        bucket: S3 bucket name.
        prefix: S3 key prefix (e.g. "debate-logs").
        run_id: The run ID for S3 path scoping.
        poll_interval: Seconds between upload sweeps (default 30).
        github_username: Override for the user-scoped S3 path component.
    """

    def __init__(
        self,
        event_logger: EventLogger,
        bucket: str,
        prefix: str,
        run_id: str,
        *,
        poll_interval: float = 30.0,
        github_username: Optional[str] = None,
    ) -> None:
        self._event_logger = event_logger
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._run_id = run_id
        self._poll_interval = poll_interval
        self._username = github_username or _get_github_username()

        self._uploaded_path = event_logger.run_dir / "uploaded_segments.json"
        self._uploaded: set[str] = self._load_uploaded()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._s3_client: Any = None  # lazy-initialized boto3 client

    # ── S3 key helpers ──────────────────────────────────────────────

    def _s3_key(self, segment_name: str) -> str:
        """Build the full S3 key for a segment file."""
        return f"{self._prefix}/{self._username}/{self._run_id}/events/{segment_name}"

    def _s3_artifact_key(self, relative_path: str) -> str:
        """Build the full S3 key for an artifact file."""
        return f"{self._prefix}/{self._username}/{self._run_id}/{relative_path}"

    def _manifest_key(self) -> str:
        """Build the S3 key for the run manifest."""
        return f"{self._prefix}/{self._username}/{self._run_id}/manifest.json"

    # ── Uploaded tracking (idempotency) ─────────────────────────────

    def _load_uploaded(self) -> set[str]:
        """Load the set of already-uploaded segment filenames from disk."""
        if self._uploaded_path.exists():
            try:
                data = json.loads(self._uploaded_path.read_text(encoding="utf-8"))
                return set(data.get("uploaded", []))
            except (json.JSONDecodeError, OSError):
                pass
        return set()

    def _save_uploaded(self) -> None:
        """Persist the uploaded set to disk."""
        try:
            self._uploaded_path.write_text(
                json.dumps({"uploaded": sorted(self._uploaded)}, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("Failed to save uploaded_segments.json: %s", e)

    # ── S3 operations ───────────────────────────────────────────────

    def _get_s3_client(self) -> Any:
        """Lazy-initialize the boto3 S3 client."""
        if self._s3_client is None:
            import boto3
            self._s3_client = boto3.client("s3")
        return self._s3_client

    def _s3_object_exists(self, key: str) -> bool:
        """Check if an S3 object exists (HEAD request)."""
        try:
            self._get_s3_client().head_object(Bucket=self._bucket, Key=key)
            return True
        except Exception:
            return False

    def _upload_file(self, local_path: Path, key: str) -> bool:
        """Upload a file to S3. Returns True on success."""
        try:
            self._get_s3_client().upload_file(
                str(local_path), self._bucket, key,
            )
            return True
        except Exception as e:
            logger.warning("S3 upload failed for %s: %s", key, e)
            return False

    # ── Sync logic ──────────────────────────────────────────────────

    def _sync_once(self) -> int:
        """Upload all closed segments that haven't been uploaded yet.

        Returns the number of segments uploaded in this sweep.
        """
        closed = self._event_logger.get_closed_segments()
        uploaded_count = 0

        for seg_path in closed:
            seg_name = seg_path.name
            if seg_name in self._uploaded:
                continue

            key = self._s3_key(seg_name)

            # Idempotent: check S3 before uploading
            if self._s3_object_exists(key):
                self._uploaded.add(seg_name)
                self._save_uploaded()
                continue

            if self._upload_file(seg_path, key):
                self._uploaded.add(seg_name)
                self._save_uploaded()
                uploaded_count += 1
                logger.info("Uploaded segment %s → s3://%s/%s", seg_name, self._bucket, key)

        return uploaded_count

    # ── Background thread ───────────────────────────────────────────

    def _run_loop(self) -> None:
        """Background thread main loop."""
        while not self._stop_event.is_set():
            try:
                self._sync_once()
            except Exception as e:
                logger.warning("S3 sync sweep failed: %s", e)
            self._stop_event.wait(self._poll_interval)

    def start(self) -> None:
        """Start the background sync thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="s3-sync-worker",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "S3SyncWorker started: bucket=%s prefix=%s/%s/%s",
            self._bucket, self._prefix, self._username, self._run_id,
        )

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the background thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def finalize(self) -> None:
        """Final sync: upload remaining closed segments + the last segment + manifest.

        Call this after EventLogger.close() so that ALL segments are closed.
        """
        # Final sweep: upload all remaining segments (all are closed now)
        all_segments = self._event_logger.get_all_segments()
        for seg_path in all_segments:
            seg_name = seg_path.name
            if seg_name in self._uploaded:
                continue
            key = self._s3_key(seg_name)
            if self._s3_object_exists(key):
                self._uploaded.add(seg_name)
                self._save_uploaded()
                continue
            if self._upload_file(seg_path, key):
                self._uploaded.add(seg_name)
                self._save_uploaded()
                logger.info("Uploaded segment %s → s3://%s/%s", seg_name, self._bucket, key)

        # Upload artifact files
        artifacts_dir = self._event_logger.run_dir / "artifacts" / "llm_calls"
        artifact_files: list[str] = []
        if artifacts_dir.is_dir():
            for artifact_path in sorted(artifacts_dir.glob("*.json")):
                relative = f"artifacts/llm_calls/{artifact_path.name}"
                key = self._s3_artifact_key(relative)
                if not self._s3_object_exists(key):
                    if self._upload_file(artifact_path, key):
                        logger.info(
                            "Uploaded artifact %s → s3://%s/%s",
                            artifact_path.name, self._bucket, key,
                        )
                artifact_files.append(relative)

        # Upload manifest with status=complete
        manifest = {
            "run_id": self._run_id,
            "username": self._username,
            "status": "complete",
            "segments": sorted(self._uploaded),
            "artifacts": artifact_files,
            "total_events": self._event_logger._logical_clock,
        }
        manifest_path = self._event_logger.run_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        key = self._manifest_key()
        if self._upload_file(manifest_path, key):
            logger.info("Uploaded manifest → s3://%s/%s", self._bucket, key)
