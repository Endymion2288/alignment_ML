"""T01 immutable artifact lifecycle: unique run ID, no overwrite, atomic finalize."""

from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class ArtifactStoreError(RuntimeError):
    """Raised when a run directory or file violates the immutable contract."""


def new_run_id(campaign: str, *, clock: datetime | None = None) -> str:
    stamp = (clock or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{campaign}_{stamp}_{secrets.token_hex(4)}"


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    dest = Path(path)
    if dest.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()


class ImmutableArtifactStore:
    """Exclusive run directory.  Incomplete until finalize() writes COMPLETE.json."""

    def __init__(self, run_dir: Path, *, run_id: str, campaign: str):
        self.run_dir = Path(run_dir)
        self.run_id = str(run_id)
        self.campaign = str(campaign)
        self._finalized = False

    @classmethod
    def begin(cls, campaign_root: str | Path, campaign: str) -> "ImmutableArtifactStore":
        root = Path(campaign_root)
        root.mkdir(parents=True, exist_ok=True)
        run_id = new_run_id(campaign)
        run_dir = root / run_id
        if run_dir.exists():
            raise FileExistsError(f"run directory already exists: {run_dir}")
        run_dir.mkdir(parents=False)
        store = cls(run_dir, run_id=run_id, campaign=campaign)
        write_json_atomic(
            run_dir / "run_manifest.json",
            {
                "kind": "run_manifest",
                "campaign": campaign,
                "run_id": run_id,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "complete": False,
                "geometry_write_allowed": False,
            },
        )
        return store

    def write_json(self, relative: str, payload: Mapping[str, Any]) -> Path:
        if self._finalized:
            raise ArtifactStoreError("cannot write after finalize")
        dest = self.run_dir / relative
        if dest.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {dest}")
        write_json_atomic(dest, payload)
        return dest

    def is_complete(self) -> bool:
        return (self.run_dir / "COMPLETE.json").is_file()

    def finalize(self, extra: Mapping[str, Any] | None = None) -> Path:
        if self.is_complete():
            raise FileExistsError(f"run already finalized: {self.run_dir}")
        payload = {
            "kind": "run_complete",
            "campaign": self.campaign,
            "run_id": self.run_id,
            "finalized_utc": datetime.now(timezone.utc).isoformat(),
            "complete": True,
            **dict(extra or {}),
        }
        path = self.write_json("COMPLETE.json", payload)
        self._finalized = True
        return path
