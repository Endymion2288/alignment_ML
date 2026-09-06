"""T00 evidence registry: hash-locked claims, no sealed-test reads."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from alignment.cad_survey_nov22 import git_head_sha
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)

SCHEMA_VERSION = "master-audit-reconciliation-v1"
DEFAULT_CONFIG = "configs/audit_evidence_registry_v1.yaml"
ALLOWED_KINDS = (
    "inference",
    "paired_response_analysis",
    "real_data_monitoring",
)
AUDIT_BASELINE_SHA = "0c8a6ca34c6c8b4f505dc153dd0ca8cc7949ce56"


class EvidenceRegistryError(ValueError):
    """Raised when the T00 registry contract is violated."""


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise EvidenceRegistryError("T00 config must be a mapping")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceRegistryError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != "T00":
        raise EvidenceRegistryError("task must be T00")
    if bool(config.get("geometry_write_allowed", True)):
        raise EvidenceRegistryError("geometry_write_allowed must be false")
    if bool(config.get("do_not_reread_sealed_test", False)) is not True:
        raise EvidenceRegistryError("do_not_reread_sealed_test must be true")
    if bool(config.get("do_not_backfill_post_audit_into_master_audit", False)) is not True:
        raise EvidenceRegistryError("backfill into the master audit is forbidden")
    if str(config.get("audit_baseline_sha")) != AUDIT_BASELINE_SHA:
        raise EvidenceRegistryError("audit_baseline_sha must stay pinned")
    return dict(config)


def _status(path: Path, expected: str) -> dict[str, Any]:
    if not path.is_file():
        return {
            "exists": False,
            "status": "MISSING",
            "actual_sha256": None,
            "sha256_match": False,
        }
    actual = sha256_file(path)
    match = actual == str(expected)
    return {
        "exists": True,
        "status": "PASS" if match else "HASH_MISMATCH",
        "actual_sha256": actual,
        "sha256_match": match,
    }


def verify_records(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    root = project_root()
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    rows: list[dict[str, Any]] = []
    for spec in config["evidence"]:
        evidence_id = str(spec["id"])
        kind = str(spec["kind"])
        relative = str(spec["relative_path"])
        expected = str(spec["expected_sha256"])
        if evidence_id in seen_ids:
            raise EvidenceRegistryError(f"duplicate evidence id: {evidence_id}")
        if relative in seen_paths:
            raise EvidenceRegistryError(f"duplicate evidence path: {relative}")
        if kind not in ALLOWED_KINDS:
            raise EvidenceRegistryError(f"unknown evidence kind: {kind}")
        if len(expected) != 64:
            raise EvidenceRegistryError(f"missing/invalid sha256 for {evidence_id}")
        if "sealed" in relative.lower() and "/test" in f"/{relative.lower()}":
            raise EvidenceRegistryError("registry must not point at sealed test event files")
        seen_ids.add(evidence_id)
        seen_paths.add(relative)
        path = resolve_under_root(root, relative)
        checked = _status(path, expected)
        rows.append(
            {
                "id": evidence_id,
                "kind": kind,
                "relative_path": relative,
                "expected_sha256": expected,
                "lineage": spec.get("lineage", "audit_baseline"),
                "claim": spec.get("claim"),
                "frozen_negative": bool(spec.get("frozen_negative", False)),
                **checked,
            }
        )
    return rows


def verify_e09_manifest_files(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Re-hash the 13 Operating Protocol V1 freeze files.  No event content."""
    root = project_root()
    e09 = next((item for item in config["evidence"] if item["id"] == "E09"), None)
    if e09 is None or not bool(e09.get("rehash_manifest_files", True)):
        return []
    manifest_path = resolve_under_root(root, str(e09["relative_path"]))
    if not manifest_path.is_file():
        return [{"status": "MISSING", "path": str(manifest_path)}]
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for item in payload.get("files", []):
        path = Path(str(item["path"]))
        expected = str(item["sha256"])
        checked = _status(path, expected)
        rows.append(
            {
                "role": item.get("role"),
                "path": str(path),
                "expected_sha256": expected,
                **checked,
            }
        )
    return rows


def dirty_tree_sha(root: Path | None = None) -> str:
    """Hash of `git status --porcelain` plus HEAD.  Empty tree is still hashed."""
    import subprocess

    cwd = project_root() if root is None else Path(root)
    head = git_head_sha(cwd) or "unknown"
    try:
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        porcelain = "UNAVAILABLE\n"
    return hashlib.sha256(f"{head}\n{porcelain}".encode("utf-8")).hexdigest()


def runtime_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    pins = dict(config.get("software_provenance_pins") or {})
    return {
        "kind": "external_runtime_manifest",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head_sha": git_head_sha(),
        "audit_baseline_sha": AUDIT_BASELINE_SHA,
        "dirty_tree_sha256": dirty_tree_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "calypso_git_sha": pins.get("calypso_git_sha"),
        "athena_release": pins.get("athena_release"),
        "acts_version": pins.get("acts_version"),
        "historical_runtime_unverified": bool(
            pins.get("historical_runtime_unverified", True)
        ),
        "note": pins.get("note"),
        "geometry_write_allowed": False,
        "sealed_test_opened": False,
    }


def reconcile(
    config: Mapping[str, Any], records: list[Mapping[str, Any]], e09: list[Mapping[str, Any]]
) -> dict[str, Any]:
    all_pass = all(row.get("status") == "PASS" for row in records) and all(
        row.get("status") == "PASS" for row in e09 if "status" in row
    )
    return {
        "kind": "claim_reconciliation",
        "audit_baseline_sha": AUDIT_BASELINE_SHA,
        "current_head_sha": git_head_sha(),
        "commits_after_audit_baseline": None,
        "do_not_backfill_post_audit_into_master_audit": True,
        "pipeline_entries": dict(config["pipeline_entries"]),
        "findings": dict(config["findings"]),
        "post_audit_lineage": config["post_audit_lineage"],
        "evidence_all_pass": bool(all_pass),
        "e09_files_all_pass": all(row.get("status") == "PASS" for row in e09),
        "n_evidence": len(records),
        "n_e09_files": len(e09),
        "missing_or_mismatch": [
            row["id"] for row in records if row.get("status") != "PASS"
        ],
        "verdict": "PASS" if all_pass else "INCOMPLETE",
        "geometry_write_allowed": False,
        "measurement_model_validated": False,
        "real_data_alignment_authorized": False,
    }


def build_registry(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    loaded = load_config() if config is None else dict(config)
    records = verify_records(loaded)
    e09 = verify_e09_manifest_files(loaded)
    runtime = runtime_manifest(loaded)
    claims = reconcile(loaded, records, e09)
    try:
        import subprocess

        count = subprocess.run(
            ["git", "rev-list", "--count", f"{AUDIT_BASELINE_SHA}..HEAD"],
            cwd=str(project_root()),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        claims["commits_after_audit_baseline"] = int(count)
    except (OSError, subprocess.CalledProcessError, ValueError):
        claims["commits_after_audit_baseline"] = None
    return {
        "evidence_registry": {
            "kind": "evidence_registry",
            "schema_version": SCHEMA_VERSION,
            "records": records,
        },
        "claim_reconciliation": claims,
        "external_runtime_manifest": runtime,
        "e09_file_rehash": {"kind": "e09_file_rehash", "files": e09},
    }
