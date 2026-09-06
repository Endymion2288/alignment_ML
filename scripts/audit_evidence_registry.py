#!/usr/bin/env python3
"""T00 evidence-registry driver.  Metadata and hashes only; no production."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from alignment.cad_survey_nov22 import git_head_sha
from alignment.gauge_fixed_real_data_diagnostic import _json_ready
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from datasets.evidence_registry import DEFAULT_CONFIG, build_registry, load_config


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    payload = build_registry(config)
    root = resolve_under_root(project_root(), str(config["output_root"]))
    config_sha = sha256_file(config_path)
    stamp = {
        "kind": "t00_config_validation",
        "task": "T00",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "git_head_sha": git_head_sha(),
        "schema_version": config["schema_version"],
        "geometry_write_allowed": False,
        "sealed_test_opened": False,
        "production_ran": False,
    }
    payload["evidence_registry"]["config_sha256"] = config_sha
    payload["claim_reconciliation"]["config_sha256"] = config_sha
    payload["external_runtime_manifest"]["config_sha256"] = config_sha
    _write_json(root / "config_validation.json", stamp)
    _write_json(root / "evidence_registry.json", payload["evidence_registry"])
    _write_json(root / "claim_reconciliation.json", payload["claim_reconciliation"])
    _write_json(root / "external_runtime_manifest.json", payload["external_runtime_manifest"])
    _write_json(root / "e09_file_rehash.json", payload["e09_file_rehash"])
    verdict = str(payload["claim_reconciliation"]["verdict"])
    print(f"T00 verdict={verdict} output={root}")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
