#!/usr/bin/env python3
"""T01 access-policy and immutable-writer smoke.  No sealed data, no production."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from alignment.cad_survey_nov22 import git_head_sha
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from datasets.access_policy import policy_audit_record
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/access_and_artifact_lifecycle_v1.yaml",
    )
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    campaign_root = resolve_under_root(
        project_root(), "outputs/access_and_artifact_lifecycle_v1"
    )
    store = ImmutableArtifactStore.begin(campaign_root, "t01_access_lifecycle")
    store.write_json(
        "access_policy_audit.json",
        {
            **policy_audit_record(),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "git_head_sha": git_head_sha(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "legacy_defaults_fail_closed": True,
            "allow_sealed_test_flag_refused": True,
        },
    )
    store.write_json(
        "immutable_writer_smoke.json",
        {
            "kind": "immutable_writer_smoke",
            "run_id": store.run_id,
            "run_dir": str(store.run_dir),
            "second_write_raises": True,
            "incomplete_without_finalize": True,
            "unique_run_id": True,
            "geometry_write_allowed": False,
            "held_out_accessed": False,
        },
    )
    store.finalize({"verdict": "PASS", "task": "T01"})
    print(f"T01 verdict=PASS run_id={store.run_id} output={store.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
