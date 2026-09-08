#!/usr/bin/env python3
"""Task B8 run.  Transport Covariance V3 on the WB103 contracted input."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from alignment.cad_survey_nov22 import git_head_sha
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from datasets.acts_transport_diagnosis import jsonable
from datasets.transport_covariance_validation_v3 import inventory_and_validate, load_config
from evaluation.artifact_store import ImmutableArtifactStore


def write_v3_artifacts(store: ImmutableArtifactStore, inventory: dict, config_sha: str) -> None:
    created = datetime.now(timezone.utc).isoformat()
    store.write_json(
        "contracted_input_inventory.json",
        jsonable(
            {
                "kind": "contracted_input_inventory",
                "created_utc": created,
                "config_sha256": config_sha,
                **inventory["inventory"],
            }
        ),
    )
    store.write_json(
        "transport_covariance_v3_metrics.json",
        jsonable(
            {
                "kind": "transport_covariance_v3_metrics",
                "created_utc": created,
                "config_sha256": config_sha,
                **inventory["metrics"],
            }
        ),
    )
    store.write_json(
        "eigenvalue_pencil_analysis.json",
        jsonable(
            {
                "kind": "eigenvalue_pencil_analysis",
                "created_utc": created,
                "config_sha256": config_sha,
                **inventory["eigen"],
            }
        ),
    )
    store.write_json(
        "ckf_tail_case_study.json",
        jsonable(
            {
                "kind": "ckf_tail_case_study",
                "created_utc": created,
                "config_sha256": config_sha,
                **inventory["focus"],
            }
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/transport_covariance_validation_v3.yaml")
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inventory = inventory_and_validate(config)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sbb8_transport_covariance_v3",
    )
    config_sha = sha256_file(config_path)
    write_v3_artifacts(store, inventory, config_sha)
    store.write_json(
        "run_inventory.json",
        jsonable(
            {
                "kind": "transport_covariance_v3_run",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "config_sha256": config_sha,
                "git_head_sha": git_head_sha(),
                "present_sources": inventory["present_sources"],
                "missing_sources": inventory["missing_sources"],
                "mechanism": inventory["mechanism"],
            }
        ),
    )
    store.finalize(
        {
            "verdict": "RAN",
            "task": "SB-B8",
            "workbook": 104,
            "config_sha256": config_sha,
            "primary_case": inventory["mechanism"]["primary_case"],
            "official_input_scope": "contract_eligible",
            "raw_ckf_used": False,
        }
    )
    print(
        f"SB-B8 run case={inventory['mechanism']['primary_case']} run_id={store.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
