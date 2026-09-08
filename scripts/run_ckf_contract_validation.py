#!/usr/bin/env python3
"""Task B7 run.  Apply the reconstruction contract and replay WB98 closure."""

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
from datasets.ckf_contract_validation import (
    SAMPLE_A,
    SAMPLE_B,
    inventory_and_validate,
    load_config,
)
from evaluation.artifact_store import ImmutableArtifactStore


def write_validation_artifacts(store: ImmutableArtifactStore, inventory: dict, config_sha: str) -> None:
    created = datetime.now(timezone.utc).isoformat()
    store.write_json(
        "ckf_contract_sample_comparison.json",
        jsonable(
            {
                "kind": "ckf_contract_sample_comparison",
                "created_utc": created,
                "config_sha256": config_sha,
                **inventory["comparison"],
            }
        ),
    )
    store.write_json(
        "ckf_contract_closure_replay.json",
        jsonable(
            {
                "kind": "ckf_contract_closure_replay",
                "created_utc": created,
                "config_sha256": config_sha,
                "raw_CKF": inventory["closure_replay"][SAMPLE_A],
                "contract_eligible": inventory["closure_replay"][SAMPLE_B],
                "filter_then_pass_claimed": False,
                "covariance_model_fixed": False,
            }
        ),
    )
    store.write_json(
        "ckf_longtrack_residual_failure.json",
        jsonable(
            {
                "kind": "ckf_longtrack_residual_failure",
                "created_utc": created,
                "config_sha256": config_sha,
                **inventory["remaining_tails"],
            }
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ckf_reconstruction_contract_v1.yaml")
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inventory = inventory_and_validate(config)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sbb7_ckf_contract_validation",
    )
    config_sha = sha256_file(config_path)
    write_validation_artifacts(store, inventory, config_sha)
    store.write_json(
        "run_inventory.json",
        jsonable(
            {
                "kind": "ckf_contract_validation_run",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "config_sha256": config_sha,
                "git_head_sha": git_head_sha(),
                "present_sources": inventory["present_sources"],
                "missing_sources": inventory["missing_sources"],
                "n_dump_rows": inventory["n_dump_rows"],
                "mechanism": inventory["mechanism"],
            }
        ),
    )
    store.finalize(
        {
            "verdict": "RAN",
            "task": "SB-B7",
            "workbook": 103,
            "config_sha256": config_sha,
            "primary_case": inventory["mechanism"]["primary_case"],
            "closure_pass": False,
            "filter_then_pass_claimed": False,
        }
    )
    print(
        f"SB-B7 run case={inventory['mechanism']['primary_case']} "
        f"run_id={store.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
