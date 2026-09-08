#!/usr/bin/env python3
"""Task B7 audit.  Official contract validation with frozen WB98 replay."""

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
    decide,
    inherit_frozen_stage,
    inventory_and_validate,
    load_config,
)
from evaluation.artifact_store import ImmutableArtifactStore
from scripts.run_ckf_contract_validation import write_validation_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ckf_reconstruction_contract_v1.yaml")
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    inventory = inventory_and_validate(config)
    decision = decide(inventory, inherited)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sbb7_ckf_contract_validation",
    )
    created = datetime.now(timezone.utc).isoformat()
    config_sha = sha256_file(config_path)
    write_validation_artifacts(store, inventory, config_sha)
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json(
        "ckf_contract_validation_decision.json",
        jsonable(
            {
                "created_utc": created,
                "config_sha256": config_sha,
                "git_head_sha": git_head_sha(),
                "present_sources": inventory["present_sources"],
                "missing_sources": inventory["missing_sources"],
                "n_dump_rows": inventory["n_dump_rows"],
                "frozen_1pct_rescreened": inventory["frozen_1pct_rescreened"],
                **decision,
            }
        ),
    )
    store.finalize(
        {
            "verdict": decision["verdict"],
            "task": "SB-B7",
            "workbook": 103,
            "decision": decision["decision"],
            "primary_case": decision["primary_case"],
            "next_step": decision["next_step"],
            "config_sha256": config_sha,
            "inherited_wb102_decision_sha256": decision["inherited_wb102_decision_sha256"],
            "closure_pass": False,
            "filter_then_pass_claimed": False,
            "covariance_model_fixed": False,
            "closure_under_contracted_input_scope": decision[
                "closure_under_contracted_input_scope"
            ],
        }
    )
    print(
        f"SB-B7 verdict={decision['verdict']} case={decision['primary_case']} "
        f"next={decision['next_step']} run_id={store.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
