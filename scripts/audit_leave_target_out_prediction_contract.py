#!/usr/bin/env python3
"""Task B10 audit.  Official leave-target-out prediction contract."""

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
from datasets.leave_target_out_prediction_contract import (
    decide,
    inherit_frozen_stage,
    inventory_and_audit,
    load_config,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/leave_target_out_prediction_contract_v1.yaml"
    )
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    inventory = inventory_and_audit(config)
    decision = decide(inventory, inherited)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sbb10_leave_target_out",
    )
    created = datetime.now(timezone.utc).isoformat()
    config_sha = sha256_file(config_path)
    common = {
        "created_utc": created,
        "config_sha256": config_sha,
        "git_head_sha": git_head_sha(),
    }
    store.write_json(
        "leave_target_out_dependency_contract.json",
        jsonable({"kind": "leave_target_out_dependency_contract", **common, **inventory["interfaces"]}),
    )
    store.write_json(
        "leave_target_out_state_inventory.json",
        jsonable({"kind": "leave_target_out_state_inventory", **common, **inventory["inventory"]}),
    )
    store.write_json(
        "leave_target_out_covariance_contract.json",
        jsonable({"kind": "leave_target_out_covariance_contract", **common, **inventory["covariance"]}),
    )
    store.write_json(
        "target_independence_audit.json",
        jsonable({"kind": "target_independence_audit", **common, **inventory["independence"]}),
    )
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json(
        "leave_target_out_prediction_decision.json",
        jsonable(
            {
                **common,
                "present_sources": inventory["present_sources"],
                "missing_sources": inventory["missing_sources"],
                "n_contracted": inventory["n_contracted"],
                "n_official_pairs": inventory["n_official_pairs"],
                **decision,
            }
        ),
    )
    store.finalize(
        {
            "verdict": decision["verdict"],
            "task": "SB-B10",
            "workbook": 106,
            "decision": decision["decision"],
            "primary_case": decision["primary_case"],
            "next_step": decision["next_step"],
            "config_sha256": config_sha,
            "inherited_wb105_decision_sha256": decision["inherited_wb105_decision_sha256"],
            "inherited_wb103_contract_sha256": decision["inherited_wb103_contract_sha256"],
            "official_input_scope": "contract_eligible",
            "raw_ckf_used": False,
            "measurement_model_v2_entered": False,
            "measurement_model_v2_authorized": False,
            "lto_states_materialized": False,
            "independence_proven": False,
            "focus_identity_retained": True,
        }
    )
    print(
        f"SB-B10 verdict={decision['verdict']} case={decision['primary_case']} "
        f"next={decision['next_step']} run_id={store.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
