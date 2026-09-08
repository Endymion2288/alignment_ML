#!/usr/bin/env python3
"""Task B13 audit.  Independent leave-target-out state materialization."""

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
from datasets.leave_target_out_state_materialization import (
    decide,
    inherit_frozen_stage,
    inventory_and_audit,
    load_config,
    plugin_library_path,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/leave_target_out_state_materialization_v1.yaml"
    )
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    inventory = inventory_and_audit(config)
    decision = decide(inventory, inherited)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sbb13_leave_target_out_state",
    )
    created = datetime.now(timezone.utc).isoformat()
    config_sha = sha256_file(config_path)
    helper = plugin_library_path(config)
    common = {
        "created_utc": created,
        "config_sha256": config_sha,
        "git_head_sha": git_head_sha(),
        "helper_sha256": sha256_file(helper) if helper.is_file() else None,
    }
    store.write_json(
        "leave_target_out_measurement_removal_contract.json",
        jsonable(
            {
                "kind": "leave_target_out_measurement_removal_contract",
                **common,
                **inventory["measurement_removal"],
            }
        ),
    )
    store.write_json(
        "leave_target_out_state_inventory.json",
        jsonable(
            {
                "kind": "leave_target_out_state_inventory",
                **common,
                **inventory["states"],
                **inventory["dumps"],
                "dump_wide_states": inventory.get("dump_states"),
            }
        ),
    )
    store.write_json(
        "leave_target_out_covariance_contract.json",
        jsonable(
            {
                "kind": "leave_target_out_covariance_contract",
                **common,
                **inventory["covariance"],
            }
        ),
    )
    store.write_json(
        "leave_target_out_denominator.json",
        jsonable(
            {
                "kind": "leave_target_out_denominator",
                **common,
                **inventory["denominator"],
            }
        ),
    )
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json(
        "leave_target_out_state_materialization_decision.json",
        jsonable(
            {
                **common,
                "present_sources": inventory["present_sources"],
                "missing_sources": inventory["missing_sources"],
                "n_contracted": inventory["n_contracted"],
                "n_official_pairs": inventory["n_official_pairs"],
                "n_raw": inventory["n_raw"],
                "n_ineligible": inventory["n_ineligible"],
                **decision,
            }
        ),
    )
    store.finalize(
        {
            "verdict": decision["verdict"],
            "task": "SB-B13",
            "workbook": 109,
            "decision": decision["decision"],
            "primary_case": decision["primary_case"],
            "next_step": decision["next_step"],
            "config_sha256": config_sha,
            "inherited_wb108_decision_sha256": decision["inherited_wb108_decision_sha256"],
            "inherited_wb107_decision_sha256": decision["inherited_wb107_decision_sha256"],
            "inherited_wb106_decision_sha256": decision["inherited_wb106_decision_sha256"],
            "inherited_wb103_contract_sha256": decision["inherited_wb103_contract_sha256"],
            "official_input_scope": "contract_eligible",
            "raw_ckf_used": False,
            "measurement_model_v2_entered": False,
            "measurement_model_v2_authorized": False,
            "transport_covariance_validated": False,
            "independence_proven": decision["independence_proven"],
            "lto_states_materialized": decision["lto_states_materialized"],
            "official_WB107_Cin_reused": decision["official_WB107_Cin_reused"],
            "seed_covariance_used_as_output_covariance": decision[
                "seed_covariance_used_as_output_covariance"
            ],
            "b14_authorized": decision["b14_authorized"],
            "b15_authorized": False,
            "target_exclusion_proven": decision["target_exclusion_proven"],
            "focus_identity_retained": True,
            "kalman_fitter_tool_fit_called": False,
        }
    )
    print(
        f"SB-B13 verdict={decision['verdict']} case={decision['primary_case']} "
        f"next={decision['next_step']} run_id={store.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
