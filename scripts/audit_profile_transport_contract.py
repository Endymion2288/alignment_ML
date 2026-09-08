#!/usr/bin/env python3
"""Task B14T audit.  Standalone measurement transport contract."""

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
from datasets.profile_transport_contract import (
    decide,
    inherit_frozen_stage,
    inventory_and_audit,
    load_config,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/profile_transport_contract_v1.yaml")
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    inventory = inventory_and_audit(config)
    decision = decide(inventory, inherited)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sbb14t_profile_transport",
    )
    created = datetime.now(timezone.utc).isoformat()
    config_sha = sha256_file(config_path)
    helper_sha = sha256_file(
        resolve_under_root(
            project_root(),
            "alignment/leave_target_out_dump/CkfLeaveTargetOutDumpAlg.cxx",
        )
    )
    common = {
        "created_utc": created,
        "config_sha256": config_sha,
        "git_head_sha": git_head_sha(),
        "current_helper_sha256": helper_sha,
        "wb109_dumps_not_overwritten": True,
        "wb114_dumps_not_overwritten": True,
        "wb115_dumps_not_overwritten": True,
        "prior_introduced": False,
        "ridge_added": False,
        "statistical_model_unchanged": True,
        "profile_math_rewritten": False,
        "measurement_update_in_evaluator": False,
    }
    store.write_json(
        "transport_failure_taxonomy.json",
        jsonable({"kind": "transport_failure_taxonomy", **common, **inventory["taxonomy"]}),
    )
    store.write_json(
        "acts_kalman_transport_semantics.json",
        jsonable({"kind": "acts_kalman_transport_semantics", **common, **inventory["acts_semantics"]}),
    )
    store.write_json(
        "magnet_crossing_transport_audit.json",
        jsonable({"kind": "magnet_crossing_transport_audit", **common, **inventory["magnet"]}),
    )
    store.write_json(
        "transport_state_unit_contract.json",
        jsonable({"kind": "transport_state_unit_contract", **common, **inventory["units"]}),
    )
    store.write_json(
        "magnet_stepper_pathology.json",
        jsonable({"kind": "magnet_stepper_pathology", **common, **inventory["pathology"]}),
    )
    store.write_json(
        "stereo_surface_geometry_audit.json",
        jsonable({"kind": "stereo_surface_geometry_audit", **common, **inventory["stereo"]}),
    )
    store.write_json(
        "measurement_surface_projection_contract.json",
        jsonable({"kind": "measurement_surface_projection_contract", **common, **inventory["projection"]}),
    )
    store.write_json(
        "standalone_likelihood_transport_contract.json",
        jsonable({"kind": "standalone_likelihood_transport_contract", **common, **inventory["likelihood_contract"]}),
    )
    store.write_json(
        "sequential_vs_direct_prediction.json",
        jsonable({"kind": "sequential_vs_direct_prediction", **common, **inventory["sequential_vs_direct"]}),
    )
    store.write_json(
        "transport_jacobian_spotcheck.json",
        jsonable({"kind": "transport_jacobian_spotcheck", **common, **inventory["jacobian"]}),
    )
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json(
        "profile_transport_contract_decision.json",
        jsonable(
            {
                **common,
                "present_sources": inventory["present_sources"],
                "n_contracted": inventory["n_contracted"],
                "n_official_pairs": inventory["n_official_pairs"],
                "n_raw": inventory["n_raw"],
                "n_ineligible": inventory["n_ineligible"],
                "exclusion": inventory["exclusion"],
                "smoke_gate": inventory["smoke_gate"],
                "denominator": inventory["denominator"],
                "evaluations": inventory["evaluations"],
                "wb115_consistency": inventory["wb115_consistency"],
                **decision,
            }
        ),
    )
    store.finalize(
        {
            "verdict": decision["verdict"],
            "task": "SB-B14T",
            "workbook": 116,
            "decision": decision["decision"],
            "primary_case": decision["primary_case"],
            "next_step": decision["next_step"],
            "config_sha256": config_sha,
            "inherited_wb115_decision_sha256": decision["inherited_wb115_decision_sha256"],
            "inherited_wb114_decision_sha256": decision["inherited_wb114_decision_sha256"],
            "inherited_wb113_decision_sha256": decision["inherited_wb113_decision_sha256"],
            "inherited_wb112_decision_sha256": decision["inherited_wb112_decision_sha256"],
            "inherited_wb111_decision_sha256": decision["inherited_wb111_decision_sha256"],
            "inherited_wb110_decision_sha256": decision["inherited_wb110_decision_sha256"],
            "inherited_wb109_decision_sha256": decision["inherited_wb109_decision_sha256"],
            "inherited_wb103_contract_sha256": decision["inherited_wb103_contract_sha256"],
            "official_input_scope": "contract_eligible",
            "raw_ckf_used": False,
            "measurement_model_v2_entered": False,
            "measurement_model_v2_authorized": False,
            "b15_authorized": False,
            "b14m_reopen_authorized": decision["b14m_reopen_authorized"],
            "do_not_force_5d_lto_covariance": True,
            "prior_introduced": False,
            "statistical_model_unchanged": True,
            "smoke_gate_passed": decision["smoke_gate_passed"],
            "full_sample_authorized": False,
            "wb109_dumps_not_overwritten": True,
            "wb114_dumps_not_overwritten": True,
            "wb115_dumps_not_overwritten": True,
            "focus_identity_retained": True,
            "jacobian_contract_established": decision["jacobian_contract_established"],
        }
    )
    print(
        f"SB-B14T verdict={decision['verdict']} case={decision['primary_case']} "
        f"gate={decision['smoke_gate_passed']} next={decision['next_step']} "
        f"run_id={store.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
