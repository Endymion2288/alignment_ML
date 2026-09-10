#!/usr/bin/env python3
"""Task B14J audit.  Supporting-plane Jacobian continuity contract."""

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
from datasets.supporting_plane_jacobian_continuity import (
    decide,
    inherit_frozen_stage,
    inventory_and_audit,
    load_config,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/supporting_plane_jacobian_continuity_v1.yaml"
    )
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    inventory = inventory_and_audit(config)
    decision = decide(inventory, inherited)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sbb14j_jacobian_continuity",
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
        "wb116_dumps_not_overwritten": True,
        "prior_introduced": False,
        "ridge_added": False,
        "statistical_model_unchanged": True,
        "profile_math_rewritten": False,
        "measurement_update_in_evaluator": False,
        "do_not_select_best_step": True,
        "do_not_switch_to_direct": True,
        "five_d_cin_not_the_objective": True,
    }
    store.write_json(
        "jacobian_continuity_ladder.json",
        jsonable({"kind": "jacobian_continuity_ladder", **common, **inventory["jacobian"]}),
    )
    store.write_json(
        "sequential_vs_direct_jacobian.json",
        jsonable(
            {
                "kind": "sequential_vs_direct_jacobian",
                **common,
                **inventory["sequential_vs_direct"],
            }
        ),
    )
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json(
        "supporting_plane_jacobian_continuity_decision.json",
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
                **decision,
            }
        ),
    )
    store.finalize(
        {
            "verdict": decision["verdict"],
            "task": "SB-B14J",
            "workbook": 117,
            "decision": decision["decision"],
            "primary_case": decision["primary_case"],
            "next_step": decision["next_step"],
            "config_sha256": config_sha,
            "inherited_wb116_decision_sha256": decision["inherited_wb116_decision_sha256"],
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
            "restart_invariance_authorized": decision["restart_invariance_authorized"],
            "do_not_force_5d_lto_covariance": True,
            "prior_introduced": False,
            "statistical_model_unchanged": True,
            "smoke_gate_passed": decision["smoke_gate_passed"],
            "full_sample_authorized": False,
            "wb109_dumps_not_overwritten": True,
            "wb114_dumps_not_overwritten": True,
            "wb115_dumps_not_overwritten": True,
            "wb116_dumps_not_overwritten": True,
            "focus_identity_retained": True,
            "jacobian_contract_established": decision["jacobian_contract_established"],
            "do_not_select_best_step": True,
            "do_not_switch_to_direct": True,
            "five_d_cin_not_the_objective": True,
        }
    )
    print(
        f"SB-B14J verdict={decision['verdict']} case={decision['primary_case']} "
        f"gate={decision['smoke_gate_passed']} jac={decision['jacobian_contract_established']} "
        f"next={decision['next_step']} run_id={store.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
