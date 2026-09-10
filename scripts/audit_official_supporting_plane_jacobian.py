#!/usr/bin/env python3
"""Task B14U audit.  Official supporting-plane path Jacobian vs fixed FD."""

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
from datasets.official_supporting_plane_jacobian import (
    decide,
    inherit_frozen_stage,
    inventory_and_audit,
    load_config,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/official_supporting_plane_jacobian_v1.yaml"
    )
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    inventory = inventory_and_audit(config)
    decision = decide(inventory, inherited)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sbb14u_official_jacobian",
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
        "wb117_dumps_not_overwritten": True,
        "wb118_dumps_not_overwritten": True,
        "wb119_dumps_not_overwritten": True,
        "prior_introduced": False,
        "ridge_added": False,
        "statistical_model_unchanged": True,
        "do_not_select_best_step": True,
        "do_not_select_best_tolerance": True,
        "do_not_switch_to_direct": True,
        "do_not_replace_official_likelihood": True,
        "do_not_use_dummy_cov_bounded_transportJacobian": True,
        "five_d_cin_not_the_objective": True,
        "wb119_took_wrong_jacobian": True,
        "wb119_not_a_physical_conclusion": True,
    }
    store.write_json(
        "official_path_jacobian_vs_fd_ladder.json",
        jsonable(
            {
                "kind": "official_path_jacobian_vs_fd_ladder",
                **common,
                "targets": [
                    {
                        "event": item.get("event"),
                        "target_station": item.get("target_station"),
                        "free_jacobian_available": item.get(
                            "free_jacobian_available"
                        ),
                        "loc0_matches_official": item.get("loc0_matches_official"),
                        "n_hops": item.get("n_hops"),
                        "n_available_hops": item.get("n_available_hops"),
                        "n_stepped_hops": item.get("n_stepped_hops"),
                        "n_stepped_available": item.get("n_stepped_available"),
                        "comparisons": item.get("comparisons"),
                        "fd_columns": [
                            {
                                "parameter": col.get("parameter"),
                                "ladder_converged": col.get("ladder_converged"),
                                "last_pair_relative_error": col.get(
                                    "last_pair_relative_error"
                                ),
                                "official_norm": col.get("official_norm"),
                            }
                            for col in (item.get("fd_columns") or [])
                        ],
                    }
                    for item in inventory.get("targets") or []
                ],
            }
        ),
    )
    store.write_json(
        "official_path_hop_decomposition.json",
        jsonable(
            {
                "kind": "official_path_hop_decomposition",
                **common,
                "targets": [
                    {
                        "event": item.get("event"),
                        "target_station": item.get("target_station"),
                        "loc0_matches_official": item.get("loc0_matches_official"),
                        "continuation_constructions": item.get(
                            "continuation_constructions"
                        ),
                        "branch_fallback": item.get("branch_fallback"),
                        "official_jac_transport_is_identity": item.get(
                            "official_jac_transport_is_identity"
                        ),
                        "chain": item.get("chain"),
                    }
                    for item in inventory.get("targets") or []
                ],
            }
        ),
    )
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json(
        "official_supporting_plane_jacobian_decision.json",
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
                "n_control_targets": len(inventory.get("controls") or []),
                "n_focus_targets": len(inventory.get("focus_86") or []),
                **decision,
            }
        ),
    )
    store.finalize(
        {
            "verdict": decision["verdict"],
            "task": "SB-B14U",
            "workbook": 120,
            "decision": decision["decision"],
            "primary_case": decision["primary_case"],
            "next_step": decision["next_step"],
            "config_sha256": config_sha,
            "inherited_wb119_decision_sha256": decision[
                "inherited_wb119_decision_sha256"
            ],
            "inherited_wb118_decision_sha256": decision[
                "inherited_wb118_decision_sha256"
            ],
            "official_input_scope": "contract_eligible",
            "raw_ckf_used": False,
            "measurement_model_v2_entered": False,
            "measurement_model_v2_authorized": False,
            "b15_authorized": False,
            "b14m_reopen_authorized": decision["b14m_reopen_authorized"],
            "restart_invariance_authorized": False,
            "do_not_force_5d_lto_covariance": True,
            "prior_introduced": False,
            "statistical_model_unchanged": True,
            "smoke_gate_passed": decision["smoke_gate_passed"],
            "full_sample_authorized": False,
            "wb109_dumps_not_overwritten": True,
            "wb119_dumps_not_overwritten": True,
            "focus_identity_retained": True,
            "jacobian_contract_established": decision["jacobian_contract_established"],
            "do_not_select_best_step": True,
            "do_not_select_best_tolerance": True,
            "do_not_replace_official_likelihood": True,
            "do_not_use_dummy_cov_bounded_transportJacobian": True,
            "five_d_cin_not_the_objective": True,
            "wb119_took_wrong_jacobian": True,
        }
    )
    print(
        f"SB-B14U verdict={decision['verdict']} case={decision['primary_case']} "
        f"gate={decision['smoke_gate_passed']} "
        f"contract={decision['jacobian_contract_established']} "
        f"b14m={decision['b14m_reopen_authorized']} "
        f"next={decision['next_step']} run_id={store.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
