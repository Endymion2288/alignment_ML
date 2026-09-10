#!/usr/bin/env python3
"""Task B14L audit.  ACTS transport Jacobian vs fixed FD ladder."""

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
from datasets.acts_fd_derivative_contract import (
    decide,
    inherit_frozen_stage,
    inventory_and_audit,
    load_config,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/acts_fd_derivative_contract_v1.yaml"
    )
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    inventory = inventory_and_audit(config)
    decision = decide(inventory, inherited)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sbb14l_derivative_contract",
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
        "prior_introduced": False,
        "ridge_added": False,
        "statistical_model_unchanged": True,
        "do_not_select_best_step": True,
        "do_not_select_best_tolerance": True,
        "do_not_switch_to_direct": True,
        "do_not_replace_official_likelihood": True,
        "five_d_cin_not_the_objective": True,
        "wb118_not_a_physical_conclusion": True,
    }
    store.write_json(
        "acts_chain_vs_fd_ladder.json",
        jsonable(
            {
                "kind": "acts_chain_vs_fd_ladder",
                **common,
                "targets": [
                    {
                        "event": item.get("event"),
                        "target_station": item.get("target_station"),
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
        "acts_chain_accuracy_stability.json",
        jsonable(
            {
                "kind": "acts_chain_accuracy_stability",
                **common,
                "targets": [
                    {
                        "event": item.get("event"),
                        "target_station": item.get("target_station"),
                        "stability": item.get("stability"),
                        "accuracy": [
                            {
                                "name": rung.get("name"),
                                "step_tolerance": rung.get("step_tolerance"),
                                "nominal_ok": rung.get("nominal_ok"),
                                "chi2": rung.get("chi2"),
                                "path_length_total": rung.get("path_length_total"),
                                "n_steps_total": rung.get("n_steps_total"),
                                "chain_complete": (rung.get("chain") or {}).get(
                                    "chain_complete"
                                ),
                            }
                            for rung in (item.get("accuracy") or [])
                        ],
                    }
                    for item in inventory.get("targets") or []
                ],
            }
        ),
    )
    store.write_json(
        "transport_projection_decomposition.json",
        jsonable(
            {
                "kind": "transport_projection_decomposition",
                **common,
                "targets": [
                    {
                        "event": item.get("event"),
                        "target_station": item.get("target_station"),
                        "acts_end_loc0_matches_official": item.get(
                            "acts_end_loc0_matches_official"
                        ),
                        "continuation_constructions": item.get(
                            "continuation_constructions"
                        ),
                        "branch_fallback": item.get("branch_fallback"),
                        "hop_mechanism": item.get("hop_mechanism"),
                        "comparisons": [
                            {
                                "parameter": comp.get("parameter"),
                                "projection_vs_acts_bound": comp.get(
                                    "projection_vs_acts_bound"
                                ),
                                "projection_vs_fd": comp.get("projection_vs_fd"),
                                "acts_bound_vs_fd": comp.get("acts_bound_vs_fd"),
                            }
                            for comp in (item.get("comparisons") or [])
                        ],
                    }
                    for item in inventory.get("targets") or []
                ],
            }
        ),
    )
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json(
        "acts_fd_derivative_contract_decision.json",
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
            "task": "SB-B14L",
            "workbook": 119,
            "decision": decision["decision"],
            "primary_case": decision["primary_case"],
            "next_step": decision["next_step"],
            "config_sha256": config_sha,
            "inherited_wb118_decision_sha256": decision["inherited_wb118_decision_sha256"],
            "inherited_wb117_decision_sha256": decision["inherited_wb117_decision_sha256"],
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
            "wb118_dumps_not_overwritten": True,
            "focus_identity_retained": True,
            "jacobian_contract_established": decision["jacobian_contract_established"],
            "do_not_select_best_step": True,
            "do_not_select_best_tolerance": True,
            "do_not_replace_official_likelihood": True,
            "five_d_cin_not_the_objective": True,
            "wb118_not_a_physical_conclusion": True,
        }
    )
    print(
        f"SB-B14L verdict={decision['verdict']} case={decision['primary_case']} "
        f"gate={decision['smoke_gate_passed']} "
        f"contract={decision['jacobian_contract_established']} "
        f"b14m={decision['b14m_reopen_authorized']} "
        f"next={decision['next_step']} run_id={store.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
