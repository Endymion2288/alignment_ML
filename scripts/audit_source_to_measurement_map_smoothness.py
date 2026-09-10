#!/usr/bin/env python3
"""Task B14K audit.  Source-to-measurement map smoothness root cause."""

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
from datasets.source_to_measurement_map_smoothness import (
    decide,
    inherit_frozen_stage,
    inventory_and_audit,
    load_config,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/source_to_measurement_map_smoothness_v1.yaml"
    )
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    inventory = inventory_and_audit(config)
    decision = decide(inventory, inherited)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sbb14k_map_smoothness",
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
        "prior_introduced": False,
        "ridge_added": False,
        "statistical_model_unchanged": True,
        "do_not_select_best_step": True,
        "do_not_select_best_tolerance": True,
        "do_not_switch_to_direct": True,
        "five_d_cin_not_the_objective": True,
        "wb117_not_treated_as_physical_nonsmoothness": True,
    }
    store.write_json(
        "source_to_measurement_stagewise_jacobian.json",
        jsonable(
            {
                "kind": "source_to_measurement_stagewise_jacobian",
                **common,
                "targets": [
                    {
                        "event": item.get("event"),
                        "target_station": item.get("target_station"),
                        "stagewise": item.get("stagewise"),
                        "first_failing_stages": item.get("first_failing_stages"),
                        "first_failing_hops": item.get("first_failing_hops"),
                        "first_failing_operations": item.get("first_failing_operations"),
                    }
                    for item in inventory.get("targets") or []
                ],
            }
        ),
    )
    store.write_json(
        "transport_repeatability.json",
        jsonable(
            {
                "kind": "transport_repeatability",
                **common,
                "targets": [
                    {
                        "event": item.get("event"),
                        "target_station": item.get("target_station"),
                        **(item.get("repeatability") or {}),
                    }
                    for item in inventory.get("targets") or []
                ],
            }
        ),
    )
    store.write_json(
        "transport_resolution_vs_fd_signal.json",
        jsonable(
            {
                "kind": "transport_resolution_vs_fd_signal",
                **common,
                "targets": [
                    {
                        "event": item.get("event"),
                        "target_station": item.get("target_station"),
                        **(item.get("resolution") or {}),
                    }
                    for item in inventory.get("targets") or []
                ],
            }
        ),
    )
    store.write_json(
        "propagator_accuracy_convergence.json",
        jsonable(
            {
                "kind": "propagator_accuracy_convergence",
                **common,
                "targets": [
                    {
                        "event": item.get("event"),
                        "target_station": item.get("target_station"),
                        **(item.get("accuracy") or {}),
                    }
                    for item in inventory.get("targets") or []
                ],
            }
        ),
    )
    store.write_json(
        "supporting_plane_projection_smoothness.json",
        jsonable(
            {
                "kind": "supporting_plane_projection_smoothness",
                **common,
                "targets": [
                    {
                        "event": item.get("event"),
                        "target_station": item.get("target_station"),
                        **(item.get("projection") or {}),
                    }
                    for item in inventory.get("targets") or []
                ],
            }
        ),
    )
    store.write_json(
        "fd_scale_dimensionless_audit.json",
        jsonable(
            {
                "kind": "fd_scale_dimensionless_audit",
                **common,
                "targets": [
                    {
                        "event": item.get("event"),
                        "target_station": item.get("target_station"),
                        **(item.get("fd_scale") or {}),
                    }
                    for item in inventory.get("targets") or []
                ],
            }
        ),
    )
    store.write_json(
        "acts_derivative_capability_inventory.json",
        jsonable(
            {
                "kind": "acts_derivative_capability_inventory",
                **common,
                "targets": [
                    {
                        "event": item.get("event"),
                        "target_station": item.get("target_station"),
                        **(item.get("acts_derivative") or {}),
                    }
                    for item in inventory.get("targets") or []
                ],
            }
        ),
    )
    store.write_json(
        "focus_vs_control_transport_smoothness.json",
        jsonable(
            {
                "kind": "focus_vs_control_transport_smoothness",
                **common,
                **(inventory.get("focus_vs_control") or {}),
            }
        ),
    )
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json(
        "source_to_measurement_map_smoothness_decision.json",
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
                "n_focus_targets": len(inventory.get("focus_86") or []),
                "n_control_targets": len(inventory.get("control_44") or []),
                **decision,
            }
        ),
    )
    store.finalize(
        {
            "verdict": decision["verdict"],
            "task": "SB-B14K",
            "workbook": 118,
            "decision": decision["decision"],
            "primary_case": decision["primary_case"],
            "next_step": decision["next_step"],
            "config_sha256": config_sha,
            "inherited_wb117_decision_sha256": decision["inherited_wb117_decision_sha256"],
            "inherited_wb116_decision_sha256": decision["inherited_wb116_decision_sha256"],
            "inherited_wb115_decision_sha256": decision["inherited_wb115_decision_sha256"],
            "inherited_wb114_decision_sha256": decision["inherited_wb114_decision_sha256"],
            "inherited_wb109_decision_sha256": decision["inherited_wb109_decision_sha256"],
            "inherited_wb103_contract_sha256": decision["inherited_wb103_contract_sha256"],
            "official_input_scope": "contract_eligible",
            "raw_ckf_used": False,
            "measurement_model_v2_entered": False,
            "measurement_model_v2_authorized": False,
            "b15_authorized": False,
            "b14m_reopen_authorized": False,
            "restart_invariance_authorized": False,
            "do_not_force_5d_lto_covariance": True,
            "prior_introduced": False,
            "statistical_model_unchanged": True,
            "smoke_gate_passed": decision["smoke_gate_passed"],
            "full_sample_authorized": False,
            "wb109_dumps_not_overwritten": True,
            "wb117_dumps_not_overwritten": True,
            "focus_identity_retained": True,
            "jacobian_contract_established": False,
            "do_not_select_best_step": True,
            "do_not_select_best_tolerance": True,
            "five_d_cin_not_the_objective": True,
            "wb117_not_treated_as_physical_nonsmoothness": True,
        }
    )
    print(
        f"SB-B14K verdict={decision['verdict']} case={decision['primary_case']} "
        f"gate={decision['smoke_gate_passed']} next={decision['next_step']} "
        f"run_id={store.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
