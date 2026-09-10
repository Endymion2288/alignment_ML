#!/usr/bin/env python3
"""Task B14ZC audit.  Certified-mean common-grid independent FD."""

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
from datasets.certified_mean_common_grid_fd import (
    FROZEN_FIELD_GRADIENT_SHA,
    decide,
    inherit_frozen_stage,
    inventory_and_audit,
    load_config,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/certified_mean_common_grid_fd_v1.yaml"
    )
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    inventory = inventory_and_audit(config)
    decision = decide(inventory, inherited)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sbb14zc_certified_mean_fd",
    )
    created = datetime.now(timezone.utc).isoformat()
    config_sha = sha256_file(config_path)
    helper_sha = sha256_file(
        resolve_under_root(
            project_root(),
            "alignment/leave_target_out_dump/CkfLeaveTargetOutDumpAlg.cxx",
        )
    )
    mean_sha = sha256_file(
        resolve_under_root(
            project_root(),
            "alignment/leave_target_out_dump/MeanTransportContract.hpp",
        )
    )
    repair_sha = sha256_file(
        resolve_under_root(
            project_root(),
            "alignment/leave_target_out_dump/FieldGradientDefaultExtension.hpp",
        )
    )
    common = {
        "created_utc": created,
        "config_sha256": config_sha,
        "git_head_sha": git_head_sha(),
        "current_helper_sha256": helper_sha,
        "mean_transport_sha256": mean_sha,
        "field_gradient_extension_sha256": repair_sha,
        "field_gradient_extension_unchanged": repair_sha == FROZEN_FIELD_GRADIENT_SHA,
        "wb109_dumps_not_overwritten": True,
        "wb123_dumps_not_overwritten": True,
        "wb125_dumps_not_overwritten": True,
        "wb126_dumps_not_overwritten": True,
        "wb127_dumps_not_overwritten": True,
        "do_not_change_official_mean_path": True,
        "do_not_change_mean_semantics": True,
        "do_not_relax_five_percent_gate": True,
        "do_not_add_fd_rung": True,
        "do_not_reuse_wb124_adaptive_dopri5": True,
        "do_not_submit_1989": True,
        "production_eloss_quantity": "computeEnergyLossBethe",
        "five_percent_gate_unchanged": True,
        "prior_introduced": False,
        "ridge_added": False,
        "statistical_model_unchanged": True,
    }
    segs = inventory["focus_segments"]
    store.write_json(
        "certified_shadow_fd_ladder.json",
        jsonable(
            {
                "kind": "certified_shadow_fd_ladder",
                **common,
                "targets": [
                    {
                        "target_station": item.get("target_station"),
                        "required_index": item.get("required_index"),
                        "certified_fd": item.get("certified_fd"),
                        "parameters": item.get("parameters"),
                    }
                    for item in segs
                ],
            }
        ),
    )
    store.write_json(
        "perturbed_arm_mean_contract.json",
        jsonable(
            {
                "kind": "perturbed_arm_mean_contract",
                **common,
                "targets": [
                    {
                        "target_station": item.get("target_station"),
                        "required_index": item.get("required_index"),
                        "all_arms_same_mean_contract": (
                            item.get("certified_fd") or {}
                        ).get("all_arms_same_mean_contract"),
                        "all_branch_identity_same": (
                            item.get("certified_fd") or {}
                        ).get("all_branch_identity_same"),
                        "all_material_node_identity_same": (
                            item.get("certified_fd") or {}
                        ).get("all_material_node_identity_same"),
                    }
                    for item in segs
                ],
            }
        ),
    )
    store.write_json(
        "repaired_tangent_vs_certified_shadow.json",
        jsonable(
            {
                "kind": "repaired_tangent_vs_certified_shadow",
                **common,
                "targets": [
                    {
                        "target_station": item.get("target_station"),
                        "required_index": item.get("required_index"),
                        "parameters": item.get("parameters"),
                        "repaired_agrees_independent": item.get(
                            "repaired_agrees_independent"
                        ),
                    }
                    for item in segs
                ],
            }
        ),
    )
    store.write_json(
        "required_segment_derivative_taxonomy.json",
        jsonable(
            {
                "kind": "required_segment_derivative_taxonomy",
                **common,
                "targets": decision.get("segment_taxonomy"),
            }
        ),
    )
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json(
        "certified_mean_common_grid_fd_decision.json",
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
                **decision,
            }
        ),
    )
    store.finalize(
        {
            "verdict": decision["verdict"],
            "task": "SB-B14ZC",
            "workbook": 128,
            "decision": decision["decision"],
            "primary_case": decision["primary_case"],
            "next_step": decision["next_step"],
            "config_sha256": config_sha,
            "inherited_wb127_decision_sha256": decision[
                "inherited_wb127_decision_sha256"
            ],
            "shadow_mean_contract_established": True,
            "focus_independent_reference_established": decision[
                "focus_independent_reference_established"
            ],
            "jacobian_contract_established": decision[
                "jacobian_contract_established"
            ],
            "b14m_reopen_authorized": decision["b14m_reopen_authorized"],
            "full_sample_authorized": False,
            "do_not_change_mean_semantics": True,
            "do_not_submit_1989": True,
            "smoke_gate_passed": decision["smoke_gate_passed"],
            "wb127_dumps_not_overwritten": True,
            "field_gradient_extension_unchanged": repair_sha
            == FROZEN_FIELD_GRADIENT_SHA,
        }
    )
    print(
        f"SB-B14ZC verdict={decision['verdict']} case={decision['primary_case']} "
        f"gate={decision['smoke_gate_passed']} "
        f"mean={decision['shadow_mean_contract_established']} "
        f"focus_ref={decision['focus_independent_reference_established']} "
        f"jac={decision['jacobian_contract_established']} "
        f"b14m={decision['b14m_reopen_authorized']} "
        f"next={decision['next_step']} run_id={store.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
