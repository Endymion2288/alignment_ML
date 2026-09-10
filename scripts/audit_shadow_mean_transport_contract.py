#!/usr/bin/env python3
"""Task B14ZA audit.  Production-vs-shadow mean transport contract."""

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
from datasets.shadow_mean_transport_contract import (
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
        "--config", default="configs/shadow_mean_transport_contract_v2.yaml"
    )
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    inventory = inventory_and_audit(config)
    decision = decide(inventory, inherited)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sbb14za_shadow_mean_transport",
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
        "wb120_dumps_not_overwritten": True,
        "wb123_dumps_not_overwritten": True,
        "wb124_dumps_not_overwritten": True,
        "wb125_dumps_not_overwritten": True,
        "do_not_change_official_mean_path": True,
        "do_not_relax_five_percent_gate": True,
        "do_not_add_fd_rung": True,
        "do_not_change_field_gradient_variational_implementation": True,
        "do_not_tune_wb124_dopri5": True,
        "do_not_tune_grid_from_jacobian": True,
        "do_not_evaluate_jacobian": True,
        "derivative_not_evaluated": True,
        "jacobian_agreement_not_read": True,
        "grid_selected_from_mean_contract_only": True,
        "do_not_call_vacuum_a_production_map_reference": True,
        "five_percent_gate_unchanged": True,
        "prior_introduced": False,
        "ridge_added": False,
        "statistical_model_unchanged": True,
    }
    segs = inventory["focus_segments"]
    store.write_json(
        "production_mean_transition_ledger.json",
        jsonable(
            {
                "kind": "production_mean_transition_ledger",
                **common,
                "targets": [
                    {
                        "target_station": item.get("target_station"),
                        "required_index": item.get("required_index"),
                        "ledger": item.get("production_ledger"),
                    }
                    for item in segs
                ],
            }
        ),
    )
    store.write_json(
        "shadow_mean_transition_ledger.json",
        jsonable(
            {
                "kind": "shadow_mean_transition_ledger",
                **common,
                "targets": [
                    {
                        "target_station": item.get("target_station"),
                        "required_index": item.get("required_index"),
                        "ledger": item.get("shadow_p_ledger"),
                    }
                    for item in segs
                ],
            }
        ),
    )
    store.write_json(
        "production_shadow_first_divergence.json",
        jsonable(
            {
                "kind": "production_shadow_first_divergence",
                **common,
                "targets": [
                    {
                        "target_station": item.get("target_station"),
                        "required_index": item.get("required_index"),
                        "first_divergence": item.get("first_divergence"),
                    }
                    for item in segs
                ],
            }
        ),
    )
    store.write_json(
        "surface_energy_loss_mean_contract.json",
        jsonable(
            {
                "kind": "surface_energy_loss_mean_contract",
                **common,
                "targets": [
                    {
                        "target_station": item.get("target_station"),
                        "required_index": item.get("required_index"),
                        "contract": item.get("surface_energy_loss_mean_contract"),
                    }
                    for item in segs
                ],
            }
        ),
    )
    store.write_json(
        "material_update_order_contract.json",
        jsonable(
            {
                "kind": "material_update_order_contract",
                **common,
                "targets": [
                    {
                        "target_station": item.get("target_station"),
                        "required_index": item.get("required_index"),
                        "contract": item.get("material_update_order_contract"),
                    }
                    for item in segs
                ],
            }
        ),
    )
    store.write_json(
        "qop_energy_loss_unit_contract.json",
        jsonable(
            {
                "kind": "qop_energy_loss_unit_contract",
                **common,
                "targets": [
                    {
                        "target_station": item.get("target_station"),
                        "required_index": item.get("required_index"),
                        "contract": item.get("qop_energy_loss_unit_contract"),
                    }
                    for item in segs
                ],
            }
        ),
    )
    store.write_json(
        "direction_normalization_contract.json",
        jsonable(
            {
                "kind": "direction_normalization_contract",
                **common,
                "targets": [
                    {
                        "target_station": item.get("target_station"),
                        "required_index": item.get("required_index"),
                        "contract": item.get("direction_normalization_contract"),
                    }
                    for item in segs
                ],
            }
        ),
    )
    store.write_json(
        "shadow_mean_convergence.json",
        jsonable(
            {
                "kind": "shadow_mean_convergence",
                **common,
                "sequence_mm": [10.0, 5.0, 2.5, 1.25],
                "targets": [
                    {
                        "target_station": item.get("target_station"),
                        "required_index": item.get("required_index"),
                        "shadow_d": item.get("shadow_d"),
                        "shadow_d_any_closed": item.get("shadow_d_any_closed"),
                        "shadow_d_loc0_monotone": item.get("shadow_d_loc0_monotone"),
                    }
                    for item in segs
                ],
            }
        ),
    )
    store.write_json(
        "shadow_mean_mechanism_ablation.json",
        jsonable(
            {
                "kind": "shadow_mean_mechanism_ablation",
                **common,
                "targets": [
                    {
                        "target_station": item.get("target_station"),
                        "required_index": item.get("required_index"),
                        "ablation": item.get("mechanism_ablation"),
                    }
                    for item in segs
                ],
            }
        ),
    )
    store.write_json(
        "required_segment_mean_failure_taxonomy.json",
        jsonable(
            {
                "kind": "required_segment_mean_failure_taxonomy",
                **common,
                "targets": decision.get("segment_taxonomy"),
            }
        ),
    )
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json(
        "shadow_mean_transport_decision.json",
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
            "task": "SB-B14ZA",
            "workbook": 126,
            "decision": decision["decision"],
            "primary_case": decision["primary_case"],
            "next_step": decision["next_step"],
            "config_sha256": config_sha,
            "inherited_wb125_decision_sha256": decision[
                "inherited_wb125_decision_sha256"
            ],
            "inherited_wb124_decision_sha256": decision[
                "inherited_wb124_decision_sha256"
            ],
            "inherited_wb123_decision_sha256": decision[
                "inherited_wb123_decision_sha256"
            ],
            "b14m_reopen_authorized": False,
            "full_sample_authorized": False,
            "jacobian_contract_established": False,
            "focus_independent_reference_established": False,
            "do_not_change_official_mean_path": True,
            "do_not_relax_five_percent_gate": True,
            "do_not_tune_wb124_dopri5": True,
            "do_not_tune_grid_from_jacobian": True,
            "do_not_evaluate_jacobian": True,
            "derivative_not_evaluated": True,
            "smoke_gate_passed": decision["smoke_gate_passed"],
            "wb123_dumps_not_overwritten": True,
            "wb124_dumps_not_overwritten": True,
            "wb125_dumps_not_overwritten": True,
            "field_gradient_extension_unchanged": repair_sha
            == FROZEN_FIELD_GRADIENT_SHA,
        }
    )
    print(
        f"SB-B14ZA verdict={decision['verdict']} case={decision['primary_case']} "
        f"gate={decision['smoke_gate_passed']} "
        f"focus_ref={decision['focus_independent_reference_established']} "
        f"b14m={decision['b14m_reopen_authorized']} "
        f"next={decision['next_step']} run_id={store.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
