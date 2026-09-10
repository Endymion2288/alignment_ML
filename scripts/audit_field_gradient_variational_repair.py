#!/usr/bin/env python3
"""Task B14X audit.  Field-gradient tangent repair and recontract."""

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
from datasets.field_gradient_variational_repair import (
    decide,
    inherit_frozen_stage,
    inventory_and_audit,
    load_config,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/field_gradient_variational_repair_v1.yaml"
    )
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    inventory = inventory_and_audit(config)
    decision = decide(inventory, inherited)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sbb14x_field_gradient_repair",
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
        "wb120_dumps_not_overwritten": True,
        "wb121_dumps_not_overwritten": True,
        "wb122_dumps_not_overwritten": True,
        "do_not_change_official_mean_path": True,
        "do_not_relax_five_percent_gate": True,
        "do_not_add_fd_rung": True,
        "do_not_tune_field_step_from_track_jacobian": True,
        "five_percent_gate_unchanged": True,
        "prior_introduced": False,
        "ridge_added": False,
        "statistical_model_unchanged": True,
    }
    store.write_json(
        "acts_field_gradient_variational_source_audit.json",
        jsonable({"kind": "acts_field_gradient_variational_source_audit", **common,
                  **inventory["source_audit"]}),
    )
    store.write_json(
        "field_gradient_variational_equation_contract.json",
        jsonable({"kind": "field_gradient_variational_equation_contract", **common,
                  **inventory["equation"]}),
    )
    store.write_json(
        "magnetic_field_gradient_contract.json",
        jsonable(
            {
                "kind": "magnetic_field_gradient_contract",
                **common,
                "official_api": "FASERMagneticFieldWrapper::getFieldGradient",
                "field_gradient_fd_step_mm": 1.0,
                "do_not_tune_field_step_from_track_jacobian": True,
                "controls": inventory["controls"],
                "focus": inventory["focus"],
                "field_samples": inventory.get("field_contract"),
            }
        ),
    )
    store.write_json(
        "field_gradient_variational_repair_invariance.json",
        jsonable({"kind": "field_gradient_variational_repair_invariance", **common,
                  "targets": inventory["invariance"]}),
    )
    store.write_json(
        "control1_field_gradient_coupling_repair.json",
        jsonable(
            {
                "kind": "control1_field_gradient_coupling_repair",
                **common,
                "targets": [item for item in inventory["controls"] if item.get("event") == "100043/1"],
            }
        ),
    )
    store.write_json(
        "long_magnet_segment_fd_contract.json",
        jsonable({"kind": "long_magnet_segment_fd_contract", **common,
                  "controls": inventory.get("control_segments"),
                  "focus": inventory["focus_segments"]}),
    )
    store.write_json(
        "focus86_repaired_derivative_reference.json",
        jsonable({"kind": "focus86_repaired_derivative_reference", **common,
                  "targets": inventory["focus_segments"]}),
    )
    store.write_json(
        "material_tangent_continuity_check.json",
        jsonable({"kind": "material_tangent_continuity_check", **common,
                  "targets": inventory["material"]}),
    )
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json(
        "field_gradient_variational_decision.json",
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
            "task": "SB-B14X",
            "workbook": 123,
            "decision": decision["decision"],
            "primary_case": decision["primary_case"],
            "next_step": decision["next_step"],
            "config_sha256": config_sha,
            "inherited_wb122_decision_sha256": decision["inherited_wb122_decision_sha256"],
            "b14m_reopen_authorized": decision["b14m_reopen_authorized"],
            "full_sample_authorized": False,
            "jacobian_contract_established": decision["jacobian_contract_established"],
            "do_not_change_official_mean_path": True,
            "do_not_relax_five_percent_gate": True,
            "smoke_gate_passed": decision["smoke_gate_passed"],
            "wb122_dumps_not_overwritten": True,
        }
    )
    print(
        f"SB-B14X verdict={decision['verdict']} case={decision['primary_case']} "
        f"gate={decision['smoke_gate_passed']} "
        f"c1={decision['control_1_pass']} focus_ref={decision['focus_independent_reference_established']} "
        f"b14m={decision['b14m_reopen_authorized']} "
        f"next={decision['next_step']} run_id={store.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
