#!/usr/bin/env python3
"""Task B14W audit.  Missing loc1 coupling and independent 86 reference."""

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
from datasets.independent_derivative_reference import (
    decide,
    inherit_frozen_stage,
    inventory_and_audit,
    load_config,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/independent_derivative_reference_v1.yaml"
    )
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    inventory = inventory_and_audit(config)
    decision = decide(inventory, inherited)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sbb14w_independent_reference",
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
        "do_not_change_jacobian_implementation": True,
        "do_not_relax_five_percent_gate": True,
        "do_not_add_fd_rung": True,
        "do_not_shrink_fd_step": True,
        "five_percent_gate_unchanged": True,
        "small_physical_effect_does_not_pass_contract": True,
        "prior_introduced": False,
        "ridge_added": False,
        "statistical_model_unchanged": True,
    }
    store.write_json(
        "control1_loc1_missing_coupling.json",
        jsonable({"kind": "control1_loc1_missing_coupling", **common, "targets": inventory["control1"]}),
    )
    store.write_json(
        "material_variational_reset_audit.json",
        jsonable({"kind": "material_variational_reset_audit", **common, "targets": inventory["material_audits"]}),
    )
    store.write_json(
        "segment_derivative_closure.json",
        jsonable({"kind": "segment_derivative_closure", **common, "targets": inventory["segment_audits"]}),
    )
    store.write_json(
        "focus86_independent_derivative_reference.json",
        jsonable(
            {
                "kind": "focus86_independent_derivative_reference",
                **common,
                "targets": inventory["focus_references"],
            }
        ),
    )
    store.write_json(
        "focus86_per_hit_derivative_breakdown.json",
        jsonable(
            {
                "kind": "focus86_per_hit_derivative_breakdown",
                **common,
                "targets": inventory["focus_per_hits"],
            }
        ),
    )
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json(
        "derivative_reference_decision.json",
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
            "task": "SB-B14W",
            "workbook": 122,
            "decision": decision["decision"],
            "primary_case": decision["primary_case"],
            "next_step": decision["next_step"],
            "config_sha256": config_sha,
            "inherited_wb121_decision_sha256": decision["inherited_wb121_decision_sha256"],
            "b14m_reopen_authorized": False,
            "full_sample_authorized": False,
            "jacobian_contract_established": False,
            "do_not_change_jacobian_implementation": True,
            "do_not_relax_five_percent_gate": True,
            "do_not_add_fd_rung": True,
            "smoke_gate_passed": decision["smoke_gate_passed"],
            "wb120_dumps_not_overwritten": True,
            "wb121_dumps_not_overwritten": True,
        }
    )
    print(
        f"SB-B14W verdict={decision['verdict']} case={decision['primary_case']} "
        f"gate={decision['smoke_gate_passed']} "
        f"coupling={decision['missing_deterministic_transport_coupling_identified']} "
        f"focus_ref={decision['focus_independent_reference_established']} "
        f"b14m={decision['b14m_reopen_authorized']} "
        f"next={decision['next_step']} run_id={store.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
