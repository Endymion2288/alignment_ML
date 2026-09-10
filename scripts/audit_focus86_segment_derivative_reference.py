#!/usr/bin/env python3
"""Task B14Y audit.  86 required-hop independent segment reference."""

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
from datasets.focus86_segment_derivative_reference import (
    decide,
    inherit_frozen_stage,
    inventory_and_audit,
    load_config,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/focus86_segment_derivative_reference_v1.yaml"
    )
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    inventory = inventory_and_audit(config)
    decision = decide(inventory, inherited)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sbb14y_focus86_segment_reference",
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
        "wb123_dumps_not_overwritten": True,
        "do_not_change_official_mean_path": True,
        "do_not_relax_five_percent_gate": True,
        "do_not_add_fd_rung": True,
        "do_not_change_field_gradient_variational_implementation": True,
        "do_not_tune_independent_tolerance_from_jacobian": True,
        "five_percent_gate_unchanged": True,
        "prior_introduced": False,
        "ridge_added": False,
        "statistical_model_unchanged": True,
    }
    store.write_json(
        "required_hop_start_states.json",
        jsonable({"kind": "required_hop_start_states", **common,
                  "targets": inventory["focus_segments"]}),
    )
    store.write_json(
        "production_mean_segment_fd.json",
        jsonable({"kind": "production_mean_segment_fd", **common,
                  "targets": inventory["focus_segments"]}),
    )
    store.write_json(
        "independent_segment_reference.json",
        jsonable({"kind": "independent_segment_reference", **common,
                  "targets": inventory["focus_segments"]}),
    )
    store.write_json(
        "triple_comparison_repaired_productionfd_independent.json",
        jsonable({"kind": "triple_comparison", **common,
                  "targets": inventory["focus_segments"]}),
    )
    store.write_json(
        "control_0_1_37_gate_replay.json",
        jsonable({"kind": "control_0_1_37_gate_replay", **common,
                  "targets": inventory["controls"],
                  "invariance": inventory["invariance"]}),
    )
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json(
        "focus86_segment_derivative_decision.json",
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
            "task": "SB-B14Y",
            "workbook": 124,
            "decision": decision["decision"],
            "primary_case": decision["primary_case"],
            "next_step": decision["next_step"],
            "config_sha256": config_sha,
            "inherited_wb123_decision_sha256": decision["inherited_wb123_decision_sha256"],
            "b14m_reopen_authorized": decision["b14m_reopen_authorized"],
            "full_sample_authorized": False,
            "jacobian_contract_established": decision["jacobian_contract_established"],
            "focus_independent_reference_established": decision[
                "focus_independent_reference_established"
            ],
            "do_not_change_official_mean_path": True,
            "do_not_relax_five_percent_gate": True,
            "smoke_gate_passed": decision["smoke_gate_passed"],
            "wb123_dumps_not_overwritten": True,
        }
    )
    print(
        f"SB-B14Y verdict={decision['verdict']} case={decision['primary_case']} "
        f"gate={decision['smoke_gate_passed']} "
        f"focus_ref={decision['focus_independent_reference_established']} "
        f"b14m={decision['b14m_reopen_authorized']} "
        f"next={decision['next_step']} run_id={store.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
