#!/usr/bin/env python3
"""Task B14R audit.  LTO fit-covariance provenance / information."""

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
from datasets.lto_fit_covariance_provenance import (
    decide,
    inherit_frozen_stage,
    inventory_and_audit,
    load_config,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lto_fit_covariance_provenance_v1.yaml")
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    inventory = inventory_and_audit(config)
    decision = decide(inventory, inherited)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sbb14r_lto_fit_provenance",
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
    }
    store.write_json(
        "lto_fit_information_inventory.json",
        jsonable({"kind": "lto_fit_information_inventory", **common, **inventory["fit_information"]}),
    )
    store.write_json(
        "lto_seed_covariance_sensitivity.json",
        jsonable({"kind": "lto_seed_covariance_sensitivity", **common, **inventory["seed_sensitivity"]}),
    )
    store.write_json(
        "lto_covariance_state_provenance.json",
        jsonable({"kind": "lto_covariance_state_provenance", **common, **inventory["provenance"]}),
    )
    store.write_json(
        "lto_parameter_information_audit.json",
        jsonable({"kind": "lto_parameter_information_audit", **common, **inventory["information"]}),
    )
    store.write_json(
        "lto_pull_information_correlation.json",
        jsonable({"kind": "lto_pull_information_correlation", **common, **inventory["pulls"]}),
    )
    store.write_json(
        "focus_identity_lto_fit_provenance.json",
        jsonable({"kind": "focus_identity_lto_fit_provenance", **common, **inventory["focus"]}),
    )
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json(
        "lto_fit_covariance_provenance_decision.json",
        jsonable(
            {
                **common,
                "present_sources": inventory["present_sources"],
                "missing_sources": inventory["missing_sources"],
                "n_contracted": inventory["n_contracted"],
                "n_official_pairs": inventory["n_official_pairs"],
                "n_raw": inventory["n_raw"],
                "n_ineligible": inventory["n_ineligible"],
                **decision,
            }
        ),
    )
    store.finalize(
        {
            "verdict": decision["verdict"],
            "task": "SB-B14R",
            "workbook": 111,
            "decision": decision["decision"],
            "primary_case": decision["primary_case"],
            "next_step": decision["next_step"],
            "config_sha256": config_sha,
            "inherited_wb110_decision_sha256": decision["inherited_wb110_decision_sha256"],
            "inherited_wb109_decision_sha256": decision["inherited_wb109_decision_sha256"],
            "inherited_wb103_contract_sha256": decision["inherited_wb103_contract_sha256"],
            "official_input_scope": "contract_eligible",
            "raw_ckf_used": False,
            "measurement_model_v2_entered": False,
            "measurement_model_v2_authorized": False,
            "transport_covariance_validated": False,
            "b15_authorized": False,
            "lto_cin_contract_established": False,
            "focus_identity_retained": True,
            "seed_scale_selected_from_closure": False,
            "current_helper_sha256": helper_sha,
            "wb109_dumps_not_overwritten": True,
        }
    )
    print(
        f"SB-B14R verdict={decision['verdict']} case={decision['primary_case']} "
        f"next={decision['next_step']} run_id={store.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
