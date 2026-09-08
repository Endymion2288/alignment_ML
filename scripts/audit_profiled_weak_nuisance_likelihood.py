#!/usr/bin/env python3
"""Task B14M audit.  Profiled weak-nuisance measurement likelihood."""

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
from datasets.profiled_weak_nuisance_likelihood import (
    decide,
    inherit_frozen_stage,
    inventory_and_audit,
    load_config,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/profiled_weak_nuisance_likelihood_v1.yaml"
    )
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    inventory = inventory_and_audit(config)
    decision = decide(inventory, inherited)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sbb14m_profiled_likelihood",
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
        "prior_introduced": False,
        "ridge_added": False,
        "marginalization_executed": False,
    }
    store.write_json(
        "measurement_likelihood_contract.json",
        jsonable({"kind": "measurement_likelihood_contract", **common, **inventory["likelihood_contract"]}),
    )
    store.write_json(
        "profiled_state_partition.json",
        jsonable({"kind": "profiled_state_partition", **common, **inventory["partition"]}),
    )
    store.write_json(
        "profile_likelihood_numerical_validation.json",
        jsonable(
            {
                "kind": "profile_likelihood_numerical_validation",
                **common,
                **inventory["numerical_validation"],
            }
        ),
    )
    store.write_json(
        "profile_nuisance_identifiability.json",
        jsonable({"kind": "profile_nuisance_identifiability", **common, **inventory["identifiability"]}),
    )
    store.write_json(
        "profile_seed_invariance.json",
        jsonable({"kind": "profile_seed_invariance", **common, **inventory["seed_invariance"]}),
    )
    store.write_json(
        "profiled_target_prediction.json",
        jsonable({"kind": "profiled_target_prediction", **common, **inventory["predictions"]}),
    )
    store.write_json(
        "profiled_prediction_uncertainty_contract.json",
        jsonable(
            {
                "kind": "profiled_prediction_uncertainty_contract",
                **common,
                **inventory["uncertainty_contract"],
            }
        ),
    )
    store.write_json(
        "profiled_prediction_calibration.json",
        jsonable({"kind": "profiled_prediction_calibration", **common, **inventory["calibration"]}),
    )
    store.write_json(
        "focus_identity_profile_likelihood.json",
        jsonable({"kind": "focus_identity_profile_likelihood", **common, **inventory["focus"]}),
    )
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json(
        "profiled_measurement_likelihood_decision.json",
        jsonable(
            {
                **common,
                "present_sources": inventory["present_sources"],
                "missing_sources": inventory["missing_sources"],
                "n_contracted": inventory["n_contracted"],
                "n_official_pairs": inventory["n_official_pairs"],
                "n_raw": inventory["n_raw"],
                "n_ineligible": inventory["n_ineligible"],
                "exclusion": inventory["exclusion"],
                "profile_dumps": inventory["profile_dumps"],
                **decision,
            }
        ),
    )
    store.finalize(
        {
            "verdict": decision["verdict"],
            "task": "SB-B14M",
            "workbook": 114,
            "decision": decision["decision"],
            "primary_case": decision["primary_case"],
            "next_step": decision["next_step"],
            "config_sha256": config_sha,
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
            "transport_covariance_validated": False,
            "b15_authorized": False,
            "lto_cin_contract_established": False,
            "prior_contract_established": False,
            "prior_introduced": False,
            "do_not_force_5d_lto_covariance": True,
            "profiling_executed": True,
            "marginalization_executed": False,
            "focus_identity_retained": True,
            "wb109_dumps_not_overwritten": True,
            "acts_profile_materialized": decision["acts_profile_materialized"],
            "synthetic_profile_passed": decision["synthetic_profile_passed"],
        }
    )
    print(
        f"SB-B14M verdict={decision['verdict']} case={decision['primary_case']} "
        f"next={decision['next_step']} run_id={store.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
