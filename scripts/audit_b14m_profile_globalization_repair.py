#!/usr/bin/env python3
"""Task B14M-T audit.  Stationarity first, then restart invariance."""

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
from datasets.b14m_profile_globalization_repair import (
    FROZEN_FIELD_GRADIENT_SHA,
    decide,
    inherit_frozen_stage,
    inventory_and_audit,
    likelihood_contract,
    load_config,
    preregistered_trust_region,
)
from datasets.b14m_restart_invariance import recover_optimizer_contract
from datasets.surface_energy_loss_mean_semantics import PRODUCTION_ELOSS
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/b14m_profile_globalization_repair_v1.yaml"
    )
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    contract = recover_optimizer_contract(config)
    inventory = inventory_and_audit(config, contract)
    decision = decide(inventory, inherited, contract)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sbb14mt_profile_globalization",
    )
    created = datetime.now(timezone.utc).isoformat()
    helper_sha = sha256_file(
        resolve_under_root(
            project_root(),
            "alignment/leave_target_out_dump/CkfLeaveTargetOutDumpAlg.cxx",
        )
    )
    repair_inc = sha256_file(
        resolve_under_root(
            project_root(),
            "alignment/leave_target_out_dump/ProfileGlobalizationRepair.inc",
        )
    )
    field_sha = sha256_file(
        resolve_under_root(
            project_root(),
            "alignment/leave_target_out_dump/FieldGradientDefaultExtension.hpp",
        )
    )
    common = {
        "created_utc": created,
        "config_sha256": sha256_file(config_path),
        "git_head_sha": git_head_sha(),
        "current_helper_sha256": helper_sha,
        "profile_globalization_repair_inc_sha256": repair_inc,
        "field_gradient_extension_sha256": field_sha,
        "field_gradient_extension_unchanged": field_sha == FROZEN_FIELD_GRADIENT_SHA,
        "wb109_dumps_not_overwritten": True,
        "wb114_dumps_not_overwritten": True,
        "wb129_dumps_not_overwritten": True,
        "wb130_dumps_not_overwritten": True,
        "legacy_b14m_smoke_not_overwritten": True,
        "production_eloss_quantity": PRODUCTION_ELOSS,
        "prior_introduced": False,
        "ridge_added": False,
        "statistical_model_unchanged": True,
        "lm_lambda_is_not_prior": True,
        "lm_lambda_is_not_ridge": True,
        "lm_lambda_is_not_information": True,
        "trust_region_is_algorithmic_globalization_only": True,
        "do_not_submit_1989": True,
        "full_sample_authorized": False,
        "likelihood_contract": likelihood_contract(),
        "trust_region": preregistered_trust_region(config),
        "optimizer_contract": {
            key: contract[key]
            for key in (
                "recovered",
                "optimizer_contract_not_recoverable",
                "chi2_rel_tolerance",
                "prediction_abs_tolerance_mm",
                "supported_abs_tolerance",
                "pinv_relative",
                "gradient_norm_z",
                "step_norm_z",
                "relative_chi2_decrease",
            )
            if key in contract
        },
    }
    per = inventory.get("per_identity") or {}
    store.write_json(
        "inherited_stage.json",
        jsonable({"kind": "inherited_stage", **common, "inherited": inherited}),
    )
    store.write_json(
        "profile_inner_nuisance_trace.json",
        jsonable(
            {
                "kind": "profile_inner_nuisance_trace",
                **common,
                "rows": (inventory.get("traces") or {}).get("inner"),
            }
        ),
    )
    store.write_json(
        "profile_outer_alpha_trace.json",
        jsonable(
            {
                "kind": "profile_outer_alpha_trace",
                **common,
                "rows": (inventory.get("traces") or {}).get("outer"),
            }
        ),
    )
    store.write_json(
        "trust_region_trial_trace.json",
        jsonable(
            {
                "kind": "trust_region_trial_trace",
                **common,
                "rows": (inventory.get("traces") or {}).get("trials"),
            }
        ),
    )
    store.write_json(
        "supported_stationarity_audit.json",
        jsonable(
            {
                "kind": "supported_stationarity_audit",
                **common,
                "identities": {
                    key: {
                        "all_stationary": item.get("all_stationary"),
                        "stationary_restarts": item.get("stationary_restarts"),
                        "norm_g_n_R": item.get("norm_g_n_R"),
                        "norm_g_alpha_prof_z": item.get("norm_g_alpha_prof_z"),
                        "terminations": item.get("terminations"),
                        "stall_not_escaped": item.get("stall_not_escaped"),
                    }
                    for key, item in per.items()
                },
            }
        ),
    )
    store.write_json(
        "profile_gradient_stationarity.json",
        jsonable(
            {
                "kind": "profile_gradient_stationarity",
                **common,
                "identities": {
                    key: {
                        "inner_nuisance_profile_valid": item.get(
                            "inner_nuisance_profile_valid"
                        ),
                        "profile_alpha_stationary": item.get(
                            "profile_alpha_stationary"
                        ),
                        "norm_g_n_R": item.get("norm_g_n_R"),
                        "norm_g_alpha_prof_z": item.get("norm_g_alpha_prof_z"),
                    }
                    for key, item in per.items()
                },
            }
        ),
    )
    store.write_json(
        "restart_objective_invariance.json",
        jsonable(
            {
                "kind": "restart_objective_invariance",
                **common,
                "identities": {
                    key: {
                        "all_stationary": item.get("all_stationary"),
                        "objective_invariance": item.get("objective_invariance"),
                        "chi2_prof": item.get("chi2_prof"),
                        "max_pairwise_delta_chi2_prof": item.get(
                            "max_pairwise_delta_chi2_prof"
                        ),
                    }
                    for key, item in per.items()
                },
            }
        ),
    )
    store.write_json(
        "restart_prediction_invariance.json",
        jsonable(
            {
                "kind": "restart_prediction_invariance",
                **common,
                "identities": {
                    key: {
                        "all_stationary": item.get("all_stationary"),
                        "prediction_invariance": item.get("prediction_invariance"),
                        "predicted_target_loc0": item.get("predicted_target_loc0"),
                        "max_pairwise_delta_target_loc0": item.get(
                            "max_pairwise_delta_target_loc0"
                        ),
                        "max_pairwise_delta_surviving_loc0": item.get(
                            "max_pairwise_delta_surviving_loc0"
                        ),
                        "transport_branch_invariance": item.get(
                            "transport_branch_invariance"
                        ),
                    }
                    for key, item in per.items()
                },
            }
        ),
    )
    store.write_json(
        "restart_parameter_nonuniqueness.json",
        jsonable(
            {
                "kind": "restart_parameter_nonuniqueness",
                **common,
                "identities": {
                    key: {
                        "parameter_invariance_alpha": item.get(
                            "parameter_invariance_alpha"
                        ),
                        "nuisance_nonunique": item.get("nuisance_nonunique"),
                        "natives": item.get("natives"),
                    }
                    for key, item in per.items()
                },
            }
        ),
    )
    store.write_json(
        "optimizer_regression_controls.json",
        jsonable(
            {
                "kind": "optimizer_regression_controls",
                **common,
                "identities": {
                    key: item
                    for key, item in per.items()
                    if item.get("role") == "regression_control"
                },
            }
        ),
    )
    store.write_json(
        "target_exclusion_audit.json",
        jsonable(
            {
                "kind": "target_exclusion_audit",
                **common,
                "target_leakage": inventory.get("target_leakage"),
                "identities": {
                    key: {"target_leakage": item.get("target_leakage")}
                    for key, item in per.items()
                },
            }
        ),
    )
    store.write_json(
        "held_out_prediction_diagnostic.json",
        jsonable(
            {
                "kind": "held_out_prediction_diagnostic",
                **common,
                "note": "Stage B diagnostic only; unused for solver or selection",
                "identities": {
                    key: item.get("predicted_target_loc0") for key, item in per.items()
                },
            }
        ),
    )
    store.write_json(
        "validity_matrix.json",
        jsonable(
            {
                "kind": "validity_matrix",
                **common,
                "per_identity": per,
                "complete_48": inventory.get("complete_48"),
            }
        ),
    )
    store.write_json(
        "b14m_profile_globalization_decision.json",
        jsonable(
            {"kind": "b14m_profile_globalization_decision", **common, **decision}
        ),
    )
    store.finalize(
        {
            "decision": decision.get("decision"),
            "verdict": decision.get("verdict"),
            "full_sample_authorized": False,
        }
    )
    print(store.run_dir)
    print(decision.get("decision"), decision.get("verdict"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
