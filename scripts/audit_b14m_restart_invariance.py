#!/usr/bin/env python3
"""Task B14M-R audit.  Profile smoke reopen and restart invariance."""

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
from datasets.b14m_restart_invariance import (
    FROZEN_FIELD_GRADIENT_SHA,
    decide,
    inherit_frozen_stage,
    inventory_and_audit,
    load_config,
    recover_optimizer_contract,
)
from datasets.surface_energy_loss_mean_semantics import PRODUCTION_ELOSS
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/b14m_restart_invariance_v2.yaml"
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
        "sbb14mr_restart_invariance",
    )
    created = datetime.now(timezone.utc).isoformat()
    config_sha = sha256_file(config_path)
    helper_sha = sha256_file(
        resolve_under_root(
            project_root(),
            "alignment/leave_target_out_dump/CkfLeaveTargetOutDumpAlg.cxx",
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
        "field_gradient_extension_sha256": repair_sha,
        "field_gradient_extension_unchanged": repair_sha == FROZEN_FIELD_GRADIENT_SHA,
        "wb109_dumps_not_overwritten": True,
        "wb114_dumps_not_overwritten": True,
        "wb128_dumps_not_overwritten": True,
        "legacy_b14m_smoke_not_overwritten": True,
        "production_eloss_quantity": PRODUCTION_ELOSS,
        "prior_introduced": False,
        "ridge_added": False,
        "statistical_model_unchanged": True,
        "do_not_submit_1989": True,
        "full_sample_authorized": False,
        "optimizer_contract": {
            key: contract[key]
            for key in (
                "recovered",
                "optimizer_contract_not_recoverable",
                "chi2_rel_tolerance",
                "prediction_abs_tolerance_mm",
                "supported_abs_tolerance",
                "max_iterations",
                "pinv_relative",
                "schur_abs_tol",
                "schur_rel_tol",
            )
            if key in contract
        },
    }
    store.write_json(
        "inherited_stage.json",
        jsonable({"kind": "inherited_stage", **common, "inherited": inherited}),
    )
    store.write_json(
        "b14m_nominal_profile_trace.json",
        jsonable(
            {
                "kind": "b14m_nominal_profile_trace",
                **common,
                "rows": [
                    {
                        "event": item.get("event"),
                        "target_station": item.get("target_station"),
                        "chi2_prof": (item.get("chi2_prof") or {}).get("nominal"),
                        "predicted_target_loc0": (
                            item.get("predicted_target_loc0") or {}
                        ).get("nominal"),
                        "termination": (item.get("terminations") or {}).get("nominal"),
                        "hessian_rank": (item.get("hessian_ranks") or {}).get("nominal"),
                    }
                    for item in inventory.get("identities") or []
                ],
            }
        ),
    )
    store.write_json(
        "b14m_restart_profile_trace.json",
        jsonable(
            {
                "kind": "b14m_restart_profile_trace",
                **common,
                "identities": inventory.get("identities") or [],
            }
        ),
    )
    store.write_json(
        "restart_objective_invariance.json",
        jsonable(
            {
                "kind": "restart_objective_invariance",
                **common,
                "chi2_rel_tolerance": contract.get("chi2_rel_tolerance"),
                "identities": [
                    {
                        "event": item.get("event"),
                        "target_station": item.get("target_station"),
                        "objective_invariance": item.get("objective_invariance"),
                        "max_pairwise_delta_chi2_prof": item.get(
                            "max_pairwise_delta_chi2_prof"
                        ),
                        "chi2_prof": item.get("chi2_prof"),
                    }
                    for item in inventory.get("identities") or []
                ],
            }
        ),
    )
    store.write_json(
        "restart_prediction_invariance.json",
        jsonable(
            {
                "kind": "restart_prediction_invariance",
                **common,
                "prediction_abs_tolerance_mm": contract.get(
                    "prediction_abs_tolerance_mm"
                ),
                "identities": [
                    {
                        "event": item.get("event"),
                        "target_station": item.get("target_station"),
                        "prediction_invariance": item.get("prediction_invariance"),
                        "max_pairwise_delta_target_loc0": item.get(
                            "max_pairwise_delta_target_loc0"
                        ),
                        "max_pairwise_delta_surviving_loc0": item.get(
                            "max_pairwise_delta_surviving_loc0"
                        ),
                    }
                    for item in inventory.get("identities") or []
                ],
            }
        ),
    )
    store.write_json(
        "restart_parameter_nonuniqueness.json",
        jsonable(
            {
                "kind": "restart_parameter_nonuniqueness",
                **common,
                "identities": [
                    {
                        "event": item.get("event"),
                        "target_station": item.get("target_station"),
                        "parameter_invariance": item.get("parameter_invariance"),
                        "parameter_invariance_alpha": item.get(
                            "parameter_invariance_alpha"
                        ),
                        "nuisance_nonunique": item.get("nuisance_nonunique"),
                        "natives": item.get("natives"),
                    }
                    for item in inventory.get("identities") or []
                ],
            }
        ),
    )
    store.write_json(
        "restart_transport_branch_invariance.json",
        jsonable(
            {
                "kind": "restart_transport_branch_invariance",
                **common,
                "identities": [
                    {
                        "event": item.get("event"),
                        "target_station": item.get("target_station"),
                        "transport_branch_invariance": item.get(
                            "transport_branch_invariance"
                        ),
                    }
                    for item in inventory.get("identities") or []
                ],
            }
        ),
    )
    store.write_json(
        "profile_schur_consistency.json",
        jsonable({"kind": "profile_schur_consistency", **common, "checks": inventory.get("schur")}),
    )
    store.write_json(
        "target_exclusion_audit.json",
        jsonable(
            {
                "kind": "target_exclusion_audit",
                **common,
                "exclusion": inventory.get("exclusion"),
            }
        ),
    )
    store.write_json(
        "held_out_prediction_diagnostic.json",
        jsonable(
            {
                "kind": "held_out_prediction_diagnostic",
                **common,
                "stage": "B",
                "not_used_for_gate": True,
                "diagnostics": inventory.get("held_out"),
            }
        ),
    )
    store.write_json(
        "b14m_restart_invariance_decision.json",
        jsonable(
            {
                "kind": "b14m_restart_invariance_decision",
                **common,
                **decision,
            }
        ),
    )
    store.finalize()
    print(store.run_dir)
    print(decision.get("decision"), decision.get("verdict"))
    return 0 if decision.get("verdict") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
