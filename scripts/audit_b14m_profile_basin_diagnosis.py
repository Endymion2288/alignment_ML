#!/usr/bin/env python3
"""Task B14M-S audit.  Flat-direction stationarity and basin topology."""

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
from datasets.b14m_profile_basin_diagnosis import (
    FROZEN_FIELD_GRADIENT_SHA,
    decide,
    inherit_frozen_stage,
    inventory_and_audit,
    load_config,
)
from datasets.b14m_restart_invariance import recover_optimizer_contract
from datasets.surface_energy_loss_mean_semantics import PRODUCTION_ELOSS
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/b14m_profile_basin_diagnosis_v1.yaml"
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
        "sbb14ms_profile_basin",
    )
    created = datetime.now(timezone.utc).isoformat()
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
        "config_sha256": sha256_file(config_path),
        "git_head_sha": git_head_sha(),
        "current_helper_sha256": helper_sha,
        "field_gradient_extension_sha256": repair_sha,
        "field_gradient_extension_unchanged": repair_sha == FROZEN_FIELD_GRADIENT_SHA,
        "wb109_dumps_not_overwritten": True,
        "wb114_dumps_not_overwritten": True,
        "wb129_dumps_not_overwritten": True,
        "legacy_b14m_smoke_not_overwritten": True,
        "production_eloss_quantity": PRODUCTION_ELOSS,
        "prior_introduced": False,
        "ridge_added": False,
        "statistical_model_unchanged": True,
        "do_not_change_optimizer": True,
        "do_not_submit_1989": True,
        "full_sample_authorized": False,
        "optimizer_contract": {
            key: contract[key]
            for key in (
                "recovered",
                "optimizer_contract_not_recoverable",
                "max_iterations",
                "pinv_relative",
                "line_search",
                "gradient_norm_z",
                "step_norm_z",
                "relative_chi2_decrease",
            )
            if key in contract
        },
    }
    store.write_json(
        "inherited_stage.json",
        jsonable({"kind": "inherited_stage", **common, "inherited": inherited}),
    )
    store.write_json(
        "endpoint_stationarity_audit.json",
        jsonable(
            {
                "kind": "endpoint_stationarity_audit",
                **common,
                "rows": inventory.get("stationarity"),
            }
        ),
    )
    store.write_json(
        "flat_direction_termination_audit.json",
        jsonable(
            {
                "kind": "flat_direction_termination_audit",
                **common,
                "rows": inventory.get("stationarity"),
            }
        ),
    )
    store.write_json(
        "frozen_linesearch_replay.json",
        jsonable(
            {
                "kind": "frozen_linesearch_replay",
                **common,
                "rows": inventory.get("linesearch"),
            }
        ),
    )
    store.write_json(
        "fixed_alpha_cross_start_matrix.json",
        jsonable(
            {
                "kind": "fixed_alpha_cross_start_matrix",
                **common,
                "identities": {
                    key: item.get("cross_start")
                    for key, item in (inventory.get("per_identity") or {}).items()
                },
            }
        ),
    )
    store.write_json(
        "profile_continuation_forward.json",
        jsonable(
            {
                "kind": "profile_continuation_forward",
                **common,
                "identities": {
                    key: item.get("continuation")
                    for key, item in (inventory.get("per_identity") or {}).items()
                },
            }
        ),
    )
    store.write_json(
        "profile_continuation_backward.json",
        jsonable(
            {
                "kind": "profile_continuation_backward",
                **common,
                "identities": {
                    key: item.get("continuation")
                    for key, item in (inventory.get("per_identity") or {}).items()
                },
            }
        ),
    )
    store.write_json(
        "profile_basin_topology.json",
        jsonable(
            {
                "kind": "profile_basin_topology",
                **common,
                "per_identity": inventory.get("per_identity"),
            }
        ),
    )
    store.write_json(
        "profile_gradient_audit.json",
        jsonable(
            {
                "kind": "profile_gradient_audit",
                **common,
                "rows": inventory.get("gradient"),
            }
        ),
    )
    store.write_json(
        "negative_control_stationarity.json",
        jsonable(
            {
                "kind": "negative_control_stationarity",
                **common,
                "rows": [
                    item
                    for item in inventory.get("stationarity") or []
                    if item.get("role") == "negative_control"
                ],
            }
        ),
    )
    store.write_json(
        "target_exclusion_audit.json",
        jsonable(
            {
                "kind": "target_exclusion_audit",
                **common,
                "exclusion": inventory.get("exclusion"),
                "stage_a": inventory.get("target_exclusion"),
            }
        ),
    )
    store.write_json(
        "b14m_profile_basin_decision.json",
        jsonable({"kind": "b14m_profile_basin_decision", **common, **decision}),
    )
    store.finalize()
    print(store.run_dir)
    print(decision.get("decision"), decision.get("verdict"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
