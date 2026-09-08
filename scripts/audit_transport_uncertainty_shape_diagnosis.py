#!/usr/bin/env python3
"""Task B9 audit.  Official transport / material / state-uncertainty diagnosis."""

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
from datasets.transport_uncertainty_shape_diagnosis import (
    decide,
    inherit_frozen_stage,
    inventory_and_diagnose,
    load_config,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/transport_uncertainty_shape_diagnosis_v1.yaml")
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    inventory = inventory_and_diagnose(config)
    decision = decide(inventory, inherited)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sbb9_transport_uncertainty_shape",
    )
    created = datetime.now(timezone.utc).isoformat()
    config_sha = sha256_file(config_path)
    common = {
        "created_utc": created,
        "config_sha256": config_sha,
        "git_head_sha": git_head_sha(),
    }
    store.write_json(
        "covariance_budget_decomposition.json",
        jsonable({"kind": "covariance_budget_decomposition", **common, **inventory["budget"]}),
    )
    store.write_json(
        "shared_measurement_dependency.json",
        jsonable({"kind": "shared_measurement_dependency", **common, **inventory["dependency"]}),
    )
    store.write_json(
        "ckf_input_covariance_shape.json",
        jsonable({"kind": "ckf_input_covariance_shape", **common, **inventory["cin_shape"]}),
    )
    store.write_json(
        "material_shape_contribution.json",
        jsonable({"kind": "material_shape_contribution", **common, **inventory["material"]}),
    )
    store.write_json(
        "transport_linearization_shape_audit.json",
        jsonable(
            {"kind": "transport_linearization_shape_audit", **common, **inventory["linearization"]}
        ),
    )
    store.write_json(
        "focus_identity_case_study.json",
        jsonable({"kind": "focus_identity_case_study", **common, **inventory["focus"]}),
    )
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json(
        "transport_uncertainty_shape_decision.json",
        jsonable(
            {
                **common,
                "present_sources": inventory["present_sources"],
                "missing_sources": inventory["missing_sources"],
                "n_contracted": inventory["n_contracted"],
                "n_official_pairs": inventory["n_official_pairs"],
                **decision,
            }
        ),
    )
    store.finalize(
        {
            "verdict": decision["verdict"],
            "task": "SB-B9",
            "workbook": 105,
            "decision": decision["decision"],
            "primary_case": decision["primary_case"],
            "next_step": decision["next_step"],
            "config_sha256": config_sha,
            "inherited_wb104_decision_sha256": decision["inherited_wb104_decision_sha256"],
            "inherited_wb103_contract_sha256": decision["inherited_wb103_contract_sha256"],
            "official_input_scope": "contract_eligible",
            "raw_ckf_used": False,
            "measurement_model_v2_entered": False,
            "measurement_model_v2_authorized": False,
            "focus_identity_retained": True,
        }
    )
    print(
        f"SB-B9 verdict={decision['verdict']} case={decision['primary_case']} "
        f"next={decision['next_step']} run_id={store.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
