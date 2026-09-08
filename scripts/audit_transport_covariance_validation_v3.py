#!/usr/bin/env python3
"""Task B8 audit.  Official Transport Covariance V3 decision."""

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
from datasets.transport_covariance_validation_v3 import (
    decide,
    inherit_frozen_stage,
    inventory_and_validate,
    load_config,
)
from evaluation.artifact_store import ImmutableArtifactStore
from scripts.run_transport_covariance_validation_v3 import write_v3_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/transport_covariance_validation_v3.yaml")
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    inventory = inventory_and_validate(config)
    decision = decide(inventory, inherited)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "sbb8_transport_covariance_v3",
    )
    created = datetime.now(timezone.utc).isoformat()
    config_sha = sha256_file(config_path)
    write_v3_artifacts(store, inventory, config_sha)
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json(
        "transport_covariance_v3_decision.json",
        jsonable(
            {
                "created_utc": created,
                "config_sha256": config_sha,
                "git_head_sha": git_head_sha(),
                "present_sources": inventory["present_sources"],
                "missing_sources": inventory["missing_sources"],
                **decision,
            }
        ),
    )
    store.finalize(
        {
            "verdict": decision["verdict"],
            "task": "SB-B8",
            "workbook": 104,
            "decision": decision["decision"],
            "primary_case": decision["primary_case"],
            "next_step": decision["next_step"],
            "config_sha256": config_sha,
            "inherited_wb103_decision_sha256": decision["inherited_wb103_decision_sha256"],
            "inherited_wb103_contract_sha256": decision["inherited_wb103_contract_sha256"],
            "official_input_scope": "contract_eligible",
            "raw_ckf_used": False,
            "measurement_model_v2_entered": False,
            "measurement_model_v2_authorized": decision["measurement_model_v2_authorized"],
            "focus_identity_retained": True,
        }
    )
    print(
        f"SB-B8 verdict={decision['verdict']} case={decision['primary_case']} "
        f"next={decision['next_step']} run_id={store.run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
