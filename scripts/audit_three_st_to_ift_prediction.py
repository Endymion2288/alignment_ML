#!/usr/bin/env python3
"""Yasu Stage 1 audit.  Fails closed if dumps or IFT independence are absent."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from alignment.cad_survey_nov22 import git_head_sha
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from datasets.three_st_to_ift_prediction import (
    decide,
    inherit_frozen_stage,
    inventory_split,
    load_config,
    provenance_hashes,
    residual_definition,
    state_definition,
    verify_pinned_calypso_sources,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/three_st_to_ift_prediction_v1.yaml")
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    pins = verify_pinned_calypso_sources(config)
    construction = inventory_split(config, "construction")
    validation = inventory_split(config, "validation")
    dumps_materialized = bool(construction["dumps_present"] and validation["dumps_present"])
    decision = decide(
        construction,
        validation,
        pins,
        inherited,
        dumps_materialized=dumps_materialized,
    )
    decision.update(provenance_hashes(config))
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "yasu_s1_three_st_to_ift_prediction",
    )
    created = datetime.now(timezone.utc).isoformat()
    config_sha = sha256_file(config_path)
    store.write_json(
        "state_definition.json",
        {
            "kind": "three_st_to_ift_state_definition",
            "created_utc": created,
            "config_sha256": config_sha,
            **state_definition(),
        },
    )
    store.write_json(
        "residual_definition.json",
        {
            "kind": "three_st_to_ift_residual_definition",
            "created_utc": created,
            "config_sha256": config_sha,
            **residual_definition(),
        },
    )
    store.write_json("pinned_calypso_sources.json", pins)
    store.write_json(
        "source_inventory.json",
        {
            "kind": "three_st_to_ift_source_inventory",
            "new_source_campaign_started": False,
            "construction": construction,
            "validation": validation,
        },
    )
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json("three_st_to_ift_prediction_contract.json", decision)
    store.finalize(
        {
            "verdict": decision["verdict"],
            "task": "YASU-S1",
            "workbook": 118,
            "decision": decision["decision"],
            "mechanism": decision["mechanism"],
            "config_sha256": config_sha,
            "mc_qp_calibration_authorized": decision["mc_qp_calibration_authorized"],
            "residual_conditional_authorized": decision["residual_conditional_authorized"],
        }
    )
    print(store.run_dir)
    print(decision["verdict"], decision["decision"], decision["mechanism"])
    print(
        "mc_qp_calibration_authorized",
        decision["mc_qp_calibration_authorized"],
        "residual_conditional_authorized",
        decision["residual_conditional_authorized"],
    )
    return 0 if decision["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
