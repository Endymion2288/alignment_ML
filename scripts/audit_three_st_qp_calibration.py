#!/usr/bin/env python3
"""Yasu Stage 2 audit.  Fails closed if dumps, matching, or gates fail."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from alignment.cad_survey_nov22 import git_head_sha
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from datasets.three_st_qp_calibration import (
    decide,
    inherit_frozen_stage,
    inventory_split,
    load_config,
    primary_quantities,
    provenance_hashes,
    state_definition,
    truth_definition,
    verify_pinned_calypso_sources,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/three_st_qp_calibration_v1.yaml")
    parser.add_argument("--campaign", choices=("smoke", "batch"), default="smoke")
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    pins = verify_pinned_calypso_sources(config)
    construction = inventory_split(config, "construction", args.campaign)
    validation = inventory_split(config, "validation", args.campaign)
    dumps_materialized = bool(construction["dumps_present"] and validation["dumps_present"])
    decision = decide(
        construction,
        validation,
        pins,
        inherited,
        dumps_materialized=dumps_materialized,
        campaign=args.campaign,
        config=config,
    )
    decision.update(provenance_hashes(config))
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        f"yasu_s2_three_st_qp_calibration_{args.campaign}",
    )
    created = datetime.now(timezone.utc).isoformat()
    config_sha = sha256_file(config_path)
    store.write_json(
        "state_definition.json",
        {
            "kind": "three_st_qp_state_definition",
            "created_utc": created,
            "config_sha256": config_sha,
            **state_definition(),
        },
    )
    store.write_json(
        "truth_definition.json",
        {
            "kind": "three_st_qp_truth_definition",
            "created_utc": created,
            "config_sha256": config_sha,
            **truth_definition(),
        },
    )
    store.write_json(
        "preregistration.json",
        {
            "kind": "three_st_qp_preregistration",
            "created_utc": created,
            "config_sha256": config_sha,
            "code_sha": git_head_sha(project_root()),
            "campaign": args.campaign,
            "binning": config["binning"],
            "gates": config["gates"],
            "sample_rules": config["sample_rules"],
            "primary_quantities": primary_quantities(),
            "frozen_before_batch_inspection": True,
        },
    )
    store.write_json("pinned_calypso_sources.json", pins)
    store.write_json(
        "source_inventory.json",
        {
            "kind": "three_st_qp_source_inventory",
            "new_source_campaign_started": False,
            "campaign": args.campaign,
            "construction": {
                "split": construction["split"],
                "sources": construction["sources"],
                "dumps_present": construction["dumps_present"],
                "events": construction["events"],
                "tracks": {
                    key: value
                    for key, value in construction["tracks"].items()
                    if key != "primary_rows"
                },
            },
            "validation": {
                "split": validation["split"],
                "sources": validation["sources"],
                "dumps_present": validation["dumps_present"],
                "events": validation["events"],
                "tracks": {
                    key: value
                    for key, value in validation["tracks"].items()
                    if key != "primary_rows"
                },
            },
        },
    )
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json("three_st_qp_calibration_contract.json", decision)
    store.finalize(
        {
            "verdict": decision["verdict"],
            "task": "YASU-S2",
            "workbook": 119,
            "campaign": args.campaign,
            "decision": decision["decision"],
            "mechanism": decision["mechanism"],
            "contract_verdict": decision["contract_verdict"],
            "physics_verdict": decision["physics_verdict"],
            "config_sha256": config_sha,
            "code_sha": git_head_sha(project_root()),
            "batch_htcondor_authorized": decision["batch_htcondor_authorized"],
            "residual_conditional_authorized": decision["residual_conditional_authorized"],
            "three_st_qp_trusted_observable": decision["three_st_qp_trusted_observable"],
        }
    )
    print(store.run_dir)
    print(
        decision["verdict"],
        decision["decision"],
        decision["contract_verdict"],
        decision["physics_verdict"],
        decision["mechanism"],
    )
    print(
        "trusted_observable",
        decision["three_st_qp_trusted_observable"],
        "residual_conditional_authorized",
        decision["residual_conditional_authorized"],
        "batch_htcondor_authorized",
        decision["batch_htcondor_authorized"],
    )
    if args.campaign == "smoke":
        return 0 if decision["contract_verdict"] == "PASS" else 1
    return 0 if decision["decision"] == "three_st_qp_calibration_established" else 1


if __name__ == "__main__":
    raise SystemExit(main())
