#!/usr/bin/env python3
"""Yasu-S2D audit.  Diagnoses WB119 FAIL; never flips S2 or opens S3."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from alignment.cad_survey_nov22 import git_head_sha
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from datasets.three_st_qp_calibration import provenance_hashes
from datasets.three_st_qp_failure_diagnosis import (
    DECISION_RECORDED,
    decide,
    inherit_frozen_stage,
    inventory_split,
    load_config,
    primary_quantities,
    verify_pinned_calypso_sources,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/three_st_qp_failure_diagnosis_v1.yaml")
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
        f"yasu_s2d_three_st_qp_failure_diagnosis_{args.campaign}",
    )
    created = datetime.now(timezone.utc).isoformat()
    config_sha = sha256_file(config_path)
    store.write_json(
        "preregistration.json",
        {
            "kind": "three_st_qp_failure_diagnosis_preregistration",
            "created_utc": created,
            "config_sha256": config_sha,
            "code_sha": git_head_sha(project_root()),
            "campaign": args.campaign,
            "binning": config["binning"],
            "gates": config["gates"],
            "sample_rules": config["sample_rules"],
            "focus_identities": config["focus_identities"],
            "primary_quantities": primary_quantities(),
            "frozen_before_batch_inspection": True,
            "three_st_qp_trusted_observable": False,
            "residual_conditional_authorized": False,
        },
    )
    store.write_json("pinned_calypso_sources.json", pins)
    store.write_json(
        "source_inventory.json",
        {
            "kind": "three_st_qp_failure_diagnosis_inventory",
            "new_source_campaign_started": False,
            "campaign": args.campaign,
            "construction": {
                "split": construction["split"],
                "sources": construction["sources"],
                "dumps_present": construction["dumps_present"],
                "events": construction["events"],
                "n_records": construction["tracks"]["n_records"],
                "n_primary": construction["tracks"]["n_primary"],
                "n_flip": construction["tracks"]["n_flip"],
            },
            "validation": {
                "split": validation["split"],
                "sources": validation["sources"],
                "dumps_present": validation["dumps_present"],
                "events": validation["events"],
                "n_records": validation["tracks"]["n_records"],
                "n_primary": validation["tracks"]["n_primary"],
                "n_flip": validation["tracks"]["n_flip"],
            },
        },
    )
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json("three_st_qp_failure_diagnosis_contract.json", decision)
    store.finalize(
        {
            "verdict": decision["verdict"],
            "task": "YASU-S2D",
            "workbook": 120,
            "campaign": args.campaign,
            "decision": decision["decision"],
            "mechanism": decision["mechanism"],
            "contract_verdict": decision["contract_verdict"],
            "diagnosis_verdict": decision["diagnosis_verdict"],
            "mechanism_verdict": decision["mechanism_verdict"],
            "config_sha256": config_sha,
            "code_sha": git_head_sha(project_root()),
            "three_st_qp_trusted_observable": False,
            "residual_conditional_authorized": False,
            "new_reconstruction_dump_authorized": decision["new_reconstruction_dump_authorized"],
        }
    )
    print(store.run_dir)
    print(
        decision["verdict"],
        decision["decision"],
        decision["contract_verdict"],
        decision["diagnosis_verdict"],
        decision["mechanism_verdict"],
    )
    print(
        "trusted_observable",
        decision["three_st_qp_trusted_observable"],
        "residual_conditional_authorized",
        decision["residual_conditional_authorized"],
        "new_dump",
        decision["new_reconstruction_dump_authorized"],
    )
    if args.campaign == "smoke":
        return 0 if decision["contract_verdict"] == "PASS" else 1
    return 0 if decision["decision"] == DECISION_RECORDED else 1


if __name__ == "__main__":
    raise SystemExit(main())
