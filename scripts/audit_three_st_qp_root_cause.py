#!/usr/bin/env python3
"""Yasu-S2E audit.  Splits curvature vs covariance.  Never flips S2 or opens S3."""

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
from datasets.three_st_qp_failure_diagnosis import verify_pinned_calypso_sources
from datasets.three_st_qp_root_cause_audit import (
    DECISION_RECORDED,
    conversion_chain,
    decide,
    diagnostic_export_contract,
    inherit_frozen_stage,
    inventory_split,
    load_config,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/three_st_qp_root_cause_audit_v1.yaml")
    parser.add_argument("--campaign", choices=("smoke", "batch"), default="smoke")
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    pins = verify_pinned_calypso_sources(config)
    construction = inventory_split(config, "construction", args.campaign)
    validation = inventory_split(config, "validation", args.campaign)
    decision = decide(
        construction,
        validation,
        pins,
        inherited,
        dumps_materialized=bool(construction["dumps_present"] and validation["dumps_present"]),
        campaign=args.campaign,
        config=config,
    )
    decision.update(provenance_hashes(config))
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        f"yasu_s2e_three_st_qp_root_cause_{args.campaign}",
    )
    created = datetime.now(timezone.utc).isoformat()
    config_sha = sha256_file(config_path)
    store.write_json(
        "preregistration.json",
        {
            "kind": "three_st_qp_root_cause_preregistration",
            "created_utc": created,
            "config_sha256": config_sha,
            "code_sha": git_head_sha(project_root()),
            "campaign": args.campaign,
            "binning": config["binning"],
            "gates": config["gates"],
            "clean_subsample": config["clean_subsample"],
            "focus_identities": config["focus_identities"],
            "conversion_chain": conversion_chain(),
            "diagnostic_export_contract": diagnostic_export_contract(),
            "frozen_before_batch_inspection": True,
            "three_st_qp_trusted_observable": False,
            "residual_conditional_authorized": False,
        },
    )
    store.write_json("pinned_calypso_sources.json", pins)
    store.write_json(
        "source_inventory.json",
        {
            "kind": "three_st_qp_root_cause_inventory",
            "campaign": args.campaign,
            "construction": {
                "sources": construction["sources"],
                "dumps_present": construction["dumps_present"],
                "n_primary": construction["tracks"]["n_primary"],
                "n_flip": construction["tracks"]["n_flip"],
            },
            "validation": {
                "sources": validation["sources"],
                "dumps_present": validation["dumps_present"],
                "n_primary": validation["tracks"]["n_primary"],
                "n_flip": validation["tracks"]["n_flip"],
            },
        },
    )
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json("three_st_qp_root_cause_contract.json", decision)
    store.finalize(
        {
            "verdict": decision["verdict"],
            "task": "YASU-S2E",
            "workbook": 121,
            "campaign": args.campaign,
            "decision": decision["decision"],
            "A": decision["A"],
            "B": decision["B"],
            "C": decision["C"],
            "D": decision["D"],
            "config_sha256": config_sha,
            "code_sha": git_head_sha(project_root()),
            "three_st_qp_trusted_observable": False,
            "residual_conditional_authorized": False,
            "new_reconstruction_dump_authorized": False,
        }
    )
    print(store.run_dir)
    print(
        decision["verdict"],
        decision["decision"],
        decision["diagnosis_verdict"],
        "A",
        decision["A"],
        "B",
        decision["B"],
        "C",
        decision["C"],
        "D",
        decision["D"],
    )
    print(
        "trusted",
        decision["three_st_qp_trusted_observable"],
        "s3",
        decision["residual_conditional_authorized"],
        "new_dump",
        decision["new_reconstruction_dump_authorized"],
        "info_limit",
        decision["information_limit_remains_if_covariance_fixed"],
    )
    if args.campaign == "smoke":
        return 0 if decision["contract_verdict"] == "PASS" else 1
    return 0 if decision["decision"] == DECISION_RECORDED else 1


if __name__ == "__main__":
    raise SystemExit(main())
