#!/usr/bin/env python3
"""Yasu-S2G audit.  Extreme tails and split/source instability.  Never flips S2 or opens S3."""

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
from datasets.three_st_qp_tail_source_audit import (
    DECISION_RECORDED,
    decide,
    inherit_frozen_stage,
    inventory_split,
    load_config,
    official_quantities,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/three_st_qp_tail_source_audit_v1.yaml")
    parser.add_argument("--campaign", choices=("smoke", "batch"), default="smoke")
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    construction = inventory_split(config, "construction", args.campaign)
    validation = inventory_split(config, "validation", args.campaign)
    decision = decide(
        construction,
        validation,
        inherited,
        dumps_materialized=bool(construction["dumps_present"] and validation["dumps_present"]),
        campaign=args.campaign,
        config=config,
    )
    decision.update(provenance_hashes(config))
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        f"yasu_s2g_three_st_qp_tail_source_{args.campaign}",
    )
    created = datetime.now(timezone.utc).isoformat()
    config_sha = sha256_file(config_path)
    store.write_json(
        "preregistration.json",
        {
            "kind": "three_st_qp_tail_source_preregistration",
            "created_utc": created,
            "config_sha256": config_sha,
            "code_sha": git_head_sha(project_root()),
            "campaign": args.campaign,
            "binning": config["binning"],
            "gates": config["gates"],
            "clean_subsample": config["clean_subsample"],
            "focus_identities": config["focus_identities"],
            "official_quantities": official_quantities(),
            "frozen_before_batch_inspection": True,
            "three_st_qp_trusted_observable": False,
            "residual_conditional_authorized": False,
            "official_mean_is_untrimmed": True,
            "reweight_is_diagnostic_only": True,
            "influence_is_diagnostic_only": True,
        },
    )
    store.write_json(
        "source_inventory.json",
        {
            "kind": "three_st_qp_tail_source_inventory",
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
    store.write_json("three_st_qp_tail_source_contract.json", decision)
    store.finalize(
        {
            "verdict": decision["verdict"],
            "task": "YASU-S2G",
            "workbook": 123,
            "campaign": args.campaign,
            "decision": decision["decision"],
            "official_mechanism": decision["official_mechanism"],
            "repeatable_reconstruction_bias": decision["repeatable_reconstruction_bias"]["repeatable"],
            "config_sha256": config_sha,
            "code_sha": git_head_sha(project_root()),
            "three_st_qp_trusted_observable": False,
            "residual_conditional_authorized": False,
            "new_reconstruction_dump_authorized": decision["new_reconstruction_dump_authorized"],
            "official_mean_trimmed": False,
        }
    )
    print(store.run_dir)
    print(
        decision["verdict"],
        decision["decision"],
        decision["diagnosis_verdict"],
        decision["official_mechanism"],
        "repeatable",
        decision["repeatable_reconstruction_bias"]["repeatable"],
    )
    print(
        "trusted",
        decision["three_st_qp_trusted_observable"],
        "s3",
        decision["residual_conditional_authorized"],
        "new_dump",
        decision["new_reconstruction_dump_authorized"],
        "trimmed",
        decision["official_mean_trimmed"],
    )
    for name, item in decision["structured_verdicts"].items():
        print(name, item.get("status"))
    if args.campaign == "smoke":
        return 0 if decision["contract_verdict"] == "PASS" else 1
    return 0 if decision["decision"] == DECISION_RECORDED else 1


if __name__ == "__main__":
    raise SystemExit(main())
