#!/usr/bin/env python3
"""Yasu-S2H audit.  CKF/KF-refit provenance on ≤20 frozen identities."""

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
from datasets.three_st_qp_refit_provenance import (
    DECISION_RECORDED,
    conversion_chain,
    decide,
    inherit_frozen_stage,
    inventory_campaign,
    load_config,
    official_quantities,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/three_st_qp_refit_provenance_v1.yaml")
    parser.add_argument("--campaign", choices=("smoke", "batch"), default="smoke")
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    report = inventory_campaign(config, args.campaign)
    decision = decide(
        report,
        inherited,
        dumps_materialized=bool(report["dumps_present"]),
        campaign=args.campaign,
        config=config,
    )
    decision.update(provenance_hashes(config))
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        f"yasu_s2h_three_st_qp_refit_provenance_{args.campaign}",
    )
    created = datetime.now(timezone.utc).isoformat()
    config_sha = sha256_file(config_path)
    store.write_json(
        "preregistration.json",
        {
            "kind": "three_st_qp_refit_provenance_preregistration",
            "created_utc": created,
            "config_sha256": config_sha,
            "code_sha": git_head_sha(project_root()),
            "campaign": args.campaign,
            "binning": config["binning"],
            "gates": config["gates"],
            "focus_identities": config["focus_identities"],
            "official_quantities": official_quantities(),
            "conversion_chain": conversion_chain(),
            "frozen_before_dump_inspection": True,
            "three_st_qp_trusted_observable": False,
            "residual_conditional_authorized": False,
            "do_not_replace_focus_after_seeing_results": True,
        },
    )
    store.write_json(
        "source_inventory.json",
        {
            "kind": "three_st_qp_refit_provenance_inventory",
            "campaign": args.campaign,
            "sources_present": report.get("sources_present"),
            "dumps_present": report.get("dumps_present"),
            "n_tracks": report.get("n_tracks"),
            "n_s2_front": report.get("n_s2_front"),
            "n_s1_front": report.get("n_s1_front"),
            "wb119_join": report.get("wb119_join"),
        },
    )
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json("three_st_qp_refit_provenance_contract.json", decision)
    store.finalize(
        {
            "verdict": decision["verdict"],
            "task": "YASU-S2H",
            "workbook": 124,
            "campaign": args.campaign,
            "decision": decision["decision"],
            "official_mechanism": decision["official_mechanism"],
            "config_sha256": config_sha,
            "code_sha": git_head_sha(project_root()),
            "three_st_qp_trusted_observable": False,
            "residual_conditional_authorized": False,
            "s2_flipped_to_pass": False,
            "focus_replaced_after_results": False,
            "large_dump_submitted": False,
        }
    )
    print(store.run_dir)
    print(
        decision["verdict"],
        decision["decision"],
        decision["diagnosis_verdict"],
        decision["official_mechanism"],
    )
    print(
        "trusted",
        decision["three_st_qp_trusted_observable"],
        "s3",
        decision["residual_conditional_authorized"],
        "n_tracks",
        decision["n_tracks"],
        "n_s2",
        decision["n_s2_front"],
        "n_hole",
        decision["n_ckf_target_hole"],
        "n_mot",
        decision["n_first_mot"],
    )
    if decision.get("fallback_note"):
        print(decision["fallback_note"])
    for name, item in (decision.get("structured_verdicts") or {}).items():
        print(name, item.get("status"))
    if args.campaign == "smoke":
        return 0 if decision["contract_verdict"] == "PASS" else 1
    return 0 if decision["decision"] == DECISION_RECORDED else 1


if __name__ == "__main__":
    raise SystemExit(main())
