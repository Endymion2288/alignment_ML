#!/usr/bin/env python3
"""Yasu-S2K audit.  Field-integral-normalized 3ST bending response."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import subprocess
from pathlib import Path

from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)


def git_head_sha(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"
from datasets.faser_field_table import lock_field_unit_sign_contract
from datasets.three_st_qp_calibration import provenance_hashes
from datasets.three_st_qp_field_normalized_bending import (
    conversion_chain,
    decide,
    inherit_frozen_stage,
    inventory_campaign,
    load_config,
    load_official_field,
    official_quantities,
    verify_frozen_dumps,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/three_st_qp_field_normalized_bending_v1.yaml")
    parser.add_argument("--campaign", choices=("smoke", "batch"), default="smoke")
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_frozen_stage(config)
    field = load_official_field(config)
    field_contract = lock_field_unit_sign_contract(field)
    dump_check = verify_frozen_dumps(config, args.campaign)
    report = inventory_campaign(config, args.campaign, field)
    decision = decide(
        report,
        inherited,
        field_contract,
        dumps_materialized=bool(report["dumps_present"]),
        campaign=args.campaign,
        config=config,
    )
    decision.update(provenance_hashes(config))
    decision["frozen_dump_check"] = dump_check
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        f"yasu_s2k_three_st_qp_field_normalized_bending_{args.campaign}",
    )
    created = datetime.now(timezone.utc).isoformat()
    config_sha = sha256_file(config_path)
    store.write_json(
        "preregistration.json",
        {
            "kind": "three_st_qp_field_normalized_bending_preregistration",
            "created_utc": created,
            "config_sha256": config_sha,
            "code_sha": git_head_sha(project_root()),
            "campaign": args.campaign,
            "gates": config["gates"],
            "binning": config["binning"],
            "path": config["path"],
            "conversion": config["conversion"],
            "observable": config["observable"],
            "official_quantities": official_quantities(),
            "conversion_chain": conversion_chain(),
            "frozen_before_proxy_vs_truth": True,
            "three_st_qp_trusted_observable": False,
            "residual_conditional_authorized": False,
            "official_qp_like_jacobian_authorized": False,
        },
    )
    store.write_json("field_unit_sign_contract.json", field_contract)
    store.write_json(
        "source_inventory.json",
        {
            "kind": "three_st_qp_field_normalized_bending_inventory",
            "campaign": args.campaign,
            "sources_present": report.get("sources_present"),
            "dumps_present": report.get("dumps_present"),
            "frozen_dump_check": dump_check,
            "n_tracks": report.get("n_tracks"),
            "n_clean_18hit": report.get("n_clean_18hit"),
        },
    )
    store.write_json("inherited_stage.json", {"kind": "inherited_stage", **inherited})
    store.write_json("three_st_qp_field_normalized_bending_contract.json", decision)
    store.finalize(
        {
            "verdict": decision["verdict"],
            "task": "YASU-S2K",
            "workbook": 127,
            "campaign": args.campaign,
            "decision": decision["decision"],
            "verdicts": decision["verdicts"],
            "config_sha256": config_sha,
            "code_sha": git_head_sha(project_root()),
            "three_st_qp_trusted_observable": False,
            "residual_conditional_authorized": False,
            "official_qp_like_jacobian_authorized": False,
            "s2_flipped_to_pass": False,
            "trusted_momentum": False,
        }
    )
    print(store.run_dir)
    print(decision["verdict"], decision["decision"], decision["diagnosis_verdict"])
    print("verdicts", decision["verdicts"])
    print(
        "trusted",
        decision["three_st_qp_trusted_observable"],
        "s3",
        decision["residual_conditional_authorized"],
        "official_jac",
        decision["official_qp_like_jacobian_authorized"],
        "n_clean",
        decision["n_clean_18hit"],
    )
    clean = decision.get("clean_18hit") or {}
    print(
        "sign_agree",
        clean.get("sign_agree"),
        "spearman",
        clean.get("spearman_proxy_vs_qp_truth"),
        "ols",
        clean.get("ols_proxy_vs_qp_truth"),
    )
    print("missingness", decision.get("missingness_all"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
