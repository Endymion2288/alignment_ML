#!/usr/bin/env python3
"""Yasu-S3A Jacobian audit.  Mean-response only.  Never claims a weak mode."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from datasets.yasu_s3a_jacobian_closure import (
    association_contract,
    decide,
    evaluate_controls,
    evaluate_track_jacobian,
    focus_manifest,
    geometry_response_definition,
    identity_contract,
    inherit_frozen_stage,
    inventory_dumps,
    jsonable,
    load_config,
    load_focus_tracks,
    provenance_hashes,
    residual_definition,
    state_definition,
)
from evaluation.artifact_store import ImmutableArtifactStore


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/yasu_s3a_jacobian_closure_v1.yaml")
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    if bool(config.get("htcondor_authorized")):
        raise SystemExit("S3A must not submit HTCondor")
    inherited = inherit_frozen_stage(config)
    controls = evaluate_controls()
    dumps = inventory_dumps(config)
    tracks = load_focus_tracks(config) if dumps["dumps_present"] else []
    track_results = [evaluate_track_jacobian(track, config) for track in tracks]
    decision = decide(
        inherited=inherited,
        controls=controls,
        track_results=track_results,
        dumps_materialized=bool(dumps["dumps_present"]),
        config=config,
    )
    decision.update(provenance_hashes(config))
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "yasu_s3a_jacobian_closure",
    )
    created = datetime.now(timezone.utc).isoformat()
    config_sha = sha256_file(config_path)
    code_sha = git_head_sha(project_root())
    store.write_json(
        "preregistration.json",
        jsonable({
            "kind": "yasu_s3a_jacobian_preregistration",
            "created_utc": created,
            "config_sha256": config_sha,
            "code_sha": code_sha,
            "frozen_before_jacobian": True,
            "focus_manifest": focus_manifest(config),
            "residual_definition": residual_definition(),
            "state_definition": state_definition(),
            "geometry_response_definition": geometry_response_definition(),
            "identity_contract": identity_contract(),
            "association_contract": association_contract(),
            "finite_difference": decision["finite_difference"],
            "three_st_qp_trusted_observable": False,
            "qp_bending_proxy_used": False,
            "native_5x5_weighted": False,
            "four_station_association_used": False,
            "htcondor_submitted": False,
            "weak_mode_claimed": False,
            "s2k_isolated": True,
        }),
    )
    store.write_json("focus_manifest.json", jsonable(focus_manifest(config)))
    store.write_json("analytic_controls.json", jsonable(controls))
    store.write_json("source_inventory.json", jsonable({"kind": "yasu_s3a_source_inventory", **dumps}))
    store.write_json("inherited_stage.json", jsonable({"kind": "inherited_stage", **inherited}))
    store.write_json(
        "track_jacobians.json",
        jsonable({
            "kind": "yasu_s3a_track_jacobians",
            "n": len(track_results),
            "tracks": track_results,
            "mean_response_only": True,
            "high_correlation_is_not_a_weak_mode": True,
        }),
    )
    store.write_json("yasu_s3a_jacobian_contract.json", jsonable(decision))
    store.finalize(
        jsonable({
            "verdict": decision["verdict"],
            "task": "YASU-S3A",
            "workbook": 128,
            "decision": decision["decision"],
            "config_sha256": config_sha,
            "code_sha": code_sha,
            "three_st_qp_trusted_observable": False,
            "residual_conditional_authorized": False,
            "qp_bending_proxy_used": False,
            "native_5x5_weighted": False,
            "four_station_association_used": False,
            "htcondor_submitted": False,
            "weak_mode_claimed": False,
            "authorize_profiled_schur_next": decision["authorize_profiled_schur_next"],
            "s2k_isolated": True,
            "dumps_materialized": decision["dumps_materialized"],
            "legal_final_claim": decision["legal_final_claim"],
            "forbidden_claim": decision["forbidden_claim"],
        }),
    )
    print(store.run_dir)
    print(decision["verdict"], decision["decision"])
    print("dumps", decision["dumps_materialized"], "n_tracks", decision["n_tracks"])
    print("authorize_profiled_schur_next", decision["authorize_profiled_schur_next"])
    print("weak_mode_claimed", decision["weak_mode_claimed"])
    return 0 if decision["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
