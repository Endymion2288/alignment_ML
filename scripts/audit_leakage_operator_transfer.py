#!/usr/bin/env python3
"""Remeasure leakage operator A on the large-statistics FD-only Jacobian bank.

This is analysis-only.  It never writes geometry, never retunes A, and never
opens the sealed test.  Axial station and C_dx finite differences are
linearized at the all-zero reference.  Any 10% gate failure is recorded as
contract instability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alignment.calibration_modes import evaluate_A_stability, load_mode_validity_contract
from alignment.capture_criteria import load_capture_criteria
from scripts.audit_6dof_identifiability import _source_entries
from scripts.audit_hierarchical_v1_leakage import (
    analyze_route_corner,
    analyze_truth_corner,
    attach_route_target,
    attach_truth_target,
    load_route_fd_cache,
    load_truth_fd_cache,
    _collect_A_rows,
    _drop_private,
)
from scripts.run_refit_multidof_closure import _json_ready, _read_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--iteration-manifest", required=True)
    parser.add_argument("--capture-criteria-station", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-point", default="iteration_00_reference")
    parser.add_argument("--synthetic-root", default=None)
    parser.add_argument("--v2-backbone-train", default=None)
    parser.add_argument("--v2-backbone-validation", default=None)
    parser.add_argument("--scan-root-train", default=None)
    parser.add_argument("--scan-root-validation", default=None)
    parser.add_argument("--skip-route-selected", action="store_true")
    parser.add_argument("--skip-truth-selected", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    contract = load_mode_validity_contract(Path(args.contract).expanduser().resolve())
    criteria = load_capture_criteria(Path(args.capture_criteria_station).expanduser().resolve())
    iteration = _read_json(Path(args.iteration_manifest).expanduser().resolve())
    if iteration.get("test_data_accessed") is not False or int(iteration.get("q_over_p_mode", -1)) != 0:
        raise ValueError("iteration manifest is not a sealed mode-0 bank")
    target = str(args.target_point)
    corners: list[dict] = []
    by_split = {
        split: [entry for entry in _source_entries(iteration) if str(entry.get("split")) == split]
        for split in ("train", "validation")
    }
    fd_cache: dict[str, dict] = {}
    if not args.skip_truth_selected:
        for split, entries in by_split.items():
            for entry in entries:
                print(f"load truth FD cache {entry['source_id']}", flush=True)
                fd_cache[str(entry["source_id"])] = load_truth_fd_cache(entry, min_truth_match_fraction=0.99)
            observed = {str(entry["source_id"]): entry for entry in entries}
            print(f"truth {split} {target}", flush=True)
            banks = [
                attach_truth_target(fd_cache[str(entry["source_id"])], observed[str(entry["source_id"])], target)
                for entry in entries
            ]
            result = analyze_truth_corner(banks, criteria=criteria)
            corners.append(
                {
                    "observation_kind": "truth_selected",
                    "split": split,
                    "target": target,
                    "geometry": "iteration_00",
                    "result": result,
                }
            )
    if not args.skip_route_selected:
        required = (
            args.synthetic_root,
            args.v2_backbone_train,
            args.v2_backbone_validation,
            args.scan_root_train,
            args.scan_root_validation,
        )
        if any(item is None for item in required):
            raise ValueError("route-selected A transfer requires synthetic root, V2 backbones, and scan roots")
        route_scan = {"train": Path(args.scan_root_train), "validation": Path(args.scan_root_validation)}
        route_backbone = {
            "train": Path(args.v2_backbone_train),
            "validation": Path(args.v2_backbone_validation),
        }
        synth = Path(args.synthetic_root).expanduser().resolve()
        for split in ("train", "validation"):
            print(f"load route FD cache {split}", flush=True)
            cache = load_route_fd_cache(
                split=split,
                scan_root=route_scan[split],
                synthetic_root=synth,
                backbone=route_backbone[split],
            )
            print(f"route {split} {target}", flush=True)
            bank = attach_route_target(
                cache,
                target_point=target,
                target_synthetic_root=synth,
                target_scan_root=None,
            )
            result = analyze_route_corner(bank, criteria=criteria)
            corners.append(
                {
                    "observation_kind": "route_selected",
                    "split": split,
                    "target": target,
                    "geometry": "iteration_00",
                    "result": result,
                }
            )
    rows = _collect_A_rows(corners)
    scored = []
    for row in rows:
        stability = evaluate_A_stability(
            contract,
            measured_A_dx=float(row["A6_dx"]),
            measured_A_ry=float(row["A6_ry"]),
        )
        scored.append({**row, **stability})
    all_stable = bool(scored) and all(bool(item["stable"]) for item in scored)
    report = {
        "schema_version": "faser-ift-leakage-operator-transfer-v1",
        "contract": str(Path(args.contract).expanduser().resolve()),
        "iteration_manifest": str(Path(args.iteration_manifest).expanduser().resolve()),
        "target": target,
        "n_corners": len(scored),
        "all_stable": all_stable,
        "do_not_retune": True,
        "geometry_write_candidate": False,
        "joint_station_cdx_injection": False,
        "residual_reduction_is_not_alignment_success": True,
        "test_data_accessed": False,
        "corners": scored,
        "pooled_operators": {
            f"{corner['observation_kind']}_{corner['split']}": _drop_private(corner["result"]["pooled"])["operators"][
                "production_six_dof_survey_dz"
            ]
            for corner in corners
        },
        "if_unstable": "mark contract instability; do not update A, capture, V2, or DoF",
    }
    (output / "A_stability.json").write_text(
        json.dumps(_json_ready(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output / "A_stability.json"), "all_stable": all_stable, "n_corners": len(scored)}, indent=2))


if __name__ == "__main__":
    main()
