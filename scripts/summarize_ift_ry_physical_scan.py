#!/usr/bin/env python3
"""Summarize real IFT-R_y refit/Acts response across a frozen scan plan.

This command is intentionally upstream of association models.  It certifies
the physical candidate graph, conditions/ACTS response, and station-0 state
response at each true payload before synthetic overlay or learned scoring is
allowed to consume the bank.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from alignment.payload import load_station_rigid_alignment_payload
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events
from evaluation.field_propagation import evaluate_field_propagation
from scripts.audit_ift_ry_refit_response import _propagation_response, _tracklet_response
from scripts.audit_physical_route_candidate_graph import (
    _acts_surface_summary,
    _condition_chain_checks,
    audit_candidate_graph,
)


def _paths(point_root: Path) -> dict[str, Path]:
    return {
        "tracklets": point_root / "refit" / "tracklets.root",
        "propagations": point_root / "refit" / "propagations.root",
        "payload": point_root / "payload" / "alignment_payload.json",
        "log": point_root / "logs" / "refit.log",
        "failure": point_root / "failure.json",
    }


def _completed(paths: Mapping[str, Path]) -> bool:
    return not paths["failure"].exists() and all(
        paths[name].is_file() for name in ("tracklets", "propagations", "payload", "log")
    )


def _mode0_evaluation(tracklets: Path, propagations: Path):
    return evaluate_field_propagation(
        load_events(tracklets, require_mc_labels=True),
        load_propagation_records(propagations),
        require_truth_match=True,
        q_over_p_mode=0,
        min_truth_match_fraction=0.99,
    )


def _entry(
    point: Mapping[str, Any],
    point_root: Path,
    nominal_events,
    nominal_evaluation,
) -> dict[str, object]:
    paths = _paths(point_root)
    payload = load_station_rigid_alignment_payload(paths["payload"])
    events = load_events(paths["tracklets"], require_mc_labels=True)
    records = load_propagation_records(paths["propagations"])
    candidate = audit_candidate_graph(events, records, chi2_gate=None)
    surface = _acts_surface_summary(records)
    condition_chain = _condition_chain_checks(paths["log"])
    displaced_evaluation = _mode0_evaluation(paths["tracklets"], paths["propagations"])
    if float(point["condition_magnitude"]) == 0.0:
        tracklet_response = None
        propagation_response = None
    else:
        _, tracklet_response = _tracklet_response(nominal_events, events)
        _, propagation_response = _propagation_response(nominal_evaluation, displaced_evaluation)
    station0 = None if tracklet_response is None else tracklet_response["by_station"].get("0")
    pair01 = None if propagation_response is None else propagation_response["by_station_pair"].get("0->1")
    chain_recall = candidate["candidate_complete_truth_chain_recall"]
    return {
        "name": str(point["name"]),
        "direction_trial": str(point["direction_trial"]),
        "ry_mrad": float(point["ry_mrad"]),
        "abs_ry_mrad": float(point["condition_magnitude"]),
        "joint_dx_mm": float(point["joint_translation_xy_mm"][0]),
        "joint_dy_mm": float(point["joint_translation_xy_mm"][1]),
        "payload_manifest": str(payload.manifest_path),
        "station0_transform": list(payload.transform_for_station(0)),
        "candidate_complete_truth_chain_recall": chain_recall,
        "candidate_capture_success": bool(chain_recall is not None and float(chain_recall) >= 0.99),
        "condition_chain": condition_chain,
        "acts_surface_response": surface,
        "tracklet_state_response": tracklet_response,
        "mode0_propagation_response": propagation_response,
        "station0_state_response": station0,
        "pair01_mode0_response": pair01,
    }


def _flat_row(entry: Mapping[str, object]) -> dict[str, object]:
    checks = entry.get("condition_chain")
    if not isinstance(checks, Mapping):
        checks = {}
    check_values = checks.get("checks") if isinstance(checks.get("checks"), Mapping) else {}
    station0 = entry.get("station0_state_response")
    if not isinstance(station0, Mapping):
        station0 = {}
    delta = station0.get("state_delta") if isinstance(station0.get("state_delta"), Mapping) else {}
    values: dict[str, object] = {
        name: entry[name]
        for name in (
            "name",
            "direction_trial",
            "ry_mrad",
            "abs_ry_mrad",
            "joint_dx_mm",
            "joint_dy_mm",
            "candidate_complete_truth_chain_recall",
            "candidate_capture_success",
        )
    }
    values.update(
        {
            "acts_layer_overlap_error_count": checks.get("acts_layer_overlap_error_count"),
            "acts_surface_error_count": checks.get("acts_surface_error_count"),
            "condition_chain_loaded": bool(check_values) and all(bool(value) for value in check_values.values()),
        }
    )
    for label in ("x_mm", "y_mm", "z_mm", "tx", "ty"):
        summary = delta.get(label) if isinstance(delta.get(label), Mapping) else {}
        values[f"station0_delta_{label}_mean"] = summary.get("mean")
        values[f"station0_delta_{label}_rms"] = summary.get("rms")
    return values


def _by_magnitude(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[float, list[dict[str, object]]] = defaultdict(list)
    for entry in entries:
        grouped[float(entry["abs_ry_mrad"])].append(entry)
    rows = []
    for magnitude, group in sorted(grouped.items()):
        recalls = [
            float(entry["candidate_complete_truth_chain_recall"])
            for entry in group
            if entry["candidate_complete_truth_chain_recall"] is not None
        ]
        rows.append(
            {
                "abs_ry_mrad": magnitude,
                "trials": len(group),
                "candidate_capture_fraction": float(
                    np.mean([bool(entry["candidate_capture_success"]) for entry in group])
                ),
                "candidate_truth_chain_recall_mean": None if not recalls else float(np.mean(recalls)),
                "candidate_truth_chain_recall_min": None if not recalls else float(np.min(recalls)),
                "candidate_truth_chain_recall_max": None if not recalls else float(np.max(recalls)),
            }
        )
    return rows


def _plot(output: Path, entries: list[dict[str, object]]) -> None:
    pure = [entry for entry in entries if np.isclose(float(entry["joint_dx_mm"]), 0.0) and np.isclose(float(entry["joint_dy_mm"]), 0.0)]
    ordered = sorted(pure, key=lambda entry: float(entry["ry_mrad"]))
    x = np.asarray([float(entry["ry_mrad"]) for entry in ordered])
    recall = np.asarray(
        [float(entry["candidate_complete_truth_chain_recall"]) for entry in ordered], dtype=np.float64
    )
    dx = np.asarray(
        [
            np.nan
            if not isinstance(entry.get("station0_state_response"), Mapping)
            else entry["station0_state_response"]["state_delta"]["x_mm"]["mean"]
            for entry in ordered
        ],
        dtype=np.float64,
    )
    dtx = np.asarray(
        [
            np.nan
            if not isinstance(entry.get("station0_state_response"), Mapping)
            else entry["station0_state_response"]["state_delta"]["tx"]["mean"]
            for entry in ordered
        ],
        dtype=np.float64,
    )
    figure, axes = plt.subplots(3, 1, figsize=(7.0, 9.0), sharex=True, constrained_layout=True)
    axes[0].plot(x, recall, marker="o", color="#267a85")
    axes[0].set_ylabel("raw truth-chain recall")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].plot(x, dx, marker="o", color="#b95b31")
    axes[1].set_ylabel("IFT mean delta x [mm]")
    axes[2].plot(x, dtx, marker="o", color="#5f7c44")
    axes[2].set_ylabel("IFT mean delta tx")
    axes[2].set_xlabel("injected IFT R_y [mrad]")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    scan_root = Path(args.scan_root).expanduser().resolve()
    plan = json.loads((scan_root / "scan_plan.json").read_text(encoding="utf-8"))
    if plan.get("scan_mode") != "ift_ry_rotation":
        raise ValueError("scan plan is not an IFT R_y physical rotation scan")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    points = plan.get("points")
    if not isinstance(points, list):
        raise ValueError("scan plan lacks points")
    zero = next(point for point in points if float(point["condition_magnitude"]) == 0.0)
    zero_paths = _paths(scan_root / str(zero["relative_point_dir"]))
    if not _completed(zero_paths):
        raise RuntimeError("nominal physical point is incomplete")
    nominal_events = load_events(zero_paths["tracklets"], require_mc_labels=True)
    nominal_evaluation = _mode0_evaluation(zero_paths["tracklets"], zero_paths["propagations"])
    entries: list[dict[str, object]] = []
    incomplete: list[str] = []
    for point in points:
        point_root = scan_root / str(point["relative_point_dir"])
        paths = _paths(point_root)
        if not _completed(paths):
            incomplete.append(str(point["name"]))
            continue
        entries.append(_entry(point, point_root, nominal_events, nominal_evaluation))
    if incomplete and not args.allow_incomplete:
        raise RuntimeError("physical IFT R_y scan is incomplete: " + ", ".join(incomplete))
    output.mkdir(parents=True, exist_ok=True)
    flat = [_flat_row(entry) for entry in entries]
    with (output / "physical_rotation_scan.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in flat for key in row}))
        writer.writeheader()
        writer.writerows(flat)
    by_magnitude = _by_magnitude(entries)
    payload = {
        "method": "physical_ift_ry_refit_acts_scan_summary",
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "q_over_p_mode": 0,
        "scan_root": str(scan_root),
        "reference_station_ids": [1, 2, 3],
        "movable_station_ids": [0],
        "completed_points": len(entries),
        "incomplete_points": incomplete,
        "by_point": entries,
        "by_abs_ry_mrad": by_magnitude,
    }
    (output / "physical_rotation_scan.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    _plot(output / "physical_rotation_scan.png", entries)
    print(json.dumps({"output_dir": str(output), "completed_points": len(entries), "incomplete": incomplete}, indent=2))


if __name__ == "__main__":
    main()
