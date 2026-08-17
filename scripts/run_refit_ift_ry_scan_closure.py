#!/usr/bin/env python3
"""Run a held-out, multi-payload physical IFT ``R_y`` closure.

The calibration response curve and the held-out point are all read from a
completed real ``/Tracker/Align`` scan.  This command never shifts a state or
residual in memory: each response was created by the persisted SCT cluster ->
segment refit -> Acts mode-0 chain before this script reads it.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from alignment.payload import load_station_rigid_alignment_payload
from alignment.rotation_closure import solve_ift_ry_polynomial_response
from scripts.run_refit_ift_ry_closure import _aligned_station0_rows, _assert_ift_only, _evaluation


def _point_map(scan_root: Path) -> dict[str, Mapping[str, Any]]:
    plan_path = scan_root / "scan_plan.json"
    with plan_path.open(encoding="utf-8") as handle:
        plan = json.load(handle)
    if not isinstance(plan, Mapping) or plan.get("scan_mode") != "ift_ry_rotation":
        raise ValueError("scan root is not an IFT R_y physical scan")
    points = plan.get("points")
    if not isinstance(points, list):
        raise ValueError("scan plan has no point list")
    result: dict[str, Mapping[str, Any]] = {}
    for raw in points:
        if not isinstance(raw, Mapping):
            raise ValueError("scan plan contains an invalid point")
        name = str(raw.get("name", ""))
        if not name or name in result:
            raise ValueError("scan plan has an invalid or duplicate point name")
        result[name] = raw
    return result


def _paths(scan_root: Path, point: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    root = scan_root / str(point["relative_point_dir"])
    tracklets = root / "refit" / "tracklets.root"
    propagations = root / "refit" / "propagations.root"
    payload = root / "payload" / "alignment_payload.json"
    for path in (tracklets, propagations, payload):
        if not path.is_file():
            raise FileNotFoundError(f"physical scan point is incomplete: {path}")
    return tracklets, propagations, payload


def _load_point(scan_root: Path, point: Mapping[str, Any], *, label: str):
    tracklets, propagations, payload_path = _paths(scan_root, point)
    payload = load_station_rigid_alignment_payload(payload_path)
    angle = _assert_ift_only(payload, label=label, require_pure_rotation=True)
    return {
        "name": str(point["name"]),
        "angle_mrad": angle,
        "tracklets": tracklets,
        "propagations": propagations,
        "payload": payload,
        "evaluation": _evaluation(str(tracklets), str(propagations), 0.99),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", required=True)
    parser.add_argument("--observed-point", required=True)
    parser.add_argument(
        "--calibration-point",
        action="append",
        required=True,
        help="Completed pure-R_y point used to calibrate the response; repeat this option.",
    )
    parser.add_argument("--polynomial-degree", type=int, default=3)
    parser.add_argument("--search-min-mrad", type=float, default=-80.0)
    parser.add_argument("--search-max-mrad", type=float, default=80.0)
    parser.add_argument("--capture-tolerance-mrad", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if not np.isfinite(args.capture_tolerance_mrad) or args.capture_tolerance_mrad < 0.0:
        parser.error("--capture-tolerance-mrad must be finite and non-negative")
    scan_root = Path(args.scan_root).expanduser().resolve()
    points = _point_map(scan_root)
    if args.observed_point not in points:
        parser.error("--observed-point is absent from the physical scan plan")
    if len(set(args.calibration_point)) != len(args.calibration_point):
        parser.error("--calibration-point values must be unique")
    if args.observed_point in set(args.calibration_point):
        parser.error("the held-out observed point cannot be a calibration point")
    unknown = sorted(set(args.calibration_point) - set(points))
    if unknown:
        parser.error("calibration point(s) absent from scan plan: " + ", ".join(unknown))
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    calibration = [_load_point(scan_root, points[name], label=f"calibration '{name}'") for name in args.calibration_point]
    observed = _load_point(scan_root, points[args.observed_point], label="held-out observed point")
    angles = np.asarray([entry["angle_mrad"] for entry in calibration], dtype=np.float64)
    if not np.any(np.isclose(angles, 0.0, rtol=0.0, atol=1.0e-12)):
        raise ValueError("multi-payload physical closure requires nominal ry_0 in calibration")
    if len(np.unique(angles)) != angles.size:
        raise ValueError("calibration payloads contain duplicate R_y values")

    evaluations = tuple(entry["evaluation"] for entry in calibration) + (observed["evaluation"],)
    keys, indexes, overlap = _aligned_station0_rows(*evaluations)
    residuals: list[np.ndarray] = []
    nominal_covariance: np.ndarray | None = None
    nominal_index = int(np.flatnonzero(np.isclose(angles, 0.0, rtol=0.0, atol=1.0e-12))[0])
    for position, (evaluation, index) in enumerate(zip(evaluations, indexes)):
        rows = np.asarray([index[key] for key in keys], dtype=np.intp)
        residuals.append(np.asarray(evaluation.residual[rows], dtype=np.float64))
        if position == nominal_index:
            nominal_covariance = np.asarray(evaluation.combined_covariance[rows], dtype=np.float64)
    if nominal_covariance is None:  # pragma: no cover - checked above
        raise RuntimeError("nominal covariance was not recovered")
    fit = solve_ift_ry_polynomial_response(
        angles,
        np.stack(residuals[:-1], axis=0),
        residuals[-1],
        nominal_covariance,
        polynomial_degree=int(args.polynomial_degree),
        search_interval_mrad=(float(args.search_min_mrad), float(args.search_max_mrad)),
    )
    injected = float(observed["angle_mrad"])
    error = float(fit.recovered_ry_mrad - injected)
    with (output / "closure_pairs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("run_id", "event_id", "source_tracklet_id", "target_tracklet_id", "q_over_p_mode"),
        )
        writer.writeheader()
        writer.writerows(
            {
                "run_id": key[0],
                "event_id": key[1],
                "source_tracklet_id": key[2],
                "target_tracklet_id": key[3],
                "q_over_p_mode": key[4],
            }
            for key in keys
        )
    summary = {
        "method": "truth_fixed_held_out_physical_polynomial_ift_ry_closure",
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "q_over_p_mode": 0,
        "reference_station_ids": [1, 2, 3],
        "movable_station_ids": [0],
        "observed_point": str(observed["name"]),
        "calibration_points": [str(entry["name"]) for entry in calibration],
        "calibration_ry_mrad": [float(value) for value in angles],
        "polynomial_degree": fit.polynomial_degree,
        "search_interval_mrad": list(fit.search_interval_mrad),
        "injected_ry_mrad": injected,
        "recovered_ry_mrad": fit.recovered_ry_mrad,
        "absolute_recovery_error_mrad": abs(error),
        "signed_recovery_error_mrad": error,
        "recovered_variance_mrad2": fit.variance_mrad2,
        "response_chi2": fit.response_chi2,
        "response_ndof": fit.response_ndof,
        "used_station0_truth_pairs": fit.used_pairs,
        "truth_pair_overlap": overlap,
        "capture_tolerance_mrad": float(args.capture_tolerance_mrad),
        "capture_success": abs(error) <= float(args.capture_tolerance_mrad),
        "calibration_payloads": {
            str(entry["name"]): str(entry["payload"].manifest_path) for entry in calibration
        },
        "observed_payload": str(observed["payload"].manifest_path),
    }
    (output / "closure.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
