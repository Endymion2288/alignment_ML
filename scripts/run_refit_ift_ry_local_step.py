#!/usr/bin/env python3
"""Apply one local physical Newton/WLS step for IFT ``R_y`` closure.

The anchor and its lower/upper derivative probes must all be independently
refitted outputs.  The held-out observation is never used as a derivative
probe.  This is deliberately a response of actual alignment payloads rather
than a residual-level shifted approximation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from alignment.payload import load_station_rigid_alignment_payload
from alignment.rotation_closure import solve_ift_ry_finite_difference
from scripts.run_refit_ift_ry_closure import _aligned_station0_rows, _assert_ift_only, _evaluation


def _load(label: str, tracklets: str, propagations: str, payload_manifest: str):
    payload = load_station_rigid_alignment_payload(payload_manifest)
    angle = _assert_ift_only(payload, label=label, require_pure_rotation=True)
    return {
        "angle_mrad": angle,
        "payload": payload,
        "evaluation": _evaluation(tracklets, propagations, 0.99),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("anchor", "lower", "upper", "observed"):
        parser.add_argument(f"--{label}-tracklets", required=True)
        parser.add_argument(f"--{label}-propagations", required=True)
        parser.add_argument(f"--{label}-payload-manifest", required=True)
    parser.add_argument("--capture-tolerance-mrad", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if not np.isfinite(args.capture_tolerance_mrad) or args.capture_tolerance_mrad < 0.0:
        parser.error("--capture-tolerance-mrad must be finite and non-negative")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    anchor = _load(
        "anchor",
        args.anchor_tracklets,
        args.anchor_propagations,
        args.anchor_payload_manifest,
    )
    lower = _load(
        "lower local probe",
        args.lower_tracklets,
        args.lower_propagations,
        args.lower_payload_manifest,
    )
    upper = _load(
        "upper local probe",
        args.upper_tracklets,
        args.upper_propagations,
        args.upper_payload_manifest,
    )
    observed = _load(
        "held-out observed point",
        args.observed_tracklets,
        args.observed_propagations,
        args.observed_payload_manifest,
    )
    if not float(lower["angle_mrad"]) < float(anchor["angle_mrad"]) < float(upper["angle_mrad"]):
        raise ValueError("local physical probes must bracket the anchor R_y")
    evaluations = (
        anchor["evaluation"],
        lower["evaluation"],
        upper["evaluation"],
        observed["evaluation"],
    )
    keys, indexes, overlap = _aligned_station0_rows(*evaluations)
    values = []
    for evaluation, index in zip(evaluations, indexes):
        rows = np.asarray([index[key] for key in keys], dtype=np.intp)
        values.append(
            (
                np.asarray(evaluation.residual[rows], dtype=np.float64),
                np.asarray(evaluation.combined_covariance[rows], dtype=np.float64),
            )
        )
    anchor_residual, anchor_covariance = values[0]
    lower_residual, _ = values[1]
    upper_residual, _ = values[2]
    observed_residual, _ = values[3]
    fit = solve_ift_ry_finite_difference(
        anchor_residual,
        upper_residual,
        lower_residual,
        observed_residual,
        anchor_covariance,
        positive_ry_mrad=float(upper["angle_mrad"]) - float(anchor["angle_mrad"]),
        negative_ry_mrad=float(lower["angle_mrad"]) - float(anchor["angle_mrad"]),
    )
    recovered = float(anchor["angle_mrad"]) + fit.recovered_ry_mrad
    injected = float(observed["angle_mrad"])
    error = recovered - injected
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
        "method": "truth_fixed_local_physical_finite_difference_ift_ry_step",
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "q_over_p_mode": 0,
        "reference_station_ids": [1, 2, 3],
        "movable_station_ids": [0],
        "anchor_ry_mrad": float(anchor["angle_mrad"]),
        "lower_probe_ry_mrad": float(lower["angle_mrad"]),
        "upper_probe_ry_mrad": float(upper["angle_mrad"]),
        "recovered_increment_mrad": fit.recovered_ry_mrad,
        "injected_ry_mrad": injected,
        "recovered_ry_mrad": recovered,
        "absolute_recovery_error_mrad": abs(error),
        "signed_recovery_error_mrad": error,
        "recovered_increment_variance_mrad2": fit.variance_mrad2,
        "normal_matrix": fit.normal_matrix.tolist(),
        "normal_matrix_rank": 1,
        "normal_matrix_condition_number": 1.0,
        "response_chi2": fit.response_chi2,
        "response_ndof": fit.response_ndof,
        "used_station0_truth_pairs": fit.used_pairs,
        "truth_pair_overlap": overlap,
        "capture_tolerance_mrad": float(args.capture_tolerance_mrad),
        "capture_success": abs(error) <= float(args.capture_tolerance_mrad),
        "payloads": {
            "anchor": str(anchor["payload"].manifest_path),
            "lower": str(lower["payload"].manifest_path),
            "upper": str(upper["payload"].manifest_path),
            "observed": str(observed["payload"].manifest_path),
        },
    }
    (output / "closure.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
