#!/usr/bin/env python3
"""Truth-fixed station-0 R_y closure from physical refit/Acts outputs.

The downstream 3ST frame is held fixed.  A central finite difference from
actual ``+R_y`` and ``-R_y`` payload refits supplies the local response; no
coordinate-level residual surrogate is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from alignment.payload import load_station_rigid_alignment_payload
from alignment.rotation_closure import solve_ift_ry_finite_difference
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events
from evaluation.field_propagation import evaluate_field_propagation


def _assert_ift_only(payload, *, label: str, require_pure_rotation: bool) -> float:
    for station in (1, 2, 3):
        if not np.allclose(payload.transform_for_station(station), 0.0, rtol=0.0, atol=1.0e-15):
            raise ValueError(f"{label} moves downstream reference station {station}")
    transform = np.asarray(payload.transform_for_station(0), dtype=np.float64)
    if require_pure_rotation and not np.allclose(
        transform[[0, 1, 2, 3, 5]], 0.0, rtol=0.0, atol=1.0e-15
    ):
        raise ValueError(f"{label} must be a pure station-0 R_y payload")
    return float(transform[4] * 1.0e3)


def _evaluation(tracklets: str, propagations: str, min_truth_match_fraction: float):
    return evaluate_field_propagation(
        load_events(tracklets, require_mc_labels=True),
        load_propagation_records(propagations),
        require_truth_match=True,
        q_over_p_mode=0,
        min_truth_match_fraction=min_truth_match_fraction,
    )


def _key(evaluation, row: int) -> tuple[int, int, int, int, int]:
    return (
        int(evaluation.run_id[row]),
        int(evaluation.event_id[row]),
        int(evaluation.source_tracklet_id[row]),
        int(evaluation.target_tracklet_id[row]),
        int(evaluation.q_over_p_mode[row]),
    )


def _row_index(evaluation) -> dict[tuple[int, int, int, int, int], int]:
    index = {_key(evaluation, row): row for row in range(evaluation.size)}
    if len(index) != evaluation.size:
        raise ValueError("physical R_y closure has duplicate truth-pair identities")
    return index


def _aligned_station0_rows(*evaluations):
    indexes = [_row_index(evaluation) for evaluation in evaluations]
    common = set.intersection(*(set(index) for index in indexes))
    all_keys = set.union(*(set(index) for index in indexes))
    selected = sorted(
        key
        for key in common
        if all(int(evaluation.source_station_id[index[key]]) == 0 for evaluation, index in zip(evaluations, indexes))
    )
    if not selected:
        raise ValueError("no common station-0 truth propagation pair remains for physical R_y closure")
    return selected, indexes, {
        "all_common_truth_pairs": len(common),
        "all_union_truth_pairs": len(all_keys),
        "station0_common_truth_pairs": len(selected),
        "non_common_truth_pairs": len(all_keys - common),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("nominal", "positive", "negative", "observed"):
        parser.add_argument(f"--{label}-tracklets", required=True)
        parser.add_argument(f"--{label}-propagations", required=True)
        parser.add_argument(
            f"--{label}-payload-manifest",
            required=(label != "nominal"),
            default=None,
        )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-truth-match-fraction", type=float, default=0.99)
    parser.add_argument("--prior-sigma-mrad", type=float, default=None)
    parser.add_argument("--capture-tolerance-mrad", type=float, default=1.0)
    args = parser.parse_args()
    if not np.isfinite(args.min_truth_match_fraction) or not 0.0 <= args.min_truth_match_fraction <= 1.0:
        parser.error("--min-truth-match-fraction must be in [0, 1]")
    if not np.isfinite(args.capture_tolerance_mrad) or args.capture_tolerance_mrad < 0.0:
        parser.error("--capture-tolerance-mrad must be finite and non-negative")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    positive_payload = load_station_rigid_alignment_payload(args.positive_payload_manifest)
    negative_payload = load_station_rigid_alignment_payload(args.negative_payload_manifest)
    observed_payload = load_station_rigid_alignment_payload(args.observed_payload_manifest)
    positive_ry = _assert_ift_only(positive_payload, label="positive probe", require_pure_rotation=True)
    negative_ry = _assert_ift_only(negative_payload, label="negative probe", require_pure_rotation=True)
    observed_ry = _assert_ift_only(observed_payload, label="observed payload", require_pure_rotation=True)
    if positive_ry <= 0.0 or negative_ry >= 0.0:
        raise ValueError("finite-difference probes must bracket zero with positive and negative IFT R_y")

    evaluations = tuple(
        _evaluation(getattr(args, f"{label}_tracklets"), getattr(args, f"{label}_propagations"), args.min_truth_match_fraction)
        for label in ("nominal", "positive", "negative", "observed")
    )
    keys, indexes, overlap = _aligned_station0_rows(*evaluations)
    arrays = []
    for evaluation, index in zip(evaluations, indexes):
        rows = np.asarray([index[key] for key in keys], dtype=np.intp)
        arrays.append(
            (
                np.asarray(evaluation.residual[rows], dtype=np.float64),
                np.asarray(evaluation.combined_covariance[rows], dtype=np.float64),
                rows,
            )
        )
    nominal_residual, nominal_covariance, nominal_rows = arrays[0]
    positive_residual, _, _ = arrays[1]
    negative_residual, _, _ = arrays[2]
    observed_residual, _, _ = arrays[3]
    fit = solve_ift_ry_finite_difference(
        nominal_residual,
        positive_residual,
        negative_residual,
        observed_residual,
        nominal_covariance,
        positive_ry_mrad=positive_ry,
        negative_ry_mrad=negative_ry,
        prior_sigma_mrad=args.prior_sigma_mrad,
    )
    error = float(fit.recovered_ry_mrad - observed_ry)
    rows = []
    derivative = (positive_residual - negative_residual) / (positive_ry - negative_ry)
    response = observed_residual - nominal_residual
    prediction = derivative * fit.recovered_ry_mrad
    for output_row, (key, source_row) in enumerate(zip(keys, nominal_rows)):
        rows.append(
            {
                "run_id": key[0],
                "event_id": key[1],
                "source_tracklet_id": key[2],
                "target_tracklet_id": key[3],
                "q_over_p_mode": key[4],
                "source_station_id": int(evaluations[0].source_station_id[source_row]),
                "target_station_id": int(evaluations[0].target_station_id[source_row]),
                **{
                    f"derivative_{label}_per_mrad": float(derivative[output_row, column])
                    for column, label in enumerate(("rx_mm", "ry_mm", "rtx", "rty"))
                },
                **{
                    f"observed_response_{label}": float(response[output_row, column])
                    for column, label in enumerate(("rx_mm", "ry_mm", "rtx", "rty"))
                },
                **{
                    f"fitted_response_{label}": float(prediction[output_row, column])
                    for column, label in enumerate(("rx_mm", "ry_mm", "rtx", "rty"))
                },
            }
        )
    import csv

    with (output / "closure_pairs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "method": "truth_fixed_physical_finite_difference_ift_ry_closure",
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "q_over_p_mode": 0,
        "reference_station_ids": [1, 2, 3],
        "movable_station_ids": [0],
        "fit_weight_interpretation": (
            "Nominal combined covariance is a deterministic WLS weight for paired refit response; "
            "it is not an independent covariance of two refits of the same clusters."
        ),
        "positive_probe_ry_mrad": positive_ry,
        "negative_probe_ry_mrad": negative_ry,
        "injected_ry_mrad": observed_ry,
        "recovered_ry_mrad": fit.recovered_ry_mrad,
        "absolute_recovery_error_mrad": abs(error),
        "signed_recovery_error_mrad": error,
        "capture_tolerance_mrad": args.capture_tolerance_mrad,
        "capture_success": abs(error) <= args.capture_tolerance_mrad,
        "normal_matrix_rank": 1,
        "normal_matrix_condition_number": 1.0,
        "normal_matrix": fit.normal_matrix.tolist(),
        "right_hand_side": fit.right_hand_side.tolist(),
        "recovered_variance_mrad2": fit.variance_mrad2,
        "response_chi2": fit.response_chi2,
        "response_ndof": fit.response_ndof,
        "finite_difference_step_mrad": fit.finite_difference_step_mrad,
        "used_station0_truth_pairs": fit.used_pairs,
        "truth_pair_overlap": overlap,
        "payloads": {
            "positive": str(positive_payload.manifest_path),
            "negative": str(negative_payload.manifest_path),
            "observed": str(observed_payload.manifest_path),
        },
    }
    (output / "closure.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
