#!/usr/bin/env python3
"""Apply one truth-fixed local multi-DoF update from real refit responses.

This is the physical identifiability/closure control for the iterative loop.
The anchor, every central finite-difference probe, and the target point must
already be separate completed ``/Tracker/Align -> segment refit -> mode-0
Acts`` outputs in one frozen multi-DoF scan plan.  Truth is used only to keep
MC pair identities fixed while verifying local geometry recovery; the later
route-selected update consumes the same mathematical interface without truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from alignment.physical_jacobian import (
    parameter_values_from_station_transforms,
    solve_physical_finite_difference,
    station_transforms_with_parameter_values,
)
from scripts.run_refit_multidof_closure import (
    RESIDUAL_LABELS,
    _aligned_rows,
    _evaluation,
    _payload_for_point,
    _point_map,
    _read_json,
)


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _name_values(values: Sequence[str] | None, names: Sequence[str], *, label: str) -> np.ndarray | None:
    if not values:
        return None
    parsed: dict[str, float] = {}
    for raw in values:
        name, separator, raw_value = str(raw).partition(":")
        if not separator or not name:
            raise ValueError(f"{label} entries must have form PARAMETER:VALUE")
        if name in parsed:
            raise ValueError(f"{label} repeats parameter '{name}'")
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{label} values must be finite and non-negative")
        parsed[name] = value
    if set(parsed) != set(names):
        raise ValueError(f"{label} must specify exactly: " + ", ".join(names))
    return np.asarray([parsed[name] for name in names], dtype=np.float64)


def _probe(points: Mapping[str, Mapping[str, object]], parameter: str, sign: str, anchor: str) -> Mapping[str, object]:
    matches = [
        point
        for point in points.values()
        if point.get("finite_difference_for") == parameter
        and point.get("probe_sign") == sign
        and point.get("finite_difference_anchor") == anchor
    ]
    if len(matches) != 1:
        raise ValueError(
            f"scan plan requires exactly one {sign} finite-difference probe for '{parameter}' around '{anchor}'"
        )
    return matches[0]


def _write_csv(path: Path, rows):
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fit_diagnostic(
    fit,
    names: Sequence[str],
    *,
    observations: int,
) -> dict[str, object]:
    """Compact local-response solve diagnostics for a provenance subset."""
    return {
        "observations": int(observations),
        "normal_matrix_rank": int(fit.normal_matrix_rank),
        "fit_matrix_rank": int(fit.fit_matrix_rank),
        "normal_matrix_condition_number": fit.normal_matrix_condition_number,
        "identifiable_subspace_condition_number": fit.identifiable_subspace_condition_number,
        "recovered_local_delta": {
            name: float(fit.recovered_parameters[index]) for index, name in enumerate(names)
        },
        "recovered_sigma": {
            name: (
                math.sqrt(float(fit.covariance_native[index, index]))
                if math.isfinite(float(fit.covariance_native[index, index]))
                and float(fit.covariance_native[index, index]) >= 0.0
                else None
            )
            for index, name in enumerate(names)
        },
    }


def _subset_fit_diagnostic(
    mask: np.ndarray,
    *,
    anchor_residual: np.ndarray,
    positive_residual: np.ndarray,
    negative_residual: np.ndarray,
    target_residual: np.ndarray,
    covariance: np.ndarray,
    names: Sequence[str],
    positive_values: Sequence[float],
    negative_values: Sequence[float],
    scales: np.ndarray,
    rcond: float,
) -> dict[str, object]:
    """Solve a station-pair or leave-one-event subset without failing audit."""
    selected = np.asarray(mask, dtype=bool)
    count = int(np.count_nonzero(selected))
    if count < 1:
        return {"observations": 0, "status": "empty"}
    try:
        fit = solve_physical_finite_difference(
            anchor_residual[selected],
            positive_residual[:, selected],
            negative_residual[:, selected],
            target_residual[selected],
            covariance[selected],
            parameter_names=names,
            positive_values=positive_values,
            negative_values=negative_values,
            parameter_scales=scales,
            rcond=rcond,
        )
    except ValueError as error:
        return {"observations": count, "status": "invalid", "error": str(error)}
    return {"status": "ok", **_fit_diagnostic(fit, names, observations=count)}


def _curvature_summary(
    curvature: np.ndarray,
    covariance: np.ndarray,
    names: Sequence[str],
) -> list[dict[str, object]]:
    """Report central-difference asymmetry from independently refitted probes.

    The values are diagnostics, not a statistical hypothesis test: plus and
    minus refits share the same cluster input.  Large curvature flags a local
    step that should be damped or supported by more physical events.
    """
    result: list[dict[str, object]] = []
    for parameter_index, name in enumerate(names):
        values = np.asarray(curvature[parameter_index], dtype=np.float64)
        weighted = 0.0
        for residual, matrix in zip(values, covariance):
            weighted += float(residual @ np.linalg.solve(matrix, residual))
        result.append(
            {
                "name": str(name),
                "observations": int(values.shape[0]),
                "deterministic_weighted_curvature": weighted,
                "rms_by_residual_component": {
                    label: float(np.sqrt(np.mean(np.square(values[:, index]))))
                    for index, label in enumerate(RESIDUAL_LABELS)
                },
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", required=True)
    parser.add_argument("--anchor-point", required=True)
    parser.add_argument("--target-point", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-truth-match-fraction", type=float, default=0.99)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--prior-sigma", action="append", default=None, metavar="PARAMETER:VALUE")
    parser.add_argument("--capture-tolerance", action="append", default=None, metavar="PARAMETER:VALUE")
    parser.add_argument("--rcond", type=float, default=1.0e-10)
    parser.add_argument("--require-full-rank", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.min_truth_match_fraction) or not 0.0 <= args.min_truth_match_fraction <= 1.0:
        parser.error("--min-truth-match-fraction must be in [0, 1]")
    if not math.isfinite(args.damping) or not 0.0 < args.damping <= 1.0:
        parser.error("--damping must lie in (0, 1]")
    if not math.isfinite(args.rcond) or not 0.0 < args.rcond < 1.0:
        parser.error("--rcond must lie in (0, 1)")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    scan_root = Path(args.scan_root).expanduser().resolve()
    plan = _read_json(scan_root / "scan_plan.json")
    if plan.get("scan_mode") != "station_rigid_multidof" or int(plan.get("q_over_p_mode", -1)) != 0:
        raise ValueError("local multi-DoF step requires a mode-0 station_rigid_multidof scan")
    raw_specs = plan.get("alignment_parameter_specs")
    if not isinstance(raw_specs, list) or not raw_specs or any(not isinstance(item, Mapping) for item in raw_specs):
        raise ValueError("multi-DoF scan lacks valid alignment_parameter_specs")
    specs = [dict(item) for item in raw_specs]
    names = tuple(str(spec.get("name", "")) for spec in specs)
    if not all(names) or len(set(names)) != len(names):
        raise ValueError("multi-DoF scan has invalid parameter names")
    scales = np.asarray([float(spec["severity_scale"]) for spec in specs], dtype=np.float64)
    movable = {int(station) for station in plan.get("movable_station_ids", ())}
    if not movable:
        raise ValueError("multi-DoF scan has no movable station")
    points = _point_map(plan)
    anchor_point = points.get(str(args.anchor_point))
    target_point = points.get(str(args.target_point))
    if anchor_point is None or target_point is None:
        raise ValueError("anchor/target point is absent from scan plan")
    if anchor_point.get("point_role") == "nominal":
        raise ValueError("anchor point must be a non-reference physical payload")
    positive_points = {name: _probe(points, name, "positive", str(args.anchor_point)) for name in names}
    negative_points = {name: _probe(points, name, "negative", str(args.anchor_point)) for name in names}
    ordered_points = [anchor_point]
    for name in names:
        ordered_points.extend((positive_points[name], negative_points[name]))
    ordered_points.append(target_point)
    payloads = []
    evaluations = []
    for point in ordered_points:
        payload, tracklets, propagations = _payload_for_point(scan_root, point)
        payloads.append(payload)
        evaluations.append(_evaluation(tracklets, propagations, float(args.min_truth_match_fraction)))
    keys, indexes, overlap = _aligned_rows(evaluations, movable)
    rows_by_evaluation = [np.asarray([index[key] for key in keys], dtype=np.intp) for index in indexes]
    anchor_rows = rows_by_evaluation[0]
    anchor_residual = np.asarray(evaluations[0].residual[anchor_rows], dtype=np.float64)
    anchor_covariance = np.asarray(evaluations[0].combined_covariance[anchor_rows], dtype=np.float64)
    positive_residual = np.asarray(
        [
            evaluations[1 + 2 * index].residual[rows_by_evaluation[1 + 2 * index]]
            for index in range(len(names))
        ],
        dtype=np.float64,
    )
    negative_residual = np.asarray(
        [
            evaluations[2 + 2 * index].residual[rows_by_evaluation[2 + 2 * index]]
            for index in range(len(names))
        ],
        dtype=np.float64,
    )
    target_residual = np.asarray(evaluations[-1].residual[rows_by_evaluation[-1]], dtype=np.float64)
    anchor_values = parameter_values_from_station_transforms(specs, payloads[0].transforms)
    target_values = parameter_values_from_station_transforms(specs, payloads[-1].transforms)
    positive_values = [
        parameter_values_from_station_transforms(specs, payloads[1 + 2 * index].transforms)[name]
        for index, name in enumerate(names)
    ]
    negative_values = [
        parameter_values_from_station_transforms(specs, payloads[2 + 2 * index].transforms)[name]
        for index, name in enumerate(names)
    ]
    prior = _name_values(args.prior_sigma, names, label="--prior-sigma")
    tolerance = _name_values(args.capture_tolerance, names, label="--capture-tolerance")
    fit = solve_physical_finite_difference(
        anchor_residual,
        positive_residual,
        negative_residual,
        target_residual,
        anchor_covariance,
        parameter_names=names,
        positive_values=positive_values,
        negative_values=negative_values,
        parameter_scales=scales,
        prior_sigma_native=prior,
        rcond=float(args.rcond),
    )
    if args.require_full_rank and not fit.full_rank:
        raise RuntimeError("local physical multi-DoF normal matrix is rank deficient")
    current = np.asarray([anchor_values[name] for name in names], dtype=np.float64)
    target = np.asarray([target_values[name] for name in names], dtype=np.float64)
    expected_delta = target - current
    proposal = current + float(args.damping) * fit.recovered_parameters
    proposal_values = {name: float(proposal[index]) for index, name in enumerate(names)}
    proposed_transforms = station_transforms_with_parameter_values(specs, payloads[0].transforms, proposal_values)
    errors = fit.recovered_parameters - expected_delta
    curvature = positive_residual + negative_residual - 2.0 * anchor_residual[np.newaxis, ...]
    source_station = np.asarray(evaluations[0].source_station_id[anchor_rows], dtype=np.int64)
    target_station = np.asarray(evaluations[0].target_station_id[anchor_rows], dtype=np.int64)
    run_ids = np.asarray(evaluations[0].run_id[anchor_rows], dtype=np.int64)
    event_ids = np.asarray(evaluations[0].event_id[anchor_rows], dtype=np.int64)
    station_pair_diagnostics = []
    for source, target_station_id in sorted(set(zip(source_station.tolist(), target_station.tolist()))):
        mask = (source_station == source) & (target_station == target_station_id)
        station_pair_diagnostics.append(
            {
                "source_station_id": int(source),
                "target_station_id": int(target_station_id),
                **_subset_fit_diagnostic(
                    mask,
                    anchor_residual=anchor_residual,
                    positive_residual=positive_residual,
                    negative_residual=negative_residual,
                    target_residual=target_residual,
                    covariance=anchor_covariance,
                    names=names,
                    positive_values=positive_values,
                    negative_values=negative_values,
                    scales=scales,
                    rcond=float(args.rcond),
                ),
            }
        )
    leave_one_event_out = []
    for run_id, event_id in sorted(set(zip(run_ids.tolist(), event_ids.tolist()))):
        mask = (run_ids != run_id) | (event_ids != event_id)
        leave_one_event_out.append(
            {
                "withheld_run_id": int(run_id),
                "withheld_event_id": int(event_id),
                "withheld_observations": int(np.count_nonzero(~mask)),
                **_subset_fit_diagnostic(
                    mask,
                    anchor_residual=anchor_residual,
                    positive_residual=positive_residual,
                    negative_residual=negative_residual,
                    target_residual=target_residual,
                    covariance=anchor_covariance,
                    names=names,
                    positive_values=positive_values,
                    negative_values=negative_values,
                    scales=scales,
                    rcond=float(args.rcond),
                ),
            }
        )
    parameter_rows = []
    for index, (name, spec) in enumerate(zip(names, specs)):
        variance = float(fit.covariance_native[index, index])
        sigma = math.sqrt(variance) if math.isfinite(variance) and variance >= 0.0 else None
        parameter_rows.append(
            {
                "name": name,
                "station_id": int(spec["station_id"]),
                "component": str(spec["component"]),
                "unit": str(spec["unit"]),
                "anchor_value": float(current[index]),
                "target_value": float(target[index]),
                "expected_delta_to_target": float(expected_delta[index]),
                "recovered_local_delta": float(fit.recovered_parameters[index]),
                "local_delta_error": float(errors[index]),
                "proposed_next_value": float(proposal[index]),
                "recovered_sigma": sigma,
                "observability_fraction": float(fit.parameter_observability_fraction[index]),
                "capture_tolerance": None if tolerance is None else float(tolerance[index]),
                "capture_success": None if tolerance is None else bool(abs(errors[index]) <= tolerance[index]),
            }
        )
    response_rows = []
    for row, key in enumerate(keys):
        anchor_index = int(anchor_rows[row])
        entry = {
            "run_id": key[0],
            "event_id": key[1],
            "source_tracklet_id": key[2],
            "target_tracklet_id": key[3],
            "q_over_p_mode": key[4],
            "source_station_id": int(evaluations[0].source_station_id[anchor_index]),
            "target_station_id": int(evaluations[0].target_station_id[anchor_index]),
        }
        for residual_index, label in enumerate(RESIDUAL_LABELS):
            entry[f"anchor_residual_{label}"] = float(anchor_residual[row, residual_index])
            entry[f"target_residual_{label}"] = float(target_residual[row, residual_index])
            entry[f"response_{label}"] = float(fit.response[row, residual_index])
            entry[f"post_update_response_{label}"] = float(fit.residual_response[row, residual_index])
            for parameter_index, name in enumerate(names):
                entry[f"derivative_{name}_{label}"] = float(fit.derivative_native[row, residual_index, parameter_index])
        response_rows.append(entry)
    _write_csv(output / "local_step_pairs.csv", response_rows)
    np.savez_compressed(
        output / "local_step_arrays.npz",
        parameter_names=np.asarray(names),
        anchor_residual=anchor_residual,
        positive_residual=positive_residual,
        negative_residual=negative_residual,
        target_residual=target_residual,
        covariance=anchor_covariance,
        central_difference_curvature=curvature,
        derivative_native=fit.derivative_native,
        response=fit.response,
        predicted_response=fit.predicted_response,
        residual_response=fit.residual_response,
        normal_matrix_native=fit.normal_matrix_native,
        normal_matrix_scaled=fit.normal_matrix_scaled,
        covariance_native=fit.covariance_native,
        correlation_native=fit.correlation_native,
    )
    capture_success = None if tolerance is None else bool(all(row["capture_success"] for row in parameter_rows))
    summary = {
        "method": "truth_fixed_local_physical_multidof_finite_difference_step",
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "q_over_p_mode": 0,
        "reference_station_ids": list(plan["reference_station_ids"]),
        "movable_station_ids": list(plan["movable_station_ids"]),
        "anchor_point": str(args.anchor_point),
        "target_point": str(args.target_point),
        "damping": float(args.damping),
        "update_semantics": (
            "Local physical response from anchor to independently refitted target; the proposal is anchor + damping * "
            "recovered_local_delta in the persisted payload convention."
        ),
        "truth_pair_overlap": overlap,
        "used_truth_pairs": len(keys),
        "normal_matrix_rank": int(fit.normal_matrix_rank),
        "fit_matrix_rank_after_prior": int(fit.fit_matrix_rank),
        "normal_matrix_condition_number": fit.normal_matrix_condition_number,
        "identifiable_subspace_condition_number": fit.identifiable_subspace_condition_number,
        "parameter_covariance": fit.covariance_native,
        "parameter_correlation": fit.correlation_native,
        "response_chi2": float(fit.response_chi2),
        "response_ndof": int(fit.response_ndof),
        "finite_difference_linearity": {
            "interpretation": (
                "Central-probe curvature from independently refitted physical payloads. It is deterministic "
                "response asymmetry, not an independent-refit statistical test."
            ),
            "by_parameter": _curvature_summary(curvature, anchor_covariance, names),
        },
        "station_pair_local_fits": station_pair_diagnostics,
        "leave_one_event_out_local_fits": leave_one_event_out,
        "parameters": parameter_rows,
        "proposed_next_parameter_values": proposal_values,
        "proposed_next_station_transforms": proposed_transforms,
        "capture_success": capture_success,
        "payloads": {
            "anchor": str(payloads[0].manifest_path),
            "target": str(payloads[-1].manifest_path),
            "positive": {name: str(payloads[1 + 2 * index].manifest_path) for index, name in enumerate(names)},
            "negative": {name: str(payloads[2 + 2 * index].manifest_path) for index, name in enumerate(names)},
        },
    }
    (output / "local_step.json").write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            _json_ready(
                {
                    "output_dir": str(output),
                    "used_truth_pairs": len(keys),
                    "normal_matrix_rank": fit.normal_matrix_rank,
                    "normal_matrix_condition_number": fit.normal_matrix_condition_number,
                    "proposed_next_parameter_values": proposal_values,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
