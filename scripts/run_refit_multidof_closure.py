#!/usr/bin/env python3
"""Truth-fixed multi-DoF physical closure from a rigid refit scan bank.

This consumes a frozen ``station_rigid_multidof`` scan whose nominal,
central-difference, and observed points each executed the real
``/Tracker/Align -> SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit
-> NtupleDumper -> FaserActsExtrapolationTool(mode 0)`` chain.  Truth is used
only to hold pair identities fixed while auditing alignment identifiability;
no tracklet coordinate or residual-level injection is constructed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from alignment.payload import load_station_rigid_alignment_payload
from alignment.physical_jacobian import (
    parameter_values_from_station_transforms,
    solve_physical_finite_difference,
)
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events
from evaluation.field_propagation import evaluate_field_propagation


RESIDUAL_LABELS = ("rx_mm", "ry_mm", "rtx", "rty")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact is not a mapping: {path}")
    return dict(payload)


def _point_map(plan: Mapping[str, object]) -> dict[str, dict[str, object]]:
    raw_points = plan.get("points")
    if not isinstance(raw_points, list) or not raw_points:
        raise ValueError("physical scan plan has no points")
    result: dict[str, dict[str, object]] = {}
    for raw in raw_points:
        if not isinstance(raw, Mapping):
            raise ValueError("physical scan point is not a mapping")
        name = str(raw.get("name", ""))
        if not name or name in result:
            raise ValueError("physical scan point names must be unique")
        result[name] = dict(raw)
    return result


def _payload_for_point(scan_root: Path, point: Mapping[str, object]):
    relative = point.get("relative_point_dir")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"physical point '{point.get('name')}' lacks relative_point_dir")
    point_root = scan_root / relative
    payload_path = point_root / "payload" / "alignment_payload.json"
    tracklets = point_root / "refit" / "tracklets.root"
    propagations = point_root / "refit" / "propagations.root"
    for artifact in (payload_path, tracklets, propagations):
        if not artifact.is_file():
            raise FileNotFoundError(f"physical point '{point.get('name')}' is incomplete: {artifact}")
    payload = load_station_rigid_alignment_payload(payload_path)
    expected = point.get("injected_station_transforms")
    if not isinstance(expected, Mapping):
        raise ValueError(f"physical point '{point.get('name')}' lacks planned station transforms")
    for raw_station, raw_transform in expected.items():
        station = int(raw_station)
        transformed = np.asarray(payload.transform_for_station(station), dtype=np.float64)
        planned = np.asarray(raw_transform, dtype=np.float64)
        if planned.shape != (6,) or not np.isfinite(planned).all() or not np.allclose(
            transformed, planned, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(f"physical point '{point.get('name')}' payload does not match its frozen plan")
    return payload, tracklets, propagations


def _evaluation(tracklets: Path, propagations: Path, fraction: float):
    return evaluate_field_propagation(
        load_events(tracklets, require_mc_labels=True),
        load_propagation_records(propagations),
        require_truth_match=True,
        q_over_p_mode=0,
        min_truth_match_fraction=fraction,
    )


def _pair_key(evaluation, row: int) -> tuple[int, int, int, int, int]:
    return (
        int(evaluation.run_id[row]),
        int(evaluation.event_id[row]),
        int(evaluation.source_tracklet_id[row]),
        int(evaluation.target_tracklet_id[row]),
        int(evaluation.q_over_p_mode[row]),
    )


def _index(evaluation) -> dict[tuple[int, int, int, int, int], int]:
    result = {_pair_key(evaluation, row): row for row in range(evaluation.size)}
    if len(result) != evaluation.size:
        raise ValueError("physical multi-DoF closure has duplicate truth-pair identities")
    return result


def _aligned_rows(evaluations: Sequence[object], movable_stations: set[int]):
    indexes = [_index(evaluation) for evaluation in evaluations]
    common = set.intersection(*(set(index) for index in indexes))
    union = set.union(*(set(index) for index in indexes))
    selected = []
    for key in sorted(common):
        rows = [index[key] for index in indexes]
        source = int(evaluations[0].source_station_id[rows[0]])
        target = int(evaluations[0].target_station_id[rows[0]])
        if source not in movable_stations and target not in movable_stations:
            continue
        if any(
            int(evaluation.source_station_id[row]) != source
            or int(evaluation.target_station_id[row]) != target
            for evaluation, row in zip(evaluations[1:], rows[1:])
        ):
            raise ValueError(f"truth pair {key} changes station-pair identity across physical refits")
        selected.append(key)
    if not selected:
        raise ValueError("no common truth propagation pair involving a movable station survives closure")
    return selected, indexes, {
        "all_common_truth_pairs": len(common),
        "all_union_truth_pairs": len(union),
        "non_common_truth_pairs": len(union - common),
        "selected_pairs_involving_movable_station": len(selected),
    }


def _name_value_pairs(values: Sequence[str] | None, names: Sequence[str], *, label: str) -> np.ndarray | None:
    if not values:
        return None
    parsed: dict[str, float] = {}
    for raw in values:
        key, separator, raw_value = str(raw).partition(":")
        if not separator or not key:
            raise ValueError(f"{label} entries must have form PARAMETER:VALUE")
        if key in parsed:
            raise ValueError(f"{label} repeats parameter '{key}'")
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{label} value for '{key}' must be finite and non-negative")
        parsed[key] = value
    if set(parsed) != set(names):
        raise ValueError(f"{label} must specify exactly {list(names)}")
    return np.asarray([parsed[name] for name in names], dtype=np.float64)


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


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", required=True)
    parser.add_argument("--observed-point", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-truth-match-fraction", type=float, default=0.99)
    parser.add_argument("--prior-sigma", action="append", default=None, metavar="PARAMETER:VALUE")
    parser.add_argument("--capture-tolerance", action="append", default=None, metavar="PARAMETER:VALUE")
    parser.add_argument("--rcond", type=float, default=1.0e-10)
    parser.add_argument("--require-full-rank", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.min_truth_match_fraction) or not 0.0 <= args.min_truth_match_fraction <= 1.0:
        parser.error("--min-truth-match-fraction must be in [0, 1]")
    if not np.isfinite(args.rcond) or not 0.0 < args.rcond < 1.0:
        parser.error("--rcond must be in (0, 1)")
    scan_root = Path(args.scan_root).expanduser().resolve()
    plan = _read_json(scan_root / "scan_plan.json")
    if plan.get("scan_mode") != "station_rigid_multidof":
        raise ValueError("multi-DoF closure requires a station_rigid_multidof physical scan")
    if int(plan.get("q_over_p_mode", -1)) != 0:
        raise ValueError("multi-DoF closure requires mode-0 Acts propagation")
    raw_specs = plan.get("alignment_parameter_specs")
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("multi-DoF scan plan lacks alignment_parameter_specs")
    specs = [dict(spec) for spec in raw_specs if isinstance(spec, Mapping)]
    if len(specs) != len(raw_specs):
        raise ValueError("multi-DoF scan parameter spec must be a mapping")
    names = tuple(str(spec.get("name", "")) for spec in specs)
    if not all(names) or len(set(names)) != len(names):
        raise ValueError("multi-DoF scan has invalid parameter names")
    scales = np.asarray([float(spec["severity_scale"]) for spec in specs], dtype=np.float64)
    if not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise ValueError("multi-DoF scan has invalid parameter severity scales")
    movable = {int(station) for station in plan.get("movable_station_ids", ())}
    reference = {int(station) for station in plan.get("reference_station_ids", ())}
    if not movable or not reference or movable.intersection(reference):
        raise ValueError("multi-DoF scan has invalid reference/movable station definition")
    points = _point_map(plan)
    nominal = [point for point in points.values() if point.get("point_role") == "nominal"]
    if len(nominal) != 1:
        raise ValueError("multi-DoF scan must have exactly one nominal point")
    observed = points.get(str(args.observed_point))
    if observed is None:
        raise ValueError(f"observed point is not present in frozen scan plan: {args.observed_point}")
    if str(observed.get("point_role")) in {"nominal", "finite_difference_positive", "finite_difference_negative"}:
        raise ValueError("observed point must be an independent joint curriculum/closure payload")
    probes: dict[str, dict[str, Mapping[str, object]]] = {name: {} for name in names}
    for point in points.values():
        parameter = point.get("finite_difference_for")
        sign = point.get("probe_sign")
        if parameter is None and sign is None:
            continue
        if str(parameter) not in probes or str(sign) not in {"positive", "negative"}:
            raise ValueError("multi-DoF scan has invalid finite-difference point tags")
        holder = probes[str(parameter)]
        if str(sign) in holder:
            raise ValueError(f"multi-DoF scan has duplicate {sign} probe for '{parameter}'")
        holder[str(sign)] = point
    missing = [name for name, values in probes.items() if set(values) != {"positive", "negative"}]
    if missing:
        raise ValueError("multi-DoF scan lacks finite-difference probes for: " + ", ".join(missing))

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    ordered_points: list[Mapping[str, object]] = [nominal[0]]
    for name in names:
        ordered_points.extend((probes[name]["positive"], probes[name]["negative"]))
    ordered_points.append(observed)
    payloads = []
    evaluations = []
    for point in ordered_points:
        payload, tracklets, propagations = _payload_for_point(scan_root, point)
        payloads.append(payload)
        evaluations.append(_evaluation(tracklets, propagations, args.min_truth_match_fraction))
    keys, indexes, overlap = _aligned_rows(evaluations, movable)
    rows_by_evaluation = [np.asarray([index[key] for key in keys], dtype=np.intp) for index in indexes]
    nominal_evaluation = evaluations[0]
    nominal_rows = rows_by_evaluation[0]
    nominal_residual = np.asarray(nominal_evaluation.residual[nominal_rows], dtype=np.float64)
    nominal_covariance = np.asarray(nominal_evaluation.combined_covariance[nominal_rows], dtype=np.float64)
    positive_rows = []
    negative_rows = []
    positive_values = []
    negative_values = []
    for index, name in enumerate(names):
        positive_point_index = 1 + 2 * index
        negative_point_index = positive_point_index + 1
        positive_rows.append(
            np.asarray(evaluations[positive_point_index].residual[rows_by_evaluation[positive_point_index]], dtype=np.float64)
        )
        negative_rows.append(
            np.asarray(evaluations[negative_point_index].residual[rows_by_evaluation[negative_point_index]], dtype=np.float64)
        )
        plus_values = parameter_values_from_station_transforms(specs, payloads[positive_point_index].transforms)
        minus_values = parameter_values_from_station_transforms(specs, payloads[negative_point_index].transforms)
        positive_values.append(float(plus_values[name]))
        negative_values.append(float(minus_values[name]))
    observed_index = len(evaluations) - 1
    observed_residual = np.asarray(evaluations[observed_index].residual[rows_by_evaluation[observed_index]], dtype=np.float64)
    injected_values = parameter_values_from_station_transforms(specs, payloads[observed_index].transforms)
    prior = _name_value_pairs(args.prior_sigma, names, label="--prior-sigma")
    tolerance = _name_value_pairs(args.capture_tolerance, names, label="--capture-tolerance")
    fit = solve_physical_finite_difference(
        nominal_residual,
        np.asarray(positive_rows, dtype=np.float64),
        np.asarray(negative_rows, dtype=np.float64),
        observed_residual,
        nominal_covariance,
        parameter_names=names,
        positive_values=positive_values,
        negative_values=negative_values,
        parameter_scales=scales,
        prior_sigma_native=prior,
        rcond=float(args.rcond),
    )
    errors = fit.recovered_parameters - np.asarray([injected_values[name] for name in names], dtype=np.float64)
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
                "severity_scale": float(spec["severity_scale"]),
                "finite_difference_step": float(spec["finite_difference_step"]),
                "injected": float(injected_values[name]),
                "recovered": float(fit.recovered_parameters[index]),
                "signed_recovery_error": float(errors[index]),
                "absolute_recovery_error": float(abs(errors[index])),
                "recovered_sigma": sigma,
                "observability_fraction": float(fit.parameter_observability_fraction[index]),
                "capture_tolerance": None if tolerance is None else float(tolerance[index]),
                "capture_success": None if tolerance is None else bool(abs(errors[index]) <= tolerance[index]),
            }
        )
    closure_rows = []
    for row, key in enumerate(keys):
        source_row = int(nominal_rows[row])
        entry: dict[str, object] = {
            "run_id": key[0],
            "event_id": key[1],
            "source_tracklet_id": key[2],
            "target_tracklet_id": key[3],
            "q_over_p_mode": key[4],
            "source_station_id": int(nominal_evaluation.source_station_id[source_row]),
            "target_station_id": int(nominal_evaluation.target_station_id[source_row]),
        }
        for residual_index, label in enumerate(RESIDUAL_LABELS):
            entry[f"observed_response_{label}"] = float(fit.response[row, residual_index])
            entry[f"fitted_response_{label}"] = float(fit.predicted_response[row, residual_index])
            entry[f"postfit_response_{label}"] = float(fit.residual_response[row, residual_index])
            for parameter_index, name in enumerate(names):
                entry[f"derivative_{name}_{label}"] = float(fit.derivative_native[row, residual_index, parameter_index])
        closure_rows.append(entry)
    _write_csv(output / "closure_pairs.csv", closure_rows)
    np.savez_compressed(
        output / "closure_arrays.npz",
        parameter_names=np.asarray(names),
        nominal_residual=nominal_residual,
        observed_residual=observed_residual,
        covariance=nominal_covariance,
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
        "method": "truth_fixed_physical_multidof_finite_difference_closure",
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "q_over_p_mode": 0,
        "refit_chain": (
            "persisted SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit -> "
            "NtupleDumper -> FaserActsExtrapolationTool"
        ),
        "association": "truth_fixed_for_identifiability_audit",
        "fit_weight_interpretation": (
            "Nominal combined covariance is a deterministic WLS weight for paired physical refit "
            "responses; it is not an independent covariance of a difference of refits of the same clusters."
        ),
        "scan_root": str(scan_root),
        "observed_point": str(observed["name"]),
        "condition_axis": str(plan["condition_axis"]),
        "reference_station_ids": sorted(reference),
        "movable_station_ids": sorted(movable),
        "normal_matrix_rank": int(fit.normal_matrix_rank),
        "fit_matrix_rank_after_prior": int(fit.fit_matrix_rank),
        "normal_matrix_condition_number": fit.normal_matrix_condition_number,
        "identifiable_subspace_condition_number": fit.identifiable_subspace_condition_number,
        "normal_matrix_condition_units": "dimensionless theta/parameter_severity_scale",
        "normal_matrix_native": fit.normal_matrix_native,
        "normal_matrix_scaled": fit.normal_matrix_scaled,
        "right_hand_side_native": fit.right_hand_side_native,
        "right_hand_side_scaled": fit.right_hand_side_scaled,
        "normal_matrix_singular_values": fit.data_singular_values,
        "fit_matrix_singular_values": fit.fit_singular_values,
        "full_rank": fit.full_rank,
        "response_chi2": fit.response_chi2,
        "response_ndof": fit.response_ndof,
        "used_truth_pairs": fit.used_pairs,
        "truth_pair_overlap": overlap,
        "prior_sigma_native": fit.prior_sigma_native,
        "capture_success": capture_success,
        "parameters": parameter_rows,
        "parameter_covariance_native": fit.covariance_native,
        "parameter_correlation_native": fit.correlation_native,
        "payloads": {
            "nominal": str(payloads[0].manifest_path),
            "positive_probes": {
                name: str(payloads[1 + 2 * index].manifest_path) for index, name in enumerate(names)
            },
            "negative_probes": {
                name: str(payloads[2 + 2 * index].manifest_path) for index, name in enumerate(names)
            },
            "observed": str(payloads[observed_index].manifest_path),
        },
    }
    (output / "closure.json").write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True, allow_nan=False))
    if args.require_full_rank and not fit.full_rank:
        raise SystemExit("multi-DoF physical closure normal matrix is rank deficient")


if __name__ == "__main__":
    main()
