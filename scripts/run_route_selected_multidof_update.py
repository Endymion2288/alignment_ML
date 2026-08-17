#!/usr/bin/env python3
"""Solve one truth-free, route-selected physical multi-DoF alignment update.

Inputs are outputs of the frozen V1/V2 association backbone for an anchor
payload, its real central finite-difference probes, and a separately refitted
reference target.  The selected routes are intersected only by original
tracklet provenance.  By default the update consumes exact field-aware mode-0
ACTS residuals carried by selected route edges.  The older straight-line
leave-one-out table remains available only as an explicit diagnostic option;
it is not the default alignment objective in a magnetic field.

The default target is intended for an MC closure scan whose nominal payload is
known.  It is deliberately not labelled as a real-data correction: a data
alignment loop needs a collaboration-validated residual target and conditions
sign convention before the proposed payload may be deployed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from alignment.physical_jacobian import (
    parameter_values_from_station_transforms,
    solve_physical_finite_difference,
    station_transforms_with_parameter_values,
)
from alignment.route_selected_update import (
    align_route_selected_observations,
    read_route_selected_field_edge_observations,
    read_route_selected_observations,
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact is not a mapping: {path}")
    return dict(payload)


def _points(plan: Mapping[str, object]) -> dict[str, dict[str, object]]:
    raw = plan.get("points")
    if not isinstance(raw, list) or not raw:
        raise ValueError("physical scan plan has no points")
    result: dict[str, dict[str, object]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("physical scan point is not a mapping")
        name = str(item.get("name", ""))
        if not name or name in result:
            raise ValueError("physical scan has invalid/duplicate point names")
        result[name] = dict(item)
    return result


def _specs(plan: Mapping[str, object]) -> list[dict[str, object]]:
    raw = plan.get("alignment_parameter_specs")
    if not isinstance(raw, list) or not raw:
        raise ValueError("physical scan lacks alignment_parameter_specs")
    result = [dict(item) for item in raw if isinstance(item, Mapping)]
    if len(result) != len(raw):
        raise ValueError("alignment_parameter_specs entries must be mappings")
    names = [str(item.get("name", "")) for item in result]
    if not all(names) or len(set(names)) != len(names):
        raise ValueError("alignment_parameter_specs has invalid names")
    return result


def _parameter_paths(values: Sequence[str] | None, names: Sequence[str], *, label: str) -> dict[str, Path]:
    if not values:
        raise ValueError(f"{label} must specify exactly one PATH for every parameter")
    result: dict[str, Path] = {}
    for raw in values:
        name, separator, raw_path = str(raw).partition(":")
        if not separator or not name or not raw_path:
            raise ValueError(f"{label} entries must have form PARAMETER:PATH")
        if name in result:
            raise ValueError(f"{label} repeats parameter '{name}'")
        result[name] = Path(raw_path).expanduser().resolve()
    if set(result) != set(names):
        raise ValueError(f"{label} must specify exactly: " + ", ".join(names))
    return result


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
            raise ValueError(f"{label} parameter '{name}' must be finite and non-negative")
        parsed[name] = value
    if set(parsed) != set(names):
        raise ValueError(f"{label} must specify exactly: " + ", ".join(names))
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
    fields = sorted({str(key) for row in rows for key in row}) or ["empty"]
    data = list(rows) if rows else [{"empty": ""}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def _validate_frozen_output(
    path: Path,
    *,
    expected_point: str,
    movable: Sequence[int],
    observation_kind: str,
):
    summary_path = path / "association_summary.json"
    observation_file = {
        "field_edge": "selected_route_field_edge_residuals.csv",
        "leave_one_out": "selected_route_leave_one_out_residuals.csv",
    }.get(observation_kind)
    if observation_file is None:  # pragma: no cover - argparse constrains this
        raise ValueError(f"unsupported observation kind '{observation_kind}'")
    observation_path = path / observation_file
    if not summary_path.is_file() or not observation_path.is_file():
        raise FileNotFoundError(f"frozen association output is incomplete: {path}")
    summary = _read_json(summary_path)
    if summary.get("test_opened") is not False or summary.get("architecture_or_threshold_tuning") is not False:
        raise ValueError("route-selected update requires a frozen association output sealed from test/tuning")
    if summary.get("physical_geometry_repropagation") is not True or int(summary.get("q_over_p_mode", -1)) != 0:
        raise ValueError("route-selected update requires mode-0 physical association output")
    # This scan's synthetic materialization has exactly one payload per
    # backbone run.  Check the table directly so a point cannot accidentally
    # be paired with association results from another real conditions payload.
    with observation_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        payload_ids = {str(row.get("payload_id", "")) for row in reader if row.get("payload_id")}
    if payload_ids and payload_ids != {expected_point}:
        raise ValueError(
            f"frozen association output {path} has payload IDs {sorted(payload_ids)}, expected '{expected_point}'"
        )
    reader = (
        read_route_selected_field_edge_observations
        if observation_kind == "field_edge"
        else read_route_selected_observations
    )
    return summary, reader(observation_path, movable_station_ids=movable)


def _probe_for_parameter(points: Mapping[str, Mapping[str, object]], name: str, sign: str, anchor: str) -> str:
    matches = [
        point_name
        for point_name, point in points.items()
        if point.get("finite_difference_for") == name
        and point.get("probe_sign") == sign
        and point.get("finite_difference_anchor") == anchor
    ]
    if len(matches) != 1:
        raise ValueError(
            f"scan plan needs exactly one {sign} finite-difference probe for '{name}' around '{anchor}'"
        )
    return matches[0]


def _role_counts(bank) -> dict[str, int]:
    result: dict[str, int] = {}
    for observation in bank.values():
        key = str(observation.target_synthetic_role)
        result[key] = result.get(key, 0) + 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", required=True)
    parser.add_argument("--anchor-point", required=True)
    parser.add_argument("--anchor-association-output", required=True)
    parser.add_argument("--target-point", required=True)
    parser.add_argument("--target-association-output", required=True)
    parser.add_argument("--positive-association-output", action="append", default=None, metavar="PARAMETER:PATH")
    parser.add_argument("--negative-association-output", action="append", default=None, metavar="PARAMETER:PATH")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--observation-kind",
        choices=("field_edge", "leave_one_out"),
        default="field_edge",
        help=(
            "Use exact selected mode-0 ACTS edges (default) or the legacy straight-line leave-one-out "
            "diagnostic table."
        ),
    )
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--prior-sigma", action="append", default=None, metavar="PARAMETER:VALUE")
    parser.add_argument("--capture-tolerance", action="append", default=None, metavar="PARAMETER:VALUE")
    parser.add_argument("--rcond", type=float, default=1.0e-10)
    parser.add_argument("--require-full-rank", action="store_true")
    args = parser.parse_args()
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
        raise ValueError("route-selected update requires a mode-0 station_rigid_multidof physical scan")
    specs = _specs(plan)
    names = [str(spec["name"]) for spec in specs]
    scales = np.asarray([float(spec["severity_scale"]) for spec in specs], dtype=np.float64)
    movable = [int(value) for value in plan.get("movable_station_ids", ())]
    if not movable:
        raise ValueError("physical scan has no movable station")
    points = _points(plan)
    anchor_point = points.get(str(args.anchor_point))
    target_point = points.get(str(args.target_point))
    if anchor_point is None or target_point is None:
        raise ValueError("anchor-point and target-point must be present in the frozen scan plan")
    if anchor_point.get("point_role") == "nominal":
        raise ValueError("anchor-point must be a non-reference physical payload")
    positive_paths = _parameter_paths(args.positive_association_output, names, label="--positive-association-output")
    negative_paths = _parameter_paths(args.negative_association_output, names, label="--negative-association-output")
    positive_points = {
        name: _probe_for_parameter(points, name, "positive", str(args.anchor_point)) for name in names
    }
    negative_points = {
        name: _probe_for_parameter(points, name, "negative", str(args.anchor_point)) for name in names
    }
    anchor_summary, anchor_bank = _validate_frozen_output(
        Path(args.anchor_association_output).expanduser().resolve(),
        expected_point=str(args.anchor_point),
        movable=movable,
        observation_kind=args.observation_kind,
    )
    target_summary, target_bank = _validate_frozen_output(
        Path(args.target_association_output).expanduser().resolve(),
        expected_point=str(args.target_point),
        movable=movable,
        observation_kind=args.observation_kind,
    )
    positive_summaries = {}
    negative_summaries = {}
    positive_banks = {}
    negative_banks = {}
    for name in names:
        positive_summaries[name], positive_banks[name] = _validate_frozen_output(
            positive_paths[name],
            expected_point=positive_points[name],
            movable=movable,
            observation_kind=args.observation_kind,
        )
        negative_summaries[name], negative_banks[name] = _validate_frozen_output(
            negative_paths[name],
            expected_point=negative_points[name],
            movable=movable,
            observation_kind=args.observation_kind,
        )
    # The order is fixed: anchor, p/m for each parameter, then target.  The
    # same provenance intersection is used for every derivative and response.
    banks = [anchor_bank]
    for name in names:
        banks.extend((positive_banks[name], negative_banks[name]))
    banks.append(target_bank)
    keys, residuals, covariances, overlap = align_route_selected_observations(banks)
    anchor_residual = residuals[0]
    anchor_covariance = covariances[0]
    positive_residual = np.asarray([residuals[1 + 2 * index] for index in range(len(names))])
    negative_residual = np.asarray([residuals[2 + 2 * index] for index in range(len(names))])
    target_residual = residuals[-1]
    anchor_values = parameter_values_from_station_transforms(
        specs, anchor_point["injected_station_transforms"]  # type: ignore[arg-type]
    )
    target_values = parameter_values_from_station_transforms(
        specs, target_point["injected_station_transforms"]  # type: ignore[arg-type]
    )
    positive_values = [
        parameter_values_from_station_transforms(specs, points[positive_points[name]]["injected_station_transforms"])[name]
        for name in names
    ]
    negative_values = [
        parameter_values_from_station_transforms(specs, points[negative_points[name]]["injected_station_transforms"])[name]
        for name in names
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
        raise RuntimeError("route-selected physical update normal matrix is rank deficient")
    current = np.asarray([anchor_values[name] for name in names], dtype=np.float64)
    target = np.asarray([target_values[name] for name in names], dtype=np.float64)
    expected_delta = target - current
    recovered_delta = fit.recovered_parameters
    proposed = current + float(args.damping) * recovered_delta
    proposal_values = {name: float(proposed[index]) for index, name in enumerate(names)}
    anchor_transforms = anchor_point.get("injected_station_transforms")
    if not isinstance(anchor_transforms, Mapping):
        raise ValueError("anchor point lacks injected_station_transforms")
    proposed_transforms = station_transforms_with_parameter_values(specs, anchor_transforms, proposal_values)
    delta_error = recovered_delta - expected_delta
    parameter_rows: list[dict[str, object]] = []
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
                "recovered_local_delta": float(recovered_delta[index]),
                "local_delta_error": float(delta_error[index]),
                "proposed_next_value": float(proposed[index]),
                "recovered_sigma": sigma,
                "observability_fraction": float(fit.parameter_observability_fraction[index]),
                "capture_tolerance": None if tolerance is None else float(tolerance[index]),
                "capture_success": (
                    None if tolerance is None else bool(abs(delta_error[index]) <= tolerance[index])
                ),
            }
        )
    observation_rows: list[dict[str, object]] = []
    for row_index, key in enumerate(keys):
        sample, run_id, event_id, signature, *_ = key
        anchor_observation = anchor_bank[key]
        target_observation = target_bank[key]
        entry: dict[str, object] = {
            "sample_id": sample,
            "run_id": run_id,
            "event_id": event_id,
            "route_origin_signature": signature,
            "observation_kind": anchor_observation.observation_kind,
            "source_station_id": anchor_observation.source_station_id,
            "target_station_id": anchor_observation.target_station_id,
            "anchor_source_synthetic_role": anchor_observation.source_synthetic_role,
            "anchor_target_synthetic_role": anchor_observation.target_synthetic_role,
            "target_source_synthetic_role": target_observation.source_synthetic_role,
            "target_target_synthetic_role": target_observation.target_synthetic_role,
        }
        for residual_index, label in enumerate(("x_mm", "y_mm", "tx", "ty")):
            entry[f"anchor_residual_{label}"] = float(anchor_residual[row_index, residual_index])
            entry[f"target_residual_{label}"] = float(target_residual[row_index, residual_index])
            entry[f"response_{label}"] = float(fit.response[row_index, residual_index])
            entry[f"post_update_response_{label}"] = float(fit.residual_response[row_index, residual_index])
            for parameter_index, name in enumerate(names):
                entry[f"derivative_{name}_{label}"] = float(fit.derivative_native[row_index, residual_index, parameter_index])
        observation_rows.append(entry)
    _write_csv(output / "route_selected_update_observations.csv", observation_rows)
    np.savez_compressed(
        output / "route_selected_update_arrays.npz",
        parameter_names=np.asarray(names),
        observation_keys=np.asarray([json.dumps(key) for key in keys]),
        anchor_residual=anchor_residual,
        target_residual=target_residual,
        covariance=anchor_covariance,
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
    summary: dict[str, object] = {
        "method": "truth_free_route_selected_physical_multidof_local_update",
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "q_over_p_mode": 0,
        "test_opened": False,
        "architecture_or_threshold_tuning": False,
        "scan_root": str(scan_root),
        "anchor_point": str(args.anchor_point),
        "target_point": str(args.target_point),
        "probe_points": {name: {"positive": positive_points[name], "negative": negative_points[name]} for name in names},
        "damping": float(args.damping),
        "observation_kind": str(args.observation_kind),
        "observation_contract": (
            "Exact selected mode-0 ACTS edge residual/covariance from the physical candidate graph."
            if args.observation_kind == "field_edge"
            else "Legacy straight-line leave-one-out residual used only as an explicit diagnostic comparison."
        ),
        "update_semantics": (
            "A local physical Newton/Gauss-Newton step in payload coordinates from the anchor residual toward "
            "the independently refitted target residual.  This MC closure update has no assumed manual sign flip."
        ),
        "association_contract": {
            "anchor": anchor_summary,
            "target": target_summary,
            "positive": positive_summaries,
            "negative": negative_summaries,
            "selected_route_overlap": overlap,
            "target_synthetic_role_counts_by_bank": {
                "anchor": _role_counts(anchor_bank),
                "target": _role_counts(target_bank),
                **{f"positive_{name}": _role_counts(positive_banks[name]) for name in names},
                **{f"negative_{name}": _role_counts(negative_banks[name]) for name in names},
            },
        },
        "normal_matrix_rank": int(fit.normal_matrix_rank),
        "fit_matrix_rank": int(fit.fit_matrix_rank),
        "normal_matrix_condition_number": fit.normal_matrix_condition_number,
        "identifiable_subspace_condition_number": fit.identifiable_subspace_condition_number,
        "data_singular_values": fit.data_singular_values,
        "response_chi2": float(fit.response_chi2),
        "response_ndof": int(fit.response_ndof),
        "parameter_covariance": fit.covariance_native,
        "parameter_correlation": fit.correlation_native,
        "parameters": parameter_rows,
        "proposed_next_parameter_values": proposal_values,
        "proposed_next_station_transforms": proposed_transforms,
        "capture_success": capture_success,
    }
    (output / "route_selected_update.json").write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            _json_ready(
                {
                    "output_dir": str(output),
                    "used_selected_observations": len(keys),
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
