#!/usr/bin/env python3
"""IFT station/layer identifiability audit with two explicit gauges.

Reads a completed ``ift_layer_hierarchy`` physical scan (reference + central
FD probes for station common-mode and IFT layer internals) and builds the
truth-selected Jacobian from the real /Tracker/Align -> SegmentFitRefit ->
Acts(mode 0) responses.  No association model and no sealed test are used.

The unconstrained 20-parameter system is reported first so station↔layer
near-degeneracies are visible.  Two gauges then produce comparable physical
solutions:

* station rigid transform carries common mode; coverage-weighted layer
  corrections sum to zero;
* station rigid transform carries common mode; IFT layer 0 is the fixed
  reference plane.

Only gauged parameters that keep a full-rank, source-stable, well-conditioned
sub-block are admitted.  The report then solves that admitted subset against
each held-out physical point (truth-selected linear closure).  It does not
write a new Athena payload or open the sealed test.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from alignment.layer_hierarchy import (
    IFT_LAYER_IDS,
    expand_gauged_parameters,
    hierarchy_degeneracy_pairs,
    hierarchy_index,
    project_to_gauge,
    reduce_gauge,
    split_common_and_internal,
)
from alignment.physical_jacobian import parameter_values_from_payload, solve_physical_finite_difference
from scripts.audit_6dof_identifiability import (
    _column_cosines,
    _complete_route_mask,
    _component_sensitivity,
    _fit_summary,
    _gate,
    _masked_bank,
    _pair_sensitivity,
    _pooled_bank,
    _source_entries,
    _weighted_column_norms,
)
from scripts.run_refit_multidof_closure import (
    _aligned_rows,
    _evaluation,
    _json_ready,
    _point_map,
    _read_json,
)


SCHEMA_VERSION = "faser-ift-layer-hierarchy-identifiability-v1"


def _payload_for_point(scan_root: Path, point: Mapping[str, object]):
    from alignment.payload import load_station_rigid_alignment_payload

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
    expected_station = point.get("injected_station_transforms")
    if not isinstance(expected_station, Mapping):
        raise ValueError(f"physical point '{point.get('name')}' lacks planned station transforms")
    for raw_station, raw_transform in expected_station.items():
        actual = np.asarray(payload.transform_for_station(int(raw_station)), dtype=np.float64)
        planned = np.asarray(raw_transform, dtype=np.float64)
        if planned.shape != (6,) or not np.allclose(actual, planned, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"physical point '{point.get('name')}' station payload mismatch")
    expected_layers = point.get("injected_layer_transforms")
    if not isinstance(expected_layers, Mapping):
        raise ValueError(f"physical point '{point.get('name')}' lacks planned layer transforms")
    for raw_station, raw_layers in expected_layers.items():
        if not isinstance(raw_layers, Mapping):
            raise ValueError(f"physical point '{point.get('name')}' has invalid layer transforms")
        for raw_layer, raw_transform in raw_layers.items():
            actual = np.asarray(payload.transform_for_layer(int(raw_station), int(raw_layer)), dtype=np.float64)
            planned = np.asarray(raw_transform, dtype=np.float64)
            if planned.shape != (6,) or not np.allclose(actual, planned, rtol=0.0, atol=1.0e-12):
                raise ValueError(
                    f"physical point '{point.get('name')}' layer {raw_station}/{raw_layer} payload mismatch"
                )
    return payload, tracklets, propagations


def _ordered_fd_points(
    plan: Mapping[str, object],
    *,
    anchor_point: str,
    target_points: Sequence[str],
    only_parameters: Sequence[str] | None = None,
    require_targets: bool = True,
):
    if plan.get("scan_mode") != "ift_layer_hierarchy" or int(plan.get("q_over_p_mode", -1)) != 0:
        raise ValueError("source scan is not a mode-0 ift_layer_hierarchy physical scan")
    raw_specs = plan.get("alignment_parameter_specs")
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("source scan lacks alignment_parameter_specs")
    specs = tuple(dict(item) for item in raw_specs)
    names = tuple(str(spec["name"]) for spec in specs)
    if only_parameters is not None:
        requested = tuple(str(name) for name in only_parameters)
        unknown = set(requested) - set(names)
        if unknown or len(set(requested)) != len(requested):
            raise ValueError(f"unknown or duplicated parameter selection: {sorted(unknown)}")
        by_name = {str(spec["name"]): spec for spec in specs}
        specs = tuple(by_name[name] for name in requested)
        names = requested
    scales = np.asarray(
        [float(spec["severity_scale"]) for spec in specs],
        dtype=np.float64,
    )
    movable = {int(station) for station in plan.get("movable_station_ids", ())}
    points = _point_map(plan)
    anchor = points.get(anchor_point)
    if anchor is None:
        raise ValueError(f"scan plan lacks anchor '{anchor_point}'")
    if require_targets and not target_points:
        raise ValueError("layer identifiability audit requires at least one held-out target")
    ordered: list[Mapping[str, object]] = [anchor]
    for name in names:
        for sign in ("positive", "negative"):
            matches = [
                point
                for point in points.values()
                if point.get("finite_difference_for") == name
                and point.get("probe_sign") == sign
                and point.get("finite_difference_anchor") == anchor_point
            ]
            if len(matches) != 1:
                raise ValueError(f"scan plan needs exactly one {sign} probe for '{name}'")
            ordered.append(matches[0])
    if not target_points:
        return specs, names, scales, movable, ordered, ()
    targets: list[Mapping[str, object]] = []
    seen_targets: set[str] = set()
    for target_name in target_points:
        if target_name in seen_targets:
            continue
        seen_targets.add(str(target_name))
        target = points.get(target_name)
        if target is None:
            raise ValueError(f"scan plan lacks target '{target_name}'")
        # A finite-difference probe may be reused as a linear self-check target;
        # do not load it twice or it would drop out of the unique ordered list.
        if target is anchor or any(target is point for point in ordered[1:]):
            targets.append(target)
            continue
        targets.append(target)
        ordered.append(target)
    return specs, names, scales, movable, ordered, tuple(str(name) for name in seen_targets)


def _layer_weights_from_hit_pattern(evaluation, rows: np.ndarray) -> np.ndarray:
    """Coverage weights from IFT occupancy bits when the evaluation carries them."""
    hit_pattern = getattr(evaluation, "source_hit_pattern", None)
    if hit_pattern is None:
        return np.full(len(IFT_LAYER_IDS), 1.0 / len(IFT_LAYER_IDS), dtype=np.float64)
    patterns = np.asarray(hit_pattern)[rows]
    counts = np.zeros(len(IFT_LAYER_IDS), dtype=np.float64)
    for pattern in patterns:
        bits = int(pattern)
        for layer in IFT_LAYER_IDS:
            if bits & (1 << (2 * layer)) or bits & (1 << (2 * layer + 1)):
                counts[layer] += 1.0
    if not np.any(counts > 0.0):
        counts[:] = 1.0
    return counts / counts.sum()


def _source_bank(
    entry: Mapping[str, object],
    *,
    anchor_point: str,
    target_points: Sequence[str],
    min_truth_match_fraction: float,
    only_parameters: Sequence[str] | None = None,
) -> dict[str, Any]:
    source_id = str(entry["source_id"])
    root = Path(str(entry["physical_scan_root"])).expanduser().resolve()
    plan = _read_json(root / "scan_plan.json")
    specs, names, scales, movable, ordered, target_names = _ordered_fd_points(
        plan,
        anchor_point=anchor_point,
        target_points=target_points,
        only_parameters=only_parameters,
    )
    payloads = []
    evaluations = []
    name_to_index: dict[str, int] = {}
    for point in ordered:
        payload, tracklets, propagations = _payload_for_point(root, point)
        name_to_index[str(point["name"])] = len(evaluations)
        payloads.append(payload)
        evaluations.append(_evaluation(tracklets, propagations, min_truth_match_fraction))
    keys, indexes, overlap = _aligned_rows(evaluations, movable)
    rows = [np.asarray([index[key] for key in keys], dtype=np.intp) for index in indexes]
    parameters = len(names)
    nested_layers = [
        {
            str(station): {
                str(layer): list(payload.transform_for_layer(int(station), int(layer)))
                for layer in IFT_LAYER_IDS
            }
            for station in (0,)
        }
        for payload in payloads
    ]
    anchor_values = parameter_values_from_payload(specs, payloads[0].transforms, nested_layers[0])
    positive_values = np.asarray(
        [
            parameter_values_from_payload(specs, payloads[1 + 2 * index].transforms, nested_layers[1 + 2 * index])[name]
            for index, name in enumerate(names)
        ],
        dtype=np.float64,
    )
    negative_values = np.asarray(
        [
            parameter_values_from_payload(specs, payloads[2 + 2 * index].transforms, nested_layers[2 + 2 * index])[name]
            for index, name in enumerate(names)
        ],
        dtype=np.float64,
    )
    target_values = {}
    target_residuals = {}
    for target_name in target_names:
        index = name_to_index[target_name]
        values = parameter_values_from_payload(specs, payloads[index].transforms, nested_layers[index])
        target_values[target_name] = np.asarray([values[name] for name in names], dtype=np.float64)
        target_residuals[target_name] = np.asarray(evaluations[index].residual[rows[index]], dtype=np.float64)
    primary = target_names[0]
    anchor_evaluation = evaluations[0]
    anchor_rows = rows[0]
    return {
        "source_id": source_id,
        "split": str(entry.get("split", "")),
        "specs": specs,
        "names": names,
        "scales": scales,
        "anchor_values": np.asarray([anchor_values[name] for name in names], dtype=np.float64),
        "reference_values": target_values[primary],
        "target_names": target_names,
        "target_values": target_values,
        "target_residuals": target_residuals,
        "positive_values": positive_values,
        "negative_values": negative_values,
        "anchor_residual": np.asarray(anchor_evaluation.residual[anchor_rows], dtype=np.float64),
        "positive_residual": np.asarray(
            [evaluations[1 + 2 * index].residual[rows[1 + 2 * index]] for index in range(parameters)],
            dtype=np.float64,
        ),
        "negative_residual": np.asarray(
            [evaluations[2 + 2 * index].residual[rows[2 + 2 * index]] for index in range(parameters)],
            dtype=np.float64,
        ),
        "reference_residual": target_residuals[primary],
        "covariance": np.asarray(anchor_evaluation.combined_covariance[anchor_rows], dtype=np.float64),
        "run_id": np.asarray(anchor_evaluation.run_id[anchor_rows], dtype=np.int64),
        "event_id": np.asarray(anchor_evaluation.event_id[anchor_rows], dtype=np.int64),
        "truth_particle_id": np.asarray(anchor_evaluation.truth_particle_id[anchor_rows], dtype=np.int64),
        "source_station_id": np.asarray(anchor_evaluation.source_station_id[anchor_rows], dtype=np.int64),
        "target_station_id": np.asarray(anchor_evaluation.target_station_id[anchor_rows], dtype=np.int64),
        "overlap": overlap,
        "layer_weights": _layer_weights_from_hit_pattern(anchor_evaluation, anchor_rows),
    }


def _pool_layer_banks(banks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pooled = _pooled_bank(banks)
    pooled["specs"] = banks[0]["specs"]
    pooled["target_names"] = banks[0]["target_names"]
    pooled["target_values"] = banks[0]["target_values"]
    pooled["target_residuals"] = {
        name: np.concatenate([bank["target_residuals"][name] for bank in banks], axis=0)
        for name in banks[0]["target_names"]
    }
    pooled["layer_weights"] = np.mean(np.stack([bank["layer_weights"] for bank in banks]), axis=0)
    return pooled


def _bank_for_target(bank: Mapping[str, Any], target_name: str) -> dict[str, Any]:
    updated = dict(bank)
    updated["reference_values"] = bank["target_values"][target_name]
    updated["reference_residual"] = bank["target_residuals"][target_name]
    return updated


def _fit(bank: Mapping[str, Any], *, rcond: float, names=None, scales=None, column_transform=None, prior_sigma_native=None):
    names = tuple(bank["names"] if names is None else names)
    scales = np.asarray(bank["scales"] if scales is None else scales, dtype=np.float64)
    if column_transform is None:
        return solve_physical_finite_difference(
            bank["anchor_residual"],
            bank["positive_residual"],
            bank["negative_residual"],
            bank["reference_residual"],
            bank["covariance"],
            parameter_names=names,
            positive_values=bank["positive_values"],
            negative_values=bank["negative_values"],
            parameter_scales=scales,
            prior_sigma_native=prior_sigma_native,
            rcond=rcond,
        )
    unconstrained = solve_physical_finite_difference(
        bank["anchor_residual"],
        bank["positive_residual"],
        bank["negative_residual"],
        bank["reference_residual"],
        bank["covariance"],
        parameter_names=tuple(bank["names"]),
        positive_values=bank["positive_values"],
        negative_values=bank["negative_values"],
        parameter_scales=np.asarray(bank["scales"], dtype=np.float64),
        rcond=rcond,
    )
    derivative = unconstrained.derivative_native @ column_transform
    step = 1.0
    reduced_positive = bank["anchor_residual"][None, :, :] + step * np.moveaxis(derivative, 2, 0)
    reduced_negative = bank["anchor_residual"][None, :, :] - step * np.moveaxis(derivative, 2, 0)
    return solve_physical_finite_difference(
        bank["anchor_residual"],
        reduced_positive,
        reduced_negative,
        bank["reference_residual"],
        bank["covariance"],
        parameter_names=names,
        positive_values=np.full(len(names), step),
        negative_values=np.full(len(names), -step),
        parameter_scales=scales,
        rcond=rcond,
    )


def _expected_delta(bank: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(bank["names"], np.asarray(bank["reference_values"]) - np.asarray(bank["anchor_values"]))
    }


def _recovery_from_fit(
    bank: Mapping[str, Any],
    *,
    reduction,
    fit,
) -> dict[str, Any]:
    specs = bank["specs"]
    weights = bank["layer_weights"]
    expected_full = _expected_delta(bank)
    recovered_reduced = {
        name: float(fit.recovered_parameters[index]) for index, name in enumerate(fit.parameter_names)
    }
    if tuple(fit.parameter_names) == tuple(reduction.names):
        recovered_full = expand_gauged_parameters(recovered_reduced, reduction, specs)
        expected_reduced = project_to_gauge(expected_full, reduction, specs)
    else:
        padded = {name: 0.0 for name in reduction.names}
        padded.update(recovered_reduced)
        recovered_full = expand_gauged_parameters(padded, reduction, specs)
        expected_reduced = {
            name: project_to_gauge(expected_full, reduction, specs)[name] for name in fit.parameter_names
        }
    return {
        "expected_reduced": expected_reduced,
        "recovered_reduced": recovered_reduced,
        "expected_full": expected_full,
        "recovered_full": recovered_full,
        "expected_split": split_common_and_internal(expected_full, specs, layer_weights=weights),
        "recovered_split": split_common_and_internal(recovered_full, specs, layer_weights=weights),
        "fit": _fit_summary(fit, fit.parameter_names),
    }


def _admitted_fit(
    bank: Mapping[str, Any],
    *,
    reduction,
    reduced_scales: np.ndarray,
    admitted: Sequence[str],
    rcond: float,
):
    if not admitted:
        return None
    keep = [reduction.names.index(name) for name in admitted]
    return _fit(
        bank,
        rcond=rcond,
        names=tuple(admitted),
        scales=np.asarray([float(reduced_scales[index]) for index in keep], dtype=np.float64),
        column_transform=reduction.column_transform[:, keep],
    )


def _gauge_report(
    pooled: Mapping[str, Any],
    source_banks: Sequence[Mapping[str, Any]],
    *,
    choice: str,
    rcond: float,
    max_condition_number: float,
    max_source_spread: float,
) -> dict[str, Any]:
    specs = pooled["specs"]
    weights = pooled["layer_weights"]
    reduction = reduce_gauge(
        specs,
        choice=choice,
        layer_weights=weights,
        reference_layer=0,
        dropped_layer=2,
    )
    reduced_scales = np.asarray([float(pooled["scales"][index]) for index in reduction.keep_indices], dtype=np.float64)
    gauged_fit = _fit(
        pooled,
        rcond=rcond,
        names=reduction.names,
        scales=reduced_scales,
        column_transform=reduction.column_transform,
    )
    source_fits = [
        {
            "source_id": bank["source_id"],
            "fit": _fit(
                bank,
                rcond=rcond,
                names=reduction.names,
                scales=reduced_scales,
                column_transform=reduction.column_transform,
            ),
        }
        for bank in source_banks
    ]
    gate = _gate(
        reduction.names,
        reduced_scales,
        gauged_fit,
        source_fits,
        max_condition_number=max_condition_number,
        max_source_spread=max_source_spread,
    )
    admitted = tuple(gate["admitted_parameters"])
    target_names = tuple(pooled.get("target_names") or ("primary",))
    full_gauged = {}
    admitted_closure = {}
    for target_name in target_names:
        target_pooled = (
            _bank_for_target(pooled, target_name)
            if target_name in pooled.get("target_residuals", {})
            else pooled
        )
        full_fit = (
            gauged_fit
            if target_name == target_names[0] or target_name not in pooled.get("target_residuals", {})
            else _fit(
                target_pooled,
                rcond=rcond,
                names=reduction.names,
                scales=reduced_scales,
                column_transform=reduction.column_transform,
            )
        )
        full_gauged[target_name] = _recovery_from_fit(target_pooled, reduction=reduction, fit=full_fit)
        admitted_fit = _admitted_fit(
            target_pooled,
            reduction=reduction,
            reduced_scales=reduced_scales,
            admitted=admitted,
            rcond=rcond,
        )
        if admitted_fit is None:
            zeros = {name: 0.0 for name in pooled["names"]}
            admitted_closure[target_name] = {
                "admitted_parameters": [],
                "expected_reduced": {},
                "recovered_reduced": {},
                "expected_full": _expected_delta(target_pooled),
                "recovered_full": zeros,
                "expected_split": split_common_and_internal(
                    _expected_delta(target_pooled), specs, layer_weights=weights
                ),
                "recovered_split": split_common_and_internal(zeros, specs, layer_weights=weights),
                "fit": None,
            }
        else:
            admitted_closure[target_name] = {
                "admitted_parameters": list(admitted),
                **_recovery_from_fit(target_pooled, reduction=reduction, fit=admitted_fit),
            }
    primary = full_gauged[target_names[0]]
    return {
        "choice": choice,
        "constraint": reduction.constraint,
        "reduced_parameter_names": list(reduction.names),
        "dropped_parameter_names": list(reduction.dropped_names),
        "layer_weights": [float(value) for value in weights],
        "expected_reduced": primary["expected_reduced"],
        "recovered_reduced": primary["recovered_reduced"],
        "expected_split": primary["expected_split"],
        "recovered_split": primary["recovered_split"],
        "fit": primary["fit"],
        "identifiability_gate": gate,
        "full_gauged_recovery": full_gauged,
        "admitted_subset_closure": admitted_closure,
    }


def _default_targets(manifest: Mapping[str, object], requested: Sequence[str] | None) -> tuple[str, ...]:
    if requested:
        return tuple(dict.fromkeys(requested))
    iteration = manifest.get("alignment_iteration")
    held = iteration.get("held_out_closure_points") if isinstance(iteration, Mapping) else None
    if isinstance(held, list) and held:
        return tuple(str(name) for name in held)
    return ("iteration_00_closure_internal",)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-manifest", required=True)
    parser.add_argument("--anchor-point", default="iteration_00_reference")
    parser.add_argument(
        "--target-point",
        action="append",
        default=None,
        help="Held-out point to recover. Repeatable. Default: both closure points in the manifest.",
    )
    parser.add_argument(
        "--only-parameters",
        nargs="+",
        default=None,
        help="Restrict the Jacobian to these named parameters (and skip their unused FD points).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-truth-match-fraction", type=float, default=0.99)
    parser.add_argument("--rcond", type=float, default=1.0e-10)
    parser.add_argument("--max-condition-number", type=float, default=1.0e4)
    parser.add_argument("--max-source-spread", type=float, default=0.5)
    args = parser.parse_args()

    manifest_path = Path(args.iteration_manifest).expanduser().resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("test_data_accessed") is not False:
        raise ValueError("iteration manifest has an invalid test-access declaration")
    target_points = _default_targets(manifest, args.target_point)
    entries = _source_entries(manifest)
    banks = [
        _source_bank(
            entry,
            anchor_point=str(args.anchor_point),
            target_points=target_points,
            min_truth_match_fraction=float(args.min_truth_match_fraction),
            only_parameters=args.only_parameters,
        )
        for entry in entries
    ]
    pooled = _pool_layer_banks(banks)
    names = pooled["names"]
    specs = pooled["specs"]
    unconstrained = _fit(pooled, rcond=float(args.rcond))
    source_fits = [{"source_id": bank["source_id"], "fit": _fit(bank, rcond=float(args.rcond))} for bank in banks]
    complete_mask = _complete_route_mask(pooled)
    column_cosines = _column_cosines(unconstrained)
    degeneracy = hierarchy_degeneracy_pairs(names, column_cosines, specs)
    ungauged_gate = _gate(
        names,
        pooled["scales"],
        unconstrained,
        source_fits,
        max_condition_number=float(args.max_condition_number),
        max_source_spread=float(args.max_source_spread),
    )
    gauges = {
        choice: _gauge_report(
            pooled,
            banks,
            choice=choice,
            rcond=float(args.rcond),
            max_condition_number=float(args.max_condition_number),
            max_source_spread=float(args.max_source_spread),
        )
        for choice in ("sum_to_zero", "reference_layer")
    }
    physical_agreement = {}
    for target_name in target_points:
        physical_agreement[target_name] = {
            "internal_dx_layer0": {
                choice: gauges[choice]["full_gauged_recovery"][target_name]["recovered_split"]["layer_internal"]["layer_0"]["dx_mm"]
                for choice in gauges
            },
            "total_common_dx": {
                choice: gauges[choice]["full_gauged_recovery"][target_name]["recovered_split"]["total_common_station_plus_layer_mean"]["dx_mm"]
                for choice in gauges
            },
            "admitted_internal_dx_layer0": {
                choice: gauges[choice]["admitted_subset_closure"][target_name]["recovered_split"]["layer_internal"]["layer_0"]["dx_mm"]
                for choice in gauges
            },
            "admitted_station_dx": {
                choice: gauges[choice]["admitted_subset_closure"][target_name]["recovered_split"]["station_common"]["dx_mm"]
                for choice in gauges
            },
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "iteration_manifest": str(manifest_path),
        "anchor_point": str(args.anchor_point),
        "target_points": list(target_points),
        "parameter_names": list(names),
        "selected_parameters": list(names),
        "observation_semantics": "truth_selected_physical_edge",
        "q_over_p_mode": 0,
        "test_data_accessed": False,
        "sources": [str(bank["source_id"]) for bank in banks],
        "ungauged": {
            **_fit_summary(unconstrained, names),
            "component_sensitivity": _component_sensitivity(unconstrained, pooled),
            "station_pair_sensitivity": _pair_sensitivity(unconstrained, pooled),
            "near_degenerate_pairs_abs_cosine_ge_0.9": degeneracy,
            "identifiability_gate": ungauged_gate,
        },
        "complete_truth_route_pairs": int(complete_mask.sum()),
        "gauges": gauges,
        "physical_agreement": physical_agreement,
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "identifiability_layer.json"
    output_path.write_text(json.dumps(_json_ready(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"parameters: {len(names)}")
    print(
        "ungauged scaled normal: rank "
        f"{unconstrained.normal_matrix_rank}/{len(names)}, condition "
        f"{unconstrained.normal_matrix_condition_number}"
    )
    print(f"targets: {', '.join(target_points)}")
    for choice, payload in gauges.items():
        admitted = payload["identifiability_gate"]["admitted_parameters"]
        print(f"{choice}: admitted {', '.join(admitted) or '(none)'}")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
