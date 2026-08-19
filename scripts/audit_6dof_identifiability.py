#!/usr/bin/env python3
"""Station-0 six-DoF sensitivity and identifiability audit.

Reads one prepared multi-source physical scan (reference + anchor + central
finite-difference probes for every declared parameter, optionally held-out
joint closure points) and builds the truth-selected physical Jacobian
J = d(residual)/d(parameter) from the real /Tracker/Align -> SegmentFitRefit
-> Acts(mode 0) refit responses.  No association model, no synthetic overlay,
and no test data are involved: every observation is a truth-matched,
uniquely propagating tracklet pair involving the movable station.

Reported per parameter, both pooled and per source:
  * response sensitivity per residual component (rx/ry/rtx/rty), per adjacent
    station pair, and restricted to complete three-edge truth routes;
  * scaled normal matrix, SVD rank, condition number, singular vectors;
  * native parameter covariance/correlation and observability fraction;
  * source-to-source response spread.

A deterministic admission gate then decides which parameters may enter the
small-scale joint physical closure:
  1. stable response: per-source weighted column-norm relative spread within
     ``--max-source-spread`` of the pooled norm;
  2. resolution: data-only native sigma below the parameter severity scale;
  3. conditioning: the admitted sub-block of the scaled normal matrix stays
     full rank with condition number at most ``--max-condition-number``.
Parameters are considered in decreasing scaled-information order; a parameter
that fails any criterion is recorded as not admitted with its reason.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from alignment.physical_jacobian import (
    solve_physical_finite_difference,
    parameter_values_from_station_transforms,
)
from scripts.run_refit_multidof_closure import (
    _aligned_rows,
    _evaluation,
    _payload_for_point,
    _point_map,
    _read_json,
)


SCHEMA_VERSION = "faser-station0-6dof-identifiability-v1"
RESIDUAL_LABELS = ("rx_mm", "ry_mm", "rtx", "rty")
ADJACENT_PAIR_LABELS = ("0->1", "1->2", "2->3")


def _source_entries(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("iteration manifest has no sources")
    entries: list[dict[str, object]] = []
    for raw in sources:
        if not isinstance(raw, Mapping):
            raise ValueError("iteration manifest has an invalid source entry")
        source = dict(raw)
        if str(source.get("split", "")) not in ("train", "validation"):
            raise ValueError(f"source '{source.get('source_id')}' has an invalid split")
        entries.append(source)
    return entries


def _ordered_fd_points(
    plan: Mapping[str, object],
    *,
    anchor_point: str,
    target_point: str,
    only_parameters: Sequence[str] | None = None,
) -> tuple[tuple[dict[str, object], ...], tuple[str, ...], np.ndarray, set[int], list[Mapping[str, object]]]:
    if plan.get("scan_mode") != "station_rigid_multidof" or int(plan.get("q_over_p_mode", -1)) != 0:
        raise ValueError("source scan is not a mode-0 station_rigid_multidof physical scan")
    raw_specs = plan.get("alignment_parameter_specs")
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("source scan lacks alignment_parameter_specs")
    specs = tuple(dict(item) for item in raw_specs)
    names = tuple(str(spec["name"]) for spec in specs)
    scales = np.asarray([float(spec["severity_scale"]) for spec in specs], dtype=np.float64)
    if only_parameters is not None:
        requested = tuple(str(name) for name in only_parameters)
        unknown = set(requested) - set(names)
        if unknown or len(set(requested)) != len(requested):
            raise ValueError(f"unknown or duplicated parameter selection: {sorted(unknown)}")
        specs = tuple(spec for spec in specs if str(spec["name"]) in requested)
        names = tuple(str(spec["name"]) for spec in specs)
        scales = np.asarray([float(spec["severity_scale"]) for spec in specs], dtype=np.float64)
    movable = {int(station) for station in plan.get("movable_station_ids", ())}
    points = _point_map(plan)
    anchor = points.get(anchor_point)
    target = points.get(target_point)
    if anchor is None or target is None:
        raise ValueError(f"scan plan lacks anchor '{anchor_point}' or target '{target_point}'")
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
    ordered.append(target)
    return specs, names, scales, movable, ordered


def _source_bank(
    entry: Mapping[str, object],
    *,
    anchor_point: str,
    target_point: str,
    min_truth_match_fraction: float,
    only_parameters: Sequence[str] | None = None,
) -> dict[str, Any]:
    source_id = str(entry["source_id"])
    root = Path(str(entry["physical_scan_root"])).expanduser().resolve()
    plan = _read_json(root / "scan_plan.json")
    specs, names, scales, movable, ordered = _ordered_fd_points(
        plan, anchor_point=anchor_point, target_point=target_point, only_parameters=only_parameters
    )
    payloads = []
    evaluations = []
    for point in ordered:
        payload, tracklets, propagations = _payload_for_point(root, point)
        payloads.append(payload)
        evaluations.append(_evaluation(tracklets, propagations, min_truth_match_fraction))
    keys, indexes, overlap = _aligned_rows(evaluations, movable)
    rows = [np.asarray([index[key] for key in keys], dtype=np.intp) for index in indexes]
    parameters = len(names)
    anchor_values = parameter_values_from_station_transforms(specs, payloads[0].transforms)
    target_values = parameter_values_from_station_transforms(specs, payloads[-1].transforms)
    positive_values = np.asarray(
        [
            parameter_values_from_station_transforms(specs, payloads[1 + 2 * index].transforms)[name]
            for index, name in enumerate(names)
        ],
        dtype=np.float64,
    )
    negative_values = np.asarray(
        [
            parameter_values_from_station_transforms(specs, payloads[2 + 2 * index].transforms)[name]
            for index, name in enumerate(names)
        ],
        dtype=np.float64,
    )
    anchor_evaluation = evaluations[0]
    anchor_rows = rows[0]
    return {
        "source_id": source_id,
        "split": str(entry.get("split", "")),
        "specs": specs,
        "names": names,
        "scales": scales,
        "anchor_values": np.asarray([anchor_values[name] for name in names], dtype=np.float64),
        "reference_values": np.asarray([target_values[name] for name in names], dtype=np.float64),
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
        "reference_residual": np.asarray(evaluations[-1].residual[rows[-1]], dtype=np.float64),
        "covariance": np.asarray(anchor_evaluation.combined_covariance[anchor_rows], dtype=np.float64),
        "run_id": np.asarray(anchor_evaluation.run_id[anchor_rows], dtype=np.int64),
        "event_id": np.asarray(anchor_evaluation.event_id[anchor_rows], dtype=np.int64),
        "truth_particle_id": np.asarray(anchor_evaluation.truth_particle_id[anchor_rows], dtype=np.int64),
        "source_station_id": np.asarray(anchor_evaluation.source_station_id[anchor_rows], dtype=np.int64),
        "target_station_id": np.asarray(anchor_evaluation.target_station_id[anchor_rows], dtype=np.int64),
        "overlap": overlap,
    }


def _fit(bank: Mapping[str, Any], *, rcond: float):
    return solve_physical_finite_difference(
        bank["anchor_residual"],
        bank["positive_residual"],
        bank["negative_residual"],
        bank["reference_residual"],
        bank["covariance"],
        parameter_names=bank["names"],
        positive_values=bank["positive_values"],
        negative_values=bank["negative_values"],
        parameter_scales=bank["scales"],
        rcond=rcond,
    )


def _pooled_bank(banks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = banks[0]
    pooled = dict(first)
    for key in (
        "anchor_residual",
        "reference_residual",
        "covariance",
        "run_id",
        "event_id",
        "truth_particle_id",
        "source_station_id",
        "target_station_id",
    ):
        pooled[key] = np.concatenate([bank[key] for bank in banks], axis=0)
    pooled["positive_residual"] = np.concatenate([bank["positive_residual"] for bank in banks], axis=1)
    pooled["negative_residual"] = np.concatenate([bank["negative_residual"] for bank in banks], axis=1)
    pooled["source_id"] = "pooled"
    pooled["member_sources"] = [str(bank["source_id"]) for bank in banks]
    return pooled


def _weighted_column_norms(fit) -> np.ndarray:
    """sqrt(J_p^T W J_p) per parameter: the diagonal of the native normal matrix."""
    diagonal = np.diag(fit.normal_matrix_native)
    return np.sqrt(np.clip(diagonal, a_min=0.0, a_max=None))


def _column_cosines(fit) -> np.ndarray:
    normal = fit.normal_matrix_native
    diagonal = np.diag(normal)
    result = np.full(normal.shape, np.nan, dtype=np.float64)
    valid = diagonal > 0.0
    if np.any(valid):
        denominator = np.sqrt(np.outer(diagonal[valid], diagonal[valid]))
        result[np.ix_(valid, valid)] = normal[np.ix_(valid, valid)] / denominator
        result[np.ix_(valid, valid)] = np.clip(result[np.ix_(valid, valid)], -1.0, 1.0)
    return result


def _component_sensitivity(fit, bank: Mapping[str, object]) -> dict[str, dict[str, float]]:
    """Unweighted RMS of each Jacobian column per residual component."""
    derivative = fit.derivative_native  # [pairs, 4, parameters]
    names = bank["names"]
    result: dict[str, dict[str, float]] = {}
    for p_index, name in enumerate(names):
        column = derivative[:, :, p_index]
        result[name] = {
            label: float(np.sqrt(np.mean(np.square(column[:, c_index]))))
            for c_index, label in enumerate(RESIDUAL_LABELS)
        }
        result[name]["weighted_information_norm"] = float(
            math.sqrt(max(float(np.diag(fit.normal_matrix_native)[p_index]), 0.0))
        )
    return result


def _pair_sensitivity(fit, bank: Mapping[str, object]) -> list[dict[str, object]]:
    derivative = fit.derivative_native
    sources = np.asarray(bank["source_station_id"])
    targets = np.asarray(bank["target_station_id"])
    rows: list[dict[str, object]] = []
    for label in ADJACENT_PAIR_LABELS:
        left, right = (int(part) for part in label.split("->"))
        mask = (sources == left) & (targets == right)
        if not np.any(mask):
            continue
        sub = derivative[mask]
        row: dict[str, object] = {"station_pair": label, "pairs": int(mask.sum())}
        for p_index, name in enumerate(bank["names"]):
            row[name] = float(np.sqrt(np.mean(np.square(sub[:, :, p_index]))))
        rows.append(row)
    return rows


def _complete_route_mask(bank: Mapping[str, object]) -> np.ndarray:
    """Flag edges belonging to a truth route that spans every observed pair.

    The bank only carries edges involving a movable station (for station 0:
    the 0->1, 0->2, 0->3 trio).  A route is complete when one truth particle
    in one event contributes every pair type present in the bank, i.e. the
    track crossed all four stations and propagated successfully at each.
    """
    runs = np.asarray(bank["run_id"])
    events = np.asarray(bank["event_id"])
    truth = np.asarray(bank["truth_particle_id"])
    sources = np.asarray(bank["source_station_id"])
    targets = np.asarray(bank["target_station_id"])
    all_pairs = {(int(left), int(right)) for left, right in zip(sources.tolist(), targets.tolist())}
    coverage: dict[tuple[int, int, int], set[tuple[int, int]]] = {}
    for row in range(truth.shape[0]):
        particle = int(truth[row])
        if particle < 0:
            continue
        key = (int(runs[row]), int(events[row]), particle)
        coverage.setdefault(key, set()).add((int(sources[row]), int(targets[row])))
    complete = {key for key, pairs in coverage.items() if all_pairs.issubset(pairs)}
    return np.asarray(
        [
            (int(runs[row]), int(events[row]), int(truth[row])) in complete
            for row in range(truth.shape[0])
        ],
        dtype=bool,
    )


def _masked_bank(bank: Mapping[str, Any], mask: np.ndarray) -> dict[str, Any]:
    result = dict(bank)
    for key in (
        "anchor_residual",
        "reference_residual",
        "covariance",
        "run_id",
        "event_id",
        "truth_particle_id",
        "source_station_id",
        "target_station_id",
    ):
        result[key] = np.asarray(bank[key])[mask]
    result["positive_residual"] = np.asarray(bank["positive_residual"])[:, mask]
    result["negative_residual"] = np.asarray(bank["negative_residual"])[:, mask]
    return result


def _fit_summary(fit, names: Sequence[str]) -> dict[str, Any]:
    _, singular_vectors, right = np.linalg.svd(fit.normal_matrix_scaled, full_matrices=False)
    del singular_vectors
    sigmas = {}
    for index, name in enumerate(names):
        value = float(fit.covariance_native[index, index])
        sigmas[name] = math.sqrt(value) if math.isfinite(value) and value >= 0.0 else None
    return {
        "observations": int(fit.used_pairs),
        "normal_matrix_rank": int(fit.normal_matrix_rank),
        "full_rank": bool(fit.full_rank),
        "normal_matrix_condition_number": fit.normal_matrix_condition_number,
        "identifiable_subspace_condition_number": fit.identifiable_subspace_condition_number,
        "data_singular_values": [float(value) for value in fit.data_singular_values],
        "svd_right_singular_vectors": [
            {"singular_value": float(fit.data_singular_values[row]), "parameter_composition": {
                name: float(right[row, column]) for column, name in enumerate(names)
            }}
            for row in range(right.shape[0])
        ],
        "parameter_covariance_native": {
            name: {other: float(fit.covariance_native[i, j]) for j, other in enumerate(names)}
            for i, name in enumerate(names)
        },
        "parameter_correlation_native": {
            name: {other: float(fit.correlation_native[i, j]) for j, other in enumerate(names)}
            for i, name in enumerate(names)
        },
        "parameter_sigma_native": sigmas,
        "parameter_observability_fraction": {
            name: float(fit.parameter_observability_fraction[index]) for index, name in enumerate(names)
        },
        "response_chi2": float(fit.response_chi2),
        "response_ndof": int(fit.response_ndof),
        "recovered_anchor_to_reference": {
            name: float(fit.recovered_parameters[index]) for index, name in enumerate(names)
        },
    }


def _gate(
    names: Sequence[str],
    scales: np.ndarray,
    pooled_fit,
    source_fits: Sequence[Mapping[str, Any]],
    *,
    max_condition_number: float,
    max_source_spread: float,
) -> dict[str, Any]:
    pooled_norms = _weighted_column_norms(pooled_fit)
    per_source_norms = {
        str(entry["source_id"]): _weighted_column_norms(entry["fit"]) for entry in source_fits
    }
    scaled_information = np.diag(pooled_fit.normal_matrix_scaled)
    order = sorted(range(len(names)), key=lambda index: -float(scaled_information[index]))

    details: dict[str, Any] = {}
    for index, name in enumerate(names):
        norms = np.asarray([norms_[index] for norms_ in per_source_norms.values()], dtype=np.float64)
        pooled_norm = float(pooled_norms[index])
        spread = (
            float((norms.max() - norms.min()) / pooled_norm)
            if norms.size and pooled_norm > 0.0 and np.isfinite(norms).all()
            else None
        )
        sigma = float(pooled_fit.covariance_native[index, index])
        sigma = math.sqrt(sigma) if math.isfinite(sigma) and sigma >= 0.0 else None
        details[name] = {
            "scaled_information_rank": int(order.index(index) + 1),
            "pooled_weighted_column_norm": pooled_norm,
            "per_source_weighted_column_norm": {
                source: float(value) for source, norms_ in per_source_norms.items() for value in [norms_[index]]
            },
            "source_spread_relative": spread,
            "response_stable": spread is not None and spread <= max_source_spread,
            "sigma_native": sigma,
            "severity_scale": float(scales[index]),
            "sigma_within_severity": sigma is not None and sigma < float(scales[index]),
        }

    admitted: list[str] = []
    admitted_indices: list[int] = []
    for index in order:
        name = names[index]
        trial = admitted_indices + [index]
        submatrix = pooled_fit.normal_matrix_scaled[np.ix_(trial, trial)]
        singular = np.linalg.svd(submatrix, compute_uv=False)
        tolerance = 1.0e-10 * float(singular[0]) if singular.size else 0.0
        rank = int(np.count_nonzero(singular > tolerance))
        condition = float(singular[0] / singular[-1]) if singular.size and singular[-1] > 0.0 else None
        stable = bool(details[name]["response_stable"])
        resolved = bool(details[name]["sigma_within_severity"])
        full_rank = rank == len(trial)
        conditioned = condition is not None and condition <= max_condition_number
        verdict = stable and resolved and full_rank and conditioned
        details[name].update(
            {
                "trial_submatrix_rank": rank,
                "trial_submatrix_size": len(trial),
                "trial_full_rank": full_rank,
                "trial_condition_number": condition,
                "trial_condition_acceptable": conditioned,
                "admitted": verdict,
                "exclusion_reason": (
                    None
                    if verdict
                    else ", ".join(
                        reason
                        for reason, failed in (
                            ("unstable source response", not stable),
                            ("data sigma exceeds severity scale", not resolved),
                            ("trial submatrix rank deficient", not full_rank),
                            ("trial condition number too large", not conditioned),
                        )
                        if failed
                    )
                ),
            }
        )
        if verdict:
            admitted.append(name)
            admitted_indices.append(index)
    return {
        "max_condition_number": float(max_condition_number),
        "max_source_spread": float(max_source_spread),
        "admission_order_by_scaled_information": [names[index] for index in order],
        "admitted_parameters": admitted,
        "not_admitted_parameters": [name for name in names if name not in admitted],
        "per_parameter": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-manifest", required=True)
    parser.add_argument("--anchor-point", required=True)
    parser.add_argument("--reference-point", required=True)
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
    entries = _source_entries(manifest)

    banks = [
        _source_bank(
            entry,
            anchor_point=str(args.anchor_point),
            target_point=str(args.reference_point),
            min_truth_match_fraction=float(args.min_truth_match_fraction),
        )
        for entry in entries
    ]
    pooled = _pooled_bank(banks)
    names = pooled["names"]
    scales = pooled["scales"]

    pooled_fit = _fit(pooled, rcond=float(args.rcond))
    source_fits = [{"source_id": bank["source_id"], "fit": _fit(bank, rcond=float(args.rcond))} for bank in banks]

    complete_mask = _complete_route_mask(pooled)
    route_fit = _fit(_masked_bank(pooled, complete_mask), rcond=float(args.rcond))

    column_cosines = _column_cosines(pooled_fit)
    degenerate_pairs = []
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            if j <= i:
                continue
            cosine = float(column_cosines[i, j])
            if math.isfinite(cosine) and abs(cosine) >= 0.9:
                degenerate_pairs.append({"pair": [left, right], "response_column_cosine": cosine})

    gate = _gate(
        names,
        scales,
        pooled_fit,
        source_fits,
        max_condition_number=float(args.max_condition_number),
        max_source_spread=float(args.max_source_spread),
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "iteration_manifest": str(manifest_path),
        "anchor_point": str(args.anchor_point),
        "reference_point": str(args.reference_point),
        "parameter_names": list(names),
        "parameter_scales": [float(value) for value in scales],
        "anchor_values": {name: float(value) for name, value in zip(names, pooled["anchor_values"])},
        "reference_values": {name: float(value) for name, value in zip(names, pooled["reference_values"])},
        "finite_difference_steps": {
            name: float(0.5 * (plus - minus))
            for name, plus, minus in zip(names, pooled["positive_values"], pooled["negative_values"])
        },
        "observation_semantics": "truth_selected_physical_edge",
        "q_over_p_mode": 0,
        "test_data_accessed": False,
        "sources": [str(bank["source_id"]) for bank in banks],
        "pooled": {
            **_fit_summary(pooled_fit, names),
            "component_sensitivity": _component_sensitivity(pooled_fit, pooled),
            "station_pair_sensitivity": _pair_sensitivity(pooled_fit, pooled),
            "response_column_cosine": {
                name: {other: float(column_cosines[i, j]) for j, other in enumerate(names)}
                for i, name in enumerate(names)
            },
            "near_degenerate_pairs_abs_cosine_ge_0.9": degenerate_pairs,
        },
        "complete_truth_route_subset": {
            "pairs": int(complete_mask.sum()),
            "fraction_of_pooled": float(complete_mask.mean()) if complete_mask.size else None,
            **_fit_summary(route_fit, names),
        },
        "per_source": [
            {
                "source_id": str(bank["source_id"]),
                "split": str(bank["split"]),
                "overlap": dict(bank["overlap"]),
                **_fit_summary(entry["fit"], names),
                "component_sensitivity": _component_sensitivity(entry["fit"], bank),
                "station_pair_sensitivity": _pair_sensitivity(entry["fit"], bank),
            }
            for bank, entry in zip(banks, source_fits)
        ],
        "identifiability_gate": gate,
    }

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "identifiability_6dof.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"parameters: {', '.join(names)}")
    print(f"pooled truth pairs: {pooled_fit.used_pairs} (complete routes: {int(complete_mask.sum())})")
    print(
        "scaled normal matrix: rank "
        f"{pooled_fit.normal_matrix_rank}/{len(names)}, condition "
        f"{pooled_fit.normal_matrix_condition_number}"
    )
    print(f"admitted: {', '.join(gate['admitted_parameters']) or '(none)'}")
    for name in gate["not_admitted_parameters"]:
        print(f"excluded: {name} ({gate['per_parameter'][name]['exclusion_reason']})")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
