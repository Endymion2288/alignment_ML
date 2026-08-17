#!/usr/bin/env python3
"""Run source-disjoint finite-difference closure for arbitrary joint payloads.

Unlike the anchor-centred iterative step, this command evaluates one or more
independently refitted joint curriculum/held-out points relative to a physical
reference point.  It pools only original xAOD sources within a split.  Train
and validation therefore remain source-disjoint diagnostics even when their
random observed transform vectors intentionally differ.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from alignment.physical_jacobian import parameter_values_from_station_transforms, solve_physical_finite_difference
from scripts.run_refit_multidof_closure import (
    RESIDUAL_LABELS,
    _aligned_rows,
    _evaluation,
    _payload_for_point,
    _point_map,
    _read_json,
)


SPLITS = ("train", "validation")


@dataclass(frozen=True)
class ClosureBank:
    split: str
    observed_point: str
    names: tuple[str, ...]
    specs: tuple[dict[str, object], ...]
    scales: np.ndarray
    reference_values: np.ndarray
    observed_values: np.ndarray
    positive_values: np.ndarray
    negative_values: np.ndarray
    reference_residual: np.ndarray
    positive_residual: np.ndarray
    negative_residual: np.ndarray
    observed_residual: np.ndarray
    covariance: np.ndarray
    source_ids: np.ndarray
    source_station_ids: np.ndarray
    target_station_ids: np.ndarray
    source_counts: tuple[dict[str, object], ...]

    @property
    def observations(self) -> int:
        return int(self.reference_residual.shape[0])


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
    values = list(rows) if rows else [{"empty": ""}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("physical_geometry_repropagation") is not True or payload.get("coordinate_surrogate") is True:
        raise ValueError("physical manifest lacks real-geometry closure contract")
    if int(payload.get("q_over_p_mode", -1)) != 0:
        raise ValueError("multi-source physical closure requires mode-0")
    if "test" not in {str(value) for value in payload.get("forbidden_splits", ())}:
        raise ValueError("physical manifest must explicitly forbid test")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("physical manifest has no sources")
    return payload


def _parse_tolerances(values: Sequence[str] | None, names: Sequence[str]) -> np.ndarray | None:
    if not values:
        return None
    parsed: dict[str, float] = {}
    for raw in values:
        name, separator, text = str(raw).partition(":")
        if not separator or not name or name in parsed:
            raise ValueError("--capture-tolerance entries must have unique PARAMETER:VALUE form")
        value = float(text)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("--capture-tolerance values must be finite and non-negative")
        parsed[name] = value
    if set(parsed) != set(names):
        raise ValueError("--capture-tolerance must specify exactly: " + ", ".join(names))
    return np.asarray([parsed[name] for name in names], dtype=np.float64)


def _probe(points: Mapping[str, Mapping[str, object]], name: str, sign: str, reference: str) -> Mapping[str, object]:
    matches = []
    for point in points.values():
        if point.get("finite_difference_for") != name or point.get("probe_sign") != sign:
            continue
        tagged_anchor = point.get("finite_difference_anchor")
        if tagged_anchor is None or tagged_anchor == reference:
            matches.append(point)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {sign} probe for '{name}' around '{reference}'")
    return matches[0]


def _canonical_plan(plan: Mapping[str, object]) -> str:
    fields = (
        "scan_mode",
        "q_over_p_mode",
        "station_ids",
        "reference_station_ids",
        "movable_station_ids",
        "condition_axis",
        "alignment_parameter_specs",
        "points",
    )
    return json.dumps({field: plan.get(field) for field in fields}, sort_keys=True, separators=(",", ":"))


def _source_records(manifest: Mapping[str, object], split: str) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    for raw in manifest["sources"]:
        if not isinstance(raw, Mapping):
            raise ValueError("physical manifest has an invalid source record")
        source = dict(raw)
        source_split = str(source.get("split", ""))
        if source_split == "test":
            raise ValueError("sealed test source appears in physical manifest")
        xAOD = str(Path(str(source.get("input_xaod", ""))).expanduser().resolve())
        if xAOD in seen_paths:
            raise ValueError("original xAOD is reused in physical manifest")
        seen_paths.add(xAOD)
        if source_split == split:
            selected.append(source)
    if not selected:
        raise ValueError(f"physical manifest has no source in split '{split}'")
    return selected


def _load_bank(
    manifest: Mapping[str, object],
    *,
    split: str,
    reference_point: str,
    observed_point: str,
    min_truth_match_fraction: float,
) -> ClosureBank:
    reference_residual: list[np.ndarray] = []
    positive_residual: list[np.ndarray] = []
    negative_residual: list[np.ndarray] = []
    observed_residual: list[np.ndarray] = []
    covariance: list[np.ndarray] = []
    source_ids: list[np.ndarray] = []
    source_station_ids: list[np.ndarray] = []
    target_station_ids: list[np.ndarray] = []
    source_counts: list[dict[str, object]] = []
    canonical: str | None = None
    specs: tuple[dict[str, object], ...] | None = None
    names: tuple[str, ...] | None = None
    scales: np.ndarray | None = None
    values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    for source in _source_records(manifest, split):
        source_id = str(source.get("source_id", ""))
        root = Path(str(source.get("physical_scan_root", ""))).expanduser().resolve()
        plan = _read_json(root / "scan_plan.json")
        if plan.get("scan_mode") != "station_rigid_multidof" or int(plan.get("q_over_p_mode", -1)) != 0:
            raise ValueError(f"source '{source_id}' is not a mode-0 station-rigid multi-DoF scan")
        plan_signature = _canonical_plan(plan)
        raw_specs = plan.get("alignment_parameter_specs")
        if not isinstance(raw_specs, list) or not raw_specs or any(not isinstance(item, Mapping) for item in raw_specs):
            raise ValueError("scan plan has invalid alignment parameter specs")
        local_specs = tuple(dict(item) for item in raw_specs)
        local_names = tuple(str(spec.get("name", "")) for spec in local_specs)
        local_scales = np.asarray([float(spec["severity_scale"]) for spec in local_specs], dtype=np.float64)
        movable = {int(station) for station in plan.get("movable_station_ids", ())}
        if not movable or not all(local_names) or len(set(local_names)) != len(local_names):
            raise ValueError("scan plan has invalid movable stations or parameter names")
        points = _point_map(plan)
        reference = points.get(reference_point)
        observed = points.get(observed_point)
        if reference is None or observed is None:
            raise ValueError(f"source '{source_id}' lacks reference/observed point")
        if reference.get("point_role") != "nominal":
            raise ValueError("reference point must be the independently refitted nominal payload")
        if observed.get("point_role") in {"nominal", "finite_difference_positive", "finite_difference_negative"}:
            raise ValueError("observed point must be an independent joint curriculum/closure payload")
        ordered: list[Mapping[str, object]] = [reference]
        for name in local_names:
            ordered.extend((_probe(points, name, "positive", reference_point), _probe(points, name, "negative", reference_point)))
        ordered.append(observed)
        if canonical is None:
            canonical = plan_signature
            specs = local_specs
            names = local_names
            scales = local_scales
        elif canonical != plan_signature:
            raise ValueError(f"source '{source_id}' does not share the physical multi-DoF plan")
        assert specs is not None and names is not None and scales is not None
        payloads = []
        evaluations = []
        for point in ordered:
            payload, tracklets, propagations = _payload_for_point(root, point)
            payloads.append(payload)
            evaluations.append(_evaluation(tracklets, propagations, min_truth_match_fraction))
        keys, indexes, overlap = _aligned_rows(evaluations, movable)
        rows = [np.asarray([index[key] for key in keys], dtype=np.intp) for index in indexes]
        local_values = (
            np.asarray(
                [parameter_values_from_station_transforms(specs, payloads[0].transforms)[name] for name in names],
                dtype=np.float64,
            ),
            np.asarray(
                [parameter_values_from_station_transforms(specs, payloads[-1].transforms)[name] for name in names],
                dtype=np.float64,
            ),
            np.asarray(
                [
                    parameter_values_from_station_transforms(specs, payloads[1 + 2 * index].transforms)[name]
                    for index, name in enumerate(names)
                ],
                dtype=np.float64,
            ),
            np.asarray(
                [
                    parameter_values_from_station_transforms(specs, payloads[2 + 2 * index].transforms)[name]
                    for index, name in enumerate(names)
                ],
                dtype=np.float64,
            ),
        )
        if values is None:
            values = local_values
        elif any(not np.allclose(left, right, rtol=0.0, atol=1.0e-12) for left, right in zip(values, local_values)):
            raise ValueError(f"source '{source_id}' payload values differ inside split '{split}'")
        reference_rows = rows[0]
        reference_residual.append(np.asarray(evaluations[0].residual[reference_rows], dtype=np.float64))
        positive_residual.append(
            np.asarray([evaluations[1 + 2 * index].residual[rows[1 + 2 * index]] for index in range(len(names))])
        )
        negative_residual.append(
            np.asarray([evaluations[2 + 2 * index].residual[rows[2 + 2 * index]] for index in range(len(names))])
        )
        observed_residual.append(np.asarray(evaluations[-1].residual[rows[-1]], dtype=np.float64))
        covariance.append(np.asarray(evaluations[0].combined_covariance[reference_rows], dtype=np.float64))
        source_ids.append(np.full(len(keys), source_id, dtype=object))
        source_station_ids.append(np.asarray(evaluations[0].source_station_id[reference_rows], dtype=np.int64))
        target_station_ids.append(np.asarray(evaluations[0].target_station_id[reference_rows], dtype=np.int64))
        source_counts.append({"source_id": source_id, **overlap, "used_truth_pairs": len(keys)})
    assert specs is not None and names is not None and scales is not None and values is not None
    return ClosureBank(
        split=split,
        observed_point=observed_point,
        names=names,
        specs=specs,
        scales=scales,
        reference_values=values[0],
        observed_values=values[1],
        positive_values=values[2],
        negative_values=values[3],
        reference_residual=np.concatenate(reference_residual, axis=0),
        positive_residual=np.concatenate(positive_residual, axis=1),
        negative_residual=np.concatenate(negative_residual, axis=1),
        observed_residual=np.concatenate(observed_residual, axis=0),
        covariance=np.concatenate(covariance, axis=0),
        source_ids=np.concatenate(source_ids, axis=0),
        source_station_ids=np.concatenate(source_station_ids, axis=0),
        target_station_ids=np.concatenate(target_station_ids, axis=0),
        source_counts=tuple(source_counts),
    )


def _curvature(bank: ClosureBank) -> list[dict[str, object]]:
    values = bank.positive_residual + bank.negative_residual - 2.0 * bank.reference_residual[np.newaxis, ...]
    result: list[dict[str, object]] = []
    for index, name in enumerate(bank.names):
        component = values[index]
        weighted = sum(float(row @ np.linalg.solve(matrix, row)) for row, matrix in zip(component, bank.covariance))
        result.append(
            {
                "name": name,
                "observations": bank.observations,
                "deterministic_weighted_curvature": weighted,
                "rms_by_residual_component": {
                    label: float(np.sqrt(np.mean(np.square(component[:, position]))))
                    for position, label in enumerate(RESIDUAL_LABELS)
                },
            }
        )
    return result


def _run_bank(bank: ClosureBank, *, rcond: float, tolerance: np.ndarray | None) -> tuple[dict[str, object], object]:
    fit = solve_physical_finite_difference(
        bank.reference_residual,
        bank.positive_residual,
        bank.negative_residual,
        bank.observed_residual,
        bank.covariance,
        parameter_names=bank.names,
        positive_values=bank.positive_values,
        negative_values=bank.negative_values,
        parameter_scales=bank.scales,
        rcond=rcond,
    )
    expected = bank.observed_values - bank.reference_values
    errors = fit.recovered_parameters - expected
    parameters = []
    for index, (name, spec) in enumerate(zip(bank.names, bank.specs)):
        variance = float(fit.covariance_native[index, index])
        parameters.append(
            {
                "name": name,
                "station_id": int(spec["station_id"]),
                "component": str(spec["component"]),
                "unit": str(spec["unit"]),
                "reference_value": float(bank.reference_values[index]),
                "observed_value": float(bank.observed_values[index]),
                "expected_delta": float(expected[index]),
                "recovered_delta": float(fit.recovered_parameters[index]),
                "absolute_recovery_error": float(abs(errors[index])),
                "recovery_error": float(errors[index]),
                "recovered_sigma": math.sqrt(variance) if variance >= 0.0 and math.isfinite(variance) else None,
                "observability_fraction": float(fit.parameter_observability_fraction[index]),
                "capture_tolerance": None if tolerance is None else float(tolerance[index]),
                "capture_success": None if tolerance is None else bool(abs(errors[index]) <= tolerance[index]),
            }
        )
    summary = {
        "split": bank.split,
        "observed_point": bank.observed_point,
        "source_count": len({str(value) for value in bank.source_ids}),
        "used_truth_pairs": bank.observations,
        "normal_matrix_rank": int(fit.normal_matrix_rank),
        "fit_matrix_rank": int(fit.fit_matrix_rank),
        "normal_matrix_condition_number": fit.normal_matrix_condition_number,
        "identifiable_subspace_condition_number": fit.identifiable_subspace_condition_number,
        "parameter_covariance": fit.covariance_native,
        "parameter_correlation": fit.correlation_native,
        "response_chi2": float(fit.response_chi2),
        "response_ndof": int(fit.response_ndof),
        "finite_difference_linearity": _curvature(bank),
        "source_overlap": bank.source_counts,
        "parameters": parameters,
        "capture_success": None if tolerance is None else bool(all(row["capture_success"] for row in parameters)),
    }
    return summary, fit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-manifest", required=True)
    parser.add_argument("--reference-point", required=True)
    parser.add_argument("--observed-point", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", action="append", choices=SPLITS, default=None)
    parser.add_argument("--min-truth-match-fraction", type=float, default=0.99)
    parser.add_argument("--capture-tolerance", action="append", default=None, metavar="PARAMETER:VALUE")
    parser.add_argument("--rcond", type=float, default=1.0e-10)
    parser.add_argument("--require-full-rank", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.min_truth_match_fraction) or not 0.0 <= args.min_truth_match_fraction <= 1.0:
        parser.error("--min-truth-match-fraction must be in [0, 1]")
    if not np.isfinite(args.rcond) or not 0.0 < args.rcond < 1.0:
        parser.error("--rcond must be in (0, 1)")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.physical_manifest).expanduser().resolve()
    manifest = _load_manifest(manifest_path)
    rows: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    for split in tuple(args.split or SPLITS):
        tolerance: np.ndarray | None = None
        for observed in args.observed_point:
            bank = _load_bank(
                manifest,
                split=split,
                reference_point=str(args.reference_point),
                observed_point=str(observed),
                min_truth_match_fraction=float(args.min_truth_match_fraction),
            )
            if tolerance is None:
                tolerance = _parse_tolerances(args.capture_tolerance, bank.names)
            elif args.capture_tolerance is not None and tolerance.shape != (len(bank.names),):
                raise ValueError("closure points have incompatible parameter definitions")
            summary, fit = _run_bank(bank, rcond=float(args.rcond), tolerance=tolerance)
            if args.require_full_rank and int(summary["normal_matrix_rank"]) != len(bank.names):
                raise RuntimeError(f"{split}/{observed} physical closure normal matrix is rank deficient")
            results.append(summary)
            for parameter in summary["parameters"]:
                assert isinstance(parameter, Mapping)
                rows.append({"split": split, "observed_point": str(observed), **parameter})
            stem = f"{split}_{observed}"
            np.savez_compressed(
                output / f"{stem}_arrays.npz",
                parameter_names=np.asarray(bank.names),
                reference_residual=bank.reference_residual,
                positive_residual=bank.positive_residual,
                negative_residual=bank.negative_residual,
                observed_residual=bank.observed_residual,
                covariance=bank.covariance,
                derivative_native=fit.derivative_native,
                normal_matrix_native=fit.normal_matrix_native,
                normal_matrix_scaled=fit.normal_matrix_scaled,
                covariance_native=fit.covariance_native,
                correlation_native=fit.correlation_native,
            )
    _write_csv(output / "closure_parameters.csv", rows)
    payload = {
        "method": "source_disjoint_truth_fixed_station_rigid_multidof_physical_closure",
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "q_over_p_mode": 0,
        "test_data_accessed": False,
        "physical_manifest": str(manifest_path),
        "reference_point": str(args.reference_point),
        "results": results,
    }
    (output / "closure.json").write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_ready({"output_dir": str(output), "closures": len(results), "test_data_accessed": False}), indent=2))


if __name__ == "__main__":
    main()
