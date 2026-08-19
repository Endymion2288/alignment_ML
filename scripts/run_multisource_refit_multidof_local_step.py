#!/usr/bin/env python3
"""Aggregate a source-disjoint physical multi-DoF alignment closure.

All inputs must already be independent real conditions payloads followed by
SCT cluster-to-segment refitting and mode-0 Acts propagation.  The train split
defines a local finite-difference update at an anchor payload.  The validation
split is then used only to evaluate that fixed update and to report an
independent diagnostic fit; it never changes the train update.  Test sources
are rejected before any physical file is opened.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from alignment.capture_criteria import attach_capture_and_prior, load_capture_criteria
from alignment.physical_jacobian import (
    PhysicalJacobianFit,
    parameter_values_from_station_transforms,
    solve_physical_finite_difference,
    station_transforms_with_parameter_values,
)
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events
from scripts.audit_physical_route_candidate_graph import ADJACENT_PAIRS, _condition_chain_checks, audit_candidate_graph
from scripts.run_refit_multidof_closure import (
    RESIDUAL_LABELS,
    _aligned_rows,
    _evaluation,
    _payload_for_point,
    _point_map,
    _read_json,
)


SCHEMA_VERSION = "faser-multisource-physical-alignment-iteration-v1"
SPLITS = ("train", "validation")


@dataclass(frozen=True)
class ResponseBank:
    """Physical finite-difference responses pooled only across source files."""

    split: str
    parameter_names: tuple[str, ...]
    specs: tuple[dict[str, object], ...]
    parameter_scales: np.ndarray
    anchor_values: np.ndarray
    target_values: np.ndarray
    positive_values: np.ndarray
    negative_values: np.ndarray
    anchor_transforms: Mapping[str, Sequence[float]]
    anchor_residual: np.ndarray
    positive_residual: np.ndarray
    negative_residual: np.ndarray
    target_residual: np.ndarray
    covariance: np.ndarray
    source_ids: np.ndarray
    run_ids: np.ndarray
    event_ids: np.ndarray
    source_station_ids: np.ndarray
    target_station_ids: np.ndarray
    source_tracklet_ids: np.ndarray
    target_tracklet_ids: np.ndarray
    source_overlap: tuple[dict[str, object], ...]

    @property
    def observations(self) -> int:
        return int(self.anchor_residual.shape[0])


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
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"not a multi-source alignment iteration manifest: {path}")
    if payload.get("physical_geometry_repropagation") is not True or payload.get("coordinate_surrogate") is not False:
        raise ValueError("iteration manifest lacks the real-geometry contract")
    if int(payload.get("q_over_p_mode", -1)) != 0:
        raise ValueError("multi-source closure requires mode-0 Acts propagation")
    if tuple(payload.get("allowed_splits", ())) != SPLITS or "test" not in set(payload.get("forbidden_splits", ())):
        raise ValueError("iteration manifest does not strictly seal test data")
    if payload.get("test_data_accessed") is not False:
        raise ValueError("iteration manifest has an invalid test access declaration")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("iteration manifest has no sources")
    return payload


def _parse_named_nonnegative(
    values: Sequence[str] | None,
    names: Sequence[str],
    *,
    label: str,
    require_all: bool = True,
) -> np.ndarray | None:
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
    unknown = set(parsed) - set(names)
    if unknown:
        raise ValueError(f"{label} has unknown parameter(s): " + ", ".join(sorted(unknown)))
    if require_all and set(parsed) != set(names):
        raise ValueError(f"{label} must specify exactly: " + ", ".join(names))
    return np.asarray([parsed[name] if name in parsed else math.nan for name in names], dtype=np.float64)


def _plan_contract(plan: Mapping[str, object]) -> dict[str, object]:
    """Select all scan fields that must agree across physical source files."""
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
    return {field: plan.get(field) for field in fields}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _probe(
    points: Mapping[str, Mapping[str, object]],
    *,
    parameter: str,
    sign: str,
    anchor: str,
) -> Mapping[str, object]:
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


def _required_plan_fields(
    plan: Mapping[str, object],
    *,
    anchor_point: str,
    target_point: str,
) -> tuple[
    tuple[dict[str, object], ...],
    tuple[str, ...],
    np.ndarray,
    set[int],
    list[Mapping[str, object]],
]:
    if plan.get("scan_mode") != "station_rigid_multidof" or int(plan.get("q_over_p_mode", -1)) != 0:
        raise ValueError("source scan is not a mode-0 station_rigid_multidof physical scan")
    raw_specs = plan.get("alignment_parameter_specs")
    if not isinstance(raw_specs, list) or not raw_specs or any(not isinstance(item, Mapping) for item in raw_specs):
        raise ValueError("source scan lacks valid alignment_parameter_specs")
    specs = tuple(dict(item) for item in raw_specs)
    names = tuple(str(spec.get("name", "")) for spec in specs)
    if not all(names) or len(set(names)) != len(names):
        raise ValueError("source scan has invalid parameter names")
    scales = np.asarray([float(spec["severity_scale"]) for spec in specs], dtype=np.float64)
    if not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise ValueError("source scan has invalid severity scales")
    movable = {int(station) for station in plan.get("movable_station_ids", ())}
    reference = {int(station) for station in plan.get("reference_station_ids", ())}
    if not movable or not reference or movable.intersection(reference):
        raise ValueError("source scan has invalid movable/reference station sets")
    points = _point_map(plan)
    anchor = points.get(anchor_point)
    target = points.get(target_point)
    if anchor is None or target is None:
        raise ValueError("source scan does not contain requested anchor/target point")
    if anchor.get("point_role") in {"nominal", "finite_difference_positive", "finite_difference_negative"}:
        raise ValueError("anchor must be an independently refitted non-reference alignment payload")
    ordered: list[Mapping[str, object]] = [anchor]
    for name in names:
        ordered.extend(
            (
                _probe(points, parameter=name, sign="positive", anchor=anchor_point),
                _probe(points, parameter=name, sign="negative", anchor=anchor_point),
            )
        )
    ordered.append(target)
    return specs, names, scales, movable, ordered


def _source_entries(manifest: Mapping[str, object], split: str) -> list[dict[str, object]]:
    if split not in SPLITS:
        raise ValueError(f"unknown split '{split}'")
    result: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for raw in manifest["sources"]:
        if not isinstance(raw, Mapping):
            raise ValueError("iteration manifest has an invalid source entry")
        source = dict(raw)
        source_id = str(source.get("source_id", ""))
        source_split = str(source.get("split", ""))
        if not source_id or source_id in seen_ids:
            raise ValueError("iteration manifest source IDs are absent or duplicated")
        seen_ids.add(source_id)
        if source_split not in SPLITS:
            raise ValueError(f"source '{source_id}' has invalid split '{source_split}'")
        xAOD = str(Path(str(source.get("input_xaod", ""))).expanduser().resolve())
        if xAOD in seen_paths:
            raise ValueError("an original xAOD appears more than once in the iteration manifest")
        seen_paths.add(xAOD)
        if source_split == split:
            config = Path(str(source.get("physical_scan_config", ""))).expanduser().resolve()
            root = Path(str(source.get("physical_scan_root", ""))).expanduser().resolve()
            if not config.is_file() or root != config.parent / "physical_scan":
                raise ValueError(f"source '{source_id}' physical scan config/root is invalid")
            result.append({**source, "physical_scan_config": str(config), "physical_scan_root": str(root)})
    if not result:
        raise ValueError(f"iteration manifest has no '{split}' source")
    return result


def _load_bank(
    manifest: Mapping[str, object],
    *,
    split: str,
    anchor_point: str,
    target_point: str,
    min_truth_match_fraction: float,
) -> ResponseBank:
    entries = _source_entries(manifest, split)
    anchor_residual: list[np.ndarray] = []
    positive_residual: list[np.ndarray] = []
    negative_residual: list[np.ndarray] = []
    target_residual: list[np.ndarray] = []
    covariance: list[np.ndarray] = []
    source_ids: list[np.ndarray] = []
    run_ids: list[np.ndarray] = []
    event_ids: list[np.ndarray] = []
    source_station_ids: list[np.ndarray] = []
    target_station_ids: list[np.ndarray] = []
    source_tracklet_ids: list[np.ndarray] = []
    target_tracklet_ids: list[np.ndarray] = []
    overlaps: list[dict[str, object]] = []

    reference_contract: str | None = None
    specs: tuple[dict[str, object], ...] | None = None
    names: tuple[str, ...] | None = None
    scales: np.ndarray | None = None
    values_reference: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Mapping[str, Sequence[float]]] | None = None
    for entry in entries:
        source_id = str(entry["source_id"])
        root = Path(str(entry["physical_scan_root"])).expanduser().resolve()
        plan = _read_json(root / "scan_plan.json")
        source_specs, source_names, source_scales, movable, ordered_points = _required_plan_fields(
            plan,
            anchor_point=anchor_point,
            target_point=target_point,
        )
        contract = _canonical_json(_plan_contract(plan))
        if reference_contract is None:
            reference_contract = contract
            specs = source_specs
            names = source_names
            scales = source_scales
        elif contract != reference_contract:
            raise ValueError(f"source '{source_id}' does not share the frozen physical probe plan")
        assert specs is not None and names is not None and scales is not None
        payloads = []
        evaluations = []
        for point in ordered_points:
            payload, tracklets, propagations = _payload_for_point(root, point)
            payloads.append(payload)
            evaluations.append(_evaluation(tracklets, propagations, min_truth_match_fraction))
        keys, indexes, overlap = _aligned_rows(evaluations, movable)
        rows_by_evaluation = [np.asarray([index[key] for key in keys], dtype=np.intp) for index in indexes]
        if not keys:
            raise ValueError(f"source '{source_id}' has no common movable-station truth pairs")
        local_anchor_values = parameter_values_from_station_transforms(specs, payloads[0].transforms)
        local_target_values = parameter_values_from_station_transforms(specs, payloads[-1].transforms)
        local_positive_values = np.asarray(
            [
                parameter_values_from_station_transforms(specs, payloads[1 + 2 * index].transforms)[name]
                for index, name in enumerate(names)
            ],
            dtype=np.float64,
        )
        local_negative_values = np.asarray(
            [
                parameter_values_from_station_transforms(specs, payloads[2 + 2 * index].transforms)[name]
                for index, name in enumerate(names)
            ],
            dtype=np.float64,
        )
        local_values = (
            np.asarray([local_anchor_values[name] for name in names], dtype=np.float64),
            np.asarray([local_target_values[name] for name in names], dtype=np.float64),
            local_positive_values,
            local_negative_values,
            payloads[0].transforms,
        )
        if values_reference is None:
            values_reference = local_values
        else:
            for observed, expected in zip(local_values[:4], values_reference[:4]):
                if not np.allclose(observed, expected, rtol=0.0, atol=1.0e-12):
                    raise ValueError(f"source '{source_id}' payload values differ from the common plan")
            if _canonical_json(local_values[4]) != _canonical_json(values_reference[4]):
                raise ValueError(f"source '{source_id}' anchor transform differs from the common plan")
        anchor_rows = rows_by_evaluation[0]
        anchor_residual.append(np.asarray(evaluations[0].residual[anchor_rows], dtype=np.float64))
        positive_residual.append(
            np.asarray(
                [
                    evaluations[1 + 2 * index].residual[rows_by_evaluation[1 + 2 * index]]
                    for index in range(len(names))
                ],
                dtype=np.float64,
            )
        )
        negative_residual.append(
            np.asarray(
                [
                    evaluations[2 + 2 * index].residual[rows_by_evaluation[2 + 2 * index]]
                    for index in range(len(names))
                ],
                dtype=np.float64,
            )
        )
        target_residual.append(np.asarray(evaluations[-1].residual[rows_by_evaluation[-1]], dtype=np.float64))
        covariance.append(np.asarray(evaluations[0].combined_covariance[anchor_rows], dtype=np.float64))
        source_ids.append(np.full(len(keys), source_id, dtype=object))
        run_ids.append(np.asarray(evaluations[0].run_id[anchor_rows], dtype=np.int64))
        event_ids.append(np.asarray(evaluations[0].event_id[anchor_rows], dtype=np.int64))
        source_station_ids.append(np.asarray(evaluations[0].source_station_id[anchor_rows], dtype=np.int64))
        target_station_ids.append(np.asarray(evaluations[0].target_station_id[anchor_rows], dtype=np.int64))
        source_tracklet_ids.append(np.asarray(evaluations[0].source_tracklet_id[anchor_rows], dtype=np.int64))
        target_tracklet_ids.append(np.asarray(evaluations[0].target_tracklet_id[anchor_rows], dtype=np.int64))
        overlaps.append({"source_id": source_id, **overlap, "used_truth_pairs": len(keys)})
    assert specs is not None and names is not None and scales is not None and values_reference is not None
    return ResponseBank(
        split=split,
        parameter_names=names,
        specs=specs,
        parameter_scales=scales,
        anchor_values=values_reference[0],
        target_values=values_reference[1],
        positive_values=values_reference[2],
        negative_values=values_reference[3],
        anchor_transforms=values_reference[4],
        anchor_residual=np.concatenate(anchor_residual, axis=0),
        positive_residual=np.concatenate(positive_residual, axis=1),
        negative_residual=np.concatenate(negative_residual, axis=1),
        target_residual=np.concatenate(target_residual, axis=0),
        covariance=np.concatenate(covariance, axis=0),
        source_ids=np.concatenate(source_ids, axis=0),
        run_ids=np.concatenate(run_ids, axis=0),
        event_ids=np.concatenate(event_ids, axis=0),
        source_station_ids=np.concatenate(source_station_ids, axis=0),
        target_station_ids=np.concatenate(target_station_ids, axis=0),
        source_tracklet_ids=np.concatenate(source_tracklet_ids, axis=0),
        target_tracklet_ids=np.concatenate(target_tracklet_ids, axis=0),
        source_overlap=tuple(overlaps),
    )


def _fit_bank(
    bank: ResponseBank,
    *,
    prior_sigma: np.ndarray | None,
    rcond: float,
) -> PhysicalJacobianFit:
    return solve_physical_finite_difference(
        bank.anchor_residual,
        bank.positive_residual,
        bank.negative_residual,
        bank.target_residual,
        bank.covariance,
        parameter_names=bank.parameter_names,
        positive_values=bank.positive_values,
        negative_values=bank.negative_values,
        parameter_scales=bank.parameter_scales,
        prior_sigma_native=prior_sigma,
        rcond=rcond,
    )


def _fit_diagnostic(fit: PhysicalJacobianFit, names: Sequence[str]) -> dict[str, object]:
    return {
        "observations": int(fit.used_pairs),
        "normal_matrix_rank": int(fit.normal_matrix_rank),
        "fit_matrix_rank": int(fit.fit_matrix_rank),
        "normal_matrix_condition_number": fit.normal_matrix_condition_number,
        "identifiable_subspace_condition_number": fit.identifiable_subspace_condition_number,
        "recovered_local_delta": {name: float(fit.recovered_parameters[index]) for index, name in enumerate(names)},
        "recovered_sigma": {
            name: (
                math.sqrt(float(fit.covariance_native[index, index]))
                if math.isfinite(float(fit.covariance_native[index, index]))
                and float(fit.covariance_native[index, index]) >= 0.0
                else None
            )
            for index, name in enumerate(names)
        },
        "parameter_observability_fraction": {
            name: float(fit.parameter_observability_fraction[index]) for index, name in enumerate(names)
        },
    }


def _chi2(values: np.ndarray, covariance: np.ndarray) -> float:
    total = 0.0
    for residual, matrix in zip(values, covariance):
        total += float(residual @ np.linalg.solve(matrix, residual))
    return total


def _apply_update(
    bank: ResponseBank,
    derivative_native: np.ndarray,
    update: np.ndarray,
) -> dict[str, object]:
    if derivative_native.shape != (bank.observations, len(RESIDUAL_LABELS), len(bank.parameter_names)):
        raise ValueError("physical derivative does not match its response bank")
    predicted = np.einsum("nrp,p->nr", derivative_native, update)
    # The passed update is determined on the fit split.  This function only
    # evaluates that already-frozen update against a physical response bank.
    response = bank.target_residual - bank.anchor_residual
    post = response - predicted
    baseline_chi2 = _chi2(response, bank.covariance)
    post_chi2 = _chi2(post, bank.covariance)
    return {
        "baseline_response_chi2": baseline_chi2,
        "post_update_response_chi2": post_chi2,
        "response_chi2_reduction": baseline_chi2 - post_chi2,
        "response_chi2_reduction_fraction": (
            None if baseline_chi2 <= 0.0 else float(1.0 - post_chi2 / baseline_chi2)
        ),
        "baseline_response_rms": {
            label: float(np.sqrt(np.mean(np.square(response[:, index]))))
            for index, label in enumerate(RESIDUAL_LABELS)
        },
        "post_update_response_rms": {
            label: float(np.sqrt(np.mean(np.square(post[:, index]))))
            for index, label in enumerate(RESIDUAL_LABELS)
        },
    }


def _curvature_summary(bank: ResponseBank) -> list[dict[str, object]]:
    curvature = bank.positive_residual + bank.negative_residual - 2.0 * bank.anchor_residual[np.newaxis, ...]
    result: list[dict[str, object]] = []
    for parameter_index, name in enumerate(bank.parameter_names):
        values = curvature[parameter_index]
        result.append(
            {
                "name": name,
                "observations": int(values.shape[0]),
                "deterministic_weighted_curvature": _chi2(values, bank.covariance),
                "rms_by_residual_component": {
                    label: float(np.sqrt(np.mean(np.square(values[:, index]))))
                    for index, label in enumerate(RESIDUAL_LABELS)
                },
            }
        )
    return result


def _subset_fit(
    bank: ResponseBank,
    mask: np.ndarray,
    *,
    rcond: float,
) -> dict[str, object]:
    if int(np.count_nonzero(mask)) < 1:
        return {"observations": 0, "status": "empty"}
    sub = ResponseBank(
        **{
            **bank.__dict__,
            "anchor_residual": bank.anchor_residual[mask],
            "positive_residual": bank.positive_residual[:, mask],
            "negative_residual": bank.negative_residual[:, mask],
            "target_residual": bank.target_residual[mask],
            "covariance": bank.covariance[mask],
            "source_ids": bank.source_ids[mask],
            "run_ids": bank.run_ids[mask],
            "event_ids": bank.event_ids[mask],
            "source_station_ids": bank.source_station_ids[mask],
            "target_station_ids": bank.target_station_ids[mask],
            "source_tracklet_ids": bank.source_tracklet_ids[mask],
            "target_tracklet_ids": bank.target_tracklet_ids[mask],
        }
    )
    try:
        fit = _fit_bank(sub, prior_sigma=None, rcond=rcond)
    except ValueError as error:
        return {"observations": int(np.count_nonzero(mask)), "status": "invalid", "error": str(error)}
    return {"observations": int(np.count_nonzero(mask)), "status": "ok", **_fit_diagnostic(fit, bank.parameter_names)}


def _source_diagnostics(bank: ResponseBank, *, rcond: float) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for source_id in sorted({str(value) for value in bank.source_ids}):
        mask = bank.source_ids == source_id
        result.append({"source_id": source_id, **_subset_fit(bank, mask, rcond=rcond)})
    return result


def _station_pair_diagnostics(bank: ResponseBank, *, rcond: float) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    pairs = sorted(set(zip(bank.source_station_ids.tolist(), bank.target_station_ids.tolist())))
    for source, target in pairs:
        mask = (bank.source_station_ids == source) & (bank.target_station_ids == target)
        result.append(
            {
                "source_station_id": int(source),
                "target_station_id": int(target),
                **_subset_fit(bank, mask, rcond=rcond),
            }
        )
    return result


def _candidate_metrics(
    manifest: Mapping[str, object],
    *,
    chi2_gate: float | None,
) -> list[dict[str, object]]:
    """Aggregate raw physical candidate retention at every frozen plan point."""
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for raw in manifest["sources"]:
        if not isinstance(raw, Mapping):
            raise ValueError("iteration manifest has an invalid source entry")
        source = dict(raw)
        split = str(source["split"])
        source_id = str(source["source_id"])
        root = Path(str(source["physical_scan_root"])).expanduser().resolve()
        plan = _read_json(root / "scan_plan.json")
        if plan.get("scan_mode") != "station_rigid_multidof" or int(plan.get("q_over_p_mode", -1)) != 0:
            raise ValueError(f"source '{source_id}' does not have a valid mode-0 multi-DoF plan")
        for raw_point in plan.get("points", []):
            if not isinstance(raw_point, Mapping):
                raise ValueError("scan plan has an invalid point")
            point = dict(raw_point)
            point_root = root / str(point["relative_point_dir"])
            events = load_events(point_root / "refit" / "tracklets.root", require_mc_labels=True)
            records = load_propagation_records(point_root / "refit" / "propagations.root")
            graph = audit_candidate_graph(events, records, chi2_gate=chi2_gate)
            chain = _condition_chain_checks(point_root / "logs" / "refit.log")
            # Surface errors and layer-overlap messages are counted directly
            # from the real refit log.  They are recorded separately from
            # truth-edge availability because an unrelated failed surface
            # attempt must not be silently reclassified as a missing edge.
            if chain is None:  # pragma: no cover - a physical point has a log
                raise RuntimeError("physical refit log audit unexpectedly returned None")
            surface_errors = int(chain["acts_surface_error_count"])
            layer_overlap_errors = int(chain["acts_layer_overlap_error_count"])
            key = (split, str(point["name"]))
            aggregate = grouped.setdefault(
                key,
                {
                    "split": split,
                    "point_name": str(point["name"]),
                    "point_role": str(point.get("point_role", "")),
                    "direction_trial": str(point.get("direction_trial", "")),
                    "alignment_parameter_values": dict(point.get("alignment_parameter_values", {})),
                    "sources": 0,
                    "events": 0,
                    "condition_chain_complete_sources": 0,
                    "acts_surface_error_count": 0,
                    "acts_layer_overlap_error_count": 0,
                    "complete_truth_chains": 0,
                    "candidate_retained_complete_truth_chains": 0,
                    "physical_candidate_edges": {f"{left}->{right}": 0 for left, right in ADJACENT_PAIRS},
                    "truth_pairs_with_unique_endpoints": {f"{left}->{right}": 0 for left, right in ADJACENT_PAIRS},
                    "candidate_retained_truth_pairs": {f"{left}->{right}": 0 for left, right in ADJACENT_PAIRS},
                },
            )
            aggregate["sources"] = int(aggregate["sources"]) + 1
            aggregate["events"] = int(aggregate["events"]) + int(graph["events"])
            aggregate["condition_chain_complete_sources"] = int(
                aggregate["condition_chain_complete_sources"]
            ) + int(all(bool(value) for value in chain["checks"].values()))
            aggregate["acts_surface_error_count"] = int(aggregate["acts_surface_error_count"]) + surface_errors
            aggregate["acts_layer_overlap_error_count"] = int(
                aggregate["acts_layer_overlap_error_count"]
            ) + layer_overlap_errors
            aggregate["complete_truth_chains"] = int(aggregate["complete_truth_chains"]) + int(
                graph["complete_truth_chains"]
            )
            aggregate["candidate_retained_complete_truth_chains"] = int(
                aggregate["candidate_retained_complete_truth_chains"]
            ) + int(graph["candidate_retained_complete_truth_chains"])
            by_pair = graph["by_station_pair"]
            assert isinstance(by_pair, Mapping)
            for left, right in ADJACENT_PAIRS:
                label = f"{left}->{right}"
                values = by_pair[label]
                assert isinstance(values, Mapping)
                for field in (
                    "physical_candidate_edges",
                    "truth_pairs_with_unique_endpoints",
                    "candidate_retained_truth_pairs",
                ):
                    holder = aggregate[field]
                    assert isinstance(holder, Mapping)
                    holder[label] = int(holder[label]) + int(values[field])
    rows: list[dict[str, object]] = []
    for _, aggregate in sorted(grouped.items()):
        chains = int(aggregate["complete_truth_chains"])
        retained_chains = int(aggregate["candidate_retained_complete_truth_chains"])
        sources = int(aggregate["sources"])
        row = {
            **aggregate,
            "candidate_chi2_gate": chi2_gate,
            "candidate_complete_truth_chain_recall": None if chains == 0 else float(retained_chains / chains),
            "condition_chain_complete_fraction": (
                None
                if sources == 0
                else float(int(aggregate["condition_chain_complete_sources"]) / sources)
            ),
        }
        for left, right in ADJACENT_PAIRS:
            label = f"{left}->{right}"
            denominator = int(aggregate["truth_pairs_with_unique_endpoints"][label])
            retained = int(aggregate["candidate_retained_truth_pairs"][label])
            row[f"{label}_candidate_truth_edge_recall"] = None if denominator == 0 else float(retained / denominator)
            row[f"{label}_physical_candidate_edges"] = int(aggregate["physical_candidate_edges"][label])
        rows.append(row)
    return rows


def _flat_candidate_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    flat: list[dict[str, object]] = []
    for row in rows:
        result = {
            key: value
            for key, value in row.items()
            if not isinstance(value, Mapping)
        }
        values = row.get("alignment_parameter_values")
        if isinstance(values, Mapping):
            result.update({str(key): value for key, value in values.items()})
        flat.append(result)
    return flat


def _response_rows(bank: ResponseBank, fit: PhysicalJacobianFit) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(bank.observations):
        row: dict[str, object] = {
            "split": bank.split,
            "source_id": str(bank.source_ids[index]),
            "run_id": int(bank.run_ids[index]),
            "event_id": int(bank.event_ids[index]),
            "source_station_id": int(bank.source_station_ids[index]),
            "target_station_id": int(bank.target_station_ids[index]),
            "source_tracklet_id": int(bank.source_tracklet_ids[index]),
            "target_tracklet_id": int(bank.target_tracklet_ids[index]),
        }
        for component, label in enumerate(RESIDUAL_LABELS):
            row[f"anchor_residual_{label}"] = float(bank.anchor_residual[index, component])
            row[f"target_residual_{label}"] = float(bank.target_residual[index, component])
            row[f"response_{label}"] = float(fit.response[index, component])
            for parameter, name in enumerate(bank.parameter_names):
                row[f"derivative_{name}_{label}"] = float(fit.derivative_native[index, component, parameter])
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-manifest", required=True)
    parser.add_argument("--anchor-point", required=True)
    parser.add_argument("--target-point", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fit-split", choices=SPLITS, default="train")
    parser.add_argument("--held-out-split", choices=SPLITS, default="validation")
    parser.add_argument("--min-truth-match-fraction", type=float, default=0.99)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--prior-sigma", action="append", default=None, metavar="PARAMETER:VALUE")
    parser.add_argument("--capture-tolerance", action="append", default=None, metavar="PARAMETER:VALUE")
    parser.add_argument(
        "--capture-criteria",
        default=None,
        help="Pre-registered dual engineering+pull capture JSON selected on train only.",
    )
    parser.add_argument("--candidate-chi2-gate", type=float, default=25.0)
    parser.add_argument("--rcond", type=float, default=1.0e-10)
    parser.add_argument("--require-full-rank", action="store_true")
    args = parser.parse_args()
    if args.capture_criteria is not None and args.capture_tolerance is not None:
        parser.error("provide only one of --capture-criteria or --capture-tolerance")
    if args.fit_split == args.held_out_split:
        parser.error("--fit-split and --held-out-split must differ")
    if not np.isfinite(args.min_truth_match_fraction) or not 0.0 <= args.min_truth_match_fraction <= 1.0:
        parser.error("--min-truth-match-fraction must be in [0, 1]")
    if not np.isfinite(args.damping) or not 0.0 < args.damping <= 1.0:
        parser.error("--damping must be in (0, 1]")
    if not np.isfinite(args.rcond) or not 0.0 < args.rcond < 1.0:
        parser.error("--rcond must be in (0, 1)")
    if args.candidate_chi2_gate is not None and (
        not np.isfinite(args.candidate_chi2_gate) or args.candidate_chi2_gate <= 0.0
    ):
        parser.error("--candidate-chi2-gate must be finite and positive")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.iteration_manifest).expanduser().resolve()
    manifest = _read_manifest(manifest_path)
    fit_bank = _load_bank(
        manifest,
        split=args.fit_split,
        anchor_point=str(args.anchor_point),
        target_point=str(args.target_point),
        min_truth_match_fraction=float(args.min_truth_match_fraction),
    )
    held_out_bank = _load_bank(
        manifest,
        split=args.held_out_split,
        anchor_point=str(args.anchor_point),
        target_point=str(args.target_point),
        min_truth_match_fraction=float(args.min_truth_match_fraction),
    )
    if fit_bank.parameter_names != held_out_bank.parameter_names or not np.allclose(
        fit_bank.parameter_scales, held_out_bank.parameter_scales, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("fit and held-out physical banks have incompatible parameter definitions")
    for fit_values, held_out_values in zip(
        (fit_bank.anchor_values, fit_bank.target_values, fit_bank.positive_values, fit_bank.negative_values),
        (held_out_bank.anchor_values, held_out_bank.target_values, held_out_bank.positive_values, held_out_bank.negative_values),
    ):
        if not np.allclose(fit_values, held_out_values, rtol=0.0, atol=1.0e-12):
            raise ValueError("fit and held-out banks do not share the frozen physical payload values")
    names = fit_bank.parameter_names
    prior = _parse_named_nonnegative(args.prior_sigma, names, label="--prior-sigma", require_all=False)
    tolerance = _parse_named_nonnegative(args.capture_tolerance, names, label="--capture-tolerance")
    criteria = None if args.capture_criteria is None else load_capture_criteria(Path(args.capture_criteria).expanduser().resolve())
    fit = _fit_bank(fit_bank, prior_sigma=prior, rcond=float(args.rcond))
    held_out_fit = _fit_bank(held_out_bank, prior_sigma=prior, rcond=float(args.rcond))
    if args.require_full_rank and (not fit.full_rank or not held_out_fit.full_rank):
        raise RuntimeError("fit or held-out physical normal matrix is rank deficient")
    expected_delta = fit_bank.target_values - fit_bank.anchor_values
    proposal = fit_bank.anchor_values + float(args.damping) * fit.recovered_parameters
    errors = fit.recovered_parameters - expected_delta
    proposed_values = {name: float(proposal[index]) for index, name in enumerate(names)}
    proposed_transforms = station_transforms_with_parameter_values(
        fit_bank.specs,
        fit_bank.anchor_transforms,
        proposed_values,
    )
    held_out_application = _apply_update(
        held_out_bank,
        held_out_fit.derivative_native,
        float(args.damping) * fit.recovered_parameters,
    )
    fit_application = _apply_update(
        fit_bank,
        fit.derivative_native,
        float(args.damping) * fit.recovered_parameters,
    )
    parameter_rows = []
    for index, (name, spec) in enumerate(zip(names, fit_bank.specs)):
        variance = float(fit.covariance_native[index, index])
        sigma = math.sqrt(variance) if math.isfinite(variance) and variance >= 0.0 else None
        parameter_rows.append(
            {
                "name": name,
                "station_id": int(spec["station_id"]),
                "component": str(spec["component"]),
                "unit": str(spec["unit"]),
                "anchor_value": float(fit_bank.anchor_values[index]),
                "target_value": float(fit_bank.target_values[index]),
                "expected_delta_to_target": float(expected_delta[index]),
                "fit_split_recovered_local_delta": float(fit.recovered_parameters[index]),
                "fit_split_local_delta_error": float(errors[index]),
                "held_out_independent_recovered_local_delta": float(held_out_fit.recovered_parameters[index]),
                "held_out_independent_local_delta_error": float(
                    held_out_fit.recovered_parameters[index] - expected_delta[index]
                ),
                "proposed_next_value": float(proposal[index]),
                "recovered_sigma": sigma,
                "observability_fraction": float(fit.parameter_observability_fraction[index]),
                "capture_tolerance": None if tolerance is None else float(tolerance[index]),
                "capture_success": None if tolerance is None else bool(abs(errors[index]) <= tolerance[index]),
                "held_out_independent_capture_success": (
                    None
                    if tolerance is None
                    else bool(
                        abs(held_out_fit.recovered_parameters[index] - expected_delta[index])
                        <= tolerance[index]
                    )
                ),
            }
        )
    held_out_errors = held_out_fit.recovered_parameters - expected_delta
    held_out_sigmas = []
    for index in range(len(names)):
        variance = float(held_out_fit.covariance_native[index, index])
        held_out_sigmas.append(
            math.sqrt(variance) if math.isfinite(variance) and variance >= 0.0 else None
        )
    parameter_rows, capture_aggregate, prior_rows = attach_capture_and_prior(
        parameter_rows,
        names=names,
        errors=[float(value) for value in errors],
        fit_sigmas=[row["recovered_sigma"] for row in parameter_rows],
        criteria=criteria,
        normal_matrix_native=fit.normal_matrix_native,
        covariance_native=fit.covariance_native,
        prior_sigma_native=prior,
    )
    _, held_out_capture_aggregate, held_out_prior_rows = attach_capture_and_prior(
        [{} for _ in names],
        names=names,
        errors=[float(value) for value in held_out_errors],
        fit_sigmas=held_out_sigmas,
        criteria=criteria,
        normal_matrix_native=held_out_fit.normal_matrix_native,
        covariance_native=held_out_fit.covariance_native,
        prior_sigma_native=prior,
    )
    capture_parameters = tolerance is not None and all(row["capture_success"] for row in parameter_rows)
    held_out_capture_parameters = tolerance is not None and all(
        row["held_out_independent_capture_success"] for row in parameter_rows
    )
    heldout_reduces_response = bool(held_out_application["response_chi2_reduction"] > 0.0)
    if capture_aggregate is not None:
        capture_success = bool(
            capture_aggregate["capture_success"]
            and held_out_capture_aggregate["capture_success"]
            and heldout_reduces_response
        )
    else:
        capture_success = (
            None
            if tolerance is None
            else bool(capture_parameters and held_out_capture_parameters and heldout_reduces_response)
        )
    candidate_metrics = _candidate_metrics(manifest, chi2_gate=args.candidate_chi2_gate)
    _write_csv(output / "candidate_metrics.csv", _flat_candidate_rows(candidate_metrics))
    _write_csv(output / "fit_split_response_pairs.csv", _response_rows(fit_bank, fit))
    _write_csv(output / "held_out_response_pairs.csv", _response_rows(held_out_bank, held_out_fit))
    source_diagnostics = {
        args.fit_split: _source_diagnostics(fit_bank, rcond=float(args.rcond)),
        args.held_out_split: _source_diagnostics(held_out_bank, rcond=float(args.rcond)),
    }
    station_pair_diagnostics = {
        args.fit_split: _station_pair_diagnostics(fit_bank, rcond=float(args.rcond)),
        args.held_out_split: _station_pair_diagnostics(held_out_bank, rcond=float(args.rcond)),
    }
    _write_csv(
        output / "source_fit_diagnostics.csv",
        [
            {"split": split, **row}
            for split, rows in source_diagnostics.items()
            for row in rows
        ],
    )
    _write_csv(
        output / "station_pair_fit_diagnostics.csv",
        [
            {"split": split, **row}
            for split, rows in station_pair_diagnostics.items()
            for row in rows
        ],
    )
    np.savez_compressed(
        output / "multisource_local_step_arrays.npz",
        parameter_names=np.asarray(names),
        fit_anchor_residual=fit_bank.anchor_residual,
        fit_positive_residual=fit_bank.positive_residual,
        fit_negative_residual=fit_bank.negative_residual,
        fit_target_residual=fit_bank.target_residual,
        fit_covariance=fit_bank.covariance,
        fit_derivative_native=fit.derivative_native,
        fit_normal_matrix_native=fit.normal_matrix_native,
        fit_normal_matrix_scaled=fit.normal_matrix_scaled,
        fit_covariance_native=fit.covariance_native,
        fit_correlation_native=fit.correlation_native,
        held_out_anchor_residual=held_out_bank.anchor_residual,
        held_out_positive_residual=held_out_bank.positive_residual,
        held_out_negative_residual=held_out_bank.negative_residual,
        held_out_target_residual=held_out_bank.target_residual,
        held_out_covariance=held_out_bank.covariance,
        held_out_derivative_native=held_out_fit.derivative_native,
        held_out_normal_matrix_native=held_out_fit.normal_matrix_native,
        held_out_normal_matrix_scaled=held_out_fit.normal_matrix_scaled,
        held_out_covariance_native=held_out_fit.covariance_native,
        held_out_correlation_native=held_out_fit.correlation_native,
    )
    summary = {
        "method": "source_disjoint_truth_fixed_local_physical_multidof_finite_difference_step",
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "q_over_p_mode": 0,
        "iteration_manifest": str(manifest_path),
        "test_data_accessed": False,
        "fit_split_name": args.fit_split,
        "held_out_split_name": args.held_out_split,
        "anchor_point": str(args.anchor_point),
        "target_point": str(args.target_point),
        "damping": float(args.damping),
        "candidate_chi2_gate": args.candidate_chi2_gate,
        "update_semantics": (
            "The proposal is determined only by the fit split's real physical finite-difference responses. "
            "The held-out split consumes that frozen proposal only for closure evaluation; its independent fit is diagnostic."
        ),
        "fit_split": {
            "source_count": len({str(value) for value in fit_bank.source_ids}),
            "source_overlap": fit_bank.source_overlap,
            "finite_difference_linearity": _curvature_summary(fit_bank),
            "fit": _fit_diagnostic(fit, names),
            "application": fit_application,
        },
        "held_out_split": {
            "source_count": len({str(value) for value in held_out_bank.source_ids}),
            "source_overlap": held_out_bank.source_overlap,
            "finite_difference_linearity": _curvature_summary(held_out_bank),
            "independent_fit_diagnostic_only": _fit_diagnostic(held_out_fit, names),
            "frozen_fit_split_update_application": held_out_application,
        },
        "parameter_covariance": fit.covariance_native,
        "parameter_correlation": fit.correlation_native,
        "parameters": parameter_rows,
        "prior_audit": prior_rows,
        "held_out_prior_audit": held_out_prior_rows,
        "capture_aggregate": capture_aggregate,
        "held_out_capture_aggregate": held_out_capture_aggregate,
        "capture_criteria": None if args.capture_criteria is None else str(Path(args.capture_criteria).expanduser().resolve()),
        "proposed_next_parameter_values": proposed_values,
        "proposed_next_station_transforms": proposed_transforms,
        "candidate_metrics": candidate_metrics,
        "source_fit_diagnostics": source_diagnostics,
        "station_pair_fit_diagnostics": station_pair_diagnostics,
        "capture_success": capture_success,
        "capture_success_definition": (
            "With --capture-criteria: train and held-out free parameters pass the pre-registered "
            "statistical+coverage gate, and the frozen fit-split update reduces held-out response chi2. "
            "Survey-constrained dz is excluded from capture.  With --capture-tolerance: fit and held-out "
            "independent parameter recoveries lie within declared absolute tolerances, and the frozen "
            "fit-split update reduces held-out physical response chi2."
        ),
    }
    (output / "multisource_local_step.json").write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            _json_ready(
                {
                    "output_dir": str(output),
                    "fit_observations": fit_bank.observations,
                    "held_out_observations": held_out_bank.observations,
                    "fit_normal_matrix_rank": fit.normal_matrix_rank,
                    "fit_normal_matrix_condition_number": fit.normal_matrix_condition_number,
                    "proposed_next_parameter_values": proposed_values,
                    "capture_success": capture_success,
                    "test_data_accessed": False,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
