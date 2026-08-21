#!/usr/bin/env python3
"""Quantitative station–C_dx leakage / identifiability audit for hierarchical V1.

Uses the already-produced iteration-00 finite-difference Jacobian and the
iteration-00 / remaining (999493, 999500) observation banks.  No new physical
refit is submitted.  ``C_dx`` is never written to a payload: it is only a
Jacobian column used to form the leakage operator

    A = (J_s^T W J_s)^{-1} J_s^T W j_c

and a train-only nuisance-robust station estimator (weighted projection /
Schur profiling).  Validation is a frozen transfer of that procedure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from alignment.capture_criteria import attach_capture_and_prior, load_capture_criteria
from alignment.hierarchical_v1 import C_DX, STATION_SURVEY_PARAMETER
from alignment.hierarchical_v1_leakage import (
    STATION_FIVE,
    compare_station_estimators,
    cdx_budget_from_operator,
    default_survey_prior,
    estimator_diagnostics,
    leakage_operator_from_normal,
    predicted_station_bias,
    realizability_from_budget,
    schur_profiled_station_normal,
    slice_parameter_bank,
)
from alignment.layer_hierarchy import IFT_LAYER_IDS
from alignment.physical_jacobian import parameter_values_from_payload, solve_physical_finite_difference
from alignment.route_selected_update import align_route_selected_observations, apply_observation_statistics
from scripts.audit_6dof_identifiability import _fit_summary, _source_entries
from scripts.audit_layer_identifiability import (
    _aligned_rows,
    _evaluation,
    _fit,
    _layer_weights_from_hit_pattern,
    _ordered_fd_points,
    _payload_for_point,
    _point_map,
    _pool_layer_banks,
)
from scripts.run_refit_multidof_closure import _json_ready, _read_json
from scripts.run_route_selected_multidof_update import (
    _load_anchor_selected_payload_bank,
    _point_parameter_values,
    _points,
    _probe_for_parameter,
    _select_parameter_specs,
    _specs,
)


SCHEMA_VERSION = "faser-ift-hierarchical-v1-leakage-audit-v1"
JOINT_NAMES = (
    "ift_dx_mm",
    "ift_dy_mm",
    "ift_dz_mm",
    "ift_rx_mrad",
    "ift_ry_mrad",
    "ift_rz_mrad",
    C_DX,
)
STATION_SIX = JOINT_NAMES[:-1]
ANCHOR_POINT = "iteration_00_reference"
TARGETS = ("iteration_00_start", "iteration_00_heldout_00")
RCOND = 1.0e-10
MIN_TRUTH_MATCH = 0.99


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _rms(values) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(array))))


def _source_spread(values: Sequence[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return float("nan")
    return float(np.max(array) - np.min(array))


def _prior_array(names: Sequence[str]) -> np.ndarray:
    prior = np.full(len(names), np.nan, dtype=np.float64)
    if STATION_SURVEY_PARAMETER in names:
        prior[list(names).index(STATION_SURVEY_PARAMETER)] = default_survey_prior()[STATION_SURVEY_PARAMETER]
    return prior


def _nested_layers(payload) -> dict[str, dict[str, list[float]]]:
    return {
        str(station): {
            str(layer): list(payload.transform_for_layer(int(station), int(layer)))
            for layer in IFT_LAYER_IDS
        }
        for station in (0,)
    }


def _bank_from_evaluations(
    *,
    source_id: str,
    split: str,
    specs,
    names,
    scales,
    payloads,
    evaluations,
    rows,
    overlap,
    target_name: str,
    target_index: int,
) -> dict[str, Any]:
    parameters = len(names)
    nested = [_nested_layers(payload) for payload in payloads]
    anchor_values = parameter_values_from_payload(specs, payloads[0].transforms, nested[0])
    positive_values = np.asarray(
        [
            parameter_values_from_payload(
                specs, payloads[1 + 2 * index].transforms, nested[1 + 2 * index]
            )[name]
            for index, name in enumerate(names)
        ],
        dtype=np.float64,
    )
    negative_values = np.asarray(
        [
            parameter_values_from_payload(
                specs, payloads[2 + 2 * index].transforms, nested[2 + 2 * index]
            )[name]
            for index, name in enumerate(names)
        ],
        dtype=np.float64,
    )
    target_values = parameter_values_from_payload(
        specs, payloads[target_index].transforms, nested[target_index]
    )
    target_residual = np.asarray(evaluations[target_index].residual[rows[target_index]], dtype=np.float64)
    anchor_evaluation = evaluations[0]
    anchor_rows = rows[0]
    reference_values = np.asarray([target_values[name] for name in names], dtype=np.float64)
    return {
        "source_id": source_id,
        "split": split,
        "specs": specs,
        "names": names,
        "scales": scales,
        "anchor_values": np.asarray([anchor_values[name] for name in names], dtype=np.float64),
        "reference_values": reference_values,
        "target_names": (target_name,),
        "target_values": {target_name: reference_values},
        "target_residuals": {target_name: target_residual},
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
        "reference_residual": target_residual,
        "covariance": np.asarray(anchor_evaluation.combined_covariance[anchor_rows], dtype=np.float64),
        "run_id": np.asarray(anchor_evaluation.run_id[anchor_rows], dtype=np.int64),
        "event_id": np.asarray(anchor_evaluation.event_id[anchor_rows], dtype=np.int64),
        "truth_particle_id": np.asarray(anchor_evaluation.truth_particle_id[anchor_rows], dtype=np.int64),
        "source_station_id": np.asarray(anchor_evaluation.source_station_id[anchor_rows], dtype=np.int64),
        "target_station_id": np.asarray(anchor_evaluation.target_station_id[anchor_rows], dtype=np.int64),
        "overlap": overlap,
        "layer_weights": _layer_weights_from_hit_pattern(anchor_evaluation, anchor_rows),
    }


def load_truth_fd_cache(entry: Mapping[str, object], *, min_truth_match_fraction: float) -> dict[str, Any]:
    source_id = str(entry["source_id"])
    root = Path(str(entry["physical_scan_root"])).expanduser().resolve()
    plan = _read_json(root / "scan_plan.json")
    specs, names, scales, movable, ordered, _ = _ordered_fd_points(
        plan,
        anchor_point=ANCHOR_POINT,
        target_points=(),
        only_parameters=JOINT_NAMES,
        require_targets=False,
    )
    payloads = []
    evaluations = []
    for point in ordered:
        payload, tracklets, propagations = _payload_for_point(root, point)
        payloads.append(payload)
        evaluations.append(_evaluation(tracklets, propagations, min_truth_match_fraction))
        print(f"  truth FD {source_id} {point['name']}", flush=True)
    return {
        "source_id": source_id,
        "split": str(entry.get("split", "")),
        "jacobian_root": root,
        "specs": specs,
        "names": names,
        "scales": scales,
        "movable": movable,
        "ordered": list(ordered),
        "payloads": payloads,
        "evaluations": evaluations,
    }


def attach_truth_target(cache: Mapping[str, Any], observed_entry: Mapping[str, object], target_point: str) -> dict[str, Any]:
    observed_root = Path(str(observed_entry["physical_scan_root"])).expanduser().resolve()
    target = _point_map(_read_json(observed_root / "scan_plan.json")).get(target_point)
    if target is None:
        raise ValueError(f"{observed_entry['source_id']} lacks target {target_point}")
    payload, tracklets, propagations = _payload_for_point(observed_root, target)
    payloads = list(cache["payloads"]) + [payload]
    evaluations = list(cache["evaluations"]) + [_evaluation(tracklets, propagations, MIN_TRUTH_MATCH)]
    keys, indexes, overlap = _aligned_rows(evaluations, cache["movable"])
    rows = [np.asarray([index[key] for key in keys], dtype=np.intp) for index in indexes]
    return _bank_from_evaluations(
        source_id=str(cache["source_id"]),
        split=str(cache["split"]),
        specs=cache["specs"],
        names=cache["names"],
        scales=cache["scales"],
        payloads=payloads,
        evaluations=evaluations,
        rows=rows,
        overlap=overlap,
        target_name=target_point,
        target_index=len(payloads) - 1,
    )


def _operator_bundle(fit, names: Sequence[str], scales) -> dict[str, Any]:
    five = leakage_operator_from_normal(
        fit.normal_matrix_native,
        names,
        station_names=STATION_FIVE,
        parameter_scales=scales,
        rcond=RCOND,
    )
    six = leakage_operator_from_normal(
        fit.normal_matrix_native,
        names,
        station_names=STATION_SIX,
        prior_sigma_native=default_survey_prior(),
        parameter_scales=scales,
        rcond=RCOND,
    )
    return {"five_dof": five.as_json(), "production_six_dof_survey_dz": six.as_json(), "_five": five, "_six": six}


def _fit_station_estimators(bank: Mapping[str, Any], *, criteria) -> dict[str, Any]:
    names = tuple(bank["names"])
    joint = _fit(bank, rcond=RCOND, prior_sigma_native=_prior_array(names))
    operators = _operator_bundle(joint, names, bank["scales"])
    ordinary_bank = slice_parameter_bank(bank, STATION_SIX)
    ordinary_fit = _fit(
        ordinary_bank,
        rcond=RCOND,
        prior_sigma_native=_prior_array(STATION_SIX),
    )
    profiled_normal, _, _ = schur_profiled_station_normal(
        joint.normal_matrix_native,
        names,
        station_names=STATION_SIX,
        prior_sigma_native=default_survey_prior(),
    )
    ordinary_diag = estimator_diagnostics(
        ordinary_fit.normal_matrix_native,
        STATION_SIX,
        parameter_scales=ordinary_bank["scales"],
        prior_sigma_native=default_survey_prior(),
        rcond=RCOND,
    )
    profiled_diag = estimator_diagnostics(
        profiled_normal,
        STATION_SIX,
        parameter_scales=ordinary_bank["scales"],
        rcond=RCOND,
    )
    # Profiled recovery is the station block of the joint 7-parameter fit
    # (C_dx is a nuisance column, not a written DoF).
    expected_joint = {
        name: float(bank["reference_values"][index] - bank["anchor_values"][index])
        for index, name in enumerate(names)
    }
    ordinary_recovered = {
        name: float(ordinary_fit.recovered_parameters[index]) for index, name in enumerate(STATION_SIX)
    }
    profiled_recovered = {
        name: float(joint.recovered_parameters[list(names).index(name)]) for name in STATION_SIX
    }
    nuisance_cdx = float(joint.recovered_parameters[list(names).index(C_DX)])
    c_dx = expected_joint[C_DX]
    six_op = operators["_six"]
    predicted = predicted_station_bias(six_op, c_dx)

    def _rows(recovered: Mapping[str, float], fit_like) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        errors = {name: recovered[name] - expected_joint[name] for name in STATION_SIX}
        sigmas = []
        covariance = fit_like.covariance_native if hasattr(fit_like, "covariance_native") else fit_like["covariance_native"]
        normal = fit_like.normal_matrix_native if hasattr(fit_like, "normal_matrix_native") else profiled_normal
        prior = _prior_array(STATION_SIX)
        for index, name in enumerate(STATION_SIX):
            variance = float(covariance[index, index])
            sigmas.append(math.sqrt(variance) if math.isfinite(variance) and variance >= 0.0 else None)
        parameter_rows = [
            {
                "name": name,
                "expected": expected_joint[name],
                "recovered": recovered[name],
                "error": errors[name],
                "sigma": sigmas[index],
                "predicted_leakage_bias": predicted.get(name),
                "error_minus_predicted_bias": errors[name] - predicted[name],
            }
            for index, name in enumerate(STATION_SIX)
        ]
        parameter_rows, capture_aggregate, _prior_rows = attach_capture_and_prior(
            parameter_rows,
            names=STATION_SIX,
            errors=[errors[name] for name in STATION_SIX],
            fit_sigmas=sigmas,
            criteria=criteria,
            normal_matrix_native=normal,
            covariance_native=covariance,
            prior_sigma_native=prior,
        )
        return parameter_rows, capture_aggregate

    ordinary_rows, ordinary_capture = _rows(ordinary_recovered, ordinary_fit)
    profiled_rows, profiled_capture = _rows(profiled_recovered, profiled_diag)
    dx_error = ordinary_recovered["ift_dx_mm"] - expected_joint["ift_dx_mm"]
    ry_error = ordinary_recovered["ift_ry_mrad"] - expected_joint["ift_ry_mrad"]
    empirical = {
        "injected_C_dx_mm": c_dx,
        "dx_error_over_C_dx": None if c_dx == 0.0 else dx_error / c_dx,
        "ry_error_over_C_dx": None if c_dx == 0.0 else ry_error / c_dx,
        "A_production_dx": six_op.A_native_per_mm["ift_dx_mm"],
        "A_production_ry": six_op.A_native_per_mm["ift_ry_mrad"],
        "A_five_dx": operators["_five"].A_native_per_mm["ift_dx_mm"],
        "A_five_ry": operators["_five"].A_native_per_mm["ift_ry_mrad"],
    }
    five_index = [list(STATION_SIX).index(name) for name in STATION_FIVE]
    five_ordinary = estimator_diagnostics(
        ordinary_fit.normal_matrix_native[np.ix_(five_index, five_index)],
        STATION_FIVE,
        parameter_scales=np.asarray(ordinary_bank["scales"], dtype=np.float64)[five_index],
        rcond=RCOND,
    )
    five_profiled = estimator_diagnostics(
        profiled_normal[np.ix_(five_index, five_index)],
        STATION_FIVE,
        parameter_scales=np.asarray(ordinary_bank["scales"], dtype=np.float64)[five_index],
        rcond=RCOND,
    )
    return {
        "observations": int(joint.used_pairs),
        "joint_fit": _fit_summary(joint, names),
        "operators": {key: value for key, value in operators.items() if not key.startswith("_")},
        "ordinary_station": {
            "fit": _fit_summary(ordinary_fit, STATION_SIX),
            "diagnostics": {key: ordinary_diag[key] for key in ordinary_diag if key != "covariance_native"},
            "parameters": ordinary_rows,
            "capture_aggregate": ordinary_capture,
            "prefit_residual_rms": _rms(ordinary_fit.response),
            "postfit_residual_rms": _rms(ordinary_fit.residual_response),
        },
        "profiled_station": {
            "diagnostics": {key: profiled_diag[key] for key in profiled_diag if key != "covariance_native"},
            "parameters": profiled_rows,
            "capture_aggregate": profiled_capture,
            "nuisance_C_dx_recovered_mm": nuisance_cdx,
            "nuisance_C_dx_not_written": True,
        },
        "estimator_comparison": {
            **compare_station_estimators(ordinary=ordinary_diag, profiled=profiled_diag),
            "five_dof_ordinary": {key: five_ordinary[key] for key in five_ordinary if key != "covariance_native"},
            "five_dof_profiled": {key: five_profiled[key] for key in five_profiled if key != "covariance_native"},
            "five_dof_profiled_full_rank": five_profiled["full_rank"],
            "five_dof_sigma_inflation": five_profiled["sigma_native"]
            and {
                name: None
                if five_ordinary["sigma_native"].get(name) in (None, 0.0) or five_profiled["sigma_native"].get(name) is None
                else float(five_profiled["sigma_native"][name]) / float(five_ordinary["sigma_native"][name])
                for name in STATION_FIVE
            },
        },
        "empirical_vs_operator": empirical,
        "predicted_ordinary_bias": predicted,
        "_six": six_op,
        "_ordinary_recovered": ordinary_recovered,
        "_profiled_recovered": profiled_recovered,
        "_expected": expected_joint,
    }


def _drop_private(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if not str(key).startswith("_")}


def analyze_truth_corner(
    banks: Sequence[Mapping[str, Any]],
    *,
    criteria,
) -> dict[str, Any]:
    pooled = _pool_layer_banks(list(banks))
    pooled_result = _fit_station_estimators(pooled, criteria=criteria)
    per_source = []
    for bank in banks:
        source_result = _fit_station_estimators(bank, criteria=criteria)
        per_source.append(
            {
                "source_id": bank["source_id"],
                "split": bank["split"],
                "observations": source_result["observations"],
                "A_production_dx": source_result["empirical_vs_operator"]["A_production_dx"],
                "A_production_ry": source_result["empirical_vs_operator"]["A_production_ry"],
                "ordinary_recovered": source_result["_ordinary_recovered"],
                "profiled_recovered": source_result["_profiled_recovered"],
                "ordinary_capture_success": None
                if source_result["ordinary_station"]["capture_aggregate"] is None
                else bool(source_result["ordinary_station"]["capture_aggregate"]["capture_success"]),
                "profiled_capture_success": None
                if source_result["profiled_station"]["capture_aggregate"] is None
                else bool(source_result["profiled_station"]["capture_aggregate"]["capture_success"]),
            }
        )
    six_names = list(STATION_SIX)
    source_spread_ordinary = {
        name: _source_spread([row["ordinary_recovered"][name] for row in per_source]) for name in six_names
    }
    source_spread_profiled = {
        name: _source_spread([row["profiled_recovered"][name] for row in per_source]) for name in six_names
    }
    return {
        "kind": "truth_selected",
        "observations": pooled_result["observations"],
        "pooled": _drop_private(pooled_result),
        "per_source": per_source,
        "source_spread_ordinary": source_spread_ordinary,
        "source_spread_profiled": source_spread_profiled,
        "A_production_dx_source_spread": _source_spread([row["A_production_dx"] for row in per_source]),
        "A_production_ry_source_spread": _source_spread([row["A_production_ry"] for row in per_source]),
    }


def _synthetic_sample_dir(synthetic_root: Path, split: str, payload_id: str) -> Path:
    path = synthetic_root / "samples" / split / payload_id
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def load_route_fd_cache(
    *,
    split: str,
    scan_root: Path,
    synthetic_root: Path,
    backbone: Path,
) -> dict[str, Any]:
    plan = _read_json(scan_root / "scan_plan.json")
    specs = _select_parameter_specs(_specs(plan), JOINT_NAMES)
    names = [str(spec["name"]) for spec in specs]
    scales = np.asarray([float(spec["severity_scale"]) for spec in specs], dtype=np.float64)
    movable = [int(value) for value in plan.get("movable_station_ids", ())]
    points = _points(plan)
    anchor_table = backbone / "selected_route_field_edge_residuals.csv"
    if not anchor_table.is_file():
        raise FileNotFoundError(anchor_table)

    def _payload(path: Path, expected: str):
        return _load_anchor_selected_payload_bank(
            path,
            expected_point=expected,
            movable=movable,
            anchor_table=anchor_table,
            q_over_p_mode=0,
        )

    print(f"  route FD {split} {ANCHOR_POINT}", flush=True)
    _anchor_summary, anchor_bank = _payload(_synthetic_sample_dir(synthetic_root, split, ANCHOR_POINT), ANCHOR_POINT)
    fd_banks = [anchor_bank]
    fd_summaries = []
    positive_points = {}
    negative_points = {}
    for name in names:
        positive_points[name] = _probe_for_parameter(points, name, "positive", ANCHOR_POINT)
        negative_points[name] = _probe_for_parameter(points, name, "negative", ANCHOR_POINT)
        print(f"  route FD {split} {positive_points[name]}", flush=True)
        _summary_p, bank_p = _payload(
            _synthetic_sample_dir(synthetic_root, split, positive_points[name]), positive_points[name]
        )
        print(f"  route FD {split} {negative_points[name]}", flush=True)
        _summary_m, bank_m = _payload(
            _synthetic_sample_dir(synthetic_root, split, negative_points[name]), negative_points[name]
        )
        fd_banks.extend((bank_p, bank_m))
        fd_summaries.append((_summary_p, _summary_m))
    anchor_values = _point_parameter_values(specs, points[ANCHOR_POINT])
    positive_values = [_point_parameter_values(specs, points[positive_points[name]])[name] for name in names]
    negative_values = [_point_parameter_values(specs, points[negative_points[name]])[name] for name in names]
    return {
        "split": split,
        "plan": plan,
        "specs": specs,
        "names": names,
        "scales": scales,
        "movable": movable,
        "points": points,
        "fd_banks": fd_banks,
        "anchor_values": np.asarray([anchor_values[name] for name in names], dtype=np.float64),
        "positive_values": np.asarray(positive_values, dtype=np.float64),
        "negative_values": np.asarray(negative_values, dtype=np.float64),
        "payload_loader": _payload,
    }


def attach_route_target(
    cache: Mapping[str, Any],
    *,
    target_point: str,
    target_synthetic_root: Path,
    target_scan_root: Path | None,
) -> dict[str, Any]:
    points = dict(cache["points"])
    if target_scan_root is not None:
        points.update(_points(_read_json(target_scan_root / "scan_plan.json")))
    if target_point not in points:
        raise ValueError(f"route-selected target {target_point} missing from scan plan")
    print(f"  route target {cache['split']} {target_point}", flush=True)
    _summary, target_bank = cache["payload_loader"](
        _synthetic_sample_dir(target_synthetic_root, str(cache["split"]), target_point),
        target_point,
    )
    banks = list(cache["fd_banks"]) + [target_bank]
    keys, residuals, covariances, overlap = align_route_selected_observations(banks)
    physical_edge_keys = [cache["fd_banks"][0][key].physical_edge_key for key in keys]
    keys, residuals, covariances, statistics = apply_observation_statistics(
        keys, residuals, covariances, physical_edge_keys, "physical_edge_deduplicated"
    )
    names = cache["names"]
    target_values = _point_parameter_values(cache["specs"], points[target_point])
    reference_values = np.asarray([target_values[name] for name in names], dtype=np.float64)
    return {
        "source_id": f"pooled_{cache['split']}",
        "split": cache["split"],
        "specs": cache["specs"],
        "names": tuple(names),
        "scales": cache["scales"],
        "anchor_values": cache["anchor_values"],
        "reference_values": reference_values,
        "target_names": (target_point,),
        "target_values": {target_point: reference_values},
        "target_residuals": {target_point: residuals[-1]},
        "positive_values": cache["positive_values"],
        "negative_values": cache["negative_values"],
        "anchor_residual": residuals[0],
        "positive_residual": np.asarray([residuals[1 + 2 * index] for index in range(len(names))]),
        "negative_residual": np.asarray([residuals[2 + 2 * index] for index in range(len(names))]),
        "reference_residual": residuals[-1],
        "covariance": covariances[0],
        "overlap": overlap,
        "observation_statistics": statistics,
        "layer_weights": np.full(len(IFT_LAYER_IDS), 1.0 / len(IFT_LAYER_IDS), dtype=np.float64),
    }


def analyze_route_corner(bank: Mapping[str, Any], *, criteria) -> dict[str, Any]:
    result = _fit_station_estimators(bank, criteria=criteria)
    return {
        "kind": "route_selected",
        "observations": result["observations"],
        "observation_statistics": bank.get("observation_statistics"),
        "pooled": _drop_private(result),
    }


def _read_empirical_truth(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = _read_json(path)
    leak = payload.get("leakage_station_absorbs_C_dx") or {}
    return {
        "path": str(path),
        "dx_error_over_C_dx": leak.get("dx_error_over_injected_C_dx"),
        "ry_error_over_C_dx": leak.get("ry_error_over_injected_C_dx"),
        "capture_success": payload.get("capture_success"),
        "injected_C_dx_mm": leak.get("injected_C_dx_mm"),
    }


def _read_empirical_route(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = _read_json(path)
    hierarchy = payload.get("hierarchy_internal_audit") or {}
    leak = hierarchy.get("leakage_station_absorbs_C_dx") or {}
    return {
        "path": str(path),
        "dx_error_over_C_dx": leak.get("dx_error_over_injected_C_dx"),
        "ry_error_over_C_dx": leak.get("ry_error_over_C_dx") if "ry_error_over_C_dx" in leak else leak.get(
            "ry_error_over_injected_C_dx"
        ),
        "capture_success": hierarchy.get("capture_success"),
        "injected_C_dx_mm": leak.get("injected_C_dx_mm"),
    }


def _budget_from_criteria(operator_json: Mapping[str, Any], criteria: Mapping[str, Any]) -> dict[str, Any]:
    from alignment.hierarchical_v1_leakage import LeakageOperatorResult

    dummy = LeakageOperatorResult(
        station_names=tuple(operator_json["station_names"]),
        contrast_name=str(operator_json["contrast_name"]),
        A_native_per_mm=dict(operator_json["A_native_per_mm_C_dx"]),
        A_per_um=dict(operator_json["A_native_per_um_C_dx"]),
        disguised_same_unit_per_um=dict(operator_json["disguised_um_or_urad_per_um_C_dx"]),
        column_cosine=dict(operator_json["column_cosine"]),
        subspace_r2=float(operator_json["subspace_r2"]),
        subspace_cosine=float(operator_json["subspace_cosine"]),
        rank=int(operator_json["rank"]),
        full_rank=bool(operator_json["full_rank"]),
        condition_scaled=operator_json.get("condition_scaled"),
        raw_condition_scaled=operator_json.get("raw_condition_scaled"),
        singular_values_scaled=tuple(operator_json.get("singular_values_scaled") or ()),
        used_prior_sigma_native=dict(operator_json.get("used_prior_sigma_native") or {}),
        normal_ss_native=np.eye(len(operator_json["station_names"])),
        normal_sc_native=np.zeros(len(operator_json["station_names"])),
        normal_cc_native=float(operator_json.get("normal_cc_native") or 1.0),
    )
    engineering = {}
    statistical = {}
    for name in dummy.station_names:
        per = criteria["per_parameter"].get(name, {})
        if per.get("engineering_tolerance") is not None:
            engineering[name] = float(per["engineering_tolerance"])
        if per.get("registered_sigma") is not None:
            statistical[name] = float(per.get("statistical_k", 3.0)) * float(per["registered_sigma"])
    return cdx_budget_from_operator(dummy, engineering_tolerance=engineering, statistical_limit=statistical)


def _collect_A_rows(corners: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for corner in corners:
        pooled = corner["result"]["pooled"]
        five = pooled["operators"]["five_dof"]
        six = pooled["operators"]["production_six_dof_survey_dz"]
        empirical = pooled["empirical_vs_operator"]
        rows.append(
            {
                "observation_kind": corner["observation_kind"],
                "split": corner["split"],
                "target": corner["target"],
                "geometry": corner["geometry"],
                "observations": corner["result"]["observations"],
                "A5_dx": five["A_native_per_mm_C_dx"]["ift_dx_mm"],
                "A5_dy": five["A_native_per_mm_C_dx"]["ift_dy_mm"],
                "A5_rx": five["A_native_per_mm_C_dx"]["ift_rx_mrad"],
                "A5_ry": five["A_native_per_mm_C_dx"]["ift_ry_mrad"],
                "A5_rz": five["A_native_per_mm_C_dx"]["ift_rz_mrad"],
                "A6_dx": six["A_native_per_mm_C_dx"]["ift_dx_mm"],
                "A6_dy": six["A_native_per_mm_C_dx"]["ift_dy_mm"],
                "A6_dz": six["A_native_per_mm_C_dx"]["ift_dz_mm"],
                "A6_rx": six["A_native_per_mm_C_dx"]["ift_rx_mrad"],
                "A6_ry": six["A_native_per_mm_C_dx"]["ift_ry_mrad"],
                "A6_rz": six["A_native_per_mm_C_dx"]["ift_rz_mrad"],
                "A5_subspace_r2": five["subspace_r2"],
                "A6_subspace_r2": six["subspace_r2"],
                "A6_subspace_cosine": six["subspace_cosine"],
                "empirical_dx_over_C_dx": empirical["dx_error_over_C_dx"],
                "empirical_ry_over_C_dx": empirical["ry_error_over_C_dx"],
                "ordinary_capture": (pooled["ordinary_station"]["capture_aggregate"] or {}).get("capture_success"),
                "profiled_capture": (pooled["profiled_station"]["capture_aggregate"] or {}).get("capture_success"),
                "ordinary_condition": pooled["estimator_comparison"]["ordinary_condition_scaled"],
                "profiled_condition": pooled["estimator_comparison"]["profiled_condition_scaled"],
                "condition_ratio": pooled["estimator_comparison"]["condition_ratio_profiled_over_ordinary"],
                "raw_condition_ratio": pooled["estimator_comparison"].get(
                    "raw_condition_ratio_profiled_over_ordinary"
                ),
                "sigma_inflation_dx": pooled["estimator_comparison"]["sigma_inflation_profiled_over_ordinary"].get(
                    "ift_dx_mm"
                ),
                "sigma_inflation_ry": pooled["estimator_comparison"]["sigma_inflation_profiled_over_ordinary"].get(
                    "ift_ry_mrad"
                ),
                "profiled_sigma_dx": pooled["estimator_comparison"]["sigma_native_profiled"].get("ift_dx_mm"),
                "profiled_sigma_ry": pooled["estimator_comparison"]["sigma_native_profiled"].get("ift_ry_mrad"),
                "profiled_full_rank": pooled["estimator_comparison"]["profiled_full_rank"],
                "ordinary_full_rank": pooled["estimator_comparison"]["ordinary_full_rank"],
                "five_dof_profiled_full_rank": pooled["estimator_comparison"].get("five_dof_profiled_full_rank"),
                "five_dof_profiled_condition": (
                    (pooled["estimator_comparison"].get("five_dof_profiled") or {}).get("condition_scaled")
                ),
                "five_dof_ordinary_condition": (
                    (pooled["estimator_comparison"].get("five_dof_ordinary") or {}).get("condition_scaled")
                ),
            }
        )
    return rows


def _range_stats(values: Sequence[float]) -> dict[str, float | None]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return {"n": 0, "mean": None, "min": None, "max": None, "spread": None, "rel_spread": None}
    mean = float(np.mean(finite))
    spread = float(max(finite) - min(finite))
    return {
        "n": len(finite),
        "mean": mean,
        "min": float(min(finite)),
        "max": float(max(finite)),
        "spread": spread,
        "rel_spread": None if mean == 0.0 else abs(spread / mean),
    }


def _decision(rows: Sequence[Mapping[str, Any]], *, budget, realizability, leftover_um) -> dict[str, Any]:
    iter00 = [row for row in rows if row["geometry"] == "iteration_00"]
    train = [row for row in iter00 if row["split"] == "train"]
    validation = [row for row in iter00 if row["split"] == "validation"]
    a_dx = _range_stats([row["A6_dx"] for row in iter00])
    a_ry = _range_stats([row["A6_ry"] for row in iter00])
    empirical_dx = _range_stats([row["empirical_dx_over_C_dx"] for row in iter00])
    empirical_ry = _range_stats([row["empirical_ry_over_C_dx"] for row in iter00])
    jacobian_matches_empirical = (
        a_dx["mean"] is not None
        and empirical_dx["mean"] is not None
        and abs(float(a_dx["mean"]) - float(empirical_dx["mean"])) / max(abs(float(empirical_dx["mean"])), 1.0e-12)
        < 0.15
        and a_dx.get("rel_spread") is not None
        and float(a_dx["rel_spread"]) < 0.20
    )
    def _all_profiled_capture(subset):
        flags = [row["profiled_capture"] for row in subset]
        return bool(flags) and all(flag is True for flag in flags)

    def _precision_ok(subset):
        dx = [row["profiled_sigma_dx"] for row in subset]
        ry = [row["profiled_sigma_ry"] for row in subset]
        if not dx or any(value is None for value in dx + ry):
            return False
        return all(float(value) <= 0.1 for value in dx) and all(float(value) <= 1.0 for value in ry)

    def _rank_ok(subset):
        return bool(subset) and all(row.get("five_dof_profiled_full_rank") is True for row in subset)

    def _condition_exploded(subset):
        ratios = [
            row.get("raw_condition_ratio") or row["condition_ratio"]
            for row in subset
            if (row.get("raw_condition_ratio") or row["condition_ratio"]) is not None
        ]
        infl = [row["sigma_inflation_dx"] for row in subset if row["sigma_inflation_dx"] is not None]
        rank_loss = any(row.get("five_dof_profiled_full_rank") is False for row in subset)
        if rank_loss:
            return True
        if not ratios:
            return True
        return max(float(value) for value in ratios) >= 50.0 or (
            bool(infl) and max(float(value) for value in infl) >= 10.0
        )

    train_rank_ok = _rank_ok(train)
    train_precision_ok = _precision_ok(train)
    train_capture_ok = _all_profiled_capture(train)
    val_capture_ok = _all_profiled_capture(validation)
    condition_exploded = _condition_exploded(train)
    dx_ry_projected_away = condition_exploded or not train_precision_ok or not train_rank_ok
    allow_physical = (
        train_rank_ok
        and train_precision_ok
        and (not condition_exploded)
        and train_capture_ok
        and val_capture_ok
    )
    close_rescue = not allow_physical
    return {
        "A6_dx_across_iteration00_corners": a_dx,
        "A6_ry_across_iteration00_corners": a_ry,
        "empirical_dx_over_C_dx_across_iteration00_corners": empirical_dx,
        "empirical_ry_over_C_dx_across_iteration00_corners": empirical_ry,
        "minus_59_minus_32_is_stable_jacobian_geometry": jacobian_matches_empirical,
        "sequential_hierarchy_statistically_realizable": realizability["sequential_hierarchy_statistically_realizable"],
        "train_profiled_full_rank": train_rank_ok,
        "train_profiled_precision_acceptable": train_precision_ok,
        "train_profiled_condition_or_sigma_exploded": condition_exploded,
        "train_profiled_capture_all_corners": train_capture_ok,
        "validation_profiled_capture_frozen_transfer": val_capture_ok,
        "dx_ry_information_projected_away": dx_ry_projected_away,
        "allow_small_physical_verification": allow_physical,
        "close_hierarchical_v1_rescue": close_rescue,
        "failure_kind": (
            "estimator_cross_talk_removable_by_nuisance_projection"
            if allow_physical
            else "incompatible_precision_of_physical_observation_space"
        ),
        "frozen_conclusion_if_closed": (
            "station 5-DoF + survey-dz and IFT 1-D C_dx are each measurable in isolation, "
            "but under the current FASER track sample/reconstruction they are not jointly "
            "solvable hierarchical parameters and must be used as independent calibration modes"
        ),
        "leftover_C_dx_um": leftover_um,
        "binding_engineering_budget": budget.get("binding_engineering"),
        "binding_statistical_budget": budget.get("binding_statistical"),
    }


def main() -> None:
    root = _root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iteration-manifest",
        default=str(root / "outputs/mc24_ift_hierarchical_v1_iteration00_trainval_physical_v1/iteration_manifest.json"),
    )
    parser.add_argument(
        "--remaining-after-station-manifest",
        default=str(
            root / "outputs/mc24_ift_hierarchical_v1_station_then_cdx_remaining_after_station_v1/iteration_manifest.json"
        ),
    )
    parser.add_argument(
        "--remaining-reverse-manifest",
        default=str(
            root / "outputs/mc24_ift_hierarchical_v1_cdx_then_station_remaining_after_station_v1/iteration_manifest.json"
        ),
    )
    parser.add_argument(
        "--synthetic-root",
        default=str(root / "outputs/mc24_ift_hierarchical_v1_iteration00_synthetic_trainval_v1"),
    )
    parser.add_argument(
        "--remaining-after-station-synthetic-root",
        default=str(
            root / "outputs/mc24_ift_hierarchical_v1_station_then_cdx_remaining_after_station_synthetic_trainval_v1"
        ),
    )
    parser.add_argument(
        "--remaining-reverse-synthetic-root",
        default=str(
            root / "outputs/mc24_ift_hierarchical_v1_cdx_then_station_remaining_after_station_synthetic_trainval_v1"
        ),
    )
    parser.add_argument(
        "--v2-backbone-train",
        default=str(root / "outputs/mc24_ift_hierarchical_v1_iteration00_v2_backbone_train_v1/iteration_00_reference"),
    )
    parser.add_argument(
        "--v2-backbone-validation",
        default=str(
            root / "outputs/mc24_ift_hierarchical_v1_iteration00_v2_backbone_validation_v1/iteration_00_reference"
        ),
    )
    parser.add_argument(
        "--scan-root-train",
        default=str(
            root
            / "outputs/mc24_ift_hierarchical_v1_iteration00_trainval_physical_v1/sources/mc24_100043_00200_00299/physical_scan"
        ),
    )
    parser.add_argument(
        "--scan-root-validation",
        default=str(
            root
            / "outputs/mc24_ift_hierarchical_v1_iteration00_trainval_physical_v1/sources/mc24_100047_00000_00049/physical_scan"
        ),
    )
    parser.add_argument(
        "--remaining-after-station-scan-root-train",
        default=str(
            root
            / "outputs/mc24_ift_hierarchical_v1_station_then_cdx_remaining_after_station_v1/sources/mc24_100043_00200_00299/physical_scan"
        ),
    )
    parser.add_argument(
        "--remaining-after-station-scan-root-validation",
        default=str(
            root
            / "outputs/mc24_ift_hierarchical_v1_station_then_cdx_remaining_after_station_v1/sources/mc24_100047_00000_00049/physical_scan"
        ),
    )
    parser.add_argument(
        "--remaining-reverse-scan-root-train",
        default=str(
            root
            / "outputs/mc24_ift_hierarchical_v1_cdx_then_station_remaining_after_station_v1/sources/mc24_100043_00200_00299/physical_scan"
        ),
    )
    parser.add_argument(
        "--remaining-reverse-scan-root-validation",
        default=str(
            root
            / "outputs/mc24_ift_hierarchical_v1_cdx_then_station_remaining_after_station_v1/sources/mc24_100047_00000_00049/physical_scan"
        ),
    )
    parser.add_argument(
        "--capture-criteria-station",
        default=str(root / "outputs/mc24_ift_5dof_survey_dz_capture_criteria_train_v1/capture_criteria.json"),
    )
    parser.add_argument(
        "--capture-criteria-cdx",
        default=str(root / "outputs/mc24_ift_layer_contrast_2d_capture_criteria_train_v1/capture_criteria.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(root / "outputs/mc24_ift_hierarchical_v1_leakage_identifiability_audit_v1"),
    )
    parser.add_argument("--skip-remaining", action="store_true")
    parser.add_argument("--skip-route-selected", action="store_true")
    parser.add_argument("--skip-truth-selected", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    station_criteria = load_capture_criteria(Path(args.capture_criteria_station))
    cdx_criteria = load_capture_criteria(Path(args.capture_criteria_cdx))
    iteration = _read_json(Path(args.iteration_manifest))
    remaining_station = None if args.skip_remaining else _read_json(Path(args.remaining_after_station_manifest))
    remaining_reverse = None if args.skip_remaining else _read_json(Path(args.remaining_reverse_manifest))
    geometries = [("iteration_00", None, Path(args.synthetic_root), None, None)]
    if remaining_station is not None:
        geometries.append(
            (
                "remaining_after_station_999493",
                remaining_station,
                Path(args.remaining_after_station_synthetic_root),
                Path(args.remaining_after_station_scan_root_train),
                Path(args.remaining_after_station_scan_root_validation),
            )
        )
    if remaining_reverse is not None:
        geometries.append(
            (
                "remaining_reverse_999500",
                remaining_reverse,
                Path(args.remaining_reverse_synthetic_root),
                Path(args.remaining_reverse_scan_root_train),
                Path(args.remaining_reverse_scan_root_validation),
            )
        )

    corners: list[dict[str, Any]] = []
    if not args.skip_truth_selected:
        by_split = {
            split: [entry for entry in _source_entries(iteration) if str(entry.get("split")) == split]
            for split in ("train", "validation")
        }
        observed_by_geometry = {"iteration_00": None}
        if remaining_station is not None:
            observed_by_geometry["remaining_after_station_999493"] = remaining_station
        if remaining_reverse is not None:
            observed_by_geometry["remaining_reverse_999500"] = remaining_reverse
        fd_cache: dict[str, dict[str, Any]] = {}
        for split, entries in by_split.items():
            for entry in entries:
                print(f"load truth FD cache {entry['source_id']}", flush=True)
                fd_cache[str(entry["source_id"])] = load_truth_fd_cache(entry, min_truth_match_fraction=MIN_TRUTH_MATCH)
            for geometry_name, observed_manifest, _synth, _sr_t, _sr_v in geometries:
                observed_entries = {
                    str(item["source_id"]): item
                    for item in _source_entries(iteration if observed_manifest is None else observed_manifest)
                    if str(item.get("split")) == split
                }
                for target in TARGETS:
                    print(f"truth {split} {geometry_name} {target}", flush=True)
                    banks = [
                        attach_truth_target(fd_cache[str(entry["source_id"])], observed_entries[str(entry["source_id"])], target)
                        for entry in entries
                    ]
                    result = analyze_truth_corner(banks, criteria=station_criteria)
                    corners.append(
                        {
                            "observation_kind": "truth_selected",
                            "split": split,
                            "target": target,
                            "geometry": geometry_name,
                            "result": result,
                        }
                    )

    if not args.skip_route_selected:
        route_scan = {"train": Path(args.scan_root_train), "validation": Path(args.scan_root_validation)}
        route_backbone = {
            "train": Path(args.v2_backbone_train),
            "validation": Path(args.v2_backbone_validation),
        }
        remaining_scan = {
            "iteration_00": {"train": None, "validation": None},
            "remaining_after_station_999493": {
                "train": Path(args.remaining_after_station_scan_root_train),
                "validation": Path(args.remaining_after_station_scan_root_validation),
            },
            "remaining_reverse_999500": {
                "train": Path(args.remaining_reverse_scan_root_train),
                "validation": Path(args.remaining_reverse_scan_root_validation),
            },
        }
        for split in ("train", "validation"):
            print(f"load route FD cache {split}", flush=True)
            cache = load_route_fd_cache(
                split=split,
                scan_root=route_scan[split],
                synthetic_root=Path(args.synthetic_root),
                backbone=route_backbone[split],
            )
            for geometry_name, _obs, synth_root, _sr_t, _sr_v in geometries:
                target_scan = remaining_scan[geometry_name][split]
                for target in TARGETS:
                    print(f"route {split} {geometry_name} {target}", flush=True)
                    bank = attach_route_target(
                        cache,
                        target_point=target,
                        target_synthetic_root=synth_root,
                        target_scan_root=target_scan,
                    )
                    result = analyze_route_corner(bank, criteria=station_criteria)
                    corners.append(
                        {
                            "observation_kind": "route_selected",
                            "split": split,
                            "target": target,
                            "geometry": geometry_name,
                            "result": result,
                        }
                    )

    rows = _collect_A_rows(corners)
    iter00_route_train_start = next(
        (
            row
            for row in rows
            if row["observation_kind"] == "route_selected"
            and row["split"] == "train"
            and row["target"] == "iteration_00_start"
            and row["geometry"] == "iteration_00"
        ),
        rows[0] if rows else None,
    )
    primary_operator = None
    if iter00_route_train_start is not None:
        matching = [
            corner
            for corner in corners
            if corner["observation_kind"] == "route_selected"
            and corner["split"] == "train"
            and corner["target"] == "iteration_00_start"
            and corner["geometry"] == "iteration_00"
        ]
        if matching:
            primary_operator = matching[0]["result"]["pooled"]["operators"]["production_six_dof_survey_dz"]
        elif corners:
            primary_operator = corners[0]["result"]["pooled"]["operators"]["production_six_dof_survey_dz"]
    budget = None if primary_operator is None else _budget_from_criteria(primary_operator, station_criteria)
    leftover_points = _read_json(
        root / "outputs/mc24_ift_hierarchical_v1_cdx_then_station_remaining_after_station_v1/remaining_hierarchical_v1_points.json"
    )
    leftover_um = [abs(float(point["alignment_parameter_values"][C_DX])) * 1.0e3 for point in leftover_points["points"]]
    registered_cdx_um = 1.0e3 * float(cdx_criteria["per_parameter"][C_DX]["registered_sigma"])
    realizability = realizability_from_budget(
        binding_engineering_um=None if budget is None else (budget["binding_engineering"] or {}).get("max_abs_C_dx_um"),
        binding_statistical_um=None if budget is None else (budget["binding_statistical"] or {}).get("max_abs_C_dx_um"),
        registered_cdx_sigma_um=registered_cdx_um,
        leftover_cdx_um=leftover_um,
    )
    empirical_existing = {
        "truth_selected": {
            f"{split}_{target.split('iteration_00_')[-1]}": _read_empirical_truth(
                root
                / f"outputs/mc24_ift_hierarchical_v1_truth_{split}_{target.split('iteration_00_')[-1]}_station_v1/hierarchical_v1_truth_selected.json"
            )
            for split in ("train", "validation")
            for target in TARGETS
        },
        "route_selected": {
            f"{split}_{target.split('iteration_00_')[-1]}": _read_empirical_route(
                root
                / f"outputs/mc24_ift_hierarchical_v1_route_selected_{split}_{target.split('iteration_00_')[-1]}_station_v1/route_selected_update.json"
            )
            for split in ("train", "validation")
            for target in TARGETS
        },
    }
    decision = _decision(rows, budget=budget or {}, realizability=realizability, leftover_um=leftover_um)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "q_over_p_mode": 0,
        "test_data_accessed": False,
        "validation_used_in_registration": False,
        "new_physical_refit": False,
        "new_layer_or_module_dof": False,
        "payload_unchanged": True,
        "joint_station_cdx_newton_for_production": False,
        "C_dx_used_only_as_analysis_nuisance_column": True,
        "jacobian_linearized_at": ANCHOR_POINT,
        "observation_statistics": "physical_edge_deduplicated",
        "station_capture_criteria": str(Path(args.capture_criteria_station).resolve()),
        "cdx_capture_criteria": str(Path(args.capture_criteria_cdx).resolve()),
        "registered_C_dx_sigma_um": registered_cdx_um,
        "reverse_path_leftover_C_dx_um": leftover_um,
        "primary_leakage_operator_production_six_dof": primary_operator,
        "cdx_budget_from_frozen_station_capture": budget,
        "realizability": realizability,
        "existing_empirical_leakage": empirical_existing,
        "corners": [
            {
                "observation_kind": corner["observation_kind"],
                "split": corner["split"],
                "target": corner["target"],
                "geometry": corner["geometry"],
                "result": corner["result"],
            }
            for corner in corners
        ],
        "decision": decision,
    }
    (output / "hierarchical_v1_leakage_audit.json").write_text(
        json.dumps(_json_ready(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if rows:
        fields = list(rows[0].keys())
        with (output / "leakage_operator_table.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "output_dir": str(output),
        "n_corners": len(corners),
        "decision": decision,
        "realizability": realizability,
        "primary_A6_per_um_disguise": None
        if primary_operator is None
        else primary_operator["disguised_um_or_urad_per_um_C_dx"],
    }
    (output / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
