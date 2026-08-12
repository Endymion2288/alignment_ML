#!/usr/bin/env python3
"""Evaluate V2 complete-route queries with the existing unit-capacity solver.

Only validation physical payloads are opened.  Edge scores, calibration and
adjacent thresholds are frozen from a completed V2 validation artifact.  The
new complete-route score is calibrated and its threshold/dustbin utility are
selected on validation only; no test event, artifact or source path is read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from baselines.route_assignment import RouteAssignmentConfig
from datasets.physical_curriculum import CurriculumSample, load_synthetic_curriculum_manifest
from scripts.config_loader import load_yaml_with_base
from training.curriculum_mlp import CandidateSet, build_candidate_sets
from training.geometry_aware_transformer import (
    ADJACENT_STATION_PAIRS,
    ALL_STATION_PAIRS,
    apply_frozen_transformer_calibration,
    build_transformer_graph_bundle,
    candidate_score_metrics,
    source_disjoint_audit,
)
from training.route_assignment import (
    RouteAssignmentContext,
    evaluate_adjacent_route_assignment_sets,
    prepare_route_assignment_context,
)
from training.route_aware_transformer import (
    calibrate_route_query_scores,
    load_route_aware_transformer_artifact,
    predict_route_aware_scores,
    route_query_score_maps_by_event,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "geometry_aware_transformer_v2_direct_routes.yaml"
MAGNITUDES = (0.0, 0.1, 1.0, 5.0, 10.0, 50.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    if not fields:
        fields = ["empty"]
        rows = [{"empty": ""}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_value(dict(row)))


def _require_mapping(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration requires mapping '{key}'")
    return dict(value)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON payload must be a mapping: {path}")
    return dict(payload)


def _sample_key(sample: CurriculumSample) -> tuple[str, str]:
    return str(sample.source_id), str(sample.payload_id)


def _event_key(candidate_set: CandidateSet) -> tuple[str, str, int, int]:
    return (
        str(candidate_set.sample.source_id),
        str(candidate_set.sample.payload_id),
        int(candidate_set.event.run_id),
        int(candidate_set.event.event_id),
    )


def _capture_success(route: Mapping[str, object], criteria: Mapping[str, object]) -> bool:
    checks = (
        ("complete_track_efficiency", ">=", "minimum_complete_track_efficiency"),
        ("complete_track_purity", ">=", "minimum_complete_track_purity"),
        ("track_fake_rate", "<=", "maximum_track_fake_rate"),
    )
    for metric, relation, criterion in checks:
        value = route.get(metric)
        if value is None:
            return False
        bound = float(criteria[criterion])
        if relation == ">=" and float(value) < bound:
            return False
        if relation == "<=" and float(value) > bound:
            return False
    return True


def _parse_thresholds(raw: Mapping[str, object]) -> dict[tuple[int, int], float]:
    result: dict[tuple[int, int], float] = {}
    for pair in ADJACENT_STATION_PAIRS:
        key = f"{pair[0]}->{pair[1]}"
        if key not in raw:
            raise ValueError(f"frozen V2 edge controls lack threshold {key}")
        value = float(raw[key])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid frozen V2 edge threshold {key}")
        result[pair] = value
    return result


def _load_frozen_v2(v2_dir: Path) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    contract = _read_json(v2_dir / "validation_run_contract.json")
    if contract.get("loaded_event_splits") != ["train", "validation"]:
        raise ValueError("V2 checkpoint was not produced from exactly train and validation")
    if contract.get("test_events_loaded") is not False or contract.get("test_artifacts_opened") is not False:
        raise ValueError("V2 artifact violates the sealed-test boundary")
    calibration_payload = _read_json(v2_dir / "calibration.json")
    if calibration_payload.get("fit_split") != "validation_only" or calibration_payload.get("test_opened") is not False:
        raise ValueError("V2 edge calibration is not validation-only")
    calibration = calibration_payload.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("V2 edge calibration payload is malformed")
    selected = _read_json(v2_dir / "validation_selected_operating_point.json")
    if selected.get("selection_split") != "validation_only" or selected.get("test_opened") is not False:
        raise ValueError("V2 operating point is not validation-only")
    if selected.get("method") != "adjacent_contiguous_unit_capacity_set_packing":
        raise ValueError("V2 operating point uses an unexpected route solver")
    return dict(contract), dict(calibration), dict(selected)


def _validate_input_contract(
    contract: Mapping[str, object], manifest: Mapping[str, object], samples: Sequence[CurriculumSample]
) -> None:
    if tuple(contract.get("allowed_splits", ())) != ("validation",):
        raise ValueError("direct V2 study must load only validation")
    if "test" not in tuple(contract.get("forbidden_splits", ())):
        raise ValueError("direct V2 study must explicitly forbid test")
    if manifest.get("physical_geometry_repropagation") is not True or int(manifest.get("q_over_p_mode", -1)) != 0:
        raise ValueError("direct V2 requires physical mode-0 repropagation")
    if {sample.split for sample in samples} != {"validation"}:
        raise ValueError("direct V2 loader returned a non-validation split")
    observed = tuple(sorted({float(sample.magnitude_mm) for sample in samples}))
    if observed != MAGNITUDES:
        raise ValueError("direct V2 validation set is missing a physical curriculum magnitude")


def _group_by_magnitude(
    candidate_sets: Sequence[CandidateSet], scores: Sequence[np.ndarray]
) -> dict[float, tuple[list[CandidateSet], list[np.ndarray]]]:
    grouped: dict[float, tuple[list[CandidateSet], list[np.ndarray]]] = {}
    for candidate_set, values in zip(candidate_sets, scores):
        magnitude = float(candidate_set.sample.magnitude_mm)
        sets, grouped_scores = grouped.setdefault(magnitude, ([], []))
        sets.append(candidate_set)
        grouped_scores.append(np.asarray(values, dtype=np.float64))
    if tuple(sorted(grouped)) != MAGNITUDES:
        raise ValueError("candidate-set magnitude grouping is incomplete")
    return grouped


def _rank(
    evaluations: Mapping[float, Mapping[str, object]],
    criteria: Mapping[str, object],
    threshold: float,
    penalty: float,
    context_weight: float,
) -> tuple[float, ...]:
    captures: dict[float, bool] = {}
    routes: list[Mapping[str, object]] = []
    for magnitude in MAGNITUDES:
        evaluation = evaluations[magnitude]
        route = evaluation.get("route")
        if not isinstance(route, Mapping):
            raise RuntimeError("direct V2 route evaluation is malformed")
        captures[magnitude] = _capture_success(route, criteria)
        routes.append(route)

    def mean_metric(name: str) -> float:
        values = [float(route[name]) for route in routes if route.get(name) is not None]
        return float(np.mean(values)) if values else -math.inf

    captured = [magnitude for magnitude, value in captures.items() if value]
    return (
        float(captures[0.0]),
        float(sum(captures.values())),
        max(captured) if captured else -1.0,
        mean_metric("complete_track_efficiency"),
        mean_metric("complete_track_purity"),
        -mean_metric("track_fake_rate"),
        -float(context_weight),
        float(threshold),
        -float(penalty),
    )


def _evaluate_all_magnitudes(
    grouped: Mapping[float, tuple[list[CandidateSet], list[np.ndarray]]],
    contexts: Mapping[float, RouteAssignmentContext],
    route_score_maps: Mapping[tuple[str, str, int, int], Mapping[tuple[int, ...], float]] | None,
    config: RouteAssignmentConfig,
    bins: int,
) -> dict[float, dict[str, object]]:
    result: dict[float, dict[str, object]] = {}
    for magnitude in MAGNITUDES:
        sets, scores = grouped[magnitude]
        maps = None if route_score_maps is None else _maps_for_sets(sets, route_score_maps)
        result[magnitude] = evaluate_adjacent_route_assignment_sets(
            sets,
            scores,
            config,
            calibration_bins=bins,
            context=contexts[magnitude],
            complete_route_scores_by_event=maps,
        )
    return result


def _select_direct_controls(
    grouped: Mapping[float, tuple[list[CandidateSet], list[np.ndarray]]],
    contexts: Mapping[float, RouteAssignmentContext],
    route_score_maps: Mapping[tuple[str, str, int, int], Mapping[tuple[int, ...], float]],
    edge_thresholds: Mapping[tuple[int, int], float],
    threshold_grid: Sequence[float],
    penalty_grid: Sequence[float],
    composition: str,
    context_weight_grid: Sequence[float],
    criteria: Mapping[str, object],
    maximum_hypotheses: int,
    bins: int,
) -> tuple[RouteAssignmentConfig, dict[float, dict[str, object]], list[dict[str, object]]]:
    best: tuple[tuple[float, ...], RouteAssignmentConfig, dict[float, dict[str, object]]] | None = None
    rows: list[dict[str, object]] = []
    if composition not in {"replace", "residual"}:
        raise ValueError("complete_route_score_composition must be 'replace' or 'residual'")
    for threshold in threshold_grid:
        if not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("direct route-query threshold grid contains an invalid value")
        for penalty in penalty_grid:
            for context_weight in context_weight_grid:
                if not math.isfinite(float(context_weight)) or float(context_weight) < 0.0:
                    raise ValueError("direct route-query context-weight grid contains an invalid value")
                config = RouteAssignmentConfig(
                    score_threshold_by_pair=dict(edge_thresholds),
                    unmatched_penalty=float(penalty),
                    station_path=(0, 1, 2, 3),
                    maximum_hypotheses=maximum_hypotheses,
                    complete_route_score_threshold=float(threshold),
                    complete_route_score_composition=composition,
                    complete_route_context_weight=float(context_weight),
                )
                try:
                    evaluations = _evaluate_all_magnitudes(
                        grouped, contexts, route_score_maps, config, bins
                    )
                except RuntimeError as error:
                    if "route hypothesis count exceeds maximum_hypotheses" not in str(error):
                        raise
                    rows.append(
                        {
                            "complete_route_score_threshold": float(threshold),
                            "unmatched_penalty": float(penalty),
                            "complete_route_score_composition": composition,
                            "complete_route_context_weight": float(context_weight),
                            "invalid_reason": "route_hypothesis_limit_exceeded",
                        }
                    )
                    continue
                rank = _rank(
                    evaluations,
                    criteria,
                    float(threshold),
                    float(penalty),
                    float(context_weight),
                )
                captures = {
                    str(magnitude): _capture_success(evaluations[magnitude]["route"], criteria)
                    for magnitude in MAGNITUDES
                }
                rows.append(
                    {
                        "complete_route_score_threshold": float(threshold),
                        "unmatched_penalty": float(penalty),
                        "complete_route_score_composition": composition,
                        "complete_route_context_weight": float(context_weight),
                        "rank": list(rank),
                        "capture_by_magnitude": captures,
                        "complete_track_efficiency_by_magnitude": {
                            str(magnitude): evaluations[magnitude]["route"]["complete_track_efficiency"]
                            for magnitude in MAGNITUDES
                        },
                        "complete_track_purity_by_magnitude": {
                            str(magnitude): evaluations[magnitude]["route"]["complete_track_purity"]
                            for magnitude in MAGNITUDES
                        },
                        "track_fake_rate_by_magnitude": {
                            str(magnitude): evaluations[magnitude]["route"]["track_fake_rate"]
                            for magnitude in MAGNITUDES
                        },
                    }
                )
                candidate = (rank, config, evaluations)
                if best is None or candidate[0] > best[0]:
                    best = candidate
    if best is None:
        raise RuntimeError("direct V2 route-query control grid produced no valid configuration")
    return best[1], best[2], rows


def _group_trial_sets(
    samples: Sequence[CurriculumSample], candidate_sets: Sequence[CandidateSet], scores: Sequence[np.ndarray]
) -> list[tuple[CurriculumSample, list[CandidateSet], list[np.ndarray]]]:
    indices_by_sample: dict[tuple[str, str], list[int]] = defaultdict(list)
    sample_by_key: dict[tuple[str, str], CurriculumSample] = {}
    for index, candidate_set in enumerate(candidate_sets):
        key = _sample_key(candidate_set.sample)
        indices_by_sample[key].append(index)
        sample_by_key.setdefault(key, candidate_set.sample)
    expected = {_sample_key(sample) for sample in samples}
    if set(indices_by_sample) != expected:
        raise ValueError("route candidate sets do not cover every validation payload")
    return [
        (
            sample_by_key[key],
            [candidate_sets[index] for index in indices],
            [scores[index] for index in indices],
        )
        for key, indices in sorted(indices_by_sample.items())
    ]


def _maps_for_sets(
    candidate_sets: Sequence[CandidateSet],
    route_score_maps: Mapping[tuple[str, str, int, int], Mapping[tuple[int, ...], float]],
) -> dict[tuple[str, str, int, int], Mapping[tuple[int, ...], float]]:
    keys = {_event_key(candidate_set) for candidate_set in candidate_sets}
    if keys != set(route_score_maps).intersection(keys):
        raise ValueError("direct route-query scores do not cover a validation event")
    return {key: route_score_maps[key] for key in keys}


def _trial_and_magnitude_rows(
    label: str,
    samples: Sequence[CurriculumSample],
    candidate_sets: Sequence[CandidateSet],
    scores: Sequence[np.ndarray],
    config: RouteAssignmentConfig,
    route_score_maps: Mapping[tuple[str, str, int, int], Mapping[tuple[int, ...], float]] | None,
    criteria: Mapping[str, object],
    bins: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    trial_rows: list[dict[str, object]] = []
    trial_details: list[dict[str, object]] = []
    for sample, sets, values in _group_trial_sets(samples, candidate_sets, scores):
        maps = None if route_score_maps is None else _maps_for_sets(sets, route_score_maps)
        evaluation = evaluate_adjacent_route_assignment_sets(
            sets,
            values,
            config,
            calibration_bins=bins,
            complete_route_scores_by_event=maps,
        )
        route = evaluation["route"]
        success = _capture_success(route, criteria)
        row = {
            "score_stream": label,
            "source_id": sample.source_id,
            "payload_id": sample.payload_id,
            "magnitude_mm": float(sample.magnitude_mm),
            "direction_trial": sample.direction_trial,
            "capture_success": success,
            **dict(route),
        }
        route_query = evaluation.get("route_query")
        if isinstance(route_query, Mapping):
            row.update({f"route_query_{key}": value for key, value in route_query.items()})
        trial_rows.append(row)
        trial_details.append(
            {
                "score_stream": label,
                "source_id": sample.source_id,
                "payload_id": sample.payload_id,
                "magnitude_mm": float(sample.magnitude_mm),
                "direction_trial": sample.direction_trial,
                "capture_success": success,
                "evaluation": evaluation,
            }
        )
    grouped = _group_by_magnitude(candidate_sets, scores)
    contexts = {
        magnitude: prepare_route_assignment_context(sets, values, bins, (0, 1, 2, 3))
        for magnitude, (sets, values) in grouped.items()
    }
    evaluations = _evaluate_all_magnitudes(grouped, contexts, route_score_maps, config, bins)
    magnitude_rows: list[dict[str, object]] = []
    for magnitude in MAGNITUDES:
        evaluation = evaluations[magnitude]
        route = evaluation["route"]
        magnitude_trials = [
            row for row in trial_rows if np.isclose(float(row["magnitude_mm"]), magnitude)
        ]
        row: dict[str, object] = {
            "score_stream": label,
            "magnitude_mm": magnitude,
            "direction_trials": len(magnitude_trials),
            "capture_successes": int(sum(bool(item["capture_success"]) for item in magnitude_trials)),
            "capture_fraction": float(
                np.mean([bool(item["capture_success"]) for item in magnitude_trials])
            ),
            **{f"pooled_{key}": value for key, value in route.items()},
        }
        query = evaluation.get("route_query")
        if isinstance(query, Mapping):
            row.update({f"pooled_route_query_{key}": value for key, value in query.items()})
        magnitude_rows.append(row)
    return trial_rows, magnitude_rows, trial_details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--v2-validation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    supplied = load_yaml_with_base(config_path)
    settings = _require_mapping(supplied, "route_aware_transformer_v2_direct_routes")
    input_contract = _require_mapping(settings, "input_contract")
    calibration_config = _require_mapping(settings, "route_query_calibration")
    selection = _require_mapping(settings, "route_query_selection")
    criteria = _require_mapping(settings, "capture_success")
    if str(calibration_config.get("method")) not in {"temperature", "platt"}:
        raise ValueError("direct V2 requires temperature or Platt route-query calibration")
    composition = str(selection.get("complete_route_score_composition", "replace"))
    context_weight_grid = tuple(
        float(value)
        for value in selection.get("complete_route_context_weights", [1.0])
    )
    if not context_weight_grid:
        raise ValueError("direct V2 requires at least one route-query context weight")
    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty direct V2 validation output")
    output_root.mkdir(parents=True, exist_ok=True)

    v2_dir = Path(args.v2_validation_dir).expanduser().resolve()
    frozen_contract, frozen_edge_calibration, frozen_selected = _load_frozen_v2(v2_dir)
    edge_thresholds = _parse_thresholds(_require_mapping(frozen_selected, "thresholds"))
    checkpoint = v2_dir / "route_aware_transformer_v2.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"V2 checkpoint is absent: {checkpoint}")
    manifest_path, samples, manifest = load_synthetic_curriculum_manifest(
        args.synthetic_manifest,
        require_all_splits=False,
        allowed_splits=("validation",),
    )
    _validate_input_contract(input_contract, manifest, samples)
    source_audit = source_disjoint_audit(samples)
    candidate_sets = build_candidate_sets(
        samples, ALL_STATION_PAIRS, chi2_gate=None, feature_set="residual_v1"
    )
    bundle = build_transformer_graph_bundle(candidate_sets, context_mode="full_event")
    model, artifact = load_route_aware_transformer_artifact(checkpoint, device=str(args.device))
    prediction = predict_route_aware_scores(
        model,
        bundle,
        artifact.node_standardizer,
        artifact.edge_standardizer,
        device=str(args.device),
        batch_size=32,
    )
    bins = int(calibration_config["bins"])
    edge_scores = apply_frozen_transformer_calibration(
        bundle.adjacent_sets, prediction.edge_scores, frozen_edge_calibration
    )
    route_scores, route_calibration = calibrate_route_query_scores(
        prediction,
        bins=bins,
        method=str(calibration_config["method"]),
    )
    route_score_maps = route_query_score_maps_by_event(prediction.route_score_sets, route_scores)
    grouped = _group_by_magnitude(list(bundle.adjacent_sets), edge_scores)
    contexts = {
        magnitude: prepare_route_assignment_context(sets, values, bins, (0, 1, 2, 3))
        for magnitude, (sets, values) in grouped.items()
    }
    selected_config, selected_evaluations, search_rows = _select_direct_controls(
        grouped,
        contexts,
        route_score_maps,
        edge_thresholds,
        tuple(float(value) for value in selection["complete_route_score_thresholds"]),
        tuple(float(value) for value in selection["unmatched_penalties"]),
        composition,
        context_weight_grid,
        criteria,
        int(selection["maximum_hypotheses"]),
        bins,
    )
    # Strict shared-control contract: edge scores, edge calibration, edge
    # thresholds and dustbin penalty are identical.  Only use of the V2
    # complete-route score is disabled.
    edge_only_control = RouteAssignmentConfig(
        score_threshold_by_pair=dict(selected_config.score_threshold_by_pair),
        unmatched_penalty=float(selected_config.unmatched_penalty),
        station_path=(0, 1, 2, 3),
        maximum_hypotheses=selected_config.maximum_hypotheses,
        complete_route_score_threshold=None,
    )
    direct_trials, direct_magnitudes, direct_details = _trial_and_magnitude_rows(
        "v2_direct_route_query",
        samples,
        list(bundle.adjacent_sets),
        list(edge_scores),
        selected_config,
        route_score_maps,
        criteria,
        bins,
    )
    control_trials, control_magnitudes, control_details = _trial_and_magnitude_rows(
        "same_checkpoint_edge_route_control",
        samples,
        list(bundle.adjacent_sets),
        list(edge_scores),
        edge_only_control,
        None,
        criteria,
        bins,
    )
    captures = {
        str(magnitude): _capture_success(selected_evaluations[magnitude]["route"], criteria)
        for magnitude in MAGNITUDES
    }
    (output_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(_json_value(supplied), sort_keys=False), encoding="utf-8"
    )
    _write_json(
        output_root / "validation_run_contract.json",
        {
            "schema_version": "faser-route-aware-transformer-v2-direct-route-validation",
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "synthetic_manifest": str(manifest_path),
            "synthetic_manifest_sha256": _sha256(manifest_path),
            "v2_validation_dir": str(v2_dir),
            "v2_checkpoint": str(checkpoint),
            "v2_checkpoint_sha256": _sha256(checkpoint),
            "loaded_event_splits": ["validation"],
            "forbidden_splits": ["test"],
            "test_events_loaded": False,
            "test_artifacts_opened": False,
            "physical_geometry_repropagation": True,
            "q_over_p_mode": 0,
            "candidate_chi2_gate": None,
            "candidate_graph": "existing_mode0_acts_physical_candidates_all_six_station_pairs",
            "route_assignment_backend": "adjacent_contiguous_unit_capacity_set_packing",
            "edge_calibration_refit": False,
            "edge_threshold_selection_refit": False,
            "route_query_calibration_fit_split": "validation_only",
            "route_query_threshold_selection_split": "validation_only",
            "route_query_score_composition": composition,
            "route_query_context_weights": list(context_weight_grid),
            "same_checkpoint_control_edge_calibration_refit": False,
            "same_checkpoint_control_edge_threshold_selection_refit": False,
            "same_checkpoint_control_dustbin_penalty_selection_refit": False,
            "source_audit": source_audit,
            "frozen_v2_contract": frozen_contract,
        },
    )
    _write_json(
        output_root / "route_query_calibration.json",
        {
            "schema_version": "faser-route-aware-transformer-v2-direct-route-calibration",
            "fit_split": "validation_only",
            "test_opened": False,
            "calibration": route_calibration,
        },
    )
    _write_json(
        output_root / "validation_selected_operating_point.json",
        {
            "schema_version": "faser-route-aware-transformer-v2-direct-route-operating-point",
            "selection_split": "validation_only",
            "test_opened": False,
            "method": "adjacent_contiguous_unit_capacity_set_packing_with_complete_route_query_utility",
            "capture_success_criteria": criteria,
            "frozen_edge_thresholds": {
                f"{left}->{right}": value for (left, right), value in edge_thresholds.items()
            },
            "complete_route_score_threshold": selected_config.complete_route_score_threshold,
            "complete_route_score_composition": selected_config.complete_route_score_composition,
            "complete_route_context_weight": selected_config.complete_route_context_weight,
            "unmatched_penalty": selected_config.unmatched_penalty,
            "capture_by_magnitude": captures,
            "rank": list(
                _rank(
                    selected_evaluations,
                    criteria,
                    float(selected_config.complete_route_score_threshold),
                    float(selected_config.unmatched_penalty),
                    float(selected_config.complete_route_context_weight),
                )
            ),
            "search_rows": search_rows,
        },
    )
    _write_csv(output_root / "validation_route_trial_metrics.csv", [*direct_trials, *control_trials])
    _write_csv(
        output_root / "validation_route_magnitude_summary.csv", [*direct_magnitudes, *control_magnitudes]
    )
    _write_json(
        output_root / "validation_results.json",
        {
            "schema_version": "faser-route-aware-transformer-v2-direct-route-validation",
            "edge_candidate_metrics": candidate_score_metrics(bundle.adjacent_sets, edge_scores, bins),
            "route_query_raw_metrics": {
                "scores": int(prediction.route_scores.size),
                "labels": int(prediction.route_labels.size),
            },
            "route_query_calibration": route_calibration,
            "score_streams": {
                "v2_direct_route_query": {
                    "route_trial_evaluations": direct_details,
                    "route_magnitude_summary": direct_magnitudes,
                },
                "same_checkpoint_edge_route_control": {
                    "route_trial_evaluations": control_details,
                    "route_magnitude_summary": control_magnitudes,
                },
            },
        },
    )
    print(
        json.dumps(
            _json_value(
                {
                    "output_dir": str(output_root),
                    "complete_route_score_threshold": selected_config.complete_route_score_threshold,
                    "complete_route_score_composition": selected_config.complete_route_score_composition,
                    "complete_route_context_weight": selected_config.complete_route_context_weight,
                    "unmatched_penalty": selected_config.unmatched_penalty,
                    "validation_selected_capture": captures,
                    "test_events_loaded": False,
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
