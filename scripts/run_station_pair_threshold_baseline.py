#!/usr/bin/env python3
"""Search validation-only station-pair score thresholds for the MLP baseline.

The candidate graph is always built from the same physical mode-0 Acts
propagations used by the standard global-assignment runner.  This program does
not train a model, alter a coordinate, or use truth in its assignment solver.
Truth labels are read only after each validation assignment to select a bounded
threshold-vector operating point and to report its metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from baselines.field_chi2_matching import pair_feature_names
from baselines.global_assignment import AssignmentConfig
from baselines.mlp_pair_classifier import load_pair_classifier
from datasets.physical_curriculum import CurriculumSample, load_synthetic_curriculum_manifest
from evaluation.metrics import assess_event_matches
from scripts.run_global_assignment_mlp_baseline import (
    DEFAULT_CONFIG,
    _apply_frozen_calibration_sets,
    _calibrate_score_sets,
    _load_config,
    _plot,
    _source_split_audit,
    _write_csv,
    _write_json,
)
from training.curriculum_mlp import (
    build_candidate_sets,
    score_candidate_sets,
    score_station_pair_ensemble,
)
from training.global_assignment import (
    evaluate_station_pair_assignment_sets,
    prepare_global_assignment_context,
)
from training.station_pair_thresholds import (
    ThresholdSearchResult,
    coordinate_descent_thresholds,
    exact_pareto_thresholds,
    quality_rank,
)


def _select_magnitude(
    sets: Sequence[object], scores: Sequence[np.ndarray], magnitude: float
) -> tuple[list[object], list[np.ndarray]]:
    selected_sets: list[object] = []
    selected_scores: list[np.ndarray] = []
    for candidate_set, values in zip(sets, scores):
        if np.isclose(float(candidate_set.sample.magnitude_mm), magnitude):
            selected_sets.append(candidate_set)
            selected_scores.append(values)
    return selected_sets, selected_scores


def _threshold_vector_payload(thresholds: Mapping[tuple[int, int], float]) -> dict[str, float]:
    return {
        f"{pair[0]}->{pair[1]}": float(value)
        for pair, value in sorted(thresholds.items())
    }


def _score_threshold_recall(
    sets: Sequence[object],
    scores: Sequence[np.ndarray],
    thresholds: Mapping[tuple[int, int], float],
) -> dict[str, object]:
    """Report truth-edge retention after score masking, evaluation only."""
    possible_by_pair: dict[tuple[int, int], int] = {}
    retained_by_pair: dict[tuple[int, int], int] = {}
    for candidate_set, values in zip(sets, scores):
        pair = tuple(int(value) for value in candidate_set.station_pair)
        if pair not in thresholds:
            raise ValueError(f"missing score threshold for station pair {pair}")
        score_values = np.asarray(values, dtype=np.float64)
        labels = np.asarray(candidate_set.labels, dtype=bool)
        if score_values.shape != labels.shape:
            raise ValueError("score shape does not match candidate labels")
        available, _ = assess_event_matches(
            candidate_set.event,
            (),
            source_station=pair[0],
            target_station=pair[1],
        )
        possible_by_pair[pair] = possible_by_pair.get(pair, 0) + available.possible_matches
        retained_by_pair[pair] = retained_by_pair.get(pair, 0) + int(
            np.count_nonzero(labels & (score_values >= float(thresholds[pair])))
        )
    by_pair: dict[str, dict[str, object]] = {}
    total_possible = 0
    total_retained = 0
    for pair in sorted(possible_by_pair):
        possible = possible_by_pair[pair]
        retained = retained_by_pair.get(pair, 0)
        total_possible += possible
        total_retained += retained
        by_pair[f"{pair[0]}->{pair[1]}"] = {
            "truth_pairs_with_unique_known_endpoints": possible,
            "truth_pairs_retained_after_score_threshold": retained,
            "score_threshold_truth_recall": None if not possible else retained / possible,
        }
    return {
        "truth_pairs_with_unique_known_endpoints": total_possible,
        "truth_pairs_retained_after_score_threshold": total_retained,
        "score_threshold_truth_recall": (
            None if not total_possible else total_retained / total_possible
        ),
        "by_station_pair": by_pair,
    }


def _configs_for_thresholds(
    thresholds: Mapping[tuple[int, int], float],
    method: str,
    unmatched_penalty: float,
    sinkhorn_temperature: float,
    sinkhorn_iterations: int,
) -> dict[tuple[int, int], AssignmentConfig]:
    return {
        pair: AssignmentConfig(
            method=method,  # type: ignore[arg-type]
            score_threshold=float(threshold),
            unmatched_penalty=float(unmatched_penalty),
            sinkhorn_temperature=float(sinkhorn_temperature),
            sinkhorn_iterations=int(sinkhorn_iterations),
        )
        for pair, threshold in thresholds.items()
    }


def _evaluate_vector(
    sets: Sequence[object],
    scores: Sequence[np.ndarray],
    thresholds: Mapping[tuple[int, int], float],
    method: str,
    unmatched_penalty: float,
    sinkhorn_temperature: float,
    sinkhorn_iterations: int,
    calibration_bins: int,
    context: object | None = None,
) -> dict[str, object]:
    configs = _configs_for_thresholds(
        thresholds,
        method=method,
        unmatched_penalty=unmatched_penalty,
        sinkhorn_temperature=sinkhorn_temperature,
        sinkhorn_iterations=sinkhorn_iterations,
    )
    resolved_context = (
        prepare_global_assignment_context(sets, scores, calibration_bins)
        if context is None
        else context
    )
    result = evaluate_station_pair_assignment_sets(
        sets,
        scores,
        configs,
        calibration_bins=calibration_bins,
        context=resolved_context,
    )
    result["score_threshold_recall"] = _score_threshold_recall(sets, scores, thresholds)
    return result


def _row_from_result(
    result: Mapping[str, object],
    thresholds: Mapping[tuple[int, int], float],
    method: str,
    unmatched_penalty: float,
    sinkhorn_temperature: float,
    magnitude_mm: float,
) -> dict[str, object]:
    association = result["association"]
    unmatched = result["unmatched"]
    matrix = result["assignment_matrix"]
    recall = result["score_threshold_recall"]
    if not all(isinstance(value, Mapping) for value in (association, unmatched, matrix, recall)):
        raise RuntimeError("invalid station-pair threshold evaluation payload")
    return {
        "assignment_method": method,
        "magnitude_mm": float(magnitude_mm),
        "unmatched_penalty": float(unmatched_penalty),
        "sinkhorn_temperature": float(sinkhorn_temperature),
        **_threshold_vector_payload(thresholds),
        **dict(association),
        **dict(unmatched),
        **dict(matrix),
        "score_threshold_truth_recall": recall["score_threshold_truth_recall"],
        "truth_pairs_retained_after_score_threshold": recall[
            "truth_pairs_retained_after_score_threshold"
        ],
    }


def _station_pair_rows(
    result: Mapping[str, object],
    thresholds: Mapping[tuple[int, int], float],
    method: str,
    unmatched_penalty: float,
    sinkhorn_temperature: float,
    magnitude_mm: float,
) -> list[dict[str, object]]:
    by_pair = result["by_station_pair"]
    recall = result["score_threshold_recall"]
    if not isinstance(by_pair, Mapping) or not isinstance(recall, Mapping):
        raise RuntimeError("station-pair evaluation lacks per-pair payload")
    recall_by_pair = recall.get("by_station_pair")
    if not isinstance(recall_by_pair, Mapping):
        raise RuntimeError("station-pair score recall payload is missing")
    rows: list[dict[str, object]] = []
    for key, payload in sorted(by_pair.items()):
        if not isinstance(payload, Mapping):
            raise RuntimeError("invalid station-pair metric payload")
        candidate = payload.get("candidate")
        association = payload.get("association")
        unmatched = payload.get("unmatched")
        pair_recall = recall_by_pair.get(key)
        if not all(
            isinstance(value, Mapping)
            for value in (candidate, association, unmatched, pair_recall)
        ):
            raise RuntimeError("incomplete station-pair metric payload")
        rows.append(
            {
                "assignment_method": method,
                "magnitude_mm": float(magnitude_mm),
                "station_pair": str(key),
                "score_threshold": float(thresholds[tuple(int(value) for value in str(key).split("->"))]),
                "unmatched_penalty": float(unmatched_penalty),
                "sinkhorn_temperature": float(sinkhorn_temperature),
                **dict(candidate),
                **dict(association),
                **dict(unmatched),
                **dict(pair_recall),
            }
        )
    return rows


def _is_primary(
    result: Mapping[str, object],
    maximum_inclusive_fake_rate: float,
    minimum_inclusive_purity: float,
    minimum_efficiency: float,
) -> bool:
    association = result.get("association")
    if not isinstance(association, Mapping):
        return False
    values = (
        association.get("association_efficiency"),
        association.get("inclusive_fake_rate"),
        association.get("inclusive_association_purity"),
    )
    if any(value is None for value in values):
        return False
    efficiency, fake_rate, purity = (float(value) for value in values)
    return (
        efficiency >= minimum_efficiency
        and fake_rate <= maximum_inclusive_fake_rate
        and purity >= minimum_inclusive_purity
    )


def _validate_search_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "method",
        "unmatched_penalties",
        "sinkhorn_temperatures",
        "threshold_grid",
        "initial_thresholds",
        "maximum_sweeps",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("station_pair_threshold_baseline is missing: " + ", ".join(missing))
    method = str(config["method"])
    if method not in {"greedy", "hungarian", "dustbin_hungarian", "sinkhorn_hungarian"}:
        raise ValueError("station_pair threshold baseline has an unsupported assignment method")
    grid = tuple(sorted({float(value) for value in config["threshold_grid"]}))
    initials = tuple(float(value) for value in config["initial_thresholds"])
    if not grid or any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in grid):
        raise ValueError("station_pair threshold grid must be finite values in [0, 1]")
    if not initials or any(value not in grid for value in initials):
        raise ValueError("station_pair threshold initial values must be in the declared grid")
    penalties = tuple(float(value) for value in config["unmatched_penalties"])
    temperatures = tuple(float(value) for value in config["sinkhorn_temperatures"])
    if not penalties or not np.isfinite(penalties).all():
        raise ValueError("station_pair unmatched penalties must be finite")
    if not temperatures or any(not np.isfinite(value) or value <= 0.0 for value in temperatures):
        raise ValueError("station_pair Sinkhorn temperatures must be positive")
    sweeps = int(config["maximum_sweeps"])
    if sweeps < 1:
        raise ValueError("station_pair maximum_sweeps must be positive")
    strategy = str(config.get("search_strategy", "coordinate_descent"))
    if strategy not in {"coordinate_descent", "exact_pareto"}:
        raise ValueError("station_pair search_strategy must be coordinate_descent or exact_pareto")
    return {
        "method": method,
        "unmatched_penalties": penalties,
        "sinkhorn_temperatures": temperatures,
        "threshold_grid": grid,
        "initial_thresholds": initials,
        "maximum_sweeps": sweeps,
        "search_strategy": strategy,
    }


def _load_shared_checkpoint(
    checkpoint_argument: str | None,
    station_pairs: tuple[tuple[int, int], ...],
    feature_set: str,
    device: str,
) -> tuple[Any, Any, dict[str, str]]:
    """Load one fixed shared MLP and validate its serialized feature contract."""
    if checkpoint_argument is None:
        raise ValueError("model_topology=shared requires --checkpoint")
    checkpoint = Path(checkpoint_argument).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"shared MLP checkpoint does not exist: {checkpoint}")
    model, artifact = load_pair_classifier(checkpoint, device=device)
    expected_features = tuple(pair_feature_names(station_pairs, feature_set))
    if artifact.station_pairs != station_pairs or artifact.feature_names != expected_features:
        raise ValueError("shared checkpoint schema does not match configured station pairs and features")
    return model, artifact, {"shared": str(checkpoint)}


def _load_station_pair_ensemble(
    checkpoint_directory_argument: str | None,
    station_pairs: tuple[tuple[int, int], ...],
    feature_set: str,
    device: str,
) -> tuple[dict[tuple[int, int], Any], dict[tuple[int, int], Any], dict[str, str]]:
    """Load fixed pair-specialised MLPs without training or opening test data."""
    if checkpoint_directory_argument is None:
        raise ValueError("model_topology=station_pair_ensemble requires --checkpoint-dir")
    checkpoint_directory = Path(checkpoint_directory_argument).expanduser().resolve()
    if not checkpoint_directory.is_dir():
        raise NotADirectoryError(
            f"station-pair MLP checkpoint directory does not exist: {checkpoint_directory}"
        )
    models: dict[tuple[int, int], Any] = {}
    artifacts: dict[tuple[int, int], Any] = {}
    paths: dict[str, str] = {}
    for pair in station_pairs:
        checkpoint = checkpoint_directory / f"mlp_pair_classifier_{pair[0]}_{pair[1]}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"station-pair MLP checkpoint does not exist: {checkpoint}")
        model, artifact = load_pair_classifier(checkpoint, device=device)
        expected_features = tuple(pair_feature_names((pair,), feature_set))
        if artifact.station_pairs != (pair,) or artifact.feature_names != expected_features:
            raise ValueError(
                f"station-pair checkpoint schema does not match configured pair {pair[0]}->{pair[1]}"
            )
        models[pair] = model
        artifacts[pair] = artifact
        paths[f"{pair[0]}->{pair[1]}"] = str(checkpoint)
    return models, artifacts, paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--checkpoint",
        help="fixed shared-MLP checkpoint; required when model_topology=shared",
    )
    checkpoint_group.add_argument(
        "--checkpoint-dir",
        help=(
            "directory containing fixed mlp_pair_classifier_<source>_<target>.pt files; "
            "required when model_topology=station_pair_ensemble"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="open the sealed test split only after this validation search meets the primary point",
    )
    args = parser.parse_args()

    manifest_path, samples, manifest = load_synthetic_curriculum_manifest(args.synthetic_manifest)
    config_path = Path(args.config).expanduser().resolve()
    root, mlp, assignment = _load_config(config_path)
    search_config_raw = root.get("station_pair_threshold_baseline")
    if not isinstance(search_config_raw, Mapping):
        raise ValueError("configuration requires station_pair_threshold_baseline")
    search_config = _validate_search_config(dict(search_config_raw))
    if int(manifest.get("q_over_p_mode", -1)) != 0:
        raise ValueError("station-pair threshold baseline supports only physical mode-0 propagation")
    model_topology = str(mlp.get("model_topology", "shared"))
    if model_topology not in {"shared", "station_pair_ensemble"}:
        raise ValueError("model_topology must be shared or station_pair_ensemble")
    calibration_scope = str(mlp.get("calibration_scope", "global"))
    if calibration_scope not in {"global", "station_pair"}:
        raise ValueError("curriculum_mlp.calibration_scope must be global or station_pair")
    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty station-pair threshold output directory")
    output_root.mkdir(parents=True, exist_ok=True)

    split_audit = _source_split_audit(samples)
    stations = tuple(sorted(int(value) for value in root["refit"]["station_ids"]))
    station_pairs = tuple((left, right) for left in stations for right in stations if left < right)
    feature_set = str(mlp.get("feature_set", "residual_v1"))
    if model_topology == "shared":
        model, artifact, checkpoint_paths = _load_shared_checkpoint(
            args.checkpoint,
            station_pairs,
            feature_set,
            device=str(mlp["device"]),
        )
        models_by_pair: dict[tuple[int, int], Any] = {}
        artifacts_by_pair: dict[tuple[int, int], Any] = {}
    else:
        models_by_pair, artifacts_by_pair, checkpoint_paths = _load_station_pair_ensemble(
            args.checkpoint_dir,
            station_pairs,
            feature_set,
            device=str(mlp["device"]),
        )
        model = None
        artifact = None

    validation_samples = [sample for sample in samples if sample.split == "validation"]
    validation_sets = build_candidate_sets(
        validation_samples, station_pairs, chi2_gate=None, feature_set=feature_set
    )
    validation_raw_scores = (
        score_candidate_sets(model, artifact, validation_sets)
        if model_topology == "shared"
        else score_station_pair_ensemble(
            models_by_pair,
            artifacts_by_pair,
            validation_sets,
            station_pairs,
            feature_set=feature_set,
        )
    )
    validation_scores, calibration = _calibrate_score_sets(
        validation_sets,
        validation_raw_scores,
        calibration_bins=int(mlp["calibration_bins"]),
        scope=calibration_scope,
    )
    nominal_sets, nominal_scores = _select_magnitude(validation_sets, validation_scores, 0.0)
    if not nominal_sets:
        raise RuntimeError("validation corpus has no nominal physical payload")
    maximum_fake_rate = float(assignment["target_inclusive_fake_rate"])
    minimum_purity = float(assignment["target_inclusive_purity"])
    minimum_efficiency = float(assignment["minimum_nominal_association_efficiency"])
    calibration_bins = int(mlp["calibration_bins"])
    sinkhorn_iterations = int(assignment["sinkhorn_iterations"])
    nominal_context = prepare_global_assignment_context(
        nominal_sets, nominal_scores, calibration_bins
    )
    nominal_pair_views: dict[tuple[int, int], tuple[list[object], list[np.ndarray], object]] = {}
    for pair in station_pairs:
        pair_sets = [
            candidate_set
            for candidate_set in nominal_sets
            if tuple(int(value) for value in candidate_set.station_pair) == pair
        ]
        pair_scores = [
            values
            for candidate_set, values in zip(nominal_sets, nominal_scores)
            if tuple(int(value) for value in candidate_set.station_pair) == pair
        ]
        if not pair_sets:
            raise RuntimeError(f"nominal validation corpus lacks station pair {pair}")
        nominal_pair_views[pair] = (
            pair_sets,
            pair_scores,
            prepare_global_assignment_context(pair_sets, pair_scores, calibration_bins),
        )

    searches: list[dict[str, object]] = []
    best: tuple[ThresholdSearchResult, float, float] | None = None
    best_rank: tuple[int, float, float, float, float] | None = None
    for penalty in search_config["unmatched_penalties"]:
        for temperature in search_config["sinkhorn_temperatures"]:
            initials: Sequence[float | None] = (
                search_config["initial_thresholds"]
                if search_config["search_strategy"] == "coordinate_descent"
                else (None,)
            )
            for initial in initials:
                def evaluator(thresholds: Mapping[tuple[int, int], float]) -> dict[str, object]:
                    return _evaluate_vector(
                        nominal_sets,
                        nominal_scores,
                        thresholds,
                        method=search_config["method"],
                        unmatched_penalty=penalty,
                        sinkhorn_temperature=temperature,
                        sinkhorn_iterations=sinkhorn_iterations,
                        calibration_bins=calibration_bins,
                        context=nominal_context,
                    )

                if search_config["search_strategy"] == "coordinate_descent":
                    if initial is None:  # pragma: no cover - strategy invariant
                        raise RuntimeError("coordinate descent lacks its initial threshold")
                    result = coordinate_descent_thresholds(
                        station_pairs,
                        threshold_grid=search_config["threshold_grid"],
                        initial_thresholds={pair: initial for pair in station_pairs},
                        evaluator=evaluator,
                        maximum_inclusive_fake_rate=maximum_fake_rate,
                        minimum_inclusive_purity=minimum_purity,
                        maximum_sweeps=search_config["maximum_sweeps"],
                    )
                else:
                    def pair_evaluator(
                        pair: tuple[int, int], threshold: float
                    ) -> dict[str, object]:
                        pair_sets, pair_scores, pair_context = nominal_pair_views[pair]
                        return _evaluate_vector(
                            pair_sets,
                            pair_scores,
                            {pair: threshold},
                            method=search_config["method"],
                            unmatched_penalty=penalty,
                            sinkhorn_temperature=temperature,
                            sinkhorn_iterations=sinkhorn_iterations,
                            calibration_bins=calibration_bins,
                            context=pair_context,
                        )

                    compact_result = exact_pareto_thresholds(
                        station_pairs,
                        threshold_grid=search_config["threshold_grid"],
                        evaluator=pair_evaluator,
                        maximum_inclusive_fake_rate=maximum_fake_rate,
                        minimum_inclusive_purity=minimum_purity,
                    )
                    result = ThresholdSearchResult(
                        thresholds=compact_result.thresholds,
                        evaluation=evaluator(compact_result.thresholds),
                        trace=compact_result.trace,
                    )
                rank = quality_rank(result.evaluation, maximum_fake_rate, minimum_purity)
                association = result.evaluation["association"]
                unmatched = result.evaluation["unmatched"]
                recall = result.evaluation["score_threshold_recall"]
                if not all(isinstance(value, Mapping) for value in (association, unmatched, recall)):
                    raise RuntimeError("threshold search did not return metrics")
                searches.append(
                    {
                        "search_strategy": search_config["search_strategy"],
                        "method": search_config["method"],
                        "unmatched_penalty": penalty,
                        "sinkhorn_temperature": temperature,
                        "initial_threshold": initial,
                        "thresholds": _threshold_vector_payload(result.thresholds),
                        "rank": list(rank),
                        "association": dict(association),
                        "unmatched": dict(unmatched),
                        "score_threshold_recall": dict(recall),
                        "trace": list(result.trace),
                    }
                )
                if best_rank is None or rank > best_rank:
                    best = (result, penalty, temperature)
                    best_rank = rank
    if best is None:  # pragma: no cover - non-empty validated search grid
        raise RuntimeError("station-pair threshold search produced no result")
    selected, selected_penalty, selected_temperature = best
    primary = _is_primary(
        selected.evaluation,
        maximum_inclusive_fake_rate=maximum_fake_rate,
        minimum_inclusive_purity=minimum_purity,
        minimum_efficiency=minimum_efficiency,
    )

    _write_json(
        output_root / "resolved_config.json",
        {
            "config_source": str(config_path),
            "synthetic_manifest": str(manifest_path),
            "model_topology": model_topology,
            "checkpoints": checkpoint_paths,
            "source_split_audit": split_audit,
            "station_pairs": [list(pair) for pair in station_pairs],
            "feature_set": feature_set,
            "calibration_scope": calibration_scope,
            "physical_geometry_repropagation": True,
            "q_over_p_mode": 0,
            "transformer_started": False,
            "search": search_config,
        },
    )
    _write_json(output_root / "validation_searches.json", {"searches": searches})
    _write_json(
        output_root / "validation_selected_operating_point.json",
        {
            "selection_split": "validation_only",
            "method": search_config["method"],
            "unmatched_penalty": selected_penalty,
            "sinkhorn_temperature": selected_temperature,
            "thresholds": _threshold_vector_payload(selected.thresholds),
            "rank": list(best_rank) if best_rank is not None else None,
            "association": selected.evaluation["association"],
            "unmatched": selected.evaluation["unmatched"],
            "assignment_matrix": selected.evaluation["assignment_matrix"],
            "score_threshold_recall": selected.evaluation["score_threshold_recall"],
            "primary_constraints_satisfied": primary,
            "test_opened": False,
        },
    )
    _write_json(
        output_root / "calibration.json",
        {"fit_split": "validation_only", "calibration": calibration},
    )

    validation_rows: list[dict[str, object]] = []
    validation_pair_rows: list[dict[str, object]] = []
    for magnitude in sorted({float(candidate_set.sample.magnitude_mm) for candidate_set in validation_sets}):
        sets, scores = _select_magnitude(validation_sets, validation_scores, magnitude)
        result = _evaluate_vector(
            sets,
            scores,
            selected.thresholds,
            method=search_config["method"],
            unmatched_penalty=selected_penalty,
            sinkhorn_temperature=selected_temperature,
            sinkhorn_iterations=sinkhorn_iterations,
            calibration_bins=calibration_bins,
        )
        validation_rows.append(
            _row_from_result(
                result,
                selected.thresholds,
                method=search_config["method"],
                unmatched_penalty=selected_penalty,
                sinkhorn_temperature=selected_temperature,
                magnitude_mm=magnitude,
            )
        )
        validation_pair_rows.extend(
            _station_pair_rows(
                result,
                selected.thresholds,
                method=search_config["method"],
                unmatched_penalty=selected_penalty,
                sinkhorn_temperature=selected_temperature,
                magnitude_mm=magnitude,
            )
        )
    _write_csv(output_root / "validation_global_matching_by_magnitude.csv", validation_rows)
    _write_csv(output_root / "validation_global_matching_by_station_pair.csv", validation_pair_rows)
    _plot(output_root / "validation_global_matching_vs_misalignment.png", validation_rows)

    test_opened = False
    test_rows: list[dict[str, object]] = []
    test_pair_rows: list[dict[str, object]] = []
    if args.evaluate_test and primary:
        test_samples = [sample for sample in samples if sample.split == "test"]
        test_sets = build_candidate_sets(
            test_samples, station_pairs, chi2_gate=None, feature_set=feature_set
        )
        test_raw_scores = (
            score_candidate_sets(model, artifact, test_sets)
            if model_topology == "shared"
            else score_station_pair_ensemble(
                models_by_pair,
                artifacts_by_pair,
                test_sets,
                station_pairs,
                feature_set=feature_set,
            )
        )
        test_scores = _apply_frozen_calibration_sets(test_sets, test_raw_scores, calibration)
        for magnitude in sorted({float(candidate_set.sample.magnitude_mm) for candidate_set in test_sets}):
            sets, scores = _select_magnitude(test_sets, test_scores, magnitude)
            result = _evaluate_vector(
                sets,
                scores,
                selected.thresholds,
                method=search_config["method"],
                unmatched_penalty=selected_penalty,
                sinkhorn_temperature=selected_temperature,
                sinkhorn_iterations=sinkhorn_iterations,
                calibration_bins=calibration_bins,
            )
            test_rows.append(
                _row_from_result(
                    result,
                    selected.thresholds,
                    method=search_config["method"],
                    unmatched_penalty=selected_penalty,
                    sinkhorn_temperature=selected_temperature,
                    magnitude_mm=magnitude,
                )
            )
            test_pair_rows.extend(
                _station_pair_rows(
                    result,
                    selected.thresholds,
                    method=search_config["method"],
                    unmatched_penalty=selected_penalty,
                    sinkhorn_temperature=selected_temperature,
                    magnitude_mm=magnitude,
                )
            )
        _write_csv(output_root / "test_global_matching_by_magnitude.csv", test_rows)
        _write_csv(output_root / "test_global_matching_by_station_pair.csv", test_pair_rows)
        _plot(output_root / "test_global_matching_vs_misalignment.png", test_rows)
        test_opened = True

    _write_json(
        output_root / "metrics.json",
        {
            "method": "pairwise_mlp_station_pair_threshold_vector_plus_global_assignment",
            "model_topology": model_topology,
            "transformer_started": False,
            "physical_geometry_repropagation": True,
            "q_over_p_mode": 0,
            "source_split_audit": split_audit,
            "primary_validation_operating_point": primary,
            "test_opened": test_opened,
            "test_withheld_reason": (
                None
                if test_opened
                else (
                    "evaluate_test_not_requested"
                    if primary
                    else "no_validation_operating_point_satisfies_primary_constraints"
                )
            ),
            "test_rows": test_rows,
        },
    )
    print(
        {
            "output_dir": str(output_root),
            "primary_validation_operating_point": primary,
            "test_opened": test_opened,
            "thresholds": _threshold_vector_payload(selected.thresholds),
        }
    )


if __name__ == "__main__":
    main()
