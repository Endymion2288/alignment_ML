#!/usr/bin/env python3
"""Train and freeze the route-aware Geometry-Aware Transformer V2.

Only source-disjoint train and validation physical payloads are opened.  The
script never resolves a test path, creates a test bank, or evaluates a test
event.  It freezes V2 calibration and route controls on validation after the
architecture/loss checkpoint has already been selected there.
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

from baselines.route_assignment import RouteAssignmentConfig, adjacent_station_pairs
from datasets.physical_curriculum import (
    CurriculumSample,
    load_synthetic_curriculum_manifest,
    uniform_condition_axis,
)
from models.route_transformer import RouteAwareTransformerConfig
from scripts.config_loader import load_yaml_with_base
from training.curriculum_mlp import CandidateSet, build_candidate_sets
from training.geometry_aware_transformer import (
    ADJACENT_STATION_PAIRS,
    ALL_STATION_PAIRS,
    apply_frozen_transformer_calibration,
    build_transformer_graph_bundle,
    calibrate_transformer_scores,
    candidate_score_metrics,
    source_disjoint_audit,
    stages_from_payload,
)
from training.route_assignment import evaluate_adjacent_route_assignment_sets
from training.route_aware_transformer import (
    RouteAwareTrainingConfig,
    predict_route_aware_scores,
    route_aware_artifact_summary,
    save_route_aware_transformer_artifact,
    train_route_aware_transformer_v2,
)
from training.transformer_route_selection import select_route_operating_point


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "geometry_aware_transformer_v2.yaml"


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


def _pair_mapping(raw: Mapping[object, object]) -> dict[tuple[int, int], float]:
    result: dict[tuple[int, int], float] = {}
    for key, value in raw.items():
        parts = str(key).split("->")
        if len(parts) != 2:
            raise ValueError(f"invalid station-pair threshold key '{key}'")
        result[(int(parts[0]), int(parts[1]))] = float(value)
    if set(result) != set(ADJACENT_STATION_PAIRS):
        raise ValueError("V2 initial thresholds must define exactly 0->1, 1->2, 2->3")
    return result


def _sample_key(sample: CurriculumSample) -> tuple[str, str]:
    return str(sample.source_id), str(sample.payload_id)


def _group_candidate_sets(
    samples: Sequence[CurriculumSample],
    candidate_sets: Sequence[CandidateSet],
    scores: Sequence[np.ndarray],
) -> list[tuple[CurriculumSample, list[CandidateSet], list[np.ndarray]]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    samples_by_key: dict[tuple[str, str], CurriculumSample] = {}
    for index, candidate_set in enumerate(candidate_sets):
        key = _sample_key(candidate_set.sample)
        grouped[key].append(index)
        samples_by_key.setdefault(key, candidate_set.sample)
    expected = {_sample_key(sample) for sample in samples}
    if set(grouped) != expected:
        raise ValueError("candidate-set groups differ from loaded physical samples")
    return [
        (
            samples_by_key[key],
            [candidate_sets[index] for index in indices],
            [scores[index] for index in indices],
        )
        for key, indices in sorted(grouped.items())
    ]


def _capture_success(route: Mapping[str, object], criteria: Mapping[str, object]) -> bool:
    checks = (
        ("complete_track_efficiency", ">=", "minimum_complete_track_efficiency"),
        ("complete_track_purity", ">=", "minimum_complete_track_purity"),
        ("track_fake_rate", "<=", "maximum_track_fake_rate"),
    )
    for metric, relation, key in checks:
        value = route.get(metric)
        if value is None:
            return False
        bound = float(criteria[key])
        if relation == ">=" and float(value) < bound:
            return False
        if relation == "<=" and float(value) > bound:
            return False
    return True


def _condition_fields(sample: CurriculumSample) -> dict[str, object]:
    return {
        "condition_axis": sample.condition_axis,
        "condition_value": float(sample.condition_value),
        "condition_magnitude": float(sample.curriculum_magnitude),
    }


def _route_trial_rows(
    label: str,
    samples: Sequence[CurriculumSample],
    candidate_sets: Sequence[CandidateSet],
    scores: Sequence[np.ndarray],
    route_config: RouteAssignmentConfig,
    criteria: Mapping[str, object],
    calibration_bins: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    for sample, sets, values in _group_candidate_sets(samples, candidate_sets, scores):
        evaluation = evaluate_adjacent_route_assignment_sets(
            sets, values, route_config, calibration_bins=calibration_bins
        )
        route = evaluation.get("route")
        if not isinstance(route, Mapping):
            raise RuntimeError("route evaluation is malformed")
        success = _capture_success(route, criteria)
        rows.append(
            {
                "score_stream": label,
                "source_id": sample.source_id,
                "payload_id": sample.payload_id,
                **_condition_fields(sample),
                "direction_trial": sample.direction_trial,
                "capture_success": success,
                **dict(route),
            }
        )
        details.append(
            {
                "source_id": sample.source_id,
                "payload_id": sample.payload_id,
                **_condition_fields(sample),
                "direction_trial": sample.direction_trial,
                "capture_success": success,
                "evaluation": evaluation,
            }
        )
    return rows, details


def _route_magnitude_summary(
    label: str,
    samples: Sequence[CurriculumSample],
    candidate_sets: Sequence[CandidateSet],
    scores: Sequence[np.ndarray],
    route_config: RouteAssignmentConfig,
    calibration_bins: int,
    trial_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    metrics = (
        "candidate_complete_truth_chain_recall",
        "score_threshold_complete_truth_chain_recall",
        "complete_track_efficiency",
        "complete_track_purity",
        "track_fake_rate",
        "missing_station_recovery",
    )
    result: list[dict[str, object]] = []
    condition_axis = uniform_condition_axis(samples)
    for magnitude in sorted({float(sample.curriculum_magnitude) for sample in samples}):
        keys = {
            _sample_key(sample)
            for sample in samples
            if np.isclose(sample.curriculum_magnitude, magnitude)
        }
        selected = [
            index for index, candidate_set in enumerate(candidate_sets)
            if _sample_key(candidate_set.sample) in keys
        ]
        evaluation = evaluate_adjacent_route_assignment_sets(
            [candidate_sets[index] for index in selected],
            [scores[index] for index in selected],
            route_config,
            calibration_bins=calibration_bins,
        )
        route = evaluation.get("route")
        if not isinstance(route, Mapping):
            raise RuntimeError("pooled route evaluation is malformed")
        trials = [
            row
            for row in trial_rows
            if np.isclose(float(row["condition_magnitude"]), magnitude)
        ]
        row: dict[str, object] = {
            "score_stream": label,
            "condition_axis": condition_axis,
            "condition_magnitude": magnitude,
            "direction_trials": len(trials),
            "capture_successes": int(sum(bool(trial["capture_success"]) for trial in trials)),
            "capture_fraction": float(np.mean([bool(trial["capture_success"]) for trial in trials])),
            **{f"pooled_{key}": value for key, value in route.items()},
        }
        for metric in metrics:
            values = np.asarray(
                [float(trial[metric]) for trial in trials if trial.get(metric) is not None],
                dtype=np.float64,
            )
            row[f"direction_mean_{metric}"] = None if not values.size else float(np.mean(values))
            row[f"direction_std_{metric}"] = None if not values.size else float(np.std(values))
        result.append(row)
    return result


def _selection_payload(result: object) -> dict[str, object]:
    return {
        "thresholds": {
            f"{left}->{right}": float(result.thresholds[(left, right)])
            for left, right in ADJACENT_STATION_PAIRS
        },
        "unmatched_penalty": float(result.unmatched_penalty),
        "rank": list(result.rank),
        "capture_by_magnitude": {
            str(magnitude): bool(value)
            for magnitude, value in sorted(result.capture_by_magnitude.items())
        },
        "evaluation_by_magnitude": {
            str(magnitude): value
            for magnitude, value in sorted(result.evaluation_by_magnitude.items())
        },
        "search_rows": list(result.search_rows),
        "selection_event_counts_by_magnitude": {
            str(magnitude): int(value)
            for magnitude, value in sorted(result.selection_event_counts_by_magnitude.items())
        },
        "full_validation_event_counts_by_magnitude": {
            str(magnitude): int(value)
            for magnitude, value in sorted(result.full_validation_event_counts_by_magnitude.items())
        },
        "full_validation_rerank_candidates_requested": int(
            result.full_validation_rerank_candidates_requested
        ),
        "full_validation_rerank_candidates_evaluated": int(
            result.full_validation_rerank_candidates_evaluated
        ),
    }


def _validate_input_contract(
    contract: Mapping[str, Any], manifest: Mapping[str, Any], samples: Sequence[CurriculumSample]
) -> None:
    if tuple(contract.get("allowed_splits", ())) != ("train", "validation"):
        raise ValueError("V2 training must allow exactly source-disjoint train and validation splits")
    if "test" not in tuple(contract.get("forbidden_splits", ())):
        raise ValueError("V2 training must explicitly forbid the test split")
    if contract.get("physical_geometry_repropagation") is not True:
        raise ValueError("V2 requires physical geometry repropagation")
    if int(contract.get("q_over_p_mode", -1)) != 0 or contract.get("candidate_chi2_gate") is not None:
        raise ValueError("V2 requires the ungated physical mode-0 candidate graph")
    if str(contract.get("feature_set")) != "residual_v1" or str(contract.get("context_mode")) != "full_event":
        raise ValueError("V2 requires the existing residual_v1 full-event graph contract")
    if tuple(int(value) for value in contract.get("station_path", ())) != (0, 1, 2, 3):
        raise ValueError("V2 requires the IFT -> S1 -> S2 -> S3 station path")
    expected_axis = str(contract.get("condition_axis", "translation_xy_mm"))
    legacy_magnitudes = contract.get("curriculum_magnitudes_mm")
    declared_magnitudes = contract.get("curriculum_condition_magnitudes", legacy_magnitudes)
    if not isinstance(declared_magnitudes, (list, tuple)) or not declared_magnitudes:
        raise ValueError("V2 configuration must declare its physical condition magnitudes")
    expected_magnitudes = tuple(sorted(float(value) for value in declared_magnitudes))
    if manifest.get("physical_geometry_repropagation") is not True or int(manifest.get("q_over_p_mode", -1)) != 0:
        raise ValueError("V2 manifest does not certify physical mode-0 repropagation")
    if {sample.split for sample in samples} != {"train", "validation"}:
        raise ValueError("V2 loader did not return exactly train and validation samples")
    if uniform_condition_axis(samples) != expected_axis:
        raise ValueError("V2 manifest condition axis differs from the frozen input contract")
    for split in ("train", "validation"):
        observed = tuple(
            sorted({float(sample.curriculum_magnitude) for sample in samples if sample.split == split})
        )
        if observed != expected_magnitudes:
            raise ValueError(f"V2 {split} sample set is missing a physical curriculum condition")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    supplied = load_yaml_with_base(config_path)
    settings = _require_mapping(supplied, "route_aware_transformer_v2")
    contract = _require_mapping(settings, "input_contract")
    architecture = _require_mapping(settings, "architecture")
    training = _require_mapping(settings, "training")
    calibration_config = _require_mapping(settings, "calibration")
    route_selection = _require_mapping(settings, "route_selection")
    criteria = _require_mapping(settings, "capture_success")
    raw_stages = settings.get("curriculum_stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("V2 configuration requires a non-empty curriculum_stages list")
    for key in (
        "minimum_complete_track_efficiency",
        "minimum_complete_track_purity",
        "maximum_track_fake_rate",
    ):
        value = float(criteria[key])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid V2 capture criterion '{key}'")
    if str(route_selection.get("method")) != "adjacent_contiguous_unit_capacity_set_packing":
        raise ValueError("V2 must retain the existing unit-capacity route assignment backend")
    if str(calibration_config.get("scope")) != "station_pair" or str(calibration_config.get("method")) not in {"temperature", "platt"}:
        raise ValueError("V2 calibration must be a station-pair temperature or Platt map")

    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty V2 validation output")
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path, samples, manifest = load_synthetic_curriculum_manifest(
        args.synthetic_manifest,
        require_all_splits=False,
        allowed_splits=("train", "validation"),
    )
    _validate_input_contract(contract, manifest, samples)
    source_audit = source_disjoint_audit(samples)
    train_samples = [sample for sample in samples if sample.split == "train"]
    validation_samples = [sample for sample in samples if sample.split == "validation"]
    train_sets = build_candidate_sets(
        train_samples, ALL_STATION_PAIRS, chi2_gate=None, feature_set="residual_v1"
    )
    validation_sets = build_candidate_sets(
        validation_samples, ALL_STATION_PAIRS, chi2_gate=None, feature_set="residual_v1"
    )
    train_bundle = build_transformer_graph_bundle(train_sets, context_mode="full_event")
    validation_bundle = build_transformer_graph_bundle(validation_sets, context_mode="full_event")

    model_config = RouteAwareTransformerConfig(
        node_feature_dim=17,
        edge_feature_dim=11,
        **architecture,
    )
    training_config = RouteAwareTrainingConfig(**training)
    stages = stages_from_payload(raw_stages)
    bins = int(calibration_config["bins"])
    model, artifact, history = train_route_aware_transformer_v2(
        train_bundle,
        validation_bundle,
        model_config,
        training_config,
        stages,
        bins,
    )
    checkpoint = output_root / "route_aware_transformer_v2.pt"
    save_route_aware_transformer_artifact(checkpoint, model, artifact)

    prediction = predict_route_aware_scores(
        model,
        validation_bundle,
        artifact.node_standardizer,
        artifact.edge_standardizer,
        device=training_config.device,
        batch_size=training_config.batch_size,
    )
    calibrated_scores, calibration = calibrate_transformer_scores(
        validation_bundle.adjacent_sets,
        prediction.edge_scores,
        bins,
        scope=str(calibration_config["scope"]),
        method=str(calibration_config["method"]),
    )
    initial_thresholds = _pair_mapping(_require_mapping(route_selection, "initial_thresholds"))
    selected = select_route_operating_point(
        validation_bundle.adjacent_sets,
        calibrated_scores,
        station_path=(0, 1, 2, 3),
        threshold_grid=tuple(float(value) for value in route_selection["threshold_grid"]),
        unmatched_penalties=tuple(float(value) for value in route_selection["unmatched_penalties"]),
        initial_thresholds=initial_thresholds,
        maximum_sweeps=int(route_selection["maximum_sweeps"]),
        capture_criteria=criteria,
        calibration_bins=bins,
        maximum_hypotheses=int(route_selection["maximum_hypotheses"]),
        selection_events_per_magnitude=int(route_selection["selection_events_per_magnitude"]),
        full_validation_rerank_candidates=int(route_selection["full_validation_rerank_candidates"]),
    )
    route_config = RouteAssignmentConfig(
        score_threshold_by_pair=dict(selected.thresholds),
        unmatched_penalty=float(selected.unmatched_penalty),
        station_path=(0, 1, 2, 3),
        maximum_hypotheses=int(route_selection["maximum_hypotheses"]),
    )

    # This is a strict same-checkpoint, same-calibration, same-threshold
    # control.  It suppresses only the route-derived edge correction at
    # inference, so any route-level difference cannot be due to re-tuning.
    base_scores_same_controls = apply_frozen_transformer_calibration(
        validation_bundle.adjacent_sets, prediction.base_edge_scores, calibration
    )
    streams = {
        "route_aware_v2": (tuple(calibrated_scores), prediction.edge_scores),
        "same_checkpoint_base_edge_control": (tuple(base_scores_same_controls), prediction.base_edge_scores),
    }
    trial_rows: list[dict[str, object]] = []
    magnitude_rows: list[dict[str, object]] = []
    detailed_streams: dict[str, object] = {}
    for label, (scored, raw) in streams.items():
        trials, details = _route_trial_rows(
            label,
            validation_samples,
            list(validation_bundle.adjacent_sets),
            list(scored),
            route_config,
            criteria,
            bins,
        )
        summaries = _route_magnitude_summary(
            label,
            validation_samples,
            list(validation_bundle.adjacent_sets),
            list(scored),
            route_config,
            bins,
            trials,
        )
        trial_rows.extend(trials)
        magnitude_rows.extend(summaries)
        detailed_streams[label] = {
            "raw_candidate_metrics": candidate_score_metrics(
                validation_bundle.adjacent_sets, raw, bins
            ),
            "score_stream_calibration": (
                calibration if label == "route_aware_v2" else "reused_route_aware_v2_frozen_calibration"
            ),
            "route_trial_evaluations": details,
            "route_magnitude_summary": summaries,
        }

    (output_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(_json_value(supplied), sort_keys=False), encoding="utf-8"
    )
    _write_json(
        output_root / "validation_run_contract.json",
        {
            "schema_version": "faser-route-aware-transformer-v2-validation",
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "synthetic_manifest": str(manifest_path),
            "synthetic_manifest_sha256": _sha256(manifest_path),
            "loaded_event_splits": ["train", "validation"],
            "forbidden_splits": ["test"],
            "test_events_loaded": False,
            "test_artifacts_opened": False,
            "physical_geometry_repropagation": True,
            "q_over_p_mode": 0,
            "condition_axis": uniform_condition_axis(samples),
            "candidate_chi2_gate": None,
            "candidate_graph": "existing_mode0_acts_physical_candidates_all_six_station_pairs",
            "route_candidates": "complete_chains_of_existing_adjacent_physical_edges_only",
            "route_assignment_backend": "adjacent_contiguous_unit_capacity_set_packing",
            "source_audit": source_audit,
            "same_control_calibration_refit": False,
            "same_control_threshold_selection": False,
        },
    )
    _write_json(output_root / "artifact_summary.json", route_aware_artifact_summary(artifact))
    _write_json(output_root / "training_history.json", {"history": history})
    _write_csv(
        output_root / "training_history.csv",
        [
            {key: value for key, value in row.items() if not isinstance(value, (dict, list, tuple))}
            for row in history
        ],
    )
    _write_json(
        output_root / "calibration.json",
        {
            "schema_version": "faser-route-aware-transformer-v2-validation-calibration",
            "fit_split": "validation_only",
            "test_opened": False,
            "calibration": calibration,
        },
    )
    _write_json(
        output_root / "validation_selected_operating_point.json",
        {
            "schema_version": "faser-route-aware-transformer-v2-route-operating-point",
            "selection_split": "validation_only",
            "test_opened": False,
            "method": "adjacent_contiguous_unit_capacity_set_packing",
            "selection_policy": "nominal_primary_then_validation_capture_count_then_maximum_magnitude_then_mean_route_quality",
            "capture_success_criteria": criteria,
            **_selection_payload(selected),
        },
    )
    _write_csv(output_root / "validation_route_trial_metrics.csv", trial_rows)
    _write_csv(output_root / "validation_route_magnitude_summary.csv", magnitude_rows)
    _write_json(
        output_root / "validation_results.json",
        {
            "schema_version": "faser-route-aware-transformer-v2-validation",
            "route_truth_candidate_metrics": {
                "rows": int(prediction.route_scores.size),
                "positive_rows": int(np.count_nonzero(prediction.route_labels)),
                "fake_endpoint_rows": int(np.count_nonzero(prediction.route_fake_endpoint)),
                "hard_negative_rows": int(np.count_nonzero(prediction.route_hard_negative)),
            },
            "route_edge_incidence": {
                "scored_adjacent_edges": int(prediction.route_edge_counts.size),
                "edges_with_complete_route_context": int(np.count_nonzero(prediction.route_edge_counts > 0.0)),
                "mean_complete_routes_per_contextual_edge": (
                    None
                    if not np.any(prediction.route_edge_counts > 0.0)
                    else float(np.mean(prediction.route_edge_counts[prediction.route_edge_counts > 0.0]))
                ),
            },
            "score_streams": detailed_streams,
        },
    )
    print(
        json.dumps(
            _json_value(
                {
                    "output_dir": str(output_root),
                    "checkpoint": str(checkpoint),
                    "validation_best_epoch": artifact.training_summary["best_global_epoch"],
                    "validation_selected_capture": {
                        str(magnitude): value
                        for magnitude, value in sorted(selected.capture_by_magnitude.items())
                    },
                    "test_events_loaded": False,
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
