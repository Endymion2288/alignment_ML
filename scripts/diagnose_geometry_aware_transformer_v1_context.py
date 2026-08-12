#!/usr/bin/env python3
"""Diagnose V1 sparse-message mechanisms on validation only.

This program intentionally refuses to load a test sample.  It runs a frozen
V1 checkpoint with deterministic message-edge/depth interventions on the
physical validation candidate graph, while reusing the checkpoint's already
frozen validation calibration and route operating point.  It is therefore a
mechanism audit, not a new architecture, calibration, or threshold search.
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

from baselines.route_assignment import RouteAssignmentConfig, adjacent_station_pairs
from datasets.physical_curriculum import CurriculumSample, load_synthetic_curriculum_manifest
from scripts.config_loader import load_yaml_with_base
from training.curriculum_mlp import CandidateSet, build_candidate_sets
from training.geometry_aware_transformer import (
    ALL_STATION_PAIRS,
    apply_frozen_transformer_calibration,
    build_transformer_graph_bundle,
    graph_bundle_summary,
    load_transformer_artifact,
)
from training.route_assignment import evaluate_adjacent_route_assignment_sets
from training.transformer_diagnostics import (
    MessageProbe,
    evaluate_probe_with_frozen_route_controls,
    predict_transformer_probe,
    score_drift_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "geometry_aware_transformer_v2_diagnostics.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
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
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    if not fields:
        fields = ["empty"]
        rows = [{"empty": ""}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_value(dict(row)))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON mapping: {path}")
    return dict(value)


def _require_mapping(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration requires mapping '{key}'")
    return dict(value)


def _parse_thresholds(
    payload: Mapping[str, Any], station_path: tuple[int, ...]
) -> dict[tuple[int, int], float]:
    raw = payload.get("thresholds")
    if not isinstance(raw, Mapping):
        raise ValueError("frozen route operating point lacks station-pair thresholds")
    result: dict[tuple[int, int], float] = {}
    for pair in adjacent_station_pairs(station_path):
        key = f"{pair[0]}->{pair[1]}"
        value = float(raw[key])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid frozen score threshold for {key}")
        result[pair] = value
    return result


def _load_frozen_v1_contract(
    root: Path,
    ablation: str,
    station_path: tuple[int, ...],
    frozen_config: Mapping[str, Any],
) -> tuple[Path, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], RouteAssignmentConfig]:
    ablation_root = (root / ablation).resolve()
    checkpoint = ablation_root / "checkpoint.pt"
    selection_path = ablation_root / "model_selection.json"
    calibration_path = ablation_root / "calibration.json"
    operating_path = ablation_root / "validation_selected_operating_point.json"
    for path in (checkpoint, selection_path, calibration_path, operating_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    selection = _read_json(selection_path)
    calibration_wrapper = _read_json(calibration_path)
    operating = _read_json(operating_path)
    if bool(frozen_config.get("require_validation_only_model_selection", True)):
        if selection.get("selection_split") != "validation_only" or selection.get("test_opened") is not False:
            raise ValueError("V1 model selection is not validation-only sealed")
        if selection.get("selected_checkpoint_sha256") != _sha256(checkpoint):
            raise ValueError("V1 checkpoint hash differs from its model-selection contract")
    calibration = calibration_wrapper.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("V1 calibration wrapper is malformed")
    if bool(frozen_config.get("require_validation_only_calibration", True)):
        if calibration_wrapper.get("fit_split") != "validation_only" or calibration_wrapper.get("test_opened") is not False:
            raise ValueError("V1 calibration is not validation-only sealed")
        if calibration.get("fit_split") != "validation_only":
            raise ValueError("V1 inner calibration is not validation-only")
    if bool(frozen_config.get("require_validation_only_operating_point", True)):
        if operating.get("selection_split") != "validation_only" or operating.get("test_opened") is not False:
            raise ValueError("V1 route operating point is not validation-only sealed")
    if operating.get("method") != "adjacent_contiguous_unit_capacity_set_packing":
        raise ValueError("V1 diagnostic expects the existing contiguous route assignment")
    thresholds = _parse_thresholds(operating, station_path)
    penalty = float(operating["unmatched_penalty"])
    if not math.isfinite(penalty):
        raise ValueError("V1 unmatched penalty is not finite")
    return (
        checkpoint,
        selection,
        dict(calibration),
        operating,
        RouteAssignmentConfig(
            score_threshold_by_pair=thresholds,
            unmatched_penalty=penalty,
            station_path=station_path,
        ),
    )


def _sample_key(sample: CurriculumSample) -> tuple[str, str]:
    return str(sample.source_id), str(sample.payload_id)


def _group_candidate_sets(
    samples: Sequence[CurriculumSample],
    sets: Sequence[CandidateSet],
    scores: Sequence[np.ndarray],
) -> list[tuple[CurriculumSample, list[CandidateSet], list[np.ndarray]]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    sample_by_key: dict[tuple[str, str], CurriculumSample] = {}
    for index, candidate_set in enumerate(sets):
        key = _sample_key(candidate_set.sample)
        grouped[key].append(index)
        sample_by_key.setdefault(key, candidate_set.sample)
    expected = {_sample_key(sample) for sample in samples}
    if set(grouped) != expected:
        raise ValueError("candidate-set grouping differs from the loaded validation manifest")
    return [
        (sample_by_key[key], [sets[index] for index in indices], [scores[index] for index in indices])
        for key, indices in sorted(grouped.items())
    ]


def _capture_success(route: Mapping[str, object], criteria: Mapping[str, Any]) -> bool:
    values = (
        ("complete_track_efficiency", ">=", "minimum_complete_track_efficiency"),
        ("complete_track_purity", ">=", "minimum_complete_track_purity"),
        ("track_fake_rate", "<=", "maximum_track_fake_rate"),
    )
    for metric, relation, criterion in values:
        observed = route.get(metric)
        if observed is None:
            return False
        bound = float(criteria[criterion])
        if relation == ">=" and float(observed) < bound:
            return False
        if relation == "<=" and float(observed) > bound:
            return False
    return True


def _route_trial_rows(
    probe: str,
    samples: Sequence[CurriculumSample],
    sets: Sequence[CandidateSet],
    scores: Sequence[np.ndarray],
    config: RouteAssignmentConfig,
    criteria: Mapping[str, Any],
    calibration_bins: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    trials: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    for sample, selected_sets, selected_scores in _group_candidate_sets(samples, sets, scores):
        evaluation = evaluate_adjacent_route_assignment_sets(
            selected_sets, selected_scores, config, calibration_bins=calibration_bins
        )
        route = evaluation["route"]
        if not isinstance(route, Mapping):  # pragma: no cover - contract assertion
            raise RuntimeError("route evaluation did not return route metrics")
        success = _capture_success(route, criteria)
        trials.append(
            {
                "probe": probe,
                "source_id": sample.source_id,
                "payload_id": sample.payload_id,
                "magnitude_mm": float(sample.magnitude_mm),
                "direction_trial": sample.direction_trial,
                "capture_success": success,
                **dict(route),
            }
        )
        details.append(
            {
                "source_id": sample.source_id,
                "payload_id": sample.payload_id,
                "magnitude_mm": float(sample.magnitude_mm),
                "direction_trial": sample.direction_trial,
                "capture_success": success,
                "evaluation": evaluation,
            }
        )
    return trials, details


def _magnitude_route_summary(
    probe: str,
    samples: Sequence[CurriculumSample],
    sets: Sequence[CandidateSet],
    scores: Sequence[np.ndarray],
    config: RouteAssignmentConfig,
    criteria: Mapping[str, Any],
    calibration_bins: int,
    trial_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    metrics = (
        "candidate_complete_truth_chain_recall",
        "score_threshold_complete_truth_chain_recall",
        "complete_track_efficiency",
        "complete_track_purity",
        "track_fake_rate",
        "missing_station_recovery",
    )
    for magnitude in sorted({float(sample.magnitude_mm) for sample in samples}):
        keys = {_sample_key(sample) for sample in samples if np.isclose(sample.magnitude_mm, magnitude)}
        indices = [index for index, candidate_set in enumerate(sets) if _sample_key(candidate_set.sample) in keys]
        evaluation = evaluate_adjacent_route_assignment_sets(
            [sets[index] for index in indices],
            [scores[index] for index in indices],
            config,
            calibration_bins=calibration_bins,
        )
        route = evaluation["route"]
        if not isinstance(route, Mapping):  # pragma: no cover - contract assertion
            raise RuntimeError("route evaluation did not return route metrics")
        trials = [row for row in trial_rows if np.isclose(float(row["magnitude_mm"]), magnitude)]
        row: dict[str, object] = {
            "probe": probe,
            "magnitude_mm": magnitude,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--v1-validation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    supplied = load_yaml_with_base(config_path)
    settings = _require_mapping(supplied, "geometry_aware_transformer_v2_diagnostics")
    contract = _require_mapping(settings, "input_contract")
    frozen_config = _require_mapping(settings, "frozen_v1")
    execution = _require_mapping(settings, "execution")
    route_settings = _require_mapping(settings, "route_control")
    if tuple(contract.get("allowed_splits", ())) != ("validation",):
        raise ValueError("V2 mechanism diagnostics must allow exactly the validation split")
    if "test" not in tuple(contract.get("forbidden_splits", ())):
        raise ValueError("V2 mechanism diagnostics must explicitly forbid the test split")
    if contract.get("physical_geometry_repropagation") is not True:
        raise ValueError("diagnostics require physical geometry repropagation")
    if int(contract.get("q_over_p_mode", -1)) != 0 or contract.get("candidate_chi2_gate") is not None:
        raise ValueError("diagnostics require the ungated physical mode-0 candidate graph")
    if str(contract.get("feature_set")) != "residual_v1" or str(contract.get("context_mode")) != "full_event":
        raise ValueError("diagnostics require the existing full-event residual_v1 graph contract")
    station_path = tuple(int(value) for value in route_settings.get("station_path", (0, 1, 2, 3)))
    if adjacent_station_pairs(station_path) != ((0, 1), (1, 2), (2, 3)):
        raise ValueError("diagnostics require the IFT -> S1 -> S2 -> S3 route path")
    criteria = _require_mapping(route_settings, "capture_success")
    probes_raw = settings.get("probes")
    if not isinstance(probes_raw, list) or not probes_raw:
        raise ValueError("diagnostics require at least one message probe")
    probes = tuple(MessageProbe.from_mapping(dict(value)) for value in probes_raw if isinstance(value, Mapping))
    if len(probes) != len(probes_raw) or len({probe.name for probe in probes}) != len(probes):
        raise ValueError("message probes must be unique mappings")
    reference_name = "depth_4_full_messages"
    if reference_name not in {probe.name for probe in probes}:
        raise ValueError(f"diagnostics require the '{reference_name}' score-drift reference probe")

    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty diagnostic output directory")
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path, samples, manifest = load_synthetic_curriculum_manifest(
        args.synthetic_manifest,
        require_all_splits=False,
        allowed_splits=("validation",),
    )
    if any(sample.split != "validation" for sample in samples):  # pragma: no cover - loader boundary
        raise RuntimeError("diagnostic loader returned a non-validation sample")
    if manifest.get("physical_geometry_repropagation") is not True or int(manifest.get("q_over_p_mode", -1)) != 0:
        raise ValueError("manifest does not certify physical mode-0 repropagation")
    loaded_sources = sorted({source for sample in samples for source in sample.source_ids})
    source_uids = {uid for sample in samples for uid in sample.source_event_uids}

    checkpoint, selection, calibration, operating, route_config = _load_frozen_v1_contract(
        Path(args.v1_validation_dir).expanduser().resolve(),
        str(frozen_config["ablation"]),
        station_path,
        frozen_config,
    )
    route_config = RouteAssignmentConfig(
        score_threshold_by_pair=route_config.score_threshold_by_pair,
        unmatched_penalty=route_config.unmatched_penalty,
        station_path=route_config.station_path,
        maximum_hypotheses=int(route_settings.get("maximum_hypotheses", 100_000)),
    )
    model, artifact = load_transformer_artifact(checkpoint, device=str(execution.get("device", "auto")))
    if artifact.context_mode != "full_event":
        raise ValueError("the selected V1 checkpoint is not the full-context model")
    if artifact.model_config.num_layers != 4 or artifact.model_config.nhead != 8:
        raise ValueError("the selected V1 checkpoint does not have the fixed V1 architecture")

    candidate_sets = build_candidate_sets(
        samples,
        ALL_STATION_PAIRS,
        chi2_gate=None,
        feature_set="residual_v1",
    )
    bundle = build_transformer_graph_bundle(candidate_sets, context_mode="full_event")
    bins = int(execution.get("calibration_bins", 15))
    device = str(execution.get("device", "auto"))
    batch_size = int(execution.get("batch_size", 64))

    by_name = {probe.name: probe for probe in probes}
    ordered_probes = (by_name[reference_name], *(probe for probe in probes if probe.name != reference_name))
    probe_results: dict[str, dict[str, object]] = {}
    all_layer_rows: list[dict[str, object]] = []
    all_state_rows: list[dict[str, object]] = []
    all_attention_rows: list[dict[str, object]] = []
    all_drift_rows: list[dict[str, object]] = []
    all_false_rows: list[dict[str, object]] = []
    all_trial_rows: list[dict[str, object]] = []
    all_summary_rows: list[dict[str, object]] = []

    reference_scores: tuple[np.ndarray, ...] | None = None
    for probe in ordered_probes:
        prediction = predict_transformer_probe(
            model,
            bundle,
            artifact.node_standardizer,
            artifact.edge_standardizer,
            probe,
            device=device,
            batch_size=batch_size,
            calibration_bins=bins,
        )
        if probe.name == reference_name:
            reference_scores = prediction.scores
        if reference_scores is None:  # pragma: no cover - ordered probe invariant
            raise RuntimeError("diagnostic reference score was not produced")
        calibrated = apply_frozen_transformer_calibration(bundle.adjacent_sets, prediction.scores, calibration)
        route_by_magnitude, false_rows = evaluate_probe_with_frozen_route_controls(
            bundle,
            calibrated,
            bins,
            route_config,
            probe_name=probe.name,
        )
        trial_rows, trial_details = _route_trial_rows(
            probe.name,
            samples,
            list(bundle.adjacent_sets),
            calibrated,
            route_config,
            criteria,
            bins,
        )
        magnitude_summary = _magnitude_route_summary(
            probe.name,
            samples,
            list(bundle.adjacent_sets),
            calibrated,
            route_config,
            criteria,
            bins,
            trial_rows,
        )
        drift = score_drift_metrics(reference_scores, prediction.scores, bundle.adjacent_sets)
        drift_rows = [
            {"probe": probe.name, "station_pair": "all", **{key: value for key, value in drift.items() if key != "by_station_pair"}},
            *[
                {"probe": probe.name, "station_pair": pair, **dict(values)}
                for pair, values in dict(drift["by_station_pair"]).items()
            ],
        ]
        all_layer_rows.extend(dict(row) for row in prediction.layer_edge_metrics)
        all_state_rows.extend(dict(row) for row in prediction.node_state_summary)
        all_attention_rows.extend(dict(row) for row in prediction.attention_summary)
        all_drift_rows.extend(drift_rows)
        all_false_rows.extend(false_rows)
        all_trial_rows.extend(trial_rows)
        all_summary_rows.extend(magnitude_summary)
        probe_results[probe.name] = {
            "probe": {
                "maximum_layers": probe.maximum_layers,
                "maximum_station_span": probe.maximum_station_span,
                "allowed_directions": list(probe.allowed_directions),
                "excluded_stations": list(probe.excluded_stations),
            },
            "message_edge_counts": prediction.message_edge_counts,
            "score_drift_from_depth_4_full_messages": drift,
            "frozen_calibrated_route_evaluation_by_magnitude": route_by_magnitude,
            "route_trial_evaluations": trial_details,
        }

    _write_csv(output_root / "layer_station_pair_edge_metrics.csv", all_layer_rows)
    _write_csv(output_root / "node_state_depth_summary.csv", all_state_rows)
    _write_csv(output_root / "attention_by_layer_station_pair.csv", all_attention_rows)
    _write_csv(output_root / "score_drift_from_full_context.csv", all_drift_rows)
    _write_csv(output_root / "fake_route_contributions.csv", all_false_rows)
    _write_csv(output_root / "frozen_route_trial_metrics.csv", all_trial_rows)
    _write_csv(output_root / "frozen_route_magnitude_summary.csv", all_summary_rows)
    _write_json(
        output_root / "diagnostic_contract.json",
        {
            "schema_version": "faser-geometry-aware-transformer-v2-mechanism-diagnostic-v1",
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "synthetic_manifest": str(manifest_path),
            "synthetic_manifest_sha256": _sha256(manifest_path),
            "loaded_splits": ["validation"],
            "forbidden_splits": ["test"],
            "test_events_loaded": False,
            "test_artifacts_opened": False,
            "validation_source_files": loaded_sources,
            "validation_source_event_uid_count": len(source_uids),
            "physical_geometry_repropagation": True,
            "q_over_p_mode": 0,
            "candidate_chi2_gate": None,
            "candidate_graph": "existing_physical_residual_v1_full_event",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "validation_model_selection": str(Path(args.v1_validation_dir).resolve() / str(frozen_config["ablation"]) / "model_selection.json"),
            "validation_model_selection_sha256": _sha256(Path(args.v1_validation_dir).resolve() / str(frozen_config["ablation"]) / "model_selection.json"),
            "validation_calibration": str(Path(args.v1_validation_dir).resolve() / str(frozen_config["ablation"]) / "calibration.json"),
            "validation_calibration_sha256": _sha256(Path(args.v1_validation_dir).resolve() / str(frozen_config["ablation"]) / "calibration.json"),
            "validation_operating_point": str(Path(args.v1_validation_dir).resolve() / str(frozen_config["ablation"]) / "validation_selected_operating_point.json"),
            "validation_operating_point_sha256": _sha256(Path(args.v1_validation_dir).resolve() / str(frozen_config["ablation"]) / "validation_selected_operating_point.json"),
            "calibration_fit_split": calibration.get("fit_split"),
            "operating_point_selection_split": operating.get("selection_split"),
            "frozen_thresholds": {f"{left}->{right}": value for (left, right), value in route_config.score_threshold_by_pair.items()},
            "frozen_unmatched_penalty": route_config.unmatched_penalty,
            "route_solver": "adjacent_contiguous_unit_capacity_set_packing",
            "graph_bundle": graph_bundle_summary(bundle),
            "probes": [probe.name for probe in probes],
            "diagnostic_only": True,
            "model_training": False,
            "calibration_refit": False,
            "threshold_selection": False,
        },
    )
    _write_json(
        output_root / "diagnostic_results.json",
        {
            "schema_version": "faser-geometry-aware-transformer-v2-mechanism-diagnostic-v1",
            "reference_probe": reference_name,
            "probe_results": probe_results,
        },
    )
    print(
        json.dumps(
            _json_value(
                {
                    "output_dir": str(output_root),
                    "probes": [probe.name for probe in probes],
                    "validation_samples": len(samples),
                    "validation_sources": loaded_sources,
                    "test_events_loaded": False,
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
