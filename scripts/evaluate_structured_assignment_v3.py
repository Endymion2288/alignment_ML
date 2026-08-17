#!/usr/bin/env python3
"""Select V3 calibration and route controls on validation physical data only.

The checkpoint is fixed before this script starts.  It opens only validation
synthetic samples, fits station-pair edge calibration there, and chooses the
edge thresholds, dustbin utility, and structured-route utility temperature on
that same declared validation set.  It never opens test sources or artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from baselines.route_assignment import RouteAssignmentConfig
from datasets.physical_curriculum import CurriculumSample, load_synthetic_curriculum_manifest
from evaluation.pairwise_metrics import binary_calibration
from scripts.config_loader import load_yaml_with_base
from scripts.evaluate_route_aware_transformer_v2_direct_routes import (
    MAGNITUDES,
    _capture_success,
    _group_by_magnitude,
    _rank,
    _select_direct_controls,
    _trial_and_magnitude_rows,
    _write_csv,
    _write_json,
)
from training.curriculum_mlp import CandidateSet, build_candidate_sets
from training.geometry_aware_transformer import (
    ALL_STATION_PAIRS,
    calibrate_transformer_scores,
    build_transformer_graph_bundle,
    candidate_score_metrics,
    source_disjoint_audit,
)
from training.route_assignment import prepare_route_assignment_context
from training.structured_assignment import (
    StructuredRouteScoreSet,
    load_structured_assignment_artifact,
    predict_structured_assignment_scores,
)
from training.transformer_route_selection import select_route_operating_point


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "geometry_aware_transformer_v3.yaml"


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


def _require_mapping(parent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration requires mapping '{key}'")
    return dict(value)


def _pair_mapping(raw: Mapping[str, object]) -> dict[tuple[int, int], float]:
    result: dict[tuple[int, int], float] = {}
    for pair in ((0, 1), (1, 2), (2, 3)):
        key = f"{pair[0]}->{pair[1]}"
        if key not in raw:
            raise ValueError(f"V3 route selection lacks initial threshold {key}")
        value = float(raw[key])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"V3 initial threshold {key} is invalid")
        result[pair] = value
    return result


def _route_utility_score_maps(
    score_sets: Sequence[StructuredRouteScoreSet], utility_temperature: float
) -> dict[tuple[str, str, int, int], dict[tuple[int, ...], float]]:
    """Convert frozen structured utilities to solver probabilities at one scale."""
    if not math.isfinite(utility_temperature) or utility_temperature <= 0.0:
        raise ValueError("V3 route utility temperature must be finite and positive")
    result: dict[tuple[str, str, int, int], dict[tuple[int, ...], float]] = {}
    for score_set in score_sets:
        key = (
            str(score_set.sample.source_id),
            str(score_set.sample.payload_id),
            int(score_set.event.run_id),
            int(score_set.event.event_id),
        )
        if key in result:
            raise ValueError("duplicate V3 route-utility event key")
        scaled = np.clip(np.asarray(score_set.utilities, dtype=np.float64) / utility_temperature, -60.0, 60.0)
        scores = 1.0 / (1.0 + np.exp(-scaled))
        table: dict[tuple[int, ...], float] = {}
        for endpoints, score in zip(score_set.endpoint_indices, scores):
            endpoint_key = tuple(int(value) for value in endpoints)
            if endpoint_key in table:
                raise ValueError("duplicate V3 physical complete-route endpoint tuple")
            if len(endpoint_key) != 4 or tuple(int(score_set.event.station_id[index]) for index in endpoint_key) != (0, 1, 2, 3):
                raise ValueError("V3 route endpoint stations are not IFT -> S1 -> S2 -> S3")
            table[endpoint_key] = float(score)
        result[key] = table
    return result


def _validate_training_contract(training_dir: Path) -> dict[str, object]:
    contract_path = training_dir / "validation_run_contract.json"
    with contract_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("V3 training contract is malformed")
    if payload.get("loaded_event_splits") != ["train", "validation"]:
        raise ValueError("V3 checkpoint did not use exactly train and validation")
    if payload.get("test_events_loaded") is not False or payload.get("test_artifacts_opened") is not False:
        raise ValueError("V3 checkpoint violates the sealed-test boundary")
    if payload.get("training_objective") != "exact_loss_augmented_structured_margin":
        raise ValueError("checkpoint was not trained with the V3 structured objective")
    return dict(payload)


def _validate_input_contract(
    contract: Mapping[str, object], manifest: Mapping[str, object], samples: Sequence[CurriculumSample]
) -> None:
    if tuple(contract.get("allowed_splits", ())) != ("train", "validation"):
        raise ValueError("V3 configuration must retain train/validation-only training scope")
    if "test" not in tuple(contract.get("forbidden_splits", ())):
        raise ValueError("V3 configuration must explicitly forbid test")
    if manifest.get("physical_geometry_repropagation") is not True or int(manifest.get("q_over_p_mode", -1)) != 0:
        raise ValueError("V3 evaluator requires physical mode-0 repropagation")
    if {sample.split for sample in samples} != {"validation"}:
        raise ValueError("V3 evaluator must load only validation samples")
    observed = tuple(sorted({float(sample.magnitude_mm) for sample in samples}))
    if observed != MAGNITUDES:
        raise ValueError("V3 validation samples are missing a physical curriculum magnitude")


def _maps_for_sets(
    candidate_sets: Sequence[CandidateSet],
    route_maps: Mapping[tuple[str, str, int, int], Mapping[tuple[int, ...], float]],
) -> dict[tuple[str, str, int, int], Mapping[tuple[int, ...], float]]:
    keys = {
        (
            str(candidate_set.sample.source_id),
            str(candidate_set.sample.payload_id),
            int(candidate_set.event.run_id),
            int(candidate_set.event.event_id),
        )
        for candidate_set in candidate_sets
    }
    if keys != set(route_maps).intersection(keys):
        raise ValueError("V3 route utility map does not cover every validation event")
    return {key: route_maps[key] for key in keys}


def _evaluate_by_magnitude(
    grouped: Mapping[float, tuple[list[CandidateSet], list[np.ndarray]]],
    contexts: Mapping[float, object],
    route_maps: Mapping[tuple[str, str, int, int], Mapping[tuple[int, ...], float]],
    config: RouteAssignmentConfig,
    bins: int,
) -> dict[float, dict[str, object]]:
    from training.route_assignment import evaluate_adjacent_route_assignment_sets

    result: dict[float, dict[str, object]] = {}
    for magnitude in MAGNITUDES:
        sets, scores = grouped[magnitude]
        result[magnitude] = evaluate_adjacent_route_assignment_sets(
            sets,
            scores,
            config,
            calibration_bins=bins,
            context=contexts[magnitude],
            complete_route_scores_by_event=_maps_for_sets(sets, route_maps),
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    supplied = load_yaml_with_base(config_path)
    settings = _require_mapping(supplied, "structured_assignment_v3")
    contract = _require_mapping(settings, "input_contract")
    calibration_config = _require_mapping(settings, "calibration")
    selection = _require_mapping(settings, "route_selection")
    criteria = _require_mapping(settings, "capture_success")
    if str(selection.get("method")) != "adjacent_contiguous_unit_capacity_set_packing_with_v3_route_utility":
        raise ValueError("V3 must retain the existing unit-capacity route backend")
    if str(calibration_config.get("scope")) != "station_pair":
        raise ValueError("V3 evaluator requires station-pair edge calibration")
    training_dir = Path(args.training_dir).expanduser().resolve()
    training_contract = _validate_training_contract(training_dir)
    checkpoint = training_dir / "structured_assignment_v3.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty V3 validation output")
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path, samples, manifest = load_synthetic_curriculum_manifest(
        args.synthetic_manifest,
        require_all_splits=False,
        allowed_splits=("validation",),
    )
    _validate_input_contract(contract, manifest, samples)
    source_audit = source_disjoint_audit(samples)
    candidate_sets = build_candidate_sets(samples, ALL_STATION_PAIRS, chi2_gate=None, feature_set="residual_v1")
    bundle = build_transformer_graph_bundle(candidate_sets, context_mode="full_event")
    model, artifact = load_structured_assignment_artifact(checkpoint, device="cuda")
    prediction = predict_structured_assignment_scores(
        model,
        bundle,
        artifact.node_standardizer,
        artifact.edge_standardizer,
        device="cuda",
        batch_size=int(settings["training"]["batch_size"]),
    )
    bins = int(calibration_config["bins"])
    edge_scores, calibration = calibrate_transformer_scores(
        bundle.adjacent_sets,
        prediction.edge_scores,
        bins,
        scope="station_pair",
        method=str(calibration_config["method"]),
    )
    initial_thresholds = _pair_mapping(_require_mapping(selection, "initial_thresholds"))
    edge_selected = select_route_operating_point(
        bundle.adjacent_sets,
        edge_scores,
        station_path=(0, 1, 2, 3),
        threshold_grid=tuple(float(value) for value in selection["threshold_grid"]),
        unmatched_penalties=tuple(float(value) for value in selection["unmatched_penalties"]),
        initial_thresholds=initial_thresholds,
        maximum_sweeps=int(selection["maximum_sweeps"]),
        capture_criteria=criteria,
        calibration_bins=bins,
        maximum_hypotheses=int(selection["maximum_hypotheses"]),
        selection_events_per_magnitude=int(selection["selection_events_per_magnitude"]),
        full_validation_rerank_candidates=int(selection["full_validation_rerank_candidates"]),
    )
    grouped = _group_by_magnitude(list(bundle.adjacent_sets), list(edge_scores))
    contexts = {
        magnitude: prepare_route_assignment_context(sets, scores, bins, (0, 1, 2, 3))
        for magnitude, (sets, scores) in grouped.items()
    }
    temperatures = tuple(float(value) for value in selection["route_utility_temperatures"])
    if not temperatures or any(not math.isfinite(value) or value <= 0.0 for value in temperatures):
        raise ValueError("V3 route utility temperature grid is invalid")
    score_thresholds = tuple(float(value) for value in selection["complete_route_score_thresholds"])
    candidates: list[dict[str, object]] = []
    for temperature in temperatures:
        route_maps = _route_utility_score_maps(prediction.route_score_sets, temperature)
        route_config, evaluations, rows = _select_direct_controls(
            grouped,
            contexts,
            route_maps,
            edge_selected.thresholds,
            score_thresholds,
            tuple(float(value) for value in selection["unmatched_penalties"]),
            "replace",
            (1.0,),
            criteria,
            int(selection["maximum_hypotheses"]),
            bins,
        )
        rank = _rank(
            evaluations,
            criteria,
            float(route_config.complete_route_score_threshold),
            float(route_config.unmatched_penalty),
            float(route_config.complete_route_context_weight),
        )
        candidates.append(
            {
                "temperature": temperature,
                "route_maps": route_maps,
                "route_config": route_config,
                "evaluations": evaluations,
                "rank": rank,
                "search_rows": rows,
            }
        )
    # Lower temperature is the deterministic conservative tie-break after the
    # predeclared validation-quality rank.
    selected = max(candidates, key=lambda value: tuple(value["rank"]) + (-float(value["temperature"]),))
    selected_config = selected["route_config"]
    selected_maps = selected["route_maps"]
    selected_evaluations = selected["evaluations"]
    if not isinstance(selected_config, RouteAssignmentConfig) or not isinstance(selected_maps, Mapping):
        raise RuntimeError("V3 route utility selection is malformed")
    edge_control = RouteAssignmentConfig(
        score_threshold_by_pair=dict(edge_selected.thresholds),
        unmatched_penalty=float(edge_selected.unmatched_penalty),
        station_path=(0, 1, 2, 3),
        maximum_hypotheses=int(selection["maximum_hypotheses"]),
    )
    v3_trials, v3_magnitudes, v3_details = _trial_and_magnitude_rows(
        "v3_structured_route_utility",
        samples,
        list(bundle.adjacent_sets),
        list(edge_scores),
        selected_config,
        selected_maps,
        criteria,
        bins,
    )
    edge_trials, edge_magnitudes, edge_details = _trial_and_magnitude_rows(
        "same_checkpoint_edge_route_control",
        samples,
        list(bundle.adjacent_sets),
        list(edge_scores),
        edge_control,
        None,
        criteria,
        bins,
    )
    utility_probabilities = 1.0 / (1.0 + np.exp(-np.clip(prediction.route_utilities, -60.0, 60.0)))
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
            "schema_version": "faser-structured-assignment-v3-validation",
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "synthetic_manifest": str(manifest_path),
            "synthetic_manifest_sha256": _sha256(manifest_path),
            "training_dir": str(training_dir),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "loaded_event_splits": ["validation"],
            "forbidden_splits": ["test"],
            "test_events_loaded": False,
            "test_artifacts_opened": False,
            "physical_geometry_repropagation": True,
            "q_over_p_mode": 0,
            "candidate_chi2_gate": None,
            "candidate_graph": "existing_mode0_acts_physical_candidates_all_six_station_pairs",
            "route_assignment_backend": "adjacent_contiguous_unit_capacity_set_packing",
            "edge_calibration_fit_split": "validation_only",
            "edge_threshold_selection_split": "validation_only",
            "route_utility_temperature_selection_split": "validation_only",
            "training_contract": training_contract,
            "source_audit": source_audit,
        },
    )
    _write_json(
        output_root / "calibration.json",
        {
            "schema_version": "faser-structured-assignment-v3-edge-calibration",
            "fit_split": "validation_only",
            "test_opened": False,
            "calibration": calibration,
        },
    )
    _write_json(
        output_root / "validation_selected_operating_point.json",
        {
            "schema_version": "faser-structured-assignment-v3-route-operating-point",
            "selection_split": "validation_only",
            "test_opened": False,
            "method": "adjacent_contiguous_unit_capacity_set_packing_with_v3_route_utility",
            "capture_success_criteria": criteria,
            "edge_thresholds": {
                f"{left}->{right}": float(edge_selected.thresholds[(left, right)])
                for left, right in ((0, 1), (1, 2), (2, 3))
            },
            "edge_unmatched_penalty": float(edge_selected.unmatched_penalty),
            "route_utility_temperature": float(selected["temperature"]),
            "complete_route_score_threshold": selected_config.complete_route_score_threshold,
            "complete_route_score_composition": selected_config.complete_route_score_composition,
            "unmatched_penalty": selected_config.unmatched_penalty,
            "capture_by_magnitude": captures,
            "rank": list(selected["rank"]),
            "edge_selection": {
                "rank": list(edge_selected.rank),
                "search_rows": list(edge_selected.search_rows),
            },
            "route_utility_search": [
                {
                    "temperature": float(candidate["temperature"]),
                    "rank": list(candidate["rank"]),
                    "search_rows": list(candidate["search_rows"]),
                }
                for candidate in candidates
            ],
        },
    )
    _write_csv(output_root / "validation_route_trial_metrics.csv", [*v3_trials, *edge_trials])
    _write_csv(output_root / "validation_route_magnitude_summary.csv", [*v3_magnitudes, *edge_magnitudes])
    _write_json(
        output_root / "validation_results.json",
        {
            "schema_version": "faser-structured-assignment-v3-validation",
            "edge_candidate_metrics": candidate_score_metrics(bundle.adjacent_sets, edge_scores, bins),
            "structured_route_utility_diagnostic": {
                "routes": int(prediction.route_utilities.size),
                "truth_assignment_routes": int(np.count_nonzero(prediction.route_truth_assignment)),
                "route_label_metrics_at_native_logistic_scale": binary_calibration(
                    utility_probabilities, prediction.route_labels, bins=bins
                ),
                "note": (
                    "The primary route quantity is a structured utility, not an independently "
                    "calibrated route probability; only its validation-selected temperature is "
                    "used to interface with the unchanged solver."
                ),
            },
            "score_streams": {
                "v3_structured_route_utility": {
                    "route_trial_evaluations": v3_details,
                    "route_magnitude_summary": v3_magnitudes,
                },
                "same_checkpoint_edge_route_control": {
                    "route_trial_evaluations": edge_details,
                    "route_magnitude_summary": edge_magnitudes,
                },
            },
        },
    )
    print(
        json.dumps(
            _json_value(
                {
                    "output_dir": str(output_root),
                    "route_utility_temperature": float(selected["temperature"]),
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
