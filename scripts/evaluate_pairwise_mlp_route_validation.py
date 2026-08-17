#!/usr/bin/env python3
"""Evaluate the frozen pairwise MLP with validation-only four-station routes.

This control deliberately keeps the existing mode-0 Acts physical candidate
graph, the trained pairwise MLP, and its validation-fitted station-pair
calibration.  It opens only validation payloads and selects one adjacent
route operating point on validation; no test asset is resolved or created.
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

from baselines.field_chi2_matching import PAIR_FEATURE_SETS, pair_feature_names
from baselines.mlp_pair_classifier import load_pair_classifier
from baselines.route_assignment import RouteAssignmentConfig
from datasets.physical_curriculum import (
    CurriculumSample,
    load_synthetic_curriculum_manifest,
    uniform_condition_axis,
)
from scripts.config_loader import load_yaml_with_base
from scripts.run_global_assignment_mlp_baseline import _apply_frozen_calibration_sets
from training.curriculum_mlp import CandidateSet, build_candidate_sets, score_candidate_sets
from training.geometry_aware_transformer import (
    ADJACENT_STATION_PAIRS,
    ALL_STATION_PAIRS,
    candidate_score_metrics,
    source_disjoint_audit,
)
from training.route_assignment import evaluate_adjacent_route_assignment_sets
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


def _parse_pair_thresholds(raw: Mapping[object, object]) -> dict[tuple[int, int], float]:
    result: dict[tuple[int, int], float] = {}
    for key, value in raw.items():
        parts = str(key).split("->")
        if len(parts) != 2:
            raise ValueError(f"invalid adjacent threshold key '{key}'")
        result[(int(parts[0]), int(parts[1]))] = float(value)
    if set(result) != set(ADJACENT_STATION_PAIRS):
        raise ValueError("initial thresholds must define exactly 0->1, 1->2, 2->3")
    return result


def _sample_key(sample: CurriculumSample) -> tuple[str, str]:
    return str(sample.source_id), str(sample.payload_id)


def _group_candidate_sets(
    samples: Sequence[CurriculumSample],
    candidate_sets: Sequence[CandidateSet],
    scores: Sequence[np.ndarray],
) -> list[tuple[CurriculumSample, list[CandidateSet], list[np.ndarray]]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    sample_by_key: dict[tuple[str, str], CurriculumSample] = {}
    for index, candidate_set in enumerate(candidate_sets):
        key = _sample_key(candidate_set.sample)
        grouped[key].append(index)
        sample_by_key.setdefault(key, candidate_set.sample)
    expected = {_sample_key(sample) for sample in samples}
    if set(grouped) != expected:
        raise ValueError("candidate-set groups differ from loaded validation samples")
    return [
        (
            sample_by_key[key],
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


def _trial_rows(
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
                "source_id": sample.source_id,
                "payload_id": sample.payload_id,
                "condition_axis": sample.condition_axis,
                "condition_value": float(
                    sample.curriculum_magnitude
                    if sample.condition_value is None
                    else sample.condition_value
                ),
                "condition_magnitude": float(sample.curriculum_magnitude),
                "direction_trial": sample.direction_trial,
                "capture_success": success,
                **dict(route),
            }
        )
        details.append(
            {
                "source_id": sample.source_id,
                "payload_id": sample.payload_id,
                "condition_axis": sample.condition_axis,
                "condition_value": float(
                    sample.curriculum_magnitude
                    if sample.condition_value is None
                    else sample.condition_value
                ),
                "condition_magnitude": float(sample.curriculum_magnitude),
                "direction_trial": sample.direction_trial,
                "capture_success": success,
                "evaluation": evaluation,
            }
        )
    return rows, details


def _magnitude_rows(
    samples: Sequence[CurriculumSample],
    candidate_sets: Sequence[CandidateSet],
    scores: Sequence[np.ndarray],
    route_config: RouteAssignmentConfig,
    calibration_bins: int,
    trials: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    tracked_metrics = (
        "candidate_complete_truth_chain_recall",
        "score_threshold_complete_truth_chain_recall",
        "complete_track_efficiency",
        "complete_track_purity",
        "track_fake_rate",
        "missing_station_recovery",
    )
    condition_axis = uniform_condition_axis(samples)
    for magnitude in sorted({float(sample.curriculum_magnitude) for sample in samples}):
        sample_keys = {
            _sample_key(sample)
            for sample in samples
            if np.isclose(float(sample.curriculum_magnitude), magnitude)
        }
        selected = [
            index
            for index, candidate_set in enumerate(candidate_sets)
            if _sample_key(candidate_set.sample) in sample_keys
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
        magnitude_trials = [
            trial
            for trial in trials
            if np.isclose(float(trial["condition_magnitude"]), magnitude)
        ]
        row: dict[str, object] = {
            "condition_axis": condition_axis,
            "condition_magnitude": magnitude,
            "direction_trials": len(magnitude_trials),
            "capture_successes": int(sum(bool(trial["capture_success"]) for trial in magnitude_trials)),
            "capture_fraction": float(
                np.mean([bool(trial["capture_success"]) for trial in magnitude_trials])
            ),
            **{f"pooled_{key}": value for key, value in route.items()},
        }
        for metric in tracked_metrics:
            values = np.asarray(
                [float(trial[metric]) for trial in magnitude_trials if trial.get(metric) is not None],
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


def _load_calibration(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("MLP calibration payload must be a mapping")
    calibration = payload.get("calibration", payload)
    if not isinstance(calibration, Mapping):
        raise ValueError("MLP calibration payload has no calibration mapping")
    if payload.get("fit_split") != "validation_only" or calibration.get("fit_split") != "validation_only":
        raise ValueError("MLP calibration must have been fit on validation only")
    if calibration.get("scope") != "station_pair":
        raise ValueError("V2 route comparison requires station-pair MLP calibration")
    return dict(payload), dict(calibration)


def _validate_contract(
    contract: Mapping[str, object], manifest: Mapping[str, object], samples: Sequence[CurriculumSample]
) -> None:
    if tuple(contract.get("allowed_splits", ())) != ("train", "validation"):
        raise ValueError("V2 configuration must declare only train and validation as allowed")
    if "test" not in tuple(contract.get("forbidden_splits", ())):
        raise ValueError("V2 configuration must explicitly forbid test")
    if manifest.get("physical_geometry_repropagation") is not True:
        raise ValueError("manifest does not certify physical geometry repropagation")
    if int(manifest.get("q_over_p_mode", -1)) != 0:
        raise ValueError("MLP route comparison requires mode-0 Acts propagation")
    if {sample.split for sample in samples} != {"validation"}:
        raise ValueError("MLP route comparison loaded a non-validation sample")
    expected_axis = str(contract.get("condition_axis", "translation_xy_mm"))
    declared = contract.get(
        "curriculum_condition_magnitudes", contract.get("curriculum_magnitudes_mm")
    )
    if not isinstance(declared, (list, tuple)) or not declared:
        raise ValueError("V2 input contract lacks physical condition magnitudes")
    if uniform_condition_axis(samples) != expected_axis:
        raise ValueError("validation physical curriculum condition axis differs from the V2 contract")
    required = tuple(sorted(float(value) for value in declared))
    observed = tuple(sorted({float(sample.curriculum_magnitude) for sample in samples}))
    if observed != required:
        raise ValueError("validation physical curriculum is missing a required condition magnitude")


def _artifact_feature_set(feature_names: tuple[str, ...]) -> str:
    """Resolve the MLP's original feature view without changing its input schema."""
    matches = [
        name
        for name in PAIR_FEATURE_SETS
        if pair_feature_names(ALL_STATION_PAIRS, feature_set=name) == feature_names
    ]
    if len(matches) != 1:
        raise ValueError("MLP checkpoint does not match one unambiguous physical feature set")
    return matches[0]


def _candidate_membership_audit(
    reference_sets: Sequence[CandidateSet], candidate_sets: Sequence[CandidateSet]
) -> dict[str, object]:
    """Prove that feature encoding did not alter the physical candidate graph."""
    if len(reference_sets) != len(candidate_sets):
        raise ValueError("feature views have different candidate-set counts")
    digest = hashlib.sha256()
    candidate_rows = 0
    for reference, candidate in zip(reference_sets, candidate_sets):
        if (
            _sample_key(reference.sample) != _sample_key(candidate.sample)
            or reference.station_pair != candidate.station_pair
            or reference.event.run_id != candidate.event.run_id
            or reference.event.event_id != candidate.event.event_id
        ):
            raise ValueError("feature views disagree on a physical event or station pair")
        reference_endpoints = [
            (int(item.source_index), int(item.target_index)) for item in reference.candidates
        ]
        candidate_endpoints = [
            (int(item.source_index), int(item.target_index)) for item in candidate.candidates
        ]
        if reference_endpoints != candidate_endpoints:
            raise ValueError("MLP feature view changed a physical candidate endpoint")
        candidate_rows += len(candidate_endpoints)
        digest.update(
            f"{reference.sample.source_id}\x1f{reference.sample.payload_id}\x1f"
            f"{reference.event.run_id}\x1f{reference.event.event_id}\x1f"
            f"{reference.station_pair}\x1f{candidate_endpoints}\n".encode()
        )
    return {
        "reference_feature_set": "residual_v1",
        "mlp_feature_view_has_identical_physical_candidate_membership": True,
        "candidate_sets": len(candidate_sets),
        "candidate_rows": candidate_rows,
        "membership_sha256": digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frozen-calibration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    supplied = load_yaml_with_base(config_path)
    settings = _require_mapping(supplied, "route_aware_transformer_v2")
    contract = _require_mapping(settings, "input_contract")
    route_selection = _require_mapping(settings, "route_selection")
    criteria = _require_mapping(settings, "capture_success")
    calibration_config = _require_mapping(settings, "calibration")
    if str(route_selection.get("method")) != "adjacent_contiguous_unit_capacity_set_packing":
        raise ValueError("V2 MLP control must use the existing unit-capacity route solver")

    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty MLP route validation output")
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path, samples, manifest = load_synthetic_curriculum_manifest(
        args.synthetic_manifest,
        require_all_splits=False,
        allowed_splits=("validation",),
    )
    _validate_contract(contract, manifest, samples)
    source_audit = source_disjoint_audit(samples)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    calibration_path = Path(args.frozen_calibration).expanduser().resolve()
    calibration_payload, calibration = _load_calibration(calibration_path)
    model, artifact = load_pair_classifier(checkpoint, device=str(args.device))
    if artifact.station_pairs != ALL_STATION_PAIRS:
        raise ValueError("MLP checkpoint station pairs are incompatible with V2 physical candidates")
    feature_set = _artifact_feature_set(artifact.feature_names)

    reference_sets = build_candidate_sets(
        samples, ALL_STATION_PAIRS, chi2_gate=None, feature_set="residual_v1"
    )
    candidate_sets = build_candidate_sets(
        samples, ALL_STATION_PAIRS, chi2_gate=None, feature_set=feature_set
    )
    membership_audit = _candidate_membership_audit(reference_sets, candidate_sets)
    raw_scores = score_candidate_sets(model, artifact, candidate_sets)
    scores = _apply_frozen_calibration_sets(candidate_sets, raw_scores, calibration)
    initial_thresholds = _parse_pair_thresholds(_require_mapping(route_selection, "initial_thresholds"))
    selected = select_route_operating_point(
        candidate_sets,
        scores,
        station_path=(0, 1, 2, 3),
        threshold_grid=tuple(float(value) for value in route_selection["threshold_grid"]),
        unmatched_penalties=tuple(float(value) for value in route_selection["unmatched_penalties"]),
        initial_thresholds=initial_thresholds,
        maximum_sweeps=int(route_selection["maximum_sweeps"]),
        capture_criteria=criteria,
        calibration_bins=int(calibration_config["bins"]),
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
    trial_rows, trial_details = _trial_rows(
        samples,
        candidate_sets,
        scores,
        route_config,
        criteria,
        int(calibration_config["bins"]),
    )
    magnitude_rows = _magnitude_rows(
        samples,
        candidate_sets,
        scores,
        route_config,
        int(calibration_config["bins"]),
        trial_rows,
    )

    (output_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(_json_value(supplied), sort_keys=False), encoding="utf-8"
    )
    _write_json(
        output_root / "validation_control_contract.json",
        {
            "schema_version": "faser-pairwise-mlp-route-validation-v2-control",
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "synthetic_manifest": str(manifest_path),
            "synthetic_manifest_sha256": _sha256(manifest_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "frozen_calibration": str(calibration_path),
            "frozen_calibration_sha256": _sha256(calibration_path),
            "loaded_event_splits": ["validation"],
            "forbidden_splits": ["test"],
            "test_events_loaded": False,
            "test_artifacts_opened": False,
            "physical_geometry_repropagation": True,
            "q_over_p_mode": 0,
            "condition_axis": uniform_condition_axis(samples),
            "candidate_chi2_gate": None,
            "candidate_graph": "existing_mode0_acts_physical_candidates_all_six_station_pairs",
            "mlp_feature_set": feature_set,
            "candidate_membership_audit": membership_audit,
            "route_assignment_backend": "adjacent_contiguous_unit_capacity_set_packing",
            "calibration_fit_split": calibration.get("fit_split"),
            "calibration_refit": False,
            "route_threshold_selection_split": "validation_only",
            "source_audit": source_audit,
        },
    )
    _write_json(
        output_root / "validation_selected_operating_point.json",
        {
            "schema_version": "faser-pairwise-mlp-route-v2-control-operating-point",
            "selection_split": "validation_only",
            "test_opened": False,
            "method": "adjacent_contiguous_unit_capacity_set_packing",
            "capture_success_criteria": criteria,
            **_selection_payload(selected),
        },
    )
    _write_csv(output_root / "validation_route_trial_metrics.csv", trial_rows)
    _write_csv(output_root / "validation_route_magnitude_summary.csv", magnitude_rows)
    _write_json(
        output_root / "validation_results.json",
        {
            "schema_version": "faser-pairwise-mlp-route-validation-v2-control",
            "raw_candidate_metrics": candidate_score_metrics(
                candidate_sets, raw_scores, int(calibration_config["bins"])
            ),
            "frozen_calibrated_candidate_metrics": candidate_score_metrics(
                candidate_sets, scores, int(calibration_config["bins"])
            ),
            "calibration_payload_fit_split": calibration_payload.get("fit_split"),
            "route_trial_evaluations": trial_details,
            "route_magnitude_summary": magnitude_rows,
        },
    )
    print(
        json.dumps(
            _json_value(
                {
                    "output_dir": str(output_root),
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
