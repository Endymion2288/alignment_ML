#!/usr/bin/env python3
"""Evaluate frozen pairwise MLP scores with adjacent four-station routes.

This is intentionally an evaluation-only program.  It loads a separately
materialized physical test corpus, verifies that it contains exactly the
configured sealed source files, applies the validation-fitted temperature map,
and uses the validation-selected adjacent thresholds and dustbin penalty.  No
model, calibration, gate, threshold, temperature, or test split is optimized
here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baselines.field_chi2_matching import pair_feature_names
from baselines.mlp_pair_classifier import load_pair_classifier
from baselines.route_assignment import RouteAssignmentConfig, adjacent_station_pairs
from datasets.physical_curriculum import CurriculumSample, load_synthetic_curriculum_manifest
from scripts.config_loader import load_yaml_with_base
from scripts.run_global_assignment_mlp_baseline import _apply_frozen_calibration_sets
from training.curriculum_mlp import build_candidate_sets, score_candidate_sets
from training.route_assignment import evaluate_adjacent_route_assignment_sets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "frozen_route_level_multidirection_test.yaml"


def _json_value(value: Any) -> Any:
    """Convert NumPy values and absent rates into strict JSON-compatible data."""
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_value(dict(row)))


def _read_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON payload must be a mapping: {resolved}")
    return resolved, dict(payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    supplied = load_yaml_with_base(path)
    root = supplied.get("physical_curriculum_mlp", supplied)
    route = supplied.get("frozen_route_level_scan")
    if not isinstance(root, Mapping) or not isinstance(route, Mapping):
        raise ValueError("configuration requires physical_curriculum_mlp and frozen_route_level_scan")
    mlp = root.get("curriculum_mlp")
    refit = root.get("refit")
    if not isinstance(mlp, Mapping) or not isinstance(refit, Mapping):
        raise ValueError("configuration requires curriculum_mlp and refit mappings")
    return dict(root), dict(mlp), dict(route)


def _configured_test_sources(root: Mapping[str, Any], split: str) -> set[str]:
    raw_sources = root.get("sources")
    if not isinstance(raw_sources, list):
        raise ValueError("configuration has no source list")
    sources = {
        str(source["id"])
        for source in raw_sources
        if isinstance(source, Mapping) and str(source.get("split")) == split
    }
    if not sources:
        raise ValueError(f"configuration has no {split} sources")
    return sources


def _audit_sealed_samples(
    samples: Sequence[CurriculumSample],
    expected_split: str,
    expected_sources: set[str],
) -> dict[str, object]:
    if not samples:
        raise ValueError("frozen route evaluation has no samples")
    unexpected_splits = sorted({sample.split for sample in samples} - {expected_split})
    if unexpected_splits:
        raise ValueError(
            "frozen route evaluation manifest contains non-evaluation split(s): "
            + ", ".join(unexpected_splits)
        )
    actual_sources = {source for sample in samples for source in sample.source_ids}
    if actual_sources != expected_sources:
        raise ValueError(
            "physical scan source files differ from the sealed configured split: "
            f"missing={sorted(expected_sources - actual_sources)}, "
            f"extra={sorted(actual_sources - expected_sources)}"
        )
    owners: dict[str, str] = {}
    for sample in samples:
        for uid in sample.source_event_uids:
            previous = owners.setdefault(uid, sample.source_id)
            if previous != sample.source_id:
                raise ValueError(f"source event provenance is duplicated across pooled samples: {uid}")
    return {
        "evaluation_split": expected_split,
        "source_split_unit": "original_xAOD_file",
        "configured_sources": sorted(expected_sources),
        "observed_sources": sorted(actual_sources),
        "source_event_uid_count": len(owners),
        "strict_source_membership_match": True,
        "no_non_evaluation_samples_loaded": True,
    }


def _parse_thresholds(
    selected: Mapping[str, Any], station_path: tuple[int, ...]
) -> dict[tuple[int, int], float]:
    pairs = adjacent_station_pairs(station_path)
    raw = selected.get("thresholds")
    if not isinstance(raw, Mapping):
        raise ValueError("frozen operating point lacks station-pair thresholds")
    values: dict[tuple[int, int], float] = {}
    for pair in pairs:
        key = f"{pair[0]}->{pair[1]}"
        if key not in raw:
            raise ValueError(f"frozen operating point lacks adjacent threshold {key}")
        threshold = float(raw[key])
        if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"invalid frozen threshold for {key}")
        values[pair] = threshold
    return values


def _frozen_artifacts(
    calibration_path: Path,
    operating_path: Path,
    route_config: Mapping[str, Any],
    station_path: tuple[int, ...],
) -> tuple[dict[str, Any], dict[str, Any], dict[tuple[int, int], float], float]:
    _, calibration_payload = _read_json(calibration_path)
    calibration = calibration_payload.get("calibration", calibration_payload)
    if not isinstance(calibration, Mapping):
        raise ValueError("frozen calibration payload is invalid")
    contract = route_config.get("frozen_contract", {})
    if not isinstance(contract, Mapping):
        raise ValueError("frozen_contract must be a mapping")
    if bool(contract.get("require_validation_only_calibration", True)):
        if calibration_payload.get("fit_split") != "validation_only" or calibration.get("fit_split") != "validation_only":
            raise ValueError("frozen calibration was not fit on validation only")
    _, selected = _read_json(operating_path)
    if bool(contract.get("require_validation_only_operating_point", True)):
        if selected.get("selection_split") != "validation_only" or selected.get("test_opened") is not False:
            raise ValueError("frozen operating point is not a sealed validation-only selection")
    required_method = str(contract.get("required_assignment_method", "sinkhorn_hungarian"))
    if str(selected.get("method")) != required_method:
        raise ValueError(
            f"frozen operating point method is {selected.get('method')!r}, expected {required_method!r}"
        )
    thresholds = _parse_thresholds(selected, station_path)
    penalty = float(selected.get("unmatched_penalty"))
    if not math.isfinite(penalty):
        raise ValueError("frozen operating point has invalid unmatched_penalty")
    return dict(calibration), dict(selected), thresholds, penalty


def _sample_groups(
    samples: Sequence[CurriculumSample],
    sets: Sequence[object],
    scores: Sequence[np.ndarray],
) -> list[tuple[CurriculumSample, list[object], list[np.ndarray]]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    sample_by_key: dict[tuple[str, str], CurriculumSample] = {}
    for index, candidate_set in enumerate(sets):
        sample = candidate_set.sample
        key = (str(sample.source_id), str(sample.payload_id))
        grouped[key].append(index)
        previous = sample_by_key.setdefault(key, sample)
        if previous != sample:
            raise ValueError(f"candidate sets disagree on sample metadata for {key}")
    expected_keys = {(sample.source_id, sample.payload_id) for sample in samples}
    if set(grouped) != expected_keys:
        raise ValueError("candidate-set sample grouping does not match manifest samples")
    return [
        (
            sample_by_key[key],
            [sets[index] for index in indices],
            [scores[index] for index in indices],
        )
        for key, indices in sorted(grouped.items())
    ]


def _capture_success(route: Mapping[str, object], criteria: Mapping[str, Any]) -> bool:
    requirements = (
        ("complete_track_efficiency", ">=", "minimum_complete_track_efficiency"),
        ("complete_track_purity", ">=", "minimum_complete_track_purity"),
        ("track_fake_rate", "<=", "maximum_track_fake_rate"),
    )
    for metric, relation, limit_name in requirements:
        value = route.get(metric)
        if value is None:
            return False
        limit = float(criteria[limit_name])
        if relation == ">=" and float(value) < limit:
            return False
        if relation == "<=" and float(value) > limit:
            return False
    return True


def _trial_row(
    sample: CurriculumSample,
    evaluation: Mapping[str, object],
    capture_success: bool,
) -> dict[str, object]:
    route = evaluation["route"]
    assignment = evaluation["assignment"]
    candidate = evaluation["candidate"]
    if not isinstance(route, Mapping) or not isinstance(assignment, Mapping) or not isinstance(candidate, Mapping):
        raise RuntimeError("route evaluation returned an invalid metric payload")
    row: dict[str, object] = {
        "source_id": sample.source_id,
        "source_ids": ";".join(sample.source_ids),
        "payload_id": sample.payload_id,
        "magnitude_mm": float(sample.magnitude_mm),
        "direction_trial": sample.direction_trial,
        "injected_offsets_xy_mm": json.dumps(sample.injected_offsets_xy_mm, sort_keys=True),
        "capture_success": bool(capture_success),
        **dict(route),
        "assignment_events": assignment["events"],
        "candidate_edges_above_threshold": assignment["candidate_edges_above_threshold"],
        "hypotheses_considered": assignment["hypotheses_considered"],
    }
    for pair, payload in sorted(candidate.items()):
        if not isinstance(payload, Mapping):
            raise RuntimeError("route candidate metric payload is invalid")
        prefix = "pair_" + str(pair).replace("->", "_").replace("-", "m")
        for key in (
            "candidate_rows",
            "positive_candidate_rows",
            "candidate_truth_recall",
            "roc_auc",
            "average_precision",
            "expected_calibration_error",
        ):
            row[f"{prefix}_{key}"] = payload.get(key)
    return row


def _station_pair_rows(
    sample: CurriculumSample,
    evaluation: Mapping[str, object],
    capture_success: bool,
) -> list[dict[str, object]]:
    payload = evaluation["by_station_pair"]
    if not isinstance(payload, Mapping):
        raise RuntimeError("route evaluation lacks station-pair metrics")
    rows: list[dict[str, object]] = []
    for pair, values in sorted(payload.items()):
        if not isinstance(values, Mapping):
            raise RuntimeError("station-pair metric payload is invalid")
        candidate = values.get("candidate")
        association = values.get("association")
        unmatched = values.get("unmatched")
        if not all(isinstance(value, Mapping) for value in (candidate, association, unmatched)):
            raise RuntimeError("station-pair metric payload is incomplete")
        rows.append(
            {
                "source_id": sample.source_id,
                "payload_id": sample.payload_id,
                "magnitude_mm": float(sample.magnitude_mm),
                "direction_trial": sample.direction_trial,
                "station_pair": str(pair),
                "capture_success": bool(capture_success),
                "score_threshold": values["score_threshold"],
                **dict(candidate),
                **dict(association),
                **dict(unmatched),
            }
        )
    return rows


def _select_magnitude(
    samples: Sequence[CurriculumSample],
    sets: Sequence[object],
    scores: Sequence[np.ndarray],
    magnitude: float,
) -> tuple[list[CurriculumSample], list[object], list[np.ndarray]]:
    selected_samples = [sample for sample in samples if np.isclose(sample.magnitude_mm, magnitude)]
    selected_keys = {(sample.source_id, sample.payload_id) for sample in selected_samples}
    selected_indices = [
        index
        for index, candidate_set in enumerate(sets)
        if (candidate_set.sample.source_id, candidate_set.sample.payload_id) in selected_keys
    ]
    return selected_samples, [sets[index] for index in selected_indices], [scores[index] for index in selected_indices]


def _direction_summary(
    trial_rows: Sequence[Mapping[str, object]],
    pooled: Mapping[str, object],
) -> dict[str, object]:
    route = pooled["route"]
    if not isinstance(route, Mapping):
        raise RuntimeError("pooled route evaluation is invalid")
    metrics = (
        "candidate_complete_truth_chain_recall",
        "score_threshold_complete_truth_chain_recall",
        "complete_track_efficiency",
        "complete_track_purity",
        "track_purity",
        "track_fake_rate",
        "fake_endpoint_route_rate",
        "duplicate_rate",
        "missing_station_recovery",
    )
    summary: dict[str, object] = {
        "magnitude_mm": float(trial_rows[0]["magnitude_mm"]),
        "direction_trials": len(trial_rows),
        "capture_successes": int(sum(bool(row["capture_success"]) for row in trial_rows)),
        "capture_fraction": float(np.mean([bool(row["capture_success"]) for row in trial_rows])),
        **{f"pooled_{key}": value for key, value in route.items()},
    }
    for metric in metrics:
        values = np.asarray(
            [float(row[metric]) for row in trial_rows if row.get(metric) is not None],
            dtype=np.float64,
        )
        summary[f"direction_mean_{metric}"] = None if not values.size else float(np.mean(values))
        summary[f"direction_std_{metric}"] = None if not values.size else float(np.std(values))
        summary[f"direction_min_{metric}"] = None if not values.size else float(np.min(values))
        summary[f"direction_max_{metric}"] = None if not values.size else float(np.max(values))
    return summary


def _plot_capture(path: Path, summaries: Sequence[Mapping[str, object]]) -> None:
    x = np.asarray([float(row["magnitude_mm"]) for row in summaries], dtype=np.float64)
    y = np.asarray([float(row["capture_fraction"]) for row in summaries], dtype=np.float64)
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(x, y, marker="o", color="#1b6ca8", label="capture fraction")
    axis.set_xscale("symlog", linthresh=0.1)
    axis.set_xlabel("Injected station misalignment magnitude [mm]")
    axis.set_ylabel("Direction-trial capture fraction")
    axis.set_ylim(-0.03, 1.03)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_quality(path: Path, summaries: Sequence[Mapping[str, object]]) -> None:
    x = np.asarray([float(row["magnitude_mm"]) for row in summaries], dtype=np.float64)
    metrics = (
        ("candidate_complete_truth_chain_recall", "raw candidate chain recall", "#2a9d8f"),
        ("score_threshold_complete_truth_chain_recall", "thresholded score chain recall", "#e76f51"),
        ("complete_track_efficiency", "complete-track efficiency", "#264653"),
        ("track_purity", "route purity", "#8a5a44"),
    )
    fig, axis = plt.subplots(figsize=(8.0, 4.7))
    for metric, label, color in metrics:
        means = np.asarray(
            [row[f"direction_mean_{metric}"] for row in summaries], dtype=np.float64
        )
        deviations = np.asarray(
            [row[f"direction_std_{metric}"] for row in summaries], dtype=np.float64
        )
        axis.errorbar(x, means, yerr=deviations, marker="o", capsize=3, label=label, color=color)
    axis.set_xscale("symlog", linthresh=0.1)
    axis.set_xlabel("Injected station misalignment magnitude [mm]")
    axis.set_ylabel("Metric (mean +/- direction spread)")
    axis.set_ylim(-0.03, 1.03)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frozen-calibration", required=True)
    parser.add_argument("--frozen-operating-point", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    root, mlp, route_config = _load_config(config_path)
    evaluation_split = str(route_config.get("evaluation_split", "test"))
    if evaluation_split != "test":
        raise ValueError("frozen route scan is reserved for the sealed test split")
    station_path = tuple(int(station) for station in route_config.get("station_path", (0, 1, 2, 3)))
    adjacent_pairs = adjacent_station_pairs(station_path)
    if int(root["refit"].get("q_over_p_mode", -1)) != 0:
        raise ValueError("frozen route scan supports only physical mode-0 propagation")
    if route_config.get("candidate_chi2_gate") is not None:
        raise ValueError("frozen route scan must use the ungated physical candidate graph")

    manifest_path, samples, manifest = load_synthetic_curriculum_manifest(
        args.synthetic_manifest,
        require_all_splits=False,
    )
    if manifest.get("physical_geometry_repropagation") is not True or int(manifest.get("q_over_p_mode", -1)) != 0:
        raise ValueError("manifest does not certify physical mode-0 geometry repropagation")
    samples = [sample for sample in samples if sample.split == evaluation_split]
    source_audit = _audit_sealed_samples(
        samples,
        evaluation_split,
        _configured_test_sources(root, evaluation_split),
    )

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    model, artifact = load_pair_classifier(checkpoint, device=str(mlp["device"]))
    configured_stations = tuple(sorted(int(value) for value in root["refit"]["station_ids"]))
    station_pairs = tuple(
        (left, right) for left in configured_stations for right in configured_stations if left < right
    )
    feature_set = str(mlp.get("feature_set", "residual_v1"))
    if artifact.station_pairs != station_pairs or artifact.feature_names != tuple(
        pair_feature_names(station_pairs, feature_set=feature_set)
    ):
        raise ValueError("checkpoint feature schema does not match the frozen physical configuration")
    if not set(adjacent_pairs).issubset(set(station_pairs)):
        raise ValueError("route station path is not a subset of the MLP station pairs")

    calibration_path = Path(args.frozen_calibration).expanduser().resolve()
    operating_path = Path(args.frozen_operating_point).expanduser().resolve()
    calibration, selected, thresholds, unmatched_penalty = _frozen_artifacts(
        calibration_path, operating_path, route_config, station_path
    )
    route_assignment = RouteAssignmentConfig(
        score_threshold_by_pair=thresholds,
        unmatched_penalty=unmatched_penalty,
        station_path=station_path,
        maximum_hypotheses=int(route_config.get("maximum_hypotheses", 100_000)),
    )
    criteria = route_config.get("capture_success")
    if not isinstance(criteria, Mapping):
        raise ValueError("frozen route scan requires a predeclared capture_success mapping")
    for key in (
        "minimum_complete_track_efficiency",
        "minimum_complete_track_purity",
        "maximum_track_fake_rate",
    ):
        value = float(criteria[key])
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"capture_success.{key} must be finite and in [0, 1]")

    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty frozen route scan output")
    output_root.mkdir(parents=True, exist_ok=True)

    sets = build_candidate_sets(samples, station_pairs, chi2_gate=None, feature_set=feature_set)
    raw_scores = score_candidate_sets(model, artifact, sets)
    calibrated_scores = _apply_frozen_calibration_sets(sets, raw_scores, calibration)
    groups = _sample_groups(samples, sets, calibrated_scores)
    calibration_bins = int(mlp["calibration_bins"])

    trial_rows: list[dict[str, object]] = []
    station_pair_rows: list[dict[str, object]] = []
    detailed_trials: list[dict[str, object]] = []
    for sample, sample_sets, sample_scores in groups:
        evaluation = evaluate_adjacent_route_assignment_sets(
            sample_sets,
            sample_scores,
            route_assignment,
            calibration_bins=calibration_bins,
        )
        success = _capture_success(evaluation["route"], criteria)
        trial_rows.append(_trial_row(sample, evaluation, success))
        station_pair_rows.extend(_station_pair_rows(sample, evaluation, success))
        detailed_trials.append(
            {
                "source_id": sample.source_id,
                "source_ids": list(sample.source_ids),
                "payload_id": sample.payload_id,
                "magnitude_mm": sample.magnitude_mm,
                "direction_trial": sample.direction_trial,
                "injected_offsets_xy_mm": dict(sample.injected_offsets_xy_mm),
                "capture_success": success,
                "evaluation": evaluation,
            }
        )

    summary_rows: list[dict[str, object]] = []
    detailed_magnitudes: list[dict[str, object]] = []
    for magnitude in sorted({float(sample.magnitude_mm) for sample in samples}):
        magnitude_samples, magnitude_sets, magnitude_scores = _select_magnitude(
            samples, sets, calibrated_scores, magnitude
        )
        pooled = evaluate_adjacent_route_assignment_sets(
            magnitude_sets,
            magnitude_scores,
            route_assignment,
            calibration_bins=calibration_bins,
        )
        directions = [row for row in trial_rows if np.isclose(float(row["magnitude_mm"]), magnitude)]
        summary = _direction_summary(directions, pooled)
        summary_rows.append(summary)
        detailed_magnitudes.append(
            {
                "magnitude_mm": magnitude,
                "source_payloads": [sample.payload_id for sample in magnitude_samples],
                "pooled_evaluation": pooled,
                "direction_summary": summary,
            }
        )

    _write_csv(output_root / "trial_metrics.csv", trial_rows)
    _write_csv(output_root / "station_pair_trial_metrics.csv", station_pair_rows)
    _write_csv(output_root / "magnitude_summary.csv", summary_rows)
    _write_json(
        output_root / "frozen_evaluation_contract.json",
        {
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "synthetic_manifest": str(manifest_path),
            "synthetic_manifest_sha256": _sha256(manifest_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "frozen_calibration": str(calibration_path),
            "frozen_calibration_sha256": _sha256(calibration_path),
            "frozen_operating_point": str(operating_path),
            "frozen_operating_point_sha256": _sha256(operating_path),
            "source_audit": source_audit,
            "q_over_p_mode": 0,
            "station_path": list(station_path),
            "adjacent_station_pairs": [f"{left}->{right}" for left, right in adjacent_pairs],
            "calibration_fit_split": calibration.get("fit_split"),
            "selected_operating_point_split": selected.get("selection_split"),
            "selected_operating_point_test_opened": selected.get("test_opened"),
            "frozen_thresholds": {f"{left}->{right}": thresholds[(left, right)] for left, right in adjacent_pairs},
            "frozen_unmatched_penalty": unmatched_penalty,
            "capture_success_criteria": dict(criteria),
            "route_solver": route_config["frozen_contract"]["route_solver"],
            "no_test_time_calibration_or_selection": True,
        },
    )
    _write_json(
        output_root / "route_scan_results.json",
        {
            "schema_version": "faser-frozen-adjacent-route-scan-v1",
            "physical_geometry_repropagation": True,
            "q_over_p_mode": 0,
            "trial_evaluations": detailed_trials,
            "magnitude_evaluations": detailed_magnitudes,
        },
    )
    _plot_capture(output_root / "capture_fraction_vs_misalignment.png", summary_rows)
    _plot_quality(output_root / "route_quality_vs_misalignment.png", summary_rows)
    print(
        json.dumps(
            _json_value(
                {
                    "output_dir": str(output_root),
                    "trials": len(trial_rows),
                    "magnitudes": len(summary_rows),
                    "capture_fraction": {
                        str(row["magnitude_mm"]): row["capture_fraction"] for row in summary_rows
                    },
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
