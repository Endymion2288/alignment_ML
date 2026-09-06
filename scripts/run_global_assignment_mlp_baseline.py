#!/usr/bin/env python3
"""Train a curriculum pairwise MLP and evaluate global assignment baselines.

This program deliberately stops before any Transformer.  It consumes only a
manifest that certifies a per-payload SCT-cluster -> segment-refit -> Acts
chain, fits the MLP on source-disjoint training files, calibrates and selects
all assignment hyperparameters on validation, and opens test files only for
the resulting fixed operating points.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from baselines.global_assignment import ASSIGNMENT_METHODS, AssignmentConfig
from baselines.field_chi2_matching import pair_feature_names
from baselines.mlp_pair_classifier import (
    PairClassifierConfig,
    load_pair_classifier,
    save_pair_classifier,
)
from baselines.multistation_assignment import MultiStationAssignmentConfig
from datasets.physical_curriculum import (
    CurriculumSample,
    condition_axis_label,
    uniform_condition_axis,
)
from evaluation.pairwise_metrics import apply_temperature, binary_calibration, calibration_report
from training.curriculum_mlp import (
    adaptive_threshold_grid,
    build_candidate_sets,
    filter_candidate_scores,
    filter_candidate_sets,
    score_candidate_sets,
    score_station_pair_ensemble,
    train_curriculum_pair_classifier,
    train_station_pair_ensemble,
)
from training.global_assignment import (
    choose_global_operating_point,
    evaluate_global_assignment_sets,
    evaluate_multistation_assignment_sets,
    scan_global_assignment,
    scan_multistation_assignment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "physical_global_assignment_muon.yaml"
MULTISTATION_FLOW_METHOD = "multistation_flow"


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge a small YAML experiment override into its base."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def _read_config(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    chain = set() if seen is None else set(seen)
    if resolved in chain:
        raise ValueError(f"cyclic base_config reference at {resolved}")
    chain.add(resolved)
    with resolved.open(encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle)
    if not isinstance(supplied, Mapping):
        raise ValueError("global-assignment MLP config must be a YAML mapping")
    payload = dict(supplied)
    base_reference = payload.pop("base_config", None)
    if base_reference is None:
        return payload
    base_path = Path(str(base_reference)).expanduser()
    if not base_path.is_absolute():
        base_path = resolved.parent / base_path
    return _deep_merge(_read_config(base_path, chain), payload)


def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    supplied = _read_config(path)
    root = supplied.get("physical_curriculum_mlp", supplied)
    if not isinstance(root, Mapping):
        raise ValueError("physical_curriculum_mlp must be a mapping")
    mlp = root.get("curriculum_mlp")
    assignment = root.get("global_assignment_mlp")
    if not isinstance(mlp, Mapping) or not isinstance(assignment, Mapping):
        raise ValueError("configuration requires curriculum_mlp and global_assignment_mlp")
    return dict(root), dict(mlp), dict(assignment)


def _load_manifest_for_scope(
    manifest_path: str | Path, *, validation_only: bool
) -> tuple[Path, list[CurriculumSample], Mapping[str, object]]:
    """Load only the assets permitted by the declared execution scope."""
    from datasets.access_policy import AccessPolicyError, AccessScope, load_curriculum_for_scope

    if not validation_only:
        raise AccessPolicyError(
            "global-assignment loader cannot open the test split; "
            "--allow-sealed-test is not a development license"
        )
    return load_curriculum_for_scope(manifest_path, AccessScope.DEVELOPMENT_VALIDATION)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _source_split_audit(samples: Sequence[CurriculumSample]) -> dict[str, object]:
    source_to_split: dict[str, str] = {}
    uids_by_split: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    sources_by_split: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    for sample in samples:
        for source_id in sample.source_ids:
            prior = source_to_split.setdefault(source_id, sample.split)
            if prior != sample.split:
                raise ValueError(f"source '{source_id}' crosses data splits")
            sources_by_split[sample.split].add(source_id)
        uids_by_split[sample.split].update(sample.source_event_uids)
    overlaps = {
        f"{left}_{right}": sorted(uids_by_split[left].intersection(uids_by_split[right]))
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    if any(overlaps.values()):
        raise ValueError("original source event leakage across train/validation/test")
    return {
        "split_unit": "original_xAOD_file",
        "sources_by_split": {name: sorted(values) for name, values in sources_by_split.items()},
        "source_event_counts_by_split": {name: len(values) for name, values in uids_by_split.items()},
        "source_event_uid_overlap": overlaps,
        "strictly_disjoint": True,
    }


def _validate_condition_contract(root: Mapping[str, Any], samples: Sequence[CurriculumSample]) -> str:
    """Reject silently relabelled or mixed physical condition axes."""
    axis = uniform_condition_axis(samples)
    contract = root.get("condition_contract")
    if contract is None:
        return axis
    if not isinstance(contract, Mapping):
        raise ValueError("condition_contract must be a mapping when supplied")
    if str(contract.get("condition_axis", "")) != axis:
        raise ValueError("MLP condition axis differs from its declared physical contract")
    declared = contract.get("curriculum_condition_magnitudes")
    if not isinstance(declared, (list, tuple)) or not declared:
        raise ValueError("MLP condition contract lacks curriculum_condition_magnitudes")
    expected = tuple(sorted(float(value) for value in declared))
    for split in ("train", "validation"):
        observed = tuple(
            sorted({float(sample.curriculum_magnitude) for sample in samples if sample.split == split})
        )
        if observed != expected:
            raise ValueError(f"MLP {split} split differs from its frozen physical condition curriculum")
    return axis


def _flatten_scores(scores: Sequence[np.ndarray], sets: Sequence[object]) -> tuple[np.ndarray, np.ndarray]:
    score_parts = [np.asarray(values, dtype=np.float64) for values in scores if values.size]
    label_parts = [
        np.asarray(candidate_set.labels, dtype=bool)
        for candidate_set, values in zip(sets, scores)
        if values.size
    ]
    if not score_parts:
        raise ValueError("no candidate scores are available")
    return np.concatenate(score_parts), np.concatenate(label_parts)


def _calibrate_score_sets(
    sets: Sequence[object],
    scores: Sequence[np.ndarray],
    calibration_bins: int,
    scope: str,
) -> tuple[list[np.ndarray], dict[str, object]]:
    """Fit validation-only score calibration globally or per station pair.

    Per-pair temperatures are important when separate small MLPs have
    different raw-logit scales.  They remain a calibration-only transform:
    ordering inside each pair matrix is unchanged, and no truth value reaches
    an assignment routine.
    """
    if len(sets) != len(scores):
        raise ValueError("candidate sets and scores are not aligned")
    if scope not in {"global", "station_pair"}:
        raise ValueError("calibration_scope must be global or station_pair")
    raw, labels = _flatten_scores(scores, sets)
    if scope == "global":
        _, report = calibration_report(raw, labels, bins=calibration_bins)
        temperature = float(report["temperature"])
        return (
            [
                apply_temperature(np.asarray(values, dtype=np.float64), temperature)
                if values.size
                else np.empty(0, dtype=np.float64)
                for values in scores
            ],
            {"scope": scope, **report},
        )

    raw_by_pair: dict[tuple[int, int], list[np.ndarray]] = {}
    labels_by_pair: dict[tuple[int, int], list[np.ndarray]] = {}
    for candidate_set, values in zip(sets, scores):
        pair = tuple(int(value) for value in candidate_set.station_pair)
        raw_by_pair.setdefault(pair, []).append(np.asarray(values, dtype=np.float64))
        labels_by_pair.setdefault(pair, []).append(np.asarray(candidate_set.labels, dtype=bool))
    reports: dict[tuple[int, int], Mapping[str, object]] = {}
    temperatures: dict[tuple[int, int], float] = {}
    for pair in sorted(raw_by_pair):
        pair_scores = np.concatenate(raw_by_pair[pair])
        pair_labels = np.concatenate(labels_by_pair[pair])
        _, report = calibration_report(pair_scores, pair_labels, bins=calibration_bins)
        reports[pair] = report
        temperatures[pair] = float(report["temperature"])
    calibrated = [
        apply_temperature(
            np.asarray(values, dtype=np.float64),
            temperatures[tuple(int(value) for value in candidate_set.station_pair)],
        )
        if values.size
        else np.empty(0, dtype=np.float64)
        for candidate_set, values in zip(sets, scores)
    ]
    calibrated_flat, _ = _flatten_scores(calibrated, sets)
    return (
        calibrated,
        {
            "scope": scope,
            "temperature": None,
            "temperature_by_station_pair": {
                f"{pair[0]}->{pair[1]}": value for pair, value in sorted(temperatures.items())
            },
            "by_station_pair": {
                f"{pair[0]}->{pair[1]}": dict(report) for pair, report in sorted(reports.items())
            },
            "before": binary_calibration(raw, labels, bins=calibration_bins),
            "after": binary_calibration(calibrated_flat, labels, bins=calibration_bins),
            "fit_split": "validation_only",
        },
    )


def _apply_frozen_calibration_sets(
    sets: Sequence[object], scores: Sequence[np.ndarray], calibration: Mapping[str, object]
) -> list[np.ndarray]:
    """Apply a validation-fitted calibration map to a disjoint split."""
    if len(sets) != len(scores):
        raise ValueError("candidate sets and scores are not aligned")
    scope = str(calibration.get("scope", "global"))
    if scope == "global":
        temperature = float(calibration["temperature"])
        return [
            apply_temperature(np.asarray(values, dtype=np.float64), temperature)
            if values.size
            else np.empty(0, dtype=np.float64)
            for values in scores
        ]
    if scope != "station_pair":
        raise ValueError("unknown frozen calibration scope")
    raw_temperatures = calibration.get("temperature_by_station_pair")
    if not isinstance(raw_temperatures, Mapping):
        raise ValueError("station-pair calibration lacks temperature map")
    calibrated: list[np.ndarray] = []
    for candidate_set, values in zip(sets, scores):
        pair = tuple(int(value) for value in candidate_set.station_pair)
        key = f"{pair[0]}->{pair[1]}"
        if key not in raw_temperatures:
            raise ValueError(f"station-pair calibration lacks temperature for {key}")
        temperature = float(raw_temperatures[key])
        calibrated.append(
            apply_temperature(np.asarray(values, dtype=np.float64), temperature)
            if values.size
            else np.empty(0, dtype=np.float64)
        )
    return calibrated


def _normalise_candidate_gates(values: Sequence[object]) -> list[float | None]:
    """Validate the declared physical candidate gates without reordering them."""
    if not values:
        raise ValueError("candidate_chi2_gates must contain at least one gate")
    result: list[float | None] = []
    for raw in values:
        gate = None if raw is None else float(raw)
        if gate is not None and (not np.isfinite(gate) or gate <= 0.0):
            raise ValueError("candidate chi2 gates must be positive or null")
        if any(
            existing is None and gate is None
            or existing is not None and gate is not None and np.isclose(existing, gate)
            for existing in result
        ):
            raise ValueError("candidate_chi2_gates contains a duplicate")
        result.append(gate)
    return result


def _gate_key(gate: float | None) -> str:
    return "ungated" if gate is None else f"chi2_le_{gate:g}"


def _row_condition_magnitude(row: Mapping[str, object]) -> float:
    """Read a condition magnitude without silently relabelling rotations as mm."""
    value = row.get("condition_magnitude")
    if value is None:
        value = row.get("magnitude_mm")
    if value is None:
        raise ValueError("assignment row lacks condition_magnitude")
    return float(value)


def _annotate_condition_row(
    row: dict[str, object], *, condition_axis: str, condition_magnitude: float
) -> None:
    """Attach generic condition metadata and retain mm only for translations."""
    row["condition_axis"] = condition_axis
    row["condition_magnitude"] = float(condition_magnitude)
    if condition_axis == "translation_xy_mm":
        row["magnitude_mm"] = float(condition_magnitude)
    else:
        row.pop("magnitude_mm", None)


def _annotate_assignment_scan_row(
    row: dict[str, object], *, condition_axis: str, selection_scope: str
) -> None:
    """Describe the condition domain represented by one scan aggregate.

    ``scan_global_assignment`` and ``scan_multistation_assignment`` return
    one aggregate row for the exact collection of sets supplied to them.  A
    nominal-only scan therefore has a known condition coordinate (zero), but
    an all-condition scan must remain explicitly aggregate rather than being
    assigned an invented magnitude.
    """
    if selection_scope == "nominal_only":
        _annotate_condition_row(
            row, condition_axis=condition_axis, condition_magnitude=0.0
        )
        return
    if selection_scope != "all_magnitudes":
        raise ValueError("assignment selection scope is unsupported")
    row["condition_axis"] = condition_axis
    row["condition_scope"] = "all_configured_magnitudes"
    row.pop("condition_magnitude", None)
    row.pop("magnitude_mm", None)


def _assignment_scan_gate_keys(
    views: Mapping[str, object], selected_key: str, scan_all_candidate_gates: bool
) -> list[str]:
    """Choose either the score-level default graph or every validation graph."""
    if selected_key not in views:
        raise ValueError("selected candidate gate is unavailable for assignment scanning")
    return list(views) if scan_all_candidate_gates else [selected_key]


def _choose_candidate_gate(
    rows: Sequence[Mapping[str, object]],
    scope: str,
    minimum_truth_recall: float,
) -> dict[str, object]:
    """Choose one physical gate before the global-assignment hyperparameter scan.

    This deliberately uses only candidate-level validation evidence.  It keeps
    the expensive score-threshold/dustbin/Sinkhorn scan conditional on one
    fixed graph definition, while every declared gate remains reported in the
    candidate-recall and calibration tables.
    """
    if not np.isfinite(minimum_truth_recall) or not 0.0 <= minimum_truth_recall <= 1.0:
        raise ValueError("minimum candidate-gate truth recall must be in [0, 1]")
    scoped = [
        row
        for row in rows
        if scope == "all_magnitudes"
        or np.isclose(_row_condition_magnitude(row), 0.0)
    ]
    eligible: list[Mapping[str, object]] = []
    for row in scoped:
        recall = row.get("candidate_truth_recall")
        average_precision = row.get("average_precision")
        if recall is None or average_precision is None:
            continue
        if float(recall) >= minimum_truth_recall:
            eligible.append(row)
    if not eligible:
        raise RuntimeError(
            "no physical candidate gate reaches the declared validation candidate-recall floor"
        )

    def ordering(row: Mapping[str, object]) -> tuple[float, float]:
        gate = row.get("candidate_chi2_gate")
        gate_width = np.inf if gate is None else float(gate)
        return float(row["average_precision"]), -gate_width

    return dict(max(eligible, key=ordering))


def _select_magnitude(
    sets: Sequence[object], scores: Sequence[np.ndarray], magnitude: float
) -> tuple[list[object], list[np.ndarray]]:
    selected_sets: list[object] = []
    selected_scores: list[np.ndarray] = []
    for candidate_set, values in zip(sets, scores):
        if np.isclose(float(candidate_set.sample.curriculum_magnitude), magnitude):
            selected_sets.append(candidate_set)
            selected_scores.append(values)
    return selected_sets, selected_scores


def _assignment_configs(
    assignment: Mapping[str, Any], thresholds: Sequence[float]
) -> list[AssignmentConfig]:
    raw_methods = tuple(str(value) for value in assignment["methods"])
    unknown = set(raw_methods) - set(ASSIGNMENT_METHODS) - {MULTISTATION_FLOW_METHOD}
    if unknown:
        raise ValueError("unsupported global assignment methods: " + ", ".join(sorted(unknown)))
    penalties = [float(value) for value in assignment["unmatched_penalties"]]
    temperatures = [float(value) for value in assignment["sinkhorn_temperatures"]]
    iterations = int(assignment["sinkhorn_iterations"])
    configs: list[AssignmentConfig] = []
    for method in raw_methods:
        if method == MULTISTATION_FLOW_METHOD:
            continue
        for threshold in thresholds:
            if method in {"greedy", "hungarian"}:
                configs.append(
                    AssignmentConfig(
                        method=method, score_threshold=float(threshold), sinkhorn_iterations=iterations
                    )
                )
            elif method == "dustbin_hungarian":
                configs.extend(
                    AssignmentConfig(
                        method=method,
                        score_threshold=float(threshold),
                        unmatched_penalty=penalty,
                        sinkhorn_iterations=iterations,
                    )
                    for penalty in penalties
                )
            else:
                configs.extend(
                    AssignmentConfig(
                        method=method,
                        score_threshold=float(threshold),
                        unmatched_penalty=penalty,
                        sinkhorn_temperature=temperature,
                        sinkhorn_iterations=iterations,
                    )
                    for penalty in penalties
                    for temperature in temperatures
                )
    return configs


def _config_from_row(row: Mapping[str, object]) -> AssignmentConfig:
    return AssignmentConfig(
        method=str(row["assignment_method"]),  # type: ignore[arg-type]
        score_threshold=float(row["score_threshold"]),
        unmatched_penalty=float(row["unmatched_penalty"]),
        sinkhorn_temperature=float(row["sinkhorn_temperature"]),
        sinkhorn_iterations=int(row["sinkhorn_iterations"]),
    )


def _multistation_config_from_row(row: Mapping[str, object]) -> MultiStationAssignmentConfig:
    return MultiStationAssignmentConfig(
        score_threshold=float(row["score_threshold"]),
        unmatched_penalty=float(row["unmatched_penalty"]),
        maximum_hypotheses=int(row.get("maximum_hypotheses", 100_000)),
    )


def _candidate_rows(
    sets: Sequence[object],
    raw_scores: Sequence[np.ndarray],
    scores: Sequence[np.ndarray],
    calibration_bins: int,
    candidate_chi2_gate: float | None,
    candidate_calibration_scope: str,
    candidate_calibration_temperature: float | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    default = AssignmentConfig(method="greedy", score_threshold=1.0)
    condition_axis = uniform_condition_axis([candidate_set.sample for candidate_set in sets])
    for magnitude in sorted({float(candidate_set.sample.curriculum_magnitude) for candidate_set in sets}):
        selected_sets, selected_scores = _select_magnitude(sets, scores, magnitude)
        _, selected_raw_scores = _select_magnitude(sets, raw_scores, magnitude)
        result = evaluate_global_assignment_sets(selected_sets, selected_scores, default, calibration_bins)
        raw_result = evaluate_global_assignment_sets(
            selected_sets, selected_raw_scores, default, calibration_bins
        )
        candidate = result["candidate"]
        raw_candidate = raw_result["candidate"]
        row: dict[str, object] = {
                "condition_axis": condition_axis,
                "condition_magnitude": magnitude,
                "candidate_chi2_gate": candidate_chi2_gate,
                "candidate_calibration_scope": candidate_calibration_scope,
                "candidate_calibration_temperature": candidate_calibration_temperature,
                "candidate_truth_recall": candidate["candidate_truth_recall"],
                "candidate_rows": candidate["candidate_rows"],
                "positive_candidate_rows": candidate["positive_candidate_rows"],
                "roc_auc": candidate["roc_auc"],
                "average_precision": candidate["average_precision"],
                "raw_brier": raw_candidate["brier"],
                "raw_negative_log_likelihood": raw_candidate["negative_log_likelihood"],
                "raw_expected_calibration_error": raw_candidate["expected_calibration_error"],
                "brier": candidate["brier"],
                "negative_log_likelihood": candidate["negative_log_likelihood"],
                "expected_calibration_error": candidate["expected_calibration_error"],
            }
        _annotate_condition_row(row, condition_axis=condition_axis, condition_magnitude=magnitude)
        rows.append(row)
    return rows


def _evaluation_rows(
    method: str,
    operating_point: Mapping[str, object],
    sets: Sequence[object],
    scores: Sequence[np.ndarray],
    calibration_bins: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    is_multistation_flow = method == MULTISTATION_FLOW_METHOD
    config = (
        _multistation_config_from_row(operating_point)
        if is_multistation_flow
        else _config_from_row(operating_point)
    )
    candidate_chi2_gate = operating_point.get("candidate_chi2_gate")
    candidate_calibration_scope = str(operating_point["candidate_calibration_scope"])
    raw_temperature = operating_point.get("candidate_calibration_temperature")
    candidate_calibration_temperature = (
        None if raw_temperature is None else float(raw_temperature)
    )
    magnitude_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    condition_axis = uniform_condition_axis([candidate_set.sample for candidate_set in sets])
    for magnitude in sorted({float(candidate_set.sample.curriculum_magnitude) for candidate_set in sets}):
        selected_sets, selected_scores = _select_magnitude(sets, scores, magnitude)
        result = (
            evaluate_multistation_assignment_sets(
                selected_sets, selected_scores, config, calibration_bins
            )
            if is_multistation_flow
            else evaluate_global_assignment_sets(
                selected_sets, selected_scores, config, calibration_bins
            )
        )
        candidate = result["candidate"]
        association = result["association"]
        unmatched = result["unmatched"]
        magnitude_row: dict[str, object] = {
                "assignment_method": method,
                "condition_axis": condition_axis,
                "condition_magnitude": magnitude,
                "candidate_chi2_gate": candidate_chi2_gate,
                "candidate_calibration_scope": candidate_calibration_scope,
                "candidate_calibration_temperature": candidate_calibration_temperature,
                "score_threshold": config.score_threshold,
                "unmatched_penalty": config.unmatched_penalty,
                "sinkhorn_temperature": (
                    None if is_multistation_flow else config.sinkhorn_temperature
                ),
                "sinkhorn_iterations": (
                    None if is_multistation_flow else config.sinkhorn_iterations
                ),
                "candidate_truth_recall": candidate["candidate_truth_recall"],
                "candidate_rows": candidate["candidate_rows"],
                "average_precision": candidate["average_precision"],
                **association,
                **unmatched,
            }
        _annotate_condition_row(
            magnitude_row, condition_axis=condition_axis, condition_magnitude=magnitude
        )
        magnitude_rows.append(magnitude_row)
        for station_pair, payload in result["by_station_pair"].items():
            pair_candidate = payload["candidate"]
            pair_association = payload["association"]
            pair_unmatched = payload["unmatched"]
            pair_row: dict[str, object] = {
                    "assignment_method": method,
                    "condition_axis": condition_axis,
                    "condition_magnitude": magnitude,
                    "station_pair": station_pair,
                    "candidate_chi2_gate": candidate_chi2_gate,
                    "candidate_calibration_scope": candidate_calibration_scope,
                    "candidate_calibration_temperature": candidate_calibration_temperature,
                    "score_threshold": config.score_threshold,
                    "unmatched_penalty": config.unmatched_penalty,
                    "sinkhorn_temperature": (
                        None if is_multistation_flow else config.sinkhorn_temperature
                    ),
                    "candidate_truth_recall": pair_candidate["candidate_truth_recall"],
                    "candidate_rows": pair_candidate["candidate_rows"],
                    "roc_auc": pair_candidate["roc_auc"],
                    "average_precision": pair_candidate["average_precision"],
                    **pair_association,
                    **pair_unmatched,
                }
            _annotate_condition_row(
                pair_row, condition_axis=condition_axis, condition_magnitude=magnitude
            )
            pair_rows.append(pair_row)
    return magnitude_rows, pair_rows


def _plot(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, constrained_layout=True)
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["assignment_method"]), []).append(row)
    metrics = (
        ("association_efficiency", "global matching efficiency"),
        ("inclusive_association_purity", "inclusive purity"),
        ("inclusive_fake_rate", "inclusive fake rate"),
        ("missing_truth_unmatched_recall", "missing-tracklet unmatched recall"),
    )
    for axis, (field, label) in zip(axes.flat, metrics):
        for method, group in sorted(grouped.items()):
            ordered = sorted(group, key=_row_condition_magnitude)
            x_values = [_row_condition_magnitude(row) for row in ordered]
            y_values = [row[field] for row in ordered]
            axis.plot(x_values, y_values, marker="o", label=method)
        axis.set_ylabel(label)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    label = condition_axis_label(str(rows[0].get("condition_axis", "translation_xy_mm")))
    axes[1, 0].set_xlabel(label)
    axes[1, 1].set_xlabel(label)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_candidate_gates(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Plot physical-gate coverage and score quality independently of matching."""
    if not rows:
        return
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.3), sharex=True, constrained_layout=True)
    grouped: dict[float | None, list[Mapping[str, object]]] = {}
    for row in rows:
        gate = row.get("candidate_chi2_gate")
        grouped.setdefault(None if gate is None else float(gate), []).append(row)
    metrics = (
        ("candidate_truth_recall", "candidate truth recall"),
        ("average_precision", "candidate average precision"),
        ("expected_calibration_error", "calibrated ECE"),
    )
    for axis, (field, label) in zip(axes, metrics):
        for gate, group in sorted(grouped.items(), key=lambda item: (item[0] is None, item[0] or 0.0)):
            ordered = sorted(group, key=_row_condition_magnitude)
            axis.plot(
                [_row_condition_magnitude(row) for row in ordered],
                [np.nan if row[field] is None else float(row[field]) for row in ordered],
                marker="o",
                label="ungated" if gate is None else f"chi2 <= {gate:g}",
            )
        axis.set_ylabel(label)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
        axis.set_xlabel(condition_axis_label(str(rows[0].get("condition_axis", "translation_xy_mm"))))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="train and select on train/validation only; never materialize test candidates",
    )
    parser.add_argument(
        "--allow-sealed-test",
        action="store_true",
        help=(
            "permit this legacy entry point to open the sealed test split; only for "
            "reproducing a historical pre-seal evaluation, never for model selection"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "reuse an existing shared pairwise-MLP checkpoint for assignment-only "
            "comparisons; its station-pair and feature schema are verified"
        ),
    )
    parser.add_argument(
        "--q-over-p-mode",
        type=int,
        default=0,
        choices=(0, 3),
        help=(
            "propagation record variant used to build the physical candidate graph; "
            "the synthetic manifest must certify the same mode"
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    root, mlp, assignment = _load_config(config_path)
    test_permitted = bool(args.allow_sealed_test) and not args.validation_only
    if not test_permitted:
        declared = root.get("allowed_splits")
        if declared is not None and tuple(str(value) for value in declared) != ("train", "validation"):
            raise ValueError("test-forbidding runs must allow exactly train and validation")
        forbidden = {str(value) for value in root.get("forbidden_splits", ())}
        if forbidden and "test" not in forbidden:
            raise ValueError("test-forbidding runs must explicitly forbid test")
    manifest_path, samples, manifest = _load_manifest_for_scope(
        args.synthetic_manifest, validation_only=not test_permitted
    )
    if int(manifest.get("q_over_p_mode", -1)) != int(args.q_over_p_mode):
        raise ValueError(
            f"synthetic manifest q_over_p_mode does not match the requested mode {args.q_over_p_mode}"
        )
    condition_axis = _validate_condition_contract(root, samples)
    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty global-assignment output directory")
    output_root.mkdir(parents=True, exist_ok=True)

    split_audit = _source_split_audit(samples)
    stations = tuple(sorted(int(value) for value in root["refit"]["station_ids"]))
    station_pairs = tuple((left, right) for left in stations for right in stations if left < right)
    candidate_gates = _normalise_candidate_gates(mlp.get("candidate_chi2_gates", [None]))
    feature_set = str(mlp.get("feature_set", "residual_v1"))
    calibration_scope = str(mlp.get("calibration_scope", "global"))
    if calibration_scope not in {"global", "station_pair"}:
        raise ValueError("curriculum_mlp.calibration_scope must be global or station_pair")
    train_samples = [sample for sample in samples if sample.split == "train"]
    validation_samples = [sample for sample in samples if sample.split == "validation"]
    test_samples = [sample for sample in samples if sample.split == "test"]
    _write_json(
        output_root / "resolved_config.json",
        {
            "config_source": str(config_path),
            "synthetic_manifest": str(manifest_path),
            "source_split_audit": split_audit,
            "station_pairs": [list(pair) for pair in station_pairs],
            "candidate_chi2_gates": candidate_gates,
            "calibration_scope": calibration_scope,
            "q_over_p_mode": int(args.q_over_p_mode),
            "condition_axis": condition_axis,
            "physical_geometry_repropagation": True,
            "model_family": "pairwise_mlp_plus_global_assignment_only",
            "validation_only": not test_permitted,
            "sealed_test_permitted": test_permitted,
            "loaded_event_splits": sorted({str(sample.split) for sample in samples}),
            "forbidden_splits": [] if test_permitted else ["test"],
            "test_events_loaded": False if not test_permitted else None,
            "reused_checkpoint": (
                None if args.checkpoint is None else str(Path(args.checkpoint).expanduser().resolve())
            ),
            "config": {"curriculum_mlp": mlp, "global_assignment_mlp": assignment},
        },
    )

    train_sets = build_candidate_sets(
        train_samples, station_pairs, chi2_gate=None, feature_set=feature_set,
        q_over_p_mode=int(args.q_over_p_mode),
    )
    validation_sets = build_candidate_sets(
        validation_samples, station_pairs, chi2_gate=None, feature_set=feature_set,
        q_over_p_mode=int(args.q_over_p_mode),
    )
    model_topology = str(mlp.get("model_topology", "shared"))
    if model_topology not in {"shared", "station_pair_ensemble"}:
        raise ValueError("model_topology must be shared or station_pair_ensemble")
    model_config = PairClassifierConfig(
        hidden_dim=int(mlp["hidden_dim"]),
        epochs=1,
        batch_size=int(mlp["batch_size"]),
        learning_rate=float(mlp["learning_rate"]),
        weight_decay=float(mlp["weight_decay"]),
        seed=int(mlp["seed"]),
        device=str(mlp["device"]),
    )
    checkpoint: Path | None = None
    checkpoint_paths: dict[str, str] = {}
    model: Any | None = None
    artifact: Any | None = None
    models_by_pair: dict[tuple[int, int], Any] = {}
    artifacts_by_pair: dict[tuple[int, int], Any] = {}
    if args.checkpoint is not None and model_topology != "shared":
        raise ValueError("--checkpoint currently supports only model_topology=shared")
    if args.checkpoint is not None:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"pairwise MLP checkpoint does not exist: {checkpoint}")
        model, artifact = load_pair_classifier(checkpoint, device=str(mlp["device"]))
        expected_features = tuple(pair_feature_names(station_pairs, feature_set))
        if artifact.station_pairs != station_pairs:
            raise ValueError("checkpoint station-pair schema does not match the configured stations")
        if artifact.feature_names != expected_features:
            raise ValueError("checkpoint feature schema does not match curriculum_mlp.feature_set")
        checkpoint_paths["shared"] = str(checkpoint)
        training_summary = {
            "reused_checkpoint": str(checkpoint),
            "original_training_summary": artifact.training_summary,
        }
        training_device = str(next(model.parameters()).device)
    elif model_topology == "shared":
        model, artifact, curriculum = train_curriculum_pair_classifier(
            train_sets,
            validation_sets,
            station_pairs,
            mlp["stages"],
            model_config,
            feature_set=feature_set,
        )
        checkpoint = output_root / "mlp_pair_classifier.pt"
        save_pair_classifier(checkpoint, model, artifact)
        checkpoint_paths["shared"] = str(checkpoint)
        training_summary: object = curriculum
        training_device = artifact.training_summary["device"]
    else:
        models_by_pair, artifacts_by_pair, curriculum_by_pair = train_station_pair_ensemble(
            train_sets,
            validation_sets,
            station_pairs,
            mlp["stages"],
            model_config,
            feature_set=feature_set,
        )
        for pair in station_pairs:
            pair_checkpoint = output_root / f"mlp_pair_classifier_{pair[0]}_{pair[1]}.pt"
            save_pair_classifier(pair_checkpoint, models_by_pair[pair], artifacts_by_pair[pair])
            checkpoint_paths[f"{pair[0]}->{pair[1]}"] = str(pair_checkpoint)
        training_summary = {
            f"{pair[0]}->{pair[1]}": value for pair, value in curriculum_by_pair.items()
        }
        training_device = sorted(
            {str(artifact.training_summary["device"]) for artifact in artifacts_by_pair.values()}
        )
    _write_json(
        output_root / "training.json",
        {
            "model_topology": model_topology,
            "checkpoints": checkpoint_paths,
            "device": training_device,
            "curriculum": training_summary,
            "standardizer_fit": "all train-split physical candidate rows only",
            "calibration_scope": calibration_scope,
        },
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
    scope = str(assignment["validation_selection_scope"])
    if scope not in {"nominal_only", "all_magnitudes"}:
        raise ValueError("validation_selection_scope must be nominal_only or all_magnitudes")

    # The MLP is trained once on the ungated physical candidate universe above.
    # Candidate gates are first screened by a predeclared validation-only
    # recall/AP criterion; the full global-assignment grid then sees one fixed
    # sparse graph, rather than repeating the same dustbin scan per gate.
    validation_rows: list[dict[str, object]] = []
    validation_candidate_rows: list[dict[str, object]] = []
    gate_calibrations: dict[str, dict[str, object]] = {}
    validation_views: dict[str, tuple[float | None, list[object], list[np.ndarray]]] = {}
    for gate in candidate_gates:
        key = _gate_key(gate)
        gated_sets = filter_candidate_sets(validation_sets, gate)
        gated_raw_scores = filter_candidate_scores(validation_sets, validation_raw_scores, gate)
        validation_flat, validation_labels = _flatten_scores(gated_raw_scores, gated_sets)
        gated_scores, calibration = _calibrate_score_sets(
            gated_sets,
            gated_raw_scores,
            calibration_bins=int(mlp["calibration_bins"]),
            scope=calibration_scope,
        )
        temperature = calibration.get("temperature")
        scalar_temperature = None if temperature is None else float(temperature)
        gate_calibrations[key] = {
            **calibration,
            "candidate_chi2_gate": gate,
            "validation_candidate_rows": int(validation_labels.size),
            "validation_positive_candidate_rows": int(np.count_nonzero(validation_labels)),
            "fit_split": "validation_only",
        }
        validation_candidate_rows.extend(
            _candidate_rows(
                gated_sets,
                gated_raw_scores,
                gated_scores,
                int(mlp["calibration_bins"]),
                candidate_chi2_gate=gate,
                candidate_calibration_scope=calibration_scope,
                candidate_calibration_temperature=scalar_temperature,
            )
        )
        validation_views[key] = (gate, gated_sets, gated_scores)

    minimum_gate_recall = float(mlp.get("candidate_gate_minimum_truth_recall", 0.99))
    candidate_gate_selection = _choose_candidate_gate(
        validation_candidate_rows,
        scope=scope,
        minimum_truth_recall=minimum_gate_recall,
    )
    selected_gate = candidate_gate_selection.get("candidate_chi2_gate")
    selected_key = _gate_key(None if selected_gate is None else float(selected_gate))
    scan_all_candidate_gates = bool(assignment.get("scan_all_candidate_gates", False))
    bipartite_gate_keys = _assignment_scan_gate_keys(
        validation_views, selected_key, scan_all_candidate_gates
    )
    validation_rows: list[dict[str, object]] = []
    bipartite_scan_configurations = 0
    for gate_key in bipartite_gate_keys:
        gate, gate_sets_all, gate_scores_all = validation_views[gate_key]
        if scope == "nominal_only":
            gate_sets, gate_scores = _select_magnitude(gate_sets_all, gate_scores_all, 0.0)
        else:
            gate_sets, gate_scores = list(gate_sets_all), list(gate_scores_all)
        thresholds = adaptive_threshold_grid(
            gate_scores,
            [
                float(value)
                for value in assignment.get("threshold_grid", mlp["threshold_grid"])
            ],
            quantiles=int(
                assignment.get(
                    "adaptive_threshold_quantiles", mlp["adaptive_threshold_quantiles"]
                ),
            ),
        )
        scan_configs = _assignment_configs(assignment, thresholds)
        bipartite_scan_configurations += len(scan_configs)
        gate_rows = (
            scan_global_assignment(
                gate_sets,
                gate_scores,
                scan_configs,
                calibration_bins=int(mlp["calibration_bins"]),
            )
            if scan_configs
            else []
        )
        calibration = gate_calibrations[gate_key]
        temperature = calibration.get("temperature")
        for row in gate_rows:
            _annotate_assignment_scan_row(
                row,
                condition_axis=condition_axis,
                selection_scope=scope,
            )
            row["candidate_chi2_gate"] = gate
            row["candidate_calibration_scope"] = calibration_scope
            row["candidate_calibration_temperature"] = (
                None if temperature is None else float(temperature)
            )
        validation_rows.extend(gate_rows)
    flow_requested = MULTISTATION_FLOW_METHOD in {
        str(method) for method in assignment["methods"]
    }
    flow_gate_keys: list[str] = []
    flow_scan_details: list[dict[str, object]] = []
    flow_scan_configurations = 0
    flow_maximum_hypotheses: int | None = None
    flow_rows: list[dict[str, object]] = []
    if flow_requested:
        # Use the same explicit all-gates switch as the bipartite methods.
        # The default remains the preselected high-recall graph, while an
        # ablation can test whether a tighter physical graph plus four-station
        # consistency is enough to recover a valid operating point.
        flow_gate_keys = _assignment_scan_gate_keys(
            validation_views, selected_key, scan_all_candidate_gates
        )
        flow_penalties = [
            float(value)
            for value in assignment.get(
                "multistation_flow_unmatched_penalties",
                assignment["unmatched_penalties"],
            )
        ]
        flow_maximum_hypotheses = int(
            assignment.get("multistation_flow_maximum_hypotheses", 100_000)
        )
        for flow_gate_key in flow_gate_keys:
            flow_gate, flow_sets_all, flow_scores_all = validation_views[flow_gate_key]
            if scope == "nominal_only":
                flow_sets, flow_scores = _select_magnitude(flow_sets_all, flow_scores_all, 0.0)
            else:
                flow_sets, flow_scores = list(flow_sets_all), list(flow_scores_all)
            flow_thresholds = adaptive_threshold_grid(
                flow_scores,
                [
                    float(value)
                    for value in assignment.get(
                        "multistation_flow_threshold_grid",
                        assignment.get("threshold_grid", mlp["threshold_grid"]),
                    )
                ],
                quantiles=int(
                    assignment.get(
                        "multistation_flow_adaptive_threshold_quantiles",
                        assignment.get(
                            "adaptive_threshold_quantiles", mlp["adaptive_threshold_quantiles"]
                        ),
                    )
                ),
            )
            gate_flow_rows = scan_multistation_assignment(
                flow_sets,
                flow_scores,
                flow_thresholds,
                flow_penalties,
                calibration_bins=int(mlp["calibration_bins"]),
                maximum_hypotheses=flow_maximum_hypotheses,
            )
            calibration = gate_calibrations[flow_gate_key]
            temperature = calibration.get("temperature")
            for row in gate_flow_rows:
                _annotate_assignment_scan_row(
                    row,
                    condition_axis=condition_axis,
                    selection_scope=scope,
                )
                row["maximum_hypotheses"] = flow_maximum_hypotheses
                row["candidate_chi2_gate"] = flow_gate
                row["candidate_calibration_scope"] = calibration_scope
                row["candidate_calibration_temperature"] = (
                    None if temperature is None else float(temperature)
                )
            flow_rows.extend(gate_flow_rows)
            validation_rows.extend(gate_flow_rows)
            flow_scan_configurations += len(flow_thresholds) * len(flow_penalties)
            flow_scan_details.append(
                {
                    "candidate_chi2_gate": flow_gate,
                    "thresholds": flow_thresholds,
                    "unmatched_penalties": flow_penalties,
                }
            )

    _write_json(
        output_root / "calibration.json",
        {
            "fit_split": "validation_only",
            "candidate_gate_calibrations": gate_calibrations,
        },
    )
    # Route-level controls require one explicit station-pair calibration map.
    # Only an ungated physical graph is compatible with the V1/V2/V3 route
    # contract, so publish this convenience artifact only when that map exists.
    if "ungated" in gate_calibrations:
        _write_json(
            output_root / "route_calibration.json",
            {
                "fit_split": "validation_only",
                "calibration": gate_calibrations["ungated"],
                "candidate_chi2_gate": None,
                "physical_candidate_graph": "ungated_mode0_acts",
            },
        )
    _write_csv(output_root / "validation_candidate_metrics_by_gate_and_magnitude.csv", validation_candidate_rows)
    _plot_candidate_gates(
        output_root / "validation_candidate_gates_vs_misalignment.png",
        validation_candidate_rows,
    )
    _write_csv(output_root / "validation_global_assignment_scan.csv", validation_rows)
    minimum_nominal_efficiency = float(
        assignment.get("minimum_nominal_association_efficiency", 0.0)
    )
    selections: dict[str, object] = {}
    for method in assignment["methods"]:
        method_rows = [row for row in validation_rows if row["assignment_method"] == method]
        selections[str(method)] = choose_global_operating_point(
            method_rows,
            maximum_inclusive_fake_rate=float(assignment["target_inclusive_fake_rate"]),
            minimum_inclusive_purity=float(assignment["target_inclusive_purity"]),
            minimum_association_efficiency=minimum_nominal_efficiency,
        )
    has_validation_operating_point = any(
        isinstance(selected, Mapping) for selected in selections.values()
    )

    # Keep the primary floor separate from a validation-only diagnostic point.
    # When the declared primary cannot be met, the latter still exposes the
    # station-pair and misalignment failure pattern under the same quality
    # constraints, without opening the test split or changing selection.
    quality_diagnostic_selections: dict[str, object] = {}
    validation_quality_rows: list[dict[str, object]] = []
    validation_quality_pair_rows: list[dict[str, object]] = []
    for method in assignment["methods"]:
        method_name = str(method)
        method_rows = [
            row for row in validation_rows if row["assignment_method"] == method_name
        ]
        selected = choose_global_operating_point(
            method_rows,
            maximum_inclusive_fake_rate=float(assignment["target_inclusive_fake_rate"]),
            minimum_inclusive_purity=float(assignment["target_inclusive_purity"]),
            minimum_association_efficiency=0.0,
        )
        quality_diagnostic_selections[method_name] = selected
        if not isinstance(selected, Mapping):
            continue
        gate = selected.get("candidate_chi2_gate")
        key = _gate_key(None if gate is None else float(gate))
        _, selected_sets, selected_scores = validation_views[key]
        summary, pairs = _evaluation_rows(
            method_name,
            selected,
            selected_sets,
            selected_scores,
            int(mlp["calibration_bins"]),
        )
        validation_quality_rows.extend(summary)
        validation_quality_pair_rows.extend(pairs)

    _write_json(
        output_root / "validation_quality_diagnostic_operating_points.json",
        {
            "selection_split": "validation_only",
            "selection_scope": scope,
            "condition_axis": condition_axis,
            "selection_condition_magnitudes": [0.0] if scope == "nominal_only" else sorted(
                {float(candidate_set.sample.curriculum_magnitude) for candidate_set in validation_sets}
            ),
            "selection_magnitudes_mm": [0.0] if scope == "nominal_only" else sorted(
                {float(candidate_set.sample.curriculum_magnitude) for candidate_set in validation_sets}
            ),
            "maximum_inclusive_fake_rate": float(assignment["target_inclusive_fake_rate"]),
            "minimum_inclusive_purity": float(assignment["target_inclusive_purity"]),
            "ignored_primary_minimum_nominal_association_efficiency": minimum_nominal_efficiency,
            "methods": quality_diagnostic_selections,
            "test_opened": False,
        },
    )
    _write_csv(
        output_root / "validation_quality_global_matching_by_magnitude.csv",
        validation_quality_rows,
    )
    _write_csv(
        output_root / "validation_quality_global_matching_by_station_pair.csv",
        validation_quality_pair_rows,
    )
    _plot(
        output_root / "validation_quality_global_matching_vs_misalignment.png",
        validation_quality_rows,
    )

    _write_json(
        output_root / "validation_operating_points.json",
        {
            "selection_scope": scope,
            "condition_axis": condition_axis,
            "selection_condition_magnitudes": [0.0] if scope == "nominal_only" else sorted(
                {float(candidate_set.sample.curriculum_magnitude) for candidate_set in validation_sets}
            ),
            "selection_magnitudes_mm": [0.0] if scope == "nominal_only" else sorted(
                {float(candidate_set.sample.curriculum_magnitude) for candidate_set in validation_sets}
            ),
            "candidate_chi2_gates": candidate_gates,
            "candidate_gate_selection": {
                "selection_rule": (
                    "maximum_average_precision_subject_to_candidate_truth_recall_floor"
                    "_then_tightest_gate"
                ),
                "scope": scope,
                "minimum_candidate_truth_recall": minimum_gate_recall,
                "selected": candidate_gate_selection,
            },
            "candidate_gate_calibrations": gate_calibrations,
            "maximum_inclusive_fake_rate": float(assignment["target_inclusive_fake_rate"]),
            "minimum_inclusive_purity": float(assignment["target_inclusive_purity"]),
            "minimum_nominal_association_efficiency": minimum_nominal_efficiency,
            "scan_all_candidate_gates": scan_all_candidate_gates,
            "assignment_scan_candidate_gates": [
                validation_views[key][0] for key in bipartite_gate_keys
            ],
            "scan_configurations": (
                bipartite_scan_configurations + flow_scan_configurations
            ),
            "multistation_flow": {
                "requested": flow_requested,
                "assignment_scan_candidate_gates": [
                    validation_views[key][0] for key in flow_gate_keys
                ],
                "scans": flow_scan_details,
                "maximum_hypotheses": flow_maximum_hypotheses,
            },
            "methods": selections,
            "quality_diagnostic_output": {
                "selection_file": "validation_quality_diagnostic_operating_points.json",
                "by_magnitude_file": "validation_quality_global_matching_by_magnitude.csv",
                "by_station_pair_file": "validation_quality_global_matching_by_station_pair.csv",
            },
        },
    )

    # Test candidates and scores are materialised only after every candidate
    # gate, calibration temperature and assignment hyperparameter above is
    # fixed on validation.  Without --allow-sealed-test the loader never
    # resolves a test path, so no test ROOT file can be opened here.
    candidate_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    if test_permitted and has_validation_operating_point:
        test_sets = build_candidate_sets(
            test_samples, station_pairs, chi2_gate=None, feature_set=feature_set,
            q_over_p_mode=int(args.q_over_p_mode),
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
        test_views: dict[str, tuple[list[object], list[np.ndarray]]] = {}
        for key, (gate, _, _) in validation_views.items():
            calibration = gate_calibrations[key]
            gated_sets = filter_candidate_sets(test_sets, gate)
            gated_raw_scores = filter_candidate_scores(test_sets, test_raw_scores, gate)
            gated_scores = _apply_frozen_calibration_sets(
                gated_sets, gated_raw_scores, calibration
            )
            test_views[key] = (gated_sets, gated_scores)
            candidate_rows.extend(
                _candidate_rows(
                    gated_sets,
                    gated_raw_scores,
                    gated_scores,
                    int(mlp["calibration_bins"]),
                    candidate_chi2_gate=gate,
                    candidate_calibration_scope=calibration_scope,
                    candidate_calibration_temperature=(
                        None
                        if calibration.get("temperature") is None
                        else float(calibration["temperature"])
                    ),
                )
            )
        _write_csv(output_root / "test_candidate_metrics_by_gate_and_magnitude.csv", candidate_rows)
        _plot_candidate_gates(
            output_root / "candidate_gates_vs_misalignment.png",
            candidate_rows,
        )
        for method, selected in selections.items():
            if selected is None:
                continue
            gate = selected.get("candidate_chi2_gate")
            key = _gate_key(None if gate is None else float(gate))
            selected_sets, selected_scores = test_views[key]
            method_summary, method_pairs = _evaluation_rows(
                method,
                selected,
                selected_sets,
                selected_scores,
                int(mlp["calibration_bins"]),
            )
            summary_rows.extend(method_summary)
            pair_rows.extend(method_pairs)
        _write_csv(output_root / "test_global_matching_by_magnitude.csv", summary_rows)
        _write_csv(output_root / "test_global_matching_by_station_pair.csv", pair_rows)
        _plot(output_root / "global_matching_vs_misalignment.png", summary_rows)

    selected_rows = [value for value in selections.values() if isinstance(value, Mapping)]
    primary = (
        max(
            selected_rows,
            key=lambda row: (
                float(row["association_efficiency"]),
                -float(row["inclusive_fake_rate"]),
            ),
        )
        if selected_rows
        else None
    )
    _write_json(
        output_root / "metrics.json",
        {
            "method": "curriculum_pairwise_mlp_plus_global_assignment",
            "transformer_started": False,
            "physical_geometry_repropagation": True,
            "q_over_p_mode": int(args.q_over_p_mode),
            "condition_axis": condition_axis,
            "source_split_audit": split_audit,
            "candidate_chi2_gates": candidate_gates,
            "candidate_gate_selection": candidate_gate_selection,
            "candidate_gate_calibrations": gate_calibrations,
            "validation_operating_points": selections,
            "validation_quality_diagnostic_operating_points": quality_diagnostic_selections,
            "primary_validation_operating_point": primary,
            "test_opened": bool(test_permitted and has_validation_operating_point),
            "test_events_loaded": bool(test_permitted and has_validation_operating_point),
            "test_withheld_reason": (
                None
                if test_permitted and has_validation_operating_point
                else (
                    "sealed_test_not_permitted"
                    if not test_permitted
                    else "no_validation_operating_point_satisfies_primary_constraints"
                )
            ),
            "test_evaluated_methods": sorted({row["assignment_method"] for row in summary_rows}),
            "test_candidate_metrics": candidate_rows,
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_root),
                "checkpoints": checkpoint_paths,
                "model_topology": model_topology,
                "validation_operating_methods": [
                    method for method, selected in selections.items() if selected is not None
                ],
                "test_opened": bool(test_permitted and has_validation_operating_point),
                "test_evaluated_methods": sorted({row["assignment_method"] for row in summary_rows}),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
