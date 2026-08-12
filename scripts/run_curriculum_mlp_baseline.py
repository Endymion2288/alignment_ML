#!/usr/bin/env python3
"""Train and evaluate a source-disjoint physical-augmentation MLP baseline.

This command is intentionally the last stage before considering a Transformer:
all examples are drawn from a manifest that certifies the full displaced
geometry refit and Acts propagation chain, and source xAOD files are split
before any synthetic overlay resampling.
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

from baselines.mlp_pair_classifier import (
    PairClassifierConfig,
    load_pair_classifier,
    save_pair_classifier,
)
from datasets.physical_curriculum import CurriculumSample, load_synthetic_curriculum_manifest
from evaluation.pairwise_metrics import apply_temperature, calibration_report
from training.curriculum_mlp import (
    build_candidate_sets,
    choose_operating_threshold,
    adaptive_threshold_grid,
    evaluate_scored_candidate_sets,
    score_candidate_sets,
    threshold_scan,
    train_curriculum_pair_classifier,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "physical_curriculum_mlp_muon.yaml"


def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle)
    if not isinstance(supplied, Mapping):
        raise ValueError("curriculum MLP config must be a YAML mapping")
    root = supplied.get("physical_curriculum_mlp", supplied)
    if not isinstance(root, Mapping):
        raise ValueError("physical_curriculum_mlp must be a mapping")
    mlp = root.get("curriculum_mlp")
    if not isinstance(mlp, Mapping):
        raise ValueError("configuration is missing curriculum_mlp")
    return dict(root), dict(mlp)


def _gate_name(value: float | None) -> str:
    if value is None:
        return "ungated"
    return "chi2_" + format(value, ".8g").replace(".", "p").replace("-", "m")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _flatten_scores(scores: Sequence[np.ndarray], sets: Sequence[object]) -> tuple[np.ndarray, np.ndarray]:
    parts_scores = [np.asarray(values, dtype=np.float64) for values in scores if values.size]
    parts_labels = [candidate_set.labels for candidate_set, values in zip(sets, scores) if values.size]
    if not parts_scores:
        raise ValueError("no candidate scores are available")
    return np.concatenate(parts_scores), np.concatenate(parts_labels)


def _apply_temperature_sets(scores: Sequence[np.ndarray], temperature: float) -> list[np.ndarray]:
    return [
        apply_temperature(np.asarray(values, dtype=np.float64), temperature)
        if values.size
        else np.empty(0, dtype=np.float64)
        for values in scores
    ]


def _select_sets(
    sets: Sequence[object], scores: Sequence[np.ndarray], predicate: object
) -> tuple[list[object], list[np.ndarray]]:
    selected_sets: list[object] = []
    selected_scores: list[np.ndarray] = []
    for candidate_set, values in zip(sets, scores):
        if predicate(candidate_set):
            selected_sets.append(candidate_set)
            selected_scores.append(values)
    return selected_sets, selected_scores


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


def _summary_rows(
    gate: float | None,
    operating_point: str,
    threshold: float,
    test_sets: Sequence[object],
    test_raw_scores: Sequence[np.ndarray],
    test_scores: Sequence[np.ndarray],
    calibration_bins: int,
) -> list[dict[str, object]]:
    magnitudes = sorted({float(candidate_set.sample.magnitude_mm) for candidate_set in test_sets})
    rows: list[dict[str, object]] = []
    for magnitude in magnitudes:
        selected_sets, selected_scores = _select_sets(
            test_sets,
            test_scores,
            lambda candidate_set, value=magnitude: np.isclose(candidate_set.sample.magnitude_mm, value),
        )
        _, selected_raw_scores = _select_sets(
            test_sets,
            test_raw_scores,
            lambda candidate_set, value=magnitude: np.isclose(candidate_set.sample.magnitude_mm, value),
        )
        result = evaluate_scored_candidate_sets(
            selected_sets, selected_scores, threshold, calibration_bins
        )
        raw_result = evaluate_scored_candidate_sets(
            selected_sets, selected_raw_scores, threshold, calibration_bins
        )
        candidate = result["candidate"]
        raw_candidate = raw_result["candidate"]
        association = result["association"]
        rows.append(
            {
                "candidate_chi2_gate": gate,
                "operating_point": operating_point,
                "score_threshold": threshold,
                "magnitude_mm": magnitude,
                "candidate_truth_recall": candidate["candidate_truth_recall"],
                "candidate_rows": candidate["candidate_rows"],
                "positive_candidate_rows": candidate["positive_candidate_rows"],
                "roc_auc": candidate["roc_auc"],
                "average_precision": candidate["average_precision"],
                "raw_brier": raw_candidate["brier"],
                "raw_negative_log_likelihood": raw_candidate["negative_log_likelihood"],
                "raw_expected_calibration_error": raw_candidate["expected_calibration_error"],
                "calibrated_brier": candidate["brier"],
                "calibrated_negative_log_likelihood": candidate["negative_log_likelihood"],
                "expected_calibration_error": candidate["expected_calibration_error"],
                "association_efficiency": association["association_efficiency"],
                "inclusive_association_purity": association["inclusive_association_purity"],
                "inclusive_fake_rate": association["inclusive_fake_rate"],
                "correct_matches": association["correct_matches"],
                "possible_matches": association["possible_matches"],
                "predicted_matches": association["predicted_matches"],
            }
        )
    return rows


def _candidate_gate_audit_rows(
    gate: float | None,
    sets: Sequence[object],
    calibration_bins: int,
) -> list[dict[str, object]]:
    """Report physical candidate recall before learned-score thresholding."""
    rows: list[dict[str, object]] = []
    magnitudes = sorted({float(candidate_set.sample.magnitude_mm) for candidate_set in sets})
    for magnitude in magnitudes:
        selected_sets, _ = _select_sets(
            sets,
            [np.empty(0, dtype=np.float64) for _ in sets],
            lambda candidate_set, value=magnitude: np.isclose(candidate_set.sample.magnitude_mm, value),
        )
        zero_scores = [np.zeros(candidate_set.labels.size, dtype=np.float64) for candidate_set in selected_sets]
        result = evaluate_scored_candidate_sets(
            selected_sets, zero_scores, score_threshold=1.1, calibration_bins=calibration_bins
        )
        candidate = result["candidate"]
        rows.append(
            {
                "candidate_chi2_gate": gate,
                "magnitude_mm": magnitude,
                "candidate_truth_recall": candidate["candidate_truth_recall"],
                "candidate_rows": candidate["candidate_rows"],
                "positive_candidate_rows": candidate["positive_candidate_rows"],
                "truth_pairs_with_unique_known_endpoints": candidate[
                    "truth_pairs_with_unique_known_endpoints"
                ],
            }
        )
    return rows


def _station_pair_rows(
    gate: float | None,
    operating_point: str,
    threshold: float,
    test_sets: Sequence[object],
    test_raw_scores: Sequence[np.ndarray],
    test_scores: Sequence[np.ndarray],
    calibration_bins: int,
    threshold_scope: str,
    validation_magnitude_mm: float | None,
) -> list[dict[str, object]]:
    result = evaluate_scored_candidate_sets(test_sets, test_scores, threshold, calibration_bins)
    raw_result = evaluate_scored_candidate_sets(
        test_sets, test_raw_scores, threshold, calibration_bins
    )
    rows: list[dict[str, object]] = []
    for station_pair, payload in result["by_station_pair"].items():
        candidate = payload["candidate"]
        raw_candidate = raw_result["by_station_pair"][station_pair]["candidate"]
        association = payload["association"]
        rows.append(
            {
                "candidate_chi2_gate": gate,
                "operating_point": operating_point,
                "score_threshold": threshold,
                "threshold_scope": threshold_scope,
                "validation_magnitude_mm": validation_magnitude_mm,
                "station_pair": station_pair,
                "candidate_truth_recall": candidate["candidate_truth_recall"],
                "candidate_rows": candidate["candidate_rows"],
                "positive_candidate_rows": candidate["positive_candidate_rows"],
                "roc_auc": candidate["roc_auc"],
                "average_precision": candidate["average_precision"],
                "raw_brier": raw_candidate["brier"],
                "raw_negative_log_likelihood": raw_candidate["negative_log_likelihood"],
                "raw_expected_calibration_error": raw_candidate["expected_calibration_error"],
                "calibrated_brier": candidate["brier"],
                "calibrated_negative_log_likelihood": candidate["negative_log_likelihood"],
                "expected_calibration_error": candidate["expected_calibration_error"],
                "association_efficiency": association["association_efficiency"],
                "inclusive_association_purity": association["inclusive_association_purity"],
                "inclusive_fake_rate": association["inclusive_fake_rate"],
            }
        )
    return rows


def _same_magnitude_sets(
    sets: Sequence[object],
    scores: Sequence[np.ndarray],
    magnitude_mm: float,
) -> tuple[list[object], list[np.ndarray]]:
    """Select one physical-payload magnitude without crossing source splits."""
    return _select_sets(
        sets,
        scores,
        lambda candidate_set, value=magnitude_mm: np.isclose(
            candidate_set.sample.magnitude_mm, value
        ),
    )


def _magnitude_conditioned_operating_rows(
    gate: float | None,
    operation_name: str,
    constraint: str,
    target: float,
    validation_sets: Sequence[object],
    validation_scores: Sequence[np.ndarray],
    test_sets: Sequence[object],
    test_raw_scores: Sequence[np.ndarray],
    test_scores: Sequence[np.ndarray],
    threshold_grid: Sequence[float],
    calibration_bins: int,
    adaptive_quantiles: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Use held-out validation payloads to fix one threshold per magnitude.

    A global threshold is retained separately for deployment realism.  This
    companion calculation makes every scan point comparable at the requested
    validation fake-rate or purity target, without using any test score while
    choosing the threshold.
    """
    test_magnitudes = sorted({float(candidate_set.sample.magnitude_mm) for candidate_set in test_sets})
    summary_rows: list[dict[str, object]] = []
    station_pair_rows: list[dict[str, object]] = []
    selections: dict[str, object] = {}
    for magnitude in test_magnitudes:
        validation_at_magnitude, validation_scores_at_magnitude = _same_magnitude_sets(
            validation_sets, validation_scores, magnitude
        )
        test_at_magnitude, test_scores_at_magnitude = _same_magnitude_sets(
            test_sets, test_scores, magnitude
        )
        _, test_raw_scores_at_magnitude = _same_magnitude_sets(
            test_sets, test_raw_scores, magnitude
        )
        scan_rows = threshold_scan(
            validation_at_magnitude,
            validation_scores_at_magnitude,
            adaptive_threshold_grid(
                validation_scores_at_magnitude,
                threshold_grid,
                quantiles=adaptive_quantiles,
            ),
            calibration_bins,
        )
        selected = choose_operating_threshold(scan_rows, constraint, target)
        selections[format(magnitude, ".8g")] = {
            "validation_magnitude_mm": magnitude,
            "constraint": constraint,
            "target": target,
            "selected": selected,
            "threshold_scan_rows": scan_rows,
        }
        if selected is None:
            continue
        threshold = float(selected["score_threshold"])
        point_rows = _summary_rows(
            gate,
            operation_name,
            threshold,
            test_at_magnitude,
            test_raw_scores_at_magnitude,
            test_scores_at_magnitude,
            calibration_bins,
        )
        for row in point_rows:
            row["threshold_scope"] = "validation_same_magnitude"
            row["validation_magnitude_mm"] = magnitude
        summary_rows.extend(point_rows)
        station_pair_rows.extend(
            _station_pair_rows(
                gate,
                operation_name,
                threshold,
                test_at_magnitude,
                test_raw_scores_at_magnitude,
                test_scores_at_magnitude,
                calibration_bins,
                threshold_scope="validation_same_magnitude",
                validation_magnitude_mm=magnitude,
            )
        )
    return summary_rows, station_pair_rows, selections


def _plot(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    selected = [
        row
        for row in rows
        if row["operating_point"].startswith("fixed_fake_rate")
        and row.get("threshold_scope") == "validation_same_magnitude"
    ]
    if not selected:
        return
    figure, axes = plt.subplots(2, 1, figsize=(7.6, 6.2), sharex=True, constrained_layout=True)
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in selected:
        key = f"gate={row['candidate_chi2_gate']}"
        grouped.setdefault(key, []).append(row)
    for name, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: float(row["magnitude_mm"]))
        x_values = [float(row["magnitude_mm"]) for row in ordered]
        efficiency = [row["association_efficiency"] for row in ordered]
        recall = [row["candidate_truth_recall"] for row in ordered]
        axes[0].plot(x_values, efficiency, marker="o", label=name)
        axes[1].plot(x_values, recall, marker="o", label=name)
    axes[0].set_ylabel("association efficiency")
    axes[1].set_ylabel("candidate recall")
    axes[1].set_xlabel("injected misalignment magnitude [mm]")
    for axis in axes:
        axis.set_ylim(-0.02, 1.02)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_candidate_metrics(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Plot gate coverage and pairwise discrimination without an assignment cut."""
    if not rows:
        return
    figure, axes = plt.subplots(2, 1, figsize=(7.6, 6.2), sharex=True, constrained_layout=True)
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        key = f"gate={row['candidate_chi2_gate']}"
        grouped.setdefault(key, []).append(row)
    for name, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: float(row["magnitude_mm"]))
        x_values = [float(row["magnitude_mm"]) for row in ordered]
        recall = [row["candidate_truth_recall"] for row in ordered]
        average_precision = [row["average_precision"] for row in ordered]
        axes[0].plot(x_values, recall, marker="o", label=name)
        axes[1].plot(x_values, average_precision, marker="o", label=name)
    axes[0].set_ylabel("candidate recall")
    axes[1].set_ylabel("average precision")
    axes[1].set_xlabel("injected misalignment magnitude [mm]")
    for axis in axes:
        axis.set_ylim(-0.02, 1.02)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _transformer_screening(rows: Sequence[Mapping[str, object]], config: Mapping[str, object]) -> dict[str, object]:
    """State the predeclared evidence test without starting a Transformer."""
    supplied = config.get("transformer_screening", {})
    if not isinstance(supplied, Mapping):
        raise ValueError("transformer_screening must be a mapping")
    minimum_recall = float(supplied.get("minimum_candidate_truth_recall", 0.95))
    minimum_drop = float(supplied.get("minimum_efficiency_drop", 0.10))
    selected = [row for row in rows if row["operating_point"].startswith("fixed_fake_rate")]
    by_gate: dict[str, list[Mapping[str, object]]] = {}
    for row in selected:
        by_gate.setdefault(str(row["candidate_chi2_gate"]), []).append(row)
    evidence: list[dict[str, object]] = []
    for gate, group in sorted(by_gate.items()):
        ordered = sorted(group, key=lambda row: float(row["magnitude_mm"]))
        low = ordered[0]
        high = ordered[-1]
        low_efficiency = low["association_efficiency"]
        high_efficiency = high["association_efficiency"]
        high_recall = high["candidate_truth_recall"]
        drop = (
            None
            if low_efficiency is None or high_efficiency is None
            else float(low_efficiency) - float(high_efficiency)
        )
        candidate_sufficient = high_recall is not None and float(high_recall) >= minimum_recall
        efficiency_lost = drop is not None and drop >= minimum_drop
        evidence.append(
            {
                "candidate_chi2_gate": gate,
                "low_magnitude_mm": low["magnitude_mm"],
                "high_magnitude_mm": high["magnitude_mm"],
                "low_association_efficiency": low_efficiency,
                "high_association_efficiency": high_efficiency,
                "efficiency_drop": drop,
                "high_candidate_truth_recall": high_recall,
                "candidate_recall_sufficient": candidate_sufficient,
                "mlp_efficiency_lost": efficiency_lost,
                "supports_global_context_hypothesis": bool(candidate_sufficient and efficiency_lost),
            }
        )
    return {
        "purpose": (
            "screening only; a true value is evidence to compare a multi-station model, "
            "not an instruction to start one automatically"
        ),
        "minimum_candidate_truth_recall": minimum_recall,
        "minimum_efficiency_drop": minimum_drop,
        "by_candidate_gate": evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--checkpoint-root",
        default=None,
        help=(
            "reuse one completed MLP checkpoint per candidate gate instead of retraining; "
            "the validation-only calibration and threshold scan are recomputed"
        ),
    )
    args = parser.parse_args()
    manifest_path, samples, manifest = load_synthetic_curriculum_manifest(args.synthetic_manifest)
    config_path = Path(args.config).expanduser().resolve()
    root_config, config = _load_config(config_path)
    if int(manifest.get("q_over_p_mode", -1)) != 0:
        raise ValueError("curriculum MLP baseline supports only physical mode-0 propagation")
    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty MLP output directory")
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = (
        None if args.checkpoint_root is None else Path(args.checkpoint_root).expanduser().resolve()
    )
    if checkpoint_root is not None and not checkpoint_root.is_dir():
        raise FileNotFoundError(f"checkpoint root does not exist: {checkpoint_root}")
    stations = tuple(sorted(int(value) for value in root_config["refit"]["station_ids"]))
    station_pairs = tuple((source, target) for source in stations for target in stations if source < target)
    train_samples = [sample for sample in samples if sample.split == "train"]
    validation_samples = [sample for sample in samples if sample.split == "validation"]
    test_samples = [sample for sample in samples if sample.split == "test"]
    split_audit = _source_split_audit(samples)
    _write_json(
        output_root / "resolved_config.json",
        {
            "config_source": str(config_path),
            "synthetic_manifest": str(manifest_path),
            "source_split_audit": split_audit,
            "station_pairs": [list(pair) for pair in station_pairs],
            "q_over_p_mode": 0,
            "physical_geometry_repropagation": True,
            "checkpoint_root": None if checkpoint_root is None else str(checkpoint_root),
            "config": config,
        },
    )
    model_config = PairClassifierConfig(
        hidden_dim=int(config["hidden_dim"]),
        epochs=1,
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        seed=int(config["seed"]),
        device=str(config["device"]),
    )
    gates: list[float | None] = []
    for raw_gate in config["candidate_chi2_gates"]:
        gate = None if raw_gate is None else float(raw_gate)
        if gate is not None and gate <= 0.0:
            raise ValueError("candidate chi2 gates must be positive or null")
        if gate not in gates:
            gates.append(gate)
    overall: dict[str, object] = {
        "method": (
            "physical_misalignment_augmentation_curriculum_mlp"
            if checkpoint_root is None
            else "physical_misalignment_augmentation_curriculum_mlp_reassessment"
        ),
        "synthetic_manifest": str(manifest_path),
        "physical_geometry_repropagation": True,
        "source_split_audit": split_audit,
        "station_pairs": [list(pair) for pair in station_pairs],
        "checkpoint_root": None if checkpoint_root is None else str(checkpoint_root),
        "candidate_gate_results": [],
    }
    all_summary_rows: list[dict[str, object]] = []
    all_pair_rows: list[dict[str, object]] = []
    all_gate_audit_rows: list[dict[str, object]] = []
    all_candidate_metric_rows: list[dict[str, object]] = []
    all_candidate_pair_rows: list[dict[str, object]] = []
    for gate in gates:
        gate_root = output_root / _gate_name(gate)
        gate_root.mkdir(parents=True)
        candidate_gate_rows: list[dict[str, object]] = []
        try:
            train_sets = (
                build_candidate_sets(train_samples, station_pairs, gate)
                if checkpoint_root is None
                else []
            )
            validation_sets = build_candidate_sets(validation_samples, station_pairs, gate)
            test_sets = build_candidate_sets(test_samples, station_pairs, gate)
            candidate_gate_rows = _candidate_gate_audit_rows(
                gate, test_sets, int(config["calibration_bins"])
            )
            _write_csv(gate_root / "candidate_gate_test_by_magnitude.csv", candidate_gate_rows)
            all_gate_audit_rows.extend(candidate_gate_rows)
            checkpoint = gate_root / "mlp_pair_classifier.pt"
            if checkpoint_root is None:
                model, artifact, curriculum = train_curriculum_pair_classifier(
                    train_sets,
                    validation_sets,
                    station_pairs,
                    config["stages"],
                    model_config,
                )
                save_pair_classifier(checkpoint, model, artifact)
            else:
                checkpoint = checkpoint_root / _gate_name(gate) / "mlp_pair_classifier.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"checkpoint is missing for gate {gate}: {checkpoint}")
                model, artifact = load_pair_classifier(checkpoint, device=str(config["device"]))
                if tuple(artifact.station_pairs) != station_pairs:
                    raise ValueError(f"checkpoint station-pair contract differs for gate {gate}")
                curriculum = [
                    {
                        "reused_checkpoint": str(checkpoint),
                        "reassessment": "no model training; validation-only recalibration and threshold scan",
                    }
                ]
            validation_raw_scores = score_candidate_sets(model, artifact, validation_sets)
            raw_flat, labels_flat = _flatten_scores(validation_raw_scores, validation_sets)
            calibrated_flat, calibration = calibration_report(
                raw_flat, labels_flat, bins=int(config["calibration_bins"])
            )
            # Reapply the fitted scalar to per-event arrays; the flattened
            # value is retained only as validation evidence in calibration.json.
            del calibrated_flat
            validation_scores = _apply_temperature_sets(
                validation_raw_scores, float(calibration["temperature"])
            )
            scan_rows = threshold_scan(
                validation_sets,
                validation_scores,
                adaptive_threshold_grid(
                    validation_scores,
                    [float(value) for value in config["threshold_grid"]],
                    quantiles=int(config["adaptive_threshold_quantiles"]),
                ),
                int(config["calibration_bins"]),
            )
            _write_csv(gate_root / "validation_threshold_scan.csv", scan_rows)
            global_operating_points = {
                f"global_fixed_fake_rate_{float(config['target_inclusive_fake_rate']):.3g}": choose_operating_threshold(
                    scan_rows,
                    "inclusive_fake_rate",
                    float(config["target_inclusive_fake_rate"]),
                ),
                f"global_fixed_purity_{float(config['target_inclusive_purity']):.3g}": choose_operating_threshold(
                    scan_rows,
                    "inclusive_association_purity",
                    float(config["target_inclusive_purity"]),
                ),
            }
            test_raw_scores = score_candidate_sets(model, artifact, test_sets)
            test_scores = _apply_temperature_sets(test_raw_scores, float(calibration["temperature"]))
            # Candidate discrimination is meaningful even if no non-empty
            # assignment can meet the strict fake-rate/purity constraint.
            candidate_metric_rows = _summary_rows(
                gate,
                "candidate_only",
                1.1,
                test_sets,
                test_raw_scores,
                test_scores,
                int(config["calibration_bins"]),
            )
            for row in candidate_metric_rows:
                row["threshold_scope"] = "no_assignment_candidate_metrics"
                row["validation_magnitude_mm"] = None
            candidate_pair_rows = _station_pair_rows(
                gate,
                "candidate_only",
                1.1,
                test_sets,
                test_raw_scores,
                test_scores,
                int(config["calibration_bins"]),
                threshold_scope="no_assignment_candidate_metrics",
                validation_magnitude_mm=None,
            )
            _write_csv(gate_root / "test_candidate_metrics_by_magnitude.csv", candidate_metric_rows)
            _write_csv(gate_root / "test_candidate_metrics_by_station_pair.csv", candidate_pair_rows)
            _write_json(
                gate_root / "calibration.json",
                {
                    **calibration,
                    "candidate_chi2_gate": gate,
                    "validation_candidate_rows": int(labels_flat.size),
                    "validation_positive_candidate_rows": int(np.count_nonzero(labels_flat)),
                },
            )
            _write_json(
                gate_root / "training.json",
                {
                    "candidate_chi2_gate": gate,
                    "curriculum": curriculum,
                    "standardizer_fit": "all train-split physical candidate rows only",
                    "checkpoint": str(checkpoint),
                },
            )
            gate_summary_rows: list[dict[str, object]] = []
            gate_pair_rows: list[dict[str, object]] = []
            for name, selected in global_operating_points.items():
                if selected is None:
                    continue
                threshold = float(selected["score_threshold"])
                global_rows = _summary_rows(
                    gate,
                    name,
                    threshold,
                    test_sets,
                    test_raw_scores,
                    test_scores,
                    int(config["calibration_bins"]),
                )
                for row in global_rows:
                    row["threshold_scope"] = "validation_all_magnitudes"
                    row["validation_magnitude_mm"] = None
                gate_summary_rows.extend(global_rows)
                gate_pair_rows.extend(
                    _station_pair_rows(
                        gate,
                        name,
                        threshold,
                        test_sets,
                        test_raw_scores,
                        test_scores,
                        int(config["calibration_bins"]),
                        threshold_scope="validation_all_magnitudes",
                        validation_magnitude_mm=None,
                    )
                )
            fixed_fake_name = f"fixed_fake_rate_{float(config['target_inclusive_fake_rate']):.3g}"
            fixed_purity_name = f"fixed_purity_{float(config['target_inclusive_purity']):.3g}"
            fixed_fake_rows, fixed_fake_pair_rows, fixed_fake_selections = _magnitude_conditioned_operating_rows(
                gate,
                fixed_fake_name,
                "inclusive_fake_rate",
                float(config["target_inclusive_fake_rate"]),
                validation_sets,
                validation_scores,
                test_sets,
                test_raw_scores,
                test_scores,
                [float(value) for value in config["threshold_grid"]],
                int(config["calibration_bins"]),
                int(config["adaptive_threshold_quantiles"]),
            )
            fixed_purity_rows, fixed_purity_pair_rows, fixed_purity_selections = _magnitude_conditioned_operating_rows(
                gate,
                fixed_purity_name,
                "inclusive_association_purity",
                float(config["target_inclusive_purity"]),
                validation_sets,
                validation_scores,
                test_sets,
                test_raw_scores,
                test_scores,
                [float(value) for value in config["threshold_grid"]],
                int(config["calibration_bins"]),
                int(config["adaptive_threshold_quantiles"]),
            )
            _write_json(
                gate_root / "validation_threshold_scan_by_magnitude.json",
                {
                    "candidate_chi2_gate": gate,
                    "temperature": float(calibration["temperature"]),
                    "fixed_fake_rate": fixed_fake_selections,
                    "fixed_purity": fixed_purity_selections,
                },
            )
            gate_summary_rows.extend(fixed_fake_rows)
            gate_summary_rows.extend(fixed_purity_rows)
            gate_pair_rows.extend(fixed_fake_pair_rows)
            gate_pair_rows.extend(fixed_purity_pair_rows)
            _write_csv(gate_root / "test_by_magnitude.csv", gate_summary_rows)
            _write_csv(gate_root / "test_by_station_pair.csv", gate_pair_rows)
            gate_result = {
                "status": "complete",
                "candidate_chi2_gate": gate,
                "checkpoint": str(checkpoint),
                "operating_points": {
                    "global_validation": global_operating_points,
                    "magnitude_conditioned_validation": {
                        fixed_fake_name: fixed_fake_selections,
                        fixed_purity_name: fixed_purity_selections,
                    },
                },
                "validation_threshold_rows": len(scan_rows),
                "candidate_gate_test_by_magnitude": candidate_gate_rows,
                "test_candidate_metrics_by_magnitude": candidate_metric_rows,
                "test_candidate_metrics_by_station_pair": candidate_pair_rows,
                "test_by_magnitude": gate_summary_rows,
                "test_by_station_pair": gate_pair_rows,
            }
            _write_json(gate_root / "metrics.json", gate_result)
            overall["candidate_gate_results"].append(gate_result)
            all_summary_rows.extend(gate_summary_rows)
            all_pair_rows.extend(gate_pair_rows)
            all_candidate_metric_rows.extend(candidate_metric_rows)
            all_candidate_pair_rows.extend(candidate_pair_rows)
        except Exception as error:
            failure = {
                "status": "failed",
                "candidate_chi2_gate": gate,
                "error": str(error),
                "candidate_gate_test_by_magnitude": candidate_gate_rows,
            }
            _write_json(gate_root / "failure.json", failure)
            overall["candidate_gate_results"].append(failure)
    _write_csv(output_root / "test_efficiency_vs_misalignment.csv", all_summary_rows)
    _write_csv(output_root / "test_station_pair_metrics.csv", all_pair_rows)
    _write_csv(output_root / "candidate_gate_test_by_magnitude.csv", all_gate_audit_rows)
    _write_csv(output_root / "test_candidate_metrics_by_magnitude.csv", all_candidate_metric_rows)
    _write_csv(output_root / "test_candidate_metrics_by_station_pair.csv", all_candidate_pair_rows)
    _plot(output_root / "test_efficiency_vs_misalignment.png", all_summary_rows)
    _plot_candidate_metrics(output_root / "candidate_metrics_vs_misalignment.png", all_candidate_metric_rows)
    overall["transformer_screening"] = _transformer_screening(all_summary_rows, config)
    _write_json(output_root / "metrics.json", overall)
    print(json.dumps(overall, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
