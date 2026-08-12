#!/usr/bin/env python3
"""Diagnose a saved pairwise MLP on validation data only.

This tool deliberately never opens test samples.  It decomposes physical
candidate scores by station pair and evaluation-only synthetic fake role, so a
failed nominal global-assignment operating point can be investigated without
using test labels to choose a remedy.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from baselines.mlp_pair_classifier import load_pair_classifier
from baselines.field_chi2_matching import PAIR_FEATURE_SETS, pair_feature_names
from datasets.physical_curriculum import load_synthetic_curriculum_manifest
from datasets.synthetic_overlay import SYNTHETIC_ROLE_NAMES, SYNTHETIC_ROLE_TRUE
from evaluation.pairwise_metrics import apply_temperature, binary_calibration
from training.curriculum_mlp import (
    build_candidate_sets,
    filter_candidate_scores,
    filter_candidate_sets,
    score_candidate_sets,
)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {
            "score_min": None,
            "score_p10": None,
            "score_p50": None,
            "score_p90": None,
            "score_p99": None,
            "score_max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    result = np.quantile(array, [0.0, 0.10, 0.50, 0.90, 0.99, 1.0])
    return {
        "score_min": float(result[0]),
        "score_p10": float(result[1]),
        "score_p50": float(result[2]),
        "score_p90": float(result[3]),
        "score_p99": float(result[4]),
        "score_max": float(result[5]),
    }


def _role_category(event: Any, source_index: int, target_index: int, label: bool) -> str:
    """Classify a candidate for evaluation without exposing a model feature."""
    if label:
        return "truth_matched"
    if event.synthetic_role is None:
        return "unlabelled_negative"
    source_role = int(event.synthetic_role[source_index])
    target_role = int(event.synthetic_role[target_index])
    if source_role == SYNTHETIC_ROLE_TRUE and target_role == SYNTHETIC_ROLE_TRUE:
        return "true_to_different_truth"
    source_name = SYNTHETIC_ROLE_NAMES.get(source_role, f"unknown_{source_role}")
    target_name = SYNTHETIC_ROLE_NAMES.get(target_role, f"unknown_{target_role}")
    return f"{source_name}->{target_name}"


def _best_candidate_precision(
    scores: np.ndarray, labels: np.ndarray
) -> dict[str, float | int | None]:
    """Report the best non-empty candidate precision at a score cut.

    This is diagnostic only, not an operating-point selector: it ignores the
    event-wise one-to-one constraint and should not be confused with a global
    assignment result.
    """
    if not scores.size:
        return {
            "best_candidate_precision": None,
            "best_candidate_precision_threshold": None,
            "best_candidate_precision_rows": 0,
            "best_candidate_precision_true_rows": 0,
        }
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    positive = np.cumsum(sorted_labels, dtype=np.int64)
    ranks = np.arange(1, sorted_labels.size + 1, dtype=np.int64)
    precision = positive / ranks
    # Restrict to boundaries between score values; otherwise a threshold would
    # include a tied subset that cannot be obtained by score thresholding.
    boundaries = np.r_[sorted_scores[1:] < sorted_scores[:-1], True]
    candidates = np.flatnonzero(boundaries)
    best = candidates[np.argmax(precision[candidates])]
    return {
        "best_candidate_precision": float(precision[best]),
        "best_candidate_precision_threshold": float(sorted_scores[best]),
        "best_candidate_precision_rows": int(ranks[best]),
        "best_candidate_precision_true_rows": int(positive[best]),
    }


def _parse_candidate_gate(value: str) -> float | None:
    if value == "ungated":
        return None
    gate = float(value)
    if not np.isfinite(gate) or gate <= 0.0:
        raise ValueError("--candidate-gate must be 'ungated' or a positive chi2 value")
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--calibration",
        default=None,
        help="validation calibration JSON; defaults to checkpoint sibling calibration.json",
    )
    parser.add_argument(
        "--calibration-key",
        default=None,
        help="key in a multi-gate calibration JSON, for example 'ungated'",
    )
    parser.add_argument(
        "--candidate-gate",
        default="ungated",
        help="physical candidate graph to diagnose: 'ungated' or a positive chi2 gate",
    )
    parser.add_argument(
        "--feature-set",
        default="residual_v1",
        choices=PAIR_FEATURE_SETS,
        help="physical pair feature schema used to train the checkpoint",
    )
    parser.add_argument("--magnitude-mm", type=float, default=0.0)
    args = parser.parse_args()

    _, samples, manifest = load_synthetic_curriculum_manifest(args.synthetic_manifest)
    if int(manifest.get("q_over_p_mode", -1)) != 0:
        raise ValueError("diagnosis supports only mode-0 physical propagation")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    calibration_path = (
        Path(args.calibration).expanduser().resolve()
        if args.calibration is not None
        else checkpoint.parent / "calibration.json"
    )
    with calibration_path.open(encoding="utf-8") as handle:
        calibration = json.load(handle)
    if "temperature" in calibration:
        temperature = float(calibration["temperature"])
        calibration_key = None
    else:
        all_gates = calibration.get("candidate_gate_calibrations")
        if not isinstance(all_gates, Mapping):
            raise ValueError("calibration JSON lacks a scalar temperature or candidate-gate map")
        calibration_key = args.calibration_key
        if calibration_key is None:
            raise ValueError("--calibration-key is required for a multi-gate calibration JSON")
        selected = all_gates.get(calibration_key)
        if not isinstance(selected, Mapping) or "temperature" not in selected:
            raise ValueError(f"calibration key '{calibration_key}' is unavailable")
        temperature = float(selected["temperature"])
    candidate_gate = _parse_candidate_gate(args.candidate_gate)
    model, artifact = load_pair_classifier(checkpoint, device="cuda")
    stations = tuple(sorted({station for pair in artifact.station_pairs for station in pair}))
    station_pairs = tuple((left, right) for left in stations for right in stations if left < right)
    validation_samples = [sample for sample in samples if sample.split == "validation"]
    if not validation_samples:
        raise ValueError("synthetic manifest has no validation samples")
    expected_features = pair_feature_names(station_pairs, feature_set=args.feature_set)
    if tuple(artifact.feature_names) != expected_features:
        raise ValueError(
            "checkpoint feature schema does not match --feature-set; use the schema recorded "
            "in its resolved configuration"
        )
    candidate_sets = build_candidate_sets(
        validation_samples,
        station_pairs,
        chi2_gate=None,
        feature_set=args.feature_set,
    )
    selected_sets = [
        candidate_set
        for candidate_set in candidate_sets
        if np.isclose(float(candidate_set.sample.magnitude_mm), args.magnitude_mm)
    ]
    if not selected_sets:
        raise ValueError(f"no validation samples at magnitude {args.magnitude_mm}")
    raw_scores = score_candidate_sets(model, artifact, selected_sets)
    if candidate_gate is not None:
        raw_scores = filter_candidate_scores(selected_sets, raw_scores, candidate_gate)
        selected_sets = filter_candidate_sets(selected_sets, candidate_gate)
    calibrated_scores = [
        apply_temperature(values, temperature) if values.size else np.empty(0, dtype=np.float64)
        for values in raw_scores
    ]

    pair_labels: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    pair_raw: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    pair_calibrated: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    category_raw: dict[tuple[tuple[int, int], str], list[float]] = defaultdict(list)
    category_calibrated: dict[tuple[tuple[int, int], str], list[float]] = defaultdict(list)
    category_count: dict[tuple[tuple[int, int], str], int] = defaultdict(int)

    for candidate_set, values_raw, values_calibrated in zip(
        selected_sets, raw_scores, calibrated_scores
    ):
        pair = candidate_set.station_pair
        labels = np.asarray(candidate_set.labels, dtype=bool)
        pair_labels[pair].append(labels)
        pair_raw[pair].append(np.asarray(values_raw, dtype=np.float64))
        pair_calibrated[pair].append(np.asarray(values_calibrated, dtype=np.float64))
        for candidate, label, score_raw, score_calibrated in zip(
            candidate_set.candidates,
            labels,
            values_raw,
            values_calibrated,
        ):
            category = _role_category(
                candidate_set.event,
                candidate.source_index,
                candidate.target_index,
                bool(label),
            )
            key = (pair, category)
            category_count[key] += 1
            category_raw[key].append(float(score_raw))
            category_calibrated[key].append(float(score_calibrated))

    pair_rows: list[dict[str, object]] = []
    for pair in sorted(pair_labels):
        labels = np.concatenate(pair_labels[pair])
        raw = np.concatenate(pair_raw[pair])
        calibrated = np.concatenate(pair_calibrated[pair])
        pair_rows.append(
            {
                "magnitude_mm": args.magnitude_mm,
                "station_pair": f"{pair[0]}->{pair[1]}",
                "candidate_rows": int(labels.size),
                "positive_candidate_rows": int(np.count_nonzero(labels)),
                **{
                    f"raw_{key}": value
                    for key, value in binary_calibration(raw, labels, bins=15).items()
                },
                **{
                    f"calibrated_{key}": value
                    for key, value in binary_calibration(calibrated, labels, bins=15).items()
                },
                **_best_candidate_precision(calibrated, labels),
            }
        )
    category_rows = []
    for (pair, category), count in sorted(category_count.items()):
        category_rows.append(
            {
                "magnitude_mm": args.magnitude_mm,
                "station_pair": f"{pair[0]}->{pair[1]}",
                "candidate_category": category,
                "candidate_rows": count,
                **{f"raw_{key}": value for key, value in _quantiles(category_raw[(pair, category)]).items()},
                **{
                    f"calibrated_{key}": value
                    for key, value in _quantiles(category_calibrated[(pair, category)]).items()
                },
            }
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty diagnosis output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "validation_candidate_metrics_by_station_pair.csv", pair_rows)
    _write_csv(output_dir / "validation_score_by_candidate_category.csv", category_rows)
    (output_dir / "diagnosis.json").write_text(
        json.dumps(
            {
                "synthetic_manifest": str(Path(args.synthetic_manifest).expanduser().resolve()),
                "checkpoint": str(checkpoint),
                "calibration": str(calibration_path),
                "split": "validation",
                "magnitude_mm": args.magnitude_mm,
                "temperature": temperature,
                "calibration_key": calibration_key,
                "candidate_chi2_gate": candidate_gate,
                "feature_set": args.feature_set,
                "physical_geometry_repropagation": True,
                "q_over_p_mode": 0,
                "test_samples_opened": False,
                "station_pairs": [list(pair) for pair in station_pairs],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "validation_candidate_sets": len(selected_sets),
                "station_pairs": len(pair_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
