#!/usr/bin/env python3
"""Model-free truth/fake chi2 separation of mode-0 vs mode-3 candidate graphs.

The frozen-backbone AUC convolves the candidate graph's intrinsic information
with the frozen V2 feature calibration.  This audit isolates the intrinsic
part: for the identical overlay events, label every candidate edge by MC truth
and compute the ROC of the raw Mahalanobis chi2 (and of the frozen V2 score
for reference) under each covariance variant.  Train-only, read-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from datasets.physical_curriculum import load_synthetic_curriculum_manifest
from training.curriculum_mlp import build_candidate_sets

ADJACENT_PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (2, 3))


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Rank-based AUC of a score where higher means more truth-like."""
    labels = labels.astype(bool)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    # Average ranks for ties.
    unique_scores, inverse, counts = np.unique(
        scores, return_inverse=True, return_counts=True
    )
    if np.any(counts > 1):
        sums = np.zeros(unique_scores.size, dtype=np.float64)
        np.add.at(sums, inverse, ranks)
        ranks = sums[inverse] / counts[inverse]
    return float((ranks[labels].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def _manifest_metrics(manifest_path: Path, split: str, mode: int) -> dict[str, Any]:
    _, samples, manifest = load_synthetic_curriculum_manifest(
        manifest_path,
        require_all_splits=False,
        allowed_splits=(split,),
    )
    if int(manifest.get("q_over_p_mode", -1)) != int(mode):
        raise ValueError(f"manifest is not a mode-{mode} candidate contract")
    candidate_sets = build_candidate_sets(
        samples,
        ADJACENT_PAIRS,
        chi2_gate=None,
        feature_set="residual_v1",
        q_over_p_mode=int(mode),
    )
    per_pair: dict[str, dict[str, Any]] = {}
    for pair in ADJACENT_PAIRS:
        chi2_values: list[float] = []
        labels: list[int] = []
        for candidate_set in candidate_sets:
            if tuple(candidate_set.station_pair) != tuple(pair):
                continue
            for candidate, label in zip(candidate_set.candidates, candidate_set.labels):
                chi2 = float(candidate.chi2)
                if not np.isfinite(chi2):
                    continue
                chi2_values.append(chi2)
                labels.append(int(label))
        chi2_array = np.asarray(chi2_values, dtype=np.float64)
        label_array = np.asarray(labels, dtype=np.int64)
        truth = label_array == 1
        fake = label_array == 0
        per_pair[f"{pair[0]}->{pair[1]}"] = {
            "candidates": int(label_array.size),
            "truth": int(truth.sum()),
            "fake": int(fake.sum()),
            "chi2_auc_truth_below_fake": _roc_auc(-chi2_array, label_array),
            "truth_chi2_median": float(np.median(chi2_array[truth])) if truth.any() else None,
            "fake_chi2_median": float(np.median(chi2_array[fake])) if fake.any() else None,
            "truth_chi2_q99": float(np.quantile(chi2_array[truth], 0.99)) if truth.any() else None,
            "fake_chi2_q01": float(np.quantile(chi2_array[fake], 0.01)) if fake.any() else None,
        }
    all_chi2 = []
    all_labels = []
    for pair in ADJACENT_PAIRS:
        for candidate_set in candidate_sets:
            if tuple(candidate_set.station_pair) != tuple(pair):
                continue
            for candidate, label in zip(candidate_set.candidates, candidate_set.labels):
                if np.isfinite(float(candidate.chi2)):
                    all_chi2.append(float(candidate.chi2))
                    all_labels.append(int(label))
    return {
        "q_over_p_mode": int(mode),
        "per_pair": per_pair,
        "overall_chi2_auc_truth_below_fake": _roc_auc(
            -np.asarray(all_chi2, dtype=np.float64), np.asarray(all_labels, dtype=np.int64)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-mode0", required=True)
    parser.add_argument("--manifest-mode3", required=True)
    parser.add_argument("--split", default="train", choices=("train", "validation"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    report = {
        "schema_version": "faser-mode3-intrinsic-separation-audit-v1",
        "split": args.split,
        "mode0": _manifest_metrics(Path(args.manifest_mode0).expanduser().resolve(), args.split, 0),
        "mode3": _manifest_metrics(Path(args.manifest_mode3).expanduser().resolve(), args.split, 3),
        "test_data_accessed": False,
    }
    (output / "intrinsic_separation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "mode0_auc": report["mode0"]["overall_chi2_auc_truth_below_fake"],
        "mode3_auc": report["mode3"]["overall_chi2_auc_truth_below_fake"],
    }, indent=2))


if __name__ == "__main__":
    main()
