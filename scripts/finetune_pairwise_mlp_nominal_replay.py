#!/usr/bin/env python3
"""Fine-tune a curriculum pairwise MLP on a small-offset replay subset.

This is a validation-only curriculum-forgetting check.  It opens train and
validation sources only, keeps the physical refit/Acts candidates unchanged,
and writes a checkpoint that can be inspected with
``diagnose_pairwise_mlp_validation.py`` before any test evaluation is allowed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Mapping

import numpy as np

from baselines.mlp_pair_classifier import (
    PairClassifierConfig,
    load_pair_classifier,
    save_pair_classifier,
    train_pair_classifier,
)
from datasets.physical_curriculum import load_synthetic_curriculum_manifest
from evaluation.pairwise_metrics import calibration_report
from training.curriculum_mlp import (
    build_candidate_sets,
    concatenate_candidate_sets,
    payload_balanced_sampling_weights,
    score_candidate_sets,
)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _source_split_audit(samples: list[object]) -> dict[str, object]:
    sources_by_split: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    uids_by_split: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    for sample in samples:
        split = str(sample.split)
        sources_by_split[split].update(str(source_id) for source_id in sample.source_ids)
        uids_by_split[split].update(str(value) for value in sample.source_event_uids)
    overlaps = {
        f"{left}_{right}": sorted(uids_by_split[left].intersection(uids_by_split[right]))
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    if any(overlaps.values()):
        raise ValueError("source event leakage across corpus splits")
    return {
        "sources_by_split": {key: sorted(value) for key, value in sources_by_split.items()},
        "source_event_counts_by_split": {key: len(value) for key, value in uids_by_split.items()},
        "source_event_uid_overlap": overlaps,
        "strictly_disjoint": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replay-maximum-magnitude-mm", type=float, default=0.1)
    parser.add_argument("--validation-magnitude-mm", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    if args.replay_maximum_magnitude_mm < 0.0 or args.validation_magnitude_mm < 0.0:
        raise ValueError("replay and validation magnitudes must be non-negative")
    if args.epochs < 1 or args.learning_rate <= 0.0:
        raise ValueError("epochs and learning rate must be positive")

    _, samples, manifest = load_synthetic_curriculum_manifest(args.synthetic_manifest)
    if int(manifest.get("q_over_p_mode", -1)) != 0:
        raise ValueError("nominal replay supports only mode-0 physical propagation")
    split_audit = _source_split_audit(samples)
    checkpoint = Path(args.initial_checkpoint).expanduser().resolve()
    model, initial_artifact = load_pair_classifier(checkpoint, device="cuda")
    station_pairs = tuple((int(left), int(right)) for left, right in initial_artifact.station_pairs)
    train_samples = [
        sample
        for sample in samples
        if sample.split == "train"
        and float(sample.magnitude_mm) <= args.replay_maximum_magnitude_mm
    ]
    validation_samples = [
        sample
        for sample in samples
        if sample.split == "validation"
        and np.isclose(float(sample.magnitude_mm), args.validation_magnitude_mm)
    ]
    if not train_samples or not validation_samples:
        raise ValueError("requested replay or validation physical samples are unavailable")
    train_sets = build_candidate_sets(train_samples, station_pairs, chi2_gate=None)
    validation_sets = build_candidate_sets(validation_samples, station_pairs, chi2_gate=None)
    train_features, train_labels = concatenate_candidate_sets(train_sets)
    validation_features, validation_labels = concatenate_candidate_sets(validation_sets)
    model_config = replace(
        initial_artifact.config,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device="cuda",
    )
    state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model, artifact = train_pair_classifier(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        feature_names=initial_artifact.feature_names,
        station_pairs=station_pairs,
        config=model_config,
        standardizer=initial_artifact.standardizer,
        initial_state_dict=state_dict,
        sampling_weights=payload_balanced_sampling_weights(train_sets),
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty replay output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_checkpoint = output_dir / "mlp_pair_classifier_nominal_replay.pt"
    save_pair_classifier(output_checkpoint, model, artifact)
    raw_scores = score_candidate_sets(model, artifact, validation_sets)
    merged_scores = np.concatenate([values for values in raw_scores if values.size])
    merged_labels = np.concatenate(
        [candidate_set.labels for candidate_set, values in zip(validation_sets, raw_scores) if values.size]
    )
    _, calibration = calibration_report(merged_scores, merged_labels, bins=15)
    _write_json(
        output_dir / "calibration.json",
        {
            **calibration,
            "fit_split": "validation",
            "fit_magnitude_mm": args.validation_magnitude_mm,
            "validation_candidate_rows": int(merged_labels.size),
            "validation_positive_candidate_rows": int(np.count_nonzero(merged_labels)),
        },
    )
    _write_json(
        output_dir / "training.json",
        {
            "initial_checkpoint": str(checkpoint),
            "checkpoint": str(output_checkpoint),
            "device": artifact.training_summary["device"],
            "replay_maximum_magnitude_mm": args.replay_maximum_magnitude_mm,
            "validation_magnitude_mm": args.validation_magnitude_mm,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "training_candidate_rows": int(train_labels.size),
            "training_positive_candidate_rows": int(np.count_nonzero(train_labels)),
            "validation_candidate_rows": int(validation_labels.size),
            "validation_positive_candidate_rows": int(np.count_nonzero(validation_labels)),
            "payload_balanced_sampling": True,
            "source_split_audit": split_audit,
            "test_samples_opened": False,
            "physical_geometry_repropagation": True,
            "q_over_p_mode": 0,
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "checkpoint": str(output_checkpoint),
                "device": artifact.training_summary["device"],
                "test_samples_opened": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
