#!/usr/bin/env python3
"""Train one small curriculum MLP per station pair on source-disjoint MC.

This is a controlled validation ablation of the shared pairwise MLP.  The
models retain exactly the existing physical feature representation and
curriculum; only parameter sharing between heterogeneous station pairs is
removed.  Test ROOT files are never opened by this command.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from baselines.mlp_pair_classifier import PairClassifierConfig, save_pair_classifier
from datasets.physical_curriculum import load_synthetic_curriculum_manifest
from evaluation.pairwise_metrics import calibration_report
from training.curriculum_mlp import (
    build_candidate_sets,
    concatenate_candidate_sets,
    score_candidate_sets,
    train_curriculum_pair_classifier,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "physical_global_assignment_muon.yaml"


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle)
    if not isinstance(supplied, Mapping):
        raise ValueError("configuration must be a YAML mapping")
    root = supplied.get("physical_curriculum_mlp", supplied)
    if not isinstance(root, Mapping) or not isinstance(root.get("curriculum_mlp"), Mapping):
        raise ValueError("configuration requires physical_curriculum_mlp.curriculum_mlp")
    return dict(root), dict(root["curriculum_mlp"])


def _source_split_audit(samples: list[object]) -> dict[str, object]:
    source_ids = {"train": set(), "validation": set(), "test": set()}
    source_uids = {"train": set(), "validation": set(), "test": set()}
    for sample in samples:
        split = str(sample.split)
        source_ids[split].update(str(source_id) for source_id in sample.source_ids)
        source_uids[split].update(str(value) for value in sample.source_event_uids)
    overlap = {
        f"{left}_{right}": sorted(source_uids[left].intersection(source_uids[right]))
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    if any(overlap.values()):
        raise ValueError("source event leakage across train/validation/test")
    return {
        "sources_by_split": {key: sorted(value) for key, value in source_ids.items()},
        "source_event_counts_by_split": {key: len(value) for key, value in source_uids.items()},
        "source_event_uid_overlap": overlap,
        "strictly_disjoint": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--validation-magnitude-mm", type=float, default=0.0)
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    root_config, mlp = _load_config(config_path)
    _, samples, manifest = load_synthetic_curriculum_manifest(args.synthetic_manifest)
    if int(manifest.get("q_over_p_mode", -1)) != 0:
        raise ValueError("station-pair MLP study supports only mode-0 physical propagation")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    split_audit = _source_split_audit(samples)
    train_samples = [sample for sample in samples if sample.split == "train"]
    validation_samples = [sample for sample in samples if sample.split == "validation"]
    if not train_samples or not validation_samples:
        raise ValueError("training and validation samples are required")
    stations = tuple(sorted(int(value) for value in root_config["refit"]["station_ids"]))
    station_pairs = tuple((left, right) for left in stations for right in stations if left < right)
    config = PairClassifierConfig(
        hidden_dim=int(mlp["hidden_dim"]),
        epochs=1,
        batch_size=int(mlp["batch_size"]),
        learning_rate=float(mlp["learning_rate"]),
        weight_decay=float(mlp["weight_decay"]),
        seed=int(mlp["seed"]),
        device=str(mlp["device"]),
    )
    pair_rows: list[dict[str, object]] = []
    pair_training: dict[str, object] = {}
    for pair_index, pair in enumerate(station_pairs):
        train_sets = build_candidate_sets(train_samples, (pair,), chi2_gate=None)
        validation_sets = build_candidate_sets(validation_samples, (pair,), chi2_gate=None)
        model, artifact, curriculum = train_curriculum_pair_classifier(
            train_sets,
            validation_sets,
            (pair,),
            mlp["stages"],
            PairClassifierConfig(
                **{**config.__dict__, "seed": config.seed + pair_index}
            ),
        )
        checkpoint = output_dir / f"mlp_pair_{pair[0]}_{pair[1]}.pt"
        save_pair_classifier(checkpoint, model, artifact)
        nominal_sets = [
            candidate_set
            for candidate_set in validation_sets
            if np.isclose(float(candidate_set.sample.magnitude_mm), args.validation_magnitude_mm)
        ]
        if not nominal_sets:
            raise ValueError(f"no validation samples at magnitude {args.validation_magnitude_mm}")
        scores = score_candidate_sets(model, artifact, nominal_sets)
        labels = np.concatenate(
            [candidate_set.labels for candidate_set, values in zip(nominal_sets, scores) if values.size]
        )
        values = np.concatenate([score for score in scores if score.size])
        _, calibration = calibration_report(values, labels, bins=int(mlp["calibration_bins"]))
        pair_name = f"{pair[0]}->{pair[1]}"
        pair_training[pair_name] = {
            "checkpoint": str(checkpoint),
            "curriculum": curriculum,
            "nominal_calibration": calibration,
        }
        pair_rows.append(
            {
                "station_pair": pair_name,
                "checkpoint": str(checkpoint),
                "validation_magnitude_mm": args.validation_magnitude_mm,
                "candidate_rows": int(labels.size),
                "positive_candidate_rows": int(np.count_nonzero(labels)),
                **calibration["before"],
                "temperature": float(calibration["temperature"]),
                "calibrated_roc_auc": calibration["after"]["roc_auc"],
                "calibrated_average_precision": calibration["after"]["average_precision"],
                "calibrated_expected_calibration_error": calibration["after"]["expected_calibration_error"],
            }
        )
    _write_csv(output_dir / "validation_nominal_candidate_metrics_by_station_pair.csv", pair_rows)
    _write_json(
        output_dir / "training.json",
        {
            "config_source": str(config_path),
            "synthetic_manifest": str(Path(args.synthetic_manifest).expanduser().resolve()),
            "device": config.device,
            "station_pairs": [list(pair) for pair in station_pairs],
            "validation_magnitude_mm": args.validation_magnitude_mm,
            "source_split_audit": split_audit,
            "pair_training": pair_training,
            "physical_geometry_repropagation": True,
            "q_over_p_mode": 0,
            "test_samples_opened": False,
            "synthetic_role_used_as_feature": False,
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "models": len(station_pairs),
                "device": config.device,
                "test_samples_opened": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
