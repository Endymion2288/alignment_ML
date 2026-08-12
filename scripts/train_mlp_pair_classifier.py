#!/usr/bin/env python3
"""Train the nominal-geometry MLP pair-classifier baseline on synthetic MC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from baselines.field_candidate_dataset import (
    build_candidate_batches,
    concatenate_batches,
    resolved_station_pairs,
)
from baselines.field_chi2_matching import pair_feature_names
from baselines.mlp_pair_classifier import (
    PairClassifierConfig,
    save_pair_classifier,
    train_pair_classifier,
)
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events


DEFAULT_CONFIG = {
    "seed": 20260810,
    "validation_fraction": 0.30,
    "hidden_dim": 64,
    "epochs": 150,
    "batch_size": 256,
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-4,
    "device": "auto",
    "q_over_p_mode": 0,
    "candidate_chi2_gate": None,
}


def _load_config(path: str | None) -> dict[str, object]:
    config = dict(DEFAULT_CONFIG)
    if path is None:
        return config
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle) or {}
    config.update(supplied.get("mlp_pair_classifier", supplied))
    return config


def _split_events(event_ids: list[int], validation_fraction: float, seed: int) -> tuple[set[int], set[int]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between zero and one")
    unique = sorted(set(event_ids))
    if len(unique) < 2:
        raise ValueError("at least two synthetic events are required for a train/validation split")
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique)
    validation_count = min(max(1, int(round(len(unique) * validation_fraction))), len(unique) - 1)
    validation = {int(value) for value in order[:validation_count]}
    training = set(unique) - validation
    return training, validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracklets", required=True, help="Nominal synthetic canonical ROOT")
    parser.add_argument("--propagations", required=True, help="Nominal synthetic field candidates ROOT")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = _load_config(args.config)
    for name in ("seed", "validation_fraction", "epochs", "device"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    if int(config["q_over_p_mode"]) != 0:
        raise ValueError("V1 MLP pair classifier is restricted to q_over_p_mode=0")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    events = load_events(args.tracklets, require_mc_labels=True)
    records = load_propagation_records(args.propagations)
    station_pairs = resolved_station_pairs(events)
    training_ids, validation_ids = _split_events(
        [event.event_id for event in events],
        validation_fraction=float(config["validation_fraction"]),
        seed=int(config["seed"]),
    )
    training_events = [event for event in events if event.event_id in training_ids]
    validation_events = [event for event in events if event.event_id in validation_ids]
    gate = config["candidate_chi2_gate"]
    candidate_gate = None if gate is None else float(gate)
    training_batches = build_candidate_batches(
        training_events, records, station_pairs=station_pairs, chi2_gate=candidate_gate
    )
    validation_batches = build_candidate_batches(
        validation_events, records, station_pairs=station_pairs, chi2_gate=candidate_gate
    )
    training_features, training_labels = concatenate_batches(training_batches)
    validation_features, validation_labels = concatenate_batches(validation_batches)
    model_config = PairClassifierConfig(
        hidden_dim=int(config["hidden_dim"]),
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        seed=int(config["seed"]),
        device=str(config["device"]),
    )
    model, artifact = train_pair_classifier(
        training_features,
        training_labels,
        validation_features,
        validation_labels,
        feature_names=pair_feature_names(station_pairs),
        station_pairs=station_pairs,
        config=model_config,
    )
    checkpoint = output_dir / "mlp_pair_classifier.pt"
    save_pair_classifier(checkpoint, model, artifact)
    metrics = {
        "method": "nominal_geometry_mlp_pair_classifier",
        "q_over_p_mode": 0,
        "tracklets": str(Path(args.tracklets).expanduser().resolve()),
        "propagations": str(Path(args.propagations).expanduser().resolve()),
        "station_pairs": [list(pair) for pair in station_pairs],
        "training_event_ids": sorted(training_ids),
        "validation_event_ids": sorted(validation_ids),
        "candidate_chi2_gate": candidate_gate,
        "checkpoint": str(checkpoint),
        "threshold": artifact.threshold,
        "feature_names": list(artifact.feature_names),
        **artifact.training_summary,
    }
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
