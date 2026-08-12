#!/usr/bin/env python3
"""Evaluate a non-Transformer unknown-association baseline on synthetic MC."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from baselines.field_candidate_dataset import resolved_station_pairs
from baselines.field_chi2_matching import (
    build_field_candidates,
    candidate_feature_matrix,
    candidate_labels,
    greedy_field_chi2_match,
    greedy_score_match,
    pair_feature_names,
)
from baselines.mlp_pair_classifier import load_pair_classifier, predict_pair_classifier
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events
from evaluation.metrics import AssociationMetrics, assess_event_matches


DEFAULT_CONFIG = {
    "q_over_p_mode": 0,
    "field_chi2_gate": 1500.0,
    "mlp_candidate_chi2_gate": None,
    "mlp_score_threshold": None,
    "target_z_tolerance_mm": 1.0e-6,
    "device": "auto",
}


def _load_config(path: str | None) -> dict[str, object]:
    config = dict(DEFAULT_CONFIG)
    if path is None:
        return config
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle) or {}
    config.update(supplied.get("unknown_association_baseline", supplied))
    return config


def _metrics_with_candidate_counts(
    metrics: AssociationMetrics,
    candidate_rows: int,
    positive_candidate_rows: int,
) -> dict[str, int | float | None]:
    return {
        **metrics.as_dict(),
        "candidate_rows": candidate_rows,
        "positive_candidate_rows": positive_candidate_rows,
    }


def _selected_event_ids(path: str | None) -> set[int] | None:
    """Load an explicit held-out event list written by the scan coordinator."""
    if path is None:
        return None
    source = Path(path).expanduser().resolve()
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    values = payload.get("event_ids") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not values:
        raise ValueError("event-id file must be a non-empty JSON list or an object with event_ids")
    try:
        return {int(value) for value in values}
    except (TypeError, ValueError) as error:
        raise ValueError("event-id file contains a non-integer event ID") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", required=True, choices=("field_chi2", "mlp"))
    parser.add_argument("--tracklets", required=True)
    parser.add_argument("--propagations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None, help="Required for --algorithm mlp")
    parser.add_argument("--chi2-gate", type=float, default=None)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--event-id-file",
        default=None,
        help="Optional JSON held-out synthetic event IDs used for every evaluated payload",
    )
    args = parser.parse_args()
    config = _load_config(args.config)
    if args.chi2_gate is not None:
        config["field_chi2_gate"] = args.chi2_gate
    if args.score_threshold is not None:
        config["mlp_score_threshold"] = args.score_threshold
    if args.device is not None:
        config["device"] = args.device
    if int(config["q_over_p_mode"]) != 0:
        raise ValueError("V1 unknown-association baselines must use q_over_p_mode=0")
    if args.algorithm == "field_chi2" and float(config["field_chi2_gate"]) <= 0.0:
        raise ValueError("field_chi2_gate must be positive")
    if args.algorithm == "mlp" and args.checkpoint is None:
        parser.error("--checkpoint is required for --algorithm mlp")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    events = load_events(args.tracklets, require_mc_labels=True)
    selected_event_ids = _selected_event_ids(args.event_id_file)
    if selected_event_ids is not None:
        events = [event for event in events if event.event_id in selected_event_ids]
    if not events:
        raise ValueError("no synthetic events remain after event-ID selection")
    records = load_propagation_records(args.propagations)
    station_pairs = resolved_station_pairs(events)
    model = None
    artifact = None
    candidate_gate: float | None
    score_threshold: float | None = None
    if args.algorithm == "field_chi2":
        candidate_gate = float(config["field_chi2_gate"])
    else:
        candidate_gate = (
            None
            if config["mlp_candidate_chi2_gate"] is None
            else float(config["mlp_candidate_chi2_gate"])
        )
        model, artifact = load_pair_classifier(args.checkpoint, device=str(config["device"]))
        if tuple(artifact.station_pairs) != tuple(station_pairs):
            raise ValueError("MLP checkpoint station-pair contract differs from evaluation data")
        if tuple(artifact.feature_names) != pair_feature_names(station_pairs):
            raise ValueError("MLP checkpoint feature contract differs from evaluation data")
        score_threshold = (
            artifact.threshold
            if config["mlp_score_threshold"] is None
            else float(config["mlp_score_threshold"])
        )

    aggregate = AssociationMetrics()
    pair_metrics: dict[tuple[int, int], AssociationMetrics] = {
        pair: AssociationMetrics() for pair in station_pairs
    }
    candidate_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    positive_candidate_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    matched_rows: list[dict[str, int | float | str]] = []
    for event in events:
        for source_station, target_station in station_pairs:
            candidates = build_field_candidates(
                event,
                records,
                source_station=source_station,
                target_station=target_station,
                chi2_gate=candidate_gate,
                q_over_p_mode=0,
                target_z_tolerance_mm=float(config["target_z_tolerance_mm"]),
            )
            labels = candidate_labels(event, candidates)
            candidate_counts[(source_station, target_station)] += len(candidates)
            positive_candidate_counts[(source_station, target_station)] += int(np.count_nonzero(labels))
            if args.algorithm == "field_chi2":
                matches = greedy_field_chi2_match(candidates)
                scores: np.ndarray | None = None
            else:
                assert model is not None and artifact is not None and score_threshold is not None
                features = candidate_feature_matrix(event, candidates, station_pairs)
                scores = predict_pair_classifier(model, artifact, features) if candidates else np.empty(0)
                matches = greedy_score_match(candidates, scores, score_threshold)
            event_metrics, assessments = assess_event_matches(
                event,
                matches,
                source_station=source_station,
                target_station=target_station,
            )
            aggregate.add(event_metrics)
            pair_metrics[(source_station, target_station)].add(event_metrics)
            candidate_lookup = {
                (candidate.source_index, candidate.target_index): row
                for row, candidate in enumerate(candidates)
            }
            for match, assessment in zip(matches, assessments):
                candidate_row = candidate_lookup[(match.source_index, match.target_index)]
                row: dict[str, int | float | str] = {
                    "run_id": event.run_id,
                    "event_id": event.event_id,
                    "source_station": source_station,
                    "target_station": target_station,
                    "source_tracklet_id": int(event.tracklet_id[match.source_index]),
                    "target_tracklet_id": int(event.tracklet_id[match.target_index]),
                    "chi2": float(match.chi2),
                    "source_truth_particle_id": ""
                    if assessment.source_truth_particle_id is None
                    else assessment.source_truth_particle_id,
                    "target_truth_particle_id": ""
                    if assessment.target_truth_particle_id is None
                    else assessment.target_truth_particle_id,
                    "truth_relation": assessment.relation,
                }
                if scores is not None:
                    row["score"] = float(scores[candidate_row])
                matched_rows.append(row)

    by_pair = {
        f"{source}->{target}": _metrics_with_candidate_counts(
            pair_metrics[(source, target)],
            candidate_counts[(source, target)],
            positive_candidate_counts[(source, target)],
        )
        for source, target in station_pairs
    }
    metrics = {
        "method": f"synthetic_unknown_association_{args.algorithm}",
        "algorithm": args.algorithm,
        "q_over_p_mode": 0,
        "tracklets": str(Path(args.tracklets).expanduser().resolve()),
        "propagations": str(Path(args.propagations).expanduser().resolve()),
        "events_processed": len(events),
        "evaluated_event_ids": [event.event_id for event in events],
        "event_id_file": (
            None if args.event_id_file is None else str(Path(args.event_id_file).expanduser().resolve())
        ),
        "station_pairs": [list(pair) for pair in station_pairs],
        "candidate_chi2_gate": candidate_gate,
        "score_threshold": score_threshold,
        "checkpoint": None if args.checkpoint is None else str(Path(args.checkpoint).expanduser().resolve()),
        "overall": _metrics_with_candidate_counts(
            aggregate,
            sum(candidate_counts.values()),
            sum(positive_candidate_counts.values()),
        ),
        "by_station_pair": by_pair,
        "metric_convention": {
            "association_efficiency": "correct matches / truth pairs with unique known endpoints",
            "inclusive_association_purity": "correct matches / all predicted matches, including fake endpoints",
            "inclusive_fake_rate": "non-correct predictions / all predicted matches, including fake endpoints",
            "association_purity": "scorable-only legacy purity; fake endpoints are excluded",
        },
    }
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    fields = sorted({field for row in matched_rows for field in row})
    with (output_dir / "matches.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(matched_rows)
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
