#!/usr/bin/env python3
"""Run the reproducible straight-line chi-square matching baseline."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baselines.chi2_matching import build_candidates, greedy_one_to_one_match
from datasets.root_loader import load_events
from evaluation.metrics import AssociationMetrics, assess_event_matches


DEFAULT_CONFIG = {
    "seed": 7,
    "source_station": 0,
    "target_station": 1,
    "chi2_gate": 50.0,
    "max_events": None,
}


def _load_config(path: str | None) -> dict:
    config = dict(DEFAULT_CONFIG)
    if path is None:
        return config
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        supplied = yaml.safe_load(stream) or {}
    config.update(supplied.get("chi2_baseline", supplied))
    return config


def _write_plot(chi2_values: list[float], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 4))
    if chi2_values:
        upper = max(1.0, float(np.percentile(chi2_values, 99.0)))
        axis.hist(chi2_values, bins=50, range=(0.0, upper), histtype="stepfilled")
    axis.set_xlabel("candidate chi2")
    axis.set_ylabel("count")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Canonical flat ROOT tracklet file")
    parser.add_argument("--config", default=None, help="YAML baseline configuration")
    parser.add_argument("--output-dir", required=True, help="Directory for metrics and plots")
    parser.add_argument("--source-station", type=int, default=None)
    parser.add_argument("--target-station", type=int, default=None)
    parser.add_argument("--chi2-gate", type=float, default=None)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = _load_config(args.config)
    for name in ("source_station", "target_station", "chi2_gate", "max_events", "seed"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(
        args.input,
        max_events=config["max_events"],
        require_mc_labels=False,
    )
    aggregate = AssociationMetrics()
    matched_rows: list[dict[str, int | float | str]] = []
    candidate_chi2: list[float] = []
    candidate_count = 0

    for event in events:
        candidates = build_candidates(
            event,
            source_station=config["source_station"],
            target_station=config["target_station"],
            chi2_gate=config["chi2_gate"],
        )
        matches = greedy_one_to_one_match(candidates)
        event_metrics, truth_assessments = assess_event_matches(
            event,
            matches,
            source_station=config["source_station"],
            target_station=config["target_station"],
        )
        aggregate.add(event_metrics)
        candidate_count += len(candidates)
        candidate_chi2.extend(candidate.chi2 for candidate in candidates)
        for match, assessment in zip(matches, truth_assessments):
            matched_rows.append(
                {
                    "run_id": event.run_id,
                    "event_id": event.event_id,
                    "source_station": config["source_station"],
                    "target_station": config["target_station"],
                    "source_tracklet_id": int(event.tracklet_id[match.source_index]),
                    "target_tracklet_id": int(event.tracklet_id[match.target_index]),
                    "chi2": match.chi2,
                    "source_truth_particle_id": (
                        ""
                        if assessment.source_truth_particle_id is None
                        else assessment.source_truth_particle_id
                    ),
                    "target_truth_particle_id": (
                        ""
                        if assessment.target_truth_particle_id is None
                        else assessment.target_truth_particle_id
                    ),
                    "truth_relation": assessment.relation,
                }
            )

    metrics = {
        **aggregate.as_dict(),
        "events_processed": len(events),
        "candidate_pairs": candidate_count,
        "chi2_gate": config["chi2_gate"],
        "source_station": config["source_station"],
        "target_station": config["target_station"],
        "input": str(Path(args.input).expanduser().resolve()),
    }
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with (output_dir / "matches.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "run_id",
            "event_id",
            "source_station",
            "target_station",
            "source_tracklet_id",
            "target_tracklet_id",
            "chi2",
            "source_truth_particle_id",
            "target_truth_particle_id",
            "truth_relation",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(matched_rows)
    _write_plot(candidate_chi2, output_dir / "candidate_chi2.png")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
