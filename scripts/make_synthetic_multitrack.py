#!/usr/bin/env python3
"""Overlay single-muon canonical events into controlled multi-track samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from datasets.root_loader import load_events
from datasets.synthetic_overlay import write_synthetic_multitrack_root


DEFAULT_CONFIG = {
    "events": 100,
    "tracks_per_event": 4,
    "stations": [0, 1, 2, 3],
    "missing_tracklet_probability": 0.10,
    "fake_mean_per_station": 0.50,
    "minimum_truth_match_fraction": 0.99,
    "synthetic_run_id": 990000,
    "seed": 12345,
}


def _load_config(path: str | None) -> dict[str, object]:
    config = dict(DEFAULT_CONFIG)
    if path is None:
        return config
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle) or {}
    config.update(supplied.get("synthetic_multitrack", supplied))
    return config


def _station_ids(value: str) -> tuple[int, ...]:
    try:
        stations = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("station IDs must be comma-separated integers") from error
    if not stations:
        raise argparse.ArgumentTypeError("at least one station ID is required")
    return stations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Canonical truth-labelled single-particle ROOT")
    parser.add_argument("--output", required=True, help="Synthetic canonical ROOT output")
    parser.add_argument("--config", default=None, help="YAML synthetic-overlay configuration")
    parser.add_argument("--events", type=int, default=None)
    parser.add_argument("--tracks-per-event", type=int, default=None)
    parser.add_argument("--stations", type=_station_ids, default=None)
    parser.add_argument("--missing-tracklet-probability", type=float, default=None)
    parser.add_argument("--fake-mean-per-station", type=float, default=None)
    parser.add_argument("--minimum-truth-match-fraction", type=float, default=None)
    parser.add_argument("--synthetic-run-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    config = _load_config(args.config)
    for name in (
        "events",
        "tracks_per_event",
        "stations",
        "missing_tracklet_probability",
        "fake_mean_per_station",
        "minimum_truth_match_fraction",
        "synthetic_run_id",
        "seed",
    ):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    summary = write_synthetic_multitrack_root(
        load_events(args.input, require_mc_labels=True),
        args.output,
        output_events=int(config["events"]),
        tracks_per_event=int(config["tracks_per_event"]),
        station_ids=tuple(int(station) for station in config["stations"]),
        missing_tracklet_probability=float(config["missing_tracklet_probability"]),
        fake_mean_per_station=float(config["fake_mean_per_station"]),
        minimum_truth_match_fraction=float(config["minimum_truth_match_fraction"]),
        seed=int(config["seed"]),
        synthetic_run_id=int(config["synthetic_run_id"]),
    )
    output = Path(args.output).expanduser().resolve()
    with output.with_suffix(".resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {**config, "input": str(Path(args.input).expanduser().resolve()), "output": str(output)},
            handle,
            sort_keys=True,
        )
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
