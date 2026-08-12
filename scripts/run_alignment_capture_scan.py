#!/usr/bin/env python3
"""Scan truth-fixed residual-level station alignment capture range."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from alignment.closure import (
    measurements_from_field_evaluation,
    run_capture_range_scan,
)
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events
from evaluation.field_propagation import evaluate_field_propagation


DEFAULT_CONFIG = {
    "reference_station": 0,
    "magnitudes_mm": [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0],
    "trials": 100,
    "seed": 12345,
    "chi2_gate": 25.0,
    "refinement_iterations": 3,
    "prior_sigma_mm": None,
    "measurement_noise_scale": 0.0,
    "capture_tolerance_mm": 0.01,
}


def _load_config(path: str | None) -> dict[str, object]:
    config = dict(DEFAULT_CONFIG)
    if path is None:
        return config
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle) or {}
    config.update(supplied.get("alignment_capture_scan", supplied))
    return config


def _parse_magnitudes(value: str) -> list[float]:
    try:
        magnitudes = [float(item) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("magnitudes must be comma-separated floats") from error
    if not magnitudes:
        raise argparse.ArgumentTypeError("at least one scan magnitude is required")
    return magnitudes


def _write_csv(output: Path, points: list[dict[str, object]]) -> None:
    fields = (
        "magnitude_mm",
        "trials",
        "capture_fraction",
        "mean_max_error_mm",
        "p95_max_error_mm",
        "mean_rms_error_mm",
        "mean_active_fraction",
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in points:
            writer.writerow({field: point[field] for field in fields})


def _write_plot(output: Path, points: list[dict[str, object]]) -> None:
    magnitude = np.asarray([point["magnitude_mm"] for point in points], dtype=np.float64)
    capture = np.asarray([point["capture_fraction"] for point in points], dtype=np.float64)
    mean_error = np.asarray([point["mean_max_error_mm"] for point in points], dtype=np.float64)
    p95_error = np.asarray([point["p95_max_error_mm"] for point in points], dtype=np.float64)
    active = np.asarray([point["mean_active_fraction"] for point in points], dtype=np.float64)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(magnitude, capture, marker="o", label="closure capture")
    axes[0].plot(magnitude, active, marker="s", label="active pair fraction")
    axes[0].set_xlabel("injected offset magnitude per movable station [mm]")
    axes[0].set_ylabel("fraction")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].legend()
    axes[1].plot(magnitude, mean_error, marker="o", label="mean max error")
    axes[1].plot(magnitude, p95_error, marker="s", label="p95 max error")
    axes[1].set_xlabel("injected offset magnitude per movable station [mm]")
    axes[1].set_ylabel("recovered alignment error [mm]")
    axes[1].legend()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracklets", required=True, help="Canonical MC tracklet ROOT file")
    parser.add_argument("--propagations", required=True, help="Canonical propagation ROOT file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None, help="YAML capture-scan configuration")
    parser.add_argument("--reference-station", type=int, default=None)
    parser.add_argument(
        "--magnitudes-mm",
        type=_parse_magnitudes,
        default=None,
    )
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--chi2-gate",
        type=float,
        default=None,
        help="Gate on baseline-subtracted two-dimensional residual increments",
    )
    parser.add_argument("--refinement-iterations", type=int, default=None)
    parser.add_argument("--prior-sigma-mm", type=float, default=None)
    parser.add_argument("--measurement-noise-scale", type=float, default=None)
    parser.add_argument("--capture-tolerance-mm", type=float, default=None)
    args = parser.parse_args()
    config = _load_config(args.config)
    for name in (
        "reference_station",
        "magnitudes_mm",
        "trials",
        "seed",
        "chi2_gate",
        "refinement_iterations",
        "prior_sigma_mm",
        "measurement_noise_scale",
        "capture_tolerance_mm",
    ):
        value = getattr(args, name)
        if value is not None:
            config[name] = value

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    events = load_events(args.tracklets, require_mc_labels=True)
    evaluation = evaluate_field_propagation(
        events, load_propagation_records(args.propagations), require_truth_match=True
    )
    measurements = measurements_from_field_evaluation(evaluation)
    summary = {
        "tracklets": str(Path(args.tracklets).expanduser().resolve()),
        "propagations": str(Path(args.propagations).expanduser().resolve()),
        "accepted_truth_matched_pairs": evaluation.size,
        "coordinate_reporting_convention": "r_injected = r_nominal + delta_target - delta_source",
        "deformed_geometry_repropagation": False,
        **run_capture_range_scan(
            measurements,
            reference_station=int(config["reference_station"]),
            magnitudes_mm=[float(value) for value in config["magnitudes_mm"]],
            trials_per_magnitude=int(config["trials"]),
            seed=int(config["seed"]),
            chi2_gate=(None if config["chi2_gate"] is None else float(config["chi2_gate"])),
            refinement_iterations=int(config["refinement_iterations"]),
            capture_tolerance_mm=float(config["capture_tolerance_mm"]),
            measurement_noise_scale=float(config["measurement_noise_scale"]),
            prior_sigma_mm=(
                None if config["prior_sigma_mm"] is None else float(config["prior_sigma_mm"])
            ),
        ),
    }
    resolved_config = {
        **config,
        "tracklets": str(Path(args.tracklets).expanduser().resolve()),
        "propagations": str(Path(args.propagations).expanduser().resolve()),
    }
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_config, handle, sort_keys=True)
    (output_dir / "capture_scan.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "capture_scan.csv", summary["points"])
    _write_plot(output_dir / "capture_scan.png", summary["points"])
    console_summary = dict(summary)
    console_summary["points"] = [
        {key: value for key, value in point.items() if key != "trial_results"}
        for point in summary["points"]
    ]
    print(json.dumps(console_summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
