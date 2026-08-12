#!/usr/bin/env python3
"""Run a reference-source capture scan for condition-payload x/y transforms.

Each trial samples global station translations and applies their analytically
equivalent coordinate increment to truth-fixed residuals.  The transform sign
and units are anchored by a Calypso-written ``/Tracker/Align`` payload, while
the scan deliberately avoids source shifts because persisted local segments
have not been refitted and repropagated from a displaced source surface.
"""

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
    select_measurements,
)
from alignment.payload import load_station_alignment_payload
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events
from evaluation.field_propagation import evaluate_field_propagation


def _magnitudes(value: str) -> list[float]:
    try:
        result = [float(item) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("--magnitudes-mm must be comma-separated floats") from error
    if not result or any(not np.isfinite(item) or item < 0.0 for item in result):
        raise argparse.ArgumentTypeError("magnitudes must be finite and non-negative")
    return result


def _write_csv(path: Path, points: list[dict[str, object]]) -> None:
    fields = (
        "magnitude_mm",
        "trials",
        "capture_fraction",
        "mean_max_error_mm",
        "p95_max_error_mm",
        "mean_rms_error_mm",
        "mean_active_fraction",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in points:
            writer.writerow({field: point[field] for field in fields})


def _write_plot(path: Path, points: list[dict[str, object]]) -> None:
    magnitude = np.asarray([point["magnitude_mm"] for point in points], dtype=np.float64)
    capture = np.asarray([point["capture_fraction"] for point in points], dtype=np.float64)
    active = np.asarray([point["mean_active_fraction"] for point in points], dtype=np.float64)
    mean_error = np.asarray([point["mean_max_error_mm"] for point in points], dtype=np.float64)
    p95_error = np.asarray([point["p95_max_error_mm"] for point in points], dtype=np.float64)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(magnitude, capture, marker="o", label="capture fraction")
    axes[0].plot(magnitude, active, marker="s", label="active pair fraction")
    axes[0].set_xlabel("translation magnitude per movable station [mm]")
    axes[0].set_ylabel("fraction")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].legend()
    axes[1].plot(magnitude, mean_error, marker="o", label="mean max error")
    axes[1].plot(magnitude, p95_error, marker="s", label="p95 max error")
    axes[1].set_xlabel("translation magnitude per movable station [mm]")
    axes[1].set_ylabel("recovered alignment error [mm]")
    axes[1].legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracklets", required=True, help="Nominal canonical MC tracklets")
    parser.add_argument("--propagations", required=True)
    parser.add_argument("--payload-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-station", type=int, default=0)
    parser.add_argument("--q-over-p-mode", type=int, default=1)
    parser.add_argument("--min-truth-match-fraction", type=float, default=0.99)
    parser.add_argument(
        "--magnitudes-mm",
        type=_magnitudes,
        default=[0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--chi2-gate", type=float, default=25.0)
    parser.add_argument("--refinement-iterations", type=int, default=3)
    parser.add_argument("--capture-tolerance-mm", type=float, default=0.01)
    parser.add_argument("--prior-sigma-mm", type=float, default=None)
    args = parser.parse_args()

    payload = load_station_alignment_payload(args.payload_manifest)
    if not np.allclose(payload.offset_for_station(args.reference_station), 0.0, rtol=0.0, atol=1.0e-15):
        raise ValueError("the reference station must have zero payload offset")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = (
        output_dir / "capture_scan.json",
        output_dir / "capture_scan.csv",
        output_dir / "capture_scan.png",
        output_dir / "resolved_config.yaml",
    )
    if any(path.exists() for path in expected_outputs):
        raise FileExistsError(f"refusing to overwrite an existing scan artifact in {output_dir}")

    evaluation = evaluate_field_propagation(
        load_events(args.tracklets, require_mc_labels=True),
        load_propagation_records(args.propagations),
        require_truth_match=True,
        q_over_p_mode=args.q_over_p_mode,
        min_truth_match_fraction=args.min_truth_match_fraction,
    )
    measurements_all = measurements_from_field_evaluation(evaluation)
    measurements = select_measurements(
        measurements_all, measurements_all.source_station_id == args.reference_station
    )
    scan = run_capture_range_scan(
        measurements,
        reference_station=args.reference_station,
        magnitudes_mm=args.magnitudes_mm,
        trials_per_magnitude=args.trials,
        seed=args.seed,
        chi2_gate=args.chi2_gate,
        refinement_iterations=args.refinement_iterations,
        capture_tolerance_mm=args.capture_tolerance_mm,
        prior_sigma_mm=args.prior_sigma_mm,
    )
    summary = {
        **scan,
        "method": "condition-payload-calibrated_coordinate-level_capture_scan",
        "coordinate_convention": "x' = x + dx_station; y' = y + dy_station",
        "reference_source_only": True,
        "deformed_geometry_repropagation": False,
        "limitation": (
            "Each trial is a controlled coordinate-level translation. It does not rerun raw-hit "
            "tracking or propagate from shifted non-reference source states."
        ),
        "payload_manifest": str(payload.manifest_path),
        "payload_sqlite": str(payload.sqlite_path),
        "tracklets": str(Path(args.tracklets).expanduser().resolve()),
        "propagations": str(Path(args.propagations).expanduser().resolve()),
        "q_over_p_mode": args.q_over_p_mode,
        "min_truth_match_fraction": args.min_truth_match_fraction,
        "accepted_truth_matched_pairs_before_reference_filter": evaluation.size,
        "accepted_reference_source_pairs": measurements.size,
    }
    resolved_config = {
        "reference_station": args.reference_station,
        "q_over_p_mode": args.q_over_p_mode,
        "min_truth_match_fraction": args.min_truth_match_fraction,
        "magnitudes_mm": args.magnitudes_mm,
        "trials": args.trials,
        "seed": args.seed,
        "chi2_gate": args.chi2_gate,
        "refinement_iterations": args.refinement_iterations,
        "capture_tolerance_mm": args.capture_tolerance_mm,
        "prior_sigma_mm": args.prior_sigma_mm,
        "tracklets": str(Path(args.tracklets).expanduser().resolve()),
        "propagations": str(Path(args.propagations).expanduser().resolve()),
        "payload_manifest": str(payload.manifest_path),
    }
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "capture_scan.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "capture_scan.csv", scan["points"])
    _write_plot(output_dir / "capture_scan.png", scan["points"])
    console_summary = dict(summary)
    console_summary["points"] = [
        {key: value for key, value in point.items() if key != "trial_results"}
        for point in scan["points"]
    ]
    print(json.dumps(console_summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
