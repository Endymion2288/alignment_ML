#!/usr/bin/env python3
"""Evaluate field-aware propagation records against truth-matched tracklets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events
from evaluation.field_propagation import (
    STATE_NAMES,
    evaluate_field_propagation,
    field_propagation_summary,
)


def _write_histograms(output_dir: Path, evaluation) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for index, (axis, name) in enumerate(zip(axes.flat, STATE_NAMES)):
        axis.hist(evaluation.line_residual[:, index], bins=30, histtype="step", label="line")
        axis.hist(
            evaluation.residual[:, index], bins=30, histtype="step", label="field-aware"
        )
        axis.set_xlabel(f"residual {name}")
        axis.set_ylabel("pairs")
        axis.legend()
    fig.savefig(output_dir / "residuals.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for index, (axis, name) in enumerate(zip(axes.flat, STATE_NAMES)):
        axis.hist(evaluation.line_pull[:, index], bins=30, histtype="step", label="line")
        axis.hist(evaluation.pull[:, index], bins=30, histtype="step", label="field-aware")
        axis.set_xlabel(f"pull {name}")
        axis.set_ylabel("pairs")
        axis.legend()
    fig.savefig(output_dir / "pulls.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    positive = np.concatenate(
        [evaluation.line_chi2[evaluation.line_chi2 > 0.0], evaluation.chi2[evaluation.chi2 > 0.0]]
    )
    if positive.size:
        low = max(float(np.min(positive)), 1.0e-6)
        high = max(float(np.max(positive)), low * 1.01)
        bins = np.geomspace(low, high, 31)
        axis.hist(evaluation.line_chi2, bins=bins, histtype="step", label="line")
        axis.hist(evaluation.chi2, bins=bins, histtype="step", label="field-aware")
        axis.set_xscale("log")
    axis.set_xlabel("4D chi2")
    axis.set_ylabel("pairs")
    axis.legend()
    fig.savefig(output_dir / "chi2.png", dpi=160)
    plt.close(fig)


def _write_pair_csv(output: Path, summary: dict[str, object]) -> None:
    pair_summary = summary["by_station_pair"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "station_pair",
                "records",
                "field_chi2_median",
                "line_chi2_median",
                "field_residual_x_median_mm",
                "line_residual_x_median_mm",
            ),
        )
        writer.writeheader()
        for station_pair, values in pair_summary.items():
            field = values["field_aware"]
            line = values["straight_line"]
            writer.writerow(
                {
                    "station_pair": station_pair,
                    "records": values["records"],
                    "field_chi2_median": field["chi2"]["median"],
                    "line_chi2_median": line["chi2"]["median"],
                    "field_residual_x_median_mm": field["residual"]["x_mm"]["median"],
                    "line_residual_x_median_mm": line["residual"]["x_mm"]["median"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracklets", required=True, help="Canonical tracklet ROOT file")
    parser.add_argument("--propagations", required=True, help="Canonical propagation ROOT file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument(
        "--allow-unverified-truth",
        action="store_true",
        help="Allow records whose endpoints cannot be verified with MC truth labels",
    )
    parser.add_argument(
        "--q-over-p-mode",
        type=int,
        choices=(0, 1),
        default=None,
        help="Select 0=SegmentFit seed q/p or 1=MC truth q/p audit records",
    )
    parser.add_argument(
        "--min-truth-match-fraction",
        type=float,
        default=None,
        help="Require this hit-level truth-match fraction at both pair endpoints",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    events = load_events(
        args.tracklets,
        max_events=args.max_events,
        require_mc_labels=not args.allow_unverified_truth,
    )
    evaluation = evaluate_field_propagation(
        events,
        load_propagation_records(args.propagations),
        require_truth_match=not args.allow_unverified_truth,
        q_over_p_mode=args.q_over_p_mode,
        min_truth_match_fraction=args.min_truth_match_fraction,
    )
    summary = {
        "tracklets": str(Path(args.tracklets).expanduser().resolve()),
        "propagations": str(Path(args.propagations).expanduser().resolve()),
        "require_truth_match": not args.allow_unverified_truth,
        "q_over_p_mode": args.q_over_p_mode,
        "min_truth_match_fraction": args.min_truth_match_fraction,
        **field_propagation_summary(evaluation),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_pair_csv(output_dir / "station_pair_summary.csv", summary)
    if evaluation.size:
        _write_histograms(output_dir, evaluation)
        np.savez_compressed(
            output_dir / "accepted_pairs.npz",
            run_id=evaluation.run_id,
            event_id=evaluation.event_id,
            source_tracklet_id=evaluation.source_tracklet_id,
            target_tracklet_id=evaluation.target_tracklet_id,
            source_station_id=evaluation.source_station_id,
            target_station_id=evaluation.target_station_id,
            truth_particle_id=evaluation.truth_particle_id,
            q_over_p_mode=evaluation.q_over_p_mode,
            residual=evaluation.residual,
            pull=evaluation.pull,
            chi2=evaluation.chi2,
            line_residual=evaluation.line_residual,
            line_pull=evaluation.line_pull,
            line_chi2=evaluation.line_chi2,
            combined_covariance=evaluation.combined_covariance,
        )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
