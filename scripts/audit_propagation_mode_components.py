#!/usr/bin/env python3
"""Audit raw residual and covariance components for propagation modes 0 and 1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events
from evaluation.field_propagation import evaluate_field_propagation
from evaluation.propagation_mode_audit import (
    COMPARISON_CSV_FIELDNAMES,
    COMPONENT_CSV_FIELDNAMES,
    comparison_csv_rows,
    comparison_summary,
    compare_mode_components,
    component_csv_rows,
    component_summary,
    components_from_field_evaluation,
)


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_component_npz(path: Path, components) -> None:
    np.savez_compressed(
        path,
        run_id=components.run_id,
        event_id=components.event_id,
        source_tracklet_id=components.source_tracklet_id,
        target_tracklet_id=components.target_tracklet_id,
        source_station_id=components.source_station_id,
        target_station_id=components.target_station_id,
        truth_particle_id=components.truth_particle_id,
        q_over_p_mode=components.q_over_p_mode,
        residual=components.residual,
        pull=components.pull,
        propagated_covariance=components.propagated_covariance,
        target_covariance=components.target_covariance,
        combined_covariance=components.combined_covariance,
        chi2=components.chi2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracklets", required=True, help="Canonical tracklet ROOT file")
    parser.add_argument("--propagations", required=True, help="Canonical propagation ROOT file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--allow-unverified-truth",
        action="store_true",
        help="Do not require matching MC truth labels at both pair endpoints",
    )
    parser.add_argument(
        "--min-truth-match-fraction",
        type=float,
        default=0.99,
        help="Minimum endpoint hit-level truth-match fraction (default: 0.99)",
    )
    parser.add_argument("--max-events", type=int, default=None)
    args = parser.parse_args()

    if not np.isfinite(args.min_truth_match_fraction) or not 0.0 <= args.min_truth_match_fraction <= 1.0:
        parser.error("--min-truth-match-fraction must be in [0, 1]")

    output_dir = Path(args.output_dir).expanduser().resolve()
    expected_outputs = tuple(
        output_dir / name
        for name in (
            "mode0_raw_components.csv",
            "mode1_raw_components.csv",
            "mode0_components.npz",
            "mode1_components.npz",
            "mode0_vs_mode1.csv",
            "metrics.json",
            "resolved_config.yaml",
        )
    )
    if any(path.exists() for path in expected_outputs):
        raise FileExistsError(f"refusing to overwrite an existing audit artifact in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    require_truth_match = not args.allow_unverified_truth
    events = load_events(
        args.tracklets,
        max_events=args.max_events,
        require_mc_labels=require_truth_match,
    )
    records = load_propagation_records(args.propagations)
    evaluations = {
        mode: evaluate_field_propagation(
            events,
            records,
            require_truth_match=require_truth_match,
            q_over_p_mode=mode,
            min_truth_match_fraction=args.min_truth_match_fraction,
        )
        for mode in (0, 1)
    }
    components = {
        mode: components_from_field_evaluation(evaluation, records)
        for mode, evaluation in evaluations.items()
    }
    comparison = compare_mode_components(components[0], components[1])

    _write_csv(
        output_dir / "mode0_raw_components.csv",
        COMPONENT_CSV_FIELDNAMES,
        component_csv_rows(components[0]),
    )
    _write_csv(
        output_dir / "mode1_raw_components.csv",
        COMPONENT_CSV_FIELDNAMES,
        component_csv_rows(components[1]),
    )
    _write_csv(
        output_dir / "mode0_vs_mode1.csv",
        COMPARISON_CSV_FIELDNAMES,
        comparison_csv_rows(comparison),
    )
    _write_component_npz(output_dir / "mode0_components.npz", components[0])
    _write_component_npz(output_dir / "mode1_components.npz", components[1])

    resolved_config = {
        "tracklets": str(Path(args.tracklets).expanduser().resolve()),
        "propagations": str(Path(args.propagations).expanduser().resolve()),
        "require_truth_match": require_truth_match,
        "min_truth_match_fraction": args.min_truth_match_fraction,
        "max_events": args.max_events,
        "modes": [0, 1],
        "chi2_decomposition": (
            "chi2_mode1 - chi2_mode0 = residual_effect_hold_mode0_covariance + "
            "covariance_effect_after_mode1_residual"
        ),
    }
    metrics = {
        **resolved_config,
        "mode0": {
            "rejected_counts": evaluations[0].rejected_counts,
            **component_summary(components[0]),
        },
        "mode1": {
            "rejected_counts": evaluations[1].rejected_counts,
            **component_summary(components[1]),
        },
        "mode0_vs_mode1": comparison_summary(comparison),
    }
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
