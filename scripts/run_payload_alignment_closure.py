#!/usr/bin/env python3
"""Close V1 station x/y alignment from a condition-payload coordinate injection.

The observed tracklet file must be produced by
``apply_station_alignment_payload.py``.  This command keeps truth association
fixed and restricts pairs to the reference station as source.  That restriction
is required while propagation records still originate from persisted nominal
source states rather than a new local-tracklet refit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from alignment.closure import (
    alignment_error_summary,
    measurements_from_field_evaluation,
    select_measurements,
    solve_alignment,
)
from alignment.payload import load_station_alignment_payload
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events
from evaluation.field_propagation import FieldPropagationEvaluation, evaluate_field_propagation


def _pair_keys(evaluation: FieldPropagationEvaluation) -> list[tuple[int, int, int, int, int]]:
    return [
        (
            int(run),
            int(event),
            int(source),
            int(target),
            int(mode),
        )
        for run, event, source, target, mode in zip(
            evaluation.run_id,
            evaluation.event_id,
            evaluation.source_tracklet_id,
            evaluation.target_tracklet_id,
            evaluation.q_over_p_mode,
        )
    ]


def _ordered_observed_residual(
    nominal: FieldPropagationEvaluation,
    observed: FieldPropagationEvaluation,
) -> np.ndarray:
    """Put observed residuals in nominal pair order and reject key drift."""
    nominal_keys = _pair_keys(nominal)
    observed_keys = _pair_keys(observed)
    if len(set(nominal_keys)) != len(nominal_keys):
        raise ValueError("nominal field evaluation contains duplicate pair identities")
    observed_index = {key: row for row, key in enumerate(observed_keys)}
    if len(observed_index) != len(observed_keys):
        raise ValueError("observed field evaluation contains duplicate pair identities")
    missing = sorted(set(nominal_keys) - set(observed_keys))
    extra = sorted(set(observed_keys) - set(nominal_keys))
    if missing or extra:
        raise ValueError(
            "nominal and observed evaluations select different pair identities; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    return np.asarray(
        [observed.residual[observed_index[key]] for key in nominal_keys], dtype=np.float64
    )


def _write_offset_plot(output: Path, summary: dict[str, object]) -> None:
    injected = summary["injected_offsets_xy_mm"]
    recovered = summary["recovered_offsets_xy_mm"]
    stations = sorted(int(station) for station in injected)
    figure, axis = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    for station in stations:
        true_value = np.asarray(
            injected[str(station)] if str(station) in injected else injected[station]
        )
        fitted_value = np.asarray(
            recovered[str(station)] if str(station) in recovered else recovered[station]
        )
        axis.scatter(*true_value, marker="o", color="tab:blue")
        axis.scatter(*fitted_value, marker="x", color="tab:orange")
        axis.annotate(f"S{station}", true_value, xytext=(4, 4), textcoords="offset points")
        axis.plot(
            [true_value[0], fitted_value[0]],
            [true_value[1], fitted_value[1]],
            color="0.55",
            linewidth=0.8,
        )
    axis.scatter([], [], marker="o", color="tab:blue", label="payload")
    axis.scatter([], [], marker="x", color="tab:orange", label="recovered")
    axis.axhline(0.0, color="0.8", linewidth=0.8)
    axis.axvline(0.0, color="0.8", linewidth=0.8)
    axis.set_xlabel("station delta x [mm]")
    axis.set_ylabel("station delta y [mm]")
    axis.set_aspect("equal", adjustable="datalim")
    axis.legend()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal-tracklets", required=True)
    parser.add_argument("--observed-tracklets", required=True)
    parser.add_argument("--propagations", required=True)
    parser.add_argument("--payload-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-station", type=int, default=0)
    parser.add_argument("--q-over-p-mode", type=int, default=1)
    parser.add_argument("--min-truth-match-fraction", type=float, default=0.99)
    parser.add_argument("--chi2-gate", type=float, default=None)
    parser.add_argument("--refinement-iterations", type=int, default=3)
    parser.add_argument("--prior-sigma-mm", type=float, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = (
        output_dir / "closure.json",
        output_dir / "resolved_config.yaml",
        output_dir / "alignment_offsets.png",
        output_dir / "closure_pairs.npz",
    )
    if any(path.exists() for path in expected_outputs):
        raise FileExistsError(f"refusing to overwrite an existing closure artifact in {output_dir}")

    payload = load_station_alignment_payload(args.payload_manifest)
    reference_offset = payload.offset_for_station(args.reference_station)
    if not np.allclose(reference_offset, 0.0, rtol=0.0, atol=1.0e-15):
        raise ValueError("the reference station must have zero payload offset")
    if not np.isfinite(args.min_truth_match_fraction) or not 0.0 <= args.min_truth_match_fraction <= 1.0:
        raise ValueError("--min-truth-match-fraction must be in [0, 1]")

    records = load_propagation_records(args.propagations)
    evaluation_kwargs = {
        "require_truth_match": True,
        "q_over_p_mode": args.q_over_p_mode,
        "min_truth_match_fraction": args.min_truth_match_fraction,
    }
    nominal_evaluation = evaluate_field_propagation(
        load_events(args.nominal_tracklets, require_mc_labels=True), records, **evaluation_kwargs
    )
    observed_evaluation = evaluate_field_propagation(
        load_events(args.observed_tracklets, require_mc_labels=True), records, **evaluation_kwargs
    )
    observed_residual = _ordered_observed_residual(nominal_evaluation, observed_evaluation)
    measurements_all = measurements_from_field_evaluation(nominal_evaluation)
    reference_mask = measurements_all.source_station_id == args.reference_station
    measurements = select_measurements(measurements_all, reference_mask)
    observed_residual = observed_residual[reference_mask]

    station_to_offset = {
        int(station): payload.offset_for_station(int(station))
        for station in measurements.station_ids
    }
    expected_increment = np.asarray(
        [
            np.asarray(station_to_offset[int(target)]) - np.asarray(station_to_offset[int(source)])
            for source, target in zip(measurements.source_station_id, measurements.target_station_id)
        ],
        dtype=np.float64,
    )
    observed_increment = observed_residual[:, :2] - measurements.nominal_residual_xy_mm
    coordinate_error = observed_increment - expected_increment

    fit = solve_alignment(
        measurements,
        observed_residual[:, :2],
        reference_station=args.reference_station,
        chi2_gate=args.chi2_gate,
        refinement_iterations=args.refinement_iterations,
        prior_sigma_mm=args.prior_sigma_mm,
    )
    summary = {
        "method": "truth-fixed_condition-payload_coordinate-level_alignment_closure",
        "coordinate_convention": "x' = x + dx_station; y' = y + dy_station",
        "reference_source_only": True,
        "deformed_geometry_repropagation": False,
        "limitation": (
            "The payload is validated in Calypso conditions, but persisted source tracklets are not "
            "refitted under the displaced geometry. Propagation records are nominal and source is fixed "
            "to the zero-offset reference station."
        ),
        "payload_manifest": str(payload.manifest_path),
        "payload_sqlite": str(payload.sqlite_path),
        "payload_pool": str(payload.pool_path),
        "payload_pool_catalog": str(payload.pool_catalog_path),
        "nominal_tracklets": str(Path(args.nominal_tracklets).expanduser().resolve()),
        "observed_tracklets": str(Path(args.observed_tracklets).expanduser().resolve()),
        "propagations": str(Path(args.propagations).expanduser().resolve()),
        "q_over_p_mode": args.q_over_p_mode,
        "min_truth_match_fraction": args.min_truth_match_fraction,
        "accepted_truth_matched_pairs_before_reference_filter": nominal_evaluation.size,
        "accepted_reference_source_pairs": measurements.size,
        "reference_station": args.reference_station,
        "chi2_gate": args.chi2_gate,
        "refinement_iterations": args.refinement_iterations,
        "prior_sigma_mm": args.prior_sigma_mm,
        "coordinate_injection_increment_max_abs_error_mm": float(np.max(np.abs(coordinate_error))),
        "coordinate_injection_increment_rms_error_mm": float(
            np.sqrt(np.mean(np.square(coordinate_error)))
        ),
        **alignment_error_summary(fit, station_to_offset),
    }
    resolved_config = {
        "reference_station": args.reference_station,
        "q_over_p_mode": args.q_over_p_mode,
        "min_truth_match_fraction": args.min_truth_match_fraction,
        "chi2_gate": args.chi2_gate,
        "refinement_iterations": args.refinement_iterations,
        "prior_sigma_mm": args.prior_sigma_mm,
        "nominal_tracklets": str(Path(args.nominal_tracklets).expanduser().resolve()),
        "observed_tracklets": str(Path(args.observed_tracklets).expanduser().resolve()),
        "propagations": str(Path(args.propagations).expanduser().resolve()),
        "payload_manifest": str(payload.manifest_path),
    }
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "closure.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_offset_plot(output_dir / "alignment_offsets.png", summary)
    np.savez_compressed(
        output_dir / "closure_pairs.npz",
        source_station_id=measurements.source_station_id,
        target_station_id=measurements.target_station_id,
        nominal_residual_xy_mm=measurements.nominal_residual_xy_mm,
        observed_residual_xy_mm=observed_residual[:, :2],
        expected_increment_xy_mm=expected_increment,
        coordinate_increment_error_xy_mm=coordinate_error,
        final_increment_residual_xy_mm=fit.final_increment_residual_xy_mm,
        final_increment_chi2=fit.final_increment_chi2,
        active_mask=fit.active_mask,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
