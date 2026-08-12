#!/usr/bin/env python3
"""Recover a station-level payload from two physically refitted MC outputs.

Unlike the earlier coordinate-level closure, both inputs here are produced by
Calypso's SegmentFit/GhostBusters chain after loading their respective
alignment conditions.  Truth association is held fixed only for validation;
the fitted alignment parameters remain global station dx/dy values.
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
    solve_alignment,
)
from alignment.payload import load_station_alignment_payload
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events
from evaluation.field_propagation import (
    FieldPropagationEvaluation,
    evaluate_field_propagation,
    field_propagation_summary,
)


def _pair_key(evaluation: FieldPropagationEvaluation, row: int) -> tuple[int, int, int, int, int]:
    return (
        int(evaluation.run_id[row]),
        int(evaluation.event_id[row]),
        int(evaluation.source_tracklet_id[row]),
        int(evaluation.target_tracklet_id[row]),
        int(evaluation.q_over_p_mode[row]),
    )


def _ordered_displaced_rows(
    nominal: FieldPropagationEvaluation,
    displaced: FieldPropagationEvaluation,
) -> np.ndarray:
    """Return rows of the displaced evaluation in nominal pair order."""
    nominal_keys = [_pair_key(nominal, row) for row in range(nominal.size)]
    displaced_keys = [_pair_key(displaced, row) for row in range(displaced.size)]
    if len(set(nominal_keys)) != len(nominal_keys):
        raise ValueError("nominal field evaluation contains duplicate pair identities")
    displaced_index = {key: row for row, key in enumerate(displaced_keys)}
    if len(displaced_index) != len(displaced_keys):
        raise ValueError("displaced field evaluation contains duplicate pair identities")
    missing = sorted(set(nominal_keys) - set(displaced_index))
    extra = sorted(set(displaced_index) - set(nominal_keys))
    if missing or extra:
        raise ValueError(
            "nominal and displaced refit outputs select different truth pair identities; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    ordered_rows = np.asarray([displaced_index[key] for key in nominal_keys], dtype=np.intp)
    if not np.array_equal(
        nominal.source_station_id, displaced.source_station_id[ordered_rows]
    ) or not np.array_equal(nominal.target_station_id, displaced.target_station_id[ordered_rows]):
        raise ValueError("station-pair identity drift after displaced-geometry refit")
    if not np.array_equal(nominal.truth_particle_id, displaced.truth_particle_id[ordered_rows]):
        raise ValueError("truth-particle identity drift after displaced-geometry refit")
    return ordered_rows


def _write_offset_plot(output: Path, summary: dict[str, object]) -> None:
    injected = summary["injected_offsets_xy_mm"]
    recovered = summary["recovered_offsets_xy_mm"]
    stations = sorted(int(station) for station in injected)
    figure, axis = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    for station in stations:
        true_value = np.asarray(injected[str(station)] if str(station) in injected else injected[station])
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


def _covariance_response_summary(
    nominal_covariance: np.ndarray,
    displaced_covariance: np.ndarray,
) -> dict[str, float]:
    difference = displaced_covariance - nominal_covariance
    nominal_sign, nominal_logdet = np.linalg.slogdet(nominal_covariance)
    displaced_sign, displaced_logdet = np.linalg.slogdet(displaced_covariance)
    if not np.all(nominal_sign > 0.0) or not np.all(displaced_sign > 0.0):
        raise ValueError("accepted combined covariance lost positive definiteness")
    return {
        "combined_covariance_max_abs_change": float(np.max(np.abs(difference))),
        "combined_covariance_rms_change": float(np.sqrt(np.mean(np.square(difference)))),
        "combined_covariance_logdet_change_mean": float(
            np.mean(displaced_logdet - nominal_logdet)
        ),
        "combined_covariance_logdet_change_max_abs": float(
            np.max(np.abs(displaced_logdet - nominal_logdet))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal-tracklets", required=True)
    parser.add_argument("--nominal-propagations", required=True)
    parser.add_argument("--displaced-tracklets", required=True)
    parser.add_argument("--displaced-propagations", required=True)
    parser.add_argument("--payload-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-station", type=int, default=0)
    parser.add_argument(
        "--q-over-p-mode",
        choices=(0, 1),
        type=int,
        default=0,
        help="0=SegmentFit seed q/p (default); 1=MC truth q/p audit mode",
    )
    parser.add_argument("--min-truth-match-fraction", type=float, default=0.99)
    parser.add_argument("--chi2-gate", type=float, default=None)
    parser.add_argument("--refinement-iterations", type=int, default=3)
    parser.add_argument("--prior-sigma-mm", type=float, default=None)
    args = parser.parse_args()

    if not np.isfinite(args.min_truth_match_fraction) or not 0.0 <= args.min_truth_match_fraction <= 1.0:
        parser.error("--min-truth-match-fraction must be in [0, 1]")
    output_dir = Path(args.output_dir).expanduser().resolve()
    expected_outputs = (
        output_dir / "closure.json",
        output_dir / "resolved_config.yaml",
        output_dir / "alignment_offsets.png",
        output_dir / "closure_pairs.npz",
    )
    if any(path.exists() for path in expected_outputs):
        raise FileExistsError(f"refusing to overwrite an existing closure artifact in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_station_alignment_payload(args.payload_manifest)
    reference_offset = payload.offset_for_station(args.reference_station)
    if not np.allclose(reference_offset, 0.0, rtol=0.0, atol=1.0e-15):
        raise ValueError("the reference station must have zero payload offset")
    evaluation_kwargs = {
        "require_truth_match": True,
        "q_over_p_mode": args.q_over_p_mode,
        "min_truth_match_fraction": args.min_truth_match_fraction,
    }
    nominal_evaluation = evaluate_field_propagation(
        load_events(args.nominal_tracklets, require_mc_labels=True),
        load_propagation_records(args.nominal_propagations),
        **evaluation_kwargs,
    )
    displaced_evaluation = evaluate_field_propagation(
        load_events(args.displaced_tracklets, require_mc_labels=True),
        load_propagation_records(args.displaced_propagations),
        **evaluation_kwargs,
    )
    displaced_rows = _ordered_displaced_rows(nominal_evaluation, displaced_evaluation)
    measurements = measurements_from_field_evaluation(nominal_evaluation)
    observed_residual = np.asarray(displaced_evaluation.residual[displaced_rows], dtype=np.float64)
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
    response_error = observed_increment - expected_increment
    fit = solve_alignment(
        measurements,
        observed_residual[:, :2],
        reference_station=args.reference_station,
        chi2_gate=args.chi2_gate,
        refinement_iterations=args.refinement_iterations,
        prior_sigma_mm=args.prior_sigma_mm,
    )
    covariance_response = _covariance_response_summary(
        nominal_evaluation.combined_covariance,
        displaced_evaluation.combined_covariance[displaced_rows],
    )
    nominal_diagnostics = field_propagation_summary(nominal_evaluation)
    displaced_diagnostics = field_propagation_summary(displaced_evaluation)
    summary = {
        "method": "truth-fixed_displaced-geometry_segment-refit_alignment_closure",
        "refit_chain": "persisted SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit -> NtupleDumper -> FaserActsExtrapolationTool",
        "physical_geometry_repropagation": True,
        "raw_hit_reconstruction": False,
        "reference_source_only": False,
        "fit_weight_interpretation": (
            "Nominal combined covariance is used as a deterministic WLS weight for the "
            "baseline-subtracted response. It is not an independent covariance for the difference "
            "of two refits of the same hits."
        ),
        "payload_manifest": str(payload.manifest_path),
        "payload_sqlite": str(payload.sqlite_path),
        "payload_pool": str(payload.pool_path),
        "payload_pool_catalog": str(payload.pool_catalog_path),
        "nominal_tracklets": str(Path(args.nominal_tracklets).expanduser().resolve()),
        "nominal_propagations": str(Path(args.nominal_propagations).expanduser().resolve()),
        "displaced_tracklets": str(Path(args.displaced_tracklets).expanduser().resolve()),
        "displaced_propagations": str(Path(args.displaced_propagations).expanduser().resolve()),
        "q_over_p_mode": args.q_over_p_mode,
        "min_truth_match_fraction": args.min_truth_match_fraction,
        "accepted_truth_matched_pairs": measurements.size,
        "active_truth_matched_pairs": int(np.count_nonzero(fit.active_mask)),
        "reference_station": args.reference_station,
        "chi2_gate": args.chi2_gate,
        "refinement_iterations": args.refinement_iterations,
        "prior_sigma_mm": args.prior_sigma_mm,
        "physical_response_increment_max_abs_error_mm": float(np.max(np.abs(response_error))),
        "physical_response_increment_rms_error_mm": float(
            np.sqrt(np.mean(np.square(response_error)))
        ),
        "residual_slope_increment_max_abs": float(
            np.max(np.abs(observed_residual[:, 2:] - nominal_evaluation.residual[:, 2:]))
        ),
        "nominal_field_aware_by_station_pair": nominal_diagnostics["by_station_pair"],
        "displaced_field_aware_by_station_pair": displaced_diagnostics["by_station_pair"],
        **covariance_response,
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
        "nominal_propagations": str(Path(args.nominal_propagations).expanduser().resolve()),
        "displaced_tracklets": str(Path(args.displaced_tracklets).expanduser().resolve()),
        "displaced_propagations": str(Path(args.displaced_propagations).expanduser().resolve()),
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
        run_id=nominal_evaluation.run_id,
        event_id=nominal_evaluation.event_id,
        source_tracklet_id=nominal_evaluation.source_tracklet_id,
        target_tracklet_id=nominal_evaluation.target_tracklet_id,
        source_station_id=measurements.source_station_id,
        target_station_id=measurements.target_station_id,
        nominal_residual_xy_mm=measurements.nominal_residual_xy_mm,
        displaced_residual_xy_mm=observed_residual[:, :2],
        observed_increment_xy_mm=observed_increment,
        expected_increment_xy_mm=expected_increment,
        physical_response_error_xy_mm=response_error,
        nominal_combined_covariance=nominal_evaluation.combined_covariance,
        displaced_combined_covariance=displaced_evaluation.combined_covariance[displaced_rows],
        final_increment_residual_xy_mm=fit.final_increment_residual_xy_mm,
        final_increment_chi2=fit.final_increment_chi2,
        active_mask=fit.active_mask,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
