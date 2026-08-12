#!/usr/bin/env python3
"""Run one truth-fixed, residual-level station alignment closure experiment."""

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
    inject_station_offsets,
    measurements_from_field_evaluation,
    sample_station_offsets,
    solve_alignment,
)
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events
from evaluation.field_propagation import evaluate_field_propagation


DEFAULT_CONFIG = {
    "reference_station": 0,
    "seed": 12345,
    "chi2_gate": None,
    "refinement_iterations": 3,
    "prior_sigma_mm": None,
    "measurement_noise_scale": 0.0,
    "magnitude_mm": None,
}


def _load_config(path: str | None) -> dict[str, object]:
    config = dict(DEFAULT_CONFIG)
    if path is None:
        return config
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle) or {}
    config.update(supplied.get("alignment_closure", supplied))
    return config


def _parse_offsets(values: list[str]) -> dict[int, tuple[float, float]]:
    offsets: dict[int, tuple[float, float]] = {}
    for value in values:
        fields = value.split(":")
        if len(fields) != 3:
            raise argparse.ArgumentTypeError(
                f"invalid --offset '{value}'; expected station:delta_x_mm:delta_y_mm"
            )
        try:
            station = int(fields[0])
            offset = (float(fields[1]), float(fields[2]))
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"invalid --offset '{value}'") from error
        if station in offsets:
            raise argparse.ArgumentTypeError(f"duplicate offset for station {station}")
        offsets[station] = offset
    return offsets


def _write_offset_plot(output: Path, summary: dict[str, object]) -> None:
    injected = summary["injected_offsets_xy_mm"]
    recovered = summary["recovered_offsets_xy_mm"]
    station_ids = sorted(int(station) for station in injected)
    figure, axis = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    for station in station_ids:
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
    axis.scatter([], [], marker="o", color="tab:blue", label="injected")
    axis.scatter([], [], marker="x", color="tab:orange", label="recovered")
    axis.set_xlabel("station delta x [mm]")
    axis.set_ylabel("station delta y [mm]")
    axis.axhline(0.0, color="0.8", linewidth=0.8)
    axis.axvline(0.0, color="0.8", linewidth=0.8)
    axis.set_aspect("equal", adjustable="datalim")
    axis.legend()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracklets", required=True, help="Canonical MC tracklet ROOT file")
    parser.add_argument("--propagations", required=True, help="Canonical propagation ROOT file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None, help="YAML closure configuration")
    parser.add_argument("--reference-station", type=int, default=None)
    parser.add_argument(
        "--offset",
        action="append",
        default=[],
        metavar="STATION:DX_MM:DY_MM",
        help="Explicit coordinate-reporting offset; repeat for several stations",
    )
    parser.add_argument(
        "--magnitude-mm",
        type=float,
        default=None,
        help="Instead of --offset, sample this fixed radial offset for each non-reference station",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--chi2-gate",
        type=float,
        default=None,
        help="Optional gate on baseline-subtracted two-dimensional residual increments",
    )
    parser.add_argument("--refinement-iterations", type=int, default=None)
    parser.add_argument("--prior-sigma-mm", type=float, default=None)
    parser.add_argument("--measurement-noise-scale", type=float, default=None)
    args = parser.parse_args()
    config = _load_config(args.config)
    for name in (
        "reference_station",
        "seed",
        "chi2_gate",
        "refinement_iterations",
        "prior_sigma_mm",
        "measurement_noise_scale",
        "magnitude_mm",
    ):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    if bool(args.offset) == (config["magnitude_mm"] is not None):
        parser.error("supply exactly one of --offset or --magnitude-mm")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    events = load_events(args.tracklets, require_mc_labels=True)
    evaluation = evaluate_field_propagation(
        events, load_propagation_records(args.propagations), require_truth_match=True
    )
    measurements = measurements_from_field_evaluation(evaluation)
    rng = np.random.default_rng(int(config["seed"]))
    if config["magnitude_mm"] is not None:
        injected = sample_station_offsets(
            measurements.station_ids,
            int(config["reference_station"]),
            float(config["magnitude_mm"]),
            rng,
        )
    else:
        injected = _parse_offsets(args.offset)
    observed = inject_station_offsets(
        measurements,
        injected,
        reference_station=int(config["reference_station"]),
        rng=rng,
        measurement_noise_scale=float(config["measurement_noise_scale"]),
    )
    fit = solve_alignment(
        measurements,
        observed,
        reference_station=int(config["reference_station"]),
        chi2_gate=(None if config["chi2_gate"] is None else float(config["chi2_gate"])),
        refinement_iterations=int(config["refinement_iterations"]),
        prior_sigma_mm=(
            None if config["prior_sigma_mm"] is None else float(config["prior_sigma_mm"])
        ),
    )
    summary = {
        "method": "truth-fixed_baseline-subtracted_residual-level_alignment_closure",
        "coordinate_reporting_convention": "r_injected = r_nominal + delta_target - delta_source",
        "deformed_geometry_repropagation": False,
        "tracklets": str(Path(args.tracklets).expanduser().resolve()),
        "propagations": str(Path(args.propagations).expanduser().resolve()),
        "accepted_truth_matched_pairs": evaluation.size,
        "reference_station": config["reference_station"],
        "chi2_gate": config["chi2_gate"],
        "refinement_iterations": config["refinement_iterations"],
        "prior_sigma_mm": config["prior_sigma_mm"],
        "measurement_noise_scale": config["measurement_noise_scale"],
        "seed": config["seed"],
        **alignment_error_summary(fit, injected),
    }
    resolved_config = {
        **config,
        "tracklets": str(Path(args.tracklets).expanduser().resolve()),
        "propagations": str(Path(args.propagations).expanduser().resolve()),
        "explicit_offsets_xy_mm": injected if args.offset else None,
    }
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_config, handle, sort_keys=True)
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
        observed_residual_xy_mm=observed,
        final_increment_residual_xy_mm=fit.final_increment_residual_xy_mm,
        final_increment_chi2=fit.final_increment_chi2,
        active_mask=fit.active_mask,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
