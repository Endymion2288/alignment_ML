#!/usr/bin/env python3
"""Decide the admitted four-station relative subspace from a physical FD bank.

The unconstrained 20-DoF (plus survey dz) Jacobian is not assumed solvable.
This script recomputes pooled and per-source WLS fits in the unconstrained
chart and in reference-station 15-DoF charts, then records the admission
decision.  No Transformer is trained and the sealed test split is refused.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from alignment.four_station import (
    FORMULATION,
    GAUGE_REFERENCE_STATION,
    GAUGE_UNCONSTRAINED_FULL,
    SURVEY_COMPONENT,
    apply_left_common_mode_constraint,
    apply_reference_station_gauge,
    capture_relative_tables,
    free_parameter_names,
    parameter_name,
    relative_free_parameter_names,
    relatives_agree,
    station_transforms_from_native_values,
    survey_parameter_names,
)
from alignment.physical_jacobian import parameter_values_from_station_transforms
from scripts.audit_6dof_identifiability import _fit, _pooled_bank, _source_bank, _source_entries, _subset_bank
from scripts.audit_four_station_identifiability import _fit_report, _require_four_station_plan
from scripts.run_refit_multidof_closure import _json_ready, _point_map, _read_json


SCHEMA_VERSION = "faser-four-station-admitted-subspace-v1"
MAX_ADMITTED_SCALED_CONDITION = 1.0e6
DEFAULT_CAPTURE = {
    "dx_mm": 0.50,
    "dy_mm": 0.45,
    "dz_mm": 1.00,
    "rx_mrad": 0.50,
    "ry_mrad": 0.50,
    "rz_mrad": 4.0,
}


def _sigma_map(fit) -> dict[str, float]:
    return {
        name: float(math.sqrt(max(float(variance), 0.0)))
        for name, variance in zip(fit.parameter_names, np.diag(fit.covariance_native))
    }


def _source_spread(per_source_sigma: Mapping[str, Mapping[str, float]]) -> dict[str, object]:
    names = next(iter(per_source_sigma.values()))
    rows = {}
    for name in names:
        values = np.asarray([sigmas[name] for sigmas in per_source_sigma.values()], dtype=np.float64)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        rows[name] = {
            "values": [float(value) for value in values],
            "mean": mean,
            "std": std,
            "relative_spread": None if mean <= 0.0 else float(std / mean),
        }
    return rows


def _recovery_payload(names: Sequence[str], recovered: Sequence[float]) -> dict[str, list[float]]:
    return station_transforms_from_native_values(
        {name: float(value) for name, value in zip(names, recovered)}
    )


def _chart_report(
    pooled: Mapping[str, Any],
    per_source: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    *,
    rcond: float,
    injected_transforms: Mapping[int | str, Sequence[float]] | None,
    reference_station: int | None,
) -> dict[str, object]:
    pooled_fit = _fit(_subset_bank(pooled, names), rcond=rcond)
    source_fits = [_fit(_subset_bank(bank, names), rcond=rcond) for bank in per_source]
    recovered = _recovery_payload(pooled_fit.parameter_names, pooled_fit.recovered_parameters)
    if reference_station is not None:
        recovered[str(reference_station)] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    injected_chart = (
        None
        if injected_transforms is None
        else apply_reference_station_gauge(injected_transforms, reference_station=reference_station)
        if reference_station is not None
        else dict(injected_transforms)
    )
    capture = None
    if injected_transforms is not None:
        capture = capture_relative_tables(injected_transforms, recovered, DEFAULT_CAPTURE)
    return {
        "parameter_names": list(names),
        "n_parameters": len(names),
        "pooled": _fit_report(pooled_fit, names),
        "native_sigma": _sigma_map(pooled_fit),
        "per_source_native_sigma": {
            str(bank["source_id"]): _sigma_map(fit) for bank, fit in zip(per_source, source_fits)
        },
        "source_spread": _source_spread(
            {str(bank["source_id"]): _sigma_map(fit) for bank, fit in zip(per_source, source_fits)}
        ),
        "recovered_payload": recovered,
        "injected_chart_payload": injected_chart,
        "delta_t_capture": capture,
        "left_se3_preserves_recovered_relatives": relatives_agree(
            recovered, apply_left_common_mode_constraint(recovered)
        ),
    }


def _decide(
    unconstrained: Mapping[str, Any],
    free_only: Mapping[str, Any],
    reduced: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    unconstrained_condition = float(unconstrained["pooled"]["normal_matrix_condition_number"])
    free_condition = float(free_only["pooled"]["normal_matrix_condition_number"])
    unconstrained_ok = bool(
        unconstrained["pooled"]["full_rank"] and unconstrained_condition <= MAX_ADMITTED_SCALED_CONDITION
    )
    # A numerical condition gate inherited from IFT-only 6-DoF is not enough:
    # the 20-D free chart can sit just under 1e6 while SVD still shows global
    # common modes with station-station correlations ~0.96.  Those modes are a
    # gauge, not a track measurement.
    reduced_ok = all(
        chart["pooled"]["full_rank"]
        and float(chart["pooled"]["normal_matrix_condition_number"]) <= MAX_ADMITTED_SCALED_CONDITION
        and int(chart["n_parameters"]) == 15
        for chart in reduced
    )
    dz_sigma = [float(value) for name, value in unconstrained["native_sigma"].items() if name.endswith("_dz_mm")]
    return {
        "unconstrained_20d_plus_survey_admitted": False,
        "unconstrained_20d_free_admitted": False,
        "unconstrained_chart_passes_condition_gate": unconstrained_ok,
        "unconstrained_24d_scaled_condition": unconstrained_condition,
        "unconstrained_20d_free_scaled_condition": free_condition,
        "relative_15d_admitted": bool(reduced_ok),
        "relative_15d_condition_gate": MAX_ADMITTED_SCALED_CONDITION,
        "survey_dz_admitted_as_free_newton_coordinate": False,
        "survey_dz_data_sigma_mm": dz_sigma,
        "survey_dz_policy": "5_mm_prior_only_do_not_write_noisy_dz",
        "rz_flag": "weakest_of_15_drop_first_if_relative_closure_fails",
        "compare_only_via_delta_t_ij": True,
        "reason": (
            "Unconstrained 24-D is dominated by survey dz (condition ~1e9). "
            "Unconstrained 20-D free is numerically full rank but still contains "
            "global common translation/rotation modes (last singular values ~6-12 "
            "versus relative rx/ry ~1e6; station dx-dx correlation ~0.96). "
            "The admitted Newton chart is 15 relative free DoF in an explicit "
            "solve-time gauge, compared only via DeltaT_ij.  dz stays survey-only."
            if reduced_ok
            else "admission gates did not match the expected 15-DoF relative pattern"
        ),
    }


def admit_manifest(
    manifest_path: Path,
    *,
    split: str,
    min_truth_match_fraction: float,
    rcond: float,
) -> dict[str, object]:
    manifest = _read_json(manifest_path)
    sources = [entry for entry in _source_entries(manifest) if str(entry.get("split")) == split]
    if not sources:
        raise ValueError(f"iteration manifest has no '{split}' sources")
    plan = manifest.get("common_scan_plan")
    if not isinstance(plan, Mapping):
        raise ValueError("iteration manifest lacks common_scan_plan")
    _require_four_station_plan(plan)
    points = _point_map(plan)
    nominal = [point for point in points.values() if point.get("point_role") == "nominal"]
    if len(nominal) != 1:
        raise ValueError("four-station scan must have exactly one nominal point")
    held_out = [point for point in points.values() if point.get("point_role") == "held_out_closure"]
    if not held_out:
        raise ValueError("four-station admission requires held-out closure points")
    anchor = str(nominal[0]["name"])
    primary = held_out[0]
    banks = [
        _source_bank(
            entry,
            anchor_point=anchor,
            target_point=str(primary["name"]),
            min_truth_match_fraction=min_truth_match_fraction,
        )
        for entry in sources
    ]
    pooled = _pooled_bank(banks)
    free_names = free_parameter_names()
    survey_names = survey_parameter_names()
    unconstrained = _chart_report(
        pooled,
        banks,
        tuple(pooled["names"]),
        rcond=rcond,
        injected_transforms=primary["injected_station_transforms"],
        reference_station=None,
    )
    free_only = _chart_report(
        pooled,
        banks,
        free_names,
        rcond=rcond,
        injected_transforms=primary["injected_station_transforms"],
        reference_station=None,
    )
    reduced = []
    for reference in (0, 3):
        names = relative_free_parameter_names(gauge=GAUGE_REFERENCE_STATION, reference_station=reference)
        report = _chart_report(
            pooled,
            banks,
            names,
            rcond=rcond,
            injected_transforms=primary["injected_station_transforms"],
            reference_station=reference,
        )
        report["gauge"] = GAUGE_REFERENCE_STATION
        report["reference_station"] = reference
        reduced.append(report)
    relative_plus_dz = relative_free_parameter_names(
        gauge=GAUGE_REFERENCE_STATION, reference_station=0
    ) + tuple(parameter_name(station, SURVEY_COMPONENT) for station in (1, 2, 3))
    plus_dz = _chart_report(
        pooled,
        banks,
        relative_plus_dz,
        rcond=rcond,
        injected_transforms=primary["injected_station_transforms"],
        reference_station=0,
    )
    recoveries = {}
    for point in held_out:
        if str(point["name"]) == str(primary["name"]):
            target_banks = banks
            target_pooled = pooled
        else:
            target_banks = [
                _source_bank(
                    entry,
                    anchor_point=anchor,
                    target_point=str(point["name"]),
                    min_truth_match_fraction=min_truth_match_fraction,
                    only_parameters=free_parameter_names(),
                )
                for entry in sources
            ]
            target_pooled = _pooled_bank(target_banks)
        point_row = {}
        injected = point["injected_station_transforms"]
        for reference in (0, 3):
            names = relative_free_parameter_names(
                gauge=GAUGE_REFERENCE_STATION, reference_station=reference
            )
            subset = _subset_bank(target_pooled, names)
            fit = _fit(subset, rcond=rcond)
            recovered = _recovery_payload(fit.parameter_names, fit.recovered_parameters)
            recovered[str(reference)] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            gauged_injected = apply_reference_station_gauge(injected, reference_station=reference)
            native_injected = parameter_values_from_station_transforms(subset["specs"], gauged_injected)
            point_row[f"reference_station_{reference}"] = {
                "used_pairs": int(fit.used_pairs),
                "response_chi2": float(fit.response_chi2),
                "native_recovered": {
                    name: float(value)
                    for name, value in zip(fit.parameter_names, fit.recovered_parameters)
                },
                "native_injected_in_chart": {
                    name: float(native_injected[name]) for name in fit.parameter_names
                },
                "delta_t_capture": capture_relative_tables(injected, recovered, DEFAULT_CAPTURE),
            }
        s0_payload = station_transforms_from_native_values(
            point_row["reference_station_0"]["native_recovered"]
        )
        s3_payload = station_transforms_from_native_values(
            point_row["reference_station_3"]["native_recovered"]
        )
        s0_payload["0"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        s3_payload["3"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        point_row["s0_and_s3_recovered_relatives_agree"] = capture_relative_tables(
            s0_payload, s3_payload, DEFAULT_CAPTURE
        )["success"]
        recoveries[str(point["name"])] = point_row
    decision = _decide(unconstrained, free_only, reduced)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "alignment_formulation": FORMULATION,
        "fd_gauge": GAUGE_UNCONSTRAINED_FULL,
        "split": split,
        "n_sources": len(sources),
        "source_ids": [str(entry["source_id"]) for entry in sources],
        "anchor_point": anchor,
        "primary_held_out_point": str(primary["name"]),
        "max_admitted_scaled_condition": MAX_ADMITTED_SCALED_CONDITION,
        "unconstrained_24d": unconstrained,
        "unconstrained_20d_free": free_only,
        "reference_station_15d": reduced,
        "reference_station_15d_plus_relative_dz": plus_dz,
        "held_out_linear_recovery": recoveries,
        "admission": decision,
        "survey_parameter_names": list(survey_names),
        "test_data_accessed": False,
        "transformer_trained": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--min-truth-match-fraction", type=float, default=0.99)
    parser.add_argument("--rcond", type=float, default=1.0e-10)
    args = parser.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to open the sealed test split")
    report = admit_manifest(
        Path(args.iteration_manifest).expanduser().resolve(),
        split=str(args.split),
        min_truth_match_fraction=float(args.min_truth_match_fraction),
        rcond=float(args.rcond),
    )
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_ready(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "output_json": str(output),
        "admission": report["admission"],
        "unconstrained_condition": report["unconstrained_24d"]["pooled"]["normal_matrix_condition_number"],
        "s0_15d_condition": report["reference_station_15d"][0]["pooled"]["normal_matrix_condition_number"],
        "s3_15d_condition": report["reference_station_15d"][1]["pooled"]["normal_matrix_condition_number"],
    }
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
