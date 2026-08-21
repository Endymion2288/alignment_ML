#!/usr/bin/env python3
"""Unknown-association 15-DoF relative closure on a four-station physical bank.

Frozen V2 selects routes once on the identity physical reference payload.
Those truth-free selected edges are re-measured in existing central-FD and
held-out candidate graphs.  Solve columns are exactly the admitted 15-DoF
reference-station chart; observation filtering keeps stations 0-3.  Capture
is gauge-invariant ``ΔT_ij``.  Residual reduction is recorded only as DQ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.four_station import (
    FORMULATION,
    GAUGE_REFERENCE_STATION,
    apply_left_common_mode_constraint,
    capture_relative_tables,
    relative_free_parameter_names,
    relatives_agree,
    require_four_station_route_selected_contract,
    station_transforms_from_native_values,
)
from alignment.physical_jacobian import (
    parameter_values_from_station_transforms,
    solve_physical_finite_difference,
)
from alignment.route_selected_update import (
    align_route_selected_observations,
    apply_observation_statistics,
    read_anchor_selected_field_edge_observations,
)
from scripts.audit_four_station_identifiability import _require_four_station_plan
from scripts.run_refit_multidof_closure import _json_ready, _point_map, _read_json
from scripts.run_route_selected_multidof_update import _probe_for_parameter, _specs


def _merge_banks(
    banks: Sequence[Mapping[tuple[object, ...], object]],
) -> dict[tuple[object, ...], object]:
    merged: dict[tuple[object, ...], object] = {}
    for bank in banks:
        clash = set(merged).intersection(bank)
        if clash:
            raise ValueError("duplicate observation keys while merging four-station sources")
        merged.update(dict(bank))
    return merged


def _assets(scan_root: Path, point: Mapping[str, object]) -> tuple[Path, Path]:
    relative = point.get("relative_point_dir")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"point '{point.get('name')}' lacks relative_point_dir")
    root = scan_root / relative / "refit"
    tracklets = root / "tracklets.root"
    propagations = root / "propagations.root"
    if not tracklets.is_file() or not propagations.is_file():
        raise FileNotFoundError(f"physical point '{point.get('name')}' is incomplete")
    return tracklets, propagations


def _remeasure(
    anchor_table: Path,
    scan_root: Path,
    point: Mapping[str, object],
) -> dict[tuple[object, ...], object]:
    tracklets, propagations = _assets(scan_root, point)
    bank, _audit = read_anchor_selected_field_edge_observations(
        anchor_table,
        tracklets,
        propagations,
        movable_station_ids=(0, 1, 2, 3),
        q_over_p_mode=0,
    )
    return bank


def _recovered_payload(names: Sequence[str], values: Sequence[float], reference_station: int) -> dict[str, list[float]]:
    payload = station_transforms_from_native_values(
        {name: float(value) for name, value in zip(names, values)}
    )
    payload[str(int(reference_station))] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return payload


def close_chart(
    *,
    iteration_manifest: Mapping[str, Any],
    observed_point: str,
    reference_station: int,
    anchor_association: Path,
    tolerances: Mapping[str, float],
    rcond: float,
) -> dict[str, object]:
    plan = iteration_manifest["common_scan_plan"]
    _require_four_station_plan(plan)
    names = require_four_station_route_selected_contract(
        plan,
        only_parameters=relative_free_parameter_names(
            gauge=GAUGE_REFERENCE_STATION, reference_station=reference_station
        ),
        observation_statistics="physical_edge_deduplicated",
    )
    specs = [spec for spec in _specs(plan) if str(spec["name"]) in set(names)]
    specs = [next(spec for spec in specs if str(spec["name"]) == name) for name in names]
    scales = np.asarray([float(spec["severity_scale"]) for spec in specs], dtype=np.float64)
    points = _point_map(plan)
    anchor_name = next(name for name, point in points.items() if point.get("point_role") == "nominal")
    observed = points[observed_point]
    sources = [entry for entry in iteration_manifest["sources"] if str(entry.get("split")) == "train"]
    if any(str(entry.get("split")) == "test" for entry in iteration_manifest["sources"]):
        raise ValueError("refusing to open the sealed test split")
    anchor_table = anchor_association / "selected_route_field_edge_residuals.csv"
    if not anchor_table.is_file():
        raise FileNotFoundError(anchor_table)

    def _pooled(point_name: str):
        return _merge_banks(
            [
                _remeasure(
                    anchor_table,
                    Path(str(entry["physical_scan_root"])).expanduser().resolve(),
                    _point_map(_read_json(Path(str(entry["physical_scan_root"])) / "scan_plan.json"))[point_name],
                )
                for entry in sources
            ]
        )

    banks = [_pooled(anchor_name)]
    positive_names = []
    negative_names = []
    for name in names:
        positive = _probe_for_parameter(points, name, "positive", anchor_name)
        negative = _probe_for_parameter(points, name, "negative", anchor_name)
        positive_names.append(positive)
        negative_names.append(negative)
        banks.append(_pooled(positive))
        banks.append(_pooled(negative))
    banks.append(_pooled(str(observed["name"])))
    keys, residuals, covariances, overlap = align_route_selected_observations(banks)
    physical_keys = [banks[0][key].physical_edge_key for key in keys]
    keys, residuals, covariances, stats = apply_observation_statistics(
        keys, residuals, covariances, physical_keys, "physical_edge_deduplicated"
    )
    fit = solve_physical_finite_difference(
        residuals[0],
        np.asarray([residuals[1 + 2 * index] for index in range(len(names))]),
        np.asarray([residuals[2 + 2 * index] for index in range(len(names))]),
        residuals[-1],
        covariances[0],
        parameter_names=names,
        positive_values=[
            parameter_values_from_station_transforms(specs, points[point]["injected_station_transforms"])[name]
            for name, point in zip(names, positive_names)
        ],
        negative_values=[
            parameter_values_from_station_transforms(specs, points[point]["injected_station_transforms"])[name]
            for name, point in zip(names, negative_names)
        ],
        parameter_scales=scales,
        rcond=rcond,
    )
    recovered = _recovered_payload(fit.parameter_names, fit.recovered_parameters, reference_station)
    injected = observed["injected_station_transforms"]
    capture = capture_relative_tables(injected, recovered, tolerances)
    anchor_chi2 = float(fit.response_chi2)  # post-fit response chi2
    return {
        "gauge": GAUGE_REFERENCE_STATION,
        "reference_station": int(reference_station),
        "parameter_names": list(names),
        "used_observations": int(fit.used_pairs),
        "unique_physical_edges": stats.get("unique_physical_edges"),
        "observation_statistics": stats,
        "truth_free_overlap": overlap,
        "normal_matrix_rank": int(fit.normal_matrix_rank),
        "full_rank": bool(fit.full_rank),
        "normal_matrix_condition_number": fit.normal_matrix_condition_number,
        "response_chi2": float(fit.response_chi2),
        "response_ndof": int(fit.response_ndof),
        "native_recovered": {
            name: float(value) for name, value in zip(fit.parameter_names, fit.recovered_parameters)
        },
        "native_sigma": {
            name: float(np.sqrt(max(float(variance), 0.0)))
            for name, variance in zip(fit.parameter_names, np.diag(fit.covariance_native))
        },
        "recovered_payload": recovered,
        "delta_t_capture": capture,
        "left_se3_preserves_relatives": relatives_agree(
            recovered, apply_left_common_mode_constraint(recovered)
        ),
        "residual_reduction_is_dq_only": True,
        "postfit_response_chi2": float(anchor_chi2),
        "positive_fd_points": positive_names,
        "negative_fd_points": negative_names,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-manifest", required=True)
    parser.add_argument("--observed-point", required=True)
    parser.add_argument("--anchor-association-output", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--operating-point",
        default="configs/physical_refit_four_station_unknown_association.yaml",
    )
    parser.add_argument("--rcond", type=float, default=1.0e-10)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to open the sealed test split")
    operating = yaml.safe_load(Path(args.operating_point).expanduser().resolve().read_text(encoding="utf-8"))
    if bool(operating.get("unconstrained_20d_admitted", False)):
        raise SystemExit("operating point must not admit unconstrained 20-D")
    manifest = _read_json(Path(args.iteration_manifest).expanduser().resolve())
    charts = {}
    recovered = {}
    for raw in operating["gauges_to_compare"]:
        reference = int(raw["reference_station"])
        chart = close_chart(
            iteration_manifest=manifest,
            observed_point=str(args.observed_point),
            reference_station=reference,
            anchor_association=Path(args.anchor_association_output).expanduser().resolve(),
            tolerances=operating["relative_delta_t_capture_tolerance"],
            rcond=float(args.rcond),
        )
        charts[f"reference_station_{reference}"] = chart
        recovered[reference] = chart["recovered_payload"]
    pairwise = capture_relative_tables(
        recovered[0], recovered[3], operating["relative_delta_t_capture_tolerance"]
    )["success"]
    capture_success = all(chart["delta_t_capture"]["success"] for chart in charts.values()) and pairwise
    rank_ok = all(chart["normal_matrix_rank"] == 15 and chart["full_rank"] for chart in charts.values())
    cond_ok = all(
        float(chart["normal_matrix_condition_number"]) <= float(operating["max_scaled_condition"])
        for chart in charts.values()
    )
    report = {
        "schema_version": "faser-four-station-unknown-association-relative-closure-v1",
        "alignment_formulation": FORMULATION,
        "association": "frozen_v2_anchor_selected_physical_edge_deduplicated",
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "q_over_p_mode": 0,
        "overlay": False,
        "observed_point": str(args.observed_point),
        "charts": charts,
        "s0_s3_recovered_relatives_agree": bool(pairwise),
        "primary_delta_t_capture_success": bool(capture_success and rank_ok and cond_ok),
        "rank_and_condition_ok": bool(rank_ok and cond_ok),
        "residual_reduction_is_dq_only": True,
        "test_data_accessed": False,
        "transformer_trained": False,
        "thresholds_retuned": False,
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_ready(report), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(output),
                "observed_point": report["observed_point"],
                "primary_delta_t_capture_success": report["primary_delta_t_capture_success"],
                "s0_s3_recovered_relatives_agree": report["s0_s3_recovered_relatives_agree"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
