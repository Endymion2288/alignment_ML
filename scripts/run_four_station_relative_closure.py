#!/usr/bin/env python3
"""Truth-selected 15-DoF relative four-station closure on a physical FD bank.

The identifiability production remains unconstrained_full (empty reference,
all four stations probed).  This driver does **not** change that payload
convention.  It drops one station's five free coordinates at solve time,
fits the admitted relative chart, and reports capture only on
``ΔT_ij = T_i^{-1} T_j``.  The historical IFT-only
``run_refit_multidof_closure.py`` still requires a non-empty reference set
and is left unchanged.

No Transformer is involved.  The sealed test split is refused.  Capture
tolerances must be loaded from the pre-registered YAML operating point.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.four_station import (
    FORMULATION,
    GAUGE_REFERENCE_STATION,
    apply_left_common_mode_constraint,
    apply_reference_station_gauge,
    capture_relative_tables,
    relative_free_parameter_names,
    relatives_agree,
    station_transforms_from_native_values,
)
from alignment.physical_jacobian import parameter_values_from_station_transforms
from scripts.audit_6dof_identifiability import _fit, _pooled_bank, _source_bank, _source_entries, _subset_bank
from scripts.audit_four_station_identifiability import _require_four_station_plan
from scripts.run_refit_multidof_closure import _json_ready, _point_map, _read_json


SCHEMA_VERSION = "faser-four-station-relative-closure-v1"


def _load_operating_point(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"operating-point YAML is not a mapping: {path}")
    if payload.get("alignment_formulation") != FORMULATION:
        raise ValueError("operating point is not four_station_v1")
    if bool(payload.get("unconstrained_20d_admitted", False)):
        raise ValueError("operating point must not admit unconstrained 20-D")
    if int(payload.get("n_admitted_free_parameters", -1)) != 15:
        raise ValueError("operating point must admit exactly 15 relative free parameters")
    tolerances = payload.get("relative_delta_t_capture_tolerance")
    if not isinstance(tolerances, Mapping):
        raise ValueError("operating point lacks relative_delta_t_capture_tolerance")
    gauges = payload.get("gauges_to_compare")
    if not isinstance(gauges, list) or not gauges:
        raise ValueError("operating point lacks gauges_to_compare")
    return dict(payload)


def _recovered_payload(names: Sequence[str], recovered: Sequence[float], reference_station: int) -> dict[str, list[float]]:
    payload = station_transforms_from_native_values(
        {name: float(value) for name, value in zip(names, recovered)}
    )
    payload[str(int(reference_station))] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return payload


def close_observed_point(
    manifest_path: Path,
    *,
    observed_point: str,
    split: str,
    operating_point: Mapping[str, Any],
    min_truth_match_fraction: float,
    rcond: float,
) -> dict[str, object]:
    if split == "test" or split in set(operating_point.get("forbidden_splits", ())):
        raise ValueError("refusing to open a forbidden split")
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
    observed = points.get(observed_point)
    if observed is None:
        raise ValueError(f"observed point is not in the frozen scan plan: {observed_point}")
    if str(observed.get("point_role")) in {
        "nominal",
        "finite_difference_positive",
        "finite_difference_negative",
    }:
        raise ValueError("observed point must be an independent held-out closure payload")
    gauges = []
    for raw in operating_point["gauges_to_compare"]:
        if dict(raw).get("gauge") != GAUGE_REFERENCE_STATION:
            raise ValueError("this driver currently compares reference_station charts only")
        gauges.append(int(raw["reference_station"]))
    all_names = relative_free_parameter_names(gauge=GAUGE_REFERENCE_STATION, reference_station=gauges[0])
    for extra in gauges[1:]:
        all_names = tuple(dict.fromkeys(all_names + relative_free_parameter_names(
            gauge=GAUGE_REFERENCE_STATION, reference_station=extra
        )))
    banks = [
        _source_bank(
            entry,
            anchor_point=str(nominal[0]["name"]),
            target_point=str(observed["name"]),
            min_truth_match_fraction=min_truth_match_fraction,
            only_parameters=all_names,
        )
        for entry in sources
    ]
    pooled = _pooled_bank(banks)
    injected = observed["injected_station_transforms"]
    tolerances = operating_point["relative_delta_t_capture_tolerance"]
    charts = {}
    recovered_payloads = {}
    for reference in gauges:
        names = relative_free_parameter_names(
            gauge=GAUGE_REFERENCE_STATION, reference_station=reference
        )
        subset = _subset_bank(pooled, names)
        fit = _fit(subset, rcond=rcond)
        recovered = _recovered_payload(fit.parameter_names, fit.recovered_parameters, reference)
        gauged_injected = apply_reference_station_gauge(injected, reference_station=reference)
        native_injected = {
            name: float(value)
            for name, value in parameter_values_from_station_transforms(
                subset["specs"], gauged_injected
            ).items()
        }
        sigma = [
            float(np.sqrt(max(float(variance), 0.0)))
            for variance in np.diag(fit.covariance_native)
        ]
        charts[f"reference_station_{reference}"] = {
            "gauge": GAUGE_REFERENCE_STATION,
            "reference_station": reference,
            "parameter_names": list(names),
            "used_pairs": int(fit.used_pairs),
            "response_chi2": float(fit.response_chi2),
            "response_ndof": int(fit.response_ndof),
            "normal_matrix_rank": int(fit.normal_matrix_rank),
            "full_rank": bool(fit.full_rank),
            "normal_matrix_condition_number": fit.normal_matrix_condition_number,
            "native_injected_in_chart": native_injected,
            "native_recovered": {
                name: float(value)
                for name, value in zip(fit.parameter_names, fit.recovered_parameters)
            },
            "native_sigma": {name: sig for name, sig in zip(fit.parameter_names, sigma)},
            "native_pull_vs_chart": {
                name: float((rec - native_injected[name]) / sig) if sig > 0.0 else None
                for name, rec, sig in zip(fit.parameter_names, fit.recovered_parameters, sigma)
            },
            "recovered_payload": recovered,
            "gauged_injected_payload": gauged_injected,
            "delta_t_capture": capture_relative_tables(injected, recovered, tolerances),
            "left_se3_preserves_relatives": relatives_agree(
                recovered, apply_left_common_mode_constraint(recovered)
            ),
        }
        recovered_payloads[reference] = recovered
    references = list(recovered_payloads)
    pairwise_agreement = True
    for left, right in zip(references, references[1:]):
        pairwise_agreement = pairwise_agreement and capture_relative_tables(
            recovered_payloads[left],
            recovered_payloads[right],
            tolerances,
        )["success"]
    capture_success = all(chart["delta_t_capture"]["success"] for chart in charts.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "truth_fixed_four_station_relative_finite_difference_closure",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "alignment_formulation": FORMULATION,
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "association": "truth_fixed_for_identifiability_audit",
        "q_over_p_mode": 0,
        "split": split,
        "source_ids": [str(entry["source_id"]) for entry in sources],
        "observed_point": str(observed["name"]),
        "operating_point": dict(operating_point),
        "charts": charts,
        "s0_s3_recovered_relatives_agree": bool(pairwise_agreement),
        "primary_delta_t_capture_success": bool(capture_success and pairwise_agreement),
        "test_data_accessed": False,
        "transformer_trained": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-manifest", required=True)
    parser.add_argument("--observed-point", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--operating-point",
        default="configs/physical_refit_four_station_relative_closure.yaml",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--min-truth-match-fraction", type=float, default=0.99)
    parser.add_argument("--rcond", type=float, default=1.0e-10)
    args = parser.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to open the sealed test split")
    report = close_observed_point(
        Path(args.iteration_manifest).expanduser().resolve(),
        observed_point=str(args.observed_point),
        split=str(args.split),
        operating_point=_load_operating_point(Path(args.operating_point).expanduser().resolve()),
        min_truth_match_fraction=float(args.min_truth_match_fraction),
        rcond=float(args.rcond),
    )
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_ready(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_json": str(output),
                "observed_point": report["observed_point"],
                "primary_delta_t_capture_success": report["primary_delta_t_capture_success"],
                "s0_s3_recovered_relatives_agree": report["s0_s3_recovered_relatives_agree"],
                "used_pairs": {
                    name: chart["used_pairs"] for name, chart in report["charts"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
