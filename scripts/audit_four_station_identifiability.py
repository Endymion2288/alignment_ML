#!/usr/bin/env python3
"""Four-station identifiability audit from a physical FD bank.

Reads a prepared four-station iteration (nominal + central FD probes on every
S0--S3 free and survey coordinate) and builds the truth-selected Jacobian
from the real /Tracker/Align -> SegmentFitRefit -> Acts(mode 0) responses.
No association model and no sealed test source are involved.

The 20-D free space is not assumed full rank.  The report records parameter
sensitivity, station-pair residual response, scaled SVD, common/relative
mode labels, and gauge-invariant ``ΔT_ij`` agreement between a reference-
station chart and the additive common-mode chart of any held-out payload.
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
    GAUGE_UNCONSTRAINED_FULL,
    STATION_IDS,
    apply_additive_common_mode_constraint,
    apply_left_common_mode_constraint,
    apply_reference_station_gauge,
    classify_scaled_singular_vector,
    relative_alignment_table,
    relatives_agree,
    station_pair_ids,
)
from scripts.audit_6dof_identifiability import _fit, _pooled_bank, _source_bank, _source_entries
from scripts.run_refit_multidof_closure import _point_map, _read_json


SCHEMA_VERSION = "faser-four-station-identifiability-v1"
RESIDUAL_LABELS = ("rx_mm", "ry_mm", "rtx", "rty")


def _require_four_station_plan(plan: Mapping[str, object]) -> None:
    if plan.get("scan_mode") != "station_rigid_multidof":
        raise ValueError("scan is not station_rigid_multidof")
    if str(plan.get("alignment_formulation", "")) != FORMULATION:
        raise ValueError("scan is not a four_station_v1 formulation")
    if str(plan.get("gauge", "")) != GAUGE_UNCONSTRAINED_FULL:
        raise ValueError("identifiability audit expects the unconstrained_full FD chart")
    if tuple(int(station) for station in plan.get("movable_station_ids", ())) != STATION_IDS:
        raise ValueError("four-station audit requires all four stations to be movable")
    if plan.get("reference_station_ids"):
        raise ValueError("unconstrained_full scan must not list a reference station")


def _pair_masks(source_station: np.ndarray, target_station: np.ndarray) -> dict[str, np.ndarray]:
    masks = {
        f"{source}_{target}": (source_station == source) & (target_station == target)
        for source, target in station_pair_ids()
    }
    masks["all"] = np.ones(source_station.shape, dtype=bool)
    return masks


def _column_sensitivity(
    derivative: np.ndarray,
    names: Sequence[str],
    source_station: np.ndarray,
    target_station: np.ndarray,
) -> dict[str, object]:
    masks = _pair_masks(source_station, target_station)
    rows: dict[str, object] = {}
    for index, name in enumerate(names):
        column = derivative[:, :, index]
        per_pair = {}
        for pair, mask in masks.items():
            if not np.any(mask):
                per_pair[pair] = None
                continue
            rms = np.sqrt(np.mean(np.square(column[mask]), axis=0))
            per_pair[pair] = {
                label: float(rms[residual]) for residual, label in enumerate(RESIDUAL_LABELS)
            }
            per_pair[pair]["weighted_norm"] = float(np.linalg.norm(rms))
        rows[str(name)] = per_pair
    return rows


def _fit_report(fit, names: Sequence[str]) -> dict[str, object]:
    vectors = np.linalg.svd(fit.normal_matrix_scaled, full_matrices=False)[2]
    modes = [
        {
            "singular_value": float(value),
            **classify_scaled_singular_vector(names, vectors[index]),
        }
        for index, value in enumerate(fit.data_singular_values)
    ]
    return {
        "parameter_names": list(names),
        "normal_matrix_rank": int(fit.normal_matrix_rank),
        "full_rank": bool(fit.full_rank),
        "normal_matrix_condition_number": fit.normal_matrix_condition_number,
        "identifiable_subspace_condition_number": fit.identifiable_subspace_condition_number,
        "data_singular_values": [float(value) for value in fit.data_singular_values],
        "parameter_observability_fraction": [float(value) for value in fit.parameter_observability_fraction],
        "native_sigma": [float(math.sqrt(max(value, 0.0))) for value in np.diag(fit.covariance_native)],
        "correlation_native": np.asarray(fit.correlation_native, dtype=np.float64).tolist(),
        "used_pairs": int(fit.used_pairs),
        "response_chi2": float(fit.response_chi2),
        "singular_vectors": modes,
    }


def _gauge_report(transforms: Mapping[int | str, Sequence[float]]) -> dict[str, object]:
    packed = {int(station): list(values) for station, values in transforms.items()}
    by_s0 = apply_reference_station_gauge(packed, reference_station=0)
    by_s3 = apply_reference_station_gauge(packed, reference_station=3)
    common = apply_left_common_mode_constraint(packed)
    additive = apply_additive_common_mode_constraint(packed)
    return {
        "payload_relative_delta_t_ij": relative_alignment_table(packed),
        "reference_station_s0_relatives": relative_alignment_table(by_s0),
        "reference_station_s3_relatives": relative_alignment_table(by_s3),
        "left_se3_common_mode_relatives": relative_alignment_table(common),
        "additive_mean_relatives": relative_alignment_table(additive),
        "s0_chart_agrees_with_payload": relatives_agree(packed, by_s0),
        "s3_chart_agrees_with_payload": relatives_agree(packed, by_s3),
        "s0_and_s3_charts_agree": relatives_agree(by_s0, by_s3),
        "left_se3_common_mode_agrees_with_payload": relatives_agree(packed, common),
        "additive_mean_is_not_automatically_a_gauge": not relatives_agree(packed, additive, atol=1.0e-8),
    }


def audit_manifest(
    manifest_path: Path,
    *,
    split: str,
    min_truth_match_fraction: float,
) -> dict[str, object]:
    manifest = _read_json(manifest_path)
    sources = [entry for entry in _source_entries(manifest) if str(entry.get("split")) == split]
    if not sources:
        raise ValueError(f"iteration manifest has no '{split}' sources")
    common_plan = manifest.get("common_scan_plan")
    if not isinstance(common_plan, Mapping):
        raise ValueError("iteration manifest lacks common_scan_plan")
    _require_four_station_plan(common_plan)
    points = _point_map(common_plan)
    nominal = [point for point in points.values() if point.get("point_role") == "nominal"]
    if len(nominal) != 1:
        raise ValueError("four-station scan must have exactly one nominal point")
    anchor_name = str(nominal[0]["name"])
    held_out = [name for name, point in points.items() if point.get("point_role") == "held_out_closure"]
    target_name = held_out[0] if held_out else anchor_name
    banks = [
        _source_bank(
            entry,
            anchor_point=anchor_name,
            target_point=target_name,
            min_truth_match_fraction=min_truth_match_fraction,
        )
        for entry in sources
    ]
    pooled = _pooled_bank(banks)
    fit = _fit(pooled, rcond=1.0e-10)
    names = tuple(pooled["names"])
    gauge_rows = {
        name: _gauge_report(points[name]["injected_station_transforms"]) for name in held_out
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "alignment_formulation": FORMULATION,
        "gauge": GAUGE_UNCONSTRAINED_FULL,
        "split": split,
        "n_sources": len(sources),
        "source_ids": [str(entry["source_id"]) for entry in sources],
        "used_pairs": int(fit.used_pairs),
        "parameter_sensitivity": _column_sensitivity(
            fit.derivative_native,
            names,
            np.asarray(pooled["source_station_id"]),
            np.asarray(pooled["target_station_id"]),
        ),
        "pooled_fit": _fit_report(fit, names),
        "held_out_gauge_invariants": gauge_rows,
        "do_not_admit_20d_without_svd": True,
        "test_data_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--min-truth-match-fraction", type=float, default=0.99)
    args = parser.parse_args()
    if args.split == "test":
        raise SystemExit("refusing to open the sealed test split")
    report = audit_manifest(
        Path(args.iteration_manifest).expanduser().resolve(),
        split=str(args.split),
        min_truth_match_fraction=float(args.min_truth_match_fraction),
    )
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(output),
                "used_pairs": report["used_pairs"],
                "rank": report["pooled_fit"]["normal_matrix_rank"],
                "condition": report["pooled_fit"]["normal_matrix_condition_number"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
