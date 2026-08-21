#!/usr/bin/env python3
"""Freeze C_dx/C_rx dual capture criteria from 3-source train route-selected 1-D solves.

Validation and sealed test are never opened.  The resulting JSON is the
pre-registered gate for the first C_dx+C_rx mini-curriculum and must be
written before that curriculum produces any 10/8 closure numbers.

C_dx is taken from the already-closed sum_to_zero relative-dx update by the
canonical map C=(layer0-layer2)/2.  C_rx is taken from the already-closed
outer_contrast relative-rx update.  Neither 1-D result is retuned.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from alignment.capture_criteria import (
    DEFAULT_COVERAGE_K,
    DEFAULT_STATISTICAL_K,
    evaluate_parameter_capture,
    overall_framework_capture,
)
from alignment.layer_contrast_capture import DEFAULT_ENGINEERING, FREE_PARAMETERS, SCHEMA_VERSION


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object at {path}")
    return dict(payload)


def _parameter_map(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("parameters")
    if not isinstance(rows, list):
        raise ValueError("route-selected update lacks parameter rows")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or "name" not in row:
            raise ValueError("route-selected parameter row is invalid")
        result[str(row["name"])] = dict(row)
    return result


def _unique_edges(payload: Mapping[str, Any]) -> int:
    stats = payload.get("observation_statistics")
    if not isinstance(stats, Mapping):
        raise ValueError("route-selected update lacks observation_statistics")
    return int(stats["unique_physical_edges"])


def _covariance_matrix(payload: Mapping[str, Any], names: tuple[str, ...]) -> list[list[float]]:
    raw = payload.get("parameter_covariance")
    if not isinstance(raw, list) or len(raw) != len(names):
        raise ValueError("route-selected update lacks a matching parameter_covariance")
    matrix = [list(row) for row in raw]
    if any(len(row) != len(names) for row in matrix):
        raise ValueError("parameter_covariance is not square")
    return matrix


def _c_dx_from_sum_to_zero(payload: Mapping[str, Any]) -> dict[str, Any]:
    parameters = _parameter_map(payload)
    names = ("ift_layer0_dx_mm", "ift_layer1_dx_mm", "ift_layer2_dx_mm")
    missing = [name for name in names if name not in parameters]
    if missing:
        raise ValueError("relative-dx update is missing " + ", ".join(missing))
    if payload.get("gauge") != "sum_to_zero":
        raise ValueError("C_dx registration expects the existing sum_to_zero relative-dx update")
    covariance = _covariance_matrix(payload, names)
    var_c = 0.25 * (covariance[0][0] + covariance[2][2] - 2.0 * covariance[0][2])
    if not math.isfinite(var_c) or var_c <= 0.0:
        raise ValueError("transformed C_dx variance is not positive")
    recovered = 0.5 * (
        float(parameters["ift_layer0_dx_mm"]["proposed_next_value"])
        - float(parameters["ift_layer2_dx_mm"]["proposed_next_value"])
    )
    expected = 0.5 * (
        float(parameters["ift_layer0_dx_mm"]["target_value"])
        - float(parameters["ift_layer2_dx_mm"]["target_value"])
    )
    return {
        "name": "C_dx",
        "registered_sigma": math.sqrt(var_c),
        "registration_error": recovered - expected,
        "expected": expected,
        "recovered": recovered,
        "unique_physical_edges": _unique_edges(payload),
        "source": "sum_to_zero_layer_covariance_map_C=(L0-L2)/2",
        "layer1_proposed": float(parameters["ift_layer1_dx_mm"]["proposed_next_value"]),
    }


def _c_rx_from_outer_contrast(payload: Mapping[str, Any]) -> dict[str, Any]:
    parameters = _parameter_map(payload)
    if payload.get("gauge") != "outer_contrast":
        raise ValueError("C_rx registration expects the existing outer_contrast relative-rx update")
    if "ift_layer0_rx_mrad" not in parameters:
        raise ValueError("relative-rx update lacks layer0 rx")
    audit = payload.get("hierarchy_internal_audit")
    if not isinstance(audit, Mapping) or "C_rx" not in list(audit.get("reduced_parameter_names") or []):
        raise ValueError("relative-rx update is not an outer_contrast C_rx solve")
    recovered = float(audit["recovered"]["layer_internal"]["layer_0"])
    expected = float(audit["expected"]["layer_internal"]["layer_0"])
    sigma = parameters["ift_layer0_rx_mrad"]["recovered_sigma"]
    if sigma is None or float(sigma) <= 0.0:
        raise ValueError("relative-rx update has no usable C_rx sigma")
    return {
        "name": "C_rx",
        "registered_sigma": float(sigma),
        "registration_error": recovered - expected,
        "expected": expected,
        "recovered": recovered,
        "unique_physical_edges": _unique_edges(payload),
        "source": "outer_contrast_route_selected_layer0_equals_C_rx",
        "layer1_proposed": float(parameters["ift_layer1_rx_mrad"]["proposed_next_value"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dx-closure", required=True, help="Existing 3-source train relative-dx route-selected JSON")
    parser.add_argument("--rx-closure", required=True, help="Existing 3-source train relative-rx contrast JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--statistical-k", type=float, default=DEFAULT_STATISTICAL_K)
    parser.add_argument("--coverage-k", type=float, default=DEFAULT_COVERAGE_K)
    args = parser.parse_args()
    dx_path = Path(args.dx_closure).expanduser().resolve()
    rx_path = Path(args.rx_closure).expanduser().resolve()
    dx = _read_json(dx_path)
    rx = _read_json(rx_path)
    for payload, label in ((dx, "dx"), (rx, "rx")):
        if payload.get("test_opened") is not False:
            raise ValueError(f"{label} closure accessed test data")
        if int(payload.get("q_over_p_mode", -1)) != 0:
            raise ValueError(f"{label} closure is not mode-0")
    c_dx = _c_dx_from_sum_to_zero(dx)
    c_rx = _c_rx_from_outer_contrast(rx)
    rows = {"C_dx": c_dx, "C_rx": c_rx}

    per_parameter: dict[str, Any] = {}
    for name in FREE_PARAMETERS:
        row = rows[name]
        per_parameter[name] = {
            "role": "free",
            "engineering_tolerance": DEFAULT_ENGINEERING[name],
            "registered_sigma": float(row["registered_sigma"]),
            "statistical_k": float(args.statistical_k),
            "coverage_k": float(args.coverage_k),
            "registration_error": float(row["registration_error"]),
            "registration_expected": float(row["expected"]),
            "registration_recovered": float(row["recovered"]),
            "registration_unique_physical_edges": int(row["unique_physical_edges"]),
            "registration_source": row["source"],
            "layer1_proposed": float(row["layer1_proposed"]),
        }

    scored = [
        evaluate_parameter_capture(
            name=name,
            error=float(rows[name]["registration_error"]),
            fit_sigma=float(rows[name]["registered_sigma"]),
            criteria={
                "per_parameter": per_parameter,
                "statistical_k": args.statistical_k,
                "coverage_k": args.coverage_k,
            },
        )
        for name in FREE_PARAMETERS
    ]
    registration_score = overall_framework_capture(scored, survey_parameters=[])
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "validation_used_in_registration": False,
        "test_data_accessed": False,
        "canonical_internal_basis": "outer_contrast",
        "free_parameters": list(FREE_PARAMETERS),
        "statistical_k": float(args.statistical_k),
        "coverage_k": float(args.coverage_k),
        "engineering_informational_only": True,
        "framework_gate": "statistical_and_this_fit_coverage",
        "payload_expansion": "L0=(+C_dx,+C_rx), L1=0, L2=(-C_dx,-C_rx); station six-vector identically 0",
        "inputs": {
            "C_dx": str(dx_path),
            "C_rx": str(rx_path),
        },
        "per_parameter": per_parameter,
        "registration_score": registration_score,
        "note": (
            "Registered from the already-closed 3-source train route-selected 1-D solves. "
            "Do not retune C_dx. Do not open validation to choose these tolerances. "
            "If the 2-D joint curriculum degenerates or validation fails, stop at the two 1-D results."
        ),
    }
    output.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **registration_score, "per_parameter": per_parameter}, indent=2))


if __name__ == "__main__":
    main()
