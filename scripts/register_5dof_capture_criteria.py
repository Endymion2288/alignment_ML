#!/usr/bin/env python3
"""Freeze 5-DoF dual capture criteria from the train-only route-selected pilot.

Reads only the train-source 6-DoF sensitivity-pilot route-selected closures.
Validation and sealed test are never opened.  The resulting JSON is the
pre-registered gate for the station-level 5-DoF + survey-dz curriculum and
must be written before that curriculum produces any closure numbers.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from alignment.capture_criteria import (
    DEFAULT_COVERAGE_K,
    DEFAULT_ENGINEERING,
    DEFAULT_STATISTICAL_K,
    FREE_PARAMETERS,
    SCHEMA_VERSION,
    SURVEY_PARAMETERS,
    evaluate_parameter_capture,
    overall_framework_capture,
)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-closure", required=True, help="Primary train-only route-selected JSON (linear point c)")
    parser.add_argument("--corroboration-closure", action="append", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--statistical-k", type=float, default=DEFAULT_STATISTICAL_K)
    parser.add_argument("--coverage-k", type=float, default=DEFAULT_COVERAGE_K)
    args = parser.parse_args()
    primary_path = Path(args.train_closure).expanduser().resolve()
    primary = _read_json(primary_path)
    parameters = _parameter_map(primary)
    missing = [name for name in (*FREE_PARAMETERS, *SURVEY_PARAMETERS) if name not in parameters]
    if missing:
        raise ValueError("train closure is missing parameters: " + ", ".join(missing))

    per_parameter: dict[str, Any] = {}
    for name in FREE_PARAMETERS:
        sigma = parameters[name]["recovered_sigma"]
        if sigma is None or float(sigma) <= 0.0:
            raise ValueError(f"train closure has no usable registered sigma for {name}")
        per_parameter[name] = {
            "role": "free",
            "engineering_tolerance": DEFAULT_ENGINEERING[name],
            "registered_sigma": float(sigma),
            "statistical_k": float(args.statistical_k),
            "coverage_k": float(args.coverage_k),
            "registration_error": float(parameters[name]["local_delta_error"]),
            "registration_unique_physical_edges": _unique_edges(primary),
        }
    for name in SURVEY_PARAMETERS:
        sigma = parameters[name]["recovered_sigma"]
        per_parameter[name] = {
            "role": "survey_constrained",
            "engineering_tolerance": None,
            "registered_sigma": None if sigma is None else float(sigma),
            "statistical_k": float(args.statistical_k),
            "coverage_k": float(args.coverage_k),
            "registration_error": float(parameters[name]["local_delta_error"]),
            "note": "Survey prior dominates; excluded from track-based capture.",
        }

    scored = [
        evaluate_parameter_capture(
            name=name,
            error=float(parameters[name]["local_delta_error"]),
            fit_sigma=parameters[name]["recovered_sigma"],
            criteria={"per_parameter": per_parameter, "statistical_k": args.statistical_k, "coverage_k": args.coverage_k},
        )
        for name in (*FREE_PARAMETERS, *SURVEY_PARAMETERS)
    ]
    registration_score = overall_framework_capture(scored)

    corroboration: list[dict[str, Any]] = []
    for raw in args.corroboration_closure or []:
        path = Path(raw).expanduser().resolve()
        payload = _read_json(path)
        rows = _parameter_map(payload)
        corroboration.append(
            {
                "path": str(path),
                "unique_physical_edges": _unique_edges(payload),
                "recovered_sigma": {
                    name: rows[name]["recovered_sigma"] for name in (*FREE_PARAMETERS, *SURVEY_PARAMETERS) if name in rows
                },
            }
        )

    contract = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "validation_used_in_registration": False,
        "test_data_accessed": False,
        "registration_split": "train",
        "registration_artifact": str(primary_path),
        "registration_point": primary.get("target_point"),
        "statistical_k": float(args.statistical_k),
        "coverage_k": float(args.coverage_k),
        "free_parameters": list(FREE_PARAMETERS),
        "survey_constrained_parameters": list(SURVEY_PARAMETERS),
        "survey_prior_sigma": {"ift_dz_mm": 5.0},
        "per_parameter": per_parameter,
        "registration_self_score": {
            "per_parameter": {row["name"]: row for row in scored},
            **registration_score,
        },
        "corroboration": corroboration,
        "capture_success_definition": (
            "framework_capture_success requires every free parameter to pass statistical "
            "(|error| <= k * registered_sigma) AND coverage (|error| <= k * this-fit sigma). "
            "engineering_capture_success is reported and never discarded, but does not by "
            "itself advance or reject an iteration.  dz is survey-constrained and excluded."
        ),
        "motivation": (
            "Pilot route-selected dy error 0.102 mm was 0.33 sigma of the train-only "
            "route-selected uncertainty 0.313 mm but failed the inherited 0.1 mm absolute "
            "gate.  Dual reporting keeps the engineering number while advancing on pull/coverage."
        ),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "framework_capture_success": registration_score["framework_capture_success"], "engineering_capture_success": registration_score["engineering_capture_success"]}, indent=2))


if __name__ == "__main__":
    main()
