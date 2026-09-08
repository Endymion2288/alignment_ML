#!/usr/bin/env python3
"""Combine Workbook-79 fold evaluations into one A vs B table.  JSON only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

FOLDS = ("holdout_family1", "holdout_family2")
METRICS = (
    "complete_track_efficiency",
    "complete_fake_rate",
    "complete_track_purity",
    "all_route_purity",
    "all_route_fake_rate",
    "fragmentation_rate",
    "unmatched_truth_chain_rate",
    "complete_truth_chains",
    "selected_complete_routes",
    "correct_complete_routes",
    "fake_complete_routes",
    "n_scored_events",
    "n_enumerated_routes",
    "metric_version",
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SystemExit(f"expected JSON object: {path}")
    return dict(payload)


def _arm_row(arm: Mapping[str, Any]) -> dict[str, Any]:
    return {key: arm.get(key) for key in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="outputs/mc24_four_station_explicit_route_energy_v1",
    )
    args = parser.parse_args()
    root = Path(args.root)
    folds = {}
    for fold in FOLDS:
        evaluation = root / fold / "evaluation.json"
        metadata = root / fold / "run_metadata.json"
        if not evaluation.is_file():
            raise SystemExit(f"missing {evaluation}")
        payload = _load(evaluation)
        meta = _load(metadata) if metadata.is_file() else {}
        folds[fold] = {
            "holdout_family": payload.get("holdout_family"),
            "git_sha": meta.get("git_sha"),
            "n_train_graphs": meta.get("n_train_graphs"),
            "n_val_graphs": meta.get("n_val_graphs"),
            "n_holdout_graphs": meta.get("n_holdout_graphs"),
            "arm_a": _arm_row(payload.get("arm_a") or {}),
            "arm_b": _arm_row(payload.get("arm_b") or {}),
            "delta_complete_efficiency_b_minus_a": payload.get("delta_complete_efficiency_b_minus_a"),
            "delta_complete_fake_rate_b_minus_a": payload.get("delta_complete_fake_rate_b_minus_a"),
            "delta_all_route_purity_b_minus_a": payload.get("delta_all_route_purity_b_minus_a"),
            "w64_saw_all_six_train_sources": payload.get("w64_saw_all_six_train_sources"),
        }
    summary = {
        "workbook": 79,
        "experiment": "explicit_route_energy_v1",
        "utility_contract": "raw_energy_v1",
        "metric_version": "route_accounting_v2",
        "arm_c_authorized": False,
        "development_00350_used": False,
        "final_blind_accessed": False,
        "sealed_test_accessed": False,
        "folds": folds,
    }
    output = root / "summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
