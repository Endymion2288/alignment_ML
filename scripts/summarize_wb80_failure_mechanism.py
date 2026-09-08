#!/usr/bin/env python3
"""Combine Workbook-80 fold evaluations and pre-registered readings.  JSON only."""

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
        default="outputs/mc24_four_station_wb80_failure_mechanism_v1",
    )
    args = parser.parse_args()
    root = Path(args.root)
    folds = {}
    for fold in FOLDS:
        evaluation = root / fold / "evaluation.json"
        metadata = root / fold / "run_metadata.json"
        audit = root / fold / "stratified_audit.json"
        reading = root / fold / "interpretation.json"
        if not evaluation.is_file():
            raise SystemExit(f"missing {evaluation}")
        payload = _load(evaluation)
        holdout = payload.get("holdout") or {}
        same_family = payload.get("same_family_val") or {}
        folds[fold] = {
            "holdout_family": payload.get("holdout_family"),
            "git_sha": _load(metadata).get("git_sha") if metadata.is_file() else None,
            "holdout": {arm: _arm_row(holdout.get(arm) or {}) for arm in ("A", "B", "C")},
            "same_family_val": {arm: _arm_row(same_family.get(arm) or {}) for arm in ("A", "B", "C")},
            "delta_holdout": payload.get("delta_holdout"),
            "delta_same_family_val": payload.get("delta_same_family_val"),
            "audit": _load(audit) if audit.is_file() else None,
            "interpretation": _load(reading) if reading.is_file() else None,
            "w64_saw_all_six_train_sources": payload.get("w64_saw_all_six_train_sources"),
        }
    summary = {
        "workbook": 80,
        "experiment": "wb80_failure_mechanism_v1",
        "utility_contract": "raw_energy_v1",
        "metric_version": "route_accounting_v2",
        "wp4_arm_c_authorized": False,
        "v5a_retrained": False,
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
