#!/usr/bin/env python3
"""Combine Workbook-81b fold evaluations and the two-fold verdict.  JSON only."""

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
    "selected_fragment_routes",
    "n_scored_events",
    "n_enumerated_routes",
    "metric_version",
)
AUDIT_KEYS = (
    "recovered_from_T_neg",
    "lost_from_T_near",
    "new_complete_fake",
    "a_selected_truth_lost",
    "a_selected_fake_retained",
    "a_selected_fake_removed",
    "t_neg_recall_A",
    "t_neg_recall_D",
    "t_near_preservation_A",
    "t_near_preservation_D",
    "selected_route_length",
    "delta_u_truth_mean",
    "delta_u_fake_mean",
    "fraction_abs_delta_near_0_25",
    "inclusion_gap_A_mean",
    "inclusion_gap_D_mean",
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
        default="outputs/mc24_four_station_wb81b_calibration_v1",
    )
    args = parser.parse_args()
    root = Path(args.root)
    folds = {}
    acceptable = []
    no_increment = []
    no_admission = []
    identity = []
    for fold in FOLDS:
        evaluation = root / fold / "evaluation.json"
        metadata = root / fold / "run_metadata.json"
        audit = root / fold / "stratified_audit.json"
        decision = root / fold / "decision.json"
        identity_path = root / fold / "identity.json"
        history = root / fold / "training_history.json"
        if not evaluation.is_file():
            raise SystemExit(f"missing {evaluation}")
        payload = _load(evaluation)
        decision_payload = _load(decision) if decision.is_file() else {}
        audit_payload = _load(audit) if audit.is_file() else {}
        folds[fold] = {
            "holdout_family": payload.get("holdout_family"),
            "git_sha": _load(metadata).get("git_sha") if metadata.is_file() else None,
            "identity_passed": (_load(identity_path).get("passed") if identity_path.is_file() else None),
            "holdout": {arm: _arm_row(payload.get(arm) or {}) for arm in ("arm_a", "arm_d")},
            "delta_complete_efficiency_d_minus_a": payload.get("delta_complete_efficiency_d_minus_a"),
            "delta_complete_fake_rate_d_minus_a": payload.get("delta_complete_fake_rate_d_minus_a"),
            "audit": {key: audit_payload.get(key) for key in AUDIT_KEYS},
            "decision": decision_payload,
            "selected_epoch": payload.get("selected_epoch"),
            "admitted": payload.get("admitted"),
            "admitted_epochs": (_load(history).get("admitted_epochs") if history.is_file() else None),
            "w64_saw_all_six_train_sources": payload.get("w64_saw_all_six_train_sources"),
            "gate_pass": None,
        }
        acceptable.append(bool(decision_payload.get("calibration_acceptable")))
        no_increment.append(bool(decision_payload.get("calibration_has_no_accepted_increment")))
        no_admission.append(bool(decision_payload.get("calibration_unacceptable_under_zero_fake_slack")))
        identity.append(bool(folds[fold]["identity_passed"]))
    supported = bool(acceptable and all(acceptable))
    summary = {
        "workbook": "81b",
        "experiment": "wb81b_calibration_v1",
        "utility_contract": "raw_energy_v1",
        "metric_version": "route_accounting_v2",
        "model_contract": "physics_constrained_calibration_d_v1",
        "not_wp4_arm_c": True,
        "not_wb80_hybrid": True,
        "v5a_retrained": False,
        "development_00350_used": False,
        "final_blind_accessed": False,
        "sealed_test_accessed": False,
        "delta_max": 0.25,
        "fake_slack": 0.0,
        "accept_eff_gain": 0.01,
        "identity_passed": bool(identity and all(identity)),
        "calibration_acceptable": supported,
        "physics_constrained_calibration_supported": supported,
        "calibration_has_no_accepted_increment": bool(
            identity and all(identity) and not supported and not any(no_admission)
        ),
        "calibration_unacceptable_under_zero_fake_slack": bool(no_admission and all(no_admission)),
        "fold_decisions": {
            fold: (folds[fold]["decision"] or {}).get("decision") for fold in FOLDS
        },
        "default_system": (
            "physics_constrained_calibration_d_v1" if supported else "frozen_W64_raw_energy_plus_exact_solver"
        ),
        "folds": folds,
        "gate_pass": None,
    }
    output = root / "summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
