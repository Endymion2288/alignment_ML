#!/usr/bin/env python3
"""Workbook 82 phase-1: truth-only single-condition common-track smoke.

Does not use W64 association.  Does not open 00350 / Final Blind / sealed.
Does not claim an alignment oracle.  Batch independent replicas stay Condor.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alignment.common_track_geometry import SOLVER_CONTRACT, identity_station_map, left_update_payload
from alignment.common_track_solver import (
    ASSOCIATION_DEFAULT_SYSTEM,
    MAX_ITERATIONS,
    MOMENTUM_PRIOR,
    PHASE1_SMOKE_COMPONENTS,
    PHASE1_SMOKE_STATIONS,
    QP_PRIOR_SIGMA,
    SURVEY_FIXED_DZ,
    StationHit,
    chi2_of_blocks,
    iterate_common_track,
    linearize_track,
    parameter_chart,
    predict_local_measurement,
    refuse_wb82_path,
    relative_lie_table,
    translation_second_difference,
)
from alignment.four_station import IDENTITY_SIX, STATION_IDS


OUTPUT_ROOT = Path("outputs/mc24_four_station_wb82_truth_only_smoke_v1")
Z_MM = {0: 0.0, 1: 1000.0, 2: 2000.0, 3: 3000.0}
N_FIT = 24
N_VAL = 8
SEED = 20260906


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _git_porcelain() -> str:
    return subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout


def _write_json(path: Path, payload: object) -> None:
    refuse_wb82_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _make_state(rng: np.random.Generator, track_id: int) -> np.ndarray:
    return np.asarray(
        [
            rng.normal(scale=2.0),
            rng.normal(scale=2.0),
            rng.normal(scale=0.003),
            rng.normal(scale=0.003),
            0.08 + 0.02 * ((track_id % 5) - 2),
        ],
        dtype=np.float64,
    )


def _hits_for_track(track_id: int, state: np.ndarray, truth_payloads, rng: np.random.Generator) -> list[StationHit]:
    hits = []
    noise = rng.normal(scale=0.02, size=(4, 4))
    for station in STATION_IDS:
        predicted = predict_local_measurement(
            state,
            truth_payloads[station],
            z_ref_mm=Z_MM[0],
            z_station_mm=Z_MM[station],
            field_y=0.35,
        )
        observed = predicted + noise[station]
        hits.append(
            StationHit(
                track_id=track_id,
                station=station,
                measurement_id=f"truth-{track_id}-s{station}",
                observed=observed,
                covariance=np.eye(4) * 0.02**2,
            )
        )
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args()
    output = Path(args.output_root)
    refuse_wb82_path(output)

    rng = np.random.default_rng(SEED)
    start = identity_station_map()
    truth = identity_station_map()
    truth[3] = left_update_payload(IDENTITY_SIX, (0.5, 0.0, 0.0, 0.0, 0.0, 0.0))
    tracks = []
    states = {}
    for track_id in range(N_FIT + N_VAL):
        state = _make_state(rng, track_id)
        states[track_id] = state
        tracks.append(_hits_for_track(track_id, state, truth, rng))
    val_ids = tuple(range(N_FIT, N_FIT + N_VAL))
    names = parameter_chart(
        survey_mode=SURVEY_FIXED_DZ,
        reference_station=0,
        components=PHASE1_SMOKE_COMPONENTS,
        stations=PHASE1_SMOKE_STATIONS,
    )
    before_blocks = [
        linearize_track(tracks[track_id], states[track_id], start, names, z_mm=Z_MM, field_y=0.35) for track_id in val_ids
    ]
    before_val = chi2_of_blocks(before_blocks)
    result = iterate_common_track(
        tracks,
        {key: value.copy() for key, value in states.items()},
        start,
        z_mm=Z_MM,
        field_y=0.35,
        survey_mode=SURVEY_FIXED_DZ,
        reference_station=0,
        validation_track_ids=val_ids,
        max_iterations=MAX_ITERATIONS,
        components=PHASE1_SMOKE_COMPONENTS,
        stations=PHASE1_SMOKE_STATIONS,
    )
    after_blocks = [
        linearize_track(tracks[track_id], result["states"][track_id], result["payloads"], names, z_mm=Z_MM, field_y=0.35)
        for track_id in val_ids
    ]
    after_val = chi2_of_blocks(after_blocks)
    lie = relative_lie_table(result["payloads"], truth)
    s3_dx_error = float(lie["per_station"]["3"][0])
    kink_x = translation_second_difference(lie["per_station"], axis=0)
    kink_y = translation_second_difference(lie["per_station"], axis=1)
    smoke_pass = bool(
        result["converged"]
        and result["automatically_passed_at_max_iterations"] is False
        and after_val < before_val
        and abs(s3_dx_error) < 0.05
        and abs(kink_x) < 0.05
        and abs(kink_y) < 0.05
    )
    payload = {
        "workbook": "82",
        "phase": "truth_only_single_condition_smoke",
        "solver_contract": SOLVER_CONTRACT,
        "association": "truth_only",
        "association_default_system": ASSOCIATION_DEFAULT_SYSTEM,
        "w64_used": False,
        "ml_association_used": False,
        "survey_mode": SURVEY_FIXED_DZ,
        "momentum_mode": MOMENTUM_PRIOR,
        "qp_prior_sigma": QP_PRIOR_SIGMA,
        "newton_components": list(PHASE1_SMOKE_COMPONENTS),
        "newton_stations": list(PHASE1_SMOKE_STATIONS),
        "update_convention": "left_se3",
        "injected": {"s3_dx_mm": 0.5},
        "n_fit_tracks": N_FIT,
        "n_validation_tracks": N_VAL,
        "field_y": 0.35,
        "z_mm": Z_MM,
        "seed": SEED,
        "converged": result["converged"],
        "reason": result["reason"],
        "message": result.get("message"),
        "iterations": result["iterations"],
        "n_retained": None if result.get("solution") is None else getattr(result["solution"], "n_retained", None),
        "n_dropped": None if result.get("solution") is None else getattr(result["solution"], "n_dropped", None),
        "automatically_passed_at_max_iterations": False,
        "validation_chi2_before": before_val,
        "validation_chi2_after": after_val,
        "s3_dx_lie_error_mm": s3_dx_error,
        "identifiable_x_kink_error_mm": kink_x,
        "identifiable_y_kink_error_mm": kink_y,
        "raw_s3_dx_within_0p05": bool(abs(s3_dx_error) < 0.05),
        "relative_lie": lie,
        "history": result["history"],
        "smoke_pass": smoke_pass,
        "alignment_oracle_qualified": False,
        "note": "Smoke is not bulk replica coverage.  oracle_qualified stays false until independent replica closure.",
        "git_sha": _git_sha(),
        "git_status_porcelain": _git_porcelain(),
        "hostname": socket.gethostname(),
        "utc": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "continue_to_15d_relative_wls": False,
        "continue_route_energy_rewrite": False,
        "continue_hybrid_expansion": False,
        "continue_residual_calibration": False,
        "final_blind_eval_authorized": False,
        "sealed_test_accessed": False,
        "development_00350_used": False,
    }
    _write_json(output / "smoke.json", payload)
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "smoke_pass",
                    "converged",
                    "s3_dx_lie_error_mm",
                    "identifiable_x_kink_error_mm",
                    "identifiable_y_kink_error_mm",
                    "validation_chi2_before",
                    "validation_chi2_after",
                    "iterations",
                )
            },
            indent=2,
        )
    )
    if not math.isfinite(s3_dx_error):
        raise SystemExit("WB82 smoke produced a non-finite Lie error")


if __name__ == "__main__":
    main()
