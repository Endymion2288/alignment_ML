#!/usr/bin/env python3
"""T02 bootstrap/numerical-contract regression.  Synthetic fixtures only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np

from alignment.cad_survey_nov22 import git_head_sha
from alignment.identifiable_subspace import FROZEN_RANK_TOLERANCE, identifiable_svd
from alignment.numerical_contract import is_spd
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment.resampling import (
    describe_event_draw,
    event_identity_keys,
    group_rows_by_event,
    rows_for_event_draw,
)
from alignment.tracker_only_identifiable_subspace import subspace_from_physical_bank
from evaluation.artifact_store import ImmutableArtifactStore


def _synthetic_derivative():
    derivative = np.zeros((6, 4, 3), dtype=np.float64)
    derivative[:, 0, 0] = 1.0
    derivative[:, 1, 1] = 0.5
    derivative[:, 0, 2] = 1.0
    return derivative


def _synthetic_bank(derivative, *, source_id="s0", scales=(5.0, 60.0, 0.12)):
    pairs, _dim, parameters = derivative.shape
    covariance = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], pairs, axis=0)
    nominal = np.zeros((pairs, 4), dtype=np.float64)
    step = 1.0
    positive = np.zeros((parameters, pairs, 4), dtype=np.float64)
    negative = np.zeros((parameters, pairs, 4), dtype=np.float64)
    for index in range(parameters):
        positive[index] = nominal + step * derivative[:, :, index]
        negative[index] = nominal - step * derivative[:, :, index]
    names = ("ift_dx_mm", "ift_ry_mrad", "C_dx")[:parameters]
    return {
        "source_id": source_id,
        "split": "train",
        "names": names,
        "scales": np.asarray(scales[:parameters], dtype=np.float64),
        "anchor_residual": nominal,
        "positive_residual": positive,
        "negative_residual": negative,
        "reference_residual": np.array(nominal, copy=True),
        "covariance": covariance,
        "run_id": np.arange(pairs, dtype=np.int64),
        "event_id": np.arange(pairs, dtype=np.int64),
        "positive_values": np.full(parameters, step, dtype=np.float64),
        "negative_values": np.full(parameters, -step, dtype=np.float64),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/bootstrap_numerical_contract_v1.yaml",
    )
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), "outputs/bootstrap_numerical_contract_v1"),
        "t02_bootstrap_contract",
    )
    derivative = _synthetic_derivative()
    bank = _synthetic_bank(derivative, source_id="srcA")
    bank["run_id"] = np.array([1, 1, 1, 1, 2, 2], dtype=np.int64)
    bank["event_id"] = np.array([10, 10, 10, 10, 20, 20], dtype=np.int64)
    groups = group_rows_by_event(event_identity_keys(bank))
    event0 = ("srcA", 1, 10)
    event1 = ("srcA", 2, 20)
    drawn = (event0, event0, event1)
    rows = rows_for_event_draw(groups, drawn)
    a0 = np.asarray(
        subspace_from_physical_bank(
            bank, pair_indices=np.asarray(groups[event0], dtype=np.int64)
        )[1]["weighted_matrix"]
    )
    a1 = np.asarray(
        subspace_from_physical_bank(
            bank, pair_indices=np.asarray(groups[event1], dtype=np.int64)
        )[1]["weighted_matrix"]
    )
    a_draw = np.asarray(
        subspace_from_physical_bank(bank, pair_indices=rows)[1]["weighted_matrix"]
    )
    multiplicity_ok = bool(
        np.allclose(a_draw.T @ a_draw, 2.0 * (a0.T @ a0) + (a1.T @ a1), atol=1.0e-12)
    )
    wide = identifiable_svd(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        parameter_names=("ift_dx_mm", "ift_dy_mm", "ift_dz_mm"),
        parameter_units=("mm", "mm", "mm"),
        parameter_scales=(5.0, 5.0, 5.0),
    )
    info = describe_event_draw(groups, drawn)
    payload = {
        "kind": "bootstrap_semantics_regression",
        "task": "T02",
        "config_sha256": sha256_file(config_path),
        "git_head_sha": git_head_sha(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rank_tolerance": float(FROZEN_RANK_TOLERANCE),
        "boolean_mask_used": False,
        "forced_draw": info.as_json(),
        "normal_equals_2n0_plus_n1": multiplicity_ok,
        "wide_null_dimension": int(wide.null_dimension),
        "wide_v_null_columns": int(wide.v_null.shape[1]),
        "non_spd_rejected": not is_spd([[1.0, 2.0], [2.0, 1.0]]),
        "historical_rank_not_reinterpreted": True,
        "geometry_write_allowed": False,
        "held_out_accessed": False,
    }
    store.write_json("bootstrap_semantics_regression.json", payload)
    verdict = (
        "PASS"
        if multiplicity_ok and wide.null_dimension == 1 and not is_spd([[1.0, 2.0], [2.0, 1.0]])
        else "FAIL"
    )
    store.finalize({"verdict": verdict, "task": "T02"})
    print(f"T02 verdict={verdict} run_id={store.run_id}")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
