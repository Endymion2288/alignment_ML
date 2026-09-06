#!/usr/bin/env python3
"""T12 toy profiled-information pilot.  Physical 18-source run is blocked."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np

from alignment.cad_survey_nov22 import git_head_sha
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment.profiled_information import (
    assert_no_information_inflation,
    solve_joint,
    solve_profiled,
)
from evaluation.artifact_store import ImmutableArtifactStore
from models.field_track_likelihood import FieldLikelihoodError, refuse_physical_run
import yaml


def _toy(seed: int):
    rng = np.random.default_rng(int(seed))
    g = rng.normal(size=(8, 2))
    h = rng.normal(size=(8, 5))
    covariance = np.eye(8) + 0.02 * rng.normal(size=(8, 8))
    covariance = 0.5 * (covariance + covariance.T) + 2.0 * np.eye(8)
    true_a = np.array([0.05, -0.02])
    true_q = rng.normal(size=5) * 0.05
    residual = g @ true_a + h @ true_q
    return g, h, residual, covariance, true_a


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/alignment_information_pilot_v1.yaml")
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    blocked = False
    block_reason = None
    try:
        refuse_physical_run(t11_contract_established=bool(config["t11_contract_established"]))
    except FieldLikelihoodError as error:
        blocked = True
        block_reason = str(error)
    g, h, residual, covariance, true_a = _toy(int(config["toy_seed"]))
    joint_a, _q, joint_normal = solve_joint(g, h, residual, covariance)
    profiled = solve_profiled(g, h, residual, covariance)
    weight = np.linalg.inv(covariance)
    assert_no_information_inflation(g.T @ weight @ g, profiled.normal_alignment)
    bias = profiled.delta_alignment - true_a
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "t12_information_pilot",
    )
    store.write_json(
        "absolute_precision.json",
        {
            "kind": "absolute_precision",
            "population": "synthetic_math_fixture",
            "uses_production_c_prop": False,
            "bias": [float(item) for item in bias],
            "delta": [float(item) for item in profiled.delta_alignment],
            "truth": [float(item) for item in true_a],
            "covariance": profiled.covariance_alignment.tolist(),
            "coverage_not_established": True,
            "physical_18_source_run": False,
        },
    )
    store.write_json(
        "fd_linearity.json",
        {
            "kind": "fd_linearity",
            "note": "toy G/H are linear by construction; no Athena refit",
            "linear_model": True,
        },
    )
    store.write_json(
        "oracle_free_closure.json",
        {
            "kind": "oracle_free_closure",
            "joint_matches_schur": bool(
                np.allclose(profiled.delta_alignment, joint_a, atol=1.0e-8)
            ),
            "reference_residual_used": False,
            "truth_qp_used": False,
        },
    )
    store.write_json(
        "aggregate/legacy_gate_reconciliation.json",
        {
            "kind": "legacy_gate_reconciliation",
            "t10_verdict": "FAIL",
            "t11_verdict": "FAIL",
            "old_operational_rank_unchanged": True,
            "old_wls_still_diagnostic": True,
        },
    )
    resource = {
        "kind": "resource_decision",
        "task": "T12",
        "verdict": "blocked_on_t11",
        "conditional_go": False,
        "unconstrained_tracker_only_stopped": True,
        "physical_18_source_run": False,
        "eighteen_source_jobs_submitted": False,
        "fallback": "real_data_residual_dq_monitoring_plus_reproducible_negative",
        "t11_contract_established": False,
        "block_reason": block_reason,
        "geometry_write_allowed": False,
        "held_out_accessed": False,
        "new_payload_written": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(config_path),
        "git_head_sha": git_head_sha(),
    }
    store.write_json("aggregate/resource_decision.json", resource)
    store.finalize({"verdict": resource["verdict"], "task": "T12", "blocked": blocked})
    print(f"T12 verdict={resource['verdict']} run_id={store.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
