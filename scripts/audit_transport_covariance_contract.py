#!/usr/bin/env python3
"""T11 transport contract.  Math fixtures + frozen WB87; no Stage B rerun."""

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
from datasets.transport_contract import (
    decide,
    geometric_jacobian,
    inherit_wb87,
    load_config,
    numerical_jacobian,
    q_over_p_per_mev_from_gev,
    signed_q_over_p,
    slope_from_direction,
    transport_covariance,
)
from evaluation.artifact_store import ImmutableArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/transport_covariance_contract_v1.yaml")
    args = parser.parse_args()
    config_path = resolve_under_root(project_root(), args.config)
    config = load_config(config_path)
    inherited = inherit_wb87(config)
    lever = 2000.0
    jac = geometric_jacobian(lever)
    numeric = numerical_jacobian(np.array([1.0, -2.0, 0.01, -0.02]), lever, 1.0e-4)
    fd_rel = float(np.linalg.norm(numeric - jac) / np.linalg.norm(jac))
    cov = np.diag([1.0, 1.0, 1.0e-6, 1.0e-6])
    transported = transport_covariance(cov, jac)
    math_pass = (
        fd_rel <= float(config["math_relative_error"])
        and np.isfinite(transported).all()
        and abs(signed_q_over_p(-1.0, 1.0e5) + 1.0e-5) < 1.0e-15
        and abs(q_over_p_per_mev_from_gev(1.0) - 0.001) < 1.0e-15
        and slope_from_direction(np.array([0.1, 0.0, 1.0]))[0] == 0.1
    )
    decision = decide(config, math_pass, inherited)
    store = ImmutableArtifactStore.begin(
        resolve_under_root(project_root(), str(config["output_root"])),
        "t11_transport_contract",
    )
    store.write_json(
        "coordinate_closure.json",
        {
            "kind": "coordinate_closure",
            "state_names": list(config["state_names"]),
            "q_over_p_native_unit": config["q_over_p_native_unit"],
            "fd_relative_error": fd_rel,
            "math_pass": math_pass,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": sha256_file(config_path),
            "git_head_sha": git_head_sha(),
        },
    )
    store.write_json(
        "transport_fd_summary.json",
        {
            "kind": "transport_fd_summary",
            "physical_transport_fd_rerun": False,
            "eighteen_source_jobs_submitted": False,
            "geometric_fd_relative_error": fd_rel,
            "inherited_wb87": inherited,
        },
    )
    store.write_json("aggregate/decision.json", decision)
    store.finalize({"verdict": decision["verdict"], "task": "T11"})
    print(f"T11 verdict={decision['verdict']} run_id={store.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
