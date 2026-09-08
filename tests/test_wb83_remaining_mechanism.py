"""WB83 remaining-alignment diagnostics.  Does not change frozen gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from alignment.wb83_qualification import PULL_MEAN_MAX, build_geometry_matrix, freeze_weak_mode, project_scalar
from alignment.wb83_remaining_mechanism import (
    DY_SLOPE_DIRECTION,
    diagnose_replica,
    generate_replica,
)


def test_dy_slope_is_partner_not_injected_weak_mode():
    weak = freeze_weak_mode()
    names = list(weak["parameter_names"])
    injected = np.asarray([weak["weak_direction"][name] for name in names], dtype=np.float64)
    partner = np.asarray([DY_SLOPE_DIRECTION.get(name, 0.0) for name in names], dtype=np.float64)
    injected = injected / np.linalg.norm(injected)
    partner = partner / np.linalg.norm(partner)
    assert abs(float(injected @ partner)) < 0.01
    assert PULL_MEAN_MAX == 0.2


def test_official_c3_half_replica_zero_is_reproduced():
    matrix = build_geometry_matrix(freeze_weak_mode())
    cell = next(item for item in matrix["cells"] if item["cell_id"] == "C3_weak_0.5x_fixed_dz")
    path = Path("outputs/mc24_four_station_wb83_truth_only_replicas_v2/replicas/C3_weak_0.5x_fixed_dz/0000_0100.jsonl")
    official = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    row = diagnose_replica(cell, 0)
    assert row["not_a_qualification_row"] is True
    assert row["seed"] == official["seed"]
    # Identity-linearized σ matches 1109806.  Iterate residual may differ at
    # ~0.01 mm because the official DAG still added 1e-12 I after symmetrize.
    assert row["identity_sigma"] == pytest.approx(official["sigma"], rel=0.0, abs=1.0e-8)
    assert row["first_step_residual"] == pytest.approx(row["iterate_residual"], rel=0.0, abs=1.0e-6)
    assert abs(row["iterate_residual"] - official["residual_coefficient"]) < 0.05
    assert abs(row["iterate_pull_identity_sigma"] - official["pull"]) < 0.01


def test_first_step_pull_is_amplitude_independent_on_one_seed():
    matrix = build_geometry_matrix(freeze_weak_mode())
    cell = next(item for item in matrix["cells"] if item["cell_id"] == "C3_weak_0.5x_fixed_dz")
    zero = diagnose_replica(cell, 0, inject_amplitude=0.0, iterate=False)
    half = diagnose_replica(cell, 0, inject_amplitude=0.5, iterate=False)
    full = diagnose_replica(cell, 0, inject_amplitude=1.0, iterate=False)
    assert zero["first_step_pull"] == pytest.approx(half["first_step_pull"], abs=1.0e-8)
    assert half["first_step_pull"] == pytest.approx(full["first_step_pull"], abs=1.0e-8)


def test_amplitude_swap_keeps_official_seed():
    matrix = build_geometry_matrix(freeze_weak_mode())
    cell = next(item for item in matrix["cells"] if item["cell_id"] == "C3_weak_0.5x_fixed_dz")
    official = generate_replica(cell, 3)
    swapped = generate_replica(cell, 3, inject_amplitude=1.0)
    assert official["seed"] == swapped["seed"]
    assert official["amplitude_injected"] == 0.5
    assert swapped["amplitude_injected"] == 1.0
    assert official["true_coefficient"] == pytest.approx(0.25)
    assert swapped["true_coefficient"] == pytest.approx(0.5)
    assert np.allclose(official["states"][0], swapped["states"][0])
