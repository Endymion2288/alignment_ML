"""Workbook 83: frozen criteria, independence, fail-closed gate."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alignment.wb83_qualification import (
    ASSOCIATION_DEFAULT_SYSTEM,
    DESIGN_SEED,
    MIN_REPLICAS_FOR_PASS,
    N_TARGET_REPLICAS,
    SOURCE_ID,
    TRANSLATION_DIRECTION,
    WB83Error,
    Z_MM,
    build_geometry_matrix,
    build_replica_manifest,
    freeze_weak_mode,
    measurement_audit,
    qualify,
    refuse_wb83_path,
    replica_seed,
    summarize_cell,
)
from alignment.common_track_solver import StationHit, iterate_common_track
from alignment.four_station import IDENTITY_SIX, STATION_IDS


def test_frozen_flags_and_target_count():
    assert ASSOCIATION_DEFAULT_SYSTEM == "frozen_W64_raw_energy_plus_exact_solver"
    assert N_TARGET_REPLICAS == 100
    assert MIN_REPLICAS_FOR_PASS == 100
    assert SOURCE_ID == "toy_independent_measurement_v1"


def test_translation_direction_is_orthogonal_to_z_slope():
    z = np.asarray([Z_MM[1], Z_MM[2], Z_MM[3]], dtype=np.float64)
    vec = np.asarray(
        [TRANSLATION_DIRECTION["s1_dx_mm"], TRANSLATION_DIRECTION["s2_dx_mm"], TRANSLATION_DIRECTION["s3_dx_mm"]]
    )
    assert abs(float(vec @ z)) < 1.0e-12
    assert float(np.linalg.norm(vec)) == pytest.approx(1.0)


def test_replica_seeds_are_unique_and_not_the_design_seed():
    weak = freeze_weak_mode()
    matrix = build_geometry_matrix(weak)
    manifest = build_replica_manifest(matrix)
    seeds = [row["seed"] for row in manifest["rows"]]
    keys = [row["replica_key"] for row in manifest["rows"]]
    assert len(seeds) == len(set(seeds))
    assert len(keys) == len(set(keys))
    assert DESIGN_SEED not in seeds
    assert replica_seed(condition_id="C0_nominal", amplitude=1.0, survey_mode="fixed_dz", replica_id=0) != replica_seed(
        condition_id="C0_nominal", amplitude=1.0, survey_mode="fixed_dz", replica_id=1
    )


def test_two_replicas_are_not_the_same_track_resample():
    from alignment.wb83_qualification import make_state

    first = make_state(np.random.default_rng(replica_seed(condition_id="C1_translation", amplitude=1.0, survey_mode="fixed_dz", replica_id=0)), 0)
    second = make_state(np.random.default_rng(replica_seed(condition_id="C1_translation", amplitude=1.0, survey_mode="fixed_dz", replica_id=1)), 0)
    assert not np.allclose(first, second)


def test_shared_measurement_audit_counts_dedup():
    observed = np.asarray([0.1, 0.0, 0.0, 0.0])
    cov = np.eye(4) * 0.01
    first = StationHit(1, 0, "same", observed, cov)
    second = StationHit(1, 0, "same", observed, cov)
    audit = measurement_audit(((first, second),))
    assert audit["n_raw_pairwise_observations"] == 2
    assert audit["n_unique_measurements"] == 1
    assert audit["deduplication_count"] == 1


def test_insufficient_replicas_are_unknown_not_pass():
    cell = {
        "cell_id": "C0_nominal_1p0_fixed_dz",
        "condition_id": "C0_nominal",
        "survey_mode": "fixed_dz",
        "amplitude": 1.0,
        "family": "nominal",
        "apply_bias_screening": True,
    }
    rows = [
        {
            "pull": 0.0,
            "residual_coefficient": 0.0,
            "sigma": 0.05,
            "covered_95": True,
            "converged": True,
            "n_dropped": 0,
        }
        for _ in range(8)
    ]
    report = summarize_cell(rows, cell)
    assert report["qualification"] == "UNKNOWN"
    gate = qualify([report], fd_ok=True, n_independent=8, forbidden_accessed=False)
    assert gate["alignment_oracle_qualified"] == "unknown"
    assert gate["qualification"] == "UNKNOWN"
    assert gate["common_track_solver_qualified_under_toy_model"] is False
    assert gate["alignment_oracle_qualified_for_physical_FASER"] is False


def test_forbidden_access_fail_closes_oracle():
    gate = qualify([], fd_ok=True, n_independent=100, forbidden_accessed=True)
    assert gate["alignment_oracle_qualified"] is False
    assert gate["qualification"] == "FAIL"
    assert gate["common_track_solver_qualified_under_toy_model"] is False
    assert gate["alignment_oracle_qualified_for_physical_FASER"] is False


def test_refuse_overlay_and_forbidden_alignment_paths():
    with pytest.raises(WB83Error, match="refuses"):
        refuse_wb83_path("outputs/family1/overlay_synthetic_v1/x.json")
    with pytest.raises(WB83Error, match="refuses"):
        refuse_wb83_path("mc24_100047_00350_00399/x.root")
    with pytest.raises(WB83Error, match="refuses"):
        refuse_wb83_path("00800_00849")
    with pytest.raises(WB83Error, match="refuses"):
        refuse_wb83_path("outputs/mc24_100116_x")


def test_iterate_max_iterations_is_not_pass():
    from alignment.common_track_geometry import left_update_payload
    from alignment.common_track_solver import predict_local_measurement
    from alignment.wb83_qualification import Z_MM

    payloads = {station: IDENTITY_SIX for station in STATION_IDS}
    truth = {station: IDENTITY_SIX for station in STATION_IDS}
    truth[3] = left_update_payload(IDENTITY_SIX, (0.5, 0.0, 0.0, 0.0, 0.0, 0.0))
    tracks = []
    states = {}
    for track_id in range(6):
        state = np.asarray([0.1 * track_id, -0.05 * track_id, 0.001, 0.0, 0.1], dtype=np.float64)
        states[track_id] = state
        hits = []
        for station in STATION_IDS:
            predicted = predict_local_measurement(
                state, truth[station], z_ref_mm=0.0, z_station_mm=Z_MM[station], field_y=0.0
            )
            hits.append(StationHit(track_id, station, f"{track_id}-{station}", predicted, np.eye(4) * 1.0e-4))
        tracks.append(hits)
    result = iterate_common_track(
        tracks,
        states,
        payloads,
        z_mm=Z_MM,
        field_y=0.0,
        validation_track_ids=(4, 5),
        max_iterations=2,
        consecutive_required=2,
        components=("dx_mm", "dy_mm"),
        stations=(3,),
    )
    assert result["automatically_passed_at_max_iterations"] is False
    if result["iterations"] >= 2 and not result["converged"]:
        assert result["reason"] == "max_iterations"


def test_rotation_second_step_is_not_killed_by_fd_asymmetry():
    from alignment.wb83_qualification import build_geometry_matrix, freeze_weak_mode, run_one_replica

    matrix = build_geometry_matrix(freeze_weak_mode())
    cell = next(item for item in matrix["cells"] if item["cell_id"] == "C2_rotation_0.5x_fixed_dz")
    row = run_one_replica(cell, 0, matrix_sha256=str(matrix["matrix_sha256"]))
    assert row["reason"] != "non_spd"
    assert row["converged"] is True


def test_finite_survey_prior_converges_instead_of_dz_walk():
    from alignment.wb83_qualification import build_geometry_matrix, freeze_weak_mode, run_one_replica

    matrix = build_geometry_matrix(freeze_weak_mode())
    cell = next(item for item in matrix["cells"] if item["cell_id"] == "C0_nominal_1p0_finite_survey_prior")
    row = run_one_replica(cell, 0, matrix_sha256=str(matrix["matrix_sha256"]))
    assert row["reason"] != "max_iterations"
    assert row["converged"] is True
    assert row["iterations"] < 10


def test_weak_mode_is_not_called_algebraic_gauge():
    weak = freeze_weak_mode()
    assert weak["frozen_before_replicas"] is True
    assert weak["not_algebraic_relative_gauge"] is True
    assert "g_i" not in str(weak["weak_direction"])
    matrix = build_geometry_matrix(weak)
    weak_cells = [cell for cell in matrix["cells"] if cell["condition_id"] == "C3_weak"]
    assert weak_cells
    assert weak_cells[0]["direction"] == weak["weak_direction"]
    assert matrix["wb82_smoke_is_not_qualification"] is True
    assert matrix["calypso_physical_replicas_available"] is False
