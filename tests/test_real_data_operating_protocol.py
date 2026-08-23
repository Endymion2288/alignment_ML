from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from alignment.calibration_modes import load_mode_validity_contract
from alignment.hierarchical_v1 import STATION_SOLVE_PARAMETERS
from alignment.real_data_operating_protocol import (
    BLOCK_CANDIDATE_GRAPH_DQ_FAILED,
    BLOCK_CANDIDATE_GRAPH_DQ_PASSED,
    BLOCK_CROSS_LEVEL_CONTAMINATED,
    BLOCK_DQ_FAILED,
    BLOCK_GEOMETRY_WRITE_CANDIDATE,
    BLOCK_VALID_FOR_STATION_MODE,
    ROLE_CALIBRATION,
    ROLE_HELD_OUT_DQ,
    WORKBOOK_03_BLOCKED_REASON,
    assign_block_status,
    compile_real_data_current_only,
    compile_real_data_station_linearization,
    cross_level_contamination_diagnostic,
)
from datasets.real_data_identity import write_real_data_identity_tracklets
from datasets.root_loader import load_events
from scripts.convert_ntuple_tracklets import convert_ntuple_tracklets
import awkward as ak
import numpy as np
import uproot


def _write_mock_enhanced_ntuple(path):
    branches = {
        "run": np.asarray([7, 7], dtype=np.int32),
        "eventID": np.asarray([10, 11], dtype=np.int32),
        "Tracklet_station_id": ak.Array([[0, 1], [0]]),
        "Tracklet_id": ak.Array([[0, 1], [0]]),
        "Tracklet_x_mm": ak.Array([[1.0, 2.0], [3.0]]),
        "Tracklet_y_mm": ak.Array([[4.0, 5.0], [6.0]]),
        "Tracklet_z_mm": ak.Array([[7.0, 8.0], [9.0]]),
        "Tracklet_tx": ak.Array([[0.01, 0.02], [0.03]]),
        "Tracklet_ty": ak.Array([[0.04, 0.05], [0.06]]),
        "Tracklet_chi2": ak.Array([[1.0, 2.0], [3.0]]),
        "Tracklet_ndof": ak.Array([[3.0, 4.0], [5.0]]),
        "Tracklet_n_hit": ak.Array([[6, 6], [6]]),
        "Tracklet_hit_pattern": ak.Array([[7, 7], [7]]),
        "Tracklet_has_covariance": ak.Array([[True, False], [True]]),
        "Tracklet_truth_particle_id": ak.Array([[101, 102], [103]]),
        "Tracklet_truth_pdg": ak.Array([[11, -11], [11]]),
        "Tracklet_truth_match_fraction": ak.Array([[1.0, 0.5], [0.75]]),
        "Tracklet_q_over_p_per_MeV": ak.Array([[1.0e-5, -2.0e-5], [3.0e-5]]),
        "Tracklet_q_over_p_from_momentum_per_MeV": ak.Array(
            [[1.0e-5, -2.0e-5], [3.0e-5]]
        ),
        "Tracklet_q_over_p_variance_per_MeV2": ak.Array([[1.0e-9, 2.0e-9], [3.0e-9]]),
        "Tracklet_has_q_over_p_covariance": ak.Array([[True, False], [True]]),
    }
    for name in (
        "xx_mm2",
        "xy_mm2",
        "xtx_mm",
        "xty_mm",
        "yy_mm2",
        "ytx_mm",
        "yty_mm",
        "txtx",
        "txty",
        "tyty",
    ):
        branches[f"Tracklet_cov_{name}"] = ak.Array([[0.01, 0.01], [0.01]])
    with uproot.recreate(path) as root_file:
        root_file["nt"] = branches


def _template():
    path = Path("configs/physical_refit_station_mode_real_data_dryrun.yaml")
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_station_linearization_is_current_plus_axial_fd_only():
    compiled, contract = compile_real_data_station_linearization(_template())
    points = compiled["physical_refit_capture_scan"]["rigid_points"]
    names = [point["name"] for point in points]
    assert names[0] == "iteration_00_current"
    assert points[0]["point_role"] == "nominal"
    assert len(points) == 1 + 2 * len(STATION_SOLVE_PARAMETERS)
    fd = [point for point in points[1:]]
    assert all(point["finite_difference_anchor"] == "iteration_00_current" for point in fd)
    joint = []
    for point in fd:
        values = point["alignment_parameter_values"]
        excited = [name for name, value in values.items() if abs(float(value)) > 1.0e-15]
        if len(excited) != 1:
            joint.append(point["name"])
    assert joint == []
    assert contract["self_nulling_update"] is True
    assert contract["official_conditions_db_write"] is False
    assert contract["cdx_fixed_by"] == "external_geometry"
    assert compiled["physical_refit_capture_scan"]["is_mc"] is False


def test_holdout_scan_is_current_geometry_only():
    compiled, contract = compile_real_data_current_only(_template())
    points = compiled["physical_refit_capture_scan"]["rigid_points"]
    assert [point["name"] for point in points] == ["iteration_00_current"]
    assert contract["geometry_estimation_allowed"] is False
    assert compiled["physical_refit_capture_scan"]["current_geometry_only"] is True
    from scripts.run_physical_refit_capture_scan import _build_plan

    holdout_plan = _build_plan(compiled["physical_refit_capture_scan"])
    assert len(holdout_plan["points"]) == 1
    station, _ = compile_real_data_station_linearization(_template())
    station_plan = _build_plan(station["physical_refit_capture_scan"])
    assert len(station_plan["points"]) == 1 + 2 * len(STATION_SOLVE_PARAMETERS)


def test_calypso_subcommand_clears_calypso_setup_markers():
    from scripts.run_physical_refit_capture_scan import _calypso_command

    command = _calypso_command("python -c 'pass'")
    for marker in (
        "Calypso_SET_UP",
        "Calypso_EXTONLY_SET_UP",
        "Calypso_RELONLY_SET_UP",
        "Athena_SET_UP",
        "AthenaExternals_SET_UP",
    ):
        assert marker in command.split("\n", 1)[0]


def test_implied_cdx_from_station_dx_rejects_above_operating_band():
    contract = load_mode_validity_contract(
        Path("outputs/mc24_ift_calibration_mode_validity_contract_v1/mode_validity_contract.json")
    )
    # |C_dx| = |dx_um / A_dx|; 0.2 mm station dx => ~3.4 µm implied C_dx.
    diagnostic = cross_level_contamination_diagnostic(
        contract, station_dx_mm=0.2, station_ry_mrad=None, run_to_run_dx_spread_mm=None
    )
    assert diagnostic["exceeds_operating_band"] is True
    assert diagnostic["mode_validity"]["geometry_write_allowed"] is False
    assert diagnostic["not_a_C_dx_measurement"] is True
    assert diagnostic["implied_Cdx_dx"] == pytest.approx(-diagnostic["implied_C_dx_um_from_station_dx"])
    assert "true_C_dx_not_independently_proven_le_statistical_budget" in diagnostic["mode_validity"]["reject_indicators"]


def test_small_implied_cdx_still_contaminated_without_independent_evidence():
    contract = load_mode_validity_contract(
        Path("outputs/mc24_ift_calibration_mode_validity_contract_v1/mode_validity_contract.json")
    )
    diagnostic = cross_level_contamination_diagnostic(
        contract, station_dx_mm=0.01, station_ry_mrad=0.005, run_to_run_dx_spread_mm=0.0
    )
    assert diagnostic["exceeds_operating_band"] is False
    assert diagnostic["mode_validity"]["geometry_write_allowed"] is False
    assert diagnostic["cdx_fixed_by_is_provenance_only"] is True
    assert "true_C_dx_not_independently_proven_le_statistical_budget" in diagnostic["mode_validity"]["reject_indicators"]


def test_unknown_implied_pattern_cannot_prove_precondition():
    contract = load_mode_validity_contract(
        Path("outputs/mc24_ift_calibration_mode_validity_contract_v1/mode_validity_contract.json")
    )
    diagnostic = cross_level_contamination_diagnostic(
        contract, station_dx_mm=None, station_ry_mrad=None, run_to_run_dx_spread_mm=None
    )
    assert diagnostic["mode_validity"]["geometry_write_allowed"] is False
    assert "true_C_dx_unknown_implied_pattern_unavailable" in diagnostic["mode_validity"]["reject_indicators"]


def test_block_status_and_no_conditions_write():
    assert (
        assign_block_status(
            role=ROLE_CALIBRATION,
            dq_failed=True,
            cross_level_contaminated=False,
            station_mode_valid=True,
            cdx_mode_valid=False,
            holdout_dq_worsened=False,
            anomalous_run_drift=False,
        )
        == BLOCK_DQ_FAILED
    )
    assert (
        assign_block_status(
            role=ROLE_CALIBRATION,
            dq_failed=False,
            cross_level_contaminated=True,
            station_mode_valid=False,
            cdx_mode_valid=False,
            holdout_dq_worsened=False,
            anomalous_run_drift=False,
        )
        == BLOCK_CROSS_LEVEL_CONTAMINATED
    )
    assert (
        assign_block_status(
            role=ROLE_HELD_OUT_DQ,
            dq_failed=False,
            cross_level_contaminated=False,
            station_mode_valid=True,
            cdx_mode_valid=False,
            holdout_dq_worsened=False,
            anomalous_run_drift=False,
        )
        == BLOCK_VALID_FOR_STATION_MODE
    )
    assert (
        assign_block_status(
            role=ROLE_CALIBRATION,
            dq_failed=False,
            cross_level_contaminated=False,
            station_mode_valid=True,
            cdx_mode_valid=True,
            holdout_dq_worsened=False,
            anomalous_run_drift=False,
        )
        == BLOCK_GEOMETRY_WRITE_CANDIDATE
    )
    assert (
        assign_block_status(
            role=ROLE_CALIBRATION,
            dq_failed=False,
            cross_level_contaminated=False,
            station_mode_valid=False,
            cdx_mode_valid=False,
            holdout_dq_worsened=False,
            anomalous_run_drift=False,
            candidate_graph_dq_passed=True,
        )
        == BLOCK_CANDIDATE_GRAPH_DQ_PASSED
    )
    assert (
        assign_block_status(
            role=ROLE_CALIBRATION,
            dq_failed=True,
            cross_level_contaminated=False,
            station_mode_valid=True,
            cdx_mode_valid=False,
            holdout_dq_worsened=False,
            anomalous_run_drift=False,
            candidate_graph_dq_failed=True,
        )
        == BLOCK_CANDIDATE_GRAPH_DQ_FAILED
    )
    with pytest.raises(ValueError, match="official conditions"):
        assign_block_status(
            role=ROLE_CALIBRATION,
            dq_failed=False,
            cross_level_contaminated=False,
            station_mode_valid=True,
            cdx_mode_valid=True,
            holdout_dq_worsened=False,
            anomalous_run_drift=False,
            official_conditions_write_requested=True,
        )


def test_identity_sample_refuses_mc_labels(tmp_path):
    ntuple = tmp_path / "enhanced.root"
    _write_mock_enhanced_ntuple(ntuple)
    with_truth = tmp_path / "with_truth.root"
    convert_ntuple_tracklets(ntuple, with_truth, include_truth=True)
    with pytest.raises(ValueError, match="MC truth"):
        write_real_data_identity_tracklets(with_truth, tmp_path / "identity.root")


def test_identity_sample_stamps_origin_without_truth(tmp_path):
    ntuple = tmp_path / "enhanced.root"
    _write_mock_enhanced_ntuple(ntuple)
    physical = tmp_path / "physical.root"
    convert_ntuple_tracklets(ntuple, physical, include_truth=False)
    identity = tmp_path / "identity.root"
    summary = write_real_data_identity_tracklets(physical, identity)
    assert summary["has_mc_labels"] is False
    events = load_events(identity, require_mc_labels=False)
    assert events
    assert events[0].truth_particle_id is None
    assert int(events[0].origin_run_id[0]) == int(events[0].run_id)
    assert int(events[0].origin_event_id[0]) == int(events[0].event_id)


def test_workbook_03_block_reason_is_explicit():
    assert "2022" in WORKBOOK_03_BLOCKED_REASON
    assert "1...N" in WORKBOOK_03_BLOCKED_REASON or "1..." in WORKBOOK_03_BLOCKED_REASON
