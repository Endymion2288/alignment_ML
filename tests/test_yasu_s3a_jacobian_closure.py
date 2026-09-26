"""Yasu-S3A Jacobian contract.  Isolated from S2K.  Never claims a weak mode."""

from __future__ import annotations

import numpy as np
import pytest

from datasets.ckf_without_ift_definition import SOURCE_COLLECTION
from datasets.yasu_s3a_jacobian_closure import (
    COLLINEAR_COSINE,
    DECISION_CONTRACT,
    FD_RUNG_FACTORS,
    FROZEN_DELTAS,
    PARAM_DX,
    PARAM_QP,
    PARAM_RY,
    PARAM_TY,
    STATUS_PASS,
    YasuS3AError,
    analytic_uniform_field_circle_control,
    analytic_zero_field_control,
    association_contract,
    decide,
    evaluate_controls,
    evaluate_track_jacobian,
    focus_manifest,
    frozen_deltas,
    geometry_response_definition,
    identity_contract,
    jacobian_from_rungs,
    load_config,
    refuse_b14m,
    refuse_four_station_association,
    refuse_free_scale,
    refuse_held_out,
    refuse_htcondor,
    refuse_native_covariance_weight,
    refuse_replace_focus,
    refuse_s2k_batch,
    refuse_s2k_proxy,
    refuse_times_two_patch,
    refuse_weak_mode_claim,
    residual_definition,
    se3_ry_dx,
    attach_digests,
    measurement_digest,
    zero_field_analytic_residual,
)


def test_config_freezes_focus_and_refuses_s2k():
    config = load_config()
    assert config["task"] == "YASU-S3A"
    assert config["workbook"] == 128
    assert config["source_collection"] == SOURCE_COLLECTION
    assert config["use_qp_bending_proxy"] is False
    assert config["use_native_5x5_jacobian_weight"] is False
    assert config["four_station_association_authorized"] is False
    assert config["s2k_batch_authorized"] is False
    assert config["times_two_patch_authorized"] is False
    assert config["htcondor_authorized"] is False
    assert config["weak_mode_claim_authorized"] is False
    assert config["do_not_replace_focus_after_seeing_results"] is True
    assert len(config["focus_identities"]) == 20
    keys = {
        (row["source_id"], row["skip_index"], row["track_index"])
        for row in config["focus_identities"]
    }
    assert len(keys) == 20
    assert tuple(config["finite_difference"]["rung_factors"]) == FD_RUNG_FACTORS
    assert frozen_deltas(config) == FROZEN_DELTAS
    roles = {row["role"] for row in config["focus_identities"]}
    assert "frozen_wb119" in roles
    assert "validation_s1_front_control" in roles
    assert "construction_s2_front_dirty" in roles
    assert sum(1 for row in config["focus_identities"] if row["reason"] == "sign_flip") == 2
    manifest = focus_manifest(config)
    assert manifest["frozen_before_jacobian"] is True
    assert manifest["n"] == 20
    assert residual_definition()["four_station_association_forbidden"] is True
    assert identity_contract()["collection"] == SOURCE_COLLECTION
    assert association_contract()["method"] == "truth_sdo_barcode"
    assert geometry_response_definition()["handwritten_station_response"] is False


def test_forbidden_repairs():
    with pytest.raises(YasuS3AError, match="qp_bending_proxy"):
        refuse_s2k_proxy()
    with pytest.raises(YasuS3AError, match="multiply"):
        refuse_times_two_patch()
    with pytest.raises(YasuS3AError, match="free scale"):
        refuse_free_scale()
    with pytest.raises(YasuS3AError, match="5x5"):
        refuse_native_covariance_weight()
    with pytest.raises(YasuS3AError, match="4ST"):
        refuse_four_station_association()
    with pytest.raises(YasuS3AError, match="HTCondor"):
        refuse_htcondor()
    with pytest.raises(YasuS3AError, match="B14M"):
        refuse_b14m()
    with pytest.raises(YasuS3AError, match="held-out"):
        refuse_held_out()
    with pytest.raises(YasuS3AError, match="weak mode"):
        refuse_weak_mode_claim()
    with pytest.raises(YasuS3AError, match="focus"):
        refuse_replace_focus()
    with pytest.raises(YasuS3AError, match="paused"):
        refuse_s2k_batch()


def test_python_recomputes_measurement_digest():
    row = attach_digests(
        {
            "mot_identifiers": ["b", "a"],
            "associated_ift_identifiers": ["c"],
            "measurement_digest": None,
            "association_digest": None,
        }
    )
    assert row["measurement_digest"] == measurement_digest(["a", "b", "c"])
    assert row["association_digest"] == measurement_digest(["c"])


def test_analytic_controls_close_before_any_dump():
    controls = evaluate_controls()
    assert controls["analytic_controls_closed"] is True
    assert controls["stop_physical_interpretation"] is False
    zero = analytic_zero_field_control()
    assert zero["q_over_p_column_near_zero"] is True
    for name in (PARAM_RY, PARAM_DX, PARAM_TY):
        assert zero["geometry_columns"][name]["convergence_ok"] is True
        assert zero["geometry_columns"][name]["linear_ok"] is True
    # u = global x: d_x and R_y (about the origin) enter loc0; t_y does not.
    assert zero["geometry_columns"][PARAM_DX]["near_zero"] is False
    assert zero["geometry_columns"][PARAM_RY]["near_zero"] is False
    assert zero["geometry_columns"][PARAM_TY]["near_zero"] is True
    circle = analytic_uniform_field_circle_control()
    assert circle["convergence_ok"] is True
    assert circle["relative_fd_vs_truth"] < 1.0e-2


def test_ry_dx_use_runtime_axes_not_yz_guess():
    origin = (0.0, 0.0, 50.0)
    direction = (0.0, 0.0, 1.0)
    surface = {
        "center_mm": np.array([0.0, 0.0, -1860.15]),
        "u": np.array([0.0, 1.0, 0.0]),
        "v": np.array([-1.0, 0.0, 0.0]),
        "n": np.array([0.0, 0.0, 1.0]),
    }
    cluster = np.array([0.0, 0.2, -1860.15])
    nominal = zero_field_analytic_residual(origin, direction, cluster, surface)
    plus_dx = zero_field_analytic_residual(
        origin, direction, cluster, surface, dx_mm=0.10
    )
    plus_ry = zero_field_analytic_residual(
        origin, direction, cluster, surface, ry_rad=1.0e-3
    )
    # Stereo u is global y, so d_x (global x) does not enter loc0 at first order.
    assert abs(plus_dx["residual_loc0_mm"] - nominal["residual_loc0_mm"]) < 1.0e-6
    # Ry about origin moves a sensor at z=-1860 mm in x, which is orthogonal to u=y.
    assert abs(plus_ry["residual_loc0_mm"] - nominal["residual_loc0_mm"]) < 1.0e-3
    transform = se3_ry_dx(ry_rad=1.0e-3, dx_mm=0.10)
    assert transform[0, 3] == pytest.approx(0.10)
    assert transform[0, 2] == pytest.approx(np.sin(1.0e-3))


def test_fd_stop_on_association_change():
    config = load_config()
    residual = {
        "identifier": "a",
        "layer": 0,
        "side": 0,
        "residual_available": True,
        "residual_loc0_mm": 0.01,
        "surface_geo_id": 1,
    }
    changed = dict(residual, identifier="b")

    def rungs_for(name, values):
        return {
            "plus": [[dict(residual, residual_loc0_mm=v)] for v in values],
            "minus": [[dict(residual, residual_loc0_mm=-v)] for v in values],
        }

    track = {
        "file_sha256": "abc",
        "source_id": "mc24_100043_00400_00499",
        "run_id": 100043,
        "event_id": 0,
        "skip_index": 0,
        "collection": SOURCE_COLLECTION,
        "track_index": 0,
        "measurement_digest": "digest",
        "ift_association": {"method": "truth_sdo_barcode"},
        "nominal_residuals": [residual],
        "fixed_state_rungs": {
            PARAM_QP: rungs_for(PARAM_QP, [0.02, 0.01, 0.005]),
            PARAM_RY: rungs_for(PARAM_RY, [0.02, 0.01, 0.005]),
            PARAM_DX: rungs_for(PARAM_DX, [0.02, 0.01, 0.005]),
            PARAM_TY: {
                "plus": [[changed], [residual], [residual]],
                "minus": [[residual], [residual], [residual]],
            },
        },
    }
    result = evaluate_track_jacobian(track, config)
    assert result["stop_physical_interpretation"] is True
    assert result["failure_class"] == "association_changed"


def test_contract_decision_without_dump_is_not_a_weak_mode():
    config = load_config()
    controls = evaluate_controls()
    inherited = {
        "workbook_119": {"decision": "three_st_qp_calibration_not_established"},
        "pinned_calypso_sources": {"all_match": True},
    }
    decision = decide(
        inherited=inherited,
        controls=controls,
        track_results=[],
        dumps_materialized=False,
        config=config,
    )
    assert decision["decision"] == DECISION_CONTRACT
    assert decision["verdict"] == STATUS_PASS
    assert decision["weak_mode_claimed"] is False
    assert decision["qp_bending_proxy_used"] is False
    assert decision["authorize_profiled_schur_next"] is False
    assert "does not establish a natural-data" in decision["legal_final_claim"]


def _closed_rungs(residual):
    nominal = float(residual["residual_loc0_mm"])
    shifts = (0.02, 0.01, 0.005)
    def side(sign):
        return [[dict(residual, residual_loc0_mm=nominal + sign * shift)] for shift in shifts]
    block = {"plus": side(1.0), "minus": side(-1.0)}
    return {
        PARAM_QP: dict(block),
        PARAM_RY: dict(block),
        PARAM_DX: dict(block),
        PARAM_TY: dict(block),
    }


def test_dirty_coverage_empty_ift_does_not_fail_stage():
    config = load_config()
    controls = evaluate_controls()
    inherited = {
        "workbook_119": {"decision": "three_st_qp_calibration_not_established"},
        "pinned_calypso_sources": {"all_match": True},
    }
    residual = {
        "identifier": "a",
        "layer": 0,
        "side": 0,
        "residual_available": True,
        "residual_loc0_mm": 0.01,
        "surface_geo_id": 1,
    }
    eligible = {
        "file_sha256": "abc",
        "source_id": "mc24_100043_00400_00499",
        "run_id": 100043,
        "event_id": 0,
        "skip_index": 0,
        "collection": SOURCE_COLLECTION,
        "track_index": 0,
        "measurement_digest": "digest",
        "role": "construction_s1_front_control",
        "ift_association": {"method": "truth_sdo_barcode"},
        "nominal_residuals": [residual],
        "fixed_state_rungs": _closed_rungs(residual),
    }
    dirty = {
        "file_sha256": "abc",
        "source_id": "mc24_100043_00400_00499",
        "run_id": 100043,
        "event_id": 50,
        "skip_index": 50,
        "collection": SOURCE_COLLECTION,
        "track_index": 0,
        "measurement_digest": "digest",
        "role": "construction_s2_front_dirty",
        "primary_failure_class": "truth_sdo_ift_unmatched",
        "ift_association": {"method": "truth_sdo_barcode", "n_truth_associated_ift_clusters": 0},
        "nominal_residuals": [],
    }
    dirty_result = evaluate_track_jacobian(dirty, config)
    eligible_result = evaluate_track_jacobian(eligible, config)
    assert dirty_result["coverage_only"] is True
    assert dirty_result["jacobian_eligible"] is False
    assert dirty_result["stop_physical_interpretation"] is True
    assert eligible_result["stop_physical_interpretation"] is False
    decision = decide(
        inherited=inherited,
        controls=controls,
        track_results=[eligible_result, dirty_result],
        dumps_materialized=True,
        config=config,
    )
    assert decision["verdict"] == STATUS_PASS
    assert decision["summary"]["n_coverage_only"] == 1
    assert decision["summary"]["n_jacobian_eligible_closed"] == 1
    assert decision["weak_mode_claimed"] is False
    assert "does not establish a natural-data" in decision["legal_final_claim"]


def test_partial_rung_unavailable_is_navigation_break():
    config = load_config()
    residual = {
        "identifier": "a",
        "layer": 0,
        "side": 0,
        "residual_available": True,
        "residual_loc0_mm": 0.01,
        "surface_geo_id": 1,
    }
    failed = dict(
        residual,
        residual_available=False,
        residual_loc0_mm=None,
        failure_class="propagate_surface_failed",
    )
    rungs = _closed_rungs(residual)
    rungs[PARAM_RY]["plus"][0] = [failed]
    track = {
        "file_sha256": "abc",
        "source_id": "mc24_100048_00000_00049",
        "run_id": 100048,
        "event_id": 1,
        "skip_index": 1,
        "collection": SOURCE_COLLECTION,
        "track_index": 0,
        "measurement_digest": "digest",
        "role": "validation_s1_front_control",
        "ift_association": {"method": "truth_sdo_barcode"},
        "nominal_residuals": [residual],
        "fixed_state_rungs": rungs,
    }
    result = evaluate_track_jacobian(track, config)
    assert result["stop_physical_interpretation"] is True
    assert "navigation_path_changed" in (result.get("failure_classes") or [])
    assert result["jacobian_eligible"] is False


def test_all_propagate_failed_is_not_empty_jacobian_closure():
    config = load_config()
    residual = {
        "identifier": "a",
        "layer": 0,
        "side": 0,
        "residual_available": False,
        "residual_loc0_mm": None,
        "surface_geo_id": 1,
        "failure_class": "propagate_surface_failed",
    }
    track = {
        "file_sha256": "abc",
        "source_id": "mc24_100048_00000_00049",
        "run_id": 100048,
        "event_id": 15,
        "skip_index": 15,
        "collection": SOURCE_COLLECTION,
        "track_index": 0,
        "measurement_digest": "digest",
        "role": "frozen_wb119",
        "ift_association": {"method": "truth_sdo_barcode"},
        "nominal_residuals": [residual],
        "fixed_state_rungs": _closed_rungs(dict(residual, residual_available=True, residual_loc0_mm=0.01)),
    }
    result = evaluate_track_jacobian(track, config)
    assert result["failure_class"] == "propagate_surface_failed"
    assert result["stop_physical_interpretation"] is True
    assert result["jacobian_eligible"] is False
    assert "fixed_state" not in result or result.get("n_residuals") == 0


def test_eligible_fd_break_fails_stage():
    config = load_config()
    controls = evaluate_controls()
    inherited = {
        "workbook_119": {"decision": "three_st_qp_calibration_not_established"},
        "pinned_calypso_sources": {"all_match": True},
    }
    residual = {
        "identifier": "a",
        "layer": 0,
        "side": 0,
        "residual_available": True,
        "residual_loc0_mm": 0.01,
        "surface_geo_id": 1,
    }
    changed = dict(residual, identifier="b")
    track = {
        "file_sha256": "abc",
        "source_id": "mc24_100043_00400_00499",
        "run_id": 100043,
        "event_id": 0,
        "skip_index": 0,
        "collection": SOURCE_COLLECTION,
        "track_index": 0,
        "measurement_digest": "digest",
        "role": "construction_s1_front_control",
        "ift_association": {"method": "truth_sdo_barcode"},
        "nominal_residuals": [residual],
        "fixed_state_rungs": {
            **_closed_rungs(residual),
            PARAM_TY: {
                "plus": [[changed], [residual], [residual]],
                "minus": [[residual], [residual], [residual]],
            },
        },
    }
    result = evaluate_track_jacobian(track, config)
    decision = decide(
        inherited=inherited,
        controls=controls,
        track_results=[result],
        dumps_materialized=True,
        config=config,
    )
    assert result["failure_class"] == "association_changed"
    assert decision["verdict"] == "FAIL"
    assert decision["authorize_profiled_schur_next"] is False


def test_high_cosine_is_not_a_weak_mode_gate():
    assert COLLINEAR_COSINE == 0.98
    nominal = np.array([0.0, 0.0])
    delta = FROZEN_DELTAS[PARAM_QP]
    plus = [nominal + 2.0 * delta * f * np.array([1.0, 0.5]) for f in FD_RUNG_FACTORS]
    minus = [nominal - 2.0 * delta * f * np.array([1.0, 0.5]) for f in FD_RUNG_FACTORS]
    report = jacobian_from_rungs(
        nominal,
        {PARAM_QP: {"plus": plus, "minus": minus}},
        deltas={PARAM_QP: delta},
    )
    assert report["columns"][PARAM_QP]["convergence_ok"] is True
    assert report["stop_physical_interpretation"] is False
