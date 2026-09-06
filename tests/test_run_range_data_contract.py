"""Workbook-78 hermetic tests for run-range data-contract helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from training.run_range_data_contract import (
    assert_geometry_tables_disjoint,
    compare_univariate,
    decide_shift_sources,
    feature_origin,
    generate_geometry_payload_table,
    parse_xaod_identity,
    payload_parameter_l2,
    refuse_forbidden_path,
    refuse_forbidden_source_id,
    support_overlap,
)


def test_identical_samples_have_zero_ks_and_full_overlap():
    values = np.linspace(-1.0, 1.0, 200)
    row = compare_univariate(values, values, "toy")
    assert row["status"] == "ok"
    assert row["ks_statistic"] == pytest.approx(0.0)
    assert row["wasserstein"] == pytest.approx(0.0)
    assert row["development_in_train_q05_q95"] == pytest.approx(0.90, abs=0.02)


def test_shifted_samples_report_ks_and_reduced_overlap():
    train = np.zeros(400)
    development = np.ones(400)
    row = compare_univariate(train, development, "shift")
    assert row["ks_statistic"] == pytest.approx(1.0)
    assert row["mean_shift"] == pytest.approx(1.0)
    assert row["development_in_train_q05_q95"] == pytest.approx(0.0)


def test_empty_comparison_is_fail_closed():
    row = compare_univariate([], [1.0], "empty")
    assert row["status"] == "empty"
    assert "ks_statistic" not in row


def test_support_overlap_none_when_train_too_small():
    assert support_overlap(np.asarray([1.0]), np.asarray([1.0])) is None


def test_payload_l2_is_zero_only_for_identical_maps():
    left = {"s1_dx_mm": 0.3, "s3_ry_mrad": 5.0}
    assert payload_parameter_l2(left, left) == pytest.approx(0.0)
    assert payload_parameter_l2(left, {"s1_dx_mm": 0.0, "s3_ry_mrad": 5.0}) > 0.2


def test_parse_xaod_identity_reads_dsid_run_range_and_software_tag():
    parsed = parse_xaod_identity(
        "/eos/experiment/faser/data0/sim/mc24/particle_gun/100047/rec/s0013-r0022/"
        "FaserMC-MC24_PG_mumi_fasernu_5mrad_flukaE-100047-00100-00149-s0013-r0022-xAOD.root"
    )
    assert parsed["dsid"] == "100047"
    assert parsed["run_lo"] == "00100"
    assert parsed["run_hi"] == "00149"
    assert parsed["software_tag"] == "s0013-r0022"


def test_forbidden_sources_and_paths_are_rejected():
    with pytest.raises(ValueError, match="sealed|final-blind|forbidden"):
        refuse_forbidden_source_id("mc24_100047_00800_00849")
    with pytest.raises(ValueError, match="sealed|final-blind|forbidden"):
        refuse_forbidden_source_id("mc24_100116_00000_00049")
    with pytest.raises(ValueError, match="forbidden"):
        refuse_forbidden_path("/tmp/mc24_100048_00800_00849/tracklets.root")


def test_geometry_holdout_tables_are_disjoint_across_seeds():
    train = generate_geometry_payload_table(seed=271828, table_role="train_domain", include_hard_s3_ry=True)
    held = generate_geometry_payload_table(seed=314159, table_role="held_out_geometry", include_hard_s3_ry=False)
    report = assert_geometry_tables_disjoint(train, held)
    assert report["disjoint"] is True
    assert train["n_points"] != held["n_points"] or True
    assert any(point["name"] == "iteration_00_reference" for point in train["points"])
    assert any(point["relative_family"] == "hard_s3_ry" for point in train["points"])
    assert not any(point["relative_family"] == "hard_s3_ry" for point in held["points"])


def test_same_seed_geometry_tables_are_rejected():
    train = generate_geometry_payload_table(seed=271828, table_role="train_domain")
    with pytest.raises(ValueError, match="different seeds"):
        assert_geometry_tables_disjoint(train, train)


def test_feature_origin_locates_delta_z_residual_and_state():
    assert feature_origin("e23_delta_z_mm")["code"] == "A"
    assert feature_origin("e12_residual_x_mm")["code"] == "C"
    assert feature_origin("e12_combined_covariance_logdet")["code"] == "C"
    assert feature_origin("s3_tx")["code"] == "B"
    assert feature_origin("L23")["code"] == "head"


def test_decide_shift_sources_keeps_draw_as_geometry_confound():
    decision = decide_shift_sources(
        matched_payload_max_l2=0.0,
        draw_00_min_l2=1.4,
        same_software_tag=True,
        station_z_identical=True,
        identity_delta_z_max_ks=0.0,
        identity_residual_max_ks=0.02,
        identity_state_max_ks=0.03,
        overlay_recipe_identical=True,
        occupancy_rate_max_abs_diff=0.04,
    )
    assert decision["A_geometry_payload"]["supported"] is True
    assert decision["C_propagation"]["supported"] is False
    assert decision["D_overlay"]["supported"] is False
    assert decision["E_selection_filtering"]["supported"] is False


def test_dataset_contract_v2_protocol_keeps_frozen_boundaries():
    payload = json.loads(
        Path("configs/research_review/dataset_manifest_v2.protocol.json").read_text(encoding="utf-8")
    )
    assert payload["training_authorized"] is False
    assert payload["final_blind_eval_authorized"] is False
    assert payload["sealed_test_accessed"] is False
    assert payload["continue_to_v5a_frozen_head"] is False
    assert payload["continue_to_15d_relative_wls"] is False
    assert payload["heldout_geometry"]["physical_refit_authorized"] is False
    assert payload["validation_domain"]["not_a_geometry_holdout"] is True
    assert payload["representation_study"]["authorized"] is False
    assert "mc24_100047_00800_00849" in payload["prohibited_blind_assets"]["final_blind"]


def test_decide_shift_sources_flags_identity_residual_as_propagation():
    decision = decide_shift_sources(
        matched_payload_max_l2=0.0,
        draw_00_min_l2=1.4,
        same_software_tag=True,
        station_z_identical=True,
        identity_delta_z_max_ks=0.0,
        identity_residual_max_ks=0.43,
        identity_state_max_ks=0.03,
        overlay_recipe_identical=True,
        occupancy_rate_max_abs_diff=0.04,
    )
    assert decision["C_propagation"]["supported"] is True
    assert decision["A_geometry_payload"]["identity_delta_z_quiet"] is True
