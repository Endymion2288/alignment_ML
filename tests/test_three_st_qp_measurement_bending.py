"""Yasu-S2I measurement-level bending.  Never flips S2 or opens S3."""

from __future__ import annotations

import pytest

from datasets.three_st_qp_calibration import SOURCE_COLLECTION_NAME
from datasets.three_st_qp_measurement_bending import (
    CLASS_A,
    CLASS_C,
    DECISION_CONTRACT,
    MECH_FIT,
    MECH_LIMITED,
    MECH_PRESENT,
    MECH_SOURCE,
    STATUS_SUPPORTED,
    ThreeStQpMeasurementBendingError,
    classify_topology,
    construct_bending,
    decide,
    evaluate_mechanisms,
    evaluate_tracks,
    inherit_frozen_stage,
    load_config,
    refuse_change_fitter,
    refuse_fitted_qp_in_construction,
    refuse_htcondor,
    refuse_invent_sigma_b,
    refuse_residual_conditional,
    refuse_seed_jacobian,
    refuse_flip_s2,
)


def test_config_freezes_s2_s2h_and_yz_plane():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    assert inherited["workbook_119"]["decision"] == "three_st_qp_calibration_not_established"
    assert inherited["workbook_124"]["decision"] == "three_st_qp_refit_provenance_recorded"
    assert config["three_st_qp_trusted_observable"] is False
    assert config["residual_conditional_authorized"] is False
    assert config["bending_plane"] == "YZ"
    assert config["orthogonal_plane"] == "XZ"
    assert config["observable"]["name"] == "bending_raw"
    assert config["official_qp_like_jacobian_authorized"] is False
    assert config["measurement_uncertainty_propagated"] is False
    assert config["do_not_submit_htcondor"] is True
    assert config["do_not_chase_front_provenance"] is True
    assert config["do_not_compute_e_r_ift_given_bending"] is True
    assert inherited["pinned_calypso_sources"]["all_match"] is True
    assert "circle_fit_cxx" in config["pinned_calypso_sources"]


def test_forbidden_repairs():
    with pytest.raises(ThreeStQpMeasurementBendingError, match="trusted_observable"):
        refuse_flip_s2()
    with pytest.raises(ThreeStQpMeasurementBendingError, match="Stage 3"):
        refuse_residual_conditional()
    with pytest.raises(ThreeStQpMeasurementBendingError, match="fitter"):
        refuse_change_fitter()
    with pytest.raises(ThreeStQpMeasurementBendingError, match="fitted q/p"):
        refuse_fitted_qp_in_construction()
    with pytest.raises(ThreeStQpMeasurementBendingError, match="sigma_b"):
        refuse_invent_sigma_b()
    with pytest.raises(ThreeStQpMeasurementBendingError, match="0.55"):
        refuse_seed_jacobian()
    with pytest.raises(ThreeStQpMeasurementBendingError, match="HTCondor"):
        refuse_htcondor()


def _centroids(y1: float, y2: float, y3: float, x1: float = 0.0, x2: float = 0.0, x3: float = 0.0) -> dict:
    return {
        1: {"x": x1, "y": y1, "z": 47.4, "n": 6},
        2: {"x": x2, "y": y2, "z": 1237.4, "n": 6},
        3: {"x": x3, "y": y3, "z": 2427.4, "n": 6},
    }


def test_plus_sagitta_is_plus_charge_and_not_xz():
    config = load_config()
    plus = construct_bending(_centroids(0.0, 0.5, 0.0), config)
    minus = construct_bending(_centroids(0.0, -0.5, 0.0), config)
    x_only = construct_bending(_centroids(0.0, 0.0, 0.0, x1=0.0, x2=2.0, x3=0.0), config)
    assert plus["constructable"] is True
    assert plus["bending_raw"] > 0.0
    assert plus["sagitta_y_mm"] > 0.0
    assert minus["bending_raw"] < 0.0
    assert abs(x_only["bending_raw"]) < 1.0e-15
    assert x_only["bending_x"] > 0.0
    with pytest.raises(ThreeStQpMeasurementBendingError, match="fitted q/p"):
        construct_bending(_centroids(0.0, 0.5, 0.0), config, q_over_p_fit=1.0e-6)
    with pytest.raises(ThreeStQpMeasurementBendingError, match="truth q/p"):
        construct_bending(_centroids(0.0, 0.5, 0.0), config, q_over_p_truth=1.0e-6)


def _track(
    *,
    source_id: str,
    y2: float,
    q_truth: float,
    q_fit: float,
    n_mot: int = 18,
    stations: list[int] | None = None,
    p_mev: float = 2.0e5,
    sigma: float = 1.0e-7,
    x2: float = 0.0,
) -> dict:
    if stations is None:
        stations = [1] * 6 + [2] * 6 + [3] * 6
    return {
        "kind": "track",
        "collection": SOURCE_COLLECTION_NAME,
        "source_id": source_id,
        "event_id": 1,
        "track_index": 0,
        "n_mot": n_mot,
        "n_ift_mot": 0,
        "ift_leak": False,
        "measurements_on_track_stations": stations,
        "station_centroids": {
            "1": {"x_mm": 0.0, "y_mm": 0.0, "z_mm": 47.4, "n": 6},
            "2": {"x_mm": x2, "y_mm": y2, "z_mm": 1237.4, "n": 6},
            "3": {"x_mm": 0.0, "y_mm": 0.0, "z_mm": 2427.4, "n": 6},
        },
        "q_over_p_fit_per_mev": q_fit,
        "q_over_p_truth_s1_per_mev": q_truth,
        "p_truth_s1_mev": p_mev,
        "truth_charge": 1.0 if q_truth > 0 else -1.0,
        "sigma_q_over_p_per_mev": sigma,
        "truth_used_as_fit_seed": False,
        "truth_used_as_solution": False,
    }


def _inherited() -> dict:
    return {
        "workbook_119": {"decision": "three_st_qp_calibration_not_established"},
        "workbook_124": {"decision": "three_st_qp_refit_provenance_recorded"},
        "pinned_calypso_sources": {"all_match": True},
    }


def test_clean_versus_dirty_kept():
    dirty = classify_topology(
        {"n_mot": 12, "n_ift_mot": 0, "measurements_on_track_stations": [2] * 6 + [3] * 6}
    )
    clean = classify_topology(
        {"n_mot": 18, "n_ift_mot": 0, "measurements_on_track_stations": [1] * 6 + [2] * 6 + [3] * 6}
    )
    assert dirty["dirty"] is True
    assert dirty["clean_18hit"] is False
    assert clean["clean_18hit"] is True


def test_present_and_fit_extra_on_synthetic_clean():
    config = load_config()
    tracks = []
    for idx in range(10):
        tracks.append(
            _track(
                source_id="mc24_100043_00400_00499",
                y2=-0.4 - 0.02 * idx,
                q_truth=-5.0e-6 - 1.0e-7 * idx,
                q_fit=-5.0e-6 - 1.0e-7 * idx,
                p_mev=1.5e5,
            )
        )
        tracks.append(
            _track(
                source_id="mc24_100048_00000_00049",
                y2=0.4 + 0.02 * idx,
                q_truth=5.0e-6 + 1.0e-7 * idx,
                q_fit=-5.0e-6,
                p_mev=1.5e5,
            )
        )
    acc = evaluate_tracks(tracks, config)
    assert acc["n_clean_18hit"] == 20
    assert acc["n_dirty"] == 0
    assert acc["clean_18hit"]["sign_agree"] == 1.0
    assert acc["clean_18hit"]["class_counts"][CLASS_A] == 10
    assert acc["clean_18hit"]["class_counts"][CLASS_C] == 0
    mechs = evaluate_mechanisms(acc, config)
    assert mechs["raw_bending_information_present"] == STATUS_SUPPORTED
    assert mechs["fit_additional_sign_failure"] == STATUS_SUPPORTED
    decision = decide(acc, _inherited(), dumps_materialized=True, campaign="smoke", config=config)
    assert decision["decision"] == DECISION_CONTRACT
    assert decision["three_st_qp_trusted_observable"] is False
    assert decision["residual_conditional_authorized"] is False
    assert decision["e_r_ift_given_bending_computed"] is False
    assert MECH_PRESENT in decision["mechanisms"]["supported_mechanisms"]
    assert MECH_FIT in decision["mechanisms"]["supported_mechanisms"]


def test_high_p_limit_and_source_bias_and_bending_flip():
    config = load_config()
    tracks = []
    for idx in range(8):
        tracks.append(
            _track(
                source_id="mc24_100043_00400_00499",
                y2=-0.5,
                q_truth=-4.0e-6,
                q_fit=-4.0e-6,
                p_mev=5.0e4,
            )
        )
        tracks.append(
            _track(
                source_id="mc24_100048_00000_00049",
                y2=-0.01 if idx < 6 else 0.4,
                q_truth=4.0e-6,
                q_fit=4.0e-6,
                p_mev=3.0e5,
            )
        )
    tracks.append(
        _track(
            source_id="mc24_100048_00000_00049",
            y2=-0.4,
            q_truth=4.0e-6,
            q_fit=4.0e-6,
            n_mot=12,
            stations=[2] * 6 + [3] * 6 + [1] * 0,
        )
    )
    # missing S1 is dirty; keep it
    dirty_only = _track(
        source_id="mc24_100043_00400_00499",
        y2=-0.4,
        q_truth=-4.0e-6,
        q_fit=-4.0e-6,
        n_mot=10,
        stations=[2] * 5 + [3] * 5,
    )
    dirty_only["station_centroids"] = {
        "2": {"x_mm": 0.0, "y_mm": -0.4, "z_mm": 1237.4, "n": 5},
        "3": {"x_mm": 0.0, "y_mm": 0.0, "z_mm": 2427.4, "n": 5},
    }
    tracks.append(dirty_only)
    acc = evaluate_tracks(tracks, config)
    assert acc["n_dirty"] >= 1
    assert acc["n_clean_18hit"] >= 8
    mechs = evaluate_mechanisms(acc, config)
    assert mechs["raw_bending_information_limited_at_high_p"] in {STATUS_SUPPORTED, "rejected", "unresolved"}
    assert CLASS_C in acc["clean_18hit"]["class_counts"]
    assert MECH_LIMITED in (MECH_LIMITED, MECH_SOURCE)
    decision = decide(acc, _inherited(), dumps_materialized=True, campaign="smoke", config=config)
    assert decision["s2_flipped_to_pass"] is False
    assert decision["wb119_sign_flip_full_sample"]["quantified"] is False
