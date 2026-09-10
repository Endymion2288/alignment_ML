"""Yasu-S2J batch bending / mapping audit.  Never flips S2 or opens S3."""

from __future__ import annotations

import pytest

from datasets.three_st_qp_calibration import SOURCE_COLLECTION_NAME
from datasets.three_st_qp_measurement_bending import construct_bending
from datasets.three_st_qp_measurement_bending_batch import (
    CLASS_A,
    DECISION_CONTRACT,
    MECH_FIT,
    MECH_LIMITED,
    MECH_MAP,
    MECH_PRESENT,
    STATUS_SUPPORTED,
    STATUS_UNRESOLVED,
    ThreeStQpMeasurementBendingBatchError,
    decide,
    decorate_track,
    evaluate_mechanisms,
    evaluate_rows,
    inherit_frozen_stage,
    load_config,
    refuse_change_bending,
    refuse_flip_s2,
    refuse_invent_sigma_b,
    refuse_qp_like,
    refuse_residual_conditional,
)


def test_config_inherits_wb125_and_freezes_observable():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    assert inherited["workbook_119"]["decision"] == "three_st_qp_calibration_not_established"
    assert inherited["workbook_125"]["decision"] == "three_st_qp_measurement_bending_contract_established"
    assert config["three_st_qp_trusted_observable"] is False
    assert config["residual_conditional_authorized"] is False
    assert config["bending_plane"] == "YZ"
    assert config["observable"]["formula"] == "atan((y2-y1)/(z2-z1)) - atan((y3-y2)/(z3-z2))"
    assert config["official_qp_like_jacobian_authorized"] is False
    assert config["do_not_change_bending_definition"] is True
    assert config["do_not_auto_submit_without_smoke"] is True
    assert len(config["mc_data"]["batch_construction_sources"]) + len(
        config["mc_data"]["batch_validation_sources"]
    ) == 9


def test_forbidden_repairs():
    with pytest.raises(ThreeStQpMeasurementBendingBatchError, match="trusted_observable"):
        refuse_flip_s2()
    with pytest.raises(ThreeStQpMeasurementBendingBatchError, match="Stage 3"):
        refuse_residual_conditional()
    with pytest.raises(ThreeStQpMeasurementBendingBatchError, match="bending_raw"):
        refuse_change_bending()
    with pytest.raises(ThreeStQpMeasurementBendingBatchError, match="q/p-like"):
        refuse_qp_like()
    with pytest.raises(ThreeStQpMeasurementBendingBatchError, match="sigma_b"):
        refuse_invent_sigma_b()


def _track(
    *,
    source_id: str,
    y2: float,
    q_truth: float,
    q_fit: float,
    p_mev: float,
    x2: float = 0.0,
    n_mot: int = 18,
) -> dict:
    return {
        "kind": "track",
        "collection": SOURCE_COLLECTION_NAME,
        "source_id": source_id,
        "n_mot": n_mot,
        "n_ift_mot": 0,
        "ift_leak": False,
        "measurements_on_track_stations": [1] * 6 + [2] * 6 + [3] * 6,
        "station_centroids": {
            "1": {"x_mm": 0.0, "y_mm": 0.0, "z_mm": 47.4, "n": 6},
            "2": {"x_mm": x2, "y_mm": y2, "z_mm": 1237.4, "n": 6},
            "3": {"x_mm": 0.0, "y_mm": 0.0, "z_mm": 2427.4, "n": 6},
        },
        "q_over_p_fit_per_mev": q_fit,
        "q_over_p_truth_s1_per_mev": q_truth,
        "p_truth_s1_mev": p_mev,
        "truth_charge": 1.0 if q_truth > 0 else -1.0,
        "sigma_q_over_p_per_mev": 1.0e-7,
    }


def _inherited() -> dict:
    return {
        "workbook_119": {"decision": "three_st_qp_calibration_not_established"},
        "workbook_125": {"decision": "three_st_qp_measurement_bending_contract_established"},
        "pinned_calypso_sources": {"all_match": True},
    }


def test_s2j_reuses_s2i_constructor():
    config = load_config()
    plus = construct_bending(
        {
            1: {"x": 0.0, "y": 0.0, "z": 47.4, "n": 6},
            2: {"x": 0.0, "y": 0.5, "z": 1237.4, "n": 6},
            3: {"x": 0.0, "y": 0.0, "z": 2427.4, "n": 6},
        },
        config,
    )
    assert plus["bending_raw"] > 0.0
    row = decorate_track(
        _track(
            source_id="mc24_100048_00000_00049",
            y2=0.5,
            q_truth=4.0e-6,
            q_fit=4.0e-6,
            p_mev=3.0e5,
        ),
        config,
    )
    assert row["t_y_chord"] == pytest.approx(0.0)
    assert row["constructable"] is True


def test_mapping_and_fit_extra_on_synthetic_clean():
    config = load_config()
    tracks = []
    for idx in range(12):
        tracks.append(
            _track(
                source_id="mc24_100043_00400_00499",
                y2=-0.5 - 0.01 * idx,
                q_truth=-5.0e-6,
                q_fit=-5.0e-6,
                p_mev=8.0e4,
            )
        )
        tracks.append(
            _track(
                source_id="mc24_100048_00000_00049",
                y2=0.5 + 0.01 * idx,
                q_truth=5.0e-6,
                q_fit=-5.0e-6,
                p_mev=2.5e6,
            )
        )
    acc = evaluate_rows(tracks, config)
    assert acc["clean_18hit"]["sign_agree"] == 1.0
    assert acc["clean_18hit"]["class_counts"][CLASS_A] == 12
    assert acc["wb119_sign_flip"]["frac_class_a_of_fit_flip"] == 1.0
    mechs = evaluate_mechanisms(acc, config)
    assert mechs[MECH_PRESENT] == STATUS_SUPPORTED
    assert mechs[MECH_FIT] == STATUS_SUPPORTED
    decision = decide(acc, _inherited(), dumps_materialized=True, campaign="smoke", config=config)
    assert decision["decision"] == DECISION_CONTRACT
    assert decision["three_st_qp_trusted_observable"] is False
    assert decision["e_r_ift_given_bending_computed"] is False
    assert MECH_PRESENT in decision["mechanisms"]["supported_mechanisms"]
    assert MECH_FIT in decision["mechanisms"]["supported_mechanisms"]


def test_matched_source_can_reject_spectrum_effect():
    config = load_config()
    tracks = []
    for _idx in range(30):
        tracks.append(
            _track(
                source_id="mc24_100043_00400_00499",
                y2=-0.40,
                q_truth=-4.0e-6,
                q_fit=-4.0e-6,
                p_mev=3.0e5,
            )
        )
        tracks.append(
            _track(
                source_id="mc24_100044_00200_00299",
                y2=0.40,
                q_truth=4.0e-6,
                q_fit=4.0e-6,
                p_mev=3.0e5,
            )
        )
        tracks.append(
            _track(
                source_id="mc24_100047_00000_00049",
                y2=-0.40,
                q_truth=-4.0e-6,
                q_fit=-4.0e-6,
                p_mev=3.0e5,
            )
        )
        tracks.append(
            _track(
                source_id="mc24_100048_00000_00049",
                y2=0.40,
                q_truth=4.0e-6,
                q_fit=4.0e-6,
                p_mev=3.0e5,
            )
        )
    acc = evaluate_rows(tracks, config)
    matched = acc["source_stability_matched"]
    assert matched["enough"] is True
    assert matched["rel_range"] is not None
    assert matched["rel_range"] < 0.50
    mechs = evaluate_mechanisms(acc, config)
    assert mechs["raw_measurement_source_dependence"] == "rejected"


def test_high_p_gate_uses_2tev_and_stays_unresolved_on_small_n():
    config = load_config()
    assert config["binning"]["high_p_limit_label"] == "p_ge_2000gev"
    tracks = []
    for idx in range(12):
        tracks.append(
            _track(
                source_id="mc24_100043_00400_00499",
                y2=-0.50,
                q_truth=-8.0e-6,
                q_fit=-8.0e-6,
                p_mev=8.0e4,
            )
        )
        tracks.append(
            _track(
                source_id="mc24_100048_00000_00049",
                y2=0.05,
                q_truth=4.0e-7,
                q_fit=-4.0e-7,
                p_mev=2.5e6,
            )
        )
    acc = evaluate_rows(tracks, config)
    assert "p_ge_2000gev" in acc["clean_18hit"]["p_bins"]
    mechs = evaluate_mechanisms(acc, config)
    assert mechs[MECH_LIMITED] == STATUS_UNRESOLVED
    assert acc["wb119_sign_flip"]["quantified"] is False
    assert mechs[MECH_MAP] == STATUS_UNRESOLVED


def test_fit_flip_contribution_splits_raw_correct_from_raw_flip():
    config = load_config()
    tracks = [
        _track(
            source_id="mc24_100048_00000_00049",
            y2=0.50,
            q_truth=5.0e-6,
            q_fit=-5.0e-6,
            p_mev=3.0e5,
        ),
        _track(
            source_id="mc24_100043_00400_00499",
            y2=0.50,
            q_truth=-5.0e-6,
            q_fit=5.0e-6,
            p_mev=3.0e5,
        ),
    ]
    acc = evaluate_rows(tracks, config, keep_identities=True)
    wb119 = acc["wb119_sign_flip"]
    assert wb119["n_fit_flip_observed"] == 2
    assert wb119["n_fit_flip_raw_correct"] == 1
    assert wb119["n_fit_flip_raw_flip"] == 1
    assert wb119["frac_class_a_of_fit_flip"] == 0.5
    assert wb119["frac_class_c_of_fit_flip"] == 0.5
    assert acc["clean_18hit"]["class_counts"][CLASS_A] == 1
