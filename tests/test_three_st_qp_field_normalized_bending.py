"""Yasu-S2K field-normalized bending.  Never flips S2 or opens S3."""

from __future__ import annotations

import math

import pytest

from datasets.faser_field_table import (
    K_GEV_PER_TM,
    FaserFieldTable,
    lock_field_unit_sign_contract,
    qp_bending_proxy_per_mev,
)
from datasets.three_st_qp_calibration import SOURCE_COLLECTION_NAME
from datasets.three_st_qp_field_normalized_bending import (
    DECISION_CONTRACT,
    STATUS_REJECTED,
    STATUS_SUPPORTED,
    ThreeStQpFieldNormalizedBendingError,
    construct_qp_bending_proxy,
    decide,
    evaluate_rows,
    evaluate_verdicts,
    inherit_frozen_stage,
    load_config,
    refuse_empirical_scale,
    refuse_fitted_qp,
    refuse_flip_s2,
    refuse_residual_conditional,
    refuse_seed_jacobian,
    refuse_two_station_surrogate,
)


def test_config_inherits_wb126_and_freezes_flags():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    assert inherited["workbook_119"]["decision"] == "three_st_qp_calibration_not_established"
    assert inherited["workbook_125"]["decision"] == "three_st_qp_measurement_bending_contract_established"
    assert inherited["workbook_126"]["decision"] == "three_st_qp_measurement_bending_batch_recorded"
    assert config["three_st_qp_trusted_observable"] is False
    assert config["residual_conditional_authorized"] is False
    assert config["official_qp_like_jacobian_authorized"] is False
    assert config["conversion"]["k_gev_per_tm"] == K_GEV_PER_TM
    assert config["conversion"]["fit_free_scale_from_truth"] is False
    assert config["path"]["name"] == "s1_s2_s3_measurement_chords"
    assert config["field"]["dipole_scale"] == 1.0
    assert len(config["mc_data"]["batch_construction_sources"]) + len(
        config["mc_data"]["batch_validation_sources"]
    ) == 9


def test_forbidden_repairs():
    with pytest.raises(ThreeStQpFieldNormalizedBendingError, match="trusted_observable"):
        refuse_flip_s2()
    with pytest.raises(ThreeStQpFieldNormalizedBendingError, match="Stage 3"):
        refuse_residual_conditional()
    with pytest.raises(ThreeStQpFieldNormalizedBendingError, match="fitted q/p"):
        refuse_fitted_qp()
    with pytest.raises(ThreeStQpFieldNormalizedBendingError, match="free scale"):
        refuse_empirical_scale()
    with pytest.raises(ThreeStQpFieldNormalizedBendingError, match="1.13"):
        refuse_seed_jacobian()
    with pytest.raises(ThreeStQpFieldNormalizedBendingError, match="two-station"):
        refuse_two_station_surrogate()


def test_proxy_formula_and_refusals():
    config = load_config()
    bending = 0.001
    integral = -1.2
    built = construct_qp_bending_proxy(bending, integral, config)
    assert built["empirical_scale_applied"] is False
    assert built["used_fitted_q_over_p"] is False
    expected = qp_bending_proxy_per_mev(bending, integral)
    assert built["qp_bending_proxy_per_mev"] == pytest.approx(expected)
    assert expected > 0.0
    with pytest.raises(ThreeStQpFieldNormalizedBendingError, match="fitted q/p"):
        construct_qp_bending_proxy(bending, integral, config, q_over_p_fit=1.0e-6)
    with pytest.raises(ThreeStQpFieldNormalizedBendingError, match="truth q/p"):
        construct_qp_bending_proxy(bending, integral, config, q_over_p_truth=1.0e-6)
    with pytest.raises(ThreeStQpFieldNormalizedBendingError, match="free scale"):
        construct_qp_bending_proxy(bending, integral, config, empirical_scale=1.4)


def _track(
    *,
    source_id: str,
    y2: float,
    q_truth: float,
    q_fit: float,
    p_mev: float,
    n_mot: int = 18,
    stations=None,
    centroids=True,
) -> dict:
    row = {
        "kind": "track",
        "collection": SOURCE_COLLECTION_NAME,
        "source_id": source_id,
        "n_mot": n_mot,
        "n_ift_mot": 0,
        "ift_leak": False,
        "measurements_on_track_stations": stations or ([1] * 6 + [2] * 6 + [3] * 6),
        "q_over_p_fit_per_mev": q_fit,
        "q_over_p_truth_s1_per_mev": q_truth,
        "p_truth_s1_mev": p_mev,
        "truth_charge": 1.0 if q_truth > 0 else -1.0,
        "sigma_q_over_p_per_mev": 1.0e-7,
        "truth_station_momenta": [
            {"station": 1, "tx": 0.0, "ty": 0.0, "available": True},
        ],
    }
    if centroids:
        row["station_centroids"] = {
            "1": {"x_mm": 0.0, "y_mm": 0.0, "z_mm": 47.4, "n": 6},
            "2": {"x_mm": 0.0, "y_mm": y2, "z_mm": 1237.4, "n": 6},
            "3": {"x_mm": 0.0, "y_mm": 0.0, "z_mm": 2427.4, "n": 6},
        }
    return row


def _inherited() -> dict:
    return {
        "workbook_119": {"decision": "three_st_qp_calibration_not_established"},
        "workbook_125": {"decision": "three_st_qp_measurement_bending_contract_established"},
        "workbook_126": {"decision": "three_st_qp_measurement_bending_batch_recorded"},
        "pinned_calypso_sources": {"all_match": True},
    }


def test_uniform_field_closes_on_synthetic_sagitta():
    config = load_config()
    field = FaserFieldTable.from_uniform(-0.55)
    # Positive sagitta => +bending_raw => +charge.  Choose q_truth from the
    # field-derived formula so slope is identically 1 without a free scale.
    tracks = []
    for idx, y2 in enumerate((0.20, 0.30, 0.45, 0.60, 0.80, 1.10, 1.40, 1.80)):
        proto = _track(
            source_id="mc24_100043_00400_00499",
            y2=y2,
            q_truth=1.0e-6,
            q_fit=1.0e-6,
            p_mev=8.0e4,
        )
        from datasets.three_st_qp_measurement_bending_batch import decorate_track

        decorated = decorate_track(proto, config)
        integral = field.path_integral_s1_s2_s3(
            {"x": 0.0, "y": 0.0, "z": 47.4},
            {"x": 0.0, "y": y2, "z": 1237.4},
            {"x": 0.0, "y": 0.0, "z": 2427.4},
            n_steps=int(config["path"]["n_steps_per_segment"]),
        )["I_yz_tm"]
        truth = qp_bending_proxy_per_mev(float(decorated["bending_raw"]), integral)
        proto["q_over_p_truth_s1_per_mev"] = truth
        proto["q_over_p_fit_per_mev"] = truth
        proto["p_truth_s1_mev"] = abs(1.0 / truth)
        proto["truth_charge"] = 1.0
        proto["source_id"] = "mc24_100043_00400_00499" if idx % 2 == 0 else "mc24_100048_00000_00049"
        tracks.append(proto)
    acc = evaluate_rows(tracks, config, field)
    ols = acc["clean_18hit"]["ols_proxy_vs_qp_truth"]
    assert acc["empirical_scale_applied"] is False
    assert acc["clean_18hit"]["sign_agree"] == 1.0
    assert ols["slope"] == pytest.approx(1.0, abs=1.0e-6)
    assert ols["intercept"] == pytest.approx(0.0, abs=1.0e-12)
    field_contract = {
        "supported": True,
        "status": "supported",
        "axis_probe_is_not_jacobian": True,
    }
    verdicts = evaluate_verdicts(acc, field_contract, config)
    assert verdicts["field_unit_sign_contract"] == STATUS_SUPPORTED
    assert verdicts["field_normalized_bending_response"] == STATUS_SUPPORTED
    decision = decide(
        acc,
        _inherited(),
        field_contract,
        dumps_materialized=True,
        campaign="smoke",
        config=config,
    )
    assert decision["decision"] == DECISION_CONTRACT
    assert decision["three_st_qp_trusted_observable"] is False
    assert decision["residual_conditional_authorized"] is False
    assert decision["official_qp_like_jacobian_authorized"] is False
    assert decision["trusted_momentum"] is False
    assert decision["empirical_scale_applied"] is False


def test_wrong_scale_rejects_response_without_refitting():
    config = load_config()
    field = FaserFieldTable.from_uniform(-0.55)
    from datasets.three_st_qp_measurement_bending_batch import decorate_track

    tracks = []
    for y2 in (0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5):
        proto = _track(
            source_id="mc24_100043_00400_00499",
            y2=y2,
            q_truth=2.0e-6,
            q_fit=2.0e-6,
            p_mev=5.0e5,
        )
        decorated = decorate_track(proto, config)
        integral = field.path_integral_s1_s2_s3(
            {"x": 0.0, "y": 0.0, "z": 47.4},
            {"x": 0.0, "y": y2, "z": 1237.4},
            {"x": 0.0, "y": 0.0, "z": 2427.4},
            n_steps=int(config["path"]["n_steps_per_segment"]),
        )["I_yz_tm"]
        true_proxy = qp_bending_proxy_per_mev(float(decorated["bending_raw"]), integral)
        proto["q_over_p_truth_s1_per_mev"] = 0.5 * true_proxy
        proto["q_over_p_fit_per_mev"] = 0.5 * true_proxy
        proto["p_truth_s1_mev"] = abs(1.0 / (0.5 * true_proxy))
        tracks.append(proto)
    acc = evaluate_rows(tracks, config, field)
    slope = acc["clean_18hit"]["ols_proxy_vs_qp_truth"]["slope"]
    assert slope is not None
    assert abs(slope - 1.0) > 0.25
    verdicts = evaluate_verdicts(acc, {"supported": True, "status": "supported"}, config)
    assert verdicts["field_normalized_bending_response"] == STATUS_REJECTED
    assert verdicts["ckf_independent_physically_scaled_transferable_3st_curvature_proxy"] is False
    assert acc["empirical_scale_applied"] is False


def test_missingness_is_selection_not_failure():
    config = load_config()
    config["gates"] = dict(config["gates"])
    config["gates"]["min_fit_flip_no_bending"] = 10
    field = FaserFieldTable.from_uniform(-0.55)
    tracks = []
    for idx in range(40):
        tracks.append(
            _track(
                source_id="mc24_100043_00400_00499",
                y2=0.4,
                q_truth=5.0e-6,
                q_fit=-5.0e-6,
                p_mev=3.0e5,
                n_mot=12,
                stations=[1] * 6 + [2] * 6,
                centroids=False,
            )
        )
    for idx in range(40):
        tracks.append(
            _track(
                source_id="mc24_100048_00000_00049",
                y2=0.4,
                q_truth=5.0e-6,
                q_fit=5.0e-6,
                p_mev=3.0e5,
            )
        )
    acc = evaluate_rows(tracks, config, field)
    missing = acc["missingness_all"]
    assert missing["two_station_surrogate_used"] is False
    assert missing["classified_as_measurement_failure"] is False
    assert missing["classified_as_reconstruction_failure"] is False
    assert missing["n_fit_flip_no_bending"] == 40
    assert missing["availability_rate_gap"] == pytest.approx(1.0)
    assert "(1, 2)" in missing["station_patterns"]
    verdicts = evaluate_verdicts(acc, {"supported": True, "status": "supported"}, config)
    assert verdicts["bending_availability_selection"] == STATUS_SUPPORTED


def test_uniform_field_sign_contract_probe():
    table = FaserFieldTable.from_uniform(-0.55)
    contract = lock_field_unit_sign_contract(table)
    assert contract["bx_sign_negative"] is True
    assert contract["axis_probe_I_negative"] is True
    assert contract["axis_probe_is_not_jacobian"] is True
    assert contract["unofficial_constants_used_as_jacobian"] is False
    assert math.isclose(contract["source_contract"]["k_gev_per_tm"], K_GEV_PER_TM)
