"""Yasu-S2E root-cause audit.  Never flips S2 or opens S3."""

from __future__ import annotations

import numpy as np
import pytest

from datasets.three_st_qp_failure_diagnosis import SOURCE_COLLECTION_NAME
from datasets.three_st_qp_root_cause_audit import (
    DECISION_CONTRACT,
    DECISION_RECORDED,
    STATUS_REJECTED,
    STATUS_SUPPORTED,
    STATUS_UNRESOLVED,
    ThreeStQpRootCauseError,
    decide,
    evaluate_tracks,
    inherit_frozen_stage,
    load_config,
    refuse_drop_nspd,
    refuse_flip_s2,
    refuse_residual_conditional,
)


def test_config_freezes_s2_and_conversion_pins():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    assert inherited["workbook_119"]["decision"] == "three_st_qp_calibration_not_established"
    assert inherited["workbook_120"]["decision"] == "three_st_qp_failure_diagnosis_recorded"
    assert config["three_st_qp_trusted_observable"] is False
    assert config["do_not_submit_reconstruction_dump"] is True
    assert config["clean_subsample"]["require_n_mot"] == 18


def test_forbidden_repairs():
    with pytest.raises(ThreeStQpRootCauseError, match="trusted_observable"):
        refuse_flip_s2()
    with pytest.raises(ThreeStQpRootCauseError, match="Stage 3"):
        refuse_residual_conditional()
    with pytest.raises(ThreeStQpRootCauseError, match="non-SPD"):
        refuse_drop_nspd()


def _cov(sigma: float = 1.0e-7) -> list[list[float]]:
    cov = np.eye(5) * 1.0e-4
    cov[4, 4] = float(sigma) ** 2
    cov[0, 4] = cov[4, 0] = 1.0e-12
    return cov.tolist()


def _track(
    *,
    q_fit: float,
    q_truth: float,
    sigma: float = 1.0e-7,
    n_mot: int = 18,
    stations: list[int] | None = None,
    chi2: float = 10.0,
    ndof: float = 13.0,
    source_id: str = "mc24_100043_00400_00499",
    event_id: int = 0,
    campaign: str = "batch",
) -> dict:
    p_truth = abs(1.0 / q_truth) if q_truth else 1.0e6
    if stations is None:
        stations = [1] * 6 + [2] * 6 + [3] * 6
    return {
        "kind": "track",
        "collection": SOURCE_COLLECTION_NAME,
        "campaign": campaign,
        "source_id": source_id,
        "run_id": 100043,
        "event_id": event_id,
        "track_index": 0,
        "native_covariance": _cov(sigma),
        "q_over_p_fit_per_mev": q_fit,
        "q_over_p_truth_s1_per_mev": q_truth,
        "sigma_q_over_p_per_mev": sigma,
        "truth_matched": True,
        "truth_reference_available": True,
        "truth_used_as_fit_seed": False,
        "truth_used_as_solution": False,
        "truth_charge": -1.0 if q_truth < 0 else 1.0,
        "p_truth_s1_mev": p_truth,
        "tx_truth": 0.002,
        "ty_truth": 0.001,
        "n_mot": n_mot,
        "n_outlier_hits": 0,
        "chi2": chi2,
        "ndof": ndof,
        "measurements_on_track_stations": stations,
    }


def _split(tracks: list[dict]) -> dict:
    return {
        "dumps_present": True,
        "events": {"n_events": max(len(tracks), 1)},
        "tracks": evaluate_tracks(tracks, load_config()),
    }


def _inherited() -> dict:
    return {
        "workbook_119": {"decision": "three_st_qp_calibration_not_established"},
        "workbook_120": {"decision": "three_st_qp_failure_diagnosis_recorded"},
    }


def test_smoke_inconclusive_never_trusts():
    minus = [_track(q_fit=-5.0e-6, q_truth=-5.0e-6, event_id=i) for i in range(4)]
    plus = [
        _track(
            q_fit=5.0e-6,
            q_truth=5.0e-6,
            source_id="mc24_100048_00000_00049",
            event_id=i,
        )
        for i in range(4)
    ]
    config = dict(load_config())
    config["frozen_wb119_smoke_denominator"] = {"n_primary": 8, "n_sign_flip": 0}
    decision = decide(
        _split(minus),
        _split(plus),
        {"all_match": True},
        _inherited(),
        dumps_materialized=True,
        campaign="smoke",
        config=config,
    )
    assert decision["decision"] == DECISION_CONTRACT
    assert decision["three_st_qp_trusted_observable"] is False
    assert decision["new_reconstruction_dump_authorized"] is False


def test_batch_can_support_a_and_d_without_flipping_s2():
    rng = np.random.default_rng(0)
    minus = []
    plus = []
    for i in range(150):
        q_t = -8.0e-6
        sigma = 1.0e-7
        q_f = q_t + float(rng.normal()) * sigma
        minus.append(_track(q_fit=q_f, q_truth=q_t, sigma=sigma, event_id=i))
        plus.append(
            _track(
                q_fit=-q_f,
                q_truth=8.0e-6,
                sigma=sigma,
                source_id="mc24_100048_00000_00049",
                event_id=i,
            )
        )
    for i in range(80):
        q_t = -2.0e-7
        sigma = 2.0e-7
        q_f = 2.0e-7 if i < 40 else q_t + 80.0 * sigma
        minus.append(_track(q_fit=q_f, q_truth=q_t, sigma=sigma, event_id=300 + i))
        plus.append(
            _track(
                q_fit=-q_t if i < 40 else 2.0e-7 + 80.0 * sigma,
                q_truth=2.0e-7,
                sigma=sigma,
                source_id="mc24_100048_00000_00049",
                event_id=300 + i,
            )
        )
    tracks = minus + plus
    n_flip = sum(
        1
        for row in tracks
        if (row["q_over_p_fit_per_mev"] > 0) != (row["q_over_p_truth_s1_per_mev"] > 0)
    )
    config = dict(load_config())
    config["frozen_wb119_batch_denominator"] = {
        "n_events": len(tracks),
        "n_without_ift_tracks": len(tracks),
        "n_primary": len(tracks),
        "n_sign_flip": n_flip,
    }
    mid = len(minus)
    decision = decide(
        _split(minus),
        _split(plus),
        {"all_match": True},
        _inherited(),
        dumps_materialized=True,
        campaign="batch",
        config=config,
    )
    assert decision["decision"] == DECISION_RECORDED
    assert decision["three_st_qp_trusted_observable"] is False
    assert decision["residual_conditional_authorized"] is False
    assert decision["C"] == STATUS_REJECTED
    assert decision["A"] in {STATUS_SUPPORTED, STATUS_UNRESOLVED}
    assert decision["nspd_tracks_dropped"] is False
    assert decision["new_reconstruction_dump_authorized"] is False
    assert decision["diagnosis"]["nspd_contribution"]["deleted"] is False
    assert mid == 230


def test_c_supports_only_gaussian_unit_rescue():
    rng = np.random.default_rng(1)
    minus = []
    plus = []
    for i in range(220):
        sigma = 1.0e-7
        q_minus = -5.0e-6 + float(rng.normal()) * (sigma * 1.0e3)
        q_plus = 5.0e-6 + float(rng.normal()) * (sigma * 1.0e3)
        minus.append(_track(q_fit=q_minus, q_truth=-5.0e-6, sigma=sigma, event_id=i))
        plus.append(
            _track(
                q_fit=q_plus,
                q_truth=5.0e-6,
                sigma=sigma,
                source_id="mc24_100048_00000_00049",
                event_id=i,
            )
        )
    tracks = minus + plus
    n_flip = sum(
        1
        for row in tracks
        if (row["q_over_p_fit_per_mev"] > 0) != (row["q_over_p_truth_s1_per_mev"] > 0)
    )
    config = dict(load_config())
    config["frozen_wb119_batch_denominator"] = {
        "n_events": len(tracks),
        "n_without_ift_tracks": len(tracks),
        "n_primary": len(tracks),
        "n_sign_flip": n_flip,
    }
    decision = decide(
        _split(minus),
        _split(plus),
        {"all_match": True},
        _inherited(),
        dumps_materialized=True,
        campaign="batch",
        config=config,
    )
    assert decision["C"] == STATUS_SUPPORTED
    assert decision["three_st_qp_trusted_observable"] is False
    assert decision["residual_conditional_authorized"] is False


def test_dirty_topology_is_reported_not_dropped():
    clean = [_track(q_fit=-5.0e-6, q_truth=-5.0e-6, event_id=i) for i in range(60)]
    dirty = [
        _track(
            q_fit=5.0e-6,
            q_truth=-5.0e-6,
            n_mot=5,
            stations=[1] * 5,
            ndof=0.0,
            chi2=0.0,
            event_id=100 + i,
        )
        for i in range(60)
    ]
    acc = evaluate_tracks(clean + dirty, load_config())
    assert acc["n_primary"] == 120
    assert acc["n_clean"] == 60
    assert acc["n_dirty"] == 60
    assert acc["n_flip"] == 60
