"""Yasu-S2D failure-mechanism diagnosis.  Never flips S2 or opens S3."""

from __future__ import annotations

import numpy as np
import pytest

from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.three_st_qp_failure_diagnosis import (
    DECISION_CONTRACT,
    DECISION_RECORDED,
    MECHANISM_COVARIANCE,
    MECHANISM_CURVATURE,
    MECHANISM_MIXED,
    MECHANISM_REFERENCE,
    SOURCE_COLLECTION_NAME,
    ThreeStQpFailureDiagnosisError,
    decide,
    evaluate_tracks,
    inherit_frozen_stage,
    load_config,
    refuse_covariance_rescale,
    refuse_drop_sign_flip,
    refuse_flip_s2,
    refuse_residual_conditional,
    refuse_truth_qoverp,
)


def test_config_freezes_s2_fail_and_bins():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    assert inherited["workbook_119"]["decision"] == "three_st_qp_calibration_not_established"
    assert inherited["workbook_117"]["decision"] == "ckf_without_ift_definition_established"
    assert config["three_st_qp_trusted_observable"] is False
    assert config["residual_conditional_authorized"] is False
    assert config["do_not_flip_s2_to_pass"] is True
    assert len(config["focus_identities"]) == 6
    assert config["binning"]["p_truth_labels"][-1] == "p_ge_2000gev"
    assert config["frozen_wb119_batch_denominator"]["n_primary"] == 3398774


def test_forbidden_repairs_are_hard_errors():
    with pytest.raises(ThreeStQpFailureDiagnosisError, match="calibration reference"):
        refuse_truth_qoverp()
    with pytest.raises(ThreeStQpFailureDiagnosisError, match="sign-flip"):
        refuse_drop_sign_flip()
    with pytest.raises(ThreeStQpFailureDiagnosisError, match="rescale"):
        refuse_covariance_rescale()
    with pytest.raises(ThreeStQpFailureDiagnosisError, match="Stage 3"):
        refuse_residual_conditional()
    with pytest.raises(ThreeStQpFailureDiagnosisError, match="trusted_observable"):
        refuse_flip_s2()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


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
    charge: float | None = None,
    p_truth: float | None = None,
    tx: float = 0.002,
    ty: float = 0.001,
    n_mot: int = 18,
    stations: list[int] | None = None,
    chi2: float = 13.0,
    ndof: float = 13.0,
    source_id: str = "mc24_100043_00400_00499",
    event_id: int = 0,
) -> dict:
    if charge is None:
        charge = 1.0 if q_truth > 0.0 else -1.0
    if p_truth is None and q_truth != 0.0:
        p_truth = abs(1.0 / q_truth)
    p_fit = abs(1.0 / q_fit) if q_fit != 0.0 else None
    if stations is None:
        stations = [1] * 6 + [2] * 6 + [3] * 6
    return {
        "kind": "track",
        "collection": SOURCE_COLLECTION_NAME,
        "source_id": source_id,
        "run_id": 100043,
        "event_id": event_id,
        "track_index": 0,
        "native_state": [0.1, -0.2, 0.01, 0.02, q_fit],
        "native_covariance": _cov(sigma),
        "has_covariance": True,
        "has_parameters": True,
        "is_truth": False,
        "truth_used_as_fit_seed": False,
        "truth_used_as_solution": False,
        "official_truth_station": 1,
        "q_over_p_fit_per_mev": q_fit,
        "q_over_p_truth_s1_per_mev": q_truth,
        "q_over_p_truth_production_per_mev": q_truth,
        "sigma_q_over_p_per_mev": sigma,
        "truth_matched": True,
        "truth_reference_available": True,
        "truth_charge": charge,
        "p_truth_s1_mev": p_truth,
        "p_fit_mev": p_fit,
        "tx": tx,
        "ty": ty,
        "tx_truth": tx,
        "ty_truth": ty,
        "z_mm": 13.47,
        "n_mot": n_mot,
        "n_outlier_hits": 0,
        "chi2": chi2,
        "ndof": ndof,
        "measurements_on_track_stations": stations,
        "truth_station_momenta": [
            {"station": 0, "p": p_truth, "available": True},
            {"station": 1, "p": p_truth, "available": True},
            {"station": 2, "p": p_truth, "available": True},
            {"station": 3, "p": p_truth, "available": True},
        ],
    }


def _inherited() -> dict:
    return {
        "workbook_119": {"decision": "three_st_qp_calibration_not_established"},
        "workbook_117": {"decision": "ckf_without_ift_definition_established"},
        "workbook_118": {"decision": "three_st_to_ift_prediction_established"},
    }


def _split_from_tracks(tracks: list[dict], *, n_events: int | None = None) -> dict:
    config = load_config()
    evaluated = evaluate_tracks(tracks, config)
    n_ev = int(n_events if n_events is not None else max(len(tracks), 1))
    return {
        "dumps_present": True,
        "events": {
            "n_events": n_ev,
            "n_events_selection_loss": 0,
            "n_events_station_decoded": n_ev,
            "n_events_sdo_map_present": n_ev,
            "n_events_truth_particles_present": n_ev,
            "n_events_sct_hits_present": n_ev,
            "n_without_ift_tracks_from_events": len(tracks),
            "tracks_per_event": float(len(tracks)) / float(n_ev),
        },
        "tracks": evaluated,
    }


def test_smoke_contract_is_inconclusive_and_never_trusts():
    minus = [
        _track(q_fit=-2.0e-6, q_truth=-2.0e-6, charge=-1.0, event_id=i) for i in range(4)
    ]
    plus = [
        _track(
            q_fit=2.0e-6,
            q_truth=2.0e-6,
            charge=1.0,
            source_id="mc24_100048_00000_00049",
            event_id=i,
        )
        for i in range(4)
    ]
    config = load_config()
    # Bypass frozen smoke denominator for this unit fixture.
    config = dict(config)
    config["frozen_wb119_smoke_denominator"] = {"n_primary": 8, "n_sign_flip": 0}
    decision = decide(
        _split_from_tracks(minus),
        _split_from_tracks(plus),
        {"all_match": True},
        _inherited(),
        dumps_materialized=True,
        campaign="smoke",
        config=config,
    )
    assert decision["verdict"] == "PASS"
    assert decision["decision"] == DECISION_CONTRACT
    assert decision["diagnosis_verdict"] == "INCONCLUSIVE"
    assert decision["three_st_qp_trusted_observable"] is False
    assert decision["residual_conditional_authorized"] is False
    assert decision["s2_flipped_to_pass"] is False


def test_batch_records_mixed_and_never_authorizes_s3():
    rng = np.random.default_rng(0)
    minus = []
    plus = []
    # Low-p calibrated tracks.
    for i in range(120):
        q_t = -5.0e-6
        sigma = 1.0e-7
        q_f = q_t + float(rng.normal()) * sigma
        minus.append(_track(q_fit=q_f, q_truth=q_t, sigma=sigma, charge=-1.0, event_id=i))
        q_t = 5.0e-6
        q_f = q_t + float(rng.normal()) * sigma
        plus.append(
            _track(
                q_fit=q_f,
                q_truth=q_t,
                sigma=sigma,
                charge=1.0,
                source_id="mc24_100048_00000_00049",
                event_id=i,
            )
        )
    # High-p sign flips + fat pulls (both curvature and covariance).
    for i in range(80):
        q_t = -2.0e-7
        sigma = 1.0e-7
        q_f = 2.0e-7 if i < 50 else q_t + 50.0 * sigma
        minus.append(
            _track(q_fit=q_f, q_truth=q_t, sigma=sigma, charge=-1.0, event_id=200 + i)
        )
        plus.append(
            _track(
                q_fit=-q_t if i < 50 else q_t + 50.0 * sigma,
                q_truth=2.0e-7,
                sigma=sigma,
                charge=1.0,
                source_id="mc24_100048_00000_00049",
                event_id=200 + i,
            )
        )
    config = load_config()
    config = dict(config)
    n = 400
    n_flip = sum(
        1
        for row in minus + plus
        if (row["q_over_p_fit_per_mev"] > 0) != (row["q_over_p_truth_s1_per_mev"] > 0)
    )
    config["frozen_wb119_batch_denominator"] = {
        "n_events": n,
        "n_without_ift_tracks": n,
        "n_primary": n,
        "n_sign_flip": n_flip,
    }
    decision = decide(
        _split_from_tracks(minus, n_events=200),
        _split_from_tracks(plus, n_events=200),
        {"all_match": True},
        _inherited(),
        dumps_materialized=True,
        campaign="batch",
        config=config,
    )
    assert decision["decision"] == DECISION_RECORDED
    assert decision["three_st_qp_trusted_observable"] is False
    assert decision["residual_conditional_authorized"] is False
    assert decision["mechanism_verdict"] in {
        MECHANISM_MIXED,
        MECHANISM_CURVATURE,
        MECHANISM_COVARIANCE,
    }
    assert MECHANISM_CURVATURE in decision["supported_mechanisms"] or MECHANISM_COVARIANCE in (
        decision["supported_mechanisms"]
    )


def test_reference_identity_failure_is_detected():
    bad = [
        _track(q_fit=-2.0e-6, q_truth=-2.0e-6, charge=-1.0, event_id=i) for i in range(220)
    ]
    for row in bad:
        row["p_truth_s1_mev"] = 1.0  # breaks q*p == 1
        row["truth_station_momenta"][1]["p"] = 1.0
    good = [
        _track(
            q_fit=2.0e-6,
            q_truth=2.0e-6,
            charge=1.0,
            source_id="mc24_100048_00000_00049",
            event_id=i,
        )
        for i in range(220)
    ]
    config = load_config()
    config = dict(config)
    config["frozen_wb119_batch_denominator"] = {
        "n_events": 440,
        "n_without_ift_tracks": 440,
        "n_primary": 440,
        "n_sign_flip": 0,
    }
    decision = decide(
        _split_from_tracks(bad, n_events=220),
        _split_from_tracks(good, n_events=220),
        {"all_match": True},
        _inherited(),
        dumps_materialized=True,
        campaign="batch",
        config=config,
    )
    assert MECHANISM_REFERENCE in decision["supported_mechanisms"]
    assert decision["mechanism_verdict"] == MECHANISM_REFERENCE
    assert decision["three_st_qp_trusted_observable"] is False
