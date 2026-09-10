"""Yasu Stage 2 3ST q/p calibration contract.  No sealed test, no truth-as-fit."""

from __future__ import annotations

import numpy as np
import pytest

from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.three_st_qp_calibration import (
    DECISION_CONTRACT,
    DECISION_ESTABLISHED,
    DECISION_NOT_ESTABLISHED,
    MECHANISM_COV,
    MECHANISM_DUMP_MISSING,
    MECHANISM_IFT_LEAK,
    MECHANISM_LTO_SUBSTITUTE,
    MECHANISM_MEAN_BIAS,
    MECHANISM_SIGN,
    OFFICIAL_TRUTH_STATION,
    SOURCE_COLLECTION_NAME,
    ThreeStQpCalibrationError,
    decide,
    evaluate_tracks,
    inherit_frozen_stage,
    load_config,
    refuse_covariance_rescale,
    refuse_drop_sign_flip,
    refuse_dummy_segmentfit,
    refuse_four_station_substitute,
    refuse_lto_substitute,
    refuse_residual_conditional,
    refuse_truth_qoverp,
    state_definition,
    truth_definition,
)


def test_config_inherits_stage0_stage1_and_freezes_bins():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    assert inherited["workbook_87"]["decision"] == "faseracts_transport_covariance_not_validated"
    assert inherited["workbook_96"]["decision"] == "ckf_qoverp_covariance_export_established"
    assert inherited["workbook_117"]["decision"] == "ckf_without_ift_definition_established"
    assert inherited["workbook_118"]["decision"] == "three_st_to_ift_prediction_established"
    assert config["source_collection"] == SOURCE_COLLECTION_NAME
    assert int(config["official_truth_station"]) == OFFICIAL_TRUTH_STATION
    assert config["residual_conditional_authorized"] is False
    assert config["do_not_drop_sign_flip_tracks"] is True
    assert config["do_not_rescale_covariance"] is True
    assert truth_definition()["not_a_fit_input"] is True
    assert state_definition()["q_over_p_signed"] is True
    assert config["binning"]["abs_q_over_p_labels"][0] == "qp_near_zero_p_ge_5tev"


def test_forbidden_repairs_are_hard_errors():
    with pytest.raises(ThreeStQpCalibrationError, match="calibration reference"):
        refuse_truth_qoverp()
    with pytest.raises(ThreeStQpCalibrationError, match="dummy"):
        refuse_dummy_segmentfit()
    with pytest.raises(ThreeStQpCalibrationError, match="LTO"):
        refuse_lto_substitute()
    with pytest.raises(ThreeStQpCalibrationError, match="4-station"):
        refuse_four_station_substitute()
    with pytest.raises(ThreeStQpCalibrationError, match="Stage 3"):
        refuse_residual_conditional()
    with pytest.raises(ThreeStQpCalibrationError, match="sign-flip"):
        refuse_drop_sign_flip()
    with pytest.raises(ThreeStQpCalibrationError, match="rescale"):
        refuse_covariance_rescale()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)
    assert MECHANISM_LTO_SUBSTITUTE == "lto_used_as_without_ift_substitute"


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
    leak: bool = False,
    matched: bool = True,
    reference: bool = True,
    is_truth: bool = False,
    truth_as_solution: bool = False,
) -> dict:
    if charge is None:
        charge = 1.0 if q_truth > 0.0 else -1.0
    if p_truth is None and q_truth != 0.0:
        p_truth = abs(1.0 / q_truth)
    return {
        "kind": "track",
        "collection": SOURCE_COLLECTION_NAME,
        "native_state": [0.1, -0.2, 0.01, 0.02, q_fit],
        "native_covariance": _cov(sigma),
        "has_covariance": True,
        "has_parameters": True,
        "is_truth": is_truth,
        "truth_used_as_fit_seed": False,
        "truth_used_as_solution": truth_as_solution,
        "station_decoded_by_fasersct_id": True,
        "n_ift_mot": 1 if leak else 0,
        "n_ift_tsos": 1 if leak else 0,
        "ift_leak": leak,
        "q_over_p_fit_per_mev": q_fit,
        "q_over_p_truth_s1_per_mev": q_truth if reference else None,
        "sigma_q_over_p_per_mev": sigma,
        "truth_matched": matched,
        "truth_reference_available": matched and reference,
        "truth_charge": charge,
        "p_truth_s1_mev": p_truth,
        "tx": tx,
        "ty": ty,
        "tx_truth": tx,
        "ty_truth": ty,
    }


def _inherited() -> dict:
    return {
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


def _gaussian_tracks(n: int, charge: float, rng: np.random.Generator) -> list[dict]:
    q_truth = 2.0e-6 * charge
    sigma = 1.0e-7
    pulls = rng.normal(0.0, 1.0, size=n)
    tracks = []
    for pull in pulls:
        q_fit = q_truth + float(pull) * sigma
        tracks.append(
            _track(
                q_fit=q_fit,
                q_truth=q_truth,
                sigma=sigma,
                charge=charge,
                tx=float(rng.normal(0.0, 0.003)),
                ty=float(rng.normal(0.0, 0.003)),
            )
        )
    return tracks


def test_smoke_contract_pass_is_physics_inconclusive():
    minus = [_track(q_fit=-2.0e-6, q_truth=-2.0e-6, charge=-1.0) for _ in range(4)]
    plus = [_track(q_fit=2.0e-6, q_truth=2.0e-6, charge=1.0) for _ in range(4)]
    decision = decide(
        _split_from_tracks(minus),
        _split_from_tracks(plus),
        {"all_match": True},
        _inherited(),
        dumps_materialized=True,
        campaign="smoke",
        config=load_config(),
    )
    assert decision["verdict"] == "PASS"
    assert decision["decision"] == DECISION_CONTRACT
    assert decision["contract_verdict"] == "PASS"
    assert decision["physics_verdict"] == "INCONCLUSIVE"
    assert decision["batch_htcondor_authorized"] is True
    assert decision["residual_conditional_authorized"] is False
    assert decision["three_st_qp_trusted_observable"] is False


def test_calibrated_batch_would_pass():
    rng = np.random.default_rng(0)
    minus = _gaussian_tracks(120, -1.0, rng)
    plus = _gaussian_tracks(120, 1.0, rng)
    decision = decide(
        _split_from_tracks(minus, n_events=120),
        _split_from_tracks(plus, n_events=120),
        {"all_match": True},
        _inherited(),
        dumps_materialized=True,
        campaign="batch",
        config=load_config(),
    )
    assert decision["verdict"] == "PASS"
    assert decision["decision"] == DECISION_ESTABLISHED
    assert decision["physics_verdict"] == "PASS"
    assert decision["residual_conditional_authorized"] is True
    assert decision["three_st_qp_trusted_observable"] is True
    assert decision["looked_at_mean_qoverp_only"] is False
    assert decision["physics"]["charge_even_delta"] is not None
    assert decision["physics"]["charge_odd_delta"] is not None


def test_missing_dump_leak_and_mean_bias_fail():
    empty = _split_from_tracks([])
    missing = decide(
        empty,
        empty,
        {"all_match": True},
        _inherited(),
        dumps_materialized=False,
        campaign="smoke",
        config=load_config(),
    )
    assert missing["verdict"] == "FAIL"
    assert missing["decision"] == DECISION_NOT_ESTABLISHED
    assert missing["mechanism"] == MECHANISM_DUMP_MISSING
    assert missing["residual_conditional_authorized"] is False

    leaked = _split_from_tracks(
        [_track(q_fit=-2.0e-6, q_truth=-2.0e-6, charge=-1.0, leak=True) for _ in range(4)]
    )
    failed = decide(
        leaked,
        leaked,
        {"all_match": True},
        _inherited(),
        dumps_materialized=True,
        campaign="smoke",
        config=load_config(),
    )
    assert failed["mechanism"] == MECHANISM_IFT_LEAK

    rng = np.random.default_rng(1)
    biased_minus = _gaussian_tracks(120, -1.0, rng)
    for row in biased_minus:
        row["q_over_p_fit_per_mev"] = float(row["q_over_p_fit_per_mev"]) + 3.0e-7
        row["native_state"][4] = row["q_over_p_fit_per_mev"]
    biased_plus = _gaussian_tracks(120, 1.0, rng)
    for row in biased_plus:
        row["q_over_p_fit_per_mev"] = float(row["q_over_p_fit_per_mev"]) + 3.0e-7
        row["native_state"][4] = row["q_over_p_fit_per_mev"]
    biased = decide(
        _split_from_tracks(biased_minus, n_events=120),
        _split_from_tracks(biased_plus, n_events=120),
        {"all_match": True},
        _inherited(),
        dumps_materialized=True,
        campaign="batch",
        config=load_config(),
    )
    assert biased["verdict"] == "FAIL"
    assert biased["physics_verdict"] == "FAIL"
    assert MECHANISM_MEAN_BIAS in biased["failure_classes"]
    assert biased["residual_conditional_authorized"] is False


def test_sign_flip_and_overcoverage_are_failures_and_tracks_are_kept():
    rng = np.random.default_rng(2)
    flipped = []
    for charge, n in ((-1.0, 120), (1.0, 120)):
        rows = _gaussian_tracks(n, charge, rng)
        for row in rows:
            row["q_over_p_fit_per_mev"] = -float(row["q_over_p_truth_s1_per_mev"])
            row["native_state"][4] = row["q_over_p_fit_per_mev"]
        flipped.append(rows)
    sign_fail = decide(
        _split_from_tracks(flipped[0], n_events=120),
        _split_from_tracks(flipped[1], n_events=120),
        {"all_match": True},
        _inherited(),
        dumps_materialized=True,
        campaign="batch",
        config=load_config(),
    )
    assert sign_fail["physics"]["n_sign_flip"] == 240
    assert MECHANISM_SIGN in sign_fail["failure_classes"]
    assert sign_fail["sign_flip_tracks_dropped"] is False

    tiny = []
    for charge in (-1.0, 1.0):
        rows = _gaussian_tracks(120, charge, rng)
        for row in rows:
            q_truth = float(row["q_over_p_truth_s1_per_mev"])
            row["q_over_p_fit_per_mev"] = q_truth + 1.0e-12
            row["native_state"][4] = row["q_over_p_fit_per_mev"]
        tiny.append(rows)
    cov_fail = decide(
        _split_from_tracks(tiny[0], n_events=120),
        _split_from_tracks(tiny[1], n_events=120),
        {"all_match": True},
        _inherited(),
        dumps_materialized=True,
        campaign="batch",
        config=load_config(),
    )
    assert MECHANISM_COV in cov_fail["failure_classes"]
    assert cov_fail["covariance_rescaled"] is False
