"""Yasu-S2F mean decomposition.  Never flips S2 or opens S3."""

from __future__ import annotations

import numpy as np
import pytest

from datasets.three_st_qp_calibration import SOURCE_COLLECTION_NAME
from datasets.three_st_qp_mean_decomposition import (
    DECISION_CONTRACT,
    DECISION_RECORDED,
    MECH_POPULATION,
    MECH_RECONSTRUCTION,
    MECH_TAIL,
    ThreeStQpMeanDecompositionError,
    decide,
    evaluate_tracks,
    inherit_frozen_stage,
    load_config,
    refuse_flip_s2,
    refuse_residual_conditional,
    refuse_reweight_as_calibration,
    refuse_trim_official_mean,
)


def test_config_freezes_s2_and_s2e():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    assert inherited["workbook_119"]["decision"] == "three_st_qp_calibration_not_established"
    assert inherited["workbook_121"]["decision"] == "three_st_qp_root_cause_recorded"
    assert config["three_st_qp_trusted_observable"] is False
    assert config["do_not_trim_or_winsorize_official_mean"] is True
    assert config["do_not_submit_reconstruction_dump"] is True


def test_forbidden_repairs():
    with pytest.raises(ThreeStQpMeanDecompositionError, match="trusted_observable"):
        refuse_flip_s2()
    with pytest.raises(ThreeStQpMeanDecompositionError, match="Stage 3"):
        refuse_residual_conditional()
    with pytest.raises(ThreeStQpMeanDecompositionError, match="untrimmed"):
        refuse_trim_official_mean()
    with pytest.raises(ThreeStQpMeanDecompositionError, match="diagnostic counterfactual"):
        refuse_reweight_as_calibration()


def _track(
    *,
    q_fit: float,
    q_truth: float,
    source_id: str = "mc24_100043_00400_00499",
    event_id: int = 0,
    campaign: str = "batch",
    n_mot: int = 18,
    stations: list[int] | None = None,
) -> dict:
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
        "q_over_p_fit_per_mev": q_fit,
        "q_over_p_truth_s1_per_mev": q_truth,
        "truth_matched": True,
        "truth_reference_available": True,
        "truth_used_as_fit_seed": False,
        "truth_used_as_solution": False,
        "truth_charge": -1.0 if q_truth < 0 else 1.0,
        "p_truth_s1_mev": abs(1.0 / q_truth) if q_truth else 1.0e6,
        "tx_truth": 0.002,
        "ty_truth": 0.001,
        "n_mot": n_mot,
        "n_outlier_hits": 0,
        "chi2": 10.0,
        "ndof": 13.0,
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
        "workbook_121": {"decision": "three_st_qp_root_cause_recorded"},
    }


def _freeze(config: dict, tracks: list[dict]) -> None:
    n_flip = sum(
        1
        for row in tracks
        if (row["q_over_p_fit_per_mev"] > 0) != (row["q_over_p_truth_s1_per_mev"] > 0)
    )
    config["frozen_wb119_batch_denominator"] = {
        "n_events": len(tracks),
        "n_without_ift_tracks": len(tracks),
        "n_primary": len(tracks),
        "n_sign_flip": n_flip,
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
        _inherited(),
        dumps_materialized=True,
        campaign="smoke",
        config=config,
    )
    assert decision["decision"] == DECISION_CONTRACT
    assert decision["three_st_qp_trusted_observable"] is False
    assert decision["residual_conditional_authorized"] is False
    assert decision["official_mean_trimmed"] is False


def test_identity_and_population_mean():
    minus = [_track(q_fit=-4.0e-6, q_truth=-4.0e-6, event_id=i) for i in range(220)]
    plus = [
        _track(
            q_fit=4.0e-6,
            q_truth=4.0e-6,
            source_id="mc24_100048_00000_00049",
            event_id=i,
        )
        for i in range(80)
    ]
    tracks = minus + plus
    config = dict(load_config())
    _freeze(config, tracks)
    decision = decide(
        _split(minus),
        _split(plus),
        _inherited(),
        dumps_materialized=True,
        campaign="batch",
        config=config,
    )
    pooled = decision["pooled"]["overall"]
    assert abs(pooled["identity_residual"]) <= 1.0e-18
    assert decision["decision"] == DECISION_RECORDED
    assert decision["three_st_qp_trusted_observable"] is False
    assert decision["pooled"]["classification"]["mechanism"] == MECH_POPULATION
    assert decision["pooled"]["charge_balanced"]["diagnostic_only"] is True
    assert decision["pooled"]["charge_balanced"]["calibration_population"] is False
    assert abs(decision["pooled"]["charge_balanced"]["mu_truth"]) < 1.0e-12
    comp = decision["pooled"]["charge_composition"]
    assert abs(comp["pi_plus"] + comp["pi_minus"] - 1.0) < 1.0e-12
    assert abs(
        comp["plus_contribution_to_mu_fit"]
        + comp["minus_contribution_to_mu_fit"]
        - pooled["mu_fit"]
    ) < 1.0e-18


def test_reconstruction_bias_on_balanced_sample():
    minus = [
        _track(q_fit=-4.0e-6 - 3.0e-6, q_truth=-4.0e-6, event_id=i) for i in range(150)
    ]
    plus = [
        _track(
            q_fit=4.0e-6 - 3.0e-6,
            q_truth=4.0e-6,
            source_id="mc24_100048_00000_00049",
            event_id=i,
        )
        for i in range(150)
    ]
    tracks = minus + plus
    config = dict(load_config())
    _freeze(config, tracks)
    decision = decide(
        _split(minus),
        _split(plus),
        _inherited(),
        dumps_materialized=True,
        campaign="batch",
        config=config,
    )
    assert decision["pooled"]["classification"]["mechanism"] == MECH_RECONSTRUCTION
    assert decision["new_reconstruction_dump_authorized"] is False


def test_tail_owns_the_mean_without_redefining_it():
    rng = np.random.default_rng(0)
    minus = []
    plus = []
    for i in range(200):
        q_t = -5.0e-6
        q_f = q_t + float(rng.normal()) * 1.0e-8
        minus.append(_track(q_fit=q_f, q_truth=q_t, event_id=i))
        plus.append(
            _track(
                q_fit=-q_t + float(rng.normal()) * 1.0e-8,
                q_truth=5.0e-6,
                source_id="mc24_100048_00000_00049",
                event_id=i,
            )
        )
    minus[0] = _track(q_fit=-100.0, q_truth=-5.0e-6, event_id=0)
    tracks = minus + plus
    config = dict(load_config())
    _freeze(config, tracks)
    decision = decide(
        _split(minus),
        _split(plus),
        _inherited(),
        dumps_materialized=True,
        campaign="batch",
        config=config,
    )
    assert decision["pooled"]["classification"]["mechanism"] == MECH_TAIL
    assert decision["pooled"]["official_mean_trimmed"] is False
    assert decision["official_mean_trimmed"] is False
    share = decision["pooled"]["tails"]["top_0.001_abs_delta"]["share_of_sum_fit"]
    assert abs(share) >= 0.50
