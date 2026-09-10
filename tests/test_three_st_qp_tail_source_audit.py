"""Yasu-S2G tail/source audit.  Never flips S2 or opens S3."""

from __future__ import annotations

import numpy as np
import pytest

from datasets.three_st_qp_calibration import SOURCE_COLLECTION_NAME
from datasets.three_st_qp_tail_source_audit import (
    DECISION_CONTRACT,
    DECISION_RECORDED,
    MECH_BROAD,
    MECH_CATASTROPHIC,
    MECH_COMPOSITION,
    MECH_REFIT,
    MECH_SOURCE,
    STATUS_REJECTED,
    STATUS_SUPPORTED,
    ThreeStQpTailSourceError,
    decide,
    evaluate_tracks,
    inherit_frozen_stage,
    load_config,
    refuse_drop_tails,
    refuse_flip_s2,
    refuse_residual_conditional,
    refuse_reweight_as_calibration,
    refuse_trim_official_mean,
)


def test_config_freezes_s2_and_s2f():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    assert inherited["workbook_119"]["decision"] == "three_st_qp_calibration_not_established"
    assert inherited["workbook_122"]["decision"] == "three_st_qp_mean_decomposition_recorded"
    assert config["three_st_qp_trusted_observable"] is False
    assert config["do_not_trim_or_winsorize_official_mean"] is True
    assert config["do_not_submit_reconstruction_dump"] is True
    assert config["do_not_treat_influence_as_official_mean"] is True
    assert config["gates"]["tail_abs_z"] == 10.0
    assert config["gates"]["extreme_abs_z"] == 100.0
    assert config["gates"]["tail_fractions"] == [0.001, 0.0001]


def test_forbidden_repairs():
    with pytest.raises(ThreeStQpTailSourceError, match="trusted_observable"):
        refuse_flip_s2()
    with pytest.raises(ThreeStQpTailSourceError, match="Stage 3"):
        refuse_residual_conditional()
    with pytest.raises(ThreeStQpTailSourceError, match="untrimmed"):
        refuse_trim_official_mean()
    with pytest.raises(ThreeStQpTailSourceError, match="diagnostic only"):
        refuse_reweight_as_calibration()
    with pytest.raises(ThreeStQpTailSourceError, match="contribution"):
        refuse_drop_tails()


def _cov(sigma: float = 1.0e-6) -> list[list[float]]:
    cov = np.eye(5) * 1.0e-4
    cov[4, 4] = float(sigma) ** 2
    cov[0, 4] = cov[4, 0] = 1.0e-12
    return cov.tolist()


def _track(
    *,
    q_fit: float,
    q_truth: float,
    sigma: float = 1.0e-6,
    source_id: str = "mc24_100043_00400_00499",
    event_id: int = 0,
    campaign: str = "batch",
    n_mot: int = 18,
    stations: list[int] | None = None,
    z_mm: float = 47.4,
    tx: float = 0.002,
    ty: float = 0.001,
    p_truth: float | None = None,
) -> dict:
    if stations is None:
        stations = [1] * 6 + [2] * 6 + [3] * 6
    if p_truth is None:
        p_truth = abs(1.0 / q_truth) if q_truth else 1.0e6
    return {
        "kind": "track",
        "collection": SOURCE_COLLECTION_NAME,
        "campaign": campaign,
        "source_id": source_id,
        "run_id": int(source_id.split("_")[1]),
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
        "tx_truth": tx,
        "ty_truth": ty,
        "z_mm": z_mm,
        "n_mot": n_mot,
        "n_outlier_hits": 0,
        "chi2": 10.0,
        "ndof": 13.0,
        "measurements_on_track": True,
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
        "workbook_122": {"decision": "three_st_qp_mean_decomposition_recorded"},
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


def _decide(construction: list[dict], validation: list[dict]) -> dict:
    tracks = construction + validation
    config = dict(load_config())
    _freeze(config, tracks)
    return decide(
        _split(construction),
        _split(validation),
        _inherited(),
        dumps_materialized=True,
        campaign="batch",
        config=config,
    )


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
    assert decision["new_reconstruction_dump_authorized"] is False


def test_identity_holds_and_influence_is_not_official():
    minus = [_track(q_fit=-4.0e-6, q_truth=-4.0e-6, event_id=i) for i in range(120)]
    plus = [
        _track(
            q_fit=4.0e-6 - 3.0e-6,
            q_truth=4.0e-6,
            source_id="mc24_100048_00000_00049",
            event_id=i,
        )
        for i in range(120)
    ]
    decision = _decide(minus, plus)
    pooled = decision["pooled"]["overall"]
    assert abs(pooled["identity_residual"]) <= 1.0e-18
    infl = decision["validation"]["report"]["tails"]["z_ge_10"]["influence"]
    assert infl["diagnostic_only"] is True
    assert infl["official_mean_redefined"] is False
    assert decision["official_mean_trimmed"] is False
    assert decision["matched_comparison"]["calibration_population"] is False


def test_catastrophic_tail_owns_validation_bias():
    body = [
        _track(
            q_fit=5.0e-6 - 1.0e-8,
            q_truth=5.0e-6,
            source_id="mc24_100048_00000_00049",
            event_id=i,
            sigma=1.0e-6,
        )
        for i in range(220)
    ]
    body[0] = _track(
        q_fit=5.0e-6 - 1.0e-3,
        q_truth=5.0e-6,
        source_id="mc24_100048_00000_00049",
        event_id=0,
        sigma=1.0e-6,
    )
    construction = [
        _track(q_fit=-5.0e-6, q_truth=-5.0e-6, event_id=i) for i in range(220)
    ]
    decision = _decide(construction, body)
    assert decision["decision"] == DECISION_RECORDED
    assert decision["structured_verdicts"][MECH_CATASTROPHIC]["status"] == STATUS_SUPPORTED
    assert decision["structured_verdicts"][MECH_BROAD]["status"] == STATUS_REJECTED
    infl = decision["validation"]["report"]["tails"]["top_0.001_abs_delta"]["influence"]
    assert infl["official_mean_redefined"] is False
    assert abs(infl["remaining_mu_bias"]) < 1.0e-7


def test_broad_shift_survives_tail_removal():
    construction = [
        _track(q_fit=-5.0e-6, q_truth=-5.0e-6, event_id=i) for i in range(220)
    ]
    validation = [
        _track(
            q_fit=5.0e-6 - 3.0e-6,
            q_truth=5.0e-6,
            source_id="mc24_100048_00000_00049",
            event_id=i,
            sigma=2.0e-6,
        )
        for i in range(220)
    ]
    decision = _decide(construction, validation)
    assert decision["structured_verdicts"][MECH_BROAD]["status"] == STATUS_SUPPORTED
    assert decision["structured_verdicts"][MECH_CATASTROPHIC]["status"] == STATUS_REJECTED
    assert decision["validation"]["report"]["tails"]["z_ge_10"]["n"] == 0


def test_source_specific_same_charge_disagreement():
    construction = [
        _track(q_fit=-5.0e-6, q_truth=-5.0e-6, event_id=i) for i in range(220)
    ]
    plus_a = [
        _track(
            q_fit=5.0e-6 - 8.0e-6,
            q_truth=5.0e-6,
            source_id="mc24_100048_00000_00049",
            event_id=i,
            sigma=2.0e-6,
        )
        for i in range(120)
    ]
    plus_b = [
        _track(
            q_fit=5.0e-6 + 1.0e-6,
            q_truth=5.0e-6,
            source_id="mc24_100044_00200_00299",
            event_id=i,
            sigma=2.0e-6,
        )
        for i in range(120)
    ]
    decision = _decide(construction, plus_a + plus_b)
    assert decision["structured_verdicts"][MECH_SOURCE]["status"] == STATUS_SUPPORTED


def test_composition_explains_split_when_matched_cells_agree():
    def _row(source: str, q_t: float, bias: float, event_id: int, p: float) -> dict:
        return _track(
            q_fit=q_t + bias,
            q_truth=q_t,
            source_id=source,
            event_id=event_id,
            sigma=2.0e-6,
            p_truth=p,
            tx=0.002,
            ty=0.001,
        )

    construction = []
    validation = []
    for i in range(160):
        construction.append(_row("mc24_100043_00400_00499", -3.33e-7, -2.0e-6, i, 3.0e6))
        validation.append(_row("mc24_100047_00000_00049", -3.33e-7, -2.0e-6, i, 3.0e6))
    for i in range(60):
        construction.append(_row("mc24_100043_00400_00499", -1.0e-5, 0.0, 200 + i, 1.0e5))
        validation.append(_row("mc24_100047_00000_00049", -1.0e-5, 0.0, 200 + i, 1.0e5))
    for i in range(60):
        construction.append(_row("mc24_100044_00200_00299", 3.33e-7, -2.0e-6, i, 3.0e6))
    for i in range(160):
        validation.append(_row("mc24_100048_00000_00049", 3.33e-7, -2.0e-6, i, 3.0e6))
    for i in range(160):
        construction.append(_row("mc24_100044_00200_00299", 1.0e-5, 0.0, 200 + i, 1.0e5))
    for i in range(60):
        validation.append(_row("mc24_100048_00000_00049", 1.0e-5, 0.0, 200 + i, 1.0e5))
    decision = _decide(construction, validation)
    assert decision["matched_comparison"]["diagnostic_only"] is True
    assert decision["structured_verdicts"][MECH_COMPOSITION]["status"] == STATUS_SUPPORTED


def test_refit_proxy_authorizes_dump_only_when_it_owns_the_bias():
    construction = [
        _track(q_fit=-5.0e-6, q_truth=-5.0e-6, event_id=i) for i in range(220)
    ]
    usual = [
        _track(
            q_fit=5.0e-6 - 1.0e-8,
            q_truth=5.0e-6,
            source_id="mc24_100048_00000_00049",
            event_id=i,
            z_mm=47.4,
        )
        for i in range(220)
    ]
    usual[0] = _track(
        q_fit=5.0e-6 - 1.0e-3,
        q_truth=5.0e-6,
        source_id="mc24_100048_00000_00049",
        event_id=0,
        z_mm=-1860.15,
    )
    authorized = _decide(construction, usual)
    assert authorized["structured_verdicts"][MECH_REFIT]["status"] == STATUS_SUPPORTED
    assert authorized["new_reconstruction_dump_authorized"] is True
    assert authorized["diagnostic_export_contract"]["authorized"] is True
    assert "kf_refit_succeeded" in authorized["diagnostic_export_contract"]["required_fields"]

    usual[0] = _track(
        q_fit=5.0e-6 - 1.0e-3,
        q_truth=5.0e-6,
        source_id="mc24_100048_00000_00049",
        event_id=0,
        z_mm=47.4,
    )
    denied = _decide(construction, usual)
    assert denied["structured_verdicts"][MECH_REFIT]["status"] == STATUS_REJECTED
    assert denied["new_reconstruction_dump_authorized"] is False
    assert denied["diagnostic_export_contract"]["authorized"] is False
