"""Yasu-S2H CKF/KF-refit provenance.  Never flips S2 or opens S3."""

from __future__ import annotations

import pytest

from datasets.three_st_qp_calibration import SOURCE_COLLECTION_NAME
from datasets.three_st_qp_refit_provenance import (
    DECISION_CONTRACT,
    DECISION_RECORDED,
    KIND_HOLE,
    KIND_MOT,
    MECH_FALLBACK,
    MECH_FRONT,
    STATUS_REJECTED,
    STATUS_SUPPORTED,
    ThreeStQpRefitProvenanceError,
    classify_persisted_front,
    decide,
    evaluate_provenance,
    evaluate_tracks,
    inherit_frozen_stage,
    load_config,
    refuse_change_fitter,
    refuse_flip_s2,
    refuse_replace_focus,
    refuse_residual_conditional,
)


def test_config_freezes_s2_s2g_and_focus():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    assert inherited["workbook_119"]["decision"] == "three_st_qp_calibration_not_established"
    assert inherited["workbook_123"]["decision"] == "three_st_qp_tail_source_recorded"
    assert config["three_st_qp_trusted_observable"] is False
    assert config["residual_conditional_authorized"] is False
    assert config["do_not_flip_s2_to_pass"] is True
    assert config["do_not_change_fitter_seed_hits_geometry"] is True
    assert config["do_not_replace_focus_after_seeing_results"] is True
    assert config["do_not_submit_large_reconstruction_dump"] is True
    assert len(config["focus_identities"]) == 20
    assert len({(r["source_id"], r["skip_index"], r["track_index"]) for r in config["focus_identities"]}) == 20
    assert sum(1 for row in config["focus_identities"] if row["role"] == "frozen_wb119") == 6
    assert inherited["pinned_calypso_sources"]["all_match"] is True


def test_forbidden_repairs():
    with pytest.raises(ThreeStQpRefitProvenanceError, match="trusted_observable"):
        refuse_flip_s2()
    with pytest.raises(ThreeStQpRefitProvenanceError, match="Stage 3"):
        refuse_residual_conditional()
    with pytest.raises(ThreeStQpRefitProvenanceError, match="fitter"):
        refuse_change_fitter()
    with pytest.raises(ThreeStQpRefitProvenanceError, match="focus"):
        refuse_replace_focus()


def _track(
    *,
    z_mm: float,
    kind: str,
    source_id: str = "mc24_100048_00000_00049",
    event_id: int = 50,
    role: str = "validation_s2_front_dirty",
    n_mot: int = 12,
    stations: list[int] | None = None,
    q_fit: float = 4.0e-6,
    q_truth: float = 5.0e-6,
    sigma: float = 1.0e-6,
    diagnostic_q: float | None = None,
    diagnostic_sigma: float | None = None,
    diagnostic_z: float | None = None,
    diagnostic_ok: bool | None = None,
    chi2: float | None = None,
    ndof: float | None = None,
) -> dict:
    if stations is None:
        stations = [2] * 6 + [3] * 6 if z_mm > 1000 else [1] * 6 + [2] * 6 + [3] * 6
    row = {
        "kind": "track",
        "collection": SOURCE_COLLECTION_NAME,
        "source_id": source_id,
        "run_id": int(source_id.split("_")[1]),
        "event_id": event_id,
        "skip_index": event_id,
        "track_index": 0,
        "role": role,
        "z_mm": z_mm,
        "persisted_front_kind": kind,
        "front_is_hole": kind == KIND_HOLE,
        "front_is_measurement": kind == KIND_MOT,
        "front_fit_quality_chi2": -99.0 if kind == KIND_HOLE else 1.2,
        "front_fit_quality_ndof": 0.0 if kind == KIND_HOLE else 1.0,
        "n_mot": n_mot,
        "n_tsos": n_mot + 1,
        "measurements_on_track_stations": stations,
        "q_over_p_fit_per_mev": q_fit,
        "q_over_p_truth_s1_per_mev": q_truth,
        "sigma_q_over_p_per_mev": sigma,
        "kf_refit_attempted": True,
        "diagnostic_refit_attempted": diagnostic_ok is not None,
        "diagnostic_refit_succeeded": diagnostic_ok,
        "diagnostic_q_over_p_per_mev": diagnostic_q,
        "diagnostic_sigma_q_over_p_per_mev": diagnostic_sigma,
        "diagnostic_z_mm": diagnostic_z,
        "truth_used_as_fit_seed": False,
        "truth_used_as_solution": False,
        "chi2": chi2,
        "ndof": ndof,
    }
    return row


def _inherited() -> dict:
    return {
        "workbook_119": {"decision": "three_st_qp_calibration_not_established"},
        "workbook_123": {"decision": "three_st_qp_tail_source_recorded"},
    }


def _decide(tracks: list[dict], campaign: str = "batch") -> dict:
    config = load_config()
    acc = evaluate_tracks(tracks, config)
    report = evaluate_provenance(acc, config)
    return decide(
        report,
        _inherited(),
        dumps_materialized=True,
        campaign=campaign,
        config=config,
    )


def test_classifies_hole99_versus_measurement():
    config = load_config()
    hole = classify_persisted_front(
        {
            "front_is_hole": True,
            "front_is_measurement": False,
            "front_fit_quality_chi2": -99.0,
            "front_fit_quality_ndof": 0.0,
        },
        config,
    )
    mot = classify_persisted_front(
        {
            "front_is_hole": False,
            "front_is_measurement": True,
            "front_fit_quality_chi2": 2.0,
            "front_fit_quality_ndof": 1.0,
        },
        config,
    )
    assert hole == KIND_HOLE
    assert mot == KIND_MOT


def test_s2_front_first_mot_rejects_fallback():
    tracks = [
        _track(z_mm=1205.0, kind=KIND_MOT, event_id=50 + i, n_mot=12)
        for i in range(5)
    ]
    tracks += [
        _track(
            z_mm=18.0,
            kind=KIND_MOT,
            event_id=i,
            role="validation_s1_front_control",
            n_mot=18,
            stations=[1] * 6 + [2] * 6 + [3] * 6,
            source_id="mc24_100043_00400_00499",
        )
        for i in range(3)
    ]
    decision = _decide(tracks)
    assoc = decision["structured_verdicts"][MECH_FALLBACK]
    assert assoc["status"] == STATUS_REJECTED
    assert assoc["legal_successful_refit_first_mot"] is True
    assert decision["decision"] == DECISION_RECORDED
    assert "successful refit" in (decision["fallback_note"] or "")
    assert decision["three_st_qp_trusted_observable"] is False
    assert decision["residual_conditional_authorized"] is False
    assert decision["s2_flipped_to_pass"] is False


def test_s2_front_hole99_supports_fallback():
    tracks = [
        _track(
            z_mm=1210.0,
            kind=KIND_HOLE,
            event_id=50 + i,
            n_mot=8,
            stations=[2] * 4 + [3] * 4,
        )
        for i in range(5)
    ]
    decision = _decide(tracks)
    assert decision["structured_verdicts"][MECH_FALLBACK]["status"] == STATUS_SUPPORTED
    assert decision["official_mechanism"] in {MECH_FALLBACK, "mixed/inconclusive"}


def test_persisted_front_mismatch_is_prioritized():
    tracks = [
        _track(
            z_mm=1205.0,
            kind=KIND_MOT,
            event_id=50 + i,
            diagnostic_ok=True,
            diagnostic_q=1.0e-6,
            diagnostic_sigma=2.0e-6,
            diagnostic_z=18.0,
        )
        for i in range(4)
    ]
    decision = _decide(tracks)
    assert decision["structured_verdicts"][MECH_FRONT]["status"] == STATUS_SUPPORTED


def test_smoke_inconclusive_never_trusts():
    tracks = [
        _track(
            z_mm=18.0,
            kind=KIND_MOT,
            event_id=i,
            role="validation_s1_front_control",
            n_mot=18,
            stations=[1] * 6 + [2] * 6 + [3] * 6,
        )
        for i in range(4)
    ]
    decision = _decide(tracks, campaign="smoke")
    assert decision["decision"] == DECISION_CONTRACT
    assert decision["diagnosis_verdict"] == "INCONCLUSIVE"
    assert decision["three_st_qp_trusted_observable"] is False
    assert decision["residual_conditional_authorized"] is False
    assert decision["s2_flipped_to_pass"] is False
    assert decision["focus_replaced_after_results"] is False
    assert decision["large_dump_submitted"] is False


def test_cannot_exceed_focus_or_use_truth_as_solution():
    config = load_config()
    tracks = [_track(z_mm=18.0, kind=KIND_MOT, event_id=i) for i in range(4)]
    acc = evaluate_tracks(tracks, config)
    report = evaluate_provenance(acc, config)
    report["n_tracks"] = 21
    oversized = decide(
        report,
        _inherited(),
        dumps_materialized=True,
        campaign="batch",
        config=config,
    )
    assert oversized["decision"] != DECISION_RECORDED
    report["n_tracks"] = 4
    report["n_truth_as_solution"] = 1
    truth = decide(
        report,
        _inherited(),
        dumps_materialized=True,
        campaign="batch",
        config=config,
    )
    assert truth["mechanism"] == "truth_used_as_fit_or_solution"
    assert truth["three_st_qp_trusted_observable"] is False
