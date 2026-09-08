"""Task B4 tail / uncertainty analysis.  No sealed test, no Q/chi2 tuning."""

from __future__ import annotations

import pytest

from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.acts_transport_tail_analysis import (
    CASE_1,
    CASE_2,
    CASE_3,
    CLASS_A,
    CLASS_B,
    CLASS_C,
    DECISION_ANALYZED,
    TailAnalysisError,
    classify_tail_event,
    decide_mechanism,
    dump_path_for_source,
    inherit_frozen_stage,
    is_wrong_momentum,
    load_config,
    load_frozen_tails,
    reconstruction_quality_report,
    refuse_covariance_rescale,
    refuse_dummy_segmentfit,
    refuse_highland_replacement,
    refuse_measurement_model_v2,
    refuse_outlier_rejection,
    refuse_process_noise_tuning,
    refuse_tail_rescreen,
    refuse_truth_qoverp,
)


def test_config_inherits_frozen_wb87_through_wb99():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    assert inherited["workbook_96"]["decision"] == "ckf_qoverp_covariance_export_established"
    assert inherited["workbook_98"]["situation"] == "acts_q_materialized_closure_failed"
    assert inherited["workbook_99"]["decision"] == "acts_transport_covariance_failure_diagnosed"
    assert inherited["workbook_99"]["primary_case"] == "high_chi2_tail_dominated"
    assert inherited["workbook_99"]["tail_sha256"].startswith("c6ad96f9")
    assert config["do_not_rescreen_wb99_tails"] is True
    assert config["do_not_replace_acts_q_with_highland"] is True
    assert config["wrong_momentum"]["abs_log10_p_ratio_min"] == 1.0
    assert "acts_transport_tail_analysis_v1" in config["output_root"]
    assert config["output_root"] != config["dump_root"]
    assert config["output_root"] != config["wb99_output_root"]


def test_frozen_tails_are_not_rescreened():
    frozen = load_frozen_tails(load_config())
    assert frozen["rescreened"] is False
    assert frozen["catalogs"]["0.99"]["n"] == 24
    assert frozen["catalogs"]["0.999"]["n"] == 3
    assert frozen["catalogs"]["0.99"]["rejected"] == 0


def test_dump_path_reads_wb98_without_overwrite():
    path = dump_path_for_source(load_config(), "mc24_100043_00200_00299")
    assert path.as_posix().endswith(
        "outputs/acts_transport_dump_v1/dumps/mc24_100043_00200_00299/ckf_acts_transport.jsonl"
    )


def test_forbidden_repairs_and_sealed_test():
    with pytest.raises(TailAnalysisError, match="truth"):
        refuse_truth_qoverp()
    with pytest.raises(TailAnalysisError, match="dummy"):
        refuse_dummy_segmentfit()
    with pytest.raises(TailAnalysisError, match="rescale"):
        refuse_covariance_rescale()
    with pytest.raises(TailAnalysisError, match="process noise"):
        refuse_process_noise_tuning()
    with pytest.raises(TailAnalysisError, match="outlier"):
        refuse_outlier_rejection()
    with pytest.raises(TailAnalysisError, match="rescreened"):
        refuse_tail_rescreen()
    with pytest.raises(TailAnalysisError, match="Highland"):
        refuse_highland_replacement()
    with pytest.raises(TailAnalysisError, match="Measurement Model V2"):
        refuse_measurement_model_v2()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def _event(**overrides):
    payload = {
        "source_id": "mc24_100043_00200_00299",
        "run_id": 100043,
        "event_id": 1,
        "track_index": 0,
        "station_pair": "(0,1)",
        "source_station": 0,
        "p_mev": 4.0e5,
        "p_truth_mev": 4.1e5,
        "abs_q_over_p": 2.5e-6,
        "q_frobenius": 0.2,
        "abs_slope": 0.002,
        "chi2_per_ndof": 10.0,
        "on_frozen_station_pair": True,
        "split": "construction",
        "pull": {"x": 0.1, "y": 0.1, "tx": 0.1, "ty": 0.1, "q_over_p": 0.2},
    }
    payload.update(overrides)
    return payload


def test_wrong_momentum_is_decade_gate():
    config = load_config()
    assert is_wrong_momentum(4.0e5, 4.0e6, config) is True
    assert is_wrong_momentum(4.0e5, 4.1e5, config) is False


def test_classify_reconstruction_transport_and_material():
    config = load_config()
    reco = classify_tail_event(
        _event(p_mev=4.0e5, p_truth_mev=4.0e6, pull={"q_over_p": -13.0}),
        config=config,
        repeated=True,
    )
    assert reco["primary_class"] == CLASS_A
    assert reco["flags"]["wrong_momentum"] is True
    transport = classify_tail_event(
        _event(),
        config=config,
        repeated=False,
        dump_row={"c0_jacobian_frobenius_rel": 0.4, "transport_jacobian": None, "process_noise": None},
    )
    assert transport["primary_class"] == CLASS_B
    material = classify_tail_event(
        _event(q_frobenius=20.0, abs_slope=0.08),
        config=config,
        repeated=False,
    )
    assert material["primary_class"] == CLASS_C
    assert material["original_residual_kept"] is True


def _classified(events):
    return {
        "catalogs": {
            "0.99": {"events": events},
            "0.999": {"events": events[:1]},
        }
    }


def test_case1_when_wrong_state_dominates():
    config = load_config()
    tails = [
        classify_tail_event(
            _event(event_id=i, p_mev=1.0e5, p_truth_mev=2.0e6, chi2_per_ndof=1000.0),
            config=config,
            repeated=False,
        )
        for i in range(4)
    ]
    official = [
        {**_event(event_id=i, p_mev=1.0e5, p_truth_mev=2.0e6, chi2_per_ndof=1000.0), "split": "construction"}
        for i in range(4)
    ] + [
        {**_event(event_id=100 + i, chi2_per_ndof=0.2, source_id="mc24_100043_00300_00399"), "split": "construction"}
        for i in range(20)
    ]
    classified = _classified(tails)
    quality = reconstruction_quality_report(official, classified, config)
    result = decide_mechanism(
        classified,
        quality,
        {"bulk_overcovered": True, "tail_non_gaussian": True},
        config,
        official,
    )
    assert result["primary_case"] == CASE_1
    assert result["next_step"] == "reconstruction_association_quality_control"
    assert result["measurement_model_v2_entered"] is False
    assert quality["events_dropped"] == 0


def test_case2_when_material_enrichment_without_wrong_p():
    config = load_config()
    tails = [
        classify_tail_event(
            _event(event_id=i, q_frobenius=20.0, abs_slope=0.08, chi2_per_ndof=50.0),
            config=config,
            repeated=False,
        )
        for i in range(6)
    ]
    official = [
        {**_event(event_id=i, q_frobenius=20.0, abs_slope=0.08, chi2_per_ndof=50.0), "split": "construction"}
        for i in range(6)
    ] + [
        {**_event(event_id=200 + i, q_frobenius=0.2, abs_slope=0.002, chi2_per_ndof=0.2), "split": "construction"}
        for i in range(60)
    ]
    classified = _classified(tails)
    quality = reconstruction_quality_report(official, classified, config)
    result = decide_mechanism(
        classified,
        quality,
        {"bulk_overcovered": True, "tail_non_gaussian": False},
        config,
        official,
    )
    assert result["primary_case"] == CASE_2
    assert result["next_step"] == "acts_material_diagnosis"


def test_case3_when_neither_wrong_state_nor_material():
    config = load_config()
    tails = [
        classify_tail_event(_event(event_id=i, chi2_per_ndof=20.0), config=config, repeated=False)
        for i in range(3)
    ]
    official = [
        {**_event(event_id=i, chi2_per_ndof=20.0), "split": "construction"} for i in range(3)
    ] + [
        {**_event(event_id=300 + i, chi2_per_ndof=0.2), "split": "construction"} for i in range(40)
    ]
    classified = _classified(tails)
    quality = reconstruction_quality_report(official, classified, config)
    result = decide_mechanism(
        classified,
        quality,
        {"bulk_overcovered": True, "tail_non_gaussian": False},
        config,
        official,
    )
    assert result["primary_case"] == CASE_3
    assert result["next_step"] == "reassess_measurement_model_v2_input_model"
    assert DECISION_ANALYZED == "acts_transport_tail_uncertainty_analyzed"
