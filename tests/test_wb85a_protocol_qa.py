"""WB85a protocol QA: synthetic only.  No physical qualification."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from alignment.common_track_solver import SURVEY_FINITE_PRIOR, SURVEY_FIXED_DZ
from alignment.physical_common_track_execution import (
    FORBIDDEN_OFFICIAL_FALLBACKS,
    OFFICIAL_ENGINE,
    OfficialPhysicalIterator,
    PhysicalExecutionError,
    RecordingPhysicalBackend,
    identity_payload,
    ingest_physical_event,
    measurements_from_event,
    refuse_official_iterate_common_track,
)
from alignment.wb85_physical_qualification_protocol import (
    ALT_MEAN_PRIMARY,
    ALT_SCALE,
    WB84_CONDITION_COUNTS,
    WB85Error,
    identifiable_strata,
    location_scale_gof,
)
from alignment.wb85a_protocol_qa import (
    IDENTIFIABLE_STRATUM_COUNTS,
    OUTPUT_ROOT,
    SCALE_DEFLATION,
    WB85AError,
    execution_readiness_report,
    fail_closed_suite,
    nuisance_covariance_report,
    project_statistical_unit,
    protocol_qa_acceptance_rule,
    run_null_calibration,
    statistical_unit_contract_report,
    validate_result_schema,
    weak_jg_contract_report,
)


REQUIRED_ARTIFACTS = (
    "wb85a_acceptance_rule.json",
    "wb85a_protocol_qa.json",
    "wb85a_synthetic_calibration.json",
    "wb85a_power_validation.json",
    "wb85a_execution_readiness.json",
    "wb85a_provenance_freeze.json",
)


def _locked(payload: dict) -> None:
    assert payload["qualification_authorized"] is False
    assert payload["executable"] is False
    assert payload["looked_at_physical_alignment_outcomes"] is False
    assert payload["alignment_oracle_qualified_for_physical_FASER"] is False
    assert payload["ml_alignment_eval_authorized"] is False


def test_acceptance_rule_is_frozen_before_calibration():
    rule = protocol_qa_acceptance_rule()
    assert rule["frozen_before_synthetic_calibration"] is True
    assert rule["may_repair_by_looking_at_physical_outcomes"] is False
    assert rule["stratum_counts"] == IDENTIFIABLE_STRATUM_COUNTS
    assert IDENTIFIABLE_STRATUM_COUNTS["identity_fixed_dz"] == 299
    assert IDENTIFIABLE_STRATUM_COUNTS["identity_finite_survey_prior"] == 303
    assert IDENTIFIABLE_STRATUM_COUNTS["identifiable_translation_fixed_dz"] == 311
    assert IDENTIFIABLE_STRATUM_COUNTS["identifiable_translation_finite_survey_prior"] == 298
    assert IDENTIFIABLE_STRATUM_COUNTS["identifiable_rotation_fixed_dz"] == 300
    assert IDENTIFIABLE_STRATUM_COUNTS["identifiable_rotation_finite_survey_prior"] == 283
    for name, count in IDENTIFIABLE_STRATUM_COUNTS.items():
        assert WB84_CONDITION_COUNTS[name] == count
    assert list(IDENTIFIABLE_STRATUM_COUNTS) == [row["stratum_id"] for row in identifiable_strata()]
    assert rule["power_validation"]["primary_mean"] == ALT_MEAN_PRIMARY
    assert rule["power_validation"]["primary_scale"] == ALT_SCALE
    assert rule["power_validation"]["alternatives_not_taken_from_wb83_failures"] is True
    assert abs(SCALE_DEFLATION - 0.763) < 0.01


def test_statistical_unit_is_one_projected_z():
    report = statistical_unit_contract_report()
    assert report["statistical_unit_contract_pass"] is True
    assert report["projected"]["n_independent_z"] == 1
    assert report["rejects_multiple_correlated_pose_z"] is True
    names = ("s1_dx_mm", "s2_dy_mm")
    direction = {"s1_dx_mm": 0.30, "s2_dy_mm": -0.20}
    cov = np.array([[0.04, 0.01], [0.01, 0.09]], dtype=np.float64)
    hat = np.array([0.31, -0.19], dtype=np.float64)
    true = np.array([0.30, -0.20], dtype=np.float64)
    projected = project_statistical_unit(hat, true, cov, names, direction)
    u = np.array([0.30, -0.20], dtype=np.float64)
    u = u / np.linalg.norm(u)
    assert projected["a_hat"] == pytest.approx(float(u @ hat))
    assert projected["sigma_a"] == pytest.approx(math_sqrt(float(u @ cov @ u)))
    assert projected["z"] == pytest.approx((float(u @ hat) - float(u @ true)) / projected["sigma_a"])


def math_sqrt(value: float) -> float:
    return float(value) ** 0.5


def test_result_schema_rejects_duplicate_event_units():
    row = {
        "physical_event_uid": "e1",
        "input_locator": "src:1:1",
        "stratum_id": "identity_fixed_dz",
        "z": 0.1,
        "theta_hat": [0.0],
        "theta_true": [0.0],
        "sigma_hat": 1.0,
    }
    with pytest.raises(WB85AError, match="more than one z"):
        validate_result_schema([row, dict(row)])


def test_nuisance_covariance_matches_joint_schur_and_projection():
    report = nuisance_covariance_report()
    assert report["survey_mode"] == SURVEY_FINITE_PRIOR
    assert report["dz_is_nuisance_not_projected_mode"] is True
    assert report["nuisance_covariance_pass"] is True
    assert report["gaps"]["joint_vs_schur"] < 1.0e-9
    assert report["gaps"]["schur_vs_projected"] < 1.0e-9
    assert report["gaps"]["schur_vs_solver"] < 1.0e-9


def test_fail_closed_gates():
    report = fail_closed_suite()
    assert report["fail_closed_tests_pass"] is True
    assert report["unknown_path"] == "UNKNOWN"
    assert report["cases"]["missing_stratum"] == "FAIL"
    assert report["cases"]["nan"] == "FAIL"
    assert report["cases"]["inf"] == "FAIL"
    assert report["cases"]["nonconvergence_gt_5pct"] == "FAIL"
    assert report["cases"]["unresolved_non_spd"] == "FAIL"
    assert report["cases"]["fd_gt_1pct"] == "FAIL"
    assert report["cases"]["large_engineering_bias"] == "FAIL"
    assert report["cases"]["dropped_identifiable_mode"] == "FAIL"
    assert report["cases"]["gross_coverage_failure"] == "FAIL"
    assert report["cases"]["weak_catastrophic_standardized_bias"] == "FAIL"
    assert report["holm_after_fail_is_diagnostic_only"] is True


def test_weak_jg_stays_out_of_primary_but_keeps_guardrail():
    report = weak_jg_contract_report()
    assert report["weak_jg_contract_pass"] is True
    assert report["weak_enters_primary_global_test"] is False
    assert report["may_add_or_drop_weak_after_seeing_physical_results"] is False
    z = {row["stratum_id"]: [0.0] * 200 for row in identifiable_strata()}
    z["weak_jg_diagnostic_fixed_dz"] = [0.0] * 200
    with pytest.raises(WB85Error, match="non-identifiable"):
        location_scale_gof(z)


def test_official_engine_refuses_toy_fallbacks():
    with pytest.raises(PhysicalExecutionError, match="not the official engine"):
        refuse_official_iterate_common_track(field_y=0.35)
    with pytest.raises(PhysicalExecutionError, match="first-step"):
        OfficialPhysicalIterator(backend=RecordingPhysicalBackend(), reuse_first_step_measurements=True)
    for needle in FORBIDDEN_OFFICIAL_FALLBACKS:
        assert needle in {
            "toy_uniform_By",
            "fixed_first_step_measurements",
            "lab_only_transport",
        }


def test_hermetic_iterator_rerefits_after_left_se3():
    backend = RecordingPhysicalBackend()
    hits = [
        {
            "station_id": station,
            "measurement_id": f"t:{station}",
            "x_mm": 0.0,
            "y_mm": 0.0,
            "tx": 0.0,
            "ty": 0.0,
            "covariance_4x4": np.eye(4).tolist(),
        }
        for station in (0, 1, 2, 3)
    ]
    event = ingest_physical_event(
        {
            "event_uid": "mc24_100047_00100_00149:100047:2002",
            "source_id": "mc24_100047_00100_00149",
            "run_id": 100047,
            "event_id": 2002,
            "xaod_entry_index": 2002,
            "condition_id": "identity_fixed_dz",
            "input_xaod": (
                "/eos/experiment/faser/data0/sim/mc24/particle_gun/100047/rec/s0013-r0022/"
                "FaserMC-MC24_PG_mumi_fasernu_5mrad_flukaE-100047-00100-00149-s0013-r0022-xAOD.root"
            ),
            "tracks": [
                {
                    "track_uid": "t",
                    "truth": {"particle_id": 3, "pdg": 13, "match_fraction": 1.0},
                    "hits": hits,
                }
            ],
        }
    )
    start = identity_payload()
    result = OfficialPhysicalIterator(backend=backend, max_iterations=1).iterate(
        event,
        start,
        measurements_from_event(event, start, engine="wb84_physical_ingest"),
        chart_kind="translation",
        survey_mode=SURVEY_FIXED_DZ,
        work_dir=Path("outputs/mc24_four_station_wb85a_protocol_qa_v1/_test_iterator"),
    )
    assert result["reused_first_step_measurements"] is False
    assert result["automatically_passed_at_max_iterations"] is False
    assert len(backend.rerefit_payloads) == 1
    assert result["history"][0]["rerefit_after_update"] is True


def test_execution_readiness_is_official_and_unauthorized():
    report = execution_readiness_report()
    _locked(report)
    assert report["engine"] == OFFICIAL_ENGINE
    assert report["physical_execution_ready"] is True
    assert report["toy_iterate_common_track_is_official"] is False


def test_small_null_calibration_is_near_nominal():
    report = run_null_calibration(n_mc=400, seed=2026090817)
    assert report["reads_physical_alignment_output"] is False
    assert 0.01 <= report["location_scale"]["rate"] <= 0.12
    assert 0.01 <= report["coverage"]["rate"] <= 0.12
    assert report["official_function_crosscheck"]["mismatches"] == 0


def test_writer_refuses_execute_and_wb84_root():
    execute = subprocess.run(
        [sys.executable, "scripts/run_wb85a_protocol_qa.py", "--execute"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert execute.returncode != 0
    outcomes = subprocess.run(
        [sys.executable, "scripts/run_wb85a_protocol_qa.py", "--read-physical-outcomes"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert outcomes.returncode != 0
    wb84 = subprocess.run(
        [
            sys.executable,
            "scripts/run_wb85a_protocol_qa.py",
            "--output-root",
            "outputs/mc24_four_station_calypso_physical_replicas_v1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert wb84.returncode != 0


def test_official_artifacts_if_present_stay_locked_and_unauthorized():
    if not OUTPUT_ROOT.exists():
        pytest.skip("official WB85a artifacts not written yet")
    for name in REQUIRED_ARTIFACTS:
        payload = json.loads((OUTPUT_ROOT / name).read_text(encoding="utf-8"))
        _locked(payload)
    verdict = json.loads((OUTPUT_ROOT / "wb85a_protocol_qa.json").read_text(encoding="utf-8"))
    assert verdict["qualification_authorized"] is False
    assert verdict["alignment_oracle_qualified_for_physical_FASER"] is False
    assert verdict["wb86_automatically_authorized"] is False
    if verdict["wb85_protocol_qa_pass"]:
        assert verdict["ready_to_authorize_physical_qualification"] is True
