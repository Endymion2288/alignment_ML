"""WB85 protocol freeze: synthetic calibration only.  No physical solve."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from alignment.calypso_physical_replica_production import PAYLOAD_FAMILIES
from alignment.common_track_solver import (
    SURVEY_FINITE_PRIOR,
    SURVEY_FIXED_DZ,
    parameter_chart,
    solver_implementation_fixed,
)
from alignment.wb85_physical_qualification_protocol import (
    ALT_MEAN_PRIMARY,
    ALT_MEAN_SMALL,
    ALPHA,
    N_MIN_PER_IDENTIFIABLE_STRATUM,
    OUTPUT_ROOT,
    PHYSICAL_PROVENANCE,
    SYNTHETIC_CALIBRATION_SEED,
    WB84_CONDITION_COUNTS,
    WB84_QA,
    WB84_QUALIFIED_EXPORTS,
    WB85Error,
    artifact_payloads,
    chart_parameter_names,
    coverage_calibration,
    evaluate_future_qualification,
    evaluate_guardrails,
    holm_reject,
    identifiable_strata,
    location_scale_gof,
    power_analysis,
    protocol_freeze_complete,
    refuse_alignment_outcome_lookup,
    refuse_execution,
    refuse_forbidden_path,
    refuse_wb83_wb84_write,
    run_physical_qualification,
    write_protocol_artifacts,
)


V1_GATE = Path("outputs/mc24_four_station_wb83_truth_only_replicas_v1/truth_oracle_qualification.json")
V2_GATE = Path("outputs/mc24_four_station_wb83_truth_only_replicas_v2/truth_oracle_qualification.json")
REQUIRED_ARTIFACTS = (
    "wb85_physical_alignment_protocol.json",
    "global_statistical_gate.json",
    "mode_chart_contract.json",
    "power_analysis.json",
    "multiplicity_contract.json",
    "qualification_schema.json",
)


def _identifiable_ids() -> list[str]:
    return [row["stratum_id"] for row in identifiable_strata()]


def _standard_normal_bundle(rng: np.random.Generator, *, n: int = 200, mu: float = 0.0, scale: float = 1.0) -> dict:
    z = {}
    coverage = {}
    identifiable = {}
    for name in _identifiable_ids():
        values = rng.normal(mu, scale, size=n)
        z[name] = values.tolist()
        covered = int(rng.binomial(n, 0.95))
        coverage[name] = (covered, n)
        identifiable[name] = {
            "n_attempted": n,
            "n_converged": n,
            "n_nan": 0,
            "n_inf": 0,
            "n_non_spd": 0,
            "fd_rel_max_observed": 0.0,
            "sign_frame_failure": False,
            "engineering_bias": 0.0,
            "empirical_coverage": covered / float(n),
            "cp_hi": 0.99,
            "n_dropped_parameters": 0,
            "dropped": False,
        }
    return {"z": z, "coverage": coverage, "identifiable_strata": identifiable, "weak_strata": {}}


def _exact_calibrated_z(*, n: int = 200) -> dict[str, list[float]]:
    unit = np.ones(n, dtype=np.float64)
    centered = np.concatenate([-unit[: n // 2], unit[n // 2 :]])
    centered = centered - centered.mean()
    centered = centered / centered.std(ddof=1)
    return {name: centered.tolist() for name in _identifiable_ids()}


def test_protocol_freeze_is_complete_and_unauthorized():
    assert protocol_freeze_complete() is True
    assert solver_implementation_fixed() is True
    for payload in artifact_payloads().values():
        assert payload["executable"] is False
        assert payload["qualification_authorized"] is False
        assert payload["alignment_oracle_qualified_for_physical_FASER"] is False
        assert payload["ml_alignment_eval_authorized"] is False
        assert payload["is_wb83_rerun"] is False
        assert payload["modifies_wb84_corpus"] is False
        assert payload["trained_on_wb83_failure_numbers"] is False


def test_provenance_and_geometry_are_wb84():
    protocol = artifact_payloads()["wb85_physical_alignment_protocol.json"]
    assert protocol["corpus"]["provenance"] == PHYSICAL_PROVENANCE
    assert protocol["corpus"]["n_qualified_exports"] == WB84_QUALIFIED_EXPORTS
    families = protocol["geometry_families"]
    assert families["n_physical_geometry_families"] == 4
    assert families["may_reselect_after_seeing_alignment"] is False
    assert families["paired_event_comparison_forbidden"] is True
    assert PAYLOAD_FAMILIES["identifiable_translation"][1][0] == 0.30
    assert PAYLOAD_FAMILIES["identifiable_translation"][2][1] == -0.20
    assert PAYLOAD_FAMILIES["identifiable_rotation"][1][5] == -0.0005
    assert PAYLOAD_FAMILIES["identifiable_rotation"][3][5] == 0.0005
    assert PAYLOAD_FAMILIES["weak_jg_diagnostic"][3][0] == 0.20
    assert PAYLOAD_FAMILIES["weak_jg_diagnostic"][3][4] == 0.0002


def test_wb84_qa_counts_were_not_redesigned():
    qa = json.loads(WB84_QA.read_text(encoding="utf-8"))
    assert qa["n_qualified_exports"] == 2394
    assert qa["qualification_authorized"] is False
    assert qa["alignment_oracle_qualified_for_physical_FASER"] is False
    for name, count in WB84_CONDITION_COUNTS.items():
        assert qa["conditions"][name]["qualified_exports"] == count
        assert count >= 200


def test_charts_match_solver_parameter_chart():
    assert chart_parameter_names("translation", SURVEY_FIXED_DZ) == parameter_chart(
        survey_mode=SURVEY_FIXED_DZ,
        reference_station=0,
        components=("dx_mm", "dy_mm"),
        stations=(1, 2, 3),
    )
    assert chart_parameter_names("translation", SURVEY_FINITE_PRIOR) == parameter_chart(
        survey_mode=SURVEY_FINITE_PRIOR,
        reference_station=0,
        components=("dx_mm", "dy_mm", "dz_mm"),
        stations=(1, 2, 3),
    )
    assert chart_parameter_names("rotation", SURVEY_FIXED_DZ) == (
        "s1_rz_mrad",
        "s3_rz_mrad",
    )
    weak = chart_parameter_names("weak_jg", SURVEY_FIXED_DZ)
    assert weak == ("s3_dx_mm", "s3_ry_mrad")
    charts = artifact_payloads()["mode_chart_contract.json"]["charts"]
    assert charts["translation_chart"]["mode_status"] == "identifiable"
    assert charts["rotation_chart"]["mode_status"] == "identifiable"
    assert charts["weak_jg_diagnostic_chart"]["mode_status"] == "weak"
    assert charts["weak_jg_diagnostic_chart"]["enters_primary_global_test"] is False
    assert charts["weak_jg_diagnostic_chart"]["ordinary_translation_threshold_forbidden"] is True
    assert charts["translation_chart"]["not_15d_relative_wls"] is True


def test_weak_mode_cannot_enter_primary_gof():
    z = _exact_calibrated_z()
    z["weak_jg_diagnostic_fixed_dz"] = z[next(iter(z))]
    with pytest.raises(WB85Error, match="non-identifiable"):
        location_scale_gof(z)


def test_exact_calibrated_residuals_accept():
    gof = location_scale_gof(_exact_calibrated_z())
    assert gof["T_LS"] == pytest.approx(0.0, abs=1.0e-12)
    assert gof["accepted"] is True
    assert gof["qualification_status"] == "PASS"


def test_persistent_mean_shift_rejects():
    z = {name: [ALT_MEAN_PRIMARY] * 200 for name in _identifiable_ids()}
    gof = location_scale_gof(z)
    assert gof["accepted"] is False
    assert gof["qualification_status"] == "FAIL"


def test_n_below_200_is_unknown_not_pass():
    z = _exact_calibrated_z(n=199)
    gof = location_scale_gof(z)
    assert gof["qualification_status"] == "UNKNOWN"
    assert gof["accepted"] is False


def test_missing_stratum_is_fail_closed():
    z = _exact_calibrated_z()
    del z["identifiable_rotation_fixed_dz"]
    with pytest.raises(WB85Error, match="missing"):
        location_scale_gof(z)


def test_synthetic_type_i_near_nominal():
    rng = np.random.default_rng(SYNTHETIC_CALIBRATION_SEED)
    rejects = 0
    trials = 400
    for _ in range(trials):
        z = {name: rng.normal(0.0, 1.0, size=200).tolist() for name in _identifiable_ids()}
        if not location_scale_gof(z)["accepted"]:
            rejects += 1
    rate = rejects / float(trials)
    assert 0.015 <= rate <= 0.10


def test_synthetic_primary_alternative_has_power():
    rng = np.random.default_rng(SYNTHETIC_CALIBRATION_SEED + 1)
    rejects = 0
    trials = 80
    names = _identifiable_ids()
    for _ in range(trials):
        z = {name: rng.normal(0.0, 1.0, size=200).tolist() for name in names}
        z[names[0]] = rng.normal(ALT_MEAN_PRIMARY, 1.0, size=200).tolist()
        if not location_scale_gof(z)["accepted"]:
            rejects += 1
    assert rejects / float(trials) >= 0.80


def test_power_analysis_is_prospective():
    power = power_analysis()
    assert power["n_per_identifiable_stratum"] == N_MIN_PER_IDENTIFIABLE_STRATUM
    assert power["post_hoc_pooling_forbidden"] is True
    assert power["alternatives"]["persistent_mean_0.35_one_stratum"]["sufficient"] is True
    assert power["alternatives"]["scale_1.25_one_stratum"]["sufficient"] is True
    assert power["alternatives"]["persistent_mean_0.25_one_stratum"]["sufficient"] is False
    assert power["alternatives"]["persistent_mean_0.25_one_stratum"]["qualification_status_if_this_were_the_claim"] == (
        "UNKNOWN"
    )
    assert power["small_alternative_is_not_a_PASS_claim"] is True
    assert ALT_MEAN_SMALL == 0.25
    assert power["min_detectable_at_80_percent"]["persistent_standardized_mean"] == pytest.approx(0.294, abs=0.01)


def test_holm_stops_at_first_nonrejection():
    assert holm_reject([0.01, 0.02, 0.04, 0.20], alpha=0.05) == [True, False, False, False]
    assert holm_reject([0.01, 0.012, 0.06], alpha=0.05) == [True, True, False]


def test_coverage_gate_is_global_not_every_cell():
    names = _identifiable_ids()
    covered = {name: (190, 200) for name in names}
    report = coverage_calibration(covered)
    assert report["accepted"] is True
    assert report["not_every_cell_cp_must_contain_0.95"] is True
    covered[names[0]] = (140, 200)
    bad = coverage_calibration(covered)
    assert bad["gross_failure"] is True
    assert bad["reports"][names[0]]["empirical_coverage"] == 0.70


def test_guardrails_fail_closed():
    names = _identifiable_ids()
    healthy = {
        name: {
            "n_attempted": 200,
            "n_converged": 200,
            "n_nan": 0,
            "n_inf": 0,
            "n_non_spd": 0,
            "fd_rel_max_observed": 0.0,
            "sign_frame_failure": False,
            "engineering_bias": 0.0,
            "empirical_coverage": 0.95,
            "cp_hi": 0.98,
            "n_dropped_parameters": 0,
            "dropped": False,
        }
        for name in names
    }
    assert evaluate_guardrails({"identifiable_strata": healthy})["failed"] is False
    dropped = dict(healthy)
    dropped = {**healthy, names[0]: {**healthy[names[0]], "dropped": True}}
    assert evaluate_guardrails({"identifiable_strata": dropped})["failed"] is True
    nan = {**healthy, names[1]: {**healthy[names[1]], "n_nan": 1}}
    assert evaluate_guardrails({"identifiable_strata": nan})["failed"] is True
    bias = {**healthy, "identifiable_translation_fixed_dz": {**healthy[names[0]], "engineering_bias": 0.20}}
    assert evaluate_guardrails({"identifiable_strata": bias})["failed"] is True
    weak = evaluate_guardrails(
        {
            "identifiable_strata": healthy,
            "weak_strata": {"weak_jg_diagnostic_fixed_dz": {"standardized_bias_dx": 6.0, "standardized_bias_ry": 0.1}},
        }
    )
    assert weak["failed"] is True


def test_future_bundle_pass_and_fail():
    rng = np.random.default_rng(0)
    z = _exact_calibrated_z()
    coverage = {name: (190, 200) for name in _identifiable_ids()}
    identifiable = _standard_normal_bundle(rng)["identifiable_strata"]
    passed = evaluate_future_qualification({"z": z, "coverage": coverage, "identifiable_strata": identifiable})
    assert passed["qualification_status"] == "PASS"
    assert passed["cannot_claim_0.25_mean_alternative"] is True
    shifted = {name: [0.35] * 200 for name in _identifiable_ids()}
    failed = evaluate_future_qualification({"z": shifted, "coverage": coverage, "identifiable_strata": identifiable})
    assert failed["qualification_status"] == "FAIL"
    assert "diagnostics" in failed
    assert failed["diagnostics"]["may_reselect_model"] is False
    assert failed["diagnostics"]["may_change_thresholds"] is False


def test_refuses_execution_and_forbidden_paths():
    with pytest.raises(WB85Error, match="freezes the protocol"):
        refuse_execution()
    with pytest.raises(WB85Error, match="freezes the protocol"):
        run_physical_qualification()
    with pytest.raises(WB85Error, match="forbidden path"):
        refuse_forbidden_path("outputs/family1/overlay_synthetic_v1/x.json")
    with pytest.raises(WB85Error, match="forbidden path"):
        refuse_forbidden_path("mc24_100047_00350_00399/x.root")
    with pytest.raises(WB85Error, match="must not rewrite"):
        refuse_wb83_wb84_write("outputs/mc24_four_station_calypso_physical_replicas_v1/corpus_qa.json")
    with pytest.raises(WB85Error, match="must not rewrite"):
        refuse_wb83_wb84_write("outputs/mc24_four_station_wb83_final_closure_v1/x.json")
    with pytest.raises(WB85Error, match="must not inspect"):
        refuse_alignment_outcome_lookup("outputs/physical_alignment_outcome.json")


def test_writer_refuses_wb84_root():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/write_wb85_physical_alignment_protocol.py",
            "--output-root",
            "outputs/mc24_four_station_calypso_physical_replicas_v1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "refuses" in (result.stderr + result.stdout).lower()


def test_writer_refuses_execute_flag():
    result = subprocess.run(
        [sys.executable, "scripts/write_wb85_physical_alignment_protocol.py", "--execute"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_artifacts_write_and_stay_locked(tmp_path: Path):
    root = write_protocol_artifacts(tmp_path / "wb85")
    names = {path.name for path in root.iterdir()}
    assert set(REQUIRED_ARTIFACTS) <= names
    for name in REQUIRED_ARTIFACTS:
        payload = json.loads((root / name).read_text(encoding="utf-8"))
        assert payload["executable"] is False
        assert payload["qualification_authorized"] is False


def test_official_artifacts_if_present_are_locked():
    if not OUTPUT_ROOT.exists():
        pytest.skip("official WB85 artifacts not written yet")
    for name in REQUIRED_ARTIFACTS:
        payload = json.loads((OUTPUT_ROOT / name).read_text(encoding="utf-8"))
        assert payload["executable"] is False
        assert payload["qualification_authorized"] is False
    protocol = json.loads((OUTPUT_ROOT / "wb85_physical_alignment_protocol.json").read_text(encoding="utf-8"))
    assert protocol["protocol_audit"] == "PASS"
    assert protocol["htcondor_qualification_authorized"] is False


def test_wb83_gates_remain_fail():
    v1 = json.loads(V1_GATE.read_text(encoding="utf-8"))
    v2 = json.loads(V2_GATE.read_text(encoding="utf-8"))
    assert v1["qualification"] == "FAIL"
    assert v2["qualification"] == "FAIL"
    assert v2["alignment_oracle_qualified_for_physical_FASER"] is False
    assert not Path("outputs/mc24_four_station_wb83_truth_only_replicas_v3").exists()


def test_m2_was_not_chosen_to_rescue_wb83():
    gate = artifact_payloads()["global_statistical_gate.json"]
    assert gate["stack_rule"]["method"] == "M2_global_location_scale_gof"
    assert gate["stack_rule"]["method_chosen_because_it_would_pass_wb83"] is False
    assert gate["wb83_cellwise_unadjusted_gate_reused"] is False
    assert gate["stack_rule"]["weak_excluded_from_primary"] is True
    assert ALPHA == 0.05
