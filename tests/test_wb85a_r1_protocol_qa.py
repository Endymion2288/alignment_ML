"""WB85a-r1: coherent joint-null, smoke fixture, and provenance.  No WB86."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from alignment.calypso_physical_replica_production import SKIP_BASE, allocation_plan
from alignment.physical_common_track_execution import CalypsoActsPhysicalBackend
from alignment.wb85_physical_qualification_protocol import ALPHA, WB84_CONDITION_COUNTS
from alignment.wb85_projected_unit import (
    GAUSSIAN_95_QUANTILE,
    official_covered_95,
    official_projected_estimate,
    official_z_and_covered_95,
)
from alignment.wb85a_protocol_qa import COMBINED_TYPE_I_TARGET, IDENTIFIABLE_STRATUM_COUNTS
from alignment.wb85a_r1_protocol_qa import (
    FROZEN_PARENT,
    HISTORICAL_WB85A,
    INDEPENDENT_UNION_EXPECTATION,
    JOINT_NULL_SEED,
    OUTPUT_ROOT,
    OVERALL_QUALIFICATION_FALSE_FAIL_CAP,
    WB85AR1Error,
    independence_is_not_assumed,
    protocol_revision_required,
    r1_acceptance_rule,
    refuse_wb85a_v1_rewrite,
    run_joint_null_calibration,
)
from alignment.wb85a_r1_runtime_smoke import (
    SMOKE_SKIP_EVENTS,
    SMOKE_SOURCE_ID,
    fixture_is_legal,
)


def _locked(payload: dict) -> None:
    assert payload["qualification_authorized"] is False
    assert payload["executable"] is False
    assert payload["looked_at_physical_alignment_outcomes"] is False
    assert payload["alignment_oracle_qualified_for_physical_FASER"] is False
    assert payload["wb86_automatically_authorized"] is False


def test_acceptance_rule_is_frozen_before_joint_mc():
    rule = r1_acceptance_rule()
    assert rule["frozen_before_any_new_monte_carlo"] is True
    assert rule["may_retune_wb85_thresholds_from_this_qa"] is False
    assert rule["may_silently_change_alpha"] is False
    assert rule["joint_null_seed"] == JOINT_NULL_SEED
    assert rule["stratum_counts"] == IDENTIFIABLE_STRATUM_COUNTS
    assert tuple(rule["stratum_counts"].values()) == (299, 303, 311, 298, 300, 283)
    for name, count in IDENTIFIABLE_STRATUM_COUNTS.items():
        assert WB84_CONDITION_COUNTS[name] == count
    joint = rule["joint_union_false_fail"]
    assert joint["official_target_is_independent_union_0.0975"] is False
    assert joint["scientific_overall_qualification_false_fail_cap"] == 0.05
    assert joint["independence_not_assumed"] is True
    assert INDEPENDENT_UNION_EXPECTATION == COMBINED_TYPE_I_TARGET
    assert abs(INDEPENDENT_UNION_EXPECTATION - 0.0975) < 1.0e-12


def test_independence_is_not_the_official_joint_target():
    report = independence_is_not_assumed()
    assert report["independence_proved"] is False
    assert report["official_joint_target_is_1_minus_1_minus_alpha_squared"] is False
    assert report["historical_0.0975_is_not_the_official_joint_null_target"] is True


def test_official_coverage_is_a_function_of_the_same_z():
    z, covered = official_z_and_covered_95(0.10, 0.0, 0.10)
    assert covered is official_covered_95(z)
    assert official_covered_95(GAUSSIAN_95_QUANTILE) is True
    assert official_covered_95(GAUSSIAN_95_QUANTILE + 1.0e-9) is False
    names = ("s1_dx_mm", "s2_dy_mm")
    direction = {"s1_dx_mm": 0.30, "s2_dy_mm": -0.20}
    cov = np.array([[0.04, 0.01], [0.01, 0.09]], dtype=np.float64)
    hat = np.array([0.31, -0.19], dtype=np.float64)
    true = np.array([0.30, -0.20], dtype=np.float64)
    projected = official_projected_estimate(hat, true, cov, names, direction)
    assert projected["covered_95"] is official_covered_95(projected["z"])
    assert projected["n_independent_z"] == 1


def test_protocol_revision_rule_is_a_priori():
    assert protocol_revision_required({"rate": 0.08, "cp95_lo": 0.074, "cp95_hi": 0.086}) is True
    assert protocol_revision_required({"rate": 0.051, "cp95_lo": 0.047, "cp95_hi": 0.055}) is False
    assert protocol_revision_required({"rate": 0.049, "cp95_lo": 0.045, "cp95_hi": 0.053}) is False
    assert OVERALL_QUALIFICATION_FALSE_FAIL_CAP == ALPHA


def test_small_joint_null_is_internally_coherent():
    report = run_joint_null_calibration(n_mc=40, seed=2026090819)
    assert report["reads_physical_alignment_output"] is False
    assert report["joint_union"]["official_target_is_independent_union_0.0975"] is False
    assert report["dependence"]["historical_0.0975_is_not_the_official_joint_null_target"] is True
    assert report["official_function_crosscheck"]["mismatches"] == 0
    assert report["wb85_thresholds_not_retuned"] is True
    assert 0.0 <= report["location_scale"]["rate"] <= 1.0
    assert 0.0 <= report["coverage"]["rate"] <= 1.0


def test_smoke_fixture_is_outside_wb84_and_prohibited_domains():
    fixture = fixture_is_legal()
    assert fixture["source_id"] == SMOKE_SOURCE_ID
    assert fixture["skip_events"] == SMOKE_SKIP_EVENTS
    assert fixture["skip_events"] < SKIP_BASE
    assert fixture["in_wb84_qualification_allocation"] is False
    assert fixture["prohibited_domain"] is False
    plan = allocation_plan()
    assert not any(
        row["source_id"] == SMOKE_SOURCE_ID and int(row["xaod_entry_index"]) == SMOKE_SKIP_EVENTS
        for row in plan["events"]
    )
    text = fixture["input_xaod"].lower()
    for needle in ("00350", "00800", "100116", "100117", "final_blind", "sealed"):
        assert needle not in text


def test_official_backend_execute_stays_false():
    backend = CalypsoActsPhysicalBackend(execute=False)
    assert backend.execute is False
    with pytest.raises(Exception, match="cannot authorize Calypso execution"):
        CalypsoActsPhysicalBackend(execute=True)


def test_writer_refuses_execute_wb84_wb85a_and_wb86():
    execute = subprocess.run(
        [sys.executable, "scripts/run_wb85a_r1_protocol_qa.py", "--execute"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert execute.returncode != 0
    outcomes = subprocess.run(
        [sys.executable, "scripts/run_wb85a_r1_protocol_qa.py", "--read-physical-outcomes"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert outcomes.returncode != 0
    wb86 = subprocess.run(
        [sys.executable, "scripts/run_wb85a_r1_protocol_qa.py", "--run-wb86"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert wb86.returncode != 0
    wb84 = subprocess.run(
        [
            sys.executable,
            "scripts/run_wb85a_r1_protocol_qa.py",
            "--output-root",
            "outputs/mc24_four_station_calypso_physical_replicas_v1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert wb84.returncode != 0
    historical = subprocess.run(
        [
            sys.executable,
            "scripts/run_wb85a_r1_protocol_qa.py",
            "--output-root",
            str(HISTORICAL_WB85A),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert historical.returncode != 0
    with pytest.raises(WB85AR1Error, match="historical WB85a"):
        refuse_wb85a_v1_rewrite(HISTORICAL_WB85A / "wb85a_protocol_qa.json")


def test_official_artifacts_if_present_stay_locked_and_unauthorized():
    if not OUTPUT_ROOT.exists():
        pytest.skip("official WB85a-r1 artifacts not written yet")
    verdict = json.loads((OUTPUT_ROOT / "wb85a_r1_protocol_qa.json").read_text(encoding="utf-8"))
    _locked(verdict)
    assert verdict["frozen_parent"] == f"4station@{FROZEN_PARENT}"
    assert verdict["wb86_not_created_or_run"] is True
    assert "generating_code" in verdict
    joint = json.loads((OUTPUT_ROOT / "wb85a_r1_joint_null_calibration.json").read_text(encoding="utf-8"))
    assert joint["joint_union"]["official_target_is_independent_union_0.0975"] is False
