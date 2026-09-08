"""WB85b FWER protocol: prospective freeze only.  No WB86."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from scipy import stats

from alignment.physical_common_track_execution import CalypsoActsPhysicalBackend
from alignment.wb85_physical_qualification_protocol import ALPHA, location_scale_gof
from alignment.wb85b_fwer_protocol import (
    ALPHA_COV,
    ALPHA_LS,
    CRITICAL_COV,
    CRITICAL_LS,
    EXPECTED_CRITICAL_COV,
    EXPECTED_CRITICAL_LS,
    FAMILYWISE_ALPHA,
    HISTORICAL_WB85A,
    WB85BError,
    apply_wb85b_statistical_gate,
    historical_revision_semantics,
    power_mean_wb85b,
    refuse_historical_rewrite,
    statistical_family,
)
from alignment.wb85b_official_runner import official_runner_contract
from alignment.wb85b_protocol_qa import (
    R1_POST_RUN_FREEZE,
    R1_RUNTIME_PARENT,
    acceptance_rule,
    wb85b_fail_closed_report,
)


def test_historical_semantics_do_not_rewrite_wb85_fwer_claim():
    hist = historical_revision_semantics()
    assert hist["does_not_claim_wb85_guaranteed_qualification_level_fwer_0.05"] is True
    assert hist["wb85"]["T_LS_alpha"] == ALPHA
    assert hist["wb85"]["explicit_qualification_level_fwer_le_0.05_claim"] is False
    assert hist["wb85a_r1"]["old_two_alpha_0.05_decision_rule_eligible_under_new_global_requirement"] is False
    assert hist["wb85a_r1"]["measured_dependence_not_used_to_set_wb85b_thresholds"] is True


def test_fixed_bonferroni_criticals():
    family = statistical_family()
    assert family["alpha_LS"] == ALPHA_LS == 0.025
    assert family["alpha_cov"] == ALPHA_COV == 0.025
    assert family["familywise_alpha"] == FAMILYWISE_ALPHA == 0.05
    assert CRITICAL_LS == pytest.approx(EXPECTED_CRITICAL_LS)
    assert CRITICAL_COV == pytest.approx(EXPECTED_CRITICAL_COV)
    assert CRITICAL_LS == pytest.approx(float(stats.chi2.ppf(0.975, 12)))
    assert CRITICAL_COV == pytest.approx(float(stats.chi2.ppf(0.975, 6)))
    assert family["not_fitted_to_wb85a_r1_correlations"] is True


def test_acceptance_rule_is_frozen_and_forbids_sweeps():
    rule = acceptance_rule()
    assert rule["frozen_before_any_new_monte_carlo"] is True
    assert rule["alpha_allocation_sweep_forbidden"] is True
    assert rule["may_retune_from_wb85a_r1_correlations"] is False
    assert rule["fwer_claim_scope"]["unconditional_overall_qualification_false_fail_le_0.05"] is False
    assert rule["fwer_claim_scope"]["conditional_on_valid_analyzable_execution"] is True


def test_wb85b_gate_ignores_wb85_alpha_005_accepted_flag():
    wb85_critical = float(stats.chi2.ppf(0.95, 12))
    t_ls = 0.5 * (wb85_critical + CRITICAL_LS)
    assert wb85_critical < t_ls < CRITICAL_LS
    gate = apply_wb85b_statistical_gate(t_ls=t_ls, t_cov=1.0)
    assert gate["reject_LS"] is False
    assert gate["primary_statistical_fail"] is False
    assert gate["wb85_alpha_0.05_accepted_flags_are_not_used"] is True
    assert location_scale_gof  # historical function still uses ALPHA=0.05 critical
    assert ALPHA == 0.05


def test_mean_shift_analytic_power_is_recomputed_under_wb85b():
    value = power_mean_wb85b(0.35, n=200)
    assert 0.89 < value < 0.90
    assert value == pytest.approx(0.8942286683713147, rel=1.0e-9)


def test_official_runner_is_not_the_r1_top_level_executor():
    contract = official_runner_contract()
    assert contract["official_runner_uses_smoke_tested_executor"] is False
    assert contract["official_runner_uses_smoke_tested_primitives"] is True
    assert contract["therefore_dedicated_official_runner_e2e_smoke_is_required"] is True
    assert CalypsoActsPhysicalBackend(execute=False).execute is False
    with pytest.raises(Exception, match="cannot authorize Calypso execution"):
        CalypsoActsPhysicalBackend(execute=True)


def test_wb85b_fail_closed_uses_new_evaluator():
    report = wb85b_fail_closed_report()
    assert report["fail_closed_tests_pass"] is True
    assert report["wb85b_cases"]["missing_stratum"] == "FAIL"
    assert report["wb85b_cases"]["n_below_200"] == "UNKNOWN"
    assert report["wb85b_guardrails_use_evaluate_wb85b_qualification"] is True


def test_r1_runtime_commit_is_not_the_post_run_freeze():
    assert R1_RUNTIME_PARENT != R1_POST_RUN_FREEZE
    assert R1_RUNTIME_PARENT.startswith("fe00c0a")


def test_writer_refuses_history_execute_and_wb86():
    with pytest.raises(WB85BError, match="historical"):
        refuse_historical_rewrite(HISTORICAL_WB85A / "wb85a_protocol_qa.json")
    execute = subprocess.run(
        [sys.executable, "scripts/run_wb85b_protocol_qa.py", "--execute"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert execute.returncode != 0
    wb86 = subprocess.run(
        [sys.executable, "scripts/run_wb85b_protocol_qa.py", "--run-wb86"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert wb86.returncode != 0
    historical = subprocess.run(
        [sys.executable, "scripts/run_wb85b_protocol_qa.py", "--output-root", str(HISTORICAL_WB85A)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert historical.returncode != 0
