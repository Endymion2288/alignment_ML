"""WB85b: FWER-controlled physical alignment qualification protocol v2.

Prospective statistical revision only.  Does not execute WB86, does not read
WB84 physical alignment outcomes, and does not rewrite WB85 / WB85a / WB85a-r1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from scipy import stats

from alignment.wb85_physical_qualification_protocol import (
    ALPHA,
    ALT_MEAN_PRIMARY,
    ALT_SCALE,
    ASSOCIATION_DEFAULT_SYSTEM,
    COVERAGE_GROSS_CP_UPPER_MIN,
    COVERAGE_GROSS_EMPIRICAL_MIN,
    ENGINEERING_RZ_MRAD,
    ENGINEERING_TRANSLATION_MM,
    FD_REL_MAX,
    MAX_NONCONVERGENCE_FRACTION,
    N_IDENTIFIABLE_STRATA,
    N_MIN_PER_IDENTIFIABLE_STRATUM,
    POWER_TARGET,
    WEAK_STANDARDIZED_CATASTROPHIC,
    coverage_calibration,
    evaluate_guardrails,
    location_scale_gof,
    refuse_forbidden_path,
    refuse_wb83_wb84_write,
    write_json,
)
from alignment.wb85a_protocol_qa import WB85AError, healthy_identifiable_row, identifiable_ids


WORKBOOK = "85b"
OUTPUT_ROOT = Path("outputs/mc24_four_station_wb85b_fwer_protocol_v1")
HISTORICAL_WB85 = Path("outputs/mc24_four_station_wb85_physical_alignment_protocol_v1")
HISTORICAL_WB85A = Path("outputs/mc24_four_station_wb85a_protocol_qa_v1")
HISTORICAL_WB85A_R1 = Path("outputs/mc24_four_station_wb85a_r1_protocol_qa_v1")
HISTORICAL_WB85A_R1_SMOKE = Path("outputs/mc24_four_station_wb85a_r1_runtime_smoke_v1")

FAMILYWISE_ALPHA = 0.05
ALPHA_LS = 0.025
ALPHA_COV = 0.025
DF_LS = 2 * N_IDENTIFIABLE_STRATA
DF_COV = N_IDENTIFIABLE_STRATA
CRITICAL_LS = float(stats.chi2.ppf(1.0 - ALPHA_LS, DF_LS))
CRITICAL_COV = float(stats.chi2.ppf(1.0 - ALPHA_COV, DF_COV))
EXPECTED_CRITICAL_LS = 23.33666415864534
EXPECTED_CRITICAL_COV = 14.44937533544792
R1_UNION_RATE = 0.0874
R1_UNION_CP = (0.08352110492598208, 0.09139931425338138)
R1_CORR_T = 0.40494538360343607
R1_CORR_REJECT = 0.19630571702375582


class WB85BError(WB85AError):
    """Fail-closed WB85b protocol contract."""


def refuse_historical_rewrite(path: object) -> None:
    text = str(path)
    blocked = (
        "wb85_physical_alignment_protocol_v1",
        "wb85a_protocol_qa_v1",
        "wb85a_r1_protocol_qa_v1",
        "wb85a_r1_runtime_smoke_v1",
    )
    if any(marker in text and "wb85b" not in text for marker in blocked):
        raise WB85BError(f"WB85b must not rewrite historical artifacts: {path}")
    refuse_wb83_wb84_write(path)
    refuse_forbidden_path(path)


def locked_flags() -> dict[str, Any]:
    return {
        "workbook": WORKBOOK,
        "executable": False,
        "qualification_authorized": False,
        "alignment_oracle_qualified_for_physical_FASER": False,
        "ml_alignment_eval_authorized": False,
        "looked_at_physical_alignment_outcomes": False,
        "final_blind_eval_authorized": False,
        "association_default_system": ASSOCIATION_DEFAULT_SYSTEM,
        "is_wb83_rerun": False,
        "modifies_wb84_corpus": False,
        "trained_on_wb83_failure_numbers": False,
        "wb86_automatically_authorized": False,
        "wb85_historical_artifact_preserved": True,
        "wb85a_historical_artifact_preserved": True,
        "wb85a_r1_historical_artifact_preserved": True,
    }


def historical_revision_semantics() -> dict[str, Any]:
    return {
        "does_not_claim_wb85_guaranteed_qualification_level_fwer_0.05": True,
        "wb85": {
            "T_LS_alpha": ALPHA,
            "T_cov_was_an_additional_global_gate": True,
            "explicit_qualification_level_fwer_le_0.05_claim": False,
            "historical_artifact_not_rewritten": True,
        },
        "wb85a_r1": {
            "prospectively_imposed_primary_statistical_fwer_cap": 0.05,
            "imposed_before_any_physical_alignment_outcome": True,
            "coherent_joint_null_union_false_fail": R1_UNION_RATE,
            "coherent_joint_null_union_cp95": list(R1_UNION_CP),
            "old_two_alpha_0.05_decision_rule_eligible_under_new_global_requirement": False,
            "measured_dependence_not_used_to_set_wb85b_thresholds": True,
            "corr_T_LS_T_cov": R1_CORR_T,
            "corr_reject_LS_reject_cov": R1_CORR_REJECT,
        },
    }


def require_frozen_criticals() -> None:
    if abs(CRITICAL_LS - EXPECTED_CRITICAL_LS) > 1.0e-12:
        raise WB85BError(f"CRITICAL_LS drifted: {CRITICAL_LS}")
    if abs(CRITICAL_COV - EXPECTED_CRITICAL_COV) > 1.0e-12:
        raise WB85BError(f"CRITICAL_COV drifted: {CRITICAL_COV}")
    if abs((ALPHA_LS + ALPHA_COV) - FAMILYWISE_ALPHA) > 1.0e-15:
        raise WB85BError("Bonferroni split does not exhaust familywise_alpha")


def statistical_family() -> dict[str, Any]:
    require_frozen_criticals()
    return {
        "primary_statistical_family": ("H_LS", "H_cov"),
        "H_LS": "projected-z location/scale calibration",
        "H_cov": "projected 95% interval coverage calibration",
        "multiplicity": "fixed_bonferroni",
        "familywise_alpha": FAMILYWISE_ALPHA,
        "alpha_LS": ALPHA_LS,
        "alpha_cov": ALPHA_COV,
        "fwer_bound": "FWER <= alpha_LS + alpha_cov <= 0.05 under arbitrary dependence",
        "not_fitted_to_wb85a_r1_correlations": True,
        "alpha_allocation_sweep_forbidden": True,
        "post_hoc_reallocation_after_physical_outcomes_forbidden": True,
        "T_LS_null": f"chi2(df={DF_LS})",
        "T_cov_null": f"chi2(df={DF_COV})",
        "critical_LS": CRITICAL_LS,
        "critical_cov": CRITICAL_COV,
        "primary_statistical_fail_if": "T_LS > critical_LS OR T_cov > critical_cov",
    }


def fwer_claim_scope() -> dict[str, Any]:
    return {
        "unconditional_overall_qualification_false_fail_le_0.05": False,
        "primary_statistical_FWER_cap": FAMILYWISE_ALPHA,
        "conditional_on_valid_analyzable_execution": True,
        "conditional_claim": (
            "Given all required strata exist, solvable n>=200, finite outputs, "
            "convergence according to the frozen contract, and SPD/FD/execution "
            "prerequisites, P(false statistical FAIL from H_LS or H_cov) <= 0.05 "
            "for a fully calibrated statistical null."
        ),
        "guardrails_are_not_members_of_the_0.05_statistical_family": (
            "missing stratum",
            "n < 200",
            "nonconvergence",
            "NaN/Inf",
            "non-SPD",
            "FD failure",
            "dropped identifiable mode",
            "runtime/provenance failure",
        ),
        "guardrail_semantics": "frozen fail-closed / UNKNOWN, independent of Bonferroni",
        "engineering_research_screening_guards_unchanged": {
            "translation_mm": ENGINEERING_TRANSLATION_MM,
            "rotation_mrad": ENGINEERING_RZ_MRAD,
            "weak_catastrophic_abs_mean_z": WEAK_STANDARDIZED_CATASTROPHIC,
            "fd_rel_max": FD_REL_MAX,
            "max_nonconvergence_fraction": MAX_NONCONVERGENCE_FRACTION,
            "coverage_gross_empirical_min": COVERAGE_GROSS_EMPIRICAL_MIN,
            "coverage_gross_cp_upper_min": COVERAGE_GROSS_CP_UPPER_MIN,
        },
    }


def frozen_physical_contract() -> dict[str, Any]:
    return {
        "wb84_corpus_membership_unchanged": True,
        "geometry_families_unchanged": True,
        "survey_tags_unchanged": True,
        "projected_scientific_modes_unchanged": True,
        "weak_jg_status_unchanged": True,
        "common_track_solver_objective_unchanged": True,
        "q_over_p_prior_unchanged": True,
        "schur_treatment_unchanged": True,
        "left_se3_update_unchanged": True,
        "max_iterations_unchanged": True,
        "calypso_segmentfit_refit_unchanged": True,
        "acts_mode0_unchanged": True,
        "relinearization_unchanged": True,
        "catastrophic_engineering_thresholds_unchanged": True,
        "truth_only_association_contract_unchanged": True,
        "no_new_model_training_association_or_physical_dataset": True,
        "n_min_per_identifiable_stratum": N_MIN_PER_IDENTIFIABLE_STRATUM,
        "identifiable_stratum_ids": identifiable_ids(),
    }


def power_mean_wb85b(mu: float, *, n: int = N_MIN_PER_IDENTIFIABLE_STRATUM) -> float:
    require_frozen_criticals()
    nc = float(n) * float(mu) ** 2
    return float(stats.ncx2.sf(CRITICAL_LS, DF_LS, nc))


def power_scale_wb85b(scale: float, *, n: int = N_MIN_PER_IDENTIFIABLE_STRATUM) -> float:
    require_frozen_criticals()
    var = float(scale) ** 2
    nc = 0.5 * (n - 1) * (var - 1.0) ** 2
    return float(stats.ncx2.sf(CRITICAL_LS, DF_LS, nc))


def analytic_power_table(*, n: int = N_MIN_PER_IDENTIFIABLE_STRATUM) -> dict[str, Any]:
    return {
        "n": n,
        "critical_LS": CRITICAL_LS,
        "alpha_LS": ALPHA_LS,
        "analytic_is_approximation": True,
        "persistent_mean_0.35_one_stratum": power_mean_wb85b(ALT_MEAN_PRIMARY, n=n),
        "scale_1.25_one_stratum": power_scale_wb85b(ALT_SCALE, n=n),
        "scale_deflation_0.763_one_stratum": power_scale_wb85b(0.7632762616558972, n=n),
        "must_meet_power_target": POWER_TARGET,
        "required_alternatives": (
            "persistent_mean_0.35_one_stratum",
            "scale_1.25_one_stratum",
        ),
    }


def apply_wb85b_statistical_gate(*, t_ls: float, t_cov: float) -> dict[str, Any]:
    require_frozen_criticals()
    ls_fail = float(t_ls) > CRITICAL_LS
    cov_fail = float(t_cov) > CRITICAL_COV
    return {
        "T_LS": float(t_ls),
        "T_cov": float(t_cov),
        "critical_LS": CRITICAL_LS,
        "critical_cov": CRITICAL_COV,
        "reject_LS": ls_fail,
        "reject_cov": cov_fail,
        "primary_statistical_fail": bool(ls_fail or cov_fail),
        "wb85_alpha_0.05_accepted_flags_are_not_used": True,
    }


def evaluate_wb85b_qualification(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """WB85b decision.  Guardrails stay outside the Bonferroni family."""
    guard = evaluate_guardrails(bundle)
    if guard["failed"]:
        return {
            "qualification_status": "FAIL",
            "reason": "catastrophic guardrail",
            "decision_family": "guardrail_not_statistical_fwer",
            "guardrails": guard,
        }
    try:
        gof = location_scale_gof(bundle["z"])
    except Exception as error:
        return {
            "qualification_status": "FAIL",
            "reason": str(error),
            "decision_family": "guardrail_not_statistical_fwer",
            "guardrails": guard,
        }
    if gof.get("qualification_status") == "UNKNOWN":
        return {
            "qualification_status": "UNKNOWN",
            "decision_family": "guardrail_not_statistical_fwer",
            "gof": gof,
            "guardrails": guard,
        }
    coverage = coverage_calibration(bundle["coverage"])
    if coverage.get("qualification_status") == "UNKNOWN":
        return {
            "qualification_status": "UNKNOWN",
            "decision_family": "guardrail_not_statistical_fwer",
            "gof": gof,
            "coverage": coverage,
            "guardrails": guard,
        }
    if coverage.get("gross_failure"):
        return {
            "qualification_status": "FAIL",
            "reason": "coverage grossly below nominal",
            "decision_family": "guardrail_not_statistical_fwer",
            "gof": gof,
            "coverage": coverage,
            "guardrails": guard,
        }
    gate = apply_wb85b_statistical_gate(t_ls=float(gof["T_LS"]), t_cov=float(coverage["T_cov"]))
    status = "FAIL" if gate["primary_statistical_fail"] else "PASS"
    return {
        "qualification_status": status,
        "decision_family": "primary_statistical_bonferroni",
        "statistical_gate": gate,
        "gof_statistic_only": {"T_LS": gof["T_LS"], "df": gof["df"]},
        "coverage_statistic_only": {"T_cov": coverage["T_cov"], "df": coverage["df"]},
        "wb85_accepted_flags_ignored": {
            "location_scale_gof.accepted": gof["accepted"],
            "coverage_calibration.accepted": coverage["accepted"],
        },
        "guardrails": guard,
    }


def protocol_freeze() -> dict[str, Any]:
    require_frozen_criticals()
    return {
        **locked_flags(),
        "schema": "wb85b_fwer_protocol_v1",
        "historical_revision_semantics": historical_revision_semantics(),
        "statistical_family": statistical_family(),
        "fwer_claim_scope": fwer_claim_scope(),
        "frozen_physical_contract": frozen_physical_contract(),
        "analytic_power": analytic_power_table(),
        "healthy_identifiable_row_schema": sorted(healthy_identifiable_row(200, 190)),
    }


def write_protocol_freeze(output_root: Path | None = None) -> Path:
    root = OUTPUT_ROOT if output_root is None else Path(output_root)
    refuse_historical_rewrite(root)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "wb85b_fwer_protocol.json", protocol_freeze())
    return root
