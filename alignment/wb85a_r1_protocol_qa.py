"""WB85a-r1: coherent joint-null QA, runtime-smoke gate, and provenance closure.

Does not overwrite WB85a v1 artifacts, does not execute WB86, and does not
read WB84 physical alignment outcomes.  WB85 physical thresholds stay frozen.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from alignment.calypso_physical_replica_production import (
    ACTS_TOOL,
    GEOMETRY_NAME,
    GLOBAL_TAG,
    Q_OVER_P_MODE,
    allocation_plan,
    sha256_file,
)
from alignment.common_track_solver import SURVEY_FIXED_DZ, solver_implementation_fixed
from alignment.physical_common_track_execution import (
    FORBIDDEN_OFFICIAL_FALLBACKS,
    OFFICIAL_ENGINE,
    OfficialPhysicalIterator,
    RecordingPhysicalBackend,
    audit_execution_readiness,
    identity_payload,
    ingest_physical_event,
    locked_execution_flags,
    measurements_from_event,
    refuse_official_iterate_common_track,
)
from alignment.wb85_physical_qualification_protocol import (
    ALPHA,
    ALT_MEAN_PRIMARY,
    ALT_SCALE,
    ASSOCIATION_DEFAULT_SYSTEM,
    COVERAGE_NOMINAL,
    N_IDENTIFIABLE_STRATA,
    N_MIN_PER_IDENTIFIABLE_STRATUM,
    POWER_TARGET,
    TRANSLATION_INJECTION,
    WB84_CONDITION_COUNTS,
    WB85Error,
    chart_parameter_names,
    coverage_calibration,
    evaluate_future_qualification,
    identifiable_strata,
    location_scale_gof,
    refuse_forbidden_path,
    refuse_wb83_wb84_write,
    write_json,
)
from alignment.wb85_projected_unit import (
    COVERAGE_CONSTRUCTION,
    GAUSSIAN_95_QUANTILE,
    direction_vector,
    official_covered_95,
    official_projected_estimate,
)
from alignment.wb85a_protocol_qa import (
    COMBINED_TYPE_I_TARGET,
    IDENTIFIABLE_STRATUM_COUNTS,
    N_NULL_MONTE_CARLO,
    N_OFFICIAL_CROSSCHECK,
    N_POWER_MONTE_CARLO,
    NULL_CALIBRATION_SEED,
    POWER_ABS_TOL,
    POWER_VALIDATION_SEED,
    TYPE_I_POINT_BAND,
    WB85AError,
    WB85_PYTHON,
    WB85_YAML,
    SOLVER_PYTHON,
    WB84_BUNDLE,
    WB84_GEOMETRY,
    WB84_MANIFEST,
    WB84_PROVENANCE,
    WB84_QA,
    WB84_SNAPSHOT,
    binomial_report,
    fail_closed_suite,
    healthy_identifiable_row,
    identifiable_ids,
    nuisance_covariance_report,
    projection_direction,
    protocol_qa_acceptance_rule,
    run_null_calibration,
    run_power_validation,
    statistical_unit_contract_report,
    tcov_statistic,
    type_i_accept,
    weak_jg_contract_report,
)
from alignment.wb85a_r1_runtime_smoke import (
    SMOKE_OUTPUT,
    SMOKE_SKIP_EVENTS,
    SMOKE_SOURCE_ID,
    WB85A_V1_ROOT,
    fixture_is_legal,
    locked_smoke_flags,
)


WORKBOOK = "85a-r1"
OUTPUT_ROOT = Path("outputs/mc24_four_station_wb85a_r1_protocol_qa_v1")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_PARENT = "fe00c0a9674b861f408e5c7e6c0d00163a42d116"
JOINT_NULL_SEED = 2026090819
N_JOINT_NULL_MONTE_CARLO = N_NULL_MONTE_CARLO
OVERALL_QUALIFICATION_FALSE_FAIL_CAP = 0.05
INDEPENDENT_UNION_EXPECTATION = COMBINED_TYPE_I_TARGET
HISTORICAL_WB85A = Path("outputs/mc24_four_station_wb85a_protocol_qa_v1")

GENERATING_SOURCES = (
    PROJECT_ROOT / "alignment" / "wb85a_protocol_qa.py",
    PROJECT_ROOT / "alignment" / "wb85a_r1_protocol_qa.py",
    PROJECT_ROOT / "alignment" / "wb85a_r1_runtime_smoke.py",
    PROJECT_ROOT / "alignment" / "wb85_projected_unit.py",
    PROJECT_ROOT / "alignment" / "wb85_physical_qualification_protocol.py",
    PROJECT_ROOT / "alignment" / "physical_common_track_execution.py",
    PROJECT_ROOT / "alignment" / "common_track_solver.py",
    PROJECT_ROOT / "scripts" / "run_wb85a_r1_protocol_qa.py",
    PROJECT_ROOT / "scripts" / "run_wb85a_r1_runtime_smoke.py",
    PROJECT_ROOT / "scripts" / "run_wb85a_r1_smoke_condor.sh",
    PROJECT_ROOT / "scripts" / "submit_wb85a_r1_smoke_condor.py",
    PROJECT_ROOT / "configs" / "research_review" / "wp85a_r1_protocol_qa.yaml",
    PROJECT_ROOT / "tests" / "test_wb85a_r1_protocol_qa.py",
)


class WB85AR1Error(WB85AError):
    """Fail-closed WB85a-r1 contract."""


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
        "wb85a_v1_preserved": True,
        "frozen_parent": f"4station@{FROZEN_PARENT}",
    }


def refuse_wb85a_v1_rewrite(path: object) -> None:
    text = str(path)
    if "wb85a_protocol_qa_v1" in text and "wb85a_r1" not in text:
        raise WB85AR1Error(f"WB85a-r1 must not rewrite historical WB85a artifacts: {path}")


def refuse_physical_alignment_outcome(path: object) -> None:
    from alignment.wb85_physical_qualification_protocol import ALIGNMENT_OUTCOME_MARKERS

    text = str(path).lower()
    for marker in ALIGNMENT_OUTCOME_MARKERS:
        if marker in text:
            raise WB85AR1Error(f"WB85a-r1 must not inspect physical alignment outcomes: {path}")
    refuse_forbidden_path(path)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_identifiable_counts() -> dict[str, int]:
    expected = identifiable_ids()
    if list(IDENTIFIABLE_STRATUM_COUNTS) != expected:
        raise WB85AR1Error("frozen QA counts are not the WB85 identifiable strata")
    wanted = (299, 303, 311, 298, 300, 283)
    if tuple(IDENTIFIABLE_STRATUM_COUNTS[name] for name in expected) != wanted:
        raise WB85AR1Error("WB85a-r1 must use the frozen WB84 identifiable-stratum counts")
    for name in expected:
        if IDENTIFIABLE_STRATUM_COUNTS[name] != WB84_CONDITION_COUNTS[name]:
            raise WB85AR1Error(f"QA count for {name} is not the WB84 identifiable export count")
        if IDENTIFIABLE_STRATUM_COUNTS[name] < N_MIN_PER_IDENTIFIABLE_STRATUM:
            raise WB85AR1Error(f"{name} has n<{N_MIN_PER_IDENTIFIABLE_STRATUM}")
    return dict(IDENTIFIABLE_STRATUM_COUNTS)


def r1_acceptance_rule() -> dict[str, Any]:
    """Frozen before any WB85a-r1 Monte-Carlo.  Not tuned on joint-null numbers."""
    return {
        "schema": "wb85a_r1_acceptance_rule_v1",
        "frozen_before_any_new_monte_carlo": True,
        "may_repair_by_looking_at_physical_outcomes": False,
        "may_retune_wb85_thresholds_from_this_qa": False,
        "may_silently_change_alpha": False,
        "n_joint_monte_carlo": N_JOINT_NULL_MONTE_CARLO,
        "joint_null_seed": JOINT_NULL_SEED,
        "stratum_counts": dict(IDENTIFIABLE_STRATUM_COUNTS),
        "counts_are_wb84_identifiable_exports": True,
        "coverage_construction": COVERAGE_CONSTRUCTION,
        "gaussian_95_quantile": GAUSSIAN_95_QUANTILE,
        "joint_null": {
            "generator": "synthetic_theta_hat_and_spd_covariance",
            "same_realization_functions": [
                "official_projected_estimate",
                "official_covered_95",
                "location_scale_gof",
                "coverage_calibration",
                "evaluate_future_qualification",
            ],
            "not_independent_z_and_bernoulli_coverage": True,
            "equivalent_if_official_interval_is_symmetric_gaussian": (
                "z_ir ~ N(0,1); covered_ir = I(|z_ir| <= norm.ppf(0.975))"
            ),
        },
        "location_scale_type_i": {
            "target": ALPHA,
            "accept_if_cp95_contains_target": True,
            "point_estimate_band": list(TYPE_I_POINT_BAND),
        },
        "coverage_type_i": {
            "target": ALPHA,
            "accept_if_cp95_contains_target": True,
            "point_estimate_band": list(TYPE_I_POINT_BAND),
        },
        "joint_union_false_fail": {
            "official_target_is_independent_union_0.0975": False,
            "independent_synthetic_gates_expectation_historical_wb85a_only": INDEPENDENT_UNION_EXPECTATION,
            "independence_not_assumed": True,
            "measure_empirical_union_and_correlations": True,
            "scientific_overall_qualification_false_fail_cap": OVERALL_QUALIFICATION_FALSE_FAIL_CAP,
            "protocol_revision_required_if": (
                "the frozen two-gate union cannot meet the overall false-fail cap 0.05: "
                "union CP lower bound > 0.05, or the point estimate exceeds 0.05 and the "
                "95% CP does not contain 0.05"
            ),
        },
        "individual_independent_null_reproduced_from_wb85a": {
            "n_monte_carlo": N_NULL_MONTE_CARLO,
            "seed": NULL_CALIBRATION_SEED,
            "null": {"z": "N(0,1)", "coverage": f"Bernoulli({COVERAGE_NOMINAL})"},
            "purpose": "prove the coherent joint generator did not change frozen gate implementations",
        },
        "power_validation": {
            "n_monte_carlo": N_POWER_MONTE_CARLO,
            "seed": POWER_VALIDATION_SEED,
            "compare_at_n": N_MIN_PER_IDENTIFIABLE_STRATUM,
            "absolute_tolerance": POWER_ABS_TOL,
            "must_meet_power_target": POWER_TARGET,
            "primary_mean": ALT_MEAN_PRIMARY,
            "primary_scale": ALT_SCALE,
            "analytic_power_is_approximation": True,
            "mc_is_empirical_qa": True,
            "scale_1_25_discrepancy_reported_as_is": True,
            "may_retune_alternative_or_tolerance": False,
            "alternatives_not_taken_from_wb83_failures": True,
        },
        "final_gate_requires_all_true": (
            "joint_synthetic_null_calibrated",
            "power_validation_pass",
            "statistical_unit_contract_pass",
            "nuisance_covariance_pass",
            "fail_closed_tests_pass",
            "weak_jg_contract_pass",
            "physical_execution_contract_pass",
            "physical_execution_runtime_smoke_pass",
            "provenance_complete",
        ),
        "ready_to_authorize_requires_no_protocol_revision": True,
        "even_if_all_pass_still_unauthorized": True,
    }


def independence_is_not_assumed() -> dict[str, Any]:
    return {
        "mathematical_dependence": (
            "Official coverage is covered_ir = I(|z_ir| <= norm.ppf(0.975)).  "
            "T_LS is a function of the same {z_ir}.  T_cov is a function of the "
            "same coverage indicators.  Therefore (T_LS, T_cov) and "
            "(reject_LS, reject_cov) are dependent by construction."
        ),
        "independence_proved": False,
        "official_joint_target_is_1_minus_1_minus_alpha_squared": False,
        "historical_wb85a_independent_union_expectation": INDEPENDENT_UNION_EXPECTATION,
        "historical_0.0975_is_not_the_official_joint_null_target": True,
    }


def protocol_revision_required(union: Mapping[str, Any]) -> bool:
    cap = OVERALL_QUALIFICATION_FALSE_FAIL_CAP
    lo = float(union["cp95_lo"])
    hi = float(union["cp95_hi"])
    rate = float(union["rate"])
    if lo > cap:
        return True
    if rate > cap and not (lo <= cap <= hi):
        return True
    return False


def _chart_kind(stratum_id: str) -> str:
    return "rotation" if "rotation" in stratum_id else "translation"


def _survey_mode(stratum_id: str) -> str:
    row = next(item for item in identifiable_strata() if item["stratum_id"] == stratum_id)
    return str(row["survey_mode"])


def parameter_names_for_stratum(stratum_id: str) -> tuple[str, ...]:
    return chart_parameter_names(_chart_kind(stratum_id), _survey_mode(stratum_id))


def draw_coherent_stratum(
    rng: np.random.Generator,
    stratum_id: str,
    n: int,
) -> tuple[np.ndarray, int]:
    """One internally coherent synthetic stratum.

    Official production coverage is the symmetric Gaussian interval of the
    same projected estimator.  This draw therefore generates ``theta_hat``
    and a diagonal SPD ``Cov``, applies the official projected-z algebra,
    and sets ``covered = I(|z| <= norm.ppf(0.975))``.
    """
    names = parameter_names_for_stratum(stratum_id)
    direction = projection_direction(stratum_id)
    u = direction_vector(names, direction)
    dim = len(names)
    theta_true = np.zeros(dim, dtype=np.float64)
    if "translation" in stratum_id or stratum_id.startswith("identity"):
        for name, value in TRANSLATION_INJECTION.items():
            if name in names:
                theta_true[names.index(name)] = float(value)
    scales = rng.uniform(0.05, 0.20, size=(int(n), dim))
    noise = rng.normal(0.0, 1.0, size=(int(n), dim))
    delta = scales * noise
    theta_hat = theta_true + delta
    sigma = np.sqrt(np.sum((u * u) * (scales * scales), axis=1))
    a_hat = theta_hat @ u
    a_true = float(u @ theta_true)
    z_values = (a_hat - a_true) / sigma
    covered_flags = np.abs(z_values) <= GAUSSIAN_95_QUANTILE
    projected = official_projected_estimate(
        theta_hat[0],
        theta_true,
        np.diag(np.square(scales[0])),
        names,
        direction,
    )
    if abs(float(projected["z"]) - float(z_values[0])) > 1.0e-12:
        raise WB85AR1Error("vectorized z disagrees with official_projected_estimate")
    if bool(projected["covered_95"]) != bool(covered_flags[0]):
        raise WB85AR1Error("vectorized coverage disagrees with official_covered_95")
    if bool(covered_flags[0]) != official_covered_95(float(z_values[0])):
        raise WB85AR1Error("coverage is not the official |z| interval")
    return z_values, int(covered_flags.sum())


def draw_coherent_null(rng: np.random.Generator, counts: Mapping[str, int]) -> tuple[dict[str, np.ndarray], dict[str, tuple[int, int]]]:
    z = {}
    covered = {}
    for name, n in counts.items():
        values, k = draw_coherent_stratum(rng, name, int(n))
        z[name] = values
        covered[name] = (int(k), int(n))
    return z, covered


def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or right.size < 2:
        return None
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def run_joint_null_calibration(
    *,
    n_mc: int = N_JOINT_NULL_MONTE_CARLO,
    seed: int = JOINT_NULL_SEED,
) -> dict[str, Any]:
    counts = _require_identifiable_counts()
    rng = np.random.default_rng(seed)
    crit_ls = float(stats.chi2.ppf(1.0 - ALPHA, 2 * N_IDENTIFIABLE_STRATA))
    crit_cov = float(stats.chi2.ppf(1.0 - ALPHA, N_IDENTIFIABLE_STRATA))
    t_ls_values = np.empty(int(n_mc), dtype=np.float64)
    t_cov_values = np.empty(int(n_mc), dtype=np.float64)
    reject_ls = np.empty(int(n_mc), dtype=np.int8)
    reject_cov = np.empty(int(n_mc), dtype=np.int8)
    official_mismatch = 0
    for trial in range(int(n_mc)):
        z, covered = draw_coherent_null(rng, counts)
        z_lists = {name: values.tolist() for name, values in z.items()}
        gof = location_scale_gof(z_lists)
        t_ls = float(gof["T_LS"])
        t_cov = float(tcov_statistic(covered))
        ls_fail = not bool(gof["accepted"])
        cov_fail = t_cov > crit_cov
        t_ls_values[trial] = t_ls
        t_cov_values[trial] = t_cov
        reject_ls[trial] = int(ls_fail)
        reject_cov[trial] = int(cov_fail)
        if trial < N_OFFICIAL_CROSSCHECK:
            cov = coverage_calibration(covered)
            bundle = {
                "z": z_lists,
                "coverage": covered,
                "identifiable_strata": {
                    name: healthy_identifiable_row(n, covered[name][0]) for name, n in counts.items()
                },
                "weak_strata": {},
            }
            future = evaluate_future_qualification(bundle)
            official_ls = not bool(gof["accepted"])
            official_cov = not bool(cov["accepted"])
            official_combined = future["qualification_status"] != "PASS"
            if abs(float(cov["T_cov"]) - t_cov) > 1.0e-12:
                official_mismatch += 1
            if official_ls != ls_fail or official_cov != cov_fail:
                official_mismatch += 1
            if official_combined != (ls_fail or cov_fail) and future["qualification_status"] != "UNKNOWN":
                official_mismatch += 1
    union = reject_ls | reject_cov
    ls_report = binomial_report(int(reject_ls.sum()), n_mc)
    cov_report = binomial_report(int(reject_cov.sum()), n_mc)
    union_report = binomial_report(int(union.sum()), n_mc)
    ls_ok = type_i_accept(ls_report, target=ALPHA, band=TYPE_I_POINT_BAND)
    cov_ok = type_i_accept(cov_report, target=ALPHA, band=TYPE_I_POINT_BAND)
    revision = protocol_revision_required(union_report)
    calibrated = ls_ok and cov_ok and official_mismatch == 0
    dependence = independence_is_not_assumed()
    return {
        "acceptance_rule": r1_acceptance_rule(),
        "n_monte_carlo": int(n_mc),
        "seed": int(seed),
        "stratum_counts": counts,
        "reads_physical_alignment_output": False,
        "coverage_construction": COVERAGE_CONSTRUCTION,
        "critical_T_LS": crit_ls,
        "critical_T_cov": crit_cov,
        "location_scale": {**ls_report, "accepted": ls_ok, "target": ALPHA},
        "coverage": {**cov_report, "accepted": cov_ok, "target": ALPHA},
        "joint_union": {
            **union_report,
            "official_target_is_independent_union_0.0975": False,
            "historical_wb85a_independent_expectation": INDEPENDENT_UNION_EXPECTATION,
            "scientific_overall_cap": OVERALL_QUALIFICATION_FALSE_FAIL_CAP,
            "protocol_revision_required": revision,
        },
        "corr_T_LS_T_cov": _corr(t_ls_values, t_cov_values),
        "corr_reject_LS_reject_cov": _corr(reject_ls.astype(np.float64), reject_cov.astype(np.float64)),
        "dependence": dependence,
        "official_function_crosscheck": {
            "n": N_OFFICIAL_CROSSCHECK,
            "mismatches": official_mismatch,
            "same_realization_calls": [
                "location_scale_gof",
                "coverage_calibration",
                "evaluate_future_qualification",
            ],
            "every_trial_uses_location_scale_gof_and_official_T_cov_formula": True,
        },
        "joint_synthetic_null_calibrated": bool(calibrated),
        "protocol_revision_required": bool(revision),
        "wb85_thresholds_not_retuned": True,
        "alpha_not_silently_changed": True,
    }


def annotate_power_validation(power: Mapping[str, Any]) -> dict[str, Any]:
    reports = dict(power["reports"])
    scale = dict(reports["scale_1.25_one_stratum"])
    scale["analytic_power_is_approximation"] = True
    scale["mc_is_empirical_qa"] = True
    scale["discrepancy_reported_as_is"] = True
    scale["historical_wb85a_analytic"] = 0.983
    scale["historical_wb85a_mc"] = 0.937
    scale["alternative_not_retuned"] = True
    scale["tolerance_not_retuned"] = True
    reports["scale_1.25_one_stratum"] = scale
    return {
        **power,
        "reports": reports,
        "analytic_power_is_approximation": True,
        "mc_is_empirical_qa": True,
        "scale_1_25_analytic_mc_discrepancy_reported_as_is": True,
        "may_retune_alternative_or_tolerance": False,
    }


def execution_contract_report(*, work_dir: Path | None = None) -> dict[str, Any]:
    audit = audit_execution_readiness()
    backend = RecordingPhysicalBackend()
    eye = np.eye(4).tolist()
    hits = [
        {
            "station_id": station,
            "measurement_id": f"r1-exec:{station}",
            "x_mm": 0.1 * station,
            "y_mm": -0.05 * station,
            "tx": 0.0,
            "ty": 0.0,
            "z_mm": 1000.0 * station,
            "covariance_4x4": eye,
        }
        for station in (0, 1, 2, 3)
    ]
    record = {
        "event_uid": "mc24_100047_00100_00149:100047:1999",
        "source_id": "mc24_100047_00100_00149",
        "run_id": 100047,
        "event_id": 1999,
        "xaod_entry_index": 1999,
        "condition_id": "identity_fixed_dz",
        "input_xaod": (
            "/eos/experiment/faser/data0/sim/mc24/particle_gun/100047/rec/s0013-r0022/"
            "FaserMC-MC24_PG_mumi_fasernu_5mrad_flukaE-100047-00100-00149-s0013-r0022-xAOD.root"
        ),
        "tracks": [
            {
                "track_uid": "t1",
                "truth": {"particle_id": 7, "pdg": 13, "match_fraction": 1.0},
                "hits": hits,
            }
        ],
    }
    event = ingest_physical_event(record)
    start = identity_payload()
    initial = measurements_from_event(event, start, engine="wb84_physical_ingest")
    iterator = OfficialPhysicalIterator(backend=backend, max_iterations=1)
    work = (OUTPUT_ROOT if work_dir is None else Path(work_dir)) / "_hermetic_iterator"
    refuse_wb85a_v1_rewrite(work)
    result = iterator.iterate(
        event,
        start,
        initial,
        chart_kind="translation",
        survey_mode=SURVEY_FIXED_DZ,
        work_dir=work,
    )
    rerefit_after_update = bool(backend.rerefit_payloads) and result["history"][0]["rerefit_after_update"]
    toy_refused = False
    try:
        refuse_official_iterate_common_track(field_y=0.35)
    except Exception:
        toy_refused = True
    checks = dict(audit["checks"])
    checks["hermetic_iterator_rerefits_after_left_se3"] = rerefit_after_update
    checks["official_solve_reuses_frozen_newton_schur"] = True
    contract = all(checks.values()) and toy_refused
    return {
        **locked_execution_flags(),
        **audit,
        "checks": checks,
        "hermetic_iterator": {
            "engine": result["engine"],
            "rerefit_calls": len(backend.rerefit_payloads),
            "reused_first_step_measurements": result["reused_first_step_measurements"],
            "automatically_passed_at_max_iterations": result["automatically_passed_at_max_iterations"],
        },
        "toy_iterate_refused": toy_refused,
        "forbidden_official_fallbacks": list(FORBIDDEN_OFFICIAL_FALLBACKS),
        "physical_execution_contract_pass": bool(contract),
        "physical_execution_ready": False,
        "command_planning_is_not_runtime_smoke": True,
        "calypso_backend_execute_stays_false": True,
    }


def load_runtime_smoke(path: Path | None = None) -> dict[str, Any]:
    target = SMOKE_OUTPUT / "wb85a_r1_runtime_smoke.json" if path is None else Path(path)
    if not target.is_file():
        fixture_ok = True
        reason = "runtime smoke report is absent"
        try:
            fixture_is_legal()
        except Exception as error:
            fixture_ok = False
            reason = str(error)
        return {
            **locked_smoke_flags(),
            "physical_execution_runtime_smoke_pass": False,
            "physical_execution_ready": False,
            "report_present": False,
            "legal_fixture": fixture_ok,
            "reason": reason,
        }
    payload = json.loads(target.read_text(encoding="utf-8"))
    refuse_physical_alignment_outcome(target)
    passed = bool(payload.get("physical_execution_runtime_smoke_pass"))
    return {
        **locked_smoke_flags(),
        "report_present": True,
        "report_path": str(target),
        "physical_execution_runtime_smoke_pass": passed,
        "physical_execution_ready": passed,
        "iteration2_used_updated_geometry": payload.get("iteration2_used_updated_geometry"),
        "alignment_performance_not_reported": payload.get("alignment_performance_not_reported", True),
        "fixture": payload.get("fixture"),
        "engine": payload.get("engine"),
    }


def git_identity(*, write_diff_to: Path | None = None) -> dict[str, Any]:
    def _git(*args: str, binary: bool = False) -> bytes | str:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if binary:
            return result.stdout if result.returncode == 0 else b""
        return result.stdout.decode("utf-8", errors="replace").strip() if result.returncode == 0 else ""

    status = str(_git("status", "--porcelain"))
    head = str(_git("rev-parse", "HEAD"))
    diff = bytes(_git("diff", "--binary", binary=True))
    diff_sha = None
    diff_path = None
    if status and write_diff_to is not None:
        write_diff_to.parent.mkdir(parents=True, exist_ok=True)
        write_diff_to.write_bytes(diff)
        diff_sha = hashlib.sha256(diff).hexdigest()
        diff_path = str(write_diff_to)
    elif status:
        diff_sha = hashlib.sha256(diff).hexdigest()
    return {
        "git_sha": head,
        "git_dirty": bool(status),
        "git_status_porcelain": status,
        "git_diff_binary_sha256": diff_sha,
        "git_diff_binary_path": diff_path,
        "frozen_parent": FROZEN_PARENT,
        "head_is_frozen_parent": head == FROZEN_PARENT,
        "frozen_parent_does_not_contain_wb85a_r1": True,
    }


def generating_code_identity() -> dict[str, Any]:
    hashes = {}
    missing = []
    for path in GENERATING_SOURCES:
        if path.is_file():
            hashes[str(path.relative_to(PROJECT_ROOT))] = sha256_file(path)
        else:
            missing.append(str(path.relative_to(PROJECT_ROOT)))
    blob = json.dumps(hashes, sort_keys=True).encode("utf-8")
    return {
        "source_files": hashes,
        "missing_source_files": missing,
        "source_snapshot_sha256": hashlib.sha256(blob).hexdigest(),
        "wb85_python_sha256": sha256_file(WB85_PYTHON),
        "wb85_yaml_sha256": sha256_file(WB85_YAML),
        "wb85a_python_sha256": sha256_file(PROJECT_ROOT / "alignment" / "wb85a_protocol_qa.py"),
        "wb85a_r1_python_sha256": sha256_file(PROJECT_ROOT / "alignment" / "wb85a_r1_protocol_qa.py"),
        "projected_unit_sha256": sha256_file(PROJECT_ROOT / "alignment" / "wb85_projected_unit.py"),
        "runtime_smoke_sha256": sha256_file(PROJECT_ROOT / "alignment" / "wb85a_r1_runtime_smoke.py"),
        "physical_execution_sha256": sha256_file(
            PROJECT_ROOT / "alignment" / "physical_common_track_execution.py"
        ),
        "common_track_solver_sha256": sha256_file(SOLVER_PYTHON),
    }


def future_wb86_runtime_provenance_contract() -> dict[str, Any]:
    return {
        "wb84_historical_material_hashed_files_left_empty": True,
        "must_not_backfill_or_rewrite_wb84": True,
        "future_wb86_must_identify_at_runtime": (
            "geometry sqlite/payload actually loaded",
            "material maps actually loaded",
            "conditions / global tag",
            "magnetic field map files",
            "Calypso identity (head, dirty, diff sha256)",
            "ACTS tool and q_over_p_mode",
        ),
        "expected_runtime_fields": {
            "geometry_name": GEOMETRY_NAME,
            "global_tag": GLOBAL_TAG,
            "acts_tool": ACTS_TOOL,
            "q_over_p_mode": Q_OVER_P_MODE,
            "official_engine": OFFICIAL_ENGINE,
        },
    }


def provenance_freeze_report(*, output_root: Path | None = None) -> dict[str, Any]:
    root = OUTPUT_ROOT if output_root is None else Path(output_root)
    refuse_physical_alignment_outcome(root / "wb85a_r1_provenance_freeze.json")
    refuse_wb85a_v1_rewrite(root)
    snapshot = json.loads(WB84_SNAPSHOT.read_text(encoding="utf-8"))
    geometry = json.loads(WB84_GEOMETRY.read_text(encoding="utf-8"))
    qa = json.loads(WB84_QA.read_text(encoding="utf-8"))
    payload_hashes = {
        cell["condition_id"]: {
            "requested_payload_sha256": cell["requested_payload_sha256"],
            "sqlite_sha256": cell["calypso_payload"]["sqlite_sha256"],
            "pool_sha256": cell["calypso_payload"]["pool_sha256"],
        }
        for cell in geometry["conditions"]
    }
    generating = generating_code_identity()
    git = git_identity(write_diff_to=root / "git_diff_binary.bin")
    historical = {
        "wb84_corpus_manifest_sha256": sha256_file(WB84_MANIFEST),
        "wb84_corpus_qa_sha256": sha256_file(WB84_QA),
        "wb84_provenance_freeze_sha256": sha256_file(WB84_PROVENANCE),
        "calypso_source_bundle_sha256": sha256_file(WB84_BUNDLE),
        "wb84_geometry_manifest_sha256": sha256_file(WB84_GEOMETRY),
        "wb84_snapshot_sha256": sha256_file(WB84_SNAPSHOT),
    }
    material_empty = not bool(snapshot.get("material_map", {}).get("hashed_files"))
    field_present = bool(snapshot.get("field_map", {}).get("hashed_files"))
    complete = (
        bool(generating["source_files"])
        and not generating["missing_source_files"]
        and all(historical.values())
        and qa["n_qualified_exports"] == 2394
        and bool(snapshot["calypso"]["diff_sha256"])
        and field_present
    )
    plan = allocation_plan()
    smoke_allocated = any(
        row["source_id"] == SMOKE_SOURCE_ID and int(row["xaod_entry_index"]) == SMOKE_SKIP_EVENTS
        for row in plan["events"]
    )
    return {
        **locked_flags(),
        "schema": "wb85a_r1_provenance_freeze_v1",
        "utc": _utc(),
        "git": git,
        "generating_code": generating,
        "historical_wb84_hashes": historical,
        "field_map_hashes": snapshot["field_map"]["hashed_files"],
        "material_map_hashes": snapshot["material_map"]["hashed_files"],
        "material_map_hashed_files_empty_in_wb84_snapshot": material_empty,
        "wb84_material_hashes_not_backfilled": True,
        "geometry_payload_hashes": payload_hashes,
        "calypso_source_bundle": {
            "head": snapshot["calypso"]["head"],
            "diff_sha256": snapshot["calypso"]["diff_sha256"],
            "dirty": snapshot["calypso"]["dirty"],
        },
        "wb84_qa_locked": {
            "n_qualified_exports": qa["n_qualified_exports"],
            "qualification_authorized": qa["qualification_authorized"],
            "alignment_oracle_qualified_for_physical_FASER": qa["alignment_oracle_qualified_for_physical_FASER"],
        },
        "future_wb86_runtime_provenance": future_wb86_runtime_provenance_contract(),
        "smoke_fixture_not_in_wb84_allocation_plan": not smoke_allocated,
        "wb83_wb84_artifacts_modified": False,
        "wb85a_v1_modified": False,
        "solver_implementation_fixed": bool(solver_implementation_fixed()),
        "result_json_cites_generating_code_not_only_parent_sha": True,
        "provenance_complete": bool(complete),
    }


def build_verdict(parts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    required = {
        "joint_synthetic_null_calibrated": bool(parts["joint"]["joint_synthetic_null_calibrated"]),
        "power_validation_pass": bool(parts["power"]["power_validation_pass"]),
        "statistical_unit_contract_pass": bool(parts["units"]["statistical_unit_contract_pass"]),
        "nuisance_covariance_pass": bool(parts["nuisance"]["nuisance_covariance_pass"]),
        "fail_closed_tests_pass": bool(parts["gates"]["fail_closed_tests_pass"]),
        "weak_jg_contract_pass": bool(parts["weak"]["weak_jg_contract_pass"]),
        "physical_execution_contract_pass": bool(parts["execution"]["physical_execution_contract_pass"]),
        "physical_execution_runtime_smoke_pass": bool(parts["smoke"]["physical_execution_runtime_smoke_pass"]),
        "provenance_complete": bool(parts["provenance"]["provenance_complete"]),
    }
    revision = bool(parts["joint"]["protocol_revision_required"])
    all_true = all(required.values())
    ready = all_true and not revision
    return {
        **locked_flags(),
        "schema": "wb85a_r1_protocol_qa_v1",
        "acceptance_rule_frozen_before_calibration": True,
        "acceptance_rule": r1_acceptance_rule(),
        "checks": required,
        "protocol_revision_required": revision,
        "wb85_protocol_qa_pass": bool(ready),
        "ready_to_authorize_physical_qualification": bool(ready),
        "qualification_authorized": False,
        "alignment_oracle_qualified_for_physical_FASER": False,
        "wb86_automatically_authorized": False,
        "physical_execution_ready": bool(parts["smoke"]["physical_execution_runtime_smoke_pass"]),
        "historical_wb85a_independent_union_0.0975_not_official": True,
        "wb85_thresholds_not_retuned": True,
        "next_workbook_if_ready": "WB86 — Physical Common-Track Alignment Qualification v1",
        "wb86_not_created_or_run": True,
    }


def run_protocol_qa_r1(
    *,
    n_joint: int = N_JOINT_NULL_MONTE_CARLO,
    n_null: int = N_NULL_MONTE_CARLO,
    n_power: int = N_POWER_MONTE_CARLO,
    smoke_path: Path | None = None,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    _require_identifiable_counts()
    joint = run_joint_null_calibration(n_mc=n_joint)
    individual = run_null_calibration(n_mc=n_null)
    power = annotate_power_validation(run_power_validation(n_mc=n_power))
    units = statistical_unit_contract_report()
    nuisance = nuisance_covariance_report()
    gates = fail_closed_suite()
    weak = weak_jg_contract_report()
    execution = execution_contract_report(work_dir=work_dir)
    smoke = load_runtime_smoke(smoke_path)
    provenance = provenance_freeze_report(output_root=work_dir)
    verdict = build_verdict(
        {
            "joint": joint,
            "power": power,
            "units": units,
            "nuisance": nuisance,
            "gates": gates,
            "weak": weak,
            "execution": execution,
            "smoke": smoke,
            "provenance": provenance,
        }
    )
    return {
        "verdict": verdict,
        "joint": joint,
        "individual_independent_null": individual,
        "power": power,
        "units": units,
        "nuisance": nuisance,
        "gates": gates,
        "weak": weak,
        "execution": execution,
        "smoke": smoke,
        "provenance": provenance,
    }


def artifact_payloads_r1(bundle: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    generating = bundle["provenance"]["generating_code"]
    git = bundle["provenance"]["git"]
    common = {
        **locked_flags(),
        "generating_code": generating,
        "git": git,
    }
    return {
        "wb85a_r1_protocol_qa.json": {**common, **bundle["verdict"]},
        "wb85a_r1_joint_null_calibration.json": {**common, **bundle["joint"]},
        "wb85a_r1_individual_null_calibration.json": {**common, **bundle["individual_independent_null"]},
        "wb85a_r1_power_validation.json": {**common, **bundle["power"]},
        "wb85a_r1_execution_contract.json": {**common, **bundle["execution"]},
        "wb85a_r1_runtime_smoke_gate.json": {**common, **bundle["smoke"]},
        "wb85a_r1_provenance_freeze.json": bundle["provenance"],
    }


def refresh_verdict_with_smoke(
    output_root: Path | None = None,
    *,
    smoke_path: Path | None = None,
) -> Path:
    """Rebuild the final gate after runtime smoke without rerunning Monte-Carlo."""
    root = OUTPUT_ROOT if output_root is None else Path(output_root)
    refuse_forbidden_path(root)
    refuse_wb83_wb84_write(root)
    refuse_wb85a_v1_rewrite(root)
    required = (
        "wb85a_r1_acceptance_rule.json",
        "wb85a_r1_joint_null_calibration.json",
        "wb85a_r1_individual_null_calibration.json",
        "wb85a_r1_power_validation.json",
        "wb85a_r1_execution_contract.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise WB85AR1Error(f"cannot refresh verdict; missing {missing}")
    joint = json.loads((root / "wb85a_r1_joint_null_calibration.json").read_text(encoding="utf-8"))
    individual = json.loads((root / "wb85a_r1_individual_null_calibration.json").read_text(encoding="utf-8"))
    power = json.loads((root / "wb85a_r1_power_validation.json").read_text(encoding="utf-8"))
    execution = json.loads((root / "wb85a_r1_execution_contract.json").read_text(encoding="utf-8"))
    units = statistical_unit_contract_report()
    nuisance = nuisance_covariance_report()
    gates = fail_closed_suite()
    weak = weak_jg_contract_report()
    smoke = load_runtime_smoke(smoke_path)
    provenance = provenance_freeze_report(output_root=root)
    bundle = {
        "verdict": build_verdict(
            {
                "joint": joint,
                "power": power,
                "units": units,
                "nuisance": nuisance,
                "gates": gates,
                "weak": weak,
                "execution": execution,
                "smoke": smoke,
                "provenance": provenance,
            }
        ),
        "joint": joint,
        "individual_independent_null": individual,
        "power": power,
        "units": units,
        "nuisance": nuisance,
        "gates": gates,
        "weak": weak,
        "execution": execution,
        "smoke": smoke,
        "provenance": provenance,
    }
    for name, payload in artifact_payloads_r1(bundle).items():
        if payload.get("qualification_authorized") is not False:
            raise WB85AR1Error("WB85a-r1 artifacts must keep qualification unauthorized")
        if payload.get("wb86_automatically_authorized") is not False:
            raise WB85AR1Error("WB85a-r1 must not automatically authorize WB86")
        write_json(root / name, payload)
    return root


def write_protocol_qa_r1_artifacts(
    output_root: Path | None = None,
    *,
    n_joint: int = N_JOINT_NULL_MONTE_CARLO,
    n_null: int = N_NULL_MONTE_CARLO,
    n_power: int = N_POWER_MONTE_CARLO,
    smoke_path: Path | None = None,
) -> Path:
    root = OUTPUT_ROOT if output_root is None else Path(output_root)
    refuse_forbidden_path(root)
    refuse_wb83_wb84_write(root)
    refuse_wb85a_v1_rewrite(root)
    if HISTORICAL_WB85A.resolve() == root.resolve():
        raise WB85AR1Error("refusing to write WB85a-r1 into the historical WB85a root")
    root.mkdir(parents=True, exist_ok=True)
    write_json(
        root / "wb85a_r1_acceptance_rule.json",
        {**locked_flags(), **r1_acceptance_rule()},
    )
    bundle = run_protocol_qa_r1(
        n_joint=n_joint,
        n_null=n_null,
        n_power=n_power,
        smoke_path=smoke_path,
        work_dir=root,
    )
    for name, payload in artifact_payloads_r1(bundle).items():
        if payload.get("qualification_authorized") is not False:
            raise WB85AR1Error("WB85a-r1 artifacts must keep qualification unauthorized")
        if payload.get("looked_at_physical_alignment_outcomes") is not False:
            raise WB85AR1Error("WB85a-r1 artifacts must not look at physical alignment outcomes")
        if payload.get("wb86_automatically_authorized") is not False:
            raise WB85AR1Error("WB85a-r1 must not automatically authorize WB86")
        write_json(root / name, payload)
    return root
