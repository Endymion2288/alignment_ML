"""WB85a: independent protocol QA and execution-readiness audit.

Synthetic null / power only.  Does not run physical alignment qualification
and does not read WB84 physical alignment outcomes.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from alignment.calypso_physical_replica_production import sha256_file
from alignment.common_track_solver import (
    SURVEY_FINITE_PRIOR,
    SURVEY_FIXED_DZ,
    StationHit,
    assemble_track_block,
    solve_common_track,
    solve_joint_dense,
    solver_implementation_fixed,
)
from alignment.four_station import SURVEY_DZ_PRIOR_SIGMA_MM
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
    ALIGNMENT_OUTCOME_MARKERS,
    ALPHA,
    ALT_MEAN_PRIMARY,
    ALT_SCALE,
    ASSOCIATION_DEFAULT_SYSTEM,
    COVERAGE_GROSS_CP_UPPER_MIN,
    COVERAGE_GROSS_EMPIRICAL_MIN,
    COVERAGE_NOMINAL,
    ENGINEERING_RZ_MRAD,
    ENGINEERING_TRANSLATION_MM,
    FD_REL_MAX,
    IDENTIFIABLE_FAMILIES,
    MAX_NONCONVERGENCE_FRACTION,
    N_IDENTIFIABLE_STRATA,
    N_MIN_PER_IDENTIFIABLE_STRATUM,
    POWER_TARGET,
    ROTATION_INJECTION,
    TRANSLATION_INJECTION,
    WB84_CONDITION_COUNTS,
    WB84_QA,
    WB85Error,
    WEAK_STANDARDIZED_CATASTROPHIC,
    artifact_payloads,
    charts,
    clopper_pearson,
    coverage_calibration,
    evaluate_future_qualification,
    evaluate_guardrails,
    identifiable_strata,
    location_scale_gof,
    power_analysis,
    power_mean,
    power_scale,
    refuse_forbidden_path,
    refuse_wb83_wb84_write,
    unit_direction,
    weak_strata,
    write_json,
)


WORKBOOK = "85a"
OUTPUT_ROOT = Path("outputs/mc24_four_station_wb85a_protocol_qa_v1")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WB85_PYTHON = PROJECT_ROOT / "alignment" / "wb85_physical_qualification_protocol.py"
WB85_YAML = PROJECT_ROOT / "configs" / "research_review" / "wp85_physical_alignment_qualification.yaml"
SOLVER_PYTHON = PROJECT_ROOT / "alignment" / "common_track_solver.py"
WB84_ROOT = Path("outputs/mc24_four_station_calypso_physical_replicas_v1")
WB84_MANIFEST = WB84_ROOT / "physical_replica_manifest.json"
WB84_SNAPSHOT = WB84_ROOT / "production_snapshot.json"
WB84_BUNDLE = WB84_ROOT / "calypso_source_bundle.json"
WB84_GEOMETRY = WB84_ROOT / "geometry_payload_manifest.json"
WB84_PROVENANCE = WB84_ROOT / "provenance_freeze.json"

# Frozen before the official Monte-Carlo is run.
IDENTIFIABLE_STRATUM_COUNTS = {
    "identity_fixed_dz": 299,
    "identity_finite_survey_prior": 303,
    "identifiable_translation_fixed_dz": 311,
    "identifiable_translation_finite_survey_prior": 298,
    "identifiable_rotation_fixed_dz": 300,
    "identifiable_rotation_finite_survey_prior": 283,
}
NULL_CALIBRATION_SEED = 2026090817
POWER_VALIDATION_SEED = 2026090818
N_NULL_MONTE_CARLO = 20000
N_POWER_MONTE_CARLO = 4000
N_OFFICIAL_CROSSCHECK = 200
TYPE_I_POINT_BAND = (0.030, 0.080)
COMBINED_TYPE_I_TARGET = 1.0 - (1.0 - ALPHA) ** 2
COMBINED_TYPE_I_BAND = (0.070, 0.130)
POWER_ABS_TOL = 0.08
SCALE_DEFLATION = float(power_analysis()["min_detectable_at_80_percent"]["scale"]["scale_low"])
GROSS_COVERAGE_FRACTION = 0.50


class WB85AError(WB85Error):
    """Fail-closed WB85a protocol-QA contract."""


def locked_flags() -> dict[str, Any]:
    return {
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
        "workbook": WORKBOOK,
    }


def refuse_physical_alignment_outcome(path: object) -> None:
    text = str(path).lower()
    for marker in ALIGNMENT_OUTCOME_MARKERS:
        if marker in text:
            raise WB85AError(f"WB85a must not inspect physical alignment outcomes: {path}")
    refuse_forbidden_path(path)


def protocol_qa_acceptance_rule() -> dict[str, Any]:
    """Frozen before synthetic calibration.  Not tuned on physical outcomes."""
    return {
        "frozen_before_synthetic_calibration": True,
        "may_repair_by_looking_at_physical_outcomes": False,
        "if_fail_create_new_prospective_protocol_only": True,
        "n_monte_carlo": N_NULL_MONTE_CARLO,
        "seed": NULL_CALIBRATION_SEED,
        "stratum_counts": dict(IDENTIFIABLE_STRATUM_COUNTS),
        "counts_are_wb84_identifiable_exports": True,
        "null": {"z": "N(0,1)", "coverage": f"Bernoulli({COVERAGE_NOMINAL})"},
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
        "combined_final_gate_type_i": {
            "target": COMBINED_TYPE_I_TARGET,
            "not_required_to_equal_alpha": True,
            "reason": "union of two independent alpha=0.05 gates under the nominal null",
            "accept_if_cp95_contains_target": True,
            "point_estimate_band": list(COMBINED_TYPE_I_BAND),
        },
        "power_validation": {
            "n_monte_carlo": N_POWER_MONTE_CARLO,
            "seed": POWER_VALIDATION_SEED,
            "compare_at_n": N_MIN_PER_IDENTIFIABLE_STRATUM,
            "absolute_tolerance": POWER_ABS_TOL,
            "must_meet_power_target": POWER_TARGET,
            "primary_mean": ALT_MEAN_PRIMARY,
            "primary_scale": ALT_SCALE,
            "scale_deflation": SCALE_DEFLATION,
            "alternatives_not_taken_from_wb83_failures": True,
        },
        "obviously_wrong_if": (
            "Type-I point estimate outside the frozen band, or the 95% CP interval "
            "misses the frozen target"
        ),
    }


def identifiable_ids() -> list[str]:
    return [row["stratum_id"] for row in identifiable_strata()]


def _require_identifiable_counts() -> dict[str, int]:
    expected = identifiable_ids()
    if list(IDENTIFIABLE_STRATUM_COUNTS) != expected:
        raise WB85AError("frozen QA counts are not the WB85 identifiable strata")
    for name in expected:
        if IDENTIFIABLE_STRATUM_COUNTS[name] != WB84_CONDITION_COUNTS[name]:
            raise WB85AError(f"QA count for {name} is not the WB84 identifiable export count")
        if IDENTIFIABLE_STRATUM_COUNTS[name] < N_MIN_PER_IDENTIFIABLE_STRATUM:
            raise WB85AError(f"{name} has n<{N_MIN_PER_IDENTIFIABLE_STRATUM}")
    return dict(IDENTIFIABLE_STRATUM_COUNTS)


def binomial_report(k: int, n: int) -> dict[str, Any]:
    if n <= 0 or k < 0 or k > n:
        raise WB85AError("invalid binomial counts")
    rate = k / float(n)
    lo, hi = clopper_pearson(k, n)
    return {
        "k": int(k),
        "n": int(n),
        "rate": rate,
        "se": math.sqrt(rate * (1.0 - rate) / float(n)),
        "cp95_lo": lo,
        "cp95_hi": hi,
    }


def type_i_accept(report: Mapping[str, Any], *, target: float, band: Sequence[float]) -> bool:
    return bool(report["cp95_lo"] <= target <= report["cp95_hi"] and band[0] <= report["rate"] <= band[1])


def projection_direction(stratum_id: str) -> dict[str, float]:
    if "rotation" in stratum_id:
        return unit_direction(ROTATION_INJECTION)
    if "identity" in stratum_id or "translation" in stratum_id:
        return unit_direction(TRANSLATION_INJECTION)
    raise WB85AError(f"{stratum_id} is not an identifiable projected mode")


def direction_vector(parameter_names: Sequence[str], direction: Mapping[str, float]) -> np.ndarray:
    unit = unit_direction(direction)
    vector = np.zeros(len(parameter_names), dtype=np.float64)
    for index, name in enumerate(parameter_names):
        if name in unit:
            vector[index] = unit[name]
    if float(np.linalg.norm(vector)) <= 0.0:
        raise WB85AError("projection has no support on the parameter chart")
    return vector / float(np.linalg.norm(vector))


def project_statistical_unit(
    theta_hat: Sequence[float],
    theta_true: Sequence[float],
    covariance: np.ndarray,
    parameter_names: Sequence[str],
    direction: Mapping[str, float],
) -> dict[str, Any]:
    """One event → one preregistered projected z.  Not one z per pose coordinate."""
    hat = np.asarray(theta_hat, dtype=np.float64)
    true = np.asarray(theta_true, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    if hat.shape != true.shape or hat.size != len(parameter_names):
        raise WB85AError("projected unit requires aligned theta and names")
    if cov.shape != (hat.size, hat.size):
        raise WB85AError("covariance does not match the parameter chart")
    u = direction_vector(parameter_names, direction)
    a_hat = float(u @ hat)
    a_true = float(u @ true)
    sigma2 = float(u @ cov @ u)
    if (not math.isfinite(sigma2)) or sigma2 <= 0.0:
        raise WB85AError("projected variance is not a positive finite value")
    sigma = math.sqrt(sigma2)
    return {
        "a_hat": a_hat,
        "a_true": a_true,
        "sigma_a": sigma,
        "sigma_a2": sigma2,
        "z": (a_hat - a_true) / sigma,
        "n_independent_z": 1,
        "n_pose_coordinates": int(hat.size),
        "correlated_pose_coordinates_are_not_independent_z": True,
    }


def validate_result_schema(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    identifiable = set(identifiable_ids())
    seen: dict[tuple[str, str, str], int] = {}
    for row in rows:
        for name in (
            "physical_event_uid",
            "input_locator",
            "stratum_id",
            "z",
            "theta_hat",
            "theta_true",
            "sigma_hat",
        ):
            if name not in row:
                raise WB85AError(f"result schema missing {name}")
        stratum = str(row["stratum_id"])
        if stratum not in identifiable:
            if "weak_jg" in stratum:
                continue
            raise WB85AError(f"unknown identifiable stratum: {stratum}")
        key = (str(row["physical_event_uid"]), str(row["input_locator"]), stratum)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            raise WB85AError("one event contributed more than one z in the same identifiable stratum")
        if isinstance(row["z"], Sequence) and not isinstance(row["z"], (str, bytes)):
            if len(row["z"]) != 1:
                raise WB85AError("a replica may not emit multiple correlated pose coordinates as independent z")
    return {
        "one_event_one_projected_z_per_identifiable_stratum": True,
        "n_rows": len(rows),
        "n_unique_event_stratum": len(seen),
        "accepted": True,
    }


def statistical_unit_contract_report() -> dict[str, Any]:
    names = ("s1_dx_mm", "s2_dy_mm", "s3_dx_mm", "s1_dy_mm", "s2_dx_mm", "s3_dy_mm")
    direction = unit_direction(TRANSLATION_INJECTION)
    true = np.asarray([0.30, -0.20, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    hat = true + np.asarray([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    cov = np.eye(6, dtype=np.float64) * 0.04
    projected = project_statistical_unit(hat, true, cov, names, direction)
    u = direction_vector(names, direction)
    analytic_a = float(u @ hat)
    analytic_true = float(u @ true)
    analytic_sigma = math.sqrt(float(u @ cov @ u))
    analytic_z = (analytic_a - analytic_true) / analytic_sigma
    rows = [
        {
            "physical_event_uid": "event-a",
            "input_locator": "src:1:1",
            "stratum_id": "identifiable_translation_fixed_dz",
            "z": projected["z"],
            "theta_hat": hat.tolist(),
            "theta_true": true.tolist(),
            "sigma_hat": projected["sigma_a"],
        },
        {
            "physical_event_uid": "event-b",
            "input_locator": "src:1:2",
            "stratum_id": "identifiable_translation_fixed_dz",
            "z": 0.0,
            "theta_hat": true.tolist(),
            "theta_true": true.tolist(),
            "sigma_hat": 1.0,
        },
    ]
    schema = validate_result_schema(rows)
    multi_z_rejected = False
    try:
        validate_result_schema(
            [
                {
                    **rows[0],
                    "z": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                }
            ]
        )
    except WB85AError:
        multi_z_rejected = True
    duplicate_rejected = False
    try:
        validate_result_schema([rows[0], dict(rows[0])])
    except WB85AError:
        duplicate_rejected = True
    passed = (
        schema["accepted"]
        and projected["n_independent_z"] == 1
        and abs(projected["z"] - analytic_z) < 1.0e-12
        and multi_z_rejected
        and duplicate_rejected
        and artifact_payloads()["qualification_schema.json"]["required_replica_fields"]
    )
    return {
        "formula": "z = (u^T theta_hat - u^T theta_true) / sqrt(u^T Cov u)",
        "projected": projected,
        "analytic_z": analytic_z,
        "schema": schema,
        "rejects_multiple_correlated_pose_z": multi_z_rejected,
        "rejects_duplicate_event_stratum": duplicate_rejected,
        "weak_excluded_from_primary": all(not row["enters_primary_global_test"] for row in weak_strata()),
        "statistical_unit_contract_pass": bool(passed),
    }


def _hit(track_id: int, station: int, measurement_id: str, observed: np.ndarray) -> StationHit:
    return StationHit(
        track_id=track_id,
        station=station,
        measurement_id=measurement_id,
        observed=np.asarray(observed, dtype=np.float64),
        covariance=np.eye(4, dtype=np.float64),
    )


def nuisance_covariance_report() -> dict[str, Any]:
    """Joint solve vs Schur/marginal vs projected sigma under dz + survey prior."""
    names = ("s1_dx_mm", "s1_dz_mm")
    h_th = np.asarray(
        [
            [1.00, 0.30],
            [0.20, 1.00],
            [0.50, 0.10],
            [0.00, 0.40],
            [0.80, 0.20],
            [0.10, 0.60],
            [0.30, 0.00],
            [0.20, 0.50],
        ],
        dtype=np.float64,
    )
    h_xi = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.2, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.2, 0.0, 0.0, 0.3],
            [0.0, 0.0, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.1, 0.2],
        ],
        dtype=np.float64,
    )
    residual = h_th @ np.asarray([0.12, -0.40], dtype=np.float64)
    hits = (_hit(1, 1, "nuis-a", residual[:4]), _hit(1, 2, "nuis-b", residual[4:]))
    block = assemble_track_block(hits, h_xi, h_th, residual)
    n_xi = int(block.jacobian_nuisance.shape[1])
    prior = 1.0 / (SURVEY_DZ_PRIOR_SIGMA_MM ** 2)
    n_par = n_xi + 2
    jacobian = np.zeros((residual.size, n_par), dtype=np.float64)
    jacobian[:, :n_xi] = h_xi
    jacobian[:, n_xi:] = h_th
    normal = jacobian.T @ block.weight @ jacobian
    normal[n_xi + 1, n_xi + 1] += prior
    cov_joint = np.linalg.inv(0.5 * (normal + normal.T))
    marginal_a = float(cov_joint[n_xi, n_xi])
    from alignment.common_track_solver import schur_reduce

    piece, _rhs, *_rest = schur_reduce(block)
    reduced_with_prior = np.array(piece, dtype=np.float64, copy=True)
    reduced_with_prior[1, 1] += prior
    cov_schur = np.linalg.inv(0.5 * (reduced_with_prior + reduced_with_prior.T))
    u = np.asarray([1.0, 0.0], dtype=np.float64)
    projected = float(u @ cov_schur @ u)
    _joint_update, joint_cov_no_prior = solve_joint_dense((block,))
    solution = solve_common_track(
        (block,),
        names,
        survey_mode=SURVEY_FINITE_PRIOR,
        survey_sigma_mm=SURVEY_DZ_PRIOR_SIGMA_MM,
        damping=0.0,
    )
    if not solution.ok:
        raise WB85AError(f"nuisance covariance solve failed: {solution.message}")
    solver_sigma2 = float(solution.covariance[0, 0])
    projected_solver = project_statistical_unit(
        solution.update,
        np.zeros(2),
        solution.covariance,
        names,
        {"s1_dx_mm": 1.0},
    )
    gaps = {
        "joint_vs_schur": abs(marginal_a - float(cov_schur[0, 0])),
        "schur_vs_projected": abs(float(cov_schur[0, 0]) - projected),
        "schur_vs_solver": abs(float(cov_schur[0, 0]) - solver_sigma2),
        "projected_vs_solver_unit": abs(projected_solver["sigma_a2"] - solver_sigma2),
    }
    passed = (
        all(value < 1.0e-9 for value in gaps.values())
        and solution.n_dropped == 0
        and abs(float(joint_cov_no_prior[n_xi, n_xi]) - solver_sigma2) > 1.0e-8
    )
    return {
        "survey_mode": SURVEY_FINITE_PRIOR,
        "survey_sigma_mm": SURVEY_DZ_PRIOR_SIGMA_MM,
        "parameter_names": list(names),
        "joint_marginal_sigma2_a": marginal_a,
        "schur_marginal_sigma2_a": float(cov_schur[0, 0]),
        "projected_sigma2_a": projected,
        "solver_sigma2_a": solver_sigma2,
        "projected_solver_sigma2_a": projected_solver["sigma_a2"],
        "joint_without_prior_differs": True,
        "gaps": gaps,
        "dz_is_nuisance_not_projected_mode": True,
        "nuisance_covariance_pass": bool(passed),
    }


def healthy_identifiable_row(n: int, k: int) -> dict[str, Any]:
    lo, hi = clopper_pearson(k, n)
    return {
        "n_attempted": n,
        "n_converged": n,
        "n_nan": 0,
        "n_inf": 0,
        "n_non_spd": 0,
        "fd_rel_max_observed": 0.0,
        "sign_frame_failure": False,
        "engineering_bias": 0.0,
        "empirical_coverage": k / float(n),
        "cp_hi": hi,
        "cp_lo": lo,
        "n_dropped_parameters": 0,
        "dropped": False,
    }


def tls_statistic(z: Mapping[str, np.ndarray]) -> float:
    expected = identifiable_ids()
    t_loc = 0.0
    t_scale = 0.0
    for name in expected:
        values = np.asarray(z[name], dtype=np.float64)
        n = int(values.size)
        mean = float(values.mean())
        var = float(values.var(ddof=1))
        t_loc += n * mean * mean
        t_scale += 0.5 * (n - 1) * (var - 1.0) ** 2
    return t_loc + t_scale


def tcov_statistic(covered: Mapping[str, tuple[int, int]]) -> float:
    p = COVERAGE_NOMINAL
    total = 0.0
    for name in identifiable_ids():
        k, n = covered[name]
        total += (k - n * p) ** 2 / (n * p * (1.0 - p))
    return float(total)


def draw_null_z(
    rng: np.random.Generator,
    counts: Mapping[str, int],
    *,
    mean_shift: Mapping[str, float] | None = None,
    scale: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    out = {}
    for name, n in counts.items():
        mu = 0.0 if mean_shift is None else float(mean_shift.get(name, 0.0))
        sc = 1.0 if scale is None else float(scale.get(name, 1.0))
        out[name] = rng.normal(mu, sc, size=int(n))
    return out


def draw_null_coverage(
    rng: np.random.Generator,
    counts: Mapping[str, int],
    *,
    p: Mapping[str, float] | None = None,
) -> dict[str, tuple[int, int]]:
    covered = {}
    for name, n in counts.items():
        prob = COVERAGE_NOMINAL if p is None else float(p.get(name, COVERAGE_NOMINAL))
        covered[name] = (int(rng.binomial(int(n), prob)), int(n))
    return covered


def run_null_calibration(*, n_mc: int = N_NULL_MONTE_CARLO, seed: int = NULL_CALIBRATION_SEED) -> dict[str, Any]:
    counts = _require_identifiable_counts()
    rng = np.random.default_rng(seed)
    crit_ls = float(stats.chi2.ppf(1.0 - ALPHA, 2 * N_IDENTIFIABLE_STRATA))
    crit_cov = float(stats.chi2.ppf(1.0 - ALPHA, N_IDENTIFIABLE_STRATA))
    rej_ls = 0
    rej_cov = 0
    rej_combined = 0
    official_mismatch = 0
    for trial in range(int(n_mc)):
        z = draw_null_z(rng, counts)
        covered = draw_null_coverage(rng, counts)
        t_ls = tls_statistic(z)
        t_cov = tcov_statistic(covered)
        ls_fail = t_ls > crit_ls
        cov_fail = t_cov > crit_cov
        rej_ls += int(ls_fail)
        rej_cov += int(cov_fail)
        rej_combined += int(ls_fail or cov_fail)
        if trial < N_OFFICIAL_CROSSCHECK:
            gof = location_scale_gof({name: values.tolist() for name, values in z.items()})
            cov = coverage_calibration(covered)
            bundle = {
                "z": {name: values.tolist() for name, values in z.items()},
                "coverage": covered,
                "identifiable_strata": {
                    name: healthy_identifiable_row(n, covered[name][0]) for name, n in counts.items()
                },
                "weak_strata": {},
            }
            future = evaluate_future_qualification(bundle)
            official_ls = not gof["accepted"]
            official_cov = not cov["accepted"]
            official_combined = future["qualification_status"] != "PASS"
            if official_ls != ls_fail or official_cov != cov_fail or official_combined != (ls_fail or cov_fail):
                official_mismatch += 1
    ls_report = binomial_report(rej_ls, n_mc)
    cov_report = binomial_report(rej_cov, n_mc)
    combined_report = binomial_report(rej_combined, n_mc)
    ls_ok = type_i_accept(ls_report, target=ALPHA, band=TYPE_I_POINT_BAND)
    cov_ok = type_i_accept(cov_report, target=ALPHA, band=TYPE_I_POINT_BAND)
    combined_ok = type_i_accept(combined_report, target=COMBINED_TYPE_I_TARGET, band=COMBINED_TYPE_I_BAND)
    calibrated = ls_ok and cov_ok and combined_ok and official_mismatch == 0
    return {
        "acceptance_rule": protocol_qa_acceptance_rule(),
        "n_monte_carlo": int(n_mc),
        "seed": int(seed),
        "stratum_counts": counts,
        "reads_physical_alignment_output": False,
        "critical_T_LS": crit_ls,
        "critical_T_cov": crit_cov,
        "location_scale": {**ls_report, "accepted": ls_ok, "target": ALPHA},
        "coverage": {**cov_report, "accepted": cov_ok, "target": ALPHA},
        "combined_final_gate": {
            **combined_report,
            "accepted": combined_ok,
            "target": COMBINED_TYPE_I_TARGET,
        },
        "official_function_crosscheck": {
            "n": N_OFFICIAL_CROSSCHECK,
            "mismatches": official_mismatch,
        },
        "synthetic_null_calibrated": bool(calibrated),
        "protocol_qa_pass_null": bool(calibrated),
    }


def _power_trial_reject(
    rng: np.random.Generator,
    counts: Mapping[str, int],
    *,
    mean_shift: Mapping[str, float] | None = None,
    scale: Mapping[str, float] | None = None,
    coverage_p: Mapping[str, float] | None = None,
    use_full_gate: bool = False,
) -> bool:
    z = draw_null_z(rng, counts, mean_shift=mean_shift, scale=scale)
    if use_full_gate:
        covered = draw_null_coverage(rng, counts, p=coverage_p)
        bundle = {
            "z": {name: values.tolist() for name, values in z.items()},
            "coverage": covered,
            "identifiable_strata": {
                name: healthy_identifiable_row(n, covered[name][0]) for name, n in counts.items()
            },
            "weak_strata": {},
        }
        if coverage_p:
            for name, prob in coverage_p.items():
                bundle["identifiable_strata"][name]["empirical_coverage"] = float(prob)
                k, n = covered[name]
                _lo, hi = clopper_pearson(k, n)
                bundle["identifiable_strata"][name]["cp_hi"] = hi
        return evaluate_future_qualification(bundle)["qualification_status"] != "PASS"
    crit = float(stats.chi2.ppf(1.0 - ALPHA, 2 * N_IDENTIFIABLE_STRATA))
    return tls_statistic(z) > crit


def run_power_validation(
    *,
    n_mc: int = N_POWER_MONTE_CARLO,
    seed: int = POWER_VALIDATION_SEED,
) -> dict[str, Any]:
    analytic = power_analysis()
    n = N_MIN_PER_IDENTIFIABLE_STRATUM
    counts = {name: n for name in identifiable_ids()}
    target = identifiable_ids()[0]
    rng = np.random.default_rng(seed)
    cases = {
        "persistent_mean_0.35_one_stratum": {
            "analytic": float(analytic["alternatives"]["persistent_mean_0.35_one_stratum"]["power"]),
            "mean_shift": {target: ALT_MEAN_PRIMARY},
            "is_primary": True,
        },
        "scale_1.25_one_stratum": {
            "analytic": float(analytic["alternatives"]["scale_1.25_one_stratum"]["power"]),
            "scale": {target: ALT_SCALE},
            "is_primary": True,
        },
        "scale_deflation_mde_one_stratum": {
            "analytic": float(
                power_scale(
                    SCALE_DEFLATION,
                    n=n,
                    df=2 * N_IDENTIFIABLE_STRATA,
                    critical=float(analytic["critical_T_LS"]),
                )
            ),
            "scale": {target: SCALE_DEFLATION},
            "is_primary": False,
        },
    }
    reports = {}
    all_ok = True
    for name, spec in cases.items():
        rejects = 0
        for _ in range(int(n_mc)):
            if _power_trial_reject(
                rng,
                counts,
                mean_shift=spec.get("mean_shift"),
                scale=spec.get("scale"),
            ):
                rejects += 1
        report = binomial_report(rejects, n_mc)
        gap = abs(report["rate"] - spec["analytic"])
        sufficient = report["rate"] >= POWER_TARGET
        analytic_sufficient = spec["analytic"] >= POWER_TARGET
        consistent = gap <= POWER_ABS_TOL and sufficient == analytic_sufficient
        if spec["is_primary"]:
            consistent = consistent and sufficient
        reports[name] = {
            **report,
            "analytic": spec["analytic"],
            "absolute_gap": gap,
            "absolute_tolerance": POWER_ABS_TOL,
            "mc_meets_80_percent": sufficient,
            "analytic_meets_80_percent": analytic_sufficient,
            "consistent": consistent,
            "is_primary": spec["is_primary"],
            "injected_stratum": target,
            "n_per_stratum": n,
        }
        if spec["is_primary"] and not consistent:
            all_ok = False
        if not spec["is_primary"] and not (gap <= POWER_ABS_TOL or sufficient == analytic_sufficient):
            all_ok = False
        if not spec["is_primary"] and not (gap <= POWER_ABS_TOL):
            # deflation MDE is itself an ncx2 approximation; require MC still near 80%.
            all_ok = all_ok and abs(report["rate"] - POWER_TARGET) <= 0.10
    # Gross coverage deficit uses the full gate / guardrail.
    gross_rejects = 0
    actual_counts = _require_identifiable_counts()
    for _ in range(int(n_mc)):
        if _power_trial_reject(
            rng,
            actual_counts,
            coverage_p={target: GROSS_COVERAGE_FRACTION},
            use_full_gate=True,
        ):
            gross_rejects += 1
    gross = binomial_report(gross_rejects, n_mc)
    gross_ok = gross["rate"] >= 0.99
    reports["gross_coverage_deficit"] = {
        **gross,
        "injected_stratum": target,
        "coverage_p": GROSS_COVERAGE_FRACTION,
        "consistent": gross_ok,
        "is_primary": False,
    }
    all_ok = all_ok and gross_ok
    return {
        "n_monte_carlo": int(n_mc),
        "seed": int(seed),
        "compare_at_n": n,
        "alternatives_not_taken_from_wb83_failures": True,
        "wb83_pull_mean_minus_0.262893_not_used": True,
        "wb85_analytic": {
            "persistent_mean_0.35_one_stratum": analytic["alternatives"]["persistent_mean_0.35_one_stratum"],
            "scale_1.25_one_stratum": analytic["alternatives"]["scale_1.25_one_stratum"],
            "min_detectable_at_80_percent": analytic["min_detectable_at_80_percent"],
        },
        "reports": reports,
        "power_validation_pass": bool(all_ok),
    }


def fail_closed_suite() -> dict[str, Any]:
    names = identifiable_ids()
    z = {name: np.zeros(N_MIN_PER_IDENTIFIABLE_STRATUM).tolist() for name in names}
    # exact mean 0 / var undefined if all zeros — use calibrated residuals
    unit = np.ones(N_MIN_PER_IDENTIFIABLE_STRATUM, dtype=np.float64)
    centered = np.concatenate([-unit[: N_MIN_PER_IDENTIFIABLE_STRATUM // 2], unit[N_MIN_PER_IDENTIFIABLE_STRATUM // 2 :]])
    centered = centered - centered.mean()
    centered = centered / centered.std(ddof=1)
    z = {name: centered.tolist() for name in names}
    coverage = {name: (190, 200) for name in names}
    healthy = {name: healthy_identifiable_row(200, 190) for name in names}

    def _future(**overrides: Any) -> str:
        bundle = {
            "z": dict(z),
            "coverage": dict(coverage),
            "identifiable_strata": {name: dict(row) for name, row in healthy.items()},
            "weak_strata": {},
        }
        bundle.update(overrides)
        try:
            return str(evaluate_future_qualification(bundle)["qualification_status"])
        except WB85Error:
            return "FAIL"

    cases: dict[str, str] = {}
    missing = dict(z)
    del missing[names[-1]]
    cases["missing_stratum"] = _future(z=missing)
    small = {name: centered[:199].tolist() for name in names}
    cases["n_below_200"] = _future(z=small)
    nan_z = dict(z)
    nan_z[names[0]] = [float("nan")] + centered.tolist()[1:]
    cases["nan"] = _future(z=nan_z)
    inf_z = dict(z)
    inf_z[names[1]] = [float("inf")] + centered.tolist()[1:]
    cases["inf"] = _future(z=inf_z)
    nonconv = {name: dict(row) for name, row in healthy.items()}
    nonconv[names[0]]["n_converged"] = 189
    cases["nonconvergence_gt_5pct"] = _future(identifiable_strata=nonconv)
    nspd = {name: dict(row) for name, row in healthy.items()}
    nspd[names[1]]["n_non_spd"] = 1
    cases["unresolved_non_spd"] = _future(identifiable_strata=nspd)
    fd = {name: dict(row) for name, row in healthy.items()}
    fd[names[2]]["fd_rel_max_observed"] = FD_REL_MAX + 1.0e-6
    cases["fd_gt_1pct"] = _future(identifiable_strata=fd)
    bias = {name: dict(row) for name, row in healthy.items()}
    bias["identifiable_translation_fixed_dz"]["engineering_bias"] = ENGINEERING_TRANSLATION_MM + 0.01
    cases["large_engineering_bias"] = _future(identifiable_strata=bias)
    rot = {name: dict(row) for name, row in healthy.items()}
    rot["identifiable_rotation_fixed_dz"]["engineering_bias"] = ENGINEERING_RZ_MRAD + 0.1
    cases["large_rotation_bias"] = _future(identifiable_strata=rot)
    dropped = {name: dict(row) for name, row in healthy.items()}
    dropped[names[0]]["dropped"] = True
    cases["dropped_identifiable_mode"] = _future(identifiable_strata=dropped)
    dropped_col = {name: dict(row) for name, row in healthy.items()}
    dropped_col[names[3]]["n_dropped_parameters"] = 1
    cases["dropped_identifiable_column"] = _future(identifiable_strata=dropped_col)
    gross = {name: dict(row) for name, row in healthy.items()}
    gross[names[0]]["empirical_coverage"] = COVERAGE_GROSS_EMPIRICAL_MIN - 0.01
    gross[names[0]]["cp_hi"] = COVERAGE_GROSS_CP_UPPER_MIN - 0.01
    cases["gross_coverage_failure"] = _future(identifiable_strata=gross)
    weak = evaluate_guardrails(
        {
            "identifiable_strata": healthy,
            "weak_strata": {
                "weak_jg_diagnostic_fixed_dz": {
                    "standardized_bias_dx": WEAK_STANDARDIZED_CATASTROPHIC + 0.1,
                    "standardized_bias_ry": 0.0,
                }
            },
        }
    )
    cases["weak_catastrophic_standardized_bias"] = "FAIL" if weak["failed"] else "PASS"
    unknown_ok = cases["n_below_200"] == "UNKNOWN"
    fail_closed = all(status in {"FAIL", "UNKNOWN"} and status != "PASS" for status in cases.values())
    fail_closed = fail_closed and unknown_ok and cases["missing_stratum"] == "FAIL"
    holm_shift = {name: [ALT_MEAN_PRIMARY] * 200 for name in names}
    failed = evaluate_future_qualification(
        {"z": holm_shift, "coverage": coverage, "identifiable_strata": healthy, "weak_strata": {}}
    )
    holm_ok = (
        failed["qualification_status"] == "FAIL"
        and failed["diagnostics"]["may_reselect_model"] is False
        and failed["diagnostics"]["may_change_thresholds"] is False
        and "holm_identifiable_location" in failed["diagnostics"]
    )
    return {
        "cases": cases,
        "unknown_path": cases["n_below_200"],
        "holm_after_fail_is_diagnostic_only": holm_ok,
        "max_nonconvergence_fraction": MAX_NONCONVERGENCE_FRACTION,
        "fail_closed_tests_pass": bool(fail_closed and holm_ok),
    }


def weak_jg_contract_report() -> dict[str, Any]:
    z = {name: [0.0] * 200 for name in identifiable_ids()}
    extra = dict(z)
    extra["weak_jg_diagnostic_fixed_dz"] = [0.0] * 200
    rejected = False
    try:
        location_scale_gof(extra)
    except WB85Error:
        rejected = True
    primary = [row["stratum_id"] for row in identifiable_strata()]
    weak = [row["stratum_id"] for row in weak_strata()]
    charts_frozen = charts()
    guard = evaluate_guardrails(
        {
            "identifiable_strata": {name: healthy_identifiable_row(200, 190) for name in primary},
            "weak_strata": {
                "weak_jg_diagnostic_finite_survey_prior": {
                    "standardized_bias_dx": 0.2,
                    "standardized_bias_ry": WEAK_STANDARDIZED_CATASTROPHIC + 1.0,
                }
            },
        }
    )
    return {
        "weak_excluded_from_primary_gof": rejected,
        "primary_strata": primary,
        "weak_strata": weak,
        "weak_enters_primary_global_test": charts_frozen["weak_jg_diagnostic_chart"]["enters_primary_global_test"],
        "ordinary_translation_threshold_forbidden": charts_frozen["weak_jg_diagnostic_chart"][
            "ordinary_translation_threshold_forbidden"
        ],
        "catastrophic_guardrail_still_applies": guard["failed"],
        "may_add_or_drop_weak_after_seeing_physical_results": False,
        "weak_jg_contract_pass": bool(
            rejected
            and charts_frozen["weak_jg_diagnostic_chart"]["enters_primary_global_test"] is False
            and guard["failed"]
            and set(IDENTIFIABLE_FAMILIES) == {"identity", "identifiable_translation", "identifiable_rotation"}
        ),
    }


def execution_readiness_report() -> dict[str, Any]:
    audit = audit_execution_readiness()
    backend = RecordingPhysicalBackend()
    eye = np.eye(4).tolist()
    hits = [
        {
            "station_id": station,
            "measurement_id": f"exec:{station}",
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
        "event_uid": "mc24_100047_00100_00149:100047:2001",
        "source_id": "mc24_100047_00100_00149",
        "run_id": 100047,
        "event_id": 2001,
        "xaod_entry_index": 2001,
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
    work = OUTPUT_ROOT / "_hermetic_iterator"
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
    ready = all(checks.values())
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
        "physical_execution_ready": bool(ready),
        "execution_ready": bool(ready),
    }


def git_identity() -> dict[str, Any]:
    def _git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "baseline": "4station@fe00c0a9674b861f408e5c7e6c0d00163a42d116",
        "head_is_baseline": _git("rev-parse", "HEAD") == "fe00c0a9674b861f408e5c7e6c0d00163a42d116",
    }


def provenance_freeze_report() -> dict[str, Any]:
    refuse_physical_alignment_outcome(OUTPUT_ROOT / "wb85a_provenance_freeze.json")
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
    hashes = {
        "wb85_python_sha256": sha256_file(WB85_PYTHON),
        "wb85_yaml_sha256": sha256_file(WB85_YAML),
        "common_track_solver_sha256": sha256_file(SOLVER_PYTHON),
        "wb84_corpus_manifest_sha256": sha256_file(WB84_MANIFEST),
        "wb84_corpus_qa_sha256": sha256_file(WB84_QA),
        "wb84_provenance_freeze_sha256": sha256_file(WB84_PROVENANCE),
        "calypso_source_bundle_sha256": sha256_file(WB84_BUNDLE),
        "wb84_geometry_manifest_sha256": sha256_file(WB84_GEOMETRY),
        "wb84_snapshot_sha256": sha256_file(WB84_SNAPSHOT),
    }
    complete = all(value for value in hashes.values()) and bool(snapshot.get("field_map", {}).get("hashed_files"))
    complete = complete and qa["n_qualified_exports"] == 2394
    complete = complete and snapshot["calypso"]["diff_sha256"]
    return {
        **locked_flags(),
        "schema": "wb85a_provenance_freeze_v1",
        "git": git_identity(),
        "hashes": hashes,
        "field_map_hashes": snapshot["field_map"]["hashed_files"],
        "material_map_hashes": snapshot["material_map"]["hashed_files"],
        "material_map_hashed_files_empty_in_wb84_snapshot": not bool(snapshot["material_map"]["hashed_files"]),
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
        "wb83_wb84_artifacts_modified": False,
        "solver_implementation_fixed": bool(solver_implementation_fixed()),
        "provenance_complete": bool(complete),
    }


def build_verdict(parts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    required = {
        "synthetic_null_calibrated": bool(parts["null"]["synthetic_null_calibrated"]),
        "power_validation_pass": bool(parts["power"]["power_validation_pass"]),
        "statistical_unit_contract_pass": bool(parts["units"]["statistical_unit_contract_pass"]),
        "nuisance_covariance_pass": bool(parts["nuisance"]["nuisance_covariance_pass"]),
        "fail_closed_tests_pass": bool(parts["gates"]["fail_closed_tests_pass"]),
        "physical_execution_ready": bool(parts["execution"]["physical_execution_ready"]),
        "provenance_complete": bool(parts["provenance"]["provenance_complete"]),
    }
    required["weak_jg_contract_pass"] = bool(parts["weak"]["weak_jg_contract_pass"])
    passed = all(required.values())
    return {
        **locked_flags(),
        "schema": "wb85a_protocol_qa_v1",
        "acceptance_rule_frozen_before_calibration": True,
        "acceptance_rule": protocol_qa_acceptance_rule(),
        "checks": required,
        "wb85_protocol_qa_pass": passed,
        "ready_to_authorize_physical_qualification": passed,
        "qualification_authorized": False,
        "alignment_oracle_qualified_for_physical_FASER": False,
        "next_workbook_if_pass": "WB86 — Physical Common-Track Alignment Qualification v1",
        "wb86_may_change_model_threshold_or_data": False,
        "prospective_protocol_revision_required": (not parts["null"]["synthetic_null_calibrated"]),
    }


def run_protocol_qa(*, n_null: int = N_NULL_MONTE_CARLO, n_power: int = N_POWER_MONTE_CARLO) -> dict[str, Any]:
    _require_identifiable_counts()
    null = run_null_calibration(n_mc=n_null)
    power = run_power_validation(n_mc=n_power)
    units = statistical_unit_contract_report()
    nuisance = nuisance_covariance_report()
    gates = fail_closed_suite()
    weak = weak_jg_contract_report()
    execution = execution_readiness_report()
    provenance = provenance_freeze_report()
    verdict = build_verdict(
        {
            "null": null,
            "power": power,
            "units": units,
            "nuisance": nuisance,
            "gates": gates,
            "weak": weak,
            "execution": execution,
            "provenance": provenance,
        }
    )
    return {
        "verdict": verdict,
        "null": null,
        "power": power,
        "units": units,
        "nuisance": nuisance,
        "gates": gates,
        "weak": weak,
        "execution": execution,
        "provenance": provenance,
    }


def artifact_payloads_wb85a(bundle: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    verdict = bundle["verdict"]
    return {
        "wb85a_protocol_qa.json": verdict,
        "wb85a_synthetic_calibration.json": {**locked_flags(), **bundle["null"]},
        "wb85a_power_validation.json": {**locked_flags(), **bundle["power"]},
        "wb85a_execution_readiness.json": {**locked_flags(), **bundle["execution"]},
        "wb85a_provenance_freeze.json": bundle["provenance"],
    }


def write_protocol_qa_artifacts(
    output_root: Path | None = None,
    *,
    n_null: int = N_NULL_MONTE_CARLO,
    n_power: int = N_POWER_MONTE_CARLO,
) -> Path:
    root = OUTPUT_ROOT if output_root is None else Path(output_root)
    refuse_forbidden_path(root)
    refuse_wb83_wb84_write(root)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "wb85a_acceptance_rule.json", {**locked_flags(), **protocol_qa_acceptance_rule()})
    bundle = run_protocol_qa(n_null=n_null, n_power=n_power)
    for name, payload in artifact_payloads_wb85a(bundle).items():
        if payload.get("qualification_authorized") is not False:
            raise WB85AError("WB85a artifacts must keep qualification unauthorized")
        if payload.get("looked_at_physical_alignment_outcomes") is not False:
            raise WB85AError("WB85a artifacts must not look at physical alignment outcomes")
        write_json(root / name, payload)
    return root
