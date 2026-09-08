"""WB85: freeze the physical common-track qualification protocol.

Design only.  Does not run alignment on the WB84 corpus, does not read
physical-corpus alignment outcomes, and does not authorize qualification.
The common-track solver objective is not modified.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats
from scipy.optimize import brentq

from alignment.calypso_physical_replica_production import (
    FAMILY_ORDER,
    FORBIDDEN_PATH_NEEDLES,
    PAYLOAD_FAMILIES,
    PROVENANCE as PHYSICAL_PROVENANCE,
    SURVEY_MODES,
    TARGET_REPLICAS_PER_CONDITION,
)
from alignment.common_track_solver import (
    DEFAULT_DAMPING,
    MAX_ITERATIONS,
    MOMENTUM_PRIOR,
    PARAMETER_SCALES,
    QP_PRIOR_SIGMA,
    SCALED_UPDATE_TOL,
    SURVEY_FINITE_PRIOR,
    SURVEY_FIXED_DZ,
    VALIDATION_REL_TOL,
    parameter_chart,
    solver_implementation_fixed,
)
from alignment.four_station import SURVEY_DZ_PRIOR_SIGMA_MM


WORKBOOK = "85"
OUTPUT_ROOT = Path("outputs/mc24_four_station_wb85_physical_alignment_protocol_v1")
ASSOCIATION_DEFAULT_SYSTEM = "frozen_W64_raw_energy_plus_exact_solver"
WB84_QA = Path("outputs/mc24_four_station_calypso_physical_replicas_v1/corpus_qa.json")
WB84_ROOT_MARKERS = (
    "wb83_truth_only",
    "wb83_final_closure",
    "calypso_physical_replicas_v1",
    "calypso_physical_replica_design_v1",
)
ALIGNMENT_OUTCOME_MARKERS = (
    "physical_alignment_outcome",
    "wb85_qualification_result",
    "common_track_physical_solve",
)

ALPHA = 0.05
COVERAGE_NOMINAL = 0.95
N_MIN_PER_IDENTIFIABLE_STRATUM = 200
MAX_NONCONVERGENCE_FRACTION = 0.05
FD_REL_MAX = 0.01
ENGINEERING_TRANSLATION_MM = 0.10
ENGINEERING_RZ_MRAD = 1.0
WEAK_STANDARDIZED_CATASTROPHIC = 5.0
COVERAGE_GROSS_EMPIRICAL_MIN = 0.70
COVERAGE_GROSS_CP_UPPER_MIN = 0.80
CONSECUTIVE_REQUIRED = 2
POWER_TARGET = 0.80
ALT_MEAN_PRIMARY = 0.35
ALT_MEAN_SMALL = 0.25
ALT_SCALE = 1.25
SYNTHETIC_CALIBRATION_SEED = 2026090721
N_IDENTIFIABLE_STRATA = 6
WB84_QUALIFIED_EXPORTS = 2394
WB84_CONDITION_COUNTS = {
    "identity_fixed_dz": 299,
    "identity_finite_survey_prior": 303,
    "identifiable_translation_fixed_dz": 311,
    "identifiable_translation_finite_survey_prior": 298,
    "identifiable_rotation_fixed_dz": 300,
    "identifiable_rotation_finite_survey_prior": 283,
    "weak_jg_diagnostic_fixed_dz": 302,
    "weak_jg_diagnostic_finite_survey_prior": 298,
}

IDENTIFIABLE_FAMILIES = ("identity", "identifiable_translation", "identifiable_rotation")
DIAGNOSTIC_FAMILIES = ("translation", "rotation", "weak-JG", "survey-prior")
TRANSLATION_INJECTION = {"s1_dx_mm": 0.30, "s2_dy_mm": -0.20}
ROTATION_INJECTION = {"s1_rz_mrad": -0.5, "s3_rz_mrad": 0.5}
WEAK_INJECTION = {"s3_dx_mm": 0.20, "s3_ry_mrad": 0.2}


class WB85Error(ValueError):
    """Fail-closed WB85 protocol contract."""


def refuse_forbidden_path(path: object) -> None:
    text = str(path).lower()
    for needle in FORBIDDEN_PATH_NEEDLES:
        if needle.lower() in text:
            raise WB85Error(f"WB85 refuses forbidden path: {path}")


def refuse_wb83_wb84_write(path: object) -> None:
    text = str(path).lower()
    for marker in WB84_ROOT_MARKERS:
        if marker in text:
            raise WB85Error(f"WB85 must not rewrite WB83/WB84 artifacts: {path}")


def refuse_alignment_outcome_lookup(path: object) -> None:
    text = str(path).lower()
    for marker in ALIGNMENT_OUTCOME_MARKERS:
        if marker in text:
            raise WB85Error(f"WB85 must not inspect physical alignment outcomes: {path}")
    raise WB85Error("WB85 must not inspect physical-corpus alignment outcomes")


def refuse_execution(reason: str = "qualification is not authorized") -> None:
    raise WB85Error(f"WB85 freezes the protocol only; {reason}")


def run_physical_qualification(*_args: object, **_kwargs: object) -> None:
    refuse_execution("common-track alignment batch is forbidden in this workbook")


def write_json(path: Path, payload: object) -> None:
    refuse_forbidden_path(path)
    refuse_wb83_wb84_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def locked_flags() -> dict[str, Any]:
    return {
        "executable": False,
        "qualification_authorized": False,
        "alignment_oracle_qualified_for_physical_FASER": False,
        "ml_alignment_eval_authorized": False,
        "final_blind_eval_authorized": False,
        "association_default_system": ASSOCIATION_DEFAULT_SYSTEM,
        "is_wb83_rerun": False,
        "modifies_wb84_corpus": False,
        "trained_on_wb83_failure_numbers": False,
        "new_physical_replicas_authorized": False,
        "looked_at_physical_alignment_outcomes": False,
    }


def unit_direction(components: Mapping[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(float(value) ** 2 for value in components.values()))
    if norm <= 0.0:
        raise WB85Error("projection direction must be non-zero")
    return {name: float(value) / norm for name, value in components.items()}


def chart_parameter_names(chart_kind: str, survey_mode: str) -> tuple[str, ...]:
    if chart_kind == "translation":
        components = ["dx_mm", "dy_mm"]
        stations = (1, 2, 3)
    elif chart_kind == "rotation":
        components = ["rz_mrad"]
        stations = (1, 3)
    elif chart_kind == "weak_jg":
        components = ["dx_mm", "ry_mrad"]
        stations = (3,)
    else:
        raise WB85Error(f"unknown chart kind: {chart_kind}")
    if survey_mode == SURVEY_FINITE_PRIOR:
        components = [*components, "dz_mm"]
    elif survey_mode != SURVEY_FIXED_DZ:
        raise WB85Error(f"unknown survey mode: {survey_mode}")
    return parameter_chart(
        survey_mode=survey_mode,
        reference_station=0,
        components=components,
        stations=stations,
    )


def frozen_geometry_families() -> dict[str, Any]:
    return {
        "n_physical_geometry_families": 4,
        "may_reselect_after_seeing_alignment": False,
        "families": {
            name: {
                "station_transforms": {
                    str(station): list(values) for station, values in sorted(PAYLOAD_FAMILIES[name].items())
                }
            }
            for name in FAMILY_ORDER
        },
        "identity": "all zero",
        "identifiable_translation": "S1 dx = +0.30 mm; S2 dy = -0.20 mm",
        "identifiable_rotation": "S1 rz = -0.5 mrad; S3 rz = +0.5 mrad",
        "weak_jg_diagnostic": "S3 dx = +0.20 mm; S3 ry = +0.2 mrad",
        "survey_tags_are_reconstruction_physics": False,
        "survey_tags": list(SURVEY_MODES),
        "survey_tags_are_downstream_analysis_modes": True,
        "paired_event_comparison_forbidden": True,
        "source": "WB84 frozen payloads; not the WB83 geometry matrix",
    }


def charts() -> dict[str, Any]:
    translation_direction = unit_direction(TRANSLATION_INJECTION)
    rotation_direction = unit_direction(ROTATION_INJECTION)
    common = {
        "reference_convention": "hold_station_0_at_identity",
        "left_se3_update": True,
        "local_nuisance": "xi=(x_mm,y_mm,tx,ty,q_over_p) Schur-eliminated per truth track",
        "q_over_p_treatment": MOMENTUM_PRIOR,
        "q_over_p_prior_sigma": QP_PRIOR_SIGMA,
        "solver": "common_track_solver WB83-v2 fixes; objective frozen",
        "symmetrized_normal": True,
        "strict_spd": True,
        "absolute_state_survey_prior": True,
        "iterative_relinearization": True,
        "max_iterations": MAX_ITERATIONS,
        "damping": DEFAULT_DAMPING,
        "scaled_update_tol": SCALED_UPDATE_TOL,
        "validation_rel_tol": VALIDATION_REL_TOL,
        "consecutive_required": CONSECUTIVE_REQUIRED,
        "automatically_pass_at_max_iterations": False,
        "robust_irls": False,
        "toy_uniform_By_forbidden_for_official_solve": True,
        "physical_prediction": (
            "Calypso refit + ACTS q_over_p_mode=0 from WB84 measurements; "
            "Newton/Schur reused; lab_transport iterate_common_track is not official"
        ),
        "not_15d_relative_wls": True,
        "pseudoinverse_does_not_imply_full_rank_qualified": True,
        "identifiable_dropped_parameter_is_fail": True,
    }
    return {
        "translation_chart": {
            **common,
            "chart_kind": "translation",
            "used_by_families": ["identity", "identifiable_translation"],
            "free_station_parameters_fixed_dz": list(chart_parameter_names("translation", SURVEY_FIXED_DZ)),
            "free_station_parameters_finite_survey_prior": list(
                chart_parameter_names("translation", SURVEY_FINITE_PRIOR)
            ),
            "held_parameters": ["s0_*", "all rx/ry/rz", "dz when survey tag is fixed_dz"],
            "stations_free": [1, 2, 3],
            "components_free": ["dx_mm", "dy_mm"],
            "projection": translation_direction,
            "identity_true_amplitude_mm": 0.0,
            "translation_true_amplitude_mm": math.sqrt(0.30**2 + 0.20**2),
            "mode_status": "identifiable",
            "enters_primary_global_test": True,
            "dz_treatment": {
                SURVEY_FIXED_DZ: "held exactly",
                SURVEY_FINITE_PRIOR: (
                    f"absolute-state MAP on free-station dz, sigma={SURVEY_DZ_PRIOR_SIGMA_MM} mm; "
                    "dz is a prior-constrained nuisance and is not the projected mode"
                ),
            },
            "survey_prior": {
                SURVEY_FIXED_DZ: "none",
                SURVEY_FINITE_PRIOR: f"absolute-state dz MAP, sigma={SURVEY_DZ_PRIOR_SIGMA_MM} mm",
            },
        },
        "rotation_chart": {
            **common,
            "chart_kind": "rotation",
            "used_by_families": ["identifiable_rotation"],
            "free_station_parameters_fixed_dz": list(chart_parameter_names("rotation", SURVEY_FIXED_DZ)),
            "free_station_parameters_finite_survey_prior": list(
                chart_parameter_names("rotation", SURVEY_FINITE_PRIOR)
            ),
            "held_parameters": ["s0_*", "s2_*", "all translations", "rx/ry", "dz when survey tag is fixed_dz"],
            "stations_free": [1, 3],
            "components_free": ["rz_mrad"],
            "projection": rotation_direction,
            "true_amplitude_mrad": math.sqrt(0.5**2 + 0.5**2),
            "mode_status": "identifiable",
            "enters_primary_global_test": True,
            "dz_treatment": {
                SURVEY_FIXED_DZ: "held exactly",
                SURVEY_FINITE_PRIOR: (
                    f"absolute-state MAP on S1/S3 dz, sigma={SURVEY_DZ_PRIOR_SIGMA_MM} mm; "
                    "dz is not the projected mode"
                ),
            },
            "survey_prior": {
                SURVEY_FIXED_DZ: "none",
                SURVEY_FINITE_PRIOR: f"absolute-state dz MAP, sigma={SURVEY_DZ_PRIOR_SIGMA_MM} mm",
            },
        },
        "weak_jg_diagnostic_chart": {
            **common,
            "chart_kind": "weak_jg",
            "used_by_families": ["weak_jg_diagnostic"],
            "free_station_parameters_fixed_dz": list(chart_parameter_names("weak_jg", SURVEY_FIXED_DZ)),
            "free_station_parameters_finite_survey_prior": list(
                chart_parameter_names("weak_jg", SURVEY_FINITE_PRIOR)
            ),
            "held_parameters": ["s0_*", "s1_*", "s2_*", "s3_dy/rx/rz", "dz when survey tag is fixed_dz"],
            "stations_free": [3],
            "components_free": ["dx_mm", "ry_mrad"],
            "mode_status": "weak",
            "enters_primary_global_test": False,
            "ordinary_translation_threshold_forbidden": True,
            "reports": [
                "standardized_weak_dx",
                "standardized_weak_ry",
                "absolute_s3_dx_mm",
                "absolute_s3_ry_mrad",
            ],
            "true_injection": dict(WEAK_INJECTION),
            "dz_treatment": {
                SURVEY_FIXED_DZ: "held exactly",
                SURVEY_FINITE_PRIOR: (
                    f"absolute-state MAP on S3 dz, sigma={SURVEY_DZ_PRIOR_SIGMA_MM} mm"
                ),
            },
            "survey_prior": {
                SURVEY_FIXED_DZ: "none",
                SURVEY_FINITE_PRIOR: f"absolute-state dz MAP, sigma={SURVEY_DZ_PRIOR_SIGMA_MM} mm",
            },
        },
    }


def identifiable_strata() -> list[dict[str, Any]]:
    rows = []
    for family in IDENTIFIABLE_FAMILIES:
        chart_name = "rotation_chart" if family == "identifiable_rotation" else "translation_chart"
        diagnostic = "rotation" if family == "identifiable_rotation" else "translation"
        for survey in SURVEY_MODES:
            rows.append(
                {
                    "stratum_id": f"{family}_{survey}",
                    "family": family,
                    "survey_mode": survey,
                    "chart": chart_name,
                    "mode_status": "identifiable",
                    "enters_primary_global_test": True,
                    "diagnostic_family": diagnostic,
                    "wb84_qualified_exports": WB84_CONDITION_COUNTS[f"{family}_{survey}"],
                }
            )
    return rows


def weak_strata() -> list[dict[str, Any]]:
    return [
        {
            "stratum_id": f"weak_jg_diagnostic_{survey}",
            "family": "weak_jg_diagnostic",
            "survey_mode": survey,
            "chart": "weak_jg_diagnostic_chart",
            "mode_status": "weak",
            "enters_primary_global_test": False,
            "diagnostic_family": "weak-JG",
            "wb84_qualified_exports": WB84_CONDITION_COUNTS[f"weak_jg_diagnostic_{survey}"],
        }
        for survey in SURVEY_MODES
    ]


def stack_rule() -> dict[str, Any]:
    return {
        "method": "M2_global_location_scale_gof",
        "method_chosen_because_it_would_pass_wb83": False,
        "method_reason": (
            "The scientific claim is joint calibration of standardized residuals, "
            "not an unadjusted intersection-union of many cellwise gates."
        ),
        "z_ir": "(theta_hat_ir - theta_true_i) / sigma_hat_ir along the preregistered projection",
        "sigma_hat": "sqrt(u^T Cov u) from the SPD solver covariance; u is the unit projection",
        "primary_null": {"E[z]": 0.0, "Var[z]": 1.0},
        "modes_in_global_test": [row["stratum_id"] for row in identifiable_strata()],
        "n_identifiable_strata": N_IDENTIFIABLE_STRATA,
        "weak_excluded_from_primary": True,
        "correlation_treatment": (
            "WB84 assigned independent event ensembles to each stratum.  "
            "Treat z_ir as independent across i and r.  Do not claim paired comparisons."
        ),
        "stacking": "per-stratum mean and unbiased variance, then sum location-scale scores",
        "not_unadjusted_cellwise_intersection": True,
        "not_wb83_every_cell_gate": True,
        "post_hoc_pooling_forbidden": True,
    }


def clopper_pearson(k: int, n: int, *, alpha: float = ALPHA) -> tuple[float, float]:
    if n <= 0 or k < 0 or k > n:
        raise WB85Error("invalid coverage counts")
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def location_scale_gof(strata_z: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Primary global location-scale GOF.  Frozen before any physical solve."""
    expected = [row["stratum_id"] for row in identifiable_strata()]
    missing = [name for name in expected if name not in strata_z]
    if missing:
        raise WB85Error(f"global GOF missing identifiable strata: {missing}")
    extra = [name for name in strata_z if name not in expected]
    if extra:
        raise WB85Error(f"global GOF received non-identifiable or unknown strata: {extra}")
    loc_terms = []
    scale_terms = []
    reports = {}
    n_total = 0
    for name in expected:
        values = np.asarray(list(strata_z[name]), dtype=np.float64)
        if values.size < N_MIN_PER_IDENTIFIABLE_STRATUM:
            return {
                "qualification_status": "UNKNOWN",
                "reason": f"{name} has n={values.size} < {N_MIN_PER_IDENTIFIABLE_STRATUM}",
                "accepted": False,
                "statistic": "T_LS = T_loc + T_scale",
            }
        if not np.isfinite(values).all():
            raise WB85Error(f"{name} contains NaN/Inf")
        n = int(values.size)
        n_total += n
        mean = float(values.mean())
        var = float(values.var(ddof=1))
        loc_terms.append(n * mean * mean)
        scale_terms.append(0.5 * (n - 1) * (var - 1.0) ** 2)
        reports[name] = {"n": n, "pull_mean": mean, "pull_width": math.sqrt(var)}
    n_strata = len(expected)
    t_loc = float(sum(loc_terms))
    t_scale = float(sum(scale_terms))
    t_ls = t_loc + t_scale
    df = 2 * n_strata
    critical = float(stats.chi2.ppf(1.0 - ALPHA, df))
    p_value = float(stats.chi2.sf(t_ls, df))
    accepted = t_ls <= critical
    return {
        "statistic": "T_LS = T_loc + T_scale",
        "T_loc": t_loc,
        "T_scale": t_scale,
        "T_LS": t_ls,
        "df": df,
        "alpha": ALPHA,
        "critical_value": critical,
        "p_value": p_value,
        "accepted": accepted,
        "n_identifiable_strata": n_strata,
        "n_total_z": n_total,
        "stratum_reports": reports,
        "qualification_status": "PASS" if accepted else "FAIL",
    }


def holm_reject(p_values: Sequence[float], *, alpha: float = ALPHA) -> list[bool]:
    values = [float(p) for p in p_values]
    if any((not math.isfinite(p)) or p < 0.0 or p > 1.0 for p in values):
        raise WB85Error("Holm p-values must lie in [0, 1]")
    order = sorted(range(len(values)), key=lambda index: values[index])
    rejected = [False] * len(values)
    for rank, index in enumerate(order):
        threshold = alpha / float(len(values) - rank)
        if values[index] <= threshold:
            rejected[index] = True
        else:
            break
    return rejected


def two_sided_normal_p(mean: float, n: int) -> float:
    z = abs(float(mean)) * math.sqrt(float(n))
    return float(2.0 * stats.norm.sf(z))


def two_sided_scale_p(variance: float, n: int) -> float:
    if n < 2:
        raise WB85Error("scale test needs n>=2")
    z = abs(float(variance) - 1.0) / math.sqrt(2.0 / float(n - 1))
    return float(2.0 * stats.norm.sf(z))


def two_sided_binom_p(k: int, n: int, *, p: float = COVERAGE_NOMINAL) -> float:
    dist = stats.binom(n, p)
    observed = float(dist.pmf(k))
    total = 0.0
    for count in range(n + 1):
        mass = float(dist.pmf(count))
        if mass <= observed + 1.0e-15:
            total += mass
    return float(min(1.0, total))


def family_restricted_gof(
    strata_z: Mapping[str, Sequence[float]],
    stratum_ids: Sequence[str],
) -> dict[str, Any]:
    subset = {name: strata_z[name] for name in stratum_ids}
    loc_terms = []
    scale_terms = []
    for name in stratum_ids:
        values = np.asarray(list(subset[name]), dtype=np.float64)
        n = int(values.size)
        mean = float(values.mean())
        var = float(values.var(ddof=1))
        loc_terms.append(n * mean * mean)
        scale_terms.append(0.5 * (n - 1) * (var - 1.0) ** 2)
    t_ls = float(sum(loc_terms) + sum(scale_terms))
    df = 2 * len(stratum_ids)
    critical = float(stats.chi2.ppf(1.0 - ALPHA, df))
    return {
        "T_LS": t_ls,
        "df": df,
        "critical_value": critical,
        "p_value": float(stats.chi2.sf(t_ls, df)),
        "accepted": t_ls <= critical,
        "strata": list(stratum_ids),
    }


def diagnostic_decomposition(strata_z: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Preregistered diagnostics after a global FAIL.  Not a reselection rule."""
    identifiable = {row["stratum_id"]: row for row in identifiable_strata()}
    translation_ids = [name for name, row in identifiable.items() if row["diagnostic_family"] == "translation"]
    rotation_ids = [name for name, row in identifiable.items() if row["diagnostic_family"] == "rotation"]
    fixed_ids = [name for name, row in identifiable.items() if row["survey_mode"] == SURVEY_FIXED_DZ]
    prior_ids = [name for name, row in identifiable.items() if row["survey_mode"] == SURVEY_FINITE_PRIOR]
    location_p = []
    location_names = []
    scale_p = []
    scale_names = []
    for name in [row["stratum_id"] for row in identifiable_strata()]:
        values = np.asarray(list(strata_z[name]), dtype=np.float64)
        location_names.append(name)
        location_p.append(two_sided_normal_p(float(values.mean()), int(values.size)))
        scale_names.append(name)
        scale_p.append(two_sided_scale_p(float(values.var(ddof=1)), int(values.size)))
    return {
        "may_reselect_model": False,
        "may_change_thresholds": False,
        "families": {
            "translation": family_restricted_gof(strata_z, translation_ids),
            "rotation": family_restricted_gof(strata_z, rotation_ids),
            "survey-prior_fixed_dz": family_restricted_gof(strata_z, fixed_ids),
            "survey-prior_finite": family_restricted_gof(strata_z, prior_ids),
            "weak-JG": {
                "enters_primary": False,
                "note": "report standardized weak-mode bias and absolute sensitivity only",
            },
        },
        "holm_identifiable_location": {
            "cells": location_names,
            "p_values": location_p,
            "rejected": holm_reject(location_p),
            "m": len(location_p),
            "alpha": ALPHA,
        },
        "holm_identifiable_scale": {
            "cells": scale_names,
            "p_values": scale_p,
            "rejected": holm_reject(scale_p),
            "m": len(scale_p),
            "alpha": ALPHA,
        },
    }


def coverage_calibration(covered: Mapping[str, tuple[int, int]]) -> dict[str, Any]:
    expected = [row["stratum_id"] for row in identifiable_strata()]
    terms = []
    reports = {}
    p_values = []
    for name in expected:
        if name not in covered:
            raise WB85Error(f"coverage missing {name}")
        k, n = covered[name]
        if n < N_MIN_PER_IDENTIFIABLE_STRATUM:
            return {
                "qualification_status": "UNKNOWN",
                "accepted": False,
                "reason": f"{name} n<{N_MIN_PER_IDENTIFIABLE_STRATUM}",
            }
        p = COVERAGE_NOMINAL
        empirical = k / float(n)
        terms.append((k - n * p) ** 2 / (n * p * (1.0 - p)))
        lo, hi = clopper_pearson(k, n)
        p_values.append(two_sided_binom_p(k, n, p=p))
        reports[name] = {
            "n": n,
            "n_covered": k,
            "empirical_coverage": empirical,
            "cp_lo": lo,
            "cp_hi": hi,
            "pull_mean_reported_elsewhere": True,
            "gross_failure": empirical < COVERAGE_GROSS_EMPIRICAL_MIN or hi < COVERAGE_GROSS_CP_UPPER_MIN,
        }
    t_cov = float(sum(terms))
    df = len(expected)
    critical = float(stats.chi2.ppf(1.0 - ALPHA, df))
    return {
        "statistic": "T_cov Pearson chi-square on identifiable strata",
        "T_cov": t_cov,
        "df": df,
        "critical_value": critical,
        "p_value": float(stats.chi2.sf(t_cov, df)),
        "accepted": t_cov <= critical,
        "alpha": ALPHA,
        "nominal": COVERAGE_NOMINAL,
        "not_every_cell_cp_must_contain_0.95": True,
        "reports": reports,
        "gross_failure": any(row["gross_failure"] for row in reports.values()),
        "holm_identifiable_coverage": {
            "cells": expected,
            "p_values": p_values,
            "rejected": holm_reject(p_values),
            "m": len(p_values),
            "alpha": ALPHA,
            "may_reselect_model": False,
            "may_change_thresholds": False,
        },
    }


def _chi2_critical(df: int) -> float:
    return float(stats.chi2.ppf(1.0 - ALPHA, df))


def power_mean(mu: float, *, n: int, df: int, critical: float) -> float:
    nc = float(n) * float(mu) ** 2
    return float(stats.ncx2.sf(critical, df, nc))


def power_scale(scale: float, *, n: int, df: int, critical: float) -> float:
    var = float(scale) ** 2
    nc = 0.5 * (n - 1) * (var - 1.0) ** 2
    return float(stats.ncx2.sf(critical, df, nc))


def min_detectable_mean(*, n: int, df: int, critical: float, target: float = POWER_TARGET) -> float:
    def objective(mu: float) -> float:
        return power_mean(mu, n=n, df=df, critical=critical) - target

    return float(brentq(objective, 1.0e-4, 2.0))


def min_detectable_scale(*, n: int, df: int, critical: float, target: float = POWER_TARGET) -> dict[str, float]:
    def objective(delta: float) -> float:
        nc = 0.5 * (n - 1) * float(delta) ** 2
        return float(stats.ncx2.sf(critical, df, nc)) - target

    delta = float(brentq(objective, 1.0e-4, 3.0))
    return {
        "abs_variance_shift": delta,
        "scale_high": math.sqrt(1.0 + delta),
        "scale_low": math.sqrt(max(1.0e-12, 1.0 - delta)),
    }


def power_analysis(*, n: int = N_MIN_PER_IDENTIFIABLE_STRATUM) -> dict[str, Any]:
    df = 2 * N_IDENTIFIABLE_STRATA
    critical = _chi2_critical(df)
    p_primary = power_mean(ALT_MEAN_PRIMARY, n=n, df=df, critical=critical)
    p_small = power_mean(ALT_MEAN_SMALL, n=n, df=df, critical=critical)
    p_scale = power_scale(ALT_SCALE, n=n, df=df, critical=critical)
    mde_mean = min_detectable_mean(n=n, df=df, critical=critical)
    mde_scale = min_detectable_scale(n=n, df=df, critical=critical)
    return {
        "n_per_identifiable_stratum": n,
        "source": "WB84 qualified exports; not redesigned from WB83 failed cells",
        "wb84_qualified_exports": WB84_QUALIFIED_EXPORTS,
        "wb84_condition_counts": dict(WB84_CONDITION_COUNTS),
        "wb84_min_per_stratum": min(WB84_CONDITION_COUNTS.values()),
        "power_uses_conservative_n_200": True,
        "surplus_replicas_do_not_change_thresholds": True,
        "post_hoc_pooling_forbidden": True,
        "alpha": ALPHA,
        "df": df,
        "critical_T_LS": critical,
        "alternatives": {
            "persistent_mean_0.35_one_stratum": {
                "power": p_primary,
                "sufficient": p_primary >= POWER_TARGET,
                "is_primary_location_alternative": True,
            },
            "persistent_mean_0.25_one_stratum": {
                "power": p_small,
                "sufficient": p_small >= POWER_TARGET,
                "is_primary_location_alternative": False,
                "qualification_status_if_this_were_the_claim": "UNKNOWN",
            },
            "scale_1.25_one_stratum": {
                "power": p_scale,
                "sufficient": p_scale >= POWER_TARGET,
                "is_primary_scale_alternative": True,
            },
        },
        "min_detectable_at_80_percent": {
            "persistent_standardized_mean": mde_mean,
            "scale": mde_scale,
            "n": n,
            "one_identifiable_stratum": True,
        },
        "unknown_if_n_below_200": True,
        "unknown_if_declared_primary_alternative_underpowered": True,
        "primary_declared_alternative": "persistent standardized mean 0.35 in one identifiable stratum",
        "small_alternative_is_not_a_PASS_claim": p_small < POWER_TARGET,
    }


def guardrails() -> dict[str, Any]:
    return {
        "global_pass_cannot_average_away_catastrophe": True,
        "any_triggers_fail": [
            "nonconvergence fraction > 0.05 in any identifiable stratum",
            "NaN / Inf",
            "unresolved non-SPD system",
            "geometry sign/frame failure",
            "FD relative error > 0.01 on an admitted column",
            "engineering translation |bias| > 0.1 mm on an identifiable translation/identity stratum",
            "engineering rz |bias| > 1 mrad on identifiable rotation",
            "unqualified dropped identifiable mode",
            "coverage grossly below nominal",
        ],
        "nonconvergence_fraction_max": MAX_NONCONVERGENCE_FRACTION,
        "fd_rel_max": FD_REL_MAX,
        "engineering_bias": {
            "translation_mm": ENGINEERING_TRANSLATION_MM,
            "rz_mrad": ENGINEERING_RZ_MRAD,
            "label": "research screening threshold; not a collaboration physics requirement",
        },
        "weak_jg": {
            "ordinary_translation_threshold_forbidden": True,
            "standardized_catastrophic": WEAK_STANDARDIZED_CATASTROPHIC,
            "also_report_absolute_sensitivity": True,
        },
        "coverage_grossly_below_nominal": {
            "empirical_lt": COVERAGE_GROSS_EMPIRICAL_MIN,
            "or_cp95_upper_lt": COVERAGE_GROSS_CP_UPPER_MIN,
            "defined_before_seeing_alignment": True,
        },
        "sign_frame_failure": (
            "identifiable injected amplitude recovered with opposite sign and |mean z|>3"
        ),
    }


def evaluate_guardrails(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Catastrophic FAIL rules.  Numeric thresholds are frozen a priori."""
    failures: list[str] = []
    identifiable_ids = [row["stratum_id"] for row in identifiable_strata()]
    rows = bundle.get("identifiable_strata", {})
    for name in identifiable_ids:
        if name not in rows or bool(rows.get(name, {}).get("dropped", False)):
            failures.append(f"unqualified dropped physical mode: {name}")
            continue
        row = rows[name]
        n_attempted = int(row["n_attempted"])
        n_converged = int(row["n_converged"])
        if n_attempted <= 0:
            failures.append(f"no attempted replicas: {name}")
            continue
        if (n_attempted - n_converged) / float(n_attempted) > MAX_NONCONVERGENCE_FRACTION:
            failures.append(f"nonconvergence above tolerance: {name}")
        if int(row.get("n_nan", 0)) > 0 or int(row.get("n_inf", 0)) > 0:
            failures.append(f"NaN/Inf: {name}")
        if int(row.get("n_non_spd", 0)) > 0:
            failures.append(f"non-SPD unresolved system: {name}")
        if float(row.get("fd_rel_max_observed", 0.0)) > FD_REL_MAX:
            failures.append(f"FD instability: {name}")
        if bool(row.get("sign_frame_failure", False)):
            failures.append(f"geometry sign/frame failure: {name}")
        if name.endswith("fixed_dz"):
            family = name[: -len("_fixed_dz")]
        elif name.endswith("finite_survey_prior"):
            family = name[: -len("_finite_survey_prior")]
        else:
            raise WB85Error(f"unrecognized identifiable stratum: {name}")
        bias = float(row.get("engineering_bias", 0.0))
        if family in ("identity", "identifiable_translation"):
            if abs(bias) > ENGINEERING_TRANSLATION_MM:
                failures.append(f"large engineering translation bias: {name}")
        if family == "identifiable_rotation":
            if abs(bias) > ENGINEERING_RZ_MRAD:
                failures.append(f"large engineering rz bias: {name}")
        empirical = float(row.get("empirical_coverage", 1.0))
        cp_hi = float(row.get("cp_hi", 1.0))
        if empirical < COVERAGE_GROSS_EMPIRICAL_MIN or cp_hi < COVERAGE_GROSS_CP_UPPER_MIN:
            failures.append(f"coverage grossly below nominal: {name}")
        if int(row.get("n_dropped_parameters", 0)) > 0:
            failures.append(f"identifiable chart used a dropped/pseudoinverse column: {name}")
    for name, row in bundle.get("weak_strata", {}).items():
        if abs(float(row.get("standardized_bias_dx", 0.0))) > WEAK_STANDARDIZED_CATASTROPHIC:
            failures.append(f"catastrophic weak-mode dx: {name}")
        if abs(float(row.get("standardized_bias_ry", 0.0))) > WEAK_STANDARDIZED_CATASTROPHIC:
            failures.append(f"catastrophic weak-mode ry: {name}")
    return {"failed": bool(failures), "failures": failures}


def evaluate_future_qualification(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen gate to a synthetic or future result bundle.

    This does not run the solver and is not an authorization to do so.
    """
    guard = evaluate_guardrails(bundle)
    if guard["failed"]:
        return {
            "qualification_status": "FAIL",
            "reason": "catastrophic guardrail",
            "guardrails": guard,
            "diagnostics_authorized": True,
        }
    try:
        gof = location_scale_gof(bundle["z"])
    except WB85Error as error:
        return {"qualification_status": "FAIL", "reason": str(error), "guardrails": guard}
    if gof.get("qualification_status") == "UNKNOWN":
        return {"qualification_status": "UNKNOWN", "gof": gof, "guardrails": guard}
    coverage = coverage_calibration(bundle["coverage"])
    if coverage.get("qualification_status") == "UNKNOWN":
        return {"qualification_status": "UNKNOWN", "gof": gof, "coverage": coverage, "guardrails": guard}
    if coverage.get("gross_failure"):
        return {
            "qualification_status": "FAIL",
            "reason": "coverage grossly below nominal",
            "gof": gof,
            "coverage": coverage,
            "guardrails": guard,
        }
    if not gof["accepted"] or not coverage["accepted"]:
        diagnostics = diagnostic_decomposition(bundle["z"])
        diagnostics["holm_identifiable_coverage"] = coverage["holm_identifiable_coverage"]
        return {
            "qualification_status": "FAIL",
            "gof": gof,
            "coverage": coverage,
            "diagnostics": diagnostics,
            "guardrails": guard,
        }
    return {
        "qualification_status": "PASS",
        "gof": gof,
        "coverage": coverage,
        "guardrails": guard,
        "cannot_claim_0.25_mean_alternative": True,
    }


def protocol_checklist() -> dict[str, bool]:
    power = power_analysis()
    return {
        "all_geometry_families_frozen": True,
        "all_charts_frozen": True,
        "global_null_frozen": True,
        "global_statistic_frozen": True,
        "multiplicity_rule_frozen": True,
        "sample_size_power_frozen": True,
        "convergence_frozen": True,
        "bias_guardrails_frozen": True,
        "coverage_rule_frozen": True,
        "unknown_rule_frozen": True,
        "no_result_dependent_choices": True,
        "solver_implementation_fixed": bool(solver_implementation_fixed()),
        "primary_location_alternative_powered": bool(
            power["alternatives"]["persistent_mean_0.35_one_stratum"]["sufficient"]
        ),
        "primary_scale_alternative_powered": bool(
            power["alternatives"]["scale_1.25_one_stratum"]["sufficient"]
        ),
    }


def protocol_freeze_complete() -> bool:
    return all(protocol_checklist().values())


def independence_contract() -> dict[str, Any]:
    return {
        "cluster_identity": ["physical_event_uid", "input_locator"],
        "one_event_one_statistical_unit": True,
        "repeated_solver_analysis_is_not_a_new_replica": True,
        "cross_condition_reuse_forbidden": True,
        "wb84_survey_tags_are_independent_ensembles_not_pairs": True,
        "provenance": PHYSICAL_PROVENANCE,
    }


def execution_flow() -> list[str]:
    return [
        "WB84 physical measurements",
        "truth association labels only",
        "common-track solve on the frozen chart",
        "apply left-SE(3) alignment update",
        "physical Calypso refit / ACTS repropagation",
        "relinearize",
        "independent validation tracks when an event has >=2 truth tracks",
    ]


def corpus_contract() -> dict[str, Any]:
    return {
        "provenance": PHYSICAL_PROVENANCE,
        "n_qualified_exports": WB84_QUALIFIED_EXPORTS,
        "n_physical_geometry_families": 4,
        "allocation_strata_all_at_least_200": True,
        "condition_counts": dict(WB84_CONDITION_COUNTS),
        "target_replicas_per_condition": TARGET_REPLICAS_PER_CONDITION,
        "counts_taken_from_wb84_qa_not_redesigned": True,
        "qa_path": str(WB84_QA),
        "may_read_alignment_outcomes_now": False,
    }


def convergence_contract() -> dict[str, Any]:
    return {
        "max_iterations": MAX_ITERATIONS,
        "damping": DEFAULT_DAMPING,
        "scaled_update_tol": SCALED_UPDATE_TOL,
        "validation_rel_tol": VALIDATION_REL_TOL,
        "consecutive_required": CONSECUTIVE_REQUIRED,
        "automatically_pass_at_max_iterations": False,
        "robust_irls": False,
        "nonconvergence_fraction_max": MAX_NONCONVERGENCE_FRACTION,
        "parameter_scales": dict(PARAMETER_SCALES),
    }


def build_global_statistical_gate() -> dict[str, Any]:
    power = power_analysis()
    return {
        **locked_flags(),
        "schema": "wb85_global_statistical_gate_v1",
        "workbook": WORKBOOK,
        "stack_rule": stack_rule(),
        "acceptance_region": {
            "T_LS_le": float(stats.chi2.ppf(1.0 - ALPHA, 2 * N_IDENTIFIABLE_STRATA)),
            "df": 2 * N_IDENTIFIABLE_STRATA,
            "alpha": ALPHA,
        },
        "coverage_gate": {
            "statistic": "T_cov",
            "df": N_IDENTIFIABLE_STRATA,
            "alpha": ALPHA,
            "critical_value": float(stats.chi2.ppf(1.0 - ALPHA, N_IDENTIFIABLE_STRATA)),
            "not_every_cell_cp_must_contain_0.95": True,
        },
        "guardrails": guardrails(),
        "unknown_rule": {
            "n_below_200_solvable": "UNKNOWN",
            "underpowered_declared_claim": "UNKNOWN",
            "primary_alternatives": power["alternatives"],
        },
        "synthetic_calibration_seed": SYNTHETIC_CALIBRATION_SEED,
        "wb83_cellwise_unadjusted_gate_reused": False,
    }


def build_mode_chart_contract() -> dict[str, Any]:
    return {
        **locked_flags(),
        "schema": "wb85_mode_chart_contract_v1",
        "workbook": WORKBOOK,
        "geometry_families": frozen_geometry_families(),
        "charts": charts(),
        "identifiable_strata": identifiable_strata(),
        "weak_strata": weak_strata(),
        "convergence": convergence_contract(),
        "solver_implementation_fixed": bool(solver_implementation_fixed()),
    }


def build_power_analysis() -> dict[str, Any]:
    return {
        **locked_flags(),
        "schema": "wb85_power_analysis_v1",
        "workbook": WORKBOOK,
        **power_analysis(),
        "independence": independence_contract(),
        "corpus": corpus_contract(),
    }


def build_multiplicity_contract() -> dict[str, Any]:
    return {
        **locked_flags(),
        "schema": "wb85_multiplicity_contract_v1",
        "workbook": WORKBOOK,
        "diagnostic_families": list(DIAGNOSTIC_FAMILIES),
        "primary_is_not_holm_cellwise": True,
        "after_global_fail": {
            "output_preregistered_family_decomposition": True,
            "holm_identifiable_location_m": N_IDENTIFIABLE_STRATA,
            "holm_identifiable_scale_m": N_IDENTIFIABLE_STRATA,
            "holm_identifiable_coverage_m": N_IDENTIFIABLE_STRATA,
            "holm_weak_report_only": True,
            "correction": "Holm",
            "alpha": ALPHA,
        },
        "diagnostics_are_not_model_reselection": True,
        "diagnostics_are_not_threshold_reselection": True,
    }


def build_qualification_schema() -> dict[str, Any]:
    return {
        **locked_flags(),
        "schema": "wb85_qualification_schema_v1",
        "workbook": WORKBOOK,
        "required_replica_fields": [
            "physical_event_uid",
            "input_locator",
            "stratum_id",
            "family",
            "survey_mode",
            "theta_hat",
            "theta_true",
            "sigma_hat",
            "z",
            "covered_95",
            "converged",
            "iterations",
            "spd_ok",
            "finite",
            "fd_rel_max",
            "n_dropped_parameters",
        ],
        "decision_order": [
            "catastrophic guardrails",
            "UNKNOWN if solvable n<200 in any identifiable stratum",
            "T_LS location-scale GOF",
            "T_cov global coverage calibration",
            "Holm diagnostics only after FAIL",
        ],
        "statuses": ["PASS", "FAIL", "UNKNOWN"],
        "execution_flow": execution_flow(),
        "official_engine": (
            "physical Calypso/ACTS linearization into the frozen Newton/Schur solver"
        ),
        "toy_iterate_common_track_is_official": False,
    }


def build_protocol() -> dict[str, Any]:
    checklist = protocol_checklist()
    return {
        **locked_flags(),
        "schema": "wb85_physical_alignment_protocol_v1",
        "workbook": WORKBOOK,
        "protocol_audit": "PASS" if all(checklist.values()) else "FAIL",
        "protocol_checklist": checklist,
        "corpus": corpus_contract(),
        "geometry_families": frozen_geometry_families(),
        "charts": {name: {k: v for k, v in chart.items() if k != "physical_prediction"} for name, chart in charts().items()},
        "stack_rule": stack_rule(),
        "guardrails": guardrails(),
        "independence": independence_contract(),
        "convergence": convergence_contract(),
        "execution_flow": execution_flow(),
        "htcondor_qualification_authorized": False,
        "answer": (
            "YES — WB84 already has a prospective, multiplicity-aware, fail-closed "
            "physical alignment qualification protocol.  Formal HTCondor qualification "
            "is not authorized by this workbook."
            if all(checklist.values())
            else "NO — protocol freeze is incomplete."
        ),
    }


def artifact_payloads() -> dict[str, dict[str, Any]]:
    return {
        "wb85_physical_alignment_protocol.json": build_protocol(),
        "global_statistical_gate.json": build_global_statistical_gate(),
        "mode_chart_contract.json": build_mode_chart_contract(),
        "power_analysis.json": build_power_analysis(),
        "multiplicity_contract.json": build_multiplicity_contract(),
        "qualification_schema.json": build_qualification_schema(),
    }


def write_protocol_artifacts(output_root: Path | None = None) -> Path:
    root = OUTPUT_ROOT if output_root is None else Path(output_root)
    refuse_forbidden_path(root)
    refuse_wb83_wb84_write(root)
    root.mkdir(parents=True, exist_ok=True)
    for name, payload in artifact_payloads().items():
        if payload.get("executable") is not False or payload.get("qualification_authorized") is not False:
            raise WB85Error("protocol artifacts must remain non-executable and unauthorized")
        write_json(root / name, payload)
    return root
