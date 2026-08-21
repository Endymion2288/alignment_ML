"""Pre-registered dual capture criteria for station-level 5-DoF + survey dz.

Absolute engineering tolerances and a normalized pull/coverage gate are
kept simultaneously.  The registered route-selected sigmas come from the
train-only 6-DoF pilot; validation never participates in the freeze.
``dz`` is survey-constrained and cannot make or break track-based capture.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "faser-5dof-survey-dz-capture-criteria-v1"
FREE_PARAMETERS: tuple[str, ...] = (
    "ift_dx_mm",
    "ift_dy_mm",
    "ift_rx_mrad",
    "ift_ry_mrad",
    "ift_rz_mrad",
)
SURVEY_PARAMETERS: tuple[str, ...] = ("ift_dz_mm",)
DEFAULT_STATISTICAL_K = 3.0
DEFAULT_COVERAGE_K = 3.0
DEFAULT_ENGINEERING: dict[str, float] = {
    "ift_dx_mm": 0.1,
    "ift_dy_mm": 0.1,
    "ift_rx_mrad": 1.0,
    "ift_ry_mrad": 1.0,
    "ift_rz_mrad": 1.0,
}


def load_capture_criteria(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"not a capture-criteria contract: {path}")
    schema = payload.get("schema_version")
    if schema == "faser-ift-layer-contrast-2d-capture-criteria-v1":
        from alignment.layer_contrast_capture import validate_layer_contrast_capture_criteria

        return validate_layer_contrast_capture_criteria(payload, path)
    if schema != SCHEMA_VERSION:
        raise ValueError(f"not a 5-DoF capture-criteria contract: {path}")
    if payload.get("validation_used_in_registration") is not False:
        raise ValueError("capture criteria must declare validation_used_in_registration=false")
    if payload.get("test_data_accessed") is not False:
        raise ValueError("capture criteria must not access test data")
    return dict(payload)


def evaluate_parameter_capture(
    *,
    name: str,
    error: float,
    fit_sigma: float | None,
    criteria: Mapping[str, Any],
) -> dict[str, Any]:
    """Score one parameter against the frozen dual gate.

    ``statistical_capture`` uses the pre-registered train-pilot sigma.
    ``coverage_capture`` uses this-fit sigma.  Survey parameters are scored
    for the record but do not enter overall track-based capture.
    """
    per = criteria["per_parameter"][name]
    role = str(per["role"])
    abs_tol = per.get("engineering_tolerance")
    registered = per.get("registered_sigma")
    statistical_k = float(per.get("statistical_k", criteria.get("statistical_k", DEFAULT_STATISTICAL_K)))
    coverage_k = float(per.get("coverage_k", criteria.get("coverage_k", DEFAULT_COVERAGE_K)))
    abs_error = abs(float(error))
    engineering = None if abs_tol is None else bool(abs_error <= float(abs_tol))
    statistical = None if registered is None else bool(abs_error <= statistical_k * float(registered))
    coverage_limit = None if fit_sigma is None else coverage_k * float(fit_sigma)
    coverage = None if coverage_limit is None else bool(abs_error <= coverage_limit)
    pull_registered = None if registered in (None, 0.0) else float(error) / float(registered)
    pull_fit = None if fit_sigma in (None, 0.0) else float(error) / float(fit_sigma)
    track_capture = None if role != "free" else bool(statistical is True and coverage is True)
    return {
        "name": name,
        "role": role,
        "error": float(error),
        "fit_sigma": None if fit_sigma is None else float(fit_sigma),
        "engineering_tolerance": None if abs_tol is None else float(abs_tol),
        "engineering_capture": engineering,
        "registered_sigma": None if registered is None else float(registered),
        "statistical_k": statistical_k,
        "statistical_capture": statistical,
        "pull_registered": pull_registered,
        "coverage_k": coverage_k,
        "coverage_capture": coverage,
        "pull_fit": pull_fit,
        "track_capture": track_capture,
    }


def overall_framework_capture(
    per_parameter: Sequence[Mapping[str, Any]],
    *,
    survey_parameters: Sequence[str] | None = None,
) -> dict[str, Any]:
    free = [row for row in per_parameter if row.get("role") == "free"]
    if not free:
        raise ValueError("capture criteria produced no free parameters")
    engineering = all(row.get("engineering_capture") is True for row in free)
    statistical = all(row.get("statistical_capture") is True for row in free)
    coverage = all(row.get("coverage_capture") is True for row in free)
    framework = all(row.get("track_capture") is True for row in free)
    excluded = list(SURVEY_PARAMETERS if survey_parameters is None else survey_parameters)
    return {
        "engineering_capture_success": engineering,
        "statistical_capture_success": statistical,
        "coverage_capture_success": coverage,
        "framework_capture_success": framework,
        "capture_success": framework,
        "survey_parameters_excluded_from_capture": excluded,
    }


def prior_contributions(
    names: Sequence[str],
    *,
    normal_matrix_native: np.ndarray,
    covariance_native: np.ndarray,
    prior_sigma_native: np.ndarray | None,
) -> list[dict[str, Any]]:
    """Separate track (data) information from the survey prior per parameter.

    ``prior_information_fraction`` uses the diagonal information split
    ``I_prior / (I_prior + N_ii)``.  A recovery whose posterior is prior-
    dominated must not be reported as a track-based measurement.
    """
    diagonal = np.diag(np.asarray(normal_matrix_native, dtype=np.float64))
    posterior = np.diag(np.asarray(covariance_native, dtype=np.float64))
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        data_info = float(diagonal[index])
        data_sigma = math.sqrt(1.0 / data_info) if math.isfinite(data_info) and data_info > 0.0 else None
        posterior_var = float(posterior[index])
        posterior_sigma = (
            math.sqrt(posterior_var) if math.isfinite(posterior_var) and posterior_var >= 0.0 else None
        )
        prior = None
        if prior_sigma_native is not None:
            raw = float(prior_sigma_native[index])
            if math.isfinite(raw) and raw > 0.0:
                prior = raw
        prior_info = 0.0 if prior is None else 1.0 / (prior * prior)
        total = data_info + prior_info
        fraction = None if total <= 0.0 else prior_info / total
        rows.append(
            {
                "name": name,
                "data_information": data_info,
                "data_sigma_marginal": data_sigma,
                "prior_sigma": prior,
                "prior_information": prior_info,
                "posterior_sigma": posterior_sigma,
                "prior_information_fraction": fraction,
                "track_dominated": None if fraction is None else bool(fraction < 0.5),
                "prior_dominated": None if fraction is None else bool(fraction >= 0.5),
            }
        )
    return rows


def attach_capture_and_prior(
    parameter_rows: Sequence[Mapping[str, Any]],
    *,
    names: Sequence[str],
    errors: Sequence[float],
    fit_sigmas: Sequence[float | None],
    criteria: Mapping[str, Any] | None,
    normal_matrix_native: np.ndarray,
    covariance_native: np.ndarray,
    prior_sigma_native: np.ndarray | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, list[dict[str, Any]]]:
    """Enrich parameter rows with dual capture and prior/posterior bookkeeping."""
    prior_rows = prior_contributions(
        names,
        normal_matrix_native=normal_matrix_native,
        covariance_native=covariance_native,
        prior_sigma_native=prior_sigma_native,
    )
    prior_by_name = {row["name"]: row for row in prior_rows}
    enriched: list[dict[str, Any]] = []
    capture_rows: list[dict[str, Any]] = []
    for row, name, error, sigma in zip(parameter_rows, names, errors, fit_sigmas):
        merged = dict(row)
        merged["prior_audit"] = prior_by_name[name]
        if criteria is not None:
            scored = evaluate_parameter_capture(
                name=str(name), error=float(error), fit_sigma=sigma, criteria=criteria
            )
            merged["capture"] = scored
            capture_rows.append(scored)
        enriched.append(merged)
    aggregate = None
    if criteria is not None:
        survey = (
            []
            if criteria.get("schema_version") == "faser-ift-layer-contrast-2d-capture-criteria-v1"
            else None
        )
        aggregate = overall_framework_capture(capture_rows, survey_parameters=survey)
    return enriched, aggregate, prior_rows
