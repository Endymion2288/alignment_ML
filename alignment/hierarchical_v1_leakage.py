"""Station–C_dx first-order leakage and C_dx-profiled station estimators.

This module is analysis-only.  It does not write an Athena payload and does
not add a physical degree of freedom.  ``C_dx`` appears only as a Jacobian
column ``j_c`` used to form the omitted-variable operator

    A = (J_s^T W J_s)^{-1} J_s^T W j_c

and, separately, to Schur-complement / weighted-project that column out of
the station residual space.

``J_s`` is the station block.  The five track-constrained DoF are the
canonical leakage coordinates; a sixth survey-``dz`` column with the frozen
5 mm prior is the production-comparable operator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from alignment.five_dof_sampling import FREE_PARAMETERS as STATION_FIVE
from alignment.hierarchical_v1 import C_DX, STATION_SOLVE_PARAMETERS, STATION_SURVEY_PARAMETER
from alignment.physical_jacobian import _pinv, _svd_rank


MICRON_MM = 1.0e-3
SURVEY_DZ_PRIOR_MM = 5.0
DEFAULT_RCOND = 1.0e-10
STATION_SIX = STATION_SOLVE_PARAMETERS


def _as_float_map(values: Mapping[str, object]) -> dict[str, float]:
    return {str(name): float(values[name]) for name in values}


def _index_map(names: Sequence[str]) -> dict[str, int]:
    mapping = {str(name): index for index, name in enumerate(names)}
    if len(mapping) != len(names):
        raise ValueError("parameter names must be unique")
    return mapping


def _block(normal: np.ndarray, row_index: Sequence[int], column_index: Sequence[int]) -> np.ndarray:
    return np.asarray(normal, dtype=np.float64)[np.ix_(list(row_index), list(column_index))]


def _apply_prior(
    normal: np.ndarray,
    names: Sequence[str],
    prior_sigma_native: Mapping[str, float] | None,
) -> np.ndarray:
    updated = np.array(normal, dtype=np.float64, copy=True)
    if not prior_sigma_native:
        return updated
    index = _index_map(names)
    for name, sigma in prior_sigma_native.items():
        if name not in index:
            raise ValueError(f"prior for unknown parameter {name!r}")
        value = float(sigma)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"prior sigma for {name!r} must be positive and finite")
        updated[index[name], index[name]] += 1.0 / (value * value)
    return 0.5 * (updated + updated.T)


def _scaled_normal(normal: np.ndarray, scales: Sequence[float]) -> np.ndarray:
    scale_matrix = np.diag(np.asarray(scales, dtype=np.float64))
    scaled = scale_matrix @ np.asarray(normal, dtype=np.float64) @ scale_matrix
    return 0.5 * (scaled + scaled.T)


def _condition_and_rank(normal_scaled: np.ndarray, *, rcond: float) -> tuple[int, float | None, float | None, np.ndarray]:
    singular, rank, _tolerance, full_condition, retained = _svd_rank(normal_scaled, rcond=rcond)
    identifiable = full_condition if full_condition is not None else retained
    raw = None
    if singular.size and float(singular[-1]) > 0.0:
        raw = float(singular[0] / singular[-1])
    return int(rank), (None if identifiable is None else float(identifiable)), raw, np.asarray(singular, dtype=np.float64)


@dataclass(frozen=True)
class LeakageOperatorResult:
    """First-order map from true ``C_dx`` into a station-only WLS correction."""

    station_names: tuple[str, ...]
    contrast_name: str
    A_native_per_mm: dict[str, float]
    A_per_um: dict[str, float]
    disguised_same_unit_per_um: dict[str, float]
    column_cosine: dict[str, float]
    subspace_r2: float
    subspace_cosine: float
    rank: int
    full_rank: bool
    condition_scaled: float | None
    raw_condition_scaled: float | None
    singular_values_scaled: tuple[float, ...]
    used_prior_sigma_native: dict[str, float]
    normal_ss_native: np.ndarray
    normal_sc_native: np.ndarray
    normal_cc_native: float

    def as_json(self) -> dict[str, Any]:
        return {
            "station_names": list(self.station_names),
            "contrast_name": self.contrast_name,
            "A_native_per_mm_C_dx": dict(self.A_native_per_mm),
            "A_native_per_um_C_dx": dict(self.A_per_um),
            "disguised_um_or_urad_per_um_C_dx": dict(self.disguised_same_unit_per_um),
            "column_cosine": dict(self.column_cosine),
            "subspace_r2": self.subspace_r2,
            "subspace_cosine": self.subspace_cosine,
            "rank": self.rank,
            "full_rank": self.full_rank,
            "condition_scaled": self.condition_scaled,
            "raw_condition_scaled": self.raw_condition_scaled,
            "singular_values_scaled": list(self.singular_values_scaled),
            "used_prior_sigma_native": dict(self.used_prior_sigma_native),
            "normal_cc_native": self.normal_cc_native,
            "B_native_C_dx_per_station_unit": self.reverse_A_native(),
        }

    def reverse_A_native(self) -> dict[str, float]:
        """Omitted-station map ``B = N_cc^{-1} N_sc`` (C_dx bias per leftover station unit)."""
        if not math.isfinite(self.normal_cc_native) or self.normal_cc_native == 0.0:
            raise ValueError("contrast normal_cc is not usable")
        return {
            name: float(self.normal_sc_native[index] / self.normal_cc_native)
            for index, name in enumerate(self.station_names)
        }


def leakage_operator_from_normal(
    normal_native: np.ndarray,
    names: Sequence[str],
    *,
    station_names: Sequence[str],
    contrast_name: str = C_DX,
    prior_sigma_native: Mapping[str, float] | None = None,
    parameter_scales: Sequence[float] | None = None,
    rcond: float = DEFAULT_RCOND,
) -> LeakageOperatorResult:
    """Build ``A = (J_s^T W J_s)^{-1} J_s^T W j_c`` from a joint normal matrix.

    ``normal_native`` is ``J^T W J`` in native units, including the contrast
    column.  Optional ``prior_sigma_native`` is added only on the station
    block (production: 5 mm on ``ift_dz_mm``).
    """
    names = tuple(str(name) for name in names)
    station_names = tuple(str(name) for name in station_names)
    if contrast_name not in names:
        raise ValueError(f"{contrast_name!r} is not in the joint normal matrix")
    missing = [name for name in station_names if name not in names]
    if missing:
        raise ValueError("station names missing from joint normal: " + ", ".join(missing))
    index = _index_map(names)
    station_index = [index[name] for name in station_names]
    contrast_index = [index[str(contrast_name)]]
    normal = np.asarray(normal_native, dtype=np.float64)
    if normal.shape != (len(names), len(names)):
        raise ValueError("normal_native shape does not match names")
    n_ss = _apply_prior(
        _block(normal, station_index, station_index),
        station_names,
        prior_sigma_native,
    )
    n_sc = _block(normal, station_index, contrast_index).reshape(len(station_names))
    n_cc = float(_block(normal, contrast_index, contrast_index)[0, 0])
    if parameter_scales is None:
        scales = np.ones(len(station_names), dtype=np.float64)
    else:
        scale_map = {str(name): float(scale) for name, scale in zip(names, parameter_scales)}
        scales = np.asarray([scale_map[name] for name in station_names], dtype=np.float64)
    scaled = _scaled_normal(n_ss, scales)
    rank, condition, raw_condition, singular = _condition_and_rank(scaled, rcond=rcond)
    _, _, tolerance, _, _ = _svd_rank(scaled, rcond=rcond)
    inverse_scaled = _pinv(scaled, tolerance=tolerance)
    inverse_native = np.diag(scales) @ inverse_scaled @ np.diag(scales)
    a_native = inverse_native @ n_sc
    diagonal = np.diag(n_ss)
    cosines: dict[str, float] = {}
    for i, name in enumerate(station_names):
        denom = math.sqrt(max(float(diagonal[i]), 0.0) * max(n_cc, 0.0))
        cosines[name] = float("nan") if denom <= 0.0 else float(n_sc[i] / denom)
    quadratic = float(n_sc @ inverse_native @ n_sc)
    if n_cc > 0.0 and math.isfinite(quadratic):
        r2 = min(max(quadratic / n_cc, 0.0), 1.0)
    else:
        r2 = float("nan")
    cosine = math.sqrt(r2) if math.isfinite(r2) and r2 >= 0.0 else float("nan")
    a_map = {name: float(value) for name, value in zip(station_names, a_native)}
    per_um = {name: value * MICRON_MM for name, value in a_map.items()}
    # Translations: A (mm/mm) equals µm of station per µm of C_dx.
    # Rotations: A (mrad/mm) equals µrad of station per µm of C_dx.
    same_unit = dict(a_map)
    return LeakageOperatorResult(
        station_names=station_names,
        contrast_name=str(contrast_name),
        A_native_per_mm=a_map,
        A_per_um=per_um,
        disguised_same_unit_per_um=same_unit,
        column_cosine=cosines,
        subspace_r2=float(r2),
        subspace_cosine=float(cosine),
        rank=rank,
        full_rank=rank == len(station_names),
        condition_scaled=condition,
        raw_condition_scaled=raw_condition,
        singular_values_scaled=tuple(float(value) for value in singular),
        used_prior_sigma_native={} if not prior_sigma_native else _as_float_map(prior_sigma_native),
        normal_ss_native=n_ss,
        normal_sc_native=np.asarray(n_sc, dtype=np.float64),
        normal_cc_native=n_cc,
    )


def schur_profiled_station_normal(
    normal_native: np.ndarray,
    names: Sequence[str],
    *,
    station_names: Sequence[str],
    contrast_name: str = C_DX,
    prior_sigma_native: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Station Schur complement after profiling unknown ``C_dx``.

    Returns ``(N_ss - N_sc N_cc^{-1} N_cs, N_sc, N_cc)`` with the optional
    station prior included in ``N_ss`` before the complement.  This is the
    normal matrix of a station estimator that is first-order insensitive to
    an unknown contrast column.  It does not introduce a written DoF.
    """
    names = tuple(str(name) for name in names)
    station_names = tuple(str(name) for name in station_names)
    index = _index_map(names)
    station_index = [index[name] for name in station_names]
    contrast_index = [index[str(contrast_name)]]
    normal = np.asarray(normal_native, dtype=np.float64)
    n_ss = _apply_prior(
        _block(normal, station_index, station_index),
        station_names,
        prior_sigma_native,
    )
    n_sc = _block(normal, station_index, contrast_index).reshape(len(station_names))
    n_cc = float(_block(normal, contrast_index, contrast_index)[0, 0])
    if not math.isfinite(n_cc) or n_cc <= 0.0:
        raise ValueError("contrast column has no positive information to profile")
    profiled = n_ss - np.outer(n_sc, n_sc) / n_cc
    return 0.5 * (profiled + profiled.T), np.asarray(n_sc, dtype=np.float64), n_cc


def profiled_right_hand_side(
    rhs_native: np.ndarray,
    names: Sequence[str],
    *,
    station_names: Sequence[str],
    contrast_name: str = C_DX,
    normal_native: np.ndarray,
) -> np.ndarray:
    """Schur right-hand side ``b_s - N_sc N_cc^{-1} b_c``."""
    index = _index_map(names)
    station_index = np.asarray([index[name] for name in station_names], dtype=np.intp)
    contrast = index[str(contrast_name)]
    n_sc = np.asarray(normal_native, dtype=np.float64)[station_index, contrast]
    n_cc = float(normal_native[contrast, contrast])
    rhs = np.asarray(rhs_native, dtype=np.float64)
    return rhs[station_index] - n_sc * (rhs[contrast] / n_cc)


def estimator_diagnostics(
    normal_native: np.ndarray,
    names: Sequence[str],
    *,
    parameter_scales: Sequence[float],
    prior_sigma_native: Mapping[str, float] | None = None,
    rcond: float = DEFAULT_RCOND,
) -> dict[str, Any]:
    """Rank, scaled condition, and native sigma for one station estimator."""
    names = tuple(str(name) for name in names)
    scales = np.asarray(parameter_scales, dtype=np.float64)
    n_ss = _apply_prior(np.asarray(normal_native, dtype=np.float64), names, prior_sigma_native)
    scaled = _scaled_normal(n_ss, scales)
    rank, condition, raw_condition, singular = _condition_and_rank(scaled, rcond=rcond)
    _, _, tolerance, _, _ = _svd_rank(scaled, rcond=rcond)
    covariance = np.diag(scales) @ _pinv(scaled, tolerance=tolerance) @ np.diag(scales)
    covariance = 0.5 * (covariance + covariance.T)
    _, _, right = np.linalg.svd(scaled, full_matrices=False)
    selected = singular > tolerance
    projector = right[selected].T @ right[selected] if np.any(selected) else np.zeros_like(scaled)
    observability = np.clip(np.diag(projector), 0.0, 1.0)
    sigma = {}
    for index, name in enumerate(names):
        variance = float(covariance[index, index])
        if float(observability[index]) < 0.5:
            sigma[name] = None
        else:
            sigma[name] = math.sqrt(variance) if math.isfinite(variance) and variance >= 0.0 else None
    return {
        "names": list(names),
        "rank": rank,
        "full_rank": rank == len(names),
        "condition_scaled": condition,
        "raw_condition_scaled": raw_condition,
        "singular_values_scaled": [float(value) for value in singular],
        "sigma_native": sigma,
        "observability_fraction": {name: float(observability[index]) for index, name in enumerate(names)},
        "covariance_native": covariance,
        "used_prior_sigma_native": {} if not prior_sigma_native else _as_float_map(prior_sigma_native),
    }


def sigma_inflation(
    ordinary_sigma: Mapping[str, float | None],
    profiled_sigma: Mapping[str, float | None],
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name, baseline in ordinary_sigma.items():
        profiled = profiled_sigma.get(name)
        if baseline in (None, 0.0) or profiled is None:
            result[name] = None
        else:
            result[name] = float(profiled) / float(baseline)
    return result


def cdx_budget_from_operator(
    operator: LeakageOperatorResult,
    *,
    engineering_tolerance: Mapping[str, float],
    statistical_limit: Mapping[str, float],
) -> dict[str, Any]:
    """Largest |C_dx| a station solve can absorb before a frozen tolerance trips.

    Limits are inverted as ``|tolerance_p / A_p|``.  The binding (minimum)
    budget is the one that station capture would actually use.
    """
    rows = []
    for name in operator.station_names:
        a_value = float(operator.A_native_per_mm[name])
        abs_a = abs(a_value)
        engineering = engineering_tolerance.get(name)
        statistical = statistical_limit.get(name)
        engineering_budget = None if engineering is None or abs_a == 0.0 else abs(float(engineering)) / abs_a
        statistical_budget = None if statistical is None or abs_a == 0.0 else abs(float(statistical)) / abs_a
        rows.append(
            {
                "name": name,
                "A_native_per_mm": a_value,
                "engineering_tolerance": None if engineering is None else float(engineering),
                "statistical_limit": None if statistical is None else float(statistical),
                "max_abs_C_dx_engineering_mm": engineering_budget,
                "max_abs_C_dx_statistical_mm": statistical_budget,
                "max_abs_C_dx_engineering_um": None if engineering_budget is None else 1.0e3 * engineering_budget,
                "max_abs_C_dx_statistical_um": None if statistical_budget is None else 1.0e3 * statistical_budget,
            }
        )

    def _binding(key: str) -> dict[str, Any] | None:
        finite = [row for row in rows if row[key] is not None and math.isfinite(row[key])]
        if not finite:
            return None
        winner = min(finite, key=lambda row: float(row[key]))
        return {
            "name": winner["name"],
            "max_abs_C_dx_mm": float(winner[key]),
            "max_abs_C_dx_um": 1.0e3 * float(winner[key]),
        }

    return {
        "per_parameter": rows,
        "binding_engineering": _binding("max_abs_C_dx_engineering_mm"),
        "binding_statistical": _binding("max_abs_C_dx_statistical_mm"),
    }


def predicted_station_bias(operator: LeakageOperatorResult, c_dx_mm: float) -> dict[str, float]:
    """``E[θ̂_s - θ_s] = A * C_dx`` for a station-only WLS that omits ``j_c``."""
    return {name: float(operator.A_native_per_mm[name]) * float(c_dx_mm) for name in operator.station_names}


def reverse_leakage_from_A_and_covariance(
    *,
    A_native_per_mm: Mapping[str, float],
    station_covariance_native: Mapping[str, Mapping[str, float]],
    normal_cc_native: float,
    station_names: Sequence[str],
) -> dict[str, float]:
    """Recover ``B = N_cc^{-1} N_ss A`` when only ``A``, ``N_cc``, and station covariance are stored.

    Production station covariance already includes the 5 mm ``dz`` prior.  ``B`` is
    the first-order C_dx bias (mm) per leftover station parameter in native units.
    """
    names = tuple(str(name) for name in station_names)
    if not math.isfinite(float(normal_cc_native)) or float(normal_cc_native) == 0.0:
        raise ValueError("normal_cc_native must be finite and non-zero")
    covariance = np.array(
        [[float(station_covariance_native[row][column]) for column in names] for row in names],
        dtype=np.float64,
    )
    a_vector = np.array([float(A_native_per_mm[name]) for name in names], dtype=np.float64)
    n_sc = np.linalg.solve(covariance, a_vector)
    return {name: float(value / float(normal_cc_native)) for name, value in zip(names, n_sc)}


def predicted_cdx_bias(B_native: Mapping[str, float], station_leftover: Mapping[str, float]) -> float:
    """``E[ĉ - c] = B · θ`` for a C_dx-only WLS that omits the station columns."""
    return float(sum(float(B_native[name]) * float(station_leftover[name]) for name in B_native if name in station_leftover))


def cdx_systematic_from_station_sigma(
    B_native: Mapping[str, float],
    station_sigma: Mapping[str, float],
    *,
    include: Sequence[str],
) -> dict[str, Any]:
    """RSS C_dx systematic if each included station leftover is one registered sigma."""
    terms: dict[str, float] = {}
    for name in include:
        if name not in B_native:
            raise ValueError(f"reverse leakage lacks {name!r}")
        if name not in station_sigma:
            raise ValueError(f"station sigma lacks {name!r}")
        terms[name] = abs(float(B_native[name])) * abs(float(station_sigma[name]))
    rss = math.sqrt(sum(value * value for value in terms.values()))
    return {
        "included_parameters": list(include),
        "per_parameter_1sigma_mm": terms,
        "rss_1sigma_mm": rss,
        "rss_3sigma_mm": 3.0 * rss,
        "rss_1sigma_um": 1.0e3 * rss,
        "rss_3sigma_um": 3.0e3 * rss,
    }


def slice_parameter_bank(bank: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    """Restrict a finite-difference bank to a column subset."""
    requested = tuple(str(name) for name in names)
    available = tuple(str(name) for name in bank["names"])
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError("bank is missing parameters: " + ", ".join(missing))
    indices = [available.index(name) for name in requested]
    updated = dict(bank)
    updated["names"] = requested
    updated["scales"] = np.asarray(bank["scales"], dtype=np.float64)[indices]
    specs = list(bank["specs"])
    updated["specs"] = [specs[index] for index in indices]
    for key in ("positive_residual", "negative_residual"):
        updated[key] = np.asarray(bank[key], dtype=np.float64)[indices]
    for key in ("positive_values", "negative_values", "anchor_values", "reference_values"):
        updated[key] = np.asarray(bank[key], dtype=np.float64)[indices]
    if "target_values" in bank:
        updated["target_values"] = {
            str(target): np.asarray(values, dtype=np.float64)[indices]
            for target, values in bank["target_values"].items()
        }
    return updated


def compare_station_estimators(
    *,
    ordinary: Mapping[str, Any],
    profiled: Mapping[str, Any],
    five_names: Sequence[str] = STATION_FIVE,
) -> dict[str, Any]:
    """Rank / condition / sigma-inflation comparison for the nuisance control."""
    ordinary_sigma = ordinary["sigma_native"]
    profiled_sigma = profiled["sigma_native"]
    inflation = sigma_inflation(ordinary_sigma, profiled_sigma)
    five = tuple(five_names)
    five_ordinary_rank = sum(1 for name in five if ordinary_sigma.get(name) not in (None, 0.0))
    five_profiled_full = all(
        profiled.get("full_rank") is True or (profiled_sigma.get(name) not in (None, 0.0)) for name in five
    )
    ordinary_cond = ordinary.get("condition_scaled")
    profiled_cond = profiled.get("condition_scaled")
    condition_ratio = None
    if (
        ordinary_cond not in (None, 0.0)
        and profiled_cond is not None
        and math.isfinite(float(ordinary_cond))
        and math.isfinite(float(profiled_cond))
    ):
        condition_ratio = float(profiled_cond) / float(ordinary_cond)
    ordinary_raw = ordinary.get("raw_condition_scaled")
    profiled_raw = profiled.get("raw_condition_scaled")
    raw_condition_ratio = None
    if (
        ordinary_raw not in (None, 0.0)
        and profiled_raw is not None
        and math.isfinite(float(ordinary_raw))
        and math.isfinite(float(profiled_raw))
    ):
        raw_condition_ratio = float(profiled_raw) / float(ordinary_raw)
    return {
        "ordinary_rank": ordinary["rank"],
        "profiled_rank": profiled["rank"],
        "ordinary_full_rank": ordinary["full_rank"],
        "profiled_full_rank": profiled["full_rank"],
        "ordinary_condition_scaled": ordinary_cond,
        "profiled_condition_scaled": profiled_cond,
        "ordinary_raw_condition_scaled": ordinary_raw,
        "profiled_raw_condition_scaled": profiled_raw,
        "condition_ratio_profiled_over_ordinary": condition_ratio,
        "raw_condition_ratio_profiled_over_ordinary": raw_condition_ratio,
        "sigma_native_ordinary": ordinary_sigma,
        "sigma_native_profiled": profiled_sigma,
        "sigma_inflation_profiled_over_ordinary": inflation,
        "five_dof_ordinary_finite_sigma": five_ordinary_rank,
        "five_dof_profiled_looks_full": five_profiled_full and profiled["full_rank"],
        "survey_parameter": STATION_SURVEY_PARAMETER,
    }


def default_survey_prior() -> dict[str, float]:
    return {STATION_SURVEY_PARAMETER: SURVEY_DZ_PRIOR_MM}


def realizability_from_budget(
    *,
    binding_engineering_um: float | None,
    binding_statistical_um: float | None,
    registered_cdx_sigma_um: float,
    leftover_cdx_um: Sequence[float],
) -> dict[str, Any]:
    """Compare the inverted station-dx budget to the layer estimator's reach."""
    leftover = [float(value) for value in leftover_cdx_um if math.isfinite(float(value))]
    leftover_max = max(leftover) if leftover else None
    leftover_min = min(leftover) if leftover else None
    engineering_ok = (
        binding_engineering_um is not None and registered_cdx_sigma_um < float(binding_engineering_um)
    )
    statistical_ok = (
        binding_statistical_um is not None and registered_cdx_sigma_um < float(binding_statistical_um)
    )
    leftover_ok = leftover_max is not None and binding_engineering_um is not None and leftover_max < float(
        binding_engineering_um
    )
    sequential_realizable = bool(engineering_ok and leftover_ok)
    return {
        "registered_C_dx_sigma_um": float(registered_cdx_sigma_um),
        "reverse_path_leftover_C_dx_um_min": leftover_min,
        "reverse_path_leftover_C_dx_um_max": leftover_max,
        "binding_engineering_C_dx_um": binding_engineering_um,
        "binding_statistical_C_dx_um": binding_statistical_um,
        "registered_sigma_below_engineering_budget": engineering_ok,
        "registered_sigma_below_statistical_budget": statistical_ok,
        "leftover_below_engineering_budget": leftover_ok,
        "sequential_hierarchy_statistically_realizable": sequential_realizable,
        "statement_if_unrealizable": (
            "under the current reconstruction/sample, simultaneous hierarchical "
            "calibration is statistically unrealizable"
        ),
    }
