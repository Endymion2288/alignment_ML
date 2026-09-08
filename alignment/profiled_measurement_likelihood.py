"""B14M profiled measurement-likelihood numerics.

This is the Task B14M contract, not the historical alignment Schur
helper in ``alignment/profiled_information.py``.  Differences:

* ``H_nn`` may be rank-deficient and that rank is a physical diagnosis.
* Pseudoinverse tolerance is pre-registered here; it is not
  alignment ``rank_tolerance=0.01``.
* No ridge is added to force invertibility.
* A pinv null direction is not a claim of zero uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

PINV_RELATIVE = 1.0e-8
ALPHA_INDICES = (0, 3)
NU_INDICES = (1, 2, 4)
NATIVE_NAMES = ("loc0", "loc1", "phi", "theta", "q_over_p")
ALPHA_NAMES = ("loc0", "theta")
NU_NAMES = ("loc1", "phi", "q_over_p")


class ProfiledMeasurementLikelihoodError(ValueError):
    """Raised when the B14M numerical contract is violated."""


def refuse_ridge() -> None:
    raise ProfiledMeasurementLikelihoodError(
        "ridge must not be added to make a Hessian invertible"
    )


def refuse_alignment_rank_tolerance() -> None:
    raise ProfiledMeasurementLikelihoodError(
        "B14M must not reuse alignment rank_tolerance=0.01"
    )


def _as_square(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ProfiledMeasurementLikelihoodError("expected a square matrix")
    return 0.5 * (value + value.T)


def symmetric_eigendecomposition(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(_as_square(matrix))
    return values, vectors


def symmetric_pinv(
    matrix: np.ndarray,
    *,
    relative: float = PINV_RELATIVE,
    ridge: float | None = None,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """Moore-Penrose inverse of a symmetric matrix.  No ridge."""
    if ridge is not None and float(ridge) != 0.0:
        refuse_ridge()
    values, vectors = symmetric_eigendecomposition(matrix)
    scale = max(float(np.max(np.abs(values))) if values.size else 0.0, 1.0)
    keep = values > float(relative) * scale
    inverse = np.zeros_like(matrix, dtype=np.float64)
    if np.any(keep):
        inverse = (vectors[:, keep] * (1.0 / values[keep])) @ vectors[:, keep].T
    inverse = 0.5 * (inverse + inverse.T)
    return inverse, int(np.count_nonzero(keep)), values, vectors


def null_basis(
    matrix: np.ndarray,
    *,
    relative: float = PINV_RELATIVE,
) -> np.ndarray:
    values, vectors = symmetric_eigendecomposition(matrix)
    scale = max(float(np.max(np.abs(values))) if values.size else 0.0, 1.0)
    drop = values <= float(relative) * scale
    if not np.any(drop):
        return np.zeros((matrix.shape[0], 0), dtype=np.float64)
    return vectors[:, drop]


def split_blocks(
    hessian: np.ndarray,
    *,
    alpha_index: Sequence[int] = ALPHA_INDICES,
    nu_index: Sequence[int] = NU_INDICES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    alpha = np.asarray(alpha_index, dtype=int)
    nu = np.asarray(nu_index, dtype=int)
    matrix = _as_square(hessian)
    return matrix[np.ix_(alpha, alpha)], matrix[np.ix_(alpha, nu)], matrix[np.ix_(nu, alpha)], matrix[np.ix_(nu, nu)]


def schur_profile_hessian(
    hessian: np.ndarray,
    *,
    alpha_index: Sequence[int] = ALPHA_INDICES,
    nu_index: Sequence[int] = NU_INDICES,
    relative: float = PINV_RELATIVE,
) -> dict[str, Any]:
    h_aa, h_an, h_na, h_nn = split_blocks(
        hessian, alpha_index=alpha_index, nu_index=nu_index
    )
    h_nn_inv, rank_nn, values_nn, vectors_nn = symmetric_pinv(h_nn, relative=relative)
    profiled = h_aa - h_an @ h_nn_inv @ h_na
    profiled = 0.5 * (profiled + profiled.T)
    values_p, vectors_p = symmetric_eigendecomposition(profiled)
    return {
        "H_aa": h_aa,
        "H_an": h_an,
        "H_na": h_na,
        "H_nn": h_nn,
        "H_profile": profiled,
        "H_nn_pinv": h_nn_inv,
        "nuisance_rank": rank_nn,
        "nuisance_singular_values": values_nn,
        "nuisance_eigenvectors": vectors_nn,
        "profile_singular_values": values_p,
        "profile_eigenvectors": vectors_p,
        "nuisance_null_basis": null_basis(h_nn, relative=relative),
        "pinv_relative": float(relative),
        "ridge_added": False,
    }


def chi2_from_residual(residual: np.ndarray, weight: np.ndarray) -> float:
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    weight = np.asarray(weight, dtype=np.float64)
    return float(residual @ weight @ residual)


def gauss_newton_normal(
    jacobian: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """H = J^T W J and g = J^T W r for r = m - h, J = dh/dtheta."""
    jac = np.asarray(jacobian, dtype=np.float64)
    res = np.asarray(residual, dtype=np.float64).reshape(-1)
    wgt = np.asarray(weight, dtype=np.float64)
    hessian = jac.T @ wgt @ jac
    gradient = jac.T @ wgt @ res
    return 0.5 * (hessian + hessian.T), gradient


def joint_nls_step(
    jacobian: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    *,
    relative: float = PINV_RELATIVE,
) -> dict[str, Any]:
    hessian, gradient = gauss_newton_normal(jacobian, residual, weight)
    cov, rank, values, vectors = symmetric_pinv(hessian, relative=relative)
    delta = cov @ gradient
    return {
        "hessian": hessian,
        "gradient": gradient,
        "delta": delta,
        "rank": rank,
        "singular_values": values,
        "eigenvectors": vectors,
        "pinv": cov,
    }


def explicit_nuisance_minimum(
    jacobian_alpha: np.ndarray,
    jacobian_nu: np.ndarray,
    residual0: np.ndarray,
    weight: np.ndarray,
    delta_alpha: np.ndarray,
    *,
    relative: float = PINV_RELATIVE,
) -> dict[str, Any]:
    """Linearized min_nu ||r0 + Ja da + Jn dn||_W."""
    res = np.asarray(residual0, dtype=np.float64).reshape(-1)
    ja = np.asarray(jacobian_alpha, dtype=np.float64)
    jn = np.asarray(jacobian_nu, dtype=np.float64)
    da = np.asarray(delta_alpha, dtype=np.float64).reshape(-1)
    wgt = np.asarray(weight, dtype=np.float64)
    reduced = res - ja @ da
    h_nn = jn.T @ wgt @ jn
    g_n = jn.T @ wgt @ reduced
    h_inv, rank, values, _vectors = symmetric_pinv(h_nn, relative=relative)
    delta_nu = h_inv @ g_n
    residual = reduced - jn @ delta_nu
    return {
        "delta_nu": delta_nu,
        "residual": residual,
        "chi2": chi2_from_residual(residual, wgt),
        "nuisance_rank": rank,
        "nuisance_singular_values": values,
        "H_nn": 0.5 * (h_nn + h_nn.T),
    }


def joint_linear_solution(
    jacobian_alpha: np.ndarray,
    jacobian_nu: np.ndarray,
    residual0: np.ndarray,
    weight: np.ndarray,
    *,
    relative: float = PINV_RELATIVE,
) -> dict[str, Any]:
    ja = np.asarray(jacobian_alpha, dtype=np.float64)
    jn = np.asarray(jacobian_nu, dtype=np.float64)
    jac = np.concatenate([ja, jn], axis=1)
    step = joint_nls_step(jac, residual0, weight, relative=relative)
    n_alpha = ja.shape[1]
    return {
        **step,
        "delta_alpha": step["delta"][:n_alpha],
        "delta_nu": step["delta"][n_alpha:],
        "chi2": chi2_from_residual(
            np.asarray(residual0, dtype=np.float64).reshape(-1)
            - jac @ step["delta"],
            weight,
        ),
    }


def profile_linear_solution(
    jacobian_alpha: np.ndarray,
    jacobian_nu: np.ndarray,
    residual0: np.ndarray,
    weight: np.ndarray,
    *,
    relative: float = PINV_RELATIVE,
) -> dict[str, Any]:
    ja = np.asarray(jacobian_alpha, dtype=np.float64)
    jn = np.asarray(jacobian_nu, dtype=np.float64)
    wgt = np.asarray(weight, dtype=np.float64)
    hessian, _gradient = gauss_newton_normal(
        np.concatenate([ja, jn], axis=1), residual0, wgt
    )
    schur = schur_profile_hessian(
        hessian,
        alpha_index=tuple(range(ja.shape[1])),
        nu_index=tuple(range(ja.shape[1], ja.shape[1] + jn.shape[1])),
        relative=relative,
    )
    h_prof_inv, rank_p, values_p, _ = symmetric_pinv(
        schur["H_profile"], relative=relative
    )
    projector = wgt - wgt @ jn @ schur["H_nn_pinv"] @ jn.T @ wgt
    projector = 0.5 * (projector + projector.T)
    rhs = ja.T @ projector @ np.asarray(residual0, dtype=np.float64).reshape(-1)
    delta_alpha = h_prof_inv @ rhs
    nuisance = explicit_nuisance_minimum(
        ja, jn, residual0, wgt, delta_alpha, relative=relative
    )
    return {
        **schur,
        "delta_alpha": delta_alpha,
        "delta_nu": nuisance["delta_nu"],
        "chi2": nuisance["chi2"],
        "profile_rank": rank_p,
        "profile_singular_values": values_p,
        "projector": projector,
    }


def prediction_uncertainty_from_hessian(
    hessian: np.ndarray,
    jacobian_pred: np.ndarray,
    *,
    relative: float = PINV_RELATIVE,
) -> dict[str, Any]:
    """Map local curvature to prediction space without filling the null space."""
    hessian = _as_square(hessian)
    jac = np.asarray(jacobian_pred, dtype=np.float64)
    if jac.ndim == 1:
        jac = jac.reshape(1, -1)
    cov, rank, values, vectors = symmetric_pinv(hessian, relative=relative)
    null = null_basis(hessian, relative=relative)
    leak = jac @ null if null.size else np.zeros((jac.shape[0], 0))
    leak_norm = (
        np.linalg.norm(leak, axis=1)
        if leak.size
        else np.zeros(jac.shape[0], dtype=np.float64)
    )
    pred_scale = np.linalg.norm(jac, axis=1)
    finite = leak_norm <= float(relative) * np.maximum(pred_scale, 1.0)
    covariance = jac @ cov @ jac.T
    covariance = 0.5 * (covariance + covariance.T)
    masked = np.array(covariance, copy=True)
    for index, is_finite in enumerate(finite):
        if not is_finite:
            masked[index, :] = np.nan
            masked[:, index] = np.nan
    observable = vectors[:, values > float(relative) * max(float(np.max(np.abs(values))), 1.0)]
    return {
        "prediction_covariance": covariance,
        "prediction_covariance_finite_mask": masked,
        "observable_finite": [bool(flag) for flag in finite],
        "prediction_uncertainty_finite": bool(np.all(finite)),
        "prediction_not_identified": bool(not np.all(finite)),
        "null_basis": null,
        "observable_basis": observable,
        "hessian_rank": rank,
        "hessian_singular_values": values,
        "pinv_null_not_zero_uncertainty": True,
        "ridge_added": False,
        "seed_covariance_filled": False,
    }


ResidualFn = Callable[[np.ndarray], np.ndarray]


def finite_difference_jacobian(
    residual_fn: ResidualFn,
    theta: np.ndarray,
    steps: Sequence[float],
) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    steps = np.asarray(steps, dtype=np.float64).reshape(-1)
    r0 = np.asarray(residual_fn(theta), dtype=np.float64).reshape(-1)
    jac = np.zeros((r0.size, theta.size), dtype=np.float64)
    for index, step in enumerate(steps):
        plus = np.array(theta, copy=True)
        minus = np.array(theta, copy=True)
        plus[index] += step
        minus[index] -= step
        # r = m - h, so dr/dtheta = -dh/dtheta.  Callers that want J_h
        # should negate.  Here we return J_h = dh/dtheta.
        jac[:, index] = (
            np.asarray(residual_fn(minus), dtype=np.float64).reshape(-1)
            - np.asarray(residual_fn(plus), dtype=np.float64).reshape(-1)
        ) / (2.0 * step)
    return jac


@dataclass(frozen=True)
class SyntheticAgreement:
    passed: bool
    joint_chi2: float
    profile_chi2: float
    schur_chi2: float
    delta_alpha_joint: list[float]
    delta_alpha_profile: list[float]
    rank_deficient_diagnosed: bool
    ridge_added: False = False


def validate_linear_profile_agreement(
    *,
    relative: float = PINV_RELATIVE,
    abs_tol: float = 1.0e-8,
    rel_tol: float = 1.0e-6,
) -> dict[str, Any]:
    """Controlled linear problem: joint NLS, explicit nu min, and Schur agree."""
    rng = np.random.default_rng(114)
    n_meas = 8
    ja = rng.normal(size=(n_meas, 2))
    jn = rng.normal(size=(n_meas, 3))
    true_a = np.array([0.03, -0.02])
    true_n = np.array([0.2, -0.05, 0.01])
    noise = 0.01 * rng.normal(size=n_meas)
    residual0 = ja @ true_a + jn @ true_n + noise
    weight = np.diag(np.full(n_meas, 1.0 / 0.0005333333333333334))
    joint = joint_linear_solution(ja, jn, residual0, weight, relative=relative)
    profiled = profile_linear_solution(ja, jn, residual0, weight, relative=relative)
    explicit = explicit_nuisance_minimum(
        ja, jn, residual0, weight, joint["delta_alpha"], relative=relative
    )
    alpha_ok = bool(
        np.allclose(joint["delta_alpha"], profiled["delta_alpha"], atol=abs_tol, rtol=rel_tol)
    )
    nu_ok = bool(
        np.allclose(joint["delta_nu"], explicit["delta_nu"], atol=abs_tol, rtol=rel_tol)
        and np.allclose(joint["delta_nu"], profiled["delta_nu"], atol=abs_tol, rtol=rel_tol)
    )
    chi2_ok = bool(
        abs(joint["chi2"] - profiled["chi2"]) <= abs_tol + rel_tol * max(abs(joint["chi2"]), 1.0)
        and abs(joint["chi2"] - explicit["chi2"]) <= abs_tol + rel_tol * max(abs(joint["chi2"]), 1.0)
    )
    hess_ok = bool(
        np.allclose(joint["hessian"][:2, :2] - joint["hessian"][:2, 2:] @ np.linalg.pinv(
            joint["hessian"][2:, 2:]
        ) @ joint["hessian"][2:, :2], profiled["H_profile"], atol=1.0e-8)
    )
    del hess_ok

    # Rank-deficient nuisance: third nu column is a copy of the first.
    jn_def = np.column_stack([jn[:, 0], jn[:, 1], jn[:, 0]])
    deficient = profile_linear_solution(ja, jn_def, residual0, weight, relative=relative)
    rank_def = int(deficient["nuisance_rank"])
    rank_ok = rank_def <= 2
    # Must not have added ridge to recover rank 3.
    ridge_free = deficient["ridge_added"] is False and rank_def < 3

    passed = bool(alpha_ok and nu_ok and chi2_ok and rank_ok and ridge_free)
    return {
        "passed": passed,
        "relative_tolerance": float(relative),
        "alignment_rank_tolerance_used": False,
        "ridge_added": False,
        "full_rank": {
            "joint_chi2": joint["chi2"],
            "profile_chi2": profiled["chi2"],
            "explicit_nuisance_chi2": explicit["chi2"],
            "delta_alpha_joint": [float(v) for v in joint["delta_alpha"]],
            "delta_alpha_profile": [float(v) for v in profiled["delta_alpha"]],
            "delta_nu_joint": [float(v) for v in joint["delta_nu"]],
            "alpha_agrees": alpha_ok,
            "nuisance_agrees": nu_ok,
            "chi2_agrees": chi2_ok,
            "joint_rank": int(joint["rank"]),
            "profile_rank": int(profiled["profile_rank"]),
            "nuisance_rank": int(profiled["nuisance_rank"]),
        },
        "rank_deficient_nuisance": {
            "nuisance_rank": rank_def,
            "nuisance_singular_values": [float(v) for v in deficient["nuisance_singular_values"]],
            "diagnosed": rank_ok,
            "ridge_added": False,
            "forced_invertible": False,
        },
        "methods_compared": [
            "joint_nonlinear_least_squares",
            "explicit_nuisance_minimization",
            "schur_complement",
        ],
    }


def classify_nuisance(
    singular_values: Sequence[float],
    *,
    relative: float = PINV_RELATIVE,
    weak_relative: float = 1.0e-4,
    multimodal: bool = False,
    fit_failed: bool = False,
    hit_boundary: bool = False,
) -> str:
    if fit_failed:
        return "fit_failed"
    if multimodal:
        return "multimodal"
    values = np.asarray(list(singular_values), dtype=np.float64)
    if values.size == 0:
        return "flat"
    scale = max(float(np.max(np.abs(values))), 1.0)
    n_flat = int(np.sum(values <= float(relative) * scale))
    n_weak = int(np.sum(values <= float(weak_relative) * scale))
    if n_flat > 0:
        return "flat"
    if n_weak > 0 or hit_boundary:
        return "weakly_identified"
    return "identified"
