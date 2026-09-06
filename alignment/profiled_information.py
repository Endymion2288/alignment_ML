"""Schur-complement profiling of track nuisance.  Toy / contract math only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from alignment.numerical_contract import require_spd


class ProfiledInformationError(ValueError):
    """Raised when a profiled block is illegal or information is inflated."""


@dataclass(frozen=True)
class ProfiledSolve:
    delta_alignment: np.ndarray
    covariance_alignment: np.ndarray
    normal_alignment: np.ndarray
    projector: np.ndarray
    chi2: float


def weight_matrix(covariance: np.ndarray) -> np.ndarray:
    return np.linalg.inv(require_spd(covariance, name="measurement R"))


def joint_normal(
    g: np.ndarray,
    h: np.ndarray,
    weight: np.ndarray,
    *,
    nq: np.ndarray | None = None,
    ns: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (Nαα, Nαq, Nqα, Nqq) without forming the residual."""
    n_aa = g.T @ weight @ g
    n_aq = g.T @ weight @ h
    n_qa = h.T @ weight @ g
    n_qq = h.T @ weight @ h
    if ns is not None:
        n_aa = n_aa + np.asarray(ns, dtype=np.float64)
    if nq is not None:
        n_qq = n_qq + np.asarray(nq, dtype=np.float64)
    return n_aa, n_aq, n_qa, n_qq


def symmetric_pinv(matrix: np.ndarray, *, relative: float = 1.0e-10) -> tuple[np.ndarray, int]:
    """Moore-Penrose inverse of a symmetric normal matrix.  Not covariance clipping."""
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    scale = max(float(np.max(np.abs(values))), 1.0)
    keep = values > float(relative) * scale
    if not np.any(keep):
        return np.zeros_like(matrix), 0
    inverse = (vectors[:, keep] * (1.0 / values[keep])) @ vectors[:, keep].T
    return 0.5 * (inverse + inverse.T), int(np.count_nonzero(keep))


def schur_normal(
    g: np.ndarray,
    h: np.ndarray,
    weight: np.ndarray,
    *,
    nq: np.ndarray | None = None,
    ns: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """P = W - W H (H^T W H + Nq)^{+} H^T W;  Nα = G^T P G + Ns."""
    _n_aa, _n_aq, _n_qa, n_qq = joint_normal(g, h, weight, nq=nq, ns=ns)
    n_qq_inv, _rank = symmetric_pinv(n_qq)
    projector = weight - weight @ h @ n_qq_inv @ h.T @ weight
    projector = 0.5 * (projector + projector.T)
    n_alpha = g.T @ projector @ g
    if ns is not None:
        n_alpha = n_alpha + np.asarray(ns, dtype=np.float64)
    n_alpha = 0.5 * (n_alpha + n_alpha.T)
    return n_alpha, projector


def solve_joint(
    g: np.ndarray,
    h: np.ndarray,
    residual: np.ndarray,
    covariance: np.ndarray,
    *,
    nq: np.ndarray | None = None,
    ns: np.ndarray | None = None,
    bs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weight = weight_matrix(covariance)
    n_aa, n_aq, n_qa, n_qq = joint_normal(g, h, weight, nq=nq, ns=ns)
    top = np.concatenate([n_aa, n_aq], axis=1)
    bottom = np.concatenate([n_qa, n_qq], axis=1)
    normal = np.concatenate([top, bottom], axis=0)
    rhs_a = g.T @ weight @ residual
    rhs_q = h.T @ weight @ residual
    if bs is not None:
        rhs_a = rhs_a + np.asarray(bs, dtype=np.float64)
    rhs = np.concatenate([rhs_a, rhs_q])
    solution = np.linalg.solve(normal, rhs)
    n_align = g.shape[1]
    return solution[:n_align], solution[n_align:], normal


def solve_profiled(
    g: np.ndarray,
    h: np.ndarray,
    residual: np.ndarray,
    covariance: np.ndarray,
    *,
    nq: np.ndarray | None = None,
    ns: np.ndarray | None = None,
    bs: np.ndarray | None = None,
) -> ProfiledSolve:
    weight = weight_matrix(covariance)
    n_alpha, projector = schur_normal(g, h, weight, nq=nq, ns=ns)
    rhs = g.T @ projector @ residual
    if bs is not None:
        rhs = rhs + np.asarray(bs, dtype=np.float64)
    n_alpha = require_spd(n_alpha, name="profiled alignment normal")
    delta = np.linalg.solve(n_alpha, rhs)
    cov = np.linalg.inv(n_alpha)
    predicted = projector @ residual - projector @ g @ delta
    chi2 = float(residual @ projector @ residual - rhs @ delta)
    del predicted
    return ProfiledSolve(
        delta_alignment=delta,
        covariance_alignment=cov,
        normal_alignment=n_alpha,
        projector=projector,
        chi2=chi2,
    )


def assert_no_information_inflation(
    fixed_q_normal: np.ndarray,
    profiled_normal: np.ndarray,
    *,
    atol: float = 1.0e-10,
) -> None:
    """Fixed-q may look more informative.  Profiling must not exceed it."""
    gap = 0.5 * (
        (fixed_q_normal - profiled_normal) + (fixed_q_normal - profiled_normal).T
    )
    eigenvalues = np.linalg.eigvalsh(gap)
    if np.min(eigenvalues) < -atol:
        raise ProfiledInformationError(
            "profiled information exceeds fixed-q information without physics"
        )


def weak_direction_status(normal: np.ndarray, *, relative: float = 1.0e-8) -> dict[str, Any]:
    values = np.linalg.eigvalsh(0.5 * (normal + normal.T))
    maximum = float(np.max(np.abs(values))) if values.size else 0.0
    floor = max(relative * maximum, np.finfo(np.float64).eps)
    n_weak = int(np.sum(values <= floor))
    return {
        "eigenvalues": [float(value) for value in values],
        "n_weak": n_weak,
        "unconstrained": bool(n_weak > 0),
        "do_not_force_zero": True,
    }
