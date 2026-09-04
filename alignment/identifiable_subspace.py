"""Tracker-only identifiable subspace from a weighted, pre-scaled Jacobian.

Survey / metrology is an external cross-check, not an alignment input.
The solver is structurally restricted to the identifiable subspace of

    A = W^{1/2} J S

and must not SVD a naked mixed-unit Jacobian.  Parameter scales ``S`` are
frozen before the SVD; singular values are never used to retune ``S`` or
the rank threshold.

Right singular vectors of ``A`` live in the dimensionless coordinates
``u = S^{-1} θ``.  They are reconstruction-observable linear combinations,
not mechanical parameters such as a lone station ``ry`` or ``C_dx``.
``V_null^T u_hat = 0`` is the minimum-norm / gauge representative in those
coordinates, not a measurement that the null modes are physically zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from alignment.five_dof_sampling import DEFAULT_SCALES
from alignment.hierarchical_v1 import C_DX, C_DX_ENVELOPE_MM, HIERARCHICAL_V1_PARAMETERS
from alignment.physical_jacobian import RESIDUAL_DIMENSION
from alignment.true_cluster_local_residual import LEAKAGE_RANK_RELATIVE_TOLERANCE


FROZEN_RANK_TOLERANCE = LEAKAGE_RANK_RELATIVE_TOLERANCE
FROZEN_NORMAL_RCOND = 1.0e-10
FROZEN_PARAMETER_SCALES: dict[str, float] = {
    **DEFAULT_SCALES,
    C_DX: float(C_DX_ENVELOPE_MM),
}
FROZEN_PARAMETER_UNITS: dict[str, str] = {
    "ift_dx_mm": "mm",
    "ift_dy_mm": "mm",
    "ift_dz_mm": "mm",
    "ift_rx_mrad": "mrad",
    "ift_ry_mrad": "mrad",
    "ift_rz_mrad": "mrad",
    C_DX: "mm",
    "station_dx_mm": "mm",
    "station_ry_mrad": "mrad",
    "C_dx_mm": "mm",
}
HIERARCHICAL_NATIVE_NAMES: tuple[str, ...] = HIERARCHICAL_V1_PARAMETERS
CLUSTER_LOCAL_NATIVE_NAMES: tuple[str, ...] = ("station_dx_mm", "station_ry_mrad", "C_dx_mm")
CLUSTER_LOCAL_SCALE_MAP: dict[str, float] = {
    "station_dx_mm": DEFAULT_SCALES["ift_dx_mm"],
    "station_ry_mrad": DEFAULT_SCALES["ift_ry_mrad"],
    "C_dx_mm": float(C_DX_ENVELOPE_MM),
}


class MixedUnitNakedJacobianError(ValueError):
    """Raised when SVD is requested on a mixed-unit Jacobian without ``S``."""


class RankThresholdRetuneError(ValueError):
    """Raised when a caller tries to set the rank cut from singular values."""


class ScaleRetuneError(ValueError):
    """Raised when a caller tries to set ``S`` from the SVD spectrum."""


@dataclass(frozen=True)
class IdentifiableSubspace:
    """SVD of ``A = W^{1/2} J S`` with a frozen rank cut and sign convention."""

    parameter_names: tuple[str, ...]
    parameter_units: tuple[str, ...]
    parameter_scales: np.ndarray
    rank_tolerance: float
    singular_values: np.ndarray
    identifiable_rank: int
    u: np.ndarray
    v: np.ndarray
    v_id: np.ndarray
    v_null: np.ndarray
    projector_id: np.ndarray
    mode_signs: tuple[int, ...]
    n_observations: int
    n_parameters: int

    @property
    def null_dimension(self) -> int:
        return int(self.n_parameters - self.identifiable_rank)

    @property
    def scale_matrix(self) -> np.ndarray:
        return np.diag(np.asarray(self.parameter_scales, dtype=np.float64))


def frozen_scales_for(names: Sequence[str], overrides: Mapping[str, float] | None = None) -> np.ndarray:
    """Return the pre-declared positive scales.  SVD spectrum is not consulted."""
    resolved = dict(FROZEN_PARAMETER_SCALES)
    resolved.update(CLUSTER_LOCAL_SCALE_MAP)
    if overrides:
        for name, value in overrides.items():
            scale = float(value)
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"scale override for {name!r} must be positive and finite")
            resolved[str(name)] = scale
    missing = [str(name) for name in names if str(name) not in resolved]
    if missing:
        raise ValueError("no frozen scale for " + ", ".join(missing))
    return np.asarray([float(resolved[str(name)]) for name in names], dtype=np.float64)


def frozen_units_for(names: Sequence[str]) -> tuple[str, ...]:
    missing = [str(name) for name in names if str(name) not in FROZEN_PARAMETER_UNITS]
    if missing:
        raise ValueError("no frozen unit for " + ", ".join(missing))
    return tuple(FROZEN_PARAMETER_UNITS[str(name)] for name in names)


def refuse_naked_mixed_unit_svd(parameter_units: Sequence[str]) -> None:
    units = tuple(str(unit) for unit in parameter_units)
    if len(set(units)) > 1:
        raise MixedUnitNakedJacobianError(
            "refusing SVD of a naked Jacobian whose columns mix units "
            f"{sorted(set(units))}; form A = W^(1/2) J S with the frozen scale matrix"
        )


def refuse_scale_or_threshold_retune(*, from_singular_values: bool) -> None:
    if from_singular_values:
        raise ScaleRetuneError("parameter scales must not be retuned from singular values")


def refuse_rank_threshold_from_spectrum() -> None:
    raise RankThresholdRetuneError(
        "rank_tolerance is frozen at "
        f"{FROZEN_RANK_TOLERANCE} and must not be set from the observed spectrum"
    )


def svd_naked_jacobian(
    jacobian: object,
    *,
    parameter_units: Sequence[str],
) -> None:
    """Explicitly refuse mixed-unit SVD of an unscaled Jacobian."""
    del jacobian
    refuse_naked_mixed_unit_svd(parameter_units)
    raise MixedUnitNakedJacobianError("naked Jacobian SVD is not part of this stage")


def covariance_left_sqrt_inverse(covariance: object) -> list[np.ndarray]:
    """Return ``M`` with ``M.T @ M = C^{-1}`` for each 4×4 residual covariance."""
    matrices = np.asarray(covariance, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (RESIDUAL_DIMENSION, RESIDUAL_DIMENSION):
        raise ValueError(
            f"covariance must have shape [pairs, {RESIDUAL_DIMENSION}, {RESIDUAL_DIMENSION}]"
        )
    if not np.isfinite(matrices).all():
        raise ValueError("covariance contains non-finite values")
    blocks: list[np.ndarray] = []
    identity = np.eye(RESIDUAL_DIMENSION, dtype=np.float64)
    for row, matrix in enumerate(matrices):
        if not np.allclose(matrix, matrix.T, rtol=1.0e-7, atol=1.0e-12):
            raise ValueError(f"covariance row {row} is not symmetric")
        try:
            factor = np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as error:
            raise ValueError(f"covariance row {row} is not positive definite") from error
        blocks.append(np.linalg.solve(factor, identity))
    return blocks


def flatten_physical_jacobian(
    derivative: object,
    covariance: object,
    scales: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Build flat ``J`` and ``A = W^{1/2} J S`` from pair-wise 4-vector derivatives."""
    jacobian = np.asarray(derivative, dtype=np.float64)
    scale = np.asarray(scales, dtype=np.float64)
    if jacobian.ndim != 3 or jacobian.shape[1] != RESIDUAL_DIMENSION:
        raise ValueError(
            f"derivative must have shape [pairs, {RESIDUAL_DIMENSION}, parameters]"
        )
    if scale.ndim != 1 or scale.shape[0] != jacobian.shape[2]:
        raise ValueError("scales must have one positive entry per Jacobian column")
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("scales must be finite and positive")
    pairs = jacobian.shape[0]
    if pairs < 1:
        raise ValueError("weighted Jacobian requires at least one residual pair")
    blocks = covariance_left_sqrt_inverse(covariance)
    if len(blocks) != pairs:
        raise ValueError("covariance pair count does not match the derivative")
    n_obs = pairs * RESIDUAL_DIMENSION
    n_par = int(jacobian.shape[2])
    flat = np.zeros((n_obs, n_par), dtype=np.float64)
    weighted = np.zeros((n_obs, n_par), dtype=np.float64)
    scale_matrix = np.diag(scale)
    for index, block in enumerate(blocks):
        sl = slice(index * RESIDUAL_DIMENSION, (index + 1) * RESIDUAL_DIMENSION)
        flat[sl] = jacobian[index]
        weighted[sl] = block @ jacobian[index] @ scale_matrix
    return weighted, flat, blocks


def flatten_diagonal_jacobian(
    jacobian: object,
    residual_variance: object,
    scales: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``A = W^{1/2} J S`` for a 1-D residual with diagonal variance."""
    values = np.asarray(jacobian, dtype=np.float64)
    variance = np.asarray(residual_variance, dtype=np.float64)
    scale = np.asarray(scales, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("1-D Jacobian must have shape [observations, parameters]")
    if variance.shape != (values.shape[0],):
        raise ValueError("residual_variance must have one entry per Jacobian row")
    if scale.shape != (values.shape[1],) or not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("scales must be finite, positive, and match the column count")
    if not np.isfinite(variance).all() or np.any(variance <= 0.0):
        raise ValueError("residual variances must be finite and positive")
    weight_sqrt = 1.0 / np.sqrt(variance)
    weighted = (weight_sqrt[:, None] * values) * scale[None, :]
    return weighted, np.asarray(values, dtype=np.float64), weight_sqrt


def apply_pair_weight_sqrt(residual: object, blocks: Sequence[np.ndarray]) -> np.ndarray:
    values = np.asarray(residual, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != RESIDUAL_DIMENSION:
        raise ValueError(f"residual must have shape [pairs, {RESIDUAL_DIMENSION}]")
    if values.shape[0] != len(blocks):
        raise ValueError("residual pair count does not match weight blocks")
    out = np.empty(values.shape, dtype=np.float64)
    for index, block in enumerate(blocks):
        out[index] = np.asarray(block, dtype=np.float64) @ values[index]
    return out.reshape(-1)


def _sign_flip_vector(vector: np.ndarray) -> tuple[np.ndarray, int]:
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).any():
        return values, 1
    index = int(np.argmax(np.abs(values)))
    if values[index] < 0.0:
        return -values, -1
    return values, 1


def apply_mode_sign_convention(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Flip each mode so the largest-magnitude right-vector entry is positive."""
    u = np.array(left, dtype=np.float64, copy=True)
    v = np.array(right, dtype=np.float64, copy=True)
    if v.ndim != 2:
        raise ValueError("right singular matrix must be two-dimensional")
    signs: list[int] = []
    for column in range(v.shape[1]):
        flipped, sign = _sign_flip_vector(v[:, column])
        v[:, column] = flipped
        if u.shape[1] > column:
            u[:, column] = sign * u[:, column]
        signs.append(int(sign))
    return u, v, tuple(signs)


def identifiable_svd(
    weighted_matrix: object,
    *,
    parameter_names: Sequence[str],
    parameter_units: Sequence[str],
    parameter_scales: Sequence[float],
    rank_tolerance: float = FROZEN_RANK_TOLERANCE,
) -> IdentifiableSubspace:
    """SVD ``A = U Σ V^T`` with the frozen relative rank cut.

    ``weighted_matrix`` must already be ``W^{1/2} J S``.  Passing a naked
    mixed-unit ``J`` is refused.
    """
    if abs(float(rank_tolerance) - float(FROZEN_RANK_TOLERANCE)) > 0.0:
        # Config may repeat the frozen constant; any other value is a retune.
        if not math.isfinite(float(rank_tolerance)) or float(rank_tolerance) != float(FROZEN_RANK_TOLERANCE):
            raise RankThresholdRetuneError(
                f"rank_tolerance must remain the frozen value {FROZEN_RANK_TOLERANCE}"
            )
    names = tuple(str(name) for name in parameter_names)
    units = tuple(str(unit) for unit in parameter_units)
    scales = np.asarray(parameter_scales, dtype=np.float64)
    matrix = np.asarray(weighted_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("weighted matrix A must be two-dimensional")
    if len(names) != matrix.shape[1] or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("parameter_names must be unique, non-empty, and match A.columns")
    if len(units) != len(names) or scales.shape != (len(names),):
        raise ValueError("units and scales must match parameter_names")
    if not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise ValueError("parameter_scales must be finite and positive")
    if not np.isfinite(matrix).all():
        raise ValueError("weighted matrix A contains non-finite values")
    # ``matrix`` is already A = W^{1/2} J S, so mixed *native* units are expected
    # and have been removed by S.  Naked mixed-unit SVD is refused elsewhere.
    left, singular, right_t = np.linalg.svd(matrix, full_matrices=False)
    right = right_t.T
    left, right, signs = apply_mode_sign_convention(left, right)
    if singular.size == 0:
        rank = 0
    else:
        maximum = float(singular[0])
        tolerance = max(float(rank_tolerance) * maximum, np.finfo(np.float64).eps)
        rank = int(np.count_nonzero(singular > tolerance))
    v_id = right[:, :rank]
    v_null = right[:, rank:]
    projector = v_id @ v_id.T if rank else np.zeros((matrix.shape[1], matrix.shape[1]), dtype=np.float64)
    projector = 0.5 * (projector + projector.T)
    return IdentifiableSubspace(
        parameter_names=names,
        parameter_units=units,
        parameter_scales=np.asarray(scales, dtype=np.float64),
        rank_tolerance=float(rank_tolerance),
        singular_values=np.asarray(singular, dtype=np.float64),
        identifiable_rank=int(rank),
        u=np.asarray(left, dtype=np.float64),
        v=np.asarray(right, dtype=np.float64),
        v_id=np.asarray(v_id, dtype=np.float64),
        v_null=np.asarray(v_null, dtype=np.float64),
        projector_id=np.asarray(projector, dtype=np.float64),
        mode_signs=signs,
        n_observations=int(matrix.shape[0]),
        n_parameters=int(matrix.shape[1]),
    )


def native_to_scaled(values: object, scales: Sequence[float]) -> np.ndarray:
    native = np.asarray(values, dtype=np.float64)
    scale = np.asarray(scales, dtype=np.float64)
    return native / scale


def scaled_to_native(values: object, scales: Sequence[float]) -> np.ndarray:
    scaled = np.asarray(values, dtype=np.float64)
    scale = np.asarray(scales, dtype=np.float64)
    return scaled * scale


def inject_identifiable(subspace: IdentifiableSubspace, amplitudes: Sequence[float]) -> np.ndarray:
    """Native ``θ = S V_id a``.  Null coordinates stay at the min-norm zero."""
    amplitudes_v = np.asarray(amplitudes, dtype=np.float64).reshape(-1)
    if amplitudes_v.shape != (subspace.identifiable_rank,):
        raise ValueError("identifiable amplitudes must match identifiable_rank")
    return scaled_to_native(subspace.v_id @ amplitudes_v, subspace.parameter_scales)


def inject_null(subspace: IdentifiableSubspace, amplitudes: Sequence[float]) -> np.ndarray:
    amplitudes_v = np.asarray(amplitudes, dtype=np.float64).reshape(-1)
    if amplitudes_v.shape != (subspace.null_dimension,):
        raise ValueError("null amplitudes must match null_dimension")
    if subspace.null_dimension == 0:
        return np.zeros(subspace.n_parameters, dtype=np.float64)
    return scaled_to_native(subspace.v_null @ amplitudes_v, subspace.parameter_scales)


def project_native(subspace: IdentifiableSubspace, native: Sequence[float]) -> np.ndarray:
    """Minimum-norm identifiable representative ``θ̂ = S P_id S^{-1} θ``."""
    scaled = native_to_scaled(native, subspace.parameter_scales)
    return scaled_to_native(subspace.projector_id @ scaled, subspace.parameter_scales)


def solve_identifiable_amplitudes(
    subspace: IdentifiableSubspace,
    weighted_residual: Sequence[float],
    *,
    generating_weighted: object | None = None,
) -> dict[str, np.ndarray]:
    """Solve only mode amplitudes ``a`` and map ``Δθ = S V_id a``.

    If ``generating_weighted`` is omitted, the residual must match the
    definition matrix ``A`` used to build ``subspace``.  If it is supplied,
    the solver still uses the frozen ``V_id`` and least-squares-fits
    ``A_gen V_id a ≈ b``.  ``V_null^T u_hat = 0`` holds by construction as
    the minimum-norm / gauge representative, not as a measurement.
    """
    residual = np.asarray(weighted_residual, dtype=np.float64).reshape(-1)
    rank = subspace.identifiable_rank
    if rank == 0:
        zeros_a = np.zeros(0, dtype=np.float64)
        zeros_u = np.zeros(subspace.n_parameters, dtype=np.float64)
        return {
            "amplitudes": zeros_a,
            "scaled": zeros_u,
            "native": scaled_to_native(zeros_u, subspace.parameter_scales),
            "predicted_weighted_residual": np.zeros_like(residual),
        }
    if generating_weighted is None:
        if residual.shape != (subspace.n_observations,):
            raise ValueError("weighted residual length must match A.rows")
        u_id = subspace.u[:, :rank]
        singular = subspace.singular_values[:rank]
        amplitudes = (u_id.T @ residual) / singular
        predicted = (u_id * singular) @ amplitudes
    else:
        generating = np.asarray(generating_weighted, dtype=np.float64)
        if generating.ndim != 2 or generating.shape[1] != subspace.n_parameters:
            raise ValueError("generating weighted matrix must have one column per parameter")
        if residual.shape != (generating.shape[0],):
            raise ValueError("weighted residual length must match the generating matrix")
        design = generating @ subspace.v_id
        amplitudes, _residuals, _rank, _singular = np.linalg.lstsq(design, residual, rcond=None)
        predicted = design @ amplitudes
    scaled = subspace.v_id @ amplitudes
    return {
        "amplitudes": np.asarray(amplitudes, dtype=np.float64),
        "scaled": np.asarray(scaled, dtype=np.float64),
        "native": scaled_to_native(scaled, subspace.parameter_scales),
        "predicted_weighted_residual": np.asarray(predicted, dtype=np.float64),
    }


def principal_angles_rad(left: object, right: object) -> np.ndarray:
    """Principal angles between column-spans.  Sign of individual vectors is irrelevant."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.zeros(0, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    qa, _ = np.linalg.qr(a, mode="reduced")
    qb, _ = np.linalg.qr(b, mode="reduced")
    overlap = np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0.0, 1.0)
    return np.arccos(overlap)


def principal_angles_deg(left: object, right: object) -> np.ndarray:
    return np.degrees(principal_angles_rad(left, right))


def projector_frobenius_distance(left: object, right: object) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    return float(np.linalg.norm(a - b, ord="fro"))


def subspace_distance(left: IdentifiableSubspace, right: IdentifiableSubspace) -> dict[str, Any]:
    if left.parameter_names != right.parameter_names:
        raise ValueError("cannot compare subspaces with different parameter names")
    if left.identifiable_rank != right.identifiable_rank:
        angles = None
        max_angle = None
    elif left.identifiable_rank == 0:
        angles = []
        max_angle = 0.0
    else:
        angles_deg = principal_angles_deg(left.v_id, right.v_id)
        angles = [float(value) for value in angles_deg]
        max_angle = float(np.max(angles_deg)) if angles_deg.size else 0.0
    return {
        "left_rank": int(left.identifiable_rank),
        "right_rank": int(right.identifiable_rank),
        "same_rank": bool(left.identifiable_rank == right.identifiable_rank),
        "identifiable_principal_angles_deg": angles,
        "max_identifiable_principal_angle_deg": max_angle,
        "projector_frobenius_distance": projector_frobenius_distance(
            left.projector_id, right.projector_id
        ),
        "compares_subspace_not_signed_vector_elements": True,
    }


def mode_compositions(subspace: IdentifiableSubspace) -> dict[str, Any]:
    """Report each mode in scaled and native coefficients.

    Native coefficients are ``S v`` so they have mixed units; they are a
    bookkeeping expansion, not separately measurable mechanical DoF.
    """
    names = subspace.parameter_names
    scales = subspace.parameter_scales

    def _rows(basis: np.ndarray, singular: Sequence[float], kind: str) -> list[dict[str, Any]]:
        rows = []
        for index in range(basis.shape[1]):
            scaled = np.asarray(basis[:, index], dtype=np.float64)
            native = scaled * scales
            leading = int(np.argmax(np.abs(scaled))) if scaled.size else 0
            rows.append(
                {
                    "index": int(index),
                    "kind": kind,
                    "singular_value": float(singular[index]) if index < len(singular) else None,
                    "fraction_of_largest": (
                        None
                        if not subspace.singular_values.size or float(subspace.singular_values[0]) <= 0.0
                        else float(singular[index] / subspace.singular_values[0])
                    ),
                    "scaled_composition": {name: float(scaled[col]) for col, name in enumerate(names)},
                    "native_composition_mixed_units": {
                        name: float(native[col]) for col, name in enumerate(names)
                    },
                    "leading_scaled_parameter": names[leading] if names else None,
                    "reconstruction_observable_linear_combination": True,
                    "not_a_mechanical_parameter": True,
                    "may_not_be_relabeled_as_single_ry_or_C_dx": True,
                }
            )
        return rows

    identifiable_singular = list(subspace.singular_values[: subspace.identifiable_rank])
    null_singular = list(subspace.singular_values[subspace.identifiable_rank :])
    return {
        "parameter_names": list(names),
        "parameter_units": list(subspace.parameter_units),
        "parameter_scales": [float(value) for value in scales],
        "identifiable_modes": _rows(subspace.v_id, identifiable_singular, "identifiable"),
        "null_modes": _rows(subspace.v_null, null_singular, "null"),
        "modes_are_reconstruction_observable_linear_combinations": True,
        "null_zero_is_minimum_norm_gauge_not_a_measurement": True,
    }


def closure_metrics(
    subspace: IdentifiableSubspace,
    *,
    q_hat_native: Sequence[float],
    q_truth_native: Sequence[float],
    weighted_residual: Sequence[float] | None = None,
    predicted_weighted_residual: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Projected identifiable / null metrics.  Per-physical-parameter error is not a gate."""
    hat = native_to_scaled(q_hat_native, subspace.parameter_scales)
    truth = native_to_scaled(q_truth_native, subspace.parameter_scales)
    projected_error = subspace.v_id.T @ (hat - truth) if subspace.identifiable_rank else np.zeros(0)
    null_leakage = subspace.v_null.T @ hat if subspace.null_dimension else np.zeros(0)
    projected_truth = subspace.projector_id @ truth
    projector_residual = hat - projected_truth
    identifiable_truth_norm = float(np.linalg.norm(subspace.v_id.T @ truth)) if subspace.identifiable_rank else 0.0
    truth_scaled_norm = float(np.linalg.norm(truth))
    payload: dict[str, Any] = {
        "projected_identifiable_error_norm": float(np.linalg.norm(projected_error)),
        "projected_identifiable_error_relative_to_projected_truth": (
            None
            if identifiable_truth_norm <= 0.0
            else float(np.linalg.norm(projected_error) / identifiable_truth_norm)
        ),
        "null_leakage_norm": float(np.linalg.norm(null_leakage)),
        "null_component_of_hat_is_gauge_representative": True,
        "null_component_is_not_a_measurement_of_zero": True,
        "projector_closure_norm": float(np.linalg.norm(projector_residual)),
        "projector_closure_relative_to_projected_truth": (
            None
            if float(np.linalg.norm(projected_truth)) <= 0.0
            else float(np.linalg.norm(projector_residual) / float(np.linalg.norm(projected_truth)))
        ),
        "native_full_parameter_error_norm_not_a_gate": float(
            np.linalg.norm(np.asarray(q_hat_native, dtype=np.float64) - np.asarray(q_truth_native, dtype=np.float64))
        ),
        "scaled_truth_norm": truth_scaled_norm,
        "per_physical_parameter_truth_error_is_not_the_gate": True,
    }
    if weighted_residual is not None and predicted_weighted_residual is not None:
        observed = np.asarray(weighted_residual, dtype=np.float64)
        predicted = np.asarray(predicted_weighted_residual, dtype=np.float64)
        pre = float(np.linalg.norm(observed))
        post = float(np.linalg.norm(observed - predicted))
        payload["prefit_weighted_residual_norm"] = pre
        payload["postfit_weighted_residual_norm"] = post
        payload["postfit_observable_improved"] = bool(post < pre) if pre > 0.0 else post == 0.0
    return payload


def subspace_as_json(subspace: IdentifiableSubspace) -> dict[str, Any]:
    return {
        "parameter_names": list(subspace.parameter_names),
        "parameter_units": list(subspace.parameter_units),
        "parameter_scales": [float(value) for value in subspace.parameter_scales],
        "rank_tolerance": float(subspace.rank_tolerance),
        "rank_tolerance_frozen": True,
        "scales_not_retuned_from_singular_values": True,
        "singular_values": [float(value) for value in subspace.singular_values],
        "identifiable_rank": int(subspace.identifiable_rank),
        "null_dimension": int(subspace.null_dimension),
        "n_observations": int(subspace.n_observations),
        "n_parameters": int(subspace.n_parameters),
        "mode_signs": list(subspace.mode_signs),
        "sign_convention": "largest_abs_right_vector_entry_positive",
        "v_id_scaled": [[float(item) for item in row] for row in subspace.v_id.T],
        "v_null_scaled": [[float(item) for item in row] for row in subspace.v_null.T],
        "projector_id_scaled": [[float(item) for item in row] for row in subspace.projector_id],
        "modes_are_reconstruction_observable_linear_combinations": True,
        "not_mechanical_parameters": True,
        **mode_compositions(subspace),
    }


class ForcedRankError(ValueError):
    """Raised when a caller truncates or forces identifiable rank."""


class FuzzyJoinError(ValueError):
    """Raised when a caller requests a non-exact cluster join."""


def refuse_forced_identifiable_rank(*, native_rank: int, requested_rank: int | None) -> None:
    if requested_rank is None:
        return
    if int(requested_rank) != int(native_rank):
        raise ForcedRankError(
            f"refusing to force identifiable rank {requested_rank} onto a native rank-{native_rank} "
            "source projector; keep the frozen-rank SVD including rank-6 sources"
        )


def refuse_fuzzy_or_nearest_neighbour_join() -> None:
    raise FuzzyJoinError(
        "cluster-local join must be exact run+event+cluster_identifier; "
        "fuzzy or residual/position nearest-neighbour matching is forbidden"
    )


def consensus_operator(projectors: Sequence[object]) -> np.ndarray:
    """Equal-weight consensus ``M = (1/N) Σ_s P_s``.  Pair-count pooling is not used."""
    matrices = [np.asarray(item, dtype=np.float64) for item in projectors]
    if not matrices:
        raise ValueError("consensus operator requires at least one projector")
    shape = matrices[0].shape
    if matrices[0].ndim != 2 or shape[0] != shape[1]:
        raise ValueError("each source projector must be square")
    for index, matrix in enumerate(matrices):
        if matrix.shape != shape:
            raise ValueError(f"projector {index} has shape {matrix.shape}, expected {shape}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"projector {index} contains non-finite values")
        if not np.allclose(matrix, matrix.T, rtol=1.0e-8, atol=1.0e-10):
            raise ValueError(f"projector {index} is not symmetric")
    stacked = np.mean(np.stack(matrices, axis=0), axis=0)
    return 0.5 * (stacked + stacked.T)


def signed_eigh_descending(matrix: object) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Symmetric eigendecomposition, eigenvalues descending, frozen mode sign convention."""
    values_matrix = np.asarray(matrix, dtype=np.float64)
    if values_matrix.ndim != 2 or values_matrix.shape[0] != values_matrix.shape[1]:
        raise ValueError("consensus matrix must be square")
    symmetric = 0.5 * (values_matrix + values_matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.float64)
    dummy_left = np.array(eigenvectors, copy=True)
    dummy_left, eigenvectors, signs = apply_mode_sign_convention(dummy_left, eigenvectors)
    del dummy_left
    return eigenvalues, eigenvectors, signs


def quadratic_persistence(vector: object, projector: object) -> float:
    """``q^T P q`` in [0, 1] for an orthogonal projector ``P``."""
    q = np.asarray(vector, dtype=np.float64).reshape(-1)
    p = np.asarray(projector, dtype=np.float64)
    if q.size != p.shape[0] or p.shape[0] != p.shape[1]:
        raise ValueError("persistence vector must match the projector order")
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        return 0.0
    q = q / norm
    value = float(q @ p @ q)
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def source_support_fraction(persistences: Sequence[float], min_persistence: float) -> float:
    if not persistences:
        return 0.0
    hits = sum(1 for value in persistences if float(value) >= float(min_persistence))
    return float(hits) / float(len(persistences))


def select_stable_core(
    eigenvalues: object,
    eigenvectors: object,
    projectors: Sequence[object],
    *,
    min_consensus_eigenvalue: float,
    min_mode_persistence: float,
    min_source_support_fraction: float,
) -> dict[str, Any]:
    """Prefix rule: leading modes stay core until the first frozen-cut failure.

    Dimension is automatic.  It is not forced to 5 and does not cherry-pick
    non-consecutive eigenvectors.
    """
    values = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    vectors = np.asarray(eigenvectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != values.shape[0]:
        raise ValueError("eigenvectors must have one column per eigenvalue")
    labels: list[str] = []
    persistences: list[list[float]] = []
    supports: list[float] = []
    stopped = False
    core_dimension = 0
    for index in range(values.shape[0]):
        per_source = [
            quadratic_persistence(vectors[:, index], projector) for projector in projectors
        ]
        support = source_support_fraction(per_source, min_mode_persistence)
        persistences.append(per_source)
        supports.append(support)
        passes = (
            not stopped
            and float(values[index]) >= float(min_consensus_eigenvalue)
            and support >= float(min_source_support_fraction)
        )
        if passes:
            core_dimension += 1
            labels.append("core")
        else:
            stopped = True
            if float(values[index]) >= float(min_consensus_eigenvalue):
                labels.append("marginal")
            else:
                labels.append("source_specific_or_null")
    v_core = vectors[:, :core_dimension]
    v_orthogonal = vectors[:, core_dimension:]
    projector = (
        v_core @ v_core.T
        if core_dimension
        else np.zeros((vectors.shape[0], vectors.shape[0]), dtype=np.float64)
    )
    projector = 0.5 * (projector + projector.T)
    return {
        "core_dimension": int(core_dimension),
        "labels": labels,
        "mode_persistence_by_mode": persistences,
        "source_support_fraction_by_mode": supports,
        "v_core": np.asarray(v_core, dtype=np.float64),
        "v_orthogonal": np.asarray(v_orthogonal, dtype=np.float64),
        "projector_core": np.asarray(projector, dtype=np.float64),
        "dimension_determined_by_eigenvalue_and_persistence": True,
        "forced_core_dimension": None,
        "prefix_rule": "leading_modes_until_first_frozen_cut_failure",
        "not_forced_to_rank_five": True,
    }


def core_contained_in_projector(
    v_core: object,
    projector: object,
) -> dict[str, Any]:
    """How much of ``V_core`` sits inside a (possibly higher-rank) source projector."""
    basis = np.asarray(v_core, dtype=np.float64)
    p = np.asarray(projector, dtype=np.float64)
    if basis.size == 0:
        return {
            "core_dimension": 0,
            "mode_persistence": [],
            "min_mode_persistence": None,
            "principal_angles_deg": [],
            "max_principal_angle_deg": 0.0,
            "missing_projector_frobenius": 0.0,
            "contained": True,
        }
    if basis.ndim == 1:
        basis = basis.reshape(-1, 1)
    persistences = [quadratic_persistence(basis[:, index], p) for index in range(basis.shape[1])]
    projected = p @ basis
    column_norms = np.linalg.norm(projected, axis=0)
    kept = column_norms > 1.0e-12
    core_projector = basis @ basis.T
    if not bool(np.any(kept)):
        angles = [90.0] * int(basis.shape[1])
        missing = projector_frobenius_distance(core_projector, np.zeros_like(p))
    else:
        kept_basis = projected[:, kept]
        angles = [float(value) for value in principal_angles_deg(basis, kept_basis)]
        while len(angles) < basis.shape[1]:
            angles.append(90.0)
        overlap_projector = p @ core_projector @ p
        overlap_projector = 0.5 * (overlap_projector + overlap_projector.T)
        missing = projector_frobenius_distance(core_projector, overlap_projector)
    max_angle = float(max(angles)) if angles else 0.0
    return {
        "core_dimension": int(basis.shape[1]),
        "mode_persistence": persistences,
        "min_mode_persistence": float(min(persistences)) if persistences else None,
        "principal_angles_deg": angles,
        "max_principal_angle_deg": max_angle,
        "missing_projector_frobenius": float(missing),
        "compares_subspace_not_signed_vector_elements": True,
    }


def identifiable_subspace_from_core(
    *,
    parameter_names: Sequence[str],
    parameter_units: Sequence[str],
    parameter_scales: Sequence[float],
    v_core: object,
    v_orthogonal: object,
    rank_tolerance: float = FROZEN_RANK_TOLERANCE,
) -> IdentifiableSubspace:
    """Wrap a frozen core as an identifiable subspace for the restricted solver.

    Singular values are placeholders.  The solver must use
    ``generating_weighted`` and ``V_core``, never workbook-68 pooled ``V_id``.
    """
    names = tuple(str(name) for name in parameter_names)
    units = tuple(str(unit) for unit in parameter_units)
    scales = np.asarray(parameter_scales, dtype=np.float64)
    core = np.asarray(v_core, dtype=np.float64)
    orthogonal = np.asarray(v_orthogonal, dtype=np.float64)
    if core.size == 0:
        core = np.zeros((len(names), 0), dtype=np.float64)
    if orthogonal.size == 0:
        orthogonal = np.zeros((len(names), 0), dtype=np.float64)
    if core.ndim == 1:
        core = core.reshape(-1, 1)
    if orthogonal.ndim == 1:
        orthogonal = orthogonal.reshape(-1, 1)
    n_parameters = len(names)
    if core.shape[0] != n_parameters or orthogonal.shape[0] != n_parameters:
        raise ValueError("core / orthogonal bases must have one row per parameter")
    rank = int(core.shape[1])
    vectors = np.concatenate([core, orthogonal], axis=1) if orthogonal.shape[1] else np.array(core, copy=True)
    if vectors.shape[1] != n_parameters:
        # Pad if numerical rank dropped.
        rest, _ = np.linalg.qr(np.eye(n_parameters) - vectors @ vectors.T)
        need = n_parameters - vectors.shape[1]
        vectors = np.concatenate([vectors, rest[:, :need]], axis=1)
    singular = np.concatenate(
        [np.ones(rank, dtype=np.float64), np.zeros(n_parameters - rank, dtype=np.float64)]
    )
    projector = core @ core.T if rank else np.zeros((n_parameters, n_parameters), dtype=np.float64)
    projector = 0.5 * (projector + projector.T)
    dummy_u = np.eye(n_parameters, dtype=np.float64)
    signs = tuple(1 for _ in range(n_parameters))
    return IdentifiableSubspace(
        parameter_names=names,
        parameter_units=units,
        parameter_scales=scales,
        rank_tolerance=float(rank_tolerance),
        singular_values=singular,
        identifiable_rank=rank,
        u=dummy_u,
        v=np.asarray(vectors, dtype=np.float64),
        v_id=np.asarray(core, dtype=np.float64),
        v_null=np.asarray(orthogonal, dtype=np.float64),
        projector_id=np.asarray(projector, dtype=np.float64),
        mode_signs=signs,
        n_observations=0,
        n_parameters=n_parameters,
    )

