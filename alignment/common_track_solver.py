"""Workbook-82 common-track alignment solver.

One local nuisance per truth track, one joint residual/covariance per track,
Schur elimination onto global station poses.  Pairwise edge chi² sums are
diagnostics, not this likelihood.  Association is truth-only in WB82 phase 1.

This module does not retrain or reread W64.  The frozen association default
remains ``frozen_W64_raw_energy_plus_exact_solver`` and is not used here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from alignment.common_track_geometry import (
    SOLVER_CONTRACT,
    UPDATE_LEFT_SE3,
    composed_station_matrix,
    left_update_payload,
    pose_error_lie,
    require_update_convention,
    rotate_vector,
    se3_log,
)
from alignment.four_station import (
    ALL_COMPONENTS,
    FREE_COMPONENTS,
    STATION_IDS,
    SURVEY_DZ_PRIOR_SIGMA_MM,
    invert_six_vector,
    parameter_name,
    require_station_id,
    six_vector_to_matrix,
)


ASSOCIATION_DEFAULT_SYSTEM = "frozen_W64_raw_energy_plus_exact_solver"
MEASUREMENT_NAMES = ("x_mm", "y_mm", "tx", "ty")
NUISANCE_NAMES = ("x_mm", "y_mm", "tx", "ty", "q_over_p")
SURVEY_FIXED_DZ = "fixed_dz"
SURVEY_FINITE_PRIOR = "finite_survey_prior"
MEASUREMENT_DIM = 4
DEFAULT_DAMPING = 1.0e-8
DEFAULT_BEND_SCALE = 2.0e-5
QP_INDEX = 4
QP_PRIOR_SIGMA = 1.0e-3
MOMENTUM_FREE = "q_over_p_free"
MOMENTUM_PRIOR = "q_over_p_prior"
MAX_ITERATIONS = 10
SCALED_UPDATE_TOL = 1.0e-3
VALIDATION_REL_TOL = 1.0e-3
PARAMETER_SCALES = {
    "dx_mm": 5.0,
    "dy_mm": 5.0,
    "dz_mm": 5.0,
    "rx_mrad": 60.0,
    "ry_mrad": 60.0,
    "rz_mrad": 60.0,
}
FORBIDDEN_PATH_NEEDLES = (
    "00800_00849",
    "mc24_100116",
    "mc24_100117",
    "00350_00399",
    "source_diversity_blind",
)


class AlignmentSolverError(ValueError):
    """Fail-closed numerical or contract violation."""


def refuse_wb82_path(path: object) -> None:
    text = str(path)
    for needle in FORBIDDEN_PATH_NEEDLES:
        if needle in text:
            raise AlignmentSolverError(f"WB82 refuses development / blind / sealed path: {text}")


def require_survey_mode(mode: str) -> str:
    text = str(mode)
    if text not in (SURVEY_FIXED_DZ, SURVEY_FINITE_PRIOR):
        raise AlignmentSolverError(f"survey mode must be {SURVEY_FIXED_DZ} or {SURVEY_FINITE_PRIOR}")
    return text


def require_finite(array: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.size and not np.isfinite(values).all():
        raise AlignmentSolverError(f"{label} contains NaN/Inf")
    return values


def require_spd(matrix: np.ndarray, label: str) -> np.ndarray:
    values = require_finite(matrix, label)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise AlignmentSolverError(f"{label} must be square")
    if not np.allclose(values, values.T, rtol=1.0e-8, atol=1.0e-12):
        raise AlignmentSolverError(f"{label} is not symmetric")
    try:
        np.linalg.cholesky(values)
    except np.linalg.LinAlgError as error:
        raise AlignmentSolverError(f"{label} is not SPD") from error
    return values


def inverse_spd(matrix: np.ndarray, label: str) -> np.ndarray:
    values = require_spd(matrix, label)
    factor = np.linalg.cholesky(values)
    identity = np.eye(values.shape[0], dtype=np.float64)
    return np.linalg.solve(factor.T, np.linalg.solve(factor, identity))


def symmetrize_normal(raw: np.ndarray, label: str = "normal") -> tuple[np.ndarray, dict[str, float]]:
    """``H = 0.5 (H_raw + H_raw.T)``.  Records antisymmetry; does not add ridge."""
    values = require_finite(np.asarray(raw, dtype=np.float64), label)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise AlignmentSolverError(f"{label} must be square")
    antisym = 0.5 * (values - values.T)
    antisymmetry_norm = float(np.linalg.norm(antisym, ord="fro"))
    raw_norm = float(np.linalg.norm(values, ord="fro"))
    relative_antisymmetry = antisymmetry_norm / raw_norm if raw_norm > 0.0 else 0.0
    symmetric = 0.5 * (values + values.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    minimum_eigenvalue = float(eigenvalues[0])
    maximum_eigenvalue = float(eigenvalues[-1])
    if minimum_eigenvalue > 0.0:
        condition_number = maximum_eigenvalue / minimum_eigenvalue
    else:
        condition_number = float("inf")
    return symmetric, {
        "antisymmetry_norm": antisymmetry_norm,
        "relative_antisymmetry": relative_antisymmetry,
        "minimum_eigenvalue": minimum_eigenvalue,
        "condition_number": condition_number,
    }


def inverse_spd_after_symmetrize(raw: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Symmetrize, then a strict Cholesky test.  No ridge / jitter / loosened atol."""
    symmetric, diagnostics = symmetrize_normal(raw, label)
    if diagnostics["minimum_eigenvalue"] <= 0.0:
        raise AlignmentSolverError(
            f"{label} is not SPD after symmetrization "
            f"(minimum_eigenvalue={diagnostics['minimum_eigenvalue']})"
        )
    inverse = inverse_spd(symmetric, label)
    return inverse, symmetric, diagnostics


def absolute_survey_prior_terms(
    a_current: float,
    a_survey: float,
    *,
    sigma: float,
) -> dict[str, float]:
    """Absolute-state MAP prior on ``a_current + theta - a_survey``.

    Objective: ``0.5 w (a_current + theta - a_survey)^2``.
    Hessian wrt ``theta`` is ``w``.  At ``theta = 0`` the gradient is
    ``w (a_current - a_survey)``.  When ``a_current == a_survey`` the
    prior gradient and Newton rhs are identically zero.
    """
    if not math_isfinite(float(sigma)) or float(sigma) <= 0.0:
        raise AlignmentSolverError("survey prior sigma must be a finite positive value")
    weight = 1.0 / (float(sigma) ** 2)
    residual = float(a_current) - float(a_survey)
    gradient = weight * residual
    return {
        "hessian": weight,
        "prior_residual": residual,
        "gradient_at_linearization": gradient,
        "rhs": -gradient,
        "chi2_at_linearization": weight * residual * residual,
    }


def solver_implementation_contract() -> dict[str, bool]:
    """WB83-v2 implementation fixes.  Independent of replica pull outcomes."""
    import inspect

    schur_source = inspect.getsource(schur_reduce)
    factor_source = inspect.getsource(inverse_spd_after_symmetrize)
    prior_source = inspect.getsource(absolute_survey_prior_terms)
    solve_source = inspect.getsource(solve_common_track)
    return {
        "jt_w_j_numerical_symmetrization": "inverse_spd_after_symmetrize" in schur_source
        and "0.5" in inspect.getsource(symmetrize_normal),
        "strict_spd_after_symmetrize": "minimum_eigenvalue" in factor_source
        and "1.0e-12 * np.eye" not in factor_source
        and "1e-12 * np.eye" not in factor_source,
        "no_ridge_on_local_normal": "1.0e-12 * np.eye" not in schur_source
        and "1e-12 * np.eye" not in schur_source,
        "absolute_state_survey_prior": "absolute_survey_prior_terms" in solve_source
        and "a_current + theta - a_survey" in prior_source,
    }


def solver_implementation_fixed() -> bool:
    return all(solver_implementation_contract().values())


TRANSLATION_COMPONENTS: tuple[str, ...] = ("dx_mm", "dy_mm")
PHASE1_SMOKE_COMPONENTS: tuple[str, ...] = TRANSLATION_COMPONENTS
PHASE1_SMOKE_STATIONS: tuple[int, ...] = (3,)


def parameter_chart(
    *,
    survey_mode: str,
    reference_station: int = 0,
    components: Sequence[str] | None = None,
    stations: Sequence[int] | None = None,
) -> tuple[str, ...]:
    """Newton coordinates.  ``fixed_dz`` drops dz; finite prior keeps it."""
    require_station_id(reference_station)
    mode = require_survey_mode(survey_mode)
    if components is None:
        chosen = ALL_COMPONENTS if mode == SURVEY_FINITE_PRIOR else FREE_COMPONENTS
    else:
        chosen = tuple(str(item) for item in components)
        if mode == SURVEY_FIXED_DZ:
            chosen = tuple(item for item in chosen if item != "dz_mm")
    if stations is None:
        movable = tuple(station for station in STATION_IDS if station != int(reference_station))
    else:
        movable = tuple(require_station_id(int(station)) for station in stations)
        if int(reference_station) in movable:
            raise AlignmentSolverError("reference station cannot appear in the free-station list")
    names: list[str] = []
    for station in movable:
        for component in chosen:
            names.append(parameter_name(station, component))
    return tuple(names)


@dataclass(frozen=True)
class StationHit:
    """One physical station measurement.  ``measurement_id`` is the tracklet identity."""

    track_id: int
    station: int
    measurement_id: str
    observed: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed", require_finite(np.asarray(self.observed, dtype=np.float64), "observed"))
        if self.observed.shape != (MEASUREMENT_DIM,):
            raise AlignmentSolverError("station hit must be a 4-vector (x, y, tx, ty)")
        object.__setattr__(self, "covariance", require_spd(np.asarray(self.covariance, dtype=np.float64), "hit covariance"))
        if self.covariance.shape != (MEASUREMENT_DIM, MEASUREMENT_DIM):
            raise AlignmentSolverError("station hit covariance must be 4x4")
        require_station_id(self.station)


@dataclass(frozen=True)
class LinearTrackBlock:
    """One track after unique-hit assembly.  Nuisance then global columns."""

    track_id: int
    residual: np.ndarray
    jacobian_nuisance: np.ndarray
    jacobian_global: np.ndarray
    weight: np.ndarray
    measurement_ids: tuple[str, ...]
    qp_prior_sigma: float | None = None
    qp_prior_mean: float | None = None
    qp_at_linearization: float | None = None


@dataclass(frozen=True)
class CommonTrackSolution:
    contract: str
    status: str
    update: np.ndarray
    covariance: np.ndarray
    parameter_names: tuple[str, ...]
    chi2: float | None
    ndof: int
    condition_spectrum: np.ndarray
    local_nuisances: dict[int, np.ndarray] = field(default_factory=dict)
    message: str = ""
    n_retained: int = 0
    n_dropped: int = 0
    spd_diagnostics: tuple[dict[str, float], ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def pairwise_independent_chi2(hits: Sequence[StationHit]) -> float:
    """Diagnostic only.  Treats every hit as an independent edge residual at zero prediction."""
    total = 0.0
    for hit in hits:
        weight = inverse_spd(hit.covariance, "pairwise covariance")
        residual = require_finite(hit.observed, "pairwise residual")
        total += float(residual @ weight @ residual)
    return total


def assemble_unique_hits(hits: Sequence[StationHit]) -> tuple[StationHit, ...]:
    """Each ``measurement_id`` appears once.  Shared tracklets are not independent edges."""
    unique: dict[str, StationHit] = {}
    for hit in hits:
        key = str(hit.measurement_id)
        if key in unique:
            previous = unique[key]
            if previous.station != hit.station or not np.allclose(previous.observed, hit.observed):
                raise AlignmentSolverError(f"measurement_id {key} is reused with conflicting data")
            if not np.allclose(previous.covariance, hit.covariance):
                raise AlignmentSolverError(f"measurement_id {key} is reused with a different covariance")
            continue
        unique[key] = hit
    if not unique:
        raise AlignmentSolverError("track has no unique station measurements")
    return tuple(unique.values())


def assemble_track_block(
    hits: Sequence[StationHit],
    jacobian_nuisance: np.ndarray,
    jacobian_global: np.ndarray,
    residual: np.ndarray,
    *,
    qp_prior_sigma: float | None = None,
    qp_prior_mean: float | None = None,
    qp_at_linearization: float | None = None,
) -> LinearTrackBlock:
    unique = assemble_unique_hits(hits)
    n_meas = MEASUREMENT_DIM * len(unique)
    residual = require_finite(residual, "track residual")
    jacobian_nuisance = require_finite(jacobian_nuisance, "nuisance Jacobian")
    jacobian_global = require_finite(jacobian_global, "global Jacobian")
    if residual.shape != (n_meas,):
        raise AlignmentSolverError("track residual does not match unique-hit dimension")
    if jacobian_nuisance.shape[0] != n_meas or jacobian_global.shape[0] != n_meas:
        raise AlignmentSolverError("track Jacobians do not match unique-hit dimension")
    blocks = [hit.covariance for hit in unique]
    covariance = np.zeros((n_meas, n_meas), dtype=np.float64)
    for index, block in enumerate(blocks):
        start = index * MEASUREMENT_DIM
        covariance[start : start + MEASUREMENT_DIM, start : start + MEASUREMENT_DIM] = block
    weight = inverse_spd(covariance, "track joint covariance")
    return LinearTrackBlock(
        track_id=int(unique[0].track_id),
        residual=residual,
        jacobian_nuisance=jacobian_nuisance,
        jacobian_global=jacobian_global,
        weight=weight,
        measurement_ids=tuple(hit.measurement_id for hit in unique),
        qp_prior_sigma=qp_prior_sigma,
        qp_prior_mean=qp_prior_mean,
        qp_at_linearization=qp_at_linearization,
    )


def _apply_qp_prior(normal_xi: np.ndarray, rhs_xi: np.ndarray, block: LinearTrackBlock) -> tuple[np.ndarray, np.ndarray]:
    """Stiffen local q/p around a beam / truth mean.  q/p stays local, never global."""
    if block.qp_prior_sigma is None:
        return normal_xi, rhs_xi
    if normal_xi.shape[0] <= QP_INDEX:
        raise AlignmentSolverError("q/p prior requires a 5-component local nuisance")
    sigma = float(block.qp_prior_sigma)
    if sigma <= 0.0 or not math_isfinite(sigma):
        raise AlignmentSolverError("q/p prior sigma must be a finite positive value")
    mean = block.qp_at_linearization if block.qp_prior_mean is None else block.qp_prior_mean
    current = block.qp_at_linearization
    if mean is None or current is None:
        raise AlignmentSolverError("q/p prior requires a mean and the linearization value")
    weight = 1.0 / (sigma * sigma)
    normal_xi = np.array(normal_xi, dtype=np.float64, copy=True)
    rhs_xi = np.array(rhs_xi, dtype=np.float64, copy=True)
    normal_xi[QP_INDEX, QP_INDEX] += weight
    rhs_xi[QP_INDEX] += weight * (float(mean) - float(current))
    return normal_xi, rhs_xi


def schur_reduce(
    block: LinearTrackBlock,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Return ``(C_red, b_red, A, A_inv, B, spd_diagnostics)`` for one track."""
    weight = block.weight
    h_xi = block.jacobian_nuisance
    h_th = block.jacobian_global
    residual = block.residual
    raw_xi = h_xi.T @ weight @ h_xi
    _symmetric_xi, antisym = symmetrize_normal(raw_xi, "local J^T W J")
    cross = h_xi.T @ weight @ h_th
    normal_th = h_th.T @ weight @ h_th
    rhs_xi = h_xi.T @ weight @ residual
    rhs_th = h_th.T @ weight @ residual
    normal_xi, rhs_xi = _apply_qp_prior(_symmetric_xi, rhs_xi, block)
    try:
        xi_inv, normal_xi, factored = inverse_spd_after_symmetrize(normal_xi, "local nuisance normal")
    except AlignmentSolverError as error:
        raise AlignmentSolverError("local nuisance system is singular / not SPD") from error
    diagnostics = {
        "antisymmetry_norm": antisym["antisymmetry_norm"],
        "relative_antisymmetry": antisym["relative_antisymmetry"],
        "minimum_eigenvalue": factored["minimum_eigenvalue"],
        "condition_number": factored["condition_number"],
    }
    reduced = normal_th - cross.T @ xi_inv @ cross
    reduced_rhs = rhs_th - cross.T @ xi_inv @ rhs_xi
    return reduced, reduced_rhs, normal_xi, xi_inv, cross, diagnostics


def _fail(status: str, names: Sequence[str], message: str) -> CommonTrackSolution:
    n_par = len(names)
    return CommonTrackSolution(
        contract=SOLVER_CONTRACT,
        status=status,
        update=np.full(n_par, np.nan, dtype=np.float64),
        covariance=np.full((n_par, n_par), np.nan, dtype=np.float64),
        parameter_names=tuple(names),
        chi2=None,
        ndof=0,
        condition_spectrum=np.asarray([], dtype=np.float64),
        message=message,
    )


def solve_common_track(
    blocks: Sequence[LinearTrackBlock],
    parameter_names: Sequence[str],
    *,
    survey_mode: str = SURVEY_FIXED_DZ,
    survey_sigma_mm: float = SURVEY_DZ_PRIOR_SIGMA_MM,
    damping: float = DEFAULT_DAMPING,
    current_update: np.ndarray | None = None,
    current_payloads: Mapping[int, Sequence[float]] | None = None,
) -> CommonTrackSolution:
    """One linearized GLS step.  MILP/association is not involved."""
    names = tuple(str(name) for name in parameter_names)
    if not names or len(set(names)) != len(names):
        return _fail("contract", names, "parameter names must be unique and non-empty")
    if not blocks:
        return _fail("contract", names, "no tracks")
    n_par = len(names)
    mode = require_survey_mode(survey_mode)
    if float(damping) < 0.0 or not math_isfinite(damping):
        return _fail("non_finite", names, "damping must be a finite non-negative scalar")
    reduced = np.zeros((n_par, n_par), dtype=np.float64)
    rhs = np.zeros(n_par, dtype=np.float64)
    chi2 = 0.0
    n_meas = 0
    n_nuisance = 0
    locals_a: list[tuple[LinearTrackBlock, np.ndarray, np.ndarray]] = []
    spd_diagnostics: list[dict[str, float]] = []
    try:
        for block in blocks:
            if block.jacobian_global.shape[1] != n_par:
                raise AlignmentSolverError("global Jacobian width does not match the parameter chart")
            piece, piece_rhs, _normal_xi, xi_inv, cross, diagnostics = schur_reduce(block)
            reduced += piece
            rhs += piece_rhs
            chi2 += float(block.residual @ block.weight @ block.residual)
            n_meas += int(block.residual.size)
            n_nuisance += int(block.jacobian_nuisance.shape[1])
            locals_a.append((block, xi_inv, cross))
            spd_diagnostics.append(diagnostics)
        if mode == SURVEY_FINITE_PRIOR:
            if not math_isfinite(survey_sigma_mm) or float(survey_sigma_mm) <= 0.0:
                raise AlignmentSolverError("finite survey prior requires a positive sigma")
            if current_update is not None:
                require_finite(current_update, "current update")
            for index, name in enumerate(names):
                if not name.endswith("_dz_mm"):
                    continue
                a_current = 0.0
                if current_payloads is not None:
                    station, _component = _split_parameter_name(name)
                    a_current = float(se3_log(six_vector_to_matrix(current_payloads[station]))[2])
                terms = absolute_survey_prior_terms(a_current, 0.0, sigma=float(survey_sigma_mm))
                reduced[index, index] += terms["hessian"]
                rhs[index] += terms["rhs"]
                chi2 += terms["chi2_at_linearization"]
        reduced = require_finite(reduced, "reduced data normal")
        scales = np.asarray([_parameter_scale(name) for name in names], dtype=np.float64)
        scaled = reduced * np.outer(scales, scales)
        if float(np.linalg.norm(scaled)) == 0.0:
            raise AlignmentSolverError("reduced normal matrix is singular / ill-conditioned")
        data_spectrum = np.linalg.svd(scaled, compute_uv=False)
        if float(data_spectrum[0]) <= 1.0e-14:
            raise AlignmentSolverError("reduced normal matrix is singular / ill-conditioned")
        rhs = require_finite(rhs, "reduced rhs")
        left, spectrum, right = np.linalg.svd(scaled + float(damping) * np.eye(n_par), full_matrices=False)
        keep = spectrum > 1.0e-12 * float(data_spectrum[0])
        if not np.any(keep):
            raise AlignmentSolverError("reduced normal matrix is singular / ill-conditioned")
        rhs_u = scales * rhs
        update_u = (right[keep].T * (1.0 / spectrum[keep])) @ (left[:, keep].T @ rhs_u)
        update = require_finite(scales * update_u, "global update")
        cov_u = (right[keep].T / spectrum[keep]) @ left[:, keep].T
        covariance = require_finite((scales[:, None] * cov_u) * scales[None, :], "scaled covariance")
        n_retained = int(np.count_nonzero(keep))
        n_dropped = int(n_par - n_retained)
        nuisances = {}
        for block, xi_inv, cross in locals_a:
            local_rhs = block.jacobian_nuisance.T @ block.weight @ block.residual
            nuisances[int(block.track_id)] = require_finite(xi_inv @ (local_rhs - cross @ update), "local nuisance")
    except AlignmentSolverError as error:
        status = "non_spd" if "SPD" in str(error) or "not SPD" in str(error) else "singular"
        if "NaN" in str(error) or "Inf" in str(error) or "non-finite" in str(error) or "non_finite" in str(error):
            status = "non_finite"
        return _fail(status, names, str(error))
    ndof = int(n_meas - n_nuisance - n_par)
    if mode == SURVEY_FINITE_PRIOR:
        ndof += sum(1 for name in names if name.endswith("_dz_mm"))
    return CommonTrackSolution(
        contract=SOLVER_CONTRACT,
        status="ok",
        update=update,
        covariance=covariance,
        parameter_names=names,
        chi2=float(chi2),
        ndof=max(ndof, 0),
        condition_spectrum=spectrum,
        local_nuisances=nuisances,
        n_retained=n_retained,
        n_dropped=n_dropped,
        spd_diagnostics=tuple(spd_diagnostics),
    )


def math_isfinite(value: float) -> bool:
    return bool(np.isfinite(value))


def solve_joint_dense(blocks: Sequence[LinearTrackBlock]) -> tuple[np.ndarray, np.ndarray]:
    """Direct stacked GLS.  Test oracle for Schur, not the production path."""
    residuals = []
    jacobians = []
    weights = []
    n_xi = 0
    n_th = blocks[0].jacobian_global.shape[1]
    offsets = []
    for block in blocks:
        offsets.append(n_xi)
        n_xi += block.jacobian_nuisance.shape[1]
    n_par = n_xi + n_th
    for block, offset in zip(blocks, offsets):
        residuals.append(block.residual)
        wide = np.zeros((block.residual.size, n_par), dtype=np.float64)
        n_local = block.jacobian_nuisance.shape[1]
        wide[:, offset : offset + n_local] = block.jacobian_nuisance
        wide[:, n_xi:] = block.jacobian_global
        jacobians.append(wide)
        weights.append(block.weight)
    residual = np.concatenate(residuals)
    jacobian = np.vstack(jacobians)
    weight = np.zeros((residual.size, residual.size), dtype=np.float64)
    cursor = 0
    for block, piece in zip(blocks, weights):
        size = block.residual.size
        weight[cursor : cursor + size, cursor : cursor + size] = piece
        cursor += size
    normal = jacobian.T @ weight @ jacobian
    rhs = jacobian.T @ weight @ residual
    covariance = inverse_spd(normal, "joint normal")
    return covariance @ rhs, covariance


def lab_transport(
    state: Sequence[float],
    z_from_mm: float,
    z_to_mm: float,
    *,
    field_y: float,
    bend_scale: float = DEFAULT_BEND_SCALE,
) -> np.ndarray:
    """Uniform-By lab transport.  ``state`` is ``(x, y, tx, ty, q/p)``."""
    values = require_finite(np.asarray(state, dtype=np.float64), "track state")
    if values.shape != (5,):
        raise AlignmentSolverError("track state must be (x, y, tx, ty, q/p)")
    delta = float(z_to_mm) - float(z_from_mm)
    kappa = float(bend_scale) * float(values[4]) * float(field_y)
    transported = np.array(
        [
            values[0] + values[2] * delta + 0.5 * kappa * delta * delta,
            values[1] + values[3] * delta,
            values[2] + kappa * delta,
            values[3],
            values[4],
        ],
        dtype=np.float64,
    )
    return require_finite(transported, "transported state")


def predict_local_measurement(
    state_at_ref: Sequence[float],
    payload: Sequence[float],
    *,
    z_ref_mm: float,
    z_station_mm: float,
    field_y: float,
    bend_scale: float = DEFAULT_BEND_SCALE,
) -> np.ndarray:
    """Predict ``(x, y, tx, ty)`` in the aligned station local frame."""
    pose = composed_station_matrix(payload, z_station_mm)
    origin = pose @ np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    lab = lab_transport(state_at_ref, z_ref_mm, float(origin[2]), field_y=field_y, bend_scale=bend_scale)
    inverse = np.linalg.inv(pose)
    point = inverse @ np.asarray([lab[0], lab[1], float(origin[2]), 1.0], dtype=np.float64)
    direction = rotate_vector(inverse, (lab[2], lab[3], 1.0))
    if abs(float(direction[2])) < 1.0e-12:
        raise AlignmentSolverError("local direction has vanishing z-component")
    tx = float(direction[0] / direction[2])
    ty = float(direction[1] / direction[2])
    return require_finite(np.asarray([point[0], point[1], tx, ty], dtype=np.float64), "local prediction")


def finite_difference_jacobian(
    function,
    values: Sequence[float],
    steps: Sequence[float],
) -> np.ndarray:
    base = require_finite(np.asarray(values, dtype=np.float64), "fd point")
    step = require_finite(np.asarray(steps, dtype=np.float64), "fd step")
    if base.shape != step.shape or base.ndim != 1:
        raise AlignmentSolverError("finite-difference point and steps must be aligned vectors")
    reference = np.asarray(function(base), dtype=np.float64)
    columns = []
    for index, increment in enumerate(step):
        if abs(float(increment)) < 1.0e-18:
            columns.append(np.zeros(reference.shape, dtype=np.float64))
            continue
        shifted = base.copy()
        shifted[index] += float(increment)
        columns.append((np.asarray(function(shifted), dtype=np.float64) - reference) / float(increment))
    return require_finite(np.stack(columns, axis=1), "finite-difference Jacobian")


def fd_column_stability(function, values: Sequence[float], step: float, *, index: int) -> dict[str, float]:
    """Relative change of one Jacobian column across ``h/2, h, 2h``."""
    base = require_finite(np.asarray(values, dtype=np.float64), "fd stability point")
    if not math_isfinite(step) or abs(float(step)) < 1.0e-18:
        raise AlignmentSolverError("fd stability step must be a finite nonzero value")

    def column(width: float) -> np.ndarray:
        steps = np.zeros(base.size, dtype=np.float64)
        steps[int(index)] = float(width)
        return finite_difference_jacobian(function, base, steps)[:, int(index)]

    half = column(0.5 * float(step))
    center = column(float(step))
    double = column(2.0 * float(step))
    denom = max(float(np.linalg.norm(center)), 1.0e-12)
    return {
        "rel_h_over_2": float(np.linalg.norm(half - center) / denom),
        "rel_2h": float(np.linalg.norm(double - center) / denom),
    }


def _twist_for_parameter(component: str, value: float) -> np.ndarray:
    twist = np.zeros(6, dtype=np.float64)
    if component in ("dx_mm", "dy_mm", "dz_mm"):
        twist[{"dx_mm": 0, "dy_mm": 1, "dz_mm": 2}[component]] = float(value)
    elif component in ("rx_mrad", "ry_mrad", "rz_mrad"):
        twist[{"rx_mrad": 3, "ry_mrad": 4, "rz_mrad": 5}[component]] = float(value) / 1000.0
    else:
        raise AlignmentSolverError(f"unsupported pose component {component}")
    return twist


def linearize_track(
    hits: Sequence[StationHit],
    state: Sequence[float],
    payloads: Mapping[int, Sequence[float]],
    parameter_names: Sequence[str],
    *,
    z_mm: Mapping[int, float],
    field_y: float,
    nuisance_steps: Sequence[float] | None = None,
    pose_step_mm: float = 1.0e-2,
    pose_step_mrad: float = 1.0e-2,
    momentum_mode: str = MOMENTUM_FREE,
    qp_prior_sigma: float = QP_PRIOR_SIGMA,
    qp_prior_mean: float | None = None,
) -> LinearTrackBlock:
    unique = assemble_unique_hits(hits)
    if 0 not in z_mm:
        raise AlignmentSolverError("z_mm must include the reference station 0")
    z_ref = float(z_mm[0])
    state = require_finite(np.asarray(state, dtype=np.float64), "linearization state")
    xi_steps = np.asarray((1.0e-3, 1.0e-3, 1.0e-6, 1.0e-6, 1.0e-5) if nuisance_steps is None else nuisance_steps, dtype=np.float64)
    packed = {int(station): tuple(float(value) for value in payloads[int(station)]) for station in payloads}

    def predict_stack(local_state: np.ndarray, poses: Mapping[int, Sequence[float]] = packed) -> np.ndarray:
        pieces = []
        for hit in unique:
            pieces.append(
                predict_local_measurement(
                    local_state,
                    poses[int(hit.station)],
                    z_ref_mm=z_ref,
                    z_station_mm=float(z_mm[int(hit.station)]),
                    field_y=field_y,
                )
            )
        return np.concatenate(pieces)

    observed = np.concatenate([hit.observed for hit in unique])
    predicted = predict_stack(state)
    residual = require_finite(observed - predicted, "linearized residual")
    jacobian_nuisance = finite_difference_jacobian(predict_stack, state, xi_steps)
    jacobian_global = np.zeros((residual.size, len(parameter_names)), dtype=np.float64)
    for column, name in enumerate(parameter_names):
        station_id, component = _split_parameter_name(str(name))
        step = float(pose_step_mrad if component.endswith("mrad") else pose_step_mm)
        shifted = dict(packed)
        shifted[station_id] = left_update_payload(packed[station_id], _twist_for_parameter(component, step))
        plus_stack = predict_stack(state, shifted)
        jacobian_global[:, column] = (plus_stack - predicted) / step
    prior_sigma = None
    prior_mean = None
    qp_now = float(state[QP_INDEX])
    if str(momentum_mode) == MOMENTUM_PRIOR:
        prior_sigma = float(qp_prior_sigma)
        prior_mean = qp_now if qp_prior_mean is None else float(qp_prior_mean)
    elif str(momentum_mode) != MOMENTUM_FREE:
        raise AlignmentSolverError(f"unsupported momentum mode: {momentum_mode}")
    return assemble_track_block(
        unique,
        jacobian_nuisance,
        jacobian_global,
        residual,
        qp_prior_sigma=prior_sigma,
        qp_prior_mean=prior_mean,
        qp_at_linearization=qp_now,
    )


def refit_local_state(
    hits: Sequence[StationHit],
    state: Sequence[float],
    payloads: Mapping[int, Sequence[float]],
    *,
    z_mm: Mapping[int, float],
    field_y: float,
    momentum_mode: str = MOMENTUM_FREE,
    qp_prior_sigma: float = QP_PRIOR_SIGMA,
    qp_prior_mean: float | None = None,
) -> np.ndarray:
    """Refit one track's ξ at a frozen geometry.  Independent of the global step."""
    current = require_finite(np.asarray(state, dtype=np.float64), "refit state")
    block = linearize_track(
        hits,
        current,
        payloads,
        (),
        z_mm=z_mm,
        field_y=field_y,
        momentum_mode=momentum_mode,
        qp_prior_sigma=qp_prior_sigma,
        qp_prior_mean=qp_prior_mean,
    )
    local = block.jacobian_nuisance.T @ block.weight @ block.jacobian_nuisance
    local = 0.5 * (local + local.T)
    rhs = block.jacobian_nuisance.T @ block.weight @ block.residual
    local, rhs = _apply_qp_prior(local, rhs, block)
    values, vecs = np.linalg.eigh(require_finite(local, "local refit"))
    peak = float(values[-1])
    if peak <= 1.0e-14:
        raise AlignmentSolverError("local refit is singular / ill-conditioned")
    keep = values > 1.0e-10 * peak
    if not np.any(keep):
        raise AlignmentSolverError("local refit is singular / ill-conditioned")
    delta = (vecs[:, keep] * (1.0 / values[keep])) @ (vecs[:, keep].T @ rhs)
    return require_finite(current + delta, "refitted state")


def _split_parameter_name(name: str) -> tuple[int, str]:
    if not name.startswith("s") or "_" not in name:
        raise AlignmentSolverError(f"not a four-station parameter name: {name}")
    station_text, component = name[1:].split("_", 1)
    return require_station_id(int(station_text)), component


def _parameter_scale(name: str) -> float:
    try:
        _, component = _split_parameter_name(str(name))
        return float(PARAMETER_SCALES[component])
    except (AlignmentSolverError, KeyError, ValueError):
        return 1.0


def apply_parameter_update(
    payloads: Mapping[int, Sequence[float]],
    parameter_names: Sequence[str],
    update: Sequence[float],
    *,
    convention: str = UPDATE_LEFT_SE3,
) -> dict[int, tuple[float, ...]]:
    """Apply one left-SE(3) Newton step.  Chart rotations are mrad; payload stores rad."""
    require_update_convention(convention)
    if convention != UPDATE_LEFT_SE3:
        raise AlignmentSolverError("WB82 production chart is left SE(3) only")
    twists: dict[int, np.ndarray] = {
        int(station): np.zeros(6, dtype=np.float64) for station in payloads
    }
    for name, value in zip(parameter_names, update):
        station, component = _split_parameter_name(str(name))
        twists[station] = twists[station] + _twist_for_parameter(component, float(value))
    return {
        int(station): left_update_payload(payloads[int(station)], twists[int(station)])
        for station in payloads
    }


def scaled_update_norm(parameter_names: Sequence[str], update: Sequence[float]) -> float:
    values = require_finite(np.asarray(update, dtype=np.float64), "update")
    scaled = []
    for name, value in zip(parameter_names, values):
        _, component = _split_parameter_name(str(name))
        scaled.append(float(value) / float(PARAMETER_SCALES[component]))
    return float(np.linalg.norm(scaled))


def chi2_of_blocks(blocks: Sequence[LinearTrackBlock]) -> float:
    return float(sum(block.residual @ block.weight @ block.residual for block in blocks))


def iterate_common_track(
    tracks: Sequence[Sequence[StationHit]],
    states: Mapping[int, Sequence[float]],
    payloads: Mapping[int, Sequence[float]],
    *,
    z_mm: Mapping[int, float],
    field_y: float,
    survey_mode: str = SURVEY_FIXED_DZ,
    reference_station: int = 0,
    validation_track_ids: Sequence[int] = (),
    max_iterations: int = MAX_ITERATIONS,
    damping: float = DEFAULT_DAMPING,
    momentum_mode: str = MOMENTUM_PRIOR,
    qp_prior_sigma: float = QP_PRIOR_SIGMA,
    qp_prior_means: Mapping[int, float] | None = None,
    components: Sequence[str] | None = None,
    stations: Sequence[int] | None = None,
    consecutive_required: int = 1,
) -> dict[str, object]:
    """Relinearized damped iterations.  Hitting max_iterations is not a PASS."""
    names = parameter_chart(
        survey_mode=survey_mode,
        reference_station=reference_station,
        components=components,
        stations=stations,
    )
    current = {int(station): tuple(float(value) for value in payloads[int(station)]) for station in payloads}
    current_states = {int(key): np.asarray(values, dtype=np.float64).copy() for key, values in states.items()}
    frozen_qp = {
        int(key): float(values[QP_INDEX]) for key, values in current_states.items()
    }
    if qp_prior_means is not None:
        frozen_qp.update({int(key): float(value) for key, value in qp_prior_means.items()})
    history = []
    validation_ids = {int(track_id) for track_id in validation_track_ids}
    previous_val = None
    converged = False
    last = None
    consecutive = 0
    need = max(1, int(consecutive_required))

    def _linearize(hits: Sequence[StationHit]) -> LinearTrackBlock:
        track_id = int(hits[0].track_id)
        return linearize_track(
            hits,
            current_states[track_id],
            current,
            names,
            z_mm=z_mm,
            field_y=field_y,
            momentum_mode=momentum_mode,
            qp_prior_sigma=qp_prior_sigma,
            qp_prior_mean=frozen_qp[track_id],
        )

    for iteration in range(1, int(max_iterations) + 1):
        fit_blocks = []
        val_blocks = []
        for hits in tracks:
            track_id = int(hits[0].track_id)
            block = _linearize(hits)
            if track_id in validation_ids:
                val_blocks.append(block)
            else:
                fit_blocks.append(block)
        if not fit_blocks:
            raise AlignmentSolverError("iteration has no fit tracks")
        solution = solve_common_track(
            fit_blocks,
            names,
            survey_mode=survey_mode,
            damping=damping,
            current_payloads=current,
        )
        last = solution
        if not solution.ok:
            return {
                "converged": False,
                "reason": solution.status,
                "message": solution.message,
                "iterations": iteration,
                "history": history,
                "solution": solution,
                "payloads": current,
                "states": current_states,
                "automatically_passed_at_max_iterations": False,
                "max_iterations": int(max_iterations),
            }
        try:
            current = apply_parameter_update(current, names, solution.update)
            for hits in tracks:
                track_id = int(hits[0].track_id)
                current_states[track_id] = refit_local_state(
                    hits,
                    current_states[track_id],
                    current,
                    z_mm=z_mm,
                    field_y=field_y,
                    momentum_mode=momentum_mode,
                    qp_prior_sigma=qp_prior_sigma,
                    qp_prior_mean=frozen_qp[track_id],
                )
        except AlignmentSolverError as error:
            status = "non_spd" if "SPD" in str(error) else "singular"
            if "NaN" in str(error) or "Inf" in str(error) or "non-finite" in str(error):
                status = "non_finite"
            return {
                "converged": False,
                "reason": status,
                "message": str(error),
                "iterations": iteration,
                "history": history,
                "solution": solution,
                "payloads": current,
                "states": current_states,
                "automatically_passed_at_max_iterations": False,
                "max_iterations": int(max_iterations),
            }
        val_hits = [hits for hits in tracks if int(hits[0].track_id) in validation_ids]
        if val_hits:
            val_obj = chi2_of_blocks(
                [
                    _linearize(hits)
                    for hits in val_hits
                ]
            )
        else:
            val_obj = chi2_of_blocks(fit_blocks)
        step = scaled_update_norm(names, solution.update)
        rel = None if previous_val is None or previous_val == 0.0 else abs(val_obj - previous_val) / abs(previous_val)
        row = {
            "iteration": iteration,
            "chi2_fit": solution.chi2,
            "validation_objective": val_obj,
            "scaled_update_norm": step,
            "validation_relative_change": rel,
        }
        history.append(row)
        previous_val = val_obj
        if step < SCALED_UPDATE_TOL and rel is not None and rel < VALIDATION_REL_TOL:
            consecutive += 1
            if consecutive >= need:
                converged = True
                break
        else:
            consecutive = 0
    return {
        "converged": converged,
        "reason": "converged" if converged else "max_iterations",
        "iterations": len(history),
        "history": history,
        "solution": last,
        "payloads": current,
        "states": current_states,
        "parameter_names": names,
        "automatically_passed_at_max_iterations": False,
        "max_iterations": int(max_iterations),
    }


def relative_lie_table(
    predicted: Mapping[int, Sequence[float]],
    truth: Mapping[int, Sequence[float]],
) -> dict[str, list[float]]:
    """Per-station Lie-log pose error, plus pairwise ``ΔT`` Lie-log."""
    stations = {}
    for station in STATION_IDS:
        stations[str(station)] = pose_error_lie(predicted[int(station)], truth[int(station)]).tolist()
    pairs = {}
    for source in STATION_IDS:
        for target in STATION_IDS:
            if source >= target:
                continue
            pred = six_vector_to_matrix(invert_six_vector(predicted[int(source)])) @ six_vector_to_matrix(predicted[int(target)])
            true = six_vector_to_matrix(invert_six_vector(truth[int(source)])) @ six_vector_to_matrix(truth[int(target)])
            pairs[f"{source}_{target}"] = se3_log(np.linalg.inv(pred) @ true).tolist()
    return {"per_station": stations, "relative": pairs}


def translation_second_difference(per_station: Mapping[str, Sequence[float]], *, axis: int) -> float:
    """``d3 - 2 d2 + d1``.  Kills the reference-chart ``a + b z`` slope mode."""
    if axis not in (0, 1):
        raise AlignmentSolverError("translation second difference is only defined for dx/dy")
    return float(per_station["3"][axis] - 2.0 * per_station["2"][axis] + per_station["1"][axis])
