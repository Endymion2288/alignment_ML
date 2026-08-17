"""Small global straight-track fitter for physical alignment iterations.

This is intentionally the V1 model, not a Kalman filter.  It fits the four
local segment states ``[x, y, tx, ty]`` at their actual station z positions
with the exported 4x4 state covariances.  Field-aware propagation remains the
candidate-construction layer; this fitter supplies a transparent global track
quality and residual objective after route association.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class GlobalTrackFit:
    """One covariance-weighted global line fit for an associated route."""

    endpoint_indices: tuple[int, ...]
    station_ids: tuple[int, ...]
    z_reference_mm: float
    z_scale_mm: float
    parameters: np.ndarray
    parameter_covariance: np.ndarray
    fitted_states: np.ndarray
    residual: np.ndarray
    chi2: float
    ndof: int
    normal_matrix_rank: int

    @property
    def reduced_chi2(self) -> float | None:
        return None if self.ndof <= 0 else float(self.chi2 / self.ndof)


@dataclass(frozen=True)
class LeaveOneOutTrackResidual:
    """One station prediction from a route fit excluding that station.

    The residual is intentionally constructed from a fit to the *other*
    selected tracklets.  Its combined covariance is therefore positive
    definite, unlike the singular in-sample residual covariance of a global
    least-squares fit.  This makes it a valid four-component observation for
    the physical finite-difference alignment update.
    """

    endpoint_index: int
    station_id: int
    z_mm: float
    prediction: np.ndarray
    prediction_covariance: np.ndarray
    residual: np.ndarray
    combined_covariance: np.ndarray
    chi2: float
    reference_fit: GlobalTrackFit


def _inverse_covariance(matrix: np.ndarray, *, label: str) -> np.ndarray:
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} covariance must be finite with shape [4, 4]")
    if not np.allclose(matrix, matrix.T, rtol=1.0e-7, atol=1.0e-12):
        raise ValueError(f"{label} covariance must be symmetric")
    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{label} covariance must be positive definite") from error
    inverse_factor = np.linalg.solve(factor, np.eye(4))
    return inverse_factor.T @ inverse_factor


def _design(delta_z_mm: float) -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.0, delta_z_mm, 0.0],
            [0.0, 1.0, 0.0, delta_z_mm],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _scaled_design(delta_z_mm: float, z_scale_mm: float) -> np.ndarray:
    """Condition the line-fit slope columns without changing physical units."""
    if not np.isfinite(z_scale_mm) or z_scale_mm <= 0.0:
        raise ValueError("z_scale_mm must be finite and positive")
    inverse_scale = 1.0 / float(z_scale_mm)
    scaled_delta = float(delta_z_mm) * inverse_scale
    return np.asarray(
        [
            [1.0, 0.0, scaled_delta, 0.0],
            [0.0, 1.0, 0.0, scaled_delta],
            [0.0, 0.0, inverse_scale, 0.0],
            [0.0, 0.0, 0.0, inverse_scale],
        ],
        dtype=np.float64,
    )


def fit_global_straight_track(
    state: object,
    covariance: object,
    z_mm: object,
    *,
    endpoint_indices: Sequence[int] | None = None,
    station_ids: Sequence[int] | None = None,
    rcond: float = 1.0e-10,
) -> GlobalTrackFit:
    """Fit one global straight track from independently reconstructed segments.

    The parameter vector is ``[x(z_ref), y(z_ref), tx, ty]``.  The caller is
    responsible for providing an association-selected route; this function
    never consults truth labels or creates candidate links.
    """
    values = np.asarray(state, dtype=np.float64)
    covariances = np.asarray(covariance, dtype=np.float64)
    positions = np.asarray(z_mm, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != (4,):
        raise ValueError("state must have shape [segments, 4]")
    count = values.shape[0]
    if count < 2 or covariances.shape != (count, 4, 4) or positions.shape != (count,):
        raise ValueError("global straight-track fit requires at least two aligned segment states/covariances/z values")
    if not np.isfinite(values).all() or not np.isfinite(positions).all():
        raise ValueError("global straight-track fit inputs must be finite")
    if not np.isfinite(rcond) or not 0.0 < rcond < 1.0:
        raise ValueError("rcond must lie in (0, 1)")
    if endpoint_indices is None:
        endpoints = tuple(range(count))
    else:
        endpoints = tuple(int(value) for value in endpoint_indices)
        if len(endpoints) != count or len(set(endpoints)) != count:
            raise ValueError("endpoint_indices must be unique and align to supplied states")
    if station_ids is None:
        stations = tuple(-1 for _ in range(count))
    else:
        stations = tuple(int(value) for value in station_ids)
        if len(stations) != count:
            raise ValueError("station_ids must align to supplied states")
    z_reference = float(np.mean(positions))
    # FASER station separations are large compared with the position/slope
    # covariance units.  Solve for [x, y, tx*z_scale, ty*z_scale] so a normal
    # matrix rank test does not discard physical slope directions merely due
    # to millimetre scale.  Results are converted back to [x, y, tx, ty].
    z_scale = max(float(np.max(np.abs(positions - z_reference))), 1.0)
    normal = np.zeros((4, 4), dtype=np.float64)
    rhs = np.zeros(4, dtype=np.float64)
    designs: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for row in range(count):
        design = _scaled_design(float(positions[row] - z_reference), z_scale)
        inverse = _inverse_covariance(covariances[row], label=f"segment {row}")
        normal += design.T @ inverse @ design
        rhs += design.T @ inverse @ values[row]
        designs.append(design)
        weights.append(inverse)
    normal = 0.5 * (normal + normal.T)
    left, singular, right = np.linalg.svd(normal, full_matrices=False)
    tolerance = max(float(singular[0]) * rcond, np.finfo(np.float64).eps) if singular.size else np.inf
    rank = int(np.count_nonzero(singular > tolerance))
    if rank < 4:
        raise ValueError("global straight-track normal matrix is rank deficient")
    inverse_normal = (right.T / singular) @ left.T
    parameters_scaled = inverse_normal @ rhs
    conversion = np.diag([1.0, 1.0, 1.0 / z_scale, 1.0 / z_scale])
    parameters = conversion @ parameters_scaled
    covariance_physical = conversion @ inverse_normal @ conversion.T
    covariance_physical = 0.5 * (covariance_physical + covariance_physical.T)
    physical_designs = [_design(float(positions[row] - z_reference)) for row in range(count)]
    fitted = np.asarray([design @ parameters for design in physical_designs], dtype=np.float64)
    residual = values - fitted
    chi2 = float(sum(row @ weight @ row for row, weight in zip(residual, weights)))
    if not np.isfinite(chi2):
        raise ValueError("global straight-track chi2 is non-finite")
    return GlobalTrackFit(
        endpoint_indices=endpoints,
        station_ids=stations,
        z_reference_mm=z_reference,
        z_scale_mm=z_scale,
        parameters=np.asarray(parameters, dtype=np.float64),
        parameter_covariance=np.asarray(covariance_physical, dtype=np.float64),
        fitted_states=fitted,
        residual=residual,
        chi2=chi2,
        ndof=int(4 * count - rank),
        normal_matrix_rank=rank,
    )


def fit_event_route(event: object, endpoint_indices: Sequence[int]) -> GlobalTrackFit:
    """Fit one selected physical route directly from an event tracklet table."""
    endpoints = tuple(int(value) for value in endpoint_indices)
    if len(endpoints) < 2 or len(set(endpoints)) != len(endpoints):
        raise ValueError("a selected event route needs at least two unique endpoints")
    try:
        state = np.asarray(event.state, dtype=np.float64)[list(endpoints)]
        covariance = np.asarray(event.covariance, dtype=np.float64)[list(endpoints)]
        z_mm = np.asarray(event.z_mm, dtype=np.float64)[list(endpoints)]
        station_ids = np.asarray(event.station_id, dtype=np.int64)[list(endpoints)]
    except AttributeError as error:
        raise ValueError("event lacks state/covariance/z_mm/station_id tracklet arrays") from error
    return fit_global_straight_track(
        state,
        covariance,
        z_mm,
        endpoint_indices=endpoints,
        station_ids=station_ids,
    )


def leave_one_out_event_route_residuals(
    event: object,
    endpoint_indices: Sequence[int],
) -> tuple[LeaveOneOutTrackResidual, ...]:
    """Predict every selected endpoint from the rest of its associated route.

    This function never uses truth labels.  It is the observation layer used
    after the frozen route solver: association chooses the endpoints, then a
    given station is compared to a global straight-line fit of the remaining
    stations.  At least three route endpoints are required so every withheld
    endpoint still has a two-segment reference fit.
    """
    endpoints = tuple(int(value) for value in endpoint_indices)
    if len(endpoints) < 3 or len(set(endpoints)) != len(endpoints):
        raise ValueError("leave-one-out global residuals require at least three unique route endpoints")
    try:
        all_state = np.asarray(event.state, dtype=np.float64)
        all_covariance = np.asarray(event.covariance, dtype=np.float64)
        all_z = np.asarray(event.z_mm, dtype=np.float64)
        all_station = np.asarray(event.station_id, dtype=np.int64)
    except AttributeError as error:
        raise ValueError("event lacks state/covariance/z_mm/station_id tracklet arrays") from error
    if any(index < 0 or index >= all_state.shape[0] for index in endpoints):
        raise ValueError("route endpoint index is outside the event tracklet table")

    result: list[LeaveOneOutTrackResidual] = []
    for endpoint in endpoints:
        reference_indices = tuple(index for index in endpoints if index != endpoint)
        reference_fit = fit_global_straight_track(
            all_state[list(reference_indices)],
            all_covariance[list(reference_indices)],
            all_z[list(reference_indices)],
            endpoint_indices=reference_indices,
            station_ids=all_station[list(reference_indices)],
        )
        design = _design(float(all_z[endpoint] - reference_fit.z_reference_mm))
        prediction = design @ reference_fit.parameters
        prediction_covariance = design @ reference_fit.parameter_covariance @ design.T
        prediction_covariance = 0.5 * (prediction_covariance + prediction_covariance.T)
        combined_covariance = all_covariance[endpoint] + prediction_covariance
        combined_covariance = 0.5 * (combined_covariance + combined_covariance.T)
        inverse = _inverse_covariance(combined_covariance, label=f"leave-one-out endpoint {endpoint}")
        residual = all_state[endpoint] - prediction
        chi2 = float(residual @ inverse @ residual)
        if not np.isfinite(chi2):
            raise ValueError("leave-one-out global residual chi2 is non-finite")
        result.append(
            LeaveOneOutTrackResidual(
                endpoint_index=endpoint,
                station_id=int(all_station[endpoint]),
                z_mm=float(all_z[endpoint]),
                prediction=np.asarray(prediction, dtype=np.float64),
                prediction_covariance=np.asarray(prediction_covariance, dtype=np.float64),
                residual=np.asarray(residual, dtype=np.float64),
                combined_covariance=np.asarray(combined_covariance, dtype=np.float64),
                chi2=chi2,
                reference_fit=reference_fit,
            )
        )
    return tuple(result)


def fit_selected_routes(
    event: object,
    routes: Iterable[object],
    *,
    complete_only: bool = False,
    station_path: Sequence[int] = (0, 1, 2, 3),
) -> tuple[GlobalTrackFit, ...]:
    """Fit route-assignment outputs without using their truth-only diagnostics."""
    expected = tuple(int(value) for value in station_path)
    result: list[GlobalTrackFit] = []
    for route in routes:
        try:
            endpoints = tuple((int(station), int(index)) for station, index in route.endpoints)
        except AttributeError as error:
            raise ValueError("route object lacks endpoint pairs") from error
        stations = tuple(station for station, _ in endpoints)
        if complete_only and stations != expected:
            continue
        result.append(fit_event_route(event, tuple(index for _, index in endpoints)))
    return tuple(result)


def global_track_fit_summary(fits: Sequence[GlobalTrackFit]) -> dict[str, float | int | None]:
    """Aggregate route-level global track quality for iteration diagnostics."""
    if not fits:
        return {
            "tracks": 0,
            "chi2": 0.0,
            "ndof": 0,
            "reduced_chi2": None,
            "complete_tracks": 0,
        }
    chi2 = float(sum(fit.chi2 for fit in fits))
    ndof = int(sum(fit.ndof for fit in fits))
    return {
        "tracks": len(fits),
        "chi2": chi2,
        "ndof": ndof,
        "reduced_chi2": None if ndof <= 0 else float(chi2 / ndof),
        "complete_tracks": int(sum(len(fit.endpoint_indices) == 4 for fit in fits)),
    }
