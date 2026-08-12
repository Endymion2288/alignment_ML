"""Controlled station-level alignment closure using truth-fixed tracklet pairs.

This module deliberately works at the residual level.  It is an intermediate
validation of the alignment objective, not a substitute for rerunning ACTS in
a physically deformed detector geometry.  Its coordinate-reporting convention
is documented in :func:`inject_station_offsets`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from evaluation.field_propagation import FieldPropagationEvaluation


XY_DIMENSIONS = 2


@dataclass(frozen=True)
class AlignmentMeasurements:
    """Field-aware pair residuals reduced to station-level x/y alignment data."""

    source_station_id: np.ndarray
    target_station_id: np.ndarray
    nominal_residual_xy_mm: np.ndarray
    covariance_xy_mm2: np.ndarray

    @property
    def size(self) -> int:
        return int(self.source_station_id.size)

    @property
    def station_ids(self) -> np.ndarray:
        if not self.size:
            return np.empty(0, dtype=np.int16)
        return np.unique(
            np.concatenate((self.source_station_id, self.target_station_id))
        ).astype(np.int16, copy=False)


@dataclass(frozen=True)
class AlignmentFitResult:
    """Result of a reference-fixed, covariance-weighted alignment solve."""

    station_ids: np.ndarray
    reference_station: int
    alignment_xy_mm: np.ndarray
    alignment_covariance_mm2: np.ndarray
    active_mask: np.ndarray
    active_counts: tuple[int, ...]
    final_increment_residual_xy_mm: np.ndarray
    final_increment_chi2: np.ndarray
    normal_matrix_rank: int
    normal_matrix_condition_number: float | None

    @property
    def size(self) -> int:
        return int(self.station_ids.size)

    @property
    def active_fraction(self) -> float:
        if not self.active_mask.size:
            return 0.0
        return float(np.mean(self.active_mask))

    def offsets_by_station(self) -> dict[int, list[float]]:
        return {
            int(station): [float(value) for value in offset]
            for station, offset in zip(self.station_ids, self.alignment_xy_mm)
        }


def _validate_measurements(measurements: AlignmentMeasurements) -> None:
    size = measurements.size
    if size == 0:
        raise ValueError("at least one truth-fixed propagation pair is required")
    if measurements.target_station_id.shape != (size,):
        raise ValueError("target_station_id must have shape (n,)")
    if measurements.nominal_residual_xy_mm.shape != (size, XY_DIMENSIONS):
        raise ValueError("nominal_residual_xy_mm must have shape (n, 2)")
    if measurements.covariance_xy_mm2.shape != (size, XY_DIMENSIONS, XY_DIMENSIONS):
        raise ValueError("covariance_xy_mm2 must have shape (n, 2, 2)")
    if np.any(measurements.source_station_id == measurements.target_station_id):
        raise ValueError("a propagation pair must connect two different stations")
    if not np.isfinite(measurements.nominal_residual_xy_mm).all():
        raise ValueError("nominal residuals contain non-finite values")
    for covariance in measurements.covariance_xy_mm2:
        if not np.isfinite(covariance).all() or not np.allclose(
            covariance, covariance.T, rtol=1.0e-7, atol=1.0e-12
        ):
            raise ValueError("alignment covariance must be finite and symmetric")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError("alignment covariance must be positive definite") from error


def measurements_from_field_evaluation(
    evaluation: FieldPropagationEvaluation,
) -> AlignmentMeasurements:
    """Build x/y alignment measurements from accepted field-aware diagnostics."""
    measurements = AlignmentMeasurements(
        source_station_id=np.asarray(evaluation.source_station_id, dtype=np.int16),
        target_station_id=np.asarray(evaluation.target_station_id, dtype=np.int16),
        nominal_residual_xy_mm=np.asarray(evaluation.residual[:, :XY_DIMENSIONS], dtype=np.float64),
        covariance_xy_mm2=np.asarray(
            evaluation.combined_covariance[:, :XY_DIMENSIONS, :XY_DIMENSIONS],
            dtype=np.float64,
        ),
    )
    _validate_measurements(measurements)
    return measurements


def select_measurements(
    measurements: AlignmentMeasurements,
    mask: np.ndarray,
) -> AlignmentMeasurements:
    """Return a validated subset of truth-fixed alignment measurements."""
    _validate_measurements(measurements)
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != (measurements.size,):
        raise ValueError("measurement selection mask must have shape (n,)")
    if not np.any(selected):
        raise ValueError("measurement selection removed every truth-fixed pair")
    result = AlignmentMeasurements(
        source_station_id=measurements.source_station_id[selected],
        target_station_id=measurements.target_station_id[selected],
        nominal_residual_xy_mm=measurements.nominal_residual_xy_mm[selected],
        covariance_xy_mm2=measurements.covariance_xy_mm2[selected],
    )
    _validate_measurements(result)
    return result


def _offset_array(
    station_ids: np.ndarray,
    reference_station: int,
    offsets_by_station: Mapping[int, Sequence[float]],
) -> np.ndarray:
    """Validate a station-keyed offset mapping and return a stable array."""
    station_set = {int(station) for station in station_ids}
    unknown = sorted(set(int(station) for station in offsets_by_station) - station_set)
    if unknown:
        raise ValueError(f"offsets supplied for unknown stations: {unknown}")
    values = np.zeros((station_ids.size, XY_DIMENSIONS), dtype=np.float64)
    for row, station in enumerate(station_ids):
        supplied = offsets_by_station.get(int(station), (0.0, 0.0))
        offset = np.asarray(supplied, dtype=np.float64)
        if offset.shape != (XY_DIMENSIONS,) or not np.isfinite(offset).all():
            raise ValueError(
                f"offset for station {int(station)} must be two finite values in mm"
            )
        values[row] = offset
    reference_row = int(np.flatnonzero(station_ids == reference_station)[0])
    if not np.allclose(values[reference_row], 0.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("the reference station offset must be exactly zero")
    return values


def sample_station_offsets(
    station_ids: Sequence[int] | np.ndarray,
    reference_station: int,
    magnitude_mm: float,
    rng: np.random.Generator,
) -> dict[int, tuple[float, float]]:
    """Sample one x/y offset vector of fixed magnitude for each movable station."""
    stations = np.asarray(sorted({int(station) for station in station_ids}), dtype=np.int16)
    if reference_station not in stations:
        raise ValueError("reference station is absent from station_ids")
    if not np.isfinite(magnitude_mm) or magnitude_mm < 0.0:
        raise ValueError("magnitude_mm must be finite and non-negative")
    offsets: dict[int, tuple[float, float]] = {}
    for station in stations:
        if int(station) == reference_station or magnitude_mm == 0.0:
            offsets[int(station)] = (0.0, 0.0)
            continue
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        offsets[int(station)] = (
            float(magnitude_mm * np.cos(angle)),
            float(magnitude_mm * np.sin(angle)),
        )
    return offsets


def inject_station_offsets(
    measurements: AlignmentMeasurements,
    offsets_by_station: Mapping[int, Sequence[float]],
    reference_station: int,
    rng: np.random.Generator | None = None,
    measurement_noise_scale: float = 0.0,
) -> np.ndarray:
    """Inject controlled offsets into nominal pair residuals.

    The convention is a station-coordinate reporting shift: a local state at
    station ``s`` is reported as ``q_s + Delta_s``.  Since the diagnostic
    residual is target minus propagated source, its injected increment is
    ``Delta_target - Delta_source``.  This does **not** alter the magnetic
    field, material, surface transforms, or ACTS propagation itself.

    ``measurement_noise_scale`` multiplies the Gaussian covariance standard
    deviation.  It is zero by default, so exact closure is testable.
    """
    _validate_measurements(measurements)
    if not np.isfinite(measurement_noise_scale) or measurement_noise_scale < 0.0:
        raise ValueError("measurement_noise_scale must be finite and non-negative")
    stations = measurements.station_ids
    if reference_station not in stations:
        raise ValueError("reference station is absent from the measurements")
    offsets = _offset_array(stations, reference_station, offsets_by_station)
    station_to_row = {int(station): row for row, station in enumerate(stations)}
    source_rows = np.asarray(
        [station_to_row[int(station)] for station in measurements.source_station_id], dtype=np.intp
    )
    target_rows = np.asarray(
        [station_to_row[int(station)] for station in measurements.target_station_id], dtype=np.intp
    )
    observed = measurements.nominal_residual_xy_mm + offsets[target_rows] - offsets[source_rows]
    if measurement_noise_scale == 0.0:
        return observed
    if rng is None:
        raise ValueError("rng is required when measurement_noise_scale is non-zero")
    noise = np.empty_like(observed)
    for row, covariance in enumerate(measurements.covariance_xy_mm2):
        noise[row] = rng.multivariate_normal(
            mean=np.zeros(XY_DIMENSIONS, dtype=np.float64),
            cov=(measurement_noise_scale**2) * covariance,
        )
    return observed + noise


def _design_matrices(
    measurements: AlignmentMeasurements,
    station_ids: np.ndarray,
    reference_station: int,
) -> tuple[np.ndarray, dict[int, int]]:
    """Create one 2-by-2N design matrix per pair with the reference removed."""
    non_reference = [int(station) for station in station_ids if int(station) != reference_station]
    parameter_row = {station: XY_DIMENSIONS * row for row, station in enumerate(non_reference)}
    design = np.zeros((measurements.size, XY_DIMENSIONS, XY_DIMENSIONS * len(non_reference)))
    identity = np.eye(XY_DIMENSIONS)
    for row, (source, target) in enumerate(
        zip(measurements.source_station_id, measurements.target_station_id)
    ):
        source_column = parameter_row.get(int(source))
        target_column = parameter_row.get(int(target))
        if source_column is not None:
            design[row, :, source_column : source_column + XY_DIMENSIONS] -= identity
        if target_column is not None:
            design[row, :, target_column : target_column + XY_DIMENSIONS] += identity
    return design, parameter_row


def _chi2_per_measurement(residual_xy_mm: np.ndarray, covariance_xy_mm2: np.ndarray) -> np.ndarray:
    values = np.empty(residual_xy_mm.shape[0], dtype=np.float64)
    for row, (residual, covariance) in enumerate(
        zip(residual_xy_mm, covariance_xy_mm2)
    ):
        values[row] = float(residual @ np.linalg.solve(covariance, residual))
    return values


def solve_alignment(
    measurements: AlignmentMeasurements,
    observed_residual_xy_mm: np.ndarray,
    reference_station: int,
    chi2_gate: float | None = None,
    refinement_iterations: int = 3,
    prior_sigma_mm: float | None = None,
) -> AlignmentFitResult:
    """Fit station x/y offsets after subtracting nominal field-aware residuals.

    Pair selection is iteratively gated on the *baseline-subtracted* increment.
    This avoids conflating present nominal geometry/model residuals with the
    deliberately injected offsets in the controlled closure experiment.
    """
    _validate_measurements(measurements)
    observed = np.asarray(observed_residual_xy_mm, dtype=np.float64)
    if observed.shape != (measurements.size, XY_DIMENSIONS) or not np.isfinite(observed).all():
        raise ValueError("observed_residual_xy_mm must be finite with shape (n, 2)")
    if reference_station not in measurements.station_ids:
        raise ValueError("reference station is absent from the measurements")
    if chi2_gate is not None and (not np.isfinite(chi2_gate) or chi2_gate <= 0.0):
        raise ValueError("chi2_gate must be positive when supplied")
    if refinement_iterations < 1:
        raise ValueError("refinement_iterations must be at least one")
    if prior_sigma_mm is not None and (
        not np.isfinite(prior_sigma_mm) or prior_sigma_mm <= 0.0
    ):
        raise ValueError("prior_sigma_mm must be positive when supplied")

    station_ids = measurements.station_ids
    design, parameter_rows = _design_matrices(measurements, station_ids, reference_station)
    parameter_count = XY_DIMENSIONS * len(parameter_rows)
    increments = observed - measurements.nominal_residual_xy_mm
    parameters = np.zeros(parameter_count, dtype=np.float64)
    active_counts: list[int] = []
    active_mask = np.ones(measurements.size, dtype=bool)
    normal = np.zeros((parameter_count, parameter_count), dtype=np.float64)

    for _ in range(refinement_iterations):
        innovation = increments - np.einsum("nij,j->ni", design, parameters)
        innovation_chi2 = _chi2_per_measurement(innovation, measurements.covariance_xy_mm2)
        active_mask = (
            np.ones(measurements.size, dtype=bool)
            if chi2_gate is None
            else innovation_chi2 <= chi2_gate
        )
        active_counts.append(int(np.count_nonzero(active_mask)))
        if not np.any(active_mask):
            normal.fill(0.0)
            break

        normal.fill(0.0)
        right_hand_side = np.zeros(parameter_count, dtype=np.float64)
        for row in np.flatnonzero(active_mask):
            design_row = design[row]
            weighted_design = np.linalg.solve(
                measurements.covariance_xy_mm2[row], design_row
            )
            normal += design_row.T @ weighted_design
            right_hand_side += design_row.T @ np.linalg.solve(
                measurements.covariance_xy_mm2[row], increments[row]
            )
        if prior_sigma_mm is not None:
            normal += np.eye(parameter_count) / (prior_sigma_mm**2)
        rank = int(np.linalg.matrix_rank(normal))
        if rank < parameter_count:
            break
        parameters = np.linalg.solve(normal, right_hand_side)

    estimated = np.zeros((station_ids.size, XY_DIMENSIONS), dtype=np.float64)
    for station, column in parameter_rows.items():
        station_row = int(np.flatnonzero(station_ids == station)[0])
        estimated[station_row] = parameters[column : column + XY_DIMENSIONS]
    final_increment_residual = increments - np.einsum("nij,j->ni", design, parameters)
    final_increment_chi2 = _chi2_per_measurement(
        final_increment_residual, measurements.covariance_xy_mm2
    )
    final_active_mask = (
        np.ones(measurements.size, dtype=bool)
        if chi2_gate is None
        else final_increment_chi2 <= chi2_gate
    )
    rank = int(np.linalg.matrix_rank(normal)) if parameter_count else 0
    condition_number: float | None = None
    if parameter_count and rank == parameter_count:
        candidate_condition = float(np.linalg.cond(normal))
        if np.isfinite(candidate_condition):
            condition_number = candidate_condition
    parameter_covariance = np.full((parameter_count, parameter_count), np.nan, dtype=np.float64)
    if parameter_count and rank == parameter_count:
        parameter_covariance = np.linalg.solve(
            normal, np.eye(parameter_count, dtype=np.float64)
        )
    full_covariance = np.full(
        (station_ids.size, XY_DIMENSIONS, station_ids.size, XY_DIMENSIONS), np.nan, dtype=np.float64
    )
    reference_row = int(np.flatnonzero(station_ids == reference_station)[0])
    full_covariance[reference_row, :, reference_row, :] = 0.0
    for station_a, column_a in parameter_rows.items():
        row_a = int(np.flatnonzero(station_ids == station_a)[0])
        for station_b, column_b in parameter_rows.items():
            row_b = int(np.flatnonzero(station_ids == station_b)[0])
            full_covariance[row_a, :, row_b, :] = parameter_covariance[
                column_a : column_a + XY_DIMENSIONS,
                column_b : column_b + XY_DIMENSIONS,
            ]
    return AlignmentFitResult(
        station_ids=station_ids,
        reference_station=reference_station,
        alignment_xy_mm=estimated,
        alignment_covariance_mm2=full_covariance,
        active_mask=final_active_mask,
        active_counts=tuple(active_counts),
        final_increment_residual_xy_mm=final_increment_residual,
        final_increment_chi2=final_increment_chi2,
        normal_matrix_rank=rank,
        normal_matrix_condition_number=condition_number,
    )


def alignment_error_summary(
    fit: AlignmentFitResult,
    injected_offsets_by_station: Mapping[int, Sequence[float]],
) -> dict[str, object]:
    """Compare a reference-fixed fit with its known injected offsets."""
    injected = _offset_array(
        fit.station_ids, fit.reference_station, injected_offsets_by_station
    )
    error = fit.alignment_xy_mm - injected
    movable = fit.station_ids != fit.reference_station
    movable_error = error[movable]
    norms = np.linalg.norm(movable_error, axis=1)
    return {
        "injected_offsets_xy_mm": {
            int(station): [float(value) for value in offset]
            for station, offset in zip(fit.station_ids, injected)
        },
        "recovered_offsets_xy_mm": fit.offsets_by_station(),
        "error_xy_mm": {
            int(station): [float(value) for value in offset]
            for station, offset in zip(fit.station_ids, error)
        },
        "absolute_error_norm_mm_by_station": {
            int(station): float(np.linalg.norm(offset))
            for station, offset in zip(fit.station_ids, error)
        },
        "movable_station_rms_error_mm": float(
            np.sqrt(np.mean(np.square(movable_error))) if movable_error.size else 0.0
        ),
        "movable_station_max_norm_error_mm": float(np.max(norms) if norms.size else 0.0),
        "active_fraction": fit.active_fraction,
        "active_counts": list(fit.active_counts),
        "normal_matrix_rank": fit.normal_matrix_rank,
        "normal_matrix_condition_number": fit.normal_matrix_condition_number,
    }


def run_capture_range_scan(
    measurements: AlignmentMeasurements,
    reference_station: int,
    magnitudes_mm: Sequence[float],
    trials_per_magnitude: int,
    seed: int,
    chi2_gate: float | None,
    refinement_iterations: int,
    capture_tolerance_mm: float,
    measurement_noise_scale: float = 0.0,
    prior_sigma_mm: float | None = None,
) -> dict[str, object]:
    """Run random-direction, truth-fixed residual-level alignment capture scan."""
    _validate_measurements(measurements)
    if trials_per_magnitude < 1:
        raise ValueError("trials_per_magnitude must be at least one")
    if not np.isfinite(capture_tolerance_mm) or capture_tolerance_mm < 0.0:
        raise ValueError("capture_tolerance_mm must be finite and non-negative")
    rng = np.random.default_rng(seed)
    points: list[dict[str, object]] = []
    for magnitude in magnitudes_mm:
        if not np.isfinite(magnitude) or magnitude < 0.0:
            raise ValueError("all scan magnitudes must be finite and non-negative")
        trial_summaries: list[dict[str, float | bool | int]] = []
        for _ in range(trials_per_magnitude):
            injected = sample_station_offsets(
                measurements.station_ids, reference_station, float(magnitude), rng
            )
            observed = inject_station_offsets(
                measurements,
                injected,
                reference_station,
                rng=rng,
                measurement_noise_scale=measurement_noise_scale,
            )
            fit = solve_alignment(
                measurements,
                observed,
                reference_station=reference_station,
                chi2_gate=chi2_gate,
                refinement_iterations=refinement_iterations,
                prior_sigma_mm=prior_sigma_mm,
            )
            summary = alignment_error_summary(fit, injected)
            max_error = float(summary["movable_station_max_norm_error_mm"])
            captured = bool(
                fit.normal_matrix_rank == XY_DIMENSIONS * (fit.size - 1)
                and max_error <= capture_tolerance_mm
            )
            trial_summaries.append(
                {
                    "captured": captured,
                    "max_error_mm": max_error,
                    "rms_error_mm": float(summary["movable_station_rms_error_mm"]),
                    "active_fraction": float(summary["active_fraction"]),
                    "normal_matrix_rank": int(summary["normal_matrix_rank"]),
                }
            )
        points.append(
            {
                "magnitude_mm": float(magnitude),
                "trials": trials_per_magnitude,
                "capture_fraction": float(np.mean([trial["captured"] for trial in trial_summaries])),
                "mean_max_error_mm": float(np.mean([trial["max_error_mm"] for trial in trial_summaries])),
                "p95_max_error_mm": float(np.percentile([trial["max_error_mm"] for trial in trial_summaries], 95.0)),
                "mean_rms_error_mm": float(np.mean([trial["rms_error_mm"] for trial in trial_summaries])),
                "mean_active_fraction": float(
                    np.mean([trial["active_fraction"] for trial in trial_summaries])
                ),
                "trial_results": trial_summaries,
            }
        )
    return {
        "method": "truth-fixed_baseline-subtracted_residual-level_alignment_closure",
        "reference_station": int(reference_station),
        "station_ids": [int(station) for station in measurements.station_ids],
        "pair_count": measurements.size,
        "chi2_gate": None if chi2_gate is None else float(chi2_gate),
        "refinement_iterations": int(refinement_iterations),
        "measurement_noise_scale": float(measurement_noise_scale),
        "capture_tolerance_mm": float(capture_tolerance_mm),
        "seed": int(seed),
        "points": points,
    }
