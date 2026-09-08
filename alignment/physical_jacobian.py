"""Multi-parameter finite-difference **response / regression diagnostic**.

This module is not a certified alignment solver, not a full alignment
closure, and not a downstream oracle.  ``continue_to_15d_relative_wls``
remains false.  Workbook 82 qualifies alignment with
``alignment/common_track_solver.py`` instead.

The functions here operate only on residuals exported after independently
writing a real ``/Tracker/Align`` payload and rerunning Calypso's segment
refit plus mode-0 Acts propagation.  They deliberately do not implement a
coordinate-level alignment surrogate.  Pairwise same-event counterfactual
differences remain a diagnostic, not a scientific validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from alignment.layer_hierarchy import (
    IFT_LAYER_IDS,
    IFT_STATION_ID,
    require_station_scope,
    spec_scope,
)


RESIDUAL_DIMENSION = 4
COMPONENT_INDEX_AND_PAYLOAD_SCALE: dict[str, tuple[int, float]] = {
    "dx_mm": (0, 1.0),
    "dy_mm": (1, 1.0),
    "dz_mm": (2, 1.0),
    "rx_mrad": (3, 1.0e3),
    "ry_mrad": (4, 1.0e3),
    "rz_mrad": (5, 1.0e3),
}


@dataclass(frozen=True)
class PhysicalJacobianFit:
    """WLS closure result in the declared native parameter units.

    ``normal_matrix_scaled`` is the primary identifiability diagnostic.  It
    corresponds to dimensionless parameters ``u = theta / parameter_scales``;
    this removes an arbitrary mm-versus-mrad unit choice from rank and
    condition-number conclusions.  The native matrix and covariance are kept
    for reproducible parameter reporting.
    """

    parameter_names: tuple[str, ...]
    parameter_scales: np.ndarray
    derivative_native: np.ndarray
    response: np.ndarray
    predicted_response: np.ndarray
    residual_response: np.ndarray
    recovered_parameters: np.ndarray
    covariance_native: np.ndarray
    correlation_native: np.ndarray
    normal_matrix_native: np.ndarray
    normal_matrix_scaled: np.ndarray
    right_hand_side_native: np.ndarray
    right_hand_side_scaled: np.ndarray
    data_singular_values: np.ndarray
    fit_singular_values: np.ndarray
    normal_matrix_rank: int
    fit_matrix_rank: int
    normal_matrix_condition_number: float | None
    identifiable_subspace_condition_number: float | None
    parameter_observability_fraction: np.ndarray
    response_chi2: float
    response_ndof: int
    used_pairs: int
    prior_sigma_native: np.ndarray | None

    @property
    def full_rank(self) -> bool:
        return self.normal_matrix_rank == len(self.parameter_names)


def _responses(values: object, *, label: str, parameters: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if parameters is None:
        if array.ndim != 2 or array.shape[1:] != (RESIDUAL_DIMENSION,):
            raise ValueError(f"{label} must have shape [pairs, {RESIDUAL_DIMENSION}]")
    elif array.ndim != 3 or array.shape[0] != parameters or array.shape[2:] != (RESIDUAL_DIMENSION,):
        raise ValueError(f"{label} must have shape [parameters, pairs, {RESIDUAL_DIMENSION}]")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return array


def _inverse_covariances(covariance: object, pairs: int) -> list[np.ndarray]:
    matrices = np.asarray(covariance, dtype=np.float64)
    if matrices.shape != (pairs, RESIDUAL_DIMENSION, RESIDUAL_DIMENSION):
        raise ValueError(f"covariance must have shape [pairs, {RESIDUAL_DIMENSION}, {RESIDUAL_DIMENSION}]")
    if not np.isfinite(matrices).all():
        raise ValueError("covariance contains non-finite values")
    result: list[np.ndarray] = []
    for row, matrix in enumerate(matrices):
        if not np.allclose(matrix, matrix.T, rtol=1.0e-7, atol=1.0e-12):
            raise ValueError(f"covariance row {row} is not symmetric")
        try:
            # Cholesky is both a positive-definiteness test and a stable
            # inverse construction for the deterministic WLS weight.
            factor = np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as error:
            raise ValueError(f"covariance row {row} is not positive definite") from error
        inverse_factor = np.linalg.solve(factor, np.eye(RESIDUAL_DIMENSION))
        result.append(inverse_factor.T @ inverse_factor)
    return result


def _svd_rank(values: np.ndarray, *, rcond: float) -> tuple[np.ndarray, int, float, float | None, float | None]:
    singular = np.linalg.svd(values, compute_uv=False)
    if singular.size == 0:
        return singular, 0, 0.0, None, None
    maximum = float(singular[0])
    tolerance = max(float(rcond) * maximum, np.finfo(np.float64).eps)
    rank = int(np.count_nonzero(singular > tolerance))
    full_condition = None
    retained_condition = None
    if rank:
        retained_condition = float(maximum / singular[rank - 1])
    if rank == values.shape[0] and rank == values.shape[1]:
        full_condition = retained_condition
    return singular, rank, tolerance, full_condition, retained_condition


def _pinv(values: np.ndarray, *, tolerance: float) -> np.ndarray:
    left, singular, right = np.linalg.svd(values, full_matrices=False)
    selected = singular > tolerance
    if not np.any(selected):
        return np.zeros(values.T.shape, dtype=np.float64)
    return (right[selected].T / singular[selected]) @ left[:, selected].T


def _correlation(covariance: np.ndarray) -> np.ndarray:
    diagonal = np.diag(covariance)
    result = np.full(covariance.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(diagonal) & (diagonal > 0.0)
    if np.any(valid):
        denominator = np.sqrt(np.outer(diagonal[valid], diagonal[valid]))
        result[np.ix_(valid, valid)] = covariance[np.ix_(valid, valid)] / denominator
        result[np.ix_(valid, valid)] = np.clip(result[np.ix_(valid, valid)], -1.0, 1.0)
    return result


def parameter_values_from_station_transforms(
    parameter_specs: Sequence[Mapping[str, object]],
    transforms: Mapping[int | str, Sequence[float]],
) -> dict[str, float]:
    """Read named native-unit parameters from a payload transform mapping."""
    require_station_scope(parameter_specs)
    result: dict[str, float] = {}
    for spec in parameter_specs:
        try:
            name = str(spec["name"])
            station = int(spec["station_id"])
            component = str(spec["component"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("alignment parameter spec is incomplete") from error
        if not name or component not in COMPONENT_INDEX_AND_PAYLOAD_SCALE:
            raise ValueError("alignment parameter spec has an invalid name or component")
        raw_transform = transforms.get(station, transforms.get(str(station)))
        values = np.asarray(raw_transform, dtype=np.float64)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError(f"station {station} has no finite six-component transform")
        index, scale = COMPONENT_INDEX_AND_PAYLOAD_SCALE[component]
        result[name] = float(values[index] * scale)
    if len(result) != len(parameter_specs):
        raise ValueError("alignment parameter specs contain duplicate names")
    return result


def station_transforms_with_parameter_values(
    parameter_specs: Sequence[Mapping[str, object]],
    base_transforms: Mapping[int | str, Sequence[float]],
    parameter_values: Mapping[str, float],
) -> dict[str, list[float]]:
    """Apply named native-unit parameters to a complete rigid payload map.

    ``base_transforms`` is normally the current physical anchor payload.  The
    function changes only components explicitly admitted by the finite-
    difference plan, preserves all reference-station components verbatim, and
    converts mrad reporting units back to the rad units required by
    ``/Tracker/Align``.  It is deliberately a payload-plan helper, not a
    coordinate transformation of exported tracklets.
    """
    require_station_scope(parameter_specs)
    result: dict[str, list[float]] = {}
    for raw_station, raw_transform in base_transforms.items():
        try:
            station = int(raw_station)
        except (TypeError, ValueError) as error:
            raise ValueError("base station transform has an invalid station key") from error
        values = np.asarray(raw_transform, dtype=np.float64)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError(f"base station {station} has no finite six-component transform")
        if str(station) in result:
            raise ValueError(f"base transforms repeat station {station}")
        result[str(station)] = [float(value) for value in values]
    if not result:
        raise ValueError("base transforms cannot be empty")
    expected = {str(spec.get("name", "")) for spec in parameter_specs}
    if not expected or "" in expected or len(expected) != len(parameter_specs):
        raise ValueError("parameter specs require unique non-empty names")
    if set(parameter_values) != expected:
        raise ValueError("parameter_values must specify exactly the configured named parameters")
    for spec in parameter_specs:
        try:
            name = str(spec["name"])
            station = int(spec["station_id"])
            component = str(spec["component"])
            value = float(parameter_values[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("alignment parameter update is incomplete/non-numeric") from error
        if component not in COMPONENT_INDEX_AND_PAYLOAD_SCALE or not math.isfinite(value):
            raise ValueError(f"alignment parameter '{name}' has an invalid component or value")
        if str(station) not in result:
            raise ValueError(f"alignment parameter '{name}' references absent station {station}")
        index, scale = COMPONENT_INDEX_AND_PAYLOAD_SCALE[component]
        result[str(station)][index] = float(value / scale)
    return result


def _copy_six_component_map(
    transforms: Mapping[int | str, Sequence[float]],
    *,
    label: str,
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for raw_key, raw_transform in transforms.items():
        values = np.asarray(raw_transform, dtype=np.float64)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError(f"{label} '{raw_key}' has no finite six-component transform")
        key = str(raw_key)
        if key in result:
            raise ValueError(f"{label} repeats key {key}")
        result[key] = [float(value) for value in values]
    return result


def _nested_layer_transforms(
    transforms: Mapping[object, object],
) -> dict[str, dict[str, list[float]]]:
    """Normalise station -> layer -> [dx, dy, dz, rx, ry, rz] maps."""
    result: dict[str, dict[str, list[float]]] = {}
    for raw_station, raw_layers in transforms.items():
        station = int(raw_station)
        if station != IFT_STATION_ID:
            raise ValueError(f"layer transforms currently admit only IFT station {IFT_STATION_ID}")
        if not isinstance(raw_layers, Mapping):
            raise ValueError(f"layer transforms for station {station} must map layer ids to six-vectors")
        layers = _copy_six_component_map(raw_layers, label=f"station {station} layer")
        expected = {str(layer) for layer in IFT_LAYER_IDS}
        if set(layers) != expected:
            raise ValueError(f"station {station} must declare layer transforms for {sorted(expected)}")
        result[str(station)] = layers
    if str(IFT_STATION_ID) not in result:
        raise ValueError("layer transforms must include IFT station 0")
    return result


def parameter_values_from_payload(
    parameter_specs: Sequence[Mapping[str, object]],
    station_transforms: Mapping[int | str, Sequence[float]],
    layer_transforms: Mapping[object, object],
) -> dict[str, float]:
    """Read station and IFT-layer parameters from a hierarchy payload."""
    stations = _copy_six_component_map(station_transforms, label="station")
    layers = _nested_layer_transforms(layer_transforms)
    result: dict[str, float] = {}
    for spec in parameter_specs:
        try:
            name = str(spec["name"])
            station = int(spec["station_id"])
            component = str(spec["component"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("alignment parameter spec is incomplete") from error
        if not name or component not in COMPONENT_INDEX_AND_PAYLOAD_SCALE:
            raise ValueError("alignment parameter spec has an invalid name or component")
        index, scale = COMPONENT_INDEX_AND_PAYLOAD_SCALE[component]
        scope = spec_scope(spec)
        if scope == "station":
            values = np.asarray(stations.get(str(station)), dtype=np.float64)
            if values.shape != (6,):
                raise ValueError(f"station {station} has no finite six-component transform")
            result[name] = float(values[index] * scale)
            continue
        layer_map = layers.get(str(station))
        if layer_map is None:
            raise ValueError(f"station {station} has no layer transforms")
        if scope == "contrast":
            if "0" not in layer_map or "2" not in layer_map:
                raise ValueError(f"station {station} lacks outer IFT layers for contrast readout")
            result[name] = 0.5 * (
                float(layer_map["0"][index] * scale) - float(layer_map["2"][index] * scale)
            )
            continue
        layer = int(spec["layer_id"])
        if str(layer) not in layer_map:
            raise ValueError(f"station {station} layer {layer} has no finite six-component transform")
        result[name] = float(layer_map[str(layer)][index] * scale)
    if len(result) != len(parameter_specs):
        raise ValueError("alignment parameter specs contain duplicate names")
    return result


def payload_transforms_with_parameter_values(
    parameter_specs: Sequence[Mapping[str, object]],
    base_station_transforms: Mapping[int | str, Sequence[float]],
    base_layer_transforms: Mapping[object, object],
    parameter_values: Mapping[str, float],
) -> tuple[dict[str, list[float]], dict[str, dict[str, list[float]]]]:
    """Apply named native-unit parameters to station and IFT-layer payloads."""
    stations = _copy_six_component_map(base_station_transforms, label="base station")
    layers = _nested_layer_transforms(base_layer_transforms)
    expected = {str(spec.get("name", "")) for spec in parameter_specs}
    if not expected or "" in expected or len(expected) != len(parameter_specs):
        raise ValueError("parameter specs require unique non-empty names")
    if set(parameter_values) != expected:
        raise ValueError("parameter_values must specify exactly the configured named parameters")
    for spec in parameter_specs:
        try:
            name = str(spec["name"])
            station = int(spec["station_id"])
            component = str(spec["component"])
            value = float(parameter_values[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("alignment parameter update is incomplete/non-numeric") from error
        if component not in COMPONENT_INDEX_AND_PAYLOAD_SCALE or not math.isfinite(value):
            raise ValueError(f"alignment parameter '{name}' has an invalid component or value")
        index, scale = COMPONENT_INDEX_AND_PAYLOAD_SCALE[component]
        native = float(value / scale)
        scope = spec_scope(spec)
        if scope == "station":
            if str(station) not in stations:
                raise ValueError(f"alignment parameter '{name}' references absent station {station}")
            stations[str(station)][index] = native
            continue
        if str(station) not in layers:
            raise ValueError(f"alignment parameter '{name}' references absent station {station} layers")
        if scope == "contrast":
            layer_map = layers[str(station)]
            if "0" not in layer_map or "1" not in layer_map or "2" not in layer_map:
                raise ValueError(f"contrast parameter '{name}' requires IFT layers 0, 1, and 2")
            layer_map["0"][index] = native
            layer_map["2"][index] = -native
            continue
        layer = int(spec["layer_id"])
        if str(layer) not in layers[str(station)]:
            raise ValueError(f"alignment parameter '{name}' references absent station {station} layer {layer}")
        layers[str(station)][str(layer)][index] = native
    return stations, layers


def solve_physical_finite_difference(
    nominal_residual: object,
    positive_residual: object,
    negative_residual: object,
    observed_residual: object,
    covariance: object,
    *,
    parameter_names: Sequence[str],
    positive_values: Sequence[float],
    negative_values: Sequence[float],
    parameter_scales: Sequence[float],
    prior_sigma_native: Sequence[float] | None = None,
    rcond: float = 1.0e-10,
) -> PhysicalJacobianFit:
    """Recover several rigid alignment parameters from physical probes.

    ``positive_residual[p]`` and ``negative_residual[p]`` must come from
    separate, pure central finite-difference conditions for parameter ``p``.
    ``observed_residual`` is a distinct joint physical payload, normally a
    held-out closure point.  The covariance remains a deterministic nominal
    WLS weight for paired refit responses, not an independent covariance for
    a subtraction of reconstructions of the same clusters.
    """
    names = tuple(str(name) for name in parameter_names)
    parameters = len(names)
    if not parameters or len(set(names)) != parameters or any(not name for name in names):
        raise ValueError("parameter_names must be a non-empty unique sequence")
    if not np.isfinite(rcond) or rcond <= 0.0 or rcond >= 1.0:
        raise ValueError("rcond must be finite and lie in (0, 1)")
    nominal = _responses(nominal_residual, label="nominal_residual")
    pairs = nominal.shape[0]
    if pairs < 1:
        raise ValueError("physical finite-difference closure requires at least one pair")
    positive = _responses(positive_residual, label="positive_residual", parameters=parameters)
    negative = _responses(negative_residual, label="negative_residual", parameters=parameters)
    observed = _responses(observed_residual, label="observed_residual")
    if positive.shape[1] != pairs or negative.shape[1] != pairs or observed.shape[0] != pairs:
        raise ValueError("physical finite-difference responses have inconsistent pair counts")
    plus = np.asarray(positive_values, dtype=np.float64)
    minus = np.asarray(negative_values, dtype=np.float64)
    scales = np.asarray(parameter_scales, dtype=np.float64)
    if plus.shape != (parameters,) or minus.shape != (parameters,) or scales.shape != (parameters,):
        raise ValueError("parameter values and scales must have one entry per parameter")
    if not np.isfinite(plus).all() or not np.isfinite(minus).all() or not np.isfinite(scales).all() or np.any(scales <= 0.0):
        raise ValueError("parameter values/scales must be finite and scales positive")
    denominator = plus - minus
    if np.any(np.isclose(denominator, 0.0, rtol=0.0, atol=0.0)):
        raise ValueError("each finite-difference probe pair must have distinct parameter values")
    priors = None
    if prior_sigma_native is not None:
        priors = np.asarray(prior_sigma_native, dtype=np.float64)
        if priors.shape != (parameters,):
            raise ValueError("prior_sigma_native must have one value per parameter")
        finite_prior = np.isfinite(priors)
        if np.any(finite_prior & (priors <= 0.0)):
            raise ValueError("finite prior_sigma_native values must be positive")
        if not np.any(finite_prior):
            priors = None

    inverse_covariance = _inverse_covariances(covariance, pairs)
    # [parameter, pair, residual] -> [pair, residual, parameter]
    derivative = np.moveaxis((positive - negative) / denominator[:, None, None], 0, -1)
    response = observed - nominal
    normal_native = np.zeros((parameters, parameters), dtype=np.float64)
    rhs_native = np.zeros(parameters, dtype=np.float64)
    for row, inverse in enumerate(inverse_covariance):
        jacobian = derivative[row]
        normal_native += jacobian.T @ inverse @ jacobian
        rhs_native += jacobian.T @ inverse @ response[row]
    normal_native = 0.5 * (normal_native + normal_native.T)
    scale_matrix = np.diag(scales)
    normal_scaled = scale_matrix @ normal_native @ scale_matrix
    normal_scaled = 0.5 * (normal_scaled + normal_scaled.T)
    rhs_scaled = scale_matrix @ rhs_native
    data_singular, data_rank, data_tolerance, full_condition, retained_condition = _svd_rank(
        normal_scaled, rcond=rcond
    )
    # The observable subspace reports whether an individual native parameter
    # is constrained, even if the full block has a gauge-like degeneracy.
    _, _, right = np.linalg.svd(normal_scaled, full_matrices=False)
    selected = data_singular > data_tolerance
    projector = right[selected].T @ right[selected] if np.any(selected) else np.zeros_like(normal_scaled)
    observability = np.clip(np.diag(projector), 0.0, 1.0)

    fit_normal = np.array(normal_scaled, copy=True)
    if priors is not None:
        finite_prior = np.isfinite(priors)
        contribution = np.zeros(parameters, dtype=np.float64)
        contribution[finite_prior] = 1.0 / np.square(priors[finite_prior] / scales[finite_prior])
        fit_normal += np.diag(contribution)
    fit_normal = 0.5 * (fit_normal + fit_normal.T)
    fit_singular, fit_rank, fit_tolerance, _, _ = _svd_rank(fit_normal, rcond=rcond)
    inverse_fit_scaled = _pinv(fit_normal, tolerance=fit_tolerance)
    recovered_scaled = inverse_fit_scaled @ rhs_scaled
    recovered = scales * recovered_scaled
    covariance_native = scale_matrix @ inverse_fit_scaled @ scale_matrix
    covariance_native = 0.5 * (covariance_native + covariance_native.T)
    predicted = np.einsum("nrp,p->nr", derivative, recovered, optimize=True)
    residual = response - predicted
    chi2 = float(sum(row @ inverse @ row for row, inverse in zip(residual, inverse_covariance)))
    if not np.isfinite(chi2):
        raise ValueError("physical finite-difference response chi2 is non-finite")
    return PhysicalJacobianFit(
        parameter_names=names,
        parameter_scales=np.asarray(scales, dtype=np.float64),
        derivative_native=np.asarray(derivative, dtype=np.float64),
        response=np.asarray(response, dtype=np.float64),
        predicted_response=np.asarray(predicted, dtype=np.float64),
        residual_response=np.asarray(residual, dtype=np.float64),
        recovered_parameters=np.asarray(recovered, dtype=np.float64),
        covariance_native=np.asarray(covariance_native, dtype=np.float64),
        correlation_native=_correlation(covariance_native),
        normal_matrix_native=np.asarray(normal_native, dtype=np.float64),
        normal_matrix_scaled=np.asarray(normal_scaled, dtype=np.float64),
        right_hand_side_native=np.asarray(rhs_native, dtype=np.float64),
        right_hand_side_scaled=np.asarray(rhs_scaled, dtype=np.float64),
        data_singular_values=np.asarray(data_singular, dtype=np.float64),
        fit_singular_values=np.asarray(fit_singular, dtype=np.float64),
        normal_matrix_rank=int(data_rank),
        fit_matrix_rank=int(fit_rank),
        normal_matrix_condition_number=full_condition,
        identifiable_subspace_condition_number=retained_condition,
        parameter_observability_fraction=np.asarray(observability, dtype=np.float64),
        response_chi2=chi2,
        response_ndof=int(RESIDUAL_DIMENSION * pairs - data_rank),
        used_pairs=int(pairs),
        prior_sigma_native=None if priors is None else np.asarray(priors, dtype=np.float64),
    )
