"""IFT station/layer hierarchy: Calypso L2 keys, gauges, and physical readout.

Station-level 5-DoF remains the common-mode rigid transform.  Layer corrections
are internal to IFT (station 0, planes 0--2) and must not silently reuse a
station payload slot.  Calypso ``TrackerAlignDBTool`` stores a plane transform
under the two-digit key ``f"{station}{layer}"`` (level-2 ``/Tracker/Align/Planes``).

The canonical internal basis is the zero-common-mode outer contrast.
Hierarchical alignment V1 floats station 5-DoF and ``C_dx`` as separate
blocks, never as one six-parameter Newton step:

* station 5-DoF ``dx/dy/rx/ry/rz`` with survey-constrained ``dz``
* ``C_dx = (dx_L0 - dx_L2) / 2``

``C_rx`` is identifiable in isolation but is not a hierarchy update DoF.

Layer 1 is not floated.  Equal-weight ``sum_to_zero`` writes the same
physical family ``L = [+C, 0, -C]``.  ``reference_layer`` is the same
additive geometry only after a compensating station transform
``S = +C``, ``L = [0, -C, -2C]``.  With the station frozen it is a
different physical constraint and is retained only as a negative control,
not as a gauge cross-check.  Compare ``layer_0 - layer_2``, not labels.

See ``alignment.gauge_equivalence`` for the payload-level map and the
Calypso detector-element composition (plane-*z* conjugation).

Neither gauge is a physical measurement of ``dz``.  Layer ``dz`` is not part of
this hierarchy step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


IFT_STATION_ID = 0
IFT_LAYER_IDS = (0, 1, 2)
FREE_COMPONENTS = ("dx_mm", "dy_mm", "rx_mrad", "ry_mrad", "rz_mrad")
GAUGE_CHOICES = ("sum_to_zero", "reference_layer")
CONTRAST_CHOICE = "outer_contrast"
CANONICAL_INTERNAL_BASIS = CONTRAST_CHOICE
HIERARCHY_FIT_CHOICES = GAUGE_CHOICES + (CONTRAST_CHOICE,)
ZERO_COMMON_MODE_CHOICES = (CONTRAST_CHOICE, "sum_to_zero")
FROZEN_STATION_NEGATIVE_CONTROL = "reference_layer"
CONTRAST_COMPONENTS = ("dx_mm", "rx_mrad", "ry_mrad")
CONTRAST_SCOPE = "contrast"
SPEC_SCOPES = ("station", "layer", CONTRAST_SCOPE)
ADMITTED_CONTRAST_COMPONENTS = ("dx_mm", "rx_mrad")
NEAR_DEGENERACY_COSINE = 0.9

_LAYER_KEY_STATIONS = (0, 1, 2, 3)
_LAYER_KEY_LAYERS = (0, 1, 2)


def calypso_layer_key(station: int, layer: int) -> str:
    """Return the numeric L2 key used by ``TrackerAlignDBTool``."""
    if int(station) not in _LAYER_KEY_STATIONS or int(layer) not in _LAYER_KEY_LAYERS:
        raise ValueError(f"unsupported Calypso layer key station/layer {station}/{layer}")
    return f"{int(station)}{int(layer)}"


def parse_calypso_layer_key(key: str) -> tuple[int, int]:
    """Parse a two-digit L2 alignment constant key into ``(station, layer)``."""
    if not isinstance(key, str) or len(key) != 2 or not key.isdigit():
        raise ValueError(f"not a Calypso L2 layer key: {key!r}")
    station = int(key[0])
    layer = int(key[1])
    if station not in _LAYER_KEY_STATIONS or layer not in _LAYER_KEY_LAYERS:
        raise ValueError(f"unsupported Calypso layer key {key!r}")
    return station, layer


def is_calypso_layer_key(key: object) -> bool:
    if not isinstance(key, str):
        return False
    try:
        parse_calypso_layer_key(key)
    except ValueError:
        return False
    return True


def spec_scope(spec: Mapping[str, object]) -> str:
    scope = str(spec.get("scope", "station"))
    if scope not in SPEC_SCOPES:
        raise ValueError(f"alignment parameter '{spec.get('name')}' has unsupported scope '{scope}'")
    return scope


def is_contrast_spec(spec: Mapping[str, object]) -> bool:
    return spec_scope(spec) == CONTRAST_SCOPE


def contrast_parameter_name(component: str) -> str:
    """Return the explicit outer-contrast coordinate for one IFT component."""
    names = {"dx_mm": "C_dx", "rx_mrad": "C_rx", "ry_mrad": "C_ry"}
    if component not in names:
        raise ValueError(f"no outer-contrast coordinate for component '{component}'")
    return names[component]


def contrast_layer_six_vectors(c_dx_mm: float, c_rx_mrad: float) -> dict[str, list[float]]:
    """Expand canonical contrast coordinates into IFT layer payload six-vectors.

    Payload units are millimetres and radians.  Layer 1 and every unused
    component stay identically zero.  The station six-vector is not part of
    this expansion and must remain zero separately.
    """
    rx_rad = float(c_rx_mrad) / 1.0e3
    dx = float(c_dx_mm)
    return {
        "0": [dx, 0.0, 0.0, rx_rad, 0.0, 0.0],
        "1": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "2": [-dx, 0.0, 0.0, -rx_rad, 0.0, 0.0],
    }


def require_station_scope(parameter_specs: Sequence[Mapping[str, object]]) -> None:
    """Refuse to let a layer condition ride on a station-only helper."""
    for spec in parameter_specs:
        name = str(spec.get("name", ""))
        if spec_scope(spec) != "station" or "layer_id" in spec:
            raise ValueError(
                f"alignment parameter '{name}' is not station-scoped; "
                "do not map a layer/module condition onto a station payload"
            )


@dataclass(frozen=True)
class HierarchyIndex:
    """Index maps from the joint station+layer parameter vector."""

    names: tuple[str, ...]
    station_by_component: dict[str, int]
    layer_by_component: dict[str, dict[int, int]]

    @property
    def n_parameters(self) -> int:
        return len(self.names)


def hierarchy_index(parameter_specs: Sequence[Mapping[str, object]]) -> HierarchyIndex:
    """Classify joint station/layer specs for gauge reduction."""
    names: list[str] = []
    station_by_component: dict[str, int] = {}
    layer_by_component: dict[str, dict[int, int]] = {component: {} for component in FREE_COMPONENTS}
    seen: set[str] = set()
    for spec in parameter_specs:
        name = str(spec.get("name", ""))
        if not name or name in seen:
            raise ValueError("hierarchy specs require unique non-empty names")
        seen.add(name)
        component = str(spec.get("component", ""))
        if component not in FREE_COMPONENTS:
            raise ValueError(f"hierarchy parameter '{name}' uses non-admitted component '{component}'")
        station = int(spec["station_id"])
        if station != IFT_STATION_ID:
            raise ValueError(f"hierarchy parameter '{name}' must belong to IFT station {IFT_STATION_ID}")
        index = len(names)
        names.append(name)
        scope = spec_scope(spec)
        if scope == CONTRAST_SCOPE:
            raise ValueError(
                f"hierarchy parameter '{name}' is already in outer-contrast coordinates; "
                "do not pass C_dx/C_rx through reduce_gauge or treat reference_layer as equivalent"
            )
        if scope == "station":
            if "layer_id" in spec:
                raise ValueError(f"station parameter '{name}' must not declare layer_id")
            if component in station_by_component:
                raise ValueError(f"duplicate station component '{component}'")
            station_by_component[component] = index
            continue
        layer = int(spec["layer_id"])
        if layer not in IFT_LAYER_IDS:
            raise ValueError(f"layer parameter '{name}' has unsupported layer_id {layer}")
        if layer in layer_by_component[component]:
            raise ValueError(f"duplicate layer {layer} component '{component}'")
        layer_by_component[component][layer] = index
    if station_by_component:
        missing_station = [component for component in FREE_COMPONENTS if component not in station_by_component]
        if missing_station:
            raise ValueError("hierarchy is missing station common-mode component(s): " + ", ".join(missing_station))
    elif not any(layer_by_component[component] for component in FREE_COMPONENTS):
        raise ValueError("hierarchy has neither station nor layer parameters")
    for component, layers in layer_by_component.items():
        if not layers:
            continue
        missing = [layer for layer in IFT_LAYER_IDS if layer not in layers]
        if missing:
            raise ValueError(
                f"hierarchy component '{component}' has a partial layer set {sorted(layers)}; "
                "supply all three IFT layers or none"
            )
    return HierarchyIndex(
        names=tuple(names),
        station_by_component=station_by_component,
        layer_by_component=layer_by_component,
    )


def normalize_layer_weights(weights: Mapping[int, float] | Sequence[float] | None) -> np.ndarray:
    """Return three positive finite IFT-layer weights that sum to one."""
    if weights is None:
        values = np.ones(len(IFT_LAYER_IDS), dtype=np.float64)
    elif isinstance(weights, Mapping):
        values = np.asarray([float(weights[layer]) for layer in IFT_LAYER_IDS], dtype=np.float64)
    else:
        values = np.asarray(list(weights), dtype=np.float64)
        if values.shape != (len(IFT_LAYER_IDS),):
            raise ValueError("layer weights must have one entry per IFT layer")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("layer weights must be positive and finite")
    return values / values.sum()


@dataclass(frozen=True)
class GaugeReduction:
    """Reduced parameter basis after an explicit hierarchy gauge."""

    choice: str
    names: tuple[str, ...]
    keep_indices: tuple[int, ...]
    dropped_names: tuple[str, ...]
    dropped_indices: tuple[int, ...]
    column_transform: np.ndarray
    layer_weights: tuple[float, float, float]
    reference_layer: int | None
    constraint: str

    def apply_derivative(self, derivative: np.ndarray) -> np.ndarray:
        """Map ``[pairs, residual, n_full]`` Jacobian columns into the gauged basis."""
        values = np.asarray(derivative, dtype=np.float64)
        if values.ndim != 3 or values.shape[2] != self.column_transform.shape[0]:
            raise ValueError("derivative does not match the gauge column transform")
        return values @ self.column_transform

    def apply_vector(self, values: np.ndarray) -> np.ndarray:
        """Project a full vector that already satisfies the gauge onto reduced coordinates."""
        full = np.asarray(values, dtype=np.float64)
        if full.shape != (self.column_transform.shape[0],):
            raise ValueError("parameter vector does not match the ungauged dimension")
        return np.asarray([full[index] for index in self.keep_indices], dtype=np.float64)


def _sum_to_zero_transform(
    index: HierarchyIndex,
    weights: np.ndarray,
    *,
    dropped_layer: int,
) -> tuple[np.ndarray, list[int], list[int]]:
    dropped_weight = float(weights[list(IFT_LAYER_IDS).index(dropped_layer)])
    keep: list[int] = []
    dropped: list[int] = []
    columns: list[np.ndarray] = []
    identity = np.eye(index.n_parameters, dtype=np.float64)
    for component in FREE_COMPONENTS:
        layers = index.layer_by_component[component]
        if component in index.station_by_component:
            station_index = index.station_by_component[component]
            keep.append(station_index)
            columns.append(identity[:, station_index])
        elif not layers:
            continue
        if not layers:
            continue
        dropped_index = layers[dropped_layer]
        dropped.append(dropped_index)
        for layer in IFT_LAYER_IDS:
            if layer == dropped_layer:
                continue
            source = layers[layer]
            column = identity[:, source].copy()
            layer_weight = float(weights[list(IFT_LAYER_IDS).index(layer)])
            column[dropped_index] = -layer_weight / dropped_weight
            keep.append(source)
            columns.append(column)
    return np.column_stack(columns), keep, dropped


def _reference_layer_transform(
    index: HierarchyIndex,
    *,
    reference_layer: int,
) -> tuple[np.ndarray, list[int], list[int]]:
    keep: list[int] = []
    dropped: list[int] = []
    identity = np.eye(index.n_parameters, dtype=np.float64)
    columns: list[np.ndarray] = []
    for component in FREE_COMPONENTS:
        layers = index.layer_by_component[component]
        if component in index.station_by_component:
            station_index = index.station_by_component[component]
            keep.append(station_index)
            columns.append(identity[:, station_index])
        elif not layers:
            continue
        if not layers:
            continue
        dropped.append(layers[reference_layer])
        for layer in IFT_LAYER_IDS:
            if layer == reference_layer:
                continue
            source = layers[layer]
            keep.append(source)
            columns.append(identity[:, source])
    return np.column_stack(columns), keep, dropped


def _outer_contrast_transform(
    index: HierarchyIndex,
) -> tuple[np.ndarray, list[int], list[int], tuple[str, ...]]:
    """Map each admitted component onto C=(outer0-outer2)/2 with layer1 and common mode fixed."""
    keep: list[int] = []
    dropped: list[int] = []
    columns: list[np.ndarray] = []
    names: list[str] = []
    for component in FREE_COMPONENTS:
        if component in index.station_by_component:
            dropped.append(index.station_by_component[component])
        layers = index.layer_by_component[component]
        if not layers:
            continue
        if component not in CONTRAST_COMPONENTS:
            dropped.extend(layers[layer] for layer in IFT_LAYER_IDS)
            continue
        if set(layers) != set(IFT_LAYER_IDS):
            raise ValueError(f"outer_contrast component '{component}' requires all three IFT layers")
        column = np.zeros(index.n_parameters, dtype=np.float64)
        column[layers[0]] = 1.0
        column[layers[2]] = -1.0
        columns.append(column)
        keep.append(layers[0])
        dropped.extend((layers[1], layers[2]))
        names.append(contrast_parameter_name(component))
    if not columns:
        raise ValueError("outer_contrast requires at least one layer dx/rx/ry component")
    return np.column_stack(columns), keep, dropped, tuple(names)


def reduce_gauge(
    parameter_specs: Sequence[Mapping[str, object]],
    *,
    choice: str,
    layer_weights: Mapping[int, float] | Sequence[float] | None = None,
    reference_layer: int = 0,
    dropped_layer: int = 2,
) -> GaugeReduction:
    """Build the explicit gauge that separates station common-mode from layers."""
    if choice not in HIERARCHY_FIT_CHOICES:
        raise ValueError(f"unsupported gauge choice '{choice}'")
    index = hierarchy_index(parameter_specs)
    weights = normalize_layer_weights(layer_weights)
    if choice == CONTRAST_CHOICE:
        transform, keep, dropped, keep_names = _outer_contrast_transform(index)
        return GaugeReduction(
            choice=choice,
            names=keep_names,
            keep_indices=tuple(keep),
            dropped_names=tuple(index.names[item] for item in dropped),
            dropped_indices=tuple(dropped),
            column_transform=transform,
            layer_weights=tuple(float(value) for value in weights),
            reference_layer=None,
            constraint=(
                "station rigid transform frozen; IFT layer common mode fixed at 0; "
                "layer 1 frozen; fit C=(layer0-layer2)/2"
            ),
        )
    if choice == "sum_to_zero":
        if dropped_layer not in IFT_LAYER_IDS:
            raise ValueError(f"unsupported dropped layer {dropped_layer}")
        transform, keep, dropped = _sum_to_zero_transform(index, weights, dropped_layer=dropped_layer)
        dropped_names = tuple(index.names[item] for item in dropped)
        keep_names = tuple(index.names[item] for item in keep)
        constraint = (
            "station rigid transform carries common mode; "
            f"coverage-weighted IFT layer corrections sum to zero with dropped layer {dropped_layer}"
        )
        return GaugeReduction(
            choice=choice,
            names=keep_names,
            keep_indices=tuple(keep),
            dropped_names=dropped_names,
            dropped_indices=tuple(dropped),
            column_transform=transform,
            layer_weights=tuple(float(value) for value in weights),
            reference_layer=None,
            constraint=constraint,
        )
    if reference_layer not in IFT_LAYER_IDS:
        raise ValueError(f"unsupported reference layer {reference_layer}")
    transform, keep, dropped = _reference_layer_transform(index, reference_layer=reference_layer)
    return GaugeReduction(
        choice=choice,
        names=tuple(index.names[item] for item in keep),
        keep_indices=tuple(keep),
        dropped_names=tuple(index.names[item] for item in dropped),
        dropped_indices=tuple(dropped),
        column_transform=transform,
        layer_weights=tuple(float(value) for value in weights),
        reference_layer=int(reference_layer),
        constraint=(
            "station rigid transform carries common mode; "
            f"IFT layer {reference_layer} is the fixed reference plane"
        ),
    )


def expand_gauged_parameters(
    gauged_values: Mapping[str, float],
    reduction: GaugeReduction,
    parameter_specs: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    """Lift a gauged solution back to the full station+layer vector."""
    index = hierarchy_index(parameter_specs)
    if set(gauged_values) != set(reduction.names):
        raise ValueError("gauged values must specify exactly the reduced parameter names")
    reduced = np.asarray([float(gauged_values[name]) for name in reduction.names], dtype=np.float64)
    full = reduction.column_transform @ reduced
    return {name: float(value) for name, value in zip(index.names, full)}


def project_to_gauge(
    full_values: Mapping[str, float],
    reduction: GaugeReduction,
    parameter_specs: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    """Move layer common-mode into the station slot, then drop gauge coordinates.

    This is the comparison convention for the two gauges, not a claim that a
    layer rotation about the plane origin equals a station rotation about the
    global origin.  The Jacobian, not this additive split, decides whether
    those operators are empirically degenerate.
    """
    index = hierarchy_index(parameter_specs)
    split = split_common_and_internal(
        full_values,
        parameter_specs,
        layer_weights=reduction.layer_weights,
    )
    projected: dict[str, float] = {}
    if reduction.choice == "sum_to_zero":
        for component in FREE_COMPONENTS:
            layers = index.layer_by_component[component]
            if component in index.station_by_component:
                station_name = index.names[index.station_by_component[component]]
                projected[station_name] = split["total_common_station_plus_layer_mean"][component]
            if not layers:
                continue
            for layer in IFT_LAYER_IDS:
                name = index.names[layers[layer]]
                projected[name] = split["layer_internal"][f"layer_{layer}"][component]
    elif reduction.choice == "reference_layer":
        reference = reduction.reference_layer
        if reference is None:
            raise ValueError("reference-layer gauge is missing reference_layer")
        for component in FREE_COMPONENTS:
            layers = index.layer_by_component[component]
            if component in index.station_by_component:
                station_name = index.names[index.station_by_component[component]]
                if not layers:
                    projected[station_name] = float(full_values[station_name])
                    continue
                reference_value = float(full_values[index.names[layers[reference]]])
                projected[station_name] = float(full_values[station_name]) + reference_value
            elif not layers:
                continue
            else:
                reference_value = float(full_values[index.names[layers[reference]]])
            if not layers:
                continue
            for layer in IFT_LAYER_IDS:
                name = index.names[layers[layer]]
                projected[name] = (
                    0.0 if layer == reference else float(full_values[name]) - float(full_values[index.names[layers[reference]]])
                )
    elif reduction.choice == CONTRAST_CHOICE:
        for component in CONTRAST_COMPONENTS:
            name = contrast_parameter_name(component)
            if name not in reduction.names:
                continue
            layers = index.layer_by_component[component]
            if set(layers) != set(IFT_LAYER_IDS):
                raise ValueError(f"outer_contrast component '{component}' requires all three IFT layers")
            projected[name] = 0.5 * (
                float(full_values[index.names[layers[0]]]) - float(full_values[index.names[layers[2]]])
            )
    else:
        raise ValueError(f"unsupported gauge choice '{reduction.choice}'")
    return {name: projected[name] for name in reduction.names}


def split_common_and_internal(
    full_values: Mapping[str, float],
    parameter_specs: Sequence[Mapping[str, object]],
    *,
    layer_weights: Mapping[int, float] | Sequence[float] | None = None,
) -> dict[str, dict[str, float]]:
    """Separate station common-mode from coverage-weighted layer deformation.

    For translations the reported common mode is ``station + weighted mean(layer)``.
    That additive split is a readout convention, not a claim that layer and
    station rotations about different origins are the same physical operator.
    """
    index = hierarchy_index(parameter_specs)
    weights = normalize_layer_weights(layer_weights)
    station: dict[str, float] = {component: 0.0 for component in FREE_COMPONENTS}
    layer_mean: dict[str, float] = {component: 0.0 for component in FREE_COMPONENTS}
    internal: dict[str, dict[str, float]] = {
        f"layer_{layer}": {component: 0.0 for component in FREE_COMPONENTS} for layer in IFT_LAYER_IDS
    }
    for component in FREE_COMPONENTS:
        layers = index.layer_by_component[component]
        if component in index.station_by_component:
            station_name = index.names[index.station_by_component[component]]
            station_value = float(full_values[station_name])
        elif not layers:
            continue
        else:
            station_value = 0.0
        station[component] = station_value
        if not layers:
            layer_mean[component] = 0.0
            for layer in IFT_LAYER_IDS:
                internal[f"layer_{layer}"][component] = 0.0
            continue
        layer_values = np.asarray(
            [
                float(full_values[index.names[layers[layer]]])
                for layer in IFT_LAYER_IDS
            ],
            dtype=np.float64,
        )
        mean = float(weights @ layer_values)
        layer_mean[component] = mean
        for layer, value in zip(IFT_LAYER_IDS, layer_values):
            internal[f"layer_{layer}"][component] = float(value - mean)
    return {
        "station_common": station,
        "layer_weighted_mean": layer_mean,
        "total_common_station_plus_layer_mean": {
            component: station[component] + layer_mean[component] for component in FREE_COMPONENTS
        },
        "layer_internal": internal,
    }


def column_cosines(matrix: np.ndarray) -> np.ndarray:
    """Pairwise cosines of normal-matrix columns, NaN where a column is null."""
    normal = np.asarray(matrix, dtype=np.float64)
    if normal.ndim != 2 or normal.shape[0] != normal.shape[1]:
        raise ValueError("column cosines require a square normal matrix")
    diagonal = np.diag(normal)
    result = np.full(normal.shape, np.nan, dtype=np.float64)
    valid = diagonal > 0.0
    if np.any(valid):
        denominator = np.sqrt(np.outer(diagonal[valid], diagonal[valid]))
        result[np.ix_(valid, valid)] = normal[np.ix_(valid, valid)] / denominator
        result[np.ix_(valid, valid)] = np.clip(result[np.ix_(valid, valid)], -1.0, 1.0)
    return result


def hierarchy_degeneracy_pairs(
    names: Sequence[str],
    cosines: np.ndarray,
    parameter_specs: Sequence[Mapping[str, object]],
    *,
    threshold: float = NEAR_DEGENERACY_COSINE,
) -> list[dict[str, object]]:
    """Flag station↔layer and layer↔layer near-collinear Jacobian columns."""
    index = hierarchy_index(parameter_specs)
    if tuple(names) != index.names:
        raise ValueError("degeneracy names do not match hierarchy specs")
    rows: list[dict[str, object]] = []
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            if j <= i:
                continue
            cosine = float(cosines[i, j])
            if not np.isfinite(cosine) or abs(cosine) < threshold:
                continue
            left_spec = parameter_specs[i]
            right_spec = parameter_specs[j]
            relation = "other"
            if spec_scope(left_spec) != spec_scope(right_spec) and left_spec["component"] == right_spec["component"]:
                relation = "station_layer_same_component"
            elif (
                spec_scope(left_spec) == "layer"
                and spec_scope(right_spec) == "layer"
                and left_spec["component"] == right_spec["component"]
            ):
                relation = "layer_layer_same_component"
            rows.append(
                {
                    "pair": [left, right],
                    "response_column_cosine": cosine,
                    "relation": relation,
                    "component": str(left_spec["component"]),
                }
            )
    return rows
