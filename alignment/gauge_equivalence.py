"""Payload-level gauge map and Calypso detector-element composition.

Workbook 40 treated ``sum_to_zero``, ``reference_layer``, and ``outer_contrast``
as interchangeable coordinates of one physical subspace.  That is only true
when the station rigid transform is free to absorb layer common mode.  With
the station six-vector frozen at zero they are not the same family:

* ``outer_contrast`` / equal-weight ``sum_to_zero`` write the zero-common-mode
  outer antisymmetric deformation ``L = [+C, 0, -C]``, ``S = 0``.
* The same additive geometry in the reference-layer chart (layer 0 fixed at
  0) is ``L = [0, -C, -2C]`` **and** a compensating station transform
  ``S = +C``.  Freezing the station drops that compensation, so
  ``reference_layer`` then fits a different physical constraint.

Even with the compensating station restored, Calypso does not add the two
six-vectors.  ``TrackerAlignDBTool`` writes station L1 as a global
``T * Rz * Ry * Rx`` and conjugates plane L2 to the plane's global *z*.
GeoModel then applies each stored delta as a global-frame correction, so the
composed detector-element global delta is

``g_element = g_station * Ad_{T(z_plane)}(g_layer)``.

Translations commute with the *z* conjugation, so the additive map is an
SE(3) identity for ``dx``.  A station ``rx`` rotates about the global origin
while a layer ``rx`` rotates about the plane origin; additive station
compensation therefore does **not** reproduce the same detector-element
transform for relative ``rx``.  Frozen-station ``reference_layer`` is a
negative control of a different constraint, not a gauge cross-check of
``C_rx``.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from alignment.layer_hierarchy import (
    CANONICAL_INTERNAL_BASIS,
    FROZEN_STATION_NEGATIVE_CONTROL,
    IFT_LAYER_IDS,
    IFT_STATION_ID,
    ZERO_COMMON_MODE_CHOICES,
)
from alignment.physical_jacobian import COMPONENT_INDEX_AND_PAYLOAD_SCALE


# Station-0 reconstructed tracklet *z* from the relative-rx held-out content
# audit, together with FASERNU ``LAYERPITCH`` (geomDB ``SCTFASERGENERAL``).
# The millimetre-scale ``rx`` mismatch is dominated by the ~1.86 m lever arm,
# not by the 31.5 mm plane spacing.
IFT_STATION_Z_MM = -1860.1511662696719
IFT_LAYER_PITCH_MM = 31.5
IFT_PLANE_Z_MM = tuple(
    float(IFT_STATION_Z_MM + (int(layer) - 1) * IFT_LAYER_PITCH_MM) for layer in IFT_LAYER_IDS
)

SIX_VECTOR_COMPONENTS = ("dx_mm", "dy_mm", "dz_mm", "rx_mrad", "ry_mrad", "rz_mrad")

_SE3_ATOL = 1.0e-12
_ADDITIVE_ATOL = 1.0e-15


def zero_six() -> tuple[float, float, float, float, float, float]:
    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def native_payload_value(component: str, value: float) -> float:
    """Convert a named hierarchy coordinate into the payload six-vector unit."""
    if component not in COMPONENT_INDEX_AND_PAYLOAD_SCALE:
        raise ValueError(f"unsupported alignment component '{component}'")
    _index, scale = COMPONENT_INDEX_AND_PAYLOAD_SCALE[component]
    return float(value) / float(scale)


def six_vector_with_component(
    component: str,
    value: float,
    *,
    base: Sequence[float] | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Return a payload six-vector with one native-unit component set."""
    packed = list(zero_six() if base is None else base)
    if len(packed) != 6:
        raise ValueError("payload six-vector must have six components")
    if component not in COMPONENT_INDEX_AND_PAYLOAD_SCALE:
        raise ValueError(f"unsupported alignment component '{component}'")
    index, scale = COMPONENT_INDEX_AND_PAYLOAD_SCALE[component]
    packed[index] = float(value) / float(scale)
    return tuple(float(item) for item in packed)


def outer_contrast_payload(
    component: str,
    contrast: float,
) -> dict[str, object]:
    """Zero-common-mode outer antisymmetric deformation in payload coordinates.

    ``contrast`` is ``C = (layer0 - layer2) / 2`` in the hierarchy native unit
    (mm or mrad).  Station is identically zero; layer 1 is identically zero.
    Equal-weight ``sum_to_zero`` writes the same six-vectors.
    """
    plus = six_vector_with_component(component, contrast)
    minus = six_vector_with_component(component, -float(contrast))
    return {
        "representation": "outer_contrast",
        "component": str(component),
        "contrast": float(contrast),
        "station": zero_six(),
        "layers": {0: plus, 1: zero_six(), 2: minus},
        "same_as_equal_weight_sum_to_zero": True,
    }


def reference_layer_additive_equivalent(
    component: str,
    contrast: float,
    *,
    reference_layer: int = 0,
) -> dict[str, object]:
    """Additive chart of the same geometry with ``layer[reference] = 0``.

    Shift every plane correction by ``-C`` and compensate with station ``+C``
    in the same component.  For reference layer 0 and ``L = [+C, 0, -C]``::

        L' = [0, -C, -2C]
        S' = +C

    This is a coordinate change of the additive six-vector readout
    ``station + layer_i``.  It is **not** a claim that a station rotation
    about the global origin equals a plane rotation conjugated to plane *z*.
    """
    if int(reference_layer) not in IFT_LAYER_IDS:
        raise ValueError(f"unsupported reference layer {reference_layer}")
    physical = outer_contrast_payload(component, contrast)
    shift = six_vector_with_component(component, -float(contrast))
    layers = {
        layer: _add_six(physical["layers"][layer], shift) for layer in IFT_LAYER_IDS
    }
    if not np.allclose(layers[int(reference_layer)], 0.0, rtol=0.0, atol=_ADDITIVE_ATOL):
        raise RuntimeError("reference-layer additive map did not fix the reference plane")
    return {
        "representation": "reference_layer_with_station_compensation",
        "component": str(component),
        "contrast": float(contrast),
        "reference_layer": int(reference_layer),
        "station": six_vector_with_component(component, contrast),
        "layers": layers,
        "compensating_station": six_vector_with_component(component, contrast),
    }


def frozen_station_reference_layer_payload(
    component: str,
    contrast: float,
    *,
    reference_layer: int = 0,
) -> dict[str, object]:
    """Same layer chart as the additive equivalent, but with station frozen at 0."""
    compensated = reference_layer_additive_equivalent(
        component, contrast, reference_layer=reference_layer
    )
    return {
        "representation": "reference_layer_frozen_station",
        "component": str(component),
        "contrast": float(contrast),
        "reference_layer": int(reference_layer),
        "station": zero_six(),
        "layers": dict(compensated["layers"]),
        "compensating_station": compensated["compensating_station"],
        "different_physical_family": True,
    }


def additive_element_six(
    station: Sequence[float],
    layer: Sequence[float],
) -> tuple[float, float, float, float, float, float]:
    """Readout convention ``station + layer`` used by ``split_common_and_internal``."""
    return _add_six(station, layer)


def calypso_alignment_matrix(
    dx_mm: float,
    dy_mm: float,
    dz_mm: float,
    rx_rad: float,
    ry_rad: float,
    rz_rad: float,
) -> np.ndarray:
    """Homogeneous ``T * Rz * Ry * Rx`` matching ``TrackerAlignDBTool``."""
    return (
        _translation(float(dx_mm), float(dy_mm), float(dz_mm))
        @ _rotation_z(float(rz_rad))
        @ _rotation_y(float(ry_rad))
        @ _rotation_x(float(rx_rad))
    )


def conjugate_to_plane_z(matrix: np.ndarray, plane_z_mm: float) -> np.ndarray:
    """``T(z) * alignment * T(-z)`` as stored for Calypso L2 planes."""
    shift = _translation(0.0, 0.0, float(plane_z_mm))
    return shift @ np.asarray(matrix, dtype=np.float64) @ _translation(0.0, 0.0, -float(plane_z_mm))


def detector_element_global_delta(
    station: Sequence[float],
    layer: Sequence[float],
    plane_z_mm: float,
) -> np.ndarray:
    """Composed global-frame correction applied to one IFT detector element.

    GeoModel applies a stored global delta ``g`` by the identity
    ``T_new = g * T_nominal``.  Station L1 is stored without extra
    conjugation; plane L2 is stored already conjugated to plane *z*.  Module
    L3 is identity in this hierarchy.  The composed global correction is
    therefore ``g_station * Ad_{T(z)}(g_layer)``.
    """
    g_station = calypso_alignment_matrix(*tuple(float(value) for value in station))
    g_layer = calypso_alignment_matrix(*tuple(float(value) for value in layer))
    return g_station @ conjugate_to_plane_z(g_layer, plane_z_mm)


def transform_point(matrix: np.ndarray, point_mm: Sequence[float]) -> np.ndarray:
    homogeneous = np.asarray([float(point_mm[0]), float(point_mm[1]), float(point_mm[2]), 1.0], dtype=np.float64)
    mapped = np.asarray(matrix, dtype=np.float64) @ homogeneous
    return mapped[:3]


def matrices_close(left: np.ndarray, right: np.ndarray, *, atol: float = _SE3_ATOL) -> bool:
    return bool(np.allclose(left, right, rtol=0.0, atol=float(atol)))


def additive_elements_match(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    atol: float = _ADDITIVE_ATOL,
) -> bool:
    left_station = np.asarray(left["station"], dtype=np.float64)
    right_station = np.asarray(right["station"], dtype=np.float64)
    for layer in IFT_LAYER_IDS:
        left_element = np.asarray(additive_element_six(left_station, left["layers"][layer]), dtype=np.float64)
        right_element = np.asarray(additive_element_six(right_station, right["layers"][layer]), dtype=np.float64)
        if not np.allclose(left_element, right_element, rtol=0.0, atol=float(atol)):
            return False
    return True


def composed_elements_match(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    plane_z_mm: Sequence[float] = IFT_PLANE_Z_MM,
    atol: float = _SE3_ATOL,
) -> bool:
    for layer, plane_z in zip(IFT_LAYER_IDS, plane_z_mm):
        left_delta = detector_element_global_delta(left["station"], left["layers"][layer], float(plane_z))
        right_delta = detector_element_global_delta(right["station"], right["layers"][layer], float(plane_z))
        if not matrices_close(left_delta, right_delta, atol=atol):
            return False
    return True


def element_mismatch_table(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    plane_z_mm: Sequence[float] = IFT_PLANE_Z_MM,
    sample_y_mm: float = 50.0,
) -> dict[str, dict[str, object]]:
    """Per-plane SE(3) and sample-point differences for the audit report."""
    rows: dict[str, dict[str, object]] = {}
    for layer, plane_z in zip(IFT_LAYER_IDS, plane_z_mm):
        left_delta = detector_element_global_delta(left["station"], left["layers"][layer], float(plane_z))
        right_delta = detector_element_global_delta(right["station"], right["layers"][layer], float(plane_z))
        origin = (0.0, 0.0, float(plane_z))
        off_axis = (0.0, float(sample_y_mm), float(plane_z))
        translation = left_delta[:3, 3] - right_delta[:3, 3]
        rows[f"layer_{layer}"] = {
            "plane_z_mm": float(plane_z),
            "matrices_equal": matrices_close(left_delta, right_delta),
            "max_abs_matrix_difference": float(np.max(np.abs(left_delta - right_delta))),
            "translation_difference_mm": [float(value) for value in translation],
            "origin_image_difference_mm": [
                float(value) for value in (transform_point(left_delta, origin) - transform_point(right_delta, origin))
            ],
            "off_axis_image_difference_mm": [
                float(value)
                for value in (transform_point(left_delta, off_axis) - transform_point(right_delta, off_axis))
            ],
        }
    return rows


def audit_outer_contrast_equivalence(
    component: str,
    contrast: float,
    *,
    plane_z_mm: Sequence[float] = IFT_PLANE_Z_MM,
    reference_layer: int = 0,
) -> dict[str, object]:
    """Compare sum_to_zero / reference_layer charts of one outer-contrast deformation."""
    contrast_payload = outer_contrast_payload(component, contrast)
    compensated = reference_layer_additive_equivalent(
        component, contrast, reference_layer=reference_layer
    )
    frozen = frozen_station_reference_layer_payload(
        component, contrast, reference_layer=reference_layer
    )
    additive_with_station = additive_elements_match(contrast_payload, compensated)
    additive_frozen = additive_elements_match(contrast_payload, frozen)
    se3_with_station = composed_elements_match(contrast_payload, compensated, plane_z_mm=plane_z_mm)
    se3_frozen = composed_elements_match(contrast_payload, frozen, plane_z_mm=plane_z_mm)
    if component in {"dx_mm", "dy_mm", "dz_mm"}:
        expected_se3_with_station = True
        pivot_or_conjugation = False
    else:
        expected_se3_with_station = False
        pivot_or_conjugation = True
    return {
        "station_id": IFT_STATION_ID,
        "component": str(component),
        "contrast": float(contrast),
        "reference_layer": int(reference_layer),
        "plane_z_mm": [float(value) for value in plane_z_mm],
        "analytic_map": {
            "outer_contrast_and_equal_weight_sum_to_zero": {
                "station": list(contrast_payload["station"]),
                "layers": {str(layer): list(values) for layer, values in contrast_payload["layers"].items()},
            },
            "reference_layer_same_additive_geometry": {
                "station": list(compensated["station"]),
                "layers": {str(layer): list(values) for layer, values in compensated["layers"].items()},
                "rule": "L'_i = L_i - C; S' = S + C; reference layer 0 => L' = [0, -C, -2C], S' = +C",
            },
            "reference_layer_frozen_station": {
                "station": list(frozen["station"]),
                "layers": {str(layer): list(values) for layer, values in frozen["layers"].items()},
                "rule": "same L' as above, but S' forced to 0; not the same physical family",
            },
        },
        "additive_station_plus_layer": {
            "contrast_equals_compensated_reference_layer": additive_with_station,
            "contrast_equals_frozen_station_reference_layer": additive_frozen,
        },
        "calypso_detector_element_global_delta": {
            "contrast_equals_compensated_reference_layer": se3_with_station,
            "contrast_equals_frozen_station_reference_layer": se3_frozen,
            "expected_compensated_equivalence_for_this_component": expected_se3_with_station,
            "pivot_or_conjugation_limits_rotation_gauge": pivot_or_conjugation,
            "compensated_mismatch": element_mismatch_table(contrast_payload, compensated, plane_z_mm=plane_z_mm),
            "frozen_mismatch": element_mismatch_table(contrast_payload, frozen, plane_z_mm=plane_z_mm),
        },
        "conclusion": {
            "canonical_internal_basis": CANONICAL_INTERNAL_BASIS,
            "zero_common_mode_family": list(ZERO_COMMON_MODE_CHOICES),
            "frozen_station_reference_layer_role": "negative_control_different_physical_family",
            "additive_gauge_equivalence_requires_station_compensation": bool(
                additive_with_station and not additive_frozen
            ),
            "calypso_se3_equivalence_after_station_compensation": se3_with_station,
        },
    }


def _add_six(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float, float, float, float]:
    values = np.asarray(left, dtype=np.float64) + np.asarray(right, dtype=np.float64)
    if values.shape != (6,):
        raise ValueError("payload six-vectors must have six components")
    return tuple(float(value) for value in values)


def _translation(dx: float, dy: float, dz: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = float(dx)
    matrix[1, 3] = float(dy)
    matrix[2, 3] = float(dz)
    return matrix


def _rotation_x(rx: float) -> np.ndarray:
    cosine = float(np.cos(rx))
    sine = float(np.sin(rx))
    matrix = np.eye(4, dtype=np.float64)
    matrix[1, 1] = cosine
    matrix[1, 2] = -sine
    matrix[2, 1] = sine
    matrix[2, 2] = cosine
    return matrix


def _rotation_y(ry: float) -> np.ndarray:
    cosine = float(np.cos(ry))
    sine = float(np.sin(ry))
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = cosine
    matrix[0, 2] = sine
    matrix[2, 0] = -sine
    matrix[2, 2] = cosine
    return matrix


def _rotation_z(rz: float) -> np.ndarray:
    cosine = float(np.cos(rz))
    sine = float(np.sin(rz))
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = cosine
    matrix[0, 1] = -sine
    matrix[1, 0] = sine
    matrix[1, 1] = cosine
    return matrix
