"""Four-station rigid alignment schema, gauges, and SE(3) relative geometry.

The historical IFT-only line treated station 0 as the unique movable station
and forced stations 1--3 to the identity.  That is a physical constraint, not
a default of this module.  Here every station S0--S3 may carry a real
``/Tracker/Align`` six-vector.  The first-version free chart is the five
track-constrained components ``dx, dy, rx, ry, rz`` on each station (20 DoF).
``dz`` remains survey-constrained (4 priors) and is never admitted as a free
Newton coordinate without a new identifiability map.

Two *solve-time* charts are defined:

* ``reference_station``: one station's free six-vector is held at identity.
* ``common_mode_constraint``: all four stations stay writable; a left SE(3)
  common mode is removed (linearized as an additive six-vector mean).

Neither chart is a claim that a particular station is physically correct.
Results are compared only after conversion to the gauge-invariant relatives
``ΔT_ij = T_i^{-1} T_j``.  A formulation that writes a different detector-
element transform is recorded as a different physical constraint, not as a
gauge choice.  GeoModel applies a stored station delta by left multiplication
``T_new = g * T_nominal`` (see ``alignment.gauge_equivalence``).
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from alignment.gauge_equivalence import calypso_alignment_matrix
from alignment.physical_jacobian import COMPONENT_INDEX_AND_PAYLOAD_SCALE


SCHEMA_VERSION = "faser-four-station-alignment-v1"
STATION_IDS: tuple[int, ...] = (0, 1, 2, 3)
STATION_LABELS: dict[int, str] = {
    0: "S0_IFT",
    1: "S1",
    2: "S2",
    3: "S3",
}
FREE_COMPONENTS: tuple[str, ...] = ("dx_mm", "dy_mm", "rx_mrad", "ry_mrad", "rz_mrad")
SURVEY_COMPONENT = "dz_mm"
ALL_COMPONENTS: tuple[str, ...] = ("dx_mm", "dy_mm", "dz_mm", "rx_mrad", "ry_mrad", "rz_mrad")
FORMULATION = "four_station_v1"
LEGACY_FORMULATION = "legacy_ift_vs_fixed_downstream"
GAUGE_UNCONSTRAINED_FULL = "unconstrained_full"
GAUGE_REFERENCE_STATION = "reference_station"
GAUGE_COMMON_MODE = "common_mode_constraint"
GAUGE_CHOICES: tuple[str, ...] = (
    GAUGE_UNCONSTRAINED_FULL,
    GAUGE_REFERENCE_STATION,
    GAUGE_COMMON_MODE,
)
SURVEY_DZ_PRIOR_SIGMA_MM = 5.0
DEFAULT_SEVERITY_SCALES: dict[str, float] = {
    "dx_mm": 5.0,
    "dy_mm": 5.0,
    "dz_mm": 5.0,
    "rx_mrad": 60.0,
    "ry_mrad": 60.0,
    "rz_mrad": 60.0,
}
DEFAULT_FINITE_DIFFERENCE_STEPS: dict[str, float] = {
    "dx_mm": 0.5,
    "dy_mm": 0.5,
    "dz_mm": 2.0,
    "rx_mrad": 10.0,
    "ry_mrad": 10.0,
    "rz_mrad": 10.0,
}
COMPONENT_UNITS: dict[str, str] = {
    "dx_mm": "mm",
    "dy_mm": "mm",
    "dz_mm": "mm",
    "rx_mrad": "mrad",
    "ry_mrad": "mrad",
    "rz_mrad": "mrad",
}
PAYLOAD_COMPOSITION = "T(dx,dy,dz) * Rz(rz) * Ry(ry) * Rx(rx)"
IDENTITY_SIX: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_SE3_ATOL = 1.0e-12
_ADDITIVE_ATOL = 1.0e-15


def require_station_id(station: int) -> int:
    value = int(station)
    if value not in STATION_IDS:
        raise ValueError(f"station {station} is not one of {list(STATION_IDS)}")
    return value


def parameter_name(station: int, component: str) -> str:
    """Return the four-station native name, e.g. ``s2_ry_mrad``."""
    if component not in ALL_COMPONENTS:
        raise ValueError(f"unsupported component '{component}'")
    return f"s{require_station_id(station)}_{component}"


def parse_parameter_name(name: str) -> tuple[int, str]:
    """Parse ``s{station}_{component}`` into ``(station, component)``."""
    if not isinstance(name, str) or not name.startswith("s") or "_" not in name:
        raise ValueError(f"not a four-station parameter name: {name!r}")
    raw_station, component = name[1:].split("_", 1)
    try:
        station = int(raw_station)
    except ValueError as error:
        raise ValueError(f"not a four-station parameter name: {name!r}") from error
    if component not in ALL_COMPONENTS or station not in STATION_IDS:
        raise ValueError(f"not a four-station parameter name: {name!r}")
    return station, component


def parameter_role(component: str) -> str:
    if component == SURVEY_COMPONENT:
        return "survey_constrained"
    if component in FREE_COMPONENTS:
        return "free"
    raise ValueError(f"unsupported component '{component}'")


def identity_station_transforms() -> dict[str, list[float]]:
    return {str(station): list(IDENTITY_SIX) for station in STATION_IDS}


def default_parameter_specs(*, include_survey_dz: bool = True) -> list[dict[str, object]]:
    """Return the machine-readable 20-DoF (+ optional 4 survey dz) spec list."""
    components = ALL_COMPONENTS if include_survey_dz else FREE_COMPONENTS
    specs: list[dict[str, object]] = []
    for station in STATION_IDS:
        for component in components:
            specs.append(
                {
                    "name": parameter_name(station, component),
                    "scope": "station",
                    "station_id": int(station),
                    "component": component,
                    "unit": COMPONENT_UNITS[component],
                    "role": parameter_role(component),
                    "finite_difference_step": float(DEFAULT_FINITE_DIFFERENCE_STEPS[component]),
                    "severity_scale": float(DEFAULT_SEVERITY_SCALES[component]),
                    "survey_prior_sigma": (
                        float(SURVEY_DZ_PRIOR_SIGMA_MM) if component == SURVEY_COMPONENT else None
                    ),
                }
            )
    return specs


def free_parameter_names() -> tuple[str, ...]:
    return tuple(parameter_name(station, component) for station in STATION_IDS for component in FREE_COMPONENTS)


def survey_parameter_names() -> tuple[str, ...]:
    return tuple(parameter_name(station, SURVEY_COMPONENT) for station in STATION_IDS)


def zero_parameter_values(specs: Sequence[Mapping[str, object]]) -> dict[str, float]:
    names = [str(spec["name"]) for spec in specs]
    if not names or len(set(names)) != len(names):
        raise ValueError("parameter specs require unique non-empty names")
    return {name: 0.0 for name in names}


def station_pair_ids() -> tuple[tuple[int, int], ...]:
    """Adjacent pairs, then the long baseline, then the remaining pairs."""
    return ((0, 1), (1, 2), (2, 3), (0, 3), (0, 2), (1, 3))


def six_vector_to_matrix(transform: Sequence[float]) -> np.ndarray:
    """Map a payload six-vector onto ``T * Rz * Ry * Rx``."""
    values = np.asarray(transform, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("station transform must be a finite six-vector")
    return calypso_alignment_matrix(
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]),
        float(values[4]),
        float(values[5]),
    )


def matrix_to_six_vector(matrix: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Invert ``T * Rz * Ry * Rx`` into payload units (mm, rad)."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (4, 4) or not np.isfinite(values).all():
        raise ValueError("SE(3) matrix must be a finite 4x4 array")
    if not np.allclose(values[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=_SE3_ATOL):
        raise ValueError("SE(3) matrix has a non-homogeneous last row")
    rotation = values[:3, :3]
    sine_y = float(np.clip(-rotation[2, 0], -1.0, 1.0))
    cosine_y = float(math.sqrt(max(0.0, 1.0 - sine_y * sine_y)))
    if cosine_y > 1.0e-12:
        rx = math.atan2(rotation[2, 1], rotation[2, 2])
        ry = math.atan2(sine_y, cosine_y)
        rz = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        rx = math.atan2(-rotation[1, 2], rotation[1, 1])
        ry = math.copysign(0.5 * math.pi, sine_y)
        rz = 0.0
    return (
        float(values[0, 3]),
        float(values[1, 3]),
        float(values[2, 3]),
        float(rx),
        float(ry),
        float(rz),
    )


def invert_six_vector(transform: Sequence[float]) -> tuple[float, float, float, float, float, float]:
    return matrix_to_six_vector(np.linalg.inv(six_vector_to_matrix(transform)))


def compose_six_vectors(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float, float, float, float]:
    return matrix_to_six_vector(six_vector_to_matrix(left) @ six_vector_to_matrix(right))


def relative_six_vector(
    source: Sequence[float],
    target: Sequence[float],
) -> tuple[float, float, float, float, float, float]:
    """Return ``ΔT = T_source^{-1} T_target`` in payload units."""
    return compose_six_vectors(invert_six_vector(source), target)


def _as_station_map(
    transforms: Mapping[int | str, Sequence[float]],
) -> dict[int, tuple[float, float, float, float, float, float]]:
    result: dict[int, tuple[float, float, float, float, float, float]] = {}
    for raw_station, raw_transform in transforms.items():
        station = require_station_id(int(raw_station))
        values = np.asarray(raw_transform, dtype=np.float64)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError(f"station {station} has no finite six-vector")
        if station in result:
            raise ValueError(f"duplicate transform for station {station}")
        result[station] = tuple(float(value) for value in values)
    missing = set(STATION_IDS) - set(result)
    if missing:
        raise ValueError(f"station transform map lacks stations {sorted(missing)}")
    return result


def relative_alignment_table(
    transforms: Mapping[int | str, Sequence[float]],
) -> dict[str, list[float]]:
    """Gauge-invariant payload relatives ``ΔT_ij = T_i^{-1} T_j``."""
    packed = _as_station_map(transforms)
    return {
        f"{source}_{target}": list(relative_six_vector(packed[source], packed[target]))
        for source, target in station_pair_ids()
    }


def composed_relative_alignment_table(
    transforms: Mapping[int | str, Sequence[float]],
    nominal_z_mm: Mapping[int, float],
) -> dict[str, list[float]]:
    """Relatives of composed poses ``T_i N_i`` with ``N_i = T(0,0,z_i)``.

    ``nominal_z_mm`` is an analysis input (typically mean reconstructed
    tracklet *z*), not a frozen FASER survey constant stored in this schema.
    """
    packed = _as_station_map(transforms)
    composed: dict[int, tuple[float, float, float, float, float, float]] = {}
    for station, transform in packed.items():
        if station not in nominal_z_mm or not math.isfinite(float(nominal_z_mm[station])):
            raise ValueError(f"nominal z for station {station} is missing or non-finite")
        nominal = (0.0, 0.0, float(nominal_z_mm[station]), 0.0, 0.0, 0.0)
        composed[station] = compose_six_vectors(transform, nominal)
    return {
        f"{source}_{target}": list(relative_six_vector(composed[source], composed[target]))
        for source, target in station_pair_ids()
    }


def left_multiply_all(
    transforms: Mapping[int | str, Sequence[float]],
    common: Sequence[float],
) -> dict[str, list[float]]:
    """Apply one global SE(3) element on the left of every station delta."""
    packed = _as_station_map(transforms)
    return {str(station): list(compose_six_vectors(common, transform)) for station, transform in packed.items()}


def add_common_six_vector(
    transforms: Mapping[int | str, Sequence[float]],
    common: Sequence[float],
) -> dict[str, list[float]]:
    """Add one six-vector in the payload chart (not a claimed SE(3) gauge)."""
    packed = _as_station_map(transforms)
    common_values = np.asarray(common, dtype=np.float64)
    if common_values.shape != (6,) or not np.isfinite(common_values).all():
        raise ValueError("common six-vector must be finite and length 6")
    return {
        str(station): [float(left + right) for left, right in zip(transform, common_values)]
        for station, transform in packed.items()
    }


def apply_reference_station_gauge(
    transforms: Mapping[int | str, Sequence[float]],
    *,
    reference_station: int,
) -> dict[str, list[float]]:
    """Rewrite ``T_i' = T_ref^{-1} T_i`` so the reference station is identity."""
    packed = _as_station_map(transforms)
    reference = require_station_id(reference_station)
    inverse = invert_six_vector(packed[reference])
    return {str(station): list(compose_six_vectors(inverse, transform)) for station, transform in packed.items()}


def apply_left_common_mode_constraint(
    transforms: Mapping[int | str, Sequence[float]],
) -> dict[str, list[float]]:
    """Remove a left SE(3) element built from the four-station mean six-vector.

    ``G`` is ``T * Rz * Ry * Rx`` of the additive mean payload, then
    ``T_i' = G^{-1} T_i``.  This *is* a gauge: ``ΔT_ij`` is invariant.
    Subtracting the mean six-vector componentwise is not the same map once
    finite rotations are present; that additive chart is retained only as a
    linearized Jacobian diagnostic.
    """
    packed = _as_station_map(transforms)
    mean = np.mean(np.asarray([packed[station] for station in STATION_IDS], dtype=np.float64), axis=0)
    return left_multiply_all(packed, invert_six_vector(mean))


def apply_additive_common_mode_constraint(
    transforms: Mapping[int | str, Sequence[float]],
) -> dict[str, list[float]]:
    """Subtract the additive four-station mean six-vector.

    Linearized Jacobian common-mode, **not** a finite SE(3) gauge.  Compare
    relatives with :func:`apply_left_common_mode_constraint` before calling
    this a gauge choice.
    """
    packed = _as_station_map(transforms)
    mean = np.mean(np.asarray([packed[station] for station in STATION_IDS], dtype=np.float64), axis=0)
    return add_common_six_vector(packed, -mean)


def relatives_agree(
    left: Mapping[int | str, Sequence[float]],
    right: Mapping[int | str, Sequence[float]],
    *,
    atol: float = _SE3_ATOL,
) -> bool:
    first = relative_alignment_table(left)
    second = relative_alignment_table(right)
    return all(
        np.allclose(first[key], second[key], rtol=0.0, atol=float(atol)) for key in first
    )


def additive_translation_is_left_gauge(component_index: int, value: float) -> bool:
    """Translations commute: additive common dx/dy/dz equals left SE(3)."""
    if component_index not in (0, 1, 2):
        return False
    common = list(IDENTITY_SIX)
    common[component_index] = float(value)
    base = {
        0: (0.1, -0.2, 0.0, 0.0, 0.0, 0.0),
        1: (0.0, 0.3, 0.0, 0.0, 0.0, 0.0),
        2: (-0.4, 0.0, 0.0, 0.0, 0.0, 0.0),
        3: (0.2, 0.1, 0.0, 0.0, 0.0, 0.0),
    }
    return relatives_agree(left_multiply_all(base, common), add_common_six_vector(base, common))


def finite_rotation_additive_is_not_automatically_a_gauge() -> bool:
    """Finite common *additive* rx is not the same map as left-multiply rx."""
    common = (0.0, 0.0, 0.0, 0.05, 0.0, 0.0)
    base = {
        0: (0.4, -0.2, 0.0, 0.01, -0.02, 0.0),
        1: (-0.1, 0.3, 0.0, 0.0, 0.015, -0.01),
        2: (0.2, 0.0, 0.0, -0.012, 0.0, 0.008),
        3: (0.0, 0.15, 0.0, 0.0, 0.0, 0.0),
    }
    return not relatives_agree(
        left_multiply_all(base, common),
        add_common_six_vector(base, common),
        atol=1.0e-8,
    )


def classify_scaled_singular_vector(
    names: Sequence[str],
    vector: Sequence[float],
    *,
    threshold: float = 0.25,
) -> dict[str, object]:
    """Label a scaled-normal singular vector by common / relative structure."""
    values = np.asarray(vector, dtype=np.float64)
    if values.shape != (len(names),) or not np.isfinite(values).all():
        raise ValueError("singular vector must be finite and match parameter names")
    by_component: dict[str, dict[int, float]] = {component: {} for component in ALL_COMPONENTS}
    for name, value in zip(names, values):
        station, component = parse_parameter_name(str(name))
        by_component[component][station] = float(value)
    labels: list[str] = []
    for component in FREE_COMPONENTS:
        loadings = [by_component[component].get(station, 0.0) for station in STATION_IDS]
        if not any(abs(value) >= threshold for value in loadings):
            continue
        same_sign = all(value * loadings[0] > 0.0 for value in loadings if abs(value) >= threshold)
        if same_sign and all(abs(value) >= threshold for value in loadings):
            labels.append(f"global_common_{component}")
        elif abs(loadings[0]) >= threshold and abs(loadings[3]) >= threshold and loadings[0] * loadings[3] < 0.0:
            labels.append(f"long_baseline_{component}")
        elif any(
            abs(loadings[left]) >= threshold
            and abs(loadings[right]) >= threshold
            and loadings[left] * loadings[right] < 0.0
            for left, right in ((0, 1), (1, 2), (2, 3))
        ):
            labels.append(f"adjacent_relative_{component}")
        else:
            labels.append(f"mixed_{component}")
    if SURVEY_COMPONENT in by_component and any(
        abs(by_component[SURVEY_COMPONENT].get(station, 0.0)) >= threshold for station in STATION_IDS
    ):
        labels.append("survey_dz")
    dominant = [
        str(name)
        for name, value in sorted(zip(names, values), key=lambda item: -abs(item[1]))
        if abs(float(value)) >= threshold
    ]
    return {
        "labels": labels,
        "dominant_parameters": dominant,
        "loadings": {str(name): float(value) for name, value in zip(names, values)},
    }


def formulation_contract(
    *,
    gauge: str,
    reference_station: int | None = None,
    include_survey_dz: bool = True,
) -> dict[str, object]:
    """Machine-readable four-station contract written into manifests."""
    if gauge not in GAUGE_CHOICES:
        raise ValueError(f"unsupported four-station gauge '{gauge}'")
    if gauge == GAUGE_REFERENCE_STATION:
        if reference_station is None:
            raise ValueError("reference_station gauge requires an explicit reference_station")
        reference = require_station_id(reference_station)
        movable = [station for station in STATION_IDS if station != reference]
        reference_ids = [reference]
    elif gauge in {GAUGE_UNCONSTRAINED_FULL, GAUGE_COMMON_MODE}:
        if reference_station is not None:
            raise ValueError(f"gauge '{gauge}' forbids a privileged reference station")
        movable = list(STATION_IDS)
        reference_ids = []
    else:  # pragma: no cover - guarded by GAUGE_CHOICES
        raise ValueError(f"unsupported four-station gauge '{gauge}'")
    specs = default_parameter_specs(include_survey_dz=include_survey_dz)
    return {
        "schema_version": SCHEMA_VERSION,
        "alignment_formulation": FORMULATION,
        "gauge": gauge,
        "station_ids": list(STATION_IDS),
        "station_labels": dict(STATION_LABELS),
        "reference_station_ids": reference_ids,
        "movable_station_ids": movable,
        "free_components": list(FREE_COMPONENTS),
        "survey_component": SURVEY_COMPONENT,
        "free_parameter_names": list(free_parameter_names()),
        "survey_parameter_names": list(survey_parameter_names()) if include_survey_dz else [],
        "n_free_parameters": len(FREE_COMPONENTS) * len(STATION_IDS),
        "n_survey_parameters": len(STATION_IDS) if include_survey_dz else 0,
        "survey_dz_prior_sigma_mm": SURVEY_DZ_PRIOR_SIGMA_MM,
        "payload_composition": PAYLOAD_COMPOSITION,
        "payload_units": "translations_mm_rotations_rad",
        "reporting_units": "translations_mm_rotations_mrad",
        "geomodel_application": "T_new = g_station * T_nominal",
        "gauge_invariant_observable": "DeltaT_ij = T_i^{-1} T_j",
        "coordinate_surrogate": False,
        "physical_geometry_repropagation": True,
        "q_over_p_mode": 0,
        "no_station_is_assumed_correct": True,
        "legacy_ift_only_default": False,
        "alignment_parameter_specs": specs,
        "notes": [
            "Unconstrained_full is the FD/identifiability chart: all 20 (plus survey dz) columns are probed.",
            "Do not assume the 20-D space is full rank; SVD decides the admitted relative subspace.",
            "reference_station and common_mode_constraint are solve-time charts of the same SE(3) relatives.",
            "A chart that writes a different detector-element transform is a different physical constraint.",
        ],
    }


REPORTING_SIX_LABELS: tuple[str, ...] = (
    "dx_mm",
    "dy_mm",
    "dz_mm",
    "rx_mrad",
    "ry_mrad",
    "rz_mrad",
)
N_RELATIVE_FREE_PARAMETERS = len(FREE_COMPONENTS) * (len(STATION_IDS) - 1)


def relative_free_parameter_names(
    *,
    gauge: str,
    reference_station: int | None = None,
) -> tuple[str, ...]:
    """Return the solve-time free names for a relative four-station chart.

    ``reference_station`` drops that station's five track-constrained
    coordinates (15 DoF).  ``common_mode_constraint`` keeps all 20 free
    names; the left SE(3) common mode is a payload rewrite after the fit,
    not a silently dropped column.  ``unconstrained_full`` is the FD chart
    and is not a column-reduced solve.
    """
    if gauge == GAUGE_REFERENCE_STATION:
        reference = require_station_id(reference_station)
        return tuple(
            parameter_name(station, component)
            for station in STATION_IDS
            if station != reference
            for component in FREE_COMPONENTS
        )
    if gauge == GAUGE_COMMON_MODE:
        if reference_station is not None:
            raise ValueError("common_mode_constraint forbids a privileged reference station")
        return free_parameter_names()
    raise ValueError(f"gauge '{gauge}' is not a column-reduced four-station solve chart")


def station_transforms_from_native_values(
    values: Mapping[str, float],
    *,
    base: Mapping[int | str, Sequence[float]] | None = None,
) -> dict[str, list[float]]:
    """Write ``s{station}_{component}`` native values into a four-station payload."""
    from alignment.physical_jacobian import station_transforms_with_parameter_values

    specs = []
    for name in values:
        station, component = parse_parameter_name(name)
        specs.append(
            {
                "name": name,
                "scope": "station",
                "station_id": station,
                "component": component,
            }
        )
    return station_transforms_with_parameter_values(
        specs,
        identity_station_transforms() if base is None else base,
        dict(values),
    )


def six_vector_reporting_units(values: Sequence[float]) -> list[float]:
    packed = np.asarray(values, dtype=np.float64)
    if packed.shape != (6,) or not np.isfinite(packed).all():
        raise ValueError("six-vector must be a finite length-6 payload")
    return [
        float(packed[0]),
        float(packed[1]),
        float(packed[2]),
        float(packed[3] * 1.0e3),
        float(packed[4] * 1.0e3),
        float(packed[5] * 1.0e3),
    ]


def relative_table_reporting_units(
    transforms: Mapping[int | str, Sequence[float]],
) -> dict[str, list[float]]:
    return {
        key: six_vector_reporting_units(values)
        for key, values in relative_alignment_table(transforms).items()
    }


def relative_table_errors(
    injected: Mapping[int | str, Sequence[float]],
    recovered: Mapping[int | str, Sequence[float]],
) -> dict[str, dict[str, float]]:
    """Signed ``ΔT_ij`` errors in reporting units (mm, mrad)."""
    left = relative_table_reporting_units(injected)
    right = relative_table_reporting_units(recovered)
    errors: dict[str, dict[str, float]] = {}
    for key, injected_values in left.items():
        recovered_values = right[key]
        errors[key] = {
            label: float(recovered_values[index] - injected_values[index])
            for index, label in enumerate(REPORTING_SIX_LABELS)
        }
    return errors


def capture_relative_tables(
    injected: Mapping[int | str, Sequence[float]],
    recovered: Mapping[int | str, Sequence[float]],
    tolerances: Mapping[str, float],
) -> dict[str, object]:
    """Compare gauge-invariant relatives against a pre-registered tolerance."""
    missing = [label for label in REPORTING_SIX_LABELS if label not in tolerances]
    if missing:
        raise ValueError(f"capture tolerances lack {missing}")
    errors = relative_table_errors(injected, recovered)
    pair_rows = {}
    success = True
    for pair, pair_errors in errors.items():
        captured = {
            label: bool(abs(pair_errors[label]) <= float(tolerances[label]))
            for label in REPORTING_SIX_LABELS
        }
        pair_rows[pair] = {"signed_error": pair_errors, "captured": captured}
        success = success and all(captured.values())
    return {
        "success": bool(success),
        "tolerances": {label: float(tolerances[label]) for label in REPORTING_SIX_LABELS},
        "pairs": pair_rows,
        "injected_delta_t_ij": relative_table_reporting_units(injected),
        "recovered_delta_t_ij": relative_table_reporting_units(recovered),
    }
