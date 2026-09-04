#!/usr/bin/env python3
"""Run a real-conditions, cluster-to-segment physical capture-range scan.

Every planned point writes a fresh /Tracker/Align SQLite/POOL payload and
reruns Calypso's SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit ->
NtupleDumper chain before field-aware mode-0 propagation is evaluated.  The
only reused input is the original persisted MC xAOD; coordinate-level shifts
and cached residuals are deliberately not part of this workflow.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
SETUP_SCRIPT = PROJECT_ROOT / "scripts" / "setup_environment.sh"
PAYLOAD_WRITER = PROJECT_ROOT / "scripts" / "write_station_alignment_payload.py"
SUMMARIZER = PROJECT_ROOT / "scripts" / "summarize_physical_refit_capture_scan.py"


def _quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def _slug(value: float) -> str:
    text = format(value, ".8g")
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle)
    if not isinstance(supplied, Mapping):
        raise ValueError("scan configuration must be a YAML mapping")
    config = supplied.get("physical_refit_capture_scan", supplied)
    if not isinstance(config, Mapping):
        raise ValueError("physical_refit_capture_scan must be a YAML mapping")
    return dict(config)


def _build_translation_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    stations = tuple(sorted({int(station) for station in config["station_ids"]}))
    reference = int(config["reference_station"])
    if len(stations) < 2 or reference not in stations:
        raise ValueError("station_ids must contain the reference and at least one movable station")
    if any(station not in (0, 1, 2, 3) for station in stations):
        raise ValueError("the current conditions payload supports only stations 0 through 3")
    magnitudes = [float(value) for value in config["magnitudes_mm"]]
    if not magnitudes or any(not math.isfinite(value) or value < 0.0 for value in magnitudes):
        raise ValueError("magnitudes_mm must be a non-empty list of finite non-negative values")
    if magnitudes != sorted(magnitudes) or len(set(magnitudes)) != len(magnitudes):
        raise ValueError("magnitudes_mm must be strictly increasing")
    supplied_directions = config["direction_trials"]
    if not isinstance(supplied_directions, list) or not supplied_directions:
        raise ValueError("direction_trials must be a non-empty list")
    movable = tuple(station for station in stations if station != reference)
    directions: list[dict[str, Any]] = []
    names: set[str] = set()
    for direction in supplied_directions:
        if not isinstance(direction, Mapping):
            raise ValueError("every direction trial must be a mapping")
        name = str(direction["name"])
        if not name.replace("_", "").isalnum() or name in names:
            raise ValueError("direction trial names must be unique alphanumeric identifiers")
        names.add(name)
        raw_offsets = direction["unit_offsets_xy"]
        if not isinstance(raw_offsets, Mapping):
            raise ValueError("unit_offsets_xy must map station IDs to [dx, dy]")
        offsets = {int(station): tuple(float(value) for value in offset) for station, offset in raw_offsets.items()}
        if set(offsets) != set(movable):
            raise ValueError("each direction must specify exactly the non-reference stations")
        for station, offset in offsets.items():
            if len(offset) != 2 or not all(math.isfinite(value) for value in offset):
                raise ValueError(f"invalid unit offset for station {station}")
            if not math.isclose(math.hypot(*offset), 1.0, rel_tol=0.0, abs_tol=1.0e-8):
                raise ValueError(f"unit offset for station {station} is not normalized")
        directions.append({"name": name, "unit_offsets_xy": offsets})

    points: list[dict[str, Any]] = []
    point_index = 0
    for magnitude in magnitudes:
        # A zero injection is direction-independent, so it is one physical
        # baseline refit rather than redundant repeats.
        active_directions = directions[:1] if magnitude == 0.0 else directions
        for direction in active_directions:
            injected = {str(reference): [0.0, 0.0]}
            for station in movable:
                unit = direction["unit_offsets_xy"][station]
                injected[str(station)] = [magnitude * unit[0], magnitude * unit[1]]
            name = f"mag_{_slug(magnitude)}_{direction['name']}"
            points.append(
                {
                    "index": point_index,
                    "name": name,
                    "relative_point_dir": str(Path("points") / name),
                    "magnitude_mm": magnitude,
                    "direction_trial": direction["name"],
                    "injected_offsets_xy_mm": injected,
                }
            )
            point_index += 1
    return {
        "method": "physical_refit_capture_range_scan",
        "scan_mode": "translation_xy",
        "station_ids": list(stations),
        "reference_station": reference,
        "q_over_p_mode": int(config["q_over_p_mode"]),
        "points": points,
    }


def _station_transform(values: object, *, label: str) -> tuple[float, float, float, float, float, float]:
    """Validate one exact Calypso station transform in writer component order."""
    if not isinstance(values, (list, tuple)) or len(values) != 6:
        raise ValueError(f"{label} must be [dx, dy, dz, rx, ry, rz]")
    try:
        transform = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain numeric transform components") from error
    if not all(math.isfinite(value) for value in transform):
        raise ValueError(f"{label} must contain only finite transform components")
    return transform  # type: ignore[return-value]


# ``WriteAlignmentCfg`` accepts rotations in radians, while the alignment
# studies report them in mrad.  Keep this conversion in one explicit mapping
# so a future six-DoF scan cannot silently mix payload and reporting units.
_ALIGNMENT_COMPONENTS: dict[str, tuple[int, str, float]] = {
    "dx_mm": (0, "mm", 1.0),
    "dy_mm": (1, "mm", 1.0),
    "dz_mm": (2, "mm", 1.0),
    "rx_mrad": (3, "mrad", 1.0e3),
    "ry_mrad": (4, "mrad", 1.0e3),
    "rz_mrad": (5, "mrad", 1.0e3),
}


def _safe_identifier(value: object, *, label: str) -> str:
    identifier = str(value)
    if not identifier or not identifier.replace("_", "").isalnum():
        raise ValueError(f"{label} must be a non-empty alphanumeric identifier")
    return identifier


def _build_station_rigid_multidof_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build a frozen physical multi-DoF station-transform plan.

    The scan consumes explicit full transforms, rather than altering exported
    states.  Its first use is IFT ``dx/dy/R_y`` with stations 1--3 fixed, but
    the parameter schema deliberately covers all station rigid components so
    later ``dz/Rx/Rz`` admission is evidence-driven rather than a new payload
    format.  Every enabled degree of freedom needs a pair of pure central
    finite-difference physical probes; arbitrary joint points are reserved for
    the curriculum and held-out closure.
    """
    stations = tuple(sorted({int(station) for station in config["station_ids"]}))
    if len(stations) < 2 or any(station not in (0, 1, 2, 3) for station in stations):
        raise ValueError("station rigid multi-DoF scans support at least two stations from 0 through 3")
    raw_reference = config.get("reference_station_ids")
    raw_movable = config.get("movable_station_ids")
    if not isinstance(raw_reference, (list, tuple)) or not isinstance(raw_movable, (list, tuple)):
        raise ValueError("station rigid multi-DoF scans require reference_station_ids and movable_station_ids")
    reference_stations = tuple(sorted({int(station) for station in raw_reference}))
    movable_stations = tuple(sorted({int(station) for station in raw_movable}))
    if not reference_stations or not movable_stations:
        raise ValueError("station rigid multi-DoF scans require non-empty reference and movable station sets")
    if set(reference_stations).intersection(movable_stations) or set(reference_stations).union(movable_stations) != set(stations):
        raise ValueError("reference_station_ids and movable_station_ids must be disjoint and cover station_ids")
    if any(station not in stations for station in (*reference_stations, *movable_stations)):
        raise ValueError("reference or movable station is absent from station_ids")
    axis = str(config.get("condition_axis", "station_rigid_multidof"))
    if not axis:
        raise ValueError("station rigid multi-DoF scans require a non-empty condition_axis")

    raw_specs = config.get("alignment_parameter_specs")
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("station rigid multi-DoF scans require alignment_parameter_specs")
    specs: list[dict[str, Any]] = []
    names: set[str] = set()
    slots: set[tuple[int, str]] = set()
    for raw_spec in raw_specs:
        if not isinstance(raw_spec, Mapping):
            raise ValueError("every alignment_parameter_specs entry must be a mapping")
        name = _safe_identifier(raw_spec.get("name", ""), label="alignment parameter name")
        if name in names:
            raise ValueError(f"duplicate alignment parameter name '{name}'")
        names.add(name)
        scope = str(raw_spec.get("scope", "station"))
        if scope != "station":
            raise ValueError(
                f"alignment parameter '{name}' has scope '{scope}'; "
                "this physical payload writer currently admits station scope only"
            )
        try:
            station = int(raw_spec["station_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"alignment parameter '{name}' requires station_id") from error
        if station not in movable_stations:
            raise ValueError(f"alignment parameter '{name}' cannot move reference/unsupported station {station}")
        component = str(raw_spec.get("component", ""))
        if component not in _ALIGNMENT_COMPONENTS:
            raise ValueError(
                f"alignment parameter '{name}' has unsupported component '{component}'; "
                f"allowed={sorted(_ALIGNMENT_COMPONENTS)}"
            )
        if (station, component) in slots:
            raise ValueError(f"station {station} component '{component}' is declared more than once")
        slots.add((station, component))
        index, unit, payload_scale = _ALIGNMENT_COMPONENTS[component]
        supplied_unit = raw_spec.get("unit", unit)
        if str(supplied_unit) != unit:
            raise ValueError(f"alignment parameter '{name}' unit must be '{unit}' for component '{component}'")
        try:
            step = float(raw_spec["finite_difference_step"])
            severity_scale = float(raw_spec["severity_scale"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"alignment parameter '{name}' requires finite_difference_step and severity_scale"
            ) from error
        if not math.isfinite(step) or step <= 0.0 or not math.isfinite(severity_scale) or severity_scale <= 0.0:
            raise ValueError(f"alignment parameter '{name}' has non-positive/non-finite step or severity scale")
        specs.append(
            {
                "name": name,
                "scope": scope,
                "station_id": station,
                "component": component,
                "unit": unit,
                "transform_index": index,
                "payload_scale": payload_scale,
                "finite_difference_step": step,
                "severity_scale": severity_scale,
            }
        )

    supplied_points = config.get("rigid_points")
    if not isinstance(supplied_points, list) or not supplied_points:
        raise ValueError("station rigid multi-DoF scans require a non-empty rigid_points list")
    points: list[dict[str, Any]] = []
    point_names: set[str] = set()
    nominal_count = 0
    finite_difference: dict[str, set[str]] = {str(spec["name"]): set() for spec in specs}
    for point_index, raw_point in enumerate(supplied_points):
        if not isinstance(raw_point, Mapping):
            raise ValueError("every rigid multi-DoF point must be a mapping")
        name = _safe_identifier(raw_point.get("name", ""), label="rigid multi-DoF point name")
        if name in point_names:
            raise ValueError(f"duplicate rigid multi-DoF point name '{name}'")
        point_names.add(name)
        raw_transforms = raw_point.get("station_transforms")
        if not isinstance(raw_transforms, Mapping):
            raise ValueError(f"rigid point '{name}' requires station_transforms")
        transforms: dict[str, list[float]] = {}
        parsed_stations: set[int] = set()
        for raw_station, raw_transform in raw_transforms.items():
            try:
                station = int(raw_station)
            except (TypeError, ValueError) as error:
                raise ValueError(f"rigid point '{name}' has an invalid station key") from error
            if station not in stations or station in parsed_stations:
                raise ValueError(f"rigid point '{name}' has duplicate/unsupported station {station}")
            parsed_stations.add(station)
            transforms[str(station)] = list(
                _station_transform(raw_transform, label=f"rigid point '{name}' station {station}")
            )
        if parsed_stations != set(stations):
            raise ValueError(f"rigid point '{name}' must explicitly specify every station transform")
        for station in reference_stations:
            if not np_allclose_zero(transforms[str(station)]):
                raise ValueError(f"rigid point '{name}' moves reference station {station}")

        derived_values = {
            str(spec["name"]): float(
                transforms[str(spec["station_id"])][int(spec["transform_index"])]
                * float(spec["payload_scale"])
            )
            for spec in specs
        }
        for station in movable_stations:
            transform = transforms[str(station)]
            permitted = {
                int(spec["transform_index"])
                for spec in specs
                if int(spec["station_id"]) == station
            }
            for index, value in enumerate(transform):
                if index not in permitted and not math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1.0e-15):
                    raise ValueError(
                        f"rigid point '{name}' changes station {station} component index {index} "
                        "without an enabled alignment parameter"
                    )
        raw_values = raw_point.get("alignment_parameter_values")
        if raw_values is not None:
            if not isinstance(raw_values, Mapping) or set(str(key) for key in raw_values) != set(derived_values):
                raise ValueError(
                    f"rigid point '{name}' alignment_parameter_values must specify exactly {sorted(derived_values)}"
                )
            for parameter, expected in derived_values.items():
                try:
                    observed = float(raw_values[parameter])
                except (TypeError, ValueError, KeyError) as error:
                    raise ValueError(f"rigid point '{name}' has invalid value for '{parameter}'") from error
                if not math.isfinite(observed) or not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1.0e-12):
                    raise ValueError(
                        f"rigid point '{name}' parameter '{parameter}' disagrees with station_transforms"
                    )

        normalized = [
            derived_values[str(spec["name"])] / float(spec["severity_scale"])
            for spec in specs
        ]
        severity = float(math.sqrt(sum(value * value for value in normalized)))
        role = str(raw_point.get("point_role", "curriculum"))
        is_zero = all(math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1.0e-15) for value in derived_values.values())
        if role == "nominal":
            if not is_zero:
                raise ValueError(f"nominal rigid point '{name}' must be the all-zero payload")
            nominal_count += 1
        elif is_zero:
            raise ValueError(f"non-nominal rigid point '{name}' must contain a non-zero enabled parameter")

        finite_difference_for = raw_point.get("finite_difference_for")
        probe_sign = raw_point.get("probe_sign")
        finite_difference_anchor = raw_point.get("finite_difference_anchor")
        if finite_difference_for is not None or probe_sign is not None:
            parameter = str(finite_difference_for)
            if parameter not in derived_values or str(probe_sign) not in {"positive", "negative"}:
                raise ValueError(f"rigid point '{name}' has an invalid finite-difference tag")
            sign = str(probe_sign)
            if role not in {"finite_difference_positive", "finite_difference_negative"} or not role.endswith(sign):
                raise ValueError(f"rigid point '{name}' finite-difference point_role/sign disagree")
            if finite_difference_anchor is None:
                # The original closure bank differentiates around the
                # all-zero nominal payload.  Retain that strict convention
                # for legacy scans and make iteration-centred probes opt in
                # explicitly below.
                for other, value in derived_values.items():
                    if other != parameter and not math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1.0e-15):
                        raise ValueError(f"finite-difference point '{name}' must vary only '{parameter}'")
                if (sign == "positive" and derived_values[parameter] <= 0.0) or (
                    sign == "negative" and derived_values[parameter] >= 0.0
                ):
                    raise ValueError(f"finite-difference point '{name}' does not bracket zero for '{parameter}'")
            else:
                anchor_name = _safe_identifier(
                    finite_difference_anchor, label=f"rigid point '{name}' finite-difference anchor"
                )
                if anchor_name == name:
                    raise ValueError(f"finite-difference point '{name}' cannot anchor itself")
                finite_difference_anchor = anchor_name
            finite_difference[parameter].add(sign)
        else:
            if finite_difference_anchor is not None:
                raise ValueError(f"rigid point '{name}' has a finite-difference anchor without probe tags")
            if role.startswith("finite_difference_"):
                raise ValueError(f"rigid point '{name}' has a finite-difference role without tags")

        try:
            condition_value = float(raw_point.get("condition_value", severity))
            condition_magnitude = float(raw_point.get("condition_magnitude", severity))
        except (TypeError, ValueError) as error:
            raise ValueError(f"rigid point '{name}' has invalid condition metadata") from error
        if (
            not math.isfinite(condition_value)
            or not math.isfinite(condition_magnitude)
            or condition_magnitude < 0.0
        ):
            raise ValueError(f"rigid point '{name}' has non-finite/negative condition metadata")
        points.append(
            {
                "index": point_index,
                "name": name,
                "relative_point_dir": str(Path("points") / name),
                "direction_trial": str(raw_point.get("direction_trial", name)),
                "point_role": role,
                "finite_difference_for": None if finite_difference_for is None else str(finite_difference_for),
                "probe_sign": None if probe_sign is None else str(probe_sign),
                "finite_difference_anchor": (
                    None if finite_difference_anchor is None else str(finite_difference_anchor)
                ),
                "condition_axis": axis,
                "condition_value": condition_value,
                "condition_magnitude": condition_magnitude,
                "alignment_parameter_values": derived_values,
                "injected_station_transforms": transforms,
                **(
                    {}
                    if raw_point.get("sampling_seed") is None
                    else {"sampling_seed": int(raw_point["sampling_seed"])}
                ),
                **(
                    {}
                    if raw_point.get("curriculum_stage") is None
                    else {"curriculum_stage": str(raw_point["curriculum_stage"])}
                ),
            }
        )
    if nominal_count != 1:
        raise ValueError("station rigid multi-DoF scan must contain exactly one nominal all-zero payload")
    missing_probes = [name for name, signs in finite_difference.items() if signs != {"positive", "negative"}]
    if missing_probes and not bool(config.get("current_geometry_only", False)):
        raise ValueError(
            "station rigid multi-DoF scan requires positive/negative physical probes for: "
            + ", ".join(sorted(missing_probes))
        )
    if bool(config.get("current_geometry_only", False)):
        if nominal_count != 1:
            raise ValueError("current-geometry-only scans require exactly one nominal payload")
        extra = [point["name"] for point in points if point.get("finite_difference_for") is not None]
        if extra:
            raise ValueError("current-geometry-only scans must not include finite-difference probes")
    # Iterative alignment probes are centred on a non-zero current payload.
    # Validate the full central-difference stencil only after all point names
    # are available; it must vary exactly one named parameter around one
    # common physical anchor and must never be mistaken for a zero-centred
    # legacy closure probe.
    points_by_name = {str(point["name"]): point for point in points}
    for parameter in finite_difference:
        probes = [
            point
            for point in points
            if point["finite_difference_for"] == parameter
        ]
        anchored = [point for point in probes if point["finite_difference_anchor"] is not None]
        if not anchored:
            continue
        if len(anchored) != 2 or {str(point["probe_sign"]) for point in anchored} != {"positive", "negative"}:
            raise ValueError(
                f"iteration-centred finite differences for '{parameter}' require exactly one positive and one negative probe"
            )
        anchor_names = {str(point["finite_difference_anchor"]) for point in anchored}
        if len(anchor_names) != 1:
            raise ValueError(f"iteration-centred finite differences for '{parameter}' use inconsistent anchors")
        anchor_name = next(iter(anchor_names))
        anchor = points_by_name.get(anchor_name)
        if anchor is None:
            raise ValueError(
                f"iteration-centred finite-difference anchor '{anchor_name}' for '{parameter}' is absent"
            )
        anchor_values = anchor["alignment_parameter_values"]
        if not isinstance(anchor_values, Mapping):  # pragma: no cover - plan entries are constructed above
            raise RuntimeError("invalid constructed rigid anchor values")
        for point in anchored:
            values = point["alignment_parameter_values"]
            if not isinstance(values, Mapping):  # pragma: no cover - plan entries are constructed above
                raise RuntimeError("invalid constructed rigid finite-difference values")
            for other in finite_difference:
                if other == parameter:
                    continue
                if not math.isclose(
                    float(values[other]), float(anchor_values[other]), rel_tol=0.0, abs_tol=1.0e-12
                ):
                    raise ValueError(
                        f"finite-difference point '{point['name']}' changes '{other}' relative to anchor '{anchor_name}'"
                    )
            delta = float(values[parameter]) - float(anchor_values[parameter])
            if (point["probe_sign"] == "positive" and delta <= 0.0) or (
                point["probe_sign"] == "negative" and delta >= 0.0
            ):
                raise ValueError(
                    f"finite-difference point '{point['name']}' does not bracket anchor '{anchor_name}' for '{parameter}'"
                )
    return {
        "method": "physical_refit_station_rigid_multidof_scan",
        "scan_mode": "station_rigid_multidof",
        "station_ids": list(stations),
        "reference_station_ids": list(reference_stations),
        "movable_station_ids": list(movable_stations),
        "condition_axis": axis,
        "q_over_p_mode": int(config["q_over_p_mode"]),
        "alignment_parameter_specs": specs,
        "points": points,
    }


def _parse_layer_transform_map(
    raw_layers: object,
    *,
    point_name: str,
    station: int,
    required_layers: tuple[int, ...],
) -> dict[str, list[float]]:
    if not isinstance(raw_layers, Mapping):
        raise ValueError(f"rigid point '{point_name}' station {station} layer_transforms must be a mapping")
    parsed: dict[str, list[float]] = {}
    seen: set[int] = set()
    for raw_layer, raw_transform in raw_layers.items():
        try:
            layer = int(raw_layer)
        except (TypeError, ValueError) as error:
            raise ValueError(f"rigid point '{point_name}' has an invalid layer key") from error
        if layer not in required_layers or layer in seen:
            raise ValueError(f"rigid point '{point_name}' has duplicate/unsupported layer {layer}")
        seen.add(layer)
        parsed[str(layer)] = list(
            _station_transform(raw_transform, label=f"rigid point '{point_name}' station {station} layer {layer}")
        )
    if seen != set(required_layers):
        raise ValueError(
            f"rigid point '{point_name}' station {station} must declare layers {list(required_layers)}"
        )
    return parsed


def _hierarchy_parameter_value(
    spec: Mapping[str, Any],
    station_transforms: Mapping[str, Sequence[float]],
    layer_transforms: Mapping[str, Mapping[str, Sequence[float]]],
) -> float:
    index = int(spec["transform_index"])
    scale = float(spec["payload_scale"])
    scope = str(spec["scope"])
    station = str(int(spec["station_id"]))
    if scope == "station":
        return float(station_transforms[station][index] * scale)
    layers = layer_transforms[station]
    if scope == "contrast":
        return 0.5 * (float(layers["0"][index] * scale) - float(layers["2"][index] * scale))
    return float(layers[str(int(spec["layer_id"]))][index] * scale)


def _build_ift_layer_hierarchy_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build a joint station-common / IFT-layer-internal physical FD plan.

    Station 5-DoF remains the common-mode rigid transform.  IFT planes 0--2
    carry internal corrections.  Layer conditions are written as Calypso L2
    keys and are never silently copied into a station payload slot.  Station
    and layer ``dz`` stay identically zero except the hierarchical-V1 survey
    coordinate, which may be present in the normal equation but is written 0.
    """
    from alignment.layer_hierarchy import FREE_COMPONENTS, IFT_LAYER_IDS, IFT_STATION_ID

    stations = tuple(sorted({int(station) for station in config["station_ids"]}))
    if stations != (0, 1, 2, 3):
        raise ValueError("IFT layer hierarchy scans require station_ids [0, 1, 2, 3]")
    raw_reference = config.get("reference_station_ids")
    raw_movable = config.get("movable_station_ids")
    if not isinstance(raw_reference, (list, tuple)) or not isinstance(raw_movable, (list, tuple)):
        raise ValueError("IFT layer hierarchy scans require reference and movable station sets")
    reference_stations = tuple(sorted({int(station) for station in raw_reference}))
    movable_stations = tuple(sorted({int(station) for station in raw_movable}))
    if reference_stations != (1, 2, 3) or movable_stations != (0,):
        raise ValueError("IFT layer hierarchy scans must fix stations 1--3 and move only IFT station 0")
    raw_layers = config.get("movable_layer_ids", list(IFT_LAYER_IDS))
    if not isinstance(raw_layers, (list, tuple)):
        raise ValueError("IFT layer hierarchy scans require movable_layer_ids")
    movable_layers = tuple(sorted({int(layer) for layer in raw_layers}))
    if movable_layers != IFT_LAYER_IDS:
        raise ValueError("IFT layer hierarchy scans require movable_layer_ids [0, 1, 2]")
    if int(config.get("q_over_p_mode", -1)) != 0:
        raise ValueError("IFT layer hierarchy scans require q_over_p_mode: 0")
    axis = str(config.get("condition_axis", "ift_station_layer_hierarchy"))
    if not axis:
        raise ValueError("IFT layer hierarchy scans require a non-empty condition_axis")
    fit_basis = str(config.get("fit_basis", ""))
    hierarchical_v1 = fit_basis == "hierarchical_v1"

    raw_specs = config.get("alignment_parameter_specs")
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("IFT layer hierarchy scans require alignment_parameter_specs")
    specs: list[dict[str, Any]] = []
    names: set[str] = set()
    station_slots: set[str] = set()
    layer_slots: set[tuple[int, str]] = set()
    for raw_spec in raw_specs:
        if not isinstance(raw_spec, Mapping):
            raise ValueError("every alignment_parameter_specs entry must be a mapping")
        name = _safe_identifier(raw_spec.get("name", ""), label="alignment parameter name")
        if name in names:
            raise ValueError(f"duplicate alignment parameter name '{name}'")
        names.add(name)
        scope = str(raw_spec.get("scope", ""))
        if scope not in {"station", "layer", "contrast"}:
            raise ValueError(f"alignment parameter '{name}' must declare scope station, layer, or contrast")
        try:
            station = int(raw_spec["station_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"alignment parameter '{name}' requires station_id") from error
        if station != IFT_STATION_ID:
            raise ValueError(f"alignment parameter '{name}' must belong to IFT station {IFT_STATION_ID}")
        component = str(raw_spec.get("component", ""))
        if component == "dz_mm":
            if scope != "station" or not hierarchical_v1:
                raise ValueError(
                    f"alignment parameter '{name}' has unsupported component '{component}'; "
                    "layer hierarchy does not admit dz"
                )
        elif component not in FREE_COMPONENTS:
            raise ValueError(
                f"alignment parameter '{name}' has unsupported component '{component}'; "
                "layer hierarchy does not admit dz"
            )
        if component not in _ALIGNMENT_COMPONENTS:
            raise ValueError(f"alignment parameter '{name}' has unsupported component '{component}'")
        if (station, component) in station_slots and scope == "station":
            raise ValueError(f"station {station} component '{component}' is declared more than once")
        index, unit, payload_scale = _ALIGNMENT_COMPONENTS[component]
        supplied_unit = raw_spec.get("unit", unit)
        if str(supplied_unit) != unit:
            raise ValueError(f"alignment parameter '{name}' unit must be '{unit}' for component '{component}'")
        try:
            step = float(raw_spec["finite_difference_step"])
            severity_scale = float(raw_spec["severity_scale"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"alignment parameter '{name}' requires finite_difference_step and severity_scale"
            ) from error
        if not math.isfinite(step) or step <= 0.0 or not math.isfinite(severity_scale) or severity_scale <= 0.0:
            raise ValueError(f"alignment parameter '{name}' has non-positive/non-finite step or severity scale")
        spec: dict[str, Any] = {
            "name": name,
            "scope": scope,
            "station_id": station,
            "component": component,
            "unit": unit,
            "transform_index": index,
            "payload_scale": payload_scale,
            "finite_difference_step": step,
            "severity_scale": severity_scale,
        }
        if scope == "station":
            if "layer_id" in raw_spec:
                raise ValueError(f"station parameter '{name}' must not declare layer_id")
            station_slots.add(component)
        elif scope == "contrast":
            from alignment.layer_hierarchy import ADMITTED_CONTRAST_COMPONENTS, contrast_parameter_name

            if "layer_id" in raw_spec:
                raise ValueError(f"contrast parameter '{name}' must not declare layer_id")
            if component not in ADMITTED_CONTRAST_COMPONENTS:
                raise ValueError(
                    f"contrast parameter '{name}' does not admit component '{component}'; "
                    "relative ry, layer dy/rz, and layer1 stay frozen"
                )
            expected = contrast_parameter_name(component)
            if name != expected:
                raise ValueError(f"contrast parameter '{name}' must be named '{expected}'")
        else:
            try:
                layer = int(raw_spec["layer_id"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"layer parameter '{name}' requires layer_id") from error
            if layer not in movable_layers:
                raise ValueError(f"layer parameter '{name}' has unsupported layer_id {layer}")
            if (layer, component) in layer_slots:
                raise ValueError(f"layer {layer} component '{component}' is declared more than once")
            layer_slots.add((layer, component))
            spec["layer_id"] = layer
        specs.append(spec)
    scopes = {str(spec["scope"]) for spec in specs}
    contrast_only = scopes == {"contrast"}
    if hierarchical_v1:
        from alignment.hierarchical_v1 import C_DX, HIERARCHICAL_V1_PARAMETERS, STATION_SOLVE_PARAMETERS

        names_present = {str(spec["name"]) for spec in specs}
        if names_present != set(HIERARCHICAL_V1_PARAMETERS):
            raise ValueError(
                "hierarchical V1 requires station 5-DoF + survey dz + C_dx; "
                f"got {sorted(names_present)}"
            )
        if scopes != {"station", "contrast"}:
            raise ValueError("hierarchical V1 mixes station specs with C_dx contrast only")
        if any(str(spec["name"]) == "C_rx" for spec in specs):
            raise ValueError("hierarchical V1 does not admit C_rx")
        if C_DX not in names_present:
            raise ValueError("hierarchical V1 requires C_dx")
        missing_station = [name for name in STATION_SOLVE_PARAMETERS if name not in names_present]
        if missing_station:
            raise ValueError("hierarchical V1 is missing " + ", ".join(missing_station))
        if layer_slots:
            raise ValueError("hierarchical V1 does not admit layer-label parameters")
    elif "contrast" in scopes and not contrast_only:
        raise ValueError("contrast coordinates cannot be mixed with station or layer labels")
    if contrast_only:
        names_present = {str(spec["name"]) for spec in specs}
        if names_present not in ({"C_dx"}, {"C_dx", "C_rx"}):
            raise ValueError("contrast curriculum requires C_dx or exactly C_dx+C_rx")
    elif not hierarchical_v1:
        missing_station = [component for component in FREE_COMPONENTS if component not in station_slots]
        if missing_station:
            raise ValueError("IFT layer hierarchy is missing station common-mode: " + ", ".join(missing_station))
        missing_layers = [
            f"layer {layer} {component}"
            for component in FREE_COMPONENTS
            for layer in movable_layers
            if (layer, component) not in layer_slots
        ]
        if missing_layers:
            raise ValueError("IFT layer hierarchy is missing " + ", ".join(missing_layers))

    supplied_points = config.get("rigid_points")
    if not isinstance(supplied_points, list) or not supplied_points:
        raise ValueError("IFT layer hierarchy scans require a non-empty rigid_points list")
    points: list[dict[str, Any]] = []
    point_names: set[str] = set()
    nominal_count = 0
    finite_difference: dict[str, set[str]] = {str(spec["name"]): set() for spec in specs}
    for point_index, raw_point in enumerate(supplied_points):
        if not isinstance(raw_point, Mapping):
            raise ValueError("every hierarchy point must be a mapping")
        name = _safe_identifier(raw_point.get("name", ""), label="hierarchy point name")
        if name in point_names:
            raise ValueError(f"duplicate hierarchy point name '{name}'")
        point_names.add(name)
        raw_transforms = raw_point.get("station_transforms")
        if not isinstance(raw_transforms, Mapping):
            raise ValueError(f"hierarchy point '{name}' requires station_transforms")
        transforms: dict[str, list[float]] = {}
        parsed_stations: set[int] = set()
        for raw_station, raw_transform in raw_transforms.items():
            try:
                station = int(raw_station)
            except (TypeError, ValueError) as error:
                raise ValueError(f"hierarchy point '{name}' has an invalid station key") from error
            if station not in stations or station in parsed_stations:
                raise ValueError(f"hierarchy point '{name}' has duplicate/unsupported station {station}")
            parsed_stations.add(station)
            transforms[str(station)] = list(
                _station_transform(raw_transform, label=f"hierarchy point '{name}' station {station}")
            )
        if parsed_stations != set(stations):
            raise ValueError(f"hierarchy point '{name}' must explicitly specify every station transform")
        for station in reference_stations:
            if not np_allclose_zero(transforms[str(station)]):
                raise ValueError(f"hierarchy point '{name}' moves reference station {station}")
        raw_layer_block = raw_point.get("layer_transforms")
        if not isinstance(raw_layer_block, Mapping):
            raise ValueError(f"hierarchy point '{name}' requires layer_transforms")
        raw_ift_layers = raw_layer_block.get(IFT_STATION_ID, raw_layer_block.get(str(IFT_STATION_ID)))
        extra_stations = [
            key
            for key in raw_layer_block
            if int(key) != IFT_STATION_ID
        ]
        if extra_stations or raw_ift_layers is None:
            raise ValueError(f"hierarchy point '{name}' requires layer_transforms only for IFT station 0")
        layer_transforms = {
            str(IFT_STATION_ID): _parse_layer_transform_map(
                raw_ift_layers,
                point_name=name,
                station=IFT_STATION_ID,
                required_layers=movable_layers,
            )
        }
        derived_values = {
            str(spec["name"]): _hierarchy_parameter_value(spec, transforms, layer_transforms) for spec in specs
        }
        permitted_station = {
            int(spec["transform_index"]) for spec in specs if spec["scope"] == "station"
        }
        for index, value in enumerate(transforms[str(IFT_STATION_ID)]):
            if index not in permitted_station and not math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1.0e-15):
                raise ValueError(
                    f"hierarchy point '{name}' changes IFT station component index {index} "
                    "without an enabled station alignment parameter"
                )
        permitted_layer = {
            (int(spec["layer_id"]), int(spec["transform_index"]))
            for spec in specs
            if spec["scope"] == "layer"
        }
        for spec in specs:
            if spec["scope"] != "contrast":
                continue
            index = int(spec["transform_index"])
            permitted_layer.add((0, index))
            permitted_layer.add((2, index))
        for layer, values in layer_transforms[str(IFT_STATION_ID)].items():
            for index, value in enumerate(values):
                if (int(layer), index) not in permitted_layer and not math.isclose(
                    value, 0.0, rel_tol=0.0, abs_tol=1.0e-15
                ):
                    raise ValueError(
                        f"hierarchy point '{name}' changes IFT layer {layer} component index {index} "
                        "without an enabled layer alignment parameter"
                    )
        raw_values = raw_point.get("alignment_parameter_values")
        if raw_values is not None:
            if not isinstance(raw_values, Mapping) or set(str(key) for key in raw_values) != set(derived_values):
                raise ValueError(
                    f"hierarchy point '{name}' alignment_parameter_values must specify exactly {sorted(derived_values)}"
                )
            for parameter, expected in derived_values.items():
                try:
                    observed = float(raw_values[parameter])
                except (TypeError, ValueError, KeyError) as error:
                    raise ValueError(f"hierarchy point '{name}' has invalid value for '{parameter}'") from error
                if not math.isfinite(observed) or not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1.0e-12):
                    raise ValueError(
                        f"hierarchy point '{name}' parameter '{parameter}' disagrees with payload transforms"
                    )
        normalized = [derived_values[str(spec["name"])] / float(spec["severity_scale"]) for spec in specs]
        severity = float(math.sqrt(sum(value * value for value in normalized)))
        role = str(raw_point.get("point_role", "curriculum"))
        is_zero = all(math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1.0e-15) for value in derived_values.values())
        if role == "nominal":
            if not is_zero:
                raise ValueError(f"nominal hierarchy point '{name}' must be the all-zero payload")
            nominal_count += 1
        elif is_zero:
            if not (bool(raw_point.get("allow_identically_zero")) and role == "held_out_closure"):
                raise ValueError(f"non-nominal hierarchy point '{name}' must contain a non-zero enabled parameter")
        finite_difference_for = raw_point.get("finite_difference_for")
        probe_sign = raw_point.get("probe_sign")
        finite_difference_anchor = raw_point.get("finite_difference_anchor")
        if finite_difference_for is not None or probe_sign is not None:
            parameter = str(finite_difference_for)
            if parameter not in derived_values or str(probe_sign) not in {"positive", "negative"}:
                raise ValueError(f"hierarchy point '{name}' has an invalid finite-difference tag")
            sign = str(probe_sign)
            if role not in {"finite_difference_positive", "finite_difference_negative"} or not role.endswith(sign):
                raise ValueError(f"hierarchy point '{name}' finite-difference point_role/sign disagree")
            if finite_difference_anchor is None:
                for other, value in derived_values.items():
                    if other != parameter and not math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1.0e-15):
                        raise ValueError(f"finite-difference point '{name}' must vary only '{parameter}'")
                if (sign == "positive" and derived_values[parameter] <= 0.0) or (
                    sign == "negative" and derived_values[parameter] >= 0.0
                ):
                    raise ValueError(f"finite-difference point '{name}' does not bracket zero for '{parameter}'")
            else:
                anchor_name = _safe_identifier(
                    finite_difference_anchor, label=f"hierarchy point '{name}' finite-difference anchor"
                )
                if anchor_name == name:
                    raise ValueError(f"finite-difference point '{name}' cannot anchor itself")
                finite_difference_anchor = anchor_name
            finite_difference[parameter].add(sign)
        else:
            if finite_difference_anchor is not None:
                raise ValueError(f"hierarchy point '{name}' has a finite-difference anchor without probe tags")
            if role.startswith("finite_difference_"):
                raise ValueError(f"hierarchy point '{name}' has a finite-difference role without tags")
        try:
            condition_value = float(raw_point.get("condition_value", severity))
            condition_magnitude = float(raw_point.get("condition_magnitude", severity))
        except (TypeError, ValueError) as error:
            raise ValueError(f"hierarchy point '{name}' has invalid condition metadata") from error
        if (
            not math.isfinite(condition_value)
            or not math.isfinite(condition_magnitude)
            or condition_magnitude < 0.0
        ):
            raise ValueError(f"hierarchy point '{name}' has non-finite/negative condition metadata")
        points.append(
            {
                "index": point_index,
                "name": name,
                "relative_point_dir": str(Path("points") / name),
                "direction_trial": str(raw_point.get("direction_trial", name)),
                "point_role": role,
                "finite_difference_for": None if finite_difference_for is None else str(finite_difference_for),
                "probe_sign": None if probe_sign is None else str(probe_sign),
                "finite_difference_anchor": (
                    None if finite_difference_anchor is None else str(finite_difference_anchor)
                ),
                "condition_axis": axis,
                "condition_value": condition_value,
                "condition_magnitude": condition_magnitude,
                "alignment_parameter_values": derived_values,
                "injected_station_transforms": transforms,
                "injected_layer_transforms": layer_transforms,
            }
        )
    if nominal_count != 1:
        if bool(config.get("held_out_only", False)) and nominal_count == 0:
            pass
        else:
            raise ValueError("IFT layer hierarchy scan must contain exactly one nominal all-zero payload")
    missing_probes = [name for name, signs in finite_difference.items() if signs != {"positive", "negative"}]
    if missing_probes and not bool(config.get("held_out_only", False)):
        raise ValueError(
            "IFT layer hierarchy scan requires positive/negative physical probes for: "
            + ", ".join(sorted(missing_probes))
        )
    if bool(config.get("held_out_only", False)):
        if not any(str(point.get("point_role")) == "held_out_closure" for point in points):
            raise ValueError("held_out_only hierarchy scans require at least one held_out_closure point")
        if any(point.get("finite_difference_for") is not None for point in points):
            raise ValueError("held_out_only hierarchy scans must not reproduce finite-difference probes")
    points_by_name = {str(point["name"]): point for point in points}
    for parameter in finite_difference:
        probes = [point for point in points if point["finite_difference_for"] == parameter]
        anchored = [point for point in probes if point["finite_difference_anchor"] is not None]
        if not anchored:
            continue
        if len(anchored) != 2 or {str(point["probe_sign"]) for point in anchored} != {"positive", "negative"}:
            raise ValueError(
                f"iteration-centred finite differences for '{parameter}' require exactly one positive and one negative probe"
            )
        anchor_names = {str(point["finite_difference_anchor"]) for point in anchored}
        if len(anchor_names) != 1:
            raise ValueError(f"iteration-centred finite differences for '{parameter}' use inconsistent anchors")
        anchor_name = next(iter(anchor_names))
        anchor = points_by_name.get(anchor_name)
        if anchor is None:
            raise ValueError(
                f"iteration-centred finite-difference anchor '{anchor_name}' for '{parameter}' is absent"
            )
        anchor_values = anchor["alignment_parameter_values"]
        if not isinstance(anchor_values, Mapping):
            raise RuntimeError("invalid constructed hierarchy anchor values")
        for point in anchored:
            values = point["alignment_parameter_values"]
            if not isinstance(values, Mapping):
                raise RuntimeError("invalid constructed hierarchy finite-difference values")
            for other in finite_difference:
                if other == parameter:
                    continue
                if not math.isclose(
                    float(values[other]), float(anchor_values[other]), rel_tol=0.0, abs_tol=1.0e-12
                ):
                    raise ValueError(
                        f"finite-difference point '{point['name']}' changes '{other}' relative to anchor '{anchor_name}'"
                    )
            delta = float(values[parameter]) - float(anchor_values[parameter])
            if (point["probe_sign"] == "positive" and delta <= 0.0) or (
                point["probe_sign"] == "negative" and delta >= 0.0
            ):
                raise ValueError(
                    f"finite-difference point '{point['name']}' does not bracket anchor '{anchor_name}' for '{parameter}'"
                )
    return {
        "method": "physical_refit_ift_layer_hierarchy_scan",
        "scan_mode": "ift_layer_hierarchy",
        "fit_basis": fit_basis,
        "station_ids": list(stations),
        "reference_station_ids": list(reference_stations),
        "movable_station_ids": list(movable_stations),
        "movable_layer_ids": list(movable_layers),
        "layer_station_id": IFT_STATION_ID,
        "condition_axis": axis,
        "q_over_p_mode": 0,
        "alignment_parameter_specs": specs,
        "held_out_only": bool(config.get("held_out_only", False)),
        "points": points,
    }


def _build_ift_ry_rotation_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build an explicit physical IFT ``R_y`` bank without a coordinate surrogate.

    This mode deliberately accepts an explicit list of payload transforms rather
    than generating offsets algebraically.  That makes the downstream three
    station reference and every joint ``dx/dy + R_y`` trial part of the frozen
    plan that is written to ``/Tracker/Align``.
    """
    stations = tuple(sorted({int(station) for station in config["station_ids"]}))
    if stations != (0, 1, 2, 3):
        raise ValueError("IFT R_y scans require the four station IDs [0, 1, 2, 3]")
    raw_reference = config.get("reference_station_ids")
    raw_movable = config.get("movable_station_ids")
    if not isinstance(raw_reference, (list, tuple)) or not isinstance(raw_movable, (list, tuple)):
        raise ValueError("IFT R_y scans require reference_station_ids and movable_station_ids")
    reference_stations = tuple(sorted({int(station) for station in raw_reference}))
    movable_stations = tuple(sorted({int(station) for station in raw_movable}))
    if reference_stations != (1, 2, 3) or movable_stations != (0,):
        raise ValueError(
            "IFT R_y scans must fix downstream stations [1, 2, 3] and vary only IFT station 0"
        )
    axis = str(config.get("condition_axis", "ift_ry_mrad"))
    if axis != "ift_ry_mrad":
        raise ValueError("IFT R_y scans require condition_axis='ift_ry_mrad'")
    supplied_points = config.get("rotation_points")
    if not isinstance(supplied_points, list) or not supplied_points:
        raise ValueError("IFT R_y scans require a non-empty rotation_points list")

    points: list[dict[str, Any]] = []
    names: set[str] = set()
    zero_points = 0
    for index, raw_point in enumerate(supplied_points):
        if not isinstance(raw_point, Mapping):
            raise ValueError("every IFT R_y rotation point must be a mapping")
        name = str(raw_point.get("name", ""))
        if not name or not name.replace("_", "").isalnum() or name in names:
            raise ValueError("IFT R_y rotation point names must be unique alphanumeric identifiers")
        names.add(name)
        try:
            value_mrad = float(raw_point["ry_mrad"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"IFT R_y point '{name}' requires finite ry_mrad") from error
        if not math.isfinite(value_mrad):
            raise ValueError(f"IFT R_y point '{name}' has non-finite ry_mrad")
        raw_transforms = raw_point.get("station_transforms")
        if not isinstance(raw_transforms, Mapping):
            raise ValueError(f"IFT R_y point '{name}' requires station_transforms")
        transforms: dict[str, list[float]] = {}
        parsed_keys: set[int] = set()
        for raw_station, raw_transform in raw_transforms.items():
            try:
                station = int(raw_station)
            except (TypeError, ValueError) as error:
                raise ValueError(f"IFT R_y point '{name}' has an invalid station key") from error
            if station not in stations or station in parsed_keys:
                raise ValueError(f"IFT R_y point '{name}' has duplicate or unsupported station {station}")
            parsed_keys.add(station)
            transform = _station_transform(raw_transform, label=f"IFT R_y point '{name}' station {station}")
            transforms[str(station)] = list(transform)
        if parsed_keys != set(stations):
            raise ValueError(f"IFT R_y point '{name}' must explicitly specify every station transform")
        for station in reference_stations:
            if not np_allclose_zero(transforms[str(station)]):
                raise ValueError(
                    f"IFT R_y point '{name}' moves reference downstream station {station}"
                )
        if not math.isclose(transforms["0"][4], value_mrad * 1.0e-3, rel_tol=0.0, abs_tol=1.0e-14):
            raise ValueError(
                f"IFT R_y point '{name}' has ry={transforms['0'][4]:.17g} rad, "
                f"expected {value_mrad:.17g} mrad"
            )
        # The present study is intentionally one rotational degree of freedom
        # plus optional physical x/y translations.  Do not silently admit a
        # second rotation or a z shift under the same label.
        if not math.isclose(transforms["0"][2], 0.0, rel_tol=0.0, abs_tol=1.0e-15):
            raise ValueError(f"IFT R_y point '{name}' must keep IFT dz at zero")
        if not math.isclose(transforms["0"][3], 0.0, rel_tol=0.0, abs_tol=1.0e-15) or not math.isclose(
            transforms["0"][5], 0.0, rel_tol=0.0, abs_tol=1.0e-15
        ):
            raise ValueError(f"IFT R_y point '{name}' may vary only dx, dy, and ry")
        is_zero = all(np_allclose_zero(values) for values in transforms.values())
        zero_points += int(is_zero)
        points.append(
            {
                "index": index,
                "name": name,
                "relative_point_dir": str(Path("points") / name),
                "direction_trial": str(raw_point.get("direction_trial", name)),
                "condition_axis": axis,
                "condition_value": value_mrad,
                "condition_magnitude": abs(value_mrad),
                "ry_mrad": value_mrad,
                "injected_station_transforms": transforms,
                "joint_translation_xy_mm": [transforms["0"][0], transforms["0"][1]],
            }
        )
    if zero_points != 1:
        raise ValueError("IFT R_y scan must contain exactly one all-zero nominal payload")
    return {
        "method": "physical_refit_ift_ry_rotation_scan",
        "scan_mode": "ift_ry_rotation",
        "station_ids": list(stations),
        "reference_station_ids": list(reference_stations),
        "movable_station_ids": list(movable_stations),
        "condition_axis": axis,
        "q_over_p_mode": int(config["q_over_p_mode"]),
        "points": points,
    }


def np_allclose_zero(values: object) -> bool:
    """Avoid a NumPy dependency in this lightweight orchestration driver."""
    try:
        return all(math.isclose(float(value), 0.0, rel_tol=0.0, abs_tol=1.0e-15) for value in values)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _build_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    """Dispatch legacy translation scans and explicit rigid physical payloads."""
    mode = str(config.get("scan_mode", "translation_xy"))
    if mode == "translation_xy":
        return _build_translation_plan(config)
    if mode == "ift_ry_rotation":
        return _build_ift_ry_rotation_plan(config)
    if mode == "station_rigid_multidof":
        return _build_station_rigid_multidof_plan(config)
    if mode == "ift_layer_hierarchy":
        return _build_ift_layer_hierarchy_plan(config)
    raise ValueError(f"unsupported physical refit scan_mode '{mode}'")


def _run_shell(
    command: str,
    log_path: Path,
    dry_run: bool,
    cwd: Path = WORKSPACE_ROOT,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"[dry-run] {command}")
        return
    with log_path.open("w", encoding="utf-8") as log:
        log.write("# command\n")
        log.write(command)
        log.write("\n\n# output\n")
        log.flush()
        result = subprocess.run(
            ["bash", "-lc", command],
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def _calypso_command(command: str) -> str:
    # The driver is often launched from LCG_110_cuda for its YAML/plotting
    # dependencies.  Do not let that Python 3.13 stack leak into Athena's
    # Python 3.9/LCG_104d process.  Stale setup markers inherited from a
    # polluted submit/login shell (e.g. via Condor `getenv = True`) make the
    # Athena / AthenaExternals / Calypso InstallArea setup scripts return early
    # without exporting PYTHONPATH.  Cluster 1000432 failed payload writes
    # because Calypso_SET_UP was left set after a submit-side calypso build.
    clean_environment = (
        "unset PYTHONPATH LD_LIBRARY_PATH ROOTSYS ROOT_INCLUDE_PATH PYTHONHOME"
        " Athena_SET_UP Athena_EXTONLY_SET_UP Athena_RELONLY_SET_UP"
        " AthenaExternals_SET_UP AthenaExternals_EXTONLY_SET_UP"
        " AthenaExternals_RELONLY_SET_UP"
        " Calypso_SET_UP Calypso_EXTONLY_SET_UP Calypso_RELONLY_SET_UP"
    )
    return f"{clean_environment}\nsource {_quote(SETUP_SCRIPT)} calypso\n{command}"


def _ml_command(command: str) -> str:
    return f"source {_quote(SETUP_SCRIPT)} ml\ncd {_quote(PROJECT_ROOT)}\n{command}"


def _write_failure(point_dir: Path, phase: str, error: Exception) -> None:
    payload = {
        "status": "failed",
        "phase": phase,
        "message": str(error),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (point_dir / "failure.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _audit_has_positive_tracklets(path: Path) -> bool:
    """Empty ROOT + return code 0 is not a completed physical point."""
    if not path.is_file():
        return False
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(audit, dict):
        return False
    try:
        return int(audit.get("events", 0)) > 0 and int(audit.get("tracklets", 0)) > 0
    except (TypeError, ValueError):
        return False


def _is_complete(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _has_ntuple_tree(path: Path) -> bool:
    """Return whether a refit output is a readable NtupleDumper ROOT file.

    Athena can leave a small ROOT header behind if a job is interrupted after
    opening the output file but before the event loop writes ``nt``.  Treating
    that header as a completed refit makes ``--resume`` reuse a physically
    invalid artifact.  The scan driver runs in the ML environment, where
    uproot is available, so validate the required tree before resuming.
    """
    if not _is_complete(path):
        return False
    try:
        import uproot

        with uproot.open(path) as source:
            return "nt" in source
    except Exception:
        return False


def _discard_incomplete_refit(path: Path) -> None:
    """Remove only a known-invalid generated refit artifact before retrying."""
    if path.exists():
        path.unlink()


def _baseline_refit_dir(
    plan: Mapping[str, Any],
    scan_root: Path,
    *,
    run_alignment_closure: bool,
) -> Path:
    """Locate the zero-magnitude baseline, or skip it for held-out-only scans.

    Legacy xy/rigid closures compare every displaced point to a nominal refit
    that lives in the same scan.  A held-out-only hierarchy scan has no
    all-zero payload by construction: the Jacobian nominal already exists in
    a frozen FD bank, and this scan only produces new observed points.
    """
    if plan["scan_mode"] == "translation_xy":
        zeros = [point for point in plan["points"] if float(point["magnitude_mm"]) == 0.0]
    else:
        zeros = [point for point in plan["points"] if float(point["condition_magnitude"]) == 0.0]
    if len(zeros) > 1:
        raise ValueError("scan plan has more than one zero-magnitude baseline point")
    if len(zeros) == 1:
        return scan_root / str(zeros[0]["relative_point_dir"]) / "refit"
    if bool(plan.get("held_out_only", False)) and not run_alignment_closure:
        return scan_root / "_unused_held_out_only_baseline" / "refit"
    raise ValueError("scan plan lacks a zero-magnitude baseline point")


def _ntuple_maker_command(
    *,
    input_xaod: Path,
    payload_dir: Path,
    outfile: Path,
    nevents: int,
    skip_events: int,
    is_mc: bool,
) -> str:
    """Build the Calypso ntuple command for one physical-scan point.

    Real-data occupancy windows are selected on current-geometry SegmentFit
    refit tracklets.  NtupleDumper's default ``DoTrackFilter`` keeps only
    events with a CKF long track, and default ``StableOnly`` requires
    ``FaserLHCData.stableBeams()``.  2024 r0022 xAOD used here has SegmentFit
    occupancy but ``stableBeams()`` false on every scanned event, so both
    defaults empty the ntuple.  Alignment export therefore passes
    ``--NoTrackFilt --no_stable``.  Blinding stays on.  MC still uses
    ``--isMC``, which skips the real-data filter block.
    """
    maker = [
        "faser_ntuple_maker.py",
        _quote(input_xaod),
    ]
    if is_mc:
        maker.append("--isMC")
    else:
        maker.append("--NoTrackFilt")
        maker.append("--no_stable")
    maker.extend(
        [
            "--useIFT",
            f"--nevents {int(nevents)}",
            *(
                [f"--skip-events {int(skip_events)}"]
                if int(skip_events or 0) > 0
                else []
            ),
            "--export-tracklets",
            "--export-tracklet-propagation",
            "--refit-segments",
            f"--tracker-align-sqlite {_quote(payload_dir / 'tracker_alignment.sqlite')}",
            f"--tracker-align-pool-catalog {_quote(payload_dir / 'PoolFileCatalog.xml')}",
            f"--outfile {_quote(outfile)}",
        ]
    )
    return " ".join(maker)


def _run_point(
    point: Mapping[str, Any],
    config: Mapping[str, Any],
    scan_root: Path,
    baseline_refit: Path,
    dry_run: bool,
    resume: bool,
) -> bool:
    point_dir = scan_root / str(point["relative_point_dir"])
    point_dir.mkdir(parents=True, exist_ok=True)
    payload_dir = point_dir / "payload"
    refit_dir = point_dir / "refit"
    closure_dir = point_dir / "closure"
    manifest = payload_dir / "alignment_payload.json"
    enhanced = refit_dir / "enhanced_tracklets.root"
    tracklets = refit_dir / "tracklets.root"
    propagations = refit_dir / "propagations.root"
    audit = refit_dir / "content_audit.json"
    closure = closure_dir / "closure.json"

    try:
        if not _is_complete(manifest):
            phase = "payload"
            if "injected_station_transforms" in point:
                transforms = point["injected_station_transforms"]
                if not isinstance(transforms, Mapping):
                    raise ValueError("rigid scan point has invalid injected_station_transforms")
                arguments = " ".join(
                    "--transform "
                    + str(int(station))
                    + ":"
                    + ":".join(f"{float(component):.17g}" for component in values)
                    for station, values in sorted(transforms.items(), key=lambda item: int(item[0]))
                )
                layer_block = point.get("injected_layer_transforms")
                if layer_block is not None:
                    if not isinstance(layer_block, Mapping):
                        raise ValueError("rigid scan point has invalid injected_layer_transforms")
                    layer_arguments = []
                    for raw_station, raw_layers in sorted(layer_block.items(), key=lambda item: int(item[0])):
                        if not isinstance(raw_layers, Mapping):
                            raise ValueError("rigid scan point has invalid injected_layer_transforms")
                        for raw_layer, values in sorted(raw_layers.items(), key=lambda item: int(item[0])):
                            layer_arguments.append(
                                "--layer-transform "
                                + str(int(raw_station))
                                + ":"
                                + str(int(raw_layer))
                                + ":"
                                + ":".join(f"{float(component):.17g}" for component in values)
                            )
                    if layer_arguments:
                        arguments = arguments + " " + " ".join(layer_arguments)
            else:
                offsets = point["injected_offsets_xy_mm"]
                arguments = " ".join(
                    f"--offset {int(station)}:{float(values[0]):.17g}:{float(values[1]):.17g}"
                    for station, values in sorted(offsets.items(), key=lambda item: int(item[0]))
                )
            _run_shell(
                _calypso_command(
                    f"python {_quote(PAYLOAD_WRITER)} --output-dir {_quote(payload_dir)} {arguments}"
                ),
                point_dir / "logs" / "payload.log",
                dry_run,
                cwd=point_dir,
            )
        if not _has_ntuple_tree(enhanced):
            phase = "segment_refit"
            refit_dir.mkdir(parents=True, exist_ok=True)
            _discard_incomplete_refit(enhanced)
            input_xaod = Path(str(config["input_xaod"])).expanduser().resolve()
            command = _ntuple_maker_command(
                input_xaod=input_xaod,
                payload_dir=payload_dir,
                outfile=enhanced,
                nevents=int(config["nevents"]),
                skip_events=int(config.get("skip_events", 0) or 0),
                is_mc=bool(config.get("is_mc", True)),
            )
            _run_shell(
                _calypso_command(command), point_dir / "logs" / "refit.log", dry_run, cwd=point_dir
            )
        if not (_is_complete(tracklets) and _is_complete(propagations) and _is_complete(audit)):
            phase = "export_and_propagation"
            command = "\n".join(
                [
                    " ".join(
                        [
                            "python scripts/convert_ntuple_tracklets.py",
                            _quote(enhanced),
                            "--output",
                            _quote(tracklets),
                            *(
                                ["--include-truth"]
                                if bool(config.get("include_truth", bool(config.get("is_mc", True))))
                                else []
                            ),
                        ]
                    ),
                    " ".join(
                        [
                            "python scripts/convert_ntuple_tracklet_propagations.py",
                            _quote(enhanced),
                            "--output",
                            _quote(propagations),
                        ]
                    ),
                    " ".join(
                        [
                            "python scripts/audit_tracklets.py",
                            _quote(tracklets),
                            "--output",
                            _quote(audit),
                            *(
                                ["--require-mc-labels"]
                                if bool(config.get("require_mc_labels", bool(config.get("is_mc", True))))
                                else []
                            ),
                            *(
                                ["--physical-order"]
                                if bool(config.get("merged_rec_physical_order", False))
                                else []
                            ),
                        ]
                    ),
                ]
            )
            _run_shell(_ml_command(command), point_dir / "logs" / "export.log", dry_run)
        if bool(config.get("run_alignment_closure", True)) and not _is_complete(closure):
            phase = "alignment_closure"
            if "injected_station_transforms" in point:
                raise ValueError(
                    "translation-only run_refit_alignment_closure.py cannot be used for a rigid "
                    "physical payload; run the dedicated finite-difference closure after the scan instead"
                )
            if not dry_run and not (
                _is_complete(baseline_refit / "tracklets.root")
                and _is_complete(baseline_refit / "propagations.root")
            ):
                raise RuntimeError("the zero-offset physical baseline has not completed")
            command = " ".join(
                [
                    "python scripts/run_refit_alignment_closure.py",
                    "--nominal-tracklets",
                    _quote(baseline_refit / "tracklets.root"),
                    "--nominal-propagations",
                    _quote(baseline_refit / "propagations.root"),
                    "--displaced-tracklets",
                    _quote(tracklets),
                    "--displaced-propagations",
                    _quote(propagations),
                    "--payload-manifest",
                    _quote(manifest),
                    "--output-dir",
                    _quote(closure_dir),
                    "--reference-station",
                    str(int(config["reference_station"])),
                    "--q-over-p-mode",
                    str(int(config["q_over_p_mode"])),
                    "--min-truth-match-fraction",
                    str(float(config["min_truth_match_fraction"])),
                    "--refinement-iterations",
                    str(int(config["refinement_iterations"])),
                    "--chi2-gate",
                    str(float(config["chi2_gate"])),
                ]
                + (
                    []
                    if config.get("prior_sigma_mm") is None
                    else ["--prior-sigma-mm", str(float(config["prior_sigma_mm"]))]
                )
            )
            _run_shell(_ml_command(command), point_dir / "logs" / "closure.log", dry_run)
        if not dry_run and not _audit_has_positive_tracklets(audit):
            phase = "content_audit"
            raise RuntimeError(
                "content_audit_empty: n_tracklets must be > 0; empty ROOT is not success"
            )
    except Exception as error:
        if not dry_run:
            _write_failure(point_dir, phase=locals().get("phase", "unknown"), error=error)
        print(f"[failed] {point['name']}: {error}", file=sys.stderr)
        return False
    if not dry_run:
        failure = point_dir / "failure.json"
        if failure.exists():
            failure.unlink()
    print(f"[complete] {point['name']}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "physical_refit_capture_scan_muon.yaml"),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument(
        "--point-name",
        action="append",
        default=None,
        help="Run only this named frozen plan point; repeat for a staged physical smoke",
    )
    args = parser.parse_args()
    if args.max_points is not None and args.max_points < 1:
        parser.error("--max-points must be positive")
    if args.max_points is not None and args.point_name:
        parser.error("--max-points and --point-name cannot be used together")

    config_path = Path(args.config).expanduser().resolve()
    config = _load_config(config_path)
    if int(config.get("q_over_p_mode", -1)) != 0:
        raise ValueError("physical V1 capture scans must use q_over_p_mode=0")
    is_mc = bool(config.get("is_mc", True))
    include_truth = bool(config.get("include_truth", is_mc))
    require_mc_labels = bool(config.get("require_mc_labels", is_mc))
    if not is_mc and (include_truth or require_mc_labels):
        raise ValueError("real-data physical scans must not include or require MC labels")
    if is_mc and not include_truth:
        raise ValueError("MC physical scans must export MC labels")
    config["is_mc"] = is_mc
    config["include_truth"] = include_truth
    config["require_mc_labels"] = require_mc_labels
    merged_rec_physical_order = bool(config.get("merged_rec_physical_order", False))
    if "merged_rec_physical_order" in config and not isinstance(
        config["merged_rec_physical_order"], bool
    ):
        raise ValueError("merged_rec_physical_order must be a boolean when set")
    config["merged_rec_physical_order"] = merged_rec_physical_order
    if not Path(str(config["input_xaod"])).expanduser().is_file():
        raise FileNotFoundError(f"input xAOD is unavailable: {config['input_xaod']}")
    plan = _build_plan(config)
    rigid_scan = plan["scan_mode"] in {"ift_ry_rotation", "station_rigid_multidof", "ift_layer_hierarchy"}
    if rigid_scan and "run_alignment_closure" not in config:
        # The legacy closure is xy-only.  Make every rigid physical driver
        # safe by default until its dedicated finite-difference closure is
        # invoked explicitly after the response audit.
        config["run_alignment_closure"] = False
    if rigid_scan and bool(config.get("run_alignment_closure", False)):
        raise ValueError(
            "rigid physical scans must set run_alignment_closure=false and use a dedicated "
            "finite-difference closure explicitly after physical response validation"
        )
    scan_root = Path(args.output_dir).expanduser().resolve()
    if scan_root.exists() and any(scan_root.iterdir()) and not args.resume:
        raise FileExistsError("scan output already exists; use --resume only for the identical plan")
    scan_root.mkdir(parents=True, exist_ok=True)
    plan_path = scan_root / "scan_plan.json"
    if plan_path.is_file():
        existing_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing_plan != plan:
            raise ValueError("existing scan plan differs from the requested configuration")
    else:
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (scan_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                **config,
                "config_source": str(config_path),
                "output_dir": str(scan_root),
                "physical_geometry_repropagation": True,
                "refit_chain": (
                    "persisted SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit -> "
                    "NtupleDumper -> FaserActsExtrapolationTool"
                ),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    baseline_refit = _baseline_refit_dir(
        plan,
        scan_root,
        run_alignment_closure=bool(config.get("run_alignment_closure", True)),
    )
    points = plan["points"]
    if args.point_name:
        requested = set(str(value) for value in args.point_name)
        known = {str(point["name"]) for point in points}
        unknown = requested - known
        if unknown:
            parser.error("unknown --point-name: " + ", ".join(sorted(unknown)))
        points = [point for point in points if str(point["name"]) in requested]
    elif args.max_points is not None:
        points = points[: args.max_points]
    completed = 0
    for point in points:
        completed += int(
            _run_point(
                point,
                config,
                scan_root,
                baseline_refit,
                dry_run=args.dry_run,
                resume=args.resume,
            )
        )
    if not args.dry_run and bool(config.get("run_alignment_closure", True)):
        summary_command = " ".join(
            [
                "python scripts/summarize_physical_refit_capture_scan.py",
                "--scan-root",
                _quote(scan_root),
                "--scan-plan",
                _quote(plan_path),
                "--capture-tolerance-mm",
                str(float(config["capture_tolerance_mm"])),
            ]
        )
        _run_shell(_ml_command(summary_command), scan_root / "summary.log", dry_run=False)
    elif not args.dry_run:
        (scan_root / "physical_refit_summary.json").write_text(
            json.dumps(
                {
                    "method": str(plan["method"]),
                    "scan_mode": str(plan["scan_mode"]),
                    "physical_geometry_repropagation": True,
                    "run_alignment_closure": False,
                    "points_requested": len(points),
                    "points_completed": completed,
                    **(
                        {}
                        if plan["scan_mode"] == "translation_xy"
                        else {
                            "condition_axis": str(plan["condition_axis"]),
                            "reference_station_ids": list(plan["reference_station_ids"]),
                            "movable_station_ids": list(plan["movable_station_ids"]),
                            **(
                                {}
                                if plan["scan_mode"] != "station_rigid_multidof"
                                else {"alignment_parameter_specs": list(plan["alignment_parameter_specs"])}
                            ),
                        }
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"physical scan points completed: {completed}/{len(points)}")
    # A corpus worker must make failed payload/refit/export points visible to
    # its scheduler.  Previously a failed point only changed the printed count
    # and the parent process still returned success, which could admit an
    # incomplete physical source to a later manifest refresh.
    if completed != len(points):
        raise SystemExit(f"physical scan incomplete: completed {completed}/{len(points)} points")


if __name__ == "__main__":
    main()
