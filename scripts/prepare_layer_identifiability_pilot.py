#!/usr/bin/env python3
"""Compile the IFT station/layer identifiability pilot around converged geometry.

The station-level 5-DoF framework stays frozen: the linearization point is the
all-zero (already captured) station payload.  Each IFT plane receives independent
central finite-difference probes in dx/dy/rx/ry/rz, and the same five station
DoF are probed so the joint Jacobian can separate common mode from internal
deformation.  ``dz`` is identically zero.  This compiler never opens test data
and never maps a layer condition onto a station payload slot.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from alignment.contrast_sampling import CONTRAST_ENVELOPE, CONTRAST_PARAMETERS, inside_contrast_envelope
from alignment.layer_hierarchy import IFT_LAYER_IDS, IFT_STATION_ID, spec_scope
from alignment.physical_jacobian import payload_transforms_with_parameter_values
from scripts.config_loader import load_yaml_with_base
from scripts.prepare_multidof_alignment_iteration import _safe_name
from scripts.prepare_multisource_multidof_iteration import prepare_iteration


IDENTITY_SIX = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _read_template(path: Path) -> dict[str, Any]:
    payload = load_yaml_with_base(path)
    if not isinstance(payload, Mapping):
        raise ValueError("layer identifiability template must be a YAML mapping")
    return dict(payload)


def _parameter_specs(scan: Mapping[str, object]) -> list[dict[str, object]]:
    raw = scan.get("alignment_parameter_specs")
    if not isinstance(raw, list) or not raw:
        raise ValueError("layer identifiability template lacks alignment_parameter_specs")
    specs = [dict(item) for item in raw if isinstance(item, Mapping)]
    if len(specs) != len(raw):
        raise ValueError("alignment_parameter_specs entries must be mappings")
    names = [str(item.get("name", "")) for item in specs]
    if not all(names) or len(set(names)) != len(names):
        raise ValueError("alignment_parameter_specs requires unique non-empty names")
    return specs


def _zero_payload(scan: Mapping[str, object]) -> tuple[dict[str, list[float]], dict[str, dict[str, list[float]]]]:
    stations = {str(station): list(IDENTITY_SIX) for station in scan.get("station_ids", (0, 1, 2, 3))}
    layers = {str(IFT_STATION_ID): {str(layer): list(IDENTITY_SIX) for layer in IFT_LAYER_IDS}}
    return stations, layers


def _zero_values(names: Sequence[str]) -> dict[str, float]:
    return {name: 0.0 for name in names}


def _complete_named_values(
    supplied: Mapping[str, object],
    names: Sequence[str],
    *,
    label: str,
) -> dict[str, float]:
    unknown = set(str(key) for key in supplied) - set(names)
    if unknown:
        raise ValueError(f"{label} has unknown parameter(s): " + ", ".join(sorted(unknown)))
    values: dict[str, float] = {}
    for name in names:
        if name not in supplied:
            values[name] = 0.0
            continue
        value = float(supplied[name])
        if not math.isfinite(value):
            raise ValueError(f"{label} parameter '{name}' must be finite")
        values[name] = value
    return values


def _condition_severity(values: Mapping[str, float], specs: Sequence[Mapping[str, object]]) -> float:
    return float(
        math.sqrt(
            sum((float(values[str(spec["name"])]) / float(spec["severity_scale"])) ** 2 for spec in specs)
        )
    )


def _point(
    *,
    name: str,
    values: Mapping[str, float],
    specs: Sequence[Mapping[str, object]],
    base_station: Mapping[str, Sequence[float]],
    base_layer: Mapping[str, Mapping[str, Sequence[float]]],
    role: str,
    direction_trial: str,
    finite_difference_for: str | None = None,
    probe_sign: str | None = None,
    finite_difference_anchor: str | None = None,
) -> dict[str, object]:
    stations, layers = payload_transforms_with_parameter_values(specs, base_station, base_layer, values)
    severity = _condition_severity(values, specs)
    result: dict[str, object] = {
        "name": _safe_name(name, label="physical point name"),
        "point_role": role,
        "direction_trial": _safe_name(direction_trial, label="direction trial"),
        "alignment_parameter_values": {key: float(value) for key, value in values.items()},
        "station_transforms": stations,
        "layer_transforms": layers,
        "condition_value": severity,
        "condition_magnitude": severity,
    }
    if finite_difference_for is not None:
        result.update(
            {
                "finite_difference_for": finite_difference_for,
                "probe_sign": probe_sign,
                "finite_difference_anchor": finite_difference_anchor,
            }
        )
    return result


def _layer_component_values(
    values: Mapping[str, float],
    specs: Sequence[Mapping[str, object]],
    component: str,
) -> dict[int, float]:
    selected: dict[int, float] = {}
    for spec in specs:
        if spec_scope(spec) != "layer" or str(spec.get("component")) != component:
            continue
        selected[int(spec["layer_id"])] = float(values[str(spec["name"])])
    return selected


def _validate_contrast_heldout(
    values: Mapping[str, float],
    specs: Sequence[Mapping[str, object]],
    *,
    raw_name: str,
    allow_identically_zero: bool = False,
) -> None:
    """Refuse envelope violations, station motion, ry/dy/rz, and empty injections."""
    if not specs or any(spec_scope(spec) != "contrast" for spec in specs):
        raise ValueError(f"contrast held-out '{raw_name}' requires only C_dx/C_rx specs")
    names = {str(spec["name"]) for spec in specs}
    if names == {"C_dx"}:
        c_dx = float(values["C_dx"])
        if abs(c_dx) > float(CONTRAST_ENVELOPE["C_dx"]) + 1.0e-12:
            raise ValueError(
                f"1-D C_dx held-out '{raw_name}' must stay inside "
                f"|C_dx|<={CONTRAST_ENVELOPE['C_dx']} mm"
            )
        if math.isclose(c_dx, 0.0, rel_tol=0.0, abs_tol=1.0e-15) and not allow_identically_zero:
            raise ValueError(f"1-D C_dx held-out '{raw_name}' is identically zero")
        return
    if names != set(CONTRAST_PARAMETERS):
        raise ValueError(f"contrast held-out '{raw_name}' requires C_dx or C_dx+C_rx")
    if not inside_contrast_envelope(values):
        raise ValueError(
            f"contrast held-out '{raw_name}' must stay inside "
            f"|C_dx|<={CONTRAST_ENVELOPE['C_dx']} mm and |C_rx|<={CONTRAST_ENVELOPE['C_rx']} mrad"
        )
    active = [
        name
        for name in CONTRAST_PARAMETERS
        if not math.isclose(float(values[name]), 0.0, rel_tol=0.0, abs_tol=1.0e-15)
    ]
    if not active and not allow_identically_zero:
        raise ValueError(f"contrast held-out '{raw_name}' is identically zero")


def _validate_linear_internal_heldout(
    values: Mapping[str, float],
    specs: Sequence[Mapping[str, object]],
    *,
    raw_name: str,
) -> None:
    """Refuse dy/rz, station motion, mixed components, and source-unstable layer1 rx."""
    for spec in specs:
        name = str(spec["name"])
        value = float(values[name])
        if math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1.0e-15):
            continue
        if spec_scope(spec) == "station":
            raise ValueError(
                f"linear internal held-out '{raw_name}' must keep station correction at zero; "
                f"'{name}' is {value}"
            )
        component = str(spec.get("component"))
        if component in {"dy_mm", "rz_mrad"}:
            raise ValueError(
                f"linear internal held-out '{raw_name}' must not inject unusable layer {component}"
            )
        if component == "rx_mrad" and int(spec["layer_id"]) == 1:
            raise ValueError(
                f"linear internal held-out '{raw_name}' must keep source-unstable layer1 rx at zero"
            )
    active_components: list[str] = []
    active_layers: dict[str, dict[int, float]] = {}
    for component in ("dx_mm", "rx_mrad", "ry_mrad"):
        layers = _layer_component_values(values, specs, component)
        if any(not math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1.0e-15) for value in layers.values()):
            active_components.append(component)
            active_layers[component] = layers
    if len(active_components) != 1:
        raise ValueError(
            f"linear internal held-out '{raw_name}' must inject exactly one of layer dx, rx, or ry, "
            f"found {active_components or 'none'}"
        )
    component = active_components[0]
    active = active_layers[component]
    if not math.isclose(float(active.get(1, 0.0)), 0.0, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError(f"linear internal held-out '{raw_name}' must keep layer1 at zero")
    outer_plus = float(active.get(0, 0.0))
    outer_minus = float(active.get(2, 0.0))
    if not math.isclose(outer_plus, -outer_minus, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"linear internal held-out '{raw_name}' must be outer-antisymmetric")
    if math.isclose(outer_plus, 0.0, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError(f"linear internal held-out '{raw_name}' outer planes are identically zero")
    if component in {"rx_mrad", "ry_mrad"}:
        magnitude = abs(outer_plus)
        if magnitude < 0.6 - 1.0e-12 or magnitude > 0.8 + 1.0e-12:
            raise ValueError(
                f"linear internal held-out '{raw_name}' outer {component} must stay near identifiable "
                "precision, in 0.6–0.8 mrad"
            )


def compile_layer_identifiability_pilot(
    template: Mapping[str, object],
    *,
    iteration: int,
    current_values: Mapping[str, float],
    include_finite_differences: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    """Expand the template into reference + joint FD probes + held-out closures.

    ``include_finite_differences=False`` emits only linear internal held-out
    payloads so a later closure can reuse an already-screened Jacobian bank.
    """
    raw_scan = template.get("physical_refit_capture_scan", template)
    if not isinstance(raw_scan, Mapping):
        raise ValueError("physical_refit_capture_scan must be a mapping")
    scan = dict(raw_scan)
    if scan.get("scan_mode") != "ift_layer_hierarchy":
        raise ValueError("layer identifiability compiler requires scan_mode: ift_layer_hierarchy")
    if int(scan.get("q_over_p_mode", -1)) != 0:
        raise ValueError("layer identifiability compiler requires q_over_p_mode: 0")
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    specs = _parameter_specs(scan)
    names = [str(spec["name"]) for spec in specs]
    contrast_only = bool(specs) and all(spec_scope(spec) == "contrast" for spec in specs)
    contrast_names = set(names)
    cdx_only = contrast_only and contrast_names == {"C_dx"}
    if contrast_only and contrast_names not in (set(CONTRAST_PARAMETERS), {"C_dx"}):
        raise ValueError("contrast curriculum requires C_dx or exactly C_dx+C_rx")
    zeros = _zero_values(names)
    if any(not math.isclose(float(current_values[name]), 0.0, rel_tol=0.0, abs_tol=1.0e-15) for name in names):
        raise ValueError(
            "layer identifiability pilot linearizes at the frozen converged station (all-zero payload); "
            "non-zero --current is not accepted"
        )
    if set(current_values) != set(names):
        raise ValueError("current_values must specify every hierarchy parameter, all identically zero")
    if contrast_only:
        for spec in specs:
            parameter = str(spec["name"])
            step = float(spec["finite_difference_step"])
            if step > float(CONTRAST_ENVELOPE[parameter]) + 1.0e-12:
                raise ValueError(
                    f"contrast finite-difference step for '{parameter}' exceeds the verified linear envelope"
                )
    base_station, base_layer = _zero_payload(scan)
    prefix = f"iteration_{iteration:02d}"
    reference_name = f"{prefix}_reference"
    points: list[dict[str, object]] = []
    if include_finite_differences:
        points.append(
            _point(
                name=reference_name,
                values=zeros,
                specs=specs,
                base_station=base_station,
                base_layer=base_layer,
                role="nominal",
                direction_trial=f"{prefix}_reference",
            )
        )
        for spec in specs:
            parameter = str(spec["name"])
            step = float(spec["finite_difference_step"])
            for sign, suffix, role in (
                (1.0, "p", "finite_difference_positive"),
                (-1.0, "m", "finite_difference_negative"),
            ):
                probe = dict(zeros)
                probe[parameter] = sign * step
                points.append(
                    _point(
                        name=f"{prefix}_fd_{parameter}_{suffix}",
                        values=probe,
                        specs=specs,
                        base_station=base_station,
                        base_layer=base_layer,
                        role=role,
                        direction_trial=f"{prefix}_fd_{parameter}_{suffix}",
                        finite_difference_for=parameter,
                        probe_sign="positive" if sign > 0.0 else "negative",
                        finite_difference_anchor=reference_name,
                    )
                )
    held_out = scan.pop("held_out_closure_points", None)
    held_out_names: list[str] = []
    if held_out is not None:
        if not isinstance(held_out, list) or any(not isinstance(item, Mapping) for item in held_out):
            raise ValueError("held_out_closure_points must be a list of mappings")
        seen = {str(point["name"]) for point in points}
        for item in held_out:
            raw_name = str(item.get("name", ""))
            name = _safe_name(f"{prefix}_{raw_name}", label="held-out closure point name")
            if name in seen:
                raise ValueError(f"held-out closure point '{name}' collides with a scan point")
            seen.add(name)
            held_values = _complete_named_values(
                dict(item.get("alignment_parameter_values", {})),
                names,
                label=f"held-out closure point '{raw_name}'",
            )
            if contrast_only:
                _validate_contrast_heldout(
                    held_values,
                    specs,
                    raw_name=raw_name,
                    allow_identically_zero=bool(item.get("allow_identically_zero", False)),
                )
            elif not include_finite_differences:
                _validate_linear_internal_heldout(held_values, specs, raw_name=raw_name)
            points.append(
                _point(
                    name=name,
                    values=held_values,
                    specs=specs,
                    base_station=base_station,
                    base_layer=base_layer,
                    role="held_out_closure",
                    direction_trial=name,
                )
            )
            held_out_names.append(name)
    if not include_finite_differences and not held_out_names:
        raise ValueError("held-out-only compilation requires at least one linear internal held-out point")
    scan["rigid_points"] = points
    scan["run_alignment_closure"] = False
    scan["held_out_only"] = not include_finite_differences
    reused_jacobian = scan.get("reused_jacobian_manifest")
    contract: dict[str, object] = {
        "method": (
            "physical_ift_internal_cdx_transfer"
            if cdx_only
            else (
                "physical_ift_layer_contrast_2d_curriculum"
                if contrast_only
                else (
                    "physical_ift_layer_linear_internal_heldout"
                    if not include_finite_differences
                    else "physical_ift_layer_hierarchy_identifiability_pilot"
                )
            )
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "iteration": iteration,
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "q_over_p_mode": 0,
        "reference_point": None if not include_finite_differences else reference_name,
        "anchor_point": None if not include_finite_differences else reference_name,
        "linearization": "frozen_converged_station_nominal",
        "held_out_only": not include_finite_differences,
        "include_finite_differences": include_finite_differences,
        "reused_jacobian_manifest": None if reused_jacobian is None else str(reused_jacobian),
        "station_common_mode_parameters": [
            str(spec["name"]) for spec in specs if spec.get("scope") == "station"
        ],
        "layer_internal_parameters": [str(spec["name"]) for spec in specs if spec.get("scope") == "layer"],
        "contrast_parameters": [str(spec["name"]) for spec in specs if spec.get("scope") == "contrast"],
        "canonical_internal_basis": "outer_contrast",
        "payload_expansion": (
            "L0=(+C_dx,0), L1=0, L2=(-C_dx,0); C_rx identically 0; station six-vector identically 0"
            if cdx_only
            else "L0=(+C_dx,+C_rx), L1=0, L2=(-C_dx,-C_rx); station six-vector identically 0"
        ),
        "anchor_parameter_values": zeros,
        "finite_difference_steps": {str(spec["name"]): float(spec["finite_difference_step"]) for spec in specs},
        "held_out_closure_points": held_out_names,
        "gauge_choices": ["outer_contrast"] if contrast_only else ["sum_to_zero", "reference_layer", "outer_contrast"],
        "forbidden_components": [
            "dz_mm",
            "layer_dy_mm",
            "layer_rz_mrad",
            "layer1_rx_mrad",
            "relative_ry",
            "module",
            "reference_layer_as_gauge",
            *(["C_rx"] if cdx_only else []),
        ],
        "update_semantics": (
            "IFT-Internal Mode transfer: float only 1-D C_dx with station six-vector frozen at 0. "
            "Do not inject C_rx, relative ry, layer dy/rz, layer1, or module parameters, "
            "and do not retune registered C_dx capture."
            if cdx_only
            else (
            "C_dx+C_rx 2D mini-curriculum: joint contrast payloads with station six-vector frozen at 0. "
            "Do not treat frozen-station reference_layer as a gauge-equivalent fit, do not inject "
            "relative ry, layer dy/rz, layer1, or module parameters, and do not retune C_dx."
            if contrast_only
            else (
                "Held-out-only linear IFT internals: reuse the already-screened Jacobian; "
                "station six-vector stays 0 except in explicit mixed negative-control points."
                if not include_finite_differences
                else "Identifiability pilot: joint station/layer finite differences linearized at the frozen station."
            )
            )
        ),
    }
    return {"alignment_iteration": contract, "physical_refit_capture_scan": scan}, contract


def default_zero_current(template: Mapping[str, object]) -> dict[str, float]:
    raw_scan = template.get("physical_refit_capture_scan", template)
    if not isinstance(raw_scan, Mapping):
        raise ValueError("physical_refit_capture_scan must be a mapping")
    names = [str(spec["name"]) for spec in _parameter_specs(raw_scan)]
    return _zero_values(names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--iteration-template", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--nevents", type=int, default=100)
    parser.add_argument("--source-id", action="append", default=None)
    args = parser.parse_args()
    if args.iteration < 0:
        parser.error("--iteration must be non-negative")
    template_path = Path(args.iteration_template).expanduser().resolve()
    template = _read_template(template_path)
    current = default_zero_current(template)
    raw_scan = template.get("physical_refit_capture_scan", template)
    include_fd = True
    if isinstance(raw_scan, Mapping):
        include_fd = not bool(raw_scan.get("held_out_only", False))
    # Validate compilation before writing the immutable source bank.
    compile_layer_identifiability_pilot(
        copy.deepcopy(template),
        iteration=args.iteration,
        current_values=current,
        include_finite_differences=include_fd,
    )
    manifest = prepare_iteration(
        source_config_path=Path(args.source_config).expanduser().resolve(),
        iteration_template_path=template_path,
        output_root=Path(args.output_root).expanduser().resolve(),
        iteration=int(args.iteration),
        current_values=current,
        nevents=int(args.nevents),
        source_ids=args.source_id,
    )
    print(
        json.dumps(
            {
                "iteration_manifest": str(
                    Path(args.output_root).expanduser().resolve() / "iteration_manifest.json"
                ),
                "sources": len(manifest["sources"]),
                "points": len(manifest["common_scan_plan"]["points"]),
                "scan_mode": manifest["common_scan_plan"]["scan_mode"],
                "test_data_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
