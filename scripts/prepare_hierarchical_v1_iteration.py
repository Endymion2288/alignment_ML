#!/usr/bin/env python3
"""Compile and prepare the station 5-DoF + IFT C_dx hierarchical V1 bank.

Joint physical points mix the already-closed station 5-DoF with the already-
closed 1-D ``C_dx``.  Finite-difference probes stay axial and are linearized
at the all-zero reference.  Station ``dz`` is a survey coordinate (written 0).
``C_rx``, relative ry, layer dy/rz, layer 1, and module stay out.  This is
not a six-parameter Newton step: later block updates float one level only.
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

from alignment.five_dof_sampling import LINEAR_SEVERITY_MAX, five_dof_severity
from alignment.hierarchical_v1 import (
    C_DX,
    C_DX_ENVELOPE_MM,
    HIERARCHICAL_V1_PARAMETERS,
    STATION_FREE_PARAMETERS,
    STATION_SOLVE_PARAMETERS,
    STATION_SURVEY_PARAMETER,
    complete_hierarchical_values,
    inside_cdx_envelope,
)
from alignment.hierarchical_v1_sampling import sample_hierarchical_v1_points
from alignment.layer_hierarchy import spec_scope
from scripts.config_loader import load_yaml_with_base
from scripts.prepare_layer_identifiability_pilot import (
    _complete_named_values,
    _parameter_specs,
    _point,
    _zero_payload,
    _zero_values,
)
from scripts.prepare_multisource_multidof_iteration import prepare_iteration


FIT_BASIS = "hierarchical_v1"


def _read_template(path: Path) -> dict[str, Any]:
    payload = load_yaml_with_base(path)
    if not isinstance(payload, Mapping):
        raise ValueError("hierarchical V1 template must be a YAML mapping")
    return dict(payload)


def _validate_hierarchical_v1_specs(specs: Sequence[Mapping[str, object]]) -> None:
    names = [str(spec["name"]) for spec in specs]
    if set(names) != set(HIERARCHICAL_V1_PARAMETERS) or len(names) != len(HIERARCHICAL_V1_PARAMETERS):
        raise ValueError(
            "hierarchical V1 requires alignment_parameter_specs "
            + ", ".join(HIERARCHICAL_V1_PARAMETERS)
        )
    scopes = {str(spec.get("name")): spec_scope(spec) for spec in specs}
    for name in STATION_SOLVE_PARAMETERS:
        if scopes.get(name) != "station":
            raise ValueError(f"hierarchical V1 parameter '{name}' must have station scope")
    if scopes.get(C_DX) != "contrast":
        raise ValueError("hierarchical V1 C_dx must have contrast scope")
    if any(spec_scope(spec) == "layer" for spec in specs):
        raise ValueError("hierarchical V1 does not admit layer-label parameters")
    if any(str(spec.get("name")) == "C_rx" for spec in specs):
        raise ValueError("hierarchical V1 does not admit C_rx")


def _validate_joint_injection(
    values: Mapping[str, float],
    *,
    raw_name: str,
    require_both_levels: bool,
    allow_identically_zero: bool,
) -> None:
    completed = complete_hierarchical_values(values)
    if not math.isclose(completed[STATION_SURVEY_PARAMETER], 0.0, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError(f"hierarchical V1 point '{raw_name}' must keep station dz identically 0")
    if not inside_cdx_envelope(completed[C_DX]) and not allow_identically_zero:
        raise ValueError(
            f"hierarchical V1 point '{raw_name}' must stay inside |C_dx|<={C_DX_ENVELOPE_MM} mm"
        )
    station_severity = five_dof_severity(completed)
    if require_both_levels:
        if station_severity <= 0.0 or station_severity > LINEAR_SEVERITY_MAX + 1.0e-12:
            raise ValueError(
                f"hierarchical V1 point '{raw_name}' station 5-DoF severity {station_severity} "
                f"must lie in (0, {LINEAR_SEVERITY_MAX}]"
            )
        if abs(completed[C_DX]) <= 1.0e-15:
            raise ValueError(f"hierarchical V1 point '{raw_name}' must inject non-zero C_dx")
        return
    if allow_identically_zero:
        return
    if station_severity <= 1.0e-15 and abs(completed[C_DX]) <= 1.0e-15:
        raise ValueError(f"hierarchical V1 remaining point '{raw_name}' is identically zero")


def compile_hierarchical_v1(
    template: Mapping[str, object],
    *,
    iteration: int,
    current_values: Mapping[str, float],
    include_finite_differences: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    """Expand station 5-DoF + C_dx into reference, axial FD, and joint points."""
    raw_scan = template.get("physical_refit_capture_scan", template)
    if not isinstance(raw_scan, Mapping):
        raise ValueError("physical_refit_capture_scan must be a mapping")
    scan = dict(raw_scan)
    if scan.get("scan_mode") != "ift_layer_hierarchy":
        raise ValueError("hierarchical V1 compiler requires scan_mode: ift_layer_hierarchy")
    if str(scan.get("fit_basis", "")) != FIT_BASIS:
        raise ValueError("hierarchical V1 compiler requires fit_basis: hierarchical_v1")
    if int(scan.get("q_over_p_mode", -1)) != 0:
        raise ValueError("hierarchical V1 compiler requires q_over_p_mode: 0")
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    specs = _parameter_specs(scan)
    _validate_hierarchical_v1_specs(specs)
    names = [str(spec["name"]) for spec in specs]
    zeros = _zero_values(names)
    if set(current_values) != set(names):
        raise ValueError("current_values must specify every hierarchical V1 parameter")
    if any(not math.isclose(float(current_values[name]), 0.0, rel_tol=0.0, abs_tol=1.0e-15) for name in names):
        raise ValueError("hierarchical V1 linearizes at the all-zero reference; non-zero --current is not accepted")
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
            from scripts.prepare_multidof_alignment_iteration import _safe_name

            name = _safe_name(f"{prefix}_{raw_name}", label="held-out closure point name")
            if name in seen:
                raise ValueError(f"held-out closure point '{name}' collides with a scan point")
            seen.add(name)
            held_values = complete_hierarchical_values(
                _complete_named_values(
                    dict(item.get("alignment_parameter_values", {})),
                    names,
                    label=f"held-out closure point '{raw_name}'",
                )
            )
            allow_zero = bool(item.get("allow_identically_zero", False))
            _validate_joint_injection(
                held_values,
                raw_name=raw_name,
                require_both_levels=include_finite_differences,
                allow_identically_zero=allow_zero,
            )
            point = _point(
                name=name,
                values=held_values,
                specs=specs,
                base_station=base_station,
                base_layer=base_layer,
                role="held_out_closure",
                direction_trial=name,
            )
            if allow_zero:
                point["allow_identically_zero"] = True
            points.append(point)
            held_out_names.append(name)
    if not include_finite_differences and not held_out_names:
        raise ValueError("held-out-only hierarchical V1 compilation requires at least one remaining point")
    scan["rigid_points"] = points
    scan["run_alignment_closure"] = False
    scan["held_out_only"] = not include_finite_differences
    scan["fit_basis"] = FIT_BASIS
    contract: dict[str, object] = {
        "method": "physical_ift_hierarchical_v1_station_cdx",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "iteration": iteration,
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "q_over_p_mode": 0,
        "fit_basis": FIT_BASIS,
        "reference_point": None if not include_finite_differences else reference_name,
        "anchor_point": None if not include_finite_differences else reference_name,
        "linearization": "zero_reference_axial_finite_differences",
        "held_out_only": not include_finite_differences,
        "include_finite_differences": include_finite_differences,
        "joint_station_cdx_newton": False,
        "block_hierarchy": True,
        "station_parameters": list(STATION_SOLVE_PARAMETERS),
        "station_free_parameters": list(STATION_FREE_PARAMETERS),
        "survey_parameter": STATION_SURVEY_PARAMETER,
        "layer_parameters": [C_DX],
        "canonical_internal_basis": "outer_contrast",
        "payload_expansion": "station 5-DoF + dz=0; L0=+C_dx, L1=0, L2=-C_dx; C_rx identically 0",
        "anchor_parameter_values": zeros,
        "finite_difference_steps": {str(spec["name"]): float(spec["finite_difference_step"]) for spec in specs},
        "held_out_closure_points": held_out_names,
        "forbidden_components": [
            "C_rx",
            "relative_ry",
            "layer_dy_mm",
            "layer_rz_mrad",
            "layer1",
            "module",
            "joint_six_parameter_newton",
        ],
        "update_semantics": (
            "Block hierarchy: float station 5-DoF (dz survey prior) with C_dx fixed, "
            "write remaining station while keeping true C_dx, real-refit, then float "
            "only C_dx with station frozen.  Reverse order is a control.  Do not mix "
            "the two levels in one Newton step."
        ),
    }
    return {"alignment_iteration": contract, "physical_refit_capture_scan": scan}, contract


def _sampling_policy(template: Mapping[str, Any]) -> dict[str, Any]:
    raw = template.get("hierarchical_v1_sampling")
    if not isinstance(raw, Mapping):
        raise ValueError("iteration template lacks hierarchical_v1_sampling")
    return dict(raw)


def _held_out_entries(sampled: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": "start",
            "alignment_parameter_values": dict(sampled["start"]),
            "station_five_dof_severity": float(sampled["start_station_five_dof_severity"]),
            "regime": "start",
            "reserved_from_fd_and_operating_point": False,
        },
        {
            "name": "heldout_00",
            "alignment_parameter_values": dict(sampled["held_out"]),
            "station_five_dof_severity": float(sampled["held_out_station_five_dof_severity"]),
            "regime": "held_out",
            "reserved_from_fd_and_operating_point": True,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--iteration-template", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--nevents", type=int, default=100)
    args = parser.parse_args()
    if args.iteration < 0:
        parser.error("--iteration must be non-negative")
    template_path = Path(args.iteration_template).expanduser().resolve()
    template = _read_template(template_path)
    policy = _sampling_policy(template)
    sampled = sample_hierarchical_v1_points(
        seed=int(policy["seed"]),
        start_severity=float(policy["start_severity"]),
        held_out_severity=float(policy["held_out_severity"]),
        envelope=float(policy.get("envelope_C_dx_mm", C_DX_ENVELOPE_MM)),
        min_cdx_fraction=float(policy.get("min_cdx_fraction", 0.40)),
    )
    resolved = copy.deepcopy(template)
    scan = resolved["physical_refit_capture_scan"]
    scan["held_out_closure_points"] = [
        {"name": item["name"], "alignment_parameter_values": item["alignment_parameter_values"]}
        for item in _held_out_entries(sampled)
    ]
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing iteration root: {output_root}")
    sidecar = output_root.parent / f".{output_root.name}.resolved_template.yaml"
    sidecar.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    try:
        current = {name: 0.0 for name in HIERARCHICAL_V1_PARAMETERS}
        manifest = prepare_iteration(
            source_config_path=Path(args.source_config).expanduser().resolve(),
            iteration_template_path=sidecar,
            output_root=output_root,
            iteration=int(args.iteration),
            current_values=current,
            nevents=int(args.nevents),
        )
        resolved_path = output_root / "resolved_iteration_template.yaml"
        resolved_path.write_text(sidecar.read_text(encoding="utf-8"), encoding="utf-8")
        (output_root / "sampled_hierarchical_v1_points.json").write_text(
            json.dumps(
                {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "template": str(template_path),
                    "resolved_template": str(resolved_path),
                    "joint_station_cdx_newton": False,
                    "C_rx_forced_zero": True,
                    "station_dz_written_zero": True,
                    "validation_used_in_registration": False,
                    "test_data_accessed": False,
                    "points": _held_out_entries(sampled),
                    **sampled,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest["iteration_template"] = str(resolved_path)
        (output_root / "iteration_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    finally:
        sidecar.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "iteration_manifest": str(output_root / "iteration_manifest.json"),
                "sampled_points": str(output_root / "sampled_hierarchical_v1_points.json"),
                "start_station_five_dof_severity": sampled["start_station_five_dof_severity"],
                "start_C_dx_mm": sampled["start_C_dx_mm"],
                "held_out_station_five_dof_severity": sampled["held_out_station_five_dof_severity"],
                "held_out_C_dx_mm": sampled["held_out_C_dx_mm"],
                "held_out_points": [item["name"] for item in _held_out_entries(sampled)],
                "joint_station_cdx_newton": False,
                "test_data_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
