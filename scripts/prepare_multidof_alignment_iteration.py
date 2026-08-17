#!/usr/bin/env python3
"""Compile one real-geometry multi-DoF alignment iteration scan.

The emitted YAML is consumed unchanged by ``run_physical_refit_capture_scan``.
It contains an independently refitted zero/reference point, one current
alignment anchor, and a central finite-difference pair around that anchor for
every enabled station-level parameter.  No exported coordinate, residual, or
candidate graph is shifted by this utility.

This compiler deliberately represents the MC closure convention only: the
reference point is a known physical nominal payload.  A real-data correction
must provide a separately validated residual target and `/Tracker/Align`
update convention; this command does not guess either one.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from alignment.physical_jacobian import station_transforms_with_parameter_values


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("iteration template must be a YAML mapping")
    return dict(payload)


def _safe_name(value: str, *, label: str) -> str:
    if not value or not value.replace("_", "").isalnum():
        raise ValueError(f"{label} must be a non-empty alphanumeric identifier")
    return value


def _parameter_specs(scan: Mapping[str, object]) -> list[dict[str, object]]:
    raw = scan.get("alignment_parameter_specs")
    if not isinstance(raw, list) or not raw:
        raise ValueError("station_rigid_multidof template lacks alignment_parameter_specs")
    specs = [dict(item) for item in raw if isinstance(item, Mapping)]
    if len(specs) != len(raw):
        raise ValueError("alignment_parameter_specs entries must be mappings")
    names = [str(item.get("name", "")) for item in specs]
    if not all(names) or len(set(names)) != len(names):
        raise ValueError("alignment_parameter_specs requires unique non-empty names")
    for item in specs:
        for field in ("station_id", "component", "finite_difference_step", "severity_scale"):
            if field not in item:
                raise ValueError(f"alignment parameter '{item.get('name')}' lacks {field}")
        step = float(item["finite_difference_step"])
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError(f"alignment parameter '{item['name']}' has invalid finite_difference_step")
    return specs


def _parse_named_values(values: Sequence[str], names: Sequence[str], *, label: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for raw in values:
        name, separator, raw_value = str(raw).partition(":")
        if not separator or not name:
            raise ValueError(f"{label} entries must have form PARAMETER:VALUE")
        if name in parsed:
            raise ValueError(f"{label} repeats parameter '{name}'")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"{label} parameter '{name}' must be finite")
        parsed[name] = value
    if set(parsed) != set(names):
        raise ValueError(f"{label} must specify exactly: " + ", ".join(names))
    return {name: float(parsed[name]) for name in names}


def _values_from_update(path: Path, names: Sequence[str]) -> dict[str, float]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("alignment update JSON must be a mapping")
    supplied = payload.get("proposed_next_parameter_values")
    if not isinstance(supplied, Mapping):
        raise ValueError("alignment update JSON lacks proposed_next_parameter_values")
    return _parse_named_values(
        [f"{name}:{supplied[name]}" for name in names] if set(supplied) == set(names) else [],
        names,
        label="proposed_next_parameter_values",
    )


def _zero_reference_point(scan: Mapping[str, object]) -> dict[str, object]:
    raw_points = scan.get("rigid_points")
    if not isinstance(raw_points, list) or not raw_points:
        raise ValueError("iteration template lacks rigid_points")
    nominal = [dict(item) for item in raw_points if isinstance(item, Mapping) and item.get("point_role") == "nominal"]
    if len(nominal) != 1:
        raise ValueError("iteration template must contain exactly one nominal reference point")
    transforms = nominal[0].get("station_transforms")
    if not isinstance(transforms, Mapping):
        raise ValueError("iteration template nominal point lacks station_transforms")
    return nominal[0]


def _condition_severity(values: Mapping[str, float], specs: Sequence[Mapping[str, object]]) -> float:
    return float(
        math.sqrt(
            sum(
                (float(values[str(spec["name"])]) / float(spec["severity_scale"])) ** 2
                for spec in specs
            )
        )
    )


def _point(
    *,
    name: str,
    values: Mapping[str, float],
    specs: Sequence[Mapping[str, object]],
    base_transforms: Mapping[int | str, Sequence[float]],
    role: str,
    direction_trial: str,
    finite_difference_for: str | None = None,
    probe_sign: str | None = None,
    finite_difference_anchor: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "name": _safe_name(name, label="physical point name"),
        "point_role": role,
        "direction_trial": _safe_name(direction_trial, label="direction trial"),
        "alignment_parameter_values": {key: float(value) for key, value in values.items()},
        "station_transforms": station_transforms_with_parameter_values(specs, base_transforms, values),
        "condition_value": _condition_severity(values, specs),
        "condition_magnitude": _condition_severity(values, specs),
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


def compile_iteration(
    template: Mapping[str, object],
    *,
    iteration: int,
    current_values: Mapping[str, float],
) -> tuple[dict[str, object], dict[str, object]]:
    """Build the physical scan and its explicit iteration provenance contract."""
    raw_scan = template.get("physical_refit_capture_scan", template)
    if not isinstance(raw_scan, Mapping):
        raise ValueError("physical_refit_capture_scan must be a mapping")
    scan = dict(raw_scan)
    if scan.get("scan_mode") != "station_rigid_multidof":
        raise ValueError("iteration compiler requires scan_mode: station_rigid_multidof")
    if int(scan.get("q_over_p_mode", -1)) != 0:
        raise ValueError("iteration compiler requires field-aware Acts q_over_p_mode: 0")
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    specs = _parameter_specs(scan)
    names = [str(spec["name"]) for spec in specs]
    values = {name: float(current_values[name]) for name in names}
    nominal = _zero_reference_point(scan)
    base = nominal["station_transforms"]
    assert isinstance(base, Mapping)  # checked in _zero_reference_point
    prefix = f"iteration_{iteration:02d}"
    reference = dict(nominal)
    reference["name"] = f"{prefix}_reference"
    reference["direction_trial"] = f"{prefix}_reference"
    reference["point_role"] = "nominal"
    reference["finite_difference_for"] = None
    reference["probe_sign"] = None
    reference["finite_difference_anchor"] = None
    anchor_name = f"{prefix}_anchor"
    points: list[dict[str, object]] = [
        reference,
        _point(
            name=anchor_name,
            values=values,
            specs=specs,
            base_transforms=base,
            role="iteration_anchor",
            direction_trial=f"{prefix}_anchor",
        ),
    ]
    for spec in specs:
        parameter = str(spec["name"])
        step = float(spec["finite_difference_step"])
        for sign, suffix, role in (
            (1.0, "p", "finite_difference_positive"),
            (-1.0, "m", "finite_difference_negative"),
        ):
            probe = dict(values)
            probe[parameter] += sign * step
            points.append(
                _point(
                    name=f"{prefix}_fd_{parameter}_{suffix}",
                    values=probe,
                    specs=specs,
                    base_transforms=base,
                    role=role,
                    direction_trial=f"{prefix}_fd_{parameter}_{suffix}",
                    finite_difference_for=parameter,
                    probe_sign="positive" if sign > 0.0 else "negative",
                    finite_difference_anchor=anchor_name,
                )
            )
    scan["rigid_points"] = points
    scan["run_alignment_closure"] = False
    contract: dict[str, object] = {
        "method": "physical_multidof_alignment_iteration_scan_compiler",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "iteration": iteration,
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "q_over_p_mode": 0,
        "reference_point": str(reference["name"]),
        "anchor_point": anchor_name,
        "anchor_parameter_values": values,
        "finite_difference_steps": {str(spec["name"]): float(spec["finite_difference_step"]) for spec in specs},
        "update_semantics": (
            "MC closure only: subsequent route-selected WLS solves the local physical update from the anchor "
            "toward this independently refitted nominal reference.  It does not assert a real-data correction sign."
        ),
    }
    return {"alignment_iteration": contract, "physical_refit_capture_scan": scan}, contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-config", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--current", action="append", default=None, metavar="PARAMETER:VALUE")
    parser.add_argument("--update-json", default=None)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--output-contract", default=None)
    args = parser.parse_args()
    if (args.current is None) == (args.update_json is None):
        parser.error("provide exactly one of --current or --update-json")
    template_path = Path(args.template_config).expanduser().resolve()
    template = _read_yaml(template_path)
    raw_scan = template.get("physical_refit_capture_scan", template)
    if not isinstance(raw_scan, Mapping):
        raise ValueError("physical_refit_capture_scan must be a mapping")
    names = [str(spec["name"]) for spec in _parameter_specs(raw_scan)]
    current = (
        _parse_named_values(args.current, names, label="--current")
        if args.current is not None
        else _values_from_update(Path(str(args.update_json)).expanduser().resolve(), names)
    )
    compiled, contract = compile_iteration(template, iteration=int(args.iteration), current_values=current)
    output = Path(args.output_config).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite iteration config: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(compiled, sort_keys=False), encoding="utf-8")
    contract_path = (
        Path(args.output_contract).expanduser().resolve()
        if args.output_contract is not None
        else output.with_suffix(".iteration_contract.json")
    )
    if contract_path.exists():
        raise FileExistsError(f"refusing to overwrite iteration contract: {contract_path}")
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_config": str(output), "output_contract": str(contract_path)}, indent=2))


if __name__ == "__main__":
    main()
