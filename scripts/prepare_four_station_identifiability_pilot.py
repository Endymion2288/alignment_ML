#!/usr/bin/env python3
"""Compile the four-station identifiability pilot around nominal geometry.

Every station S0--S3 receives independent central finite-difference probes in
the five track-constrained components, plus optional survey-dz diagnostics.
The linearization point is the all-zero payload: this is a measurement of
which absolute/relative modes the current track sample can constrain, not a
claim that any station is correct.  Test sources are never resolved.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from alignment.four_station import (
    FORMULATION,
    GAUGE_UNCONSTRAINED_FULL,
    IDENTITY_SIX,
    STATION_IDS,
    default_parameter_specs,
    formulation_contract,
    free_parameter_names,
    relative_alignment_table,
    survey_parameter_names,
    zero_parameter_values,
)
from alignment.physical_jacobian import station_transforms_with_parameter_values
from scripts.prepare_multidof_alignment_iteration import _safe_name
from scripts.prepare_multisource_multidof_iteration import prepare_iteration
from scripts.config_loader import load_yaml_with_base


def _read_template(path: Path) -> dict[str, Any]:
    payload = load_yaml_with_base(path)
    if not isinstance(payload, Mapping):
        raise ValueError("four-station identifiability template must be a YAML mapping")
    return dict(payload)


def _parameter_specs(scan: Mapping[str, object]) -> list[dict[str, object]]:
    raw = scan.get("alignment_parameter_specs")
    if not isinstance(raw, list) or not raw:
        raise ValueError("four-station template lacks alignment_parameter_specs")
    specs = [dict(item) for item in raw if isinstance(item, Mapping)]
    if len(specs) != len(raw):
        raise ValueError("alignment_parameter_specs entries must be mappings")
    names = [str(item.get("name", "")) for item in specs]
    if not all(names) or len(set(names)) != len(names):
        raise ValueError("alignment_parameter_specs requires unique non-empty names")
    stations = {int(item["station_id"]) for item in specs}
    if not stations.issubset(set(STATION_IDS)) or len(stations) != len(STATION_IDS):
        raise ValueError("four-station identifiability requires parameters on stations 0, 1, 2, and 3")
    return specs


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
    base_transforms: Mapping[int | str, Sequence[float]],
    role: str,
    direction_trial: str,
    finite_difference_for: str | None = None,
    probe_sign: str | None = None,
    finite_difference_anchor: str | None = None,
) -> dict[str, object]:
    transforms = station_transforms_with_parameter_values(specs, base_transforms, values)
    result: dict[str, object] = {
        "name": _safe_name(name, label="physical point name"),
        "point_role": role,
        "direction_trial": _safe_name(direction_trial, label="direction trial"),
        "alignment_parameter_values": {key: float(value) for key, value in values.items()},
        "station_transforms": transforms,
        "relative_alignment_delta_t_ij": relative_alignment_table(transforms),
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


def compile_four_station_identifiability_pilot(
    template: Mapping[str, object],
    *,
    iteration: int,
    current_values: Mapping[str, float],
    include_finite_differences: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    """Expand the four-station template into nominal + FD + held-out points."""
    raw_scan = template.get("physical_refit_capture_scan", template)
    if not isinstance(raw_scan, Mapping):
        raise ValueError("physical_refit_capture_scan must be a mapping")
    scan = dict(raw_scan)
    if scan.get("scan_mode") != "station_rigid_multidof":
        raise ValueError("four-station compiler requires scan_mode: station_rigid_multidof")
    if str(scan.get("alignment_formulation", "")) != FORMULATION:
        raise ValueError("four-station compiler requires alignment_formulation: four_station_v1")
    if str(scan.get("gauge", "")) != GAUGE_UNCONSTRAINED_FULL:
        raise ValueError(
            "the identifiability pilot must linearize in unconstrained_full; "
            "reference_station / common_mode_constraint are solve-time charts"
        )
    if int(scan.get("q_over_p_mode", -1)) != 0:
        raise ValueError("four-station compiler requires q_over_p_mode: 0")
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    specs = _parameter_specs(scan)
    names = [str(spec["name"]) for spec in specs]
    zeros = zero_parameter_values(specs)
    if set(current_values) != set(names):
        raise ValueError("current_values must specify every four-station parameter")
    if any(not math.isclose(float(current_values[name]), 0.0, rel_tol=0.0, abs_tol=1.0e-15) for name in names):
        raise ValueError("four-station identifiability linearizes at the all-zero nominal payload")
    base = {str(station): list(IDENTITY_SIX) for station in STATION_IDS}
    prefix = f"iteration_{iteration:02d}"
    reference_name = f"{prefix}_reference"
    points: list[dict[str, object]] = []
    if include_finite_differences:
        points.append(
            _point(
                name=reference_name,
                values=zeros,
                specs=specs,
                base_transforms=base,
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
                        base_transforms=base,
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
            if all(math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1.0e-15) for value in held_values.values()):
                raise ValueError(f"held-out closure point '{raw_name}' is identically zero")
            points.append(
                _point(
                    name=name,
                    values=held_values,
                    specs=specs,
                    base_transforms=base,
                    role="held_out_closure",
                    direction_trial=name,
                )
            )
            held_out_names.append(name)
    if not include_finite_differences and not held_out_names:
        raise ValueError("held-out-only compilation requires at least one held-out point")
    scan["rigid_points"] = points
    scan["run_alignment_closure"] = False
    scan["reference_station_ids"] = []
    scan["movable_station_ids"] = list(STATION_IDS)
    contract = formulation_contract(gauge=GAUGE_UNCONSTRAINED_FULL, include_survey_dz=True)
    contract.update(
        {
            "method": "physical_four_station_identifiability_pilot",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "iteration": iteration,
            "physical_geometry_repropagation": True,
            "coordinate_surrogate": False,
            "q_over_p_mode": 0,
            "reference_point": None if not include_finite_differences else reference_name,
            "anchor_point": None if not include_finite_differences else reference_name,
            "linearization": "nominal_all_zero_four_station_payload",
            "include_finite_differences": include_finite_differences,
            "held_out_closure_points": held_out_names,
            "finite_difference_steps": {str(spec["name"]): float(spec["finite_difference_step"]) for spec in specs},
            "free_parameter_names": list(free_parameter_names()),
            "survey_parameter_names": list(survey_parameter_names()),
            "update_semantics": (
                "Identifiability only: construct the four-station Jacobian from independent "
                "physical central finite differences.  Do not admit the 20-D space without SVD."
            ),
        }
    )
    return {"alignment_iteration": contract, "physical_refit_capture_scan": scan}, contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--iteration-template", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--nevents", type=int, default=50)
    parser.add_argument("--source-id", action="append", default=None)
    args = parser.parse_args()
    template = _read_template(Path(args.iteration_template).expanduser().resolve())
    raw_scan = template.get("physical_refit_capture_scan", template)
    if not isinstance(raw_scan, Mapping):
        raise SystemExit("iteration template has no physical_refit_capture_scan")
    specs = raw_scan.get("alignment_parameter_specs") or default_parameter_specs(include_survey_dz=True)
    current = zero_parameter_values([dict(item) for item in specs if isinstance(item, Mapping)])
    manifest = prepare_iteration(
        source_config_path=Path(args.source_config).expanduser().resolve(),
        iteration_template_path=Path(args.iteration_template).expanduser().resolve(),
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
                "sources": [item["source_id"] for item in manifest["sources"]],
                "points": len(manifest["common_scan_plan"]["points"]),
                "alignment_formulation": FORMULATION,
                "gauge": GAUGE_UNCONSTRAINED_FULL,
                "test_data_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
