#!/usr/bin/env python3
"""Prepare a held-out-only remaining bank after one hierarchical V1 block step.

The Jacobian stays in the already-produced iteration-00 joint bank.  This
command only writes remaining station + C_dx payloads, then delegates to the
existing source-disjoint iteration compiler.  It never adds finite-difference
probes, never opens test, and never floats both levels in one Newton step.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from alignment.hierarchical_v1 import (
    C_DX,
    C_DX_ENVELOPE_MM,
    HIERARCHICAL_V1_PARAMETERS,
    LEVELS,
    complete_hierarchical_values,
    inside_cdx_envelope,
    remaining_after_level_step,
)
from scripts.config_loader import load_yaml_with_base
from scripts.prepare_multisource_multidof_iteration import prepare_iteration


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return payload


def _read_template(path: Path) -> dict[str, Any]:
    payload = load_yaml_with_base(path)
    if not isinstance(payload, Mapping):
        raise ValueError("iteration template must be a YAML mapping")
    return dict(payload)


def _held_out_from_remaining(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    points = payload.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("remaining JSON must list at least one point")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in points:
        if not isinstance(item, Mapping):
            raise ValueError("remaining point is not a mapping")
        name = str(item.get("name", ""))
        if not name or name in seen:
            raise ValueError(f"invalid or duplicated remaining point name {name!r}")
        seen.add(name)
        values = item.get("alignment_parameter_values")
        if not isinstance(values, Mapping):
            floated = str(item.get("floated_level", ""))
            injected = item.get("injected")
            recovered = item.get("recovered")
            if not isinstance(injected, Mapping) or not isinstance(recovered, Mapping) or not floated:
                raise ValueError(f"remaining point '{name}' lacks alignment_parameter_values")
            values = remaining_after_level_step(
                injected=injected,
                recovered=recovered,
                floated_level=floated,
            )
        values = complete_hierarchical_values(values)
        if not inside_cdx_envelope(values[C_DX]):
            raise ValueError(
                f"remaining point '{name}' leaves |C_dx|<={C_DX_ENVELOPE_MM} mm"
            )
        entries.append(
            {
                "name": name,
                "alignment_parameter_values": values,
                "allow_identically_zero": bool(item.get("allow_identically_zero", False)),
                "floated_level": item.get("floated_level"),
                "order": item.get("order"),
                "parent_target": item.get("parent_target"),
            }
        )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--iteration-template", required=True)
    parser.add_argument("--remaining-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--nevents", type=int, default=100)
    args = parser.parse_args()
    if args.iteration < 0:
        parser.error("--iteration must be non-negative")
    remaining = _read_json(Path(args.remaining_json).expanduser().resolve())
    if remaining.get("test_data_accessed") is not False:
        raise ValueError("remaining JSON must declare test_data_accessed=false")
    if remaining.get("joint_station_cdx_newton") is True:
        raise ValueError("remaining JSON must not come from a joint station+C_dx Newton step")
    held_out = _held_out_from_remaining(remaining)
    template_path = Path(args.iteration_template).expanduser().resolve()
    template = _read_template(template_path)
    resolved = copy.deepcopy(template)
    scan = resolved["physical_refit_capture_scan"]
    scan["held_out_only"] = True
    scan["fit_basis"] = "hierarchical_v1"
    resolved.pop("hierarchical_v1_sampling", None)
    scan["held_out_closure_points"] = [
        {
            "name": item["name"],
            "alignment_parameter_values": item["alignment_parameter_values"],
            "allow_identically_zero": item["allow_identically_zero"],
        }
        for item in held_out
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
        (output_root / "remaining_hierarchical_v1_points.json").write_text(
            json.dumps(
                {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "template": str(template_path),
                    "remaining_json": str(Path(args.remaining_json).expanduser().resolve()),
                    "held_out_only": True,
                    "finite_difference_probes": False,
                    "joint_station_cdx_newton": False,
                    "test_data_accessed": False,
                    "points": held_out,
                    "levels": list(LEVELS),
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
                "points": [item["name"] for item in held_out],
                "held_out_only": True,
                "joint_station_cdx_newton": False,
                "test_data_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
