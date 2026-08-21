#!/usr/bin/env python3
"""Prepare the C_dx+C_rx source-disjoint IFT layer-hierarchy mini-curriculum.

Samples joint contrast points inside the verified linear envelope, writes an
immutable iteration template with shared-geometry held-out points, then
delegates to ``prepare_multisource_multidof_iteration``.  Station six-vectors
stay identically zero.  Validation is present in the bank but must not be
used to choose capture criteria.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from alignment.contrast_sampling import contrast_radius, sample_joint_contrast_points
from scripts.config_loader import load_yaml_with_base
from scripts.prepare_multisource_multidof_iteration import prepare_iteration


def _read_template(path: Path) -> dict[str, Any]:
    payload = load_yaml_with_base(path)
    if not isinstance(payload, Mapping):
        raise ValueError("iteration template must be a YAML mapping")
    return dict(payload)


def _sampling_policy(template: Mapping[str, Any]) -> dict[str, Any]:
    raw = template.get("contrast_2d_sampling")
    if not isinstance(raw, Mapping):
        raise ValueError("iteration template lacks contrast_2d_sampling")
    return dict(raw)


def _held_out_entries(sampled: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = [
        {
            "name": "start",
            "alignment_parameter_values": dict(sampled["start"]),
            "sampling_radius": float(sampled["start_radius"]),
            "regime": "start",
            "reserved_from_fd_and_operating_point": False,
        }
    ]
    for group in ("linear", "stress", "held_out"):
        for item in sampled[group]:
            entries.append(
                {
                    "name": str(item["name"]),
                    "alignment_parameter_values": dict(item["alignment_parameter_values"]),
                    "sampling_radius": float(item["radius"]),
                    "regime": str(item["role"]),
                    "reserved_from_fd_and_operating_point": bool(
                        item.get("reserved_from_fd_and_operating_point", group == "held_out")
                    ),
                }
            )
    return entries


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
    envelope = policy.get("envelope")
    sampled = sample_joint_contrast_points(
        seed=int(policy["seed"]),
        start_radius=float(policy["start_radius"]),
        linear_radii=tuple(float(value) for value in policy["linear_radii"]),
        stress_radii=tuple(float(value) for value in policy["stress_radii"]),
        held_out_radii=tuple(float(value) for value in policy["held_out_radii"]),
        envelope=None if not isinstance(envelope, Mapping) else dict(envelope),
        min_axis_fraction=float(policy.get("min_axis_fraction", 0.25)),
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
        specs = scan["alignment_parameter_specs"]
        names = [str(spec["name"]) for spec in specs]
        current = {name: 0.0 for name in names}
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
        (output_root / "sampled_contrast_2d_points.json").write_text(
            json.dumps(
                {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "template": str(template_path),
                    "resolved_template": str(resolved_path),
                    "start_radius_realized": contrast_radius(sampled["start"], sampled["envelope"]),
                    "station_six_vector_forced_zero": True,
                    "validation_used_in_registration": False,
                    "test_data_accessed": False,
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
                "sampled_points": str(output_root / "sampled_contrast_2d_points.json"),
                "start": sampled["start"],
                "start_radius": contrast_radius(sampled["start"], sampled["envelope"]),
                "held_out_points": [item["name"] for item in _held_out_entries(sampled)],
                "reserved_held_out": [item["name"] for item in sampled["held_out"]],
                "sources_by_split": {
                    split: sum(1 for item in manifest["sources"] if item["split"] == split)
                    for split in ("train", "validation")
                },
                "points": len(manifest["common_scan_plan"]["points"]),
                "test_data_accessed": False,
                "validation_used_in_registration": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
