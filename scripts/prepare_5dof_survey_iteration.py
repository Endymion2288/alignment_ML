#!/usr/bin/env python3
"""Prepare the station-level 5-DoF + survey-dz source-disjoint iteration bank.

Samples a joint 5-DoF random start in the validated linear regime, writes an
immutable iteration template with shared-geometry held-out points, then
delegates to ``prepare_multisource_multidof_iteration``.  ``dz`` is identically
zero in every sampled misalignment and is present in the scan only so a frozen
survey prior can enter the normal equation.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from alignment.five_dof_sampling import SURVEY_PARAMETER, five_dof_severity, sample_linear_regime_points
from scripts.config_loader import load_yaml_with_base
from scripts.prepare_multisource_multidof_iteration import prepare_iteration


def _read_template(path: Path) -> dict[str, Any]:
    payload = load_yaml_with_base(path)
    if not isinstance(payload, Mapping):
        raise ValueError("iteration template must be a YAML mapping")
    return dict(payload)


def _sampling_policy(template: Mapping[str, Any]) -> dict[str, Any]:
    raw = template.get("five_dof_sampling")
    if not isinstance(raw, Mapping):
        raise ValueError("iteration template lacks five_dof_sampling")
    return dict(raw)


def _held_out_entries(sampled: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for group, role in (("linear_held_out", "linear"), ("stress_held_out", "stress")):
        for item in sampled[group]:
            entries.append(
                {
                    "name": f"closure_{item['name']}",
                    "alignment_parameter_values": dict(item["alignment_parameter_values"]),
                    "sampling_severity": float(item["severity"]),
                    "regime": role,
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
    sampled = sample_linear_regime_points(
        seed=int(policy["seed"]),
        start_severity=float(policy["start_severity"]),
        linear_held_out=tuple(float(value) for value in policy["linear_held_out_severities"]),
        stress_held_out=tuple(float(value) for value in policy["stress_held_out_severities"]),
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
        current = {name: float(sampled["start"].get(name, 0.0)) for name in names}
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
        (output_root / "sampled_5dof_points.json").write_text(
            json.dumps(
                {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "template": str(template_path),
                    "resolved_template": str(resolved_path),
                    "start_severity_realized": five_dof_severity(sampled["start"]),
                    "survey_parameter_forced_zero": SURVEY_PARAMETER,
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
                "sampled_points": str(output_root / "sampled_5dof_points.json"),
                "start_severity": five_dof_severity(sampled["start"]),
                "held_out_points": [item["name"] for item in _held_out_entries(sampled)],
                "sources_by_split": {
                    split: sum(1 for item in manifest["sources"] if item["split"] == split)
                    for split in ("train", "validation")
                },
                "points": len(manifest["common_scan_plan"]["points"]),
                "test_data_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
