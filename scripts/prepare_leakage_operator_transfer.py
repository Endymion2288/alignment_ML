#!/usr/bin/env python3
"""Prepare the analysis-only leakage-operator Jacobian transfer bank.

Axial finite-difference probes of station 5-DoF + survey dz and of C_dx are
linearized at the all-zero reference.  There is no joint station+C_dx
operating-point injection and no geometry-write candidate.  A is remeasured
here and scored against the frozen 10% contract; it is never updated.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from alignment.hierarchical_v1 import C_DX, HIERARCHICAL_V1_PARAMETERS
from scripts.config_loader import load_yaml_with_base
from scripts.prepare_hierarchical_v1_iteration import compile_hierarchical_v1
from scripts.prepare_multisource_multidof_iteration import prepare_iteration


def _read_template(path: Path) -> dict[str, Any]:
    payload = load_yaml_with_base(path)
    if not isinstance(payload, Mapping):
        raise ValueError("iteration template must be a YAML mapping")
    return dict(payload)


def _assert_fd_only(template: Mapping[str, Any]) -> None:
    scan = template.get("physical_refit_capture_scan")
    if not isinstance(scan, Mapping):
        raise ValueError("iteration template lacks physical_refit_capture_scan")
    if scan.get("held_out_closure_points"):
        raise ValueError("leakage transfer Jacobian must not carry held-out joint injections")
    if bool(scan.get("held_out_only", False)):
        raise ValueError("leakage transfer Jacobian requires axial finite differences")
    names = [str(spec["name"]) for spec in scan.get("alignment_parameter_specs", ())]
    if set(names) != set(HIERARCHICAL_V1_PARAMETERS):
        raise ValueError("leakage transfer Jacobian requires hierarchical V1 specs")


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
    _assert_fd_only(template)
    current = {name: 0.0 for name in HIERARCHICAL_V1_PARAMETERS}
    compiled, contract = compile_hierarchical_v1(
        copy.deepcopy(template),
        iteration=int(args.iteration),
        current_values=current,
        include_finite_differences=True,
    )
    points = compiled["physical_refit_capture_scan"]["rigid_points"]
    joint = []
    for point in points:
        values = dict(point.get("alignment_parameter_values") or {})
        station_excited = any(
            abs(float(values.get(name, 0.0))) > 1.0e-15
            for name in HIERARCHICAL_V1_PARAMETERS
            if name != C_DX
        )
        cdx_excited = abs(float(values.get(C_DX, 0.0))) > 1.0e-15
        if station_excited and cdx_excited:
            joint.append(str(point["name"]))
    if joint:
        raise ValueError("leakage transfer Jacobian has joint station+C_dx points: " + ", ".join(joint))
    if contract.get("held_out_closure_points"):
        raise ValueError("leakage transfer Jacobian compiled held-out operating points")
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing iteration root: {output_root}")
    sidecar = output_root.parent / f".{output_root.name}.resolved_template.yaml"
    sidecar.write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")
    try:
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
        isolation = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_only": "leakage_operator_A",
            "geometry_write_candidate": False,
            "joint_station_cdx_injection": False,
            "linearization": "all_zero_reference",
            "remeasure_A": True,
            "redefine_A": False,
            "retrain": False,
            "q_over_p_mode": 0,
            "test_data_accessed": False,
            "compiled_contract": contract,
        }
        (output_root / "isolation_contract.json").write_text(
            json.dumps(isolation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
                "isolation_contract": str(output_root / "isolation_contract.json"),
                "points": [item.get("name") for item in manifest["common_scan_plan"]["points"]],
                "n_points": len(manifest["common_scan_plan"]["points"]),
                "sources_by_split": {
                    split: sum(1 for item in manifest["sources"] if item["split"] == split)
                    for split in ("train", "validation")
                },
                "test_data_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
