#!/usr/bin/env python3
"""Prepare the isolated 1-D C_dx IFT-Internal Mode transfer bank.

Station six-vectors stay identically zero.  Only C_dx is injected, inside the
already verified |C_dx|<=0.12 mm envelope, with payload L0=+C_dx, L1=0,
L2=-C_dx.  Finite differences are axial around the all-zero reference.
Validation is present in the bank but must not register capture, A, or V2.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from alignment.hierarchical_v1 import C_DX, C_DX_ENVELOPE_MM
from alignment.layer_hierarchy import spec_scope
from scripts.config_loader import load_yaml_with_base
from scripts.prepare_layer_identifiability_pilot import compile_layer_identifiability_pilot
from scripts.prepare_multisource_multidof_iteration import prepare_iteration


def _read_template(path: Path) -> dict[str, Any]:
    payload = load_yaml_with_base(path)
    if not isinstance(payload, Mapping):
        raise ValueError("iteration template must be a YAML mapping")
    return dict(payload)


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
    scan = template.get("physical_refit_capture_scan")
    if not isinstance(scan, Mapping):
        raise ValueError("iteration template lacks physical_refit_capture_scan")
    specs = list(scan.get("alignment_parameter_specs") or [])
    names = [str(spec["name"]) for spec in specs if isinstance(spec, Mapping)]
    if names != [C_DX] or any(spec_scope(spec) != "contrast" for spec in specs):
        raise ValueError("IFT-Internal Mode transfer requires exactly contrast-scoped C_dx")
    compiled, contract = compile_layer_identifiability_pilot(
        copy.deepcopy(template),
        iteration=int(args.iteration),
        current_values={C_DX: 0.0},
    )
    if contract.get("method") != "physical_ift_internal_cdx_transfer":
        raise ValueError("C_dx transfer compiler did not emit the 1-D isolation contract")
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
            current_values={C_DX: 0.0},
            nevents=int(args.nevents),
        )
        resolved_path = output_root / "resolved_iteration_template.yaml"
        resolved_path.write_text(sidecar.read_text(encoding="utf-8"), encoding="utf-8")
        isolation = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "calibration_mode": "ift_internal",
            "floated": [C_DX],
            "station_six_vector": "identically_nominal",
            "payload_expansion": "L0=+C_dx, L1=0, L2=-C_dx",
            "C_rx": 0.0,
            "envelope_C_dx_mm": C_DX_ENVELOPE_MM,
            "joint_station_cdx_injection": False,
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
    points = list(manifest["common_scan_plan"]["points"])
    print(
        json.dumps(
            {
                "iteration_manifest": str(output_root / "iteration_manifest.json"),
                "isolation_contract": str(output_root / "isolation_contract.json"),
                "points": [item.get("name") for item in points],
                "n_points": len(points),
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
