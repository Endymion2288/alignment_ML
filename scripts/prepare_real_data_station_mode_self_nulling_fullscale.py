#!/usr/bin/env python3
"""Prepare calibration-only Station Mode FD scans on full remaining segments.

Reuses the scaling-study current-geometry reconstruction.  Holdout and
14977 stay current-only and are not submitted for finite-difference
Athena.  Official conditions are never written.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from alignment.real_data_operating_protocol import (
    ROLE_CALIBRATION,
    compile_real_data_current_only,
    compile_real_data_station_linearization,
)
from alignment.real_data_station_mode_fullscale import load_fullscale_config
from scripts.config_loader import load_yaml_with_base
from scripts.run_physical_refit_capture_scan import _build_plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_POINT = "iteration_00_current"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON mapping: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    destination.symlink_to(source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "operating_protocol_v1_real_data_station_mode_self_nulling_fullscale_v1.yaml"),
    )
    parser.add_argument(
        "--station-template",
        default=str(PROJECT_ROOT / "configs" / "physical_refit_station_mode_real_data_dryrun.yaml"),
    )
    parser.add_argument("--campaign-root", required=True)
    args = parser.parse_args()
    config = load_fullscale_config(args.config)
    frozen = _read_json(PROJECT_ROOT / str(config["frozen_windows"]))
    if frozen.get("all_windows_frozen") is not True:
        raise ValueError("fullscale Station Mode requires frozen occupancy windows")
    scaling = _read_json(PROJECT_ROOT / str(config["scaling_root"]) / "iteration_manifest.json")
    scaling_by_id = {str(row["source_id"]): row for row in scaling["sources"]}
    template = load_yaml_with_base(Path(args.station_template).expanduser().resolve())
    campaign_root = Path(args.campaign_root).expanduser().resolve()
    physical_root = campaign_root / "physical"
    if physical_root.exists() and any(physical_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty physical root: {physical_root}")
    physical_root.mkdir(parents=True, exist_ok=True)

    sources: list[dict[str, Any]] = []
    for run_key, meta in config["full_segment_sources"].items():
        run = int(run_key)
        role = str(config["blind_roles"][str(run)])
        occupancy = frozen["runs"][str(run)]
        source_id = str(meta["source_id"])
        xaod = Path(str(occupancy["input_xaod"])).expanduser().resolve()
        compiled, contract = (
            compile_real_data_station_linearization(template)
            if role == ROLE_CALIBRATION
            else compile_real_data_current_only(template)
        )
        scan = dict(compiled["physical_refit_capture_scan"])
        scan["input_xaod"] = str(xaod)
        scan["nevents"] = int(meta["nevents"])
        scan["skip_events"] = int(meta["skip_events"])
        scan["is_mc"] = False
        scan["include_truth"] = False
        scan["require_mc_labels"] = False
        scan["cdx_fixed_by"] = "external_geometry"
        scan["run_alignment_closure"] = False
        plan = _build_plan(scan)
        source_root = physical_root / "sources" / source_id
        source_root.mkdir(parents=True, exist_ok=False)
        config_path = source_root / "physical_scan_config.yaml"
        config_path.write_text(
            yaml.safe_dump({"physical_refit_capture_scan": scan}, sort_keys=False),
            encoding="utf-8",
        )
        _write_json(source_root / "linearization_contract.json", contract)
        _write_json(source_root / "scan_plan.json", plan)
        scaling_source = scaling_by_id[source_id]
        reused_current = Path(str(scaling_source["physical_scan_root"])) / "points" / CURRENT_POINT
        if not reused_current.is_dir():
            raise FileNotFoundError(f"missing reused current point: {reused_current}")
        _symlink(reused_current, source_root / "physical_scan" / "points" / CURRENT_POINT)
        sources.append(
            {
                "source_id": source_id,
                "run": run,
                "role": role,
                "split": "train" if role == ROLE_CALIBRATION else "validation",
                "segment": str(meta["segment"]),
                "skip_events": int(meta["skip_events"]),
                "nevents": int(meta["nevents"]),
                "input_xaod": str(xaod),
                "geometry_tag": config["geometry_tag"],
                "conditions_tag": config["conditions_tag"],
                "reconstruction_tag": config["reconstruction_tag"],
                "cdx_fixed_by": "external_geometry",
                "geometry_estimation_allowed": role == ROLE_CALIBRATION,
                "geometry_write_allowed": False,
                "cdx_mode_blocked": True,
                "reused_current_point": str(reused_current),
                "physical_scan_config": str(config_path),
                "physical_scan_root": str(source_root / "physical_scan"),
                "n_planned_points": len(plan["points"]),
                "needs_athena_fd": role == ROLE_CALIBRATION,
            }
        )

    created = datetime.now(timezone.utc).isoformat()
    _write_json(
        physical_root / "real_data_provenance_catalog.json",
        {
            "schema_version": SCHEMA_VERSION_PROVENANCE,
            "created_utc": created,
            "official_conditions_db_write": False,
            "mc_truth_used": False,
            "geometry_write_allowed": False,
            "cdx_mode_blocked": True,
            "joint_station_cdx_newton": False,
            "reused_scaling_current_geometry": True,
            "protocol_config": config["path"],
            "blocks": sources,
        },
    )
    _write_json(
        physical_root / "iteration_manifest.json",
        {
            "schema_version": "faser-operating-protocol-v1-real-data-station-mode-self-nulling-fullscale-physical",
            "created_utc": created,
            "physical_geometry_repropagation": True,
            "q_over_p_mode": 0,
            "is_mc": False,
            "official_conditions_db_write": False,
            "geometry_write_allowed": False,
            "current_geometry_only": False,
            "finite_difference_probes": True,
            "calibration_only_fd": True,
            "sources": sources,
        },
    )
    print(
        json.dumps(
            {
                "physical_root": str(physical_root),
                "n_sources": len(sources),
                "calibration_fd_sources": [
                    row["source_id"] for row in sources if row["needs_athena_fd"]
                ],
            },
            indent=2,
        )
    )


SCHEMA_VERSION_PROVENANCE = "faser-operating-protocol-v1-real-data-station-mode-self-nulling-fullscale-provenance"


if __name__ == "__main__":
    main()
