#!/usr/bin/env python3
"""Prepare current-geometry-only Athena jobs for monitoring expansion runs.

Uses the frozen occupancy start.  Official monitoring uses the full remaining
segment.  No FD probes, Newton, payload, or window re-pick.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from alignment.real_data_operating_protocol import compile_real_data_current_only
from alignment.real_data_residual_dq_monitoring_expansion import (
    EXPANSION_RUNS,
    load_expansion_config,
    plan_full_remaining_monitoring,
)
from alignment.real_data_v2_route_acceptance_scaling import occupancy_slice_summary, load_occupancy_table
from scripts.config_loader import load_yaml_with_base
from scripts.run_physical_refit_capture_scan import _build_plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON mapping: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "operating_protocol_v1_real_data_residual_dq_monitoring_expansion_v1.yaml"),
    )
    parser.add_argument("--frozen-windows", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    config = load_expansion_config(args.config)
    frozen = _read_json(Path(args.frozen_windows).expanduser().resolve())
    if frozen.get("all_windows_frozen") is not True:
        raise ValueError("expansion Athena requires frozen occupancy windows")
    template = load_yaml_with_base((PROJECT_ROOT / str(config["station_template"])).resolve())
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty expansion Athena root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    occupancy_forecasts: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    planned: list[dict[str, Any]] = []
    for run in EXPANSION_RUNS:
        row = frozen["runs"][str(run)]
        if row.get("window_accepted") is not True:
            raise ValueError(f"run {run} does not have a frozen occupancy window")
        item = plan_full_remaining_monitoring(
            run=run,
            segment=str(row["segment"]),
            skip_events=int(row["skip_events"]),
            n_segment_events=int(row["n_events_scanned"]),
        )
        occupancy_root = Path(str(row.get("occupancy_root") or Path(str(row["occupancy_windows_json"])).parent / "occupancy.root"))
        table = load_occupancy_table(occupancy_root)
        forecast = occupancy_slice_summary(
            table,
            skip_events=int(item["skip_events"]),
            nevents=int(item["nevents"]),
        )
        planned.append(item)
        occupancy_forecasts.append({**item, "occupancy_forecast": forecast, "occupancy_root": str(occupancy_root)})
        compiled, contract = compile_real_data_current_only(template)
        scan = dict(compiled["physical_refit_capture_scan"])
        scan["input_xaod"] = str(row["input_xaod"])
        scan["nevents"] = int(item["nevents"])
        scan["skip_events"] = int(item["skip_events"])
        scan["is_mc"] = False
        scan["include_truth"] = False
        scan["require_mc_labels"] = False
        scan["cdx_fixed_by"] = "external_geometry"
        scan["current_geometry_only"] = True
        scan["run_alignment_closure"] = False
        scan["generate_geometry_payload"] = False
        plan = _build_plan(scan)
        if len(plan["points"]) != 1:
            raise ValueError(f"{item['source_id']} must have exactly the current-geometry point")
        source_root = output_root / "sources" / item["source_id"]
        source_root.mkdir(parents=True, exist_ok=False)
        config_path = source_root / "physical_scan_config.yaml"
        config_path.write_text(
            yaml.safe_dump({"physical_refit_capture_scan": scan}, sort_keys=False),
            encoding="utf-8",
        )
        _write_json(source_root / "linearization_contract.json", contract)
        _write_json(source_root / "scan_plan.json", plan)
        sources.append(
            {
                **item,
                "split": "validation",
                "input_xaod": str(row["input_xaod"]),
                "xaod_segment": item["segment"],
                "occupancy_forecast": forecast,
                "lhc_fill": row.get("lhc_fill"),
                "geometry_tag": "FASERNU-04",
                "conditions_tag": "OFLCOND-FASER-06",
                "reconstruction_tag": "r0022",
                "cdx_fixed_by": "external_geometry",
                "geometry_estimation_allowed": False,
                "physical_scan_config": str(config_path),
                "physical_scan_root": str(source_root / "physical_scan"),
                "n_planned_points": 1,
            }
        )

    created = datetime.now(timezone.utc).isoformat()
    _write_json(
        output_root / "expansion_athena_plan.json",
        {
            "schema_version": config["schema_version"],
            "created_utc": created,
            "frozen_windows": str(Path(args.frozen_windows).resolve()),
            "official_conditions_db_write": False,
            "geometry_write_allowed": False,
            "station_calibration_mode_available": False,
            "cdx_mode_allowed": False,
            "use_residual": False,
            "generate_geometry_payload": False,
            "finite_difference_probes": False,
            "planned": planned,
            "occupancy_forecasts": occupancy_forecasts,
        },
    )
    _write_json(
        output_root / "iteration_manifest.json",
        {
            "schema_version": "faser-operating-protocol-v1-real-data-residual-dq-monitoring-expansion-physical",
            "created_utc": created,
            "physical_geometry_repropagation": True,
            "q_over_p_mode": 0,
            "is_mc": False,
            "official_conditions_db_write": False,
            "geometry_write_allowed": False,
            "current_geometry_only": True,
            "finite_difference_probes": False,
            "sources": sources,
        },
    )
    print(json.dumps({"output_root": str(output_root), "athena_sources": len(sources)}, indent=2))


if __name__ == "__main__":
    main()
