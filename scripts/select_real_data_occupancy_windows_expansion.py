#!/usr/bin/env python3
"""Apply the frozen residual-blind occupancy rule to expansion runs.

Selection uses only pre-registered count minima and skip_events order.
It never ranks windows by occupancy and never looks at residuals.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alignment.real_data_occupancy_preflight import load_window_rule
from alignment.real_data_operating_protocol import BLOCK_INSUFFICIENT_REAL_DATA_OCCUPANCY
from alignment.real_data_residual_dq_monitoring_expansion import (
    EXPANSION_RUNS,
    ROLE_MONITORING,
    load_expansion_config,
    select_expansion_window_from_scans,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--occupancy-root", required=True)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "operating_protocol_v1_real_data_residual_dq_monitoring_expansion_v1.yaml"),
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    config = load_expansion_config(args.config)
    rule = load_window_rule(PROJECT_ROOT / str(config["window_rule"]))
    occupancy_root = Path(args.occupancy_root).expanduser().resolve()
    selected: dict[str, Any] = {}
    need_next: list[dict[str, Any]] = []
    missing: list[int] = []
    for run in EXPANSION_RUNS:
        chosen = select_expansion_window_from_scans(
            run=run,
            occupancy_root=occupancy_root,
            rule=rule,
        )
        chosen["role"] = ROLE_MONITORING
        selected[str(run)] = chosen
        if chosen.get("status") == "occupancy_scan_missing":
            missing.append(run)
        elif chosen.get("status") == "need_next_segment":
            need_next.append({"run": run, "next_segment": chosen.get("next_segment")})
        elif not chosen.get("window_accepted"):
            chosen["status"] = BLOCK_INSUFFICIENT_REAL_DATA_OCCUPANCY
    all_frozen = all(item.get("window_accepted") for item in selected.values()) and not missing
    provenance = {
        "schema_version": "faser-operating-protocol-v1-real-data-occupancy-expansion-provenance",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "residual_blind": True,
        "window_rule": str((PROJECT_ROOT / str(config["window_rule"])).resolve()),
        "occupancy_root": str(occupancy_root),
        "all_windows_frozen": all_frozen,
        "need_next_segment": need_next,
        "missing_scans": missing,
        "athena_current_geometry_allowed": all_frozen,
        "runs": selected,
        "do_not_lower_minima_or_change_v2": True,
        "do_not_repick_window_from_residual": True,
        "geometry_write_allowed": False,
        "station_calibration_mode_available": False,
        "cdx_mode_allowed": False,
    }
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else occupancy_root / "frozen_windows.json"
    )
    output.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "all_windows_frozen": all_frozen, "need_next_segment": need_next}, indent=2))
    if not all_frozen:
        raise SystemExit("expansion occupancy windows are incomplete; do not start Athena")


if __name__ == "__main__":
    main()
