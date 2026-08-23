#!/usr/bin/env python3
"""Apply the frozen residual-blind occupancy window rule.

Reads occupancy_windows.json written by the preflight scan.  Selection uses
only pre-registered count minima and skip_events order.  It never ranks
windows by occupancy beyond the first pass, and it never looks at residuals,
V2 scores, or fit quality.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alignment.real_data_occupancy_preflight import (
    BLIND_ROLES,
    load_window_rule,
    select_window_for_segment,
    xaod_path_for_segment,
)
from alignment.real_data_operating_protocol import BLOCK_INSUFFICIENT_REAL_DATA_OCCUPANCY


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--occupancy-root", required=True)
    parser.add_argument(
        "--window-rule",
        default=str(
            PROJECT_ROOT / "configs" / "operating_protocol_v1_real_data_occupancy_window_rule.yaml"
        ),
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    rule = load_window_rule(Path(args.window_rule))
    occupancy_root = Path(args.occupancy_root).expanduser().resolve()
    selected: dict[str, Any] = {}
    need_next: list[dict[str, Any]] = []
    missing: list[int] = []
    start_segment = str(rule["segment_policy"]["start_segment"])
    for run, role in BLIND_ROLES.items():
        run_dir = occupancy_root / "runs" / f"{run:05d}"
        start_path = run_dir / start_segment / "occupancy_windows.json"
        if not start_path.is_file():
            missing.append(run)
            selected[str(run)] = {
                "run": run,
                "role": role,
                "window_accepted": False,
                "status": "occupancy_scan_missing",
                "segment": start_segment,
            }
            continue
        chosen = None
        segment = start_segment
        while True:
            scan_path = run_dir / segment / "occupancy_windows.json"
            if not scan_path.is_file():
                break
            with scan_path.open(encoding="utf-8") as handle:
                scan = json.load(handle)
            if scan.get("residual_blind") is not True:
                raise ValueError(f"{scan_path} is not residual-blind")
            if scan.get("v2_scoring") is not False or scan.get("alignment_perturbation") is not False:
                raise ValueError(f"{scan_path} used V2 or alignment perturbation")
            if str(scan.get("role")) != role:
                raise ValueError(f"role drift for run {run}: {scan.get('role')} vs {role}")
            result = select_window_for_segment(
                run=run,
                segment=segment,
                windows=list(scan.get("windows") or []),
                rule=rule,
            )
            result["input_xaod"] = scan.get("input_xaod") or str(xaod_path_for_segment(run, segment))
            result["occupancy_windows_json"] = str(scan_path)
            result["n_events_scanned"] = scan.get("n_events_scanned")
            chosen = result
            if result["window_accepted"]:
                break
            if result.get("status") != "need_next_segment":
                break
            segment = str(result["next_segment"])
        if chosen is None:
            missing.append(run)
            selected[str(run)] = {
                "run": run,
                "role": role,
                "window_accepted": False,
                "status": "occupancy_scan_missing",
            }
            continue
        selected[str(run)] = chosen
        if chosen.get("status") == "need_next_segment":
            need_next.append({"run": run, "next_segment": chosen["next_segment"]})
        elif not chosen.get("window_accepted"):
            chosen["status"] = BLOCK_INSUFFICIENT_REAL_DATA_OCCUPANCY

    all_frozen = all(item.get("window_accepted") for item in selected.values()) and not missing
    provenance = {
        "schema_version": "faser-operating-protocol-v1-real-data-occupancy-provenance",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "residual_blind": True,
        "window_rule": str(Path(args.window_rule).resolve()),
        "occupancy_root": str(occupancy_root),
        "all_windows_frozen": all_frozen,
        "need_next_segment": need_next,
        "missing_scans": missing,
        "wave1_physical_plan_allowed": all_frozen,
        "runs": selected,
        "insufficient_status": BLOCK_INSUFFICIENT_REAL_DATA_OCCUPANCY,
        "do_not_lower_minima_or_change_v2": True,
        "blind_roles": {str(run): role for run, role in BLIND_ROLES.items()},
    }
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else occupancy_root / "frozen_windows.json"
    )
    output.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "all_windows_frozen": all_frozen, "need_next_segment": need_next}, indent=2))
    if not all_frozen:
        raise SystemExit("frozen occupancy windows are incomplete; do not regenerate wave-1")


if __name__ == "__main__":
    main()
