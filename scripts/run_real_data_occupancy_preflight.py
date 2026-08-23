#!/usr/bin/env python3
"""Run residual-blind occupancy preflight for one 2024 r0022 xAOD segment.

Official current geometry only.  No alignment sqlite, no V2, no residual
inspection.  This command writes occupancy ROOT + 100-event window summaries;
it does not choose a window.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import uproot
import yaml

from alignment.real_data_occupancy_preflight import (
    BLIND_ROLES,
    assert_occupancy_tree_is_residual_blind,
    contiguous_windows,
    load_window_rule,
    xaod_path_for_segment,
)
from alignment.real_data_operating_protocol import WORKBOOK_03_BLOCKED_REASON
from scripts.run_physical_refit_capture_scan import _calypso_command, _quote, _run_shell


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _protocol(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    raw = payload.get("operating_protocol_v1_real_data_dryrun", payload)
    return dict(raw)


def _load_occupancy_table(path: Path) -> dict[str, np.ndarray]:
    with uproot.open(path) as source:
        tree_name = "occupancy" if "occupancy" in source else None
        if tree_name is None:
            for key in source.keys():
                if str(key).split(";")[0].endswith("occupancy") or str(key).split(";")[0] == "occupancy":
                    tree_name = str(key).split(";")[0]
                    break
        if tree_name is None:
            raise ValueError(f"occupancy tree is absent from {path}: {list(source.keys())}")
        tree = source[tree_name]
        names = [str(name) for name in tree.keys()]
        assert_occupancy_tree_is_residual_blind(names)
        arrays = tree.arrays(library="np")
    return {str(key): np.asarray(value) for key, value in arrays.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--segment", default="00000")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--nevents", type=int, default=-1)
    parser.add_argument("--skip-events", type=int, default=0)
    parser.add_argument(
        "--protocol-config",
        default=str(PROJECT_ROOT / "configs" / "operating_protocol_v1_real_data_dryrun.yaml"),
    )
    parser.add_argument(
        "--window-rule",
        default=str(
            PROJECT_ROOT / "configs" / "operating_protocol_v1_real_data_occupancy_window_rule.yaml"
        ),
    )
    parser.add_argument("--input-xaod", default="")
    parser.add_argument(
        "--role",
        default="",
        help="Set to monitoring for expansion runs outside the frozen 14973-14977 blind split",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rule = load_window_rule(Path(args.window_rule))
    protocol = _protocol(Path(args.protocol_config))
    run = int(args.run)
    if args.role:
        if str(args.role) != "monitoring":
            raise ValueError("occupancy --role must be omitted or monitoring")
        role = "monitoring"
    else:
        if run not in BLIND_ROLES:
            raise ValueError(f"run {run} is not in the frozen blind split")
        if BLIND_ROLES[run] != next(
            str(block["role"]) for block in protocol["blocks"] if int(block["run"]) == run
        ):
            raise ValueError("protocol roles drifted from the frozen occupancy window rule")
        role = BLIND_ROLES[run]
    segment = f"{int(args.segment):05d}"
    xaod = (
        Path(args.input_xaod).expanduser().resolve()
        if args.input_xaod
        else xaod_path_for_segment(run, segment).expanduser().resolve()
    )
    if "data0/rec/2022" in str(xaod.resolve()):
        raise ValueError(WORKBOOK_03_BLOCKED_REASON)
    if "rec/2024/r0022" not in str(xaod.resolve()):
        raise ValueError(f"occupancy preflight admits only 2024 r0022 xAOD, got {xaod}")
    if not xaod.is_file():
        raise FileNotFoundError(xaod)

    output_root = Path(args.output_dir).expanduser().resolve()
    run_dir = output_root / "runs" / f"{run:05d}" / segment
    for attempt in range(6):
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            break
        except FileExistsError:
            if attempt == 5:
                raise
            time.sleep(0.25 * (attempt + 1))
    occupancy_root = run_dir / "occupancy.root"
    log_path = run_dir / "logs" / "occupancy_preflight.log"
    command = " ".join(
        [
            "faser_occupancy_preflight.py",
            _quote(xaod),
            "--outfile",
            _quote(occupancy_root),
            f"--nevents {int(args.nevents)}",
            *(
                [f"--skip-events {int(args.skip_events)}"]
                if int(args.skip_events) > 0
                else []
            ),
        ]
    )
    _run_shell(_calypso_command(command), log_path, args.dry_run, cwd=PROJECT_ROOT)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "command": command}, indent=2))
        return

    table = _load_occupancy_table(occupancy_root)
    window_cfg = rule["window"]
    windows = contiguous_windows(
        table,
        nevents=int(window_cfg["nevents"]),
        skip_start=int(window_cfg["skip_events_start"]),
        skip_stride=int(window_cfg["skip_events_stride"]),
        job_skip_events=int(args.skip_events),
    )
    summary = {
        "schema_version": "faser-operating-protocol-v1-real-data-occupancy-scan",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "residual_blind": True,
        "alignment_perturbation": False,
        "v2_scoring": False,
        "geometry": "official_current",
        "run": run,
        "role": role,
        "segment": segment,
        "input_xaod": str(xaod.resolve()),
        "occupancy_root": str(occupancy_root),
        "n_events_scanned": int(np.asarray(table["n_sct_clusters"]).shape[0]),
        "job_skip_events": int(args.skip_events),
        "window_rule": str(Path(args.window_rule).resolve()),
        "windows": windows,
        "window_selection_not_applied": True,
    }
    (run_dir / "occupancy_windows.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run": run,
                "segment": segment,
                "n_events_scanned": summary["n_events_scanned"],
                "n_windows": len(windows),
                "occupancy_root": str(occupancy_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
