#!/usr/bin/env python3
"""Dump SCT cluster-local measurements for one frozen-V2 selected-route sample.

Official current geometry.  No alignment sqlite, no SegmentFit refit, no V2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alignment.module_level_residual_poc import load_selected_routes
from alignment.operating_protocol_v1_final_closure import project_root
from alignment.true_cluster_local_residual import write_event_list
from scripts.run_physical_refit_capture_scan import _calypso_command, _quote, _run_shell


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-routes", required=True)
    parser.add_argument("--input-xaod", required=True)
    parser.add_argument("--outfile", required=True)
    parser.add_argument("--event-list", required=True)
    parser.add_argument("--skip-events", type=int, required=True)
    parser.add_argument("--nevents", type=int, required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = Path(args.selected_routes).expanduser().resolve()
    xaod = Path(args.input_xaod).expanduser().resolve()
    outfile = Path(args.outfile).expanduser().resolve()
    event_list_path = Path(args.event_list).expanduser().resolve()
    log_path = Path(args.log).expanduser().resolve()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    routes = load_selected_routes(selected)
    summary = write_event_list(routes, event_list_path)
    if not xaod.is_file() and not args.dry_run:
        raise FileNotFoundError(xaod)
    command = " ".join(
        [
            "faser_cluster_local_dump.py",
            _quote(xaod),
            "--outfile",
            _quote(outfile),
            "--event-list",
            _quote(event_list_path),
            f"--nevents {int(args.nevents)}",
            f"--skip-events {int(args.skip_events)}",
        ]
    )
    _run_shell(_calypso_command(command), log_path, args.dry_run, cwd=PROJECT_ROOT)
    print(
        json.dumps(
            {
                "dry_run": bool(args.dry_run),
                "input_xaod": str(xaod),
                "event_list": summary,
                "cluster_local_root": str(outfile),
                "log": str(log_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
