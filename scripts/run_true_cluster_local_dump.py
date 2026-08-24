#!/usr/bin/env python3
"""Rebuild NtupleDumper if needed and dump SCT cluster-local measurements.

Official current geometry only.  No alignment sqlite, no SegmentFit refit,
no V2.  Restricted to frozen-V2 selected-route events.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alignment.module_level_residual_poc import load_selected_routes
from alignment.operating_protocol_v1_final_closure import project_root, resolve_under_root
from alignment.real_data_occupancy_preflight import xaod_path_for_segment
from alignment.true_cluster_local_residual import load_stage_config, write_event_list
from scripts.run_physical_refit_capture_scan import _calypso_command, _quote, _run_shell


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
CALYPSO_ROOT = WORKSPACE_ROOT / "calypso"
BUILD_DIR = CALYPSO_ROOT / "build"


def _rebuild_ntuple_dumper(dry_run: bool) -> None:
    log_path = PROJECT_ROOT / "outputs" / "true_cluster_local_residual_feasibility_v1" / "logs" / "ntuple_dumper_build.log"
    command = (
        f"cmake -S {_quote(CALYPSO_ROOT)} -B {_quote(BUILD_DIR)} "
        f"-DCMAKE_INSTALL_PREFIX={_quote(CALYPSO_ROOT / 'run')} && "
        f"cmake --build {_quote(BUILD_DIR)} --target NtupleDumper "
        "NtupleDumperPythonInstall NtupleDumperScriptsInstall --parallel "
        "${FASER_BUILD_JOBS:-8} && "
        f"bash {_quote(PROJECT_ROOT / 'scripts' / 'refresh_calypso_confdb.sh')}"
    )
    _run_shell(_calypso_command(command), log_path, dry_run, cwd=WORKSPACE_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/true_cluster_local_residual_feasibility_v1.yaml"),
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--input-xaod", default="")
    args = parser.parse_args()
    config = load_stage_config(args.config)
    output = resolve_under_root(PROJECT_ROOT, str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    selected_path = resolve_under_root(PROJECT_ROOT, str(config["selected_routes"]))
    routes = load_selected_routes(selected_path)
    event_list = write_event_list(routes, output / "selected_events.txt")
    xaod = (
        Path(args.input_xaod).expanduser().resolve()
        if args.input_xaod
        else Path(str(config.get("input_xaod") or xaod_path_for_segment(int(config["run"]), str(config["segment"]))))
    )
    if not xaod.is_file() and not args.dry_run:
        raise FileNotFoundError(xaod)
    dump_path = output / "cluster_local.root"
    log_path = output / "logs" / "cluster_local_dump.log"
    if not args.skip_build:
        _rebuild_ntuple_dumper(args.dry_run)
    command = " ".join(
        [
            "faser_cluster_local_dump.py",
            _quote(xaod),
            "--outfile",
            _quote(dump_path),
            "--event-list",
            _quote(event_list["event_list"]),
            f"--nevents {int(config['nevents'])}",
            f"--skip-events {int(config['skip_events'])}",
        ]
    )
    _run_shell(_calypso_command(command), log_path, args.dry_run, cwd=PROJECT_ROOT)
    print(
        json.dumps(
            {
                "dry_run": bool(args.dry_run),
                "input_xaod": str(xaod),
                "event_list": event_list,
                "cluster_local_root": str(dump_path),
                "log": str(log_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
