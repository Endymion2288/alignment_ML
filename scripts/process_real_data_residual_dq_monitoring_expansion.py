#!/usr/bin/env python3
"""Materialize identity + frozen V2 association for expansion Athena jobs.

Current-geometry only.  Does not retrain V2, generate FD probes, or write
a geometry payload.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.process_real_data_v2_route_acceptance_scale import main as _scaling_main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--athena-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-id", action="append", default=None)
    parser.add_argument("--run-association", action="store_true")
    args = parser.parse_args()
    argv = ["--scaling-root", str(Path(args.athena_root).expanduser().resolve()), "--device", args.device]
    if args.run_association:
        argv.append("--run-association")
    if args.source_id:
        for source_id in args.source_id:
            argv.extend(["--source-id", source_id])
    import sys

    sys.argv = ["process_real_data_v2_route_acceptance_scale.py", *argv]
    _scaling_main()


if __name__ == "__main__":
    main()
