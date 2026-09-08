#!/usr/bin/env python3
"""Run WB85b FWER protocol QA.  Does not execute WB86."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alignment.wb85_physical_qualification_protocol import WB85Error, refuse_execution
from alignment.wb85b_fwer_protocol import OUTPUT_ROOT, WB85BError, refuse_historical_rewrite
from alignment.wb85b_protocol_qa import refresh_verdict_with_smoke, write_protocol_qa_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--n-joint", type=int, default=None)
    parser.add_argument("--n-power", type=int, default=None)
    parser.add_argument("--smoke-path", type=Path, default=None)
    parser.add_argument("--refresh-smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--read-physical-outcomes", action="store_true")
    parser.add_argument("--run-wb86", action="store_true")
    args = parser.parse_args()
    if args.execute:
        refuse_execution("WB85b cannot execute physical qualification")
    if args.read_physical_outcomes:
        raise WB85BError("WB85b must not read physical alignment outcomes")
    if args.run_wb86:
        raise WB85BError("WB85b must not create, submit, or run WB86")
    refuse_historical_rewrite(args.output_root)
    if args.refresh_smoke:
        refresh_verdict_with_smoke(args.output_root, smoke_path=args.smoke_path)
        print(f"refreshed WB85b verdict under {args.output_root}")
    else:
        kwargs = {}
        if args.n_joint is not None:
            kwargs["n_joint"] = int(args.n_joint)
        if args.n_power is not None:
            kwargs["n_power"] = int(args.n_power)
        if args.smoke_path is not None:
            kwargs["smoke_path"] = args.smoke_path
        write_protocol_qa_artifacts(args.output_root, **kwargs)
        print(f"wrote WB85b protocol-QA artifacts under {args.output_root}")
    print("qualification_authorized = false")
    print("executable = false")
    print("wb86_automatically_authorized = false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (WB85BError, WB85Error) as error:
        print(f"WB85b refuses: {error}", file=sys.stderr)
        raise SystemExit(2)
