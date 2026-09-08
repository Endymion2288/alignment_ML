#!/usr/bin/env python3
"""Run WB85a-r1 joint-null QA and provenance closure.  Does not execute WB86."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alignment.wb85_physical_qualification_protocol import (
    WB85Error,
    refuse_execution,
    refuse_forbidden_path,
    refuse_wb83_wb84_write,
)
from alignment.wb85a_r1_protocol_qa import (
    OUTPUT_ROOT,
    WB85AR1Error,
    refresh_verdict_with_smoke,
    refuse_wb85a_v1_rewrite,
    write_protocol_qa_r1_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--n-joint", type=int, default=None)
    parser.add_argument("--n-null", type=int, default=None)
    parser.add_argument("--n-power", type=int, default=None)
    parser.add_argument("--smoke-path", type=Path, default=None)
    parser.add_argument(
        "--refresh-smoke",
        action="store_true",
        help="rebuild the final gate from existing Monte-Carlo artifacts plus the smoke report",
    )
    parser.add_argument("--execute", action="store_true", help="refused: WB85a-r1 does not qualify")
    parser.add_argument("--read-physical-outcomes", action="store_true", help="refused")
    parser.add_argument("--run-wb86", action="store_true", help="refused")
    args = parser.parse_args()
    if args.execute:
        refuse_execution("WB85a-r1 cannot execute physical qualification")
    if args.read_physical_outcomes:
        raise WB85AR1Error("WB85a-r1 must not read physical alignment outcomes")
    if args.run_wb86:
        raise WB85AR1Error("WB85a-r1 must not create, submit, or run WB86")
    refuse_forbidden_path(args.output_root)
    refuse_wb83_wb84_write(args.output_root)
    refuse_wb85a_v1_rewrite(args.output_root)
    kwargs = {}
    if args.n_joint is not None:
        kwargs["n_joint"] = int(args.n_joint)
    if args.n_null is not None:
        kwargs["n_null"] = int(args.n_null)
    if args.n_power is not None:
        kwargs["n_power"] = int(args.n_power)
    if args.smoke_path is not None:
        kwargs["smoke_path"] = args.smoke_path
    if args.refresh_smoke:
        root = refresh_verdict_with_smoke(args.output_root, smoke_path=args.smoke_path)
        print(f"refreshed WB85a-r1 verdict under {root}")
        print("qualification_authorized = false")
        print("executable = false")
        print("looked_at_physical_alignment_outcomes = false")
        print("wb86_automatically_authorized = false")
        return 0
    root = write_protocol_qa_r1_artifacts(args.output_root, **kwargs)
    print(f"wrote WB85a-r1 protocol-QA artifacts under {root}")
    print("qualification_authorized = false")
    print("executable = false")
    print("looked_at_physical_alignment_outcomes = false")
    print("wb86_automatically_authorized = false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (WB85AR1Error, WB85Error) as error:
        print(f"WB85a-r1 refuses: {error}", file=sys.stderr)
        raise SystemExit(2)
