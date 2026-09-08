#!/usr/bin/env python3
"""Run WB85a independent protocol QA.  Does not execute physical qualification."""

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
from alignment.wb85a_protocol_qa import (
    OUTPUT_ROOT,
    WB85AError,
    write_protocol_qa_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--n-null", type=int, default=None)
    parser.add_argument("--n-power", type=int, default=None)
    parser.add_argument("--execute", action="store_true", help="refused: WB85a does not qualify")
    parser.add_argument("--read-physical-outcomes", action="store_true", help="refused")
    args = parser.parse_args()
    if args.execute:
        refuse_execution("WB85a cannot execute physical qualification")
    if args.read_physical_outcomes:
        raise WB85AError("WB85a must not read physical alignment outcomes")
    refuse_forbidden_path(args.output_root)
    refuse_wb83_wb84_write(args.output_root)
    kwargs = {}
    if args.n_null is not None:
        kwargs["n_null"] = int(args.n_null)
    if args.n_power is not None:
        kwargs["n_power"] = int(args.n_power)
    root = write_protocol_qa_artifacts(args.output_root, **kwargs)
    print(f"wrote WB85a protocol-QA artifacts under {root}")
    print("qualification_authorized = false")
    print("executable = false")
    print("looked_at_physical_alignment_outcomes = false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (WB85AError, WB85Error) as error:
        print(f"WB85a refuses: {error}", file=sys.stderr)
        raise SystemExit(2)
