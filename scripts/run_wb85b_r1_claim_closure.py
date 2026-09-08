#!/usr/bin/env python3
"""WB85b-r1 claim correction and source-freeze closure.  Does not run WB86."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alignment.wb85_physical_qualification_protocol import WB85Error, refuse_execution
from alignment.wb85b_fwer_protocol import WB85BError
from alignment.wb85b_r1_claim_closure import (
    OUTPUT_ROOT,
    WB85BR1Error,
    refuse_wb85b_rewrite,
    write_claim_closure_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--freeze-commit", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--read-physical-outcomes", action="store_true")
    parser.add_argument("--run-wb86", action="store_true")
    parser.add_argument("--rerun-monte-carlo", action="store_true")
    parser.add_argument("--rerun-smoke", action="store_true")
    args = parser.parse_args()
    if args.execute:
        refuse_execution("WB85b-r1 cannot execute physical qualification")
    if args.read_physical_outcomes:
        raise WB85BR1Error("WB85b-r1 must not read physical alignment outcomes")
    if args.run_wb86:
        raise WB85BR1Error("WB85b-r1 must not create, submit, or run WB86")
    if args.rerun_monte_carlo:
        raise WB85BR1Error("WB85b-r1 must not rerun Monte-Carlo")
    if args.rerun_smoke:
        raise WB85BR1Error("WB85b-r1 must not rerun production smoke")
    refuse_wb85b_rewrite(args.output_root)
    write_claim_closure_artifacts(args.output_root, freeze_commit=args.freeze_commit)
    print(f"wrote WB85b-r1 claim-closure artifacts under {args.output_root}")
    print("qualification_authorized = false")
    print("executable = false")
    print("exact_finite_sample_FWER_proven = false")
    print("wb86_automatically_authorized = false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (WB85BR1Error, WB85BError, WB85Error) as error:
        print(f"WB85b-r1 refuses: {error}", file=sys.stderr)
        raise SystemExit(2)
