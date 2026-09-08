#!/usr/bin/env python3
"""Official Calypso/ACTS runtime smoke.  Plumbing only; not physics qualification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alignment.wb85_physical_qualification_protocol import WB85Error, refuse_execution
from alignment.wb85a_r1_protocol_qa import refuse_wb85a_v1_rewrite
from alignment.wb85a_r1_runtime_smoke import (
    SMOKE_OUTPUT,
    WB85AR1SmokeError,
    run_runtime_smoke_or_record_failure,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=SMOKE_OUTPUT)
    parser.add_argument("--qualify", action="store_true", help="refused")
    parser.add_argument("--run-wb86", action="store_true", help="refused")
    args = parser.parse_args()
    if args.qualify:
        refuse_execution("runtime smoke is not physical qualification")
    if args.run_wb86:
        raise WB85AR1SmokeError("WB85a-r1 must not create, submit, or run WB86")
    refuse_wb85a_v1_rewrite(args.output_root)
    report = run_runtime_smoke_or_record_failure(args.output_root)
    passed = bool(report.get("physical_execution_runtime_smoke_pass"))
    print(f"physical_execution_runtime_smoke_pass = {str(passed).lower()}")
    print(f"physical_execution_ready = {str(bool(report.get('physical_execution_ready'))).lower()}")
    print("alignment_performance_not_reported = true")
    print("qualification_authorized = false")
    if not passed:
        print(f"reason: {report.get('reason', 'smoke failed')}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (WB85AR1SmokeError, WB85Error) as error:
        print(f"WB85a-r1 smoke refuses: {error}", file=sys.stderr)
        raise SystemExit(2)
