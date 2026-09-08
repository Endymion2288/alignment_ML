#!/usr/bin/env python3
"""WB85b official qualification-runner e2e smoke.  Plumbing only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alignment.wb85_physical_qualification_protocol import WB85Error, refuse_execution
from alignment.wb85b_fwer_protocol import WB85BError, refuse_historical_rewrite
from alignment.wb85b_official_runner import (
    SMOKE_OUTPUT,
    run_official_runner_smoke_or_record_failure,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=SMOKE_OUTPUT)
    parser.add_argument("--qualify", action="store_true")
    parser.add_argument("--run-wb86", action="store_true")
    args = parser.parse_args()
    if args.qualify:
        refuse_execution("official runner smoke is not physical qualification")
    if args.run_wb86:
        raise WB85BError("WB85b must not create, submit, or run WB86")
    refuse_historical_rewrite(args.output_root)
    report = run_official_runner_smoke_or_record_failure(args.output_root)
    passed = bool(report.get("official_qualification_runner_e2e_smoke_pass"))
    print(f"official_qualification_runner_e2e_smoke_pass = {str(passed).lower()}")
    print(f"calypso_acts_engine_runtime_smoke_pass = {str(bool(report.get('calypso_acts_engine_runtime_smoke_pass'))).lower()}")
    print("alignment_performance_not_reported = true")
    print("qualification_authorized = false")
    if not passed:
        print(f"reason: {report.get('reason', 'smoke failed')}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (WB85BError, WB85Error) as error:
        print(f"WB85b official runner smoke refuses: {error}", file=sys.stderr)
        raise SystemExit(2)
