#!/usr/bin/env python3
"""WB86 RESULT_INTEGRITY.  File presence and hashes only; no pull/bias/coverage."""

from __future__ import annotations

import argparse

from alignment.wb86_physical_qualification import OUTPUT_ROOT, result_integrity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args()
    report = result_integrity(args.output_root)
    print(
        f"RESULT_INTEGRITY pass={report['result_integrity_pass']} "
        f"success={report['n_technical_success']} missing={report['n_missing']}"
    )
    if not report["result_integrity_pass"]:
        raise SystemExit("RESULT_INTEGRITY failed")


if __name__ == "__main__":
    main()
