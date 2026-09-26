#!/usr/bin/env python3
"""WB86 GLOBAL_SUMMARY.  First authorized scientific aggregate after integrity."""

from __future__ import annotations

import argparse

from alignment.wb86_physical_qualification import OUTPUT_ROOT, global_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args()
    report = global_summary(args.output_root)
    print(f"GLOBAL_SUMMARY wrote {args.output_root}/wb86_global_summary.json schema={report['schema']}")


if __name__ == "__main__":
    main()
