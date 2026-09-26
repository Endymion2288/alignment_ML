#!/usr/bin/env python3
"""WB86 per-event official physical iterate."""

from __future__ import annotations

import argparse

from alignment.wb86_event_worker import execute_one_event
from alignment.wb86_physical_qualification import OUTPUT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    report = execute_one_event(int(args.index), args.output_root)
    print(f"event {args.index} {report['technical_status']} uid={report['event_uid']}")
    # Recorded technical FAIL must still exit 0.  DAGMan treats the 2394-proc
    # shard node as one unit; a single nonzero rc aborts the remaining jobs.


if __name__ == "__main__":
    main()
