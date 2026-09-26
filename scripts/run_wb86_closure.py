#!/usr/bin/env python3
"""WB86 CLOSURE.  Freeze the oracle artifact.  ML eval stays unauthorized."""

from __future__ import annotations

import argparse

from alignment.wb86_physical_qualification import OUTPUT_ROOT, write_closure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args()
    report = write_closure(args.output_root)
    print(
        f"CLOSURE oracle={report['alignment_oracle_qualified_for_physical_FASER']} "
        f"ml={report['ml_alignment_eval_authorized']}"
    )


if __name__ == "__main__":
    main()
