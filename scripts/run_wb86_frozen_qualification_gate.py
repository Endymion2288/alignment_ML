#!/usr/bin/env python3
"""WB86 FROZEN_QUALIFICATION_GATE.  Calls evaluate_wb85b_qualification only."""

from __future__ import annotations

import argparse

from alignment.wb86_physical_qualification import OUTPUT_ROOT, frozen_qualification_gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args()
    report = frozen_qualification_gate(args.output_root)
    print(
        f"GATE overall={report['overall_qualification']} "
        f"oracle={report['alignment_oracle_qualified_for_physical_FASER']}"
    )


if __name__ == "__main__":
    main()
