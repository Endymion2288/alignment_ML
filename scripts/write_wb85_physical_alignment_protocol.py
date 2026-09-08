#!/usr/bin/env python3
"""Write the frozen WB85 physical alignment qualification protocol.

Does not run common-track alignment, does not read physical alignment
outcomes, and does not authorize qualification.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alignment.wb85_physical_qualification_protocol import (
    OUTPUT_ROOT,
    WB85Error,
    protocol_freeze_complete,
    refuse_execution,
    refuse_forbidden_path,
    refuse_wb83_wb84_write,
    write_protocol_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true", help="refused: WB85 is protocol-only")
    args = parser.parse_args()
    if args.execute:
        refuse_execution("this writer cannot execute qualification")
    refuse_forbidden_path(args.output_root)
    refuse_wb83_wb84_write(args.output_root)
    root = write_protocol_artifacts(args.output_root)
    if not protocol_freeze_complete():
        raise WB85Error("WB85 protocol freeze is incomplete")
    print(f"wrote WB85 protocol artifacts under {root}")
    print("executable = false")
    print("qualification_authorized = false")
    print("protocol_audit = PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WB85Error as error:
        print(f"WB85 refuses: {error}", file=sys.stderr)
        raise SystemExit(2)
