#!/usr/bin/env python3
"""WB86 SNAPSHOT_VERIFY.  Hash-check frozen inputs; no scientific alignment."""

from __future__ import annotations

import argparse
from pathlib import Path

from alignment.wb85_physical_qualification_protocol import write_json
from alignment.wb86_physical_qualification import OUTPUT_ROOT, require_remote_verification, snapshot_verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args()
    require_remote_verification()
    root = Path(args.output_root)
    report = snapshot_verify()
    write_json(root / "wb86_snapshot_verify.json", report)
    if not report["snapshot_verify_pass"]:
        raise SystemExit("SNAPSHOT_VERIFY failed")
    print(f"SNAPSHOT_VERIFY PASS -> {root / 'wb86_snapshot_verify.json'}")


if __name__ == "__main__":
    main()
