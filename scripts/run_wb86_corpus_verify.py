#!/usr/bin/env python3
"""WB86 CORPUS_VERIFY.  Catalog the frozen WB84 exports; no scientific alignment."""

from __future__ import annotations

import argparse
from pathlib import Path

from alignment.wb85_physical_qualification_protocol import write_json
from alignment.wb86_physical_qualification import (
    OUTPUT_ROOT,
    build_event_catalog,
    corpus_verify,
    require_remote_verification,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args()
    require_remote_verification()
    root = Path(args.output_root)
    snapshot = root / "wb86_snapshot_verify.json"
    if not snapshot.is_file():
        raise SystemExit("SNAPSHOT_VERIFY artifact is absent")
    report = corpus_verify()
    write_json(root / "wb86_corpus_verify.json", report)
    write_json(root / "wb86_event_catalog.json", build_event_catalog())
    if not report["corpus_verify_pass"]:
        raise SystemExit("CORPUS_VERIFY failed")
    print(f"CORPUS_VERIFY PASS n_events={report['n_events']}")


if __name__ == "__main__":
    main()
