#!/usr/bin/env python3
"""Verify that a re-materialized overlay contains exactly the same synthetic
events as the original production overlay.

The matched-retraining contract allows only the propagation record variant to
change between the mode-0 and mode-3 candidate datasets; the synthetic event
membership, overlay seed, candidate endpoints, and physical payload points
must be identical.  This script compares every sample's synthetic tracklet
content (run/event/tracklet/station/state/z/covariance, order-normalized)
between the two overlays.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.verify_mode3_bank_identity import _tracklet_digest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-overlay", required=True)
    parser.add_argument("--rematerialized-overlay", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    original_root = Path(args.original_overlay).expanduser().resolve()
    rematerialized_root = Path(args.rematerialized_overlay).expanduser().resolve()
    original = _read_json(original_root / "synthetic_corpus_manifest.json")
    rematerialized = _read_json(rematerialized_root / "synthetic_corpus_manifest.json")
    original_samples = {(str(s["split"]), str(s["payload_id"])): s for s in original["samples"]}
    rematerialized_samples = {(str(s["split"]), str(s["payload_id"])): s for s in rematerialized["samples"]}
    if set(original_samples) != set(rematerialized_samples):
        raise ValueError("sample membership differs between the two overlays")

    sample_reports: dict[str, Any] = {}
    all_identical = True
    for (split, payload_id) in sorted(original_samples):
        old_tracklets = _tracklet_digest(Path(str(original_samples[(split, payload_id)]["synthetic_tracklets"])))
        new_tracklets = _tracklet_digest(Path(str(rematerialized_samples[(split, payload_id)]["synthetic_tracklets"])))
        identical = old_tracklets == new_tracklets
        all_identical = all_identical and identical
        sample_reports[f"{split}/{payload_id}"] = {
            "identical": identical,
            "rows": new_tracklets["rows"],
        }

    summary = {
        "schema_version": "faser-mode3-overlay-identity-audit-v1",
        "original_overlay": str(original_root),
        "rematerialized_overlay": str(rematerialized_root),
        "all_identical": all_identical,
        "samples": sample_reports,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"all_identical": all_identical, "samples": len(sample_reports)}, indent=2))


if __name__ == "__main__":
    main()
