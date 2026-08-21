#!/usr/bin/env python3
"""Audit physical-chain completion of the calibration-mode large-stats banks.

Does not submit, poll, or retune.  A bank is complete only when every
train/validation source has a mode-0 accepted physical point for every scan
payload.  Test data is never opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.submit_multisource_multidof_iteration_condor import SCHEMA_VERSION, _complete_source


DEFAULT_BANKS = (
    "outputs/mc24_ift_station_mode_large_stats_transfer_physical_v1",
    "outputs/mc24_ift_internal_cdx_large_stats_transfer_physical_v1",
    "outputs/mc24_ift_leakage_operator_large_stats_transfer_physical_v1",
)


def _read_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"not a multi-source alignment iteration manifest: {path}")
    if payload.get("test_data_accessed") is not False or int(payload.get("q_over_p_mode", -1)) != 0:
        raise ValueError(f"manifest is not a sealed mode-0 bank: {path}")
    return dict(payload)


def audit_bank(root: Path) -> dict[str, Any]:
    manifest = _read_manifest(root / "iteration_manifest.json")
    complete: list[str] = []
    incomplete: list[str] = []
    by_split = {"train": {"complete": [], "incomplete": []}, "validation": {"complete": [], "incomplete": []}}
    for source in manifest["sources"]:
        source_id = str(source["source_id"])
        split = str(source["split"])
        source_root = Path(str(source["physical_scan_root"])).expanduser().resolve().parent
        done = _complete_source(source_root)
        bucket = complete if done else incomplete
        bucket.append(source_id)
        by_split[split]["complete" if done else "incomplete"].append(source_id)
    n_points = len(manifest.get("common_scan_plan", {}).get("points") or [])
    return {
        "root": str(root),
        "n_sources": len(manifest["sources"]),
        "n_points": n_points,
        "complete_sources": complete,
        "incomplete_sources": incomplete,
        "by_split": by_split,
        "complete": not incomplete,
        "q_over_p_mode": 0,
        "test_data_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", action="append", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    roots = [Path(item).expanduser().resolve() for item in (args.bank or DEFAULT_BANKS)]
    banks = [audit_bank(root) for root in roots]
    report = {
        "banks": banks,
        "all_complete": all(bool(item["complete"]) for item in banks),
        "residual_reduction_is_not_alignment_success": True,
        "test_data_accessed": False,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).expanduser().resolve().write_text(text, encoding="utf-8")
    print(text, end="")
    if args.require_complete and not report["all_complete"]:
        missing = {
            item["root"]: item["incomplete_sources"] for item in banks if not item["complete"]
        }
        raise SystemExit("physical transfer banks are incomplete: " + json.dumps(missing))


if __name__ == "__main__":
    main()
