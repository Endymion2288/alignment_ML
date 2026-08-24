#!/usr/bin/env python3
"""Audit that workbook-56 transfer xAODs never entered 48–55 or sealed test."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


SOURCE_RE = re.compile(r"mc24_\d{6}_\d{5}_\d{5}")
TRANSFER = ("mc24_100047_00300_00349", "mc24_100048_00300_00349")
SEALED_PREFIXES = ("mc24_100116_", "mc24_100117_")
HISTORICAL_GLOBS = (
    "configs/**/*.{yaml,yml}",
    "workbook/**/*.md",
)


def _iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.glob("configs/**/*"):
        if path.suffix in {".yaml", ".yml", ".json"}:
            files.append(path)
    for path in root.glob("workbook/**/*.md"):
        files.append(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    used: dict[str, list[str]] = {}
    skip = {
        "configs/physical_curriculum_four_station_relative_transfer_sources.yaml",
        "configs/physical_four_station_gauge_consistent_route_training.yaml",
    }
    for path in _iter_files(root):
        relative = str(path.relative_to(root))
        if relative in skip or relative.startswith("workbook/2026-08-22_56_"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in SOURCE_RE.findall(text):
            used.setdefault(match, []).append(relative)
    conflicts = {}
    for source in TRANSFER:
        hits = [path for path in used.get(source, []) if path not in skip]
        if hits:
            conflicts[source] = sorted(set(hits))
    sealed_hits = {
        source: sorted(set(paths))
        for source, paths in used.items()
        if source.startswith(SEALED_PREFIXES)
    }
    payload = {
        "schema_version": "faser-four-station-transfer-source-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "transfer_sources": list(TRANSFER),
        "unused_in_historical_artifacts": not conflicts,
        "conflicts": conflicts,
        "sealed_sources_seen_in_repo_text_only": sorted(sealed_hits),
        "test_data_accessed": False,
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if conflicts:
        raise SystemExit("transfer sources already appear in historical artifacts")
    print(json.dumps({"output_json": str(output), "ok": True}, indent=2))


if __name__ == "__main__":
    main()
