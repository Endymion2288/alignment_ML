#!/usr/bin/env python3
"""Confirm the reserved blind pair was never used in workbooks 48–63.

Listing the pair as reserved / do-not-load is allowed.  Appearing as a
loaded source_id, overlay constituent, identity-summary key, or
origin_source_id is not.  This does not open the blind files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.run_refit_multidof_closure import _json_ready
from training.source_diversity_audit import RESERVED_BLIND_SOURCES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESERVED = set(RESERVED_BLIND_SOURCES)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_mapping(payload: Any) -> dict[str, Any] | None:
    return dict(payload) if isinstance(payload, Mapping) else None


def loaded_sources_from_artifact(path: Path, payload: Any) -> set[str]:
    found: set[str] = set()
    data = _as_mapping(payload)
    if data is None:
        return found
    name = path.name
    if name in {"physical_corpus_manifest.json", "iteration_manifest.json"}:
        for item in data.get("sources") or []:
            if isinstance(item, Mapping):
                found.add(str(item.get("source_id") or item.get("id") or ""))
    if name == "synthetic_corpus_manifest.json":
        for item in data.get("samples") or data.get("sources") or []:
            if not isinstance(item, Mapping):
                continue
            found.add(str(item.get("source_id") or ""))
            for source in item.get("source_ids") or []:
                found.add(str(source))
        audit = data.get("source_split_audit")
        if isinstance(audit, Mapping):
            by_split = audit.get("sources_by_split")
            if isinstance(by_split, Mapping):
                for values in by_split.values():
                    if isinstance(values, list):
                        found.update(str(item) for item in values)
    if name == "identity_summaries.json":
        for block in data.values():
            if isinstance(block, Mapping):
                found.update(str(key) for key in block)
                found.update(str(key).removeprefix("candidate:") for key in block)
    if name in {"validation_run_contract.json"}:
        for key in ("loaded_sources", "source_ids", "development_validation_only_sources"):
            values = data.get(key)
            if isinstance(values, list):
                found.update(str(item) for item in values)
    return {item for item in found if item in RESERVED}


WB48_63_OUTPUT_ROOTS = (
    "mc24_four_station_identifiability_pilot_v1",
    "mc24_four_station_relative_association_retrain_v1",
    "mc24_four_station_relative_transfer_validation_v1",
    "mc24_four_station_gauge_consistent_route_v1",
    "mc24_four_station_dustbin_aware_route_v1",
    "mc24_four_station_hard_aware_reduction_v1",
)
JSON_NAMES = (
    "physical_corpus_manifest.json",
    "synthetic_corpus_manifest.json",
    "iteration_manifest.json",
    "identity_summaries.json",
    "validation_run_contract.json",
)


def collect_artifact_hits(root: Path) -> list[dict[str, object]]:
    hits: list[dict[str, object]] = []
    outputs = root / "outputs"
    for bank in WB48_63_OUTPUT_ROOTS:
        bank_root = outputs / bank
        if not bank_root.is_dir():
            continue
        for name in JSON_NAMES:
            for path in bank_root.rglob(name):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                found = sorted(loaded_sources_from_artifact(path, payload))
                if found:
                    hits.append({"path": str(path), "sources": found, "kind": "artifact_json"})
        for path in bank_root.rglob("truth_routes.jsonl"):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            present = set()
            for line in text.splitlines():
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, Mapping):
                    continue
                origin = str(row.get("origin_source_id") or "")
                source = str(row.get("source_id") or "")
                if origin in RESERVED:
                    present.add(origin)
                if source in RESERVED:
                    present.add(source)
            if present:
                hits.append({"path": str(path), "sources": sorted(present), "kind": "artifact_jsonl"})
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(PROJECT_ROOT))
    parser.add_argument(
        "--output-json",
        default="outputs/mc24_four_station_source_diversity_v1/blind_unused_audit.json",
    )
    args = parser.parse_args()
    root = Path(args.repo_root).expanduser().resolve()
    hits = collect_artifact_hits(root)
    payload = {
        "reserved_blind_sources": list(RESERVED_BLIND_SOURCES),
        "used_in_workbooks_48_63_artifacts": bool(hits),
        "hits": hits,
        "ok": not hits,
        "opened_blind_files": False,
        "test_data_accessed": False,
    }
    output = Path(args.output_json).expanduser().resolve()
    _write_json(output, payload)
    print(json.dumps(_json_ready(payload), indent=2))
    if hits:
        raise SystemExit("reserved blind sources appear in 48–63 artifacts")


if __name__ == "__main__":
    main()
