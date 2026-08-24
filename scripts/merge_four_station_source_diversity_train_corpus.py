#!/usr/bin/env python3
"""Merge the two current train physical sources with the four new ones.

The existing workbook-53 bank is not reproduced.  The assembled four-source
bank must share the same common_scan_plan.  Output is train-only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from datasets.physical_curriculum import PHYSICAL_CORPUS_SCHEMA
from scripts.prepare_multisource_multidof_iteration import _plan_signature
from scripts.run_refit_multidof_closure import _json_ready
from training.source_diversity_audit import (
    AUTHORIZED_NEW_TRAIN_SOURCES,
    AUTHORIZED_SIX_TRAIN_SOURCES,
    CURRENT_TRAIN_SOURCES,
    HISTORY_ONLY_SOURCES,
    RESERVED_BLIND_SOURCES,
)


CONTRACT_FIELDS = (
    "scan_mode",
    "q_over_p_mode",
    "station_ids",
    "reference_station_ids",
    "movable_station_ids",
    "condition_axis",
    "alignment_parameter_specs",
    "points",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"not a mapping: {path}")
    return dict(payload)


def common_scan_contract(plan: Mapping[str, Any]) -> str:
    return json.dumps(
        {field: plan.get(field) for field in CONTRACT_FIELDS},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def select_sources(corpus: Mapping[str, Any], allowed: set[str]) -> list[dict[str, Any]]:
    selected = []
    for raw in corpus.get("sources") or []:
        if not isinstance(raw, Mapping):
            raise ValueError("corpus source is invalid")
        source_id = str(raw.get("source_id") or "")
        if source_id not in allowed:
            continue
        if str(raw.get("split")) != "train":
            raise ValueError(f"source '{source_id}' is not train")
        selected.append(dict(raw))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-train-corpus", required=True)
    parser.add_argument("--new-train-corpus", required=True)
    parser.add_argument("--materialization-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    existing = _read_json(Path(args.existing_train_corpus).expanduser().resolve())
    new = _read_json(Path(args.new_train_corpus).expanduser().resolve())
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    old_sources = select_sources(existing, set(CURRENT_TRAIN_SOURCES))
    new_sources = select_sources(new, set(AUTHORIZED_NEW_TRAIN_SOURCES))
    if {item["source_id"] for item in old_sources} != set(CURRENT_TRAIN_SOURCES):
        raise ValueError("existing corpus does not contain the two current train sources")
    if {item["source_id"] for item in new_sources} != set(AUTHORIZED_NEW_TRAIN_SOURCES):
        raise ValueError("new corpus does not contain the four authorized train sources")
    old_plan = (existing.get("payload_bank") or {}).get("common_scan_plan")
    new_plan = (new.get("payload_bank") or {}).get("common_scan_plan")
    if not isinstance(old_plan, Mapping) or not isinstance(new_plan, Mapping):
        raise ValueError("both corpora must carry common_scan_plan")
    if common_scan_contract(old_plan) != common_scan_contract(new_plan):
        raise ValueError("new four-source bank does not share the workbook-53 common_scan_plan")
    sources = old_sources + new_sources
    seen = [str(item["source_id"]) for item in sources]
    if set(seen) != set(AUTHORIZED_SIX_TRAIN_SOURCES):
        raise ValueError("merged corpus is not the authorized six-source train set")
    forbidden = set(seen) & (HISTORY_ONLY_SOURCES | set(RESERVED_BLIND_SOURCES))
    if forbidden:
        raise ValueError("merged corpus contains history-only or reserved sources")
    xaods = [str(item["input_xaod"]) for item in sources]
    if len(set(xaods)) != len(xaods):
        raise ValueError("merged corpus reuses an original xAOD")
    payload = {
        "schema_version": PHYSICAL_CORPUS_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_source": str(Path(args.materialization_config).expanduser().resolve()),
        "iteration_manifest": None,
        "merged_from": {
            "existing_train_corpus": str(Path(args.existing_train_corpus).expanduser().resolve()),
            "new_train_corpus": str(Path(args.new_train_corpus).expanduser().resolve()),
        },
        "source_split_unit": "original_xAOD_file",
        "source_event_uid_convention": "source_id:run_id:event_id",
        "allowed_splits": ["train"],
        "forbidden_splits": ["validation", "test"],
        "transfer_validation_only": False,
        "reserved_blind_validation_only": False,
        "test_data_accessed": False,
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "refit_chain": existing.get("refit_chain") or new.get("refit_chain"),
        "q_over_p_mode": 0,
        "payload_bank": {
            "mode": "station_rigid_multidof_iteration",
            "alignment_iteration": (existing.get("payload_bank") or {}).get("alignment_iteration"),
            "common_scan_plan": _plan_signature(old_plan) if "scan_mode" in old_plan else old_plan,
            "common_scan_plan_verified_identical": True,
        },
        "source_split_audit": {
            "sources_by_split": {"train": sorted(seen), "validation": []},
            "source_event_counts_by_split": {
                "train": sum(len(item.get("source_event_uids") or []) for item in sources),
                "validation": 0,
            },
            "source_event_uid_overlap": {"train_validation": []},
            "strictly_disjoint": True,
        },
        "sources": sources,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "sources": sorted(seen),
                "common_scan_plan_verified_identical": True,
                "test_data_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
