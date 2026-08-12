#!/usr/bin/env python3
"""Assemble a checked source-disjoint corpus manifest from physical refit banks.

This is intentionally a manifest-only operation: it reuses completed payloads
only after checking that every configured original xAOD source appears exactly
once, with its declared split and complete cluster->segment->Acts assets.  No
coordinates, residuals, or geometry conditions are altered or regenerated.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from datasets.physical_curriculum import PHYSICAL_CORPUS_SCHEMA
from scripts.config_loader import load_yaml_with_base


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != PHYSICAL_CORPUS_SCHEMA:
        raise ValueError(f"not a physical curriculum manifest: {path}")
    if payload.get("physical_geometry_repropagation") is not True:
        raise ValueError(f"manifest does not certify physical geometry repropagation: {path}")
    return dict(payload)


def _configured_sources(config_path: Path) -> tuple[dict[str, tuple[str, str]], dict[str, Any]]:
    supplied = load_yaml_with_base(config_path)
    config = supplied.get("physical_curriculum_mlp", supplied)
    if not isinstance(config, Mapping):
        raise ValueError("physical_curriculum_mlp must be a mapping")
    expected: dict[str, tuple[str, str]] = {}
    for raw in config["sources"]:
        source_id = str(raw["id"])
        split = str(raw["split"])
        input_xaod = str(Path(str(raw["input_xaod"])).expanduser().resolve())
        if source_id in expected:
            raise ValueError(f"duplicate configured source '{source_id}'")
        expected[source_id] = (split, input_xaod)
    return expected, dict(config)


def _validate_entry(entry: Mapping[str, Any], expected: tuple[str, str]) -> None:
    if str(entry.get("split")) != expected[0]:
        raise ValueError(f"source '{entry.get('source_id')}' has a split mismatch")
    if str(Path(str(entry.get("input_xaod"))).expanduser().resolve()) != expected[1]:
        raise ValueError(f"source '{entry.get('source_id')}' has an xAOD path mismatch")
    uids = entry.get("source_event_uids")
    if not isinstance(uids, list) or not uids:
        raise ValueError(f"source '{entry.get('source_id')}' has no event provenance")
    points = entry.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError(f"source '{entry.get('source_id')}' has no payload points")
    for point in points:
        if not isinstance(point, Mapping) or point.get("completed") is not True:
            raise ValueError(f"source '{entry.get('source_id')}' has an incomplete physical payload")
        for field in ("physical_tracklets", "physical_propagations", "physical_payload_manifest"):
            asset = Path(str(point.get(field, ""))).expanduser().resolve()
            if not asset.is_file():
                raise FileNotFoundError(f"physical asset is missing for {entry.get('source_id')}: {asset}")


def _is_complete_entry(entry: Mapping[str, Any]) -> bool:
    """Ignore manifest placeholders from a partially generated component bank."""
    points = entry.get("points")
    return bool(
        isinstance(points, list)
        and points
        and all(isinstance(point, Mapping) and point.get("completed") is True for point in points)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-manifest", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    expected, config = _configured_sources(config_path)
    entries: dict[str, dict[str, Any]] = {}
    components: list[str] = []
    q_over_p_mode: int | None = None
    refit_chain: str | None = None
    for raw_path in args.input_manifest:
        path = Path(raw_path).expanduser().resolve()
        manifest = _load_json(path)
        components.append(str(path))
        mode = int(manifest.get("q_over_p_mode", -1))
        if q_over_p_mode is None:
            q_over_p_mode = mode
        elif q_over_p_mode != mode:
            raise ValueError("component manifests disagree on q_over_p_mode")
        chain = str(manifest.get("refit_chain", ""))
        if refit_chain is None:
            refit_chain = chain
        elif refit_chain != chain:
            raise ValueError("component manifests disagree on refit chain")
        for entry in manifest.get("sources", []):
            if not isinstance(entry, Mapping):
                raise ValueError(f"invalid source entry in {path}")
            source_id = str(entry.get("source_id", ""))
            if source_id not in expected:
                continue
            # Corpus builders retain all configured sources in a progress
            # manifest.  An incomplete placeholder is not evidence that a
            # source has been reused; the final missing-source check below
            # still fails if no component supplies a completed bank.
            if not _is_complete_entry(entry):
                continue
            if source_id in entries:
                raise ValueError(f"source '{source_id}' occurs in more than one component manifest")
            _validate_entry(entry, expected[source_id])
            entries[source_id] = dict(entry)
    missing = sorted(set(expected) - set(entries))
    if missing:
        raise ValueError("configured physical sources are missing: " + ", ".join(missing))
    if q_over_p_mode != 0:
        raise ValueError("V1 assembled corpus must use q_over_p_mode=0")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    split_sources = {split: [] for split in ("train", "validation", "test")}
    source_event_uids: dict[str, set[str]] = {split: set() for split in split_sources}
    for source_id, (split, _) in expected.items():
        entry = entries[source_id]
        split_sources[split].append(source_id)
        source_event_uids[split].update(str(value) for value in entry["source_event_uids"])
    overlaps = {
        f"{left}_{right}": sorted(source_event_uids[left].intersection(source_event_uids[right]))
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    if any(overlaps.values()):
        raise ValueError("source event UID leakage across assembled splits")
    # Reused and newly produced source banks must represent the same declared
    # geometry bank within a split.  Compare the actual payload descriptors,
    # not merely the YAML seed that was intended to generate them.
    payload_signatures: dict[str, dict[str, tuple[float, str, str]]] = {}
    for source_id, (split, _) in expected.items():
        current = {
            str(point["payload_id"]): (
                float(point["magnitude_mm"]),
                str(point["direction_trial"]),
                json.dumps(point["injected_offsets_xy_mm"], sort_keys=True),
            )
            for point in entries[source_id]["points"]
        }
        reference = payload_signatures.setdefault(split, current)
        if current != reference:
            raise ValueError(
                f"physical payload bank differs within split '{split}' for source '{source_id}'"
            )
    payload = {
        "schema_version": PHYSICAL_CORPUS_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_source": str(config_path),
        "component_manifests": components,
        "source_split_unit": "original_xAOD_file",
        "source_event_uid_convention": "source_id:run_id:event_id",
        "source_split_audit": {
            "sources_by_split": {key: sorted(value) for key, value in split_sources.items()},
            "source_event_counts_by_split": {key: len(value) for key, value in source_event_uids.items()},
            "source_event_uid_overlap": overlaps,
            "strictly_disjoint": True,
        },
        "physical_geometry_repropagation": True,
        "refit_chain": refit_chain,
        "q_over_p_mode": q_over_p_mode,
        "payload_bank": dict(config["payload_bank"]),
        "sources": [entries[source_id] for source_id in expected],
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "sources_by_split": {key: len(value) for key, value in split_sources.items()},
                "source_event_counts_by_split": {key: len(value) for key, value in source_event_uids.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
