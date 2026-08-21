#!/usr/bin/env python3
"""Build a train-only identity synthetic manifest from a four-station physical bank.

Each selected physical refit point is exposed as a curriculum sample whose
``synthetic_tracklets`` and ``field_candidates`` paths are the existing
mode-0 ``tracklets.root`` / ``propagations.root``.  No overlay is written,
no Athena job is submitted, and the sealed test split is refused.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from datasets.physical_curriculum import SYNTHETIC_CORPUS_SCHEMA
from datasets.root_loader import load_events
from scripts.audit_four_station_identifiability import _require_four_station_plan
from scripts.run_refit_multidof_closure import _point_map, _read_json


ASSOCIATION_PAYLOADS = (
    "iteration_00_reference",
    "iteration_00_closure_relative",
    "iteration_00_closure_relative_plus_common",
)


def _event_uids(source_id: str, tracklets: Path) -> list[str]:
    return [
        f"{source_id}:{int(event.run_id)}:{int(event.event_id)}"
        for event in load_events(tracklets, require_mc_labels=True)
    ]


def _sample_entry(
    *,
    source_id: str,
    split: str,
    point: Mapping[str, object],
    scan_root: Path,
    source_uids: Sequence[str],
) -> dict[str, object]:
    relative = point.get("relative_point_dir")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"point '{point.get('name')}' lacks relative_point_dir")
    point_root = scan_root / relative
    tracklets = point_root / "refit" / "tracklets.root"
    propagations = point_root / "refit" / "propagations.root"
    payload = point_root / "payload" / "alignment_payload.json"
    for artifact in (tracklets, propagations, payload):
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
    physical_uids = _event_uids(source_id, tracklets)
    if not set(physical_uids).issubset(set(source_uids)):
        raise ValueError(f"{source_id}/{point.get('name')} has events outside the source UID set")
    return {
        "source_id": source_id,
        "source_ids": [source_id],
        "split": split,
        "payload_id": str(point["name"]),
        "point_name": str(point["name"]),
        "direction_trial": str(point.get("direction_trial") or point["name"]),
        "magnitude_mm": float(point.get("condition_magnitude", 0.0)),
        "condition_axis": str(point.get("condition_axis", "four_station_identifiability_l2")),
        "condition_value": float(point.get("condition_value", point.get("condition_magnitude", 0.0))),
        "condition_magnitude": float(point.get("condition_magnitude", 0.0)),
        "injected_offsets_xy_mm": {},
        "injected_station_transforms": dict(point.get("injected_station_transforms") or {}),
        "alignment_parameter_values": dict(point.get("alignment_parameter_values") or {}),
        "source_event_uids": list(source_uids),
        "physical_event_uids": physical_uids,
        "physical_tracklets": str(tracklets),
        "physical_propagations": str(propagations),
        "physical_payload_manifest": str(payload),
        "synthetic_tracklets": str(tracklets),
        "field_candidates": str(propagations),
        "physical_geometry_repropagation": True,
        "identity_physical_bank": True,
        "overlay": False,
    }


def build_manifest(
    iteration_manifest: Mapping[str, Any],
    *,
    payload_ids: Sequence[str],
) -> dict[str, object]:
    plan = iteration_manifest.get("common_scan_plan")
    if not isinstance(plan, Mapping):
        raise ValueError("iteration manifest lacks common_scan_plan")
    _require_four_station_plan(plan)
    points = _point_map(plan)
    requested = tuple(str(name) for name in payload_ids)
    missing = [name for name in requested if name not in points]
    if missing:
        raise ValueError("requested payloads are absent from the frozen scan: " + ", ".join(missing))
    samples: list[dict[str, object]] = []
    for source in iteration_manifest.get("sources", ()):
        if not isinstance(source, Mapping):
            raise ValueError("iteration source entry is not a mapping")
        split = str(source.get("split", ""))
        if split == "test":
            raise ValueError("refusing to open the sealed test split")
        if split != "train":
            continue
        source_id = str(source["source_id"])
        scan_root = Path(str(source["physical_scan_root"])).expanduser().resolve()
        source_plan = _read_json(scan_root / "scan_plan.json")
        source_points = _point_map(source_plan)
        reference = source_points[requested[0]]
        source_uids = _event_uids(source_id, scan_root / str(reference["relative_point_dir"]) / "refit" / "tracklets.root")
        for name in requested:
            samples.append(
                _sample_entry(
                    source_id=source_id,
                    split=split,
                    point=source_points[name],
                    scan_root=scan_root,
                    source_uids=source_uids,
                )
            )
    if not samples:
        raise ValueError("identity association manifest has no train samples")
    return {
        "schema_version": SYNTHETIC_CORPUS_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "alignment_formulation": "four_station_v1",
        "identity_physical_bank": True,
        "overlay": False,
        "physical_geometry_repropagation": True,
        "q_over_p_mode": 0,
        "source_split_unit": "original_xAOD_file",
        "source_event_uid_convention": "source_id:run_id:event_id",
        "forbidden_splits": ["test"],
        "test_data_accessed": False,
        "payload_ids": list(requested),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--payload-id", action="append", default=None)
    args = parser.parse_args()
    requested = tuple(args.payload_id) if args.payload_id else ASSOCIATION_PAYLOADS
    manifest = build_manifest(
        _read_json(Path(args.iteration_manifest).expanduser().resolve()),
        payload_ids=requested,
    )
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(output),
                "n_samples": len(manifest["samples"]),
                "payload_ids": manifest["payload_ids"],
                "overlay": False,
                "test_data_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
