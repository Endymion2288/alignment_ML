"""Adapt a 6-DoF sensitivity-pilot iteration bank into a physical corpus manifest.

The pilot banks are produced by ``prepare_multidof_alignment_iteration.py``
plus a Condor physical scan, whose layout matches the curriculum physical
corpus (``sources/<source_id>/physical_scan/points/...``) but which never
writes the ``physical_corpus_manifest.json`` that the pooled synthetic
materializer and the frozen association backbone consume.  This adapter
derives that manifest from the on-disk scan plans with the same point
completion audit as ``build_physical_curriculum_corpus.py`` so downstream
tools cannot tell the two corpus kinds apart.  It never reruns a refit,
never alters a tracklet state, and never touches test data.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml
from datasets.physical_curriculum import PHYSICAL_CORPUS_SCHEMA
from scripts.build_physical_curriculum_corpus import (
    _physical_point_completion,
    _source_event_uids,
)

REFIT_CHAIN = (
    "persisted SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit -> "
    "NtupleDumper -> FaserActsExtrapolationTool(mode 0)"
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return payload


def _point_entry(scan_root: Path, point: Mapping[str, Any], station_ids: tuple[int, ...]) -> dict[str, Any]:
    point_root = scan_root / str(point["relative_point_dir"])
    tracklets = point_root / "refit" / "tracklets.root"
    propagations = point_root / "refit" / "propagations.root"
    payload_manifest = point_root / "payload" / "alignment_payload.json"
    content_audit = point_root / "refit" / "content_audit.json"
    completed, completion_status = _physical_point_completion(
        tracklets=tracklets,
        propagations=propagations,
        payload_manifest=payload_manifest,
        content_audit=content_audit,
        failure=point_root / "failure.json",
        station_ids=station_ids,
        expected_offsets_xy_mm=(
            dict(point["injected_offsets_xy_mm"]) if "injected_offsets_xy_mm" in point else None
        ),
        expected_station_transforms=(
            dict(point["injected_station_transforms"]) if "injected_station_transforms" in point else None
        ),
    )
    return {
        **dict(point),
        "payload_id": str(point["name"]),
        "physical_tracklets": str(tracklets),
        "physical_propagations": str(propagations),
        "physical_payload_manifest": str(payload_manifest),
        "physical_content_audit": str(content_audit),
        "completion_status": completion_status,
        "completed": completed,
    }


def build_corpus_manifest(iteration_manifest: Path) -> dict[str, Any]:
    iteration = _read_json(iteration_manifest)
    if iteration.get("test_data_accessed") is not False:
        raise ValueError("iteration manifest has an invalid test-access declaration")
    output_root = iteration_manifest.parent
    sources: list[dict[str, Any]] = []
    for source in iteration["sources"]:
        source_id = str(source["source_id"])
        scan_root = Path(str(source["physical_scan_root"])).expanduser().resolve()
        plan = _read_json(scan_root / "scan_plan.json")
        station_ids = tuple(int(value) for value in plan["station_ids"])
        points = [_point_entry(scan_root, point, station_ids) for point in plan["points"]]
        zero_point = next(
            point for point in plan["points"] if float(point["condition_magnitude"]) == 0.0
        )
        zero_tracklets = scan_root / "points" / str(zero_point["name"]) / "refit" / "tracklets.root"
        sources.append(
            {
                "source_id": source_id,
                "split": str(source["split"]),
                "input_xaod": str(source["input_xaod"]),
                "physical_scan_config": str(source["physical_scan_config"]),
                "physical_scan_root": str(scan_root),
                "source_event_uids": _source_event_uids(source_id, zero_tracklets),
                "points": points,
            }
        )
    splits_present = {str(source["split"]) for source in sources}
    allowed = [split for split in ("train", "validation", "test") if split in splits_present]
    forbidden = [split for split in ("train", "validation", "test") if split not in splits_present]
    scan_plan_common = iteration.get("common_scan_plan", {})
    return {
        "schema_version": PHYSICAL_CORPUS_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_source": str(iteration.get("iteration_template", "")),
        "output_root": str(output_root),
        "source_split_unit": "original_xAOD_file",
        "allowed_splits": allowed,
        "forbidden_splits": forbidden,
        "source_event_uid_convention": "source_id:run_id:event_id",
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "refit_chain": REFIT_CHAIN,
        "q_over_p_mode": int(iteration.get("q_over_p_mode", 0)),
        "condition_axis": scan_plan_common.get("condition_axis"),
        "payload_bank": {
            "mode": "station_rigid_multidof",
            "alignment_iteration": dict(iteration["alignment_iteration"]),
            "common_scan_plan": dict(scan_plan_common),
        },
        "iteration_manifest": str(iteration_manifest),
        "test_data_accessed": False,
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-manifest", required=True)
    args = parser.parse_args()
    iteration_manifest = Path(args.iteration_manifest).expanduser().resolve()
    manifest = build_corpus_manifest(iteration_manifest)
    incomplete = [
        f"{source['source_id']}:{point['payload_id']}"
        for source in manifest["sources"]
        for point in source["points"]
        if not point["completed"]
    ]
    if incomplete:
        raise RuntimeError("incomplete physical payloads: " + ", ".join(incomplete))
    output_path = iteration_manifest.parent / "physical_corpus_manifest.json"
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "manifest": str(output_path),
        "sources": len(manifest["sources"]),
        "points_per_source": sorted({len(source["points"]) for source in manifest["sources"]}),
        "events_per_source": sorted({len(source["source_event_uids"]) for source in manifest["sources"]}),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
