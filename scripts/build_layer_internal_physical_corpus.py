"""Build a physical corpus for IFT layer-internal route-selected closure.

The Jacobian finite-difference bank and the linear held-out bank live in
different output roots.  This adapter lists only the frozen-station relative-dx
points needed for unknown-association closure: the already-screened layer dx
probes plus the new outer-antisymmetric held-out.  It never reruns a refit and
never opens test data.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from datasets.physical_curriculum import PHYSICAL_CORPUS_SCHEMA
from scripts.build_physical_curriculum_corpus import (
    _physical_point_completion,
    _source_event_uids,
)


REFIT_CHAIN = (
    "persisted SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit -> "
    "NtupleDumper -> FaserActsExtrapolationTool(mode 0)"
)
LAYER_DX_JACOBIAN_POINTS = (
    "iteration_00_reference",
    "iteration_00_fd_ift_layer0_dx_mm_p",
    "iteration_00_fd_ift_layer0_dx_mm_m",
    "iteration_00_fd_ift_layer1_dx_mm_p",
    "iteration_00_fd_ift_layer1_dx_mm_m",
    "iteration_00_fd_ift_layer2_dx_mm_p",
    "iteration_00_fd_ift_layer2_dx_mm_m",
)
DEFAULT_OBSERVED_POINT = "iteration_00_closure_relative_dx"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return payload


def _point_map(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    points = plan.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("scan plan has no points")
    result: dict[str, dict[str, Any]] = {}
    for item in points:
        if not isinstance(item, Mapping):
            raise ValueError("scan plan point is not a mapping")
        name = str(item.get("name", ""))
        if not name or name in result:
            raise ValueError(f"scan plan has invalid/duplicate point name {name!r}")
        result[name] = dict(item)
    return result


def _point_entry(scan_root: Path, point: Mapping[str, Any], station_ids: tuple[int, ...]) -> dict[str, Any]:
    point_root = scan_root / str(point["relative_point_dir"])
    tracklets = point_root / "refit" / "tracklets.root"
    propagations = point_root / "refit" / "propagations.root"
    payload_manifest = point_root / "payload" / "alignment_payload.json"
    content_audit = point_root / "refit" / "content_audit.json"
    layer_transforms = point.get("injected_layer_transforms")
    completed, completion_status = _physical_point_completion(
        tracklets=tracklets,
        propagations=propagations,
        payload_manifest=payload_manifest,
        content_audit=content_audit,
        failure=point_root / "failure.json",
        station_ids=station_ids,
        expected_offsets_xy_mm=None,
        expected_station_transforms=(
            dict(point["injected_station_transforms"]) if "injected_station_transforms" in point else None
        ),
        expected_layer_transforms=dict(layer_transforms) if isinstance(layer_transforms, Mapping) else None,
    )
    return {
        **dict(point),
        "payload_id": str(point["name"]),
        "physical_scan_root": str(scan_root),
        "physical_tracklets": str(tracklets),
        "physical_propagations": str(propagations),
        "physical_payload_manifest": str(payload_manifest),
        "physical_content_audit": str(content_audit),
        "completion_status": completion_status,
        "completed": completed,
    }


def _source_index(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("iteration manifest has no sources")
    result: dict[str, dict[str, Any]] = {}
    for item in sources:
        if not isinstance(item, Mapping):
            raise ValueError("iteration manifest source is not a mapping")
        source_id = str(item.get("source_id", ""))
        if not source_id or source_id in result:
            raise ValueError(f"invalid/duplicate source_id {source_id!r}")
        if str(item.get("split", "")) == "test":
            raise ValueError(f"source '{source_id}' is a sealed test source")
        result[source_id] = dict(item)
    return result


def build_corpus_manifest(
    *,
    jacobian_manifest: Path,
    observed_manifest: Path,
    jacobian_points: Sequence[str] = LAYER_DX_JACOBIAN_POINTS,
    observed_point: str = DEFAULT_OBSERVED_POINT,
) -> dict[str, Any]:
    jacobian = _read_json(jacobian_manifest)
    observed = _read_json(observed_manifest)
    if jacobian.get("test_data_accessed") is not False or observed.get("test_data_accessed") is not False:
        raise ValueError("iteration manifest has an invalid test-access declaration")
    if int(jacobian.get("q_over_p_mode", -1)) != 0 or int(observed.get("q_over_p_mode", -1)) != 0:
        raise ValueError("layer-internal corpus requires mode-0 physical banks")
    jacobian_sources = _source_index(jacobian)
    observed_sources = _source_index(observed)
    missing = sorted(set(observed_sources) - set(jacobian_sources))
    if missing:
        raise ValueError("observed sources missing from jacobian bank: " + ", ".join(missing))

    sources: list[dict[str, Any]] = []
    for source_id, observed_entry in observed_sources.items():
        jacobian_entry = jacobian_sources[source_id]
        jacobian_root = Path(str(jacobian_entry["physical_scan_root"])).expanduser().resolve()
        observed_root = Path(str(observed_entry["physical_scan_root"])).expanduser().resolve()
        jacobian_plan = _read_json(jacobian_root / "scan_plan.json")
        observed_plan = _read_json(observed_root / "scan_plan.json")
        station_ids = tuple(int(value) for value in jacobian_plan["station_ids"])
        jacobian_points_by_name = _point_map(jacobian_plan)
        observed_points_by_name = _point_map(observed_plan)
        selected: list[dict[str, Any]] = []
        for name in jacobian_points:
            point = jacobian_points_by_name.get(name)
            if point is None:
                raise ValueError(f"jacobian bank for '{source_id}' lacks '{name}'")
            selected.append(_point_entry(jacobian_root, point, station_ids))
        held_out = observed_points_by_name.get(observed_point)
        if held_out is None:
            raise ValueError(f"observed bank for '{source_id}' lacks '{observed_point}'")
        selected.append(_point_entry(observed_root, held_out, station_ids))
        reference = next(point for point in selected if point["name"] == "iteration_00_reference")
        sources.append(
            {
                "source_id": source_id,
                "split": str(observed_entry["split"]),
                "input_xaod": str(observed_entry["input_xaod"]),
                "physical_scan_config": str(jacobian_entry["physical_scan_config"]),
                "physical_scan_root": str(jacobian_root),
                "observed_scan_root": str(observed_root),
                "source_event_uids": _source_event_uids(source_id, Path(str(reference["physical_tracklets"]))),
                "points": selected,
            }
        )
    splits_present = {str(source["split"]) for source in sources}
    allowed = [split for split in ("train", "validation", "test") if split in splits_present]
    forbidden = [split for split in ("train", "validation", "test") if split not in splits_present]
    return {
        "schema_version": PHYSICAL_CORPUS_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_source": str(observed.get("iteration_template", "")),
        "output_root": str(observed_manifest.parent),
        "source_split_unit": "original_xAOD_file",
        "allowed_splits": allowed,
        "forbidden_splits": forbidden,
        "source_event_uid_convention": "source_id:run_id:event_id",
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "refit_chain": REFIT_CHAIN,
        "q_over_p_mode": 0,
        "condition_axis": "ift_station_layer_hierarchy_l2",
        "payload_bank": {
            "mode": "ift_layer_hierarchy",
            "held_out_only_observed": True,
            "station_5dof_frozen": True,
            "jacobian_manifest": str(jacobian_manifest),
            "observed_manifest": str(observed_manifest),
            "jacobian_points": list(jacobian_points),
            "observed_point": observed_point,
        },
        "test_data_accessed": False,
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jacobian-manifest", required=True)
    parser.add_argument("--observed-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--observed-point", default=DEFAULT_OBSERVED_POINT)
    args = parser.parse_args()
    manifest = build_corpus_manifest(
        jacobian_manifest=Path(args.jacobian_manifest).expanduser().resolve(),
        observed_manifest=Path(args.observed_manifest).expanduser().resolve(),
        observed_point=str(args.observed_point),
    )
    incomplete = [
        f"{source['source_id']}:{point['payload_id']}"
        for source in manifest["sources"]
        for point in source["points"]
        if not point["completed"]
    ]
    if incomplete:
        raise RuntimeError("incomplete physical payloads: " + ", ".join(incomplete))
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(output_path),
                "sources": len(manifest["sources"]),
                "points_per_source": sorted({len(source["points"]) for source in manifest["sources"]}),
                "events_per_source": sorted({len(source["source_event_uids"]) for source in manifest["sources"]}),
                "test_data_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
