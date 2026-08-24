#!/usr/bin/env python3
"""Assemble a completed multi-source alignment iteration into a pooled manifest.

This is a read-only provenance operation.  It checks every source-specific
real payload/refit/Acts output against the frozen iteration plan and writes a
standard physical-corpus manifest for the existing pooled synthetic overlay
tool.  It never shifts coordinates, alters residuals, or opens test sources.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from datasets.physical_curriculum import PHYSICAL_CORPUS_SCHEMA
from datasets.root_loader import load_events
from scripts.build_physical_curriculum_corpus import _physical_point_completion
from scripts.prepare_multisource_multidof_iteration import iteration_split_mode
from scripts.run_refit_multidof_closure import _read_json


ITERATION_SCHEMA = "faser-multisource-physical-alignment-iteration-v1"


def _load_iteration(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != ITERATION_SCHEMA:
        raise ValueError(f"not a multi-source iteration manifest: {path}")
    if payload.get("physical_geometry_repropagation") is not True or payload.get("coordinate_surrogate") is not False:
        raise ValueError("iteration manifest does not certify physical geometry repropagation")
    if int(payload.get("q_over_p_mode", -1)) != 0:
        raise ValueError("iteration manifest is not mode-0")
    iteration_split_mode(payload, label="iteration manifest")
    if payload.get("test_data_accessed") is not False:
        raise ValueError("iteration manifest test access declaration is invalid")
    if not isinstance(payload.get("sources"), list) or not payload["sources"]:
        raise ValueError("iteration manifest has no source records")
    return payload


def _source_event_uids(source_id: str, tracklets: Path) -> list[str]:
    events = load_events(tracklets, require_mc_labels=True)
    if not events:
        raise ValueError(f"source '{source_id}' has no events in its reference refit")
    return [f"{source_id}:{event.run_id}:{event.event_id}" for event in events]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-manifest", required=True)
    parser.add_argument(
        "--materialization-config",
        required=True,
        help="Train/validation-only YAML providing the synthetic overlay policy.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    iteration_path = Path(args.iteration_manifest).expanduser().resolve()
    iteration = _load_iteration(iteration_path)
    config = Path(args.materialization_config).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    sources: list[dict[str, object]] = []
    all_uids: dict[str, set[str]] = {"train": set(), "validation": set()}
    xAOD_paths: dict[str, str] = {}
    expected_plan = iteration.get("common_scan_plan")
    if not isinstance(expected_plan, Mapping):
        raise ValueError("iteration manifest lacks common_scan_plan")
    expected_points = expected_plan.get("points")
    if not isinstance(expected_points, list) or not expected_points:
        raise ValueError("iteration manifest has no common physical points")
    expected_names = [str(point.get("name", "")) for point in expected_points if isinstance(point, Mapping)]
    if len(expected_names) != len(expected_points) or not all(expected_names) or len(set(expected_names)) != len(expected_names):
        raise ValueError("iteration common plan has invalid point names")
    contract_fields = (
        "scan_mode",
        "q_over_p_mode",
        "station_ids",
        "reference_station_ids",
        "movable_station_ids",
        "condition_axis",
        "alignment_parameter_specs",
        "points",
    )
    expected_contract = json.dumps(
        {field: expected_plan.get(field) for field in contract_fields},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    for raw_source in iteration["sources"]:
        if not isinstance(raw_source, Mapping):
            raise ValueError("iteration manifest has invalid source record")
        source = dict(raw_source)
        source_id = str(source.get("source_id", ""))
        split = str(source.get("split", ""))
        if not source_id or split not in all_uids:
            raise ValueError("iteration source ID/split is invalid")
        xAOD = str(Path(str(source.get("input_xaod", ""))).expanduser().resolve())
        if xAOD in xAOD_paths:
            raise ValueError(f"original xAOD is reused by '{source_id}' and '{xAOD_paths[xAOD]}'")
        xAOD_paths[xAOD] = source_id
        root = Path(str(source.get("physical_scan_root", ""))).expanduser().resolve()
        plan = _read_json(root / "scan_plan.json")
        if plan.get("scan_mode") != "station_rigid_multidof" or int(plan.get("q_over_p_mode", -1)) != 0:
            raise ValueError(f"source '{source_id}' does not contain a mode-0 multi-DoF scan")
        observed_contract = json.dumps(
            {field: plan.get(field) for field in contract_fields},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if observed_contract != expected_contract:
            raise ValueError(f"source '{source_id}' does not share the frozen multi-DoF physical plan")
        points = plan.get("points")
        if not isinstance(points, list) or [str(point.get("name", "")) for point in points if isinstance(point, Mapping)] != expected_names:
            raise ValueError(f"source '{source_id}' does not share the frozen point ordering")
        station_ids = plan.get("station_ids")
        if not isinstance(station_ids, list):
            raise ValueError(f"source '{source_id}' scan lacks station IDs")
        entries: list[dict[str, object]] = []
        for raw_point in points:
            if not isinstance(raw_point, Mapping):
                raise ValueError("source scan point is invalid")
            point = dict(raw_point)
            relative = str(point.get("relative_point_dir", ""))
            point_root = root / relative
            transforms = point.get("injected_station_transforms")
            if not isinstance(transforms, Mapping):
                raise ValueError(f"source '{source_id}' point '{point.get('name')}' lacks station transforms")
            completed, status = _physical_point_completion(
                tracklets=point_root / "refit" / "tracklets.root",
                propagations=point_root / "refit" / "propagations.root",
                payload_manifest=point_root / "payload" / "alignment_payload.json",
                content_audit=point_root / "refit" / "content_audit.json",
                failure=point_root / "failure.json",
                station_ids=tuple(int(station) for station in station_ids),
                expected_offsets_xy_mm=None,
                expected_station_transforms=dict(transforms),
            )
            if not completed:
                raise RuntimeError(
                    f"source '{source_id}' point '{point.get('name')}' is not physically complete: {status}"
                )
            entries.append(
                {
                    **point,
                    "payload_id": str(point["name"]),
                    "physical_tracklets": str(point_root / "refit" / "tracklets.root"),
                    "physical_propagations": str(point_root / "refit" / "propagations.root"),
                    "physical_payload_manifest": str(point_root / "payload" / "alignment_payload.json"),
                    "physical_content_audit": str(point_root / "refit" / "content_audit.json"),
                    "completed": True,
                    "completion_status": "accepted",
                }
            )
        reference_name = str(iteration["alignment_iteration"]["reference_point"])
        reference = next((item for item in entries if item["payload_id"] == reference_name), None)
        if reference is None:
            raise ValueError(f"source '{source_id}' has no iteration reference point")
        uids = _source_event_uids(source_id, Path(str(reference["physical_tracklets"])))
        overlap = all_uids[split].intersection(uids)
        if overlap:
            raise ValueError(f"source-event provenance overlaps within {split}: {sorted(overlap)[:3]}")
        all_uids[split].update(uids)
        sources.append(
            {
                "source_id": source_id,
                "split": split,
                "input_xaod": xAOD,
                "physical_scan_config": str(source["physical_scan_config"]),
                "physical_scan_root": str(root),
                "source_event_uids": uids,
                "points": entries,
            }
        )
    cross_overlap = all_uids["train"].intersection(all_uids["validation"])
    if cross_overlap:
        raise ValueError("train/validation source-event provenance overlaps")
    payload = {
        "schema_version": PHYSICAL_CORPUS_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_source": str(config),
        "iteration_manifest": str(iteration_path),
        "source_split_unit": "original_xAOD_file",
        "source_event_uid_convention": "source_id:run_id:event_id",
        "allowed_splits": list(iteration.get("allowed_splits") or ["train", "validation"]),
        "forbidden_splits": list(iteration.get("forbidden_splits") or ["test"]),
        "transfer_validation_only": bool(iteration.get("transfer_validation_only")),
        "reserved_blind_validation_only": bool(iteration.get("reserved_blind_validation_only")),
        "test_data_accessed": False,
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "refit_chain": iteration["refit_chain"],
        "q_over_p_mode": 0,
        "payload_bank": {
            "mode": "station_rigid_multidof_iteration",
            "alignment_iteration": iteration["alignment_iteration"],
            "common_scan_plan": expected_plan,
        },
        "source_split_audit": {
            "sources_by_split": {
                split: sorted(str(item["source_id"]) for item in sources if item["split"] == split)
                for split in ("train", "validation")
            },
            "source_event_counts_by_split": {split: len(values) for split, values in all_uids.items()},
            "source_event_uid_overlap": {"train_validation": []},
            "strictly_disjoint": True,
        },
        "sources": sources,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "sources_by_split": payload["source_split_audit"]["sources_by_split"],
                "source_event_counts_by_split": payload["source_split_audit"]["source_event_counts_by_split"],
                "test_data_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
