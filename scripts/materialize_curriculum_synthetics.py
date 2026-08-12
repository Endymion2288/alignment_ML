#!/usr/bin/env python3
"""Materialize source-disjoint synthetic overlays from physical refit outputs."""

from __future__ import annotations

import argparse
import json
import zlib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from datasets.physical_curriculum import PHYSICAL_CORPUS_SCHEMA, SYNTHETIC_CORPUS_SCHEMA
from datasets.root_loader import load_events
from datasets.synthetic_field_propagation import write_synthetic_field_candidate_root
from datasets.synthetic_overlay import write_synthetic_multitrack_root


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("physical corpus manifest must be a mapping")
    if payload.get("schema_version") != PHYSICAL_CORPUS_SCHEMA:
        raise ValueError("unexpected physical corpus manifest schema")
    if payload.get("physical_geometry_repropagation") is not True:
        raise ValueError("physical corpus does not certify geometry repropagation")
    return dict(payload)


def _load_synthetic_config(path: Path | None, physical: Mapping[str, Any]) -> dict[str, Any]:
    if path is None:
        config_source = Path(str(physical["config_source"])).expanduser().resolve()
        with config_source.open(encoding="utf-8") as handle:
            supplied = yaml.safe_load(handle)
        config = supplied.get("physical_curriculum_mlp", supplied)
    else:
        with path.open(encoding="utf-8") as handle:
            supplied = yaml.safe_load(handle)
        config = supplied.get("physical_curriculum_mlp", supplied)
    if not isinstance(config, Mapping):
        raise ValueError("synthetic configuration must be a mapping")
    synthetic = config.get("synthetic_multitrack")
    field = config.get("field_candidate_export")
    if not isinstance(synthetic, Mapping) or not isinstance(field, Mapping):
        raise ValueError("configuration must define synthetic_multitrack and field_candidate_export")
    return {"synthetic_multitrack": dict(synthetic), "field_candidate_export": dict(field)}


def _stable_seed(base: int, source_id: str, payload_id: str) -> int:
    return int((base + zlib.crc32(f"{source_id}/{payload_id}".encode("utf-8"))) % (2**31 - 1))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    physical_path = Path(args.physical_manifest).expanduser().resolve()
    physical = _load_json(physical_path)
    config = _load_synthetic_config(
        None if args.config is None else Path(args.config).expanduser().resolve(), physical
    )
    synthetic = config["synthetic_multitrack"]
    field = config["field_candidate_export"]
    if int(field.get("q_over_p_mode", 0)) != 0:
        raise ValueError("physical synthetic candidate export must use q_over_p_mode=0")
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, object]] = []
    source_ids_by_split: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    for source_index, source in enumerate(physical["sources"]):
        if not isinstance(source, Mapping):
            raise ValueError("invalid source entry in physical corpus")
        source_id = str(source["source_id"])
        split = str(source["split"])
        if split not in source_ids_by_split:
            raise ValueError(f"invalid source split '{split}'")
        source_ids_by_split[split].add(source_id)
        source_uids = list(source.get("source_event_uids", []))
        if not source_uids:
            raise ValueError(f"physical source '{source_id}' has no refitted source-event provenance")
        for point_index, point in enumerate(source["points"]):
            if not point.get("completed", False):
                raise RuntimeError(f"physical point is incomplete: {source_id}/{point['name']}")
            payload_id = str(point["payload_id"])
            physical_tracklets = Path(str(point["physical_tracklets"])).expanduser().resolve()
            physical_propagations = Path(str(point["physical_propagations"])).expanduser().resolve()
            physical_payload = Path(str(point["physical_payload_manifest"])).expanduser().resolve()
            if not (physical_tracklets.is_file() and physical_propagations.is_file() and physical_payload.is_file()):
                raise FileNotFoundError(f"physical refit assets are missing for {source_id}/{payload_id}")
            physical_events = load_events(physical_tracklets, require_mc_labels=True)
            point_uids = [f"{source_id}:{event.run_id}:{event.event_id}" for event in physical_events]
            if not set(point_uids).issubset(set(source_uids)):
                raise RuntimeError(
                    f"physical payload {source_id}/{payload_id} contains an event outside its source split"
                )
            sample_root = output_root / "samples" / source_id / payload_id
            tracklets = sample_root / "synthetic_tracklets.root"
            candidates = sample_root / "field_candidates.root"
            seed = _stable_seed(int(synthetic["seed"]), source_id, payload_id)
            run_id = int(synthetic["synthetic_run_id_base"]) + source_index * 1000 + point_index
            if not (args.resume and tracklets.is_file()):
                overlay = write_synthetic_multitrack_root(
                    physical_events,
                    tracklets,
                    output_events=int(synthetic["events_per_payload"]),
                    tracks_per_event=int(synthetic["tracks_per_event"]),
                    station_ids=tuple(int(value) for value in synthetic["stations"]),
                    missing_tracklet_probability=float(synthetic["missing_tracklet_probability"]),
                    fake_mean_per_station=float(synthetic["fake_mean_per_station"]),
                    random_easy_fake_mean_per_station=(
                        None
                        if "random_easy_fake_mean_per_station" not in synthetic
                        else float(synthetic["random_easy_fake_mean_per_station"])
                    ),
                    random_easy_min_chi2=(
                        None
                        if synthetic.get("random_easy_min_chi2") is None
                        else float(synthetic["random_easy_min_chi2"])
                    ),
                    hard_negative_mean_per_target_station=float(
                        synthetic.get("hard_negative_mean_per_target_station", 0.0)
                    ),
                    hard_negative_chi2_min=float(synthetic.get("hard_negative_chi2_min", 1.0)),
                    hard_negative_chi2_max=float(synthetic.get("hard_negative_chi2_max", 100.0)),
                    hard_negative_max_trials=int(synthetic.get("hard_negative_max_trials", 256)),
                    physical_propagations=physical_propagations,
                    target_z_tolerance_mm=float(field["target_z_tolerance_mm"]),
                    allow_duplicate_source_tracks=bool(
                        synthetic.get("allow_duplicate_source_tracks", False)
                    ),
                    minimum_truth_match_fraction=float(synthetic["minimum_truth_match_fraction"]),
                    seed=seed,
                    synthetic_run_id=run_id,
                )
                _write_json(sample_root / "overlay_summary.json", asdict(overlay))
            if not (args.resume and candidates.is_file()):
                candidate_summary = write_synthetic_field_candidate_root(
                    synthetic_tracklets=tracklets,
                    source_propagations=physical_propagations,
                    destination=candidates,
                    q_over_p_mode=int(field["q_over_p_mode"]),
                    target_z_tolerance_mm=float(field["target_z_tolerance_mm"]),
                )
                _write_json(sample_root / "field_candidate_summary.json", asdict(candidate_summary))
                if candidate_summary.target_z_mismatch:
                    raise RuntimeError(f"target-z mismatch in {source_id}/{payload_id}")
            _write_json(
                sample_root / "resolved_config.json",
                {
                    "source_id": source_id,
                    "source_ids": [source_id],
                    "split": split,
                    "payload_id": payload_id,
                    "physical_tracklets": str(physical_tracklets),
                    "physical_propagations": str(physical_propagations),
                    "physical_payload_manifest": str(physical_payload),
                    "synthetic_tracklets": str(tracklets),
                    "field_candidates": str(candidates),
                    "synthetic_seed": seed,
                    "synthetic_run_id": run_id,
                    "physical_geometry_repropagation": True,
                    "physical_event_uids": point_uids,
                },
            )
            samples.append(
                {
                    "source_id": source_id,
                    "split": split,
                    "payload_id": payload_id,
                    "point_name": str(point["name"]),
                    "magnitude_mm": float(point["magnitude_mm"]),
                    "direction_trial": str(point["direction_trial"]),
                    "injected_offsets_xy_mm": dict(point["injected_offsets_xy_mm"]),
                    "source_event_uids": source_uids,
                    "physical_event_uids": point_uids,
                    "physical_tracklets": str(physical_tracklets),
                    "physical_propagations": str(physical_propagations),
                    "physical_payload_manifest": str(physical_payload),
                    "synthetic_tracklets": str(tracklets),
                    "field_candidates": str(candidates),
                    "physical_geometry_repropagation": True,
                }
            )
    for first_split, first_sources in source_ids_by_split.items():
        for second_split, second_sources in source_ids_by_split.items():
            if first_split < second_split and first_sources.intersection(second_sources):
                raise RuntimeError("source split overlap while materializing synthetic corpus")
    manifest = {
        "schema_version": SYNTHETIC_CORPUS_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "physical_corpus_manifest": str(physical_path),
        "physical_geometry_repropagation": True,
        "source_split_unit": "original_xAOD_file",
        "source_event_uid_convention": physical["source_event_uid_convention"],
        "q_over_p_mode": int(field["q_over_p_mode"]),
        "synthetic_multitrack": synthetic,
        "field_candidate_export": field,
        "samples": samples,
    }
    manifest_path = output_root / "synthetic_corpus_manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "samples": len(samples)}, indent=2))


if __name__ == "__main__":
    main()
