#!/usr/bin/env python3
"""Materialize pooled, source-disjoint physical synthetic overlays.

The original MC24 chunks reuse run/event numbers.  For each split and each
identical real conditions payload, this command namespaces those provenance
IDs before pooling physical tracklets and exact mode-0 Acts records.  Pooling
never crosses train/validation/test and never alters a tracklet state,
covariance, residual, prediction, or alignment payload.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import zlib

from datasets.physical_curriculum import PHYSICAL_CORPUS_SCHEMA, SYNTHETIC_CORPUS_SCHEMA
from datasets.pooled_physical import (
    merge_propagation_records,
    namespace_physical_source,
    write_pooled_propagations_root,
    write_pooled_tracklets_root,
)
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events
from datasets.synthetic_field_propagation import write_synthetic_field_candidate_root
from datasets.synthetic_overlay import write_synthetic_multitrack_root
from scripts.config_loader import load_yaml_with_base


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != PHYSICAL_CORPUS_SCHEMA:
        raise ValueError("unexpected physical corpus manifest")
    if payload.get("physical_geometry_repropagation") is not True:
        raise ValueError("physical corpus does not certify geometry repropagation")
    return dict(payload)


def _load_config(path: Path | None, physical: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    source = Path(str(physical["config_source"])).expanduser().resolve() if path is None else path
    supplied = load_yaml_with_base(source)
    root = supplied.get("physical_curriculum_mlp", supplied)
    if not isinstance(root, Mapping):
        raise ValueError("physical_curriculum_mlp must be a mapping")
    synthetic = root.get("synthetic_multitrack")
    field = root.get("field_candidate_export")
    if not isinstance(synthetic, Mapping) or not isinstance(field, Mapping):
        raise ValueError("configuration must define synthetic_multitrack and field_candidate_export")
    return {"synthetic": dict(synthetic), "field": dict(field)}


def _stable_seed(
    base: int,
    split: str,
    payload_id: str,
    magnitude_mm: float,
    scope: str,
) -> int:
    """Derive a deterministic overlay seed without touching physical states.

    The legacy ``payload`` scope gives every geometry payload an independent
    synthetic overlay.  A direction scan instead uses
    ``magnitude_shared_across_direction_trials`` so all directions at one
    magnitude begin with the same source-track/fake RNG stream.  Any later
    difference is then attributable to a real refit/Acts response or to a
    geometry-dependent physical-fake acceptance, not a new arbitrary seed.
    """
    if scope == "payload":
        identity = f"pooled/{split}/{payload_id}"
    elif scope == "magnitude_shared_across_direction_trials":
        identity = f"pooled/{split}/magnitude/{float(magnitude_mm):.17g}"
    else:
        raise ValueError(
            "synthetic_multitrack.overlay_seed_scope must be 'payload' or "
            "'magnitude_shared_across_direction_trials'"
        )
    return int((base + zlib.crc32(identity.encode("utf-8"))) % (2**31 - 1))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _payload_signature(point: Mapping[str, Any]) -> tuple[float, str, str]:
    return (
        float(point["magnitude_mm"]),
        str(point["direction_trial"]),
        json.dumps(point["injected_offsets_xy_mm"], sort_keys=True),
    )


def _groups(
    physical: Mapping[str, Any],
    selected_splits: set[str] | None = None,
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, set[str]]]:
    """Group complete source points, optionally for one sealed split only.

    A partial physical corpus is useful for a frozen test scan while the
    train/validation sources are deliberately not regenerated.  Filtering is
    performed before completeness checks, so an unprocessed source from a
    different split can never enter the resulting synthetic manifest.
    """
    if selected_splits is not None:
        invalid = selected_splits - {"train", "validation", "test"}
        if invalid:
            raise ValueError("unknown selected split(s): " + ", ".join(sorted(invalid)))
        if not selected_splits:
            raise ValueError("selected_splits cannot be empty")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    sources_by_split: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    for source in physical["sources"]:
        if not isinstance(source, Mapping):
            raise ValueError("physical corpus has an invalid source entry")
        source_id = str(source["source_id"])
        split = str(source["split"])
        if split not in sources_by_split:
            raise ValueError(f"invalid split '{split}'")
        if selected_splits is not None and split not in selected_splits:
            continue
        sources_by_split[split].add(source_id)
        for point in source["points"]:
            if not isinstance(point, Mapping) or point.get("completed") is not True:
                raise ValueError(f"physical point is incomplete: {source_id}")
            payload_id = str(point["payload_id"])
            groups.setdefault((split, payload_id), []).append(
                {"source": dict(source), "point": dict(point)}
            )
    for (split, payload_id), members in groups.items():
        actual = {str(member["source"]["source_id"]) for member in members}
        if actual != sources_by_split[split]:
            raise ValueError(
                f"pooled payload {split}/{payload_id} does not contain every split source"
            )
        signatures = {_payload_signature(member["point"]) for member in members}
        if len(signatures) != 1:
            raise ValueError(f"split '{split}' has inconsistent real payloads for {payload_id}")
    return groups, sources_by_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "validation", "test"),
        default=None,
        help="Materialize only this split; repeat only for an explicitly selected subset",
    )
    args = parser.parse_args()
    physical_path = Path(args.physical_manifest).expanduser().resolve()
    physical = _load_json(physical_path)
    config = _load_config(
        None if args.config is None else Path(args.config).expanduser().resolve(), physical
    )
    synthetic = config["synthetic"]
    field = config["field"]
    if int(field.get("q_over_p_mode", 0)) != 0:
        raise ValueError("pooled materialization supports only mode-0 propagation")
    if str(synthetic.get("source_pooling", "within_split_same_payload")) != "within_split_same_payload":
        raise ValueError("pooled materializer requires source_pooling=within_split_same_payload")
    if not bool(synthetic.get("scale_events_by_pool_sources", True)):
        raise ValueError("pooled materializer requires scale_events_by_pool_sources=true")
    overlay_seed_scope = str(synthetic.get("overlay_seed_scope", "payload"))
    if overlay_seed_scope not in {"payload", "magnitude_shared_across_direction_trials"}:
        raise ValueError(
            "synthetic_multitrack.overlay_seed_scope must be 'payload' or "
            "'magnitude_shared_across_direction_trials'"
        )
    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise FileExistsError("refusing to overwrite a non-empty pooled synthetic output")
    output_root.mkdir(parents=True, exist_ok=True)
    selected_splits = None if args.split is None else set(args.split)
    groups, sources_by_split = _groups(physical, selected_splits=selected_splits)
    if not groups:
        raise ValueError("no physical payload groups remain after split selection")
    samples: list[dict[str, object]] = []
    for group_index, ((split, payload_id), members) in enumerate(sorted(groups.items())):
        point = members[0]["point"]
        source_ids = sorted(str(member["source"]["source_id"]) for member in members)
        sample_root = output_root / "samples" / split / payload_id
        pooled_tracklets = sample_root / "pooled_physical_tracklets.root"
        pooled_propagations = sample_root / "pooled_physical_propagations.root"
        descriptor_path = sample_root / "pooled_physical_descriptor.json"
        synthetic_tracklets = sample_root / "synthetic_tracklets.root"
        candidates = sample_root / "field_candidates.root"
        all_events = []
        all_records = []
        origin_namespaces: list[dict[str, object]] = []
        physical_event_uids: list[str] = []
        for source_index, member in enumerate(sorted(members, key=lambda value: str(value["source"]["source_id"]))):
            source = member["source"]
            member_point = member["point"]
            source_id = str(source["source_id"])
            tracklets_path = Path(str(member_point["physical_tracklets"])).expanduser().resolve()
            propagations_path = Path(str(member_point["physical_propagations"])).expanduser().resolve()
            if not tracklets_path.is_file() or not propagations_path.is_file():
                raise FileNotFoundError(f"pooled physical asset is missing for {source_id}/{payload_id}")
            events = load_events(tracklets_path, require_mc_labels=True)
            records = load_propagation_records(propagations_path)
            namespace_base = 9_000_000_000 + group_index * 10_000 + source_index * 100
            namespaced_events, namespaced_records, namespace_map = namespace_physical_source(
                events, records, namespace_base
            )
            all_events.extend(namespaced_events)
            all_records.append(namespaced_records)
            source_uids = {str(value) for value in source["source_event_uids"]}
            member_uids = [f"{source_id}:{event.run_id}:{event.event_id}" for event in events]
            if not set(member_uids).issubset(source_uids):
                raise ValueError(f"physical events escape their source provenance: {source_id}")
            physical_event_uids.extend(member_uids)
            origin_namespaces.append(
                {
                    "source_id": source_id,
                    "physical_tracklets": str(tracklets_path),
                    "physical_propagations": str(propagations_path),
                    "run_id_mapping": namespace_map,
                }
            )
        merged_records = merge_propagation_records(all_records)
        descriptor = {
            "source_pooling": "within_split_same_payload",
            "split": split,
            "payload_id": payload_id,
            "source_ids": source_ids,
            "magnitude_mm": float(point["magnitude_mm"]),
            "direction_trial": str(point["direction_trial"]),
            "injected_offsets_xy_mm": dict(point["injected_offsets_xy_mm"]),
            "origin_namespaces": origin_namespaces,
            "physical_geometry_repropagation": True,
            "q_over_p_mode": 0,
            "overlay_seed_scope": overlay_seed_scope,
        }
        if not (args.resume and pooled_tracklets.is_file()):
            write_pooled_tracklets_root(all_events, pooled_tracklets, descriptor)
        if not (args.resume and pooled_propagations.is_file()):
            write_pooled_propagations_root(merged_records, pooled_propagations, descriptor)
        _write_json(descriptor_path, descriptor)
        event_count = int(synthetic["events_per_payload"]) * len(source_ids)
        run_id = int(synthetic["synthetic_run_id_base"]) + group_index
        overlay_seed = _stable_seed(
            int(synthetic["seed"]),
            split,
            payload_id,
            float(point["magnitude_mm"]),
            overlay_seed_scope,
        )
        if not (args.resume and synthetic_tracklets.is_file()):
            overlay = write_synthetic_multitrack_root(
                all_events,
                synthetic_tracklets,
                output_events=event_count,
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
                hard_negative_min_truth_chi2_ratio=(
                    None
                    if synthetic.get("hard_negative_min_truth_chi2_ratio") is None
                    else float(synthetic["hard_negative_min_truth_chi2_ratio"])
                ),
                hard_negative_max_truth_chi2_ratio=(
                    None
                    if synthetic.get("hard_negative_max_truth_chi2_ratio") is None
                    else float(synthetic["hard_negative_max_truth_chi2_ratio"])
                ),
                physical_propagations=pooled_propagations,
                target_z_tolerance_mm=float(field["target_z_tolerance_mm"]),
                allow_duplicate_source_tracks=bool(synthetic.get("allow_duplicate_source_tracks", False)),
                minimum_truth_match_fraction=float(synthetic["minimum_truth_match_fraction"]),
                seed=overlay_seed,
                synthetic_run_id=run_id,
            )
            _write_json(sample_root / "overlay_summary.json", asdict(overlay))
        if not (args.resume and candidates.is_file()):
            candidate_summary = write_synthetic_field_candidate_root(
                synthetic_tracklets=synthetic_tracklets,
                source_propagations=pooled_propagations,
                destination=candidates,
                q_over_p_mode=0,
                target_z_tolerance_mm=float(field["target_z_tolerance_mm"]),
            )
            _write_json(sample_root / "field_candidate_summary.json", asdict(candidate_summary))
            # Some physical source chunks already have incomplete exporter
            # coverage.  Preserve and audit that real coverage limitation;
            # only a target-plane mismatch would indicate an invalid copied
            # Acts state for the pooled geometry.
            if candidate_summary.target_z_mismatch:
                raise RuntimeError(f"pooled field candidate export has a target-z mismatch: {split}/{payload_id}")
        _write_json(
            sample_root / "resolved_config.json",
            {
                **descriptor,
                "pooled_physical_tracklets": str(pooled_tracklets),
                "pooled_physical_propagations": str(pooled_propagations),
                "synthetic_tracklets": str(synthetic_tracklets),
                "field_candidates": str(candidates),
                "synthetic_run_id": run_id,
                "synthetic_events": event_count,
                "overlay_seed": overlay_seed,
            },
        )
        source_event_uids = sorted(
            {str(value) for member in members for value in member["source"]["source_event_uids"]}
        )
        samples.append(
            {
                "source_id": f"pooled_{split}",
                "source_ids": source_ids,
                "split": split,
                "payload_id": payload_id,
                "point_name": str(point["name"]),
                "magnitude_mm": float(point["magnitude_mm"]),
                "direction_trial": str(point["direction_trial"]),
                "injected_offsets_xy_mm": dict(point["injected_offsets_xy_mm"]),
                "source_event_uids": source_event_uids,
                "physical_event_uids": sorted(physical_event_uids),
                "physical_tracklets": str(pooled_tracklets),
                "physical_propagations": str(pooled_propagations),
                "physical_payload_manifest": str(descriptor_path),
                "synthetic_tracklets": str(synthetic_tracklets),
                "field_candidates": str(candidates),
                "physical_geometry_repropagation": True,
                "source_pooling": "within_split_same_payload",
                "origin_namespace_descriptor": str(descriptor_path),
            }
        )
    manifest = {
        "schema_version": SYNTHETIC_CORPUS_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "physical_corpus_manifest": str(physical_path),
        "physical_geometry_repropagation": True,
        "source_split_unit": "original_xAOD_file",
        "source_event_uid_convention": physical["source_event_uid_convention"],
        "q_over_p_mode": 0,
        "source_pooling": "within_split_same_payload",
        "sources_by_split": {key: sorted(value) for key, value in sources_by_split.items()},
        "materialized_splits": sorted(
            {str(sample["split"]) for sample in samples}
        ),
        "synthetic_multitrack": synthetic,
        "field_candidate_export": field,
        "samples": samples,
    }
    manifest_path = output_root / "synthetic_corpus_manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "samples": len(samples)}, indent=2))


if __name__ == "__main__":
    main()
