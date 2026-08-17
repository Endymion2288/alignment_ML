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
from typing import Any, Mapping, Sequence
import zlib

import numpy as np

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


def _load_resumable_synthetic_manifest(
    path: Path,
    physical_manifest: Path,
) -> list[dict[str, object]]:
    """Load prior samples only when they belong to the same physical corpus."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SYNTHETIC_CORPUS_SCHEMA:
        raise ValueError("existing synthetic manifest has an unexpected schema")
    previous_physical = Path(str(payload.get("physical_corpus_manifest", ""))).expanduser().resolve()
    if previous_physical != physical_manifest.resolve():
        raise ValueError("existing synthetic manifest belongs to a different physical corpus")
    if payload.get("physical_geometry_repropagation") is not True or int(payload.get("q_over_p_mode", -1)) != 0:
        raise ValueError("existing synthetic manifest violates the mode-0 physical contract")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or any(not isinstance(sample, Mapping) for sample in raw_samples):
        raise ValueError("existing synthetic manifest has invalid samples")
    return [dict(sample) for sample in raw_samples]


def _merge_resumed_samples(
    existing: Sequence[Mapping[str, object]],
    generated: Sequence[Mapping[str, object]],
    replaced_splits: set[str],
) -> list[dict[str, object]]:
    """Replace only materialized splits while preserving other valid splits.

    This is what makes ``--split train`` followed by ``--split validation
    --resume`` safe: the second invocation adds validation samples instead of
    replacing the train-only manifest.
    """
    allowed = {"train", "validation", "test"}
    if not replaced_splits or not replaced_splits <= allowed:
        raise ValueError("resumed materialization has invalid replacement splits")
    result = [dict(sample) for sample in existing if str(sample.get("split")) not in replaced_splits]
    result.extend(dict(sample) for sample in generated)
    identities: set[tuple[str, str]] = set()
    for sample in result:
        split = str(sample.get("split", ""))
        payload_id = str(sample.get("payload_id", ""))
        if split not in allowed or not payload_id:
            raise ValueError("synthetic manifest sample has invalid split or payload ID")
        identity = (split, payload_id)
        if identity in identities:
            raise ValueError(f"synthetic manifest has duplicate sample identity: {split}/{payload_id}")
        identities.add(identity)
    return sorted(result, key=lambda sample: (str(sample["split"]), str(sample["payload_id"])))


def _sources_by_split_from_samples(samples: Sequence[Mapping[str, object]]) -> dict[str, list[str]]:
    result: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    for sample in samples:
        split = str(sample["split"])
        source_ids = sample.get("source_ids", [sample.get("source_id")])
        if not isinstance(source_ids, (list, tuple)):
            raise ValueError("synthetic manifest sample has invalid source_ids")
        result[split].update(str(source_id) for source_id in source_ids)
    return {split: sorted(source_ids) for split, source_ids in result.items()}


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
    condition_axis_or_magnitude: str | float,
    condition_magnitude_or_scope: float | str,
    scope: str | None = None,
) -> int:
    """Derive a deterministic overlay seed without touching physical states.

    The legacy ``payload`` scope gives every geometry payload an independent
    synthetic overlay.  A direction scan instead uses
    ``magnitude_shared_across_direction_trials`` so all directions at one
    condition magnitude begin with the same source-track/fake RNG stream.
    Any later difference is then attributable to a real refit/Acts response
    or to a geometry-dependent physical-fake acceptance, not a new arbitrary
    seed.  The condition axis is part of the seed identity, so a rotation is
    never silently treated as a translation of the same numeric value.  The
    alignment-iteration scope intentionally uses one stream across every
    payload in an iteration bank: it lets truth-free selected routes be
    intersected across the central point and physical finite-difference
    probes.  It is only appropriate for a dedicated closed-loop bank, not for
    independent curriculum augmentation samples.
    """
    # Keep the five-argument translation contract used by already materialized
    # curriculum manifests.  New callers pass an explicit condition axis so a
    # 40 mm translation and a 40 mrad rotation cannot share an RNG stream.
    legacy_translation_call = scope is None
    if legacy_translation_call:
        condition_axis = "translation_xy_mm"
        condition_magnitude = float(condition_axis_or_magnitude)
        scope = str(condition_magnitude_or_scope)
    else:
        condition_axis = str(condition_axis_or_magnitude)
        condition_magnitude = float(condition_magnitude_or_scope)

    if scope == "payload":
        identity = f"pooled/{split}/{payload_id}"
    elif scope in {
        "magnitude_shared_across_direction_trials",
        "condition_magnitude_shared_across_direction_trials",
    }:
        if legacy_translation_call:
            identity = f"pooled/{split}/magnitude/{condition_magnitude:.17g}"
        else:
            identity = f"pooled/{split}/condition/{condition_axis}/{condition_magnitude:.17g}"
    elif scope == "alignment_iteration_shared_across_payloads":
        identity = f"pooled/{split}/alignment_iteration"
    else:
        raise ValueError(
            "synthetic_multitrack.overlay_seed_scope must be 'payload', "
            "'magnitude_shared_across_direction_trials', or "
            "'condition_magnitude_shared_across_direction_trials', or "
            "'alignment_iteration_shared_across_payloads'"
        )
    return int((base + zlib.crc32(identity.encode("utf-8"))) % (2**31 - 1))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _condition_metadata(point: Mapping[str, Any]) -> dict[str, object]:
    """Read one explicit physical condition without unit relabelling."""
    axis = str(point.get("condition_axis", "translation_xy_mm"))
    if not axis:
        raise ValueError("physical point has an empty condition_axis")
    raw_magnitude = point.get("condition_magnitude", point.get("magnitude_mm"))
    if raw_magnitude is None:
        raise ValueError(f"physical point '{point.get('name')}' has no condition magnitude")
    try:
        magnitude = float(raw_magnitude)
        value = float(point.get("condition_value", raw_magnitude))
    except (TypeError, ValueError) as error:
        raise ValueError(f"physical point '{point.get('name')}' has invalid condition metadata") from error
    if not np.isfinite(magnitude) or magnitude < 0.0 or not np.isfinite(value):
        raise ValueError(f"physical point '{point.get('name')}' has non-finite condition metadata")
    transforms = point.get("injected_station_transforms", {})
    offsets = point.get("injected_offsets_xy_mm", {})
    parameter_values = point.get("alignment_parameter_values", {})
    if (
        not isinstance(transforms, Mapping)
        or not isinstance(offsets, Mapping)
        or not isinstance(parameter_values, Mapping)
    ):
        raise ValueError(f"physical point '{point.get('name')}' has invalid payload metadata")
    if axis == "ift_ry_mrad" and not transforms:
        raise ValueError(f"IFT R_y physical point '{point.get('name')}' lacks station transforms")
    if axis == "ift_dx_dy_ry_joint_l2" and not transforms:
        raise ValueError(f"joint rigid physical point '{point.get('name')}' lacks station transforms")
    return {
        "condition_axis": axis,
        "condition_value": value,
        "condition_magnitude": magnitude,
        "injected_station_transforms": dict(transforms),
        "injected_offsets_xy_mm": dict(offsets),
        "alignment_parameter_values": dict(parameter_values),
    }


def _payload_signature(point: Mapping[str, Any]) -> tuple[str, float, float, str, str, str, str]:
    condition = _condition_metadata(point)
    return (
        str(condition["condition_axis"]),
        float(condition["condition_value"]),
        float(condition["condition_magnitude"]),
        str(point["direction_trial"]),
        json.dumps(condition["injected_station_transforms"], sort_keys=True),
        json.dumps(condition["injected_offsets_xy_mm"], sort_keys=True),
        json.dumps(condition["alignment_parameter_values"], sort_keys=True),
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
    if overlay_seed_scope not in {
        "payload",
        "magnitude_shared_across_direction_trials",
        "condition_magnitude_shared_across_direction_trials",
        "alignment_iteration_shared_across_payloads",
    }:
        raise ValueError(
            "synthetic_multitrack.overlay_seed_scope must be 'payload', "
            "'magnitude_shared_across_direction_trials', or "
            "'condition_magnitude_shared_across_direction_trials', or "
            "'alignment_iteration_shared_across_payloads'"
        )
    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise FileExistsError("refusing to overwrite a non-empty pooled synthetic output")
    output_root.mkdir(parents=True, exist_ok=True)
    selected_splits = None if args.split is None else set(args.split)
    physical_allowed_splits = {str(split) for split in physical.get("allowed_splits", ("train", "validation", "test"))}
    if selected_splits is not None and not selected_splits <= physical_allowed_splits:
        disallowed = sorted(selected_splits - physical_allowed_splits)
        raise ValueError("selected split is excluded by the physical corpus: " + ", ".join(disallowed))
    manifest_path = output_root / "synthetic_corpus_manifest.json"
    previous_samples: list[dict[str, object]] = []
    if args.resume and manifest_path.is_file():
        previous_samples = _load_resumable_synthetic_manifest(manifest_path, physical_path)
    groups, sources_by_split = _groups(physical, selected_splits=selected_splits)
    if not groups:
        raise ValueError("no physical payload groups remain after split selection")
    samples: list[dict[str, object]] = []
    for group_index, ((split, payload_id), members) in enumerate(sorted(groups.items())):
        point = members[0]["point"]
        condition = _condition_metadata(point)
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
            "condition_axis": condition["condition_axis"],
            "condition_value": condition["condition_value"],
            "condition_magnitude": condition["condition_magnitude"],
            "direction_trial": str(point["direction_trial"]),
            "injected_station_transforms": condition["injected_station_transforms"],
            "injected_offsets_xy_mm": condition["injected_offsets_xy_mm"],
            "alignment_parameter_values": condition["alignment_parameter_values"],
            "origin_namespaces": origin_namespaces,
            "physical_geometry_repropagation": True,
            "q_over_p_mode": 0,
            "overlay_seed_scope": overlay_seed_scope,
        }
        if str(condition["condition_axis"]) == "translation_xy_mm":
            # Read-only compatibility for historical translation consumers.
            descriptor["magnitude_mm"] = float(condition["condition_magnitude"])
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
            str(condition["condition_axis"]),
            float(condition["condition_magnitude"]),
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
                "condition_axis": condition["condition_axis"],
                "condition_value": condition["condition_value"],
                "condition_magnitude": condition["condition_magnitude"],
                "direction_trial": str(point["direction_trial"]),
                "injected_station_transforms": condition["injected_station_transforms"],
                "injected_offsets_xy_mm": condition["injected_offsets_xy_mm"],
                "alignment_parameter_values": condition["alignment_parameter_values"],
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
        if str(condition["condition_axis"]) == "translation_xy_mm":
            samples[-1]["magnitude_mm"] = float(condition["condition_magnitude"])
    replaced_splits = {str(sample["split"]) for sample in samples}
    final_samples = _merge_resumed_samples(previous_samples, samples, replaced_splits)
    manifest = {
        "schema_version": SYNTHETIC_CORPUS_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "physical_corpus_manifest": str(physical_path),
        "physical_geometry_repropagation": True,
        "source_split_unit": "original_xAOD_file",
        "source_event_uid_convention": physical["source_event_uid_convention"],
        "q_over_p_mode": 0,
        "source_pooling": "within_split_same_payload",
        "sources_by_split": _sources_by_split_from_samples(final_samples),
        "materialized_splits": sorted(
            {str(sample["split"]) for sample in final_samples}
        ),
        "synthetic_multitrack": synthetic,
        "field_candidate_export": field,
        "condition_axes": sorted({str(sample.get("condition_axis", "translation_xy_mm")) for sample in final_samples}),
        "samples": final_samples,
    }
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "samples_generated_this_run": len(samples),
                "samples_in_manifest": len(final_samples),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
