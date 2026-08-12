"""Manifest contracts for source-disjoint physical-payload curriculum data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SPLITS = ("train", "validation", "test")
PHYSICAL_CORPUS_SCHEMA = "faser-physical-curriculum-corpus-v1"
SYNTHETIC_CORPUS_SCHEMA = "faser-curriculum-synthetic-corpus-v1"


@dataclass(frozen=True)
class CurriculumSample:
    """One synthetic overlay backed by one physical refit payload."""

    source_id: str
    source_ids: tuple[str, ...]
    split: str
    payload_id: str
    magnitude_mm: float
    direction_trial: str
    injected_offsets_xy_mm: Mapping[str, object]
    source_event_uids: tuple[str, ...]
    physical_event_uids: tuple[str, ...]
    physical_tracklets: Path
    physical_propagations: Path
    physical_payload_manifest: Path
    synthetic_tracklets: Path
    field_candidates: Path


def _read_mapping(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"manifest must be a JSON mapping: {source}")
    return source, dict(payload)


def _require_path(sample: Mapping[str, Any], name: str) -> Path:
    value = sample.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"sample is missing path field '{name}'")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"sample path does not exist for {name}: {path}")
    return path


def load_synthetic_curriculum_manifest(
    path: str | Path,
    *,
    require_all_splits: bool = True,
    allowed_splits: Sequence[str] | None = None,
) -> tuple[Path, list[CurriculumSample], dict[str, Any]]:
    """Load a physical synthetic manifest and enforce its provenance contract.

    Training and validation callers retain the default three-way split
    requirement.  ``allowed_splits`` creates an explicit read boundary: paths
    and event assets from every excluded split are never resolved.  This is
    used by post-test architecture work to keep an already sealed test source
    out of loader access altogether.  A sealed, test-only physical scan is
    allowed to load with ``require_all_splits=False`` so it cannot be
    accidentally mixed with training assets merely to satisfy a manifest-shape
    check.
    """
    source, payload = _read_mapping(path)
    if payload.get("schema_version") != SYNTHETIC_CORPUS_SCHEMA:
        raise ValueError("unexpected curriculum synthetic manifest schema version")
    if payload.get("physical_geometry_repropagation") is not True:
        raise ValueError("manifest does not certify physical geometry repropagation")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("manifest contains no synthetic curriculum samples")
    requested_splits = (
        set(SPLITS)
        if allowed_splits is None
        else {str(value) for value in allowed_splits}
    )
    if not requested_splits or not requested_splits.issubset(SPLITS):
        raise ValueError("allowed_splits must be a non-empty subset of train/validation/test")
    if require_all_splits and allowed_splits is not None:
        raise ValueError(
            "require_all_splits=True is incompatible with an explicit allowed_splits boundary"
        )
    source_split: dict[str, str] = {}
    samples: list[CurriculumSample] = []
    for raw in raw_samples:
        if not isinstance(raw, Mapping):
            raise ValueError("manifest sample is not a mapping")
        source_id = str(raw.get("source_id", ""))
        split = str(raw.get("split", ""))
        payload_id = str(raw.get("payload_id", ""))
        if not source_id or not payload_id or split not in SPLITS:
            raise ValueError("manifest sample has invalid source_id, split, or payload_id")
        # Validate split/source ownership from manifest metadata before the
        # access boundary is applied.  Do not resolve excluded sample paths.
        raw_source_ids = raw.get("source_ids", [source_id])
        if (
            not isinstance(raw_source_ids, list)
            or not raw_source_ids
            or not all(isinstance(value, str) and value for value in raw_source_ids)
        ):
            raise ValueError(f"sample '{source_id}' has invalid constituent source_ids")
        constituent_source_ids = tuple(sorted(set(raw_source_ids)))
        for constituent_source_id in constituent_source_ids:
            prior_split = source_split.setdefault(constituent_source_id, split)
            if prior_split != split:
                raise ValueError(
                    f"source '{constituent_source_id}' appears in multiple splits"
                )
        if split not in requested_splits:
            continue
        uids = raw.get("source_event_uids")
        if not isinstance(uids, list) or not uids or not all(isinstance(uid, str) for uid in uids):
            raise ValueError(f"source '{source_id}' has no explicit original event provenance")
        physical_uids = raw.get("physical_event_uids")
        if (
            not isinstance(physical_uids, list)
            or not physical_uids
            or not all(isinstance(uid, str) for uid in physical_uids)
        ):
            raise ValueError(
                f"sample {source_id}/{payload_id} has no explicit refitted-event provenance"
            )
        if not set(physical_uids).issubset(set(uids)):
            raise ValueError(
                f"sample {source_id}/{payload_id} contains a refitted event outside its source"
            )
        if raw.get("physical_geometry_repropagation") is not True:
            raise ValueError(f"sample {source_id}/{payload_id} is not a physical refit")
        samples.append(
            CurriculumSample(
                source_id=source_id,
                source_ids=constituent_source_ids,
                split=split,
                payload_id=payload_id,
                magnitude_mm=float(raw["magnitude_mm"]),
                direction_trial=str(raw["direction_trial"]),
                injected_offsets_xy_mm=dict(raw["injected_offsets_xy_mm"]),
                source_event_uids=tuple(uids),
                physical_event_uids=tuple(physical_uids),
                physical_tracklets=_require_path(raw, "physical_tracklets"),
                physical_propagations=_require_path(raw, "physical_propagations"),
                physical_payload_manifest=_require_path(raw, "physical_payload_manifest"),
                synthetic_tracklets=_require_path(raw, "synthetic_tracklets"),
                field_candidates=_require_path(raw, "field_candidates"),
            )
        )
    if require_all_splits:
        present_splits = {sample.split for sample in samples}
        missing = set(SPLITS) - present_splits
        if missing:
            raise ValueError("manifest has no samples for split(s): " + ", ".join(sorted(missing)))
    elif allowed_splits is not None:
        present_splits = {sample.split for sample in samples}
        missing = requested_splits - present_splits
        if missing:
            raise ValueError(
                "manifest has no samples for requested split(s): " + ", ".join(sorted(missing))
            )
    # A UID is namespaced by its source file deliberately.  Original xAOD run
    # and event numbers can repeat across production chunks.
    event_owner: dict[str, str] = {}
    for sample in samples:
        for uid in sample.source_event_uids:
            owner = event_owner.setdefault(uid, sample.source_id)
            if owner != sample.source_id:
                raise ValueError(f"original source event appears in multiple source files: {uid}")
    return source, samples, payload
