"""Manifest contracts for source-disjoint physical-payload curriculum data."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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
    # Legacy translation corpora store only ``magnitude_mm``.  New physical
    # corpora name the condition explicitly so a rotation can never be
    # reported as a displacement in millimetres.
    condition_axis: str = "translation_xy_mm"
    condition_value: float | None = None
    condition_magnitude: float | None = None
    injected_station_transforms: Mapping[str, object] = field(default_factory=dict)
    alignment_parameter_values: Mapping[str, object] = field(default_factory=dict)

    @property
    def curriculum_magnitude(self) -> float:
        """Non-negative curriculum coordinate in the declared condition unit."""
        return float(self.magnitude_mm if self.condition_magnitude is None else self.condition_magnitude)


def uniform_condition_axis(samples: Sequence[CurriculumSample]) -> str:
    """Require a single explicitly named physical condition axis per study."""
    axes = {str(sample.condition_axis) for sample in samples}
    if len(axes) != 1:
        raise ValueError("a curriculum study cannot mix physical condition axes")
    return next(iter(axes))


def condition_axis_label(axis: str) -> str:
    """Human-facing axis label for plots and tables, kept separate from IDs."""
    labels = {
        "translation_xy_mm": "injected translation magnitude [mm]",
        "ift_ry_mrad": "injected IFT R_y [mrad]",
        "ift_dx_dy_ry_joint_l2": "injected joint IFT transform severity [normalized]",
        "four_station_relative_l2": "injected 15-DoF relative four-station condition [unit box]",
    }
    return labels.get(str(axis), f"injected {axis}")


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
        condition_axis = str(raw.get("condition_axis", "translation_xy_mm"))
        if not condition_axis:
            raise ValueError(f"sample {source_id}/{payload_id} has an empty condition axis")
        try:
            raw_legacy_magnitude = raw.get("magnitude_mm")
            raw_condition_magnitude = raw.get("condition_magnitude")
            if raw_legacy_magnitude is None and raw_condition_magnitude is None:
                raise KeyError("magnitude_mm or condition_magnitude")
            magnitude_mm = float(
                raw_legacy_magnitude
                if raw_legacy_magnitude is not None
                else raw_condition_magnitude
            )
            condition_value = float(raw.get("condition_value", magnitude_mm))
            condition_magnitude = float(
                raw_condition_magnitude
                if raw_condition_magnitude is not None
                else abs(condition_value)
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"sample {source_id}/{payload_id} has invalid condition metadata") from error
        if condition_magnitude < 0.0:
            raise ValueError(f"sample {source_id}/{payload_id} has a negative condition magnitude")
        raw_transforms = raw.get("injected_station_transforms", {})
        if not isinstance(raw_transforms, Mapping):
            raise ValueError(f"sample {source_id}/{payload_id} has invalid station transforms")
        raw_parameter_values = raw.get("alignment_parameter_values", {})
        if not isinstance(raw_parameter_values, Mapping):
            raise ValueError(f"sample {source_id}/{payload_id} has invalid alignment parameter values")
        raw_offsets = raw.get("injected_offsets_xy_mm", {})
        if not isinstance(raw_offsets, Mapping):
            raise ValueError(f"sample {source_id}/{payload_id} has invalid translation offsets")
        if condition_axis == "ift_ry_mrad" and not raw_transforms:
            raise ValueError(
                f"sample {source_id}/{payload_id} is an IFT R_y sample without station transforms"
            )
        if condition_axis == "ift_dx_dy_ry_joint_l2" and not raw_transforms:
            raise ValueError(
                f"sample {source_id}/{payload_id} is a joint rigid sample without station transforms"
            )
        samples.append(
            CurriculumSample(
                source_id=source_id,
                source_ids=constituent_source_ids,
                split=split,
                payload_id=payload_id,
                magnitude_mm=magnitude_mm,
                direction_trial=str(raw["direction_trial"]),
                injected_offsets_xy_mm=dict(raw_offsets),
                source_event_uids=tuple(uids),
                physical_event_uids=tuple(physical_uids),
                physical_tracklets=_require_path(raw, "physical_tracklets"),
                physical_propagations=_require_path(raw, "physical_propagations"),
                physical_payload_manifest=_require_path(raw, "physical_payload_manifest"),
                synthetic_tracklets=_require_path(raw, "synthetic_tracklets"),
                field_candidates=_require_path(raw, "field_candidates"),
                condition_axis=condition_axis,
                condition_value=condition_value,
                condition_magnitude=condition_magnitude,
                injected_station_transforms=dict(raw_transforms),
                alignment_parameter_values=dict(raw_parameter_values),
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
