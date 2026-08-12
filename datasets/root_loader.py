"""ROOT loader for the canonical flat tracklet schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import uproot

from .schema import (
    CANONICAL_TREE_NAME,
    DatasetSchemaError,
    MC_LABEL_FIELDS,
    assert_finite_covariances,
    covariance_from_columns,
    required_fields,
    validate_tracklet_tree,
)


@dataclass(frozen=True)
class EventTracklets:
    """All local tracklets belonging to one (run, event) pair."""

    run_id: int
    event_id: int
    station_id: np.ndarray
    tracklet_id: np.ndarray
    z_mm: np.ndarray
    state: np.ndarray
    covariance: np.ndarray
    chi2: np.ndarray
    ndof: np.ndarray
    n_hit: np.ndarray
    hit_pattern: np.ndarray
    truth_particle_id: Optional[np.ndarray] = None
    truth_pdg: Optional[np.ndarray] = None
    truth_match_fraction: Optional[np.ndarray] = None
    q_over_p_per_mev: Optional[np.ndarray] = None
    q_over_p_from_momentum_per_mev: Optional[np.ndarray] = None
    q_over_p_variance_per_mev2: Optional[np.ndarray] = None
    has_q_over_p_covariance: Optional[np.ndarray] = None
    truth_q_over_p_per_mev: Optional[np.ndarray] = None
    origin_run_id: Optional[np.ndarray] = None
    origin_event_id: Optional[np.ndarray] = None
    origin_tracklet_id: Optional[np.ndarray] = None
    synthetic_role: Optional[np.ndarray] = None
    synthetic_hard_anchor_station: Optional[np.ndarray] = None
    synthetic_hard_anchor_chi2: Optional[np.ndarray] = None

    def indices_for_station(self, station: int) -> np.ndarray:
        return np.flatnonzero(self.station_id == station)

    @property
    def size(self) -> int:
        return int(self.station_id.size)


def _event_from_indices(
    columns: dict[str, np.ndarray],
    covariance: np.ndarray,
    indices: np.ndarray,
    has_mc_labels: bool,
) -> EventTracklets:
    first = int(indices[0])
    state = np.column_stack(
        [
            columns["x_mm"][indices],
            columns["y_mm"][indices],
            columns["tx"][indices],
            columns["ty"][indices],
        ]
    ).astype(np.float64, copy=False)
    event = EventTracklets(
        run_id=int(columns["run_id"][first]),
        event_id=int(columns["event_id"][first]),
        station_id=np.asarray(columns["station_id"][indices], dtype=np.int16),
        tracklet_id=np.asarray(columns["tracklet_id"][indices], dtype=np.int32),
        z_mm=np.asarray(columns["z_mm"][indices], dtype=np.float64),
        state=state,
        covariance=np.asarray(covariance[indices], dtype=np.float64),
        chi2=np.asarray(columns["chi2"][indices], dtype=np.float64),
        ndof=np.asarray(columns["ndof"][indices], dtype=np.float64),
        n_hit=np.asarray(columns["n_hit"][indices], dtype=np.int16),
        hit_pattern=np.asarray(columns["hit_pattern"][indices], dtype=np.uint64),
        truth_particle_id=(
            np.asarray(columns["truth_particle_id"][indices], dtype=np.int64)
            if has_mc_labels
            else None
        ),
        truth_pdg=(
            np.asarray(columns["truth_pdg"][indices], dtype=np.int32)
            if has_mc_labels
            else None
        ),
        truth_match_fraction=(
            np.asarray(columns["truth_match_fraction"][indices], dtype=np.float64)
            if has_mc_labels
            else None
        ),
        q_over_p_per_mev=(
            np.asarray(columns["q_over_p_per_mev"][indices], dtype=np.float64)
            if "q_over_p_per_mev" in columns
            else None
        ),
        q_over_p_from_momentum_per_mev=(
            np.asarray(
                columns["q_over_p_from_momentum_per_mev"][indices], dtype=np.float64
            )
            if "q_over_p_from_momentum_per_mev" in columns
            else None
        ),
        q_over_p_variance_per_mev2=(
            np.asarray(
                columns["q_over_p_variance_per_mev2"][indices], dtype=np.float64
            )
            if "q_over_p_variance_per_mev2" in columns
            else None
        ),
        has_q_over_p_covariance=(
            np.asarray(columns["has_q_over_p_covariance"][indices], dtype=bool)
            if "has_q_over_p_covariance" in columns
            else None
        ),
        truth_q_over_p_per_mev=(
            np.asarray(columns["truth_q_over_p_per_mev"][indices], dtype=np.float64)
            if "truth_q_over_p_per_mev" in columns
            else None
        ),
        origin_run_id=(
            np.asarray(columns["origin_run_id"][indices], dtype=np.int64)
            if "origin_run_id" in columns
            else None
        ),
        origin_event_id=(
            np.asarray(columns["origin_event_id"][indices], dtype=np.int64)
            if "origin_event_id" in columns
            else None
        ),
        origin_tracklet_id=(
            np.asarray(columns["origin_tracklet_id"][indices], dtype=np.int32)
            if "origin_tracklet_id" in columns
            else None
        ),
        synthetic_role=(
            np.asarray(columns["synthetic_role"][indices], dtype=np.int8)
            if "synthetic_role" in columns
            else None
        ),
        synthetic_hard_anchor_station=(
            np.asarray(columns["synthetic_hard_anchor_station"][indices], dtype=np.int8)
            if "synthetic_hard_anchor_station" in columns
            else None
        ),
        synthetic_hard_anchor_chi2=(
            np.asarray(columns["synthetic_hard_anchor_chi2"][indices], dtype=np.float64)
            if "synthetic_hard_anchor_chi2" in columns
            else None
        ),
    )
    if np.unique(event.tracklet_id).size != event.size:
        raise DatasetSchemaError(
            f"duplicate tracklet_id in run={event.run_id}, event={event.event_id}"
        )
    if not np.isfinite(event.state).all() or not np.isfinite(event.z_mm).all():
        raise DatasetSchemaError(
            f"non-finite state in run={event.run_id}, event={event.event_id}"
        )
    assert_finite_covariances(event.covariance)
    return event


def load_events(
    root_path: str | Path,
    tree_name: str = CANONICAL_TREE_NAME,
    max_events: Optional[int] = None,
    require_mc_labels: bool = False,
) -> list[EventTracklets]:
    """Load canonical tracklets grouped by event.

    This initial loader intentionally reads a complete debug-scale sample into
    memory. Training-scale sharding and streaming will be added only after the
    physics baseline has been validated.
    """
    path = Path(root_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    with uproot.open(path) as root_file:
        if tree_name not in root_file:
            raise DatasetSchemaError(f"tree '{tree_name}' is absent from {path}")
        tree = root_file[tree_name]
        report = validate_tracklet_tree(tree)
        report.require_valid(require_mc_labels=require_mc_labels)
        fields = list(required_fields(require_mc_labels=require_mc_labels))
        if report.has_mc_labels and not require_mc_labels:
            fields.extend(MC_LABEL_FIELDS)
        available = set(report.available_fields)
        q_over_p_fields = (
            "q_over_p_per_mev",
            "q_over_p_from_momentum_per_mev",
            "q_over_p_variance_per_mev2",
            "has_q_over_p_covariance",
            "truth_q_over_p_per_mev",
        )
        fields.extend(field for field in q_over_p_fields if field in available)
        origin_fields = ("origin_run_id", "origin_event_id", "origin_tracklet_id")
        present_origin_fields = tuple(field for field in origin_fields if field in available)
        if present_origin_fields and len(present_origin_fields) != len(origin_fields):
            missing_origin_fields = sorted(set(origin_fields) - set(present_origin_fields))
            raise DatasetSchemaError(
                "synthetic provenance is incomplete; missing: "
                + ", ".join(missing_origin_fields)
            )
        fields.extend(present_origin_fields)
        synthetic_role_fields = (
            "synthetic_role",
            "synthetic_hard_anchor_station",
            "synthetic_hard_anchor_chi2",
        )
        present_synthetic_role_fields = tuple(
            field for field in synthetic_role_fields if field in available
        )
        if present_synthetic_role_fields and len(present_synthetic_role_fields) != len(
            synthetic_role_fields
        ):
            missing_synthetic_role_fields = sorted(
                set(synthetic_role_fields) - set(present_synthetic_role_fields)
            )
            raise DatasetSchemaError(
                "synthetic fake provenance is incomplete; missing: "
                + ", ".join(missing_synthetic_role_fields)
            )
        fields.extend(present_synthetic_role_fields)
        raw_columns = tree.arrays(fields, library="np")

    if isinstance(raw_columns, dict):
        columns = {name: np.asarray(raw_columns[name]) for name in fields}
    elif isinstance(raw_columns, np.ndarray) and raw_columns.dtype.names is not None:
        columns = {name: np.asarray(raw_columns[name]) for name in fields}
    else:
        raise DatasetSchemaError(
            "uproot returned an unsupported column container for canonical tracklets"
        )

    if not columns["run_id"].size:
        return []

    covariance = covariance_from_columns(columns)
    order = np.lexsort(
        (
            np.asarray(columns["tracklet_id"]),
            np.asarray(columns["event_id"]),
            np.asarray(columns["run_id"]),
        )
    )
    for name in tuple(columns):
        columns[name] = np.asarray(columns[name])[order]
    covariance = covariance[order]

    run_ids = np.asarray(columns["run_id"])
    event_ids = np.asarray(columns["event_id"])
    boundaries = np.flatnonzero(
        (run_ids[1:] != run_ids[:-1]) | (event_ids[1:] != event_ids[:-1])
    ) + 1
    groups = np.split(np.arange(run_ids.size), boundaries)

    has_mc_labels = report.has_mc_labels
    events: list[EventTracklets] = []
    for group in groups:
        events.append(_event_from_indices(columns, covariance, group, has_mc_labels))
        if max_events is not None and len(events) >= max_events:
            break
    return events
