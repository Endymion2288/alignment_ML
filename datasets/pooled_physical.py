"""Namespaced pooling of physically refitted tracklets and Acts records.

MC production chunks reuse their original run/event numbers.  A synthetic pool
may therefore combine source files only after replacing the *provenance* run
identifier by a deterministic namespace.  Tracklet states, covariances and
Acts predictions are copied bit-for-bit; no coordinate or residual surrogate
is introduced by this module.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import uproot

from .propagation_loader import (
    PROPAGATION_SCHEMA_VERSION,
    PROPAGATION_TREE_NAME,
    PropagationRecords,
)
from .root_loader import EventTracklets
from .schema import CANONICAL_TREE_NAME, SCHEMA_VERSION, covariance_columns


def namespace_physical_source(
    events: Sequence[EventTracklets],
    records: PropagationRecords,
    namespace_base: int,
) -> tuple[list[EventTracklets], PropagationRecords, list[dict[str, int]]]:
    """Return a source-local origin namespace for colliding run/event IDs.

    ``namespace_base`` is supplied by the caller from a deterministic source
    ordering.  Each original run is mapped to a distinct 64-bit identifier;
    event and tracklet IDs are left untouched because the namespaced run now
    disambiguates their source file.
    """
    if not events:
        raise ValueError("cannot namespace an empty physical event collection")
    if namespace_base < 0:
        raise ValueError("namespace_base must be non-negative")
    original_runs = sorted(
        {int(event.run_id) for event in events}.union(int(value) for value in records.run_id)
    )
    if not original_runs:
        raise ValueError("physical source contains no run identifiers")
    mapping = {run_id: int(namespace_base + offset) for offset, run_id in enumerate(original_runs)}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("namespace mapping is not one-to-one")
    namespaced_events = [replace(event, run_id=mapping[int(event.run_id)]) for event in events]
    try:
        namespaced_runs = np.asarray(
            [mapping[int(run_id)] for run_id in records.run_id], dtype=np.int64
        )
    except KeyError as error:  # pragma: no cover - guarded by the run union above
        raise ValueError("propagation run has no namespace mapping") from error
    namespaced_records = replace(records, run_id=namespaced_runs)
    descriptor = [
        {"original_run_id": original, "namespaced_run_id": namespaced}
        for original, namespaced in sorted(mapping.items())
    ]
    return namespaced_events, namespaced_records, descriptor


def merge_propagation_records(records_by_source: Sequence[PropagationRecords]) -> PropagationRecords:
    """Concatenate already namespaced physical propagation records."""
    if not records_by_source:
        raise ValueError("at least one physical propagation collection is required")
    fields = (
        "run_id",
        "event_id",
        "source_tracklet_id",
        "target_tracklet_id",
        "source_station_id",
        "target_station_id",
        "truth_particle_id",
        "target_z_mm",
        "prediction",
        "covariance",
        "success",
        "has_covariance",
        "q_over_p_mode",
        "source_q_over_p_per_mev",
    )
    combined: dict[str, np.ndarray] = {}
    for field in fields:
        values = [getattr(records, field) for records in records_by_source]
        if any(value is None for value in values):
            raise ValueError(f"physical propagation field '{field}' is absent")
        combined[field] = np.concatenate([np.asarray(value) for value in values], axis=0)
    return PropagationRecords(**combined)


def _all_optional(events: Sequence[EventTracklets], name: str) -> bool:
    present = [getattr(event, name) is not None for event in events]
    if any(present) and not all(present):
        raise ValueError(f"optional tracklet field '{name}' is inconsistently present")
    return bool(present and all(present))


def write_pooled_tracklets_root(
    events: Sequence[EventTracklets],
    destination: str | Path,
    provenance: Mapping[str, object],
) -> None:
    """Persist a namespaced, pre-overlay physical pool in canonical ROOT form."""
    if not events:
        raise ValueError("cannot write an empty pooled tracklet collection")
    required_mc = ("truth_particle_id", "truth_pdg", "truth_match_fraction")
    if not all(_all_optional(events, field) for field in required_mc):
        raise ValueError("pooled physical tracklets require complete MC labels")
    optional = (
        "q_over_p_per_mev",
        "q_over_p_from_momentum_per_mev",
        "q_over_p_variance_per_mev2",
        "has_q_over_p_covariance",
        "truth_q_over_p_per_mev",
    )
    include_optional = tuple(field for field in optional if _all_optional(events, field))
    columns: dict[str, list[np.ndarray]] = {
        "run_id": [],
        "event_id": [],
        "station_id": [],
        "tracklet_id": [],
        "z_mm": [],
        "x_mm": [],
        "y_mm": [],
        "tx": [],
        "ty": [],
        "chi2": [],
        "ndof": [],
        "n_hit": [],
        "hit_pattern": [],
        "truth_particle_id": [],
        "truth_pdg": [],
        "truth_match_fraction": [],
        **{field: [] for field in include_optional},
    }
    covariances: list[np.ndarray] = []
    for event in events:
        columns["run_id"].append(np.full(event.size, event.run_id, dtype=np.int64))
        columns["event_id"].append(np.full(event.size, event.event_id, dtype=np.int64))
        columns["station_id"].append(np.asarray(event.station_id, dtype=np.int16))
        columns["tracklet_id"].append(np.asarray(event.tracklet_id, dtype=np.int32))
        columns["z_mm"].append(np.asarray(event.z_mm, dtype=np.float64))
        columns["x_mm"].append(np.asarray(event.state[:, 0], dtype=np.float64))
        columns["y_mm"].append(np.asarray(event.state[:, 1], dtype=np.float64))
        columns["tx"].append(np.asarray(event.state[:, 2], dtype=np.float64))
        columns["ty"].append(np.asarray(event.state[:, 3], dtype=np.float64))
        columns["chi2"].append(np.asarray(event.chi2, dtype=np.float64))
        columns["ndof"].append(np.asarray(event.ndof, dtype=np.float64))
        columns["n_hit"].append(np.asarray(event.n_hit, dtype=np.int16))
        columns["hit_pattern"].append(np.asarray(event.hit_pattern, dtype=np.uint64))
        for field in required_mc:
            value = getattr(event, field)
            assert value is not None
            columns[field].append(np.asarray(value))
        for field in include_optional:
            value = getattr(event, field)
            assert value is not None
            columns[field].append(np.asarray(value))
        covariances.append(np.asarray(event.covariance, dtype=np.float64))
    root_columns: dict[str, np.ndarray] = {}
    dtypes = {
        "run_id": np.int64,
        "event_id": np.int64,
        "station_id": np.int16,
        "tracklet_id": np.int32,
        "z_mm": np.float64,
        "x_mm": np.float64,
        "y_mm": np.float64,
        "tx": np.float64,
        "ty": np.float64,
        "chi2": np.float64,
        "ndof": np.float64,
        "n_hit": np.int16,
        "hit_pattern": np.uint64,
        "truth_particle_id": np.int64,
        "truth_pdg": np.int32,
        "truth_match_fraction": np.float64,
        "q_over_p_per_mev": np.float64,
        "q_over_p_from_momentum_per_mev": np.float64,
        "q_over_p_variance_per_mev2": np.float64,
        "has_q_over_p_covariance": np.bool_,
        "truth_q_over_p_per_mev": np.float64,
    }
    for field, values in columns.items():
        root_columns[field] = np.concatenate(values).astype(dtypes[field], copy=False)
    root_columns.update(covariance_columns(np.concatenate(covariances, axis=0)))
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(path) as root_file:
        root_file[CANONICAL_TREE_NAME] = root_columns
        root_file["metadata"] = {
            "schema_version": np.asarray([SCHEMA_VERSION]),
            "coordinate_unit": np.asarray(["mm"]),
            "generator": np.asarray(["datasets.pooled_physical.write_pooled_tracklets_root"]),
            "provenance": np.asarray([json.dumps(dict(provenance), sort_keys=True)]),
        }


def write_pooled_propagations_root(
    records: PropagationRecords,
    destination: str | Path,
    provenance: Mapping[str, object],
) -> None:
    """Persist namespaced, exact physical Acts records in the canonical schema."""
    if not records.size:
        raise ValueError("cannot write an empty pooled propagation collection")
    modes = records.q_over_p_mode
    q_over_p = records.source_q_over_p_per_mev
    if modes is None or q_over_p is None:
        raise ValueError("pooled propagation records require q/p mode fields")
    columns: dict[str, np.ndarray] = {
        "run_id": np.asarray(records.run_id, dtype=np.int64),
        "event_id": np.asarray(records.event_id, dtype=np.int64),
        "source_tracklet_id": np.asarray(records.source_tracklet_id, dtype=np.int32),
        "target_tracklet_id": np.asarray(records.target_tracklet_id, dtype=np.int32),
        "source_station_id": np.asarray(records.source_station_id, dtype=np.int16),
        "target_station_id": np.asarray(records.target_station_id, dtype=np.int16),
        "truth_particle_id": np.asarray(records.truth_particle_id, dtype=np.int64),
        "target_z_mm": np.asarray(records.target_z_mm, dtype=np.float64),
        "pred_x_mm": np.asarray(records.prediction[:, 0], dtype=np.float64),
        "pred_y_mm": np.asarray(records.prediction[:, 1], dtype=np.float64),
        "pred_tx": np.asarray(records.prediction[:, 2], dtype=np.float64),
        "pred_ty": np.asarray(records.prediction[:, 3], dtype=np.float64),
        "success": np.asarray(records.success, dtype=np.bool_),
        "has_covariance": np.asarray(records.has_covariance, dtype=np.bool_),
        "q_over_p_mode": np.asarray(modes, dtype=np.int8),
        "source_q_over_p_per_mev": np.asarray(q_over_p, dtype=np.float64),
    }
    columns.update({f"pred_{field}": values for field, values in covariance_columns(records.covariance).items()})
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(path) as root_file:
        root_file[PROPAGATION_TREE_NAME] = columns
        root_file["metadata"] = {
            "schema_version": np.asarray([PROPAGATION_SCHEMA_VERSION]),
            "coordinate_unit": np.asarray(["mm"]),
            "generator": np.asarray(["datasets.pooled_physical.write_pooled_propagations_root"]),
            "provenance": np.asarray([json.dumps(dict(provenance), sort_keys=True)]),
        }
