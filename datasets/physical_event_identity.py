"""Workbook-76 physical-event identity for merged-rec FD banks.

Merged MC24 rec files reuse generator-job event numbers, so
``(run_id, event_id)`` is not a unique physical-event key: each rec file
merges ten generator jobs that restart event numbering.  Any cross-geometry
FD join on sorted ``(run, event)`` groups would merge distinct physical
events and corrupt the Jacobian.

This module builds the exact physical-event identity from the enhanced
NtupleDumper file that every FD refit point writes next to its converted
``tracklets.root`` / ``propagations.root``.  The ntuple has one entry per
processed event in file order, so the occurrence of each entry among
earlier entries with the same ``(run, eventID)`` is the absolute physical
occurrence.  Per-entry tracklet/propagation vector lengths bridge each flat
row block to its ntuple entry, which keeps the two converted files
consistent even when an event contributes rows to only one of them.

The synthetic unique event id is ``event_id + occurrence * stride``.  For
files without collisions every occurrence is zero and the augmented id
equals the raw event id, leaving canonical behaviour bit-identical.  A
duplicate physical-event identity after augmentation is a hard failure
(enforced downstream by the duplicate-key checks in
``evaluate_field_propagation`` and the FD bank alignment).

No fuzzy join, no nearest-neighbour join, no residual-proximity join, and
no sorted ``(run, event)`` grouping is used anywhere in this contract.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import uproot

from .propagation_loader import PropagationRecords, load_propagation_records
from .root_loader import (
    PHYSICAL_EVENT_UID_STRIDE,
    EventTracklets,
    _event_from_indices,
    _read_tracklet_columns,
)
from .schema import DatasetSchemaError

NTUPLE_TREE_NAME = "nt"
NTUPLE_RUN_BRANCH = "run"
NTUPLE_EVENT_BRANCH = "eventID"
NTUPLE_TRACKLET_COUNT_BRANCH = "Tracklet_z_mm"
NTUPLE_PROPAGATION_COUNT_BRANCH = "TrackletPropagation_success"


@dataclass(frozen=True)
class NtupleEventIndex:
    """Per-entry physical-event index of one enhanced ntuple (file order)."""

    source: str
    run_id: np.ndarray
    event_id: np.ndarray
    occurrence: np.ndarray
    uid: np.ndarray
    n_tracklet_rows: np.ndarray
    n_propagation_rows: np.ndarray

    @property
    def size(self) -> int:
        return int(self.run_id.size)

    @property
    def n_physical_events(self) -> int:
        return self.size

    @property
    def n_collided_event_ids(self) -> int:
        return int(np.count_nonzero(self.occurrence > 0))


def _entry_count(arrays: dict, branch: str) -> np.ndarray:
    import awkward as ak

    return np.asarray(ak.to_numpy(ak.num(arrays[branch], axis=1)), dtype=np.int64)


def _entry_level_occurrence_uids(
    run_id: np.ndarray,
    event_id: np.ndarray,
    *,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-ENTRY occurrence-augmented uids for the enhanced ntuple.

    Unlike the row-level primitive in ``datasets.root_loader`` (where
    maximal consecutive equal-``(run, event)`` row blocks are one event),
    every ntuple entry IS one physical event: three consecutive entries
    with the same ``(run, eventID)`` are three distinct physical events
    from three merged generator jobs.  Returns ``(occurrence, uid)``.
    """
    runs = np.asarray(run_id, dtype=np.int64)
    events = np.asarray(event_id, dtype=np.int64)
    if runs.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if int(events.min()) < 0 or int(events.max()) >= int(stride):
        raise DatasetSchemaError(
            f"eventID range [{int(events.min())}, {int(events.max())}] exceeds "
            f"physical-identity stride {int(stride)}"
        )
    occurrence = np.zeros(runs.size, dtype=np.int64)
    seen: dict[tuple[int, int], int] = {}
    for row in range(runs.size):
        key = (int(runs[row]), int(events[row]))
        occurrence[row] = seen.get(key, 0)
        seen[key] = int(occurrence[row]) + 1
    uid = events + occurrence * np.int64(stride)
    return occurrence, uid


def read_ntuple_event_index(
    enhanced_path: str | Path,
    *,
    tree_name: str = NTUPLE_TREE_NAME,
    stride: int = PHYSICAL_EVENT_UID_STRIDE,
) -> NtupleEventIndex:
    """Read per-entry physical-event identity from an enhanced ntuple."""
    path = Path(enhanced_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with uproot.open(path) as root_file:
        if tree_name not in root_file:
            raise DatasetSchemaError(f"tree '{tree_name}' is absent from {path}")
        tree = root_file[tree_name]
        available = {str(name).split(";")[0] for name in tree.keys()}
        required = {
            NTUPLE_RUN_BRANCH,
            NTUPLE_EVENT_BRANCH,
            NTUPLE_TRACKLET_COUNT_BRANCH,
            NTUPLE_PROPAGATION_COUNT_BRANCH,
        }
        missing = sorted(required - available)
        if missing:
            raise DatasetSchemaError(
                f"enhanced ntuple {path} lacks physical-identity branches: {missing}"
            )
        arrays = tree.arrays(
            [
                NTUPLE_RUN_BRANCH,
                NTUPLE_EVENT_BRANCH,
                NTUPLE_TRACKLET_COUNT_BRANCH,
                NTUPLE_PROPAGATION_COUNT_BRANCH,
            ],
            library="ak",
        )
    run_id = np.asarray(arrays[NTUPLE_RUN_BRANCH].to_numpy(), dtype=np.int64)
    event_id = np.asarray(arrays[NTUPLE_EVENT_BRANCH].to_numpy(), dtype=np.int64)
    occurrence, uid = _entry_level_occurrence_uids(run_id, event_id, stride=stride)
    return NtupleEventIndex(
        source=str(path),
        run_id=run_id,
        event_id=event_id,
        occurrence=occurrence,
        uid=uid,
        n_tracklet_rows=_entry_count(arrays, NTUPLE_TRACKLET_COUNT_BRANCH),
        n_propagation_rows=_entry_count(arrays, NTUPLE_PROPAGATION_COUNT_BRANCH),
    )


def flat_row_uids(
    index: NtupleEventIndex,
    *,
    kind: str,
    n_rows: int,
    run_ids: np.ndarray,
    event_ids: np.ndarray,
) -> np.ndarray:
    """Assign per-row physical-event uids to a converted flat file.

    ``kind`` is ``"tracklets"`` or ``"propagations"``.  Entries with zero
    rows of that kind contribute no block.  Every row block must match its
    bridged ntuple entry in ``(run_id, event_id)`` exactly; any mismatch or
    length disagreement is a hard failure.
    """
    if kind == "tracklets":
        counts = index.n_tracklet_rows
    elif kind == "propagations":
        counts = index.n_propagation_rows
    else:
        raise ValueError(f"unknown flat-row kind '{kind}'")
    runs = np.asarray(run_ids, dtype=np.int64)
    events = np.asarray(event_ids, dtype=np.int64)
    if runs.shape != events.shape:
        raise DatasetSchemaError("flat-row run/event length mismatch")
    if int(counts.sum()) != int(n_rows):
        raise DatasetSchemaError(
            f"{kind} flat row count {int(n_rows)} does not match the enhanced "
            f"ntuple bridge total {int(counts.sum())} ({index.source})"
        )
    uids = np.empty(int(n_rows), dtype=np.int64)
    cursor = 0
    for entry in range(index.size):
        count = int(counts[entry])
        if count == 0:
            continue
        block = slice(cursor, cursor + count)
        if not (
            np.all(runs[block] == index.run_id[entry])
            and np.all(events[block] == index.event_id[entry])
        ):
            raise DatasetSchemaError(
                f"{kind} flat rows {block.start}:{block.stop} do not match ntuple "
                f"entry {entry} (run={int(index.run_id[entry])}, "
                f"event={int(index.event_id[entry])}) in {index.source}"
            )
        uids[block] = index.uid[entry]
        cursor += count
    return uids


def load_events_ntuple_identity(
    tracklets_path: str | Path,
    index: NtupleEventIndex,
    *,
    require_mc_labels: bool = True,
) -> list[EventTracklets]:
    """Load tracklets in file order with ntuple-bridged physical-event uids.

    The event split is driven by the enhanced-ntuple per-entry tracklet
    counts, NOT by maximal consecutive equal-``(run, event)`` row blocks:
    in merged-rec files two distinct physical events with the same
    ``(run_id, event_id)`` are written as adjacent row blocks, so any
    block-based grouping would merge them and trip the duplicate-tracklet
    check.  Each ntuple entry's rows become one ``EventTracklets`` whose
    ``event_id`` is the occurrence-augmented uid; the per-event duplicate
    tracklet-id check still applies inside each physical event.
    """
    columns, covariance, has_mc_labels = _read_tracklet_columns(
        tracklets_path, require_mc_labels=require_mc_labels
    )
    n_rows = int(columns["run_id"].size) if columns["run_id"].size else 0
    uids = flat_row_uids(
        index,
        kind="tracklets",
        n_rows=n_rows,
        run_ids=np.asarray(columns["run_id"], dtype=np.int64),
        event_ids=np.asarray(columns["event_id"], dtype=np.int64),
    )
    out: list[EventTracklets] = []
    cursor = 0
    for entry in range(index.size):
        count = int(index.n_tracklet_rows[entry])
        if count == 0:
            continue
        rows = np.arange(cursor, cursor + count, dtype=np.int64)
        event = _event_from_indices(columns, covariance, rows, has_mc_labels)
        out.append(replace(event, event_id=int(index.uid[entry])))
        cursor += count
    return out


def load_propagation_records_ntuple_identity(
    propagations_path: str | Path,
    index: NtupleEventIndex,
) -> PropagationRecords:
    """Load propagation records with ntuple-bridged physical-event uids."""
    records = load_propagation_records(propagations_path)
    uids = flat_row_uids(
        index,
        kind="propagations",
        n_rows=records.size,
        run_ids=records.run_id,
        event_ids=records.event_id,
    )
    return replace(records, event_id=uids)


def assert_no_duplicate_physical_identity(events: Iterable[EventTracklets]) -> None:
    """Hard failure if any (run_id, uid, tracklet_id) identity repeats."""
    seen: set[tuple[int, int, int]] = set()
    for event in events:
        for row in range(int(event.size)):
            key = (int(event.run_id), int(event.event_id), int(event.tracklet_id[row]))
            if key in seen:
                raise DatasetSchemaError(f"duplicate physical-event tracklet identity: {key}")
            seen.add(key)
