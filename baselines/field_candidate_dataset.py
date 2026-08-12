"""Shared candidate collection for classical and MLP association baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from baselines.field_chi2_matching import (
    FieldCandidate,
    build_field_candidates,
    candidate_feature_matrix,
    candidate_labels,
    station_pairs_from_events,
)
from datasets.propagation_loader import PropagationRecords
from datasets.root_loader import EventTracklets


@dataclass(frozen=True)
class CandidateBatch:
    """Candidates for one synthetic event and one ordered station pair."""

    event: EventTracklets
    station_pair: tuple[int, int]
    candidates: tuple[FieldCandidate, ...]
    features: np.ndarray
    labels: np.ndarray


def resolved_station_pairs(
    events: Sequence[EventTracklets],
    station_pairs: Sequence[tuple[int, int]] | None = None,
) -> tuple[tuple[int, int], ...]:
    """Validate or infer the fixed forward station-pair list."""
    pairs = (
        tuple((int(source), int(target)) for source, target in station_pairs)
        if station_pairs is not None
        else station_pairs_from_events(events)
    )
    if not pairs or any(source >= target for source, target in pairs):
        raise ValueError("station_pairs must contain ordered forward station pairs")
    if len(set(pairs)) != len(pairs):
        raise ValueError("station_pairs contains duplicates")
    return tuple(sorted(pairs))


def build_candidate_batches(
    events: Iterable[EventTracklets],
    records: PropagationRecords,
    station_pairs: Sequence[tuple[int, int]],
    chi2_gate: float | None = None,
) -> list[CandidateBatch]:
    """Build feature-bearing candidate batches without using truth for matching."""
    # The iterable is materialized once here so callers can safely provide a generator.
    event_list = list(events)
    pairs = resolved_station_pairs(event_list, station_pairs)
    batches: list[CandidateBatch] = []
    for event in event_list:
        for source_station, target_station in pairs:
            candidates = build_field_candidates(
                event,
                records,
                source_station=source_station,
                target_station=target_station,
                chi2_gate=chi2_gate,
            )
            if not candidates:
                continue
            batches.append(
                CandidateBatch(
                    event=event,
                    station_pair=(source_station, target_station),
                    candidates=tuple(candidates),
                    features=candidate_feature_matrix(event, candidates, pairs),
                    labels=candidate_labels(event, candidates),
                )
            )
    return batches


def concatenate_batches(batches: Sequence[CandidateBatch]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate supervised rows while preserving caller-owned event splits."""
    if not batches:
        raise ValueError("no candidate batches are available")
    return (
        np.concatenate([batch.features for batch in batches], axis=0),
        np.concatenate([batch.labels for batch in batches], axis=0),
    )
