"""Traditional geometry-only chi-square matching baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from datasets.root_loader import EventTracklets
from geometry.propagation import mahalanobis_chi2, propagate_line


@dataclass(frozen=True)
class CandidatePair:
    source_index: int
    target_index: int
    chi2: float
    residual: np.ndarray


@dataclass(frozen=True)
class Match:
    source_index: int
    target_index: int
    chi2: float


def build_candidates(
    event: EventTracklets,
    source_station: int,
    target_station: int,
    chi2_gate: float,
) -> list[CandidatePair]:
    """Create gated bipartite candidates for one ordered station pair."""
    if source_station == target_station:
        raise ValueError("source_station and target_station must differ")
    if chi2_gate <= 0.0:
        raise ValueError("chi2_gate must be positive")

    candidates: list[CandidatePair] = []
    source_indices = event.indices_for_station(source_station)
    target_indices = event.indices_for_station(target_station)
    for source_index in source_indices:
        for target_index in target_indices:
            delta_z_mm = float(event.z_mm[target_index] - event.z_mm[source_index])
            predicted_state, predicted_covariance = propagate_line(
                event.state[source_index],
                event.covariance[source_index],
                delta_z_mm,
            )
            residual = event.state[target_index] - predicted_state
            covariance = predicted_covariance + event.covariance[target_index]
            chi2 = mahalanobis_chi2(residual, covariance)
            if chi2 <= chi2_gate:
                candidates.append(
                    CandidatePair(
                        source_index=int(source_index),
                        target_index=int(target_index),
                        chi2=chi2,
                        residual=residual,
                    )
                )
    return candidates


def greedy_one_to_one_match(candidates: list[CandidatePair]) -> list[Match]:
    """Apply the conventional ascending-chi2 greedy one-to-one assignment."""
    used_sources: set[int] = set()
    used_targets: set[int] = set()
    matches: list[Match] = []
    for candidate in sorted(
        candidates,
        key=lambda value: (value.chi2, value.source_index, value.target_index),
    ):
        if candidate.source_index in used_sources or candidate.target_index in used_targets:
            continue
        used_sources.add(candidate.source_index)
        used_targets.add(candidate.target_index)
        matches.append(
            Match(
                source_index=candidate.source_index,
                target_index=candidate.target_index,
                chi2=candidate.chi2,
            )
        )
    return matches

