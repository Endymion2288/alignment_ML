"""Exact small-event multi-station assignment from pairwise score matrices.

This remains a baseline rather than a learned global model.  It consumes the
same truth-free pairwise MLP score matrices as the bipartite Hungarian rules,
forms candidate paths spanning two or more stations, and selects mutually
disjoint paths with a unit-capacity mixed-integer programme.  The event sizes
of the controlled synthetic overlays are deliberately small, so an exact
solve is preferable to an unvalidated greedy approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from baselines.field_chi2_matching import FieldCandidate, ScoredMatch
from baselines.global_assignment import ScoreMatrix
from datasets.root_loader import EventTracklets


@dataclass(frozen=True)
class MultiStationAssignmentConfig:
    """Truth-free controls for the exact multi-station set-packing solve."""

    score_threshold: float
    unmatched_penalty: float = 0.0
    maximum_hypotheses: int = 100_000


@dataclass(frozen=True)
class MultiStationAssignmentResult:
    """Selected path edges and explicitly unmatched station endpoints."""

    matches_by_pair: Mapping[tuple[int, int], tuple[ScoredMatch, ...]]
    unmatched_by_station: Mapping[int, tuple[int, ...]]
    candidate_edges_above_threshold: int
    hypotheses_considered: int
    selected_hypotheses: int


@dataclass(frozen=True)
class _TrackHypothesis:
    endpoints: tuple[tuple[int, int], ...]
    matches: tuple[tuple[tuple[int, int], ScoredMatch], ...]
    utility: float


def _validate_config(config: MultiStationAssignmentConfig) -> None:
    if not np.isfinite(config.score_threshold) or not 0.0 <= config.score_threshold <= 1.0:
        raise ValueError("score_threshold must be finite and in [0, 1]")
    if not np.isfinite(config.unmatched_penalty):
        raise ValueError("unmatched_penalty must be finite")
    if config.maximum_hypotheses < 1:
        raise ValueError("maximum_hypotheses must be positive")


def _matrix_edge_lookup(
    matrix: ScoreMatrix,
    candidates: Sequence[FieldCandidate],
    threshold: float,
) -> dict[tuple[int, int], ScoredMatch]:
    """Materialise only declared physical edges above the score threshold."""
    result: dict[tuple[int, int], ScoredMatch] = {}
    for left, source_index in enumerate(matrix.source_indices):
        for right, target_index in enumerate(matrix.target_indices):
            score = float(matrix.values[left, right])
            if not np.isfinite(score) or score < threshold:
                continue
            candidate_row = int(matrix.candidate_rows[left, right])
            if candidate_row < 0:
                raise RuntimeError("finite score matrix entry lacks a physical candidate row")
            candidate = candidates[candidate_row]
            result[(int(source_index), int(target_index))] = ScoredMatch(
                source_index=int(source_index),
                target_index=int(target_index),
                chi2=float(candidate.chi2),
                score=score,
            )
    return result


def _edge(
    station_left: int,
    index_left: int,
    station_right: int,
    index_right: int,
    edge_lookups: Mapping[tuple[int, int], Mapping[tuple[int, int], ScoredMatch]],
) -> ScoredMatch | None:
    if station_left >= station_right:
        raise ValueError("multi-station paths must use ordered station pairs")
    return edge_lookups.get((station_left, station_right), {}).get((index_left, index_right))


def _hypotheses(
    event: EventTracklets,
    stations: Sequence[int],
    edge_lookups: Mapping[tuple[int, int], Mapping[tuple[int, int], ScoredMatch]],
    unmatched_penalty: float,
    maximum_hypotheses: int,
) -> list[_TrackHypothesis]:
    """Enumerate internally compatible paths over every observed station subset."""
    paths: list[_TrackHypothesis] = []
    ordered_stations = tuple(sorted({int(station) for station in stations}))
    indices_by_station = {
        station: tuple(int(index) for index in event.indices_for_station(station).tolist())
        for station in ordered_stations
    }
    for count in range(2, len(ordered_stations) + 1):
        for path_stations in combinations(ordered_stations, count):
            if any(not indices_by_station[station] for station in path_stations):
                continue
            for path_indices in product(*(indices_by_station[station] for station in path_stations)):
                matches: list[tuple[tuple[int, int], ScoredMatch]] = []
                log_odds = 0.0
                compatible = True
                for left in range(count):
                    for right in range(left + 1, count):
                        source_station = path_stations[left]
                        target_station = path_stations[right]
                        match = _edge(
                            source_station,
                            path_indices[left],
                            target_station,
                            path_indices[right],
                            edge_lookups,
                        )
                        if match is None:
                            compatible = False
                            break
                        probability = float(np.clip(match.score, 1.0e-6, 1.0 - 1.0e-6))
                        log_odds += float(np.log(probability) - np.log1p(-probability))
                        matches.append(((source_station, target_station), match))
                    if not compatible:
                        break
                if not compatible:
                    continue
                # Relative to leaving every participating endpoint unmatched,
                # a selected path gains its edge log-odds and avoids one
                # dustbin cost for each endpoint.  This agrees with the
                # two-station dustbin objective when ``count == 2``.
                utility = log_odds + count * unmatched_penalty
                if utility <= 0.0:
                    continue
                paths.append(
                    _TrackHypothesis(
                        endpoints=tuple(zip(path_stations, path_indices)),
                        matches=tuple(matches),
                        utility=float(utility),
                    )
                )
                if len(paths) > maximum_hypotheses:
                    raise RuntimeError(
                        "multi-station hypothesis count exceeds maximum_hypotheses; "
                        "tighten the score threshold or reduce overlay occupancy"
                    )
    return paths


def _select_disjoint_hypotheses(
    hypotheses: Sequence[_TrackHypothesis],
) -> tuple[_TrackHypothesis, ...]:
    if not hypotheses:
        return ()
    endpoints = sorted({endpoint for hypothesis in hypotheses for endpoint in hypothesis.endpoints})
    endpoint_row = {endpoint: row for row, endpoint in enumerate(endpoints)}
    incidence = np.zeros((len(endpoints), len(hypotheses)), dtype=np.float64)
    for column, hypothesis in enumerate(hypotheses):
        for endpoint in hypothesis.endpoints:
            incidence[endpoint_row[endpoint], column] = 1.0
    # The epsilon only stabilises exact objective ties and is many orders below
    # score precision; it cannot change a physically distinct solution.
    utilities = np.asarray([hypothesis.utility for hypothesis in hypotheses], dtype=np.float64)
    tie_break = np.arange(len(hypotheses), dtype=np.float64) * 1.0e-12
    result = milp(
        c=-(utilities - tie_break),
        integrality=np.ones(len(hypotheses), dtype=np.int8),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(incidence, -np.inf, np.ones(len(endpoints))),
        options={"disp": False},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"multi-station assignment MILP failed: {result.message}")
    return tuple(
        hypothesis
        for hypothesis, selected in zip(hypotheses, result.x)
        if float(selected) > 0.5
    )


def multistation_score_assignment(
    event: EventTracklets,
    station_matrices: Mapping[tuple[int, int], tuple[ScoreMatrix, Sequence[FieldCandidate]]],
    config: MultiStationAssignmentConfig,
) -> MultiStationAssignmentResult:
    """Solve a truth-free, unit-capacity four-station association problem.

    ``station_matrices`` is keyed by ordered station pair and must originate
    from the same event.  Missing candidate edges remain absent; no coordinate
    surrogate or truth label is introduced here.
    """
    _validate_config(config)
    if not station_matrices:
        raise ValueError("station_matrices cannot be empty")
    stations = sorted({station for pair in station_matrices for station in pair})
    edge_lookups = {
        pair: _matrix_edge_lookup(matrix, candidates, config.score_threshold)
        for pair, (matrix, candidates) in station_matrices.items()
    }
    candidate_edges = int(sum(len(values) for values in edge_lookups.values()))
    hypotheses = _hypotheses(
        event,
        stations,
        edge_lookups,
        config.unmatched_penalty,
        config.maximum_hypotheses,
    )
    selected = _select_disjoint_hypotheses(hypotheses)
    matches_by_pair: dict[tuple[int, int], list[ScoredMatch]] = {
        pair: [] for pair in station_matrices
    }
    used_by_station: dict[int, set[int]] = {station: set() for station in stations}
    for hypothesis in selected:
        for station, index in hypothesis.endpoints:
            used_by_station[station].add(index)
        for pair, match in hypothesis.matches:
            matches_by_pair[pair].append(match)
    unmatched_by_station = {
        station: tuple(
            index
            for index in event.indices_for_station(station).tolist()
            if int(index) not in used_by_station[station]
        )
        for station in stations
    }
    return MultiStationAssignmentResult(
        matches_by_pair={pair: tuple(values) for pair, values in matches_by_pair.items()},
        unmatched_by_station=unmatched_by_station,
        candidate_edges_above_threshold=candidate_edges,
        hypotheses_considered=len(hypotheses),
        selected_hypotheses=len(selected),
    )
