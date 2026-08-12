"""Truth-evaluation metrics for adjacent four-station route assignment.

The functions here never feed truth back to the route solver.  They separate
the physical candidate graph, thresholded frozen MLP scores, and selected
unit-capacity routes so a failure can be attributed to the appropriate stage.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from baselines.field_chi2_matching import FieldCandidate
from baselines.global_assignment import ScoreMatrix
from baselines.route_assignment import RouteAssignmentResult, adjacent_station_pairs
from datasets.root_loader import EventTracklets


@dataclass
class RouteMetrics:
    """Additive, truth-only accounting for route-level evaluation."""

    events: int = 0
    events_without_truth: int = 0
    complete_truth_chains: int = 0
    candidate_retained_complete_truth_chains: int = 0
    score_retained_complete_truth_chains: int = 0
    selected_complete_routes: int = 0
    correct_complete_routes: int = 0
    selected_routes: int = 0
    truth_consistent_routes: int = 0
    routes_with_fake_endpoint: int = 0
    mixed_or_ambiguous_routes: int = 0
    duplicate_routes: int = 0
    missing_station_boundaries: int = 0
    correctly_dustbinned_missing_station_boundaries: int = 0

    def add(self, other: "RouteMetrics") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, int(getattr(self, name)) + int(getattr(other, name)))

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float | None:
        return None if not denominator else float(numerator / denominator)

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "route_events": self.events,
            "route_events_without_truth": self.events_without_truth,
            "complete_truth_chains": self.complete_truth_chains,
            "candidate_retained_complete_truth_chains": self.candidate_retained_complete_truth_chains,
            "candidate_complete_truth_chain_recall": self._ratio(
                self.candidate_retained_complete_truth_chains, self.complete_truth_chains
            ),
            "score_retained_complete_truth_chains": self.score_retained_complete_truth_chains,
            "score_threshold_complete_truth_chain_recall": self._ratio(
                self.score_retained_complete_truth_chains, self.complete_truth_chains
            ),
            "selected_complete_routes": self.selected_complete_routes,
            "correct_complete_routes": self.correct_complete_routes,
            "complete_track_efficiency": self._ratio(
                self.correct_complete_routes, self.complete_truth_chains
            ),
            "complete_track_purity": self._ratio(
                self.correct_complete_routes, self.selected_complete_routes
            ),
            "selected_routes": self.selected_routes,
            "truth_consistent_routes": self.truth_consistent_routes,
            "track_purity": self._ratio(self.truth_consistent_routes, self.selected_routes),
            # The primary route fake rate follows the inclusive convention
            # used by the pairwise baselines: every non-truth-consistent
            # selected route is false, whether it contains a synthetic fake
            # endpoint or joins incompatible genuine tracklets.
            "track_fake_rate": self._ratio(
                self.selected_routes - self.truth_consistent_routes,
                self.selected_routes,
            ),
            "routes_with_fake_endpoint": self.routes_with_fake_endpoint,
            "fake_endpoint_route_rate": self._ratio(
                self.routes_with_fake_endpoint, self.selected_routes
            ),
            "mixed_or_ambiguous_routes": self.mixed_or_ambiguous_routes,
            "duplicate_routes": self.duplicate_routes,
            "duplicate_rate": self._ratio(self.duplicate_routes, self.selected_routes),
            "missing_station_boundaries": self.missing_station_boundaries,
            "correctly_dustbinned_missing_station_boundaries": self.correctly_dustbinned_missing_station_boundaries,
            "missing_station_recovery": self._ratio(
                self.correctly_dustbinned_missing_station_boundaries,
                self.missing_station_boundaries,
            ),
            "complete_four_station_truth_chain_retention": self._ratio(
                self.correct_complete_routes, self.complete_truth_chains
            ),
        }


def _unique_truth_by_station(
    event: EventTracklets, station_path: Sequence[int]
) -> tuple[dict[int, dict[int, int]], dict[int, dict[int, int]]]:
    """Return unique truth-to-index and index-to-truth maps by station."""
    if event.truth_particle_id is None:
        return {}, {}
    by_truth: dict[int, dict[int, int]] = {}
    by_index: dict[int, dict[int, int]] = {}
    for station in station_path:
        indices = event.indices_for_station(int(station)).tolist()
        labels = [int(event.truth_particle_id[int(index)]) for index in indices]
        counts = Counter(label for label in labels if label >= 0)
        unique = {
            label: int(index)
            for index, label in zip(indices, labels)
            if label >= 0 and counts[label] == 1
        }
        by_truth[int(station)] = unique
        by_index[int(station)] = {index: truth for truth, index in unique.items()}
    return by_truth, by_index


def _complete_truth_chains(
    unique_by_truth: Mapping[int, Mapping[int, int]], station_path: Sequence[int]
) -> dict[int, tuple[tuple[int, int], ...]]:
    if not station_path:
        return {}
    common = set(unique_by_truth[int(station_path[0])])
    for station in station_path[1:]:
        common.intersection_update(unique_by_truth[int(station)])
    return {
        int(truth): tuple((int(station), int(unique_by_truth[int(station)][truth])) for station in station_path)
        for truth in common
    }


def assess_complete_route_score_retention(
    event: EventTracklets,
    station_path: Sequence[int],
    complete_route_scores: Mapping[tuple[int, ...], float],
    threshold: float | None,
) -> dict[str, int | float | None]:
    """Audit a V2 complete-route score map without changing assignment.

    The map is truth-free at inference and contains only chains already made
    from three adjacent physical candidate edges.  Truth is used solely here
    to distinguish raw candidate retention from retention after the frozen
    route-score threshold.
    """
    stations = tuple(int(station) for station in station_path)
    if threshold is not None and (
        not np.isfinite(float(threshold)) or float(threshold) < 0.0 or float(threshold) > 1.0
    ):
        raise ValueError("complete route score threshold must be finite and in [0, 1]")
    if event.truth_particle_id is None:
        return {
            "route_events": 1,
            "route_events_without_truth": 1,
            "complete_truth_chains": 0,
            "raw_route_candidate_retained_complete_truth_chains": 0,
            "raw_route_candidate_complete_truth_chain_recall": None,
            "route_score_retained_complete_truth_chains": 0,
            "route_score_threshold_complete_truth_chain_recall": None,
            "route_score_threshold": threshold,
        }
    unique_by_truth, _ = _unique_truth_by_station(event, stations)
    complete = _complete_truth_chains(unique_by_truth, stations)
    raw_retained = 0
    threshold_retained = 0
    for endpoints in complete.values():
        key = tuple(int(index) for _, index in endpoints)
        score = complete_route_scores.get(key)
        if score is None:
            continue
        value = float(score)
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("complete route score map contains an invalid probability")
        raw_retained += 1
        threshold_retained += int(threshold is None or value >= float(threshold))
    denominator = len(complete)
    ratio = lambda numerator: None if not denominator else float(numerator / denominator)
    return {
        "route_events": 1,
        "route_events_without_truth": 0,
        "complete_truth_chains": denominator,
        "raw_route_candidate_retained_complete_truth_chains": raw_retained,
        "raw_route_candidate_complete_truth_chain_recall": ratio(raw_retained),
        "route_score_retained_complete_truth_chains": threshold_retained,
        "route_score_threshold_complete_truth_chain_recall": ratio(threshold_retained),
        "route_score_threshold": threshold,
    }


def _candidate_edge_sets(
    station_matrices: Mapping[tuple[int, int], tuple[ScoreMatrix, Sequence[FieldCandidate]]]
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    return {
        pair: {(int(candidate.source_index), int(candidate.target_index)) for candidate in candidates}
        for pair, (_, candidates) in station_matrices.items()
    }


def _matrix_score_lookup(matrix: ScoreMatrix) -> dict[tuple[int, int], float]:
    result: dict[tuple[int, int], float] = {}
    for row, source_index in enumerate(matrix.source_indices):
        for column, target_index in enumerate(matrix.target_indices):
            value = float(matrix.values[row, column])
            if np.isfinite(value):
                result[(int(source_index), int(target_index))] = value
    return result


def assess_adjacent_route_assignment(
    event: EventTracklets,
    result: RouteAssignmentResult,
    station_matrices: Mapping[tuple[int, int], tuple[ScoreMatrix, Sequence[FieldCandidate]]],
    station_path: Sequence[int],
    score_threshold_by_pair: Mapping[tuple[int, int], float],
) -> RouteMetrics:
    """Assess one route solve without changing any truth-free decision.

    A complete truth chain has exactly one known truth endpoint in every
    station.  Candidate retention asks whether all three physical adjacent
    edges exist.  Score retention additionally applies the already frozen
    MLP thresholds.  Selected-route metrics are then evaluated independently.
    """
    stations = tuple(int(station) for station in station_path)
    pairs = adjacent_station_pairs(stations)
    if set(station_matrices) != set(pairs):
        raise ValueError("route metric matrices do not match the adjacent station path")
    if {tuple(int(value) for value in pair) for pair in score_threshold_by_pair} != set(pairs):
        raise ValueError("route metric thresholds do not match the adjacent station path")
    metrics = RouteMetrics(events=1)
    if event.truth_particle_id is None:
        metrics.events_without_truth = 1
        return metrics

    unique_by_truth, unique_by_index = _unique_truth_by_station(event, stations)
    complete = _complete_truth_chains(unique_by_truth, stations)
    metrics.complete_truth_chains = len(complete)
    candidate_edges = _candidate_edge_sets(station_matrices)
    score_edges = {
        pair: _matrix_score_lookup(station_matrices[pair][0]) for pair in pairs
    }

    for endpoints in complete.values():
        by_station = {station: index for station, index in endpoints}
        raw_retained = all(
            (by_station[source], by_station[target]) in candidate_edges[(source, target)]
            for source, target in pairs
        )
        score_retained = raw_retained and all(
            score_edges[(source, target)].get((by_station[source], by_station[target]), -np.inf)
            >= float(score_threshold_by_pair[(source, target)])
            for source, target in pairs
        )
        metrics.candidate_retained_complete_truth_chains += int(raw_retained)
        metrics.score_retained_complete_truth_chains += int(score_retained)

    truth_routes: defaultdict[int, int] = defaultdict(int)
    expected_complete = {endpoints: truth for truth, endpoints in complete.items()}
    for route in result.routes:
        metrics.selected_routes += 1
        endpoint_tuple = tuple((int(station), int(index)) for station, index in route.endpoints)
        complete_route = tuple(station for station, _ in endpoint_tuple) == stations
        metrics.selected_complete_routes += int(complete_route)
        labels = [unique_by_index.get(station, {}).get(index) for station, index in endpoint_tuple]
        raw_labels = [int(event.truth_particle_id[index]) for _, index in endpoint_tuple]
        has_fake = any(label < 0 for label in raw_labels)
        metrics.routes_with_fake_endpoint += int(has_fake)
        truth_id: int | None = None
        if not has_fake and all(label is not None for label in labels) and len(set(labels)) == 1:
            truth_id = int(labels[0])
            metrics.truth_consistent_routes += 1
            truth_routes[truth_id] += 1
        else:
            metrics.mixed_or_ambiguous_routes += 1
        if complete_route and truth_id is not None and expected_complete.get(endpoint_tuple) == truth_id:
            metrics.correct_complete_routes += 1

    metrics.duplicate_routes = sum(max(count - 1, 0) for count in truth_routes.values())

    selected_edges = {
        pair: {(int(match.source_index), int(match.target_index)) for match in result.matches_by_pair[pair]}
        for pair in pairs
    }
    for source_station, target_station in pairs:
        source_unique = unique_by_truth[source_station]
        target_unique = unique_by_truth[target_station]
        for truth, source_index in source_unique.items():
            if truth in target_unique:
                continue
            metrics.missing_station_boundaries += 1
            crossed = any(left == source_index for left, _ in selected_edges[(source_station, target_station)])
            metrics.correctly_dustbinned_missing_station_boundaries += int(not crossed)
        for truth, target_index in target_unique.items():
            if truth in source_unique:
                continue
            metrics.missing_station_boundaries += 1
            crossed = any(right == target_index for _, right in selected_edges[(source_station, target_station)])
            metrics.correctly_dustbinned_missing_station_boundaries += int(not crossed)
    return metrics
