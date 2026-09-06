"""Versioned route accounting that does not treat correct fragments as fakes.

Historical ``RouteMetrics.track_fake_rate`` uses ``selected - truth_consistent``
and is therefore already all-route, not complete-only.  Several later
evaluation scripts instead set ``fake_selected_routes = selected_routes -
correct_complete_routes``, which counts a correct 2/3-station fragment as a
fake.  ``route_accounting_v2`` keeps the historical ``RouteMetrics`` object
untouched and reports the split counters new gates must use:

* complete fake rate / complete purity / complete efficiency
* fragmentation of unmatched complete truth
* unmatched truth nodes
* clone / mixed conflict
* all-route purity and fake rate

Ratios are ``None`` when the denominator is zero so fail-closed gates cannot
treat a missing rate as a pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from baselines.field_chi2_matching import FieldCandidate
from baselines.global_assignment import ScoreMatrix
from baselines.route_assignment import Route, RouteAssignmentResult, adjacent_station_pairs
from datasets.root_loader import EventTracklets
from evaluation.route_metrics import (
    RouteMetrics,
    _complete_truth_chains,
    _unique_truth_by_station,
    assess_adjacent_route_assignment,
)


ACCOUNTING_VERSION = "route_accounting_v2"


def _ratio(numerator: int, denominator: int) -> float | None:
    if int(denominator) <= 0:
        return None
    return float(numerator) / float(denominator)


def _endpoint_tuple(route: Route) -> tuple[tuple[int, int], ...]:
    return tuple((int(station), int(index)) for station, index in route.endpoints)


def _route_truth_id(
    event: EventTracklets,
    unique_by_index: Mapping[int, Mapping[int, int]],
    endpoints: Sequence[tuple[int, int]],
) -> tuple[int | None, bool, bool]:
    """Return ``(truth_id, truth_consistent, has_fake_endpoint)``."""
    if event.truth_particle_id is None:
        return None, False, False
    raw_labels = [int(event.truth_particle_id[index]) for _, index in endpoints]
    has_fake = any(label < 0 for label in raw_labels)
    unique_labels = [unique_by_index.get(station, {}).get(index) for station, index in endpoints]
    consistent = (
        not has_fake
        and all(label is not None for label in unique_labels)
        and len(set(unique_labels)) == 1
    )
    truth_id = int(unique_labels[0]) if consistent else None
    return truth_id, consistent, has_fake


@dataclass
class RouteAccountingV2:
    """Additive, versioned accounting for one or more events."""

    metric_version: str = ACCOUNTING_VERSION
    events: int = 0
    events_without_truth: int = 0
    complete_truth_chains: int = 0
    selected_routes: int = 0
    selected_complete_routes: int = 0
    selected_fragment_routes: int = 0
    correct_complete_routes: int = 0
    fake_complete_routes: int = 0
    truth_consistent_routes: int = 0
    truth_consistent_fragments: int = 0
    incorrect_fragments: int = 0
    fragmented_truth_chains: int = 0
    unmatched_truth_chains: int = 0
    unmatched_truth_nodes: int = 0
    mixed_or_ambiguous_routes: int = 0
    duplicate_routes: int = 0
    routes_with_fake_endpoint: int = 0
    historical: RouteMetrics = field(default_factory=RouteMetrics)

    def add(self, other: "RouteAccountingV2") -> None:
        if other.metric_version != self.metric_version:
            raise ValueError("cannot add route accounting objects from different metric versions")
        skip = {"metric_version", "historical"}
        for name in self.__dataclass_fields__:
            if name in skip:
                continue
            setattr(self, name, int(getattr(self, name)) + int(getattr(other, name)))
        self.historical.add(other.historical)

    def as_dict(self) -> dict[str, int | float | str | None]:
        return {
            "metric_version": self.metric_version,
            "route_events": self.events,
            "route_events_without_truth": self.events_without_truth,
            "complete_truth_chains": self.complete_truth_chains,
            "selected_routes": self.selected_routes,
            "selected_complete_routes": self.selected_complete_routes,
            "selected_fragment_routes": self.selected_fragment_routes,
            "correct_complete_routes": self.correct_complete_routes,
            "fake_complete_routes": self.fake_complete_routes,
            "complete_track_efficiency": _ratio(self.correct_complete_routes, self.complete_truth_chains),
            "complete_track_purity": _ratio(self.correct_complete_routes, self.selected_complete_routes),
            "complete_fake_rate": _ratio(self.fake_complete_routes, self.selected_complete_routes),
            "truth_consistent_routes": self.truth_consistent_routes,
            "truth_consistent_fragments": self.truth_consistent_fragments,
            "incorrect_fragments": self.incorrect_fragments,
            "fragmented_truth_chains": self.fragmented_truth_chains,
            "fragmentation_rate": _ratio(self.fragmented_truth_chains, self.complete_truth_chains),
            "unmatched_truth_chains": self.unmatched_truth_chains,
            "unmatched_truth_chain_rate": _ratio(self.unmatched_truth_chains, self.complete_truth_chains),
            "unmatched_truth_nodes": self.unmatched_truth_nodes,
            "mixed_or_ambiguous_routes": self.mixed_or_ambiguous_routes,
            "conflict_rate": _ratio(self.mixed_or_ambiguous_routes, self.selected_routes),
            "duplicate_routes": self.duplicate_routes,
            "clone_rate": _ratio(self.duplicate_routes, self.selected_routes),
            "routes_with_fake_endpoint": self.routes_with_fake_endpoint,
            "all_route_purity": _ratio(self.truth_consistent_routes, self.selected_routes),
            "all_route_fake_rate": _ratio(
                self.selected_routes - self.truth_consistent_routes, self.selected_routes
            ),
            # Documented historical mis-count.  Not a supported gate metric.
            "legacy_selected_minus_correct_complete": self.selected_routes - self.correct_complete_routes,
        }


def assess_route_accounting_v2(
    event: EventTracklets,
    result: RouteAssignmentResult,
    station_matrices: Mapping[tuple[int, int], tuple[ScoreMatrix, Sequence[FieldCandidate]]],
    station_path: Sequence[int],
    score_threshold_by_pair: Mapping[tuple[int, int], float],
) -> RouteAccountingV2:
    """Assess one solve.  Truth is used only after assignment is finished."""
    historical = assess_adjacent_route_assignment(
        event,
        result,
        station_matrices,
        station_path,
        score_threshold_by_pair,
    )
    stations = tuple(int(station) for station in station_path)
    accounting = RouteAccountingV2(events=1, historical=historical)
    if event.truth_particle_id is None:
        accounting.events_without_truth = 1
        accounting.selected_routes = int(historical.selected_routes)
        accounting.selected_complete_routes = int(historical.selected_complete_routes)
        accounting.selected_fragment_routes = int(
            historical.selected_routes - historical.selected_complete_routes
        )
        return accounting

    unique_by_truth, unique_by_index = _unique_truth_by_station(event, stations)
    complete = _complete_truth_chains(unique_by_truth, stations)
    accounting.complete_truth_chains = len(complete)
    expected_complete = {endpoints: truth for truth, endpoints in complete.items()}
    consistent_by_truth: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    covered_truth_nodes: set[tuple[int, int]] = set()

    for route in result.routes:
        endpoints = _endpoint_tuple(route)
        complete_route = tuple(station for station, _ in endpoints) == stations
        accounting.selected_routes += 1
        accounting.selected_complete_routes += int(complete_route)
        accounting.selected_fragment_routes += int(not complete_route)
        truth_id, consistent, has_fake = _route_truth_id(event, unique_by_index, endpoints)
        accounting.routes_with_fake_endpoint += int(has_fake)
        if consistent and truth_id is not None:
            accounting.truth_consistent_routes += 1
            consistent_by_truth.setdefault(truth_id, []).append(endpoints)
            covered_truth_nodes.update(endpoints)
            if complete_route and expected_complete.get(endpoints) == truth_id:
                accounting.correct_complete_routes += 1
            elif not complete_route:
                accounting.truth_consistent_fragments += 1
        else:
            accounting.mixed_or_ambiguous_routes += 1
            if complete_route:
                accounting.fake_complete_routes += 1
            else:
                accounting.incorrect_fragments += 1

    accounting.duplicate_routes = sum(max(len(rows) - 1, 0) for rows in consistent_by_truth.values())

    all_unique_truth_nodes = {
        (int(station), int(index))
        for station, by_truth in unique_by_truth.items()
        for index in by_truth.values()
    }
    accounting.unmatched_truth_nodes = len(all_unique_truth_nodes - covered_truth_nodes)

    for truth_id, endpoints in complete.items():
        selected_for_truth = consistent_by_truth.get(int(truth_id), ())
        if endpoints in selected_for_truth:
            continue
        if selected_for_truth:
            accounting.fragmented_truth_chains += 1
        else:
            accounting.unmatched_truth_chains += 1

    if accounting.correct_complete_routes != historical.correct_complete_routes:
        raise RuntimeError("route_accounting_v2 complete-correct count drifted from RouteMetrics")
    if accounting.truth_consistent_routes != historical.truth_consistent_routes:
        raise RuntimeError("route_accounting_v2 truth-consistent count drifted from RouteMetrics")
    if (
        accounting.fake_complete_routes + accounting.correct_complete_routes
        != accounting.selected_complete_routes
    ):
        raise RuntimeError("complete routes must split into correct or fake")
    if (
        accounting.truth_consistent_fragments + accounting.incorrect_fragments
        != accounting.selected_fragment_routes
    ):
        raise RuntimeError("fragment routes must split into consistent or incorrect")
    return accounting
