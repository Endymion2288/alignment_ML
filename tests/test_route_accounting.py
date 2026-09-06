"""Workbook-77 route_accounting_v2: fragments are not complete fakes."""

from __future__ import annotations

import numpy as np

from baselines.field_chi2_matching import FieldCandidate, ScoredMatch
from baselines.global_assignment import full_score_matrix
from baselines.route_assignment import Route, RouteAssignmentResult
from datasets.root_loader import EventTracklets
from evaluation.route_accounting import ACCOUNTING_VERSION, assess_route_accounting_v2
from evaluation.review_gates import GateSpec, evaluate_gates


ADJACENT = ((0, 1), (1, 2), (2, 3))
STATIONS = (0, 1, 2, 3)


def _event() -> EventTracklets:
    return EventTracklets(
        run_id=1,
        event_id=2,
        station_id=np.repeat(np.arange(4, dtype=np.int16), 2),
        tracklet_id=np.arange(8, dtype=np.int32),
        z_mm=np.repeat(np.arange(4, dtype=np.float64), 2),
        state=np.zeros((8, 4), dtype=np.float64),
        covariance=np.tile(np.eye(4, dtype=np.float64), (8, 1, 1)),
        chi2=np.ones(8, dtype=np.float64),
        ndof=np.ones(8, dtype=np.float64),
        n_hit=np.full(8, 3, dtype=np.int16),
        hit_pattern=np.full(8, 0b111, dtype=np.uint64),
        truth_particle_id=np.asarray([1, 2, 1, 2, 1, 2, 1, 2], dtype=np.int64),
        truth_pdg=np.full(8, 13, dtype=np.int32),
        truth_match_fraction=np.ones(8, dtype=np.float64),
    )


def _candidate(source: int, target: int, source_station: int, target_station: int) -> FieldCandidate:
    return FieldCandidate(
        source_index=source,
        target_index=target,
        source_station=source_station,
        target_station=target_station,
        chi2=1.0,
        residual=np.zeros(4, dtype=np.float64),
        pull=np.zeros(4, dtype=np.float64),
        combined_covariance=np.eye(4, dtype=np.float64),
    )


def _matrices(event: EventTracklets):
    result = {}
    for source_station, target_station in ADJACENT:
        source = event.indices_for_station(source_station).tolist()
        target = event.indices_for_station(target_station).tolist()
        candidates = [
            _candidate(source[0], target[0], source_station, target_station),
            _candidate(source[1], target[1], source_station, target_station),
        ]
        result[(source_station, target_station)] = (
            full_score_matrix(event, candidates, [0.9, 0.9], source_station, target_station),
            candidates,
        )
    return result


def _match(source: int, target: int) -> ScoredMatch:
    return ScoredMatch(source_index=source, target_index=target, chi2=1.0, score=0.9)


def _route(endpoints: tuple[tuple[int, int], ...], matches: tuple[tuple[tuple[int, int], ScoredMatch], ...]) -> Route:
    return Route(endpoints=endpoints, matches=matches, utility=1.0)


def _result(routes: list[Route]) -> RouteAssignmentResult:
    matches_by_pair = {pair: [] for pair in ADJACENT}
    used = {station: set() for station in STATIONS}
    for route in routes:
        for pair, match in route.matches:
            matches_by_pair[pair].append(match)
        for station, index in route.endpoints:
            used[station].add(index)
    return RouteAssignmentResult(
        routes=tuple(routes),
        matches_by_pair={pair: tuple(values) for pair, values in matches_by_pair.items()},
        unmatched_by_station={
            station: tuple(index for index in (0, 1, 2, 3, 4, 5, 6, 7) if index not in used[station] and index // 2 == station)
            for station in STATIONS
        },
        candidate_edges_above_threshold=6,
        candidate_edges_above_threshold_by_pair={pair: 2 for pair in ADJACENT},
        hypotheses_considered=len(routes),
        selected_routes=len(routes),
    )


def _thresholds() -> dict[tuple[int, int], float]:
    return {pair: 0.5 for pair in ADJACENT}


def test_complete_truth_selection_has_zero_complete_fake_and_no_fragmentation():
    event = _event()
    routes = [
        _route(
            ((0, 0), (1, 2), (2, 4), (3, 6)),
            (((0, 1), _match(0, 2)), ((1, 2), _match(2, 4)), ((2, 3), _match(4, 6))),
        ),
        _route(
            ((0, 1), (1, 3), (2, 5), (3, 7)),
            (((0, 1), _match(1, 3)), ((1, 2), _match(3, 5)), ((2, 3), _match(5, 7))),
        ),
    ]
    metrics = assess_route_accounting_v2(event, _result(routes), _matrices(event), STATIONS, _thresholds()).as_dict()
    assert metrics["metric_version"] == ACCOUNTING_VERSION
    assert metrics["correct_complete_routes"] == 2
    assert metrics["fake_complete_routes"] == 0
    assert metrics["complete_fake_rate"] == 0.0
    assert metrics["selected_fragment_routes"] == 0
    assert metrics["fragmented_truth_chains"] == 0
    assert metrics["unmatched_truth_chains"] == 0
    assert metrics["all_route_purity"] == 1.0
    assert metrics["all_route_fake_rate"] == 0.0


def test_correct_disjoint_fragments_are_not_complete_fakes():
    event = _event()
    routes = [
        _route(((0, 0), (1, 2)), (((0, 1), _match(0, 2)),)),
        _route(((2, 4), (3, 6)), (((2, 3), _match(4, 6)),)),
    ]
    accounting = assess_route_accounting_v2(event, _result(routes), _matrices(event), STATIONS, _thresholds())
    metrics = accounting.as_dict()
    assert metrics["complete_truth_chains"] == 2
    assert metrics["selected_routes"] == 2
    assert metrics["selected_complete_routes"] == 0
    assert metrics["correct_complete_routes"] == 0
    assert metrics["fake_complete_routes"] == 0
    assert metrics["complete_fake_rate"] is None
    assert metrics["truth_consistent_fragments"] == 2
    assert metrics["incorrect_fragments"] == 0
    assert metrics["fragmented_truth_chains"] == 1
    assert metrics["unmatched_truth_chains"] == 1
    assert metrics["all_route_purity"] == 1.0
    assert metrics["all_route_fake_rate"] == 0.0
    # Historical script bug: both correct fragments become "fakes".
    assert metrics["legacy_selected_minus_correct_complete"] == 2
    assert accounting.historical.selected_routes - accounting.historical.truth_consistent_routes == 0


def test_mixed_complete_route_is_a_complete_fake_not_a_fragment_error():
    event = _event()
    routes = [
        _route(
            ((0, 0), (1, 3), (2, 4), (3, 6)),
            (((0, 1), _match(0, 3)), ((1, 2), _match(3, 4)), ((2, 3), _match(4, 6))),
        )
    ]
    matrices = _matrices(event)
    # The mixed 0->3 edge is absent from the default truth-only matrices.
    extra = _candidate(0, 3, 0, 1)
    old_matrix, old_candidates = matrices[(0, 1)]
    candidates = list(old_candidates) + [extra]
    scores = [0.9, 0.9, 0.8]
    matrices = dict(matrices)
    matrices[(0, 1)] = (full_score_matrix(event, candidates, scores, 0, 1), candidates)
    metrics = assess_route_accounting_v2(event, _result(routes), matrices, STATIONS, _thresholds()).as_dict()
    assert metrics["selected_complete_routes"] == 1
    assert metrics["correct_complete_routes"] == 0
    assert metrics["fake_complete_routes"] == 1
    assert metrics["complete_fake_rate"] == 1.0
    assert metrics["mixed_or_ambiguous_routes"] == 1
    assert metrics["all_route_fake_rate"] == 1.0
    assert metrics["unmatched_truth_chains"] == 2


def test_missing_truth_leaves_rates_undefined_for_fail_closed_gates():
    event = EventTracklets(
        run_id=1,
        event_id=2,
        station_id=np.repeat(np.arange(4, dtype=np.int16), 2),
        tracklet_id=np.arange(8, dtype=np.int32),
        z_mm=np.repeat(np.arange(4, dtype=np.float64), 2),
        state=np.zeros((8, 4), dtype=np.float64),
        covariance=np.tile(np.eye(4, dtype=np.float64), (8, 1, 1)),
        chi2=np.ones(8, dtype=np.float64),
        ndof=np.ones(8, dtype=np.float64),
        n_hit=np.full(8, 3, dtype=np.int16),
        hit_pattern=np.full(8, 0b111, dtype=np.uint64),
    )
    routes = [_route(((0, 0), (1, 2)), (((0, 1), _match(0, 2)),))]
    metrics = assess_route_accounting_v2(event, _result(routes), _matrices(event), STATIONS, _thresholds()).as_dict()
    assert metrics["route_events_without_truth"] == 1
    assert metrics["complete_fake_rate"] is None
    assert metrics["all_route_purity"] is None
    report = evaluate_gates(
        metrics,
        [GateSpec(name="complete_fake", metric="complete_fake_rate", op="<=", bound=0.05)],
        required_metric_version=ACCOUNTING_VERSION,
    )
    assert report["gate_pass"] is False
    assert report["fail_closed"] is True
