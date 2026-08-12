from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from baselines.field_chi2_matching import FieldCandidate
from baselines.global_assignment import full_score_matrix
from baselines.route_assignment import RouteAssignmentConfig, adjacent_route_assignment
from datasets.root_loader import EventTracklets
from evaluation.route_metrics import assess_adjacent_route_assignment
from evaluation.route_metrics import RouteMetrics
from training.route_assignment import evaluate_adjacent_route_assignment_sets


ADJACENT = ((0, 1), (1, 2), (2, 3))


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
        scores = [0.9, 0.9]
        if (source_station, target_station) == (0, 1):
            # A plausible cross edge must lose to the two fully supported
            # route chains under the fixed global route objective.
            candidates.append(_candidate(source[0], target[1], source_station, target_station))
            scores.append(0.8)
        result[(source_station, target_station)] = (
            full_score_matrix(event, candidates, scores, source_station, target_station),
            candidates,
        )
    return result


def _config(penalty: float = 0.0) -> RouteAssignmentConfig:
    return RouteAssignmentConfig(
        score_threshold_by_pair={pair: 0.5 for pair in ADJACENT},
        unmatched_penalty=penalty,
    )


def test_adjacent_route_assignment_prefers_two_complete_consistent_routes():
    event = _event()
    result = adjacent_route_assignment(event, _matrices(event), _config())

    assert result.selected_routes == 2
    assert all(len(route.endpoints) == 4 for route in result.routes)
    assert not any(result.unmatched_by_station.values())
    for pair, matches in result.matches_by_pair.items():
        assert {(match.source_index % 2, match.target_index % 2) for match in matches} == {
            (0, 0),
            (1, 1),
        }, pair


def test_adjacent_route_assignment_can_leave_everything_in_dustbin():
    event = _event()
    result = adjacent_route_assignment(event, _matrices(event), _config(penalty=-10.0))

    assert result.selected_routes == 0
    assert {station: len(rows) for station, rows in result.unmatched_by_station.items()} == {
        0: 2,
        1: 2,
        2: 2,
        3: 2,
    }


def test_complete_route_query_scores_replace_independent_edge_utility_only_for_full_routes():
    event = _event()
    matrices = _matrices(event)
    direct_scores = {
        # Truth chains remain high while the only cross-chain candidate is
        # below the direct route-score threshold.
        (0, 2, 4, 6): 0.99,
        (1, 3, 5, 7): 0.99,
        (0, 3, 5, 7): 0.001,
    }
    config = RouteAssignmentConfig(
        score_threshold_by_pair={pair: 0.5 for pair in ADJACENT},
        unmatched_penalty=0.0,
        complete_route_score_threshold=0.01,
    )
    result = adjacent_route_assignment(event, matrices, config, complete_route_scores=direct_scores)

    complete = [route for route in result.routes if len(route.endpoints) == 4]
    assert {tuple(index for _, index in route.endpoints) for route in complete} == {
        (0, 2, 4, 6),
        (1, 3, 5, 7),
    }
    assert all(route.complete_route_score is not None for route in complete)


def test_complete_route_query_residual_weight_zero_is_the_edge_only_solver():
    event = _event()
    matrices = _matrices(event)
    direct_scores = {
        (0, 2, 4, 6): 0.01,
        (1, 3, 5, 7): 0.01,
        (0, 3, 5, 7): 0.01,
    }
    edge_only = adjacent_route_assignment(event, matrices, _config())
    residual = adjacent_route_assignment(
        event,
        matrices,
        RouteAssignmentConfig(
            score_threshold_by_pair={pair: 0.5 for pair in ADJACENT},
            unmatched_penalty=0.0,
            complete_route_score_composition="residual",
            complete_route_context_weight=0.0,
        ),
        complete_route_scores=direct_scores,
    )

    assert [route.endpoints for route in residual.routes] == [route.endpoints for route in edge_only.routes]
    np.testing.assert_allclose(
        [route.utility for route in residual.routes],
        [route.utility for route in edge_only.routes],
    )


def test_complete_route_query_residual_preserves_edge_evidence_against_partial_decomposition():
    event = _event()
    matrices = _matrices(event)
    direct_scores = {
        # Direct replacement turns a 0.5 route query into zero log-odds.  The
        # edge-only evidence is deliberately much stronger, so replacement
        # loses to partial routes while a small residual correction retains the
        # complete truth chains.
        (0, 2, 4, 6): 0.5,
        (1, 3, 5, 7): 0.5,
        (0, 3, 5, 7): 0.5,
    }
    replacement = adjacent_route_assignment(
        event,
        matrices,
        RouteAssignmentConfig(
            score_threshold_by_pair={pair: 0.5 for pair in ADJACENT},
            unmatched_penalty=0.0,
            complete_route_score_composition="replace",
        ),
        complete_route_scores=direct_scores,
    )
    residual = adjacent_route_assignment(
        event,
        matrices,
        RouteAssignmentConfig(
            score_threshold_by_pair={pair: 0.5 for pair in ADJACENT},
            unmatched_penalty=0.0,
            complete_route_score_composition="residual",
            complete_route_context_weight=0.1,
        ),
        complete_route_scores=direct_scores,
    )

    assert not any(len(route.endpoints) == 4 for route in replacement.routes)
    assert sum(len(route.endpoints) == 4 for route in residual.routes) == 2


def test_route_metrics_separate_candidate_score_and_selected_retention():
    event = _event()
    matrices = _matrices(event)
    result = adjacent_route_assignment(event, matrices, _config())
    metrics = assess_adjacent_route_assignment(
        event,
        result,
        matrices,
        station_path=(0, 1, 2, 3),
        score_threshold_by_pair={pair: 0.5 for pair in ADJACENT},
    ).as_dict()

    assert metrics["complete_truth_chains"] == 2
    assert metrics["candidate_complete_truth_chain_recall"] == 1.0
    assert metrics["score_threshold_complete_truth_chain_recall"] == 1.0
    assert metrics["complete_track_efficiency"] == 1.0
    assert metrics["complete_track_purity"] == 1.0
    assert metrics["track_fake_rate"] == 0.0


def test_route_evaluator_uses_only_adjacent_station_pairs():
    event = _event()
    matrices = _matrices(event)
    sample = SimpleNamespace(source_id="test_source", payload_id="mag_0_test_00")
    sets = []
    scores = []
    for pair, (matrix, candidates) in matrices.items():
        source_rows = {index: row for row, index in enumerate(matrix.source_indices)}
        target_rows = {index: row for row, index in enumerate(matrix.target_indices)}
        values = np.asarray(
            [
                matrix.values[source_rows[candidate.source_index], target_rows[candidate.target_index]]
                for candidate in candidates
            ],
            dtype=np.float64,
        )
        sets.append(
            SimpleNamespace(
                sample=sample,
                event=event,
                station_pair=pair,
                candidates=tuple(candidates),
                labels=np.asarray(
                    [candidate.source_index % 2 == candidate.target_index % 2 for candidate in candidates],
                    dtype=bool,
                ),
            )
        )
        scores.append(values)

    evaluation = evaluate_adjacent_route_assignment_sets(sets, scores, _config(), calibration_bins=5)

    assert set(evaluation["by_station_pair"]) == {"0->1", "1->2", "2->3"}
    assert evaluation["route"]["complete_track_efficiency"] == 1.0


def test_route_fake_rate_is_inclusive_and_fake_endpoint_rate_is_separate():
    metrics = RouteMetrics(
        selected_routes=10,
        truth_consistent_routes=8,
        routes_with_fake_endpoint=1,
    ).as_dict()

    assert metrics["track_purity"] == 0.8
    assert metrics["track_fake_rate"] == 0.2
    assert metrics["fake_endpoint_route_rate"] == 0.1
