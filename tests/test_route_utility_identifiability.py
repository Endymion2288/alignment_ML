from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from baselines.field_chi2_matching import FieldCandidate
from baselines.global_assignment import full_score_matrix
from baselines.route_assignment import RouteAssignmentConfig, _matrix_edge_lookup, adjacent_route_assignment
from datasets.root_loader import EventTracklets
from training.route_operating_audit import DUSTBIN_UTILITY, audit_event_truth_chains, pair_tables_from_sets
from training.route_utility_identifiability import (
    attach_truth_route_identifiability,
    classify_route_topology,
    classify_utility_decision,
    enumerate_threshold_feasible_routes,
    training_negative_mining_coverage,
)


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
        origin_run_id=np.full(8, 10, dtype=np.int64),
        origin_event_id=np.full(8, 20, dtype=np.int64),
        origin_tracklet_id=np.asarray([1, 2, 1, 2, 1, 2, 1, 2], dtype=np.int64),
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


def _lookups(event: EventTracklets, scores_by_pair: dict[tuple[int, int], list[float]], threshold: float):
    lookups = {}
    for source_station, target_station in ADJACENT:
        source = event.indices_for_station(source_station).tolist()
        target = event.indices_for_station(target_station).tolist()
        candidates = [
            _candidate(source[0], target[0], source_station, target_station),
            _candidate(source[1], target[1], source_station, target_station),
        ]
        scores = list(scores_by_pair[(source_station, target_station)])
        matrix = full_score_matrix(event, candidates, scores, source_station, target_station)
        lookups[(source_station, target_station)] = _matrix_edge_lookup(matrix, candidates, threshold)
    return lookups


def test_topology_names_prefix_suffix_and_two_station_fragments():
    assert classify_route_topology(((0, 0), (1, 2), (2, 4))) == "three_station_prefix"
    assert classify_route_topology(((1, 2), (2, 4), (3, 6))) == "three_station_suffix"
    assert classify_route_topology(((2, 4), (3, 6))) == "two_station_23"


def test_dustbin_swallows_when_truth_beats_negative_fragments():
    assert (
        classify_utility_decision(
            selected=False,
            score_retained=True,
            candidate_retained=True,
            u_truth=-2.0,
            u_best_competitor=-8.0,
            u_dustbin=0.0,
        )
        == "truth_beats_wrong_routes_but_loses_to_dustbin"
    )


def test_fragment_win_requires_positive_rival_above_dustbin():
    assert (
        classify_utility_decision(
            selected=False,
            score_retained=True,
            candidate_retained=True,
            u_truth=1.0,
            u_best_competitor=2.5,
            u_dustbin=0.0,
        )
        == "truth_loses_to_fragment"
    )


def test_margin_over_dustbin_and_rivals_with_no_selection_is_solver_mismatch():
    assert (
        classify_utility_decision(
            selected=False,
            score_retained=True,
            candidate_retained=True,
            u_truth=1.5,
            u_best_competitor=0.2,
            u_dustbin=0.0,
        )
        == "margin_satisfied_solver_does_not_select"
    )


def test_training_miner_misses_dustbin_when_any_feasible_fragment_exists():
    coverage = training_negative_mining_coverage("three_station_suffix", 3)
    assert coverage["topology_in_training_enumerator"] is True
    assert coverage["dustbin_in_training_negative_set"] is False
    assert coverage["dustbin_winner_missing_from_training_negatives"] is True


def test_low_probability_complete_truth_is_dustbined_despite_ranking():
    event = _event()
    # All truth edges pass 0.001 but have p~0.2, so Σlogit + 4*(-1) << 0.
    scores = {
        (0, 1): [0.20, 0.05],
        (1, 2): [0.20, 0.05],
        (2, 3): [0.20, 0.05],
    }
    lookups = _lookups(event, scores, 0.001)
    config = RouteAssignmentConfig(
        score_threshold_by_pair={pair: 0.001 for pair in ADJACENT},
        unmatched_penalty=-1.0,
    )
    matrices = {
        pair: (
            full_score_matrix(
                event,
                [
                    _candidate(event.indices_for_station(pair[0])[0], event.indices_for_station(pair[1])[0], *pair),
                    _candidate(event.indices_for_station(pair[0])[1], event.indices_for_station(pair[1])[1], *pair),
                ],
                scores[pair],
                *pair,
            ),
            [
                _candidate(event.indices_for_station(pair[0])[0], event.indices_for_station(pair[1])[0], *pair),
                _candidate(event.indices_for_station(pair[0])[1], event.indices_for_station(pair[1])[1], *pair),
            ],
        )
        for pair in ADJACENT
    }
    result = adjacent_route_assignment(event, matrices, config)
    assert result.selected_routes == 0
    sets = []
    raw = []
    for pair, (matrix, candidates) in matrices.items():
        labels = np.asarray(
            [
                event.truth_particle_id[candidate.source_index] == event.truth_particle_id[candidate.target_index]
                for candidate in candidates
            ],
            dtype=bool,
        )
        values = np.asarray(scores[pair], dtype=np.float64)
        sets.append(
            SimpleNamespace(
                sample=SimpleNamespace(source_id="src", payload_id="nominal"),
                event=event,
                station_pair=pair,
                candidates=tuple(candidates),
                labels=labels,
            )
        )
        raw.append(values)
    tables = pair_tables_from_sets(sets, raw, raw)[("src", "nominal", 1, 2)]
    truth_rows = audit_event_truth_chains(event, matrices, tables, config, result)
    feasible = enumerate_threshold_feasible_routes(event, lookups, -1.0)
    assert any(route["topology"] == "three_station_suffix" for route in feasible)
    attached = [attach_truth_route_identifiability(row, feasible, -1.0) for row in truth_rows]
    assert all(row["decision_class"] == "truth_beats_wrong_routes_but_loses_to_dustbin" for row in attached)
    assert all(row["dustbin_is_production_winner"] for row in attached)
    assert all(row["dustbin_winner_missed_by_training_miner"] for row in attached)
    assert all(row["u_truth"] < DUSTBIN_UTILITY for row in attached)
    # The truth-consistent 3-station fragment is also below dustbin, but less
    # negative than the complete route: unmatched_penalty * n_stations dominates
    # when logits are negative.  Production still selects nothing.
    assert all(row["u_best_competitor"] > row["u_truth"] for row in attached)
    assert all(row["truth_outranks_overlapping_competitors"] is False for row in attached)
    assert all(
        str(row["best_competitor_topology"]).startswith(("two_station", "three_station"))
        for row in attached
    )
    assert all(row["best_competitor_production_admitted"] is False for row in attached)
