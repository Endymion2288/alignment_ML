from __future__ import annotations

import numpy as np
import pytest

from baselines.field_chi2_matching import FieldCandidate
from baselines.global_assignment import full_score_matrix
from baselines.route_assignment import RouteAssignmentConfig, _CONTINUATION_TIE_BREAK, adjacent_route_assignment
from datasets.root_loader import EventTracklets
from evaluation.route_metrics import _unique_truth_by_station
from training.route_operating_audit import DUSTBIN_UTILITY, pair_tables_from_sets, solver_log_odds
from training.solver_hard_negative_audit import (
    PACKING_MARGIN,
    align_origin_twins,
    attach_solver_hard_negative,
    audit_workbook59_miner_does_not_unroll_solver,
    classify_hard_negative,
    miner_rivals,
    overlapping_production_hypotheses,
    production_hypotheses,
    recommend_next_objective,
    summarize_twin_efficiency_gap,
    twin_alignment_key,
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


def _pair_tables_and_sets(event: EventTracklets, scores_by_pair: dict[tuple[int, int], list[float]]):
    sets = []
    raw = []
    for source_station, target_station in ADJACENT:
        source = event.indices_for_station(source_station).tolist()
        target = event.indices_for_station(target_station).tolist()
        candidates = [
            _candidate(source[0], target[0], source_station, target_station),
            _candidate(source[1], target[1], source_station, target_station),
        ]
        scores = np.asarray(scores_by_pair[(source_station, target_station)], dtype=np.float64)
        labels = np.asarray(
            [
                event.truth_particle_id[source[0]] == event.truth_particle_id[target[0]],
                event.truth_particle_id[source[1]] == event.truth_particle_id[target[1]],
            ],
            dtype=bool,
        )
        sample = type("S", (), {"source_id": "src", "payload_id": "iteration_00_draw_01"})()
        candidate_set = type(
            "C",
            (),
            {
                "sample": sample,
                "event": event,
                "station_pair": (source_station, target_station),
                "candidates": candidates,
                "labels": labels,
            },
        )()
        sets.append(candidate_set)
        raw.append(scores)
    tables = pair_tables_from_sets(sets, raw, raw)
    key = next(iter(tables))
    return tables[key], sets, raw


def test_own_partial_and_cross_truth_composition():
    event = _event()
    _, unique = _unique_truth_by_station(event, (0, 1, 2, 3))
    truth = ((0, 0), (1, 2), (2, 4), (3, 6))
    own = classify_hard_negative(((2, 4), (3, 6)), event, unique, truth_endpoints=truth, truth_id=1)
    assert own["topology"] == "two_station_23"
    assert own["composition"] == "own_truth_partial"
    assert own["involves_2_3"] is True
    other = classify_hard_negative(((0, 1), (1, 3), (2, 5)), event, unique, truth_endpoints=truth, truth_id=1)
    assert other["composition"] == "other_truth_consistent_partial"
    mixed = classify_hard_negative(
        ((0, 0), (1, 3), (2, 4), (3, 6)), event, unique, truth_endpoints=truth, truth_id=1
    )
    assert mixed["composition"] == "mixed_route"
    assert mixed["cross_truth_endpoint_conflict"] is True


def test_production_hypotheses_are_solver_admitted_only():
    event = _event()
    scores = {
        (0, 1): [0.90, 0.20],
        (1, 2): [0.90, 0.20],
        (2, 3): [0.90, 0.20],
    }
    tables, sets, raw = _pair_tables_and_sets(event, scores)
    config = RouteAssignmentConfig(
        score_threshold_by_pair={pair: 0.001 for pair in ADJACENT},
        unmatched_penalty=-1.0,
    )
    matrices = {}
    for candidate_set, values in zip(sets, raw):
        pair = tuple(int(item) for item in candidate_set.station_pair)
        matrices[pair] = (
            full_score_matrix(event, candidate_set.candidates, values.tolist(), pair[0], pair[1]),
            candidate_set.candidates,
        )
    routes = production_hypotheses(event, matrices, config)
    assert routes
    assert all(float(route.utility) > DUSTBIN_UTILITY for route in routes)
    truth = ((0, 0), (1, 2), (2, 4), (3, 6))
    overlapping = overlapping_production_hypotheses(routes, truth)
    assert overlapping
    assert all(
        tuple((int(station), int(index)) for station, index in route.endpoints) != truth
        for route in overlapping
    )


def test_miner_sees_solver_winner_when_edges_exist_and_misses_when_absent():
    event = _event()
    scores = {
        (0, 1): [0.88, 0.12],
        (1, 2): [0.88, 0.12],
        (2, 3): [0.88, 0.88],
    }
    tables, _, _ = _pair_tables_and_sets(event, scores)
    truth = ((0, 0), (1, 2), (2, 4), (3, 6))
    rivals = miner_rivals(tables, truth)
    feasible = {tuple((item["endpoints"][0]["station"], item["endpoints"][0]["index"]),) for item in rivals}
    two_station = [item for item in rivals if item["topology"] == "two_station_23" and item["threshold_feasible"]]
    assert two_station
    missing_tables = {
        pair: {
            key: record
            for key, record in table.items()
            if not (pair == (2, 3) and key == (5, 7))
        }
        for pair, table in tables.items()
    }
    reduced = miner_rivals(missing_tables, truth)
    assert not any(
        item["topology"] == "two_station_23"
        and item["endpoints"][0]["index"] == 5
        and item["threshold_feasible"]
        for item in reduced
    )
    del feasible


def test_missed_training_signal_when_easy_miner_hides_solver_winner():
    event = _event()
    _, unique = _unique_truth_by_station(event, (0, 1, 2, 3))
    sample = type("S", (), {"source_id": "src", "payload_id": "iteration_00_draw_01"})()
    sets = []
    raw = []
    for pair, extras in (
        ((0, 1), []),
        ((1, 2), []),
        ((2, 3), [((5, 6), 0.99)]),
    ):
        source = event.indices_for_station(pair[0]).tolist()
        target = event.indices_for_station(pair[1]).tolist()
        candidates = [
            _candidate(source[0], target[0], pair[0], pair[1]),
            _candidate(source[1], target[1], pair[0], pair[1]),
        ]
        scores = [0.881, 0.12]
        labels = [
            bool(event.truth_particle_id[source[0]] == event.truth_particle_id[target[0]]),
            bool(event.truth_particle_id[source[1]] == event.truth_particle_id[target[1]]),
        ]
        for (src, tgt), score in extras:
            candidates.append(_candidate(src, tgt, pair[0], pair[1]))
            scores.append(score)
            labels.append(bool(event.truth_particle_id[src] == event.truth_particle_id[tgt]))
        sets.append(
            type(
                "C",
                (),
                {
                    "sample": sample,
                    "event": event,
                    "station_pair": pair,
                    "candidates": candidates,
                    "labels": np.asarray(labels, dtype=bool),
                },
            )()
        )
        raw.append(np.asarray(scores, dtype=np.float64))
    tables = pair_tables_from_sets(sets, raw, raw)
    pair_tables = next(iter(tables.values()))
    config = RouteAssignmentConfig(
        score_threshold_by_pair={pair: 0.001 for pair in ADJACENT},
        unmatched_penalty=-1.0,
    )
    matrices = {}
    for candidate_set, values in zip(sets, raw):
        pair = tuple(int(item) for item in candidate_set.station_pair)
        matrices[pair] = (
            full_score_matrix(event, candidate_set.candidates, values.tolist(), pair[0], pair[1]),
            candidate_set.candidates,
        )
    hypotheses = production_hypotheses(event, matrices, config)
    assigned = adjacent_route_assignment(event, matrices, config)
    hidden = {
        pair: {key: record for key, record in table.items() if key != (5, 6)}
        for pair, table in pair_tables.items()
    }
    u_truth = 3 * solver_log_odds(0.881) - 4.0
    row = attach_solver_hard_negative(
        {
            "endpoints": [
                {"station": 0, "index": 0},
                {"station": 1, "index": 2},
                {"station": 2, "index": 4},
                {"station": 3, "index": 6},
            ],
            "complete_truth_route_utility": u_truth,
            "selected": False,
            "truth_id": 1,
            "sample_id": "src",
            "payload_id": "iteration_00_draw_01",
            "run_id": 1,
            "event_id": 2,
        },
        event=event,
        unique_by_index=unique,
        pair_tables=hidden,
        hypotheses=hypotheses,
        selected_routes=assigned.routes,
        n_complete_truth_in_event=2,
        margin=PACKING_MARGIN,
    )
    assert u_truth > 1.0
    assert row["production_fragment_winner"] is True
    assert row["winner_in_mined_negative_set"] is False
    assert row["winner_selected_as_strongest_miner"] is False
    assert row["missed_training_signal"] is True
    assert row["dustbin_aware_margin_loss"] <= 1.0e-6
    assert row["oracle_loss_exceeds_miner"] is True


def test_recommend_uncovered_winner_opens_solver_in_the_loop():
    decision = recommend_next_objective(
        {
            "production_fragment_winners": 10,
            "winner_in_mined_negative_set": 3,
            "winner_selected_as_strongest_miner": 2,
            "missed_training_signal": 4,
        }
    )
    assert decision["continue_to_15d_relative_wls"] is False
    assert decision["pre_register_solver_in_the_loop_control"] is True
    assert decision["next_step"] == "design_solver_in_the_loop_hard_negative_control"


def test_recommend_full_coverage_does_not_train_second_model():
    decision = recommend_next_objective(
        {
            "production_fragment_winners": 10,
            "winner_in_mined_negative_set": 10,
            "winner_selected_as_strongest_miner": 10,
            "missed_training_signal": 0,
        }
    )
    assert decision["pre_register_solver_in_the_loop_control"] is False
    assert decision["next_step"] == "diagnose_loss_weighting_per_event_reduction_and_route_length_bias"
    assert decision["new_checkpoint_authorized"] is False


def test_production_hypotheses_replace_complete_route_scores_and_leave_fragments():
    event = _event()
    scores = {
        (0, 1): [0.90, 0.20],
        (1, 2): [0.90, 0.20],
        (2, 3): [0.90, 0.20],
    }
    _, sets, raw = _pair_tables_and_sets(event, scores)
    config = RouteAssignmentConfig(
        score_threshold_by_pair={pair: 0.001 for pair in ADJACENT},
        unmatched_penalty=-1.0,
        complete_route_score_composition="replace",
    )
    matrices = {}
    for candidate_set, values in zip(sets, raw):
        pair = tuple(int(item) for item in candidate_set.station_pair)
        matrices[pair] = (
            full_score_matrix(event, candidate_set.candidates, values.tolist(), pair[0], pair[1]),
            candidate_set.candidates,
        )
    complete_map = {(0, 2, 4, 6): 0.999, (1, 3, 5, 7): 0.999}
    routes_none = production_hypotheses(event, matrices, config)
    routes_v4 = production_hypotheses(event, matrices, config, complete_route_scores=complete_map)
    truth = ((0, 0), (1, 2), (2, 4), (3, 6))
    r4_none = next(route for route in routes_none if tuple(route.endpoints) == truth)
    r4_v4 = next(route for route in routes_v4 if tuple(route.endpoints) == truth)
    assert r4_none.complete_route_score is None
    assert r4_v4.complete_route_score == pytest.approx(0.999)
    expected = solver_log_odds(0.999) + 4.0 * (-1.0) + 3.0 * _CONTINUATION_TIE_BREAK
    assert r4_v4.utility == pytest.approx(expected)
    assert r4_v4.utility != pytest.approx(r4_none.utility)
    frags_none = [route for route in routes_none if len(route.endpoints) < 4]
    frags_v4 = [route for route in routes_v4 if len(route.endpoints) < 4]
    assert len(frags_none) == len(frags_v4)
    for left, right in zip(frags_none, frags_v4):
        assert left.endpoints == right.endpoints
        assert left.utility == pytest.approx(right.utility)
        assert right.complete_route_score is None


def test_competing_four_station_winner_physical_uses_l_corrected_not_edge_hops():
    event = _event()
    _, unique = _unique_truth_by_station(event, (0, 1, 2, 3))
    sample = type("S", (), {"source_id": "src", "payload_id": "iteration_00_draw_00"})()
    sets = []
    raw = []
    extras = {
        (2, 3): [((4, 7), 0.20)],
    }
    for pair in ADJACENT:
        source = event.indices_for_station(pair[0]).tolist()
        target = event.indices_for_station(pair[1]).tolist()
        candidates = [
            _candidate(source[0], target[0], pair[0], pair[1]),
            _candidate(source[1], target[1], pair[0], pair[1]),
        ]
        scores = [0.90, 0.20]
        labels = [
            bool(event.truth_particle_id[source[0]] == event.truth_particle_id[target[0]]),
            bool(event.truth_particle_id[source[1]] == event.truth_particle_id[target[1]]),
        ]
        for (src, tgt), score in extras.get(pair, []):
            candidates.append(_candidate(src, tgt, pair[0], pair[1]))
            scores.append(score)
            labels.append(bool(event.truth_particle_id[src] == event.truth_particle_id[tgt]))
        sets.append(
            type(
                "C",
                (),
                {
                    "sample": sample,
                    "event": event,
                    "station_pair": pair,
                    "candidates": candidates,
                    "labels": np.asarray(labels, dtype=bool),
                },
            )()
        )
        raw.append(np.asarray(scores, dtype=np.float64))
    tables = pair_tables_from_sets(sets, raw, raw)
    pair_tables = next(iter(tables.values()))
    config = RouteAssignmentConfig(
        score_threshold_by_pair={pair: 0.001 for pair in ADJACENT},
        unmatched_penalty=-1.0,
        complete_route_score_composition="replace",
    )
    matrices = {}
    for candidate_set, values in zip(sets, raw):
        pair = tuple(int(item) for item in candidate_set.station_pair)
        matrices[pair] = (
            full_score_matrix(event, candidate_set.candidates, values.tolist(), pair[0], pair[1]),
            candidate_set.candidates,
        )
    complete_map = {
        (0, 2, 4, 6): 0.985,
        (1, 3, 5, 7): 0.50,
        (0, 2, 4, 7): 0.999,
    }
    routes_none = production_hypotheses(event, matrices, config)
    routes_v4 = production_hypotheses(event, matrices, config, complete_route_scores=complete_map)
    competitor = ((0, 0), (1, 2), (2, 4), (3, 7))
    assert not any(tuple(route.endpoints) == competitor for route in routes_none)
    winner = next(route for route in routes_v4 if tuple(route.endpoints) == competitor)
    assert winner.complete_route_score == pytest.approx(0.999)
    assigned = adjacent_route_assignment(event, matrices, config, complete_route_scores=complete_map)
    u_truth = solver_log_odds(0.985) + 4.0 * (-1.0)
    row = attach_solver_hard_negative(
        {
            "endpoints": [
                {"station": 0, "index": 0},
                {"station": 1, "index": 2},
                {"station": 2, "index": 4},
                {"station": 3, "index": 6},
            ],
            "complete_truth_route_utility": u_truth,
            "selected": False,
            "truth_id": 1,
            "sample_id": "src",
            "payload_id": "iteration_00_draw_00",
            "run_id": 1,
            "event_id": 2,
        },
        event=event,
        unique_by_index=unique,
        pair_tables=pair_tables,
        hypotheses=routes_v4,
        selected_routes=assigned.routes,
        n_complete_truth_in_event=2,
        margin=PACKING_MARGIN,
    )
    packing_physical = float(winner.utility) - _CONTINUATION_TIE_BREAK * 3.0
    hop_physical = 2.0 * solver_log_odds(0.90) + solver_log_odds(0.20) + 4.0 * (-1.0)
    assert packing_physical == pytest.approx(solver_log_odds(0.999) + 4.0 * (-1.0))
    assert packing_physical != pytest.approx(hop_physical)
    assert row["u_best_solver_fragment"] == pytest.approx(packing_physical)
    assert row["solver_competitor"]["physical_utility"] == pytest.approx(packing_physical)
    assert row["solver_competitor"]["n_stations"] == 4


def _origin_row(truth_id: int, *, selected: bool, u_truth: float, u_fragment: float, run_id: int = 1, event_id: int = 2):
    # Overlay origin_tracklet_id is the station index, so two tracks in one
    # original event share this signature and must be split by truth_id.
    return {
        "truth_id": truth_id,
        "run_id": run_id,
        "event_id": event_id,
        "sample_id": "pooled_validation",
        "origin_signature": ((9000000000, 24, 0), (9000000000, 24, 1), (9000000000, 24, 2), (9000000000, 24, 3)),
        "selected": selected,
        "u_truth": u_truth,
        "u_best_solver_fragment": u_fragment,
        "solver_competitor": {"topology": "two_station_23", "composition": "own_truth_partial"},
        "winner_selected_as_strongest_miner": False,
    }


def test_twin_alignment_uses_origin_plus_truth_id_not_synthetic_event():
    chart = [
        _origin_row(1, selected=True, u_truth=1.2, u_fragment=0.4, event_id=4),
        _origin_row(2, selected=True, u_truth=0.8, u_fragment=0.3, event_id=4),
    ]
    twin = [
        _origin_row(2, selected=False, u_truth=0.5, u_fragment=0.6, event_id=17),
        _origin_row(1, selected=False, u_truth=0.9, u_fragment=0.45, event_id=17),
    ]
    aligned, stats = align_origin_twins(chart, twin)
    assert stats["aligned"] == 2
    assert stats["unmatched_chart"] == 0
    assert stats["unmatched_twin"] == 0
    by_truth = {int(row["truth_id"]): row for row in aligned}
    assert by_truth[1]["delta_u_truth_twin_minus_chart"] == pytest.approx(-0.3)
    assert by_truth[2]["delta_u_fragment_twin_minus_chart"] == pytest.approx(0.3)
    assert twin_alignment_key(chart[0])[1] == 1


def test_twin_gap_attributes_truth_drop_or_fragment_rise():
    aligned = [
        {
            "chart_selected": True,
            "twin_selected": False,
            "delta_u_truth_twin_minus_chart": -0.40,
            "delta_u_fragment_twin_minus_chart": 0.05,
        },
        {
            "chart_selected": True,
            "twin_selected": False,
            "delta_u_truth_twin_minus_chart": -0.30,
            "delta_u_fragment_twin_minus_chart": 0.02,
        },
        {
            "chart_selected": True,
            "twin_selected": True,
            "delta_u_truth_twin_minus_chart": 0.01,
            "delta_u_fragment_twin_minus_chart": 0.01,
        },
    ]
    summary = summarize_twin_efficiency_gap(aligned)
    assert summary["chart_selected_twin_lost"] == 2
    assert summary["attribution"] == "truth_utility_drop"


def test_workbook59_miner_source_does_not_call_solver():
    audit = audit_workbook59_miner_does_not_unroll_solver()
    assert audit["calls_production_solver"] is False
    assert audit["uses_max_of_enumerated_rivals"] is True
