"""Workbook 65: tests for the read-only blind failure localization audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from baselines.field_chi2_matching import FieldCandidate
from baselines.global_assignment import full_score_matrix
from baselines.route_assignment import RouteAssignmentConfig, adjacent_route_assignment
from datasets.root_loader import EventTracklets
from scripts.audit_four_station_blind_failure_localization import (
    CHECKPOINT_SHA256,
    LOSS_STAGE_TO_PROBLEM,
    _failure_problems,
    _namespace_map,
    _source_uid,
)
from training.route_operating_audit import (
    complete_route_packing_utility,
    solver_log_odds,
)
from training.route_reduction_audit import production_margin
from training.source_diversity_audit import (
    RESERVED_BLIND_SOURCES,
    assert_sources_allowed,
    is_sealed_source,
)


ADJACENT = ((0, 1), (1, 2), (2, 3))


def test_checkpoint_sha_matches_frozen_workbook64_contract():
    assert CHECKPOINT_SHA256 == "0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236"


def test_loss_stage_map_covers_every_pre_registered_stage():
    assert set(LOSS_STAGE_TO_PROBLEM) == {
        "candidate_missing",
        "below_station_pair_threshold",
        "utility_nonpositive",
        "packing_competition",
        "selected",
    }


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
        origin_run_id=np.full(8, 9000000000, dtype=np.int64),
        origin_event_id=np.full(8, 44, dtype=np.int64),
        origin_tracklet_id=np.asarray([1, 2, 1, 2, 1, 2, 1, 2], dtype=np.int64),
    )


def _matrices(event: EventTracklets, scores_by_pair):
    result = {}
    for source_station, target_station in ADJACENT:
        source = event.indices_for_station(source_station).tolist()
        target = event.indices_for_station(target_station).tolist()
        candidates = tuple(
            FieldCandidate(
                source_index=source[index],
                target_index=target[index],
                source_station=source_station,
                target_station=target_station,
                chi2=1.0,
                residual=np.zeros(4, dtype=np.float64),
                pull=np.zeros(4, dtype=np.float64),
                combined_covariance=np.eye(4, dtype=np.float64),
            )
            for index in range(2)
        )
        result[(source_station, target_station)] = (
            full_score_matrix(
                event,
                candidates,
                list(scores_by_pair[(source_station, target_station)]),
                source_station,
                target_station,
            ),
            candidates,
        )
    return result


def test_production_margin_matches_u_truth_minus_max_fragment_dustbin():
    assert production_margin(1.5, 0.7) == pytest.approx(0.8)
    assert production_margin(1.5, -2.0) == pytest.approx(1.5)
    assert production_margin(1.5, None) == pytest.approx(1.5)
    assert production_margin(None, 0.7) is None


def test_complete_route_utility_is_sum_of_log_odds_plus_four_penalties():
    probabilities = [0.9, 0.8, 0.7]
    penalty = -1.0
    utility = complete_route_packing_utility(probabilities, penalty)
    expected = (
        solver_log_odds(0.9)
        + solver_log_odds(0.8)
        + solver_log_odds(0.7)
        + 4 * penalty
    )
    assert utility == pytest.approx(expected)
    # The frozen operating point: p=1/3 on all three edges sits exactly at
    # the dustbin (3*log(1/2 ratio... )) — verify sign behaviour explicitly.
    assert complete_route_packing_utility([0.5, 0.5, 0.5], -1.0) == pytest.approx(3 * 0.0 - 4.0)


def test_solver_packing_prefers_truth_when_all_edges_rank_first():
    event = _event()
    scores = {
        (0, 1): [0.9, 0.1],
        (1, 2): [0.9, 0.1],
        (2, 3): [0.9, 0.1],
    }
    matrices = _matrices(event, scores)
    config = RouteAssignmentConfig(
        score_threshold_by_pair={pair: 0.001 for pair in ADJACENT},
        unmatched_penalty=-1.0,
    )
    result = adjacent_route_assignment(event, matrices, config)
    selected_complete = [
        tuple(station for station, _ in route.endpoints) for route in result.routes
    ]
    assert (0, 1, 2, 3) in selected_complete
    truth_route = next(
        route
        for route in result.routes
        if tuple(station for station, _ in route.endpoints) == (0, 1, 2, 3)
        and [int(event.truth_particle_id[route.endpoints[i][1]]) for i in range(4)].count(1) == 4
    )
    assert truth_route.utility > 0.0


def test_failure_problems_map_threshold_loss_to_scale_when_edges_rank_first():
    edge_rank_one = {
        "loss_stage": "below_station_pair_threshold",
        "edges": [
            {
                "in_candidate_graph": True,
                "truth_rank_among_source_calibrated": 1,
                "truth_rank_among_pair_calibrated": 1,
            }
        ],
    }
    assert _failure_problems(edge_rank_one) == ["C"]


def test_failure_problems_map_threshold_loss_to_ranking_and_scale_when_rank_slips():
    edge_rank_slipped = {
        "loss_stage": "below_station_pair_threshold",
        "edges": [
            {
                "in_candidate_graph": True,
                "truth_rank_among_source_calibrated": 2,
                "truth_rank_among_pair_calibrated": 1,
            }
        ],
    }
    assert _failure_problems(edge_rank_slipped) == ["B", "C"]


def test_failure_problems_pass_selected_routes_through():
    assert _failure_problems({"loss_stage": "selected"}) == []
    assert _failure_problems({"loss_stage": "candidate_missing"}) == ["A"]
    assert _failure_problems({"loss_stage": "utility_nonpositive"}) == ["C"]
    assert _failure_problems({"loss_stage": "packing_competition"}) == ["D"]


def test_namespace_map_and_source_uid_recover_blind_uids(tmp_path: Path):
    descriptor = tmp_path / "pooled_physical_descriptor.json"
    descriptor.write_text(
        json.dumps(
            {
                "origin_namespaces": [
                    {
                        "source_id": "mc24_100047_00350_00399",
                        "run_id_mapping": [
                            {"namespaced_run_id": 9000000000, "original_run_id": 100047}
                        ],
                    },
                    {
                        "source_id": "mc24_100048_00350_00399",
                        "run_id_mapping": [
                            {"namespaced_run_id": 9000000100, "original_run_id": 100048}
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    class _Sample:
        physical_payload_manifest = descriptor

    mapping = _namespace_map([_Sample()])
    assert mapping[9000000000] == "mc24_100047_00350_00399"
    row = {"origin_signature": ((9000000000, 44, 0), (9000000000, 44, 7))}
    assert _source_uid(row, mapping) == "mc24_100047_00350_00399:9000000000:44"


def test_sealed_sources_are_never_admitted_and_blind_needs_explicit_opt_in():
    assert is_sealed_source("mc24_100116_00050_00099")
    with pytest.raises(ValueError):
        assert_sources_allowed(["mc24_100116_00050_00099"])
    with pytest.raises(ValueError):
        # Reserved blind sources stay forbidden without the explicit opt-in.
        assert_sources_allowed(set(RESERVED_BLIND_SOURCES))
    assert_sources_allowed(set(RESERVED_BLIND_SOURCES), allow_reserved_blind=True)
