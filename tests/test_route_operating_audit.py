from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from baselines.field_chi2_matching import FieldCandidate
from baselines.global_assignment import full_score_matrix
from baselines.route_assignment import RouteAssignmentConfig, adjacent_route_assignment
from datasets.root_loader import EventTracklets
from training.route_operating_audit import (
    DUSTBIN_UTILITY,
    align_gauge_twins,
    audit_event_truth_chains,
    classify_truth_chain_loss,
    complete_route_packing_utility,
    pair_tables_from_sets,
    solver_log_odds,
    summarize_truth_rows,
)


ADJACENT = ((0, 1), (1, 2), (2, 3))


def _event(*, origin: bool = False) -> EventTracklets:
    kwargs = dict(
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
    if origin:
        kwargs["origin_run_id"] = np.full(8, 10, dtype=np.int64)
        kwargs["origin_event_id"] = np.full(8, 20, dtype=np.int64)
        kwargs["origin_tracklet_id"] = np.asarray([1, 2, 1, 2, 1, 2, 1, 2], dtype=np.int64)
    return EventTracklets(**kwargs)


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


def _matrices(event: EventTracklets, scores_by_pair: dict[tuple[int, int], list[float]] | None = None):
    result = {}
    for source_station, target_station in ADJACENT:
        source = event.indices_for_station(source_station).tolist()
        target = event.indices_for_station(target_station).tolist()
        candidates = [
            _candidate(source[0], target[0], source_station, target_station),
            _candidate(source[1], target[1], source_station, target_station),
        ]
        scores = [0.9, 0.9] if scores_by_pair is None else list(scores_by_pair[(source_station, target_station)])
        result[(source_station, target_station)] = (
            full_score_matrix(event, candidates, scores, source_station, target_station),
            candidates,
        )
    return result


def _tables(event: EventTracklets, matrices):
    sets = []
    raw = []
    calibrated = []
    for pair, (matrix, candidates) in matrices.items():
        labels = np.asarray(
            [
                event.truth_particle_id[candidate.source_index]
                == event.truth_particle_id[candidate.target_index]
                and event.truth_particle_id[candidate.source_index] >= 0
                for candidate in candidates
            ],
            dtype=bool,
        )
        scores = np.asarray(
            [
                float(matrix.values[list(matrix.source_indices).index(candidate.source_index)][
                    list(matrix.target_indices).index(candidate.target_index)
                ])
                for candidate in candidates
            ],
            dtype=np.float64,
        )
        sets.append(
            SimpleNamespace(
                sample=SimpleNamespace(source_id="src", payload_id="chart"),
                event=event,
                station_pair=pair,
                candidates=tuple(candidates),
                labels=labels,
            )
        )
        raw.append(scores)
        calibrated.append(scores)
    grouped = pair_tables_from_sets(sets, raw, calibrated)
    key = ("src", "chart", int(event.run_id), int(event.event_id))
    return grouped[key]


def _config(penalty: float = 0.0, threshold: float = 0.5) -> RouteAssignmentConfig:
    return RouteAssignmentConfig(
        score_threshold_by_pair={pair: float(threshold) for pair in ADJACENT},
        unmatched_penalty=float(penalty),
    )


def test_complete_route_utility_matches_solver_log_odds_plus_dustbin_term():
    utility = complete_route_packing_utility([0.9, 0.8, 0.7], unmatched_penalty=0.5)
    expected = solver_log_odds(0.9) + solver_log_odds(0.8) + solver_log_odds(0.7) + 4 * 0.5
    assert utility == pytest.approx(expected)
    assert DUSTBIN_UTILITY == 0.0


def test_loss_stage_priority_is_threshold_then_dustbin_then_competition():
    assert (
        classify_truth_chain_loss(
            candidate_retained=True,
            score_retained=False,
            failed_pairs=["2->3"],
            packing_utility=3.0,
            selected=False,
        )
        == "below_station_pair_threshold"
    )
    assert (
        classify_truth_chain_loss(
            candidate_retained=True,
            score_retained=True,
            failed_pairs=[],
            packing_utility=-0.1,
            selected=False,
        )
        == "utility_nonpositive"
    )
    assert (
        classify_truth_chain_loss(
            candidate_retained=True,
            score_retained=True,
            failed_pairs=[],
            packing_utility=2.0,
            selected=False,
        )
        == "packing_competition"
    )
    assert (
        classify_truth_chain_loss(
            candidate_retained=True,
            score_retained=True,
            failed_pairs=[],
            packing_utility=2.0,
            selected=True,
        )
        == "selected"
    )


def test_audit_marks_selected_complete_truth_chains():
    event = _event()
    matrices = _matrices(event)
    config = _config(penalty=0.0, threshold=0.5)
    result = adjacent_route_assignment(event, matrices, config)
    rows = audit_event_truth_chains(event, matrices, _tables(event, matrices), config, result)
    summary = summarize_truth_rows(rows)
    assert summary["complete_truth_chains"] == 2
    assert summary["loss_stage_counts"]["selected"] == 2
    assert all(row["truth_edges_all_source_rank_one"] for row in rows)


def test_audit_attributes_threshold_loss_to_the_failing_station_pair():
    event = _event()
    scores = {
        (0, 1): [0.9, 0.9],
        (1, 2): [0.9, 0.9],
        (2, 3): [0.2, 0.9],
    }
    matrices = _matrices(event, scores)
    config = _config(penalty=0.0, threshold=0.5)
    result = adjacent_route_assignment(event, matrices, config)
    rows = audit_event_truth_chains(event, matrices, _tables(event, matrices), config, result)
    by_truth = {int(row["truth_id"]): row for row in rows}
    assert by_truth[1]["loss_stage"] == "below_station_pair_threshold"
    assert by_truth[1]["failed_pairs"] == ["2->3"]
    assert by_truth[1]["selected"] is False
    assert by_truth[2]["loss_stage"] == "selected"


def test_audit_attributes_dustbin_when_complete_utility_is_nonpositive():
    event = _event()
    scores = {
        (0, 1): [0.55, 0.55],
        (1, 2): [0.55, 0.55],
        (2, 3): [0.55, 0.55],
    }
    matrices = _matrices(event, scores)
    config = _config(penalty=-1.0, threshold=0.5)
    result = adjacent_route_assignment(event, matrices, config)
    assert result.selected_routes == 0
    rows = audit_event_truth_chains(event, matrices, _tables(event, matrices), config, result)
    assert all(row["loss_stage"] == "utility_nonpositive" for row in rows)
    assert all(row["complete_truth_route_utility"] < 0.0 for row in rows)


def test_gauge_twin_alignment_requires_shared_origin_and_event_ids():
    event = _event(origin=True)
    matrices = _matrices(event)
    config = _config()
    result = adjacent_route_assignment(event, matrices, config)
    chart = audit_event_truth_chains(event, matrices, _tables(event, matrices), config, result)
    twin = [dict(row, payload_id="twin") for row in chart]
    for row in chart:
        row["payload_id"] = "chart"
    aligned = align_gauge_twins(chart, twin)
    assert len(aligned) == 2
    assert all(not item["stage_changed"] for item in aligned)


def test_train_only_calibration_tag_cannot_be_read_as_validation_platt():
    from training.operating_layer_control import (
        historical_packing_config,
        nominal_quality_ok,
        tag_train_only_calibration,
    )

    tagged = tag_train_only_calibration(
        {
            "fit_split": "validation_only",
            "by_station_pair": {"2->3": {"fit_split": "validation_only", "slope": 0.7}},
        }
    )
    assert tagged["fit_split"] == "train_only"
    assert tagged["by_station_pair"]["2->3"]["fit_split"] == "train_only"
    config = historical_packing_config(
        unmatched_penalty=-1.0,
        pair_thresholds={"0->1": 0.001, "1->2": 0.001, "2->3": 0.001},
        maximum_hypotheses=1000,
    )
    assert config.unmatched_penalty == -1.0
    assert config.score_threshold_by_pair[(1, 2)] == 0.001
    two_station = complete_route_packing_utility([0.40], unmatched_penalty=0.5, n_stations=2)
    two_station_historical = complete_route_packing_utility([0.40], unmatched_penalty=-1.0, n_stations=2)
    assert two_station > 0.0
    assert two_station_historical < 0.0
    assert nominal_quality_ok(
        {"track_fake_rate": 0.049, "complete_track_purity": 0.95},
        maximum_track_fake_rate=0.05,
        minimum_complete_track_purity=0.95,
    )["ok"]
    assert not nominal_quality_ok(
        {"track_fake_rate": 0.103, "complete_track_purity": 0.948},
        maximum_track_fake_rate=0.05,
        minimum_complete_track_purity=0.95,
    )["ok"]
