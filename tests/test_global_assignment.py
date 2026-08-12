from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from baselines.field_chi2_matching import FieldCandidate, candidate_feature_matrix, pair_feature_names
from baselines.global_assignment import (
    AssignmentConfig,
    full_score_matrix,
    global_score_assignment,
)
from datasets.root_loader import EventTracklets
from training.global_assignment import (
    choose_global_operating_point,
    evaluate_global_assignment_sets,
    evaluate_station_pair_assignment_sets,
    prepare_global_assignment_context,
)


def _event() -> EventTracklets:
    return EventTracklets(
        run_id=1,
        event_id=2,
        station_id=np.asarray([0, 0, 1, 1], dtype=np.int16),
        tracklet_id=np.arange(4, dtype=np.int32),
        z_mm=np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
        state=np.zeros((4, 4), dtype=np.float64),
        covariance=np.tile(np.eye(4, dtype=np.float64), (4, 1, 1)),
        chi2=np.ones(4, dtype=np.float64),
        ndof=np.ones(4, dtype=np.float64),
        n_hit=np.full(4, 3, dtype=np.int16),
        hit_pattern=np.full(4, 0b111, dtype=np.uint64),
        truth_particle_id=np.asarray([1, 2, 1, 2], dtype=np.int64),
        truth_pdg=np.full(4, 13, dtype=np.int32),
        truth_match_fraction=np.ones(4, dtype=np.float64),
    )


def _candidate(source: int, target: int, chi2: float) -> FieldCandidate:
    return FieldCandidate(
        source_index=source,
        target_index=target,
        source_station=0,
        target_station=1,
        chi2=chi2,
        residual=np.zeros(4, dtype=np.float64),
        pull=np.zeros(4, dtype=np.float64),
        combined_covariance=np.eye(4, dtype=np.float64),
    )


def test_hungarian_uses_global_score_matrix_not_greedy_order():
    event = _event()
    candidates = (
        _candidate(0, 2, 1.0),
        _candidate(0, 3, 2.0),
        _candidate(1, 2, 3.0),
        _candidate(1, 3, 4.0),
    )
    scores = np.asarray([0.90, 0.80, 0.85, 0.10])
    result = global_score_assignment(
        event,
        candidates,
        scores,
        0,
        1,
        AssignmentConfig(method="hungarian", score_threshold=0.0),
    )
    assert {(match.source_index, match.target_index) for match in result.matches} == {(0, 3), (1, 2)}
    assert not result.unmatched_sources
    assert not result.unmatched_targets


def test_dustbin_assignment_can_leave_low_score_endpoints_unmatched():
    event = _event()
    candidates = (_candidate(0, 2, 1.0), _candidate(1, 3, 1.0))
    result = global_score_assignment(
        event,
        candidates,
        np.asarray([0.40, 0.45]),
        0,
        1,
        AssignmentConfig(
            method="dustbin_hungarian",
            score_threshold=0.0,
            unmatched_penalty=-2.0,
        ),
    )
    assert not result.matches
    assert result.unmatched_sources == (0, 1)
    assert result.unmatched_targets == (2, 3)


def test_sinkhorn_rounding_respects_dustbin_and_candidate_mask():
    event = _event()
    candidates = (
        _candidate(0, 2, 1.0),
        _candidate(0, 3, 2.0),
        _candidate(1, 2, 3.0),
    )
    scores = np.asarray([0.90, 0.80, 0.85])
    result = global_score_assignment(
        event,
        candidates,
        scores,
        0,
        1,
        AssignmentConfig(
            method="sinkhorn_hungarian",
            score_threshold=0.5,
            unmatched_penalty=1.0,
            sinkhorn_temperature=0.2,
            sinkhorn_iterations=80,
        ),
    )
    pairs = {(match.source_index, match.target_index) for match in result.matches}
    assert pairs == {(0, 3), (1, 2)}


def test_full_score_matrix_keeps_absent_edges_explicit():
    event = _event()
    matrix = full_score_matrix(event, (_candidate(0, 2, 1.0),), [0.7], 0, 1)
    assert matrix.values.shape == (2, 2)
    assert matrix.values[0, 0] == 0.7
    assert np.isnan(matrix.values[0, 1])
    assert np.isnan(matrix.values[1, 0])


def test_state_augmented_pair_feature_schema_keeps_refitted_endpoint_states():
    event = _event()
    event.state[0] = np.asarray([1.0, 2.0, 0.1, 0.2])
    event.state[2] = np.asarray([3.0, 4.0, 0.3, 0.4])
    candidate = _candidate(0, 2, 1.0)

    features = candidate_feature_matrix(
        event,
        (candidate,),
        ((0, 1),),
        feature_set="state_augmented_v2",
    )

    assert features.shape == (1, len(pair_feature_names(((0, 1),), "state_augmented_v2")))
    np.testing.assert_array_equal(features[0, 15:23], np.asarray([1.0, 2.0, 0.1, 0.2, 3.0, 4.0, 0.3, 0.4]))
    assert features[0, -1] == 1.0


def test_state_hitpattern_feature_schema_expands_persisted_layer_side_bits():
    event = _event()
    event.hit_pattern[0] = np.uint64(0b010101)
    event.hit_pattern[2] = np.uint64(0b101010)
    candidate = _candidate(0, 2, 1.0)

    names = pair_feature_names(((0, 1),), "state_hitpattern_v4")
    values = candidate_feature_matrix(
        event,
        (candidate,),
        ((0, 1),),
        feature_set="state_hitpattern_v4",
    )[0]
    by_name = dict(zip(names, values))

    assert values.shape == (len(names),)
    assert [by_name[f"source_hit_layer_{layer}_side_{side}"] for layer in range(3) for side in range(2)] == [
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
    ]
    assert [by_name[f"target_hit_layer_{layer}_side_{side}"] for layer in range(3) for side in range(2)] == [
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
    ]


def test_physics_covariance_feature_schema_keeps_acts_prediction_and_covariance_shape():
    event = _event()
    event.state[0] = np.asarray([1.0, 2.0, 0.1, 0.2])
    event.state[2] = np.asarray([3.0, 4.0, 0.3, 0.4])
    event.covariance[0] = np.diag([4.0, 9.0, 16.0, 25.0])
    event.covariance[2] = np.diag([1.0, 4.0, 9.0, 16.0])
    candidate = FieldCandidate(
        source_index=0,
        target_index=2,
        source_station=0,
        target_station=1,
        chi2=1.0,
        residual=np.asarray([0.5, -0.5, 0.1, -0.1]),
        pull=np.zeros(4, dtype=np.float64),
        combined_covariance=np.asarray(
            [
                [4.0, 1.0, 0.0, 0.0],
                [1.0, 9.0, 0.0, 0.0],
                [0.0, 0.0, 16.0, 2.0],
                [0.0, 0.0, 2.0, 25.0],
            ],
            dtype=np.float64,
        ),
    )
    names = pair_feature_names(((0, 1),), "physics_covariance_v3")
    values = candidate_feature_matrix(
        event,
        (candidate,),
        ((0, 1),),
        feature_set="physics_covariance_v3",
    )[0]
    by_name = dict(zip(names, values))

    assert values.shape == (len(names),)
    np.testing.assert_allclose(
        [
            by_name["prediction_x_mm"],
            by_name["prediction_y_mm"],
            by_name["prediction_tx"],
            by_name["prediction_ty"],
        ],
        [2.5, 4.5, 0.2, 0.5],
    )
    np.testing.assert_allclose(
        [
            by_name["combined_sigma_x_mm"],
            by_name["combined_sigma_y_mm"],
            by_name["combined_sigma_tx"],
            by_name["combined_sigma_ty"],
            by_name["combined_corr_x_y"],
            by_name["combined_corr_tx_ty"],
        ],
        [2.0, 3.0, 4.0, 5.0, 1.0 / 6.0, 0.1],
    )
    np.testing.assert_allclose(
        [by_name["source_sigma_x_mm"], by_name["target_sigma_ty"]],
        [2.0, 4.0],
    )
    assert by_name["station_pair_0_1"] == 1.0


def test_global_operating_point_requires_both_purity_and_fake_constraints():
    rows = [
        {
            "assignment_method": "hungarian",
            "score_threshold": 0.2,
            "association_efficiency": 0.9,
            "inclusive_fake_rate": 0.10,
            "inclusive_association_purity": 0.90,
            "predicted_matches": 10,
            "missing_truth_unmatched_recall": 0.5,
        },
        {
            "assignment_method": "hungarian",
            "score_threshold": 0.7,
            "association_efficiency": 0.7,
            "inclusive_fake_rate": 0.04,
            "inclusive_association_purity": 0.96,
            "predicted_matches": 8,
            "missing_truth_unmatched_recall": 0.5,
        },
    ]
    selected = choose_global_operating_point(rows, 0.05, 0.95)
    assert selected is not None
    assert selected["score_threshold"] == 0.7


def test_global_operating_point_can_require_minimum_efficiency():
    rows = [
        {
            "assignment_method": "hungarian",
            "score_threshold": 0.7,
            "association_efficiency": 0.69,
            "inclusive_fake_rate": 0.04,
            "inclusive_association_purity": 0.96,
            "predicted_matches": 8,
            "missing_truth_unmatched_recall": 0.5,
        }
    ]
    assert choose_global_operating_point(rows, 0.05, 0.95, 0.70) is None


def test_prepared_assignment_context_preserves_truth_free_assignment_result():
    event = _event()
    candidates = (
        _candidate(0, 2, 1.0),
        _candidate(0, 3, 2.0),
        _candidate(1, 2, 3.0),
        _candidate(1, 3, 4.0),
    )
    candidate_set = SimpleNamespace(
        event=event,
        station_pair=(0, 1),
        candidates=candidates,
        labels=np.asarray([True, False, False, True]),
    )
    scores = [np.asarray([0.90, 0.10, 0.20, 0.80])]
    config = AssignmentConfig(method="hungarian", score_threshold=0.0)

    direct = evaluate_global_assignment_sets([candidate_set], scores, config, calibration_bins=5)
    context = prepare_global_assignment_context([candidate_set], scores, calibration_bins=5)
    cached = evaluate_global_assignment_sets(
        [candidate_set], scores, config, calibration_bins=5, context=context
    )

    assert cached == direct


def test_station_pair_assignment_uses_declared_pair_configuration():
    event = _event()
    candidates = (
        _candidate(0, 2, 1.0),
        _candidate(0, 3, 2.0),
        _candidate(1, 2, 3.0),
        _candidate(1, 3, 4.0),
    )
    candidate_set = SimpleNamespace(
        event=event,
        station_pair=(0, 1),
        candidates=candidates,
        labels=np.asarray([True, False, False, True]),
    )
    scores = [np.asarray([0.90, 0.10, 0.20, 0.80])]
    config = AssignmentConfig(method="hungarian", score_threshold=0.0)

    result = evaluate_station_pair_assignment_sets(
        [candidate_set],
        scores,
        {(0, 1): config},
        calibration_bins=5,
    )

    assert result["association"]["correct_matches"] == 2
    assert result["by_station_pair"]["0->1"]["config"]["score_threshold"] == 0.0
