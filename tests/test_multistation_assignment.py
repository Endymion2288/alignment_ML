from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from baselines.field_chi2_matching import FieldCandidate
from baselines.global_assignment import full_score_matrix
from baselines.multistation_assignment import (
    MultiStationAssignmentConfig,
    multistation_score_assignment,
)
from datasets.root_loader import EventTracklets
from training.global_assignment import evaluate_multistation_assignment_sets


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
    matrices = {}
    for source_station in range(4):
        for target_station in range(source_station + 1, 4):
            source = event.indices_for_station(source_station).tolist()
            target = event.indices_for_station(target_station).tolist()
            candidates = [
                _candidate(source[0], target[0], source_station, target_station),
                _candidate(source[1], target[1], source_station, target_station),
            ]
            scores = [0.8, 0.8]
            if (source_station, target_station) == (0, 1):
                # A highly scored two-station cross edge is locally tempting,
                # but it is unsupported by every other station pair.
                candidates.append(_candidate(source[0], target[1], source_station, target_station))
                scores.append(0.99)
            matrices[(source_station, target_station)] = (
                full_score_matrix(event, candidates, scores, source_station, target_station),
                candidates,
            )
    return matrices


def test_multistation_assignment_prefers_consistent_four_station_paths():
    event = _event()
    result = multistation_score_assignment(
        event,
        _matrices(event),
        MultiStationAssignmentConfig(score_threshold=0.5, unmatched_penalty=0.0),
    )

    assert result.selected_hypotheses == 2
    assert not any(result.unmatched_by_station.values())
    for matches in result.matches_by_pair.values():
        assert {(match.source_index % 2, match.target_index % 2) for match in matches} == {
            (0, 0),
            (1, 1),
        }


def test_multistation_assignment_can_choose_all_dustbins():
    event = _event()
    result = multistation_score_assignment(
        event,
        _matrices(event),
        MultiStationAssignmentConfig(score_threshold=0.5, unmatched_penalty=-10.0),
    )

    assert result.selected_hypotheses == 0
    assert all(not matches for matches in result.matches_by_pair.values())
    assert {station: len(rows) for station, rows in result.unmatched_by_station.items()} == {
        0: 2,
        1: 2,
        2: 2,
        3: 2,
    }


def test_multistation_evaluator_uses_the_same_pairwise_truth_metrics():
    event = _event()
    matrices = _matrices(event)
    sets = []
    scores = []
    sample = SimpleNamespace(source_id="validation_source", payload_id="mag_0_validation_00")
    for pair, (matrix, candidates) in matrices.items():
        source_lookup = {index: row for row, index in enumerate(matrix.source_indices)}
        target_lookup = {index: row for row, index in enumerate(matrix.target_indices)}
        values = np.asarray(
            [
                matrix.values[source_lookup[candidate.source_index], target_lookup[candidate.target_index]]
                for candidate in candidates
            ],
            dtype=np.float64,
        )
        labels = np.asarray(
            [candidate.source_index % 2 == candidate.target_index % 2 for candidate in candidates],
            dtype=bool,
        )
        sets.append(
            SimpleNamespace(
                sample=sample,
                event=event,
                station_pair=pair,
                candidates=tuple(candidates),
                labels=labels,
            )
        )
        scores.append(values)

    result = evaluate_multistation_assignment_sets(
        sets,
        scores,
        MultiStationAssignmentConfig(score_threshold=0.5, unmatched_penalty=0.0),
        calibration_bins=5,
    )

    assert result["association"]["possible_matches"] == 12
    assert result["association"]["correct_matches"] == 12
    assert result["association"]["association_efficiency"] == 1.0
    assert set(result["by_station_pair"]) == {"0->1", "0->2", "0->3", "1->2", "1->3", "2->3"}
