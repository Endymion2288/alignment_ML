"""Validation and evaluation helpers for pairwise scores with global assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from baselines.global_assignment import (
    AssignmentConfig,
    ScoreMatrix,
    full_score_matrix,
    global_score_assignment_from_matrix,
)
from baselines.multistation_assignment import (
    MultiStationAssignmentConfig,
    MultiStationAssignmentResult,
    multistation_score_assignment,
)
from evaluation.metrics import (
    AssociationMetrics,
    UnmatchedEndpointMetrics,
    assess_event_matches,
    assess_event_unmatched_endpoints,
)
from evaluation.pairwise_metrics import binary_calibration


@dataclass(frozen=True)
class GlobalAssignmentContext:
    """Truth-free score matrices plus validation-only static candidate metrics."""

    matrices: tuple[ScoreMatrix, ...]
    candidate: dict[str, object]
    candidate_by_pair: dict[tuple[int, int], dict[str, object]]


@dataclass(frozen=True)
class _MultiStationEventGroup:
    """All ordered station-pair matrices for one physical synthetic event."""

    event: object
    set_indices_by_pair: Mapping[tuple[int, int], int]
    station_matrices: Mapping[tuple[int, int], tuple[ScoreMatrix, Sequence[object]]]


@dataclass(frozen=True)
class MultiStationAssignmentContext:
    """Static pair matrices grouped into event-level flow inputs."""

    global_context: GlobalAssignmentContext
    groups: tuple[_MultiStationEventGroup, ...]


def _candidate_metric_block(
    labels: Sequence[np.ndarray], scores: Sequence[np.ndarray], bins: int
) -> dict[str, object]:
    nonempty_labels = [np.asarray(value, dtype=bool) for value in labels if value.size]
    nonempty_scores = [np.asarray(value, dtype=np.float64) for value in scores if value.size]
    if not nonempty_labels:
        return {
            "candidate_rows": 0,
            "positive_candidate_rows": 0,
            "roc_auc": None,
            "average_precision": None,
            "brier": None,
            "negative_log_likelihood": None,
            "expected_calibration_error": None,
            "calibration_bins": bins,
            "occupied_calibration_bins": 0,
        }
    merged_labels = np.concatenate(nonempty_labels)
    merged_scores = np.concatenate(nonempty_scores)
    return {
        "candidate_rows": int(merged_labels.size),
        "positive_candidate_rows": int(np.count_nonzero(merged_labels)),
        **binary_calibration(merged_scores, merged_labels, bins=bins),
    }


def prepare_global_assignment_context(
    sets: Sequence[object],
    scores: Sequence[np.ndarray],
    calibration_bins: int,
) -> GlobalAssignmentContext:
    """Build immutable score matrices and candidate metrics once per score view.

    The matrices contain no MC labels.  Truth labels are read only here to
    establish the candidate-recall denominator and are never passed into an
    assignment routine.  A validation grid can reuse this context for every
    threshold, dustbin penalty and Sinkhorn temperature.
    """
    if len(sets) != len(scores):
        raise ValueError("candidate sets and scores are not aligned")
    labels_by_pair: dict[tuple[int, int], list[np.ndarray]] = {}
    scores_by_pair: dict[tuple[int, int], list[np.ndarray]] = {}
    possible_by_pair: dict[tuple[int, int], int] = {}
    positive_by_pair: dict[tuple[int, int], int] = {}
    matrices: list[ScoreMatrix] = []
    for candidate_set, raw_scores in zip(sets, scores):
        values = np.asarray(raw_scores, dtype=np.float64)
        labels = np.asarray(candidate_set.labels, dtype=bool)
        if values.shape != labels.shape:
            raise ValueError("score shape does not match a candidate set")
        source_station, target_station = tuple(int(value) for value in candidate_set.station_pair)
        pair = (source_station, target_station)
        matrices.append(
            full_score_matrix(
                candidate_set.event,
                candidate_set.candidates,
                values,
                source_station=source_station,
                target_station=target_station,
            )
        )
        labels_by_pair.setdefault(pair, []).append(labels)
        scores_by_pair.setdefault(pair, []).append(values)
        available, _ = assess_event_matches(
            candidate_set.event,
            (),
            source_station=source_station,
            target_station=target_station,
        )
        possible_by_pair[pair] = possible_by_pair.get(pair, 0) + available.possible_matches
        positive_by_pair[pair] = positive_by_pair.get(pair, 0) + int(np.count_nonzero(labels))

    candidate_by_pair: dict[tuple[int, int], dict[str, object]] = {}
    for pair in sorted(labels_by_pair):
        candidate = _candidate_metric_block(
            labels_by_pair[pair], scores_by_pair[pair], calibration_bins
        )
        possible = possible_by_pair[pair]
        candidate["candidate_truth_recall"] = (
            None if not possible else float(positive_by_pair[pair] / possible)
        )
        candidate["truth_pairs_with_unique_known_endpoints"] = possible
        candidate_by_pair[pair] = candidate
    all_candidate = _candidate_metric_block(
        [np.asarray(candidate_set.labels, dtype=bool) for candidate_set in sets],
        [np.asarray(value, dtype=np.float64) for value in scores],
        calibration_bins,
    )
    all_possible = int(sum(possible_by_pair.values()))
    all_positive = int(sum(positive_by_pair.values()))
    all_candidate["candidate_truth_recall"] = (
        None if not all_possible else float(all_positive / all_possible)
    )
    all_candidate["truth_pairs_with_unique_known_endpoints"] = all_possible
    return GlobalAssignmentContext(
        matrices=tuple(matrices),
        candidate=all_candidate,
        candidate_by_pair=candidate_by_pair,
    )


def prepare_multistation_assignment_context(
    sets: Sequence[object],
    scores: Sequence[np.ndarray],
    calibration_bins: int,
) -> MultiStationAssignmentContext:
    """Group fixed physical pair matrices by synthetic event for flow solves."""
    global_context = prepare_global_assignment_context(sets, scores, calibration_bins)
    grouped: dict[tuple[str, str, int, int], dict[str, object]] = {}
    for index, (candidate_set, matrix) in enumerate(zip(sets, global_context.matrices)):
        sample = candidate_set.sample
        event = candidate_set.event
        key = (
            str(sample.source_id),
            str(sample.payload_id),
            int(event.run_id),
            int(event.event_id),
        )
        pair = tuple(int(value) for value in candidate_set.station_pair)
        payload = grouped.setdefault(
            key,
            {
                "event": event,
                "indices": {},
                "matrices": {},
            },
        )
        if payload["event"] is not event:
            raise ValueError("one multi-station event key resolves to different event objects")
        indices = payload["indices"]
        matrices = payload["matrices"]
        if not isinstance(indices, dict) or not isinstance(matrices, dict):  # pragma: no cover
            raise RuntimeError("invalid multi-station grouping payload")
        if pair in indices:
            raise ValueError("duplicate station-pair matrix in one multi-station event")
        indices[pair] = index
        matrices[pair] = (matrix, candidate_set.candidates)
    groups = tuple(
        _MultiStationEventGroup(
            event=payload["event"],
            set_indices_by_pair=dict(payload["indices"]),
            station_matrices=dict(payload["matrices"]),
        )
        for _, payload in sorted(grouped.items())
    )
    return MultiStationAssignmentContext(global_context=global_context, groups=groups)


def evaluate_global_assignment_sets(
    sets: Sequence[object],
    scores: Sequence[np.ndarray],
    config: AssignmentConfig,
    calibration_bins: int,
    context: GlobalAssignmentContext | None = None,
    include_station_pair: bool = True,
) -> dict[str, object]:
    """Evaluate one fixed assignment point, including dustbin recall.

    The function accepts the existing ``CandidateSet`` protocol without
    importing it directly, avoiding a dependency cycle with curriculum MLP
    training.  Each object must expose ``event``, ``station_pair``,
    ``candidates`` and ``labels``.
    """
    if len(sets) != len(scores):
        raise ValueError("candidate sets and scores are not aligned")
    if context is None:
        context = prepare_global_assignment_context(sets, scores, calibration_bins)
    elif len(context.matrices) != len(sets):
        raise ValueError("assignment context does not align with candidate sets")
    aggregate_association = AssociationMetrics()
    aggregate_unmatched = UnmatchedEndpointMetrics()
    association_by_pair: dict[tuple[int, int], AssociationMetrics] = {}
    unmatched_by_pair: dict[tuple[int, int], UnmatchedEndpointMetrics] = {}
    matrix_edges = 0
    matrix_rows = 0
    matrix_columns = 0

    for candidate_set, matrix in zip(sets, context.matrices):
        source_station, target_station = tuple(int(value) for value in candidate_set.station_pair)
        pair = (source_station, target_station)
        result = global_score_assignment_from_matrix(
            matrix,
            candidate_set.candidates,
            config=config,
        )
        association, _ = assess_event_matches(
            candidate_set.event,
            result.matches,
            source_station=source_station,
            target_station=target_station,
        )
        unmatched = assess_event_unmatched_endpoints(
            candidate_set.event,
            result.matches,
            source_station=source_station,
            target_station=target_station,
        )
        aggregate_association.add(association)
        aggregate_unmatched.add(unmatched)
        if include_station_pair:
            association_by_pair.setdefault(pair, AssociationMetrics()).add(association)
            unmatched_by_pair.setdefault(pair, UnmatchedEndpointMetrics()).add(unmatched)
        matrix_edges += result.candidate_edges_above_threshold
        matrix_rows += result.matrix_shape[0]
        matrix_columns += result.matrix_shape[1]

    def decorate(pair: tuple[int, int]) -> dict[str, object]:
        return {
            "candidate": context.candidate_by_pair[pair],
            "association": association_by_pair[pair].as_dict(),
            "unmatched": unmatched_by_pair[pair].as_dict(),
        }

    return {
        "candidate": context.candidate,
        "association": aggregate_association.as_dict(),
        "unmatched": aggregate_unmatched.as_dict(),
        "assignment_matrix": {
            "candidate_edges_above_threshold": matrix_edges,
            "station_pair_matrix_rows": matrix_rows,
            "station_pair_matrix_columns": matrix_columns,
        },
        "by_station_pair": {
            f"{pair[0]}->{pair[1]}": decorate(pair)
            for pair in sorted(association_by_pair)
        } if include_station_pair else {},
    }


def evaluate_station_pair_assignment_sets(
    sets: Sequence[object],
    scores: Sequence[np.ndarray],
    configs_by_pair: Mapping[tuple[int, int], AssignmentConfig],
    calibration_bins: int,
    context: GlobalAssignmentContext | None = None,
    include_station_pair: bool = True,
) -> dict[str, object]:
    """Evaluate one fixed assignment configuration per ordered station pair.

    The score matrices remain entirely truth-blind.  This is useful for a
    validation-only threshold-vector baseline when score distributions differ
    by propagation distance.  Every configured station pair is required to
    avoid silently applying a default threshold to one detector pair.
    """
    if len(sets) != len(scores):
        raise ValueError("candidate sets and scores are not aligned")
    if context is None:
        context = prepare_global_assignment_context(sets, scores, calibration_bins)
    elif len(context.matrices) != len(sets):
        raise ValueError("assignment context does not align with candidate sets")
    required_pairs = {
        tuple(int(value) for value in candidate_set.station_pair) for candidate_set in sets
    }
    supplied_pairs = {tuple(int(value) for value in pair) for pair in configs_by_pair}
    if required_pairs != supplied_pairs:
        missing = sorted(required_pairs - supplied_pairs)
        extra = sorted(supplied_pairs - required_pairs)
        raise ValueError(
            "station-pair assignment configuration mismatch: "
            f"missing={missing}, extra={extra}"
        )
    aggregate_association = AssociationMetrics()
    aggregate_unmatched = UnmatchedEndpointMetrics()
    association_by_pair: dict[tuple[int, int], AssociationMetrics] = {}
    unmatched_by_pair: dict[tuple[int, int], UnmatchedEndpointMetrics] = {}
    matrix_edges = 0
    matrix_rows = 0
    matrix_columns = 0

    for candidate_set, matrix in zip(sets, context.matrices):
        pair = tuple(int(value) for value in candidate_set.station_pair)
        result = global_score_assignment_from_matrix(
            matrix,
            candidate_set.candidates,
            config=configs_by_pair[pair],
        )
        association, _ = assess_event_matches(
            candidate_set.event,
            result.matches,
            source_station=pair[0],
            target_station=pair[1],
        )
        unmatched = assess_event_unmatched_endpoints(
            candidate_set.event,
            result.matches,
            source_station=pair[0],
            target_station=pair[1],
        )
        aggregate_association.add(association)
        aggregate_unmatched.add(unmatched)
        if include_station_pair:
            association_by_pair.setdefault(pair, AssociationMetrics()).add(association)
            unmatched_by_pair.setdefault(pair, UnmatchedEndpointMetrics()).add(unmatched)
        matrix_edges += result.candidate_edges_above_threshold
        matrix_rows += result.matrix_shape[0]
        matrix_columns += result.matrix_shape[1]

    def decorate(pair: tuple[int, int]) -> dict[str, object]:
        return {
            "candidate": context.candidate_by_pair[pair],
            "association": association_by_pair[pair].as_dict(),
            "unmatched": unmatched_by_pair[pair].as_dict(),
            "config": {
                "method": configs_by_pair[pair].method,
                "score_threshold": configs_by_pair[pair].score_threshold,
                "unmatched_penalty": configs_by_pair[pair].unmatched_penalty,
                "sinkhorn_temperature": configs_by_pair[pair].sinkhorn_temperature,
                "sinkhorn_iterations": configs_by_pair[pair].sinkhorn_iterations,
            },
        }

    return {
        "candidate": context.candidate,
        "association": aggregate_association.as_dict(),
        "unmatched": aggregate_unmatched.as_dict(),
        "assignment_matrix": {
            "candidate_edges_above_threshold": matrix_edges,
            "station_pair_matrix_rows": matrix_rows,
            "station_pair_matrix_columns": matrix_columns,
        },
        "by_station_pair": {
            f"{pair[0]}->{pair[1]}": decorate(pair)
            for pair in sorted(association_by_pair)
        }
        if include_station_pair
        else {},
    }


def evaluate_multistation_assignment_sets(
    sets: Sequence[object],
    scores: Sequence[np.ndarray],
    config: MultiStationAssignmentConfig,
    calibration_bins: int,
    context: MultiStationAssignmentContext | None = None,
    include_station_pair: bool = True,
) -> dict[str, object]:
    """Evaluate exact four-station flow using only pairwise score matrices.

    The flow solver sees no labels.  Truth is consulted solely afterwards by
    the same association and unmatched-endpoint evaluators used for every
    bipartite baseline.
    """
    if len(sets) != len(scores):
        raise ValueError("candidate sets and scores are not aligned")
    if context is None:
        context = prepare_multistation_assignment_context(sets, scores, calibration_bins)
    elif len(context.global_context.matrices) != len(sets):
        raise ValueError("multi-station assignment context does not align with candidate sets")
    aggregate_association = AssociationMetrics()
    aggregate_unmatched = UnmatchedEndpointMetrics()
    association_by_pair: dict[tuple[int, int], AssociationMetrics] = {}
    unmatched_by_pair: dict[tuple[int, int], UnmatchedEndpointMetrics] = {}
    result_by_set_index: dict[int, MultiStationAssignmentResult] = {}
    candidate_edges = 0
    hypotheses_considered = 0
    selected_hypotheses = 0
    matrix_rows = 0
    matrix_columns = 0
    for group in context.groups:
        result = multistation_score_assignment(group.event, group.station_matrices, config)
        candidate_edges += result.candidate_edges_above_threshold
        hypotheses_considered += result.hypotheses_considered
        selected_hypotheses += result.selected_hypotheses
        for pair, index in group.set_indices_by_pair.items():
            result_by_set_index[index] = result
            matrix, _ = group.station_matrices[pair]
            matrix_rows += matrix.values.shape[0]
            matrix_columns += matrix.values.shape[1]

    for index, candidate_set in enumerate(sets):
        try:
            result = result_by_set_index[index]
        except KeyError as error:  # pragma: no cover - grouping invariant
            raise RuntimeError("multi-station result is missing a candidate set") from error
        pair = tuple(int(value) for value in candidate_set.station_pair)
        matches = result.matches_by_pair.get(pair, ())
        association, _ = assess_event_matches(
            candidate_set.event,
            matches,
            source_station=pair[0],
            target_station=pair[1],
        )
        unmatched = assess_event_unmatched_endpoints(
            candidate_set.event,
            matches,
            source_station=pair[0],
            target_station=pair[1],
        )
        aggregate_association.add(association)
        aggregate_unmatched.add(unmatched)
        if include_station_pair:
            association_by_pair.setdefault(pair, AssociationMetrics()).add(association)
            unmatched_by_pair.setdefault(pair, UnmatchedEndpointMetrics()).add(unmatched)

    def decorate(pair: tuple[int, int]) -> dict[str, object]:
        return {
            "candidate": context.global_context.candidate_by_pair[pair],
            "association": association_by_pair[pair].as_dict(),
            "unmatched": unmatched_by_pair[pair].as_dict(),
        }

    return {
        "candidate": context.global_context.candidate,
        "association": aggregate_association.as_dict(),
        "unmatched": aggregate_unmatched.as_dict(),
        "assignment_matrix": {
            "candidate_edges_above_threshold": candidate_edges,
            "station_pair_matrix_rows": matrix_rows,
            "station_pair_matrix_columns": matrix_columns,
            "hypotheses_considered": hypotheses_considered,
            "selected_hypotheses": selected_hypotheses,
        },
        "by_station_pair": {
            f"{pair[0]}->{pair[1]}": decorate(pair)
            for pair in sorted(association_by_pair)
        }
        if include_station_pair
        else {},
    }


def scan_global_assignment(
    sets: Sequence[object],
    scores: Sequence[np.ndarray],
    configs: Sequence[AssignmentConfig],
    calibration_bins: int,
) -> list[dict[str, object]]:
    """Run declared global-assignment operating points on validation only."""
    context = prepare_global_assignment_context(sets, scores, calibration_bins)
    rows: list[dict[str, object]] = []
    for config in configs:
        evaluation = evaluate_global_assignment_sets(
            sets,
            scores,
            config,
            calibration_bins,
            context=context,
            include_station_pair=False,
        )
        candidate = evaluation["candidate"]
        association = evaluation["association"]
        unmatched = evaluation["unmatched"]
        matrix = evaluation["assignment_matrix"]
        rows.append(
            {
                "assignment_method": config.method,
                "score_threshold": config.score_threshold,
                "unmatched_penalty": config.unmatched_penalty,
                "sinkhorn_temperature": config.sinkhorn_temperature,
                "sinkhorn_iterations": config.sinkhorn_iterations,
                "candidate_truth_recall": candidate["candidate_truth_recall"],
                "candidate_rows": candidate["candidate_rows"],
                "average_precision": candidate["average_precision"],
                **association,
                **unmatched,
                **matrix,
            }
        )
    return rows


def scan_multistation_assignment(
    sets: Sequence[object],
    scores: Sequence[np.ndarray],
    score_thresholds: Sequence[float],
    unmatched_penalties: Sequence[float],
    calibration_bins: int,
    maximum_hypotheses: int = 100_000,
) -> list[dict[str, object]]:
    """Scan exact multi-station flow controls on validation only."""
    context = prepare_multistation_assignment_context(sets, scores, calibration_bins)
    rows: list[dict[str, object]] = []
    for threshold in score_thresholds:
        for penalty in unmatched_penalties:
            config = MultiStationAssignmentConfig(
                score_threshold=float(threshold),
                unmatched_penalty=float(penalty),
                maximum_hypotheses=maximum_hypotheses,
            )
            evaluation = evaluate_multistation_assignment_sets(
                sets,
                scores,
                config,
                calibration_bins,
                context=context,
                include_station_pair=False,
            )
            candidate = evaluation["candidate"]
            association = evaluation["association"]
            unmatched = evaluation["unmatched"]
            matrix = evaluation["assignment_matrix"]
            rows.append(
                {
                    "assignment_method": "multistation_flow",
                    "score_threshold": config.score_threshold,
                    "unmatched_penalty": config.unmatched_penalty,
                    "sinkhorn_temperature": None,
                    "sinkhorn_iterations": None,
                    "candidate_truth_recall": candidate["candidate_truth_recall"],
                    "candidate_rows": candidate["candidate_rows"],
                    "average_precision": candidate["average_precision"],
                    **association,
                    **unmatched,
                    **matrix,
                }
            )
    return rows


def choose_global_operating_point(
    rows: Sequence[Mapping[str, object]],
    maximum_inclusive_fake_rate: float,
    minimum_inclusive_purity: float,
    minimum_association_efficiency: float = 0.0,
) -> dict[str, object] | None:
    """Pick one non-empty validation point under quality and efficiency limits."""
    if (
        not np.isfinite(maximum_inclusive_fake_rate)
        or not np.isfinite(minimum_inclusive_purity)
        or not np.isfinite(minimum_association_efficiency)
        or maximum_inclusive_fake_rate < 0.0
        or maximum_inclusive_fake_rate > 1.0
        or minimum_inclusive_purity < 0.0
        or minimum_inclusive_purity > 1.0
        or minimum_association_efficiency < 0.0
        or minimum_association_efficiency > 1.0
    ):
        raise ValueError("global operating limits must be finite values in [0, 1]")
    passing: list[Mapping[str, object]] = []
    for row in rows:
        efficiency = row.get("association_efficiency")
        fake_rate = row.get("inclusive_fake_rate")
        purity = row.get("inclusive_association_purity")
        predicted = int(row.get("predicted_matches") or 0)
        if efficiency is None or fake_rate is None or purity is None or not predicted:
            continue
        if (
            float(fake_rate) <= maximum_inclusive_fake_rate
            and float(purity) >= minimum_inclusive_purity
            and float(efficiency) >= minimum_association_efficiency
        ):
            passing.append(row)
    if not passing:
        return None
    selected = max(
        passing,
        key=lambda row: (
            float(row["association_efficiency"]),
            -float(row["inclusive_fake_rate"]),
            float(row["missing_truth_unmatched_recall"] or 0.0),
            float(row["score_threshold"]),
        ),
    )
    return dict(selected)
