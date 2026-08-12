"""Sparse-score global one-to-one assignment baselines.

The pair MLP scores only physically propagated candidate edges.  This module
turns those sparse edges into an explicit score matrix over all tracklets in a
station pair, then compares three deterministic assignment rules:

* conventional Hungarian on the thresholded bipartite score matrix;
* Sinkhorn-normalised dustbin matrix followed by Hungarian rounding; and
* Hungarian with source/target-specific dustbins, equivalent to a unit-capacity
  bipartite min-cost-flow with optional unmatched endpoints.

MC truth is never consulted by these routines.  It remains evaluation-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from baselines.field_chi2_matching import FieldCandidate, ScoredMatch
from datasets.root_loader import EventTracklets


AssignmentMethod = Literal[
    "greedy",
    "hungarian",
    "sinkhorn_hungarian",
    "dustbin_hungarian",
]

ASSIGNMENT_METHODS: tuple[AssignmentMethod, ...] = (
    "greedy",
    "hungarian",
    "sinkhorn_hungarian",
    "dustbin_hungarian",
)


@dataclass(frozen=True)
class AssignmentConfig:
    """One fixed operating point for a station-pair score matrix."""

    method: AssignmentMethod
    score_threshold: float
    unmatched_penalty: float = 0.0
    sinkhorn_temperature: float = 1.0
    sinkhorn_iterations: int = 60


@dataclass(frozen=True)
class ScoreMatrix:
    """Dense view of sparse MLP scores, with NaN for absent candidate edges."""

    source_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    values: np.ndarray
    candidate_rows: np.ndarray


@dataclass(frozen=True)
class GlobalAssignmentResult:
    """Predicted associations plus every endpoint left unmatched."""

    matches: tuple[ScoredMatch, ...]
    unmatched_sources: tuple[int, ...]
    unmatched_targets: tuple[int, ...]
    candidate_edges_above_threshold: int
    matrix_shape: tuple[int, int]


def _validate_config(config: AssignmentConfig) -> None:
    if config.method not in ASSIGNMENT_METHODS:
        raise ValueError(f"unsupported global assignment method '{config.method}'")
    if not np.isfinite(config.score_threshold) or not 0.0 <= config.score_threshold <= 1.0:
        raise ValueError("score_threshold must be finite and in [0, 1]")
    if not np.isfinite(config.unmatched_penalty):
        raise ValueError("unmatched_penalty must be finite")
    if not np.isfinite(config.sinkhorn_temperature) or config.sinkhorn_temperature <= 0.0:
        raise ValueError("sinkhorn_temperature must be finite and positive")
    if config.sinkhorn_iterations < 1:
        raise ValueError("sinkhorn_iterations must be positive")


def _station_indices(event: EventTracklets, station: int) -> tuple[int, ...]:
    return tuple(int(index) for index in event.indices_for_station(station).tolist())


def full_score_matrix(
    event: EventTracklets,
    candidates: Sequence[FieldCandidate],
    scores: Sequence[float],
    source_station: int,
    target_station: int,
) -> ScoreMatrix:
    """Materialise the complete station-pair score matrix without truth access."""
    if source_station >= target_station:
        raise ValueError("assignment requires an ordered forward station pair")
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(candidates),) or not np.isfinite(values).all():
        raise ValueError("scores must be finite and align with candidates")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("assignment scores must be probabilities in [0, 1]")
    source_indices = _station_indices(event, source_station)
    target_indices = _station_indices(event, target_station)
    source_row = {index: row for row, index in enumerate(source_indices)}
    target_row = {index: row for row, index in enumerate(target_indices)}
    matrix = np.full((len(source_indices), len(target_indices)), np.nan, dtype=np.float64)
    candidate_rows = np.full((len(source_indices), len(target_indices)), -1, dtype=np.int64)
    for row, candidate in enumerate(candidates):
        if candidate.source_station != source_station or candidate.target_station != target_station:
            raise ValueError("candidate station pair differs from requested score matrix")
        try:
            left = source_row[candidate.source_index]
            right = target_row[candidate.target_index]
        except KeyError as error:
            raise ValueError("candidate endpoint is absent from its declared station") from error
        previous = int(candidate_rows[left, right])
        if previous < 0:
            matrix[left, right] = values[row]
            candidate_rows[left, right] = row
            continue
        # A duplicate physical candidate should not normally occur.  Failing
        # closed would discard usable data, so retain a deterministic best edge
        # while making the choice explicit and independent of truth labels.
        old = candidates[previous]
        if values[row] > matrix[left, right] or (
            np.isclose(values[row], matrix[left, right]) and candidate.chi2 < old.chi2
        ):
            matrix[left, right] = values[row]
            candidate_rows[left, right] = row
    return ScoreMatrix(
        source_indices=source_indices,
        target_indices=target_indices,
        values=matrix,
        candidate_rows=candidate_rows,
    )


def _thresholded_edges(matrix: ScoreMatrix, threshold: float) -> np.ndarray:
    return np.isfinite(matrix.values) & (matrix.values >= threshold)


def _to_scored_match(
    matrix: ScoreMatrix,
    candidates: Sequence[FieldCandidate],
    left: int,
    right: int,
) -> ScoredMatch:
    candidate_row = int(matrix.candidate_rows[left, right])
    if candidate_row < 0:
        raise RuntimeError("attempted to materialise an absent candidate edge")
    candidate = candidates[candidate_row]
    return ScoredMatch(
        source_index=candidate.source_index,
        target_index=candidate.target_index,
        chi2=float(candidate.chi2),
        score=float(matrix.values[left, right]),
    )


def _result(
    matrix: ScoreMatrix,
    matches: Sequence[ScoredMatch],
    above_threshold: np.ndarray,
) -> GlobalAssignmentResult:
    used_sources = {match.source_index for match in matches}
    used_targets = {match.target_index for match in matches}
    return GlobalAssignmentResult(
        matches=tuple(matches),
        unmatched_sources=tuple(index for index in matrix.source_indices if index not in used_sources),
        unmatched_targets=tuple(index for index in matrix.target_indices if index not in used_targets),
        candidate_edges_above_threshold=int(np.count_nonzero(above_threshold)),
        matrix_shape=tuple(int(value) for value in matrix.values.shape),
    )


def _hungarian(matrix: ScoreMatrix, candidates: Sequence[FieldCandidate], threshold: float) -> GlobalAssignmentResult:
    allowed = _thresholded_edges(matrix, threshold)
    if not matrix.source_indices or not matrix.target_indices or not np.any(allowed):
        return _result(matrix, (), allowed)
    # A conventional rectangular Hungarian solve has no explicit dustbin.  An
    # unavailable edge receives a cost beyond any allowed probability and is
    # dropped after rounding; thresholding is its only abstention mechanism.
    blocked_cost = 1.0e6
    cost = np.full(matrix.values.shape, blocked_cost, dtype=np.float64)
    cost[allowed] = -matrix.values[allowed]
    rows, columns = linear_sum_assignment(cost)
    matches = [
        _to_scored_match(matrix, candidates, int(left), int(right))
        for left, right in zip(rows.tolist(), columns.tolist())
        if bool(allowed[left, right])
    ]
    return _result(matrix, matches, allowed)


def _greedy(
    matrix: ScoreMatrix,
    candidates: Sequence[FieldCandidate],
    threshold: float,
) -> GlobalAssignmentResult:
    """Reference one-to-one rule operating on an already-built score matrix."""
    allowed = _thresholded_edges(matrix, threshold)
    rows, columns = np.nonzero(allowed)
    order = sorted(
        zip(rows.tolist(), columns.tolist()),
        key=lambda edge: (
            -float(matrix.values[edge[0], edge[1]]),
            float(candidates[int(matrix.candidate_rows[edge[0], edge[1]])].chi2),
            int(matrix.source_indices[edge[0]]),
            int(matrix.target_indices[edge[1]]),
        ),
    )
    used_sources: set[int] = set()
    used_targets: set[int] = set()
    matches: list[ScoredMatch] = []
    for left, right in order:
        source_index = int(matrix.source_indices[left])
        target_index = int(matrix.target_indices[right])
        if source_index in used_sources or target_index in used_targets:
            continue
        used_sources.add(source_index)
        used_targets.add(target_index)
        matches.append(_to_scored_match(matrix, candidates, left, right))
    return _result(matrix, matches, allowed)


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1.0e-6, 1.0 - 1.0e-6)
    return np.log(clipped) - np.log1p(-clipped)


def _dustbin_cost_matrix(matrix: ScoreMatrix, allowed: np.ndarray, penalty: float) -> np.ndarray:
    """Create the square assignment form of a unit-capacity dustbin flow."""
    n_source, n_target = matrix.values.shape
    size = n_source + n_target
    blocked_cost = max(1.0e4, 100.0 + 10.0 * abs(float(penalty)))
    cost = np.full((size, size), blocked_cost, dtype=np.float64)
    if n_source and n_target:
        real_cost = -_logit(np.where(allowed, matrix.values, 0.5))
        real_block = cost[:n_source, :n_target]
        real_block[allowed] = real_cost[allowed]
    # A source-specific and a target-specific dummy preserve one-to-one
    # capacity.  The lower-right block couples unused dummy capacity at zero.
    for source in range(n_source):
        cost[source, n_target + source] = penalty
    for target in range(n_target):
        cost[n_source + target, target] = penalty
    cost[n_source:, n_target:] = 0.0
    return cost


def _dustbin_hungarian(
    matrix: ScoreMatrix,
    candidates: Sequence[FieldCandidate],
    threshold: float,
    penalty: float,
) -> GlobalAssignmentResult:
    allowed = _thresholded_edges(matrix, threshold)
    if not matrix.source_indices or not matrix.target_indices:
        return _result(matrix, (), allowed)
    cost = _dustbin_cost_matrix(matrix, allowed, penalty)
    rows, columns = linear_sum_assignment(cost)
    n_source, n_target = matrix.values.shape
    matches = [
        _to_scored_match(matrix, candidates, int(left), int(right))
        for left, right in zip(rows.tolist(), columns.tolist())
        if left < n_source and right < n_target and bool(allowed[left, right])
    ]
    return _result(matrix, matches, allowed)


def _sinkhorn_round(
    matrix: ScoreMatrix,
    candidates: Sequence[FieldCandidate],
    threshold: float,
    penalty: float,
    temperature: float,
    iterations: int,
) -> GlobalAssignmentResult:
    allowed = _thresholded_edges(matrix, threshold)
    if not matrix.source_indices or not matrix.target_indices:
        return _result(matrix, (), allowed)
    cost = _dustbin_cost_matrix(matrix, allowed, penalty)
    # The dustbin construction guarantees at least one finite-capacity edge
    # in every row and column.  Shifting before exponentiation is equivalent
    # to the log-domain initialisation but avoids millions of tiny
    # ``scipy.special.logsumexp`` calls during validation hyperparameter
    # scans.  Blocked edges underflow to zero by construction.
    log_kernel = -cost / temperature
    log_kernel -= float(np.max(log_kernel))
    soft_assignment = np.exp(log_kernel)
    for _ in range(iterations):
        row_sum = np.sum(soft_assignment, axis=1, keepdims=True)
        if np.any(row_sum <= 0.0) or not np.isfinite(row_sum).all():
            raise RuntimeError("Sinkhorn row normalisation encountered an invalid dustbin matrix")
        soft_assignment /= row_sum
        column_sum = np.sum(soft_assignment, axis=0, keepdims=True)
        if np.any(column_sum <= 0.0) or not np.isfinite(column_sum).all():
            raise RuntimeError("Sinkhorn column normalisation encountered an invalid dustbin matrix")
        soft_assignment /= column_sum
    # Rounding the soft global assignment still observes the full dustbin
    # matrix.  The tiny deterministic term breaks exact float ties stably.
    tie_break = np.arange(soft_assignment.size, dtype=np.float64).reshape(soft_assignment.shape)
    rows, columns = linear_sum_assignment(-soft_assignment + tie_break * 1.0e-12)
    n_source, n_target = matrix.values.shape
    matches = [
        _to_scored_match(matrix, candidates, int(left), int(right))
        for left, right in zip(rows.tolist(), columns.tolist())
        if left < n_source and right < n_target and bool(allowed[left, right])
    ]
    return _result(matrix, matches, allowed)


def global_score_assignment_from_matrix(
    matrix: ScoreMatrix,
    candidates: Sequence[FieldCandidate],
    config: AssignmentConfig,
) -> GlobalAssignmentResult:
    """Assign a prebuilt truth-free score matrix at one operating point."""
    _validate_config(config)
    if config.method == "greedy":
        return _greedy(matrix, candidates, config.score_threshold)
    if config.method == "hungarian":
        return _hungarian(matrix, candidates, config.score_threshold)
    if config.method == "dustbin_hungarian":
        return _dustbin_hungarian(
            matrix,
            candidates,
            config.score_threshold,
            config.unmatched_penalty,
        )
    return _sinkhorn_round(
        matrix,
        candidates,
        config.score_threshold,
        config.unmatched_penalty,
        config.sinkhorn_temperature,
        config.sinkhorn_iterations,
    )


def global_score_assignment(
    event: EventTracklets,
    candidates: Sequence[FieldCandidate],
    scores: Sequence[float],
    source_station: int,
    target_station: int,
    config: AssignmentConfig,
) -> GlobalAssignmentResult:
    """Run one fixed global assignment without examining MC labels."""
    matrix = full_score_matrix(
        event,
        candidates,
        scores,
        source_station=source_station,
        target_station=target_station,
    )
    return global_score_assignment_from_matrix(matrix, candidates, config)
