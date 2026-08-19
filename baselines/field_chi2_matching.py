"""Field-aware candidate building and conventional matching baselines.

These helpers consume mode-0 propagation records produced by
FaserActsExtrapolationTool.  They intentionally do not inspect MC truth while
forming candidates or selecting matches; truth remains evaluation-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from baselines.chi2_matching import Match
from datasets.propagation_loader import PropagationRecords
from datasets.root_loader import EventTracklets
from geometry.propagation import mahalanobis_chi2


PAIR_FEATURE_SETS: tuple[str, ...] = (
    "residual_v1",
    "state_augmented_v2",
    "physics_covariance_v3",
    "state_hitpattern_v4",
)


_RECORD_ROW_CACHE: dict[
    int,
    tuple[PropagationRecords, dict[tuple[int, int, int, int, int], tuple[int, ...]]],
] = {}


@dataclass(frozen=True)
class FieldCandidate:
    """One source-target pair scored with a propagated Acts covariance."""

    source_index: int
    target_index: int
    source_station: int
    target_station: int
    chi2: float
    residual: np.ndarray
    pull: np.ndarray
    combined_covariance: np.ndarray


@dataclass(frozen=True)
class ScoredMatch:
    """A one-to-one association with an optional learned score."""

    source_index: int
    target_index: int
    chi2: float
    score: float


def station_pairs_from_events(events: Iterable[EventTracklets]) -> tuple[tuple[int, int], ...]:
    """Return ordered forward station pairs represented by the event sample."""
    pairs: set[tuple[int, int]] = set()
    for event in events:
        stations = sorted(set(int(station) for station in event.station_id.tolist()))
        pairs.update((source, target) for source in stations for target in stations if source < target)
    return tuple(sorted(pairs))


def _tracklet_indices(event: EventTracklets) -> dict[int, int]:
    return {int(tracklet_id): row for row, tracklet_id in enumerate(event.tracklet_id)}


def _records_for_event_pair(
    records: PropagationRecords,
    event: EventTracklets,
    source_station: int,
    target_station: int,
    q_over_p_mode: int,
) -> tuple[int, ...]:
    """Index flat candidate records once instead of rescanning every event."""
    cache_key = id(records)
    cached = _RECORD_ROW_CACHE.get(cache_key)
    if cached is None or cached[0] is not records:
        rows_by_key: dict[tuple[int, int, int, int, int], list[int]] = {}
        for row in range(records.size):
            mode = int(records.q_over_p_mode[row]) if records.q_over_p_mode is not None else 0
            key = (
                int(records.run_id[row]),
                int(records.event_id[row]),
                int(records.source_station_id[row]),
                int(records.target_station_id[row]),
                mode,
            )
            rows_by_key.setdefault(key, []).append(row)
        cached = (
            records,
            {key: tuple(rows) for key, rows in rows_by_key.items()},
        )
        _RECORD_ROW_CACHE[cache_key] = cached
    return cached[1].get(
        (event.run_id, event.event_id, source_station, target_station, q_over_p_mode),
        (),
    )


def _valid_covariance(covariance: np.ndarray) -> bool:
    values = np.asarray(covariance, dtype=np.float64)
    return bool(
        values.shape == (4, 4)
        and np.isfinite(values).all()
        and np.allclose(values, values.T, rtol=1.0e-7, atol=1.0e-12)
        and np.all(np.diag(values) > 0.0)
    )


def build_field_candidates(
    event: EventTracklets,
    records: PropagationRecords,
    source_station: int,
    target_station: int,
    chi2_gate: float | None = None,
    q_over_p_mode: int = 0,
    target_z_tolerance_mm: float = 1.0e-6,
) -> list[FieldCandidate]:
    """Build all valid field-aware candidates for one ordered station pair."""
    if source_station >= target_station:
        raise ValueError("field-aware baseline accepts only ordered forward station pairs")
    if q_over_p_mode not in (0, 1, 2, 3):
        raise ValueError("field-aware candidates require a known q_over_p_mode record variant")
    if chi2_gate is not None and (not np.isfinite(chi2_gate) or chi2_gate <= 0.0):
        raise ValueError("chi2_gate must be positive when supplied")
    if not np.isfinite(target_z_tolerance_mm) or target_z_tolerance_mm < 0.0:
        raise ValueError("target_z_tolerance_mm must be finite and non-negative")
    indices = _tracklet_indices(event)
    candidates: list[FieldCandidate] = []
    for record_row in _records_for_event_pair(
        records,
        event,
        source_station,
        target_station,
        q_over_p_mode,
    ):
        mode = int(records.q_over_p_mode[record_row]) if records.q_over_p_mode is not None else 0
        if mode != q_over_p_mode:
            continue
        if (
            int(records.source_station_id[record_row]) != source_station
            or int(records.target_station_id[record_row]) != target_station
            or not bool(records.success[record_row])
            or not bool(records.has_covariance[record_row])
        ):
            continue
        source_index = indices.get(int(records.source_tracklet_id[record_row]))
        target_index = indices.get(int(records.target_tracklet_id[record_row]))
        if source_index is None or target_index is None:
            continue
        if not np.isclose(
            float(event.z_mm[target_index]),
            float(records.target_z_mm[record_row]),
            rtol=0.0,
            atol=target_z_tolerance_mm,
        ):
            continue
        propagated_covariance = records.covariance[record_row]
        combined_covariance = propagated_covariance + event.covariance[target_index]
        if not _valid_covariance(propagated_covariance) or not _valid_covariance(combined_covariance):
            continue
        residual = event.state[target_index] - records.prediction[record_row]
        try:
            chi2 = mahalanobis_chi2(residual, combined_covariance)
        except ValueError:
            continue
        if chi2_gate is not None and chi2 > chi2_gate:
            continue
        pull = residual / np.sqrt(np.diag(combined_covariance))
        candidates.append(
            FieldCandidate(
                source_index=source_index,
                target_index=target_index,
                source_station=source_station,
                target_station=target_station,
                chi2=chi2,
                residual=np.asarray(residual, dtype=np.float64),
                pull=np.asarray(pull, dtype=np.float64),
                combined_covariance=np.asarray(combined_covariance, dtype=np.float64),
            )
        )
    return candidates


def greedy_field_chi2_match(candidates: Sequence[FieldCandidate]) -> list[Match]:
    """Conventional ascending-chi2 one-to-one assignment."""
    used_sources: set[int] = set()
    used_targets: set[int] = set()
    matches: list[Match] = []
    for candidate in sorted(
        candidates,
        key=lambda value: (value.chi2, value.source_index, value.target_index),
    ):
        if candidate.source_index in used_sources or candidate.target_index in used_targets:
            continue
        used_sources.add(candidate.source_index)
        used_targets.add(candidate.target_index)
        matches.append(
            Match(
                source_index=candidate.source_index,
                target_index=candidate.target_index,
                chi2=candidate.chi2,
            )
        )
    return matches


def candidate_labels(event: EventTracklets, candidates: Sequence[FieldCandidate]) -> np.ndarray:
    """Return truth labels for supervision only, never for candidate selection."""
    if event.truth_particle_id is None:
        raise ValueError("pair-classifier supervision requires MC truth labels")
    return np.asarray(
        [
            int(event.truth_particle_id[candidate.source_index]) >= 0
            and int(event.truth_particle_id[candidate.source_index])
            == int(event.truth_particle_id[candidate.target_index])
            for candidate in candidates
        ],
        dtype=bool,
    )


def pair_feature_names(
    station_pairs: Sequence[tuple[int, int]], feature_set: str = "residual_v1"
) -> tuple[str, ...]:
    """Names for the fixed low-capacity MLP pair representation."""
    if feature_set not in PAIR_FEATURE_SETS:
        raise ValueError(f"unsupported pair feature set '{feature_set}'")
    common = (
        "residual_x_mm",
        "residual_y_mm",
        "residual_tx",
        "residual_ty",
        "pull_x",
        "pull_y",
        "pull_tx",
        "pull_ty",
        "log1p_chi2",
        "combined_covariance_logdet",
        "delta_z_mm",
        "source_chi2_per_ndof",
        "target_chi2_per_ndof",
        "source_n_hit",
        "target_n_hit",
    )
    if feature_set in {"state_augmented_v2", "physics_covariance_v3", "state_hitpattern_v4"}:
        common = (
            *common,
            "source_x_mm",
            "source_y_mm",
            "source_tx",
            "source_ty",
            "target_x_mm",
            "target_y_mm",
            "target_tx",
            "target_ty",
        )
    if feature_set == "physics_covariance_v3":
        common = (
            *common,
            "prediction_x_mm",
            "prediction_y_mm",
            "prediction_tx",
            "prediction_ty",
            "combined_sigma_x_mm",
            "combined_sigma_y_mm",
            "combined_sigma_tx",
            "combined_sigma_ty",
            "combined_corr_x_y",
            "combined_corr_x_tx",
            "combined_corr_x_ty",
            "combined_corr_y_tx",
            "combined_corr_y_ty",
            "combined_corr_tx_ty",
            "source_sigma_x_mm",
            "source_sigma_y_mm",
            "source_sigma_tx",
            "source_sigma_ty",
            "target_sigma_x_mm",
            "target_sigma_y_mm",
            "target_sigma_tx",
            "target_sigma_ty",
        )
    if feature_set == "state_hitpattern_v4":
        # NtupleDumper persists the six SCT layer-side occupancy bits as
        # bit 2*layer+side.  Keep them separate rather than treating the
        # bitfield integer as an ordinal detector-quality variable.
        common = (
            *common,
            *(f"source_hit_layer_{layer}_side_{side}" for layer in range(3) for side in range(2)),
            *(f"target_hit_layer_{layer}_side_{side}" for layer in range(3) for side in range(2)),
        )
    return (*common, *(f"station_pair_{source}_{target}" for source, target in station_pairs))


def candidate_feature_matrix(
    event: EventTracklets,
    candidates: Sequence[FieldCandidate],
    station_pairs: Sequence[tuple[int, int]],
    feature_set: str = "residual_v1",
) -> np.ndarray:
    """Create geometry and local-quality features for the MLP baseline."""
    pairs = tuple(station_pairs)
    if feature_set not in PAIR_FEATURE_SETS:
        raise ValueError(f"unsupported pair feature set '{feature_set}'")
    pair_index = {pair: row for row, pair in enumerate(pairs)}
    if len(pair_index) != len(pairs):
        raise ValueError("station_pairs contains duplicates")
    features = np.empty(
        (len(candidates), len(pair_feature_names(pairs, feature_set=feature_set))), dtype=np.float64
    )
    for row, candidate in enumerate(candidates):
        sign, logdet = np.linalg.slogdet(candidate.combined_covariance)
        if sign <= 0.0 or not np.isfinite(logdet):
            raise ValueError("candidate combined covariance is not positive definite")
        source_ndof = max(float(event.ndof[candidate.source_index]), 1.0)
        target_ndof = max(float(event.ndof[candidate.target_index]), 1.0)
        values = [
            *candidate.residual.tolist(),
            *candidate.pull.tolist(),
            float(np.log1p(max(candidate.chi2, 0.0))),
            float(logdet),
            float(event.z_mm[candidate.target_index] - event.z_mm[candidate.source_index]),
            float(event.chi2[candidate.source_index] / source_ndof),
            float(event.chi2[candidate.target_index] / target_ndof),
            float(event.n_hit[candidate.source_index]),
            float(event.n_hit[candidate.target_index]),
        ]
        if feature_set in {"state_augmented_v2", "physics_covariance_v3", "state_hitpattern_v4"}:
            values.extend(
                [
                    *event.state[candidate.source_index].tolist(),
                    *event.state[candidate.target_index].tolist(),
                ]
            )
        if feature_set == "physics_covariance_v3":
            diagonal = np.sqrt(np.diag(candidate.combined_covariance))
            correlations = candidate.combined_covariance / np.outer(diagonal, diagonal)
            source_sigma = np.sqrt(np.diag(event.covariance[candidate.source_index]))
            target_sigma = np.sqrt(np.diag(event.covariance[candidate.target_index]))
            prediction = event.state[candidate.target_index] - candidate.residual
            values.extend(
                [
                    *prediction.tolist(),
                    *diagonal.tolist(),
                    float(correlations[0, 1]),
                    float(correlations[0, 2]),
                    float(correlations[0, 3]),
                    float(correlations[1, 2]),
                    float(correlations[1, 3]),
                    float(correlations[2, 3]),
                    *source_sigma.tolist(),
                    *target_sigma.tolist(),
                ]
            )
        if feature_set == "state_hitpattern_v4":
            source_pattern = int(event.hit_pattern[candidate.source_index])
            target_pattern = int(event.hit_pattern[candidate.target_index])
            values.extend(
                [
                    *[float((source_pattern >> bit) & 1) for bit in range(6)],
                    *[float((target_pattern >> bit) & 1) for bit in range(6)],
                ]
            )
        one_hot = np.zeros(len(pairs), dtype=np.float64)
        try:
            one_hot[pair_index[(candidate.source_station, candidate.target_station)]] = 1.0
        except KeyError as error:
            raise ValueError("candidate station pair is absent from station_pairs") from error
        features[row] = np.asarray([*values, *one_hot.tolist()], dtype=np.float64)
    return features


def greedy_score_match(
    candidates: Sequence[FieldCandidate],
    scores: Sequence[float],
    score_threshold: float,
) -> list[ScoredMatch]:
    """Apply a score threshold then descending-score one-to-one assignment."""
    if not np.isfinite(score_threshold):
        raise ValueError("score_threshold must be finite")
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(candidates),) or not np.isfinite(values).all():
        raise ValueError("scores must be finite and align with candidates")
    used_sources: set[int] = set()
    used_targets: set[int] = set()
    matches: list[ScoredMatch] = []
    order = sorted(
        range(len(candidates)),
        key=lambda row: (
            -float(values[row]),
            candidates[row].chi2,
            candidates[row].source_index,
            candidates[row].target_index,
        ),
    )
    for row in order:
        candidate = candidates[row]
        score = float(values[row])
        if score < score_threshold:
            continue
        if candidate.source_index in used_sources or candidate.target_index in used_targets:
            continue
        used_sources.add(candidate.source_index)
        used_targets.add(candidate.target_index)
        matches.append(
            ScoredMatch(
                source_index=candidate.source_index,
                target_index=candidate.target_index,
                chi2=candidate.chi2,
                score=score,
            )
        )
    return matches
