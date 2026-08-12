"""Validation of field-aware local-tracklet propagation against truth pairs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from datasets.propagation_loader import PropagationRecords
from datasets.root_loader import EventTracklets
from geometry.propagation import mahalanobis_chi2, propagate_line


STATE_NAMES = ("x_mm", "y_mm", "tx", "ty")


@dataclass(frozen=True)
class FieldPropagationEvaluation:
    """Accepted truth-matched records and their field/line diagnostics."""

    run_id: np.ndarray
    event_id: np.ndarray
    source_tracklet_id: np.ndarray
    target_tracklet_id: np.ndarray
    source_station_id: np.ndarray
    target_station_id: np.ndarray
    truth_particle_id: np.ndarray
    q_over_p_mode: np.ndarray
    residual: np.ndarray
    pull: np.ndarray
    chi2: np.ndarray
    line_residual: np.ndarray
    line_pull: np.ndarray
    line_chi2: np.ndarray
    combined_covariance: np.ndarray
    rejected_counts: dict[str, int]

    @property
    def size(self) -> int:
        return int(self.chi2.size)


def _stats(values: np.ndarray) -> dict[str, int | float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    result: dict[str, int | float | None] = {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "mean": None,
        "std": None,
        "rms": None,
        "median": None,
        "p95": None,
        "min": None,
        "max": None,
    }
    if finite.size:
        result.update(
            {
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "rms": float(np.sqrt(np.mean(np.square(finite)))),
                "median": float(np.median(finite)),
                "p95": float(np.percentile(finite, 95.0)),
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
            }
        )
    return result


def _valid_covariance(covariance: np.ndarray) -> bool:
    values = np.asarray(covariance, dtype=np.float64)
    return bool(
        values.shape == (4, 4)
        and np.isfinite(values).all()
        and np.allclose(values, values.T, rtol=1.0e-7, atol=1.0e-12)
        and np.all(np.diag(values) > 0.0)
    )


def _event_tracklet_index(
    events: Iterable[EventTracklets],
) -> dict[tuple[int, int, int], tuple[EventTracklets, int]]:
    index: dict[tuple[int, int, int], tuple[EventTracklets, int]] = {}
    for event in events:
        for row, tracklet_id in enumerate(event.tracklet_id):
            key = (event.run_id, event.event_id, int(tracklet_id))
            if key in index:
                raise ValueError(f"duplicate canonical tracklet identity: {key}")
            index[key] = (event, row)
    return index


def _empty_evaluation(rejected_counts: Counter[str]) -> FieldPropagationEvaluation:
    empty_i64 = np.empty(0, dtype=np.int64)
    empty_i16 = np.empty(0, dtype=np.int16)
    empty_state = np.empty((0, 4), dtype=np.float64)
    return FieldPropagationEvaluation(
        run_id=empty_i64,
        event_id=empty_i64.copy(),
        source_tracklet_id=np.empty(0, dtype=np.int32),
        target_tracklet_id=np.empty(0, dtype=np.int32),
        source_station_id=empty_i16,
        target_station_id=empty_i16.copy(),
        truth_particle_id=empty_i64.copy(),
        q_over_p_mode=np.empty(0, dtype=np.int8),
        residual=empty_state,
        pull=empty_state.copy(),
        chi2=np.empty(0, dtype=np.float64),
        line_residual=empty_state.copy(),
        line_pull=empty_state.copy(),
        line_chi2=np.empty(0, dtype=np.float64),
        combined_covariance=np.empty((0, 4, 4), dtype=np.float64),
        rejected_counts=dict(sorted(rejected_counts.items())),
    )


def evaluate_field_propagation(
    events: Iterable[EventTracklets],
    records: PropagationRecords,
    require_truth_match: bool = True,
    q_over_p_mode: int | None = None,
    min_truth_match_fraction: float | None = None,
) -> FieldPropagationEvaluation:
    """Join propagation records to canonical tracklets and calculate pulls.

    The residual convention is target local state minus propagated source state.
    The covariance used for pulls and chi-square is the sum of independent
    target-tracklet and propagated-source covariances.
    """
    tracklet_index = _event_tracklet_index(events)
    rejected: Counter[str] = Counter(total_records=records.size)
    accepted: dict[str, list[np.ndarray | int | float]] = {
        "run_id": [],
        "event_id": [],
        "source_tracklet_id": [],
        "target_tracklet_id": [],
        "source_station_id": [],
        "target_station_id": [],
        "truth_particle_id": [],
        "q_over_p_mode": [],
        "residual": [],
        "pull": [],
        "chi2": [],
        "line_residual": [],
        "line_pull": [],
        "line_chi2": [],
        "combined_covariance": [],
    }

    for row in range(records.size):
        record_mode = (
            int(records.q_over_p_mode[row])
            if records.q_over_p_mode is not None
            else 0
        )
        if q_over_p_mode is not None and record_mode != q_over_p_mode:
            rejected["q_over_p_mode_filtered"] += 1
            continue
        if not records.success[row]:
            rejected["propagation_not_successful"] += 1
            continue
        if not records.has_covariance[row]:
            rejected["propagation_covariance_absent"] += 1
            continue
        source_key = (
            int(records.run_id[row]),
            int(records.event_id[row]),
            int(records.source_tracklet_id[row]),
        )
        target_key = (
            int(records.run_id[row]),
            int(records.event_id[row]),
            int(records.target_tracklet_id[row]),
        )
        source_lookup = tracklet_index.get(source_key)
        target_lookup = tracklet_index.get(target_key)
        if source_lookup is None or target_lookup is None:
            rejected["tracklet_not_in_input"] += 1
            continue
        source_event, source_index = source_lookup
        target_event, target_index = target_lookup
        if source_event is not target_event:
            rejected["source_target_event_mismatch"] += 1
            continue
        if (
            int(source_event.station_id[source_index]) != int(records.source_station_id[row])
            or int(target_event.station_id[target_index])
            != int(records.target_station_id[row])
        ):
            rejected["station_id_mismatch"] += 1
            continue
        if not np.isclose(
            target_event.z_mm[target_index], records.target_z_mm[row], rtol=0.0, atol=1.0e-6
        ):
            rejected["target_z_mismatch"] += 1
            continue
        if require_truth_match:
            if source_event.truth_particle_id is None:
                rejected["truth_labels_absent"] += 1
                continue
            source_truth = int(source_event.truth_particle_id[source_index])
            target_truth = int(source_event.truth_particle_id[target_index])
            record_truth = int(records.truth_particle_id[row])
            if (
                source_truth < 0
                or target_truth < 0
                or source_truth != target_truth
                or source_truth != record_truth
            ):
                rejected["truth_mismatch"] += 1
                continue
        if min_truth_match_fraction is not None:
            if source_event.truth_match_fraction is None:
                rejected["truth_match_fraction_absent"] += 1
                continue
            source_fraction = float(source_event.truth_match_fraction[source_index])
            target_fraction = float(source_event.truth_match_fraction[target_index])
            if (
                not np.isfinite(source_fraction)
                or not np.isfinite(target_fraction)
                or source_fraction < min_truth_match_fraction
                or target_fraction < min_truth_match_fraction
            ):
                rejected["truth_match_fraction_below_threshold"] += 1
                continue

        prediction = records.prediction[row]
        prediction_covariance = records.covariance[row]
        target_state = target_event.state[target_index]
        target_covariance = target_event.covariance[target_index]
        if not _valid_covariance(prediction_covariance):
            rejected["invalid_propagated_covariance"] += 1
            continue
        combined_covariance = prediction_covariance + target_covariance
        if not _valid_covariance(combined_covariance):
            rejected["invalid_combined_covariance"] += 1
            continue
        residual = target_state - prediction
        if not np.isfinite(residual).all():
            rejected["nonfinite_field_residual"] += 1
            continue
        try:
            chi2 = mahalanobis_chi2(residual, combined_covariance)
        except ValueError:
            rejected["singular_field_covariance"] += 1
            continue
        pull = residual / np.sqrt(np.diag(combined_covariance))

        source_state = source_event.state[source_index]
        source_covariance = source_event.covariance[source_index]
        line_prediction, line_covariance = propagate_line(
            source_state,
            source_covariance,
            float(target_event.z_mm[target_index] - source_event.z_mm[source_index]),
        )
        line_combined_covariance = line_covariance + target_covariance
        if not _valid_covariance(line_combined_covariance):
            rejected["invalid_line_combined_covariance"] += 1
            continue
        line_residual = target_state - line_prediction
        try:
            line_chi2 = mahalanobis_chi2(line_residual, line_combined_covariance)
        except ValueError:
            rejected["singular_line_covariance"] += 1
            continue
        line_pull = line_residual / np.sqrt(np.diag(line_combined_covariance))

        for name, value in (
            ("run_id", records.run_id[row]),
            ("event_id", records.event_id[row]),
            ("source_tracklet_id", records.source_tracklet_id[row]),
            ("target_tracklet_id", records.target_tracklet_id[row]),
            ("source_station_id", records.source_station_id[row]),
            ("target_station_id", records.target_station_id[row]),
            ("truth_particle_id", records.truth_particle_id[row]),
            ("q_over_p_mode", record_mode),
            ("residual", residual),
            ("pull", pull),
            ("chi2", chi2),
            ("line_residual", line_residual),
            ("line_pull", line_pull),
            ("line_chi2", line_chi2),
            ("combined_covariance", combined_covariance),
        ):
            accepted[name].append(value)
        rejected["accepted"] += 1

    if not accepted["chi2"]:
        return _empty_evaluation(rejected)
    return FieldPropagationEvaluation(
        run_id=np.asarray(accepted["run_id"], dtype=np.int64),
        event_id=np.asarray(accepted["event_id"], dtype=np.int64),
        source_tracklet_id=np.asarray(accepted["source_tracklet_id"], dtype=np.int32),
        target_tracklet_id=np.asarray(accepted["target_tracklet_id"], dtype=np.int32),
        source_station_id=np.asarray(accepted["source_station_id"], dtype=np.int16),
        target_station_id=np.asarray(accepted["target_station_id"], dtype=np.int16),
        truth_particle_id=np.asarray(accepted["truth_particle_id"], dtype=np.int64),
        q_over_p_mode=np.asarray(accepted["q_over_p_mode"], dtype=np.int8),
        residual=np.asarray(accepted["residual"], dtype=np.float64),
        pull=np.asarray(accepted["pull"], dtype=np.float64),
        chi2=np.asarray(accepted["chi2"], dtype=np.float64),
        line_residual=np.asarray(accepted["line_residual"], dtype=np.float64),
        line_pull=np.asarray(accepted["line_pull"], dtype=np.float64),
        line_chi2=np.asarray(accepted["line_chi2"], dtype=np.float64),
        combined_covariance=np.asarray(accepted["combined_covariance"], dtype=np.float64),
        rejected_counts=dict(sorted(rejected.items())),
    )


def _diagnostic_summary(
    residual: np.ndarray,
    pull: np.ndarray,
    chi2: np.ndarray,
) -> dict[str, object]:
    return {
        "residual": {
            name: _stats(residual[:, index]) for index, name in enumerate(STATE_NAMES)
        },
        "pull": {name: _stats(pull[:, index]) for index, name in enumerate(STATE_NAMES)},
        "chi2": _stats(chi2),
    }


def field_propagation_summary(
    evaluation: FieldPropagationEvaluation,
) -> dict[str, object]:
    """Create JSON-safe global and station-pair summaries."""
    summary: dict[str, object] = {
        "accepted_records": evaluation.size,
        "q_over_p_mode_counts": {
            str(mode): int(np.count_nonzero(evaluation.q_over_p_mode == mode))
            for mode in sorted(set(map(int, evaluation.q_over_p_mode.tolist())))
        },
        "rejected_counts": evaluation.rejected_counts,
        "field_aware": _diagnostic_summary(
            evaluation.residual, evaluation.pull, evaluation.chi2
        ),
        "straight_line": _diagnostic_summary(
            evaluation.line_residual, evaluation.line_pull, evaluation.line_chi2
        ),
        "by_station_pair": {},
    }
    pair_summary: dict[str, object] = {}
    for source_station, target_station in sorted(
        set(zip(evaluation.source_station_id.tolist(), evaluation.target_station_id.tolist()))
    ):
        mask = (evaluation.source_station_id == source_station) & (
            evaluation.target_station_id == target_station
        )
        pair_summary[f"{source_station}->{target_station}"] = {
            "records": int(np.count_nonzero(mask)),
            "field_aware": _diagnostic_summary(
                evaluation.residual[mask], evaluation.pull[mask], evaluation.chi2[mask]
            ),
            "straight_line": _diagnostic_summary(
                evaluation.line_residual[mask],
                evaluation.line_pull[mask],
                evaluation.line_chi2[mask],
            ),
        }
    summary["by_station_pair"] = pair_summary
    return summary
