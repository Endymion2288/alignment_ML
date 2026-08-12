"""Component-level diagnostics for comparing field-propagation q/p modes.

The ordinary field-propagation evaluator reports the final 4D residual and
chi-square.  This module preserves the propagated and target covariance terms
separately, so a mode-to-mode chi-square change can be decomposed into a raw
residual change and a covariance change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from datasets.propagation_loader import PropagationRecords
from evaluation.field_propagation import FieldPropagationEvaluation, STATE_NAMES
from geometry.propagation import mahalanobis_chi2


COMPONENT_CSV_FIELDNAMES = (
    "run_id",
    "event_id",
    "source_tracklet_id",
    "target_tracklet_id",
    "source_station_id",
    "target_station_id",
    "truth_particle_id",
    "q_over_p_mode",
    "rx_mm",
    "ry_mm",
    "rtx",
    "rty",
    "pull_x_mm",
    "pull_y_mm",
    "pull_tx",
    "pull_ty",
    "propagated_sigma_x_mm",
    "propagated_sigma_y_mm",
    "propagated_sigma_tx",
    "propagated_sigma_ty",
    "target_sigma_x_mm",
    "target_sigma_y_mm",
    "target_sigma_tx",
    "target_sigma_ty",
    "combined_sigma_x_mm",
    "combined_sigma_y_mm",
    "combined_sigma_tx",
    "combined_sigma_ty",
    "combined_covariance_det",
    "combined_covariance_logdet",
    "chi2",
)

COMPARISON_CSV_FIELDNAMES = (
    "run_id",
    "event_id",
    "source_tracklet_id",
    "target_tracklet_id",
    "source_station_id",
    "target_station_id",
    "truth_particle_id",
    "chi2_mode0",
    "chi2_mode1_residual_mode0_covariance",
    "chi2_mode1",
    "chi2_mode0_residual_mode1_covariance",
    "chi2_change_mode1_minus_mode0",
    "chi2_residual_effect_hold_mode0_covariance",
    "chi2_covariance_effect_after_mode1_residual",
    "combined_covariance_det_mode0",
    "combined_covariance_det_mode1",
    "combined_covariance_logdet_mode0",
    "combined_covariance_logdet_mode1",
    "combined_covariance_logdet_change_mode1_minus_mode0",
)


@dataclass(frozen=True)
class PropagationModeComponents:
    """Accepted field-propagation records with covariance terms separated."""

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
    propagated_covariance: np.ndarray
    target_covariance: np.ndarray
    combined_covariance: np.ndarray
    chi2: np.ndarray

    @property
    def size(self) -> int:
        return int(self.chi2.size)


@dataclass(frozen=True)
class PropagationModeComparison:
    """Same-pair q/p mode comparison with an exact chi-square decomposition."""

    run_id: np.ndarray
    event_id: np.ndarray
    source_tracklet_id: np.ndarray
    target_tracklet_id: np.ndarray
    source_station_id: np.ndarray
    target_station_id: np.ndarray
    truth_particle_id: np.ndarray
    chi2_mode0: np.ndarray
    chi2_mode1_residual_mode0_covariance: np.ndarray
    chi2_mode1: np.ndarray
    chi2_mode0_residual_mode1_covariance: np.ndarray
    residual_effect: np.ndarray
    covariance_effect: np.ndarray
    determinant_mode0: np.ndarray
    determinant_mode1: np.ndarray
    logdet_mode0: np.ndarray
    logdet_mode1: np.ndarray
    mode0_only_count: int
    mode1_only_count: int

    @property
    def size(self) -> int:
        return int(self.chi2_mode0.size)


def _record_mode(records: PropagationRecords, row: int) -> int:
    if records.q_over_p_mode is None:
        return 0
    return int(records.q_over_p_mode[row])


def _record_identity(records: PropagationRecords, row: int) -> tuple[int, int, int, int, int]:
    return (
        int(records.run_id[row]),
        int(records.event_id[row]),
        int(records.source_tracklet_id[row]),
        int(records.target_tracklet_id[row]),
        _record_mode(records, row),
    )


def _component_identity(
    components: PropagationModeComponents, row: int
) -> tuple[int, int, int, int, int]:
    return (
        int(components.run_id[row]),
        int(components.event_id[row]),
        int(components.source_tracklet_id[row]),
        int(components.target_tracklet_id[row]),
        int(components.q_over_p_mode[row]),
    )


def _comparison_identity(
    components: PropagationModeComponents, row: int
) -> tuple[int, int, int, int, int, int]:
    return (
        int(components.run_id[row]),
        int(components.event_id[row]),
        int(components.source_tracklet_id[row]),
        int(components.target_tracklet_id[row]),
        int(components.source_station_id[row]),
        int(components.target_station_id[row]),
    )


def _covariance_determinants(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(covariance, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 4):
        raise ValueError("covariance must have shape (n, 4, 4)")
    if not values.shape[0]:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    sign, logdet = np.linalg.slogdet(values)
    if not np.all(np.isfinite(logdet)) or not np.all(sign > 0.0):
        raise ValueError("combined covariance determinant is not positive and finite")
    with np.errstate(over="raise"):
        determinant = np.exp(logdet)
    return determinant, logdet


def _empty_components() -> PropagationModeComponents:
    empty_i64 = np.empty(0, dtype=np.int64)
    empty_i32 = np.empty(0, dtype=np.int32)
    empty_i16 = np.empty(0, dtype=np.int16)
    empty_i8 = np.empty(0, dtype=np.int8)
    empty_state = np.empty((0, 4), dtype=np.float64)
    empty_covariance = np.empty((0, 4, 4), dtype=np.float64)
    return PropagationModeComponents(
        run_id=empty_i64,
        event_id=empty_i64.copy(),
        source_tracklet_id=empty_i32,
        target_tracklet_id=empty_i32.copy(),
        source_station_id=empty_i16,
        target_station_id=empty_i16.copy(),
        truth_particle_id=empty_i64.copy(),
        q_over_p_mode=empty_i8,
        residual=empty_state,
        pull=empty_state.copy(),
        propagated_covariance=empty_covariance,
        target_covariance=empty_covariance.copy(),
        combined_covariance=empty_covariance.copy(),
        chi2=np.empty(0, dtype=np.float64),
    )


def components_from_field_evaluation(
    evaluation: FieldPropagationEvaluation,
    records: PropagationRecords,
) -> PropagationModeComponents:
    """Recover propagated and target covariance terms for accepted records.

    ``evaluate_field_propagation`` has already checked every record identity,
    covariance, truth label, and target state.  This function joins its
    accepted identities back to the source propagation record and derives the
    target covariance as ``S_combined - S_propagated``.
    """
    if not evaluation.size:
        return _empty_components()

    record_index: dict[tuple[int, int, int, int, int], int] = {}
    for row in range(records.size):
        key = _record_identity(records, row)
        if key in record_index:
            raise ValueError(f"duplicate propagation record identity: {key}")
        record_index[key] = row

    propagated_covariance: list[np.ndarray] = []
    target_covariance: list[np.ndarray] = []
    for row in range(evaluation.size):
        key = (
            int(evaluation.run_id[row]),
            int(evaluation.event_id[row]),
            int(evaluation.source_tracklet_id[row]),
            int(evaluation.target_tracklet_id[row]),
            int(evaluation.q_over_p_mode[row]),
        )
        record_row = record_index.get(key)
        if record_row is None:
            raise ValueError(f"accepted evaluation record is absent from propagation input: {key}")
        propagated = np.asarray(records.covariance[record_row], dtype=np.float64)
        target = np.asarray(evaluation.combined_covariance[row] - propagated, dtype=np.float64)
        target = 0.5 * (target + target.T)
        if (
            not np.isfinite(propagated).all()
            or not np.isfinite(target).all()
            or np.any(np.diag(propagated) <= 0.0)
            or np.any(np.diag(target) <= 0.0)
        ):
            raise ValueError(f"invalid covariance terms for accepted propagation record: {key}")
        propagated_covariance.append(propagated)
        target_covariance.append(target)

    return PropagationModeComponents(
        run_id=np.asarray(evaluation.run_id, dtype=np.int64),
        event_id=np.asarray(evaluation.event_id, dtype=np.int64),
        source_tracklet_id=np.asarray(evaluation.source_tracklet_id, dtype=np.int32),
        target_tracklet_id=np.asarray(evaluation.target_tracklet_id, dtype=np.int32),
        source_station_id=np.asarray(evaluation.source_station_id, dtype=np.int16),
        target_station_id=np.asarray(evaluation.target_station_id, dtype=np.int16),
        truth_particle_id=np.asarray(evaluation.truth_particle_id, dtype=np.int64),
        q_over_p_mode=np.asarray(evaluation.q_over_p_mode, dtype=np.int8),
        residual=np.asarray(evaluation.residual, dtype=np.float64),
        pull=np.asarray(evaluation.pull, dtype=np.float64),
        propagated_covariance=np.asarray(propagated_covariance, dtype=np.float64),
        target_covariance=np.asarray(target_covariance, dtype=np.float64),
        combined_covariance=np.asarray(evaluation.combined_covariance, dtype=np.float64),
        chi2=np.asarray(evaluation.chi2, dtype=np.float64),
    )


def _component_index(
    components: PropagationModeComponents,
) -> dict[tuple[int, int, int, int, int, int], int]:
    index: dict[tuple[int, int, int, int, int, int], int] = {}
    for row in range(components.size):
        key = _comparison_identity(components, row)
        if key in index:
            raise ValueError(f"duplicate accepted propagation identity: {key}")
        index[key] = row
    return index


def compare_mode_components(
    mode0: PropagationModeComponents,
    mode1: PropagationModeComponents,
) -> PropagationModeComparison:
    """Compare mode 0 and 1 over shared truth-matched pair identities.

    For each shared pair the exact additive decomposition is

    ``chi2_1 - chi2_0 = (r_1^T S_0^-1 r_1 - chi2_0) +
    (chi2_1 - r_1^T S_0^-1 r_1)``.

    The first term is the raw-residual effect at fixed mode-0 covariance.  The
    second term is the covariance effect after the residual switch.
    """
    if np.any(mode0.q_over_p_mode != 0):
        raise ValueError("the first argument must contain only q/p mode 0 records")
    if np.any(mode1.q_over_p_mode != 1):
        raise ValueError("the second argument must contain only q/p mode 1 records")
    mode0_index = _component_index(mode0)
    mode1_index = _component_index(mode1)
    shared_keys = sorted(set(mode0_index) & set(mode1_index))

    metadata: dict[str, list[int]] = {
        "run_id": [],
        "event_id": [],
        "source_tracklet_id": [],
        "target_tracklet_id": [],
        "source_station_id": [],
        "target_station_id": [],
        "truth_particle_id": [],
    }
    chi2_mode0: list[float] = []
    chi2_after_residual_switch: list[float] = []
    chi2_mode1: list[float] = []
    chi2_mode0_residual_mode1_covariance: list[float] = []
    residual_effect: list[float] = []
    covariance_effect: list[float] = []
    determinant_mode0: list[float] = []
    determinant_mode1: list[float] = []
    logdet_mode0: list[float] = []
    logdet_mode1: list[float] = []
    mode0_determinant, mode0_logdet = _covariance_determinants(mode0.combined_covariance)
    mode1_determinant, mode1_logdet = _covariance_determinants(mode1.combined_covariance)

    for key in shared_keys:
        mode0_row = mode0_index[key]
        mode1_row = mode1_index[key]
        if int(mode0.truth_particle_id[mode0_row]) != int(mode1.truth_particle_id[mode1_row]):
            raise ValueError(f"truth particle identity drift between q/p modes: {key}")
        for name, values in metadata.items():
            values.append(int(getattr(mode0, name)[mode0_row]))
        baseline = float(mode0.chi2[mode0_row])
        switched_residual = mahalanobis_chi2(
            mode1.residual[mode1_row], mode0.combined_covariance[mode0_row]
        )
        final = float(mode1.chi2[mode1_row])
        reverse_counterfactual = mahalanobis_chi2(
            mode0.residual[mode0_row], mode1.combined_covariance[mode1_row]
        )
        chi2_mode0.append(baseline)
        chi2_after_residual_switch.append(switched_residual)
        chi2_mode1.append(final)
        chi2_mode0_residual_mode1_covariance.append(reverse_counterfactual)
        residual_effect.append(switched_residual - baseline)
        covariance_effect.append(final - switched_residual)
        determinant_mode0.append(float(mode0_determinant[mode0_row]))
        determinant_mode1.append(float(mode1_determinant[mode1_row]))
        logdet_mode0.append(float(mode0_logdet[mode0_row]))
        logdet_mode1.append(float(mode1_logdet[mode1_row]))

    return PropagationModeComparison(
        run_id=np.asarray(metadata["run_id"], dtype=np.int64),
        event_id=np.asarray(metadata["event_id"], dtype=np.int64),
        source_tracklet_id=np.asarray(metadata["source_tracklet_id"], dtype=np.int32),
        target_tracklet_id=np.asarray(metadata["target_tracklet_id"], dtype=np.int32),
        source_station_id=np.asarray(metadata["source_station_id"], dtype=np.int16),
        target_station_id=np.asarray(metadata["target_station_id"], dtype=np.int16),
        truth_particle_id=np.asarray(metadata["truth_particle_id"], dtype=np.int64),
        chi2_mode0=np.asarray(chi2_mode0, dtype=np.float64),
        chi2_mode1_residual_mode0_covariance=np.asarray(
            chi2_after_residual_switch, dtype=np.float64
        ),
        chi2_mode1=np.asarray(chi2_mode1, dtype=np.float64),
        chi2_mode0_residual_mode1_covariance=np.asarray(
            chi2_mode0_residual_mode1_covariance, dtype=np.float64
        ),
        residual_effect=np.asarray(residual_effect, dtype=np.float64),
        covariance_effect=np.asarray(covariance_effect, dtype=np.float64),
        determinant_mode0=np.asarray(determinant_mode0, dtype=np.float64),
        determinant_mode1=np.asarray(determinant_mode1, dtype=np.float64),
        logdet_mode0=np.asarray(logdet_mode0, dtype=np.float64),
        logdet_mode1=np.asarray(logdet_mode1, dtype=np.float64),
        mode0_only_count=len(set(mode0_index) - set(mode1_index)),
        mode1_only_count=len(set(mode1_index) - set(mode0_index)),
    )


def _stats(values: np.ndarray) -> dict[str, int | float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    summary: dict[str, int | float | None] = {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "mean": None,
        "median": None,
        "p95": None,
        "min": None,
        "max": None,
    }
    if finite.size:
        summary.update(
            {
                "mean": float(np.mean(finite)),
                "median": float(np.median(finite)),
                "p95": float(np.percentile(finite, 95.0)),
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
            }
        )
    return summary


def component_summary(components: PropagationModeComponents) -> dict[str, object]:
    """Return JSON-safe component statistics for one q/p mode."""
    determinant, logdet = _covariance_determinants(components.combined_covariance)
    propagated_sigma = np.sqrt(np.diagonal(components.propagated_covariance, axis1=1, axis2=2))
    target_sigma = np.sqrt(np.diagonal(components.target_covariance, axis1=1, axis2=2))
    combined_sigma = np.sqrt(np.diagonal(components.combined_covariance, axis1=1, axis2=2))
    return {
        "accepted_records": components.size,
        "residual": {name: _stats(components.residual[:, index]) for index, name in enumerate(STATE_NAMES)},
        "pull": {name: _stats(components.pull[:, index]) for index, name in enumerate(STATE_NAMES)},
        "propagated_sigma": {
            name: _stats(propagated_sigma[:, index]) for index, name in enumerate(STATE_NAMES)
        },
        "target_sigma": {
            name: _stats(target_sigma[:, index]) for index, name in enumerate(STATE_NAMES)
        },
        "combined_sigma": {
            name: _stats(combined_sigma[:, index]) for index, name in enumerate(STATE_NAMES)
        },
        "combined_covariance_determinant": _stats(determinant),
        "combined_covariance_logdet": _stats(logdet),
        "chi2": _stats(components.chi2),
    }


def comparison_summary(comparison: PropagationModeComparison) -> dict[str, object]:
    """Return JSON-safe aggregate mode-0 versus mode-1 diagnostics."""
    chi2_change = comparison.chi2_mode1 - comparison.chi2_mode0
    return {
        "paired_records": comparison.size,
        "mode0_only_records": comparison.mode0_only_count,
        "mode1_only_records": comparison.mode1_only_count,
        "chi2_mode0": _stats(comparison.chi2_mode0),
        "chi2_mode1_residual_mode0_covariance": _stats(
            comparison.chi2_mode1_residual_mode0_covariance
        ),
        "chi2_mode1": _stats(comparison.chi2_mode1),
        "chi2_change_mode1_minus_mode0": _stats(chi2_change),
        "chi2_residual_effect_hold_mode0_covariance": _stats(comparison.residual_effect),
        "chi2_covariance_effect_after_mode1_residual": _stats(comparison.covariance_effect),
        "combined_covariance_logdet_change_mode1_minus_mode0": _stats(
            comparison.logdet_mode1 - comparison.logdet_mode0
        ),
    }


def component_csv_rows(components: PropagationModeComponents) -> Iterator[dict[str, int | float]]:
    """Yield one raw audit row per accepted propagation record."""
    determinant, logdet = _covariance_determinants(components.combined_covariance)
    propagated_sigma = np.sqrt(np.diagonal(components.propagated_covariance, axis1=1, axis2=2))
    target_sigma = np.sqrt(np.diagonal(components.target_covariance, axis1=1, axis2=2))
    combined_sigma = np.sqrt(np.diagonal(components.combined_covariance, axis1=1, axis2=2))
    labels = ("x_mm", "y_mm", "tx", "ty")
    residual_labels = ("rx_mm", "ry_mm", "rtx", "rty")
    for row in range(components.size):
        result: dict[str, int | float] = {
            "run_id": int(components.run_id[row]),
            "event_id": int(components.event_id[row]),
            "source_tracklet_id": int(components.source_tracklet_id[row]),
            "target_tracklet_id": int(components.target_tracklet_id[row]),
            "source_station_id": int(components.source_station_id[row]),
            "target_station_id": int(components.target_station_id[row]),
            "truth_particle_id": int(components.truth_particle_id[row]),
            "q_over_p_mode": int(components.q_over_p_mode[row]),
            "combined_covariance_det": float(determinant[row]),
            "combined_covariance_logdet": float(logdet[row]),
            "chi2": float(components.chi2[row]),
        }
        for index, label in enumerate(labels):
            result[residual_labels[index]] = float(components.residual[row, index])
            result[f"pull_{label}"] = float(components.pull[row, index])
            result[f"propagated_sigma_{label}"] = float(propagated_sigma[row, index])
            result[f"target_sigma_{label}"] = float(target_sigma[row, index])
            result[f"combined_sigma_{label}"] = float(combined_sigma[row, index])
        yield result


def comparison_csv_rows(
    comparison: PropagationModeComparison,
) -> Iterator[dict[str, int | float]]:
    """Yield one same-pair mode comparison row."""
    for row in range(comparison.size):
        yield {
            "run_id": int(comparison.run_id[row]),
            "event_id": int(comparison.event_id[row]),
            "source_tracklet_id": int(comparison.source_tracklet_id[row]),
            "target_tracklet_id": int(comparison.target_tracklet_id[row]),
            "source_station_id": int(comparison.source_station_id[row]),
            "target_station_id": int(comparison.target_station_id[row]),
            "truth_particle_id": int(comparison.truth_particle_id[row]),
            "chi2_mode0": float(comparison.chi2_mode0[row]),
            "chi2_mode1_residual_mode0_covariance": float(
                comparison.chi2_mode1_residual_mode0_covariance[row]
            ),
            "chi2_mode1": float(comparison.chi2_mode1[row]),
            "chi2_mode0_residual_mode1_covariance": float(
                comparison.chi2_mode0_residual_mode1_covariance[row]
            ),
            "chi2_change_mode1_minus_mode0": float(
                comparison.chi2_mode1[row] - comparison.chi2_mode0[row]
            ),
            "chi2_residual_effect_hold_mode0_covariance": float(
                comparison.residual_effect[row]
            ),
            "chi2_covariance_effect_after_mode1_residual": float(
                comparison.covariance_effect[row]
            ),
            "combined_covariance_det_mode0": float(comparison.determinant_mode0[row]),
            "combined_covariance_det_mode1": float(comparison.determinant_mode1[row]),
            "combined_covariance_logdet_mode0": float(comparison.logdet_mode0[row]),
            "combined_covariance_logdet_mode1": float(comparison.logdet_mode1[row]),
            "combined_covariance_logdet_change_mode1_minus_mode0": float(
                comparison.logdet_mode1[row] - comparison.logdet_mode0[row]
            ),
        }
