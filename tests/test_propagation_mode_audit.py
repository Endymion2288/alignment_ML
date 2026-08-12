from __future__ import annotations

import numpy as np

from datasets.propagation_loader import PropagationRecords
from evaluation.field_propagation import FieldPropagationEvaluation
from evaluation.propagation_mode_audit import (
    PropagationModeComponents,
    compare_mode_components,
    components_from_field_evaluation,
)


def _components(mode: int, residual: np.ndarray, covariance: np.ndarray) -> PropagationModeComponents:
    chi2 = float(residual @ np.linalg.solve(covariance, residual))
    return PropagationModeComponents(
        run_id=np.asarray([7], dtype=np.int64),
        event_id=np.asarray([11], dtype=np.int64),
        source_tracklet_id=np.asarray([2], dtype=np.int32),
        target_tracklet_id=np.asarray([5], dtype=np.int32),
        source_station_id=np.asarray([0], dtype=np.int16),
        target_station_id=np.asarray([3], dtype=np.int16),
        truth_particle_id=np.asarray([17], dtype=np.int64),
        q_over_p_mode=np.asarray([mode], dtype=np.int8),
        residual=np.asarray([residual], dtype=np.float64),
        pull=np.asarray([residual / np.sqrt(np.diag(covariance))], dtype=np.float64),
        propagated_covariance=np.asarray([0.5 * covariance], dtype=np.float64),
        target_covariance=np.asarray([0.5 * covariance], dtype=np.float64),
        combined_covariance=np.asarray([covariance], dtype=np.float64),
        chi2=np.asarray([chi2], dtype=np.float64),
    )


def test_mode_comparison_decomposes_chi2_change_exactly():
    mode0 = _components(
        0,
        np.asarray([2.0, 0.0, 0.0, 0.0]),
        np.diag([4.0, 9.0, 16.0, 25.0]),
    )
    mode1 = _components(
        1,
        np.asarray([3.0, 0.0, 0.0, 0.0]),
        np.diag([1.0, 9.0, 16.0, 25.0]),
    )

    comparison = compare_mode_components(mode0, mode1)

    assert comparison.size == 1
    assert comparison.mode0_only_count == 0
    assert comparison.mode1_only_count == 0
    assert np.allclose(comparison.chi2_mode0, [1.0])
    assert np.allclose(comparison.chi2_mode1_residual_mode0_covariance, [2.25])
    assert np.allclose(comparison.chi2_mode1, [9.0])
    assert np.allclose(comparison.residual_effect, [1.25])
    assert np.allclose(comparison.covariance_effect, [6.75])
    assert np.allclose(
        comparison.chi2_mode1 - comparison.chi2_mode0,
        comparison.residual_effect + comparison.covariance_effect,
    )


def test_component_audit_recovers_target_covariance_from_combined_term():
    propagated = np.diag([1.0, 2.0, 3.0, 4.0])
    target = np.diag([2.0, 2.0, 2.0, 2.0])
    evaluation = FieldPropagationEvaluation(
        run_id=np.asarray([7], dtype=np.int64),
        event_id=np.asarray([11], dtype=np.int64),
        source_tracklet_id=np.asarray([2], dtype=np.int32),
        target_tracklet_id=np.asarray([5], dtype=np.int32),
        source_station_id=np.asarray([0], dtype=np.int16),
        target_station_id=np.asarray([3], dtype=np.int16),
        truth_particle_id=np.asarray([17], dtype=np.int64),
        q_over_p_mode=np.asarray([0], dtype=np.int8),
        residual=np.zeros((1, 4), dtype=np.float64),
        pull=np.zeros((1, 4), dtype=np.float64),
        chi2=np.asarray([0.0]),
        line_residual=np.zeros((1, 4), dtype=np.float64),
        line_pull=np.zeros((1, 4), dtype=np.float64),
        line_chi2=np.asarray([0.0]),
        combined_covariance=np.asarray([propagated + target]),
        rejected_counts={"accepted": 1},
    )
    records = PropagationRecords(
        run_id=np.asarray([7], dtype=np.int64),
        event_id=np.asarray([11], dtype=np.int64),
        source_tracklet_id=np.asarray([2], dtype=np.int32),
        target_tracklet_id=np.asarray([5], dtype=np.int32),
        source_station_id=np.asarray([0], dtype=np.int16),
        target_station_id=np.asarray([3], dtype=np.int16),
        truth_particle_id=np.asarray([17], dtype=np.int64),
        target_z_mm=np.asarray([10.0]),
        prediction=np.zeros((1, 4), dtype=np.float64),
        covariance=np.asarray([propagated]),
        success=np.asarray([True]),
        has_covariance=np.asarray([True]),
        q_over_p_mode=np.asarray([0], dtype=np.int8),
    )

    components = components_from_field_evaluation(evaluation, records)

    assert components.size == 1
    assert np.allclose(components.propagated_covariance[0], propagated)
    assert np.allclose(components.target_covariance[0], target)
