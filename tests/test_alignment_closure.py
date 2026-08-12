from __future__ import annotations

import numpy as np

from alignment.closure import (
    AlignmentMeasurements,
    alignment_error_summary,
    inject_station_offsets,
    run_capture_range_scan,
    solve_alignment,
)


def _measurements() -> AlignmentMeasurements:
    source = np.asarray([0, 0, 0, 1, 1, 2], dtype=np.int16)
    target = np.asarray([1, 2, 3, 2, 3, 3], dtype=np.int16)
    nominal = np.asarray(
        [[2.0, -1.0], [3.0, 4.0], [-2.0, 1.5], [1.0, 2.5], [-3.0, 0.5], [0.2, -0.1]],
        dtype=np.float64,
    )
    covariance = np.tile(np.diag([0.04, 0.09]), (source.size, 1, 1))
    return AlignmentMeasurements(source, target, nominal, covariance)


def test_reference_fixed_alignment_closes_exactly_after_baseline_subtraction():
    measurements = _measurements()
    injected = {0: (0.0, 0.0), 1: (0.10, -0.06), 2: (-0.08, 0.12), 3: (0.05, 0.10)}
    observed = inject_station_offsets(measurements, injected, reference_station=0)

    fit = solve_alignment(measurements, observed, reference_station=0, refinement_iterations=3)
    summary = alignment_error_summary(fit, injected)

    assert fit.normal_matrix_rank == 6
    assert fit.normal_matrix_condition_number is not None
    assert fit.normal_matrix_condition_number > 1.0
    assert fit.active_counts == (6, 6, 6)
    assert np.allclose(fit.alignment_xy_mm[0], 0.0)
    assert summary["movable_station_max_norm_error_mm"] < 1.0e-12
    assert np.allclose(fit.final_increment_residual_xy_mm, 0.0, atol=1.0e-12)


def test_capture_scan_is_reproducible_for_unconstrained_gate():
    result = run_capture_range_scan(
        _measurements(),
        reference_station=0,
        magnitudes_mm=[0.0, 0.2],
        trials_per_magnitude=4,
        seed=17,
        chi2_gate=None,
        refinement_iterations=3,
        capture_tolerance_mm=1.0e-10,
    )

    assert [point["capture_fraction"] for point in result["points"]] == [1.0, 1.0]
    assert all(point["mean_active_fraction"] == 1.0 for point in result["points"])
