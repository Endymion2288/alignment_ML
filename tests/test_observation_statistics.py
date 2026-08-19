"""Unit tests for the physical-edge observation statistics semantics."""

from __future__ import annotations

import numpy as np
import pytest

from alignment.route_selected_update import apply_observation_statistics


def _system() -> tuple[list[tuple], list[np.ndarray], list[np.ndarray], list[tuple]]:
    """Three physical edges; the first is replicated three times."""
    rng = np.random.default_rng(11)
    keys = [("s", 1, 10, "sig-a", 0, 1), ("s", 1, 20, "sig-b", 0, 1), ("s", 1, 30, "sig-c", 0, 1), ("s", 1, 40, "sig-d", 0, 1)]
    physical = [("phys-a",), ("phys-a",), ("phys-a",), ("phys-b",)]
    residuals = []
    covariances = []
    for _bank in range(2):
        base = rng.normal(size=(4, 4))
        # Replicas of phys-a are numerically identical within a bank.
        base[1] = base[0]
        base[2] = base[0]
        residuals.append(base)
        cov = np.tile(np.eye(4)[None], (4, 1, 1))
        cov = cov * rng.uniform(0.5, 1.5, size=(4, 1, 1))
        cov[1] = cov[0]
        cov[2] = cov[0]
        covariances.append(cov)
    return keys, residuals, covariances, physical


def test_replica_weighted_is_identity() -> None:
    keys, residuals, covariances, physical = _system()
    out_keys, out_residuals, out_covariances, audit = apply_observation_statistics(
        keys, residuals, covariances, physical, "replica_weighted"
    )
    assert out_keys == keys
    assert audit["observations"] == 4
    assert audit["unique_physical_edges"] == 2
    for before, after in zip(residuals, out_residuals):
        np.testing.assert_array_equal(before, after)
    for before, after in zip(covariances, out_covariances):
        np.testing.assert_array_equal(before, after)


def test_deduplication_keeps_one_representative() -> None:
    keys, residuals, covariances, physical = _system()
    out_keys, out_residuals, out_covariances, audit = apply_observation_statistics(
        keys, residuals, covariances, physical, "physical_edge_deduplicated"
    )
    assert len(out_keys) == 2
    assert out_keys[0] == ("s", 1, 10, "sig-a", 0, 1)  # deterministic sorted-first
    assert audit["unique_physical_edges"] == 2
    assert audit["replicas_before_deduplication"] == 4
    assert audit["replica_multiplicity_histogram"] == {"1": 1, "3": 1}
    assert audit["max_replica_residual_spread"] == pytest.approx(0.0)
    for bank_index, bank in enumerate(out_residuals):
        np.testing.assert_array_equal(bank[0], residuals[bank_index][0])
        np.testing.assert_array_equal(bank[1], residuals[bank_index][3])


def test_inverse_multiplicity_matches_deduplication_in_wls() -> None:
    keys, residuals, covariances, physical = _system()
    dedup = apply_observation_statistics(
        keys, residuals, covariances, physical, "physical_edge_deduplicated"
    )
    inverse = apply_observation_statistics(
        keys, residuals, covariances, physical, "physical_edge_inverse_multiplicity_weighted"
    )

    def _solve(keys_, residuals_, covariances_):
        # Single-parameter toy WLS: derivative e_x for every edge.
        design = np.zeros((len(keys_), 4, 1))
        design[:, 0, 0] = 1.0
        normal = np.zeros((1, 1))
        rhs = np.zeros(1)
        for row in range(len(keys_)):
            inverse_cov = np.linalg.inv(covariances_[0][row])
            normal += design[row].T @ inverse_cov @ design[row]
            rhs += design[row].T @ inverse_cov @ residuals_[0][row]
        return float(np.linalg.solve(normal, rhs)[0])

    delta_dedup = _solve(*dedup[:3])
    delta_inverse = _solve(*inverse[:3])
    assert delta_dedup == pytest.approx(delta_inverse, rel=1e-12)
    # And both differ from the replica-weighted control (which triple-weights
    # the first physical edge).
    control = apply_observation_statistics(
        keys, residuals, covariances, physical, "replica_weighted"
    )
    assert abs(_solve(*control[:3]) - delta_dedup) > 1e-9


def test_missing_provenance_rejected_for_normalized_semantics() -> None:
    keys, residuals, covariances, _physical = _system()
    physical = [("phys-a",), None, ("phys-a",), ("phys-b",)]
    with pytest.raises(ValueError, match="physical edge"):
        apply_observation_statistics(
            keys, residuals, covariances, physical, "physical_edge_deduplicated"
        )
    # The legacy control does not require provenance.
    apply_observation_statistics(keys, residuals, covariances, physical, "replica_weighted")


def test_unknown_semantics_rejected() -> None:
    keys, residuals, covariances, physical = _system()
    with pytest.raises(ValueError, match="unknown observation statistics"):
        apply_observation_statistics(keys, residuals, covariances, physical, "dedup")
