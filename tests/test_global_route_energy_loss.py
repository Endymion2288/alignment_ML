"""Global inclusion-gap / loss-augmented hinge is not a larger local margin."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from evaluation.route_counterfactuals import inclusion_gap
from models.route_energy import (
    CANONICAL_ENERGY_VERSION,
    RouteEnergyRecord,
    RouteEnergyTable,
    route_kind,
)
from training.global_route_energy_loss import (
    hamming_loss_augmentation,
    inclusion_gap_hinge,
    loss_augmented_structured_hinge,
)
from training.gauge_consistent_route import packing_route_competition_loss


def _record(route_id: int, endpoints: tuple, energy: float) -> RouteEnergyRecord:
    n_stations = len(endpoints)
    return RouteEnergyRecord(
        route_id=route_id,
        endpoints=endpoints,
        energy=energy,
        n_stations=n_stations,
        kind=route_kind(n_stations),
        contract=CANONICAL_ENERGY_VERSION,
    )


def _two_rival_table() -> RouteEnergyTable:
    return RouteEnergyTable(
        version=CANONICAL_ENERGY_VERSION,
        unmatched_penalty=-1.0,
        records=(
            _record(0, (0, 1, 2, 3), 10.0),
            _record(1, (0, 4), 6.0),
            _record(2, (2, 5), 6.0),
        ),
    )


def test_local_single_rival_hinge_can_be_zero_while_the_set_gap_is_violated():
    table = _two_rival_table()
    local = 10.0 - 6.0
    assert local > 1.0
    assert inclusion_gap(table, 0).inclusion_gap == pytest.approx(-2.0)
    report = inclusion_gap_hinge(table, [0], margin=0.0)
    assert report["loss"] == pytest.approx(2.0)
    assert report["violation_fraction"] == 1.0


def test_loss_augmented_inference_selects_the_compatible_rival_set():
    table = _two_rival_table()
    energies = torch.tensor([10.0, 6.0, 6.0], dtype=torch.float64, requires_grad=True)
    result = loss_augmented_structured_hinge(
        energies,
        [record.endpoints for record in table.records],
        [True, False, False],
        margin=1.0,
    )
    np.testing.assert_array_equal(result["competitor_selected"], np.asarray([False, True, True]))
    assert result["raw_margin"] == pytest.approx(5.0)
    assert float(result["loss"].detach()) == pytest.approx(5.0)
    result["loss"].backward()
    assert energies.grad is not None
    # Gradients flow through the selected competitor minus the target set.
    np.testing.assert_allclose(energies.grad.detach().cpu().numpy(), np.asarray([-1.0, 1.0, 1.0]))


def test_hamming_augmentation_penalizes_false_inclusions_and_omissions():
    np.testing.assert_allclose(hamming_loss_augmentation([True, False, False]), np.asarray([-1.0, 1.0, 1.0]))


def test_packing_route_competition_loss_stays_on_the_legacy_single_rival_path():
    # The historical helper remains imported and is not this interface.
    assert packing_route_competition_loss.__doc__ is not None
    assert "strongest feasible rival" in packing_route_competition_loss.__doc__
