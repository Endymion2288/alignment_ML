"""Exact inclusion-gap oracles for the two-compatible-rival counter-example."""

from __future__ import annotations

import pytest

from evaluation.route_counterfactuals import (
    MAX_BRUTE_FORCE_ROUTES,
    brute_force_inclusion_gap,
    inclusion_gap,
    single_rival_margin,
)
from models.route_energy import (
    CANONICAL_ENERGY_VERSION,
    RouteEnergyRecord,
    RouteEnergyTable,
    route_kind,
)


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
    # Truth utility 10; two compatible fakes of 6 that occupy different endpoints.
    return RouteEnergyTable(
        version=CANONICAL_ENERGY_VERSION,
        unmatched_penalty=-1.0,
        records=(
            _record(0, (0, 1, 2, 3), 10.0),
            _record(1, (0, 4), 6.0),
            _record(2, (2, 5), 6.0),
        ),
    )


def test_single_rival_margin_misses_compatible_set_competition():
    table = _two_rival_table()
    assert single_rival_margin(table, 0) == pytest.approx(4.0)
    gap = inclusion_gap(table, 0)
    assert gap.forced_in_objective == pytest.approx(10.0)
    assert gap.forced_out_objective == pytest.approx(12.0)
    assert gap.inclusion_gap == pytest.approx(-2.0)
    assert gap.forced_out_selected == (1, 2)


def test_inclusion_gap_matches_brute_force_on_the_two_rival_event():
    table = _two_rival_table()
    for route_id in (0, 1, 2):
        exact = inclusion_gap(table, route_id)
        assert exact.inclusion_gap == pytest.approx(brute_force_inclusion_gap(table, route_id))


def test_brute_force_refuses_oversized_tables():
    records = tuple(_record(index, (index, index + 100), 1.0) for index in range(MAX_BRUTE_FORCE_ROUTES + 1))
    table = RouteEnergyTable(
        version=CANONICAL_ENERGY_VERSION,
        unmatched_penalty=-1.0,
        records=records,
    )
    with pytest.raises(ValueError, match="brute-force"):
        brute_force_inclusion_gap(table, 0)


def test_unknown_route_id_is_rejected():
    with pytest.raises(ValueError, match="route_id"):
        inclusion_gap(_two_rival_table(), 99)
