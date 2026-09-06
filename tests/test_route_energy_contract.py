"""Workbook-77 raw_energy_v1 contract: identity, saturation, and length mix.

These tests are deterministic and do not load ROOT, development, final-blind,
or sealed-test assets.  Historical ``legacy_prob_clip_logit_v1`` is exercised
only to prove that the documented zero-correction identity violation still
reproduces and that new experiments cannot silently use that decoder.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from baselines.route_assignment import solve_unit_capacity_route_packing
from models.route_energy import (
    CANONICAL_ENERGY_VERSION,
    DUSTBIN_ENERGY,
    LEGACY_DECODER_VERSION,
    LOGIT_CLIP,
    assign_from_energy_table,
    build_route_energy_table,
    canonical_route_energy,
    diagnostic_probability,
    legacy_clip_logit,
    legacy_complete_replace_energy,
    legacy_complete_residual_energy,
    legacy_fragment_energy,
)


UNMATCHED = -1.0
EDGE8 = (8.0, 8.0, 8.0)
COMPLETE = ((0, 0), (1, 1), (2, 2), (3, 3))
PREFIX3 = ((0, 0), (1, 1), (2, 2))
SUFFIX3 = ((1, 1), (2, 2), (3, 3))
PAIR01 = ((0, 0), (1, 1))
PAIR12 = ((1, 1), (2, 2))
PAIR23 = ((2, 2), (3, 3))


def _mixed_length_routes(edge_logits=EDGE8, correction: float = 0.0):
    e01, e12, e23 = edge_logits
    return (
        (PAIR01, (e01,), 0.0),
        (PAIR12, (e12,), 0.0),
        (PAIR23, (e23,), 0.0),
        (PREFIX3, (e01, e12), 0.0),
        (SUFFIX3, (e12, e23), 0.0),
        (COMPLETE, edge_logits, correction),
    )


def test_canonical_zero_correction_is_the_sum_of_raw_edge_energies():
    complete = canonical_route_energy(EDGE8, unmatched_penalty=UNMATCHED, correction=0.0)
    prefix = canonical_route_energy(EDGE8[:2], unmatched_penalty=UNMATCHED)
    assert complete == pytest.approx(20.0)
    assert prefix == pytest.approx(13.0)
    assert complete > prefix


def test_legacy_zero_correction_breaks_complete_versus_fragment_identity():
    # Review counter-example: three logits of 8 and unmatched = -1.
    complete = legacy_complete_replace_energy(sum(EDGE8), unmatched_penalty=UNMATCHED)
    prefix = legacy_fragment_energy(EDGE8[:2], unmatched_penalty=UNMATCHED)
    assert complete == pytest.approx(LOGIT_CLIP + 4.0 * UNMATCHED)
    assert complete == pytest.approx(9.815510557964274)
    assert prefix == pytest.approx(13.0)
    assert complete < prefix


def test_legacy_residual_weight_zero_matches_per_edge_clip_identity():
    edge_only = legacy_fragment_energy(EDGE8, unmatched_penalty=UNMATCHED, n_stations=4)
    residual = legacy_complete_residual_energy(
        EDGE8, sum(EDGE8), weight=0.0, unmatched_penalty=UNMATCHED
    )
    assert residual == pytest.approx(edge_only)
    assert residual == pytest.approx(20.0)


def test_saturated_logits_stay_raw_in_canonical_and_clip_in_legacy():
    saturated = (100.0, -100.0, 100.0)
    canonical = canonical_route_energy(saturated, unmatched_penalty=UNMATCHED)
    assert canonical == pytest.approx(100.0 - 100.0 + 100.0 + 4.0 * UNMATCHED)
    assert legacy_clip_logit(100.0) == pytest.approx(LOGIT_CLIP)
    assert legacy_clip_logit(-100.0) == pytest.approx(-LOGIT_CLIP)
    replaced = legacy_complete_replace_energy(sum(saturated), unmatched_penalty=UNMATCHED)
    assert replaced == pytest.approx(LOGIT_CLIP + 4.0 * UNMATCHED)
    assert replaced != pytest.approx(canonical)


def test_nan_and_inf_are_rejected_fail_closed():
    with pytest.raises(ValueError, match="finite"):
        canonical_route_energy((8.0, math.nan, 8.0), unmatched_penalty=UNMATCHED)
    with pytest.raises(ValueError, match="finite"):
        canonical_route_energy(EDGE8, unmatched_penalty=math.inf)
    with pytest.raises(ValueError, match="finite"):
        legacy_clip_logit(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        build_route_energy_table(((COMPLETE, (8.0, 8.0, float("inf"))),), unmatched_penalty=UNMATCHED)


def test_two_three_and_four_station_routes_share_one_energy_space():
    table = build_route_energy_table(_mixed_length_routes(), unmatched_penalty=UNMATCHED)
    by_kind = {record.kind: record.energy for record in table.records}
    assert set(by_kind) == {"pair", "fragment3", "complete4"}
    assert by_kind["pair"] == pytest.approx(6.0)
    assert by_kind["fragment3"] == pytest.approx(13.0)
    assert by_kind["complete4"] == pytest.approx(20.0)
    assert table.version == CANONICAL_ENERGY_VERSION


def test_canonical_zero_correction_assignment_selects_the_complete_route():
    table = build_route_energy_table(_mixed_length_routes(), unmatched_penalty=UNMATCHED)
    result = assign_from_energy_table(table)
    selected = [record.kind for record, flag in zip(table.records, result.selected) if flag]
    assert selected == ["complete4"]
    assert result.objective == pytest.approx(20.0)


def test_legacy_zero_correction_assignment_selects_the_fragment():
    table = build_route_energy_table(
        _mixed_length_routes(),
        unmatched_penalty=UNMATCHED,
        contract=LEGACY_DECODER_VERSION,
    )
    result = assign_from_energy_table(table, allow_legacy=True)
    selected = [record.kind for record, flag in zip(table.records, result.selected) if flag]
    assert "complete4" not in selected
    assert "fragment3" in selected
    complete = next(record for record in table.records if record.kind == "complete4")
    assert complete.energy == pytest.approx(9.815510557964274)


def test_new_experiments_cannot_silently_solve_a_legacy_table():
    table = build_route_energy_table(
        _mixed_length_routes(),
        unmatched_penalty=UNMATCHED,
        contract=LEGACY_DECODER_VERSION,
    )
    with pytest.raises(ValueError, match="legacy_prob_clip_logit_v1"):
        assign_from_energy_table(table)


def test_canonical_assignment_matches_the_generic_unit_capacity_solver():
    table = build_route_energy_table(_mixed_length_routes(), unmatched_penalty=UNMATCHED)
    direct = solve_unit_capacity_route_packing(table.endpoint_rows(), table.energies())
    wrapped = assign_from_energy_table(table)
    np.testing.assert_array_equal(wrapped.selected, direct.selected)
    assert wrapped.objective == pytest.approx(direct.objective)


def test_zero_correction_does_not_change_canonical_energy_or_assignment():
    plain = build_route_energy_table(_mixed_length_routes(correction=0.0), unmatched_penalty=UNMATCHED)
    omitted = build_route_energy_table(
        tuple((endpoints, edges) for endpoints, edges, _correction in _mixed_length_routes()),
        unmatched_penalty=UNMATCHED,
    )
    np.testing.assert_allclose(plain.energies(), omitted.energies())
    assert assign_from_energy_table(plain).objective == pytest.approx(
        assign_from_energy_table(omitted).objective
    )


def test_diagnostic_probability_is_not_required_for_assignment():
    table = build_route_energy_table(
        _mixed_length_routes(),
        unmatched_penalty=UNMATCHED,
        include_diagnostic_probability=True,
    )
    assert all(record.diagnostic_probability is not None for record in table.records)
    assert diagnostic_probability(20.0) == pytest.approx(1.0 / (1.0 + math.exp(-20.0)))
    result = assign_from_energy_table(table)
    assert result.objective == pytest.approx(20.0)
    assert table.dustbin_energy == DUSTBIN_ENERGY


def test_empty_table_assigns_the_dustbin():
    table = build_route_energy_table((), unmatched_penalty=UNMATCHED)
    result = assign_from_energy_table(table)
    assert result.selected.size == 0
    assert result.objective == pytest.approx(0.0)
