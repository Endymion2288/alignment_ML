"""Exact forced-in / forced-out oracles for a versioned route-energy table.

The local packing-competition loss compares a truth route with its single
strongest conflicting rival.  That margin can stay positive while two
compatible rivals jointly beat the truth set.  The inclusion gap

    M_r = max_{Y ∋ r} U(Y) − max_{Y ∌ r} U(Y)

is the exact set-packing quantity.  Ties are compared on objective value,
not on a particular selected set.  Brute-force enumeration is provided only
as a small-event oracle; it fail-closes above ``MAX_BRUTE_FORCE_ROUTES``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np

from baselines.route_assignment import solve_unit_capacity_route_packing
from models.route_energy import RouteEnergyTable, require_finite


MAX_BRUTE_FORCE_ROUTES = 16


def _record_index(table: RouteEnergyTable, route_id: int) -> int:
    for index, record in enumerate(table.records):
        if int(record.route_id) == int(route_id):
            return index
    raise ValueError(f"route_id {route_id} is not in the energy table")


def _conflicts(left: Sequence[Hashable], right: Sequence[Hashable]) -> bool:
    return bool(set(left).intersection(right))


def _solve(endpoints: Sequence[tuple[Hashable, ...]], energies: Sequence[float]):
    values = np.asarray(list(energies), dtype=np.float64)
    if values.size and not np.isfinite(values).all():
        raise ValueError("counterfactual energies must be finite")
    return solve_unit_capacity_route_packing(list(endpoints), values)


@dataclass(frozen=True)
class RouteCounterfactual:
    route_id: int
    forced_in_objective: float
    forced_out_objective: float
    inclusion_gap: float
    single_rival_margin: float | None
    forced_in_selected: tuple[int, ...]
    forced_out_selected: tuple[int, ...]


def forced_in_packing(table: RouteEnergyTable, route_id: int) -> tuple[float, np.ndarray]:
    """Exact best feasible set that contains ``route_id``."""
    index = _record_index(table, route_id)
    record = table.records[index]
    rest_endpoints: list[tuple[Hashable, ...]] = []
    rest_energies: list[float] = []
    rest_ids: list[int] = []
    for other in table.records:
        if int(other.route_id) == int(route_id):
            continue
        if _conflicts(record.endpoints, other.endpoints):
            continue
        rest_endpoints.append(other.endpoints)
        rest_energies.append(float(other.energy))
        rest_ids.append(int(other.route_id))
    packing = _solve(rest_endpoints, rest_energies)
    selected = [int(route_id)]
    selected.extend(route for route, flag in zip(rest_ids, packing.selected) if flag)
    return float(record.energy + packing.objective), np.asarray(sorted(selected), dtype=np.int64)


def forced_out_packing(table: RouteEnergyTable, route_id: int) -> tuple[float, np.ndarray]:
    """Exact best feasible set that excludes ``route_id``."""
    endpoints = [record.endpoints for record in table.records if int(record.route_id) != int(route_id)]
    energies = [float(record.energy) for record in table.records if int(record.route_id) != int(route_id)]
    ids = [int(record.route_id) for record in table.records if int(record.route_id) != int(route_id)]
    packing = _solve(endpoints, energies)
    selected = [route for route, flag in zip(ids, packing.selected) if flag]
    return float(packing.objective), np.asarray(sorted(selected), dtype=np.int64)


def single_rival_margin(table: RouteEnergyTable, route_id: int) -> float | None:
    """Local margin versus the strongest single conflicting rival, or dustbin 0."""
    index = _record_index(table, route_id)
    record = table.records[index]
    strongest: float | None = None
    for other in table.records:
        if int(other.route_id) == int(route_id):
            continue
        if not _conflicts(record.endpoints, other.endpoints):
            continue
        energy = float(other.energy)
        strongest = energy if strongest is None else max(strongest, energy)
    rival = 0.0 if strongest is None else max(float(strongest), 0.0)
    return float(record.energy - rival)


def inclusion_gap(table: RouteEnergyTable, route_id: int) -> RouteCounterfactual:
    """Exact set-packing inclusion gap for one route."""
    _record_index(table, route_id)
    forced_in, forced_in_selected = forced_in_packing(table, route_id)
    forced_out, forced_out_selected = forced_out_packing(table, route_id)
    require_finite("forced_in_objective", forced_in)
    require_finite("forced_out_objective", forced_out)
    return RouteCounterfactual(
        route_id=int(route_id),
        forced_in_objective=float(forced_in),
        forced_out_objective=float(forced_out),
        inclusion_gap=float(forced_in - forced_out),
        single_rival_margin=single_rival_margin(table, route_id),
        forced_in_selected=tuple(int(value) for value in forced_in_selected.tolist()),
        forced_out_selected=tuple(int(value) for value in forced_out_selected.tolist()),
    )


def inclusion_gaps(table: RouteEnergyTable) -> dict[int, RouteCounterfactual]:
    return {int(record.route_id): inclusion_gap(table, record.route_id) for record in table.records}


def brute_force_best_objective(
    endpoints: Sequence[Sequence[Hashable]],
    energies: Sequence[float],
    *,
    require_index: int | None = None,
    forbid_index: int | None = None,
) -> tuple[float, np.ndarray]:
    """Enumerate every feasible subset.  Empty dustbin assignment scores 0."""
    values = np.asarray(list(energies), dtype=np.float64)
    rows = [tuple(row) for row in endpoints]
    if values.shape != (len(rows),):
        raise ValueError("brute-force energies must align with endpoints")
    if values.size and not np.isfinite(values).all():
        raise ValueError("brute-force energies must be finite")
    if len(rows) > MAX_BRUTE_FORCE_ROUTES:
        raise ValueError(
            f"brute-force oracle refuses tables larger than {MAX_BRUTE_FORCE_ROUTES} routes"
        )
    if require_index is not None and (require_index < 0 or require_index >= len(rows)):
        raise ValueError("require_index is outside the route table")
    if forbid_index is not None and (forbid_index < 0 or forbid_index >= len(rows)):
        raise ValueError("forbid_index is outside the route table")
    # The empty dustbin set is feasible only when no route is required.
    best = float("-inf") if require_index is not None else 0.0
    best_mask = np.zeros(len(rows), dtype=bool)
    limit = 1 << len(rows)
    for bits in range(limit):
        chosen = [index for index in range(len(rows)) if bits & (1 << index)]
        if require_index is not None and require_index not in chosen:
            continue
        if forbid_index is not None and forbid_index in chosen:
            continue
        used: set[Hashable] = set()
        feasible = True
        for index in chosen:
            row = set(rows[index])
            if used.intersection(row):
                feasible = False
                break
            used.update(row)
        if not feasible:
            continue
        objective = float(values[chosen].sum()) if chosen else 0.0
        mask = np.zeros(len(rows), dtype=bool)
        mask[chosen] = True
        if objective > best + 1.0e-15:
            best = objective
            best_mask = mask
            continue
        if abs(objective - best) <= 1.0e-15 and tuple(mask.tolist()) < tuple(best_mask.tolist()):
            best = objective
            best_mask = mask
    if require_index is not None and not math.isfinite(best):
        raise RuntimeError("required route has no feasible set")
    return float(best), best_mask


def brute_force_inclusion_gap(table: RouteEnergyTable, route_id: int) -> float:
    index = _record_index(table, route_id)
    forced_in, _ = brute_force_best_objective(
        table.endpoint_rows(), table.energies(), require_index=index
    )
    forced_out, _ = brute_force_best_objective(
        table.endpoint_rows(), table.energies(), forbid_index=index
    )
    return float(forced_in - forced_out)
