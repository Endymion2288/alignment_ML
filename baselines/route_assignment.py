"""Adjacent-station, unit-capacity route assignment for four-station tracks.

The solver deliberately consumes only the three directed adjacent score
matrices ``IFT -> S1 -> S2 -> S3``.  It does not infer a direct long-baseline
edge, inspect truth labels, or require a complete clique as the earlier
multi-station baseline does.  Enumerating the small contiguous route set and
solving a binary set-packing problem is equivalent to a unit-capacity
min-cost-flow with optional dustbins for the controlled synthetic occupancy.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Hashable, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from baselines.field_chi2_matching import FieldCandidate, ScoredMatch
from baselines.global_assignment import ScoreMatrix
from datasets.root_loader import EventTracklets


# A deterministic continuity preference resolves exact decomposition ties in
# the route flow objective (for example, one 0-1-2-3 route versus disjoint
# 0-1 and 2-3 pieces with identical edge log-odds).  It is much smaller than
# the probability precision used by the MLP and never turns a non-positive
# dustbin improvement into a selected route.
_CONTINUATION_TIE_BREAK = 1.0e-9
_ORDERED_HYPEREDGE_DP_MAX_STATES = 200_000


@dataclass(frozen=True)
class RouteAssignmentConfig:
    """Fixed, truth-free controls for a contiguous four-station route solve."""

    score_threshold_by_pair: Mapping[tuple[int, int], float]
    unmatched_penalty: float = 0.0
    station_path: tuple[int, ...] = (0, 1, 2, 3)
    maximum_hypotheses: int = 100_000
    # Optional V2 route-query control.  A complete four-station route query
    # can either replace the independent edge utility (the original direct
    # query study) or act as a residual correction around it.  The latter
    # preserves the edge evidence that partial routes use, so a complete route
    # is not unfairly disadvantaged against a decomposition into local pieces.
    # Partial routes always retain their physical edge scores for
    # missing-station/dustbin recovery.
    complete_route_score_threshold: float | None = None
    complete_route_score_composition: str = "replace"
    complete_route_context_weight: float = 1.0


@dataclass(frozen=True)
class Route:
    """One selected contiguous path and its adjacent physical edges."""

    endpoints: tuple[tuple[int, int], ...]
    matches: tuple[tuple[tuple[int, int], ScoredMatch], ...]
    utility: float
    complete_route_score: float | None = None


@dataclass(frozen=True)
class RouteAssignmentResult:
    """Route selection plus explicit dustbin endpoints by station."""

    routes: tuple[Route, ...]
    matches_by_pair: Mapping[tuple[int, int], tuple[ScoredMatch, ...]]
    unmatched_by_station: Mapping[int, tuple[int, ...]]
    candidate_edges_above_threshold: int
    candidate_edges_above_threshold_by_pair: Mapping[tuple[int, int], int]
    hypotheses_considered: int
    selected_routes: int


@dataclass(frozen=True)
class UnitCapacityPackingResult:
    """Exact selection for a fixed set of pre-existing route hypotheses.

    This generic helper is intentionally lower-level than ``Route``.  It is
    used by V3 structured learning to obtain a loss-augmented competing
    assignment from the *same* physical route candidates that inference uses.
    ``endpoint_rows`` contains arbitrary hashable endpoint IDs; it does not
    create a candidate edge, inspect truth, or assume a station topology.
    """

    selected: np.ndarray
    objective: float
    endpoints: tuple[Hashable, ...]
    endpoint_loads: np.ndarray


def adjacent_station_pairs(station_path: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Return ordered physical edges for a route path."""
    stations = tuple(int(station) for station in station_path)
    if len(stations) < 2 or len(set(stations)) != len(stations):
        raise ValueError("station_path must contain at least two unique stations")
    if any(left >= right for left, right in zip(stations, stations[1:])):
        raise ValueError("station_path must be strictly forward ordered")
    return tuple(zip(stations, stations[1:]))


def _solve_ordered_hyperedge_component(
    rows: Sequence[tuple[Hashable, ...]],
    utilities: np.ndarray,
) -> np.ndarray | None:
    """Exactly solve a small fixed-order multipartite hyperedge component.

    V3 complete routes are ordered IFT/S1/S2/S3 tuples.  When every endpoint
    occurs in exactly one tuple position, routes can be processed one IFT
    endpoint group at a time while a bit mask tracks occupied later-station
    endpoints.  This is an exact dynamic programme, not a relaxation.  The
    generic MILP remains the fallback for mixed-length routes, non-partitioned
    endpoint IDs, and unusually large state spaces.
    """
    if not rows or utilities.shape != (len(rows),):
        raise ValueError("ordered hyperedge utilities must align with component rows")
    width = len(rows[0])
    if width < 2 or any(len(row) != width for row in rows):
        return None
    position_by_endpoint: dict[Hashable, int] = {}
    for row in rows:
        for position, endpoint in enumerate(row):
            prior = position_by_endpoint.setdefault(endpoint, position)
            if prior != position:
                return None

    by_first_endpoint: dict[Hashable, list[int]] = {}
    for route, row in enumerate(rows):
        by_first_endpoint.setdefault(row[0], []).append(route)
    later_endpoints = tuple(
        sorted(
            (endpoint for endpoint, position in position_by_endpoint.items() if position > 0),
            key=repr,
        )
    )
    # A dynamic programme over more than 200k masks is not a reliable fast
    # path.  Returning None preserves the established exact MILP behaviour.
    if len(later_endpoints) >= 63 or 2 ** len(later_endpoints) > _ORDERED_HYPEREDGE_DP_MAX_STATES:
        return None
    bit_by_endpoint = {endpoint: bit for bit, endpoint in enumerate(later_endpoints)}
    masks = np.zeros(len(rows), dtype=np.uint64)
    for route, row in enumerate(rows):
        mask = 0
        for endpoint in row[1:]:
            mask |= 1 << bit_by_endpoint[endpoint]
        masks[route] = np.uint64(mask)

    # ``records`` retain one predecessor per reachable occupancy state.  The
    # augmented utilities already carry the generic solver's deterministic
    # tiny tie-break, so a strict comparison is sufficient.
    current: dict[int, float] = {0: 0.0}
    layers: list[dict[int, tuple[float, int, int]]] = []
    for endpoint in sorted(by_first_endpoint, key=repr):
        records: dict[int, tuple[float, int, int]] = {}
        for occupied, utility in current.items():
            records[occupied] = (utility, occupied, -1)
        for occupied, utility in current.items():
            for route in by_first_endpoint[endpoint]:
                route_mask = int(masks[route])
                if occupied & route_mask:
                    continue
                updated = occupied | route_mask
                candidate = utility + float(utilities[route])
                prior = records.get(updated)
                if prior is None or candidate > prior[0]:
                    records[updated] = (candidate, occupied, route)
        if len(records) > _ORDERED_HYPEREDGE_DP_MAX_STATES:
            return None
        layers.append(records)
        current = {occupied: record[0] for occupied, record in records.items()}
    selected = np.zeros(len(rows), dtype=bool)
    occupied = max(current, key=lambda mask: current[mask])
    for records in reversed(layers):
        _, predecessor, route = records[occupied]
        if route >= 0:
            selected[route] = True
        occupied = predecessor
    return selected


def solve_unit_capacity_route_packing(
    endpoint_rows: Sequence[Sequence[Hashable]] | np.ndarray,
    utilities: Sequence[float] | np.ndarray,
) -> UnitCapacityPackingResult:
    """Solve a unit-capacity route set-packing problem exactly.

    The returned Boolean mask is aligned to the supplied route rows.  Empty
    selection is always feasible, so non-positive utilities are excluded from
    the optimization rather than being selected merely by a deterministic tie.
    This is important for the structured hinge: a loss-augmented competitor
    must improve the objective rather than be an arbitrary zero-utility route.
    """
    raw_rows = list(endpoint_rows)
    values = np.asarray(utilities, dtype=np.float64)
    if values.ndim != 1 or values.size != len(raw_rows):
        raise ValueError("route utilities must be a finite vector aligned with endpoint rows")
    if not np.isfinite(values).all():
        raise ValueError("route utilities must be finite")
    if not raw_rows:
        return UnitCapacityPackingResult(
            selected=np.empty(0, dtype=bool),
            objective=0.0,
            endpoints=(),
            endpoint_loads=np.empty(0, dtype=np.int64),
        )

    normalized_rows: list[tuple[Hashable, ...]] = []
    for row in raw_rows:
        endpoints = tuple(row)
        if not endpoints:
            raise ValueError("every route must contain at least one endpoint")
        try:
            unique = set(endpoints)
        except TypeError as error:
            raise ValueError("route endpoint IDs must be hashable") from error
        if len(unique) != len(endpoints):
            raise ValueError("one route repeats an endpoint")
        normalized_rows.append(endpoints)

    # Selecting a non-positive route can never improve against the explicit
    # dustbin/empty assignment.  Solve the positive subset and reconstruct the
    # full candidate-aligned mask afterwards.
    active = np.flatnonzero(values > 0.0)
    selected = np.zeros(values.shape, dtype=bool)
    if not active.size:
        endpoints = tuple(sorted({endpoint for row in normalized_rows for endpoint in row}, key=repr))
        return UnitCapacityPackingResult(
            selected=selected,
            objective=0.0,
            endpoints=endpoints,
            endpoint_loads=np.zeros(len(endpoints), dtype=np.int64),
        )

    active_rows = [normalized_rows[int(index)] for index in active]
    endpoints = tuple(sorted({endpoint for row in active_rows for endpoint in row}, key=repr))
    endpoint_row = {endpoint: row for row, endpoint in enumerate(endpoints)}

    # A synthetic overlay can contain many mutually disconnected trajectory
    # groups.  Unit-capacity constraints never couple such groups, so solving
    # each route-conflict component independently is exactly equivalent to one
    # monolithic MILP.  This keeps V3's loss-augmented oracle exact while
    # avoiding one solver invocation over unrelated tracks in every event.
    memberships: dict[Hashable, list[int]] = {}
    for column, route in enumerate(active_rows):
        for endpoint in route:
            memberships.setdefault(endpoint, []).append(column)
    visited = np.zeros(active.size, dtype=bool)
    components: list[np.ndarray] = []
    for root in range(active.size):
        if visited[root]:
            continue
        pending = [int(root)]
        visited[root] = True
        component: list[int] = []
        while pending:
            column = pending.pop()
            component.append(column)
            for endpoint in active_rows[column]:
                for neighbor in memberships[endpoint]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        pending.append(neighbor)
        components.append(np.asarray(sorted(component), dtype=np.int64))

    active_utilities = values[active]
    # Keep exact objective differences intact.  The bounded perturbation only
    # resolves mathematically identical set-packings reproducibly.  Its index
    # remains global to the active route list even when the problem splits.
    tie_break = np.linspace(0.0, 1.0e-12, active.size, dtype=np.float64)
    selected_active = np.zeros(active.size, dtype=bool)
    for component in components:
        component_rows = [active_rows[int(column)] for column in component]
        component_endpoints = tuple(
            sorted({endpoint for route in component_rows for endpoint in route}, key=repr)
        )
        component_endpoint_row = {
            endpoint: row for row, endpoint in enumerate(component_endpoints)
        }
        incidence = np.zeros(
            (len(component_endpoints), component.size), dtype=np.float64
        )
        for local_column, route in enumerate(component_rows):
            for endpoint in route:
                incidence[component_endpoint_row[endpoint], local_column] = 1.0

        # If no endpoint is shared, all active routes have positive utility
        # and can be selected directly.  This is common for clean multi-track
        # events and is still the exact empty-dustbin optimum.
        if component.size == 1 or np.all(incidence.sum(axis=1) <= 1.0):
            selected_active[component] = True
            continue
        adjusted_utilities = active_utilities[component] - tie_break[component]
        ordered_solution = _solve_ordered_hyperedge_component(
            component_rows, adjusted_utilities
        )
        if ordered_solution is not None:
            selected_active[component] = ordered_solution
            continue
        result = milp(
            c=-adjusted_utilities,
            integrality=np.ones(component.size, dtype=np.int8),
            bounds=Bounds(0.0, 1.0),
            constraints=LinearConstraint(
                incidence, -np.inf, np.ones(len(component_endpoints))
            ),
            options={"disp": False},
        )
        if not result.success or result.x is None:
            raise RuntimeError(f"unit-capacity route-packing MILP failed: {result.message}")
        selected_active[component] = np.asarray(result.x > 0.5, dtype=bool)

    selected[active] = selected_active
    loads = np.zeros(len(endpoints), dtype=np.float64)
    for route, chosen in zip(active_rows, selected_active):
        if chosen:
            for endpoint in route:
                loads[endpoint_row[endpoint]] += 1.0
    if np.any(loads > 1.0 + 1.0e-8):  # pragma: no cover - MILP contract
        raise RuntimeError("unit-capacity route-packing solver violated an endpoint constraint")
    return UnitCapacityPackingResult(
        selected=selected,
        objective=float(np.dot(values, selected.astype(np.float64))),
        endpoints=endpoints,
        endpoint_loads=np.rint(loads).astype(np.int64),
    )


def _validate_config(config: RouteAssignmentConfig) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    station_path = tuple(int(station) for station in config.station_path)
    pairs = adjacent_station_pairs(station_path)
    supplied = {tuple(int(value) for value in pair) for pair in config.score_threshold_by_pair}
    expected = set(pairs)
    if supplied != expected:
        raise ValueError(
            "score_threshold_by_pair must define exactly the adjacent route pairs: "
            f"missing={sorted(expected - supplied)}, extra={sorted(supplied - expected)}"
        )
    for pair, threshold in config.score_threshold_by_pair.items():
        value = float(threshold)
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"route score threshold for {pair} must be finite and in [0, 1]")
    if not np.isfinite(config.unmatched_penalty):
        raise ValueError("unmatched_penalty must be finite")
    if config.maximum_hypotheses < 1:
        raise ValueError("maximum_hypotheses must be positive")
    if config.complete_route_score_threshold is not None:
        threshold = float(config.complete_route_score_threshold)
        if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
            raise ValueError("complete route score threshold must be finite and in [0, 1]")
    if config.complete_route_score_composition not in {"replace", "residual"}:
        raise ValueError("complete_route_score_composition must be 'replace' or 'residual'")
    if not np.isfinite(config.complete_route_context_weight) or config.complete_route_context_weight < 0.0:
        raise ValueError("complete_route_context_weight must be finite and non-negative")
    return station_path, pairs


def _matrix_edge_lookup(
    matrix: ScoreMatrix,
    candidates: Sequence[FieldCandidate],
    threshold: float,
) -> dict[tuple[int, int], ScoredMatch]:
    """Materialise only declared physical candidate edges above threshold."""
    result: dict[tuple[int, int], ScoredMatch] = {}
    for left, source_index in enumerate(matrix.source_indices):
        for right, target_index in enumerate(matrix.target_indices):
            score = float(matrix.values[left, right])
            if not np.isfinite(score) or score < threshold:
                continue
            candidate_row = int(matrix.candidate_rows[left, right])
            if candidate_row < 0:
                raise RuntimeError("finite route score matrix entry lacks a physical candidate row")
            candidate = candidates[candidate_row]
            result[(int(source_index), int(target_index))] = ScoredMatch(
                source_index=int(source_index),
                target_index=int(target_index),
                chi2=float(candidate.chi2),
                score=score,
            )
    return result


def _route_hypotheses(
    event: EventTracklets,
    station_path: tuple[int, ...],
    edge_lookups: Mapping[tuple[int, int], Mapping[tuple[int, int], ScoredMatch]],
    unmatched_penalty: float,
    maximum_hypotheses: int,
    complete_route_scores: Mapping[tuple[int, ...], float] | None = None,
    complete_route_score_threshold: float | None = None,
    complete_route_score_composition: str = "replace",
    complete_route_context_weight: float = 1.0,
) -> list[Route]:
    """Enumerate compatible contiguous routes with two or more stations."""
    indices_by_station = {
        station: tuple(int(index) for index in event.indices_for_station(station).tolist())
        for station in station_path
    }
    routes: list[Route] = []
    for start in range(len(station_path) - 1):
        for stop in range(start + 2, len(station_path) + 1):
            route_stations = station_path[start:stop]
            if any(not indices_by_station[station] for station in route_stations):
                continue
            for route_indices in product(*(indices_by_station[station] for station in route_stations)):
                matches: list[tuple[tuple[int, int], ScoredMatch]] = []
                log_odds = 0.0
                for source_station, target_station, source_index, target_index in zip(
                    route_stations,
                    route_stations[1:],
                    route_indices,
                    route_indices[1:],
                ):
                    match = edge_lookups[(source_station, target_station)].get(
                        (source_index, target_index)
                    )
                    if match is None:
                        break
                    probability = float(np.clip(match.score, 1.0e-6, 1.0 - 1.0e-6))
                    log_odds += float(np.log(probability) - np.log1p(-probability))
                    matches.append(((source_station, target_station), match))
                else:
                    complete_route_score: float | None = None
                    if (
                        complete_route_scores is not None
                        and len(route_stations) == len(station_path)
                    ):
                        key = tuple(int(index) for index in route_indices)
                        if key not in complete_route_scores:
                            raise RuntimeError(
                                "complete physical route has no route-query score; "
                                "the V2 score map and adjacent candidate graph disagree"
                            )
                        complete_route_score = float(complete_route_scores[key])
                        if not np.isfinite(complete_route_score) or not 0.0 <= complete_route_score <= 1.0:
                            raise ValueError("complete route-query score must be finite and in [0, 1]")
                        if (
                            complete_route_score_threshold is not None
                            and complete_route_score < complete_route_score_threshold
                        ):
                            continue
                        probability = float(np.clip(complete_route_score, 1.0e-6, 1.0 - 1.0e-6))
                        route_log_odds = float(np.log(probability) - np.log1p(-probability))
                        if complete_route_score_composition == "replace":
                            # A calibrated V2 route score is the probability
                            # of the whole chain, so use its one route log-odds
                            # rather than adding marginal edge log-odds.
                            log_odds = route_log_odds
                        elif complete_route_score_composition == "residual":
                            # Preserve the local-edge route utility and inject
                            # the contextual route query as a residual.  At
                            # weight zero this is exactly the edge-only solver;
                            # at weight one it recovers the direct replacement
                            # formulation.  This avoids an artificial utility
                            # loss when a full chain competes with two partial
                            # chains built from the same high-score edges.
                            log_odds += float(complete_route_context_weight) * (
                                route_log_odds - log_odds
                            )
                        else:  # pragma: no cover - validated by _validate_config
                            raise RuntimeError("unsupported complete route score composition")
                    # Relative to dustbinning all participating endpoints,
                    # each selected adjacent edge improves the objective by
                    # its log-odds while the route avoids one dustbin cost per
                    # endpoint.  This is the contiguous-chain analogue of
                    # the bipartite dustbin assignment objective.
                    utility = log_odds + len(route_stations) * unmatched_penalty
                    if utility > 0.0:
                        routes.append(
                            Route(
                                endpoints=tuple(zip(route_stations, route_indices)),
                                matches=tuple(matches),
                                utility=float(
                                    utility
                                    + _CONTINUATION_TIE_BREAK * (len(route_stations) - 1)
                                ),
                                complete_route_score=complete_route_score,
                            )
                        )
                        if len(routes) > maximum_hypotheses:
                            raise RuntimeError(
                                "route hypothesis count exceeds maximum_hypotheses; "
                                "tighten a frozen score threshold or reduce overlay occupancy"
                            )
    return routes


def _select_disjoint_routes(routes: Sequence[Route]) -> tuple[Route, ...]:
    """Solve the endpoint-capacity set packing exactly for one small event."""
    if not routes:
        return ()
    packing = solve_unit_capacity_route_packing(
        [route.endpoints for route in routes],
        [route.utility for route in routes],
    )
    return tuple(route for route, selected in zip(routes, packing.selected) if bool(selected))


def adjacent_route_assignment(
    event: EventTracklets,
    station_matrices: Mapping[tuple[int, int], tuple[ScoreMatrix, Sequence[FieldCandidate]]],
    config: RouteAssignmentConfig,
    complete_route_scores: Mapping[tuple[int, ...], float] | None = None,
) -> RouteAssignmentResult:
    """Solve truth-free adjacent route assignment with optional dustbins.

    Every matrix must be from the same event.  Missing physical candidates are
    absent edges rather than synthesized coordinates.  Any endpoint not used
    by a selected route is explicitly returned as a dustbin assignment.
    """
    station_path, pairs = _validate_config(config)
    supplied_pairs = {tuple(int(value) for value in pair) for pair in station_matrices}
    expected_pairs = set(pairs)
    if supplied_pairs != expected_pairs:
        raise ValueError(
            "station_matrices must contain exactly the adjacent route pairs: "
            f"missing={sorted(expected_pairs - supplied_pairs)}, extra={sorted(supplied_pairs - expected_pairs)}"
        )
    edge_lookups: dict[tuple[int, int], dict[tuple[int, int], ScoredMatch]] = {}
    for pair in pairs:
        matrix, candidates = station_matrices[pair]
        edge_lookups[pair] = _matrix_edge_lookup(
            matrix,
            candidates,
            float(config.score_threshold_by_pair[pair]),
        )
    routes = _route_hypotheses(
        event,
        station_path,
        edge_lookups,
        config.unmatched_penalty,
        config.maximum_hypotheses,
        complete_route_scores,
        config.complete_route_score_threshold,
        config.complete_route_score_composition,
        config.complete_route_context_weight,
    )
    selected = _select_disjoint_routes(routes)
    used_by_station: dict[int, set[int]] = {station: set() for station in station_path}
    matches_by_pair: dict[tuple[int, int], list[ScoredMatch]] = {pair: [] for pair in pairs}
    for route in selected:
        for station, index in route.endpoints:
            used_by_station[station].add(index)
        for pair, match in route.matches:
            matches_by_pair[pair].append(match)
    unmatched_by_station = {
        station: tuple(
            index
            for index in event.indices_for_station(station).tolist()
            if int(index) not in used_by_station[station]
        )
        for station in station_path
    }
    edge_counts = {pair: len(edge_lookups[pair]) for pair in pairs}
    return RouteAssignmentResult(
        routes=selected,
        matches_by_pair={pair: tuple(values) for pair, values in matches_by_pair.items()},
        unmatched_by_station=unmatched_by_station,
        candidate_edges_above_threshold=int(sum(edge_counts.values())),
        candidate_edges_above_threshold_by_pair=edge_counts,
        hypotheses_considered=len(routes),
        selected_routes=len(selected),
    )
