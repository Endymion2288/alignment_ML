"""Truth-free 2/3/4-route features and assignment for Workbook 79 Arm B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from baselines.field_chi2_matching import FieldCandidate, ScoredMatch
from baselines.global_assignment import ScoreMatrix
from baselines.route_assignment import Route, RouteAssignmentResult, adjacent_station_pairs
from evaluation.route_accounting import RouteAccountingV2, assess_route_accounting_v2
from models.explicit_route_energy import (
    ADJACENT_PAIRS,
    FEATURE_DIM,
    NODE_STATE_NAMES,
    N_NODE_SLOTS,
    STATION_PATH,
    route_feature_names,
)
from models.route_energy import (
    CANONICAL_ENERGY_VERSION,
    RouteEnergyRecord,
    RouteEnergyTable,
    assign_from_energy_table,
    canonical_route_energy,
    route_kind,
)
from training.geometry_aware_transformer import (
    ADJACENT_STATION_PAIRS,
    ALL_STATION_PAIRS,
    EDGE_FEATURE_NAMES,
    TransformerGraph,
)
from training.route_aware_transformer import enumerate_physical_complete_route_chains


_NODE_OFFSET = len(ADJACENT_PAIRS) * len(EDGE_FEATURE_NAMES)

FORBIDDEN_PATH_NEEDLES = (
    "00800_00849",
    "mc24_100116",
    "mc24_100117",
    "00350_00399",
    "source_diversity_blind",
)


def refuse_forbidden_experiment_path(path: object) -> None:
    text = str(path)
    for needle in FORBIDDEN_PATH_NEEDLES:
        if needle in text:
            raise ValueError(f"WB79 refuses development / final-blind / sealed path: {text}")


@dataclass(frozen=True)
class PhysicalRoute:
    """One contiguous physical route already present in the adjacent graph."""

    stations: tuple[int, ...]
    node_indices: tuple[int, ...]
    score_edge_indices: tuple[int, ...]
    truth_consistent: bool


def _adjacent_pair_ids() -> tuple[int, ...]:
    """Score-edge pair IDs are indices into ALL_STATION_PAIRS, not 0/1/2."""
    index = {pair: row for row, pair in enumerate(ALL_STATION_PAIRS)}
    return tuple(index[pair] for pair in ADJACENT_PAIRS)


def _adjacent_edge_maps(graph: TransformerGraph) -> list[dict[int, list[tuple[int, int]]]]:
    pair_ids = _adjacent_pair_ids()
    maps: list[dict[int, list[tuple[int, int]]]] = [dict(), dict(), dict()]
    for row, (source, target, pair_id) in enumerate(
        zip(
            graph.score_edge_source.tolist(),
            graph.score_edge_destination.tolist(),
            graph.score_edge_station_pair.tolist(),
        )
    ):
        try:
            adjacent_index = pair_ids.index(int(pair_id))
        except ValueError as error:
            raise ValueError("score graph includes a non-adjacent output edge") from error
        source_station, target_station = ADJACENT_PAIRS[adjacent_index]
        if int(graph.station_ids[int(source)]) != source_station or int(graph.station_ids[int(target)]) != target_station:
            raise ValueError("adjacent score edge station IDs disagree with its declared pair")
        maps[adjacent_index].setdefault(int(source), []).append((int(target), int(row)))
    return maps


def enumerate_contiguous_physical_routes(graph: TransformerGraph) -> tuple[PhysicalRoute, ...]:
    """Walk existing adjacent edges into all contiguous 2/3/4-station routes.

    Truth IDs never create or delete a candidate.  Labels are attached after
    the physical set is fixed.
    """
    maps = _adjacent_edge_maps(graph)
    truth = None if graph.event.truth_particle_id is None else np.asarray(graph.event.truth_particle_id)
    collected: list[PhysicalRoute] = []

    def _append(nodes: tuple[int, ...], edges: tuple[int, ...]) -> None:
        stations = tuple(int(graph.station_ids[index]) for index in nodes)
        consistent = False
        if truth is not None:
            labels = truth[list(nodes)]
            consistent = bool(np.all(labels >= 0) and np.all(labels == labels[0]))
        collected.append(
            PhysicalRoute(
                stations=stations,
                node_indices=nodes,
                score_edge_indices=edges,
                truth_consistent=consistent,
            )
        )

    for source, rows in maps[0].items():
        for target, edge in rows:
            _append((source, target), (edge,))
    for source, rows in maps[1].items():
        for target, edge in rows:
            _append((source, target), (edge,))
    for source, rows in maps[2].items():
        for target, edge in rows:
            _append((source, target), (edge,))
    for source_zero, second_rows in maps[0].items():
        for source_one, edge_zero_one in second_rows:
            for source_two, edge_one_two in maps[1].get(source_one, ()):
                _append((source_zero, source_one, source_two), (edge_zero_one, edge_one_two))
    for source_one, third_rows in maps[1].items():
        for source_two, edge_one_two in third_rows:
            for source_three, edge_two_three in maps[2].get(source_two, ()):
                _append((source_one, source_two, source_three), (edge_one_two, edge_two_three))
    complete_nodes, complete_edges = enumerate_physical_complete_route_chains(graph)
    for nodes, edges in zip(complete_nodes.tolist(), complete_edges.tolist()):
        _append(tuple(int(value) for value in nodes), tuple(int(value) for value in edges))
    return tuple(collected)


def route_feature_matrix(graph: TransformerGraph, routes: Sequence[PhysicalRoute]) -> np.ndarray:
    """Station-indexed raw physical features.  Missing hops/stations are zero."""
    names = route_feature_names()
    if len(names) != FEATURE_DIM:
        raise RuntimeError("feature name table drifted from FEATURE_DIM")
    matrix = np.zeros((len(routes), FEATURE_DIM), dtype=np.float64)
    pair_slot = {pair: index for index, pair in enumerate(ADJACENT_PAIRS)}
    event = graph.event
    for row, route in enumerate(routes):
        for edge_index in route.score_edge_indices:
            pair_id = int(graph.score_edge_station_pair[int(edge_index)])
            pair = ALL_STATION_PAIRS[pair_id]
            if pair not in pair_slot:
                raise ValueError("score-edge pair is not an adjacent hop")
            slot = pair_slot[pair]
            start = slot * len(EDGE_FEATURE_NAMES)
            values = np.asarray(graph.score_edge_features[int(edge_index)], dtype=np.float64)
            if values.shape != (len(EDGE_FEATURE_NAMES),) or not np.isfinite(values).all():
                raise ValueError("score-edge features are not a finite raw_energy physical vector")
            matrix[row, start : start + len(EDGE_FEATURE_NAMES)] = values
        present = np.ones(len(STATION_PATH), dtype=np.float64)
        for node_index in route.node_indices:
            station = int(graph.station_ids[int(node_index)])
            if station not in STATION_PATH:
                raise ValueError("route node is outside the four-station path")
            present[station] = 0.0
            state = np.asarray(event.state[int(node_index)], dtype=np.float64)
            if state.shape != (len(NODE_STATE_NAMES),) or not np.isfinite(state).all():
                raise ValueError("tracklet state is not a finite x/y/tx/ty vector")
            offset = _NODE_OFFSET + station * len(NODE_STATE_NAMES)
            matrix[row, offset : offset + len(NODE_STATE_NAMES)] = state
        matrix[row, _NODE_OFFSET + N_NODE_SLOTS * len(NODE_STATE_NAMES)] = float(len(route.stations))
        matrix[row, FEATURE_DIM - N_NODE_SLOTS :] = present
        if not np.isfinite(matrix[row]).all():
            raise ValueError("route feature construction produced a non-finite row")
    return matrix


def target_mask(routes: Sequence[PhysicalRoute]) -> np.ndarray:
    """Complete truth-consistent four-station routes only."""
    return np.asarray(
        [bool(route.truth_consistent and len(route.stations) == 4) for route in routes],
        dtype=bool,
    )


def energy_table_from_utilities(
    routes: Sequence[PhysicalRoute],
    utilities: Sequence[float],
    *,
    unmatched_penalty: float,
) -> RouteEnergyTable:
    if len(routes) != len(utilities):
        raise ValueError("utilities must align with routes")
    records = []
    for route_id, (route, energy) in enumerate(zip(routes, utilities)):
        records.append(
            RouteEnergyRecord(
                route_id=route_id,
                endpoints=tuple((int(station), int(index)) for station, index in zip(route.stations, route.node_indices)),
                energy=float(energy),
                n_stations=len(route.stations),
                kind=route_kind(len(route.stations)),
                contract=CANONICAL_ENERGY_VERSION,
                correction=0.0,
            )
        )
    return RouteEnergyTable(
        version=CANONICAL_ENERGY_VERSION,
        unmatched_penalty=float(unmatched_penalty),
        records=tuple(records),
    )


def w64_edge_logit_matrix(
    graph: TransformerGraph,
    routes: Sequence[PhysicalRoute],
    base_logits_by_owner_row: Mapping[tuple[int, int], float],
) -> np.ndarray:
    """Per-hop raw W64 logits aligned to ``ADJACENT_PAIRS``.  Missing hops are 0."""
    values = np.zeros((len(routes), len(ADJACENT_PAIRS)), dtype=np.float64)
    pair_slot = {pair: index for index, pair in enumerate(ADJACENT_PAIRS)}
    for row, route in enumerate(routes):
        for edge_index in route.score_edge_indices:
            pair = ALL_STATION_PAIRS[int(graph.score_edge_station_pair[int(edge_index)])]
            if pair not in pair_slot:
                raise ValueError("score-edge pair is not an adjacent hop")
            owner = int(graph.score_owner[int(edge_index)])
            score_row = int(graph.score_row[int(edge_index)])
            values[row, pair_slot[pair]] = float(base_logits_by_owner_row[(owner, score_row)])
    return values


def w64_route_utilities(
    graph: TransformerGraph,
    routes: Sequence[PhysicalRoute],
    base_logits_by_owner_row: Mapping[tuple[int, int], float],
    unmatched_penalty: float,
) -> np.ndarray:
    """Arm A: ``sum(raw W64 adjacent logits) + n_stations * penalty``.  No clip."""
    values = []
    for route in routes:
        edges = []
        for edge_index in route.score_edge_indices:
            owner = int(graph.score_owner[int(edge_index)])
            row = int(graph.score_row[int(edge_index)])
            logit = base_logits_by_owner_row[(owner, row)]
            edges.append(float(logit))
        values.append(
            canonical_route_energy(edges, unmatched_penalty=unmatched_penalty, n_stations=len(route.stations))
        )
    return np.asarray(values, dtype=np.float64)


def _score_matrix(candidates: Sequence[FieldCandidate], scores: Sequence[float]) -> ScoreMatrix:
    sources = tuple(sorted({int(item.source_index) for item in candidates}))
    targets = tuple(sorted({int(item.target_index) for item in candidates}))
    values = np.full((len(sources), len(targets)), np.nan, dtype=np.float64)
    source_pos = {index: row for row, index in enumerate(sources)}
    target_pos = {index: column for column, index in enumerate(targets)}
    for candidate, score in zip(candidates, scores):
        values[source_pos[int(candidate.source_index)], target_pos[int(candidate.target_index)]] = float(score)
    return ScoreMatrix(
        source_indices=sources,
        target_indices=targets,
        values=values,
        candidate_rows=np.arange(len(candidates), dtype=np.int64),
    )


def station_matrices_from_graph(graph: TransformerGraph, adjacent_sets: Sequence[object]) -> dict:
    by_pair: dict[tuple[int, int], object] = {}
    for candidate_set in adjacent_sets:
        if candidate_set.event is graph.event and candidate_set.sample is graph.sample:
            by_pair[tuple(int(value) for value in candidate_set.station_pair)] = candidate_set
    matrices = {}
    for pair in adjacent_station_pairs(STATION_PATH):
        if pair not in by_pair:
            raise ValueError(f"graph is missing adjacent candidate set {pair}")
        candidates = tuple(by_pair[pair].candidates)
        matrices[pair] = (_score_matrix(candidates, np.ones(len(candidates), dtype=np.float64)), candidates)
    return matrices


def assignment_from_table(
    graph: TransformerGraph,
    routes: Sequence[PhysicalRoute],
    table: RouteEnergyTable,
    adjacent_sets: Sequence[object],
) -> RouteAssignmentResult:
    packing = assign_from_energy_table(table, allow_legacy=False)
    selected_routes: list[Route] = []
    matches_by_pair: dict[tuple[int, int], list[ScoredMatch]] = {pair: [] for pair in adjacent_station_pairs(STATION_PATH)}
    matrices = station_matrices_from_graph(graph, adjacent_sets)
    candidate_lookup: dict[tuple[int, int, int, int], FieldCandidate] = {}
    for pair, (_, candidates) in matrices.items():
        for candidate in candidates:
            candidate_lookup[(pair[0], pair[1], int(candidate.source_index), int(candidate.target_index))] = candidate
    for route, selected, record in zip(routes, packing.selected.tolist(), table.records):
        if not bool(selected):
            continue
        matches = []
        for hop, (source_station, target_station) in enumerate(zip(route.stations, route.stations[1:])):
            source_index = int(route.node_indices[hop])
            target_index = int(route.node_indices[hop + 1])
            candidate = candidate_lookup[(int(source_station), int(target_station), source_index, target_index)]
            match = ScoredMatch(
                source_index=source_index,
                target_index=target_index,
                chi2=float(candidate.chi2),
                score=1.0,
            )
            matches.append(((int(source_station), int(target_station)), match))
            matches_by_pair[(int(source_station), int(target_station))].append(match)
        selected_routes.append(
            Route(
                endpoints=record.endpoints,
                matches=tuple(matches),
                utility=float(record.energy),
            )
        )
    used = {int(index) for route in selected_routes for _, index in route.endpoints}
    unmatched = {
        station: tuple(
            int(index)
            for index in graph.event.indices_for_station(station).tolist()
            if int(index) not in used
        )
        for station in STATION_PATH
    }
    n_candidates = sum(len(candidates) for _, candidates in matrices.values())
    return RouteAssignmentResult(
        routes=tuple(selected_routes),
        matches_by_pair={pair: tuple(matches) for pair, matches in matches_by_pair.items()},
        unmatched_by_station=unmatched,
        candidate_edges_above_threshold=n_candidates,
        candidate_edges_above_threshold_by_pair={
            pair: len(candidates) for pair, (_, candidates) in matrices.items()
        },
        hypotheses_considered=len(routes),
        selected_routes=len(selected_routes),
    )


def account_event(
    graph: TransformerGraph,
    routes: Sequence[PhysicalRoute],
    table: RouteEnergyTable,
    adjacent_sets: Sequence[object],
) -> RouteAccountingV2:
    result = assignment_from_table(graph, routes, table, adjacent_sets)
    matrices = station_matrices_from_graph(graph, adjacent_sets)
    thresholds = {pair: 0.0 for pair in adjacent_station_pairs(STATION_PATH)}
    return assess_route_accounting_v2(graph.event, result, matrices, STATION_PATH, thresholds)
