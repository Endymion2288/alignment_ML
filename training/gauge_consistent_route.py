"""Gauge-twin consistency and packing-utility route-competition auxiliaries.

These terms sit on frozen-architecture V2 outputs.  They do not change the
candidate builder, the unit-capacity solver, or the historical edge-BCE /
route-query body.  Packing utilities use the same clipped log-odds convention
as ``baselines.route_assignment._route_hypotheses``.
"""

from __future__ import annotations

import copy
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

from models.route_transformer import RouteAwareSparseTransformer, RouteAwareTransformerConfig
from training.geometry_aware_transformer import (
    ADJACENT_STATION_PAIRS,
    CurriculumStage,
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    TransformerGraph,
    TransformerGraphBundle,
    fit_graph_standardizers,
    graph_bundle_summary,
    resolve_device,
)
from training.route_aware_transformer import (
    RouteCandidateTable,
    RouteAwareTrainingConfig,
    RouteAwareTransformerArtifact,
    _adjacent_pair_ids,
    _edge_positive_weight,
    _forward_route_batch,
    _make_route_batch,
    _route_loss_components,
    _seed_everything,
    _stage_graphs,
    _validate_stages,
    _validate_training_config,
    enumerate_complete_route_candidates,
    materialize_route_candidate_tables,
)
from training.route_operating_audit import (
    DUSTBIN_UTILITY,
    SOLVER_PROBABILITY_FLOOR,
    origin_key,
)


PLUS_COMMON_SUFFIX = "_plus_common"
LOGIT_CLIP = math.log((1.0 - SOLVER_PROBABILITY_FLOOR) / SOLVER_PROBABILITY_FLOOR)
IDENTITY_PLATT = {
    "0->1": {"slope": 1.0, "intercept": 0.0},
    "1->2": {"slope": 1.0, "intercept": 0.0},
    "2->3": {"slope": 1.0, "intercept": 0.0},
    "0->2": {"slope": 1.0, "intercept": 0.0},
    "0->3": {"slope": 1.0, "intercept": 0.0},
    "1->3": {"slope": 1.0, "intercept": 0.0},
}


@dataclass(frozen=True)
class GaugeConsistentAuxConfig:
    """Pre-registered auxiliary packing / gauge controls."""

    packing_margin: float = 1.0
    pair_threshold: float = 0.001
    unmatched_penalty: float = -1.0
    relative_target_vs_edge: float = 1.0
    epsilon: float = 1.0e-6
    weight_clip: tuple[float, float] = (0.05, 20.0)
    enable_dustbin_aware_route_margin: bool = False
    route_competition_reduction: str = "mean"
    copy_frozen_aux_weights: Mapping[str, float] | None = None
    zero_mean_threshold: float = 1.0e-6
    zero_mean_fallback_weight: float = 1.0


def payload_gauge_role(payload_id: str) -> tuple[str, str] | None:
    """Return ``(family, role)`` for a chart/twin payload, else ``None``."""
    name = str(payload_id)
    if name.endswith(PLUS_COMMON_SUFFIX):
        return name[: -len(PLUS_COMMON_SUFFIX)], "left_se3_control"
    if name.endswith("_reference"):
        return None
    return name, "s0_sampling_chart"


def clipped_packing_logits(logits: torch.Tensor) -> torch.Tensor:
    """Match solver log-odds of ``clip(sigmoid(z), 1e-6, 1-1e-6)``."""
    return logits.clamp(-LOGIT_CLIP, LOGIT_CLIP)


def packing_utility(edge_logits: torch.Tensor, unmatched_penalty: float, n_stations: int) -> torch.Tensor:
    """``sum(clipped logits) + n_stations * unmatched_penalty`` versus dustbin 0."""
    if int(edge_logits.numel()) != int(n_stations) - 1:
        raise ValueError("packing utility expects one logit per adjacent hop")
    return clipped_packing_logits(edge_logits).sum() + float(n_stations) * float(unmatched_penalty)


def identity_calibration_payload() -> dict[str, object]:
    """Frozen identity Platt: inference scores equal ``sigmoid(raw logits)``."""
    reports = {
        pair: {
            "slope": 1.0,
            "intercept": 0.0,
            "fit_split": "identity_frozen_pre_training",
            "identity_map": True,
        }
        for pair in IDENTITY_PLATT
    }
    return {
        "scope": "station_pair",
        "method": "platt",
        "fit_split": "identity_frozen_pre_training",
        "identity_map": True,
        "by_station_pair": reports,
        "platt_by_station_pair": dict(IDENTITY_PLATT),
    }


def origin_edge_key(
    event, source_index: int, target_index: int, pair: tuple[int, int]
) -> tuple[object, ...] | None:
    source = origin_key(event, int(source_index))
    target = origin_key(event, int(target_index))
    if source is None or target is None:
        return None
    return (int(pair[0]), int(pair[1]), source, target)


def origin_route_key(event, nodes: Sequence[int]) -> tuple[object, ...] | None:
    keys = [origin_key(event, int(index)) for index in nodes]
    if any(item is None for item in keys):
        return None
    return tuple(keys)


def pair_gauge_twin_graphs(
    graphs: Sequence[TransformerGraph],
) -> tuple[list[tuple[TransformerGraph, TransformerGraph]], list[TransformerGraph]]:
    """Pair chart/twin graphs that share event IDs and relative family."""
    buckets: dict[tuple[int, int, str], dict[str, TransformerGraph]] = {}
    leftovers: list[TransformerGraph] = []
    for graph in graphs:
        parsed = payload_gauge_role(graph.sample.payload_id)
        if parsed is None:
            leftovers.append(graph)
            continue
        family, role = parsed
        key = (int(graph.event.run_id), int(graph.event.event_id), family)
        bucket = buckets.setdefault(key, {})
        if role in bucket:
            raise ValueError(f"duplicate gauge role '{role}' for {key}")
        bucket[role] = graph
    pairs: list[tuple[TransformerGraph, TransformerGraph]] = []
    for bucket in buckets.values():
        chart = bucket.get("s0_sampling_chart")
        twin = bucket.get("left_se3_control")
        if chart is not None and twin is not None:
            pairs.append((chart, twin))
            continue
        leftovers.extend(bucket.values())
    return pairs, leftovers


def split_graph_edge_logits(
    graphs: Sequence[TransformerGraph], edge_logits: torch.Tensor
) -> dict[int, torch.Tensor]:
    offset = 0
    result: dict[int, torch.Tensor] = {}
    for graph in graphs:
        width = int(graph.score_labels.size)
        result[id(graph)] = edge_logits[offset : offset + width]
        offset += width
    if offset != int(edge_logits.numel()):
        raise RuntimeError("batched edge logits are not aligned with graph score tables")
    return result


def split_graph_route_logits(
    graphs: Sequence[TransformerGraph],
    route_logits: torch.Tensor,
    route_tables: Mapping[int, RouteCandidateTable] | None = None,
) -> dict[int, torch.Tensor]:
    offset = 0
    result: dict[int, torch.Tensor] = {}
    for graph in graphs:
        table = (
            enumerate_complete_route_candidates(graph)
            if route_tables is None
            else route_tables[id(graph)]
        )
        width = int(table.size)
        result[id(graph)] = route_logits[offset : offset + width]
        offset += width
    if offset != int(route_logits.numel()):
        raise RuntimeError("batched route logits are not aligned with graph route tables")
    return result


def _adjacent_edge_maps(graph: TransformerGraph) -> list[dict[int, list[tuple[int, int]]]]:
    pair_ids = _adjacent_pair_ids()
    maps: list[dict[int, list[tuple[int, int]]]] = [defaultdict(list), defaultdict(list), defaultdict(list)]
    for row, (source, target, pair_id) in enumerate(
        zip(
            graph.score_edge_source.tolist(),
            graph.score_edge_destination.tolist(),
            graph.score_edge_station_pair.tolist(),
        )
    ):
        adjacent_index = pair_ids.index(int(pair_id))
        source_station, target_station = ADJACENT_STATION_PAIRS[adjacent_index]
        if int(graph.station_ids[int(source)]) != source_station or int(graph.station_ids[int(target)]) != target_station:
            raise ValueError("adjacent score edge station IDs disagree with its declared pair")
        maps[adjacent_index][int(source)].append((int(target), int(row)))
    return maps


def enumerate_contiguous_packing_routes(
    graph: TransformerGraph,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """All 2/3/4-station contiguous routes already present as physical edges."""
    maps = _adjacent_edge_maps(graph)
    routes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for mapping in maps:
        for source, items in mapping.items():
            for target, edge in items:
                routes.append(((int(source), int(target)), (int(edge),)))
    for source_zero, first in maps[0].items():
        for source_one, edge_zero_one in first:
            for source_two, edge_one_two in maps[1].get(source_one, ()):
                routes.append(
                    (
                        (int(source_zero), int(source_one), int(source_two)),
                        (int(edge_zero_one), int(edge_one_two)),
                    )
                )
                for source_three, edge_two_three in maps[2].get(source_two, ()):
                    routes.append(
                        (
                            (int(source_zero), int(source_one), int(source_two), int(source_three)),
                            (int(edge_zero_one), int(edge_one_two), int(edge_two_three)),
                        )
                    )
    for source_one, second in maps[1].items():
        for source_two, edge_one_two in second:
            for source_three, edge_two_three in maps[2].get(source_two, ()):
                routes.append(
                    (
                        (int(source_one), int(source_two), int(source_three)),
                        (int(edge_one_two), int(edge_two_three)),
                    )
                )
    return routes


def _origin_edge_logit_table(
    graph: TransformerGraph, logits: torch.Tensor, *, truth_only: bool = False
) -> dict[tuple[object, ...], torch.Tensor]:
    if logits.shape != (graph.score_labels.size,):
        raise ValueError("edge logits are not aligned with the graph score table")
    pair_ids = _adjacent_pair_ids()
    table: dict[tuple[object, ...], torch.Tensor] = {}
    for row, (source, target, pair_id) in enumerate(
        zip(graph.score_edge_source, graph.score_edge_destination, graph.score_edge_station_pair)
    ):
        if truth_only and not bool(graph.score_labels[int(row)]):
            continue
        pair = ADJACENT_STATION_PAIRS[pair_ids.index(int(pair_id))]
        key = origin_edge_key(graph.event, int(source), int(target), pair)
        if key is None:
            continue
        if key in table:
            raise ValueError("duplicate origin-aligned adjacent edge")
        table[key] = logits[int(row)]
    return table


def gauge_twin_consistency_loss(
    chart: TransformerGraph,
    twin: TransformerGraph,
    chart_logits: torch.Tensor,
    twin_logits: torch.Tensor,
    chart_route_logits: torch.Tensor | None = None,
    twin_route_logits: torch.Tensor | None = None,
    chart_route_table: RouteCandidateTable | None = None,
    twin_route_table: RouteCandidateTable | None = None,
) -> torch.Tensor:
    """MSE on origin-matched raw logits and complete-route packing utilities.

    Unpaired candidates without a shared origin signature are ignored.
    """
    chart_edges = _origin_edge_logit_table(chart, chart_logits)
    twin_edges = _origin_edge_logit_table(twin, twin_logits)
    shared = sorted(set(chart_edges) & set(twin_edges), key=str)
    terms: list[torch.Tensor] = []
    if shared:
        delta = torch.stack([chart_edges[key] - twin_edges[key] for key in shared])
        terms.append(torch.mean(delta.square()))
    chart_table = (
        enumerate_complete_route_candidates(chart)
        if chart_route_table is None
        else chart_route_table
    )
    twin_table = (
        enumerate_complete_route_candidates(twin)
        if twin_route_table is None
        else twin_route_table
    )
    chart_routes: dict[tuple[object, ...], torch.Tensor] = {}
    for i, (nodes, edges, label) in enumerate(
        zip(chart_table.node_indices, chart_table.score_edge_indices, chart_table.labels)
    ):
        if not bool(label):
            continue
        key = origin_route_key(chart.event, nodes)
        if key is None:
            continue
        if chart_route_logits is not None:
            chart_routes[key] = clipped_packing_logits(chart_route_logits[i]) + 4.0 * (-1.0)
        else:
            index = torch.as_tensor(list(edges), device=chart_logits.device, dtype=torch.long)
            chart_routes[key] = packing_utility(chart_logits[index], -1.0, 4)
    twin_routes: dict[tuple[object, ...], torch.Tensor] = {}
    for i, (nodes, edges, label) in enumerate(
        zip(twin_table.node_indices, twin_table.score_edge_indices, twin_table.labels)
    ):
        if not bool(label):
            continue
        key = origin_route_key(twin.event, nodes)
        if key is None:
            continue
        if twin_route_logits is not None:
            twin_routes[key] = clipped_packing_logits(twin_route_logits[i]) + 4.0 * (-1.0)
        else:
            index = torch.as_tensor(list(edges), device=twin_logits.device, dtype=torch.long)
            twin_routes[key] = packing_utility(twin_logits[index], -1.0, 4)
    shared_routes = sorted(set(chart_routes) & set(twin_routes), key=str)
    if shared_routes:
        delta = torch.stack([chart_routes[key] - twin_routes[key] for key in shared_routes])
        terms.append(torch.mean(delta.square()))
    if not terms:
        return chart_logits.sum() * 0.0
    return torch.stack(terms).mean()


def _truth_complete_routes(graph: TransformerGraph) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    table = enumerate_complete_route_candidates(graph)
    result = []
    for nodes, edges, label in zip(table.node_indices, table.score_edge_indices, table.labels):
        if not bool(label):
            continue
        result.append((tuple(int(value) for value in nodes), tuple(int(value) for value in edges)))
    return result


def packing_route_competition_loss(
    graph: TransformerGraph,
    edge_logits: torch.Tensor,
    *,
    threshold: float,
    unmatched_penalty: float,
    margin: float,
    include_dustbin: bool = False,
    reduction: str = "mean",
    route_logits: torch.Tensor | None = None,
    route_table: RouteCandidateTable | None = None,
) -> torch.Tensor:
    """Local margin of the complete truth route over the strongest feasible rival.

    The rival set is the production contiguous-route enumerator: 2-station
    fragments, 3-station prefix/suffix, mixed four-station routes, and any
    other solver-feasible hypothesis that shares an endpoint.  The solver
    itself is not unrolled.

    Workbook-56 (``include_dustbin=False``) uses ``max(feasible rivals)`` and
    substitutes dustbin only when that set is empty.  Workbook-59
    dustbin-aware margin uses ``max(feasible rivals, 0)``.  Workbook-61
    may replace the per-event ``mean`` over truth routes with ``max``;
    the per-route competitor and the production utility are unchanged.
    """
    if reduction not in {"mean", "max"}:
        raise ValueError("route competition reduction must be 'mean' or 'max'")
    if edge_logits.shape != (graph.score_labels.size,):
        raise ValueError("edge logits are not aligned with the graph score table")
    truths = _truth_complete_routes(graph)
    if not truths:
        return edge_logits.sum() * 0.0
    routes = enumerate_contiguous_packing_routes(graph)
    probabilities = torch.sigmoid(edge_logits)
    dustbin = edge_logits.new_tensor(float(DUSTBIN_UTILITY))
    masked_out = edge_logits.new_tensor(float("-inf"))

    route_map: dict[tuple[int, ...], torch.Tensor] = {}
    if route_logits is not None:
        table = (
            enumerate_complete_route_candidates(graph)
            if route_table is None
            else route_table
        )
        if table.size:
            if route_logits.shape != (table.size,):
                raise ValueError("route logits are not aligned with the graph route table")
            for i, nodes in enumerate(table.node_indices):
                route_map[tuple(int(v) for v in nodes)] = route_logits[i]

    terms: list[torch.Tensor] = []
    for truth_nodes, truth_edges in truths:
        truth_set = set(truth_nodes)
        if tuple(truth_nodes) in route_map:
            truth_utility = (
                clipped_packing_logits(route_map[tuple(truth_nodes)])
                + 4.0 * float(unmatched_penalty)
            )
        else:
            truth_index = torch.as_tensor(list(truth_edges), device=edge_logits.device, dtype=torch.long)
            truth_utility = packing_utility(edge_logits.index_select(0, truth_index), unmatched_penalty, 4)
        competitor_utilities: list[torch.Tensor] = []
        for nodes, edges in routes:
            if nodes == truth_nodes:
                continue
            if not truth_set.intersection(nodes):
                continue
            index = torch.as_tensor(list(edges), device=edge_logits.device, dtype=torch.long)
            feasible = torch.all(probabilities.index_select(0, index) >= float(threshold))
            if len(nodes) == 4 and tuple(nodes) in route_map:
                utility = (
                    clipped_packing_logits(route_map[tuple(nodes)])
                    + 4.0 * float(unmatched_penalty)
                )
            else:
                selected = edge_logits.index_select(0, index)
                utility = packing_utility(selected, unmatched_penalty, len(nodes))
            competitor_utilities.append(torch.where(feasible, utility, masked_out))
        if competitor_utilities:
            strongest = torch.stack(competitor_utilities).max()
            strongest = torch.where(torch.isfinite(strongest), strongest, dustbin)
        else:
            strongest = dustbin
        if include_dustbin:
            strongest = torch.maximum(strongest, dustbin)
        terms.append(torch.relu(strongest + float(margin) - truth_utility))
    stacked = torch.stack(terms)
    if reduction == "max":
        return stacked.max()
    return stacked.mean()


def dustbin_aware_route_margin_loss(
    graph: TransformerGraph,
    edge_logits: torch.Tensor,
    *,
    threshold: float,
    unmatched_penalty: float,
    margin: float,
    reduction: str = "mean",
    route_logits: torch.Tensor | None = None,
    route_table: RouteCandidateTable | None = None,
) -> torch.Tensor:
    """Production-boundary margin: ``U_truth > max(U_fragment, 0) + m``."""
    return packing_route_competition_loss(
        graph,
        edge_logits,
        threshold=threshold,
        unmatched_penalty=unmatched_penalty,
        margin=margin,
        include_dustbin=True,
        reduction=reduction,
        route_logits=route_logits,
        route_table=route_table,
    )


def origin_matched_score_shifts(
    chart: TransformerGraph,
    twin: TransformerGraph,
    chart_logits: np.ndarray,
    twin_logits: np.ndarray,
    unmatched_penalty: float = -1.0,
) -> dict[str, object]:
    """Same-origin truth-edge logit and complete-route utility shifts."""
    chart_values = np.asarray(chart_logits, dtype=np.float64)
    twin_values = np.asarray(twin_logits, dtype=np.float64)
    if chart_values.shape != (chart.score_labels.size,) or twin_values.shape != (twin.score_labels.size,):
        raise ValueError("score-shift arrays are not aligned with graph score tables")
    chart_edges = _origin_edge_logit_table(chart, torch.as_tensor(chart_values), truth_only=True)
    twin_edges = _origin_edge_logit_table(twin, torch.as_tensor(twin_values), truth_only=True)
    shared = sorted(set(chart_edges) & set(twin_edges), key=str)
    edge_abs = [abs(float(chart_edges[key]) - float(twin_edges[key])) for key in shared]
    chart_table = enumerate_complete_route_candidates(chart)
    twin_table = enumerate_complete_route_candidates(twin)
    chart_routes = {}
    for nodes, edges, label in zip(chart_table.node_indices, chart_table.score_edge_indices, chart_table.labels):
        if not bool(label):
            continue
        key = origin_route_key(chart.event, nodes)
        if key is None:
            continue
        chart_routes[key] = float(packing_utility(torch.as_tensor(chart_values[list(edges)]), unmatched_penalty, 4))
    twin_routes = {}
    for nodes, edges, label in zip(twin_table.node_indices, twin_table.score_edge_indices, twin_table.labels):
        if not bool(label):
            continue
        key = origin_route_key(twin.event, nodes)
        if key is None:
            continue
        twin_routes[key] = float(packing_utility(torch.as_tensor(twin_values[list(edges)]), unmatched_penalty, 4))
    shared_routes = sorted(set(chart_routes) & set(twin_routes), key=str)
    utility_abs = [abs(chart_routes[key] - twin_routes[key]) for key in shared_routes]

    def _summary(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "median": None, "p90": None, "mean": None}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(array.size),
            "median": float(np.median(array)),
            "p90": float(np.quantile(array, 0.90)),
            "mean": float(np.mean(array)),
        }

    return {
        "matched_truth_or_origin_edges": _summary(edge_abs),
        "matched_complete_truth_routes": _summary(utility_abs),
        "edge_abs_shifts": edge_abs,
        "utility_abs_shifts": utility_abs,
    }


def clip_aux_weight(value: float, bounds: tuple[float, float]) -> float:
    low, high = bounds
    return min(max(float(value), float(low)), float(high))


def scale_aux_weight(edge_mean: float, aux_mean: float, config: GaugeConsistentAuxConfig) -> float:
    """Train-only scale match.  Zero-mean terms use the fallback, not clip max."""
    target = float(config.relative_target_vs_edge) * max(float(edge_mean), float(config.epsilon))
    if float(aux_mean) <= float(config.zero_mean_threshold):
        return float(config.zero_mean_fallback_weight)
    return clip_aux_weight(target / float(aux_mean), config.weight_clip)


def normalize_aux_weights(
    edge_mean: float,
    gauge_mean: float,
    competition_mean: float,
    config: GaugeConsistentAuxConfig,
) -> dict[str, float]:
    """Workbook-56 scale match.  Zero-mean terms still hit the clip ceiling."""
    target = float(config.relative_target_vs_edge) * max(float(edge_mean), float(config.epsilon))
    gauge = clip_aux_weight(target / max(float(gauge_mean), float(config.epsilon)), config.weight_clip)
    competition = clip_aux_weight(
        target / max(float(competition_mean), float(config.epsilon)), config.weight_clip
    )
    return {
        "gauge_twin_consistency_weight": gauge,
        "packing_route_competition_weight": competition,
        "edge_mean": float(edge_mean),
        "gauge_twin_mean": float(gauge_mean),
        "packing_route_competition_mean": float(competition_mean),
    }


def normalize_dustbin_aware_aux_weights(
    edge_mean: float,
    gauge_mean: float,
    competition_mean: float,
    dustbin_aware_mean: float,
    config: GaugeConsistentAuxConfig,
) -> dict[str, float]:
    """Workbook-59 weights: zero-mean gauge does not inflate to the clip ceiling."""
    return {
        "gauge_twin_consistency_weight": scale_aux_weight(edge_mean, gauge_mean, config),
        "packing_route_competition_weight": scale_aux_weight(edge_mean, competition_mean, config),
        "dustbin_aware_route_margin_weight": scale_aux_weight(edge_mean, dustbin_aware_mean, config),
        "edge_mean": float(edge_mean),
        "gauge_twin_mean": float(gauge_mean),
        "packing_route_competition_mean": float(competition_mean),
        "dustbin_aware_route_margin_mean": float(dustbin_aware_mean),
        "zero_mean_threshold": float(config.zero_mean_threshold),
        "zero_mean_fallback_weight": float(config.zero_mean_fallback_weight),
    }


def _iter_training_batches(
    pairs: Sequence[tuple[TransformerGraph, TransformerGraph]],
    leftovers: Sequence[TransformerGraph],
    batch_size: int,
    rng: np.random.Generator,
) -> Iterable[list[TransformerGraph]]:
    units: list[tuple[TransformerGraph, ...]] = [pair for pair in pairs] + [(graph,) for graph in leftovers]
    if not units:
        raise ValueError("training stage has no graphs")
    order = rng.permutation(len(units))
    current: list[TransformerGraph] = []
    for index in order:
        graphs = list(units[int(index)])
        if current and len(current) + len(graphs) > int(batch_size):
            yield current
            current = []
        current.extend(graphs)
    if current:
        yield current


def _aux_losses_for_batch(
    graphs: Sequence[TransformerGraph],
    edge_logits: torch.Tensor,
    pairs_in_stage: Sequence[tuple[TransformerGraph, TransformerGraph]],
    aux: GaugeConsistentAuxConfig,
    route_logits: torch.Tensor | None = None,
    route_tables: Mapping[int, RouteCandidateTable] | None = None,
) -> dict[str, torch.Tensor]:
    by_id = split_graph_edge_logits(graphs, edge_logits)
    by_route_id = (
        split_graph_route_logits(graphs, route_logits, route_tables)
        if route_logits is not None
        else {}
    )
    present = {id(graph) for graph in graphs}
    gauge_terms = []
    for chart, twin in pairs_in_stage:
        if id(chart) in present and id(twin) in present:
            chart_r_logits = by_route_id.get(id(chart))
            twin_r_logits = by_route_id.get(id(twin))
            chart_r_table = route_tables.get(id(chart)) if route_tables is not None else None
            twin_r_table = route_tables.get(id(twin)) if route_tables is not None else None
            gauge_terms.append(
                gauge_twin_consistency_loss(
                    chart,
                    twin,
                    by_id[id(chart)],
                    by_id[id(twin)],
                    chart_route_logits=chart_r_logits,
                    twin_route_logits=twin_r_logits,
                    chart_route_table=chart_r_table,
                    twin_route_table=twin_r_table,
                )
            )
    if gauge_terms:
        gauge = torch.stack(gauge_terms).mean()
    else:
        gauge = edge_logits.sum() * 0.0
    competition_terms = [
        packing_route_competition_loss(
            graph,
            by_id[id(graph)],
            threshold=aux.pair_threshold,
            unmatched_penalty=aux.unmatched_penalty,
            margin=aux.packing_margin,
            reduction=aux.route_competition_reduction,
            route_logits=by_route_id.get(id(graph)),
            route_table=route_tables.get(id(graph)) if route_tables is not None else None,
        )
        for graph in graphs
    ]
    competition = torch.stack(competition_terms).mean() if competition_terms else edge_logits.sum() * 0.0
    losses = {
        "gauge_twin": gauge,
        "packing_route_competition": competition,
        "dustbin_aware_route_margin": edge_logits.sum() * 0.0,
    }
    if aux.enable_dustbin_aware_route_margin:
        dustbin_terms = [
            dustbin_aware_route_margin_loss(
                graph,
                by_id[id(graph)],
                threshold=aux.pair_threshold,
                unmatched_penalty=aux.unmatched_penalty,
                margin=aux.packing_margin,
                reduction=aux.route_competition_reduction,
                route_logits=by_route_id.get(id(graph)),
                route_table=route_tables.get(id(graph)) if route_tables is not None else None,
            )
            for graph in graphs
        ]
        losses["dustbin_aware_route_margin"] = (
            torch.stack(dustbin_terms).mean() if dustbin_terms else edge_logits.sum() * 0.0
        )
    return losses


def estimate_aux_loss_scales(
    model: RouteAwareSparseTransformer,
    graphs: Sequence[TransformerGraph],
    pairs: Sequence[tuple[TransformerGraph, TransformerGraph]],
    leftovers: Sequence[TransformerGraph],
    node_standardizer,
    edge_standardizer,
    route_tables,
    training_config: RouteAwareTrainingConfig,
    aux: GaugeConsistentAuxConfig,
    device: torch.device,
    edge_positive_weight: float,
) -> dict[str, float]:
    """One frozen train-only pass.  Weights are written before optimizer steps."""
    model.eval()
    totals = {"edge": 0.0, "gauge_twin": 0.0, "packing_route_competition": 0.0, "dustbin_aware_route_margin": 0.0}
    batches = 0
    rng = np.random.default_rng(training_config.seed)
    with torch.no_grad():
        for batch_graphs in _iter_training_batches(pairs, leftovers, training_config.batch_size, rng):
            batch = _make_route_batch(
                batch_graphs, node_standardizer, edge_standardizer, device, route_tables
            )
            output = _forward_route_batch(model, batch)
            body = _route_loss_components(output, batch, training_config, edge_positive_weight)
            aux_losses = _aux_losses_for_batch(
                batch_graphs, output.edge_logits, pairs, aux,
                route_logits=output.route_logits, route_tables=route_tables
            )
            totals["edge"] += float(body["edge"].detach().cpu())
            totals["gauge_twin"] += float(aux_losses["gauge_twin"].detach().cpu())
            totals["packing_route_competition"] += float(
                aux_losses["packing_route_competition"].detach().cpu()
            )
            if aux.enable_dustbin_aware_route_margin:
                totals["dustbin_aware_route_margin"] += float(
                    aux_losses["dustbin_aware_route_margin"].detach().cpu()
                )
            batches += 1
    if batches < 1:
        raise RuntimeError("aux scale estimation saw no train batches")
    means = {key: value / batches for key, value in totals.items()}
    if aux.enable_dustbin_aware_route_margin:
        return normalize_dustbin_aware_aux_weights(
            means["edge"],
            means["gauge_twin"],
            means["packing_route_competition"],
            means["dustbin_aware_route_margin"],
            aux,
        )
    return normalize_aux_weights(
        means["edge"], means["gauge_twin"], means["packing_route_competition"], aux
    )


def _aux_weight_contract_payload(
    weight_contract: Mapping[str, object],
    aux: GaugeConsistentAuxConfig,
) -> dict[str, object]:
    return {
        "schema_version": "faser-four-station-gauge-consistent-aux-weights-v1",
        "algorithm": "train_only_deterministic_mean_scale",
        "relative_target_vs_edge": aux.relative_target_vs_edge,
        "epsilon": aux.epsilon,
        "weight_clip": list(aux.weight_clip),
        "packing_margin": aux.packing_margin,
        "pair_threshold": aux.pair_threshold,
        "unmatched_penalty": aux.unmatched_penalty,
        "enable_dustbin_aware_route_margin": aux.enable_dustbin_aware_route_margin,
        "route_competition_reduction": aux.route_competition_reduction,
        "zero_mean_threshold": aux.zero_mean_threshold,
        "zero_mean_fallback_weight": aux.zero_mean_fallback_weight,
        **dict(weight_contract),
        "written_before_optimizer_steps": True,
        "development_validation_used": False,
    }


def train_gauge_consistent_v2(
    train_bundle: TransformerGraphBundle,
    model_config: RouteAwareTransformerConfig,
    training_config: RouteAwareTrainingConfig,
    stages: Sequence[CurriculumStage],
    aux: GaugeConsistentAuxConfig,
    on_weight_contract: Callable[[Mapping[str, object]], None] | None = None,
) -> tuple[RouteAwareSparseTransformer, RouteAwareTransformerArtifact, list[dict[str, object]], dict[str, object]]:
    """Fixed 30-epoch GPU-only V2 with pre-registered packing/gauge auxiliaries."""
    if tuple(train_bundle.station_path) != (0, 1, 2, 3):
        raise ValueError("V2 is fixed to the four-station IFT -> S1 -> S2 -> S3 path")
    if str(training_config.device) != "cuda":
        raise ValueError("gauge-consistent V2 training is GPU-only; CUDA must be requested")
    _validate_training_config(training_config)
    validated_stages = _validate_stages(stages)
    _seed_everything(training_config.seed)
    device = resolve_device("cuda")
    node_standardizer, edge_standardizer = fit_graph_standardizers(train_bundle)
    model = RouteAwareSparseTransformer(model_config).to(device)
    train_route_tables = materialize_route_candidate_tables(train_bundle.graphs)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    first_graphs = _stage_graphs(train_bundle, validated_stages[0].maximum_magnitude_mm)
    first_pairs, first_leftovers = pair_gauge_twin_graphs(first_graphs)
    edge_positive_weight, _, _ = _edge_positive_weight(
        first_graphs, validated_stages[0].maximum_magnitude_mm, training_config.maximum_edge_positive_weight
    )
    if aux.copy_frozen_aux_weights is not None:
        required = (
            "gauge_twin_consistency_weight",
            "packing_route_competition_weight",
            "dustbin_aware_route_margin_weight",
        )
        missing = [key for key in required if key not in aux.copy_frozen_aux_weights]
        if missing:
            raise ValueError("frozen aux weights missing: " + ", ".join(missing))
        weight_contract = {
            "algorithm": "copy_frozen_workbook59_aux_weights",
            "source_checkpoint_sha256": "6565d028e01e561105d4ce467d10d5c82e5a2435daad6a69c47a07920defb5dc",
            "scale_estimation_skipped": True,
            "gauge_twin_consistency_weight": float(aux.copy_frozen_aux_weights["gauge_twin_consistency_weight"]),
            "packing_route_competition_weight": float(
                aux.copy_frozen_aux_weights["packing_route_competition_weight"]
            ),
            "dustbin_aware_route_margin_weight": float(
                aux.copy_frozen_aux_weights["dustbin_aware_route_margin_weight"]
            ),
        }
    else:
        weight_contract = estimate_aux_loss_scales(
            model,
            first_graphs,
            first_pairs,
            first_leftovers,
            node_standardizer,
            edge_standardizer,
            train_route_tables,
            training_config,
            aux,
            device,
            edge_positive_weight,
        )
    gauge_weight = float(weight_contract["gauge_twin_consistency_weight"])
    competition_weight = float(weight_contract["packing_route_competition_weight"])
    dustbin_weight = float(weight_contract.get("dustbin_aware_route_margin_weight") or 0.0)
    weight_payload = _aux_weight_contract_payload(weight_contract, aux)
    if on_weight_contract is not None:
        on_weight_contract(weight_payload)
    history: list[dict[str, object]] = []
    global_epoch = 0
    last_state: Mapping[str, torch.Tensor] | None = None

    for stage_index, stage in enumerate(validated_stages):
        graphs = _stage_graphs(train_bundle, stage.maximum_magnitude_mm)
        pairs, leftovers = pair_gauge_twin_graphs(graphs)
        edge_positive_weight, positives, negatives = _edge_positive_weight(
            graphs, stage.maximum_magnitude_mm, training_config.maximum_edge_positive_weight
        )
        for stage_epoch in range(stage.epochs):
            rng = np.random.default_rng(training_config.seed + global_epoch)
            totals = {
                "loss": 0.0,
                "edge": 0.0,
                "route_consistency": 0.0,
                "one_to_one_competition": 0.0,
                "fake_route_penalty": 0.0,
                "gauge_twin": 0.0,
                "packing_route_competition": 0.0,
                "dustbin_aware_route_margin": 0.0,
            }
            batches = 0
            model.train()
            for batch_graphs in _iter_training_batches(pairs, leftovers, training_config.batch_size, rng):
                batch = _make_route_batch(
                    batch_graphs, node_standardizer, edge_standardizer, device, train_route_tables
                )
                optimizer.zero_grad(set_to_none=True)
                output = _forward_route_batch(model, batch)
                body = _route_loss_components(output, batch, training_config, edge_positive_weight)
                aux_losses = _aux_losses_for_batch(
                    batch_graphs, output.edge_logits, pairs, aux,
                    route_logits=output.route_logits, route_tables=train_route_tables
                )
                total = (
                    body["total"]
                    + gauge_weight * aux_losses["gauge_twin"]
                    + competition_weight * aux_losses["packing_route_competition"]
                    + dustbin_weight * aux_losses["dustbin_aware_route_margin"]
                )
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                totals["loss"] += float(total.detach().cpu())
                totals["edge"] += float(body["edge"].detach().cpu())
                totals["route_consistency"] += float(body["route_consistency"].detach().cpu())
                totals["one_to_one_competition"] += float(body["one_to_one_competition"].detach().cpu())
                totals["fake_route_penalty"] += float(body["fake_route_penalty"].detach().cpu())
                totals["gauge_twin"] += float(aux_losses["gauge_twin"].detach().cpu())
                totals["packing_route_competition"] += float(
                    aux_losses["packing_route_competition"].detach().cpu()
                )
                if "dustbin_aware_route_margin" in aux_losses:
                    totals["dustbin_aware_route_margin"] += float(
                        aux_losses["dustbin_aware_route_margin"].detach().cpu()
                    )
                batches += 1
            last_state = copy.deepcopy(model.state_dict())
            history.append(
                {
                    "phase": "gauge_consistent_route",
                    "global_epoch": global_epoch,
                    "stage_index": stage_index,
                    "stage_name": stage.name,
                    "stage_epoch": stage_epoch,
                    "maximum_magnitude_mm": stage.maximum_magnitude_mm,
                    "train_graphs": len(graphs),
                    "gauge_twin_pairs": len(pairs),
                    "training_positive_edge_rows": positives,
                    "training_negative_edge_rows": negatives,
                    "edge_positive_weight": edge_positive_weight,
                    "training_loss": totals["loss"] / max(batches, 1),
                    "training_edge_loss": totals["edge"] / max(batches, 1),
                    "training_route_consistency_loss": totals["route_consistency"] / max(batches, 1),
                    "training_one_to_one_competition_loss": totals["one_to_one_competition"] / max(batches, 1),
                    "training_fake_route_penalty": totals["fake_route_penalty"] / max(batches, 1),
                    "training_gauge_twin_loss": totals["gauge_twin"] / max(batches, 1),
                    "training_packing_route_competition_loss": totals["packing_route_competition"]
                    / max(batches, 1),
                    "training_dustbin_aware_route_margin_loss": totals["dustbin_aware_route_margin"]
                    / max(batches, 1),
                    "checkpoint_rule": "last_completed_epoch_of_fixed_30_epoch_budget",
                    "is_best": True,
                }
            )
            print(json.dumps(history[-1], sort_keys=False), flush=True)
            global_epoch += 1
    if last_state is None:
        raise RuntimeError("gauge-consistent V2 did not complete any epoch")
    model.load_state_dict(last_state)
    train_route_summary = {
        "graphs": len(train_bundle.graphs),
        "complete_route_tables": len(train_route_tables),
    }
    artifact = RouteAwareTransformerArtifact(
        node_feature_names=NODE_FEATURE_NAMES,
        edge_feature_names=EDGE_FEATURE_NAMES,
        all_station_pairs=train_bundle.all_station_pairs,
        output_station_pairs=ADJACENT_STATION_PAIRS,
        node_standardizer=node_standardizer,
        edge_standardizer=edge_standardizer,
        model_config=model_config,
        context_mode="full_event",
        training_summary={
            "best_global_epoch": global_epoch - 1,
            "training_global_epochs_completed": global_epoch,
            "checkpoint_selection": "last_completed_epoch_of_fixed_30_epoch_budget",
            "development_validation_used": False,
            "device": str(device),
            "train_bundle": graph_bundle_summary(train_bundle),
            "train_route_candidates": train_route_summary,
            "aux_loss_weights": weight_contract,
            "loss": {
                "edge": "weighted_focal_binary_cross_entropy",
                "route_truth_consistency": "weighted_focal_binary_cross_entropy",
                "one_to_one_competition": "endpoint_incident_route_logsumexp",
                "fake_route_penalty": "softplus_on_fake_endpoint_routes",
                "gauge_twin_consistency": "mse_origin_matched_raw_logits_and_complete_route_utilities",
                "packing_route_competition": "relu_local_packing_utility_margin",
                "dustbin_aware_route_margin": (
                    "relu_max_fragment_or_dustbin_plus_margin_minus_truth"
                    if aux.enable_dustbin_aware_route_margin
                    else "disabled"
                ),
                "edge_loss_weight": training_config.edge_loss_weight,
                "route_consistency_weight": training_config.route_consistency_weight,
                "one_to_one_competition_weight": training_config.one_to_one_competition_weight,
                "fake_route_penalty_weight": training_config.fake_route_penalty_weight,
                "gauge_twin_consistency_weight": gauge_weight,
                "packing_route_competition_weight": competition_weight,
                "dustbin_aware_route_margin_weight": dustbin_weight,
                "packing_route_competition_margin": aux.packing_margin,
                "truth_or_synthetic_provenance_is_model_feature": False,
            },
            "operating_convention": {
                "score_stream": "raw_sigmoid_identity_calibration",
                "pair_thresholds": {"0->1": 0.001, "1->2": 0.001, "2->3": 0.001},
                "unmatched_penalty": aux.unmatched_penalty,
                "selection_split": "pre_registered_frozen_historical_packing",
            },
            "early_stopping": {
                "selection_split": "train_only",
                "route_solver_used": False,
                "calibration_used": False,
                "threshold_used": False,
            },
        },
    )
    return model, artifact, history, weight_payload
