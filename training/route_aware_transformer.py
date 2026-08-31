"""Train and evaluate the validation-only route-aware Transformer V2.

V2 deliberately starts from the same mode-0 Acts physical candidate graph as
V1.  A route candidate is an explicit complete IFT -> S1 -> S2 -> S3 chain of
three existing adjacent candidate edges.  It is a learned contextual query
whose calibrated score can be injected as a residual utility into the existing
unit-capacity route assignment backend; it never creates a new candidate edge.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from baselines.mlp_pair_classifier import FeatureStandardizer
from datasets.synthetic_overlay import SYNTHETIC_ROLE_FIELD_HARD_FAKE
from evaluation.pairwise_metrics import (
    binary_calibration,
    calibration_report,
    platt_calibration_report,
)
from models.route_transformer import (
    RelativeRouteSparseTransformer,
    RelativeRouteTransformerConfig,
    RouteAwareSparseTransformer,
    RouteAwareTransformerConfig,
    freeze_backbone_and_edge_scorer,
)
from training.curriculum_mlp import CandidateSet
from training.geometry_aware_transformer import (
    ADJACENT_STATION_PAIRS,
    ALL_STATION_PAIRS,
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    CurriculumStage,
    TransformerGraph,
    TransformerGraphBundle,
    _make_batch,
    _seed_everything,
    _weighted_focal_bce,
    candidate_score_metrics,
    fit_graph_standardizers,
    graph_bundle_summary,
    resolve_device,
)


@dataclass(frozen=True)
class RouteCandidateTable:
    """Complete-route candidates for one full-event physical graph."""

    node_indices: np.ndarray
    score_edge_indices: np.ndarray
    labels: np.ndarray
    fake_endpoint: np.ndarray
    hard_negative: np.ndarray

    @property
    def size(self) -> int:
        return int(self.labels.size)


@dataclass(frozen=True)
class RouteScoreSet:
    """V2 route-query scores for one physical full-event graph.

    ``endpoint_indices`` are original event-row indices in station order
    IFT/S1/S2/S3.  They identify only pre-existing complete adjacent physical
    chains and can therefore be consumed by the existing unit-capacity route
    solver without constructing a new coordinate-level candidate.
    """

    sample: object
    event: object
    endpoint_indices: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    fake_endpoint: np.ndarray
    hard_negative: np.ndarray

    @property
    def size(self) -> int:
        return int(self.scores.size)


@dataclass(frozen=True)
class RouteAwareTrainingConfig:
    """Optimization and loss controls, kept separate from V2 architecture."""

    batch_size: int = 8
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    seed: int = 20260812
    device: str = "auto"
    early_stopping_patience: int = 6
    focal_gamma: float = 1.5
    hard_negative_weight: float = 2.0
    maximum_edge_positive_weight: float = 30.0
    maximum_route_positive_weight: float = 80.0
    edge_loss_weight: float = 1.0
    route_consistency_weight: float = 1.0
    one_to_one_competition_weight: float = 0.25
    fake_route_penalty_weight: float = 0.25


@dataclass(frozen=True)
class RouteAwareTransformerArtifact:
    """Persisted inference and train-only normalization contract for V2."""

    node_feature_names: tuple[str, ...]
    edge_feature_names: tuple[str, ...]
    all_station_pairs: tuple[tuple[int, int], ...]
    output_station_pairs: tuple[tuple[int, int], ...]
    node_standardizer: FeatureStandardizer
    edge_standardizer: FeatureStandardizer
    model_config: RouteAwareTransformerConfig
    context_mode: str
    training_summary: Mapping[str, object]


@dataclass(frozen=True)
class RouteAwarePrediction:
    """Reconstructed edge scores and route truth-only diagnostic arrays."""

    edge_scores: tuple[np.ndarray, ...]
    base_edge_scores: tuple[np.ndarray, ...]
    route_scores: np.ndarray
    route_labels: np.ndarray
    route_fake_endpoint: np.ndarray
    route_hard_negative: np.ndarray
    route_edge_counts: np.ndarray
    route_score_sets: tuple[RouteScoreSet, ...]


@dataclass(frozen=True)
class _RouteTensorBatch:
    """One standard V1 graph batch with its explicit route-query tables."""

    base: Any
    route_node_indices: torch.Tensor
    route_score_edge_indices: torch.Tensor
    route_labels: torch.Tensor
    route_fake_endpoint: torch.Tensor
    route_hard_negative: torch.Tensor


def _adjacent_pair_ids() -> tuple[int, int, int]:
    index = {pair: row for row, pair in enumerate(ALL_STATION_PAIRS)}
    return tuple(index[pair] for pair in ADJACENT_STATION_PAIRS)


def _empty_route_table() -> RouteCandidateTable:
    return RouteCandidateTable(
        node_indices=np.empty((0, 4), dtype=np.int64),
        score_edge_indices=np.empty((0, 3), dtype=np.int64),
        labels=np.empty(0, dtype=bool),
        fake_endpoint=np.empty(0, dtype=bool),
        hard_negative=np.empty(0, dtype=bool),
    )


def enumerate_complete_route_candidates(graph: TransformerGraph) -> RouteCandidateTable:
    """Materialize only complete chains already present in physical adjacencies.

    The graph must be ``full_event`` so local node indices are original event
    rows.  No candidate is cut by score, chi2, truth, synthetic role, or a
    learned feature.  Truth/synthetic provenance below creates supervision
    labels and audit buckets only after the physical route set is fixed.
    """
    if graph.node_features.shape[0] != graph.event.size:
        raise ValueError("route candidates require a full-event graph node table")
    expected_station_ids = np.asarray(graph.event.station_id, dtype=np.int64)
    if not np.array_equal(graph.station_ids, expected_station_ids):
        raise ValueError("route candidates require full-event station ordering")
    if graph.event.truth_particle_id is None:
        raise ValueError("route-aware V2 training requires MC truth particle IDs")
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
            raise ValueError("V2 score graph includes a non-adjacent output edge") from error
        source_station, target_station = ADJACENT_STATION_PAIRS[adjacent_index]
        if int(graph.station_ids[int(source)]) != source_station or int(graph.station_ids[int(target)]) != target_station:
            raise ValueError("adjacent score edge station IDs disagree with its declared pair")
        maps[adjacent_index].setdefault(int(source), []).append((int(target), row))

    node_rows: list[tuple[int, int, int, int]] = []
    edge_rows: list[tuple[int, int, int]] = []
    for source_zero, second_rows in maps[0].items():
        for source_one, edge_zero_one in second_rows:
            for source_two, edge_one_two in maps[1].get(source_one, ()):
                for source_three, edge_two_three in maps[2].get(source_two, ()):
                    node_rows.append((source_zero, source_one, source_two, source_three))
                    edge_rows.append((edge_zero_one, edge_one_two, edge_two_three))
    if not node_rows:
        return _empty_route_table()
    nodes = np.asarray(node_rows, dtype=np.int64)
    edges = np.asarray(edge_rows, dtype=np.int64)
    truth = np.asarray(graph.event.truth_particle_id, dtype=np.int64)[nodes]
    labels = np.all(truth >= 0, axis=1) & np.all(truth == truth[:, :1], axis=1)
    fake_endpoint = np.any(truth < 0, axis=1)
    roles = graph.event.synthetic_role
    hard_negative = np.zeros(nodes.shape[0], dtype=bool)
    if roles is not None:
        hard_negative |= np.any(
            np.asarray(roles, dtype=np.int64)[nodes] == SYNTHETIC_ROLE_FIELD_HARD_FAKE,
            axis=1,
        )
    hard_negative |= np.any(np.asarray(graph.score_hard_negative, dtype=bool)[edges], axis=1)
    hard_negative &= ~labels
    return RouteCandidateTable(
        node_indices=nodes,
        score_edge_indices=edges,
        labels=np.asarray(labels, dtype=bool),
        fake_endpoint=np.asarray(fake_endpoint, dtype=bool),
        hard_negative=np.asarray(hard_negative, dtype=bool),
    )


def route_candidate_table_summary(tables: Iterable[RouteCandidateTable]) -> dict[str, int]:
    """Truth-only candidate composition used for V2 loss auditing."""
    graphs = 0
    routes = 0
    positive = 0
    fake = 0
    hard = 0
    for table in tables:
        graphs += 1
        routes += table.size
        positive += int(np.count_nonzero(table.labels))
        fake += int(np.count_nonzero(table.fake_endpoint))
        hard += int(np.count_nonzero(table.hard_negative))
    return {
        "graphs": graphs,
        "complete_route_candidates": routes,
        "positive_truth_consistent_routes": positive,
        "fake_endpoint_routes": fake,
        "hard_negative_routes": hard,
    }


def materialize_route_candidate_tables(
    graphs: Sequence[TransformerGraph],
) -> dict[int, RouteCandidateTable]:
    """Cache immutable physical route chains once per train/validation graph.

    Repeatedly re-enumerating the same route candidates every curriculum epoch
    would make V2 compute-bound in Python.  The cache is intentionally local
    to the caller's supplied graph sequence, so a train/validation run never
    materializes a route table for an excluded test graph.
    """
    result: dict[int, RouteCandidateTable] = {}
    for graph in graphs:
        key = id(graph)
        if key in result:  # pragma: no cover - graph bundles are unique by construction
            raise ValueError("route candidate cache received a duplicate graph object")
        result[key] = enumerate_complete_route_candidates(graph)
    return result


def _make_route_batch(
    graphs: Sequence[TransformerGraph],
    node_standardizer: FeatureStandardizer,
    edge_standardizer: FeatureStandardizer,
    device: torch.device,
    route_tables: Mapping[int, RouteCandidateTable] | None = None,
) -> _RouteTensorBatch:
    """Attach graph-local route candidates to one existing sparse graph batch."""
    base = _make_batch(graphs, node_standardizer, edge_standardizer, device)
    node_offset = 0
    score_offset = 0
    route_nodes: list[np.ndarray] = []
    route_edges: list[np.ndarray] = []
    route_labels: list[np.ndarray] = []
    route_fake: list[np.ndarray] = []
    route_hard: list[np.ndarray] = []
    for graph in graphs:
        table = (
            enumerate_complete_route_candidates(graph)
            if route_tables is None
            else route_tables[id(graph)]
        )
        if table.size:
            route_nodes.append(table.node_indices + node_offset)
            route_edges.append(table.score_edge_indices + score_offset)
            route_labels.append(np.asarray(table.labels, dtype=np.float32))
            route_fake.append(np.asarray(table.fake_endpoint, dtype=bool))
            route_hard.append(np.asarray(table.hard_negative, dtype=bool))
        node_offset += graph.node_features.shape[0]
        score_offset += graph.score_labels.size

    def cat(values: list[np.ndarray], dtype: np.dtype[Any], width: int) -> np.ndarray:
        if values:
            return np.concatenate(values, axis=0).astype(dtype, copy=False)
        return np.empty((0, width), dtype=dtype) if width > 1 else np.empty(0, dtype=dtype)

    return _RouteTensorBatch(
        base=base,
        route_node_indices=torch.from_numpy(cat(route_nodes, np.int64, 4)).to(device),
        route_score_edge_indices=torch.from_numpy(cat(route_edges, np.int64, 3)).to(device),
        route_labels=torch.from_numpy(cat(route_labels, np.float32, 1)).to(device),
        route_fake_endpoint=torch.from_numpy(cat(route_fake, np.bool_, 1)).to(device),
        route_hard_negative=torch.from_numpy(cat(route_hard, np.bool_, 1)).to(device),
    )


def _forward_route_batch(
    model: RouteAwareSparseTransformer, batch: _RouteTensorBatch
):
    base = batch.base
    return model(
        base.node_features,
        base.station_ids,
        base.message_edge_source,
        base.message_edge_destination,
        base.message_edge_features,
        base.message_edge_chi2,
        base.message_edge_station_pair,
        base.message_edge_direction,
        base.score_edge_source,
        base.score_edge_destination,
        base.score_edge_features,
        base.score_edge_station_pair,
        batch.route_node_indices,
        batch.route_score_edge_indices,
    )


def _segment_logsumexp(values: torch.Tensor, segments: torch.Tensor, size: int) -> torch.Tensor:
    """Stable logsumexp over route incidences of each physical node."""
    if values.ndim != 1 or segments.shape != values.shape:
        raise ValueError("segment logsumexp inputs are not aligned vectors")
    if not values.numel():
        return values.new_full((size,), -torch.inf)
    maximum = values.new_full((size,), -torch.inf)
    maximum.scatter_reduce_(0, segments, values, reduce="amax", include_self=True)
    exponent = torch.exp(values - maximum[segments])
    normalizer = values.new_zeros((size,))
    normalizer.scatter_add_(0, segments, exponent)
    return maximum + torch.log(normalizer.clamp_min(torch.finfo(values.dtype).tiny))


def _route_competition_loss(
    route_logits: torch.Tensor,
    route_labels: torch.Tensor,
    route_node_indices: torch.Tensor,
    node_count: int,
) -> torch.Tensor:
    """Make each true route beat every competing route at its four endpoints."""
    if not route_logits.numel() or not torch.any(route_labels > 0.5):
        return route_logits.sum() * 0.0
    positive = route_labels > 0.5
    incidence_nodes = route_node_indices.reshape(-1)
    incidence_logits = route_logits.repeat_interleave(4)
    denominators = _segment_logsumexp(incidence_logits, incidence_nodes, node_count)
    # Every positive full truth chain appears once per station endpoint.  The
    # average keeps the loss scale invariant under the fixed four stations.
    return torch.mean(
        -route_logits[positive]
        + torch.mean(denominators[route_node_indices[positive]], dim=1)
    )


def _route_loss_components(
    output: Any,
    batch: _RouteTensorBatch,
    config: RouteAwareTrainingConfig,
    edge_positive_weight: float,
) -> dict[str, torch.Tensor]:
    """V2 edge, route-consistency, competition and fake-route objectives."""
    edge_loss = _weighted_focal_bce(
        output.edge_logits,
        batch.base.score_labels,
        batch.base.score_hard_negative,
        edge_positive_weight,
        config.focal_gamma,
        config.hard_negative_weight,
    )
    if output.route_logits.numel():
        positives = int(torch.count_nonzero(batch.route_labels > 0.5).item())
        negatives = int(batch.route_labels.numel() - positives)
        route_positive_weight = (
            min(negatives / positives, config.maximum_route_positive_weight)
            if positives
            else 1.0
        )
        route_consistency = _weighted_focal_bce(
            output.route_logits,
            batch.route_labels,
            batch.route_hard_negative,
            route_positive_weight,
            config.focal_gamma,
            config.hard_negative_weight,
        )
        competition = _route_competition_loss(
            output.route_logits,
            batch.route_labels,
            batch.route_node_indices,
            int(batch.base.node_features.shape[0]),
        )
        fake_mask = batch.route_fake_endpoint
        fake_penalty = (
            nn.functional.softplus(output.route_logits[fake_mask]).mean()
            if torch.any(fake_mask)
            else output.route_logits.sum() * 0.0
        )
    else:
        route_positive_weight = 1.0
        route_consistency = edge_loss * 0.0
        competition = edge_loss * 0.0
        fake_penalty = edge_loss * 0.0
    total = (
        config.edge_loss_weight * edge_loss
        + config.route_consistency_weight * route_consistency
        + config.one_to_one_competition_weight * competition
        + config.fake_route_penalty_weight * fake_penalty
    )
    return {
        "total": total,
        "edge": edge_loss,
        "route_consistency": route_consistency,
        "one_to_one_competition": competition,
        "fake_route_penalty": fake_penalty,
        "route_positive_weight": total.new_tensor(route_positive_weight),
    }


def _validate_training_config(config: RouteAwareTrainingConfig) -> None:
    if config.batch_size < 1 or config.early_stopping_patience < 1:
        raise ValueError("V2 batch_size and early_stopping_patience must be positive")
    if config.learning_rate <= 0.0 or config.weight_decay < 0.0:
        raise ValueError("V2 learning_rate must be positive and weight_decay non-negative")
    for name in (
        "maximum_edge_positive_weight",
        "maximum_route_positive_weight",
        "edge_loss_weight",
        "route_consistency_weight",
        "one_to_one_competition_weight",
        "fake_route_penalty_weight",
        "hard_negative_weight",
    ):
        if float(getattr(config, name)) <= 0.0:
            raise ValueError(f"V2 {name} must be positive")
    if config.focal_gamma < 0.0:
        raise ValueError("V2 focal_gamma must be non-negative")


def _validate_stages(stages: Sequence[CurriculumStage]) -> tuple[CurriculumStage, ...]:
    result = tuple(stages)
    if not result:
        raise ValueError("V2 requires at least one physical curriculum stage")
    previous = -math.inf
    for stage in result:
        if stage.epochs < 1 or not math.isfinite(stage.maximum_magnitude_mm):
            raise ValueError("V2 curriculum stage is invalid")
        if stage.maximum_magnitude_mm < previous:
            raise ValueError("V2 curriculum maximum magnitudes must be non-decreasing")
        previous = stage.maximum_magnitude_mm
    return result


def _stage_graphs(bundle: TransformerGraphBundle, maximum_magnitude_mm: float) -> list[TransformerGraph]:
    graphs = [
        graph
        for graph in bundle.graphs
        if float(graph.sample.curriculum_magnitude) <= maximum_magnitude_mm + 1.0e-12
    ]
    if not graphs:
        raise ValueError("V2 curriculum stage selects no physical graphs")
    return graphs


def _edge_positive_weight(graphs: Sequence[TransformerGraph], maximum: float, cap: float) -> tuple[float, int, int]:
    labels = np.concatenate([graph.score_labels for graph in graphs])
    positives = int(np.count_nonzero(labels))
    negatives = int(labels.size - positives)
    if not positives or not negatives:
        raise ValueError(f"V2 curriculum through {maximum:g} condition units lacks both edge label classes")
    return min(negatives / positives, cap), positives, negatives


def _route_metrics_from_prediction(prediction: RouteAwarePrediction, bins: int) -> dict[str, object]:
    if not prediction.route_scores.size:
        return {
            "rows": 0,
            "positive_rows": 0,
            "fake_endpoint_rows": 0,
            "hard_negative_rows": 0,
            "average_precision": None,
            "roc_auc": None,
        }
    metrics = dict(binary_calibration(prediction.route_scores, prediction.route_labels, bins=bins))
    metrics.update(
        {
            "fake_endpoint_rows": int(np.count_nonzero(prediction.route_fake_endpoint)),
            "hard_negative_rows": int(np.count_nonzero(prediction.route_hard_negative)),
        }
    )
    return metrics


def calibrate_route_query_scores(
    prediction: RouteAwarePrediction,
    *,
    bins: int,
    method: str = "platt",
) -> tuple[tuple[np.ndarray, ...], Mapping[str, object]]:
    """Fit one validation-only monotonic calibration map for full-route scores.

    Route candidates have a single physical meaning independent of adjacent
    station pair, so their calibration is global.  The caller owns the split
    boundary; this helper deliberately accepts only in-memory predictions.
    """
    if method not in {"temperature", "platt"}:
        raise ValueError("route-query calibration method must be 'temperature' or 'platt'")
    if not prediction.route_score_sets:
        raise ValueError("route-query calibration requires at least one physical graph")
    reconstructed = np.concatenate([score_set.scores for score_set in prediction.route_score_sets])
    labels = np.concatenate([score_set.labels for score_set in prediction.route_score_sets])
    if not np.array_equal(reconstructed, prediction.route_scores) or not np.array_equal(labels, prediction.route_labels):
        raise RuntimeError("route-query score sets are not aligned with the flat prediction arrays")
    if not reconstructed.size:
        raise ValueError("route-query calibration requires at least one complete physical route")
    report_function = calibration_report if method == "temperature" else platt_calibration_report
    calibrated, report = report_function(reconstructed, labels, bins=bins)
    result: list[np.ndarray] = []
    offset = 0
    for score_set in prediction.route_score_sets:
        stop = offset + score_set.size
        result.append(np.asarray(calibrated[offset:stop], dtype=np.float64))
        offset = stop
    if offset != calibrated.size:  # pragma: no cover - fixed by concatenation above
        raise RuntimeError("route-query calibration split left a score tail")
    return tuple(result), {
        "scope": "complete_four_station_route",
        "method": method,
        "fit_split": "validation_only",
        **dict(report),
    }


def route_query_score_maps_by_event(
    route_score_sets: Sequence[RouteScoreSet],
    scores: Sequence[np.ndarray],
) -> dict[tuple[str, str, int, int], dict[tuple[int, ...], float]]:
    """Map calibrated route-query scores onto existing physical event chains."""
    if len(route_score_sets) != len(scores):
        raise ValueError("route score sets and calibrated route scores are not aligned")
    result: dict[tuple[str, str, int, int], dict[tuple[int, ...], float]] = {}
    for score_set, values in zip(route_score_sets, scores):
        score = np.asarray(values, dtype=np.float64)
        if score.shape != (score_set.size,) or (score.size and not np.isfinite(score).all()):
            raise ValueError("route-query score shape or values are invalid")
        key = (
            str(score_set.sample.source_id),
            str(score_set.sample.payload_id),
            int(score_set.event.run_id),
            int(score_set.event.event_id),
        )
        if key in result:
            raise ValueError("duplicate route-query event key")
        table: dict[tuple[int, ...], float] = {}
        for endpoints, value in zip(score_set.endpoint_indices, score):
            endpoint_key = tuple(int(index) for index in endpoints)
            if len(endpoint_key) != 4 or any(index < 0 or index >= score_set.event.size for index in endpoint_key):
                raise ValueError("route-query endpoint lies outside its physical event")
            if tuple(int(score_set.event.station_id[index]) for index in endpoint_key) != (0, 1, 2, 3):
                raise ValueError("route-query endpoint stations are not IFT -> S1 -> S2 -> S3")
            if endpoint_key in table:
                raise ValueError("duplicate complete physical route endpoint tuple")
            table[endpoint_key] = float(value)
        result[key] = table
    return result


def predict_route_aware_scores(
    model: RouteAwareSparseTransformer,
    bundle: TransformerGraphBundle,
    node_standardizer: FeatureStandardizer,
    edge_standardizer: FeatureStandardizer,
    *,
    device: str | torch.device = "auto",
    batch_size: int = 8,
    route_tables: Mapping[int, RouteCandidateTable] | None = None,
) -> RouteAwarePrediction:
    """Run V2 on physical graphs and reconstruct all original adjacent rows."""
    if bundle.context_mode != "full_event":
        raise ValueError("route-aware V2 requires full-event physical graph context")
    if batch_size < 1:
        raise ValueError("V2 prediction batch_size must be positive")
    resolved_device = resolve_device(str(device)) if not isinstance(device, torch.device) else device
    resolved_tables = (
        materialize_route_candidate_tables(bundle.graphs) if route_tables is None else route_tables
    )
    if set(resolved_tables) != {id(graph) for graph in bundle.graphs}:
        raise ValueError("V2 route table cache does not match the supplied physical graph bundle")
    edge_scores = [np.full(candidate.labels.shape, np.nan, dtype=np.float64) for candidate in bundle.adjacent_sets]
    base_scores = [np.full(candidate.labels.shape, np.nan, dtype=np.float64) for candidate in bundle.adjacent_sets]
    route_scores: list[np.ndarray] = []
    route_labels: list[np.ndarray] = []
    route_fake: list[np.ndarray] = []
    route_hard: list[np.ndarray] = []
    route_counts: list[np.ndarray] = []
    route_score_sets: list[RouteScoreSet | None] = [None] * len(bundle.graphs)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(bundle.graphs), batch_size):
            batch = _make_route_batch(
                bundle.graphs[start : start + batch_size],
                node_standardizer,
                edge_standardizer,
                resolved_device,
                resolved_tables,
            )
            output = _forward_route_batch(model, batch)
            values = torch.sigmoid(output.edge_logits).detach().cpu().numpy().astype(np.float64)
            base_values = torch.sigmoid(output.base_edge_logits).detach().cpu().numpy().astype(np.float64)
            if values.shape != batch.base.score_owner.shape or base_values.shape != values.shape:
                raise RuntimeError("V2 edge score reconstruction lost candidate rows")
            for value, base, owner, row in zip(
                values, base_values, batch.base.score_owner, batch.base.score_row
            ):
                if np.isfinite(edge_scores[int(owner)][int(row)]):
                    raise RuntimeError("V2 edge score reconstruction duplicated a candidate row")
                edge_scores[int(owner)][int(row)] = float(value)
                base_scores[int(owner)][int(row)] = float(base)
            route_values = torch.sigmoid(output.route_logits).detach().cpu().numpy().astype(np.float64)
            route_offset = 0
            for graph_offset, graph in enumerate(bundle.graphs[start : start + batch_size]):
                table = resolved_tables[id(graph)]
                stop = route_offset + table.size
                values_for_graph = route_values[route_offset:stop]
                if values_for_graph.shape != table.labels.shape:
                    raise RuntimeError("V2 route score reconstruction lost a physical route row")
                route_score_sets[start + graph_offset] = RouteScoreSet(
                    sample=graph.sample,
                    event=graph.event,
                    endpoint_indices=np.asarray(table.node_indices, dtype=np.int64),
                    scores=values_for_graph,
                    labels=np.asarray(table.labels, dtype=bool),
                    fake_endpoint=np.asarray(table.fake_endpoint, dtype=bool),
                    hard_negative=np.asarray(table.hard_negative, dtype=bool),
                )
                route_offset = stop
            if route_offset != route_values.size:
                raise RuntimeError("V2 route score reconstruction has an unexpected batch tail")
            if output.route_logits.numel():
                route_scores.append(route_values)
                route_labels.append(batch.route_labels.detach().cpu().numpy().astype(bool))
                route_fake.append(batch.route_fake_endpoint.detach().cpu().numpy().astype(bool))
                route_hard.append(batch.route_hard_negative.detach().cpu().numpy().astype(bool))
            route_counts.append(output.route_edge_counts.detach().cpu().numpy().astype(np.float64))
    for values in (*edge_scores, *base_scores):
        if values.size and not np.isfinite(values).all():
            raise RuntimeError("V2 edge score reconstruction left a candidate row unfilled")
    if any(value is None for value in route_score_sets):  # pragma: no cover - batch-loop invariant
        raise RuntimeError("V2 route score reconstruction omitted a physical graph")
    return RouteAwarePrediction(
        edge_scores=tuple(edge_scores),
        base_edge_scores=tuple(base_scores),
        route_scores=(np.concatenate(route_scores) if route_scores else np.empty(0, dtype=np.float64)),
        route_labels=(np.concatenate(route_labels) if route_labels else np.empty(0, dtype=bool)),
        route_fake_endpoint=(np.concatenate(route_fake) if route_fake else np.empty(0, dtype=bool)),
        route_hard_negative=(np.concatenate(route_hard) if route_hard else np.empty(0, dtype=bool)),
        route_edge_counts=(np.concatenate(route_counts) if route_counts else np.empty(0, dtype=np.float64)),
        route_score_sets=tuple(value for value in route_score_sets if value is not None),
    )


def train_route_aware_transformer_v2(
    train_bundle: TransformerGraphBundle,
    validation_bundle: TransformerGraphBundle,
    model_config: RouteAwareTransformerConfig,
    training_config: RouteAwareTrainingConfig,
    stages: Sequence[CurriculumStage],
    calibration_bins: int,
) -> tuple[RouteAwareSparseTransformer, RouteAwareTransformerArtifact, list[dict[str, object]]]:
    """Train V2 only on source-disjoint train graphs and early-stop on validation.

    The predeclared early-stop rank is route truth-consistency AP followed by
    adjacent edge AP.  It needs no calibration, threshold, route solver, or
    test sample and is therefore appropriate for architecture/loss selection.
    """
    if train_bundle.context_mode != "full_event" or validation_bundle.context_mode != "full_event":
        raise ValueError("route-aware V2 requires full-event train and validation graph bundles")
    if train_bundle.all_station_pairs != validation_bundle.all_station_pairs:
        raise ValueError("V2 train and validation candidate schemas differ")
    if tuple(train_bundle.station_path) != (0, 1, 2, 3):
        raise ValueError("V2 is fixed to the four-station IFT -> S1 -> S2 -> S3 path")
    _validate_training_config(training_config)
    validated_stages = _validate_stages(stages)
    if calibration_bins < 1:
        raise ValueError("V2 calibration_bins must be positive")

    _seed_everything(training_config.seed)
    device = resolve_device(training_config.device)
    node_standardizer, edge_standardizer = fit_graph_standardizers(train_bundle)
    model = RouteAwareSparseTransformer(model_config).to(device)
    train_route_tables = materialize_route_candidate_tables(train_bundle.graphs)
    validation_route_tables = materialize_route_candidate_tables(validation_bundle.graphs)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    best_state: Mapping[str, torch.Tensor] | None = None
    best_rank: tuple[float, float] = (-math.inf, -math.inf)
    best_epoch = -1
    global_epoch = 0
    history: list[dict[str, object]] = []

    for stage_index, stage in enumerate(validated_stages):
        graphs = _stage_graphs(train_bundle, stage.maximum_magnitude_mm)
        edge_positive_weight, positives, negatives = _edge_positive_weight(
            graphs, stage.maximum_magnitude_mm, training_config.maximum_edge_positive_weight
        )
        without_improvement = 0
        for stage_epoch in range(stage.epochs):
            order = np.random.default_rng(training_config.seed + global_epoch).permutation(len(graphs))
            totals = {
                "loss": 0.0,
                "edge": 0.0,
                "route_consistency": 0.0,
                "one_to_one_competition": 0.0,
                "fake_route_penalty": 0.0,
                "route_candidates": 0,
                "route_positive": 0,
                "route_fake": 0,
                "route_hard": 0,
            }
            batches = 0
            model.train()
            for offset in range(0, len(order), training_config.batch_size):
                batch_graphs = [graphs[int(index)] for index in order[offset : offset + training_config.batch_size]]
                batch = _make_route_batch(
                    batch_graphs,
                    node_standardizer,
                    edge_standardizer,
                    device,
                    train_route_tables,
                )
                optimizer.zero_grad(set_to_none=True)
                output = _forward_route_batch(model, batch)
                losses = _route_loss_components(
                    output, batch, training_config, edge_positive_weight
                )
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                totals["loss"] += float(losses["total"].detach().cpu())
                totals["edge"] += float(losses["edge"].detach().cpu())
                totals["route_consistency"] += float(losses["route_consistency"].detach().cpu())
                totals["one_to_one_competition"] += float(losses["one_to_one_competition"].detach().cpu())
                totals["fake_route_penalty"] += float(losses["fake_route_penalty"].detach().cpu())
                totals["route_candidates"] += int(batch.route_labels.numel())
                totals["route_positive"] += int(torch.count_nonzero(batch.route_labels > 0.5).item())
                totals["route_fake"] += int(torch.count_nonzero(batch.route_fake_endpoint).item())
                totals["route_hard"] += int(torch.count_nonzero(batch.route_hard_negative).item())
                batches += 1

            validation_prediction = predict_route_aware_scores(
                model,
                validation_bundle,
                node_standardizer,
                edge_standardizer,
                device=device,
                batch_size=training_config.batch_size,
                route_tables=validation_route_tables,
            )
            validation_edge = candidate_score_metrics(
                validation_bundle.adjacent_sets,
                validation_prediction.edge_scores,
                calibration_bins,
            )
            validation_route = _route_metrics_from_prediction(validation_prediction, calibration_bins)
            route_ap = validation_route.get("average_precision")
            edge_ap = validation_edge.get("average_precision")
            rank = (
                -math.inf if route_ap is None else float(route_ap),
                -math.inf if edge_ap is None else float(edge_ap),
            )
            improved = rank > best_rank
            if improved:
                best_rank = rank
                best_epoch = global_epoch
                best_state = copy.deepcopy(model.state_dict())
                without_improvement = 0
            else:
                without_improvement += 1
            history.append(
                {
                    "phase": "route_aware_joint",
                    "global_epoch": global_epoch,
                    "stage_index": stage_index,
                    "stage_name": stage.name,
                    "stage_epoch": stage_epoch,
                    "maximum_magnitude_mm": stage.maximum_magnitude_mm,
                    "train_graphs": len(graphs),
                    "training_positive_edge_rows": positives,
                    "training_negative_edge_rows": negatives,
                    "edge_positive_weight": edge_positive_weight,
                    "training_loss": totals["loss"] / max(batches, 1),
                    "training_edge_loss": totals["edge"] / max(batches, 1),
                    "training_route_consistency_loss": totals["route_consistency"] / max(batches, 1),
                    "training_one_to_one_competition_loss": totals["one_to_one_competition"] / max(batches, 1),
                    "training_fake_route_penalty": totals["fake_route_penalty"] / max(batches, 1),
                    "training_route_candidates": totals["route_candidates"],
                    "training_positive_routes": totals["route_positive"],
                    "training_fake_endpoint_routes": totals["route_fake"],
                    "training_hard_negative_routes": totals["route_hard"],
                    "validation_edge": validation_edge,
                    "validation_route": validation_route,
                    "validation_route_average_precision": route_ap,
                    "validation_edge_average_precision": edge_ap,
                    "validation_rank": list(rank),
                    "is_best": improved,
                }
            )
            global_epoch += 1
            if without_improvement >= training_config.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("V2 did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    # Route-candidate accounting is recorded after the selected checkpoint;
    # it only inspects fixed physical candidate chains and MC labels.
    train_route_summary = route_candidate_table_summary(train_route_tables.values())
    validation_route_summary = route_candidate_table_summary(validation_route_tables.values())
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
            "best_validation_route_average_precision": best_rank[0],
            "best_validation_edge_average_precision": best_rank[1],
            "best_global_epoch": best_epoch,
            "training_global_epochs_completed": global_epoch,
            "device": str(device),
            "train_bundle": graph_bundle_summary(train_bundle),
            "validation_bundle": graph_bundle_summary(validation_bundle),
            "train_route_candidates": train_route_summary,
            "validation_route_candidates": validation_route_summary,
            "loss": {
                "edge": "weighted_focal_binary_cross_entropy",
                "route_truth_consistency": "weighted_focal_binary_cross_entropy",
                "one_to_one_competition": "endpoint_incident_route_logsumexp",
                "fake_route_penalty": "softplus_on_fake_endpoint_routes",
                "edge_loss_weight": training_config.edge_loss_weight,
                "route_consistency_weight": training_config.route_consistency_weight,
                "one_to_one_competition_weight": training_config.one_to_one_competition_weight,
                "fake_route_penalty_weight": training_config.fake_route_penalty_weight,
                "hard_negative_weight": training_config.hard_negative_weight,
                "focal_gamma": training_config.focal_gamma,
                "truth_or_synthetic_provenance_is_model_feature": False,
            },
            "early_stopping": {
                "selection_split": "validation_only",
                "rank": "route_average_precision_then_edge_average_precision",
                "route_solver_used": False,
                "calibration_used": False,
                "threshold_used": False,
            },
        },
    )
    return model, artifact, history


def save_route_aware_transformer_artifact(
    path: str | Path,
    model: RouteAwareSparseTransformer,
    artifact: RouteAwareTransformerArtifact,
) -> None:
    """Save a V2 checkpoint with only train-derived feature standardizers."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "faser-route-aware-transformer-v2",
            "model_state_dict": model.state_dict(),
            "node_feature_names": list(artifact.node_feature_names),
            "edge_feature_names": list(artifact.edge_feature_names),
            "all_station_pairs": [list(pair) for pair in artifact.all_station_pairs],
            "output_station_pairs": [list(pair) for pair in artifact.output_station_pairs],
            "node_standardizer_mean": artifact.node_standardizer.mean,
            "node_standardizer_scale": artifact.node_standardizer.scale,
            "edge_standardizer_mean": artifact.edge_standardizer.mean,
            "edge_standardizer_scale": artifact.edge_standardizer.scale,
            "model_config": artifact.model_config.as_dict(),
            "context_mode": artifact.context_mode,
            "training_summary": dict(artifact.training_summary),
        },
        target,
    )


def load_route_aware_transformer_artifact(
    path: str | Path,
    *,
    device: str = "auto",
) -> tuple[RouteAwareSparseTransformer, RouteAwareTransformerArtifact]:
    """Restore an immutable V2 inference contract without test data access."""
    source = Path(path).expanduser().resolve()
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with the LCG PyTorch build.
        payload = torch.load(source, map_location="cpu")
    if payload.get("schema_version") != "faser-route-aware-transformer-v2":
        raise ValueError("unexpected route-aware Transformer checkpoint schema")
    model_config = RouteAwareTransformerConfig(**dict(payload["model_config"]))
    model = RouteAwareSparseTransformer(model_config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(resolve_device(device))
    artifact = RouteAwareTransformerArtifact(
        node_feature_names=tuple(payload["node_feature_names"]),
        edge_feature_names=tuple(payload["edge_feature_names"]),
        all_station_pairs=tuple(tuple(int(value) for value in pair) for pair in payload["all_station_pairs"]),
        output_station_pairs=tuple(
            tuple(int(value) for value in pair) for pair in payload["output_station_pairs"]
        ),
        node_standardizer=FeatureStandardizer(
            mean=np.asarray(payload["node_standardizer_mean"], dtype=np.float64),
            scale=np.asarray(payload["node_standardizer_scale"], dtype=np.float64),
        ),
        edge_standardizer=FeatureStandardizer(
            mean=np.asarray(payload["edge_standardizer_mean"], dtype=np.float64),
            scale=np.asarray(payload["edge_standardizer_scale"], dtype=np.float64),
        ),
        model_config=model_config,
        context_mode=str(payload["context_mode"]),
        training_summary=dict(payload["training_summary"]),
    )
    if artifact.node_feature_names != NODE_FEATURE_NAMES or artifact.edge_feature_names != EDGE_FEATURE_NAMES:
        raise ValueError("V2 checkpoint feature schema is incompatible with the physical candidate graph")
    if artifact.all_station_pairs != ALL_STATION_PAIRS or artifact.output_station_pairs != ADJACENT_STATION_PAIRS:
        raise ValueError("V2 checkpoint station-pair schema is incompatible with route assignment")
    if artifact.context_mode != "full_event":
        raise ValueError("route-aware V2 checkpoint is not full-event context")
    return model, artifact


def route_aware_artifact_summary(artifact: RouteAwareTransformerArtifact) -> dict[str, object]:
    """JSON-safe checkpoint evidence for the validation-only V2 contract."""
    return {
        "schema_version": "faser-route-aware-transformer-v2",
        "node_feature_names": list(artifact.node_feature_names),
        "edge_feature_names": list(artifact.edge_feature_names),
        "all_station_pairs": [f"{left}->{right}" for left, right in artifact.all_station_pairs],
        "output_station_pairs": [f"{left}->{right}" for left, right in artifact.output_station_pairs],
        "model_config": asdict(artifact.model_config),
        "context_mode": artifact.context_mode,
        "training_summary": dict(artifact.training_summary),
    }
