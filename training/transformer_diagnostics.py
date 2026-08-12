"""Validation-only probes for sparse Transformer message-passing mechanisms.

The helpers in this module are intentionally post-test safe: callers provide a
bundle that contains only train or validation graphs, and every probe changes
only the supplied sparse message edges or executed block depth.  Candidate rows,
physical Acts residuals/covariances, score output edges, calibration and route
assignment controls remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from baselines.route_assignment import RouteAssignmentConfig, adjacent_route_assignment
from training.curriculum_mlp import CandidateSet
from training.geometry_aware_transformer import (
    ADJACENT_STATION_PAIRS,
    ALL_STATION_PAIRS,
    TransformerGraph,
    TransformerGraphBundle,
    _make_batch,
    candidate_score_metrics,
    resolve_device,
)
from training.route_assignment import (
    evaluate_adjacent_route_assignment_sets,
    prepare_route_assignment_context,
)


@dataclass(frozen=True)
class MessageProbe:
    """A deterministic sparse-message intervention for one frozen model."""

    name: str
    maximum_layers: int
    maximum_station_span: int | None = None
    allowed_directions: tuple[int, ...] = (0, 1)
    excluded_stations: tuple[int, ...] = ()

    @staticmethod
    def from_mapping(raw: Mapping[str, object]) -> "MessageProbe":
        name = str(raw.get("name", ""))
        if not name:
            raise ValueError("message probe requires a non-empty name")
        maximum_layers = int(raw.get("maximum_layers", 4))
        span_raw = raw.get("maximum_station_span")
        maximum_station_span = None if span_raw is None else int(span_raw)
        directions_raw = raw.get("allowed_directions", (0, 1))
        if not isinstance(directions_raw, Sequence) or isinstance(directions_raw, (str, bytes)):
            raise ValueError(f"message probe '{name}' allowed_directions must be a sequence")
        directions = tuple(sorted({int(value) for value in directions_raw}))
        if not directions or any(value not in {0, 1} for value in directions):
            raise ValueError(f"message probe '{name}' directions must be drawn from 0/1")
        excluded_raw = raw.get("excluded_stations", ())
        if not isinstance(excluded_raw, Sequence) or isinstance(excluded_raw, (str, bytes)):
            raise ValueError(f"message probe '{name}' excluded_stations must be a sequence")
        excluded = tuple(sorted({int(value) for value in excluded_raw}))
        if any(value < 0 or value > 3 for value in excluded):
            raise ValueError(f"message probe '{name}' excludes an unsupported station")
        if maximum_layers < 0 or maximum_layers > 4:
            raise ValueError(f"message probe '{name}' maximum_layers must be in [0, 4]")
        if maximum_station_span is not None and maximum_station_span < 1:
            raise ValueError(f"message probe '{name}' maximum_station_span must be positive")
        return MessageProbe(
            name=name,
            maximum_layers=maximum_layers,
            maximum_station_span=maximum_station_span,
            allowed_directions=directions,
            excluded_stations=excluded,
        )


@dataclass(frozen=True)
class ProbePrediction:
    """Scores and representation summaries for a single message intervention."""

    scores: tuple[np.ndarray, ...]
    depth_scores: Mapping[int, tuple[np.ndarray, ...]]
    layer_edge_metrics: tuple[Mapping[str, object], ...]
    node_state_summary: tuple[Mapping[str, object], ...]
    attention_summary: tuple[Mapping[str, object], ...]
    message_edge_counts: Mapping[str, int]


def _message_mask(graph: TransformerGraph, probe: MessageProbe) -> np.ndarray:
    """Select existing message edges without changing score/output edges."""
    rows = graph.message_edge_source.size
    if not rows:
        return np.empty(0, dtype=bool)
    source_station = graph.station_ids[graph.message_edge_source]
    target_station = graph.station_ids[graph.message_edge_destination]
    mask = np.isin(graph.message_edge_direction, np.asarray(probe.allowed_directions))
    if probe.maximum_station_span is not None:
        mask &= np.abs(source_station - target_station) <= probe.maximum_station_span
    if probe.excluded_stations:
        excluded = np.asarray(probe.excluded_stations, dtype=np.int64)
        mask &= ~np.isin(source_station, excluded)
        mask &= ~np.isin(target_station, excluded)
    return np.asarray(mask, dtype=bool)


def apply_message_probe(
    bundle: TransformerGraphBundle, probe: MessageProbe
) -> TransformerGraphBundle:
    """Copy a graph bundle while filtering only declared sparse messages."""
    if probe.maximum_layers > 4:
        raise ValueError("Transformer V1 exposes at most four sparse blocks")
    graphs: list[TransformerGraph] = []
    for graph in bundle.graphs:
        mask = _message_mask(graph, probe)
        graphs.append(
            replace(
                graph,
                message_edge_source=graph.message_edge_source[mask],
                message_edge_destination=graph.message_edge_destination[mask],
                message_edge_features=graph.message_edge_features[mask],
                message_edge_chi2=graph.message_edge_chi2[mask],
                message_edge_station_pair=graph.message_edge_station_pair[mask],
                message_edge_direction=graph.message_edge_direction[mask],
            )
        )
    return TransformerGraphBundle(
        graphs=tuple(graphs),
        adjacent_sets=bundle.adjacent_sets,
        all_station_pairs=bundle.all_station_pairs,
        station_path=bundle.station_path,
        context_mode=bundle.context_mode,
    )


def _empty_score_arrays(bundle: TransformerGraphBundle) -> list[np.ndarray]:
    return [
        np.full(candidate_set.labels.shape, np.nan, dtype=np.float64)
        for candidate_set in bundle.adjacent_sets
    ]


def _rank_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if first.size < 2:
        return None
    if np.isclose(np.std(first), 0.0) or np.isclose(np.std(second), 0.0):
        return None
    rank_first = np.empty(first.size, dtype=np.float64)
    rank_second = np.empty(second.size, dtype=np.float64)
    rank_first[np.argsort(first, kind="mergesort")] = np.arange(first.size, dtype=np.float64)
    rank_second[np.argsort(second, kind="mergesort")] = np.arange(second.size, dtype=np.float64)
    value = float(np.corrcoef(rank_first, rank_second)[0, 1])
    return value if math.isfinite(value) else None


def score_drift_metrics(
    reference: Sequence[np.ndarray], values: Sequence[np.ndarray], candidate_sets: Sequence[CandidateSet]
) -> dict[str, object]:
    """Measure score movement relative to a frozen full-depth reference."""
    if not (len(reference) == len(values) == len(candidate_sets)):
        raise ValueError("score drift inputs are not aligned")
    rows: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] = {}
    all_reference: list[np.ndarray] = []
    all_values: list[np.ndarray] = []
    for expected, actual, candidate_set in zip(reference, values, candidate_sets):
        left = np.asarray(expected, dtype=np.float64)
        right = np.asarray(actual, dtype=np.float64)
        if left.shape != right.shape or left.shape != candidate_set.labels.shape:
            raise ValueError("score drift arrays differ from candidate rows")
        if left.size and (not np.isfinite(left).all() or not np.isfinite(right).all()):
            raise ValueError("score drift arrays are non-finite")
        rows.setdefault(candidate_set.station_pair, []).append((left, right))
        all_reference.append(left)
        all_values.append(right)

    def summarize(parts: Sequence[tuple[np.ndarray, np.ndarray]]) -> dict[str, float | None]:
        left = np.concatenate([value[0] for value in parts]) if parts else np.empty(0)
        right = np.concatenate([value[1] for value in parts]) if parts else np.empty(0)
        if not left.size:
            return {
                "rows": 0,
                "mean_signed_score_drift": None,
                "mean_absolute_score_drift": None,
                "rms_score_drift": None,
                "score_rank_correlation": None,
            }
        delta = right - left
        return {
            "rows": int(left.size),
            "mean_signed_score_drift": float(np.mean(delta)),
            "mean_absolute_score_drift": float(np.mean(np.abs(delta))),
            "rms_score_drift": float(np.sqrt(np.mean(np.square(delta)))),
            "score_rank_correlation": _rank_correlation(left, right),
        }

    result: dict[str, object] = summarize(list(zip(all_reference, all_values)))
    result["by_station_pair"] = {
        f"{pair[0]}->{pair[1]}": summarize(parts)
        for pair, parts in sorted(rows.items())
    }
    return result


def _state_rows(
    node_states_by_depth: Sequence[torch.Tensor],
    graphs: Sequence[TransformerGraph],
    *,
    probe_name: str,
) -> list[dict[str, object]]:
    """Aggregate graph-local state movement and cosine concentration by layer."""
    accumulators: dict[tuple[int, int], dict[str, float]] = {}
    arrays = [state.detach().cpu().numpy().astype(np.float64, copy=False) for state in node_states_by_depth]
    offset = 0
    for graph in graphs:
        size = graph.node_features.shape[0]
        station_ids = graph.station_ids
        initial = arrays[0][offset : offset + size]
        for depth, values in enumerate(arrays):
            current = values[offset : offset + size]
            for station in np.unique(station_ids):
                mask = station_ids == station
                rows = current[mask]
                baseline = initial[mask]
                key = (depth, int(station))
                accumulator = accumulators.setdefault(
                    key,
                    {
                        "nodes": 0.0,
                        "norm_sum": 0.0,
                        "update_l2_sum": 0.0,
                        "within_cosine_sum": 0.0,
                        "within_cosine_pairs": 0.0,
                    },
                )
                accumulator["nodes"] += float(rows.shape[0])
                accumulator["norm_sum"] += float(np.linalg.norm(rows, axis=1).sum())
                accumulator["update_l2_sum"] += float(np.linalg.norm(rows - baseline, axis=1).sum())
                if rows.shape[0] >= 2:
                    normalized = rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1.0e-12)
                    cosine = normalized @ normalized.T
                    upper = cosine[np.triu_indices(rows.shape[0], k=1)]
                    accumulator["within_cosine_sum"] += float(np.sum(upper))
                    accumulator["within_cosine_pairs"] += float(upper.size)
        offset += size
    result: list[dict[str, object]] = []
    for (depth, station), value in sorted(accumulators.items()):
        nodes = value["nodes"]
        pairs = value["within_cosine_pairs"]
        result.append(
            {
                "probe": probe_name,
                "depth": depth,
                "station": station,
                "nodes": int(nodes),
                "mean_node_norm": None if not nodes else value["norm_sum"] / nodes,
                "mean_update_l2_from_depth0": None if not nodes else value["update_l2_sum"] / nodes,
                "mean_within_station_cosine": (
                    None if not pairs else value["within_cosine_sum"] / pairs
                ),
                "within_station_cosine_pairs": int(pairs),
            }
        )
    return result


def _attention_rows(
    attention_by_layer: Sequence[torch.Tensor],
    message_station_pair: torch.Tensor,
    message_direction: torch.Tensor,
    *,
    probe_name: str,
) -> list[dict[str, object]]:
    """Aggregate attention values over the existing edge policy by layer/pair."""
    result: list[dict[str, object]] = []
    if not attention_by_layer:
        return result
    pair_values = message_station_pair.detach().cpu().numpy().astype(np.int64, copy=False)
    direction_values = message_direction.detach().cpu().numpy().astype(np.int64, copy=False)
    for layer, attention in enumerate(attention_by_layer, start=1):
        values = attention.detach().cpu().numpy().astype(np.float64, copy=False)
        if values.shape[0] != pair_values.size:
            raise RuntimeError("attention trace is not aligned with message edges")
        for pair_id, pair in enumerate(ALL_STATION_PAIRS):
            for direction in (0, 1):
                mask = (pair_values == pair_id) & (direction_values == direction)
                if not np.any(mask):
                    continue
                selected = values[mask]
                result.append(
                    {
                        "probe": probe_name,
                        "layer": layer,
                        "station_pair": f"{pair[0]}->{pair[1]}",
                        "direction": "forward" if direction == 0 else "backward",
                        "message_edges": int(np.count_nonzero(mask)),
                        "mean_attention": float(np.mean(selected)),
                        "std_attention": float(np.std(selected)),
                        "mean_head_attention_std": float(np.mean(np.std(selected, axis=1))),
                    }
                )
    return result


def _message_edge_counts(bundle: TransformerGraphBundle) -> dict[str, int]:
    counts: dict[str, int] = {}
    for graph in bundle.graphs:
        for pair_id, direction in zip(
            graph.message_edge_station_pair.tolist(), graph.message_edge_direction.tolist()
        ):
            pair = ALL_STATION_PAIRS[int(pair_id)]
            key = f"{pair[0]}->{pair[1]}:{'forward' if int(direction) == 0 else 'backward'}"
            counts[key] = counts.get(key, 0) + 1
    return {key: int(value) for key, value in sorted(counts.items())}


def predict_transformer_probe(
    model: torch.nn.Module,
    bundle: TransformerGraphBundle,
    node_standardizer: object,
    edge_standardizer: object,
    probe: MessageProbe,
    *,
    device: str = "auto",
    batch_size: int = 64,
    calibration_bins: int = 15,
) -> ProbePrediction:
    """Score a frozen model under a validation-only message intervention."""
    if batch_size < 1:
        raise ValueError("diagnostic batch_size must be positive")
    probed = apply_message_probe(bundle, probe)
    resolved_device = resolve_device(device)
    score_by_depth = {
        depth: _empty_score_arrays(probed) for depth in range(probe.maximum_layers + 1)
    }
    layer_metrics: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    attention_rows: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(probed.graphs), batch_size):
            graphs = probed.graphs[start : start + batch_size]
            batch = _make_batch(graphs, node_standardizer, edge_standardizer, resolved_device)
            _, trace = model.forward_with_trace(
                batch.node_features,
                batch.station_ids,
                batch.message_edge_source,
                batch.message_edge_destination,
                batch.message_edge_features,
                batch.message_edge_chi2,
                batch.message_edge_station_pair,
                batch.message_edge_direction,
                batch.score_edge_source,
                batch.score_edge_destination,
                batch.score_edge_features,
                batch.score_edge_station_pair,
                maximum_layers=probe.maximum_layers,
            )
            for depth, logits in enumerate(trace["logits_by_depth"]):
                probabilities = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
                destination = score_by_depth[depth]
                for value, owner, row in zip(probabilities, batch.score_owner, batch.score_row):
                    if np.isfinite(destination[int(owner)][int(row)]):
                        raise RuntimeError("diagnostic score reconstruction duplicated a candidate row")
                    destination[int(owner)][int(row)] = float(value)
            state_rows.extend(
                _state_rows(trace["node_states_by_depth"], graphs, probe_name=probe.name)
            )
            attention_rows.extend(
                _attention_rows(
                    trace["attention_by_layer"],
                    batch.message_edge_station_pair,
                    batch.message_edge_direction,
                    probe_name=probe.name,
                )
            )
    depth_scores: dict[int, tuple[np.ndarray, ...]] = {}
    for depth, values in score_by_depth.items():
        for score in values:
            if score.size and not np.isfinite(score).all():
                raise RuntimeError("diagnostic score reconstruction left a candidate row unfilled")
        depth_scores[depth] = tuple(values)
        metrics = candidate_score_metrics(probed.adjacent_sets, values, calibration_bins)
        for pair, pair_metrics in sorted(metrics["by_station_pair"].items()):
            layer_metrics.append(
                {
                    "probe": probe.name,
                    "depth": depth,
                    "station_pair": pair,
                    "average_precision": pair_metrics["average_precision"],
                    "roc_auc": pair_metrics["roc_auc"],
                    "expected_calibration_error": pair_metrics["expected_calibration_error"],
                    "negative_log_likelihood": pair_metrics["negative_log_likelihood"],
                    "rows": pair_metrics["rows"],
                    "positive_rows": pair_metrics["positive_rows"],
                }
            )
    return ProbePrediction(
        scores=depth_scores[probe.maximum_layers],
        depth_scores=depth_scores,
        layer_edge_metrics=tuple(layer_metrics),
        node_state_summary=tuple(state_rows),
        attention_summary=tuple(attention_rows),
        message_edge_counts=_message_edge_counts(probed),
    )


def _route_false_contributions(
    candidate_sets: Sequence[CandidateSet],
    scores: Sequence[np.ndarray],
    config: RouteAssignmentConfig,
    calibration_bins: int,
    *,
    probe_name: str,
) -> list[dict[str, object]]:
    """Break false selected routes into fake-endpoint and mixed-truth causes."""
    context = prepare_route_assignment_context(
        candidate_sets, scores, calibration_bins, config.station_path
    )
    counts: dict[tuple[float, str, str], int] = {}
    for group in context.groups:
        result = adjacent_route_assignment(group.event, group.station_matrices, config)
        magnitude = float(group.sample.magnitude_mm)
        for route in result.routes:
            endpoints = tuple(int(index) for _, index in route.endpoints)
            if group.event.truth_particle_id is None:
                category = "truth_unavailable"
            else:
                truth_ids = [int(group.event.truth_particle_id[index]) for index in endpoints]
                if all(value >= 0 for value in truth_ids) and len(set(truth_ids)) == 1:
                    category = "truth_consistent_route"
                elif any(value < 0 for value in truth_ids):
                    category = "fake_endpoint_route"
                else:
                    category = "mixed_or_ambiguous_route"
            counts[(magnitude, category, "all")] = counts.get((magnitude, category, "all"), 0) + 1
            if category != "truth_consistent_route":
                for pair, _ in route.matches:
                    key = f"{pair[0]}->{pair[1]}"
                    counts[(magnitude, "false_route_edge", key)] = (
                        counts.get((magnitude, "false_route_edge", key), 0) + 1
                    )
    return [
        {
            "probe": probe_name,
            "magnitude_mm": magnitude,
            "category": category,
            "station_pair": pair,
            "count": count,
        }
        for (magnitude, category, pair), count in sorted(counts.items())
    ]


def evaluate_probe_with_frozen_route_controls(
    bundle: TransformerGraphBundle,
    scores: Sequence[np.ndarray],
    calibration_bins: int,
    route_config: RouteAssignmentConfig,
    *,
    probe_name: str,
) -> tuple[dict[float, Mapping[str, object]], list[dict[str, object]]]:
    """Evaluate a score intervention with unchanged calibration/route controls."""
    by_magnitude: dict[float, tuple[list[CandidateSet], list[np.ndarray]]] = {}
    for candidate_set, score in zip(bundle.adjacent_sets, scores):
        magnitude = float(candidate_set.sample.magnitude_mm)
        sets, values = by_magnitude.setdefault(magnitude, ([], []))
        sets.append(candidate_set)
        values.append(np.asarray(score, dtype=np.float64))
    metrics: dict[float, Mapping[str, object]] = {}
    for magnitude, (sets, values) in sorted(by_magnitude.items()):
        metrics[magnitude] = evaluate_adjacent_route_assignment_sets(
            sets, values, route_config, calibration_bins=calibration_bins
        )
    false_rows = _route_false_contributions(
        bundle.adjacent_sets,
        scores,
        route_config,
        calibration_bins,
        probe_name=probe_name,
    )
    return metrics, false_rows

