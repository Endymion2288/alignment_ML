"""Physical-candidate graph construction and training for Transformer V1.

This module deliberately consumes the same ``CandidateSet`` objects used by
the pairwise MLP baseline.  Candidate construction therefore remains the
mode-0 Acts propagation path, including the refitted tracklet state and its
combined covariance.  Truth and synthetic fake provenance appear only in the
training labels/loss weights and evaluation helpers, never in node or edge
features supplied to the network.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from baselines.field_chi2_matching import FieldCandidate
from baselines.mlp_pair_classifier import FeatureStandardizer
from datasets.physical_curriculum import CurriculumSample
from datasets.root_loader import EventTracklets
from datasets.synthetic_overlay import SYNTHETIC_ROLE_FIELD_HARD_FAKE
from evaluation.pairwise_metrics import (
    apply_platt_scaling,
    apply_temperature,
    binary_calibration,
    calibration_report,
    platt_calibration_report,
)
from models.transformer import (
    GeometryAwareSparseTransformer,
    SparseTransformerConfig,
    model_config_from_payload,
)
from training.curriculum_mlp import CandidateSet


ALL_STATION_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (source, target) for source in range(4) for target in range(source + 1, 4)
)
ADJACENT_STATION_PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (2, 3))
NODE_FEATURE_NAMES: tuple[str, ...] = (
    "x_mm",
    "y_mm",
    "tx",
    "ty",
    "z_mm",
    "log_sigma_x_mm",
    "log_sigma_y_mm",
    "log_sigma_tx",
    "log_sigma_ty",
    "log1p_local_chi2_per_ndof",
    "log1p_n_hit",
    *(f"hit_layer_{layer}_side_{side}" for layer in range(3) for side in range(2)),
)
EDGE_FEATURE_NAMES: tuple[str, ...] = (
    "residual_x_mm",
    "residual_y_mm",
    "residual_tx",
    "residual_ty",
    "pull_x",
    "pull_y",
    "pull_tx",
    "pull_ty",
    "log1p_chi2",
    "combined_covariance_logdet",
    "delta_z_mm",
)


@dataclass(frozen=True)
class CurriculumStage:
    """One source-disjoint physical misalignment curriculum stage."""

    name: str
    maximum_magnitude_mm: float
    epochs: int


@dataclass(frozen=True)
class TransformerTrainingConfig:
    """Training controls separate from the persisted model architecture."""

    batch_size: int = 16
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    seed: int = 20260812
    device: str = "auto"
    early_stopping_patience: int = 6
    focal_gamma: float = 1.5
    hard_negative_weight: float = 2.0
    maximum_positive_weight: float = 30.0
    local_pretrain_learning_rate: float | None = None


@dataclass(frozen=True)
class TransformerGraph:
    """One sparse graph or one isolated adjacent-pair graph for an event."""

    sample: CurriculumSample
    event: EventTracklets
    node_features: np.ndarray
    station_ids: np.ndarray
    message_edge_source: np.ndarray
    message_edge_destination: np.ndarray
    message_edge_features: np.ndarray
    message_edge_chi2: np.ndarray
    message_edge_station_pair: np.ndarray
    message_edge_direction: np.ndarray
    score_edge_source: np.ndarray
    score_edge_destination: np.ndarray
    score_edge_features: np.ndarray
    score_edge_station_pair: np.ndarray
    score_labels: np.ndarray
    score_hard_negative: np.ndarray
    score_owner: np.ndarray
    score_row: np.ndarray


@dataclass(frozen=True)
class TransformerGraphBundle:
    """Graphs plus the ordered adjacent candidate sets whose scores they fill."""

    graphs: tuple[TransformerGraph, ...]
    adjacent_sets: tuple[CandidateSet, ...]
    all_station_pairs: tuple[tuple[int, int], ...]
    station_path: tuple[int, ...]
    context_mode: str


@dataclass(frozen=True)
class TransformerArtifact:
    """Complete inference contract saved alongside a Transformer checkpoint."""

    node_feature_names: tuple[str, ...]
    edge_feature_names: tuple[str, ...]
    all_station_pairs: tuple[tuple[int, int], ...]
    output_station_pairs: tuple[tuple[int, int], ...]
    node_standardizer: FeatureStandardizer
    edge_standardizer: FeatureStandardizer
    model_config: SparseTransformerConfig
    context_mode: str
    training_summary: Mapping[str, object]


def resolve_device(requested: str) -> torch.device:
    """Resolve the declared CPU/GPU device without silently changing a request."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("a CUDA device was requested but CUDA is unavailable")
    return device


def _event_key(candidate_set: CandidateSet) -> tuple[str, str, int, int]:
    return (
        str(candidate_set.sample.source_id),
        str(candidate_set.sample.payload_id),
        int(candidate_set.event.run_id),
        int(candidate_set.event.event_id),
    )


def _node_features(event: EventTracklets, indices: np.ndarray) -> np.ndarray:
    rows = np.asarray(indices, dtype=np.int64)
    if rows.ndim != 1 or not rows.size:
        raise ValueError("a Transformer graph must contain at least one tracklet")
    covariance_diagonal = np.diagonal(event.covariance[rows], axis1=1, axis2=2)
    if np.any(covariance_diagonal <= 0.0) or not np.isfinite(covariance_diagonal).all():
        raise ValueError("tracklet covariance diagonal is invalid")
    quality = event.chi2[rows] / np.maximum(event.ndof[rows], 1.0)
    values = [
        event.state[rows],
        event.z_mm[rows, None],
        np.log(np.sqrt(covariance_diagonal)),
        np.log1p(np.maximum(quality, 0.0))[:, None],
        np.log1p(np.maximum(event.n_hit[rows], 0.0))[:, None],
        np.asarray(
            [[(int(pattern) >> bit) & 1 for bit in range(6)] for pattern in event.hit_pattern[rows]],
            dtype=np.float64,
        ),
    ]
    result = np.concatenate(values, axis=1).astype(np.float64, copy=False)
    if result.shape[1] != len(NODE_FEATURE_NAMES) or not np.isfinite(result).all():
        raise ValueError("node-feature construction produced an invalid matrix")
    return result


def geometric_edge_features(
    event: EventTracklets, candidate: FieldCandidate, reverse: bool = False
) -> np.ndarray:
    """Build the verified residual/covariance geometry vector for one edge.

    The reverse message is not a new physics candidate.  It is the opposite
    message direction of the same forward Acts candidate, with signed residual,
    pull and delta-z components so a receiving node can distinguish direction.
    """
    sign, logdet = np.linalg.slogdet(candidate.combined_covariance)
    if sign <= 0.0 or not np.isfinite(logdet):
        raise ValueError("candidate combined covariance is not positive definite")
    if not np.isfinite(candidate.chi2) or candidate.chi2 < 0.0:
        raise ValueError("candidate chi2 is invalid")
    direction = -1.0 if reverse else 1.0
    values = np.asarray(
        [
            *(direction * np.asarray(candidate.residual, dtype=np.float64)).tolist(),
            *(direction * np.asarray(candidate.pull, dtype=np.float64)).tolist(),
            float(np.log1p(candidate.chi2)),
            float(logdet),
            direction
            * float(event.z_mm[candidate.target_index] - event.z_mm[candidate.source_index]),
        ],
        dtype=np.float64,
    )
    if values.shape != (len(EDGE_FEATURE_NAMES),) or not np.isfinite(values).all():
        raise ValueError("geometric edge-feature construction produced invalid values")
    return values


def _hard_negative_mask(event: EventTracklets, candidate: FieldCandidate, label: bool) -> bool:
    """Flag a hard fake for loss weighting only; it is never a model feature."""
    if label or event.synthetic_role is None:
        return False
    role = event.synthetic_role
    return bool(
        int(role[candidate.source_index]) == SYNTHETIC_ROLE_FIELD_HARD_FAKE
        or int(role[candidate.target_index]) == SYNTHETIC_ROLE_FIELD_HARD_FAKE
    )


def _pair_index(station_pairs: Sequence[tuple[int, int]]) -> dict[tuple[int, int], int]:
    result = {tuple(int(value) for value in pair): index for index, pair in enumerate(station_pairs)}
    if len(result) != len(station_pairs):
        raise ValueError("station-pair list contains duplicates")
    return result


def _build_graph(
    candidate_sets: Mapping[tuple[int, int], CandidateSet],
    output_owner: Mapping[int, int],
    message_pairs: Sequence[tuple[int, int]],
    output_pairs: Sequence[tuple[int, int]],
    node_indices: np.ndarray,
    pair_to_index: Mapping[tuple[int, int], int],
) -> TransformerGraph:
    first = candidate_sets[next(iter(candidate_sets))]
    event = first.event
    sample = first.sample
    nodes = np.asarray(node_indices, dtype=np.int64)
    local_index = np.full(event.size, -1, dtype=np.int64)
    local_index[nodes] = np.arange(nodes.size, dtype=np.int64)

    message_source: list[int] = []
    message_destination: list[int] = []
    message_features: list[np.ndarray] = []
    message_chi2: list[float] = []
    message_pairs_index: list[int] = []
    message_direction: list[int] = []
    for pair in message_pairs:
        candidate_set = candidate_sets[pair]
        for candidate in candidate_set.candidates:
            source = int(local_index[candidate.source_index])
            target = int(local_index[candidate.target_index])
            if source < 0 or target < 0:
                raise ValueError("candidate endpoint is absent from its graph node table")
            pair_id = pair_to_index[pair]
            message_source.extend((source, target))
            message_destination.extend((target, source))
            message_features.extend(
                (geometric_edge_features(event, candidate, reverse=False), geometric_edge_features(event, candidate, reverse=True))
            )
            message_chi2.extend((float(candidate.chi2), float(candidate.chi2)))
            message_pairs_index.extend((pair_id, pair_id))
            message_direction.extend((0, 1))

    score_source: list[int] = []
    score_destination: list[int] = []
    score_features: list[np.ndarray] = []
    score_pairs_index: list[int] = []
    score_labels: list[bool] = []
    score_hard_negative: list[bool] = []
    score_owner: list[int] = []
    score_row: list[int] = []
    for pair in output_pairs:
        candidate_set = candidate_sets[pair]
        owner = output_owner[id(candidate_set)]
        for row, (candidate, label) in enumerate(zip(candidate_set.candidates, candidate_set.labels)):
            source = int(local_index[candidate.source_index])
            target = int(local_index[candidate.target_index])
            if source < 0 or target < 0:
                raise ValueError("output candidate endpoint is absent from its graph node table")
            score_source.append(source)
            score_destination.append(target)
            score_features.append(geometric_edge_features(event, candidate, reverse=False))
            score_pairs_index.append(pair_to_index[pair])
            score_labels.append(bool(label))
            score_hard_negative.append(_hard_negative_mask(event, candidate, bool(label)))
            score_owner.append(owner)
            score_row.append(row)

    def edge_array(values: list[np.ndarray]) -> np.ndarray:
        return (
            np.asarray(values, dtype=np.float64).reshape(-1, len(EDGE_FEATURE_NAMES))
            if values
            else np.empty((0, len(EDGE_FEATURE_NAMES)), dtype=np.float64)
        )

    return TransformerGraph(
        sample=sample,
        event=event,
        node_features=_node_features(event, nodes),
        station_ids=np.asarray(event.station_id[nodes], dtype=np.int64),
        message_edge_source=np.asarray(message_source, dtype=np.int64),
        message_edge_destination=np.asarray(message_destination, dtype=np.int64),
        message_edge_features=edge_array(message_features),
        message_edge_chi2=np.asarray(message_chi2, dtype=np.float64),
        message_edge_station_pair=np.asarray(message_pairs_index, dtype=np.int64),
        message_edge_direction=np.asarray(message_direction, dtype=np.int64),
        score_edge_source=np.asarray(score_source, dtype=np.int64),
        score_edge_destination=np.asarray(score_destination, dtype=np.int64),
        score_edge_features=edge_array(score_features),
        score_edge_station_pair=np.asarray(score_pairs_index, dtype=np.int64),
        score_labels=np.asarray(score_labels, dtype=bool),
        score_hard_negative=np.asarray(score_hard_negative, dtype=bool),
        score_owner=np.asarray(score_owner, dtype=np.int64),
        score_row=np.asarray(score_row, dtype=np.int64),
    )


def build_transformer_graph_bundle(
    candidate_sets: Sequence[CandidateSet],
    station_path: Sequence[int] = (0, 1, 2, 3),
    context_mode: str = "full_event",
) -> TransformerGraphBundle:
    """Turn existing physical candidate sets into sparse Transformer graphs.

    ``full_event`` uses all six forward station-pair candidate sets as message
    edges and scores only adjacent outputs.  ``station_pair`` is the explicit
    no-multi-station-context ablation: each adjacent pair is a separate
    two-station graph while retaining the same encoder, candidate rows, labels
    and route assignment downstream.
    """
    stations = tuple(int(station) for station in station_path)
    if stations != (0, 1, 2, 3):
        raise ValueError("Transformer V1 is fixed to the IFT -> S1 -> S2 -> S3 path")
    if context_mode not in {"full_event", "station_pair"}:
        raise ValueError("context_mode must be 'full_event' or 'station_pair'")
    pair_to_index = _pair_index(ALL_STATION_PAIRS)
    grouped: dict[tuple[str, str, int, int], dict[tuple[int, int], CandidateSet]] = {}
    for candidate_set in candidate_sets:
        pair = tuple(int(value) for value in candidate_set.station_pair)
        if pair not in pair_to_index:
            raise ValueError(f"candidate set has unsupported station pair {pair}")
        rows = grouped.setdefault(_event_key(candidate_set), {})
        if pair in rows:
            raise ValueError("duplicate station-pair candidate set for one physical event")
        rows[pair] = candidate_set
    if not grouped:
        raise ValueError("no candidate sets were supplied")

    adjacent_sets = tuple(
        candidate_set
        for candidate_set in candidate_sets
        if tuple(int(value) for value in candidate_set.station_pair) in ADJACENT_STATION_PAIRS
    )
    if not adjacent_sets:
        raise ValueError("candidate graph has no adjacent station-pair sets")
    output_owner = {id(candidate_set): row for row, candidate_set in enumerate(adjacent_sets)}
    graphs: list[TransformerGraph] = []
    for key, by_pair in sorted(grouped.items()):
        missing = set(ALL_STATION_PAIRS) - set(by_pair)
        if missing:
            raise ValueError(f"candidate graph {key} is missing physical pair(s): {sorted(missing)}")
        event_ids = {id(candidate_set.event) for candidate_set in by_pair.values()}
        sample_ids = {id(candidate_set.sample) for candidate_set in by_pair.values()}
        if len(event_ids) != 1 or len(sample_ids) != 1:
            raise ValueError("one candidate graph key resolves to inconsistent event/sample objects")
        event = next(iter(by_pair.values())).event
        if context_mode == "full_event":
            graphs.append(
                _build_graph(
                    by_pair,
                    output_owner,
                    ALL_STATION_PAIRS,
                    ADJACENT_STATION_PAIRS,
                    np.arange(event.size, dtype=np.int64),
                    pair_to_index,
                )
            )
            continue
        for pair in ADJACENT_STATION_PAIRS:
            indices = np.concatenate(
                (event.indices_for_station(pair[0]), event.indices_for_station(pair[1]))
            ).astype(np.int64, copy=False)
            if not indices.size:
                raise ValueError("station-pair ablation graph has no endpoint tracklets")
            graphs.append(
                _build_graph(
                    by_pair,
                    output_owner,
                    (pair,),
                    (pair,),
                    indices,
                    pair_to_index,
                )
            )
    return TransformerGraphBundle(
        graphs=tuple(graphs),
        adjacent_sets=adjacent_sets,
        all_station_pairs=ALL_STATION_PAIRS,
        station_path=stations,
        context_mode=context_mode,
    )


def fit_graph_standardizers(bundle: TransformerGraphBundle) -> tuple[FeatureStandardizer, FeatureStandardizer]:
    """Fit node and edge transforms only from the supplied training graphs."""
    node_values = np.concatenate([graph.node_features for graph in bundle.graphs], axis=0)
    edge_parts = [graph.message_edge_features for graph in bundle.graphs if graph.message_edge_features.size]
    if not edge_parts:
        raise ValueError("training candidate graphs contain no physical message edges")
    edge_values = np.concatenate(edge_parts, axis=0)
    return FeatureStandardizer.fit(node_values), FeatureStandardizer.fit(edge_values)


def graph_bundle_summary(bundle: TransformerGraphBundle) -> dict[str, object]:
    """Compact graph and label accounting retained with each train/test run."""
    return {
        "context_mode": bundle.context_mode,
        "graphs": len(bundle.graphs),
        "adjacent_candidate_sets": len(bundle.adjacent_sets),
        "nodes": int(sum(graph.node_features.shape[0] for graph in bundle.graphs)),
        "message_edges_directed": int(sum(graph.message_edge_source.size for graph in bundle.graphs)),
        "output_edges": int(sum(graph.score_labels.size for graph in bundle.graphs)),
        "positive_output_edges": int(sum(np.count_nonzero(graph.score_labels) for graph in bundle.graphs)),
        "hard_negative_output_edges": int(
            sum(np.count_nonzero(graph.score_hard_negative) for graph in bundle.graphs)
        ),
    }


@dataclass(frozen=True)
class _TensorBatch:
    node_features: torch.Tensor
    station_ids: torch.Tensor
    message_edge_source: torch.Tensor
    message_edge_destination: torch.Tensor
    message_edge_features: torch.Tensor
    message_edge_chi2: torch.Tensor
    message_edge_station_pair: torch.Tensor
    message_edge_direction: torch.Tensor
    score_edge_source: torch.Tensor
    score_edge_destination: torch.Tensor
    score_edge_features: torch.Tensor
    score_edge_station_pair: torch.Tensor
    score_labels: torch.Tensor
    score_hard_negative: torch.Tensor
    score_owner: np.ndarray
    score_row: np.ndarray


def _float32(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all() or np.any(np.abs(array) > np.finfo(np.float32).max):
        raise ValueError(f"{name} cannot be represented as finite float32")
    return array.astype(np.float32)


def _make_batch(
    graphs: Sequence[TransformerGraph],
    node_standardizer: FeatureStandardizer,
    edge_standardizer: FeatureStandardizer,
    device: torch.device,
) -> _TensorBatch:
    if not graphs:
        raise ValueError("cannot create a Transformer batch from no graphs")
    node_offset = 0
    node_features: list[np.ndarray] = []
    station_ids: list[np.ndarray] = []
    message_source: list[np.ndarray] = []
    message_destination: list[np.ndarray] = []
    message_features: list[np.ndarray] = []
    message_chi2: list[np.ndarray] = []
    message_pair: list[np.ndarray] = []
    message_direction: list[np.ndarray] = []
    score_source: list[np.ndarray] = []
    score_destination: list[np.ndarray] = []
    score_features: list[np.ndarray] = []
    score_pair: list[np.ndarray] = []
    score_labels: list[np.ndarray] = []
    score_hard_negative: list[np.ndarray] = []
    score_owner: list[np.ndarray] = []
    score_row: list[np.ndarray] = []
    for graph in graphs:
        node_features.append(node_standardizer.transform(graph.node_features))
        station_ids.append(np.asarray(graph.station_ids, dtype=np.int64))
        message_source.append(graph.message_edge_source + node_offset)
        message_destination.append(graph.message_edge_destination + node_offset)
        if graph.message_edge_features.size:
            message_features.append(edge_standardizer.transform(graph.message_edge_features))
        message_chi2.append(_float32(graph.message_edge_chi2, "message chi2"))
        message_pair.append(graph.message_edge_station_pair)
        message_direction.append(graph.message_edge_direction)
        score_source.append(graph.score_edge_source + node_offset)
        score_destination.append(graph.score_edge_destination + node_offset)
        if graph.score_edge_features.size:
            score_features.append(edge_standardizer.transform(graph.score_edge_features))
        score_pair.append(graph.score_edge_station_pair)
        score_labels.append(np.asarray(graph.score_labels, dtype=np.float32))
        score_hard_negative.append(np.asarray(graph.score_hard_negative, dtype=bool))
        score_owner.append(graph.score_owner)
        score_row.append(graph.score_row)
        node_offset += graph.node_features.shape[0]

    def cat(values: list[np.ndarray], dtype: np.dtype[Any], shape: tuple[int, ...] = ()) -> np.ndarray:
        if values:
            return np.concatenate(values, axis=0).astype(dtype, copy=False)
        return np.empty((0, *shape), dtype=dtype)

    return _TensorBatch(
        node_features=torch.from_numpy(cat(node_features, np.float32)).to(device),
        station_ids=torch.from_numpy(cat(station_ids, np.int64)).to(device),
        message_edge_source=torch.from_numpy(cat(message_source, np.int64)).to(device),
        message_edge_destination=torch.from_numpy(cat(message_destination, np.int64)).to(device),
        message_edge_features=torch.from_numpy(
            cat(message_features, np.float32, (len(EDGE_FEATURE_NAMES),))
        ).to(device),
        message_edge_chi2=torch.from_numpy(cat(message_chi2, np.float32)).to(device),
        message_edge_station_pair=torch.from_numpy(cat(message_pair, np.int64)).to(device),
        message_edge_direction=torch.from_numpy(cat(message_direction, np.int64)).to(device),
        score_edge_source=torch.from_numpy(cat(score_source, np.int64)).to(device),
        score_edge_destination=torch.from_numpy(cat(score_destination, np.int64)).to(device),
        score_edge_features=torch.from_numpy(
            cat(score_features, np.float32, (len(EDGE_FEATURE_NAMES),))
        ).to(device),
        score_edge_station_pair=torch.from_numpy(cat(score_pair, np.int64)).to(device),
        score_labels=torch.from_numpy(cat(score_labels, np.float32)).to(device),
        score_hard_negative=torch.from_numpy(cat(score_hard_negative, np.bool_)).to(device),
        score_owner=cat(score_owner, np.int64),
        score_row=cat(score_row, np.int64),
    )


def _forward_batch(model: GeometryAwareSparseTransformer, batch: _TensorBatch) -> torch.Tensor:
    return model(
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
    )


def _forward_local_batch(model: GeometryAwareSparseTransformer, batch: _TensorBatch) -> torch.Tensor:
    """Score output candidates with the optional local residual head only."""
    return model.local_edge_logits(
        batch.node_features,
        batch.score_edge_source,
        batch.score_edge_destination,
        batch.score_edge_features,
        batch.score_edge_station_pair,
    )


def _predict_transformer_scores(
    model: GeometryAwareSparseTransformer,
    bundle: TransformerGraphBundle,
    node_standardizer: FeatureStandardizer,
    edge_standardizer: FeatureStandardizer,
    device: str | torch.device = "auto",
    batch_size: int = 16,
    *,
    local_only: bool,
) -> list[np.ndarray]:
    """Return sigmoid edge scores aligned to the bundle's adjacent candidate sets."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if local_only and not model.use_local_edge_residual:
        raise ValueError("local-only scoring requires use_local_edge_residual=True")
    resolved_device = resolve_device(str(device)) if not isinstance(device, torch.device) else device
    result = [np.full(candidate_set.labels.shape, np.nan, dtype=np.float64) for candidate_set in bundle.adjacent_sets]
    model.eval()
    with torch.no_grad():
        for start in range(0, len(bundle.graphs), batch_size):
            batch = _make_batch(
                bundle.graphs[start : start + batch_size],
                node_standardizer,
                edge_standardizer,
                resolved_device,
            )
            logits = _forward_local_batch(model, batch) if local_only else _forward_batch(model, batch)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
            if probabilities.shape != batch.score_owner.shape:
                raise RuntimeError("Transformer score reconstruction lost candidate rows")
            for value, owner, row in zip(probabilities, batch.score_owner, batch.score_row):
                if np.isfinite(result[int(owner)][int(row)]):
                    raise RuntimeError("Transformer score reconstruction duplicated a candidate row")
                result[int(owner)][int(row)] = float(value)
    for values in result:
        if values.size and not np.isfinite(values).all():
            raise RuntimeError("Transformer score reconstruction left candidate rows unfilled")
    return result


def predict_transformer_scores(
    model: GeometryAwareSparseTransformer,
    bundle: TransformerGraphBundle,
    node_standardizer: FeatureStandardizer,
    edge_standardizer: FeatureStandardizer,
    device: str | torch.device = "auto",
    batch_size: int = 16,
) -> list[np.ndarray]:
    """Return full sparse-context sigmoid scores for adjacent candidate edges."""
    return _predict_transformer_scores(
        model,
        bundle,
        node_standardizer,
        edge_standardizer,
        device=device,
        batch_size=batch_size,
        local_only=False,
    )


def predict_local_transformer_scores(
    model: GeometryAwareSparseTransformer,
    bundle: TransformerGraphBundle,
    node_standardizer: FeatureStandardizer,
    edge_standardizer: FeatureStandardizer,
    device: str | torch.device = "auto",
    batch_size: int = 16,
) -> list[np.ndarray]:
    """Return local residual scores before any sparse graph message passing."""
    return _predict_transformer_scores(
        model,
        bundle,
        node_standardizer,
        edge_standardizer,
        device=device,
        batch_size=batch_size,
        local_only=True,
    )


def candidate_score_metrics(
    candidate_sets: Sequence[CandidateSet], scores: Sequence[np.ndarray], bins: int
) -> dict[str, object]:
    """Candidate AUC/PR/calibration accounting for the adjacent score heads."""
    if len(candidate_sets) != len(scores):
        raise ValueError("candidate sets and scores are not aligned")
    score_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    by_pair: dict[tuple[int, int], tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for candidate_set, values in zip(candidate_sets, scores):
        score = np.asarray(values, dtype=np.float64)
        if score.shape != candidate_set.labels.shape or (score.size and not np.isfinite(score).all()):
            raise ValueError("candidate score shape or values are invalid")
        if not score.size:
            continue
        score_parts.append(score)
        labels = np.asarray(candidate_set.labels, dtype=bool)
        label_parts.append(labels)
        holder = by_pair.setdefault(candidate_set.station_pair, ([], []))
        holder[0].append(score)
        holder[1].append(labels)
    if not score_parts:
        raise ValueError("no Transformer candidate scores are available")
    result: dict[str, object] = binary_calibration(
        np.concatenate(score_parts), np.concatenate(label_parts), bins=bins
    )
    result["by_station_pair"] = {
        f"{pair[0]}->{pair[1]}": binary_calibration(
            np.concatenate(parts[0]), np.concatenate(parts[1]), bins=bins
        )
        for pair, parts in sorted(by_pair.items())
    }
    return result


def calibrate_transformer_scores(
    candidate_sets: Sequence[CandidateSet],
    scores: Sequence[np.ndarray],
    calibration_bins: int,
    scope: str = "station_pair",
    method: str = "temperature",
) -> tuple[list[np.ndarray], dict[str, object]]:
    """Fit a validation-only score-calibration map for adjacent Transformer scores.

    ``temperature`` remains the V1 default.  ``platt`` is a strictly monotonic
    slope-plus-intercept map for class-weighted edge logits; it is fitted only
    from validation labels and then persisted for frozen test inference.
    """
    if len(candidate_sets) != len(scores):
        raise ValueError("candidate sets and scores are not aligned")
    if scope not in {"global", "station_pair"}:
        raise ValueError("calibration scope must be 'global' or 'station_pair'")
    if method not in {"temperature", "platt"}:
        raise ValueError("calibration method must be 'temperature' or 'platt'")
    report_function = calibration_report if method == "temperature" else platt_calibration_report
    nonempty = [
        (candidate_set, np.asarray(values, dtype=np.float64))
        for candidate_set, values in zip(candidate_sets, scores)
        if np.asarray(values).size
    ]
    if not nonempty:
        raise ValueError("cannot calibrate an empty Transformer score collection")
    for candidate_set, values in nonempty:
        if values.shape != candidate_set.labels.shape or not np.isfinite(values).all():
            raise ValueError("Transformer calibration score shape or values are invalid")
    raw = np.concatenate([values for _, values in nonempty])
    labels = np.concatenate([candidate_set.labels for candidate_set, _ in nonempty])
    if scope == "global":
        _, report = report_function(raw, labels, bins=calibration_bins)
        calibrated = [
            (
                apply_temperature(np.asarray(values, dtype=np.float64), float(report["temperature"]))
                if method == "temperature"
                else apply_platt_scaling(
                    np.asarray(values, dtype=np.float64),
                    float(report["slope"]),
                    float(report["intercept"]),
                )
            )
            if np.asarray(values).size
            else np.empty(0, dtype=np.float64)
            for values in scores
        ]
        return calibrated, {
            "scope": scope,
            "method": method,
            **dict(report),
            "fit_split": "validation_only",
        }

    raw_by_pair: dict[tuple[int, int], list[np.ndarray]] = {}
    labels_by_pair: dict[tuple[int, int], list[np.ndarray]] = {}
    for candidate_set, values in nonempty:
        pair = tuple(int(value) for value in candidate_set.station_pair)
        raw_by_pair.setdefault(pair, []).append(values)
        labels_by_pair.setdefault(pair, []).append(np.asarray(candidate_set.labels, dtype=bool))
    parameters: dict[tuple[int, int], Mapping[str, object]] = {}
    reports: dict[tuple[int, int], Mapping[str, object]] = {}
    for pair in sorted(raw_by_pair):
        _, report = report_function(
            np.concatenate(raw_by_pair[pair]),
            np.concatenate(labels_by_pair[pair]),
            bins=calibration_bins,
        )
        parameters[pair] = dict(report)
        reports[pair] = report
    calibrated = []
    for candidate_set, values in zip(candidate_sets, scores):
        value = np.asarray(values, dtype=np.float64)
        if not value.size:
            calibrated.append(np.empty(0, dtype=np.float64))
            continue
        pair = tuple(int(entry) for entry in candidate_set.station_pair)
        report = parameters[pair]
        calibrated.append(
            apply_temperature(value, float(report["temperature"]))
            if method == "temperature"
            else apply_platt_scaling(value, float(report["slope"]), float(report["intercept"]))
        )
    calibrated_nonempty = [values for values in calibrated if values.size]
    payload: dict[str, object] = {
        "scope": scope,
        "method": method,
        "by_station_pair": {
            f"{left}->{right}": dict(reports[(left, right)])
            for left, right in sorted(reports)
        },
        "before": binary_calibration(raw, labels, bins=calibration_bins),
        "after": binary_calibration(
            np.concatenate(calibrated_nonempty), labels, bins=calibration_bins
        ),
        "fit_split": "validation_only",
    }
    if method == "temperature":
        payload.update(
            {
                "temperature": None,
                "temperature_by_station_pair": {
                    f"{left}->{right}": float(parameters[(left, right)]["temperature"])
                    for left, right in sorted(parameters)
                },
            }
        )
    else:
        payload.update(
            {
                "platt_by_station_pair": {
                    f"{left}->{right}": {
                        "slope": float(parameters[(left, right)]["slope"]),
                        "intercept": float(parameters[(left, right)]["intercept"]),
                    }
                    for left, right in sorted(parameters)
                }
            }
        )
    return calibrated, payload


def apply_frozen_transformer_calibration(
    candidate_sets: Sequence[CandidateSet],
    scores: Sequence[np.ndarray],
    calibration: Mapping[str, object],
) -> list[np.ndarray]:
    """Apply a saved validation-only calibration map to a disjoint split."""
    if len(candidate_sets) != len(scores):
        raise ValueError("candidate sets and scores are not aligned")
    scope = str(calibration.get("scope", "station_pair"))
    method = str(calibration.get("method", "temperature"))
    if method not in {"temperature", "platt"}:
        raise ValueError("unknown frozen Transformer calibration method")
    if scope == "global":
        if method == "temperature":
            temperature = float(calibration["temperature"])
            transform = lambda values: apply_temperature(np.asarray(values, dtype=np.float64), temperature)
        else:
            slope = float(calibration["slope"])
            intercept = float(calibration["intercept"])
            transform = lambda values: apply_platt_scaling(
                np.asarray(values, dtype=np.float64), slope, intercept
            )
        return [
            transform(values)
            if np.asarray(values).size
            else np.empty(0, dtype=np.float64)
            for values in scores
        ]
    if scope != "station_pair":
        raise ValueError("unknown frozen Transformer calibration scope")
    key_name = "temperature_by_station_pair" if method == "temperature" else "platt_by_station_pair"
    raw_parameters = calibration.get(key_name)
    if not isinstance(raw_parameters, Mapping):
        raise ValueError(f"station-pair calibration lacks a {method} parameter map")
    calibrated: list[np.ndarray] = []
    for candidate_set, values in zip(candidate_sets, scores):
        value = np.asarray(values, dtype=np.float64)
        if not value.size:
            calibrated.append(np.empty(0, dtype=np.float64))
            continue
        pair = tuple(int(entry) for entry in candidate_set.station_pair)
        key = f"{pair[0]}->{pair[1]}"
        if key not in raw_parameters:
            raise ValueError(f"frozen Transformer calibration lacks {method} parameters for {key}")
        if method == "temperature":
            calibrated.append(apply_temperature(value, float(raw_parameters[key])))
        else:
            parameters = raw_parameters[key]
            if not isinstance(parameters, Mapping):
                raise ValueError(f"frozen Transformer Platt parameters for {key} are invalid")
            calibrated.append(
                apply_platt_scaling(
                    value, float(parameters["slope"]), float(parameters["intercept"])
                )
            )
    return calibrated


def _weighted_focal_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    hard_negative: torch.Tensor,
    positive_weight: float,
    focal_gamma: float,
    hard_negative_weight: float,
) -> torch.Tensor:
    if logits.shape != labels.shape or labels.shape != hard_negative.shape:
        raise ValueError("loss tensors are not aligned")
    if not logits.numel():
        return logits.sum() * 0.0
    if positive_weight <= 0.0 or focal_gamma < 0.0 or hard_negative_weight <= 0.0:
        raise ValueError("loss weights must be positive and focal_gamma non-negative")
    bce = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    probabilities = torch.sigmoid(logits)
    true_probability = torch.where(labels > 0.5, probabilities, 1.0 - probabilities)
    focal = (1.0 - true_probability).pow(focal_gamma)
    class_weight = torch.where(labels > 0.5, torch.as_tensor(positive_weight, device=logits.device), 1.0)
    hard_weight = torch.where(
        hard_negative,
        torch.as_tensor(hard_negative_weight, device=logits.device),
        1.0,
    )
    weights = class_weight * hard_weight
    return torch.sum(bce * focal * weights) / torch.sum(weights).clamp_min(1.0)


def _stage_graphs(bundle: TransformerGraphBundle, maximum_magnitude_mm: float) -> list[TransformerGraph]:
    selected = [
        graph for graph in bundle.graphs if float(graph.sample.magnitude_mm) <= maximum_magnitude_mm + 1.0e-12
    ]
    if not selected:
        raise ValueError("curriculum stage selects no physical graphs")
    return selected


def _positive_weight(graphs: Sequence[TransformerGraph], maximum: float) -> tuple[float, int, int]:
    labels = np.concatenate([graph.score_labels for graph in graphs])
    positives = int(np.count_nonzero(labels))
    negatives = int(labels.size - positives)
    if not positives or not negatives:
        raise ValueError(f"curriculum stage through {maximum:g} mm lacks both label classes")
    return negatives / positives, positives, negatives


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _validated_curriculum_stages(
    stages: Sequence[CurriculumStage] | None,
    *,
    label: str,
    required: bool,
) -> tuple[CurriculumStage, ...]:
    """Validate one monotonic physical curriculum without inspecting test data."""
    values = tuple(stages or ())
    if required and not values:
        raise ValueError(f"{label} requires at least one curriculum stage")
    previous_limit = -math.inf
    for stage in values:
        if stage.epochs < 1 or not math.isfinite(stage.maximum_magnitude_mm):
            raise ValueError(f"{label} has an invalid epoch count or magnitude")
        if stage.maximum_magnitude_mm < previous_limit:
            raise ValueError(f"{label} magnitudes must be non-decreasing")
        previous_limit = stage.maximum_magnitude_mm
    return values


def _run_curriculum_phase(
    model: GeometryAwareSparseTransformer,
    train_bundle: TransformerGraphBundle,
    validation_bundle: TransformerGraphBundle,
    node_standardizer: FeatureStandardizer,
    edge_standardizer: FeatureStandardizer,
    training_config: TransformerTrainingConfig,
    stages: Sequence[CurriculumStage],
    calibration_bins: int,
    optimizer: torch.optim.Optimizer,
    trainable_parameters: Sequence[nn.Parameter],
    *,
    phase: str,
    local_only: bool,
    global_epoch_start: int,
    keep_initial_model: bool,
) -> tuple[Mapping[str, torch.Tensor], float, int, list[dict[str, object]], int]:
    """Train one validation-early-stopped curriculum phase.

    ``local_only`` is deliberately restricted to the local decoder.  It never
    traverses a message edge and therefore cannot leak multi-station context
    into the pretraining phase.  The later joint phase begins from its best
    checkpoint and may retain that checkpoint if context optimization does not
    improve validation candidate AP.
    """
    if not stages:
        raise ValueError("a curriculum phase requires at least one stage")
    if not trainable_parameters:
        raise ValueError("a curriculum phase requires trainable parameters")
    predictor = predict_local_transformer_scores if local_only else predict_transformer_scores
    forward = _forward_local_batch if local_only else _forward_batch
    device = resolve_device(training_config.device)
    history: list[dict[str, object]] = []
    best_state: Mapping[str, torch.Tensor] | None = None
    best_ap = -math.inf
    best_epoch = -1
    global_epoch = int(global_epoch_start)

    if keep_initial_model:
        validation_scores = predictor(
            model,
            validation_bundle,
            node_standardizer,
            edge_standardizer,
            device=device,
            batch_size=training_config.batch_size,
        )
        validation_metrics = candidate_score_metrics(
            validation_bundle.adjacent_sets, validation_scores, calibration_bins
        )
        validation_ap = validation_metrics.get("average_precision")
        if validation_ap is None:
            raise RuntimeError("initial validation candidate average precision is undefined")
        best_state = copy.deepcopy(model.state_dict())
        best_ap = float(validation_ap)
        best_epoch = global_epoch
        history.append(
            {
                "phase": phase,
                "global_epoch": global_epoch,
                "stage_index": None,
                "stage_name": "initial_local_pretrain_checkpoint",
                "stage_epoch": None,
                "maximum_magnitude_mm": None,
                "train_graphs": None,
                "training_positive_rows": None,
                "training_negative_rows": None,
                "positive_weight_before_cap": None,
                "positive_weight_used": None,
                "training_loss": None,
                "validation_candidate": validation_metrics,
                "validation_average_precision": float(validation_ap),
                "is_best": True,
                "is_initial_checkpoint": True,
            }
        )
        global_epoch += 1

    for stage_index, stage in enumerate(stages):
        selected = _stage_graphs(train_bundle, stage.maximum_magnitude_mm)
        raw_positive_weight, positives, negatives = _positive_weight(selected, stage.maximum_magnitude_mm)
        positive_weight = min(raw_positive_weight, training_config.maximum_positive_weight)
        without_improvement = 0
        for stage_epoch in range(stage.epochs):
            order = np.random.default_rng(training_config.seed + global_epoch).permutation(len(selected))
            model.train()
            weighted_loss = 0.0
            batches = 0
            for offset in range(0, len(order), training_config.batch_size):
                graphs = [selected[int(index)] for index in order[offset : offset + training_config.batch_size]]
                batch = _make_batch(graphs, node_standardizer, edge_standardizer, device)
                optimizer.zero_grad(set_to_none=True)
                logits = forward(model, batch)
                loss = _weighted_focal_bce(
                    logits,
                    batch.score_labels,
                    batch.score_hard_negative,
                    positive_weight,
                    training_config.focal_gamma,
                    training_config.hard_negative_weight,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=5.0)
                optimizer.step()
                weighted_loss += float(loss.detach().cpu())
                batches += 1
            validation_scores = predictor(
                model,
                validation_bundle,
                node_standardizer,
                edge_standardizer,
                device=device,
                batch_size=training_config.batch_size,
            )
            validation_metrics = candidate_score_metrics(
                validation_bundle.adjacent_sets, validation_scores, calibration_bins
            )
            validation_ap = validation_metrics.get("average_precision")
            if validation_ap is None:
                raise RuntimeError("validation candidate average precision is undefined")
            improved = float(validation_ap) > best_ap + 1.0e-12
            if improved:
                best_ap = float(validation_ap)
                best_epoch = global_epoch
                best_state = copy.deepcopy(model.state_dict())
                without_improvement = 0
            else:
                without_improvement += 1
            history.append(
                {
                    "phase": phase,
                    "global_epoch": global_epoch,
                    "stage_index": stage_index,
                    "stage_name": stage.name,
                    "stage_epoch": stage_epoch,
                    "maximum_magnitude_mm": stage.maximum_magnitude_mm,
                    "train_graphs": len(selected),
                    "training_positive_rows": positives,
                    "training_negative_rows": negatives,
                    "positive_weight_before_cap": raw_positive_weight,
                    "positive_weight_used": positive_weight,
                    "training_loss": weighted_loss / max(batches, 1),
                    "validation_candidate": validation_metrics,
                    "validation_average_precision": float(validation_ap),
                    "is_best": improved,
                    "is_initial_checkpoint": False,
                }
            )
            global_epoch += 1
            if without_improvement >= training_config.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("Transformer curriculum phase did not produce a validation checkpoint")
    return best_state, best_ap, best_epoch, history, global_epoch


def train_transformer_v1(
    train_bundle: TransformerGraphBundle,
    validation_bundle: TransformerGraphBundle,
    model_config: SparseTransformerConfig,
    training_config: TransformerTrainingConfig,
    stages: Sequence[CurriculumStage],
    calibration_bins: int,
    local_pretrain_stages: Sequence[CurriculumStage] | None = None,
) -> tuple[GeometryAwareSparseTransformer, TransformerArtifact, list[dict[str, object]]]:
    """Train V1 only on train graphs and stop using validation candidate AP.

    Every epoch visits source-event graphs equally within its physical payload
    stage.  This retains the existing 0/0.1/1/5/10/50 mm curriculum without
    treating high-multiplicity candidate events as more training examples.

    When configured, a local edge-residual head is first trained on exactly
    the same train-split physical curriculum.  It consumes persisted endpoint
    state and the verified physical candidate vector but not graph messages,
    truth features, or a pretrained MLP score.  The sparse-context phase then
    starts from this validation-selected state with a zero-initialized context
    score terminal layer, so multi-station context is a genuine residual.
    """
    if train_bundle.context_mode != validation_bundle.context_mode:
        raise ValueError("train and validation context modes differ")
    if train_bundle.all_station_pairs != validation_bundle.all_station_pairs:
        raise ValueError("train and validation station-pair schemas differ")
    if training_config.batch_size < 1 or training_config.early_stopping_patience < 1:
        raise ValueError("batch_size and early_stopping_patience must be positive")
    if training_config.learning_rate <= 0.0 or training_config.weight_decay < 0.0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if (
        training_config.local_pretrain_learning_rate is not None
        and training_config.local_pretrain_learning_rate <= 0.0
    ):
        raise ValueError("local_pretrain_learning_rate must be positive when supplied")
    if training_config.maximum_positive_weight <= 0.0:
        raise ValueError("maximum_positive_weight must be positive")
    if calibration_bins < 1:
        raise ValueError("calibration_bins must be positive")
    joint_stages = _validated_curriculum_stages(
        stages, label="sparse-context Transformer training", required=True
    )
    local_stages = _validated_curriculum_stages(
        local_pretrain_stages, label="local edge-residual pretraining", required=False
    )
    if local_stages and not model_config.use_local_edge_residual:
        raise ValueError("local edge-residual pretraining requires use_local_edge_residual=True")

    _seed_everything(training_config.seed)
    device = resolve_device(training_config.device)
    node_standardizer, edge_standardizer = fit_graph_standardizers(train_bundle)
    model = GeometryAwareSparseTransformer(model_config).to(device)
    history: list[dict[str, object]] = []
    global_epoch = 0
    local_summary: dict[str, object] = {
        "enabled": bool(local_stages),
        "stages": [asdict(stage) for stage in local_stages],
    }
    if local_stages:
        local_parameters = tuple(
            parameter
            for module in (model.local_edge_score, model.score_pair_embedding)
            for parameter in module.parameters()
        )
        local_optimizer = torch.optim.AdamW(
            local_parameters,
            lr=(
                training_config.learning_rate
                if training_config.local_pretrain_learning_rate is None
                else training_config.local_pretrain_learning_rate
            ),
            weight_decay=training_config.weight_decay,
        )
        local_state, local_ap, local_epoch, local_history, global_epoch = _run_curriculum_phase(
            model,
            train_bundle,
            validation_bundle,
            node_standardizer,
            edge_standardizer,
            training_config,
            local_stages,
            calibration_bins,
            local_optimizer,
            local_parameters,
            phase="local_edge_pretrain",
            local_only=True,
            global_epoch_start=global_epoch,
            keep_initial_model=False,
        )
        model.load_state_dict(local_state)
        history.extend(local_history)
        local_summary.update(
            {
                "best_validation_average_precision": local_ap,
                "best_global_epoch": local_epoch,
                "trainable_modules": ["local_edge_score", "score_pair_embedding"],
                "uses_message_edges": False,
                "learning_rate": (
                    training_config.learning_rate
                    if training_config.local_pretrain_learning_rate is None
                    else training_config.local_pretrain_learning_rate
                ),
            }
        )

    joint_parameters = tuple(model.parameters())
    joint_optimizer = torch.optim.AdamW(
        joint_parameters,
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    best_state, best_ap, best_epoch, joint_history, global_epoch = _run_curriculum_phase(
        model,
        train_bundle,
        validation_bundle,
        node_standardizer,
        edge_standardizer,
        training_config,
        joint_stages,
        calibration_bins,
        joint_optimizer,
        joint_parameters,
        phase="sparse_context_joint",
        local_only=False,
        global_epoch_start=global_epoch,
        keep_initial_model=bool(local_stages),
    )
    history.extend(joint_history)
    model.load_state_dict(best_state)
    artifact = TransformerArtifact(
        node_feature_names=NODE_FEATURE_NAMES,
        edge_feature_names=EDGE_FEATURE_NAMES,
        all_station_pairs=train_bundle.all_station_pairs,
        output_station_pairs=ADJACENT_STATION_PAIRS,
        node_standardizer=node_standardizer,
        edge_standardizer=edge_standardizer,
        model_config=model_config,
        context_mode=train_bundle.context_mode,
        training_summary={
            "best_validation_average_precision": best_ap,
            "best_global_epoch": best_epoch,
            "training_global_epochs_completed": global_epoch,
            "device": str(device),
            "train_bundle": graph_bundle_summary(train_bundle),
            "validation_bundle": graph_bundle_summary(validation_bundle),
            "loss": {
                "type": "weighted_focal_binary_cross_entropy",
                "focal_gamma": training_config.focal_gamma,
                "hard_negative_weight": training_config.hard_negative_weight,
                "maximum_positive_weight": training_config.maximum_positive_weight,
                "hard_negative_is_model_feature": False,
            },
            "sampling_unit": "physical_event_graph_equal_weight",
            "local_edge_pretraining": local_summary,
            "sparse_context_phase": {
                "stages": [asdict(stage) for stage in joint_stages],
                "initial_checkpoint_was_local_pretrain": bool(local_stages),
                "context_score_terminal_zero_initialized": bool(model_config.use_local_edge_residual),
            },
        },
    )
    return model, artifact, history


def save_transformer_artifact(
    path: str | Path, model: GeometryAwareSparseTransformer, artifact: TransformerArtifact
) -> None:
    """Save model weights and the exact train-only feature-normalization contract."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "faser-geometry-aware-transformer-v1",
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


def load_transformer_artifact(
    path: str | Path, device: str = "auto"
) -> tuple[GeometryAwareSparseTransformer, TransformerArtifact]:
    """Load a V1 artifact while retaining its saved architecture and scalers."""
    source = Path(path).expanduser().resolve()
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with older LCG PyTorch builds.
        payload = torch.load(source, map_location="cpu")
    if payload.get("schema_version") != "faser-geometry-aware-transformer-v1":
        raise ValueError("unexpected Transformer checkpoint schema")
    model_config = model_config_from_payload(payload["model_config"])
    model = GeometryAwareSparseTransformer(model_config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(resolve_device(device))
    artifact = TransformerArtifact(
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
        raise ValueError("Transformer checkpoint feature schema is not the V1 physical schema")
    if artifact.all_station_pairs != ALL_STATION_PAIRS or artifact.output_station_pairs != ADJACENT_STATION_PAIRS:
        raise ValueError("Transformer checkpoint station-pair schema is incompatible with V1 routing")
    return model, artifact


def source_disjoint_audit(samples: Sequence[CurriculumSample]) -> dict[str, object]:
    """Verify source-file and original-event separation without loading ROOT data."""
    source_to_split: dict[str, str] = {}
    sources_by_split: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    uids_by_split: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    for sample in samples:
        if sample.split not in sources_by_split:
            raise ValueError(f"unexpected split '{sample.split}'")
        for source_id in sample.source_ids:
            previous = source_to_split.setdefault(source_id, sample.split)
            if previous != sample.split:
                raise ValueError(f"source '{source_id}' crosses data splits")
            sources_by_split[sample.split].add(source_id)
        uids_by_split[sample.split].update(sample.source_event_uids)
    overlaps = {
        f"{left}_{right}": sorted(uids_by_split[left].intersection(uids_by_split[right]))
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    if any(overlaps.values()):
        raise ValueError("original source event leakage across train/validation/test")
    return {
        "split_unit": "original_xAOD_file",
        "sources_by_split": {split: sorted(values) for split, values in sources_by_split.items()},
        "source_event_counts_by_split": {split: len(values) for split, values in uids_by_split.items()},
        "source_event_uid_overlap": overlaps,
        "strictly_disjoint": True,
    }


def training_config_from_mapping(payload: Mapping[str, object]) -> TransformerTrainingConfig:
    """Parse declared YAML controls while rejecting undeclared training fields."""
    return TransformerTrainingConfig(**dict(payload))


def stages_from_payload(values: Sequence[Mapping[str, object]]) -> tuple[CurriculumStage, ...]:
    """Parse the existing physical curriculum stage declaration."""
    return tuple(CurriculumStage(**dict(value)) for value in values)


def artifact_summary(artifact: TransformerArtifact) -> dict[str, object]:
    """JSON-safe checkpoint summary for frozen validation/test contracts."""
    return {
        "schema_version": "faser-geometry-aware-transformer-v1",
        "node_feature_names": list(artifact.node_feature_names),
        "edge_feature_names": list(artifact.edge_feature_names),
        "all_station_pairs": [f"{left}->{right}" for left, right in artifact.all_station_pairs],
        "output_station_pairs": [f"{left}->{right}" for left, right in artifact.output_station_pairs],
        "model_config": asdict(artifact.model_config),
        "context_mode": artifact.context_mode,
        "training_summary": dict(artifact.training_summary),
    }
