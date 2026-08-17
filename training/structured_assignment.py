"""V3 structured learning for the existing physical four-station route graph.

The V3 objective deliberately keeps the V2 route encoder and the exact
unit-capacity route solver.  It changes only supervision: rather than fitting
independent edge or route probabilities, every physical event is compared with
its strongest loss-augmented feasible route assignment.  No truth label,
synthetic role, or candidate created here is exposed as a model feature.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from baselines.mlp_pair_classifier import FeatureStandardizer
from baselines.route_assignment import solve_unit_capacity_route_packing
from models.route_transformer import RouteAwareSparseTransformer, RouteAwareTransformerConfig
from training.geometry_aware_transformer import (
    ADJACENT_STATION_PAIRS,
    ALL_STATION_PAIRS,
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    CurriculumStage,
    TransformerGraph,
    TransformerGraphBundle,
    _seed_everything,
    fit_graph_standardizers,
    graph_bundle_summary,
    resolve_device,
)
from training.route_aware_transformer import (
    RouteCandidateTable,
    _forward_route_batch,
    _make_route_batch,
    materialize_route_candidate_tables,
    route_candidate_table_summary,
)


@dataclass(frozen=True)
class StructuredAssignmentTrainingConfig:
    """Optimization controls for V3; the four-block encoder is unchanged."""

    batch_size: int = 4
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    seed: int = 20260813
    device: str = "auto"
    early_stopping_patience: int = 5
    margin: float = 1.0
    mixed_truth_penalty: float = 1.0
    fake_endpoint_penalty: float = 1.5
    hard_negative_penalty: float = 0.25
    duplicate_truth_penalty: float = 1.0
    endpoint_conflict_penalty: float = 0.50
    soft_assignment_weight: float = 0.0
    soft_assignment_temperature: float = 1.0
    soft_assignment_iterations: int = 8
    soft_assignment_dual_step: float = 0.50
    soft_assignment_capacity_weight: float = 0.25


@dataclass(frozen=True)
class StructuredAssignmentTarget:
    """Truth-only target and competing-route audit masks for one event.

    ``truth_assignment`` is an exact maximum-cardinality unit-capacity subset
    of truth-consistent complete routes.  It is deliberately distinct from
    the raw route label: duplicate truth routes that conflict at an endpoint
    are tracked as competing alternatives rather than silently treated as
    independent positives.
    """

    truth_assignment: np.ndarray
    mixed_truth: np.ndarray
    fake_endpoint: np.ndarray
    hard_negative: np.ndarray
    duplicate_truth: np.ndarray
    endpoint_conflict: np.ndarray

    @property
    def size(self) -> int:
        return int(self.truth_assignment.size)

    @property
    def truth_route_count(self) -> int:
        return int(np.count_nonzero(self.truth_assignment))


@dataclass(frozen=True)
class StructuredAssignmentArtifact:
    """V3 checkpoint contract with train-derived normalizers only."""

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
class StructuredRouteScoreSet:
    """Uncalibrated V3 complete-route utilities for one physical graph."""

    sample: object
    event: object
    endpoint_indices: np.ndarray
    utilities: np.ndarray
    labels: np.ndarray
    truth_assignment: np.ndarray
    fake_endpoint: np.ndarray
    hard_negative: np.ndarray

    @property
    def size(self) -> int:
        return int(self.utilities.size)


@dataclass(frozen=True)
class StructuredAssignmentPrediction:
    """Raw edge probabilities and structured complete-route utilities."""

    edge_scores: tuple[np.ndarray, ...]
    base_edge_scores: tuple[np.ndarray, ...]
    route_utilities: np.ndarray
    route_labels: np.ndarray
    route_truth_assignment: np.ndarray
    route_fake_endpoint: np.ndarray
    route_hard_negative: np.ndarray
    route_score_sets: tuple[StructuredRouteScoreSet, ...]


def _endpoint_rows(table: RouteCandidateTable) -> tuple[tuple[int, int, int, int], ...]:
    nodes = np.asarray(table.node_indices, dtype=np.int64)
    if nodes.ndim != 2 or nodes.shape != (table.size, 4):
        raise ValueError("structured route table has invalid endpoint shape")
    return tuple(tuple(int(value) for value in row) for row in nodes)


def build_structured_assignment_target(table: RouteCandidateTable) -> StructuredAssignmentTarget:
    """Construct the event-level truth assignment and competitor categories."""
    labels = np.asarray(table.labels, dtype=bool)
    fake = np.asarray(table.fake_endpoint, dtype=bool)
    hard = np.asarray(table.hard_negative, dtype=bool)
    if labels.shape != (table.size,) or fake.shape != labels.shape or hard.shape != labels.shape:
        raise ValueError("structured route table supervision arrays are misaligned")
    if np.any(labels & fake):
        raise ValueError("a truth-consistent route cannot contain a fake endpoint")
    rows = _endpoint_rows(table)
    truth_assignment = solve_unit_capacity_route_packing(rows, labels.astype(np.float64)).selected
    selected_endpoints = {
        endpoint
        for selected, row in zip(truth_assignment, rows)
        if bool(selected)
        for endpoint in row
    }
    endpoint_conflict = np.asarray(
        [
            (not bool(selected)) and bool(set(row) & selected_endpoints)
            for selected, row in zip(truth_assignment, rows)
        ],
        dtype=bool,
    )
    duplicate_truth = labels & ~truth_assignment
    # Every negative route is either fake-ended or a mixed-truth route.  The
    # masks are exclusive here; hard-negative and endpoint-conflict flags are
    # additive severity tags used only in the loss augmentation.
    mixed_truth = ~labels & ~fake
    if np.any(mixed_truth & fake) or np.any(duplicate_truth & ~labels):  # pragma: no cover
        raise RuntimeError("structured route category partition is inconsistent")
    return StructuredAssignmentTarget(
        truth_assignment=np.asarray(truth_assignment, dtype=bool),
        mixed_truth=np.asarray(mixed_truth, dtype=bool),
        fake_endpoint=fake,
        hard_negative=hard,
        duplicate_truth=np.asarray(duplicate_truth, dtype=bool),
        endpoint_conflict=endpoint_conflict,
    )


def materialize_structured_assignment_targets(
    graphs: Sequence[TransformerGraph],
    route_tables: Mapping[int, RouteCandidateTable] | None = None,
) -> dict[int, StructuredAssignmentTarget]:
    """Cache truth-only V3 targets for exactly the caller's graph sequence."""
    tables = materialize_route_candidate_tables(graphs) if route_tables is None else route_tables
    expected = {id(graph) for graph in graphs}
    if set(tables) != expected:
        raise ValueError("structured target route tables do not match the supplied graph sequence")
    return {id(graph): build_structured_assignment_target(tables[id(graph)]) for graph in graphs}


def structured_target_summary(targets: Sequence[StructuredAssignmentTarget]) -> dict[str, int]:
    """Truth-only category accounting for reproducible V3 loss audits."""
    result = {
        "graphs": 0,
        "complete_route_candidates": 0,
        "truth_assignment_routes": 0,
        "mixed_truth_routes": 0,
        "fake_endpoint_routes": 0,
        "hard_negative_routes": 0,
        "duplicate_truth_routes": 0,
        "endpoint_conflict_routes": 0,
    }
    for target in targets:
        result["graphs"] += 1
        result["complete_route_candidates"] += target.size
        result["truth_assignment_routes"] += int(np.count_nonzero(target.truth_assignment))
        result["mixed_truth_routes"] += int(np.count_nonzero(target.mixed_truth))
        result["fake_endpoint_routes"] += int(np.count_nonzero(target.fake_endpoint))
        result["hard_negative_routes"] += int(np.count_nonzero(target.hard_negative))
        result["duplicate_truth_routes"] += int(np.count_nonzero(target.duplicate_truth))
        result["endpoint_conflict_routes"] += int(np.count_nonzero(target.endpoint_conflict))
    return result


def route_assignment_utilities(output: object, batch: object) -> torch.Tensor:
    """Return the complete-route utility used by V3's final route comparison.

    A route retains the three adjacent physical edge utilities and receives a
    learned route-context residual.  Thus the structured training utility is
    compatible with the existing route solver's complete-route replacement
    interface, while partial missing-station routes still use edge utilities.
    """
    edge_indices = getattr(batch, "route_score_edge_indices")
    route_logits = getattr(output, "route_logits")
    edge_logits = getattr(output, "edge_logits")
    if edge_indices.ndim != 2 or edge_indices.shape[1] != 3:
        raise ValueError("V3 route score-edge indices must have shape [routes, 3]")
    if route_logits.ndim != 1 or route_logits.shape[0] != edge_indices.shape[0]:
        raise ValueError("V3 route logits are not aligned with route candidates")
    if not route_logits.numel():
        return route_logits
    return edge_logits[edge_indices].sum(dim=1) + route_logits


def _loss_augmentation(
    target: StructuredAssignmentTarget,
    config: StructuredAssignmentTrainingConfig,
) -> np.ndarray:
    """Decomposable loss term for exact loss-augmented route inference."""
    values = np.zeros(target.size, dtype=np.float64)
    # Selecting a target truth route removes one miss from the Hamming-like
    # assignment loss.  The constant number of target routes is added later.
    values[target.truth_assignment] -= 1.0
    values[target.mixed_truth] += float(config.mixed_truth_penalty)
    values[target.fake_endpoint] += float(config.fake_endpoint_penalty)
    values[target.duplicate_truth] += float(config.duplicate_truth_penalty)
    # These two flags intentionally add severity to fake/mixed/duplicate
    # routes.  They make the strongest competitor explicitly cover hard and
    # endpoint-conflicting alternatives without changing the candidate graph.
    values[target.hard_negative & ~target.truth_assignment] += float(
        config.hard_negative_penalty
    )
    values[target.endpoint_conflict & ~target.truth_assignment] += float(
        config.endpoint_conflict_penalty
    )
    return values


def structured_margin_loss(
    route_utilities: torch.Tensor,
    route_tables: Sequence[RouteCandidateTable],
    targets: Sequence[StructuredAssignmentTarget],
    config: StructuredAssignmentTrainingConfig,
) -> dict[str, object]:
    """Exact event-level structured hinge against the strongest competitor.

    For each physical event ``Y`` is the truth global unit-capacity assignment
    and ``Y_hat`` maximizes ``U(theta, A) + Delta(A, Y)`` over the same route
    set.  Gradients flow through the selected route utilities only; MILP is
    correctly treated as the discrete loss-augmented inference oracle.
    """
    if route_utilities.ndim != 1:
        raise ValueError("structured route utilities must be a vector")
    if len(route_tables) != len(targets):
        raise ValueError("structured route tables and targets are misaligned")
    expected = sum(table.size for table in route_tables)
    if int(route_utilities.numel()) != expected:
        raise ValueError("structured route utility vector has an unexpected graph tail")
    losses: list[torch.Tensor] = []
    raw_margins: list[float] = []
    violations = 0
    competitor_selected = 0
    competitor_truth = 0
    competitor_mixed = 0
    competitor_fake = 0
    competitor_hard = 0
    competitor_duplicate = 0
    competitor_conflict = 0
    offset = 0
    for table, target in zip(route_tables, targets):
        if table.size != target.size:
            raise ValueError("structured target does not align with its route table")
        stop = offset + table.size
        local_utilities = route_utilities[offset:stop]
        offset = stop
        if not table.size:
            losses.append(route_utilities.sum() * 0.0)
            raw_margins.append(0.0)
            continue
        augmentation = _loss_augmentation(target, config)
        detached = local_utilities.detach().to("cpu", dtype=torch.float64).numpy()
        competitor = solve_unit_capacity_route_packing(
            _endpoint_rows(table), detached + float(config.margin) * augmentation
        ).selected
        target_mask = torch.as_tensor(target.truth_assignment, dtype=local_utilities.dtype, device=local_utilities.device)
        competitor_mask = torch.as_tensor(competitor, dtype=local_utilities.dtype, device=local_utilities.device)
        target_utility = torch.sum(local_utilities * target_mask)
        competitor_utility = torch.sum(local_utilities * competitor_mask)
        loss_delta = float(config.margin) * (
            float(target.truth_route_count) + float(np.dot(augmentation, competitor.astype(np.float64)))
        )
        raw_margin = competitor_utility + local_utilities.new_tensor(loss_delta) - target_utility
        hinge = torch.relu(raw_margin)
        losses.append(hinge)
        raw_margin_value = float(raw_margin.detach().cpu())
        raw_margins.append(raw_margin_value)
        violations += int(raw_margin_value > 0.0)
        competitor_selected += int(np.count_nonzero(competitor))
        competitor_truth += int(np.count_nonzero(competitor & target.truth_assignment))
        competitor_mixed += int(np.count_nonzero(competitor & target.mixed_truth))
        competitor_fake += int(np.count_nonzero(competitor & target.fake_endpoint))
        competitor_hard += int(np.count_nonzero(competitor & target.hard_negative))
        competitor_duplicate += int(np.count_nonzero(competitor & target.duplicate_truth))
        competitor_conflict += int(np.count_nonzero(competitor & target.endpoint_conflict))
    if offset != int(route_utilities.numel()):  # pragma: no cover - sum invariant above
        raise RuntimeError("structured route utility offset did not consume the batch")
    total = torch.stack(losses).mean() if losses else route_utilities.sum() * 0.0
    events = len(route_tables)
    return {
        "loss": total,
        "events": events,
        "mean_raw_margin": 0.0 if not raw_margins else float(np.mean(raw_margins)),
        "violation_fraction": 0.0 if not events else float(violations / events),
        "competitor_selected_routes": competitor_selected,
        "competitor_truth_routes": competitor_truth,
        "competitor_mixed_truth_routes": competitor_mixed,
        "competitor_fake_endpoint_routes": competitor_fake,
        "competitor_hard_negative_routes": competitor_hard,
        "competitor_duplicate_truth_routes": competitor_duplicate,
        "competitor_endpoint_conflict_routes": competitor_conflict,
    }


def soft_unit_capacity_assignment_surrogate(
    route_utilities: torch.Tensor,
    route_tables: Sequence[RouteCandidateTable],
    targets: Sequence[StructuredAssignmentTarget],
    config: StructuredAssignmentTrainingConfig,
) -> torch.Tensor:
    """A differentiable capacity-normalized soft-assignment control.

    Routes are four-endpoint hyperedges, so ordinary matrix Sinkhorn is not an
    exact relaxation.  This control instead performs differentiable dual
    endpoint-pressure updates and penalizes capacity excess.  It is retained
    solely as an explicitly labelled soft-assignment comparison; V3's primary
    objective remains the exact structured hinge above.
    """
    if route_utilities.ndim != 1:
        raise ValueError("soft-assignment route utilities must be a vector")
    if len(route_tables) != len(targets):
        raise ValueError("soft-assignment route tables and targets are misaligned")
    if int(route_utilities.numel()) != sum(table.size for table in route_tables):
        raise ValueError("soft-assignment route utility vector is misaligned")
    losses: list[torch.Tensor] = []
    offset = 0
    for table, target in zip(route_tables, targets):
        stop = offset + table.size
        local_utilities = route_utilities[offset:stop]
        offset = stop
        if not table.size:
            losses.append(route_utilities.sum() * 0.0)
            continue
        nodes = torch.as_tensor(table.node_indices, dtype=torch.long, device=local_utilities.device)
        node_count = int(nodes.max().item()) + 1
        dual = local_utilities.new_zeros((node_count,))
        scaled = local_utilities / float(config.soft_assignment_temperature)
        probability = torch.sigmoid(scaled)
        loads = local_utilities.new_zeros((node_count,))
        for _ in range(int(config.soft_assignment_iterations)):
            probability = torch.sigmoid(scaled - dual[nodes].sum(dim=1))
            loads = local_utilities.new_zeros((node_count,))
            loads.scatter_add_(0, nodes.reshape(-1), probability.repeat_interleave(4))
            dual = dual + float(config.soft_assignment_dual_step) * torch.relu(loads - 1.0)
        truth = torch.as_tensor(
            target.truth_assignment, dtype=local_utilities.dtype, device=local_utilities.device
        )
        bce = nn.functional.binary_cross_entropy(
            probability.clamp(1.0e-6, 1.0 - 1.0e-6), truth
        )
        capacity = torch.mean(torch.relu(loads - 1.0).square())
        losses.append(bce + float(config.soft_assignment_capacity_weight) * capacity)
    return torch.stack(losses).mean() if losses else route_utilities.sum() * 0.0


def _optional_soft_assignment_loss(
    route_utilities: torch.Tensor,
    route_tables: Sequence[RouteCandidateTable],
    targets: Sequence[StructuredAssignmentTarget],
    config: StructuredAssignmentTrainingConfig,
) -> torch.Tensor:
    """Evaluate the labelled soft control only when it contributes to loss."""
    if float(config.soft_assignment_weight) > 0.0:
        return soft_unit_capacity_assignment_surrogate(route_utilities, route_tables, targets, config)
    return route_utilities.sum() * 0.0


def _validate_training_config(config: StructuredAssignmentTrainingConfig) -> None:
    if config.batch_size < 1 or config.early_stopping_patience < 1:
        raise ValueError("V3 batch_size and early_stopping_patience must be positive")
    if config.learning_rate <= 0.0 or config.weight_decay < 0.0:
        raise ValueError("V3 learning_rate must be positive and weight_decay non-negative")
    if config.margin <= 0.0 or not math.isfinite(config.margin):
        raise ValueError("V3 margin must be finite and positive")
    for name in (
        "mixed_truth_penalty",
        "fake_endpoint_penalty",
        "hard_negative_penalty",
        "duplicate_truth_penalty",
        "endpoint_conflict_penalty",
        "soft_assignment_weight",
        "soft_assignment_capacity_weight",
    ):
        value = float(getattr(config, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"V3 {name} must be finite and non-negative")
    if config.soft_assignment_temperature <= 0.0 or not math.isfinite(config.soft_assignment_temperature):
        raise ValueError("V3 soft_assignment_temperature must be finite and positive")
    if config.soft_assignment_iterations < 1 or config.soft_assignment_dual_step <= 0.0:
        raise ValueError("V3 soft-assignment iterations and dual step must be positive")


def _validate_stages(stages: Sequence[CurriculumStage]) -> tuple[CurriculumStage, ...]:
    result = tuple(stages)
    if not result:
        raise ValueError("V3 requires at least one physical curriculum stage")
    previous = -math.inf
    for stage in result:
        if stage.epochs < 1 or not math.isfinite(stage.maximum_magnitude_mm):
            raise ValueError("V3 curriculum stage is invalid")
        if stage.maximum_magnitude_mm < previous:
            raise ValueError("V3 curriculum maximum magnitudes must be non-decreasing")
        previous = stage.maximum_magnitude_mm
    return result


def _stage_graphs(bundle: TransformerGraphBundle, maximum_magnitude_mm: float) -> list[TransformerGraph]:
    graphs = [
        graph
        for graph in bundle.graphs
        if float(graph.sample.magnitude_mm) <= maximum_magnitude_mm + 1.0e-12
    ]
    if not graphs:
        raise ValueError("V3 curriculum stage selects no physical graphs")
    return graphs


def _loss_batch(
    model: RouteAwareSparseTransformer,
    graphs: Sequence[TransformerGraph],
    route_tables: Mapping[int, RouteCandidateTable],
    targets: Mapping[int, StructuredAssignmentTarget],
    node_standardizer: FeatureStandardizer,
    edge_standardizer: FeatureStandardizer,
    device: torch.device,
    config: StructuredAssignmentTrainingConfig,
) -> dict[str, object]:
    batch = _make_route_batch(
        graphs, node_standardizer, edge_standardizer, device, route_tables
    )
    output = _forward_route_batch(model, batch)
    utilities = route_assignment_utilities(output, batch)
    tables = [route_tables[id(graph)] for graph in graphs]
    local_targets = [targets[id(graph)] for graph in graphs]
    structured = structured_margin_loss(utilities, tables, local_targets, config)
    # The primary V3 run has an exact structured-margin objective only.  Do
    # not materialize the optional differentiable relaxation when its weight
    # is zero: it adds a full endpoint-dual iteration without any gradient or
    # objective contribution.
    soft = _optional_soft_assignment_loss(utilities, tables, local_targets, config)
    total = structured["loss"] + float(config.soft_assignment_weight) * soft
    return {**structured, "soft_assignment_loss": soft, "total": total, "utilities": utilities}


def train_structured_assignment_v3(
    train_bundle: TransformerGraphBundle,
    validation_bundle: TransformerGraphBundle,
    model_config: RouteAwareTransformerConfig,
    training_config: StructuredAssignmentTrainingConfig,
    stages: Sequence[CurriculumStage],
) -> tuple[RouteAwareSparseTransformer, StructuredAssignmentArtifact, list[dict[str, object]]]:
    """Train V3 on train sources and select checkpoints only on validation."""
    if train_bundle.context_mode != "full_event" or validation_bundle.context_mode != "full_event":
        raise ValueError("V3 requires full-event train and validation graph bundles")
    if train_bundle.all_station_pairs != validation_bundle.all_station_pairs:
        raise ValueError("V3 train and validation candidate schemas differ")
    if tuple(train_bundle.station_path) != (0, 1, 2, 3):
        raise ValueError("V3 is fixed to the IFT -> S1 -> S2 -> S3 station path")
    _validate_training_config(training_config)
    validated_stages = _validate_stages(stages)
    _seed_everything(training_config.seed)
    device = resolve_device(training_config.device)
    node_standardizer, edge_standardizer = fit_graph_standardizers(train_bundle)
    model = RouteAwareSparseTransformer(model_config).to(device)
    train_tables = materialize_route_candidate_tables(train_bundle.graphs)
    validation_tables = materialize_route_candidate_tables(validation_bundle.graphs)
    train_targets = materialize_structured_assignment_targets(train_bundle.graphs, train_tables)
    validation_targets = materialize_structured_assignment_targets(
        validation_bundle.graphs, validation_tables
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay
    )
    best_state: Mapping[str, torch.Tensor] | None = None
    best_rank = (math.inf, math.inf)
    best_epoch = -1
    global_epoch = 0
    history: list[dict[str, object]] = []

    for stage_index, stage in enumerate(validated_stages):
        graphs = _stage_graphs(train_bundle, stage.maximum_magnitude_mm)
        without_improvement = 0
        for stage_epoch in range(stage.epochs):
            order = np.random.default_rng(training_config.seed + global_epoch).permutation(len(graphs))
            totals = {
                "loss": 0.0,
                "structured": 0.0,
                "soft": 0.0,
                "raw_margin": 0.0,
                "violations": 0.0,
                "events": 0,
                "competitor_selected": 0,
                "competitor_mixed": 0,
                "competitor_fake": 0,
                "competitor_duplicate": 0,
                "competitor_conflict": 0,
            }
            batches = 0
            model.train()
            for offset in range(0, len(order), training_config.batch_size):
                batch_graphs = [
                    graphs[int(index)] for index in order[offset : offset + training_config.batch_size]
                ]
                optimizer.zero_grad(set_to_none=True)
                losses = _loss_batch(
                    model,
                    batch_graphs,
                    train_tables,
                    train_targets,
                    node_standardizer,
                    edge_standardizer,
                    device,
                    training_config,
                )
                total = losses["total"]
                if not isinstance(total, torch.Tensor):  # pragma: no cover - typed construction
                    raise RuntimeError("V3 loss is not a tensor")
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                totals["loss"] += float(total.detach().cpu())
                structured_loss = losses["loss"]
                soft_loss = losses["soft_assignment_loss"]
                if not isinstance(structured_loss, torch.Tensor) or not isinstance(soft_loss, torch.Tensor):
                    raise RuntimeError("V3 component loss is not a tensor")
                totals["structured"] += float(structured_loss.detach().cpu())
                totals["soft"] += float(soft_loss.detach().cpu())
                totals["raw_margin"] += float(losses["mean_raw_margin"])
                totals["violations"] += float(losses["violation_fraction"]) * int(losses["events"])
                totals["events"] += int(losses["events"])
                totals["competitor_selected"] += int(losses["competitor_selected_routes"])
                totals["competitor_mixed"] += int(losses["competitor_mixed_truth_routes"])
                totals["competitor_fake"] += int(losses["competitor_fake_endpoint_routes"])
                totals["competitor_duplicate"] += int(losses["competitor_duplicate_truth_routes"])
                totals["competitor_conflict"] += int(losses["competitor_endpoint_conflict_routes"])
                batches += 1

            model.eval()
            validation_rows: list[dict[str, object]] = []
            with torch.no_grad():
                for offset in range(0, len(validation_bundle.graphs), training_config.batch_size):
                    validation_rows.append(
                        _loss_batch(
                            model,
                            validation_bundle.graphs[offset : offset + training_config.batch_size],
                            validation_tables,
                            validation_targets,
                            node_standardizer,
                            edge_standardizer,
                            device,
                            training_config,
                        )
                    )
            validation_events = sum(int(row["events"]) for row in validation_rows)
            validation_hinge = float(
                np.mean([float(row["loss"].detach().cpu()) for row in validation_rows])
            )
            validation_violation = (
                0.0
                if not validation_events
                else float(
                    sum(float(row["violation_fraction"]) * int(row["events"]) for row in validation_rows)
                    / validation_events
                )
            )
            # The rank is predeclared and independent of calibration,
            # threshold grids, or final route-assignment metrics.
            rank = (validation_hinge, validation_violation)
            improved = rank < best_rank
            if improved:
                best_rank = rank
                best_epoch = global_epoch
                best_state = copy.deepcopy(model.state_dict())
                without_improvement = 0
            else:
                without_improvement += 1
            history.append(
                {
                    "phase": "structured_assignment_margin",
                    "global_epoch": global_epoch,
                    "stage_index": stage_index,
                    "stage_name": stage.name,
                    "stage_epoch": stage_epoch,
                    "maximum_magnitude_mm": stage.maximum_magnitude_mm,
                    "train_graphs": len(graphs),
                    "training_loss": totals["loss"] / max(batches, 1),
                    "training_structured_hinge": totals["structured"] / max(batches, 1),
                    "training_soft_assignment": totals["soft"] / max(batches, 1),
                    "training_mean_raw_margin": totals["raw_margin"] / max(batches, 1),
                    "training_violation_fraction": (
                        0.0 if not totals["events"] else totals["violations"] / totals["events"]
                    ),
                    "training_competitor_selected_routes": totals["competitor_selected"],
                    "training_competitor_mixed_truth_routes": totals["competitor_mixed"],
                    "training_competitor_fake_endpoint_routes": totals["competitor_fake"],
                    "training_competitor_duplicate_truth_routes": totals["competitor_duplicate"],
                    "training_competitor_endpoint_conflict_routes": totals["competitor_conflict"],
                    "validation_structured_hinge": validation_hinge,
                    "validation_violation_fraction": validation_violation,
                    "validation_rank": list(rank),
                    "is_best": improved,
                }
            )
            global_epoch += 1
            if without_improvement >= training_config.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("V3 did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    artifact = StructuredAssignmentArtifact(
        node_feature_names=NODE_FEATURE_NAMES,
        edge_feature_names=EDGE_FEATURE_NAMES,
        all_station_pairs=train_bundle.all_station_pairs,
        output_station_pairs=ADJACENT_STATION_PAIRS,
        node_standardizer=node_standardizer,
        edge_standardizer=edge_standardizer,
        model_config=model_config,
        context_mode="full_event",
        training_summary={
            "best_validation_structured_hinge": best_rank[0],
            "best_validation_violation_fraction": best_rank[1],
            "best_global_epoch": best_epoch,
            "training_global_epochs_completed": global_epoch,
            "device": str(device),
            "train_bundle": graph_bundle_summary(train_bundle),
            "validation_bundle": graph_bundle_summary(validation_bundle),
            "train_route_candidates": route_candidate_table_summary(train_tables.values()),
            "validation_route_candidates": route_candidate_table_summary(validation_tables.values()),
            "train_structured_targets": structured_target_summary(list(train_targets.values())),
            "validation_structured_targets": structured_target_summary(list(validation_targets.values())),
            "loss": {
                "primary": "exact_loss_augmented_unit_capacity_structured_hinge",
                "truth_assignment": "maximum_cardinality_truth_consistent_unit_capacity_routes",
                "strongest_competitor": "exact_unit_capacity_milp_on_existing_complete_physical_routes",
                "mixed_truth_penalty": training_config.mixed_truth_penalty,
                "fake_endpoint_penalty": training_config.fake_endpoint_penalty,
                "hard_negative_penalty": training_config.hard_negative_penalty,
                "duplicate_truth_penalty": training_config.duplicate_truth_penalty,
                "endpoint_conflict_penalty": training_config.endpoint_conflict_penalty,
                "margin": training_config.margin,
                "soft_assignment_control": "capacity_normalized_hyperedge_surrogate",
                "soft_assignment_weight": training_config.soft_assignment_weight,
                "independent_edge_bce_used": False,
                "independent_route_bce_used": False,
                "truth_or_synthetic_provenance_is_model_feature": False,
            },
            "early_stopping": {
                "selection_split": "validation_only",
                "rank": "minimum_structured_hinge_then_minimum_violation_fraction",
                "route_solver_used": "loss_augmented_training_oracle_only",
                "calibration_used": False,
                "threshold_used": False,
            },
        },
    )
    return model, artifact, history


def predict_structured_assignment_scores(
    model: RouteAwareSparseTransformer,
    bundle: TransformerGraphBundle,
    node_standardizer: FeatureStandardizer,
    edge_standardizer: FeatureStandardizer,
    *,
    device: str | torch.device = "auto",
    batch_size: int = 4,
    route_tables: Mapping[int, RouteCandidateTable] | None = None,
    targets: Mapping[int, StructuredAssignmentTarget] | None = None,
) -> StructuredAssignmentPrediction:
    """Reconstruct raw V3 route utilities without any calibration or threshold."""
    if bundle.context_mode != "full_event":
        raise ValueError("V3 prediction requires full-event physical graph context")
    if batch_size < 1:
        raise ValueError("V3 prediction batch_size must be positive")
    resolved_device = resolve_device(str(device)) if not isinstance(device, torch.device) else device
    tables = materialize_route_candidate_tables(bundle.graphs) if route_tables is None else route_tables
    target_map = materialize_structured_assignment_targets(bundle.graphs, tables) if targets is None else targets
    expected = {id(graph) for graph in bundle.graphs}
    if set(tables) != expected or set(target_map) != expected:
        raise ValueError("V3 prediction route tables or targets do not match the supplied graph bundle")
    edge_scores = [np.full(candidate.labels.shape, np.nan, dtype=np.float64) for candidate in bundle.adjacent_sets]
    base_scores = [np.full(candidate.labels.shape, np.nan, dtype=np.float64) for candidate in bundle.adjacent_sets]
    route_utilities: list[np.ndarray] = []
    route_labels: list[np.ndarray] = []
    route_truth: list[np.ndarray] = []
    route_fake: list[np.ndarray] = []
    route_hard: list[np.ndarray] = []
    score_sets: list[StructuredRouteScoreSet | None] = [None] * len(bundle.graphs)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(bundle.graphs), batch_size):
            graphs = bundle.graphs[start : start + batch_size]
            batch = _make_route_batch(graphs, node_standardizer, edge_standardizer, resolved_device, tables)
            output = _forward_route_batch(model, batch)
            values = torch.sigmoid(output.edge_logits).detach().cpu().numpy().astype(np.float64)
            base_values = torch.sigmoid(output.base_edge_logits).detach().cpu().numpy().astype(np.float64)
            for value, base, owner, row in zip(values, base_values, batch.base.score_owner, batch.base.score_row):
                if np.isfinite(edge_scores[int(owner)][int(row)]):
                    raise RuntimeError("V3 edge score reconstruction duplicated a candidate row")
                edge_scores[int(owner)][int(row)] = float(value)
                base_scores[int(owner)][int(row)] = float(base)
            utilities = route_assignment_utilities(output, batch).detach().cpu().numpy().astype(np.float64)
            route_offset = 0
            for graph_offset, graph in enumerate(graphs):
                table = tables[id(graph)]
                target = target_map[id(graph)]
                stop = route_offset + table.size
                local = utilities[route_offset:stop]
                if local.shape != (table.size,):
                    raise RuntimeError("V3 route utility reconstruction lost a physical route row")
                score_sets[start + graph_offset] = StructuredRouteScoreSet(
                    sample=graph.sample,
                    event=graph.event,
                    endpoint_indices=np.asarray(table.node_indices, dtype=np.int64),
                    utilities=local,
                    labels=np.asarray(table.labels, dtype=bool),
                    truth_assignment=np.asarray(target.truth_assignment, dtype=bool),
                    fake_endpoint=np.asarray(table.fake_endpoint, dtype=bool),
                    hard_negative=np.asarray(table.hard_negative, dtype=bool),
                )
                route_offset = stop
            if route_offset != utilities.size:
                raise RuntimeError("V3 route utility reconstruction has an unexpected batch tail")
            if utilities.size:
                route_utilities.append(utilities)
                route_labels.append(batch.route_labels.detach().cpu().numpy().astype(bool))
                route_fake.append(batch.route_fake_endpoint.detach().cpu().numpy().astype(bool))
                route_hard.append(batch.route_hard_negative.detach().cpu().numpy().astype(bool))
                route_truth.extend(
                    target_map[id(graph)].truth_assignment for graph in graphs
                )
    for values in (*edge_scores, *base_scores):
        if values.size and not np.isfinite(values).all():
            raise RuntimeError("V3 edge score reconstruction left a candidate row unfilled")
    if any(value is None for value in score_sets):  # pragma: no cover - batch invariant
        raise RuntimeError("V3 route utility reconstruction omitted a physical graph")
    return StructuredAssignmentPrediction(
        edge_scores=tuple(edge_scores),
        base_edge_scores=tuple(base_scores),
        route_utilities=(np.concatenate(route_utilities) if route_utilities else np.empty(0, dtype=np.float64)),
        route_labels=(np.concatenate(route_labels) if route_labels else np.empty(0, dtype=bool)),
        route_truth_assignment=(np.concatenate(route_truth) if route_truth else np.empty(0, dtype=bool)),
        route_fake_endpoint=(np.concatenate(route_fake) if route_fake else np.empty(0, dtype=bool)),
        route_hard_negative=(np.concatenate(route_hard) if route_hard else np.empty(0, dtype=bool)),
        route_score_sets=tuple(value for value in score_sets if value is not None),
    )


def save_structured_assignment_artifact(
    path: str | Path,
    model: RouteAwareSparseTransformer,
    artifact: StructuredAssignmentArtifact,
) -> None:
    """Persist a V3 checkpoint without calibration, thresholds, or test data."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "faser-structured-assignment-transformer-v3",
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


def load_structured_assignment_artifact(
    path: str | Path,
    *,
    device: str = "auto",
) -> tuple[RouteAwareSparseTransformer, StructuredAssignmentArtifact]:
    """Restore a V3 checkpoint without accessing corpus assets."""
    source = Path(path).expanduser().resolve()
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with the LCG PyTorch build.
        payload = torch.load(source, map_location="cpu")
    if payload.get("schema_version") != "faser-structured-assignment-transformer-v3":
        raise ValueError("unexpected structured-assignment V3 checkpoint schema")
    model_config = RouteAwareTransformerConfig(**dict(payload["model_config"]))
    model = RouteAwareSparseTransformer(model_config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(resolve_device(device))
    artifact = StructuredAssignmentArtifact(
        node_feature_names=tuple(payload["node_feature_names"]),
        edge_feature_names=tuple(payload["edge_feature_names"]),
        all_station_pairs=tuple(tuple(int(value) for value in pair) for pair in payload["all_station_pairs"]),
        output_station_pairs=tuple(tuple(int(value) for value in pair) for pair in payload["output_station_pairs"]),
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
        raise ValueError("V3 checkpoint feature schema is incompatible with the physical candidate graph")
    if artifact.all_station_pairs != ALL_STATION_PAIRS or artifact.output_station_pairs != ADJACENT_STATION_PAIRS:
        raise ValueError("V3 checkpoint station-pair schema is incompatible with route assignment")
    if artifact.context_mode != "full_event":
        raise ValueError("structured-assignment V3 checkpoint is not full-event context")
    return model, artifact


def structured_assignment_artifact_summary(artifact: StructuredAssignmentArtifact) -> dict[str, object]:
    """Return a JSON-safe V3 checkpoint contract summary."""
    return {
        "schema_version": "faser-structured-assignment-transformer-v3",
        "node_feature_names": list(artifact.node_feature_names),
        "edge_feature_names": list(artifact.edge_feature_names),
        "all_station_pairs": [f"{left}->{right}" for left, right in artifact.all_station_pairs],
        "output_station_pairs": [f"{left}->{right}" for left, right in artifact.output_station_pairs],
        "model_config": asdict(artifact.model_config),
        "context_mode": artifact.context_mode,
        "training_summary": dict(artifact.training_summary),
    }
