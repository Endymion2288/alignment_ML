"""Route-aware sparse Transformer head for four-station FASER association.

The backbone remains the V1 geometry-aware sparse Transformer and receives
only the existing physical candidate graph.  This module adds a route query on
explicit IFT -> S1 -> S2 -> S3 chains made of those already-declared adjacent
candidate edges.  It never synthesizes a coordinate-level edge.  The route
query is emitted alongside adjacent edge scores and can be consumed as a
calibrated residual utility by the existing unit-capacity route assignment
backend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .transformer import GeometryAwareSparseTransformer, SparseTransformerConfig


@dataclass(frozen=True)
class RouteAwareTransformerConfig:
    """Fixed V2 backbone plus the small explicit route-query decoder."""

    node_feature_dim: int
    edge_feature_dim: int
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    ffn_dim: int = 256
    dropout: float = 0.10
    num_stations: int = 4
    num_station_pairs: int = 6
    station_embedding_dim: int = 16
    station_pair_embedding_dim: int = 16
    direction_embedding_dim: int = 4
    use_geometric_bias: bool = True
    use_chi2_physics_term: bool = True
    chi2_lambda: float = 1.0
    chi2_tau: float = 1.0e6
    use_local_edge_residual: bool = True
    route_hidden_dim: int = 128
    route_pair_embedding_dim: int = 16
    route_dropout: float = 0.10
    use_relative_route_representation: bool = False
    use_additive_route_correction: bool = True

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def backbone_config(self) -> SparseTransformerConfig:
        """Build the unchanged four-block geometry-aware sparse backbone."""
        return SparseTransformerConfig(
            node_feature_dim=self.node_feature_dim,
            edge_feature_dim=self.edge_feature_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout,
            num_stations=self.num_stations,
            num_station_pairs=self.num_station_pairs,
            station_embedding_dim=self.station_embedding_dim,
            station_pair_embedding_dim=self.station_pair_embedding_dim,
            direction_embedding_dim=self.direction_embedding_dim,
            use_geometric_bias=self.use_geometric_bias,
            use_chi2_physics_term=self.use_chi2_physics_term,
            chi2_lambda=self.chi2_lambda,
            chi2_tau=self.chi2_tau,
            use_local_edge_residual=self.use_local_edge_residual,
        )


@dataclass(frozen=True)
class RouteAwareTransformerOutput:
    """Adjacent edge scores plus explicit complete-route logits."""

    edge_logits: torch.Tensor
    base_edge_logits: torch.Tensor
    route_logits: torch.Tensor
    route_edge_counts: torch.Tensor


class RouteAwareSparseTransformer(nn.Module):
    """V1 physical graph backbone with a learned complete-route query.

    ``route_node_indices`` has shape ``[R, 4]`` in station order 0, 1, 2, 3.
    ``route_score_edge_indices`` has shape ``[R, 3]`` and points into the
    supplied adjacent score-edge table for 0->1, 1->2 and 2->3.  Both tables
    are constructed outside the network from real physical candidates.
    """

    def __init__(self, config: RouteAwareTransformerConfig) -> None:
        super().__init__()
        if config.route_hidden_dim < 2 or config.route_pair_embedding_dim < 1:
            raise ValueError("route hidden and pair-embedding dimensions must be positive")
        if not 0.0 <= config.route_dropout < 1.0:
            raise ValueError("route dropout must be in [0, 1)")
        # Constructing the V1 backbone also enforces d=128, 8 heads, 4 blocks,
        # and the 128 -> 256 -> 128 FFN contract.
        self.config = config
        self.backbone = GeometryAwareSparseTransformer(config.backbone_config())
        self.route_query = nn.Parameter(torch.empty(config.d_model))
        self.route_node_key = nn.Linear(config.d_model, config.d_model, bias=False)
        self.route_edge_projection = nn.Sequential(
            nn.Linear(config.edge_feature_dim, config.d_model),
            nn.GELU(),
        )
        self.route_pair_embedding = nn.Embedding(
            config.num_station_pairs, config.route_pair_embedding_dim
        )
        route_input_dim = (
            # Query-pooled route token + the ordered four endpoint states.
            config.d_model + 4 * config.d_model
            # Ordered adjacent physical edge representations and pair IDs.
            + 3 * config.d_model + 3 * config.route_pair_embedding_dim
            # Preserve the backbone's local adjacent edge evidence.
            + 3
        )
        self.route_encoder = nn.Sequential(
            nn.Linear(route_input_dim, config.route_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.route_dropout),
            nn.Linear(config.route_hidden_dim, config.route_hidden_dim),
            nn.GELU(),
        )
        self.route_score = nn.Linear(config.route_hidden_dim, 1)
        nn.init.zeros_(self.route_score.weight)
        nn.init.zeros_(self.route_score.bias)
        # A zero terminal makes the initial V2 edge interface exactly the V1
        # backbone interface.  The route loss can learn a route score first;
        # only then does edge supervision admit a route-aware correction.
        self.route_edge_correction = nn.Sequential(
            nn.Linear(3, config.route_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.route_hidden_dim // 2, 1),
        )
        terminal = self.route_edge_correction[-1]
        if not isinstance(terminal, nn.Linear):  # pragma: no cover - construction invariant
            raise RuntimeError("route correction terminal must be linear")
        nn.init.zeros_(terminal.weight)
        nn.init.zeros_(terminal.bias)
        nn.init.normal_(self.route_query, mean=0.0, std=config.d_model**-0.5)

    def _validate_routes(
        self,
        route_node_indices: torch.Tensor,
        route_score_edge_indices: torch.Tensor,
        *,
        node_count: int,
        score_edge_count: int,
    ) -> int:
        if route_node_indices.ndim != 2 or route_node_indices.shape[1] != 4:
            raise ValueError("route node indices must have shape [routes, 4]")
        if route_score_edge_indices.shape != (route_node_indices.shape[0], 3):
            raise ValueError("route score-edge indices must have shape [routes, 3]")
        routes = int(route_node_indices.shape[0])
        if routes and (
            torch.any(route_node_indices < 0)
            or torch.any(route_node_indices >= node_count)
            or torch.any(route_score_edge_indices < 0)
            or torch.any(route_score_edge_indices >= score_edge_count)
        ):
            raise ValueError("route candidate endpoint or score edge is outside its graph batch")
        return routes

    @staticmethod
    def _route_edge_mean(
        route_logits: torch.Tensor,
        route_score_edge_indices: torch.Tensor,
        edge_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate complete-route evidence back onto existing edge rows."""
        if not int(route_logits.numel()):
            return route_logits.new_zeros((edge_count,)), route_logits.new_zeros((edge_count,))
        flat_indices = route_score_edge_indices.reshape(-1)
        flat_logits = route_logits.repeat_interleave(3)
        totals = route_logits.new_zeros((edge_count,))
        counts = route_logits.new_zeros((edge_count,))
        totals.scatter_add_(0, flat_indices, flat_logits)
        counts.scatter_add_(0, flat_indices, torch.ones_like(flat_logits))
        return totals / counts.clamp_min(1.0), counts

    def forward(
        self,
        node_features: torch.Tensor,
        station_ids: torch.Tensor,
        message_edge_source: torch.Tensor,
        message_edge_destination: torch.Tensor,
        message_edge_features: torch.Tensor,
        message_edge_chi2: torch.Tensor,
        message_edge_station_pair: torch.Tensor,
        message_edge_direction: torch.Tensor,
        score_edge_source: torch.Tensor,
        score_edge_destination: torch.Tensor,
        score_edge_features: torch.Tensor,
        score_edge_station_pair: torch.Tensor,
        route_node_indices: torch.Tensor,
        route_score_edge_indices: torch.Tensor,
    ) -> RouteAwareTransformerOutput:
        """Return route-aware adjacent logits without changing candidate rows."""
        node_states = self.backbone.encode_node_states(
            node_features,
            station_ids,
            message_edge_source,
            message_edge_destination,
            message_edge_features,
            message_edge_chi2,
            message_edge_station_pair,
            message_edge_direction,
        )
        base_edge_logits = self.backbone.score_node_states(
            node_states,
            node_features,
            score_edge_source,
            score_edge_destination,
            score_edge_features,
            score_edge_station_pair,
        )
        routes = self._validate_routes(
            route_node_indices,
            route_score_edge_indices,
            node_count=int(node_states.shape[0]),
            score_edge_count=int(base_edge_logits.numel()),
        )
        if not routes:
            return RouteAwareTransformerOutput(
                edge_logits=base_edge_logits,
                base_edge_logits=base_edge_logits,
                route_logits=base_edge_logits.new_empty((0,)),
                route_edge_counts=base_edge_logits.new_zeros(base_edge_logits.shape),
            )

        route_nodes = node_states[route_node_indices]
        if self.config.use_relative_route_representation:
            # Anchor 4-station route endpoint states to Station 0 in latent space
            ref_node = route_nodes[:, 0:1]
            diff_nodes = route_nodes[:, 1:] - ref_node
            rel_route_nodes = torch.cat([ref_node, diff_nodes], dim=1)
            route_node_features = rel_route_nodes.reshape(routes, -1)
        else:
            route_node_features = route_nodes.reshape(routes, -1)

        node_keys = self.route_node_key(route_nodes)
        query_logits = (node_keys * self.route_query[None, None, :]).sum(dim=-1)
        query_weights = torch.softmax(query_logits / (self.config.d_model**0.5), dim=1)
        pooled_nodes = torch.sum(route_nodes * query_weights[:, :, None], dim=1)
        route_edge_features = score_edge_features[route_score_edge_indices]
        route_edge_states = self.route_edge_projection(route_edge_features)
        route_pair_states = self.route_pair_embedding(score_edge_station_pair[route_score_edge_indices])
        route_base_logits = base_edge_logits[route_score_edge_indices]
        route_input = torch.cat(
            (
                pooled_nodes,
                route_node_features,
                route_edge_states.reshape(routes, -1),
                route_pair_states.reshape(routes, -1),
                route_base_logits,
            ),
            dim=-1,
        )
        delta_route_logits = self.route_score(self.route_encoder(route_input)).squeeze(-1)
        if self.config.use_additive_route_correction:
            edge_sum_logits = route_base_logits.sum(dim=-1)
            route_logits = edge_sum_logits + delta_route_logits
        else:
            route_logits = delta_route_logits
        route_mean, route_counts = self._route_edge_mean(
            route_logits, route_score_edge_indices, int(base_edge_logits.numel())
        )
        correction_features = torch.stack(
            (
                base_edge_logits,
                route_mean,
                torch.log1p(route_counts),
            ),
            dim=-1,
        )
        correction = self.route_edge_correction(correction_features).squeeze(-1)
        edge_logits = base_edge_logits + correction * (route_counts > 0.0).to(correction.dtype)
        return RouteAwareTransformerOutput(
            edge_logits=edge_logits,
            base_edge_logits=base_edge_logits,
            route_logits=route_logits,
            route_edge_counts=route_counts,
        )


@dataclass(frozen=True)
class RelativeRouteTransformerConfig(RouteAwareTransformerConfig):
    """Configuration for Relative Route Transformer V4."""

    use_relative_route_representation: bool = True


class RelativeRouteSparseTransformer(RouteAwareSparseTransformer):
    """Relative Route Transformer V4 for explicit 4-station route scoring."""

    def __init__(self, config: RouteAwareTransformerConfig) -> None:
        if not config.use_relative_route_representation:
            config = RouteAwareTransformerConfig(
                **{**config.as_dict(), "use_relative_route_representation": True}
            )
        super().__init__(config)


def freeze_backbone_and_edge_scorer(
    model: RouteAwareSparseTransformer,
) -> dict[str, object]:
    """Freeze V2 backbone and edge scorer, keeping only complete-route head trainable.

    Trainable route-head components:
      - route_query
      - route_node_key
      - route_edge_projection
      - route_pair_embedding
      - route_encoder
      - route_score

    Frozen components:
      - node_encoder
      - node_layers (Transformer encoder layers)
      - edge_encoder
      - edge_score
      - local_edge_residual
      - route_edge_correction
    """
    ROUTE_HEAD_PREFIXES = (
        "route_query",
        "route_node_key",
        "route_edge_projection",
        "route_pair_embedding",
        "route_encoder",
        "route_score",
    )
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    n_trainable = 0
    n_frozen = 0

    for name, param in model.named_parameters():
        is_route_head = any(name == prefix or name.startswith(f"{prefix}.") for prefix in ROUTE_HEAD_PREFIXES)
        if is_route_head:
            param.requires_grad = True
            trainable_names.append(name)
            n_trainable += param.numel()
        else:
            param.requires_grad = False
            frozen_names.append(name)
            n_frozen += param.numel()

    return {
        "trainable_parameter_names": tuple(trainable_names),
        "frozen_parameter_names": tuple(frozen_names),
        "n_trainable_parameters": int(n_trainable),
        "n_frozen_parameters": int(n_frozen),
    }
