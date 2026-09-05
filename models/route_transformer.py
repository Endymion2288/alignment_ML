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

import math
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
    # Route representation mode.  "absolute" (default / historical V2/V4/V5A)
    # consumes the absolute/global node latent route representation: the
    # query-pooled route token, the four endpoint latent states s0..s3, the
    # learned adjacent-edge projections, the station-pair embeddings, and the
    # base edge logits.  "physical_pair_relative" (Workbook-74 Physical
    # Pair-Relative Route Encoder) drops the absolute node latent entirely and
    # consumes only the Workbook-73 R_phys tensor: the three adjacent physical
    # pair-relative edge observables (3 x edge_feature_dim) plus the frozen
    # Workbook-64 production edge logits L01/L12/L23 (3).  It never reads
    # s0..s3, the route query, the learned edge projection, or pair embeddings.
    route_representation_mode: str = "absolute"
    # Historical V2 checkpoints omit this key and must keep the original
    # route-query -> route_edge_correction path.  RelativeRoute V4 YAML sets
    # the additive complete-route correction explicitly.
    use_additive_route_correction: bool = False
    # Workbook-72 V5A bounded residual.  When None (default / historical V4)
    # the additive complete-route correction is unbounded:
    #     delta = raw_delta.
    # When set to a positive float B (V5A), the raw head output is squashed to
    # the solver-semantic O(1) scale before entering L_corrected:
    #     delta_bounded = B * tanh(raw_delta / B)  in [-B, +B].
    # tanh(0)=0 and d/dx[B*tanh(x/B)] = sech^2(x/B) = 1 at x=0, so zero-init
    # and the unit slope at the origin are both preserved exactly.  B=4 matches
    # the production dustbin boundary (U_complete = L_corrected - 4).
    route_correction_bound: float | None = None

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
    delta_route_logits: torch.Tensor | None = None


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
        route_mode = str(config.route_representation_mode)
        if route_mode not in ("absolute", "physical_pair_relative"):
            raise ValueError(
                "route_representation_mode must be 'absolute' or 'physical_pair_relative', "
                f"got {route_mode!r}"
            )
        # Constructing the V1 backbone also enforces d=128, 8 heads, 4 blocks,
        # and the 128 -> 256 -> 128 FFN contract.
        self.config = config
        self.backbone = GeometryAwareSparseTransformer(config.backbone_config())
        if route_mode == "physical_pair_relative":
            # Physical Pair-Relative Route Encoder: the route input is exactly
            # the Workbook-73 R_phys tensor (3 adjacent physical pair-relative
            # edge observables + 3 frozen edge logits).  No absolute node
            # latent, route query, learned edge projection, or pair embedding
            # is constructed, so the trainable head cannot consume them.
            route_input_dim = 3 * config.edge_feature_dim + 3
        else:
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
        if route_mode != "physical_pair_relative":
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
        production_edge_logits: torch.Tensor | None = None,
    ) -> RouteAwareTransformerOutput:
        """Return route-aware adjacent logits without changing candidate rows.

        ``production_edge_logits`` is only consulted by the
        ``physical_pair_relative`` route representation: it is the frozen
        Workbook-64 production adjacent-edge logits (L01/L12/L23), supplied by
        the ``RelativeRouteV4Inference`` wrapper so the route head consumes the
        frozen production edge scorer rather than re-estimating adjacent edge
        logits.  When ``None`` (standalone forward) the backbone's own base
        edge logits are used as a fallback.
        """
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
                delta_route_logits=base_edge_logits.new_empty((0,)),
            )

        # The additive edge-correction path always keys on the backbone's own
        # base edge logits (frozen Workbook-64 weights), independent of the
        # route representation mode.
        route_base_logits = base_edge_logits[route_score_edge_indices]
        if self.config.route_representation_mode == "physical_pair_relative":
            # Workbook-74 Physical Pair-Relative Route Encoder.  The route input
            # is exactly the Workbook-73 R_phys tensor: the three adjacent
            # physical pair-relative edge observables (the standardized features
            # entering the frozen edge scorer) plus the frozen Workbook-64
            # production edge logits L01/L12/L23.  No absolute node latent,
            # route query, learned edge projection, or pair embedding is read.
            route_edge_observables = score_edge_features[route_score_edge_indices]
            route_edge_logits_src = (
                production_edge_logits if production_edge_logits is not None else base_edge_logits
            )
            route_production_logits = route_edge_logits_src[route_score_edge_indices]
            route_input = torch.cat(
                (
                    route_edge_observables.reshape(routes, -1),
                    route_production_logits,
                ),
                dim=-1,
            )
        else:
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
        # Workbook-72 V5A bounded residual: squash the raw head output onto the
        # solver-semantic O(1) scale before it enters L_corrected.  Applied here
        # (in the head) so that every downstream consumer -- the additive
        # route_logits below, the RelativeRouteV4Inference wrapper, the
        # solver-aware losses, and the production solver -- sees the bounded
        # delta.  None (historical V4) leaves the correction unbounded.
        bound = self.config.route_correction_bound
        if bound is not None:
            bound_value = float(bound)
            if not math.isfinite(bound_value) or bound_value <= 0.0:
                raise ValueError("route_correction_bound must be a positive finite float")
            delta_route_logits = bound_value * torch.tanh(delta_route_logits / bound_value)
        if self.config.use_additive_route_correction:
            # Trainable delta must not enter the frozen Workbook-64 edge
            # correction.  Historical V2 computed route_edge_correction from
            # the route-query output; under the additive contract that query
            # is delta_route_logit and would move production edges.
            route_for_edge_correction = route_base_logits.sum(dim=-1).detach()
        else:
            route_for_edge_correction = delta_route_logits
        route_mean, route_counts = self._route_edge_mean(
            route_for_edge_correction, route_score_edge_indices, int(base_edge_logits.numel())
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
        if self.config.use_additive_route_correction:
            production_edge_sum = edge_logits[route_score_edge_indices].sum(dim=-1)
            route_logits = production_edge_sum + delta_route_logits
        else:
            route_logits = delta_route_logits
        return RouteAwareTransformerOutput(
            edge_logits=edge_logits,
            base_edge_logits=base_edge_logits,
            route_logits=route_logits,
            route_edge_counts=route_counts,
            delta_route_logits=delta_route_logits,
        )


@dataclass(frozen=True)
class RelativeRouteTransformerConfig(RouteAwareTransformerConfig):
    """Configuration for Relative Route Transformer V4."""

    use_relative_route_representation: bool = True
    use_additive_route_correction: bool = True


class RelativeRouteSparseTransformer(RouteAwareSparseTransformer):
    """Relative Route Transformer V4 for explicit 4-station route scoring."""

    def __init__(self, config: RouteAwareTransformerConfig) -> None:
        if not config.use_relative_route_representation or not config.use_additive_route_correction:
            config = RelativeRouteTransformerConfig(
                **{
                    **config.as_dict(),
                    "use_relative_route_representation": True,
                    "use_additive_route_correction": True,
                }
            )
        super().__init__(config)


class RelativeRouteV4Inference(nn.Module):
    """Compose frozen Workbook-64 production edges with a trainable route head.

    Production adjacent logits are exactly the frozen Workbook-64 forward
    (historical route-query -> route_edge_correction).  The trainable head
    emits only ``delta_route_logit``.  Complete-route logits consumed by the
    solver-aware objective and by ``complete_route_scores`` are

        L_corrected = L_edge_W64 + delta_route_logit

    where ``L_edge_W64`` is the sum of the three frozen production adjacent
    logits on that physical chain.  Fragments never see ``delta``.
    """

    def __init__(
        self,
        frozen_workbook64: RouteAwareSparseTransformer,
        trainable: RouteAwareSparseTransformer,
    ) -> None:
        super().__init__()
        if frozen_workbook64.config.use_additive_route_correction:
            raise ValueError("frozen Workbook-64 replica must use the historical V2 edge path")
        if not trainable.config.use_additive_route_correction:
            raise ValueError("RelativeRoute V4 trainable head must use additive complete-route correction")
        self.frozen_workbook64 = frozen_workbook64
        self.trainable = trainable
        for param in self.frozen_workbook64.parameters():
            param.requires_grad = False
        self.frozen_workbook64.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Dropout in the frozen Workbook-64 replica would change production
        # adjacent logits; keep that replica in eval regardless of wrapper mode.
        self.frozen_workbook64.eval()
        return self

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
        with torch.no_grad():
            frozen_out = self.frozen_workbook64(
                node_features,
                station_ids,
                message_edge_source,
                message_edge_destination,
                message_edge_features,
                message_edge_chi2,
                message_edge_station_pair,
                message_edge_direction,
                score_edge_source,
                score_edge_destination,
                score_edge_features,
                score_edge_station_pair,
                route_node_indices,
                route_score_edge_indices,
            )
        train_out = self.trainable(
            node_features,
            station_ids,
            message_edge_source,
            message_edge_destination,
            message_edge_features,
            message_edge_chi2,
            message_edge_station_pair,
            message_edge_direction,
            score_edge_source,
            score_edge_destination,
            score_edge_features,
            score_edge_station_pair,
            route_node_indices,
            route_score_edge_indices,
            # Frozen Workbook-64 production adjacent-edge logits.  Only the
            # physical_pair_relative route representation consumes these (as
            # L01/L12/L23); the absolute/relative modes ignore them.  They are
            # detached (frozen_out is computed under no_grad), so they enter
            # the route head as frozen input features, not as a gradient path.
            production_edge_logits=frozen_out.edge_logits,
        )
        delta = train_out.delta_route_logits
        if delta is None:
            delta = frozen_out.route_logits.new_zeros(frozen_out.route_logits.shape)
        if int(delta.numel()):
            production_edge_sum = frozen_out.edge_logits[route_score_edge_indices].sum(dim=-1)
            route_logits = production_edge_sum + delta
        else:
            route_logits = frozen_out.route_logits.new_empty((0,))
        return RouteAwareTransformerOutput(
            edge_logits=frozen_out.edge_logits,
            base_edge_logits=frozen_out.base_edge_logits,
            route_logits=route_logits,
            route_edge_counts=frozen_out.route_edge_counts,
            delta_route_logits=delta,
        )


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
