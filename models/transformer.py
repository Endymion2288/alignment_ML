"""Minimal geometry-aware sparse Transformer for FASER tracklet association."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import nn

from .encoder import TrackletStateEncoder
from .geometric_attention import SparseGeometricAttention


@dataclass(frozen=True)
class SparseTransformerConfig:
    """Architecture and physics-prior settings persisted with a checkpoint."""

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
    use_local_edge_residual: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class SparseTransformerBlock(nn.Module):
    """Pre-normalized sparse attention and 128 -> 256 -> 128 FFN block."""

    def __init__(self, config: SparseTransformerConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = SparseGeometricAttention(
            d_model=config.d_model,
            nhead=config.nhead,
            edge_feature_dim=config.edge_feature_dim,
            num_station_pairs=config.num_station_pairs,
            station_pair_embedding_dim=config.station_pair_embedding_dim,
            direction_embedding_dim=config.direction_embedding_dim,
            use_geometric_bias=config.use_geometric_bias,
            use_chi2_physics_term=config.use_chi2_physics_term,
            chi2_lambda=config.chi2_lambda,
            chi2_tau=config.chi2_tau,
        )
        self.feed_forward_norm = nn.LayerNorm(config.d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.d_model),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        node_states: torch.Tensor,
        edge_source: torch.Tensor,
        edge_destination: torch.Tensor,
        edge_features: torch.Tensor,
        edge_chi2: torch.Tensor,
        edge_station_pair: torch.Tensor,
        edge_direction: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        attention_output = self.attention(
            self.attention_norm(node_states),
            edge_source,
            edge_destination,
            edge_features,
            edge_chi2,
            edge_station_pair,
            edge_direction,
            return_attention=return_attention,
        )
        if return_attention:
            message, attention = attention_output
        else:
            message = attention_output
        node_states = node_states + self.dropout(message)
        result = node_states + self.dropout(self.feed_forward(self.feed_forward_norm(node_states)))
        return (result, attention) if return_attention else result


class GeometryAwareSparseTransformer(nn.Module):
    """Score physical candidate edges after sparse multi-station message passing."""

    def __init__(self, config: SparseTransformerConfig) -> None:
        super().__init__()
        if config.num_layers != 4:
            raise ValueError("Transformer V1 is fixed to four sparse blocks")
        if config.nhead != 8:
            raise ValueError("Transformer V1 is fixed to eight attention heads")
        if config.d_model != 128 or config.ffn_dim != 256:
            raise ValueError("Transformer V1 requires d_model=128 and FFN 128->256->128")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.config = config
        self.encoder = TrackletStateEncoder(
            node_feature_dim=config.node_feature_dim,
            d_model=config.d_model,
            num_stations=config.num_stations,
            station_embedding_dim=config.station_embedding_dim,
        )
        self.blocks = nn.ModuleList(
            [SparseTransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.score_pair_embedding = nn.Embedding(
            config.num_station_pairs, config.station_pair_embedding_dim
        )
        self.edge_score = nn.Sequential(
            nn.Linear(
                2 * config.d_model + config.edge_feature_dim + config.station_pair_embedding_dim,
                config.d_model,
            ),
            nn.GELU(),
            nn.Linear(config.d_model, 1),
        )
        # A four-layer message-passing stack should not be forced to recreate
        # a local two-endpoint discriminant after mixing information across a
        # sparse event graph.  This optional residual is trained jointly from
        # the same standardized persisted endpoint state and geometry vector;
        # it contains neither truth nor a separately pretrained MLP score.
        self.use_local_edge_residual = bool(config.use_local_edge_residual)
        if self.use_local_edge_residual:
            self.local_edge_score = nn.Sequential(
                nn.Linear(
                    2 * config.node_feature_dim
                    + config.edge_feature_dim
                    + config.station_pair_embedding_dim,
                    config.d_model,
                ),
                nn.GELU(),
                nn.Linear(config.d_model, 1),
            )
            # Start joint context training as a residual correction to the
            # locally pretrained edge logit.  The zero terminal layer leaves
            # the local association ranking intact at the first joint step.
            terminal = self.edge_score[-1]
            if not isinstance(terminal, nn.Linear):  # pragma: no cover - construction invariant
                raise RuntimeError("edge score terminal layer must be linear")
            nn.init.zeros_(terminal.weight)
            nn.init.zeros_(terminal.bias)

    def local_edge_logits(
        self,
        node_features: torch.Tensor,
        score_edge_source: torch.Tensor,
        score_edge_destination: torch.Tensor,
        score_edge_features: torch.Tensor,
        score_edge_station_pair: torch.Tensor,
    ) -> torch.Tensor:
        """Return the optional local residual edge logit without graph messages."""
        if not self.use_local_edge_residual:
            raise RuntimeError("local edge logits are disabled by this model configuration")
        edge_count = int(score_edge_source.numel())
        if (
            score_edge_destination.shape != (edge_count,)
            or score_edge_station_pair.shape != (edge_count,)
            or score_edge_features.shape != (edge_count, self.config.edge_feature_dim)
        ):
            raise ValueError("local score-edge tensors do not match the model schema")
        if edge_count == 0:
            return node_features.new_empty((0,))
        pair_embedding = self.score_pair_embedding(score_edge_station_pair)
        local_features = torch.cat(
            (
                node_features[score_edge_source],
                node_features[score_edge_destination],
                score_edge_features,
                pair_embedding,
            ),
            dim=-1,
        )
        return self.local_edge_score(local_features).squeeze(-1)

    def _validate_score_edges(
        self,
        score_edge_source: torch.Tensor,
        score_edge_destination: torch.Tensor,
        score_edge_features: torch.Tensor,
        score_edge_station_pair: torch.Tensor,
    ) -> int:
        """Validate the declared output edges and return their count."""
        edge_count = int(score_edge_source.numel())
        if (
            score_edge_destination.shape != (edge_count,)
            or score_edge_station_pair.shape != (edge_count,)
            or score_edge_features.shape != (edge_count, self.config.edge_feature_dim)
        ):
            raise ValueError("score-edge tensors do not match the model schema")
        if edge_count and (
            torch.any(score_edge_station_pair < 0)
            or torch.any(score_edge_station_pair >= self.config.num_station_pairs)
        ):
            raise ValueError("score station-pair ID is outside the configured embedding range")
        return edge_count

    def encode_node_states(
        self,
        node_features: torch.Tensor,
        station_ids: torch.Tensor,
        message_edge_source: torch.Tensor,
        message_edge_destination: torch.Tensor,
        message_edge_features: torch.Tensor,
        message_edge_chi2: torch.Tensor,
        message_edge_station_pair: torch.Tensor,
        message_edge_direction: torch.Tensor,
        *,
        maximum_layers: int | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Encode nodes through a declared prefix of sparse message blocks.

        The optional trace is inference-oriented.  It exposes the encoded
        state at depth zero and after every executed block, plus the attention
        weights on exactly the supplied message edges.  It never creates an
        implicit all-pairs edge or changes the standard ``forward`` path.
        """
        layers = self.config.num_layers if maximum_layers is None else int(maximum_layers)
        if layers < 0 or layers > self.config.num_layers:
            raise ValueError("maximum_layers is outside the configured sparse-block range")
        node_states = self.encoder(node_features, station_ids)
        states: list[torch.Tensor] = [node_states] if return_attention else []
        weights: list[torch.Tensor] = []
        for block in self.blocks[:layers]:
            if return_attention:
                node_states, attention = block(
                    node_states,
                    message_edge_source,
                    message_edge_destination,
                    message_edge_features,
                    message_edge_chi2,
                    message_edge_station_pair,
                    message_edge_direction,
                    return_attention=True,
                )
                states.append(node_states)
                weights.append(attention)
            else:
                node_states = block(
                    node_states,
                    message_edge_source,
                    message_edge_destination,
                    message_edge_features,
                    message_edge_chi2,
                    message_edge_station_pair,
                    message_edge_direction,
                )
        if return_attention:
            return node_states, tuple(states), tuple(weights)
        return node_states

    def score_node_states(
        self,
        node_states: torch.Tensor,
        node_features: torch.Tensor,
        score_edge_source: torch.Tensor,
        score_edge_destination: torch.Tensor,
        score_edge_features: torch.Tensor,
        score_edge_station_pair: torch.Tensor,
    ) -> torch.Tensor:
        """Score the declared output candidate edges from encoded node states."""
        edge_count = self._validate_score_edges(
            score_edge_source,
            score_edge_destination,
            score_edge_features,
            score_edge_station_pair,
        )
        if edge_count == 0:
            return node_states.new_empty((0,))
        pair_embedding = self.score_pair_embedding(score_edge_station_pair)
        score_features = torch.cat(
            (
                node_states[score_edge_source],
                node_states[score_edge_destination],
                score_edge_features,
                pair_embedding,
            ),
            dim=-1,
        )
        logits = self.edge_score(score_features).squeeze(-1)
        if self.use_local_edge_residual:
            logits = logits + self.local_edge_logits(
                node_features,
                score_edge_source,
                score_edge_destination,
                score_edge_features,
                score_edge_station_pair,
            )
        return logits

    def forward_with_trace(
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
        *,
        maximum_layers: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, ...]]]:
        """Return logits and depth-resolved states/attention for diagnostics."""
        _, states, attention = self.encode_node_states(
            node_features,
            station_ids,
            message_edge_source,
            message_edge_destination,
            message_edge_features,
            message_edge_chi2,
            message_edge_station_pair,
            message_edge_direction,
            maximum_layers=maximum_layers,
            return_attention=True,
        )
        logits_by_depth = tuple(
            self.score_node_states(
                state,
                node_features,
                score_edge_source,
                score_edge_destination,
                score_edge_features,
                score_edge_station_pair,
            )
            for state in states
        )
        return logits_by_depth[-1], {
            "node_states_by_depth": states,
            "logits_by_depth": logits_by_depth,
            "attention_by_layer": attention,
        }

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
    ) -> torch.Tensor:
        """Return logits only for requested (normally adjacent) candidate edges."""
        node_states = self.encode_node_states(
            node_features,
            station_ids,
            message_edge_source,
            message_edge_destination,
            message_edge_features,
            message_edge_chi2,
            message_edge_station_pair,
            message_edge_direction,
        )
        return self.score_node_states(
            node_states,
            node_features,
            score_edge_source,
            score_edge_destination,
            score_edge_features,
            score_edge_station_pair,
        )


def model_config_from_payload(payload: Mapping[str, object]) -> SparseTransformerConfig:
    """Restore a checkpoint architecture while rejecting undeclared fields."""
    return SparseTransformerConfig(**dict(payload))
