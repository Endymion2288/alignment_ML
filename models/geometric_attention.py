"""Sparse edge attention with optional geometry-aware logit terms."""

from __future__ import annotations

import math

import torch
from torch import nn


class SparseGeometricAttention(nn.Module):
    """Multi-head attention evaluated only on supplied physical graph edges."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        edge_feature_dim: int,
        num_station_pairs: int,
        station_pair_embedding_dim: int = 16,
        direction_embedding_dim: int = 4,
        use_geometric_bias: bool = True,
        use_chi2_physics_term: bool = True,
        chi2_lambda: float = 1.0,
        chi2_tau: float = 1.0e6,
    ) -> None:
        super().__init__()
        if d_model < 1 or nhead < 1 or d_model % nhead:
            raise ValueError("d_model must be divisible by a positive nhead")
        if edge_feature_dim < 1 or num_station_pairs < 1:
            raise ValueError("edge feature and station-pair dimensions must be positive")
        if station_pair_embedding_dim < 1 or direction_embedding_dim < 1:
            raise ValueError("edge embedding dimensions must be positive")
        if not math.isfinite(chi2_lambda) or chi2_lambda < 0.0:
            raise ValueError("chi2_lambda must be finite and non-negative")
        if not math.isfinite(chi2_tau) or chi2_tau <= 0.0:
            raise ValueError("chi2_tau must be finite and positive")

        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.head_dim = d_model // nhead
        self.edge_feature_dim = int(edge_feature_dim)
        self.num_station_pairs = int(num_station_pairs)
        self.use_geometric_bias = bool(use_geometric_bias)
        self.use_chi2_physics_term = bool(use_chi2_physics_term)
        self.chi2_lambda = float(chi2_lambda)
        self.chi2_tau = float(chi2_tau)

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.station_pair_embedding = nn.Embedding(num_station_pairs, station_pair_embedding_dim)
        self.direction_embedding = nn.Embedding(2, direction_embedding_dim)
        self.geometric_bias = nn.Sequential(
            nn.Linear(
                edge_feature_dim + station_pair_embedding_dim + direction_embedding_dim,
                d_model,
            ),
            nn.GELU(),
            nn.Linear(d_model, nhead),
        )

    @staticmethod
    def _sparse_softmax(
        logits: torch.Tensor, edge_destination: torch.Tensor, node_count: int
    ) -> torch.Tensor:
        """Softmax independently over incoming candidate edges at each node."""
        if not logits.numel():
            return logits
        heads = logits.shape[1]
        index = edge_destination[:, None].expand(-1, heads)
        maximum = torch.full(
            (node_count, heads),
            -torch.inf,
            dtype=logits.dtype,
            device=logits.device,
        )
        maximum.scatter_reduce_(0, index, logits, reduce="amax", include_self=True)
        exponent = torch.exp(logits - maximum[edge_destination])
        normalizer = torch.zeros(
            (node_count, heads), dtype=logits.dtype, device=logits.device
        )
        normalizer.scatter_add_(0, index, exponent)
        return exponent / normalizer[edge_destination].clamp_min(torch.finfo(logits.dtype).tiny)

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
        """Aggregate messages along only the declared candidate edges.

        ``edge_source -> edge_destination`` is a directed view of an existing
        physical candidate.  The caller may supply its reverse representation,
        but no all-pairs or implicit self edges are created here.
        """
        if node_states.ndim != 2 or node_states.shape[1] != self.d_model:
            raise ValueError("node states do not match the attention dimension")
        edge_count = int(edge_source.numel())
        expected_shapes = (
            (edge_destination, (edge_count,)),
            (edge_chi2, (edge_count,)),
            (edge_station_pair, (edge_count,)),
            (edge_direction, (edge_count,)),
        )
        if edge_source.ndim != 1 or any(value.shape != shape for value, shape in expected_shapes):
            raise ValueError("sparse edge index tensors are inconsistent")
        if edge_features.shape != (edge_count, self.edge_feature_dim):
            raise ValueError("edge features do not match the attention schema")
        if edge_count == 0:
            result = torch.zeros_like(node_states)
            return (result, node_states.new_empty((0, self.nhead))) if return_attention else result
        if (
            torch.any(edge_source < 0)
            or torch.any(edge_destination < 0)
            or torch.any(edge_source >= node_states.shape[0])
            or torch.any(edge_destination >= node_states.shape[0])
        ):
            raise ValueError("sparse edge endpoints are outside the node table")
        if torch.any(edge_station_pair < 0) or torch.any(edge_station_pair >= self.num_station_pairs):
            raise ValueError("station-pair edge ID is outside the configured embedding range")
        if torch.any(edge_direction < 0) or torch.any(edge_direction > 1):
            raise ValueError("edge direction must be encoded as 0 or 1")

        nodes = node_states.shape[0]
        query = self.query(node_states).reshape(nodes, self.nhead, self.head_dim)
        key = self.key(node_states).reshape(nodes, self.nhead, self.head_dim)
        value = self.value(node_states).reshape(nodes, self.nhead, self.head_dim)
        logits = (query[edge_destination] * key[edge_source]).sum(dim=-1) / math.sqrt(
            self.head_dim
        )

        if self.use_geometric_bias:
            geometry = torch.cat(
                (
                    edge_features,
                    self.station_pair_embedding(edge_station_pair),
                    self.direction_embedding(edge_direction),
                ),
                dim=-1,
            )
            logits = logits + self.geometric_bias(geometry)
        if self.use_chi2_physics_term and self.chi2_lambda:
            logits = logits - (
                self.chi2_lambda / (2.0 * self.chi2_tau)
            ) * edge_chi2[:, None]

        attention = self._sparse_softmax(logits, edge_destination, nodes)
        weighted_values = value[edge_source] * attention[:, :, None]
        aggregate = torch.zeros_like(value)
        aggregate.scatter_add_(
            0,
            edge_destination[:, None, None].expand(-1, self.nhead, self.head_dim),
            weighted_values,
        )
        result = self.output(aggregate.reshape(nodes, self.d_model))
        return (result, attention) if return_attention else result
