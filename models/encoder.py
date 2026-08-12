"""Tracklet-state encoder used by the sparse association Transformer."""

from __future__ import annotations

import torch
from torch import nn


class TrackletStateEncoder(nn.Module):
    """Encode persisted local-tracklet state with a learned station embedding.

    The numerical features are standardized by a train-split-only transform
    before entering this module.  Station identity is deliberately represented
    separately so that the encoder does not infer detector ordering from a
    nominal coordinate convention.
    """

    def __init__(
        self,
        node_feature_dim: int,
        d_model: int = 128,
        num_stations: int = 4,
        station_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        if node_feature_dim < 1 or d_model < 1 or num_stations < 1 or station_embedding_dim < 1:
            raise ValueError("encoder dimensions must be positive")
        self.node_feature_dim = int(node_feature_dim)
        self.d_model = int(d_model)
        self.num_stations = int(num_stations)
        self.station_embedding = nn.Embedding(num_stations, station_embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(node_feature_dim + station_embedding_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, node_features: torch.Tensor, station_ids: torch.Tensor) -> torch.Tensor:
        if node_features.ndim != 2 or node_features.shape[1] != self.node_feature_dim:
            raise ValueError("node features do not match the encoder schema")
        if station_ids.ndim != 1 or station_ids.shape[0] != node_features.shape[0]:
            raise ValueError("station IDs do not align with node features")
        if station_ids.dtype != torch.long:
            station_ids = station_ids.to(dtype=torch.long)
        if torch.any(station_ids < 0) or torch.any(station_ids >= self.num_stations):
            raise ValueError("station ID is outside the configured embedding range")
        return self.network(torch.cat((node_features, self.station_embedding(station_ids)), dim=-1))
