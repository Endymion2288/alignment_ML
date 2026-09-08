"""Workbook-80 ablation C: physical route features plus raw W64 edge logits.

This is not wp4 representation-study Arm C.  It is a low-capacity hybrid
energy scorer used only to attribute WB79 Arm B's holdout gap.
"""

from __future__ import annotations

import torch
from torch import nn

from models.explicit_route_energy import FEATURE_DIM, HIDDEN_WIDTH, route_feature_names


MODEL_CONTRACT = "hybrid_route_energy_c_v1"
FEATURE_VERSION = "raw_physical_plus_w64_logits_v1"
W64_LOGIT_NAMES = (
    "w64_logit_e01",
    "w64_logit_e12",
    "w64_logit_e23",
    "w64_logit_sum",
    "w64_n_edges",
)


def hybrid_feature_names() -> tuple[str, ...]:
    return route_feature_names() + W64_LOGIT_NAMES


HYBRID_FEATURE_DIM = len(hybrid_feature_names())
if HYBRID_FEATURE_DIM != FEATURE_DIM + len(W64_LOGIT_NAMES):
    raise RuntimeError("hybrid feature layout drifted from the physical contract")


class HybridRouteEnergyScorer(nn.Module):
    """Same width-64 MLP as Arm B, wider input, raw energy out."""

    def __init__(self, feature_dim: int = HYBRID_FEATURE_DIM, hidden_width: int = HIDDEN_WIDTH) -> None:
        super().__init__()
        if int(feature_dim) != HYBRID_FEATURE_DIM:
            raise ValueError(f"hybrid feature_dim must be {HYBRID_FEATURE_DIM}")
        if int(hidden_width) > 128 or int(hidden_width) < 1:
            raise ValueError("hybrid hidden width must be in [1, 128]")
        width = int(hidden_width)
        self.feature_dim = int(feature_dim)
        self.hidden_width = width
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or int(features.shape[1]) != self.feature_dim:
            raise ValueError(f"features must have shape [routes, {self.feature_dim}]")
        return self.net(features).reshape(-1)
