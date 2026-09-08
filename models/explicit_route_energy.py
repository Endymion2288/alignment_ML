"""Workbook-79 Arm B: low-capacity raw physical route-energy scorer.

The scorer emits one finite raw energy per 2/3/4-station route.  It does not
apply sigmoid, clip, or logit, and it does not read W64 latents.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from training.geometry_aware_transformer import EDGE_FEATURE_NAMES


MODEL_CONTRACT = "explicit_route_energy_b_v1"
FEATURE_VERSION = "raw_physical_route_v1"
HIDDEN_WIDTH = 64
STATION_PATH = (0, 1, 2, 3)
ADJACENT_PAIRS = ((0, 1), (1, 2), (2, 3))
N_EDGE_SLOTS = len(ADJACENT_PAIRS)
N_NODE_SLOTS = len(STATION_PATH)
NODE_STATE_NAMES = ("x_mm", "y_mm", "tx", "ty")


def route_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for source, target in ADJACENT_PAIRS:
        prefix = f"e{source}{target}_"
        names.extend(prefix + name for name in EDGE_FEATURE_NAMES)
    for station in STATION_PATH:
        names.extend(f"s{station}_{name}" for name in NODE_STATE_NAMES)
    names.append("route_length")
    names.extend(f"missing_station_{station}" for station in STATION_PATH)
    return tuple(names)


FEATURE_DIM = len(route_feature_names())
if FEATURE_DIM != N_EDGE_SLOTS * len(EDGE_FEATURE_NAMES) + N_NODE_SLOTS * len(NODE_STATE_NAMES) + 1 + N_NODE_SLOTS:
    raise RuntimeError("raw_physical_route_v1 feature layout is inconsistent")


class ExplicitRouteEnergyScorer(nn.Module):
    """Two-layer MLP, width 64, raw energy out.  No probability head."""

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_width: int = HIDDEN_WIDTH) -> None:
        super().__init__()
        if int(feature_dim) != FEATURE_DIM:
            raise ValueError(f"Arm B feature_dim must be {FEATURE_DIM}")
        if int(hidden_width) > 128 or int(hidden_width) < 1:
            raise ValueError("Arm B hidden width must be in [1, 128]")
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


def raw_route_utilities(
    cores: torch.Tensor,
    n_stations: Sequence[int],
    unmatched_penalty: float,
) -> torch.Tensor:
    """``U(r) = core(r) + n_stations * unmatched_penalty``.  No clip."""
    if cores.ndim != 1 or int(cores.numel()) != len(n_stations):
        raise ValueError("cores must align with n_stations")
    penalty = float(unmatched_penalty)
    if not torch.isfinite(cores).all():
        raise ValueError("route cores must be finite")
    stations = torch.as_tensor(list(n_stations), dtype=cores.dtype, device=cores.device)
    return cores + stations * cores.new_tensor(penalty)
