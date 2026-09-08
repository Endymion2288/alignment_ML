"""Workbook-81b Arm D: linear physics residual on frozen W64 energy.

This is not wp4 Arm C, not WB79 Arm B, and not WB80 hybrid C.
g_θ is a single linear map.  ΔU = 0.25 * tanh(g_θ(x_phys)).
"""

from __future__ import annotations

import torch
from torch import nn

from models.explicit_route_energy import FEATURE_DIM
from training.wb81_calibration_contract import DELTA_MAX


MODEL_CONTRACT = "physics_constrained_calibration_d_v1"
FEATURE_VERSION = "raw_physical_route_v1"


class PhysicsConstrainedCalibrator(nn.Module):
    """Linear g_θ on 54-D physical features.  Zero init is exact identity."""

    def __init__(self, feature_dim: int = FEATURE_DIM) -> None:
        super().__init__()
        if int(feature_dim) != FEATURE_DIM:
            raise ValueError(f"calibrator feature_dim must be {FEATURE_DIM}")
        self.feature_dim = int(feature_dim)
        self.delta_max = DELTA_MAX
        self.g = nn.Linear(self.feature_dim, 1, bias=True)
        nn.init.zeros_(self.g.weight)
        nn.init.zeros_(self.g.bias)

    def raw_scores(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or int(features.shape[1]) != self.feature_dim:
            raise ValueError(f"features must have shape [routes, {self.feature_dim}]")
        return self.g(features).reshape(-1)

    def delta(self, features: torch.Tensor) -> torch.Tensor:
        return self.delta_max * torch.tanh(self.raw_scores(features))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.delta(features)
