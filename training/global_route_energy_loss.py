"""Exact set-packing loss interface on a versioned route-energy table.

This is not a larger local margin.  Training compares a target feasible set
with the strongest loss-augmented feasible set under the same unit-capacity
solver that inference uses.  Gradients flow through route energies only; the
MILP remains a discrete oracle.

The helper is intentionally unused by current V4/V5/W64 training.  New
explicit route-energy experiments must call this interface instead of
``packing_route_competition_loss``.
"""

from __future__ import annotations

from typing import Hashable, Sequence

import numpy as np
import torch

from baselines.route_assignment import solve_unit_capacity_route_packing
from evaluation.route_counterfactuals import inclusion_gap
from models.route_energy import RouteEnergyTable, require_finite


def hamming_loss_augmentation(target_mask: Sequence[bool]) -> np.ndarray:
    """Per-route addend for Hamming ``Δ(y, y*) = ||y - y*||_1``.

    Maximising ``U(y) + Δ`` is equivalent to adding ``1 - 2 y*_r`` to each
    route energy.  The constant ``||y*||_1`` is restored in the hinge.
    """
    target = np.asarray(list(target_mask), dtype=np.bool_)
    if target.ndim != 1:
        raise ValueError("target_mask must be one-dimensional")
    return (1.0 - 2.0 * target.astype(np.float64)).astype(np.float64)


def inclusion_gap_hinge(
    table: RouteEnergyTable,
    target_route_ids: Sequence[int],
    *,
    margin: float = 0.0,
) -> dict[str, float]:
    """Hinge on the exact inclusion gap of each target route.

    ``L = mean relu(margin - M_r)``.  A positive local single-rival margin
    does not reduce this loss when a compatible rival *set* still wins.
    """
    gap_margin = require_finite("margin", margin)
    if not target_route_ids:
        raise ValueError("inclusion-gap hinge requires at least one target route")
    gaps = []
    violations = 0
    for route_id in target_route_ids:
        gap = inclusion_gap(table, route_id).inclusion_gap
        gaps.append(float(gap))
        violations += int(gap < gap_margin)
    loss = float(np.mean([max(gap_margin - gap, 0.0) for gap in gaps]))
    return {
        "loss": loss,
        "mean_inclusion_gap": float(np.mean(gaps)),
        "min_inclusion_gap": float(np.min(gaps)),
        "violation_fraction": float(violations / len(gaps)),
        "n_targets": float(len(gaps)),
    }


def loss_augmented_structured_hinge(
    energies: torch.Tensor,
    endpoints: Sequence[Sequence[Hashable]],
    target_mask: Sequence[bool],
    *,
    augmentation: np.ndarray | None = None,
    margin: float = 1.0,
) -> dict[str, object]:
    """Exact structured hinge ``relu(U(Ŷ) + Δ(Ŷ, Y*) − U(Y*))``.

    ``Ŷ`` maximises ``U + margin * Δ`` over the same unit-capacity feasible
    sets used at inference.  ``augmentation`` defaults to Hamming addends.
    """
    if energies.ndim != 1:
        raise ValueError("route energies must be a vector")
    rows = [tuple(row) for row in endpoints]
    if int(energies.numel()) != len(rows):
        raise ValueError("route energies must align with endpoints")
    target = np.asarray(list(target_mask), dtype=np.bool_)
    if target.shape != (len(rows),):
        raise ValueError("target_mask must align with endpoints")
    if int(np.count_nonzero(target)) == 0:
        zero = energies.sum() * 0.0
        return {
            "loss": zero,
            "raw_margin": 0.0,
            "competitor_selected": np.zeros(len(rows), dtype=bool),
            "target_utility": 0.0,
            "competitor_utility": 0.0,
        }
    scale = require_finite("margin", margin)
    if scale < 0.0:
        raise ValueError("structured-hinge margin must be non-negative")
    addends = hamming_loss_augmentation(target) if augmentation is None else np.asarray(augmentation, dtype=np.float64)
    if addends.shape != (len(rows),) or not np.isfinite(addends).all():
        raise ValueError("loss augmentation must be a finite per-route vector")
    detached = energies.detach().to("cpu", dtype=torch.float64).numpy()
    if not np.isfinite(detached).all():
        raise ValueError("route energies must be finite")
    competitor = solve_unit_capacity_route_packing(rows, detached + scale * addends).selected
    target_tensor = torch.as_tensor(target, dtype=energies.dtype, device=energies.device)
    competitor_tensor = torch.as_tensor(competitor, dtype=energies.dtype, device=energies.device)
    target_utility = torch.sum(energies * target_tensor)
    competitor_utility = torch.sum(energies * competitor_tensor)
    hamming_constant = float(np.count_nonzero(target)) if augmentation is None else 0.0
    delta_hat = scale * (
        float(np.dot(addends, competitor.astype(np.float64))) + hamming_constant
    )
    raw_margin = competitor_utility + energies.new_tensor(delta_hat) - target_utility
    return {
        "loss": torch.relu(raw_margin),
        "raw_margin": float(raw_margin.detach().cpu()),
        "competitor_selected": np.asarray(competitor, dtype=bool),
        "target_utility": float(target_utility.detach().cpu()),
        "competitor_utility": float(competitor_utility.detach().cpu()),
    }
