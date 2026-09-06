"""Field-aware global track likelihood structure.

This module does not attach production FaserActs C_prop.  A physical run is
blocked while T11 is failed.  Derivatives G=dh/dα and H=dh/dq are supplied
by the caller at the current anchor; they are not taken from a paired
reference residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from alignment.profiled_information import (
    ProfiledSolve,
    schur_normal,
    solve_profiled,
    weight_matrix,
)
from datasets.alignment_measurements import AlignmentMeasurement, require_measurement_ids


TRACK_STATE_NAMES = ("x_mm", "y_mm", "tx", "ty", "q_over_p_per_mev")
ALIGNMENT_NATIVE_NAMES = (
    "ift_dx_mm",
    "ift_dy_mm",
    "ift_dz_mm",
    "ift_rx_mrad",
    "ift_ry_mrad",
    "ift_rz_mrad",
    "C_dx_mm",
)


class FieldLikelihoodError(ValueError):
    """Raised when a physical likelihood is requested without a T11 pass."""


@dataclass(frozen=True)
class TrackBlock:
    measurements: tuple[AlignmentMeasurement, ...]
    residual: np.ndarray
    g: np.ndarray
    h: np.ndarray
    covariance: np.ndarray
    transport_f: np.ndarray | None = None
    process_noise_q: np.ndarray | None = None


def refuse_physical_run(*, t11_contract_established: bool) -> None:
    if not t11_contract_established:
        raise FieldLikelihoodError(
            "T11 transport contract is not established; "
            "refusing a physical measurement-level pilot that would use unverified C_prop"
        )


def validate_block(block: TrackBlock) -> None:
    for item in block.measurements:
        require_measurement_ids(item)
        if item.uses_production_c_prop:
            raise FieldLikelihoodError("block uses production C_prop")
    if block.residual.shape[0] != block.g.shape[0] or block.g.shape[0] != block.h.shape[0]:
        raise FieldLikelihoodError("G, H, and residual row counts must match")
    if block.h.shape[1] < 5:
        raise FieldLikelihoodError("track q must include x, y, tx, ty, q/p")


def profile_track(block: TrackBlock) -> ProfiledSolve:
    validate_block(block)
    return solve_profiled(block.g, block.h, block.residual, block.covariance)


def accumulate_alignment_normal(blocks: Sequence[TrackBlock]) -> dict[str, Any]:
    normals = []
    rhs = []
    for block in blocks:
        validate_block(block)
        weight = weight_matrix(block.covariance)
        n_alpha, projector = schur_normal(block.g, block.h, weight)
        normals.append(n_alpha)
        rhs.append(block.g.T @ projector @ block.residual)
    normal = np.sum(normals, axis=0)
    vector = np.sum(rhs, axis=0)
    return {
        "normal": normal,
        "rhs": vector,
        "n_tracks": len(blocks),
        "uses_production_c_prop": False,
    }
