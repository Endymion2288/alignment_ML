"""Canonical flat ROOT schema for FASER local tracklets.

Every row in the "tracklets" tree represents one reconstructed local
tracklet. State order is always [x_mm, y_mm, tx, ty] and covariance entries
refer to that exact order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CANONICAL_TREE_NAME = "tracklets"
SCHEMA_VERSION = "faser-tracklets-v1"
STATE_FIELDS = ("x_mm", "y_mm", "tx", "ty")

# Upper-triangular elements of Cov[x_mm, y_mm, tx, ty].
COVARIANCE_FIELDS = (
    "cov_xx_mm2",
    "cov_xy_mm2",
    "cov_xtx_mm",
    "cov_xty_mm",
    "cov_yy_mm2",
    "cov_ytx_mm",
    "cov_yty_mm",
    "cov_txtx",
    "cov_txty",
    "cov_tyty",
)

REQUIRED_TRACKLET_FIELDS = (
    "run_id",
    "event_id",
    "station_id",
    "tracklet_id",
    "x_mm",
    "y_mm",
    "z_mm",
    "tx",
    "ty",
    *COVARIANCE_FIELDS,
    "chi2",
    "ndof",
    "n_hit",
    "hit_pattern",
)

# MC labels are required only by association-supervised stages.
MC_LABEL_FIELDS = (
    "truth_particle_id",
    "truth_pdg",
    "truth_match_fraction",
)

OPTIONAL_TRACKLET_FIELDS = (
    "module_ids",
    "raw_hit_ids",
    "source_file_id",
    # Synthetic overlays retain the pre-overlay tracklet identity so a
    # field-aware prediction exported from the same physical payload can be
    # associated with each synthetic source tracklet without a coordinate
    # approximation.
    "origin_run_id",
    "origin_event_id",
    "origin_tracklet_id",
    # Synthetic background provenance is evaluation-only.  Keeping it in the
    # file makes the fake/missing audit reproducible without exposing it to
    # the association model.
    "synthetic_role",
    "synthetic_hard_anchor_station",
    "synthetic_hard_anchor_chi2",
    # These are optional because the MC22 electron smoke-test exports predate
    # the q/p audit branches.  Consumers must therefore tolerate their absence.
    "q_over_p_per_mev",
    "q_over_p_from_momentum_per_mev",
    "q_over_p_variance_per_mev2",
    "has_q_over_p_covariance",
    "truth_q_over_p_per_mev",
)


class DatasetSchemaError(ValueError):
    """Raised when a file cannot safely be interpreted as canonical tracklets."""


@dataclass(frozen=True)
class SchemaReport:
    tree_name: str
    available_fields: tuple[str, ...]
    missing_required_fields: tuple[str, ...]
    missing_mc_label_fields: tuple[str, ...]
    missing_optional_fields: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.missing_required_fields

    @property
    def has_mc_labels(self) -> bool:
        return not self.missing_mc_label_fields

    def require_valid(self, require_mc_labels: bool = False) -> None:
        problems: list[str] = []
        if self.missing_required_fields:
            problems.append(
                "missing required fields: " + ", ".join(self.missing_required_fields)
            )
        if require_mc_labels and self.missing_mc_label_fields:
            problems.append(
                "missing MC label fields: " + ", ".join(self.missing_mc_label_fields)
            )
        if problems:
            raise DatasetSchemaError("; ".join(problems))


def _field_names(tree: object) -> tuple[str, ...]:
    """Return ROOT branch names without importing uproot at module import time."""
    keys = getattr(tree, "keys")()
    return tuple(str(key).split(";")[0] for key in keys)


def validate_tracklet_tree(tree: object) -> SchemaReport:
    """Check a ROOT tree against the canonical input contract."""
    available = _field_names(tree)
    present = set(available)
    return SchemaReport(
        tree_name=str(getattr(tree, "name", CANONICAL_TREE_NAME)).split(";")[0],
        available_fields=available,
        missing_required_fields=tuple(
            field for field in REQUIRED_TRACKLET_FIELDS if field not in present
        ),
        missing_mc_label_fields=tuple(
            field for field in MC_LABEL_FIELDS if field not in present
        ),
        missing_optional_fields=tuple(
            field for field in OPTIONAL_TRACKLET_FIELDS if field not in present
        ),
    )


def covariance_from_columns(columns: dict[str, np.ndarray]) -> np.ndarray:
    """Build a symmetric (n, 4, 4) covariance tensor from flat ROOT columns."""
    n_rows = len(columns["cov_xx_mm2"])
    covariance = np.empty((n_rows, 4, 4), dtype=np.float64)
    covariance[:, 0, 0] = columns["cov_xx_mm2"]
    covariance[:, 0, 1] = covariance[:, 1, 0] = columns["cov_xy_mm2"]
    covariance[:, 0, 2] = covariance[:, 2, 0] = columns["cov_xtx_mm"]
    covariance[:, 0, 3] = covariance[:, 3, 0] = columns["cov_xty_mm"]
    covariance[:, 1, 1] = columns["cov_yy_mm2"]
    covariance[:, 1, 2] = covariance[:, 2, 1] = columns["cov_ytx_mm"]
    covariance[:, 1, 3] = covariance[:, 3, 1] = columns["cov_yty_mm"]
    covariance[:, 2, 2] = columns["cov_txtx"]
    covariance[:, 2, 3] = covariance[:, 3, 2] = columns["cov_txty"]
    covariance[:, 3, 3] = columns["cov_tyty"]
    return covariance


def covariance_columns(covariance: np.ndarray) -> dict[str, np.ndarray]:
    """Flatten a symmetric (n, 4, 4) covariance tensor for ROOT output."""
    values = np.asarray(covariance, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 4):
        raise ValueError("covariance must have shape (n, 4, 4)")
    return {
        "cov_xx_mm2": values[:, 0, 0],
        "cov_xy_mm2": values[:, 0, 1],
        "cov_xtx_mm": values[:, 0, 2],
        "cov_xty_mm": values[:, 0, 3],
        "cov_yy_mm2": values[:, 1, 1],
        "cov_ytx_mm": values[:, 1, 2],
        "cov_yty_mm": values[:, 1, 3],
        "cov_txtx": values[:, 2, 2],
        "cov_txty": values[:, 2, 3],
        "cov_tyty": values[:, 3, 3],
    }


def required_fields(require_mc_labels: bool = False) -> tuple[str, ...]:
    """Return field names that must be read for the requested stage."""
    if require_mc_labels:
        return (*REQUIRED_TRACKLET_FIELDS, *MC_LABEL_FIELDS)
    return REQUIRED_TRACKLET_FIELDS


def assert_finite_covariances(covariance: np.ndarray) -> None:
    """Reject invalid covariance payloads before any matching is attempted."""
    values = np.asarray(covariance, dtype=np.float64)
    if not np.isfinite(values).all():
        raise DatasetSchemaError("covariance contains non-finite values")
    if not np.allclose(values, np.swapaxes(values, 1, 2), rtol=1e-7, atol=1e-12):
        raise DatasetSchemaError("covariance is not symmetric")
    diagonal = np.diagonal(values, axis1=1, axis2=2)
    if np.any(diagonal <= 0.0):
        raise DatasetSchemaError("covariance diagonal must be strictly positive")
