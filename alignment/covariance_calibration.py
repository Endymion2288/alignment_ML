"""Auditable diagonal calibration of mode-0 propagated covariances.

The mode-0 ACTS exporter's propagated covariance is mis-calibrated and
anisotropic (see the iteration-0 mechanism diagnosis): x/tx variances are
underestimated while y/ty variances are vastly overestimated.  This module
derives per-(station-pair, component) diagonal scale factors from train-only
truth-matched pull widths and applies them to loaded propagation records as
``C' = D^(1/2) C D^(1/2)`` with ``D_cc = scale_c``.  Factors are derived only
from the train split; applying a frozen factor set to any split is a pure
covariance transformation that never changes candidate endpoint definitions,
residuals, route selections, or truth labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from datasets.propagation_loader import PropagationRecords

SCHEMA_VERSION = "faser-mode0-covariance-calibration-v1"
COMPONENTS = ("x_mm", "y_mm", "tx", "ty")


@dataclass(frozen=True)
class CovarianceCalibration:
    """Frozen per-(station-pair, component) diagonal variance scale factors."""

    factors: dict[tuple[int, int], dict[str, float]]
    provenance: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "factors": {
                f"{pair[0]}->{pair[1]}": dict(components)
                for pair, components in sorted(self.factors.items())
            },
            "provenance": dict(self.provenance),
        }

    @staticmethod
    def from_json(payload: Mapping[str, Any]) -> "CovarianceCalibration":
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported covariance calibration schema")
        factors = {}
        for label, components in payload.get("factors", {}).items():
            source, _, target = str(label).partition("->")
            pair = (int(source), int(target))
            row = {}
            for component, scale in components.items():
                if str(component) not in COMPONENTS:
                    raise ValueError(f"unknown calibration component '{component}'")
                value = float(scale)
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError(f"calibration scale for {label}/{component} must be positive")
                row[str(component)] = value
            factors[pair] = row
        if not factors:
            raise ValueError("covariance calibration has no factors")
        return CovarianceCalibration(factors=factors, provenance=payload.get("provenance", {}))

    def scale_matrix(self, source_station: int, target_station: int) -> np.ndarray:
        components = self.factors.get((int(source_station), int(target_station)))
        if components is None:
            return np.eye(4)
        diagonal = np.asarray(
            [components.get(component, 1.0) for component in COMPONENTS], dtype=np.float64
        )
        return np.diag(diagonal)


def derive_from_pull_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    width_field: str = "robust_sigma",
    provenance: Mapping[str, Any],
) -> CovarianceCalibration:
    """Freeze variance scale factors from pooled pull widths (width^2)."""
    factors: dict[tuple[int, int], dict[str, float]] = {}
    for row in rows:
        if str(row.get("component")) not in COMPONENTS:
            continue
        width = row.get(width_field)
        if width is None:
            raise ValueError(f"pull row lacks '{width_field}': {row}")
        width = float(width)
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError(f"pull width must be positive: {row}")
        pair_label = str(row["station_pair"])
        source, _, target = pair_label.partition("->")
        pair = (int(source), int(target))
        factors.setdefault(pair, {})[str(row["component"])] = width * width
    if not factors:
        raise ValueError("no calibration factors derivable from the pull rows")
    return CovarianceCalibration(factors=factors, provenance=provenance)


def apply_to_records(
    records: PropagationRecords, calibration: CovarianceCalibration
) -> PropagationRecords:
    """Return records with diagonally rescaled covariances (records unchanged)."""
    covariance = np.asarray(records.covariance, dtype=np.float64).copy()
    for pair, components in calibration.factors.items():
        mask = (records.source_station_id == pair[0]) & (records.target_station_id == pair[1])
        if not np.any(mask):
            continue
        diagonal = np.asarray(
            [components.get(component, 1.0) for component in COMPONENTS], dtype=np.float64
        )
        d_half = np.sqrt(diagonal)
        blocks = covariance[mask]
        covariance[mask] = blocks * d_half[None, :, None] * d_half[None, None, :]
    return replace(records, covariance=covariance)


def load_covariance_calibration(path: str | Path) -> CovarianceCalibration:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return CovarianceCalibration.from_json(json.loads(source.read_text(encoding="utf-8")))


def write_covariance_calibration(calibration: CovarianceCalibration, path: str | Path) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(calibration.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
