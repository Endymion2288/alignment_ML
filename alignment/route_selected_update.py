"""Truth-free route-selected observations for physical alignment iterations.

The frozen association backbone can emit either a diagnostic straight-line
leave-one-out residual or, for the physical alignment objective, the exact
mode-0 ACTS residual of every selected adjacent route edge.  This module
aligns those records across a central payload, central-difference probes, and
a separately refitted reference target using only synthetic/original tracklet
provenance.  It intentionally does not inspect truth particle IDs or MC
association labels.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


STATE_LABELS = ("x_mm", "y_mm", "tx", "ty")
_COVARIANCE_LABELS = (
    "xx_mm2",
    "xy_mm2",
    "xtx_mm",
    "xty_mm",
    "yy_mm2",
    "ytx_mm",
    "yty_mm",
    "txtx",
    "txty",
    "tyty",
)


@dataclass(frozen=True)
class RouteSelectedObservation:
    """One truth-free, selected physical residual observation."""

    key: tuple[object, ...]
    residual: np.ndarray
    covariance: np.ndarray
    chi2: float
    route_endpoint_count: int
    source_station_id: int | None
    target_station_id: int
    source_synthetic_role: int | None
    target_synthetic_role: int
    observation_kind: str


def _float(row: Mapping[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"route observation lacks finite field '{field}'") from error
    if not np.isfinite(value):
        raise ValueError(f"route observation field '{field}' is non-finite")
    return value


def _integer(row: Mapping[str, str], field: str) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"route observation lacks integer field '{field}'") from error
    return value


def _covariance(row: Mapping[str, str]) -> np.ndarray:
    values = {label: _float(row, f"combined_cov_{label}") for label in _COVARIANCE_LABELS}
    matrix = np.asarray(
        [
            [values["xx_mm2"], values["xy_mm2"], values["xtx_mm"], values["xty_mm"]],
            [values["xy_mm2"], values["yy_mm2"], values["ytx_mm"], values["yty_mm"]],
            [values["xtx_mm"], values["ytx_mm"], values["txtx"], values["txty"]],
            [values["xty_mm"], values["yty_mm"], values["txty"], values["tyty"]],
        ],
        dtype=np.float64,
    )
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("route observation combined covariance is not positive definite") from error
    return matrix


def read_route_selected_observations(
    path: str | Path,
    *,
    movable_station_ids: Sequence[int],
) -> dict[tuple[object, ...], RouteSelectedObservation]:
    """Read legacy straight-line endpoint observations for movable stations."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    movable = {int(value) for value in movable_station_ids}
    if not movable:
        raise ValueError("movable_station_ids cannot be empty")
    result: dict[tuple[object, ...], RouteSelectedObservation] = {}
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("route observation CSV has no header")
        for row in reader:
            station = _integer(row, "target_station_id")
            if station not in movable:
                continue
            count = _integer(row, "route_endpoint_count")
            if count < 3:
                raise ValueError("leave-one-out route observation has fewer than three endpoints")
            signature = str(row.get("route_origin_signature", ""))
            sample = str(row.get("sample_id", ""))
            if not signature or not sample:
                raise ValueError("route observation lacks sample or route provenance signature")
            key = (sample, _integer(row, "run_id"), _integer(row, "event_id"), signature, station)
            if key in result:
                raise ValueError("route-selected observation table has duplicate endpoint provenance")
            residual = np.asarray([_float(row, f"residual_{label}") for label in STATE_LABELS], dtype=np.float64)
            result[key] = RouteSelectedObservation(
                key=key,
                residual=residual,
                covariance=_covariance(row),
                chi2=_float(row, "chi2"),
                route_endpoint_count=count,
                source_station_id=None,
                target_station_id=station,
                source_synthetic_role=None,
                target_synthetic_role=_integer(row, "target_synthetic_role"),
                observation_kind="leave_one_out_straight_line_diagnostic",
            )
    return result


def read_route_selected_field_edge_observations(
    path: str | Path,
    *,
    movable_station_ids: Sequence[int],
) -> dict[tuple[object, ...], RouteSelectedObservation]:
    """Read selected, exact field-aware route edges incident on movable stations.

    A residual is always represented on its physical target surface, while a
    movable source station can affect its ACTS propagation.  Therefore an edge
    is retained when *either* endpoint belongs to the active alignment block.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    movable = {int(value) for value in movable_station_ids}
    if not movable:
        raise ValueError("movable_station_ids cannot be empty")
    result: dict[tuple[object, ...], RouteSelectedObservation] = {}
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("field-edge observation CSV has no header")
        for row in reader:
            source_station = _integer(row, "source_station_id")
            target_station = _integer(row, "target_station_id")
            if source_station not in movable and target_station not in movable:
                continue
            count = _integer(row, "route_endpoint_count")
            if count < 2:
                raise ValueError("selected field-edge observation has fewer than two route endpoints")
            signature = str(row.get("route_origin_signature", ""))
            sample = str(row.get("sample_id", ""))
            if not signature or not sample:
                raise ValueError("field-edge observation lacks sample or route provenance signature")
            key = (
                sample,
                _integer(row, "run_id"),
                _integer(row, "event_id"),
                signature,
                source_station,
                target_station,
            )
            if key in result:
                raise ValueError("field-edge observation table has duplicate selected-edge provenance")
            residual = np.asarray(
                [_float(row, f"residual_{label}") for label in STATE_LABELS], dtype=np.float64
            )
            result[key] = RouteSelectedObservation(
                key=key,
                residual=residual,
                covariance=_covariance(row),
                chi2=_float(row, "chi2"),
                route_endpoint_count=count,
                source_station_id=source_station,
                target_station_id=target_station,
                source_synthetic_role=_integer(row, "source_synthetic_role"),
                target_synthetic_role=_integer(row, "target_synthetic_role"),
                observation_kind="selected_field_aware_acts_edge",
            )
    return result


def align_route_selected_observations(
    banks: Sequence[Mapping[tuple[object, ...], RouteSelectedObservation]],
) -> tuple[list[tuple[object, ...]], list[np.ndarray], list[np.ndarray], dict[str, object]]:
    """Intersect provenance keys without selecting or filtering by truth labels."""
    if not banks:
        raise ValueError("at least one route observation bank is required")
    key_sets = [set(bank) for bank in banks]
    common = set.intersection(*key_sets)
    union = set.union(*key_sets)
    if not common:
        raise ValueError("no truth-free selected route endpoint survives every physical refit")
    keys = sorted(common)
    residuals = [np.asarray([bank[key].residual for key in keys], dtype=np.float64) for bank in banks]
    covariances = [np.asarray([bank[key].covariance for key in keys], dtype=np.float64) for bank in banks]
    anchor_count = len(key_sets[0])
    overlap = {
        "observation_count_by_bank": [len(values) for values in key_sets],
        "common_selected_observations": len(common),
        "union_selected_observations": len(union),
        "anchor_common_fraction": None if not anchor_count else float(len(common) / anchor_count),
        "bank_common_fraction": [
            None if not values else float(len(common) / len(values)) for values in key_sets
        ],
    }
    return keys, residuals, covariances, overlap
