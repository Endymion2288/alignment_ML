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

from baselines.field_chi2_matching import build_field_candidates
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events


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
    # Stable identity of the underlying real physical edge:
    # (source_origin_run_id, source_origin_event_id, source_origin_tracklet_id,
    #  target_origin_run_id, target_origin_event_id, target_origin_tracklet_id,
    #  source_station_id, target_station_id).  The overlay may embed one
    #  physical edge into several synthetic events/routes; every replica of the
    #  same physical edge carries the same value here.  None for legacy readers
    #  that do not resolve endpoint provenance.
    physical_edge_key: tuple[object, ...] | None = None


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
            physical_edge_key = None
            if "source_origin_run_id" in row and "target_origin_run_id" in row:
                physical_edge_key = (
                    _integer(row, "source_origin_run_id"),
                    _integer(row, "source_origin_event_id"),
                    _integer(row, "source_origin_tracklet_id"),
                    _integer(row, "target_origin_run_id"),
                    _integer(row, "target_origin_event_id"),
                    _integer(row, "target_origin_tracklet_id"),
                    source_station,
                    target_station,
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
                physical_edge_key=physical_edge_key,
            )
    return result


def read_anchor_selected_field_edge_observations(
    anchor_table_path: str | Path,
    synthetic_tracklets_path: str | Path,
    field_candidates_path: str | Path,
    *,
    movable_station_ids: Sequence[int],
    covariance_calibration: Any = None,
    q_over_p_mode: int = 0,
) -> tuple[dict[tuple[object, ...], RouteSelectedObservation], dict[str, object]]:
    """Measure anchor-selected route edges inside one other physical payload.

    The frozen backbone selects routes once at the anchor payload.  Route
    selection is geometry-dependent, so intersecting per-payload selections
    across finite-difference probes loses essentially every route.  Instead
    this reader keeps the anchor's truth-free route set fixed and recomputes
    each selected edge's exact mode-0 ACTS residual/covariance from the
    candidate graph of the payload under study, matching endpoints only by
    original (pre-overlay) tracklet provenance.  Observations are keyed by the
    anchor table's provenance key so the banks of every payload align.
    """
    anchor_path = Path(anchor_table_path).expanduser().resolve()
    if not anchor_path.is_file():
        raise FileNotFoundError(anchor_path)
    movable = {int(value) for value in movable_station_ids}
    if not movable:
        raise ValueError("movable_station_ids cannot be empty")

    anchor_rows: list[
        tuple[tuple[object, ...], tuple[int, int, int], tuple[int, int, int], int, int, int, int, int]
    ] = []
    with anchor_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("anchor selected-edge table has no header")
        for row in reader:
            source_station = _integer(row, "source_station_id")
            target_station = _integer(row, "target_station_id")
            if source_station not in movable and target_station not in movable:
                continue
            count = _integer(row, "route_endpoint_count")
            if count < 2:
                raise ValueError("anchor-selected field edge has fewer than two route endpoints")
            signature = str(row.get("route_origin_signature", ""))
            sample = str(row.get("sample_id", ""))
            if not signature or not sample:
                raise ValueError("anchor field edge lacks sample or route provenance signature")
            key = (sample, _integer(row, "run_id"), _integer(row, "event_id"), signature, source_station, target_station)
            source_origin = (
                _integer(row, "source_origin_run_id"),
                _integer(row, "source_origin_event_id"),
                _integer(row, "source_origin_tracklet_id"),
            )
            target_origin = (
                _integer(row, "target_origin_run_id"),
                _integer(row, "target_origin_event_id"),
                _integer(row, "target_origin_tracklet_id"),
            )
            anchor_rows.append(
                (
                    key,
                    source_origin,
                    target_origin,
                    count,
                    source_station,
                    target_station,
                    _integer(row, "source_synthetic_role"),
                    _integer(row, "target_synthetic_role"),
                )
            )

    events = load_events(Path(synthetic_tracklets_path).expanduser().resolve(), require_mc_labels=True)
    records = load_propagation_records(Path(field_candidates_path).expanduser().resolve())
    if covariance_calibration is not None:
        from alignment.covariance_calibration import apply_to_records

        records = apply_to_records(records, covariance_calibration)

    # The overlay is generated per payload, so the same original tracklet can
    # land in a different synthetic event in another payload.  Index every
    # embedding of every origin and pair endpoints in any shared event.
    origin_embeddings: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    for event_index, event in enumerate(events):
        if event.origin_run_id is None or event.origin_event_id is None or event.origin_tracklet_id is None:
            raise ValueError("payload synthetic tracklets lack origin provenance fields")
        for row in range(int(event.station_id.shape[0])):
            origin = (
                int(event.origin_run_id[row]),
                int(event.origin_event_id[row]),
                int(event.origin_tracklet_id[row]),
            )
            origin_embeddings.setdefault(origin, []).append((event_index, row))

    event_pair_candidates: dict[tuple[int, int, int], dict[tuple[int, int], object]] = {}

    def _candidate(event_index: int, source_station: int, target_station: int, source_row: int, target_row: int):
        cache_key = (event_index, source_station, target_station)
        if cache_key not in event_pair_candidates:
            candidates = build_field_candidates(
                events[event_index],
                records,
                source_station,
                target_station,
                chi2_gate=None,
                q_over_p_mode=q_over_p_mode,
            )
            event_pair_candidates[cache_key] = {
                (int(candidate.source_index), int(candidate.target_index)): candidate
                for candidate in candidates
            }
        return event_pair_candidates[cache_key].get((source_row, target_row))

    result: dict[tuple[object, ...], RouteSelectedObservation] = {}
    missing_origin = 0
    missing_candidate = 0
    for (
        key,
        source_origin,
        target_origin,
        count,
        source_station,
        target_station,
        source_role,
        target_role,
    ) in anchor_rows:
        source_embeddings = origin_embeddings.get(source_origin, [])
        target_embeddings = origin_embeddings.get(target_origin, [])
        if not source_embeddings or not target_embeddings:
            missing_origin += 1
            continue
        candidate = None
        target_by_event: dict[int, int] = {}
        for event_index, row in target_embeddings:
            target_by_event.setdefault(event_index, row)
        for event_index, source_row in source_embeddings:
            target_row = target_by_event.get(event_index)
            if target_row is None:
                continue
            candidate = _candidate(event_index, source_station, target_station, source_row, target_row)
            if candidate is not None:
                break
        if candidate is None:
            missing_candidate += 1
            continue
        if key in result:
            raise ValueError("anchor-selected observation maps to duplicate payload edges")
        result[key] = RouteSelectedObservation(
            key=key,
            residual=np.asarray(candidate.residual, dtype=np.float64),
            covariance=np.asarray(candidate.combined_covariance, dtype=np.float64),
            chi2=float(candidate.chi2),
            route_endpoint_count=count,
            source_station_id=source_station,
            target_station_id=target_station,
            source_synthetic_role=source_role,
            target_synthetic_role=target_role,
            observation_kind="anchor_selected_field_aware_acts_edge",
            physical_edge_key=(
                *source_origin,
                *target_origin,
                source_station,
                target_station,
            ),
        )
    audit = {
        "anchor_selected_edges": len(anchor_rows),
        "payload_observations": len(result),
        "missing_origin_in_payload": missing_origin,
        "missing_candidate_edge": missing_candidate,
        "anchor_edge_availability": None if not anchor_rows else float(len(result) / len(anchor_rows)),
    }
    return result, audit


OBSERVATION_STATISTICS = (
    "replica_weighted",
    "physical_edge_deduplicated",
    "physical_edge_inverse_multiplicity_weighted",
)


def apply_observation_statistics(
    keys: Sequence[tuple[object, ...]],
    residuals: Sequence[np.ndarray],
    covariances: Sequence[np.ndarray],
    physical_edge_keys: Sequence[tuple[object, ...] | None],
    semantics: str,
) -> tuple[list[tuple[object, ...]], list[np.ndarray], list[np.ndarray], dict[str, object]]:
    """Normalize the statistical weight of overlay-reused physical edges.

    The synthetic overlay embeds one real physical edge into several synthetic
    events, so the frozen route selection can place replica observations of the
    same physical edge (identical origin endpoints and station pair) into the
    normal equation multiple times.  Three pre-defined semantics:

    - ``replica_weighted``: every replica enters independently (legacy control).
    - ``physical_edge_deduplicated``: one deterministic representative per
      physical edge (sorted-first key); replica residuals/covariances are
      audited for near-identity.
    - ``physical_edge_inverse_multiplicity_weighted``: every replica kept with
      its covariance inflated by the replica multiplicity, so each physical
      edge's total statistical weight sums to one.  For identical replicas
      this is exactly equivalent to deduplication in the WLS normal equation.

    No truth information is used anywhere; grouping uses only the recorded
    endpoint provenance.
    """
    if semantics not in OBSERVATION_STATISTICS:
        raise ValueError(f"unknown observation statistics semantics '{semantics}'")
    if len(keys) != len(physical_edge_keys):
        raise ValueError("physical edge keys must align with the observation keys")
    n_banks = len(residuals)
    if len(covariances) != n_banks:
        raise ValueError("residuals and covariances must have the same bank count")

    if semantics == "replica_weighted":
        audit = {
            "semantics": semantics,
            "observations": len(keys),
            "unique_physical_edges": len(set(physical_edge_keys)),
        }
        return list(keys), [np.asarray(r) for r in residuals], [np.asarray(c) for c in covariances], audit

    if any(value is None for value in physical_edge_keys):
        raise ValueError(
            f"observation statistics '{semantics}' requires resolved physical edge "
            "provenance on every observation"
        )
    groups: dict[tuple[object, ...], list[int]] = {}
    for index, physical_key in enumerate(physical_edge_keys):
        groups.setdefault(physical_key, []).append(index)
    multiplicities = [len(indices) for indices in groups.values()]
    multiplicity_histogram = {
        str(multiplicity): multiplicities.count(multiplicity)
        for multiplicity in sorted(set(multiplicities))
    }

    if semantics == "physical_edge_deduplicated":
        representatives = sorted(sorted(indices)[0] for indices in groups.values())
        # Audit replica agreement: replicas of one physical edge re-measure the
        # same tracklets through the same propagation record and should be
        # numerically identical across payloads.
        max_residual_spread = 0.0
        for indices in groups.values():
            if len(indices) < 2:
                continue
            for bank_residuals in residuals:
                values = np.asarray(bank_residuals)[indices]
                spread = float(np.max(np.abs(values - values[0])))
                max_residual_spread = max(max_residual_spread, spread)
        index_array = np.asarray(representatives, dtype=int)
        audit = {
            "semantics": semantics,
            "observations": int(index_array.size),
            "unique_physical_edges": len(groups),
            "replicas_before_deduplication": len(keys),
            "replica_multiplicity_histogram": multiplicity_histogram,
            "max_replica_residual_spread": max_residual_spread,
        }
        return (
            [keys[index] for index in representatives],
            [np.asarray(bank)[index_array] for bank in residuals],
            [np.asarray(bank)[index_array] for bank in covariances],
            audit,
        )

    # physical_edge_inverse_multiplicity_weighted
    multiplicity_per_row = np.asarray(
        [len(groups[physical_key]) for physical_key in physical_edge_keys], dtype=np.float64
    )
    scaled_covariances = [
        np.asarray(bank) * multiplicity_per_row[:, None, None] for bank in covariances
    ]
    audit = {
        "semantics": semantics,
        "observations": len(keys),
        "unique_physical_edges": len(groups),
        "replica_multiplicity_histogram": multiplicity_histogram,
        "total_weight_per_physical_edge": 1.0,
    }
    return list(keys), [np.asarray(bank) for bank in residuals], scaled_covariances, audit


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
