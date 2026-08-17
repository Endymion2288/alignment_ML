"""Field-aware route-consistency observables for physical alignment loops.

The route solver chooses an ordered chain of existing mode-0 ACTS candidate
edges.  This module turns those chosen edges into an event-level consistency
record without replacing the already validated propagation with a straight-line
approximation.  A sum of edge chi-square values is useful as a *diagnostic*
only: neighbouring edges share a local segment and are therefore not an
independent global likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from baselines.field_chi2_matching import FieldCandidate


@dataclass(frozen=True)
class FieldAwareRouteEdge:
    """One selected adjacent edge with its original ACTS residual payload."""

    source_index: int
    target_index: int
    source_station_id: int
    target_station_id: int
    score: float
    chi2: float
    residual: np.ndarray
    pull: np.ndarray
    combined_covariance: np.ndarray


@dataclass(frozen=True)
class FieldAwareRouteConsistency:
    """Selected route represented by its physical adjacent ACTS edges."""

    endpoint_indices: tuple[int, ...]
    endpoint_stations: tuple[int, ...]
    edges: tuple[FieldAwareRouteEdge, ...]

    @property
    def edge_chi2_sum(self) -> float:
        return float(sum(edge.chi2 for edge in self.edges))

    @property
    def edge_count(self) -> int:
        return len(self.edges)


def candidate_lookup(
    candidates_by_pair: Mapping[tuple[int, int], Sequence[FieldCandidate]],
) -> dict[tuple[int, int, int, int], FieldCandidate]:
    """Index physical candidate edges by station pair and event-local endpoints."""
    result: dict[tuple[int, int, int, int], FieldCandidate] = {}
    for raw_pair, candidates in candidates_by_pair.items():
        pair = (int(raw_pair[0]), int(raw_pair[1]))
        if pair[0] >= pair[1]:
            raise ValueError("field-aware route candidate pair must be forward ordered")
        for candidate in candidates:
            if (int(candidate.source_station), int(candidate.target_station)) != pair:
                raise ValueError("candidate station pair does not match its lookup group")
            key = (pair[0], pair[1], int(candidate.source_index), int(candidate.target_index))
            if key in result:
                raise ValueError("duplicate physical candidate endpoints in one route event")
            result[key] = candidate
    return result


def selected_field_aware_route(
    route: object,
    candidates: Mapping[tuple[int, int, int, int], FieldCandidate],
) -> FieldAwareRouteConsistency:
    """Recover the exact physical ACTS edges used by one selected route.

    ``route.matches`` is created by the unit-capacity route solver from the
    same candidate graph, so a missing lookup is a contract violation rather
    than an invitation to make a new propagation or coordinate approximation.
    """
    try:
        raw_endpoints = tuple(route.endpoints)
        raw_matches = tuple(route.matches)
    except AttributeError as error:
        raise ValueError("route lacks endpoints or selected adjacent matches") from error
    if len(raw_endpoints) < 2 or len(raw_matches) != len(raw_endpoints) - 1:
        raise ValueError("route does not contain a contiguous selected edge chain")
    endpoint_stations = tuple(int(station) for station, _ in raw_endpoints)
    endpoint_indices = tuple(int(index) for _, index in raw_endpoints)
    if len(set(endpoint_indices)) != len(endpoint_indices):
        raise ValueError("route repeats an event-local endpoint")

    edges: list[FieldAwareRouteEdge] = []
    for edge_index, (raw_pair, match) in enumerate(raw_matches):
        pair = (int(raw_pair[0]), int(raw_pair[1]))
        expected_source_station = endpoint_stations[edge_index]
        expected_target_station = endpoint_stations[edge_index + 1]
        expected_source_index = endpoint_indices[edge_index]
        expected_target_index = endpoint_indices[edge_index + 1]
        if pair != (expected_source_station, expected_target_station):
            raise ValueError("route match station pair disagrees with route endpoints")
        try:
            source_index = int(match.source_index)
            target_index = int(match.target_index)
            score = float(match.score)
        except AttributeError as error:
            raise ValueError("route match lacks source/target endpoints or score") from error
        if (source_index, target_index) != (expected_source_index, expected_target_index):
            raise ValueError("route match endpoints disagree with route endpoints")
        candidate = candidates.get((pair[0], pair[1], source_index, target_index))
        if candidate is None:
            raise ValueError("selected route edge is absent from the physical candidate graph")
        if not np.isfinite(score) or not np.isfinite(candidate.chi2) or candidate.chi2 < 0.0:
            raise ValueError("selected route edge score or chi2 is invalid")
        residual = np.asarray(candidate.residual, dtype=np.float64)
        pull = np.asarray(candidate.pull, dtype=np.float64)
        covariance = np.asarray(candidate.combined_covariance, dtype=np.float64)
        if residual.shape != (4,) or pull.shape != (4,) or not np.isfinite(residual).all() or not np.isfinite(pull).all():
            raise ValueError("selected route edge state residual is invalid")
        if covariance.shape != (4, 4) or not np.isfinite(covariance).all():
            raise ValueError("selected route edge covariance is invalid")
        if not np.allclose(covariance, covariance.T, rtol=1.0e-7, atol=1.0e-12):
            raise ValueError("selected route edge covariance is not symmetric")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError("selected route edge covariance is not positive definite") from error
        edges.append(
            FieldAwareRouteEdge(
                source_index=source_index,
                target_index=target_index,
                source_station_id=pair[0],
                target_station_id=pair[1],
                score=score,
                chi2=float(candidate.chi2),
                residual=residual,
                pull=pull,
                combined_covariance=covariance,
            )
        )
    return FieldAwareRouteConsistency(
        endpoint_indices=endpoint_indices,
        endpoint_stations=endpoint_stations,
        edges=tuple(edges),
    )


def field_aware_route_summary(
    routes: Sequence[FieldAwareRouteConsistency],
) -> dict[str, float | int | str | None]:
    """Summarize selected physical edges without treating them as independent."""
    edges = [edge for route in routes for edge in route.edges]
    chi2 = np.asarray([edge.chi2 for edge in edges], dtype=np.float64)
    return {
        "routes": len(routes),
        "complete_four_station_routes": int(
            sum(len(route.endpoint_indices) == 4 for route in routes)
        ),
        "selected_field_aware_edges": len(edges),
        "edge_chi2_sum": float(np.sum(chi2)) if chi2.size else 0.0,
        "edge_chi2_mean": float(np.mean(chi2)) if chi2.size else None,
        "edge_chi2_median": float(np.median(chi2)) if chi2.size else None,
        "statistical_interpretation": (
            "Adjacent edge residuals share local tracklets; edge_chi2_sum is a route-consistency diagnostic, "
            "not an independent global likelihood."
        ),
    }
