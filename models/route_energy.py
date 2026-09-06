"""Versioned route-energy contract for exact unit-capacity packing.

Historical V2/V4/V5 complete-route decoding converts a route score through
``sigmoid → clip → logit`` while fragments keep a sum of per-edge clipped
logits.  Those two maps do not commute:

    clip(z1 + z2 + z3)  ≠  clip(z1) + clip(z2) + clip(z3)

so a zero route correction can change the exact packing.  New experiments
must use ``raw_energy_v1``: every 2/3/4-station hypothesis carries a raw
energy, the solver maximises that energy, and probability is never an
intermediate quantity.  ``legacy_prob_clip_logit_v1`` remains only so
historical assignments can be replayed without rewriting artifacts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np

from baselines.route_assignment import UnitCapacityPackingResult, solve_unit_capacity_route_packing


CANONICAL_ENERGY_VERSION = "raw_energy_v1"
LEGACY_DECODER_VERSION = "legacy_prob_clip_logit_v1"
SOLVER_PROBABILITY_FLOOR = 1.0e-6
LOGIT_CLIP = math.log((1.0 - SOLVER_PROBABILITY_FLOOR) / SOLVER_PROBABILITY_FLOOR)
DUSTBIN_ENERGY = 0.0
SUPPORTED_ENERGY_VERSIONS = frozenset({CANONICAL_ENERGY_VERSION, LEGACY_DECODER_VERSION})


def require_finite(name: str, value: float) -> float:
    """Reject NaN/Inf before they can enter a solver table."""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite float, got {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite float, got {value!r}")
    return number


def require_finite_sequence(name: str, values: Sequence[float]) -> tuple[float, ...]:
    if values is None:
        raise ValueError(f"{name} must be a finite sequence")
    return tuple(require_finite(f"{name}[{index}]", value) for index, value in enumerate(values))


def route_kind(n_stations: int) -> str:
    stations = int(n_stations)
    if stations == 2:
        return "pair"
    if stations == 3:
        return "fragment3"
    if stations == 4:
        return "complete4"
    if stations < 2:
        raise ValueError("a route requires at least two stations")
    return f"length{stations}"


def canonical_route_energy(
    edge_energies: Sequence[float],
    *,
    unmatched_penalty: float,
    n_stations: int | None = None,
    correction: float = 0.0,
) -> float:
    """Raw energy: ``sum(edge energies) + correction + n_stations * penalty``.

    No sigmoid, clip, or logit is applied.  ``n_stations`` defaults to
    ``len(edge_energies) + 1`` and must match that identity when supplied.
    """
    edges = require_finite_sequence("edge_energies", edge_energies)
    if not edges:
        raise ValueError("edge_energies must contain one energy per adjacent hop")
    penalty = require_finite("unmatched_penalty", unmatched_penalty)
    delta = require_finite("correction", correction)
    stations = len(edges) + 1 if n_stations is None else int(n_stations)
    if stations != len(edges) + 1:
        raise ValueError("n_stations must equal len(edge_energies) + 1")
    return float(sum(edges) + delta + stations * penalty)


def legacy_clip_logit(raw_logit: float) -> float:
    """``logit(clip(sigmoid(z), ε, 1-ε))`` for finite real ``z``.

    This is exactly ``clamp(z, -LOGIT_CLIP, LOGIT_CLIP)``.  Historical
    complete-route decode applies it to the *sum* of edge logits; fragments
    apply it per edge.  Those two uses are not interchangeable.
    """
    return float(min(max(require_finite("raw_logit", raw_logit), -LOGIT_CLIP), LOGIT_CLIP))


def legacy_fragment_energy(
    edge_logits: Sequence[float],
    *,
    unmatched_penalty: float,
    n_stations: int | None = None,
) -> float:
    """Historical W64 / fragment path: per-edge clip, then the packing offset."""
    clipped = tuple(legacy_clip_logit(logit) for logit in require_finite_sequence("edge_logits", edge_logits))
    return canonical_route_energy(clipped, unmatched_penalty=unmatched_penalty, n_stations=n_stations)


def legacy_complete_replace_energy(
    route_logit: float,
    *,
    unmatched_penalty: float,
    n_stations: int = 4,
) -> float:
    """Historical complete-route replace decode: clip the *joint* logit once."""
    stations = int(n_stations)
    if stations < 2:
        raise ValueError("a complete-route replace energy requires at least two stations")
    penalty = require_finite("unmatched_penalty", unmatched_penalty)
    return float(legacy_clip_logit(route_logit) + stations * penalty)


def legacy_complete_residual_energy(
    edge_logits: Sequence[float],
    route_logit: float,
    *,
    weight: float,
    unmatched_penalty: float,
) -> float:
    """Historical residual mix: ``U_edge + w * (U_replace_core - U_edge_core)``.

    ``weight = 0`` recovers the per-edge clipped fragment/complete identity.
    ``weight = 1`` recovers replace decode.
    """
    edges = require_finite_sequence("edge_logits", edge_logits)
    mix = require_finite("weight", weight)
    if mix < 0.0:
        raise ValueError("complete_route_context_weight must be finite and non-negative")
    edge_core = float(sum(legacy_clip_logit(logit) for logit in edges))
    route_core = legacy_clip_logit(route_logit)
    mixed = edge_core + mix * (route_core - edge_core)
    penalty = require_finite("unmatched_penalty", unmatched_penalty)
    return float(mixed + (len(edges) + 1) * penalty)


def diagnostic_probability(energy: float) -> float:
    """Optional reporting map.  Never consumed by the packing solver."""
    value = require_finite("energy", energy)
    clipped = min(max(value, -80.0), 80.0)
    return float(1.0 / (1.0 + math.exp(-clipped)))


@dataclass(frozen=True)
class RouteEnergyRecord:
    """One solver-facing hypothesis in a versioned energy table."""

    route_id: int
    endpoints: tuple[Hashable, ...]
    energy: float
    n_stations: int
    kind: str
    contract: str
    correction: float = 0.0
    edge_energies: tuple[float, ...] = ()
    diagnostic_probability: float | None = None

    def __post_init__(self) -> None:
        if int(self.route_id) < 0:
            raise ValueError("route_id must be non-negative")
        if not self.endpoints:
            raise ValueError("every route must contain at least one endpoint")
        if len(set(self.endpoints)) != len(self.endpoints):
            raise ValueError("one route repeats an endpoint")
        require_finite("energy", self.energy)
        require_finite("correction", self.correction)
        if int(self.n_stations) != len(self.endpoints):
            raise ValueError("n_stations must equal the number of endpoints")
        if self.kind != route_kind(self.n_stations):
            raise ValueError(f"kind {self.kind!r} does not match n_stations={self.n_stations}")
        if self.contract not in SUPPORTED_ENERGY_VERSIONS:
            raise ValueError(f"unsupported energy contract {self.contract!r}")
        if self.edge_energies:
            require_finite_sequence("edge_energies", self.edge_energies)
            if len(self.edge_energies) != int(self.n_stations) - 1:
                raise ValueError("edge_energies must contain one value per adjacent hop")
        if self.diagnostic_probability is not None:
            probability = require_finite("diagnostic_probability", self.diagnostic_probability)
            if probability < 0.0 or probability > 1.0:
                raise ValueError("diagnostic_probability must lie in [0, 1]")


@dataclass(frozen=True)
class RouteEnergyTable:
    """Immutable event-level energy table consumed by the exact solver."""

    version: str
    unmatched_penalty: float
    records: tuple[RouteEnergyRecord, ...]
    dustbin_energy: float = DUSTBIN_ENERGY

    def __post_init__(self) -> None:
        if self.version not in SUPPORTED_ENERGY_VERSIONS:
            raise ValueError(f"unsupported energy contract {self.version!r}")
        require_finite("unmatched_penalty", self.unmatched_penalty)
        require_finite("dustbin_energy", self.dustbin_energy)
        seen: set[int] = set()
        for record in self.records:
            if record.contract != self.version:
                raise ValueError("route contract does not match the table version")
            if record.route_id in seen:
                raise ValueError("duplicate route_id in energy table")
            seen.add(int(record.route_id))

    @property
    def size(self) -> int:
        return len(self.records)

    def energies(self) -> np.ndarray:
        return np.asarray([record.energy for record in self.records], dtype=np.float64)

    def endpoint_rows(self) -> list[tuple[Hashable, ...]]:
        return [record.endpoints for record in self.records]


def build_route_energy_table(
    routes: Sequence[tuple[tuple[Hashable, ...], Sequence[float]] | tuple[tuple[Hashable, ...], Sequence[float], float]],
    *,
    unmatched_penalty: float,
    contract: str = CANONICAL_ENERGY_VERSION,
    dustbin_energy: float = DUSTBIN_ENERGY,
    include_diagnostic_probability: bool = False,
) -> RouteEnergyTable:
    """Build a versioned table from ``(endpoints, edge_energies[, correction])``.

    Canonical tables add the optional correction in raw energy space.
    Legacy tables ignore the correction argument and apply per-edge clip
    (fragments) or joint-sum clip (complete four-station replace decode)
    so historical identity violations remain reproducible.
    """
    if contract not in SUPPORTED_ENERGY_VERSIONS:
        raise ValueError(f"unsupported energy contract {contract!r}")
    penalty = require_finite("unmatched_penalty", unmatched_penalty)
    records: list[RouteEnergyRecord] = []
    for route_id, item in enumerate(routes):
        if len(item) == 2:
            endpoints, edge_energies = item
            correction = 0.0
        elif len(item) == 3:
            endpoints, edge_energies, correction = item
        else:
            raise ValueError("each route must be (endpoints, edge_energies[, correction])")
        endpoint_tuple = tuple(endpoints)
        edges = require_finite_sequence("edge_energies", edge_energies)
        if len(endpoint_tuple) != len(edges) + 1:
            raise ValueError("endpoints must contain one more entry than edge_energies")
        n_stations = len(endpoint_tuple)
        delta = require_finite("correction", correction)
        if contract == CANONICAL_ENERGY_VERSION:
            energy = canonical_route_energy(
                edges,
                unmatched_penalty=penalty,
                n_stations=n_stations,
                correction=delta,
            )
            stored_edges = edges
        elif n_stations == 4 and abs(delta) <= 0.0:
            energy = legacy_complete_replace_energy(
                float(sum(edges)),
                unmatched_penalty=penalty,
                n_stations=n_stations,
            )
            stored_edges = tuple(legacy_clip_logit(logit) for logit in edges)
        elif n_stations == 4:
            energy = legacy_complete_residual_energy(
                edges,
                float(sum(edges) + delta),
                weight=1.0,
                unmatched_penalty=penalty,
            )
            stored_edges = tuple(legacy_clip_logit(logit) for logit in edges)
        else:
            if abs(delta) > 0.0:
                raise ValueError("legacy fragment energies do not accept a route correction")
            energy = legacy_fragment_energy(edges, unmatched_penalty=penalty, n_stations=n_stations)
            stored_edges = tuple(legacy_clip_logit(logit) for logit in edges)
        probability = diagnostic_probability(energy) if include_diagnostic_probability else None
        records.append(
            RouteEnergyRecord(
                route_id=route_id,
                endpoints=endpoint_tuple,
                energy=energy,
                n_stations=n_stations,
                kind=route_kind(n_stations),
                contract=contract,
                correction=delta,
                edge_energies=stored_edges,
                diagnostic_probability=probability,
            )
        )
    return RouteEnergyTable(
        version=contract,
        unmatched_penalty=penalty,
        records=tuple(records),
        dustbin_energy=require_finite("dustbin_energy", dustbin_energy),
    )


def assign_from_energy_table(
    table: RouteEnergyTable,
    *,
    allow_legacy: bool = False,
) -> UnitCapacityPackingResult:
    """Exact unit-capacity packing on raw energies.

    Legacy tables are rejected unless ``allow_legacy=True`` so a historical
    replay cannot silently become a new-experiment decoder.
    """
    if table.version == LEGACY_DECODER_VERSION and not allow_legacy:
        raise ValueError(
            "legacy_prob_clip_logit_v1 cannot drive new experiments; "
            "pass allow_legacy=True only for historical replay"
        )
    if table.version not in SUPPORTED_ENERGY_VERSIONS:
        raise ValueError(f"unsupported energy contract {table.version!r}")
    if table.dustbin_energy != DUSTBIN_ENERGY:
        raise ValueError("non-zero dustbin energy is not part of raw_energy_v1")
    if not table.records:
        return solve_unit_capacity_route_packing([], np.empty(0, dtype=np.float64))
    return solve_unit_capacity_route_packing(table.endpoint_rows(), table.energies())
