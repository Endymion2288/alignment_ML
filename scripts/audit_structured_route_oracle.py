#!/usr/bin/env python3
"""Audit V3's exact route-packing workload without opening a test split.

The report measures pre-existing physical complete-route candidates and their
endpoint-conflict components.  It also benchmarks the unchanged exact
unit-capacity oracle on deterministic positive utilities.  It never trains a
model, changes a candidate graph, or reads a test asset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from baselines.route_assignment import solve_unit_capacity_route_packing
from datasets.physical_curriculum import load_synthetic_curriculum_manifest
from training.curriculum_mlp import build_candidate_sets
from training.geometry_aware_transformer import (
    ALL_STATION_PAIRS,
    build_transformer_graph_bundle,
    source_disjoint_audit,
)
from training.route_aware_transformer import RouteCandidateTable, materialize_route_candidate_tables


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _quantiles(values: Sequence[int | float]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64)
    if not data.size:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0, "max": 0}
    integer_values = all(isinstance(value, (int, np.integer)) for value in values)
    return {
        "count": int(data.size),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p90": float(np.quantile(data, 0.90)),
        "p99": float(np.quantile(data, 0.99)),
        "max": int(np.max(data)) if integer_values else float(np.max(data)),
    }


def _component_sizes(table: RouteCandidateTable) -> list[tuple[int, int]]:
    """Return (routes, endpoints) for each exact-packing conflict component."""
    rows = np.asarray(table.node_indices, dtype=np.int64)
    if not rows.size:
        return []
    membership: dict[int, list[int]] = {}
    for route, endpoints in enumerate(rows):
        for endpoint in endpoints:
            membership.setdefault(int(endpoint), []).append(route)
    visited = np.zeros(rows.shape[0], dtype=bool)
    components: list[tuple[int, int]] = []
    for root in range(rows.shape[0]):
        if visited[root]:
            continue
        pending = [root]
        visited[root] = True
        routes = 0
        endpoints: set[int] = set()
        while pending:
            route = pending.pop()
            routes += 1
            for endpoint in rows[route]:
                key = int(endpoint)
                endpoints.add(key)
                for neighbor in membership[key]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        pending.append(neighbor)
        components.append((routes, len(endpoints)))
    return components


def _table_key(table: RouteCandidateTable, ordinal: int) -> str:
    digest = hashlib.sha256(np.asarray(table.node_indices, dtype=np.int64).tobytes()).hexdigest()
    return f"{ordinal:08d}:{digest}"


def _benchmark(tables: Sequence[RouteCandidateTable], limit: int) -> dict[str, object]:
    if limit < 0:
        raise ValueError("benchmark-events must be non-negative")
    selected = sorted(enumerate(tables), key=lambda item: _table_key(item[1], item[0]))[:limit]
    elapsed: list[float] = []
    route_counts: list[int] = []
    selected_counts: list[int] = []
    for ordinal, table in selected:
        utilities = np.linspace(0.01, 1.0, table.size, dtype=np.float64)
        start = time.perf_counter()
        result = solve_unit_capacity_route_packing(table.node_indices, utilities)
        elapsed.append(time.perf_counter() - start)
        route_counts.append(table.size)
        selected_counts.append(int(np.count_nonzero(result.selected)))
    return {
        "events": int(len(selected)),
        "utilities": "deterministic_positive_linear_0p01_to_1p0",
        "elapsed_seconds": _quantiles(elapsed),
        "route_candidates": _quantiles(route_counts),
        "selected_routes": _quantiles(selected_counts),
        "total_elapsed_seconds": float(sum(elapsed)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "validation"), action="append", default=None)
    parser.add_argument("--benchmark-events", type=int, default=64)
    args = parser.parse_args()

    splits = tuple(args.split or ("train", "validation"))
    manifest_path, samples, manifest = load_synthetic_curriculum_manifest(
        args.synthetic_manifest,
        require_all_splits=False,
        allowed_splits=splits,
    )
    if manifest.get("physical_geometry_repropagation") is not True:
        raise ValueError("structured-oracle audit requires physical repropagation")
    if int(manifest.get("q_over_p_mode", -1)) != 0:
        raise ValueError("structured-oracle audit requires mode-0 propagation")
    if any(sample.split == "test" for sample in samples):  # defensive; loader already excludes it
        raise ValueError("structured-oracle audit must not load test samples")

    candidate_sets = build_candidate_sets(
        samples, ALL_STATION_PAIRS, chi2_gate=None, feature_set="residual_v1"
    )
    bundle = build_transformer_graph_bundle(candidate_sets, context_mode="full_event")
    tables = list(materialize_route_candidate_tables(bundle.graphs).values())
    components = [component for table in tables for component in _component_sizes(table)]
    route_counts = [table.size for table in tables]
    component_routes = [routes for routes, _ in components]
    component_endpoints = [endpoints for _, endpoints in components]
    _write_json(
        Path(args.output).expanduser().resolve(),
        {
            "schema_version": "faser-structured-route-oracle-audit-v1",
            "synthetic_manifest": str(manifest_path),
            "loaded_event_splits": list(splits),
            "forbidden_splits": ["test"],
            "test_events_loaded": False,
            "physical_geometry_repropagation": True,
            "q_over_p_mode": 0,
            "candidate_graph": "existing_mode0_acts_physical_candidates_all_six_station_pairs",
            "route_candidates": "complete_chains_of_existing_adjacent_physical_edges_only",
            "source_audit": source_disjoint_audit(samples),
            "graphs": int(len(tables)),
            "complete_route_candidates": _quantiles(route_counts),
            "conflict_components": {
                "components": int(len(components)),
                "routes": _quantiles(component_routes),
                "endpoints": _quantiles(component_endpoints),
                "route_count_histogram_width_10": {
                    str(key): int(value)
                    for key, value in sorted(Counter(min(300, (count // 10) * 10) for count in component_routes).items())
                },
            },
            "exact_solver_benchmark": _benchmark(tables, int(args.benchmark_events)),
        },
    )


if __name__ == "__main__":
    main()
