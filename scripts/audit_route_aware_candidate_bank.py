#!/usr/bin/env python3
"""Audit complete physical route-candidate occupancy without opening test data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from datasets.physical_curriculum import load_synthetic_curriculum_manifest
from training.curriculum_mlp import build_candidate_sets
from training.geometry_aware_transformer import ALL_STATION_PAIRS, build_transformer_graph_bundle
from training.route_aware_transformer import enumerate_complete_route_candidates


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"graphs": 0, "routes": 0, "mean": 0.0, "p95": 0.0, "maximum": 0}
    return {
        "graphs": int(array.size),
        "routes": int(np.sum(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": int(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest_path, samples, manifest = load_synthetic_curriculum_manifest(
        args.synthetic_manifest,
        require_all_splits=False,
        allowed_splits=("train", "validation"),
    )
    if manifest.get("physical_geometry_repropagation") is not True or int(manifest.get("q_over_p_mode", -1)) != 0:
        raise ValueError("route candidate audit requires physical mode-0 payloads")
    values_by_split: dict[str, list[int]] = {"train": [], "validation": []}
    values_by_split_magnitude: dict[tuple[str, float], list[int]] = {}
    labels_by_split: dict[str, int] = {"train": 0, "validation": 0}
    fake_by_split: dict[str, int] = {"train": 0, "validation": 0}
    for split in ("train", "validation"):
        split_samples = [sample for sample in samples if sample.split == split]
        sets = build_candidate_sets(
            split_samples, ALL_STATION_PAIRS, chi2_gate=None, feature_set="residual_v1"
        )
        bundle = build_transformer_graph_bundle(sets, context_mode="full_event")
        for graph in bundle.graphs:
            table = enumerate_complete_route_candidates(graph)
            values_by_split[split].append(table.size)
            values_by_split_magnitude.setdefault((split, float(graph.sample.magnitude_mm)), []).append(table.size)
            labels_by_split[split] += int(np.count_nonzero(table.labels))
            fake_by_split[split] += int(np.count_nonzero(table.fake_endpoint))
    payload: dict[str, Any] = {
        "schema_version": "faser-route-aware-v2-candidate-bank-audit-v1",
        "synthetic_manifest": str(manifest_path),
        "synthetic_manifest_sha256": _sha256(manifest_path),
        "loaded_splits": ["train", "validation"],
        "forbidden_splits": ["test"],
        "test_events_loaded": False,
        "physical_geometry_repropagation": True,
        "q_over_p_mode": 0,
        "candidate_chi2_gate": None,
        "route_definition": "complete chains of existing adjacent physical candidates",
        "by_split": {
            split: {
                **_summary(values_by_split[split]),
                "truth_consistent_routes": labels_by_split[split],
                "fake_endpoint_routes": fake_by_split[split],
            }
            for split in ("train", "validation")
        },
        "by_split_magnitude": {
            f"{split}:{magnitude:g}": _summary(values)
            for (split, magnitude), values in sorted(values_by_split_magnitude.items())
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["by_split"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
