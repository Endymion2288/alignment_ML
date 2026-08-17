#!/usr/bin/env python3
"""Read-only audit of a source-disjoint station-rigid physical corpus.

The command is intentionally agnostic to which admitted station-level
components appear in a point's explicit transform.  It reads only completed
payload -> refit -> mode-0 Acts products and reports raw candidate retention,
field-propagation residual/pull/covariance response, and conditions/surface
audit data.  It neither creates a coordinate surrogate nor runs association.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events
from evaluation.field_propagation import evaluate_field_propagation
from scripts.audit_physical_route_candidate_graph import (
    ADJACENT_PAIRS,
    _acts_surface_summary,
    _condition_chain_checks,
    audit_candidate_graph,
)


STATION_PATH = (0, 1, 2, 3)
RESIDUAL_LABELS = ("rx_mm", "ry_mm", "rtx", "rty")


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("physical corpus manifest must be a mapping")
    if payload.get("physical_geometry_repropagation") is not True:
        raise ValueError("physical corpus does not certify geometry repropagation")
    if int(payload.get("q_over_p_mode", -1)) != 0:
        raise ValueError("multi-DoF physical corpus audit requires mode-0 Acts propagation")
    if "test" not in {str(value) for value in payload.get("forbidden_splits", ())}:
        raise ValueError("physical corpus must explicitly forbid test")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("physical corpus has no sources")
    return dict(payload)


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = sorted({str(key) for row in rows for key in row}) or ["empty"]
    values = list(rows) if rows else [{"empty": ""}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)


def _finite_stats(values: object) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"count": 0, "mean": None, "rms": None, "median": None, "p95_abs": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
        "median": float(np.median(array)),
        "p95_abs": float(np.quantile(np.abs(array), 0.95)),
    }


def _point_metadata(source: Mapping[str, object], point: Mapping[str, object]) -> dict[str, object]:
    values = point.get("alignment_parameter_values")
    transforms = point.get("injected_station_transforms")
    if not isinstance(values, Mapping) or not isinstance(transforms, Mapping):
        raise ValueError("multi-DoF physical point lacks explicit parameters/transforms")
    return {
        "source_id": str(source["source_id"]),
        "split": str(source["split"]),
        "payload_id": str(point["payload_id"]),
        "point_role": str(point.get("point_role", "")),
        "direction_trial": str(point.get("direction_trial", "")),
        "condition_axis": str(point.get("condition_axis", "")),
        "condition_value": float(point["condition_value"]),
        "condition_magnitude": float(point["condition_magnitude"]),
        "alignment_parameter_values": {str(key): float(value) for key, value in values.items()},
        "injected_station_transforms": dict(transforms),
    }


def _condition_key(metadata: Mapping[str, object]) -> tuple[str, str, str, str]:
    return (
        str(metadata["split"]),
        str(metadata["payload_id"]),
        json.dumps(metadata["alignment_parameter_values"], sort_keys=True),
        json.dumps(metadata["injected_station_transforms"], sort_keys=True),
    )


def _chain_counts(chain: Mapping[str, object] | None) -> tuple[bool, int, int]:
    if chain is None:
        return False, 0, 0
    checks = chain.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("physical conditions audit lacks chain checks")
    return (
        all(bool(value) for value in checks.values()),
        int(chain.get("acts_surface_error_count", 0)),
        int(chain.get("acts_layer_overlap_error_count", 0)),
    )


def audit_corpus(
    manifest: Mapping[str, object], *, splits: set[str], chi2_gate: float | None, min_truth_match_fraction: float
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    point_groups: dict[tuple[str, str, str, str], dict[str, object]] = {}
    pair_groups: dict[tuple[tuple[str, str, str, str], int, int], dict[str, object]] = {}
    for raw_source in manifest["sources"]:
        if not isinstance(raw_source, Mapping):
            raise ValueError("physical corpus contains an invalid source")
        source = dict(raw_source)
        split = str(source.get("split", ""))
        if split == "test":
            raise ValueError("sealed test source appears in physical corpus")
        if split not in splits:
            continue
        points = source.get("points")
        if not isinstance(points, list) or not points:
            raise ValueError(f"physical corpus source '{source.get('source_id')}' has no points")
        for raw_point in points:
            if not isinstance(raw_point, Mapping):
                raise ValueError("physical corpus contains an invalid point")
            point = dict(raw_point)
            if point.get("completed") is not True:
                raise RuntimeError(f"physical point is incomplete: {source.get('source_id')}/{point.get('payload_id')}")
            metadata = _point_metadata(source, point)
            key = _condition_key(metadata)
            tracklets = Path(str(point["physical_tracklets"])).expanduser().resolve()
            propagations = Path(str(point["physical_propagations"])).expanduser().resolve()
            if not tracklets.is_file() or not propagations.is_file():
                raise FileNotFoundError(f"physical point assets are missing: {tracklets} / {propagations}")
            events = load_events(tracklets, require_mc_labels=True)
            records = load_propagation_records(propagations)
            graph = audit_candidate_graph(events, records, chi2_gate=chi2_gate)
            evaluation = evaluate_field_propagation(
                events,
                records,
                require_truth_match=True,
                q_over_p_mode=0,
                min_truth_match_fraction=min_truth_match_fraction,
            )
            root = Path(str(source["physical_scan_root"])).expanduser().resolve()
            relative = str(point.get("relative_point_dir", ""))
            chain_ok, surface_errors, overlap_errors = _chain_counts(
                _condition_chain_checks(root / relative / "logs" / "refit.log")
            )
            group = point_groups.setdefault(
                key,
                {
                    **metadata,
                    "sources": set(),
                    "events": 0,
                    "truth_matched_propagation_pairs": 0,
                    "complete_truth_chains": 0,
                    "candidate_retained_complete_truth_chains": 0,
                    "condition_chain_complete_sources": 0,
                    "acts_surface_error_count": 0,
                    "acts_layer_overlap_error_count": 0,
                    "candidate_pairs": {
                        f"{left}->{right}": {"truth": 0, "retained": 0, "edges": 0}
                        for left, right in ADJACENT_PAIRS
                    },
                },
            )
            sources = group["sources"]
            assert isinstance(sources, set)
            sources.add(str(source["source_id"]))
            group["events"] = int(group["events"]) + int(graph["events"])
            group["truth_matched_propagation_pairs"] = int(group["truth_matched_propagation_pairs"]) + int(
                evaluation.size
            )
            group["complete_truth_chains"] = int(group["complete_truth_chains"]) + int(graph["complete_truth_chains"])
            group["candidate_retained_complete_truth_chains"] = int(
                group["candidate_retained_complete_truth_chains"]
            ) + int(graph["candidate_retained_complete_truth_chains"])
            group["condition_chain_complete_sources"] = int(group["condition_chain_complete_sources"]) + int(chain_ok)
            group["acts_surface_error_count"] = int(group["acts_surface_error_count"]) + surface_errors
            group["acts_layer_overlap_error_count"] = int(group["acts_layer_overlap_error_count"]) + overlap_errors
            graph_pairs = graph["by_station_pair"]
            assert isinstance(graph_pairs, Mapping)
            candidate_pairs = group["candidate_pairs"]
            assert isinstance(candidate_pairs, Mapping)
            for left, right in ADJACENT_PAIRS:
                label = f"{left}->{right}"
                graph_pair = graph_pairs[label]
                holder = candidate_pairs[label]
                assert isinstance(graph_pair, Mapping) and isinstance(holder, Mapping)
                holder["truth"] = int(holder["truth"]) + int(graph_pair["truth_pairs_with_unique_endpoints"])
                holder["retained"] = int(holder["retained"]) + int(graph_pair["candidate_retained_truth_pairs"])
                holder["edges"] = int(holder["edges"]) + int(graph_pair["physical_candidate_edges"])
            for left in STATION_PATH:
                for right in STATION_PATH:
                    if left >= right:
                        continue
                    mask = (evaluation.source_station_id == left) & (evaluation.target_station_id == right)
                    if not np.any(mask):
                        continue
                    pair_key = (key, left, right)
                    pair_group = pair_groups.setdefault(
                        pair_key,
                        {
                            "condition_key": key,
                            "source_station_id": left,
                            "target_station_id": right,
                            "sources": set(),
                            "residual": [],
                            "pull": [],
                            "chi2": [],
                            "logdet": [],
                        },
                    )
                    pair_sources = pair_group["sources"]
                    assert isinstance(pair_sources, set)
                    pair_sources.add(str(source["source_id"]))
                    pair_group["residual"].append(np.asarray(evaluation.residual[mask], dtype=np.float64))
                    pair_group["pull"].append(np.asarray(evaluation.pull[mask], dtype=np.float64))
                    pair_group["chi2"].append(np.asarray(evaluation.chi2[mask], dtype=np.float64))
                    signs, logdet = np.linalg.slogdet(evaluation.combined_covariance[mask])
                    if not np.all(signs > 0.0):
                        raise ValueError("combined covariance is not positive definite")
                    pair_group["logdet"].append(logdet)
    if not point_groups:
        raise ValueError("no selected physical points remain")
    point_rows: list[dict[str, object]] = []
    for _, group in sorted(point_groups.items()):
        sources = group.pop("sources")
        assert isinstance(sources, set)
        chains = int(group["complete_truth_chains"])
        retained = int(group["candidate_retained_complete_truth_chains"])
        row = {**group, "sources": len(sources), "candidate_chi2_gate": chi2_gate}
        row["candidate_complete_truth_chain_recall"] = None if chains == 0 else float(retained / chains)
        row["condition_chain_complete_fraction"] = float(
            int(group["condition_chain_complete_sources"]) / len(sources)
        )
        candidates = group.pop("candidate_pairs")
        assert isinstance(candidates, Mapping)
        for label, values in candidates.items():
            assert isinstance(values, Mapping)
            truth = int(values["truth"])
            row[f"{label}_candidate_truth_edge_recall"] = None if truth == 0 else float(int(values["retained"]) / truth)
            row[f"{label}_physical_candidate_edges"] = int(values["edges"])
        row.pop("condition_chain_complete_sources")
        point_rows.append(row)
    pair_rows: list[dict[str, object]] = []
    for (condition_key, left, right), group in sorted(pair_groups.items()):
        point_group = point_groups[condition_key]
        residual = np.concatenate(group["residual"], axis=0)
        pull = np.concatenate(group["pull"], axis=0)
        chi2 = np.concatenate(group["chi2"], axis=0)
        logdet = np.concatenate(group["logdet"], axis=0)
        row = {
            "split": point_group["split"],
            "payload_id": point_group["payload_id"],
            "point_role": point_group["point_role"],
            "condition_axis": point_group["condition_axis"],
            "condition_value": point_group["condition_value"],
            "condition_magnitude": point_group["condition_magnitude"],
            "alignment_parameter_values": json.dumps(point_group["alignment_parameter_values"], sort_keys=True),
            "source_station_id": int(left),
            "target_station_id": int(right),
            "station_pair": f"{left}->{right}",
            "sources": len(group["sources"]),
            "truth_pairs": int(residual.shape[0]),
        }
        row.update({f"chi2_{key}": value for key, value in _finite_stats(chi2).items()})
        row.update({f"combined_covariance_logdet_{key}": value for key, value in _finite_stats(logdet).items()})
        for index, label in enumerate(RESIDUAL_LABELS):
            row.update({f"residual_{label}_{key}": value for key, value in _finite_stats(residual[:, index]).items()})
            row.update({f"pull_{label}_{key}": value for key, value in _finite_stats(pull[:, index]).items()})
        pair_rows.append(row)
    return point_rows, pair_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", action="append", choices=("train", "validation"), default=None)
    parser.add_argument("--chi2-gate", type=float, default=25.0)
    parser.add_argument("--min-truth-match-fraction", type=float, default=0.99)
    args = parser.parse_args()
    if args.chi2_gate is not None and (not np.isfinite(args.chi2_gate) or args.chi2_gate <= 0.0):
        parser.error("--chi2-gate must be finite and positive")
    if not np.isfinite(args.min_truth_match_fraction) or not 0.0 <= args.min_truth_match_fraction <= 1.0:
        parser.error("--min-truth-match-fraction must be in [0, 1]")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.physical_manifest).expanduser().resolve()
    manifest = _load_manifest(manifest_path)
    point_rows, pair_rows = audit_corpus(
        manifest,
        splits=set(args.split or ("train", "validation")),
        chi2_gate=args.chi2_gate,
        min_truth_match_fraction=float(args.min_truth_match_fraction),
    )
    _write_csv(output / "point_metrics.csv", point_rows)
    _write_csv(output / "station_pair_metrics.csv", pair_rows)
    summary = {
        "method": "read_only_source_disjoint_station_rigid_multidof_physical_corpus_audit",
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "q_over_p_mode": 0,
        "test_data_accessed": False,
        "physical_manifest": str(manifest_path),
        "candidate_chi2_gate": args.chi2_gate,
        "point_metrics": point_rows,
        "station_pair_metrics": pair_rows,
    }
    (output / "audit.json").write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "point_groups": len(point_rows),
                "station_pair_groups": len(pair_rows),
                "test_data_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
