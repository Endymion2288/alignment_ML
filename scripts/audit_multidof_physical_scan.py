#!/usr/bin/env python3
"""Read-only audit for one completed station-rigid multi-DoF physical scan.

Every input point must already have been produced by the real alignment
payload, cluster-to-segment refit, and mode-0 Acts chain.  This command only
measures the resulting physical candidate graph and truth-fixed propagation
responses; it never shifts tracklet states or reuses an old residual.
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

from alignment.payload import load_station_rigid_alignment_payload
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events
from evaluation.field_propagation import evaluate_field_propagation
from scripts.audit_physical_route_candidate_graph import (
    _acts_surface_summary,
    _condition_chain_checks,
    audit_candidate_graph,
)


RESIDUAL_LABELS = ("rx_mm", "ry_mm", "rtx", "rty")
STATION_PATH = (0, 1, 2, 3)
ADJACENT_PAIRS = tuple(zip(STATION_PATH[:-1], STATION_PATH[1:]))


def _read_plan(scan_root: Path) -> dict[str, Any]:
    with (scan_root / "scan_plan.json").open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or payload.get("scan_mode") != "station_rigid_multidof":
        raise ValueError("scan root is not a station_rigid_multidof physical scan")
    if int(payload.get("q_over_p_mode", -1)) != 0:
        raise ValueError("multi-DoF physical audit requires mode-0 Acts propagation")
    return dict(payload)


def _finite_stats(values: object) -> dict[str, float | int | None]:
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


def _payload_values(
    payload, specs: Sequence[Mapping[str, object]]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for spec in specs:
        name = str(spec["name"])
        station = int(spec["station_id"])
        index = int(spec["transform_index"])
        scale = float(spec["payload_scale"])
        result[name] = float(payload.transform_for_station(station)[index] * scale)
    return result


def _point_paths(scan_root: Path, point: Mapping[str, object]) -> tuple[Path, Path, Path, Path]:
    root = scan_root / str(point["relative_point_dir"])
    return (
        root / "payload" / "alignment_payload.json",
        root / "refit" / "tracklets.root",
        root / "refit" / "propagations.root",
        root / "logs" / "refit.log",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chi2-gate", type=float, default=None)
    parser.add_argument("--min-truth-match-fraction", type=float, default=0.99)
    args = parser.parse_args()
    if args.chi2_gate is not None and (not np.isfinite(args.chi2_gate) or args.chi2_gate <= 0.0):
        parser.error("--chi2-gate must be finite and positive")
    if not np.isfinite(args.min_truth_match_fraction) or not 0.0 <= args.min_truth_match_fraction <= 1.0:
        parser.error("--min-truth-match-fraction must be in [0, 1]")
    scan_root = Path(args.scan_root).expanduser().resolve()
    plan = _read_plan(scan_root)
    raw_specs = plan.get("alignment_parameter_specs")
    if not isinstance(raw_specs, list) or not raw_specs or not all(isinstance(spec, Mapping) for spec in raw_specs):
        raise ValueError("multi-DoF scan plan has invalid alignment_parameter_specs")
    specs = [dict(spec) for spec in raw_specs]
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    point_rows: list[dict[str, object]] = []
    station_pair_rows: list[dict[str, object]] = []
    for raw_point in plan["points"]:
        if not isinstance(raw_point, Mapping):
            raise ValueError("scan plan contains invalid point")
        point = dict(raw_point)
        payload_path, tracklets_path, propagations_path, refit_log = _point_paths(scan_root, point)
        for path in (payload_path, tracklets_path, propagations_path):
            if not path.is_file():
                raise FileNotFoundError(f"physical scan point is incomplete: {path}")
        payload = load_station_rigid_alignment_payload(payload_path)
        planned = point.get("injected_station_transforms")
        if not isinstance(planned, Mapping):
            raise ValueError(f"point '{point.get('name')}' lacks planned rigid transforms")
        for raw_station, transform in planned.items():
            station = int(raw_station)
            if not np.allclose(payload.transform_for_station(station), transform, rtol=0.0, atol=1.0e-12):
                raise ValueError(f"point '{point.get('name')}' payload differs from its frozen plan")
        events = load_events(tracklets_path, require_mc_labels=True)
        records = load_propagation_records(propagations_path)
        graph = audit_candidate_graph(events, records, chi2_gate=args.chi2_gate)
        evaluation = evaluate_field_propagation(
            events,
            records,
            require_truth_match=True,
            q_over_p_mode=0,
            min_truth_match_fraction=args.min_truth_match_fraction,
        )
        surface = _acts_surface_summary(records)
        chain = _condition_chain_checks(refit_log)
        values = _payload_values(payload, specs)
        row: dict[str, object] = {
            "point_name": str(point["name"]),
            "point_role": str(point.get("point_role", "")),
            "direction_trial": str(point.get("direction_trial", "")),
            "condition_axis": str(point["condition_axis"]),
            "condition_value": float(point["condition_value"]),
            "condition_magnitude": float(point["condition_magnitude"]),
            "events": int(graph["events"]),
            "complete_truth_chains": int(graph["complete_truth_chains"]),
            "candidate_retained_complete_truth_chains": int(graph["candidate_retained_complete_truth_chains"]),
            "candidate_complete_truth_chain_recall": graph["candidate_complete_truth_chain_recall"],
            "truth_matched_propagation_pairs": int(evaluation.size),
            "acts_surface_error_count": int(chain["acts_surface_error_count"]),
            "acts_layer_overlap_error_count": int(chain["acts_layer_overlap_error_count"]),
            "condition_chain_complete": bool(all(bool(value) for value in chain["checks"].values())),
        }
        row.update(values)
        pairs = graph["by_station_pair"]
        if not isinstance(pairs, Mapping):
            raise ValueError("candidate graph lacks station-pair summary")
        for source, target in ADJACENT_PAIRS:
            label = f"{source}->{target}"
            graph_pair = pairs.get(label)
            if not isinstance(graph_pair, Mapping):
                raise ValueError(f"candidate graph lacks adjacent pair {label}")
            row[f"{label}_candidate_truth_edge_recall"] = graph_pair["candidate_truth_edge_recall"]
            row[f"{label}_physical_candidate_edges"] = int(graph_pair["physical_candidate_edges"])
        point_rows.append(row)
        for source, target in ((left, right) for left in STATION_PATH for right in STATION_PATH if left < right):
            mask = (evaluation.source_station_id == source) & (evaluation.target_station_id == target)
            if not np.any(mask):
                continue
            residual = evaluation.residual[mask]
            pull = evaluation.pull[mask]
            chi2 = evaluation.chi2[mask]
            sign, logdet = np.linalg.slogdet(evaluation.combined_covariance[mask])
            if not np.all(sign > 0.0):
                raise ValueError("truth-matched propagation combined covariance is not positive definite")
            pair_row: dict[str, object] = {
                "point_name": str(point["name"]),
                "point_role": str(point.get("point_role", "")),
                "station_pair": f"{source}->{target}",
                "truth_pairs": int(np.count_nonzero(mask)),
                "chi2": _finite_stats(chi2),
                "combined_covariance_logdet": _finite_stats(logdet),
            }
            pair_row.update(values)
            for index, label in enumerate(RESIDUAL_LABELS):
                pair_row[f"residual_{label}"] = _finite_stats(residual[:, index])
                pair_row[f"pull_{label}"] = _finite_stats(pull[:, index])
            # Flatten scalars for easy spreadsheet use while retaining the
            # structured JSON table below as the authoritative artifact.
            flattened = {
                key: value
                for key, value in pair_row.items()
                if not isinstance(value, Mapping)
            }
            for key, value in pair_row.items():
                if isinstance(value, Mapping):
                    for statistic, statistic_value in value.items():
                        flattened[f"{key}_{statistic}"] = statistic_value
            station_pair_rows.append(flattened)
    _write_csv(output / "point_metrics.csv", point_rows)
    _write_csv(output / "station_pair_metrics.csv", station_pair_rows)
    summary = {
        "method": "read_only_station_rigid_multidof_physical_audit",
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "q_over_p_mode": 0,
        "scan_root": str(scan_root),
        "condition_axis": str(plan["condition_axis"]),
        "reference_station_ids": list(plan["reference_station_ids"]),
        "movable_station_ids": list(plan["movable_station_ids"]),
        "alignment_parameter_specs": specs,
        "candidate_chi2_gate": args.chi2_gate,
        "point_count": len(point_rows),
        "point_metrics": point_rows,
        "station_pair_metrics": station_pair_rows,
    }
    (output / "audit.json").write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_ready({"point_count": len(point_rows), "output_dir": str(output)}), indent=2))


if __name__ == "__main__":
    main()
