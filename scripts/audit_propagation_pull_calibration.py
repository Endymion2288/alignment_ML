#!/usr/bin/env python3
"""Component-level pull calibration of mode-0 propagation covariances.

For truth-matched adjacent-station edges in the named physical bank points,
measure the residual (tracklet state minus mode-0 ACTS prediction) against the
combined covariance (propagated plus target tracklet).  Per station pair and
per component (x, y, tx, ty) report the pull width (std and MAD-robust),
median, |pull| quantile tails, and the mean predicted/tracklet variance
decomposition, both pooled over the selected split's sources and per source.

The default split is ``train``: the resulting scale factors are the only ones
that may be frozen into a covariance calibration.  Validation sources may be
scanned separately for read-only confirmation, never for factor derivation.
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

from geometry.propagation import mahalanobis_chi2
from baselines.field_chi2_matching import _valid_covariance
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events
from scripts.audit_physical_route_candidate_graph import (
    ADJACENT_PAIRS,
    _unique_truth_indices,
)
from scripts.run_multisource_refit_multidof_local_step import _read_json

SCHEMA_VERSION = "faser-propagation-pull-calibration-v1"
COMPONENTS = ("x_mm", "y_mm", "tx", "ty")
QUANTILES = (0.68, 0.95, 0.99)


def _robust_sigma(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return 1.4826 * mad


def _summarize(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"count": 0}
    summary: dict[str, Any] = {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "robust_sigma": _robust_sigma(values),
    }
    for quantile in QUANTILES:
        summary[f"abs_quantile_{int(quantile * 100)}"] = float(
            np.quantile(np.abs(values), quantile)
        )
    return summary


def _collect_pulls(
    tracklets: Path,
    propagations: Path,
) -> dict[tuple[str, str], dict[str, list[float]]]:
    """Return per-(pair, component) pulls and residual/variance rows."""
    events = load_events(tracklets, require_mc_labels=True)
    records = load_propagation_records(propagations)
    rows_by_source: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for row in range(records.size):
        mode = int(records.q_over_p_mode[row]) if records.q_over_p_mode is not None else 0
        if mode != 0:
            continue
        key = (int(records.run_id[row]), int(records.event_id[row]), int(records.source_tracklet_id[row]))
        rows_by_source[key].append(row)

    pulls: dict[tuple[str, str], list[float]] = defaultdict(list)
    extras: dict[tuple[str, str], list[float]] = defaultdict(list)
    for event in events:
        if event.truth_particle_id is None:
            continue
        for source_station, target_station in ADJACENT_PAIRS:
            pair_label = f"{source_station}->{target_station}"
            source_truth = _unique_truth_indices(event, source_station)
            target_truth = _unique_truth_indices(event, target_station)
            shared = sorted(set(source_truth) & set(target_truth))
            for truth_id in shared:
                source_index = source_truth[truth_id]
                target_index = target_truth[truth_id]
                key = (
                    int(event.run_id),
                    int(event.event_id),
                    int(event.tracklet_id[source_index]),
                )
                target_tracklet = int(event.tracklet_id[target_index])
                truth_rows = [
                    row
                    for row in rows_by_source.get(key, [])
                    if bool(records.success[row])
                    and bool(records.has_covariance[row])
                    and int(records.target_tracklet_id[row]) == target_tracklet
                ]
                if not truth_rows:
                    continue
                row = truth_rows[0]
                propagated = np.asarray(records.covariance[row], dtype=np.float64)
                combined = propagated + np.asarray(event.covariance[target_index], dtype=np.float64)
                if not _valid_covariance(propagated) or not _valid_covariance(combined):
                    continue
                residual = np.asarray(event.state[target_index], dtype=np.float64) - np.asarray(
                    records.prediction[row], dtype=np.float64
                )
                try:
                    chi2 = mahalanobis_chi2(residual, combined)
                except ValueError:
                    continue
                pulls[(pair_label, "mahalanobis_chi2")].append(float(chi2))
                for component_index, component in enumerate(COMPONENTS):
                    variance = float(combined[component_index, component_index])
                    if variance <= 0.0:
                        continue
                    pulls[(pair_label, component)].append(
                        float(residual[component_index] / math.sqrt(variance))
                    )
                    extras[(pair_label, f"residual_{component}")].append(
                        float(residual[component_index])
                    )
                    extras[(pair_label, f"pred_var_{component}")].append(
                        float(propagated[component_index, component_index])
                    )
                    extras[(pair_label, f"trk_var_{component}")].append(
                        float(event.covariance[target_index][component_index, component_index])
                    )
    merged: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(dict)
    for mapping in (pulls, extras):
        for key, values in mapping.items():
            merged[key].setdefault("values", []).extend(values)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-manifest", required=True)
    parser.add_argument("--point", action="append", required=True)
    parser.add_argument("--split", default="train", choices=("train", "validation"))
    parser.add_argument("--source-id", action="append", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    manifest = _read_json(Path(args.iteration_manifest).expanduser().resolve())
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    pooled: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(dict)
    per_source: dict[str, dict[tuple[str, str], dict[str, list[float]]]] = {}
    used_sources: list[str] = []
    for entry in manifest["sources"]:
        source_id = str(entry["source_id"])
        if str(entry["split"]) != args.split:
            continue
        if args.source_id is not None and source_id not in set(args.source_id):
            continue
        root = Path(str(entry["physical_scan_root"])).expanduser().resolve()
        plan = _read_json(root / "scan_plan.json")
        points = {str(point["name"]): point for point in plan["points"]}
        source_maps: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(dict)
        for point_name in args.point:
            point = points.get(point_name)
            if point is None:
                raise ValueError(f"point '{point_name}' absent from {root}/scan_plan.json")
            point_root = root / str(point["relative_point_dir"])
            tracklets = point_root / "refit" / "tracklets.root"
            propagations = point_root / "refit" / "propagations.root"
            if not (tracklets.is_file() and propagations.is_file()):
                raise FileNotFoundError(f"{source_id}/{point_name} refit products missing")
            for key, payload in _collect_pulls(tracklets, propagations).items():
                pair_label, component = key
                source_maps[(pair_label, component)].setdefault("values", []).extend(
                    payload["values"]
                )
                pooled[(pair_label, component)].setdefault("values", []).extend(payload["values"])
        per_source[source_id] = source_maps
        used_sources.append(source_id)

    pooled_rows: list[dict[str, Any]] = []
    per_source_rows: list[dict[str, Any]] = []
    for (pair_label, component), payload in sorted(pooled.items()):
        values = np.asarray(payload["values"], dtype=np.float64)
        row = {"station_pair": pair_label, "component": component, **_summarize(values)}
        pooled_rows.append(row)
    for source_id, source_maps in per_source.items():
        for (pair_label, component), payload in sorted(source_maps.items()):
            values = np.asarray(payload["values"], dtype=np.float64)
            per_source_rows.append(
                {
                    "source_id": source_id,
                    "station_pair": pair_label,
                    "component": component,
                    **_summarize(values),
                }
            )

    def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fieldnames = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    _write_csv(output / "pull_calibration_pooled.csv", pooled_rows)
    _write_csv(output / "pull_calibration_per_source.csv", per_source_rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "iteration_manifest": str(Path(args.iteration_manifest).expanduser().resolve()),
        "points": list(args.point),
        "split": args.split,
        "source_ids": used_sources,
        "test_data_accessed": False,
        "q_over_p_mode": 0,
        "rows_pooled": len(pooled_rows),
        "rows_per_source": len(per_source_rows),
    }
    (output / "pull_calibration_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), "sources": len(used_sources)}, indent=2))


if __name__ == "__main__":
    main()
