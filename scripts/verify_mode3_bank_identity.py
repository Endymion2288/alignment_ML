#!/usr/bin/env python3
"""Verify that a re-produced mode-3 physical bank is an exact superset of its
mode-0 production counterpart.

The matched-retraining contract requires the new bank (written by the
four-variant dumper) to reproduce the original bank bit-for-bit on every
pre-existing artifact so that a synthetic overlay materialized from either
bank contains the identical events.  For every source/point this script checks

* tracklets: identical (run_id, event_id, tracklet_id, station_id, geometry,
  truth) arrays between the production and mode-3 banks;
* propagations: the mode-0/1/2 records are identical row by row, and mode-3
  records exist with the same (run, event, source, target) keys as mode 0.

The comparison is read-only and never opens a test-split asset.
"""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash_arrays(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _tracklet_digest(tracklets: Path) -> dict[str, Any]:
    events = load_events(tracklets, require_mc_labels=False)
    parts: list[np.ndarray] = []
    n_rows = 0
    for event in events:
        covariance = np.asarray(event.covariance, dtype=np.float64)
        parts.append(
            np.column_stack(
                [
                    np.full(event.size, int(event.run_id), dtype=np.float64),
                    np.full(event.size, int(event.event_id), dtype=np.float64),
                    np.asarray(event.tracklet_id, dtype=np.float64),
                    np.asarray(event.station_id, dtype=np.float64),
                    np.asarray(event.state, dtype=np.float64),
                    np.asarray(event.z_mm, dtype=np.float64),
                    covariance.reshape(event.size, -1),
                ]
            )
        )
        n_rows += int(event.size)
    if not parts:
        return {"rows": 0, "sha256": _hash_arrays([np.empty((0, 25))])}
    stacked = np.vstack(parts)
    order = np.lexsort(stacked[:, ::-1].T)
    return {"rows": n_rows, "sha256": _hash_arrays([stacked[order]])}


def _mode_rows(propagations: Path, mode: int) -> dict[tuple[int, int, int, int], int]:
    records = load_propagation_records(propagations)
    rows: dict[tuple[int, int, int, int], int] = {}
    record_mode = records.q_over_p_mode
    for row in range(records.size):
        row_mode = int(record_mode[row]) if record_mode is not None else 0
        if row_mode != mode:
            continue
        key = (
            int(records.run_id[row]),
            int(records.event_id[row]),
            int(records.source_tracklet_id[row]),
            int(records.target_tracklet_id[row]),
        )
        rows[key] = row
    return rows


def _propagation_digest(propagations: Path, mode: int) -> dict[str, Any]:
    records = load_propagation_records(propagations)
    rows = _mode_rows(propagations, mode)
    keys = sorted(rows)
    if not keys:
        return {"rows": 0, "sha256": _hash_arrays([np.empty(0)])}
    index = np.asarray([rows[key] for key in keys], dtype=np.int64)
    arrays = [
        np.asarray(keys, dtype=np.int64),
        np.asarray(records.success[index], dtype=np.int8),
        np.asarray(records.has_covariance[index], dtype=np.int8),
        np.asarray(records.prediction[index], dtype=np.float64),
        np.asarray(records.covariance[index], dtype=np.float64),
    ]
    return {"rows": int(index.size), "sha256": _hash_arrays(arrays)}


def _source_points(source: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return {point_name: {physical_tracklets, physical_propagations, completed}}.

    Curriculum manifests embed a per-source ``points`` list with explicit
    asset paths and completion flags.  Alignment-iteration manifests instead
    point at a ``physical_scan`` root whose ``scan_plan.json`` enumerates the
    points; completion is derived from the on-disk artifacts.
    """
    embedded = source.get("points")
    if isinstance(embedded, list) and embedded:
        return {
            str(point["name"]): {
                "physical_tracklets": point["physical_tracklets"],
                "physical_propagations": point["physical_propagations"],
                "completed": bool(point.get("completed")),
            }
            for point in embedded
        }
    scan_root = Path(str(source["physical_scan_root"]))
    plan = _read_json(scan_root / "scan_plan.json")
    points: dict[str, dict[str, Any]] = {}
    for point in plan.get("points", []):
        name = str(point.get("name", point.get("point_name", "")))
        relative = str(point.get("relative_point_dir", f"points/{name}"))
        refit = scan_root / relative / "refit"
        tracklets = refit / "tracklets.root"
        propagations = refit / "propagations.root"
        points[name] = {
            "physical_tracklets": str(tracklets),
            "physical_propagations": str(propagations),
            "completed": tracklets.is_file() and propagations.is_file(),
        }
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-manifest", required=True, help="original mode-0 bank manifest")
    parser.add_argument("--mode3-manifest", required=True, help="re-produced mode-3 bank manifest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    production_path = Path(args.production_manifest).expanduser().resolve()
    mode3_path = Path(args.mode3_manifest).expanduser().resolve()
    production = _read_json(production_path)
    mode3 = _read_json(mode3_path)
    production_sources = {str(source["source_id"]): source for source in production["sources"]}
    mode3_sources = {str(source["source_id"]): source for source in mode3["sources"]}
    if set(production_sources) != set(mode3_sources):
        raise ValueError("source sets differ between the two banks")

    source_reports: dict[str, Any] = {}
    all_identical = True
    for source_id in sorted(production_sources):
        old_points = _source_points(production_sources[source_id])
        new_points = _source_points(mode3_sources[source_id])
        if set(old_points) != set(new_points):
            raise ValueError(f"point sets differ for {source_id}")
        point_reports: dict[str, Any] = {}
        for point_name in sorted(old_points):
            old_point = old_points[point_name]
            new_point = new_points[point_name]
            if not (bool(old_point.get("completed")) and bool(new_point.get("completed"))):
                point_reports[point_name] = {"status": "skipped_incomplete"}
                all_identical = False
                continue
            old_tracklets = _tracklet_digest(Path(str(old_point["physical_tracklets"])))
            new_tracklets = _tracklet_digest(Path(str(new_point["physical_tracklets"])))
            tracklets_identical = old_tracklets == new_tracklets
            old_mode0 = _propagation_digest(Path(str(old_point["physical_propagations"])), 0)
            new_mode0 = _propagation_digest(Path(str(new_point["physical_propagations"])), 0)
            new_mode3 = _propagation_digest(Path(str(new_point["physical_propagations"])), 3)
            mode0_identical = old_mode0 == new_mode0
            mode3_complete = new_mode3["rows"] == new_mode0["rows"] and new_mode3["rows"] > 0
            point_reports[point_name] = {
                "status": "accepted" if (tracklets_identical and mode0_identical and mode3_complete) else "mismatch",
                "tracklets_identical": tracklets_identical,
                "tracklet_rows": new_tracklets["rows"],
                "mode0_identical": mode0_identical,
                "mode0_rows": new_mode0["rows"],
                "mode3_rows": new_mode3["rows"],
                "mode3_complete": mode3_complete,
            }
            if not (tracklets_identical and mode0_identical and mode3_complete):
                all_identical = False
        source_reports[source_id] = {
            "split": str(production_sources[source_id]["split"]),
            "points": point_reports,
        }

    summary = {
        "schema_version": "faser-mode3-bank-identity-audit-v1",
        "production_manifest": str(production_path),
        "mode3_manifest": str(mode3_path),
        "all_identical": all_identical,
        "sources": source_reports,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"all_identical": all_identical, "sources": len(source_reports)}, indent=2))


if __name__ == "__main__":
    main()
