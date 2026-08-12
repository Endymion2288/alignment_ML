#!/usr/bin/env python3
"""Aggregate content audits for a physically refitted curriculum payload bank."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON payload is not a mapping: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _truth_and_covariance_ok(audit: Mapping[str, Any]) -> tuple[bool, bool]:
    truth_ok = bool(audit.get("has_mc_labels") is True)
    covariance = audit.get("covariance")
    if not isinstance(covariance, Mapping):
        return truth_ok, False
    rows = int(covariance.get("rows", 0))
    positive = int(covariance.get("positive_definite_rows", 0))
    return truth_ok, rows > 0 and rows == positive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "validation", "test"),
        default=None,
        help="Audit only the selected split(s), useful for a sealed test-only physical scan",
    )
    args = parser.parse_args()

    manifest_path = Path(args.physical_manifest).expanduser().resolve()
    manifest = _load_json(manifest_path)
    if manifest.get("physical_geometry_repropagation") is not True:
        raise ValueError("physical manifest does not certify geometry repropagation")
    if int(manifest.get("q_over_p_mode", -1)) != 0:
        raise ValueError("physical curriculum bank audit requires q_over_p_mode=0")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("physical manifest has no sources")

    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    selected_splits = None if args.split is None else set(args.split)

    rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, float], dict[str, object]] = defaultdict(
        lambda: {
            "source_ids": set(),
            "source_event_uids": set(),
            "assets_complete": True,
            "truth_labels_complete": True,
            "positive_definite_covariance": True,
            "tracklets": 0,
            "events": 0,
            "station_tracklets": defaultdict(int),
        }
    )
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("physical manifest source is not a mapping")
        source_id = str(source.get("source_id", ""))
        split = str(source.get("split", ""))
        if not source_id or split not in {"train", "validation", "test"}:
            raise ValueError("physical manifest has an invalid source identity")
        if selected_splits is not None and split not in selected_splits:
            continue
        source_uids = source.get("source_event_uids")
        if not isinstance(source_uids, list) or not source_uids:
            raise ValueError(f"source lacks event provenance: {source_id}")
        points = source.get("points")
        if not isinstance(points, list) or not points:
            raise ValueError(f"source lacks payload points: {source_id}")
        for point in points:
            if not isinstance(point, Mapping):
                raise ValueError(f"invalid physical point for {source_id}")
            tracklets = Path(str(point.get("physical_tracklets", ""))).expanduser().resolve()
            propagations = Path(str(point.get("physical_propagations", ""))).expanduser().resolve()
            payload = Path(str(point.get("physical_payload_manifest", ""))).expanduser().resolve()
            audit_path = tracklets.parent / "content_audit.json"
            assets_complete = bool(
                point.get("completed") is True
                and tracklets.is_file()
                and propagations.is_file()
                and payload.is_file()
                and audit_path.is_file()
            )
            if not assets_complete:
                raise FileNotFoundError(f"incomplete physical point: {source_id}/{point.get('payload_id')}")
            audit = _load_json(audit_path)
            truth_ok, covariance_ok = _truth_and_covariance_ok(audit)
            station_counts = audit.get("station_counts")
            if not isinstance(station_counts, Mapping):
                raise ValueError(f"content audit lacks station counts: {audit_path}")
            magnitude = float(point["magnitude_mm"])
            group = grouped[(split, magnitude)]
            group["source_ids"].add(source_id)
            group["source_event_uids"].update(str(value) for value in source_uids)
            group["assets_complete"] = bool(group["assets_complete"]) and assets_complete
            group["truth_labels_complete"] = bool(group["truth_labels_complete"]) and truth_ok
            group["positive_definite_covariance"] = bool(
                group["positive_definite_covariance"]
            ) and covariance_ok
            group["tracklets"] = int(group["tracklets"]) + int(audit.get("tracklets", 0))
            group["events"] = int(group["events"]) + int(audit.get("events", 0))
            aggregate_stations = group["station_tracklets"]
            if not isinstance(aggregate_stations, defaultdict):  # pragma: no cover
                raise RuntimeError("invalid station aggregation payload")
            row: dict[str, object] = {
                "source_id": source_id,
                "split": split,
                "payload_id": str(point["payload_id"]),
                "magnitude_mm": magnitude,
                "direction_trial": str(point["direction_trial"]),
                "assets_complete": assets_complete,
                "truth_labels_complete": truth_ok,
                "positive_definite_covariance": covariance_ok,
                "tracklets": int(audit.get("tracklets", 0)),
                "events": int(audit.get("events", 0)),
                "content_audit": str(audit_path),
            }
            for station, count in sorted(station_counts.items(), key=lambda item: int(item[0])):
                value = int(count)
                row[f"station_{int(station)}_tracklets"] = value
                aggregate_stations[int(station)] += value
            rows.append(row)

    if not rows:
        raise ValueError("no physical points match the requested split selection")
    summary_rows: list[dict[str, object]] = []
    for (split, magnitude), aggregate in sorted(grouped.items()):
        station_tracklets = aggregate["station_tracklets"]
        if not isinstance(station_tracklets, defaultdict):  # pragma: no cover
            raise RuntimeError("invalid station aggregation payload")
        row = {
            "split": split,
            "magnitude_mm": magnitude,
            "source_count": len(aggregate["source_ids"]),
            "source_event_count": len(aggregate["source_event_uids"]),
            "assets_complete": bool(aggregate["assets_complete"]),
            "truth_labels_complete": bool(aggregate["truth_labels_complete"]),
            "positive_definite_covariance": bool(aggregate["positive_definite_covariance"]),
            "tracklets": int(aggregate["tracklets"]),
            "events": int(aggregate["events"]),
        }
        row.update(
            {
                f"station_{station}_tracklets": int(count)
                for station, count in sorted(station_tracklets.items())
            }
        )
        summary_rows.append(row)

    _write_csv(output_root / "physical_refit_availability_by_source_and_payload.csv", rows)
    _write_csv(output_root / "physical_refit_availability_by_split_and_magnitude.csv", summary_rows)
    _write_json(
        output_root / "audit.json",
        {
            "physical_manifest": str(manifest_path),
            "physical_geometry_repropagation": True,
            "q_over_p_mode": 0,
            "source_split_unit": manifest.get("source_split_unit"),
            "source_split_audit": manifest.get("source_split_audit"),
            "sources_total_in_manifest": len(sources),
            "audited_splits": sorted({str(row["split"]) for row in rows}),
            "audited_sources": len({str(row["source_id"]) for row in rows}),
            "payload_points": len(rows),
            "split_magnitude_rows": len(summary_rows),
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_root),
                "audited_sources": len({str(row["source_id"]) for row in rows}),
                "payload_points": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
