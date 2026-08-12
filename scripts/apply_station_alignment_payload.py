#!/usr/bin/env python3
"""Apply a validated station alignment payload to canonical tracklet positions.

This command produces a translation-only coordinate-level injection for V1
alignment closure.  It is intentionally separate from Calypso reconstruction:
persisted local segments in the existing xAOD are not refitted from hits under
the displaced geometry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import uproot

from alignment.payload import apply_station_coordinate_offsets, load_station_alignment_payload
from datasets.schema import CANONICAL_TREE_NAME, validate_tracklet_tree


def _columns(tree: uproot.behaviors.TTree.TTree) -> dict[str, np.ndarray]:
    raw = tree.arrays(library="np")
    if isinstance(raw, dict):
        return {name: np.asarray(value).copy() for name, value in raw.items()}
    if isinstance(raw, np.ndarray) and raw.dtype.names is not None:
        return {name: np.asarray(raw[name]).copy() for name in raw.dtype.names}
    raise TypeError("uproot returned an unsupported canonical-tracklet column container")


def apply_payload_to_tracklets(
    source: str | Path,
    destination: str | Path,
    manifest: str | Path,
) -> dict[str, object]:
    """Write a canonical ROOT copy with payload dx/dy applied to x/y only."""
    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if destination_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination_path}")
    if source_path == destination_path:
        raise ValueError("source and destination must differ")
    payload = load_station_alignment_payload(manifest)

    with uproot.open(source_path) as root_file:
        if CANONICAL_TREE_NAME not in root_file:
            raise KeyError(f"tree '{CANONICAL_TREE_NAME}' is absent from {source_path}")
        tree = root_file[CANONICAL_TREE_NAME]
        report = validate_tracklet_tree(tree)
        report.require_valid(require_mc_labels=False)
        columns = _columns(tree)

    shifted_x, shifted_y = apply_station_coordinate_offsets(
        columns["station_id"], columns["x_mm"], columns["y_mm"], payload
    )
    columns["x_mm"] = shifted_x.astype(columns["x_mm"].dtype, copy=False)
    columns["y_mm"] = shifted_y.astype(columns["y_mm"].dtype, copy=False)
    station_ids = np.asarray(columns["station_id"], dtype=np.int64)
    counts = {
        int(station): int(np.count_nonzero(station_ids == station))
        for station in np.unique(station_ids)
    }
    summary = {
        "method": "condition-payload_global-station_xy_coordinate-injection",
        "source": str(source_path),
        "destination": str(destination_path),
        "payload_manifest": str(payload.manifest_path),
        "sqlite": str(payload.sqlite_path),
        "pool": str(payload.pool_path),
        "pool_catalog": str(payload.pool_catalog_path),
        "coordinate_convention": "x' = x + dx_station; y' = y + dy_station",
        "limitations": (
            "No raw-hit reconstruction, local refit, or source-state ACTS repropagation is run; "
            "this is a translation-only local-tracklet coordinate surrogate."
        ),
        "offsets_xy_mm": {str(station): list(offset) for station, offset in payload.offsets_xy_mm.items()},
        "tracklets_per_station": {str(station): count for station, count in counts.items()},
    }
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(destination_path) as root_file:
        root_file[CANONICAL_TREE_NAME] = columns
        root_file["alignment_injection"] = {
            "manifest_json": np.asarray([json.dumps(summary, sort_keys=True)]),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Canonical nominal tracklet ROOT file")
    parser.add_argument("--output", required=True, help="New canonical coordinate-injected ROOT file")
    parser.add_argument("--payload-manifest", required=True, help="alignment_payload.json")
    parser.add_argument(
        "--report",
        default=None,
        help="JSON summary path (default: OUTPUT with .injection.json suffix)",
    )
    args = parser.parse_args()
    output_path = Path(args.output).expanduser().resolve()
    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report is not None
        else output_path.with_suffix(".injection.json")
    )
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {report_path}")
    summary = apply_payload_to_tracklets(args.input, output_path, args.payload_manifest)
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
