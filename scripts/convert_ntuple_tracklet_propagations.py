#!/usr/bin/env python3
"""Flatten NtupleDumper field-aware tracklet propagation audit branches."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot

from datasets.propagation_loader import (
    PROPAGATION_SCHEMA_VERSION,
    PROPAGATION_TREE_NAME,
    REQUIRED_PROPAGATION_FIELDS,
)


EVENT_BRANCHES = {
    "run_id": "run",
    "event_id": "eventID",
}

PROPAGATION_BRANCHES = {
    "source_tracklet_id": "TrackletPropagation_source_tracklet_id",
    "target_tracklet_id": "TrackletPropagation_target_tracklet_id",
    "source_station_id": "TrackletPropagation_source_station_id",
    "target_station_id": "TrackletPropagation_target_station_id",
    "truth_particle_id": "TrackletPropagation_truth_particle_id",
    "target_z_mm": "TrackletPropagation_target_z_mm",
    "pred_x_mm": "TrackletPropagation_x_mm",
    "pred_y_mm": "TrackletPropagation_y_mm",
    "pred_tx": "TrackletPropagation_tx",
    "pred_ty": "TrackletPropagation_ty",
    "pred_cov_xx_mm2": "TrackletPropagation_cov_xx_mm2",
    "pred_cov_xy_mm2": "TrackletPropagation_cov_xy_mm2",
    "pred_cov_xtx_mm": "TrackletPropagation_cov_xtx_mm",
    "pred_cov_xty_mm": "TrackletPropagation_cov_xty_mm",
    "pred_cov_yy_mm2": "TrackletPropagation_cov_yy_mm2",
    "pred_cov_ytx_mm": "TrackletPropagation_cov_ytx_mm",
    "pred_cov_yty_mm": "TrackletPropagation_cov_yty_mm",
    "pred_cov_txtx": "TrackletPropagation_cov_txtx",
    "pred_cov_txty": "TrackletPropagation_cov_txty",
    "pred_cov_tyty": "TrackletPropagation_cov_tyty",
    "success": "TrackletPropagation_success",
    "has_covariance": "TrackletPropagation_has_covariance",
}

OPTIONAL_PROPAGATION_BRANCHES = {
    "q_over_p_mode": "TrackletPropagation_q_over_p_mode",
    "source_q_over_p_per_mev": "TrackletPropagation_source_q_over_p_per_MeV",
}

OUTPUT_DTYPES = {
    "run_id": np.int64,
    "event_id": np.int64,
    "source_tracklet_id": np.int32,
    "target_tracklet_id": np.int32,
    "source_station_id": np.int16,
    "target_station_id": np.int16,
    "truth_particle_id": np.int64,
    "target_z_mm": np.float64,
    "pred_x_mm": np.float64,
    "pred_y_mm": np.float64,
    "pred_tx": np.float64,
    "pred_ty": np.float64,
    "pred_cov_xx_mm2": np.float64,
    "pred_cov_xy_mm2": np.float64,
    "pred_cov_xtx_mm": np.float64,
    "pred_cov_xty_mm": np.float64,
    "pred_cov_yy_mm2": np.float64,
    "pred_cov_ytx_mm": np.float64,
    "pred_cov_yty_mm": np.float64,
    "pred_cov_txtx": np.float64,
    "pred_cov_txty": np.float64,
    "pred_cov_tyty": np.float64,
    "success": np.bool_,
    "has_covariance": np.bool_,
    "q_over_p_mode": np.int8,
    "source_q_over_p_per_mev": np.float64,
}


@dataclass(frozen=True)
class PropagationConversionSummary:
    source: str
    destination: str
    tree: str
    events_read: int
    records_written: int
    successful_records: int
    records_with_covariance: int


def _branch_names(tree: Any) -> set[str]:
    return {str(name).split(";")[0] for name in tree.keys()}


def _require_branches(tree: Any, branch_map: dict[str, str], description: str) -> None:
    available = _branch_names(tree)
    missing = [branch for branch in branch_map.values() if branch not in available]
    if missing:
        raise ValueError(f"{description} branches are missing: {', '.join(missing)}")


def _check_event_vector_lengths(
    arrays: dict[str, ak.Array], propagation_branches: dict[str, str]
) -> tuple[np.ndarray, ak.Array]:
    names = list(propagation_branches.values())
    anchor = arrays[names[0]]
    lengths = np.asarray(ak.to_numpy(ak.num(anchor, axis=1)), dtype=np.int64)
    for name in names[1:]:
        candidate = np.asarray(ak.to_numpy(ak.num(arrays[name], axis=1)), dtype=np.int64)
        if not np.array_equal(candidate, lengths):
            raise ValueError(f"event-wise propagation vector length mismatch: {name}")
    return lengths, anchor


def _flat_columns(
    arrays: dict[str, ak.Array], propagation_branches: dict[str, str]
) -> tuple[dict[str, np.ndarray], int]:
    lengths, anchor = _check_event_vector_lengths(arrays, propagation_branches)
    columns: dict[str, np.ndarray] = {}
    for canonical_name, event_branch in EVENT_BRANCHES.items():
        broadcast, _ = ak.broadcast_arrays(arrays[event_branch], anchor)
        columns[canonical_name] = np.asarray(ak.to_numpy(ak.flatten(broadcast, axis=1)))
    for canonical_name, event_branch in propagation_branches.items():
        columns[canonical_name] = np.asarray(
            ak.to_numpy(ak.flatten(arrays[event_branch], axis=1))
        )
    for name, values in columns.items():
        columns[name] = np.asarray(values, dtype=OUTPUT_DTYPES[name])
    return columns, int(lengths.sum())


def convert_ntuple_tracklet_propagations(
    source: str | Path,
    destination: str | Path,
    tree_name: str = "nt",
    output_tree_name: str = PROPAGATION_TREE_NAME,
    step_size: str = "100 MB",
) -> PropagationConversionSummary:
    """Convert event-wise field propagation branches to a flat pair tree."""
    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    with uproot.open(source_path) as input_file:
        if tree_name not in input_file:
            raise ValueError(f"tree '{tree_name}' is absent from {source_path}")
        tree = input_file[tree_name]
        _require_branches(tree, EVENT_BRANCHES, "event")
        _require_branches(tree, PROPAGATION_BRANCHES, "tracklet propagation")
        available = _branch_names(tree)
        optional_branches = {
            name: branch
            for name, branch in OPTIONAL_PROPAGATION_BRANCHES.items()
            if branch in available
        }
        all_propagation_branches = {**PROPAGATION_BRANCHES, **optional_branches}
        requested = [*EVENT_BRANCHES.values(), *all_propagation_branches.values()]

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        events_read = 0
        records_written = 0
        successful_records = 0
        records_with_covariance = 0
        with uproot.recreate(destination_path) as output_file:
            output_file.mktree(
                output_tree_name,
                {
                    field: OUTPUT_DTYPES[field]
                    for field in (*REQUIRED_PROPAGATION_FIELDS, *optional_branches)
                },
            )
            for arrays in tree.iterate(requested, step_size=step_size, library="ak", how=dict):
                columns, count = _flat_columns(arrays, all_propagation_branches)
                events_read += len(arrays[EVENT_BRANCHES["run_id"]])
                if count:
                    output_file[output_tree_name].extend(columns)
                    records_written += count
                    successful_records += int(np.count_nonzero(columns["success"]))
                    records_with_covariance += int(
                        np.count_nonzero(columns["has_covariance"])
                    )
            output_file["metadata"] = {
                "schema_version": np.asarray([PROPAGATION_SCHEMA_VERSION]),
                "coordinate_unit": np.asarray(["mm"]),
                "source_file": np.asarray([str(source_path)]),
                "converter": np.asarray(
                    ["scripts.convert_ntuple_tracklet_propagations"]
                ),
                "optional_fields": np.asarray([json.dumps(sorted(optional_branches))]),
            }
    return PropagationConversionSummary(
        source=str(source_path),
        destination=str(destination_path),
        tree=tree_name,
        events_read=events_read,
        records_written=records_written,
        successful_records=successful_records,
        records_with_covariance=records_with_covariance,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Enhanced NtupleDumper ROOT file")
    parser.add_argument("--output", required=True, help="Flat propagation ROOT output")
    parser.add_argument("--tree", default="nt", help="Input ROOT tree name")
    parser.add_argument(
        "--output-tree", default=PROPAGATION_TREE_NAME, help="Output ROOT tree name"
    )
    parser.add_argument("--step-size", default="100 MB", help="uproot iteration size")
    args = parser.parse_args()
    summary = convert_ntuple_tracklet_propagations(
        args.input,
        args.output,
        tree_name=args.tree,
        output_tree_name=args.output_tree,
        step_size=args.step_size,
    )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
