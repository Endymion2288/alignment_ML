#!/usr/bin/env python3
"""Convert detailed NtupleDumper tracklet branches into the flat ML schema.

The Calypso exporter keeps its normal event-wise ROOT layout.  This converter
is the only supported path from that layout to the canonical ``tracklets``
tree consumed by the association baseline.  It deliberately rejects branch
length mismatches and drops records explicitly marked as lacking a usable
covariance instead of manufacturing uncertainties.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import uproot

from datasets.schema import CANONICAL_TREE_NAME, MC_LABEL_FIELDS, SCHEMA_VERSION


EVENT_BRANCHES = {
    "run_id": "run",
    "event_id": "eventID",
}

TRACKLET_BRANCHES = {
    "station_id": "Tracklet_station_id",
    "tracklet_id": "Tracklet_id",
    "x_mm": "Tracklet_x_mm",
    "y_mm": "Tracklet_y_mm",
    "z_mm": "Tracklet_z_mm",
    "tx": "Tracklet_tx",
    "ty": "Tracklet_ty",
    "cov_xx_mm2": "Tracklet_cov_xx_mm2",
    "cov_xy_mm2": "Tracklet_cov_xy_mm2",
    "cov_xtx_mm": "Tracklet_cov_xtx_mm",
    "cov_xty_mm": "Tracklet_cov_xty_mm",
    "cov_yy_mm2": "Tracklet_cov_yy_mm2",
    "cov_ytx_mm": "Tracklet_cov_ytx_mm",
    "cov_yty_mm": "Tracklet_cov_yty_mm",
    "cov_txtx": "Tracklet_cov_txtx",
    "cov_txty": "Tracklet_cov_txty",
    "cov_tyty": "Tracklet_cov_tyty",
    "chi2": "Tracklet_chi2",
    "ndof": "Tracklet_ndof",
    "n_hit": "Tracklet_n_hit",
    "hit_pattern": "Tracklet_hit_pattern",
}

TRUTH_BRANCHES = {
    "truth_particle_id": "Tracklet_truth_particle_id",
    "truth_pdg": "Tracklet_truth_pdg",
    "truth_match_fraction": "Tracklet_truth_match_fraction",
}

# All of these branches are optional so the original MC22 electron smoke-test
# files remain valid inputs.  The MC24 field-aware exporter supplies the first
# three today; the boolean validity branch is accepted when ROOT writes it.
Q_OVER_P_BRANCHES = {
    "q_over_p_per_mev": "Tracklet_q_over_p_per_MeV",
    "q_over_p_from_momentum_per_mev": "Tracklet_q_over_p_from_momentum_per_MeV",
    "q_over_p_variance_per_mev2": "Tracklet_q_over_p_variance_per_MeV2",
    "has_q_over_p_covariance": "Tracklet_has_q_over_p_covariance",
    "truth_q_over_p_per_mev": "Tracklet_truth_q_over_p_per_MeV",
}

VALIDITY_BRANCH = "Tracklet_has_covariance"

OUTPUT_DTYPES = {
    "run_id": np.int64,
    "event_id": np.int64,
    "station_id": np.int8,
    "tracklet_id": np.int32,
    "x_mm": np.float64,
    "y_mm": np.float64,
    "z_mm": np.float64,
    "tx": np.float64,
    "ty": np.float64,
    "cov_xx_mm2": np.float64,
    "cov_xy_mm2": np.float64,
    "cov_xtx_mm": np.float64,
    "cov_xty_mm": np.float64,
    "cov_yy_mm2": np.float64,
    "cov_ytx_mm": np.float64,
    "cov_yty_mm": np.float64,
    "cov_txtx": np.float64,
    "cov_txty": np.float64,
    "cov_tyty": np.float64,
    "chi2": np.float64,
    "ndof": np.float64,
    "n_hit": np.int16,
    "hit_pattern": np.uint64,
    "truth_particle_id": np.int64,
    "truth_pdg": np.int32,
    "truth_match_fraction": np.float64,
    "q_over_p_per_mev": np.float64,
    "q_over_p_from_momentum_per_mev": np.float64,
    "q_over_p_variance_per_mev2": np.float64,
    "has_q_over_p_covariance": np.bool_,
    "truth_q_over_p_per_mev": np.float64,
}


@dataclass(frozen=True)
class ConversionSummary:
    source: str
    destination: str
    tree: str
    events_read: int
    tracklets_read: int
    tracklets_written: int
    dropped_missing_covariance: int
    includes_mc_labels: bool
    optional_fields: tuple[str, ...]


def _branch_names(tree: Any) -> set[str]:
    return {str(name).split(";")[0] for name in tree.keys()}


def _require_branches(tree: Any, branch_map: dict[str, str], description: str) -> None:
    available = _branch_names(tree)
    missing = [branch for branch in branch_map.values() if branch not in available]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{description} branches are missing: {joined}")


def _check_event_vector_lengths(
    arrays: dict[str, ak.Array], tracklet_branch_names: list[str]
) -> np.ndarray:
    anchor = arrays[tracklet_branch_names[0]]
    lengths = np.asarray(ak.to_numpy(ak.num(anchor, axis=1)), dtype=np.int64)
    for name in tracklet_branch_names[1:]:
        candidate = np.asarray(ak.to_numpy(ak.num(arrays[name], axis=1)), dtype=np.int64)
        if not np.array_equal(candidate, lengths):
            raise ValueError(
                f"event-wise vector length mismatch between "
                f"{tracklet_branch_names[0]} and {name}"
            )
    return lengths


def _flat_output_columns(
    arrays: dict[str, ak.Array],
    include_truth: bool,
    drop_missing_covariance: bool,
    optional_branches: dict[str, str],
) -> tuple[dict[str, np.ndarray], int, int]:
    tracklet_names = list(TRACKLET_BRANCHES.values())
    if include_truth:
        tracklet_names.extend(TRUTH_BRANCHES.values())
    tracklet_names.extend(optional_branches.values())
    if VALIDITY_BRANCH in arrays:
        tracklet_names.append(VALIDITY_BRANCH)

    lengths = _check_event_vector_lengths(arrays, tracklet_names)
    total = int(lengths.sum())
    anchor = arrays[TRACKLET_BRANCHES["tracklet_id"]]

    columns: dict[str, np.ndarray] = {}
    for canonical_name, event_branch in EVENT_BRANCHES.items():
        broadcast, _ = ak.broadcast_arrays(arrays[event_branch], anchor)
        columns[canonical_name] = np.asarray(ak.to_numpy(ak.flatten(broadcast, axis=1)))
    for canonical_name, tracklet_branch in TRACKLET_BRANCHES.items():
        columns[canonical_name] = np.asarray(
            ak.to_numpy(ak.flatten(arrays[tracklet_branch], axis=1))
        )
    if include_truth:
        for canonical_name, tracklet_branch in TRUTH_BRANCHES.items():
            columns[canonical_name] = np.asarray(
                ak.to_numpy(ak.flatten(arrays[tracklet_branch], axis=1))
            )
    for canonical_name, tracklet_branch in optional_branches.items():
        columns[canonical_name] = np.asarray(
            ak.to_numpy(ak.flatten(arrays[tracklet_branch], axis=1))
        )

    valid = np.ones(total, dtype=bool)
    if drop_missing_covariance and VALIDITY_BRANCH in arrays:
        valid = np.asarray(
            ak.to_numpy(ak.flatten(arrays[VALIDITY_BRANCH], axis=1)), dtype=bool
        )

    dropped = int((~valid).sum())
    for name, values in columns.items():
        columns[name] = np.asarray(values[valid], dtype=OUTPUT_DTYPES[name])
    return columns, total, dropped


def convert_ntuple_tracklets(
    source: str | Path,
    destination: str | Path,
    tree_name: str = "nt",
    output_tree_name: str = CANONICAL_TREE_NAME,
    include_truth: bool = False,
    drop_missing_covariance: bool = True,
    step_size: str = "100 MB",
) -> ConversionSummary:
    """Convert an enhanced NtupleDumper ROOT file to canonical flat rows."""
    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    with uproot.open(source_path) as input_file:
        if tree_name not in input_file:
            raise ValueError(f"tree '{tree_name}' is absent from {source_path}")
        tree = input_file[tree_name]
        _require_branches(tree, EVENT_BRANCHES, "event")
        _require_branches(tree, TRACKLET_BRANCHES, "tracklet")
        if include_truth:
            _require_branches(tree, TRUTH_BRANCHES, "MC truth")
        available = _branch_names(tree)
        optional_branches = {
            canonical_name: event_branch
            for canonical_name, event_branch in Q_OVER_P_BRANCHES.items()
            if event_branch in available
        }
        requested = [*EVENT_BRANCHES.values(), *TRACKLET_BRANCHES.values()]
        if include_truth:
            requested.extend(TRUTH_BRANCHES.values())
        requested.extend(optional_branches.values())
        if VALIDITY_BRANCH in available:
            requested.append(VALIDITY_BRANCH)

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        event_count = 0
        tracklet_count = 0
        written_count = 0
        dropped_count = 0
        output_fields = [*EVENT_BRANCHES, *TRACKLET_BRANCHES]
        if include_truth:
            output_fields.extend(MC_LABEL_FIELDS)
        output_fields.extend(optional_branches)
        output_types = {field: OUTPUT_DTYPES[field] for field in output_fields}

        with uproot.recreate(destination_path) as output_file:
            output_file.mktree(output_tree_name, output_types)
            for arrays in tree.iterate(requested, step_size=step_size, library="ak", how=dict):
                batch_columns, seen, dropped = _flat_output_columns(
                    arrays,
                    include_truth=include_truth,
                    drop_missing_covariance=drop_missing_covariance,
                    optional_branches=optional_branches,
                )
                event_count += len(arrays[EVENT_BRANCHES["run_id"]])
                tracklet_count += seen
                dropped_count += dropped
                batch_written = seen - dropped
                if batch_written:
                    output_file[output_tree_name].extend(batch_columns)
                    written_count += batch_written
            output_file["metadata"] = {
                "schema_version": np.asarray([SCHEMA_VERSION]),
                "coordinate_unit": np.asarray(["mm"]),
                "source_file": np.asarray([str(source_path)]),
                "converter": np.asarray(["scripts.convert_ntuple_tracklets"]),
                "includes_mc_labels": np.asarray([include_truth]),
                "optional_fields": np.asarray([json.dumps(sorted(optional_branches))]),
            }

    return ConversionSummary(
        source=str(source_path),
        destination=str(destination_path),
        tree=tree_name,
        events_read=event_count,
        tracklets_read=tracklet_count,
        tracklets_written=written_count,
        dropped_missing_covariance=dropped_count,
        includes_mc_labels=include_truth,
        optional_fields=tuple(sorted(optional_branches)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Enhanced NtupleDumper ROOT file")
    parser.add_argument("--output", required=True, help="Canonical ROOT output")
    parser.add_argument("--tree", default="nt", help="Input ROOT tree name")
    parser.add_argument(
        "--output-tree", default=CANONICAL_TREE_NAME, help="Output ROOT tree name"
    )
    parser.add_argument(
        "--include-truth",
        action="store_true",
        help="Require and export per-tracklet MC truth labels",
    )
    parser.add_argument(
        "--keep-missing-covariance",
        action="store_true",
        help="Keep rows marked as missing covariance (not usable by the baseline)",
    )
    parser.add_argument("--step-size", default="100 MB", help="uproot iteration size")
    args = parser.parse_args()

    summary = convert_ntuple_tracklets(
        args.input,
        args.output,
        tree_name=args.tree,
        output_tree_name=args.output_tree,
        include_truth=args.include_truth,
        drop_missing_covariance=not args.keep_missing_covariance,
        step_size=args.step_size,
    )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
