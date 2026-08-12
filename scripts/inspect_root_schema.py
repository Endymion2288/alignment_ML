#!/usr/bin/env python3
"""Inspect a ROOT tree without assuming that it follows the canonical schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import uproot

from datasets.schema import CANONICAL_TREE_NAME, validate_tracklet_tree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="ROOT input file")
    parser.add_argument("--tree", default=None, help="Tree name; defaults to tracklets or nt")
    args = parser.parse_args()

    path = Path(args.input).expanduser().resolve()
    with uproot.open(path) as root_file:
        tree_name = args.tree
        if tree_name is None:
            tree_name = CANONICAL_TREE_NAME if CANONICAL_TREE_NAME in root_file else "nt"
        if tree_name not in root_file:
            raise SystemExit(f"tree '{tree_name}' was not found in {path}")
        tree = root_file[tree_name]
        report = validate_tracklet_tree(tree)
        payload = {
            "path": str(path),
            "tree": tree_name,
            "entries": int(tree.num_entries),
            "branch_count": len(tree.keys()),
            "canonical_schema_valid": report.is_valid,
            "has_mc_labels": report.has_mc_labels,
            "missing_required_fields": list(report.missing_required_fields),
            "missing_mc_label_fields": list(report.missing_mc_label_fields),
            "missing_optional_fields": list(report.missing_optional_fields),
            "branches": {name: tree[name].typename for name in tree.keys()},
        }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

