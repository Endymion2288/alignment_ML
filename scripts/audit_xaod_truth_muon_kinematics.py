#!/usr/bin/env python3
"""Read-only xAOD truth-muon kinematics and persistence audit.

The preferred path reads split auxiliary truth branches with uproot.  Some
MC24 xAOD productions persist ``TruthParticlesAux.`` as one unsplit EDM
object, which uproot can identify from ROOT metadata but cannot decode into
``pdgId/px/py/pz`` arrays without the Calypso xAOD dictionaries.  For those
files the tool emits an explicit metadata-only result instead of falsely
calling the source incompatible.  It never infers units, geometry, or truth
content from a filename.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import awkward as ak
import numpy as np
import uproot


DEFAULT_TREE = "CollectionTree"
DEFAULT_BRANCHES = {
    "pdg_id": "TruthParticlesAux./TruthParticlesAux.pdgId",
    "px": "TruthParticlesAux./TruthParticlesAux.px",
    "py": "TruthParticlesAux./TruthParticlesAux.py",
    "pz": "TruthParticlesAux./TruthParticlesAux.pz",
}
PERSISTENCE_CONTAINERS = {
    "truth_particles_aux": "TruthParticlesAux.",
    "truth_particles": "TruthParticles",
    "sct_clusters": "SCT_ClusterContainer",
    "segment_fit": "SegmentFit",
    "segments": "Segments",
}


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {name: None for name in ("min", "p10", "p50", "p90", "max")}
    q = np.quantile(finite, (0.0, 0.10, 0.50, 0.90, 1.0))
    return {
        "min": float(q[0]),
        "p10": float(q[1]),
        "p50": float(q[2]),
        "p90": float(q[3]),
        "max": float(q[4]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="MC xAOD ROOT file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tree", default=DEFAULT_TREE)
    parser.add_argument("--max-events", type=int, default=100)
    parser.add_argument(
        "--require-kinematics",
        action="store_true",
        help="fail rather than emit metadata-only evidence for an unsplit xAOD truth container",
    )
    args = parser.parse_args()
    if args.max_events < 1:
        raise ValueError("--max-events must be positive")
    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    with uproot.open(source) as root_file:
        if args.tree not in root_file:
            raise ValueError(f"tree '{args.tree}' is absent from {source}")
        tree = root_file[args.tree]
        available = {str(name).split(";")[0] for name in tree.keys()}
        missing = [branch for branch in DEFAULT_BRANCHES.values() if branch not in available]
        containers = {
            name: {
                "branch": branch,
                "present": branch in available,
                "typename": None if branch not in available else str(tree[branch].typename),
            }
            for name, branch in PERSISTENCE_CONTAINERS.items()
        }
        metadata_only = bool(missing)
        if metadata_only and args.require_kinematics:
            raise ValueError("required split truth branches are absent: " + ", ".join(missing))
        arrays = (
            None
            if metadata_only
            else tree.arrays(
                list(DEFAULT_BRANCHES.values()), entry_stop=int(args.max_events), library="ak"
            )
        )
        total_entries = int(tree.num_entries)

    if metadata_only:
        report = {
            "schema_version": "faser-xaod-truth-muon-kinematics-audit-v2",
            "input_xaod": str(source),
            "tree": args.tree,
            "tree_entries_total": total_entries,
            "events_read": 0,
            "access_mode": "uproot_metadata_only_unsplit_xaod_container",
            "kinematics_available": False,
            "missing_split_truth_branches": missing,
            "persistence_containers": containers,
            "conclusion": (
                "ROOT metadata confirms the named persisted containers only. "
                "Truth kinematics and segment usability still require a Calypso refit/export smoke."
            ),
        }
        _write_json(output_root / "truth_muon_kinematics.json", report)
        _write_csv(output_root / "truth_muon_counts_by_event.csv", [])
        print(json.dumps({"output_dir": str(output_root), **report}, indent=2))
        return

    assert arrays is not None  # fixed by the metadata-only branch above
    pdg = arrays[DEFAULT_BRANCHES["pdg_id"]]
    px = arrays[DEFAULT_BRANCHES["px"]]
    py = arrays[DEFAULT_BRANCHES["py"]]
    pz = arrays[DEFAULT_BRANCHES["pz"]]
    muon_mask = np.abs(pdg) == 13
    muon_px = ak.to_numpy(ak.flatten(px[muon_mask], axis=None)).astype(np.float64, copy=False)
    muon_py = ak.to_numpy(ak.flatten(py[muon_mask], axis=None)).astype(np.float64, copy=False)
    muon_pz = ak.to_numpy(ak.flatten(pz[muon_mask], axis=None)).astype(np.float64, copy=False)
    momentum = np.sqrt(muon_px**2 + muon_py**2 + muon_pz**2)
    nonzero_pz = np.isfinite(muon_pz) & (np.abs(muon_pz) > 0.0)
    tx = muon_px[nonzero_pz] / muon_pz[nonzero_pz]
    ty = muon_py[nonzero_pz] / muon_pz[nonzero_pz]
    per_event_counts = ak.to_numpy(ak.sum(muon_mask, axis=1)).astype(np.int64, copy=False)

    per_event_rows = [
        {"event_entry": int(index), "truth_muon_count": int(count)}
        for index, count in enumerate(per_event_counts.tolist())
    ]
    report = {
        "schema_version": "faser-xaod-truth-muon-kinematics-audit-v2",
        "input_xaod": str(source),
        "tree": args.tree,
        "branches": DEFAULT_BRANCHES,
        "persistence_containers": containers,
        "tree_entries_total": total_entries,
        "events_read": int(len(pdg)),
        "access_mode": "uproot_split_auxiliary_branches",
        "kinematics_available": True,
        "pdg_selection": "abs(pdgId) == 13",
        "momentum_unit": "native xAOD branch unit (not inferred by this audit)",
        "truth_muon_count": int(muon_px.size),
        "events_with_at_least_one_truth_muon": int(np.count_nonzero(per_event_counts > 0)),
        "truth_muons_per_event": _quantiles(per_event_counts),
        "momentum_magnitude_native": _quantiles(momentum),
        "px_over_pz": _quantiles(tx),
        "py_over_pz": _quantiles(ty),
    }
    _write_json(output_root / "truth_muon_kinematics.json", report)
    _write_csv(output_root / "truth_muon_counts_by_event.csv", per_event_rows)
    print(json.dumps({"output_dir": str(output_root), **report}, indent=2))


if __name__ == "__main__":
    main()
