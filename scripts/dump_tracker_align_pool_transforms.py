#!/usr/bin/env python3
"""Dump official CVMFS Align POOL files to JSON.

Must be sourced with ``scripts/setup_environment.sh calypso`` so ROOT can
read AlignableTransform_p1.  Does not write conditions or geometry.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


POOL_FILES = (
    "FASER-02_Align.pool.root",
    "FASER-03_Align.pool.root",
    "FASER-04_2022_Align.pool.root",
    "FASER-05_2023_Align.pool.root",
    "FASER-05_2024_Align.pool.root",
    "FASER-06_2023_Align.pool.root",
    "FASER-06_2024_Align.pool.root",
    "FASER-06_2025_Align.pool.root",
)


def extract_alpha_beta_gamma(xx, xy, xz, _dx, _yx, yy, yz, _dy, _zx, zy, zz, _dz):
    siny = max(-1.0, min(1.0, float(xz)))
    beta = math.asin(siny)
    if yz == 0.0 and zz == 0.0:
        gamma = 0.0
        alpha = math.atan2(yy, zy)
    else:
        alpha = math.atan2(-yz, zz)
        gamma = math.atan2(-xy, xx)
        if alpha == 0.0:
            alpha = 0.0
        if gamma == 0.0:
            gamma = 0.0
    return alpha, beta, gamma


def is_identity(vals, atol=1.0e-12) -> bool:
    xx, xy, xz, dx, yx, yy, yz, dy, zx, zy, zz, dz = vals
    return (
        abs(xx - 1) < atol
        and abs(yy - 1) < atol
        and abs(zz - 1) < atol
        and abs(xy) < atol
        and abs(xz) < atol
        and abs(yx) < atol
        and abs(yz) < atol
        and abs(zx) < atol
        and abs(zy) < atol
        and abs(dx) < atol
        and abs(dy) < atol
        and abs(dz) < atol
    )


def dump_file(path: Path) -> dict:
    import ROOT

    handle = ROOT.TFile.Open(str(path))
    tree = handle.Get("ConditionsContainerAlignableTransform_p1")
    entries = []
    n_nonzero = 0
    for index in range(int(tree.GetEntries())):
        tree.GetEntry(index)
        obj = tree.AlignableTransform_p1
        tag = str(obj.m_tag)
        ids = [int(item) for item in obj.m_ids]
        trans = [float(item) for item in obj.m_trans]
        members = []
        for position, ident in enumerate(ids):
            vals = trans[12 * position : 12 * position + 12]
            rx, ry, rz = extract_alpha_beta_gamma(*vals)
            identity = is_identity(vals)
            if not identity:
                n_nonzero += 1
            members.append(
                {
                    "identifier32": ident,
                    "identifier32_hex": hex(ident),
                    "matrix_row_major_3x4": vals,
                    "dx_mm": vals[3],
                    "dy_mm": vals[7],
                    "dz_mm": vals[11],
                    "rx_rad": rx,
                    "ry_rad": ry,
                    "rz_rad": rz,
                    "is_identity": identity,
                }
            )
        entries.append(
            {
                "pool_entry": index,
                "alignable_transform_tag": tag,
                "n_members": len(members),
                "members": members,
            }
        )
    stat = path.stat()
    handle.Close()
    return {
        "path": str(path),
        "mtime": stat.st_mtime,
        "size_bytes": stat.st_size,
        "n_alignable_transforms": len(entries),
        "n_nonzero_members": n_nonzero,
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool-dir",
        default="/cvmfs/faser.cern.ch/repo/sw/database/DBRelease/current/poolcond",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        import ROOT  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "ROOT is unavailable. Source scripts/setup_environment.sh calypso first "
            f"({type(error).__name__}: {error})."
        ) from error
    pool_dir = Path(args.pool_dir)
    payload = {name: dump_file(pool_dir / name) for name in POOL_FILES}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({name: blob["n_nonzero_members"] for name, blob in payload.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
