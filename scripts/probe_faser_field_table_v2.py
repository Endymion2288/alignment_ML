#!/usr/bin/env python3
"""Read-only field-contract probe for FaserFieldTable_v2.

Records B_x samples and a straight-line ∫ B_x dz along x=y=0 if the
ROOT table is found.  Does not convert bending_raw to q/p.  Units and
the trajectory-dependent Jacobian remain unlocked.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

CANDIDATES = (
    "/cvmfs/faser.cern.ch/repo/sw/software/22.0/faser/offline/ReleaseData/v20/MagneticFieldMaps/FaserFieldTable_v2.root",
    "/cvmfs/faser.cern.ch/repo/sw/database/DBRelease/current/geomDB/FaserFieldTable_v2.root",
    "/cvmfs/faser.cern.ch/repo/sw/database/DBRelease/current/GLOBAL-BField-Maps-03/FaserFieldTable_v2.root",
    "/cvmfs/faser.cern.ch/repo/sw/database/DBRelease/current/FaserFieldTable_v2.root",
)

S1_Z_MM = 47.4
S3_Z_MM = 2427.4
SERVICE_MAP = "MagneticFieldMaps/FaserFieldTable_v2.root"


def _nearest(mesh, value: float) -> int:
    return min(range(len(mesh)), key=lambda idx: abs(float(mesh[idx]) - value))


def main() -> int:
    found = next((Path(path) for path in CANDIDATES if Path(path).is_file()), None)
    payload = {
        "kind": "faser_field_table_v2_contract",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "official_qp_like_jacobian_authorized": False,
        "path_integral_locked": False,
        "bending_converted_to_qp": False,
        "service_full_map_file": SERVICE_MAP,
        "iovdb_folders": ["/GLOBAL/BField/Maps", "/GLOBAL/BField/Scales"],
        "faser_field_svc_ascii_comment": "readMap ASCII (commented): convert m->mm and T->kT",
        "bfield_cache_scale_comment": "BFieldCache.h: m_scale is unit of stored field in kT",
        "source_txt_readme_units": "FaserBFieldData/data/README.txt: length mm, B Tesla, z in [0,4400]",
        "candidates": list(CANDIDATES),
        "found": str(found) if found else None,
    }
    if found is None:
        payload["status"] = "unresolved"
        payload["reason"] = "FaserFieldTable_v2.root not in the frozen candidate paths"
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    try:
        import ROOT  # type: ignore
    except Exception as exc:
        payload["status"] = "recorded_file_present"
        payload["reason"] = f"ROOT is not importable: {exc}"
        payload["note"] = (
            "Table file is present.  Trajectory-dependent path integral, "
            "units, and Jacobian are not locked; do not convert bending_raw."
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    handle = ROOT.TFile.Open(str(found))
    if handle is None or handle.IsZombie():
        payload["status"] = "unresolved"
        payload["reason"] = "ROOT file failed to open"
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    keys = [key.GetName() for key in handle.GetListOfKeys()]
    payload["keys"] = keys
    tree = handle.Get("BFieldMap")
    if tree is None:
        payload["status"] = "recorded_file_present"
        payload["reason"] = "BFieldMap tree absent"
        handle.Close()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    zones = []
    samples = []
    unofficial_integral = 0.0
    unofficial_n = 0
    for entry in range(int(tree.GetEntries())):
        tree.GetEntry(entry)
        zone = {
            "id": int(tree.id),
            "xmin": float(tree.xmin),
            "xmax": float(tree.xmax),
            "ymin": float(tree.ymin),
            "ymax": float(tree.ymax),
            "zmin": float(tree.zmin),
            "zmax": float(tree.zmax),
            "bscale": float(tree.bscale),
            "nmeshx": int(tree.nmeshx),
            "nmeshy": int(tree.nmeshy),
            "nmeshz": int(tree.nmeshz),
        }
        zones.append(zone)
        if not (zone["xmin"] <= 0.0 <= zone["xmax"] and zone["ymin"] <= 0.0 <= zone["ymax"]):
            continue
        meshx = [float(tree.meshx[i]) for i in range(zone["nmeshx"])]
        meshy = [float(tree.meshy[i]) for i in range(zone["nmeshy"])]
        meshz = [float(tree.meshz[i]) for i in range(zone["nmeshz"])]
        ix = _nearest(meshx, 0.0)
        iy = _nearest(meshy, 0.0)
        axis = []
        for iz, z_mm in enumerate(meshz):
            if z_mm < S1_Z_MM or z_mm > S3_Z_MM:
                continue
            idx = (ix * zone["nmeshy"] + iy) * zone["nmeshz"] + iz
            bx_stored = float(tree.fieldx[idx]) * zone["bscale"]
            axis.append((z_mm, bx_stored))
        if not axis:
            continue
        for z_mm, bx_stored in axis[:: max(1, len(axis) // 8)]:
            samples.append(
                {
                    "x": 0.0,
                    "y": 0.0,
                    "z_mesh": z_mm,
                    "bx_stored_times_bscale": bx_stored,
                    "zone_id": zone["id"],
                    "units": "unlocked_likely_kT_if_ATLAS_cache_convention",
                }
            )
        for left, right in zip(axis, axis[1:]):
            unofficial_integral += 0.5 * (left[1] + right[1]) * (right[0] - left[0])
            unofficial_n += 1
    handle.Close()
    payload["n_zones"] = len(zones)
    payload["zones"] = zones
    payload["bx_samples_x0_y0"] = samples
    payload["unofficial_straight_line_int_Bx_dz"] = {
        "value": unofficial_integral if unofficial_n else None,
        "n_segments": unofficial_n,
        "path": "x=y=0, mesh z in [S1,S3]",
        "s1_z_mm": S1_Z_MM,
        "s3_z_mm": S3_Z_MM,
        "official": False,
        "note": (
            "Not a trajectory path integral.  Stored B units follow "
            "bscale * short; BFieldCache comments say kT.  Mesh extent "
            "must still be confirmed as mm.  Do not invert to q/p."
        ),
    }
    payload["status"] = "recorded_file_present"
    payload["note"] = (
        "Table file is present.  Trajectory-dependent path integral, "
        "units, and Jacobian are not locked; do not convert bending_raw."
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
