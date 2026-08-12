#!/usr/bin/env python3
"""Write a FASERNU-04 station-level tracker-alignment conditions payload.

Run this only after sourcing ``scripts/setup_environment.sh calypso``.  The
resulting SQLite file overrides just ``/Tracker/Align`` when passed to
``faser_ntuple_maker.py --tracker-align-sqlite``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path


def parse_offset(value: str) -> tuple[int, float, float]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "offset must have the form STATION:DX_MM:DY_MM"
        )
    try:
        station = int(parts[0])
        dx_mm = float(parts[1])
        dy_mm = float(parts[2])
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "offset must have integer station and finite floating-point offsets"
        ) from error
    if station not in (0, 1, 2, 3):
        raise argparse.ArgumentTypeError("station must be one of 0, 1, 2, 3")
    if not math.isfinite(dx_mm) or not math.isfinite(dy_mm):
        raise argparse.ArgumentTypeError("offset components must be finite")
    return station, dx_mm, dy_mm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a station-level /Tracker/Align SQLite+POOL payload for "
            "a controlled FASERNU-04 x/y alignment injection."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New or empty directory receiving the SQLite, POOL, catalog, and manifest",
    )
    parser.add_argument(
        "--offset",
        action="append",
        type=parse_offset,
        required=True,
        metavar="STATION:DX_MM:DY_MM",
        help="Global station translation in mm; repeat for multiple stations",
    )
    parser.add_argument("--geometry", default="FASERNU-04")
    parser.add_argument("--global-tag", default="OFLCOND-FASER-06")
    parser.add_argument("--tag", default="TRACKER-ALIGN-ML")
    parser.add_argument("--run-start", type=int, default=1)
    parser.add_argument("--run-end", type=int, default=9_999_999)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.run_start < 0 or args.run_end < args.run_start:
        raise SystemExit("--run-start and --run-end must define a non-empty non-negative range")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.tag):
        raise SystemExit("--tag may contain only letters, digits, underscore, dot, and hyphen")

    offsets: dict[int, tuple[float, float]] = {}
    for station, dx_mm, dy_mm in args.offset:
        if station in offsets:
            raise SystemExit(f"duplicate offset for station {station}")
        offsets[station] = (dx_mm, dy_mm)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = output_dir / "tracker_alignment.sqlite"
    pool_path = output_dir / "tracker_alignment.pool.root"
    catalog_path = output_dir / "PoolFileCatalog.xml"
    manifest_path = output_dir / "alignment_payload.json"
    for path in (sqlite_path, pool_path, catalog_path, manifest_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing artifact: {path}")

    alignment_constants = {
        f"station:{station}": [dx_mm, dy_mm, 0.0, 0.0, 0.0, 0.0]
        for station, (dx_mm, dy_mm) in sorted(offsets.items())
    }

    # The POOL catalog is generated relative to the process working directory.
    os.chdir(output_dir)

    try:
        from AthenaCommon.Configurable import Configurable
        from CalypsoConfiguration.AllConfigFlags import initConfigFlags
        from CalypsoConfiguration.MainServicesConfig import MainServicesCfg
        from Campaigns.Utils import Campaign
        from WriteAlignment.WriteAlignmentConfig_Faser03 import WriteAlignmentCfg
    except ImportError as error:
        raise SystemExit(
            "Calypso Python modules are unavailable. Source "
            "alignment_ML/scripts/setup_environment.sh calypso first."
        ) from error

    Configurable.configurableRun3Behavior = True
    flags = initConfigFlags()
    # This conditions-writing job has no event input.  Avoid deferred metadata
    # discovery of Athena's placeholder input file.
    flags.Input.Files = []
    flags.Input.isMC = True
    flags.Input.MCCampaign = Campaign.Unknown
    flags.Input.ProjectName = "data21"
    flags.GeoModel.FaserVersion = args.geometry
    flags.IOVDb.GlobalTag = args.global_tag
    flags.IOVDb.DatabaseInstance = "OFLP200"
    flags.IOVDb.DBConnection = f"sqlite://;schema={sqlite_path};dbname=OFLP200"
    # We are writing a standalone alignment payload, so do not read an input
    # alignment container while constructing the nominal detector hierarchy.
    flags.GeoModel.Align.Disable = True
    flags.addFlag("WriteAlignment.PoolFileName", str(pool_path))
    flags.lock()

    accumulator = MainServicesCfg(flags)
    accumulator.merge(
        WriteAlignmentCfg(
            flags,
            alignmentConstants=alignment_constants,
            ValidRunStart=args.run_start,
            ValidEvtStart=0,
            ValidRunEnd=args.run_end,
            ValidEvtEnd=9_999_999,
            CondTag=args.tag,
        )
    )
    status = accumulator.run(maxEvents=1)
    if status.isFailure():
        return 1

    missing = [path for path in (sqlite_path, pool_path, catalog_path) if not path.is_file()]
    if missing:
        raise RuntimeError(f"payload writer completed but artifacts are missing: {missing}")

    manifest = {
        "geometry": args.geometry,
        "global_tag": args.global_tag,
        "folder": "/Tracker/Align",
        "condition_tag": args.tag,
        "run_range": [args.run_start, args.run_end],
        "station_transform_convention": {
            "frame": "global",
            "components": "[dx_mm, dy_mm, dz_mm, rx_rad, ry_rad, rz_rad]",
        },
        "alignment_constants": alignment_constants,
        "sqlite": str(sqlite_path),
        "pool": str(pool_path),
        "pool_catalog": str(catalog_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
