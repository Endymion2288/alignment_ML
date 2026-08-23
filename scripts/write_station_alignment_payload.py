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
import subprocess
import sys
from pathlib import Path

MC_COOL_INSTANCE = "OFLP200"
DATA_COOL_INSTANCE = "CONDBR3"


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


def parse_transform(value: str) -> tuple[int, tuple[float, float, float, float, float, float]]:
    """Parse one exact global station transform for the physical refit chain."""
    parts = value.split(":")
    if len(parts) != 7:
        raise argparse.ArgumentTypeError(
            "transform must have the form STATION:DX_MM:DY_MM:DZ_MM:RX_RAD:RY_RAD:RZ_RAD"
        )
    try:
        station = int(parts[0])
        components = tuple(float(component) for component in parts[1:])
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "transform must have an integer station and finite floating-point components"
        ) from error
    if station not in (0, 1, 2, 3):
        raise argparse.ArgumentTypeError("station must be one of 0, 1, 2, 3")
    if len(components) != 6 or not all(math.isfinite(component) for component in components):
        raise argparse.ArgumentTypeError("transform components must be finite")
    return station, components


def parse_layer_transform(
    value: str,
) -> tuple[int, int, tuple[float, float, float, float, float, float]]:
    """Parse one IFT/SCT plane transform for Calypso L2 ``{station}{layer}`` keys."""
    parts = value.split(":")
    if len(parts) != 8:
        raise argparse.ArgumentTypeError(
            "layer transform must have the form STATION:LAYER:DX_MM:DY_MM:DZ_MM:RX_RAD:RY_RAD:RZ_RAD"
        )
    try:
        station = int(parts[0])
        layer = int(parts[1])
        components = tuple(float(component) for component in parts[2:])
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "layer transform must have integer station/layer and finite floating-point components"
        ) from error
    if station not in (0, 1, 2, 3):
        raise argparse.ArgumentTypeError("station must be one of 0, 1, 2, 3")
    if layer not in (0, 1, 2):
        raise argparse.ArgumentTypeError("layer must be one of 0, 1, 2")
    if len(components) != 6 or not all(math.isfinite(component) for component in components):
        raise argparse.ArgumentTypeError("layer transform components must be finite")
    return station, layer, components


def _cool_database_exists(sqlite_path: Path, instance: str) -> bool:
    from PyCool import cool

    db_svc = cool.DatabaseSvcFactory.databaseService()
    try:
        database = db_svc.openDatabase(
            f"sqlite://;schema={sqlite_path};dbname={instance}", True
        )
    except Exception:
        return False
    database.closeDatabase()
    return True


def _replicate_cool_instance(sqlite_path: Path, source: str, dest: str) -> None:
    """Copy a COOL instance so MC (OFLP200) and data (CONDBR3) jobs can both read it."""
    if _cool_database_exists(sqlite_path, dest):
        return
    source_str = f"sqlite://;schema={sqlite_path};dbname={source}"
    dest_str = f"sqlite://;schema={sqlite_path};dbname={dest}"
    result = subprocess.run(
        ["AtlCoolCopy", source_str, dest_str, "-create"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"AtlCoolCopy {source}->{dest} failed ({result.returncode}): "
            f"{result.stdout}{result.stderr}"
        )
    if not _cool_database_exists(sqlite_path, dest):
        raise RuntimeError(f"AtlCoolCopy did not create {dest} in {sqlite_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a station-level /Tracker/Align SQLite+POOL payload for "
            "a controlled FASERNU-04 rigid alignment injection."
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
        default=[],
        metavar="STATION:DX_MM:DY_MM",
        help="Legacy global x/y station translation in mm; repeat for multiple stations",
    )
    parser.add_argument(
        "--transform",
        action="append",
        type=parse_transform,
        default=[],
        metavar="STATION:DX_MM:DY_MM:DZ_MM:RX_RAD:RY_RAD:RZ_RAD",
        help=(
            "Exact global station transform; translations are mm and rotations are rad. "
            "Repeat for every station represented in the payload."
        ),
    )
    parser.add_argument(
        "--layer-transform",
        action="append",
        type=parse_layer_transform,
        default=[],
        metavar="STATION:LAYER:DX_MM:DY_MM:DZ_MM:RX_RAD:RY_RAD:RZ_RAD",
        help=(
            "Exact IFT/SCT plane (L2) transform written as Calypso key '{station}{layer}'. "
            "Translations are mm and rotations are rad.  Repeat for each plane."
        ),
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

    if not args.offset and not args.transform and not args.layer_transform:
        raise SystemExit("at least one --offset, --transform, or --layer-transform is required")

    transforms: dict[int, tuple[float, float, float, float, float, float]] = {}
    for station, dx_mm, dy_mm in args.offset:
        if station in transforms:
            raise SystemExit(f"duplicate transform for station {station}")
        transforms[station] = (dx_mm, dy_mm, 0.0, 0.0, 0.0, 0.0)
    for station, transform in args.transform:
        if station in transforms:
            raise SystemExit(
                f"station {station} was specified more than once; use one --transform instead of --offset"
            )
        transforms[station] = transform
    layer_transforms: dict[tuple[int, int], tuple[float, float, float, float, float, float]] = {}
    for station, layer, transform in args.layer_transform:
        key = (station, layer)
        if key in layer_transforms:
            raise SystemExit(f"duplicate layer transform for station {station} layer {layer}")
        layer_transforms[key] = transform
    if layer_transforms and not transforms:
        raise SystemExit("layer transforms require explicit station --transform entries in the same payload")

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
        f"station:{station}": list(transform)
        for station, transform in sorted(transforms.items())
    }
    alignment_constants.update(
        {
            f"{station}{layer}": list(transform)
            for (station, layer), transform in sorted(layer_transforms.items())
        }
    )

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
            "alignment_ML/scripts/setup_environment.sh calypso first "
            f"({type(error).__name__}: {error})."
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
    # WriteAlignment emits OFLP200.  Real-data ntuple jobs use CONDBR3, and
    # IOVDbSvc.getSqliteContent opens the sqlite file with the job instance.
    _replicate_cool_instance(sqlite_path, MC_COOL_INSTANCE, DATA_COOL_INSTANCE)

    manifest = {
        "geometry": args.geometry,
        "global_tag": args.global_tag,
        "folder": "/Tracker/Align",
        "condition_tag": args.tag,
        "run_range": [args.run_start, args.run_end],
        "station_transform_convention": {
            "frame": "global",
            "components": "[dx_mm, dy_mm, dz_mm, rx_rad, ry_rad, rz_rad]",
            "composition": "T(dx,dy,dz) * Rz(rz) * Ry(ry) * Rx(rx)",
            "rotation_units": "rad",
            "implementation": "TrackerAlignDBTool::stationAlignment",
        },
        "layer_transform_convention": {
            "frame": "global_conjugated_to_plane_z",
            "components": "[dx_mm, dy_mm, dz_mm, rx_rad, ry_rad, rz_rad]",
            "composition": "T(dx,dy,dz) * Rz(rz) * Ry(ry) * Rx(rx)",
            "rotation_units": "rad",
            "calypso_key": "{station}{layer}",
            "implementation": "TrackerAlignDBTool L2 Planes",
        },
        "alignment_constants": alignment_constants,
        "sqlite": str(sqlite_path),
        "sqlite_cool_instances": [MC_COOL_INSTANCE, DATA_COOL_INSTANCE],
        "pool": str(pool_path),
        "pool_catalog": str(catalog_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
