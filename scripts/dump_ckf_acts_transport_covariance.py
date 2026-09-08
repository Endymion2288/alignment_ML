#!/usr/bin/env python3
"""Athena job: load the C++ ACTS transport dump helper.

Must be launched under ``source scripts/setup_environment.sh calypso``.
Does not construct Acts objects in Python, does not edit Calypso, and does
not write /Tracker/Align.  This file is standalone so Athena does not import
the ML-stack ``alignment`` / ``datasets`` packages.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = ROOT / "build" / "acts_transport_dump"
DEFAULT_CONFIG = ROOT / "configs" / "acts_transport_dump_v1.yaml"
MATERIAL_MAP = (
    "/cvmfs/faser.cern.ch/repo/sw/database/DBRelease/current/acts/material-maps-alma9.json"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _prepare_plugin() -> None:
    library = PLUGIN_DIR / "libCkfActsTransportDump.so"
    components = PLUGIN_DIR / "libCkfActsTransportDump.components"
    if not library.is_file() or not components.is_file():
        raise FileNotFoundError(
            "C++ dump helper is absent. "
            "Run scripts/build_ckf_acts_transport_dump.sh first."
        )
    current = os.environ.get("LD_LIBRARY_PATH", "")
    prefix = str(PLUGIN_DIR)
    if not current.split(os.pathsep) or current.split(os.pathsep)[0] != prefix:
        os.environ["LD_LIBRARY_PATH"] = (
            prefix if not current else f"{prefix}{os.pathsep}{current}"
        )


def _hashes(config: dict) -> dict[str, str]:
    tags = config["tags"]
    calypso = str(config["software_provenance"]["calypso_git_sha"])
    material = Path(str(config["material_map"]["path"]))
    material_sha = _sha256_file(material)
    return {
        "geometry_hash": hashlib.sha256(
            f"{tags['geometry']}|{calypso}".encode("utf-8")
        ).hexdigest(),
        "field_hash": hashlib.sha256(
            f"{tags['field']}|{calypso}".encode("utf-8")
        ).hexdigest(),
        "conditions_hash": hashlib.sha256(
            f"{tags['conditions']}|{tags['database_instance']}|{calypso}".encode("utf-8")
        ).hexdigest(),
        "material_hash": hashlib.sha256(
            f"{material}|{material_sha}|{calypso}".encode("utf-8")
        ).hexdigest(),
    }


def build_flags(args):
    from AthenaCommon.Configurable import Configurable
    from CalypsoConfiguration.AllConfigFlags import initConfigFlags

    Configurable.configurableRun3Behavior = True
    flags = initConfigFlags()
    flags.Input.Files = [str(Path(args.input_xaod).resolve())]
    flags.Input.isMC = True
    flags.GeoModel.FaserVersion = "FASERNU-04"
    flags.IOVDb.GlobalTag = "OFLCOND-FASER-06"
    flags.IOVDb.DatabaseInstance = "OFLP200"
    flags.Input.ProjectName = "data21"
    flags.Common.isOnline = False
    flags.Beam.NumberOfCollisions = 0.0
    flags.Detector.GeometryFaserSCT = True
    try:
        flags.TrackingGeometry.MaterialSource = MATERIAL_MAP
    except Exception:
        flags.addFlag("TrackingGeometry.MaterialSource", MATERIAL_MAP)
    if int(args.skip_events) > 0:
        flags.Exec.SkipEvents = int(args.skip_events)
    flags.lock()
    return flags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-xaod", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--nevents", type=int, default=100)
    parser.add_argument("--skip-events", type=int, default=0)
    parser.add_argument("--collection", default="CKFTrackCollection")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    _prepare_plugin()
    config = _load_yaml(Path(args.config))
    hashes = _hashes(config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    from AthenaCommon.Constants import INFO
    from AthenaConfiguration.ComponentFactory import CompFactory
    from AthenaPoolCnvSvc.PoolReadConfig import PoolReadCfg
    from CalypsoConfiguration.MainServicesConfig import MainServicesCfg
    from FaserActsGeometry.ActsGeometryConfig import ActsTrackingGeometryToolCfg
    from FaserGeoModel.FaserGeoModelConfig import FaserGeometryCfg
    from MagFieldServices.MagFieldServicesConfig import MagneticFieldSvcCfg

    try:
        alg_factory = CompFactory.getComp("CkfActsTransportDumpAlg")
    except Exception as exc:
        raise RuntimeError(
            "C++ helper is not visible to CompFactory. "
            "Rebuild scripts/build_ckf_acts_transport_dump.sh"
        ) from exc

    flags = build_flags(args)
    acc = MainServicesCfg(flags)
    acc.merge(PoolReadCfg(flags))
    acc.merge(FaserGeometryCfg(flags))
    acc.merge(MagneticFieldSvcCfg(flags))
    geo_acc, geo_tool = ActsTrackingGeometryToolCfg(flags)
    acc.merge(geo_acc)

    def _make_tool(name: str, ms: bool, eloss: bool, record: bool):
        tool = CompFactory.FaserActsExtrapolationTool(name)
        tool.MaxSteps = 10000
        tool.TrackingGeometryTool = geo_tool
        tool.InteractionMultiScatering = ms
        tool.InteractionEloss = eloss
        tool.InteractionRecord = record
        acc.addPublicTool(tool)
        return tool

    tool0 = _make_tool("ExtrapolationNoNoise", False, False, False)
    tool1 = _make_tool("ExtrapolationWithNoise", True, True, True)
    station_z = [float(config["station_z_mm"][i]) for i in range(4)]
    alg = alg_factory("CkfActsTransportDumpAlg")
    alg.OutputJsonl = str(output.resolve())
    alg.SourceId = str(args.source_id)
    alg.TrackCollection = str(args.collection)
    alg.GeometryHash = hashes["geometry_hash"]
    alg.FieldHash = hashes["field_hash"]
    alg.MaterialHash = hashes["material_hash"]
    alg.ConditionsHash = hashes["conditions_hash"]
    alg.StationZmm = station_z
    alg.TargetStations = [1, 2, 3]
    alg.ExtrapolationNoNoise = tool0
    alg.ExtrapolationWithNoise = tool1
    alg.TrackingGeometryTool = geo_tool
    alg.OutputLevel = INFO
    acc.addEventAlgo(alg, primary=True)
    sc = acc.run(maxEvents=int(args.nevents))
    if output.is_file():
        print(f"wrote {output} sha256={_sha256_file(output)}")
    else:
        print(f"dump file was not written: {output}")
        return 1
    return 0 if sc.isSuccess() else 1


if __name__ == "__main__":
    raise SystemExit(main())
