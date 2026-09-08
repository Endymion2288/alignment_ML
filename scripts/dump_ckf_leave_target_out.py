#!/usr/bin/env python3
"""Athena job: independent leave-target-out dump helper.

Must be launched under ``source scripts/setup_environment.sh calypso``.
Does not call KalmanFitterTool.fit, does not construct Acts objects in
Python, does not edit Calypso, and does not write /Tracker/Align.
This file is standalone so Athena does not import the ML-stack
``alignment`` / ``datasets`` packages.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = ROOT / "build" / "leave_target_out_dump"
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
    library = PLUGIN_DIR / "libCkfLeaveTargetOutDump.so"
    components = PLUGIN_DIR / "libCkfLeaveTargetOutDump.components"
    if not library.is_file() or not components.is_file():
        raise FileNotFoundError(
            "Independent LTO helper is absent. "
            "Run scripts/build_ckf_leave_target_out_dump.sh first."
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


def _station_z(config: dict) -> list[float]:
    raw = config["station_z_mm"]
    values = []
    for index in range(4):
        if index in raw:
            values.append(float(raw[index]))
        else:
            values.append(float(raw[str(index)]))
    return values


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
    parser.add_argument("--min-remaining-measurements", type=int, default=8)
    parser.add_argument(
        "--seed-covariance-scale",
        type=float,
        default=1.0,
        help="Pre-registered uninformative-seed scale for B14R/B14S sensitivity",
    )
    parser.add_argument(
        "--seed-covariance-direction",
        default="all",
        help="all, x, y, tx, ty, or q_over_p",
    )
    parser.add_argument(
        "--directional-seed-campaign",
        action="store_true",
        help="B14S: refit each direction at 0.1 / 1 / 10 in one job",
    )
    parser.add_argument(
        "--enable-profile-likelihood",
        action="store_true",
        help="B14M: write measurement-only profile rows",
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="B14M: skip Kalman export",
    )
    parser.add_argument(
        "--enable-profile-seed-invariance",
        action="store_true",
        help="B14M: restart from pre-registered init variants",
    )
    parser.add_argument(
        "--enable-profile-numerics",
        action="store_true",
        help="B14N: sequential transport, scaling, backtracking, per-hit audit",
    )
    parser.add_argument(
        "--profile-evaluate-only",
        action="store_true",
        help="B14N: write chi2 evaluations without optimizing",
    )
    parser.add_argument(
        "--enable-profile-transport",
        action="store_true",
        help="B14T: supporting-plane measurement transport",
    )
    parser.add_argument(
        "--profile-record-stepper-path",
        action="store_true",
        help="B14T: record downsampled stepper checkpoints",
    )
    parser.add_argument(
        "--profile-diagnostic-max-steps",
        type=int,
        default=0,
        help="B14T diagnostic maxSteps overlay; 0 unused",
    )
    parser.add_argument(
        "--select-event-ids",
        default="",
        help="Comma-separated event numbers; empty means all",
    )
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
        alg_factory = CompFactory.getComp("CkfLeaveTargetOutDumpAlg")
    except Exception as exc:
        raise RuntimeError(
            "Independent LTO helper is not visible to CompFactory. "
            "Rebuild scripts/build_ckf_leave_target_out_dump.sh"
        ) from exc

    flags = build_flags(args)
    acc = MainServicesCfg(flags)
    acc.merge(PoolReadCfg(flags))
    acc.merge(FaserGeometryCfg(flags))
    acc.merge(MagneticFieldSvcCfg(flags))
    geo_acc, geo_tool = ActsTrackingGeometryToolCfg(flags)
    acc.merge(geo_acc)

    alg = alg_factory("CkfLeaveTargetOutDumpAlg")
    alg.OutputJsonl = str(output.resolve())
    alg.SourceId = str(args.source_id)
    alg.TrackCollection = str(args.collection)
    alg.GeometryHash = hashes["geometry_hash"]
    alg.FieldHash = hashes["field_hash"]
    alg.MaterialHash = hashes["material_hash"]
    alg.ConditionsHash = hashes["conditions_hash"]
    alg.StationZmm = _station_z(config)
    alg.TargetStations = [1, 2, 3]
    alg.MinRemainingMeasurements = int(args.min_remaining_measurements)
    alg.SeedCovarianceScale = float(args.seed_covariance_scale)
    alg.SeedCovarianceDirection = str(args.seed_covariance_direction)
    alg.EnableDirectionalSeedCampaign = bool(args.directional_seed_campaign)
    alg.EnableProfileLikelihood = bool(args.enable_profile_likelihood)
    alg.ProfileOnly = bool(args.profile_only)
    alg.EnableProfileSeedInvariance = bool(args.enable_profile_seed_invariance)
    alg.EnableProfileNumerics = bool(args.enable_profile_numerics)
    alg.ProfileEvaluateOnly = bool(args.profile_evaluate_only)
    alg.EnableProfileTransport = bool(args.enable_profile_transport)
    alg.ProfileSupportingPlane = True
    alg.ProfileRecordStepperPath = bool(args.profile_record_stepper_path)
    alg.ProfileDiagnosticMaxSteps = int(args.profile_diagnostic_max_steps)
    if bool(args.enable_profile_numerics):
        alg.ProfileMaxIterations = 50
    if str(args.select_event_ids).strip():
        alg.SelectEventIds = [
            int(item) for item in str(args.select_event_ids).split(",") if item.strip()
        ]
    alg.TrackingGeometryTool = geo_tool
    alg.OutputLevel = INFO
    acc.addEventAlgo(alg, primary=True)
    sc = acc.run(maxEvents=int(args.nevents))
    if output.is_file():
        print(f"wrote {output} sha256={_sha256_file(output)}")
    else:
        print(f"LTO dump file was not written: {output}")
        return 1
    return 0 if sc.isSuccess() else 1


if __name__ == "__main__":
    raise SystemExit(main())
