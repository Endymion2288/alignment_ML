#!/usr/bin/env python3
"""Athena job: inspect official CKFTrackCollectionWithoutIFT provenance.

Must be launched under ``source scripts/setup_environment.sh calypso``.
Does not change the persisted collection, seed, geometry, field, or
covariance scale.  A live KalmanFitter call is a labeled diagnostic
re-attempt and is never written back.  This file is standalone so
Athena does not import the ML-stack packages.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = ROOT / "build" / "three_st_qp_refit_provenance_dump"
DEFAULT_CONFIG = ROOT / "configs" / "three_st_qp_refit_provenance_v1.yaml"
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
    library = PLUGIN_DIR / "libCkfThreeStQpRefitProvenance.so"
    components = PLUGIN_DIR / "libCkfThreeStQpRefitProvenance.components"
    if not library.is_file() or not components.is_file():
        raise FileNotFoundError(
            "C++ 3ST q/p refit-provenance helper is absent. "
            "Run scripts/build_ckf_three_st_qp_refit_provenance_dump.sh first."
        )
    current = os.environ.get("LD_LIBRARY_PATH", "")
    prefix = str(PLUGIN_DIR)
    if not current.split(os.pathsep) or current.split(os.pathsep)[0] != prefix:
        os.environ["LD_LIBRARY_PATH"] = (
            prefix if not current else f"{prefix}{os.pathsep}{current}"
        )
    jobopt = os.environ.get("JOBOPTSEARCHPATH", "")
    if prefix not in jobopt.split(os.pathsep):
        os.environ["JOBOPTSEARCHPATH"] = (
            prefix if not jobopt else f"{prefix}{os.pathsep}{jobopt}"
        )


def _focus_pairs(config: dict, source_id: str) -> str:
    pairs = [
        f"{int(row['event_id'])}:{int(row['track_index'])}"
        for row in config["focus_identities"]
        if row["source_id"] == source_id
    ]
    if not pairs:
        raise ValueError(f"no frozen focus pairs for {source_id}")
    if len(pairs) > 20:
        raise ValueError("focus list exceeds 20")
    return ",".join(pairs)


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
    parser.add_argument("--event-output", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--nevents", type=int, default=20)
    parser.add_argument("--skip-events", type=int, default=0)
    parser.add_argument("--campaign", default="smoke")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    _prepare_plugin()
    config = _load_yaml(Path(args.config))
    if bool(config.get("three_st_qp_trusted_observable", True)):
        raise RuntimeError("three_st_qp_trusted_observable must stay false")
    if bool(config.get("residual_conditional_authorized", True)):
        raise RuntimeError("residual_conditional_authorized must stay false")
    output = Path(args.output)
    event_output = Path(args.event_output)
    if output.exists() or event_output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing dump: {output} / {event_output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    focus_pairs = _focus_pairs(config, args.source_id)

    from AthenaCommon.Constants import INFO
    from AthenaConfiguration.ComponentFactory import CompFactory
    from AthenaPoolCnvSvc.PoolReadConfig import PoolReadCfg
    from CalypsoConfiguration.MainServicesConfig import MainServicesCfg
    from FaserGeoModel.FaserGeoModelConfig import FaserGeometryCfg
    from MagFieldServices.MagFieldServicesConfig import MagneticFieldSvcCfg

    try:
        alg_factory = CompFactory.getComp("CkfThreeStQpRefitProvenanceAlg")
    except Exception as exc:
        raise RuntimeError(
            "C++ helper is not visible to CompFactory. "
            "Rebuild scripts/build_ckf_three_st_qp_refit_provenance_dump.sh"
        ) from exc

    flags = build_flags(args)
    acc = MainServicesCfg(flags)
    acc.merge(PoolReadCfg(flags))
    acc.merge(FaserGeometryCfg(flags))
    acc.merge(MagneticFieldSvcCfg(flags))

    matching = CompFactory.TrackTruthMatchingTool("TrackTruthMatchingTool")
    fiducial = CompFactory.FiducialParticleTool("FiducialParticleTool")
    acc.addPublicTool(matching)
    acc.addPublicTool(fiducial)

    alg = alg_factory("CkfThreeStQpRefitProvenanceAlg")
    alg.OutputJsonl = str(output.resolve())
    alg.EventJsonl = str(event_output.resolve())
    alg.SourceId = str(args.source_id)
    alg.Campaign = str(args.campaign)
    alg.TrackCollection = "CKFTrackCollectionWithoutIFT"
    alg.OfficialTruthStation = 1
    alg.FocusPairs = focus_pairs
    alg.SkipEvents = int(args.skip_events)
    alg.TrackTruthMatchingTool = matching
    alg.FiducialParticleTool = fiducial
    alg.OutputLevel = INFO
    acc.addEventAlgo(alg, primary=True)
    sc = acc.run(maxEvents=int(args.nevents))
    if output.is_file() and event_output.is_file():
        print(f"wrote {output} sha256={_sha256_file(output)}")
        print(f"wrote {event_output} sha256={_sha256_file(event_output)}")
    else:
        print(f"dump file was not written: {output} / {event_output}")
        return 1
    return 0 if sc.isSuccess() else 1


if __name__ == "__main__":
    raise SystemExit(main())
