#!/usr/bin/env python3
"""Athena dump of persisted CKFTrackCollection 5x5.  No refit, no payload write.

Must be launched under ``source scripts/setup_environment.sh calypso``.
Reads the existing reconstructed xAOD only.  Does not run SegmentFit, does not
write /Tracker/Align, and does not use truth q/p.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _matrix5(cov) -> list[list[float]] | None:
    if cov is None:
        return None
    rows = []
    for i in range(5):
        rows.append([float(cov(i, j)) for j in range(5)])
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    tmp.replace(path)


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
    args = parser.parse_args()

    from AthenaCommon.Constants import INFO
    from AthenaPoolCnvSvc.PoolReadConfig import PoolReadCfg
    from AthenaPython.PyAthena import StatusCode
    from AthenaPython.PyAthenaComps import Alg
    from CalypsoConfiguration.MainServicesConfig import MainServicesCfg
    from FaserGeoModel.FaserGeoModelConfig import FaserGeometryCfg
    from MagFieldServices.MagFieldServicesConfig import MagneticFieldSvcCfg

    output = Path(args.output)
    collection = str(args.collection)
    source_id = str(args.source_id)
    rows: list[dict] = []

    class CkfQoverPDumpAlg(Alg):
        def initialize(self):
            return StatusCode.Success

        def execute(self):
            store = self.evtStore
            if not store.contains("TrackCollection", collection):
                return StatusCode.Success
            tracks = store.retrieve("TrackCollection", collection)
            event_info = store.retrieve("xAOD::EventInfo", "EventInfo")
            run_id = int(event_info.runNumber())
            event_id = int(event_info.eventNumber())
            for index, track in enumerate(tracks):
                if track is None or track.trackParameters() is None:
                    continue
                if track.trackParameters().empty():
                    continue
                params = track.trackParameters().front()
                if params is None:
                    continue
                values = params.parameters()
                native = [float(values[i]) for i in range(5)]
                cov = _matrix5(params.covariance())
                position = params.position()
                momentum = params.momentum()
                pz = float(momentum.z())
                if abs(pz) < 1.0e-18:
                    continue
                p_mev = math.sqrt(
                    float(momentum.x()) ** 2
                    + float(momentum.y()) ** 2
                    + float(momentum.z()) ** 2
                )
                quality = track.fitQuality()
                rows.append(
                    {
                        "source_id": source_id,
                        "run_id": run_id,
                        "event_id": event_id,
                        "track_index": int(index),
                        "collection": collection,
                        "parameter_type": type(params).__name__,
                        "native_state": native,
                        "native_covariance": cov,
                        "has_covariance": cov is not None,
                        "derived_state": [
                            float(position.x()),
                            float(position.y()),
                            float(momentum.x()) / pz,
                            float(momentum.y()) / pz,
                            native[4],
                        ],
                        "charge": float(params.charge()),
                        "p_mev": p_mev,
                        "chi2": float(quality.chiSquared()) if quality else None,
                        "ndof": float(quality.numberDoF()) if quality else None,
                        "is_truth": False,
                    }
                )
            return StatusCode.Success

        def finalize(self):
            _write_jsonl(output, rows)
            return StatusCode.Success

    flags = build_flags(args)
    acc = MainServicesCfg(flags)
    acc.merge(PoolReadCfg(flags))
    acc.merge(FaserGeometryCfg(flags))
    acc.merge(MagneticFieldSvcCfg(flags))
    acc.addEventAlgo(
        CkfQoverPDumpAlg("CkfQoverPDumpAlg"),
        primary=True,
    )
    acc.getEventAlgo("CkfQoverPDumpAlg").OutputLevel = INFO
    sc = acc.run(maxEvents=int(args.nevents))
    if output.is_file():
        print(f"wrote {len(rows)} tracks to {output}")
    return 0 if sc.isSuccess() else 1


if __name__ == "__main__":
    raise SystemExit(main())
