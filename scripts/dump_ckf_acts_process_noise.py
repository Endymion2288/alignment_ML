#!/usr/bin/env python3
"""Athena dump: transport persisted CKF 5x5 with ACTS process noise on and off.

Must be launched under ``source scripts/setup_environment.sh calypso``.
Reads the existing reconstructed xAOD only.  Does not refit SegmentFit, does
not write /Tracker/Align, and does not use truth q/p as Cin.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


MATERIAL_MAP = (
    "/cvmfs/faser.cern.ch/repo/sw/database/DBRelease/current/acts/material-maps-alma9.json"
)
STATION_Z_MM = {0: -1860.15, 1: 47.4, 2: 1237.4, 3: 2427.4}


def _matrix5(cov) -> list[list[float]] | None:
    if cov is None:
        return None
    return [[float(cov(i, j)) for j in range(5)] for i in range(5)]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    tmp.replace(path)


def bound_to_global_4d(loc0, loc1, phi, theta, bound_cov):
    """Map a z-plane bound chart to global (x, y, tx, ty)."""
    tx = math.tan(theta) * math.cos(phi)
    ty = math.tan(theta) * math.sin(phi)
    sec2 = 1.0 / (math.cos(theta) ** 2)
    jac = [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -math.tan(theta) * math.sin(phi), math.cos(phi) * sec2, 0.0],
        [0.0, 0.0, math.tan(theta) * math.cos(phi), math.sin(phi) * sec2, 0.0],
    ]
    cov4 = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            total = 0.0
            for a in range(5):
                for b in range(5):
                    total += jac[i][a] * bound_cov[a][b] * jac[j][b]
            cov4[i][j] = total
    return [float(loc0), float(loc1), float(tx), float(ty)], cov4


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


def _read_bound_cov(params) -> list[list[float]] | None:
    cov = params.covariance()
    if cov is None:
        return None
    try:
        matrix = cov.value() if hasattr(cov, "value") else cov
    except Exception:
        matrix = cov
    rows = []
    try:
        for i in range(5):
            rows.append([float(matrix(i, j)) for j in range(5)])
        return rows
    except Exception:
        try:
            for i in range(5):
                rows.append([float(matrix[i, j]) for j in range(5)])
            return rows
        except Exception:
            return None


def _export_propagated(params, geometry_context) -> dict:
    position = params.position(geometry_context) if geometry_context is not None else params.position()
    momentum = params.momentum()
    values = params.parameters()
    loc0 = float(values[0])
    loc1 = float(values[1])
    phi = float(values[2])
    theta = float(values[3])
    bound_cov = _read_bound_cov(params)
    if bound_cov is None:
        return {"success": False, "reason": "missing_bound_covariance"}
    state, cov4 = bound_to_global_4d(loc0, loc1, phi, theta, bound_cov)
    pz = float(momentum.z()) if hasattr(momentum, "z") else float(momentum[2])
    if abs(pz) < 1.0e-18:
        return {"success": False, "reason": "pz_zero"}
    px = float(momentum.x()) if hasattr(momentum, "x") else float(momentum[0])
    py = float(momentum.y()) if hasattr(momentum, "y") else float(momentum[1])
    return {
        "success": True,
        "state_xy_tx_ty": [float(position.x()), float(position.y()), px / pz, py / pz],
        "bound_state_xy_tx_ty": state,
        "covariance_4x4": cov4,
        "z_mm": float(position.z()),
    }


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
    from AthenaConfiguration.ComponentFactory import CompFactory
    from AthenaPoolCnvSvc.PoolReadConfig import PoolReadCfg
    from AthenaPython.PyAthena import StatusCode
    from AthenaPython.PyAthenaComps import Alg
    from CalypsoConfiguration.MainServicesConfig import MainServicesCfg
    from FaserActsGeometry.ActsGeometryConfig import ActsTrackingGeometryToolCfg
    from FaserGeoModel.FaserGeoModelConfig import FaserGeometryCfg
    from MagFieldServices.MagFieldServicesConfig import MagneticFieldSvcCfg

    output = Path(args.output)
    collection = str(args.collection)
    source_id = str(args.source_id)
    rows: list[dict] = []
    propagate_error = {"model0": None, "model1": None}

    class CkfActsProcessNoiseDumpAlg(Alg):
        def initialize(self):
            try:
                self.tool0 = self.toolsvc.retrieve("FaserActsExtrapolationTool/ExtrapolationNoNoise")
                self.tool1 = self.toolsvc.retrieve("FaserActsExtrapolationTool/ExtrapolationWithNoise")
                self.geo_tool = self.toolsvc.retrieve("FaserActsTrackingGeometryTool/TrackingGeometryTool")
            except Exception as exc:
                propagate_error["model0"] = f"tool_retrieve:{exc}"
                self.tool0 = None
                self.tool1 = None
                self.geo_tool = None
            return StatusCode.Success

        def _propagate(self, tool, start, target_z, ctx):
            raise RuntimeError("Acts BoundTrackParameters is not constructible from PyAthena")

        def _acts_start(self, params):
            raise RuntimeError(
                "Acts BoundTrackParameters requires a C++ helper; "
                "PyAthena has no safe Acts surface bindings"
            )

        def execute(self):
            store = self.evtStore
            if not store.contains("TrackCollection", collection):
                return StatusCode.Success
            tracks = store.retrieve("TrackCollection", collection)
            event_info = store.retrieve("xAOD::EventInfo", "EventInfo")
            run_id = int(event_info.runNumber())
            event_id = int(event_info.eventNumber())
            ctx = self.getContext() if hasattr(self, "getContext") else None
            try:
                from GaudiPython.Bindings import gbl

                ctx = gbl.Gaudi.Hive.currentContext()
            except Exception:
                pass
            for index, track in enumerate(tracks):
                if track is None or track.trackParameters() is None or track.trackParameters().empty():
                    continue
                params = track.trackParameters().front()
                if params is None:
                    continue
                values = params.parameters()
                native = [float(values[i]) for i in range(5)]
                cov5 = _matrix5(params.covariance())
                position = params.position()
                momentum = params.momentum()
                pz = float(momentum.z())
                if abs(pz) < 1.0e-18:
                    continue
                p_mev = math.sqrt(
                    float(momentum.x()) ** 2 + float(momentum.y()) ** 2 + float(momentum.z()) ** 2
                )
                source_z = float(position.z())
                try:
                    start = self._acts_start(params)
                except Exception as exc:
                    start = None
                    propagate_error["model0"] = f"acts_start:{exc}"
                nearest_source = min(STATION_Z_MM, key=lambda sid: abs(STATION_Z_MM[sid] - source_z))
                for target in (1, 2, 3):
                    if STATION_Z_MM[target] <= source_z and nearest_source >= target:
                        continue
                    row = {
                        "source_id": source_id,
                        "run_id": run_id,
                        "event_id": event_id,
                        "track_index": int(index),
                        "collection": collection,
                        "is_truth": False,
                        "native_state": native,
                        "native_covariance": cov5,
                        "has_covariance": cov5 is not None,
                        "q_over_p_per_mev": native[4],
                        "p_mev": p_mev,
                        "charge": float(params.charge()),
                        "source_z_mm": source_z,
                        "source_station": int(nearest_source),
                        "target_station": int(target),
                        "target_z_mm": float(STATION_Z_MM[target]),
                        "derived_state": [
                            float(position.x()),
                            float(position.y()),
                            float(momentum.x()) / pz,
                            float(momentum.y()) / pz,
                            native[4],
                        ],
                    }
                    if start is None or self.tool0 is None or self.tool1 is None:
                        row["model0_no_process_noise"] = {
                            "success": False,
                            "reason": propagate_error["model0"] or "acts_start_unavailable",
                        }
                        row["model1_acts_process_noise"] = {
                            "success": False,
                            "reason": propagate_error["model1"] or "acts_start_unavailable",
                        }
                    else:
                        try:
                            row["model0_no_process_noise"] = self._propagate(
                                self.tool0, start, STATION_Z_MM[target], ctx
                            )
                        except Exception as exc:
                            row["model0_no_process_noise"] = {
                                "success": False,
                                "reason": f"propagate0:{exc}",
                            }
                        try:
                            row["model1_acts_process_noise"] = self._propagate(
                                self.tool1, start, STATION_Z_MM[target], ctx
                            )
                        except Exception as exc:
                            row["model1_acts_process_noise"] = {
                                "success": False,
                                "reason": f"propagate1:{exc}",
                            }
                    rows.append(row)
            return StatusCode.Success

        def finalize(self):
            _write_jsonl(output, rows)
            return StatusCode.Success

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

    _make_tool("ExtrapolationNoNoise", False, False, False)
    _make_tool("ExtrapolationWithNoise", True, True, True)
    acc.addEventAlgo(CkfActsProcessNoiseDumpAlg("CkfActsProcessNoiseDumpAlg"), primary=True)
    acc.getEventAlgo("CkfActsProcessNoiseDumpAlg").OutputLevel = INFO
    sc = acc.run(maxEvents=int(args.nevents))
    if output.is_file():
        print(f"wrote {len(rows)} rows to {output}")
    return 0 if sc.isSuccess() else 1


if __name__ == "__main__":
    raise SystemExit(main())
