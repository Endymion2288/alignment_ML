#!/usr/bin/env python3
"""Athena dump of persisted CKFTrackCollectionWithoutIFT hit stations and 5x5.

Must be launched under ``source scripts/setup_environment.sh calypso``.
Reads existing reconstructed xAOD only.  Does not refit, does not write
/Tracker/Align, and does not use truth q/p.  Station IDs come from
FaserSCT_ID; if that helper cannot be retrieved the dump fails closed.
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
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing dump: {path}")
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


def _retrieve_id_helper(alg):
    store = alg.detStore
    for key in ("FaserSCT_ID", "SCT_ID"):
        try:
            helper = store.retrieve(key)
            if helper is not None:
                return helper, key
        except Exception:
            try:
                helper = store[key]
                if helper is not None:
                    return helper, key
            except Exception:
                continue
    return None, None


def _hit_from_measurement(meas, id_helper) -> dict | None:
    if meas is None:
        return None
    identify = getattr(meas, "identify", None)
    if identify is None:
        return None
    ident = identify()
    station = int(id_helper.station(ident))
    layer = int(id_helper.layer(ident))
    side = int(id_helper.side(ident))
    z_mm = None
    is_interface = None
    prep = getattr(meas, "prepRawData", None)
    cluster = prep() if prep is not None else None
    if cluster is not None:
        if hasattr(cluster, "globalPosition"):
            z_mm = float(cluster.globalPosition().z())
        det = cluster.detectorElement() if hasattr(cluster, "detectorElement") else None
        if det is not None and hasattr(det, "isInterface"):
            is_interface = bool(det.isInterface())
    return {
        "station": station,
        "layer": layer,
        "side": side,
        "layer_code": 6 * station + 2 * layer + side,
        "z_mm": z_mm,
        "is_interface": is_interface,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-xaod", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--event-output", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--nevents", type=int, default=20)
    parser.add_argument("--skip-events", type=int, default=0)
    parser.add_argument(
        "--collections",
        nargs="+",
        default=["CKFTrackCollectionWithoutIFT", "CKFTrackCollection"],
    )
    args = parser.parse_args()

    from AthenaCommon.Constants import INFO
    from AthenaPoolCnvSvc.PoolReadConfig import PoolReadCfg
    from AthenaPython.PyAthena import StatusCode
    from AthenaPython.PyAthenaComps import Alg
    from CalypsoConfiguration.MainServicesConfig import MainServicesCfg
    from FaserGeoModel.FaserGeoModelConfig import FaserGeometryCfg
    from MagFieldServices.MagFieldServicesConfig import MagneticFieldSvcCfg

    output = Path(args.output)
    event_output = Path(args.event_output)
    collections = [str(name) for name in args.collections]
    source_id = str(args.source_id)
    track_rows: list[dict] = []
    event_rows: list[dict] = []
    id_helper_holder: dict = {"helper": None, "key": None, "error": None}

    class CkfWithoutIftDefinitionDumpAlg(Alg):
        def initialize(self):
            helper, key = _retrieve_id_helper(self)
            if helper is None:
                id_helper_holder["error"] = "FaserSCT_ID not retrieved from detStore"
                return StatusCode.Failure
            id_helper_holder["helper"] = helper
            id_helper_holder["key"] = key
            return StatusCode.Success

        def execute(self):
            store = self.evtStore
            helper = id_helper_holder["helper"]
            event_info = store.retrieve("xAOD::EventInfo", "EventInfo")
            run_id = int(event_info.runNumber())
            event_id = int(event_info.eventNumber())
            present = {}
            for name in collections:
                present[name] = bool(store.contains("TrackCollection", name))
            cluster_counts = {str(i): 0 for i in range(4)}
            cluster_unavailable = False
            cluster_key = None
            for type_name, key in (
                ("Tracker::FaserSCT_ClusterContainer", "SCT_ClusterContainer"),
                ("FaserSCT_ClusterContainer", "SCT_ClusterContainer"),
            ):
                if store.contains(type_name, key):
                    cluster_key = (type_name, key)
                    break
            if cluster_key is None:
                cluster_unavailable = True
            else:
                container = store.retrieve(cluster_key[0], cluster_key[1])
                for collection in container:
                    for cluster in collection:
                        ident = cluster.identify()
                        station = int(helper.station(ident))
                        if 0 <= station <= 3:
                            cluster_counts[str(station)] += 1
            event_rows.append(
                {
                    "source_id": source_id,
                    "run_id": run_id,
                    "event_id": event_id,
                    "collections_present": present,
                    "cluster_station_counts": cluster_counts,
                    "cluster_stations_unavailable": cluster_unavailable,
                    "station_decoded_by_fasersct_id": True,
                    "id_helper_key": id_helper_holder["key"],
                    "is_truth": False,
                }
            )
            for name in collections:
                if not present[name]:
                    continue
                tracks = store.retrieve("TrackCollection", name)
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
                    mot_stations = []
                    mot_hits = []
                    if track.measurementsOnTrack() is not None:
                        for meas in track.measurementsOnTrack():
                            hit = _hit_from_measurement(meas, helper)
                            if hit is None:
                                continue
                            mot_hits.append(hit)
                            mot_stations.append(hit["station"])
                    tsos_stations = []
                    tsos_hits = []
                    n_outlier = 0
                    if track.trackStateOnSurfaces() is not None:
                        for tsos in track.trackStateOnSurfaces():
                            if tsos is None or tsos.measurementOnTrack() is None:
                                continue
                            hit = _hit_from_measurement(tsos.measurementOnTrack(), helper)
                            if hit is None:
                                continue
                            is_outlier = False
                            if hasattr(tsos, "type"):
                                try:
                                    import ROOT

                                    is_outlier = bool(
                                        tsos.type(ROOT.Trk.TrackStateOnSurface.Outlier)
                                    )
                                except Exception:
                                    is_outlier = False
                            hit = dict(hit)
                            hit["from_outlier"] = is_outlier
                            tsos_hits.append(hit)
                            tsos_stations.append(hit["station"])
                            if is_outlier:
                                n_outlier += 1
                    position = params.position()
                    momentum = params.momentum()
                    p_mev = math.sqrt(
                        float(momentum.x()) ** 2
                        + float(momentum.y()) ** 2
                        + float(momentum.z()) ** 2
                    )
                    quality = track.fitQuality()
                    track_rows.append(
                        {
                            "source_id": source_id,
                            "run_id": run_id,
                            "event_id": event_id,
                            "track_index": int(index),
                            "collection": name,
                            "parameter_type": type(params).__name__,
                            "native_state": native,
                            "native_covariance": cov,
                            "has_covariance": cov is not None,
                            "charge": float(params.charge()),
                            "p_mev": p_mev,
                            "x_mm": float(position.x()),
                            "y_mm": float(position.y()),
                            "z_mm": float(position.z()),
                            "chi2": float(quality.chiSquared()) if quality else None,
                            "ndof": float(quality.numberDoF()) if quality else None,
                            "measurements_on_track_stations": mot_stations,
                            "tsos_stations": tsos_stations,
                            "n_outlier_hits": n_outlier,
                            "mot_hits": mot_hits,
                            "tsos_hits": tsos_hits,
                            "station_decoded_by_fasersct_id": True,
                            "is_truth": False,
                        }
                    )
            return StatusCode.Success

        def finalize(self):
            _write_jsonl(output, track_rows)
            _write_jsonl(event_output, event_rows)
            return StatusCode.Success

    flags = build_flags(args)
    acc = MainServicesCfg(flags)
    acc.merge(PoolReadCfg(flags))
    acc.merge(FaserGeometryCfg(flags))
    acc.merge(MagneticFieldSvcCfg(flags))
    acc.addEventAlgo(
        CkfWithoutIftDefinitionDumpAlg("CkfWithoutIftDefinitionDumpAlg"),
        primary=True,
    )
    acc.getEventAlgo("CkfWithoutIftDefinitionDumpAlg").OutputLevel = INFO
    sc = acc.run(maxEvents=int(args.nevents))
    if output.is_file():
        print(f"wrote {len(track_rows)} tracks to {output}")
        print(f"wrote {len(event_rows)} events to {event_output}")
    if id_helper_holder["error"]:
        print(id_helper_holder["error"])
        return 2
    return 0 if sc.isSuccess() else 1


if __name__ == "__main__":
    raise SystemExit(main())
