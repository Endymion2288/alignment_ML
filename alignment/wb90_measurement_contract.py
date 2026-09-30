"""WB90 fail-fast measurement/frame preflight; never invokes an alignment solver.

Reads the released WB84/WB86 input catalog, not WB86 sealed results.  Runtime
probes use actual Athena Trk constructors and the frozen source fragments.
No reconstruction, propagation, conditions update, or physics repair occurs.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/research_review/wp90_measurement_contract_preflight.json"
CATALOG = ROOT / "outputs/mc24_four_station_wb86_physical_qualification_v1/wb86_event_catalog.json"
ALLOWLIST = ROOT / "outputs/mc24_four_station_calypso_physical_replicas_v1/source_allowlist.json"
OUTPUT = ROOT / "outputs/mc24_four_station_wb90_measurement_contract_v1"
FORBIDDEN = ("sealed", "blind", "00350_00399", "00800_00849", "100116", "100117")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_public(path: Path):
    resolved = path.resolve()
    if any(word in str(resolved).lower() for word in FORBIDDEN):
        raise ValueError(f"forbidden data path: {resolved}")
    return json.loads(resolved.read_text())


def write_new(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def selection(catalog: dict, allowlist: dict) -> list[dict]:
    allowed = {r["source_id"]: r["path"] for r in allowlist["sources"]}
    cells: dict[tuple[str, str], list[dict]] = {}
    families = {"identity", "identifiable_translation", "identifiable_rotation", "weak_jg_diagnostic"}
    identities = set()
    for row in catalog["events"]:
        if row["survey_mode"] != "fixed_dz":
            continue
        if row["source_id"] not in allowed or row["input_xaod"] != allowed[row["source_id"]]:
            raise ValueError("source not exactly on frozen allowlist")
        if any(x in row["input_xaod"].lower() for x in FORBIDDEN):
            raise ValueError("forbidden source")
        if row["family"] not in families or int(row["xaod_entry_index"]) < 2000:
            raise ValueError("unexpected family/entry")
        identity = (row["input_xaod"], int(row["xaod_entry_index"]))
        if identity in identities:
            raise ValueError("reused fixed-dz input entry")
        identities.add(identity)
        token = f'{row["source_id"]}:{row["input_xaod"]}:{row["xaod_entry_index"]}'
        cells.setdefault((row["source_id"], row["family"]), []).append(
            {**row, "selection_sha256": hashlib.sha256(token.encode()).hexdigest()}
        )
    if set(cells) != {(source, family) for source in allowed for family in families}:
        raise ValueError("missing source/family cell")
    out = []
    for key in sorted(cells):
        ordered = sorted(cells[key], key=lambda r: r["selection_sha256"])
        if len(ordered) < 2:
            raise ValueError("insufficient events; replacement prohibited")
        for role, row in zip(("development", "check"), ordered[:2]):
            out.append({**row, "role": role})
    return out


def angle_state(tx: float, ty: float) -> np.ndarray:
    return np.array([math.atan2(ty, tx), math.atan(math.hypot(tx, ty))])


def numerical_angle_jacobian(tx: float, ty: float, step: float) -> np.ndarray:
    point = np.array([tx, ty], dtype=float)
    out = np.zeros((2, 2))
    for c in range(2):
        plus, minus = point.copy(), point.copy()
        plus[c] += step
        minus[c] -= step
        diff = angle_state(*plus) - angle_state(*minus)
        diff[0] = math.atan2(math.sin(diff[0]), math.cos(diff[0]))
        out[:, c] = diff / (2 * step)
    return out


def correct_angle_jacobian(tx: float, ty: float) -> np.ndarray:
    r2 = tx * tx + ty * ty
    if r2 == 0:
        raise ValueError("undefined azimuth at zero transverse direction")
    r = math.sqrt(r2)
    return np.array([[-ty / r2, tx / r2], [tx / (r * (1 + r2)), ty / (r * (1 + r2))]])


def getstate_fragments(calypso: Path) -> dict:
    fitter = calypso / "Tracker/TrackerRecAlgs/TrackerSegmentFit/src/SegmentFitAlg.cxx"
    exporter = calypso / "PhysicsAnalysis/NtupleDumper/src/NtupleDumperAlg.cxx"
    text = fitter.read_text()
    body = text.split("SegmentFitAlg::GetState(", 1)[1]
    # Copy the actual current expressions, not a hand-coded bug approximation.
    start = body.index("double phi =")
    end = body.index("std::unique_ptr<FaserSCT_ClusterOnTrack>")
    covariance = body[start:end]
    export_text = exporter.read_text()
    export_start = export_text.index("using TrackletState =")
    export_end = export_text.index("std::optional<Acts::BoundTrackParameters>\nactsTrackletParameters(")
    return {"fitter": str(fitter.resolve()), "fitter_sha256": digest(fitter),
            "exporter": str(exporter.resolve()), "exporter_sha256": digest(exporter),
            "covariance_cpp": covariance, "export_cpp": export_text[export_start:export_end]}


def development_slopes(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        if row["role"] != "development":
            continue
        path = ROOT / row["replica_path"]
        if any(x in str(path).lower() for x in FORBIDDEN):
            raise ValueError("forbidden replica path")
        found = []
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                if record["event_uid"] == row["event_uid"]:
                    found.append(record)
        if len(found) != 1:
            raise ValueError("missing/duplicate event identity")
        record = found[0]
        if int(record["xaod_entry_index"]) != int(row["xaod_entry_index"]):
            raise ValueError("wrong input entry")
        for track in record["tracks"]:
            for hit in track["hits"]:
                out.append({"event_uid": row["event_uid"], "catalog_index": row["index"],
                            "station": int(hit["station_id"]), "tx": float(hit["tx"]),
                            "ty": float(hit["ty"]), "measurement_id": str(hit["measurement_id"]),
                            "declared_frame": hit["covariance_contract"]["frame"]})
    return out
