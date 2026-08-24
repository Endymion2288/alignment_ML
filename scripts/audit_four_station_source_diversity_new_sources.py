#!/usr/bin/env python3
"""Workbook 64: pre-production audit of the four authorized new train xAODs.

Checks provenance, source-disjointness, ROOT persistence, V3-identity
content / covariance / four-station occupancy.  Does not train, does not
open reserved blind or sealed test, and does not treat V3 identity banks
as a substitute for the four-station 15-DoF curriculum.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import uproot
import yaml

from datasets.root_loader import load_events
from scripts.audit_refitted_source_diversity import summarize_refitted_source
from scripts.audit_xaod_truth_muon_kinematics import PERSISTENCE_CONTAINERS
from scripts.run_refit_multidof_closure import _json_ready
from training.source_diversity_audit import (
    AUTHORIZED_NEW_TRAIN_SOURCES,
    CURRENT_TRAIN_SOURCES,
    DEVELOPMENT_SOURCES,
    HISTORY_ONLY_SOURCES,
    OUTLIER_SOURCES,
    RESERVED_BLIND_SOURCES,
    TRANSFER_DIAGNOSTIC_SOURCES,
    UNUSED_RESERVE_SOURCES,
    assert_sources_allowed,
    charge_from_source_id,
    is_sealed_source,
)


V3_PHYSICAL_ROOT = Path(
    "/eos/home-x/xcheng/FASER/alignment_ML/outputs/mc24_v3_expanded_trainval_physical_v1/sources"
)
V3_IDENTITY_POINTS = {
    "mc24_100043_00300_00399": "mag_0_train_00",
    "mc24_100044_00200_00299": "mag_0_train_00",
    "mc24_100047_00100_00149": "mag_0_validation_00",
    "mc24_100048_00100_00149": "mag_0_validation_00",
}
REQUIRED_CONTAINERS = ("sct_clusters", "segment_fit", "segments", "truth_particles_aux")
FORBIDDEN_OVERLAP = (
    CURRENT_TRAIN_SOURCES
    | DEVELOPMENT_SOURCES
    | TRANSFER_DIAGNOSTIC_SOURCES
    | set(RESERVED_BLIND_SOURCES)
    | set(UNUSED_RESERVE_SOURCES)
    | set(OUTLIER_SOURCES)
)
MIN_STATION_EVENTS = 1
MIN_PD_FRACTION = 0.95


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _expected_charge(source_id: str) -> str:
    charge = charge_from_source_id(source_id)
    if charge is None:
        raise ValueError(f"unrecognized charge for {source_id}")
    return charge


def audit_source_disjoint(source_ids: list[str]) -> dict[str, object]:
    seen = [str(item) for item in source_ids]
    if seen != list(AUTHORIZED_NEW_TRAIN_SOURCES):
        raise ValueError("new-train source list drifted from the workbook-63 authorization")
    assert_sources_allowed(seen)
    overlap = sorted(set(seen) & FORBIDDEN_OVERLAP)
    sealed = sorted(item for item in seen if is_sealed_source(item))
    if overlap or sealed:
        raise ValueError("new train sources are not disjoint: " + ", ".join(overlap + sealed))
    return {
        "source_ids": seen,
        "overlap_with_burned_or_reserved": overlap,
        "sealed": sealed,
        "ok": True,
    }


def audit_xaod_root(source_id: str, xaod: Path) -> dict[str, object]:
    if not xaod.is_file():
        raise FileNotFoundError(xaod)
    with uproot.open(xaod) as root_file:
        if "CollectionTree" not in root_file:
            raise ValueError(f"{source_id} lacks CollectionTree")
        tree = root_file["CollectionTree"]
        available = {str(name).split(";")[0] for name in tree.keys()}
        containers = {
            name: {
                "branch": branch,
                "present": branch in available,
                "typename": None if branch not in available else str(tree[branch].typename),
            }
            for name, branch in PERSISTENCE_CONTAINERS.items()
        }
        missing = [name for name in REQUIRED_CONTAINERS if not containers[name]["present"]]
        entries = int(tree.num_entries)
    if missing:
        raise ValueError(f"{source_id} missing persisted containers: {missing}")
    if entries < 1:
        raise ValueError(f"{source_id} CollectionTree is empty")
    return {
        "source_id": source_id,
        "input_xaod": str(xaod),
        "tree_entries_total": entries,
        "persistence_containers": containers,
        "required_containers_present": True,
        "ok": True,
    }


def audit_covariance(events) -> dict[str, object]:
    cov = np.concatenate([event.covariance for event in events], axis=0)
    finite = np.isfinite(cov).all(axis=(1, 2))
    eigenvalues = np.linalg.eigvalsh(cov[finite]) if int(np.count_nonzero(finite)) else np.empty((0, 4))
    positive = int(np.count_nonzero(np.all(eigenvalues > 0.0, axis=1))) if eigenvalues.size else 0
    rows = int(cov.shape[0])
    finite_rows = int(np.count_nonzero(finite))
    pd_fraction = None if finite_rows == 0 else float(positive / finite_rows)
    ok = rows > 0 and finite_rows == rows and pd_fraction is not None and pd_fraction >= MIN_PD_FRACTION
    return {
        "rows": rows,
        "finite_rows": finite_rows,
        "positive_definite_rows": positive,
        "positive_definite_fraction": pd_fraction,
        "minimum_eigenvalue": None if not eigenvalues.size else float(np.min(eigenvalues)),
        "ok": bool(ok),
    }


def audit_four_station(events, content: Mapping[str, Any] | None) -> dict[str, object]:
    station_events = {str(station): 0 for station in (0, 1, 2, 3)}
    for event in events:
        for station in np.unique(event.station_id):
            station_events[str(int(station))] += 1
    content_stations = {}
    if isinstance(content, Mapping):
        raw = content.get("events_with_station") or content.get("station_counts") or {}
        if isinstance(raw, Mapping):
            content_stations = {str(key): int(value) for key, value in raw.items()}
    missing = [station for station, count in station_events.items() if count < MIN_STATION_EVENTS]
    return {
        "station_event_counts": station_events,
        "content_audit_station_counts": content_stations,
        "missing_stations": missing,
        "ok": not missing,
    }


def _identity_paths(source_id: str) -> tuple[Path, Path]:
    point = V3_IDENTITY_POINTS[source_id]
    root = V3_PHYSICAL_ROOT / source_id / "physical_scan" / "points" / point / "refit"
    return root / "tracklets.root", root / "content_audit.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-config",
        default="configs/physical_curriculum_four_station_diversity_new_train_sources.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/mc24_four_station_source_diversity_v1/new_source_audit_v1",
    )
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not (output / "decision.json").is_file():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "decision.json").is_file():
        print(json.dumps({"skip": True, "decision": str(output / "decision.json")}, indent=2))
        return

    config = yaml.safe_load(Path(args.source_config).read_text(encoding="utf-8"))
    block = config.get("physical_curriculum_mlp", config)
    sources = list(block["sources"])
    source_ids = [str(item["id"]) for item in sources]
    disjoint = audit_source_disjoint(source_ids)
    xaod_seen: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for item in sources:
        source_id = str(item["id"])
        xaod = Path(str(item["input_xaod"])).expanduser().resolve()
        if str(xaod) in xaod_seen:
            raise ValueError(f"{source_id} reuses xAOD of {xaod_seen[str(xaod)]}")
        xaod_seen[str(xaod)] = source_id
        root_report = audit_xaod_root(source_id, xaod)
        tracklets, content_path = _identity_paths(source_id)
        if not tracklets.is_file():
            raise FileNotFoundError(tracklets)
        events = load_events(tracklets, require_mc_labels=True)
        identity = summarize_refitted_source(source_id, events, min_truth_match_fraction=0.99)
        content = json.loads(content_path.read_text(encoding="utf-8")) if content_path.is_file() else None
        covariance = audit_covariance(events)
        stations = audit_four_station(events, content)
        expected = _expected_charge(source_id)
        pdg = identity.get("truth_pdg_counts") or {}
        charge_ok = (
            (expected == "mu_minus" and int(pdg.get("13", 0)) > 0 and int(pdg.get("-13", 0)) == 0)
            or (expected == "mu_plus" and int(pdg.get("-13", 0)) > 0 and int(pdg.get("13", 0)) == 0)
        )
        ok = bool(root_report["ok"] and covariance["ok"] and stations["ok"] and charge_ok)
        if not ok:
            failures.append(source_id)
        rows.append(
            {
                "source_id": source_id,
                "split": item.get("split"),
                "expected_charge": expected,
                "charge_ok": charge_ok,
                "input_xaod": str(xaod),
                "v3_identity_tracklets": str(tracklets),
                "v3_identity_is_not_four_station_curriculum": True,
                "root": root_report,
                "identity": identity,
                "content_audit": content,
                "covariance": covariance,
                "four_station": stations,
                "ok": ok,
            }
        )
        _write_json(output / f"{source_id}.json", rows[-1])

    passed = not failures
    decision = {
        "control_id": "retrained_v2_source_disjoint_diversity_v1",
        "workbook": 64,
        "passed": passed,
        "failures": failures,
        "source_disjoint": disjoint,
        "authorize_physical_production": passed,
        "v3_identity_substitutes_for_four_station_curriculum": False,
        "reserved_blind_loaded": False,
        "sealed_test_opened": False,
        "test_data_accessed": False,
    }
    _write_json(output / "sources.json", {"sources": rows})
    _write_json(output / "decision.json", decision)
    print(json.dumps(_json_ready(decision), indent=2))
    if not passed:
        raise SystemExit("new-source pre-production audit failed")


if __name__ == "__main__":
    main()
