#!/usr/bin/env python3
"""Produce the calypso_physical_independent_v1 corpus.  No alignment qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import uproot


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alignment.calypso_physical_replica_production import (  # noqa: E402
    ACTS_TOOL,
    ASSOCIATION_DEFAULT_SYSTEM,
    AUTHORIZED_SOURCES,
    EXPECTED_YIELD,
    GEOMETRY_NAME,
    GLOBAL_TAG,
    MIN_TRUTH_MATCH_FRACTION,
    N_GEOMETRY_CONDITIONS,
    OUTPUT_ROOT,
    PRODUCTION_CHUNK_EVENTS,
    PROVENANCE,
    Q_OVER_P_MODE,
    REFIT_TAG,
    REQUIRED_INPUT_INDEPENDENT_EVENTS,
    SKIP_BASE,
    TARGET_REPLICAS_PER_CONDITION,
    PhysicalReplicaProductionError,
    allocation_plan,
    audit_covariance,
    covariance_matrix,
    empty_reject_counts,
    frozen_conditions,
    independence_audit,
    locked_qualification_flags,
    qa_gate,
    refuse_forbidden_path,
    replica_uid,
    sha256_file,
    sha256_payload,
    write_json,
)
from alignment.wb83_qualification import refuse_wb83_path  # noqa: E402


SETUP = PROJECT_ROOT / "scripts" / "setup_environment.sh"
PAYLOAD_WRITER = PROJECT_ROOT / "scripts" / "write_station_alignment_payload.py"
CALYPSO_ROOT = PROJECT_ROOT.parent / "calypso"
FIELD_ROOT = Path("/cvmfs/faser.cern.ch/repo/sw/software/22.0/faser/offline/ReleaseData")
MATERIAL_ROOT = Path("/cvmfs/faser.cern.ch/repo/sw/database/DBRelease/current")


def _has_ntuple_tree(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with uproot.open(path) as handle:
            return "nt" in handle
    except Exception:
        return False


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _output(root: Path) -> Path:
    refuse_forbidden_path(root)
    refuse_wb83_path(root)
    if "wb83" in str(root).lower():
        raise PhysicalReplicaProductionError("refusing to write into a WB83 path")
    return root


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _hash_paths(paths: list[Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in paths:
        if path.is_file():
            refuse_forbidden_path(path)
            hashes[str(path)] = sha256_file(path)
    return hashes


def _calypso_bundle() -> dict[str, Any]:
    head = _git(CALYPSO_ROOT, "rev-parse", "HEAD")
    describe = _git(CALYPSO_ROOT, "describe", "--always", "--dirty")
    porcelain = _git(CALYPSO_ROOT, "status", "--porcelain")
    dirty_files: list[str] = []
    hashes: dict[str, str] = {}
    for line in porcelain.splitlines():
        raw = line[3:].strip()
        if not raw or raw.endswith("/"):
            continue
        rel = raw.split(" -> ")[-1]
        if rel.startswith("build/") or rel.startswith("run/"):
            continue
        dirty_files.append(rel)
        path = CALYPSO_ROOT / rel
        if path.is_file():
            hashes[rel] = sha256_file(path)
    diff = _git(CALYPSO_ROOT, "diff", "--", *[name for name in dirty_files if (CALYPSO_ROOT / name).is_file()])
    return {
        "head": head,
        "describe": describe,
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files,
        "dirty_file_sha256": hashes,
        "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff else None,
        "cannot_use_dirty_head_as_reproducibility_identity": True,
    }


def cmd_snapshot(output: Path) -> dict[str, Any]:
    output = _output(output)
    calypso = _calypso_bundle()
    ml_head = _git(PROJECT_ROOT, "rev-parse", "HEAD")
    ml_dirty = bool(_git(PROJECT_ROOT, "status", "--porcelain"))
    production_files = [
        PROJECT_ROOT / "alignment" / "calypso_physical_replica_production.py",
        PROJECT_ROOT / "scripts" / "run_calypso_physical_production.py",
        PROJECT_ROOT / "scripts" / "write_station_alignment_payload.py",
        PROJECT_ROOT / "scripts" / "convert_ntuple_tracklets.py",
        PROJECT_ROOT / "scripts" / "convert_ntuple_tracklet_propagations.py",
        PROJECT_ROOT / "scripts" / "setup_environment.sh",
    ]
    field_candidates = []
    if FIELD_ROOT.is_dir():
        field_candidates = sorted(path for path in FIELD_ROOT.rglob("*Mag*") if path.is_file())[:8]
        field_candidates += sorted(path for path in FIELD_ROOT.rglob("*Field*") if path.is_file())[:8]
    material_candidates = []
    if MATERIAL_ROOT.exists():
        if MATERIAL_ROOT.is_file() or MATERIAL_ROOT.is_symlink():
            material_candidates = [MATERIAL_ROOT]
        elif MATERIAL_ROOT.is_dir():
            material_candidates = [MATERIAL_ROOT / name for name in ("version.txt", "current", "setup.py") if (MATERIAL_ROOT / name).exists()]
    snapshot = {
        "schema": "calypso_physical_independent_v1_production_snapshot",
        "utc": _utc(),
        "provenance": PROVENANCE,
        "refit_tag": REFIT_TAG,
        "alignment_ml": {
            "git_sha": ml_head,
            "dirty": ml_dirty,
            "production_file_sha256": {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in production_files if path.is_file()},
        },
        "calypso": calypso,
        "athena_version": "24.0.41",
        "lcg_view": "LCG_104d_ATLAS_7",
        "platform": "x86_64-el9-gcc13-opt",
        "compiler_runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
        },
        "tracker_align_configuration": {
            "folder": "/Tracker/Align",
            "geometry": GEOMETRY_NAME,
            "global_tag": GLOBAL_TAG,
            "writer": str(PAYLOAD_WRITER),
        },
        "acts": {
            "tool": ACTS_TOOL,
            "q_over_p_mode": Q_OVER_P_MODE,
            "physical_geometry_repropagation": True,
        },
        "truth_matching_contract": {
            "source": "mc_truth_match",
            "min_truth_match_fraction": MIN_TRUTH_MATCH_FRACTION,
            "measurement_construction_requires_truth": False,
        },
        "covariance_export_contract": {
            "matrix": "4x4_x_y_tx_ty",
            "silent_repair_forbidden": True,
            "admission_requires_spd": True,
        },
        "field_map": {
            "path": str(FIELD_ROOT),
            "exists": FIELD_ROOT.exists(),
            "hashed_files": _hash_paths(field_candidates),
        },
        "material_map": {
            "path": str(MATERIAL_ROOT),
            "exists": MATERIAL_ROOT.exists(),
            "hashed_files": _hash_paths(material_candidates),
        },
        "container_lcg_setup": {
            "setup_script": str(SETUP),
            "setup_sha256": sha256_file(SETUP),
        },
        "frozen_size": {
            "n_geometry_conditions": N_GEOMETRY_CONDITIONS,
            "target_replicas_per_condition": TARGET_REPLICAS_PER_CONDITION,
            "expected_yield": EXPECTED_YIELD,
            "required_input_independent_events": REQUIRED_INPUT_INDEPENDENT_EVENTS,
            "trained_on_wb83_failure_numbers": False,
        },
        **locked_qualification_flags(),
    }
    write_json(output / "production_snapshot.json", snapshot)
    write_json(output / "calypso_source_bundle.json", calypso)
    return snapshot


def cmd_allowlist(output: Path) -> dict[str, Any]:
    output = _output(output)
    sources = []
    for source in AUTHORIZED_SOURCES:
        path = Path(source["path"])
        refuse_forbidden_path(path)
        if not path.is_file():
            raise PhysicalReplicaProductionError(f"authorized xAOD missing: {path}")
        stat = path.stat()
        with uproot.open(path) as handle:
            tree = handle["POOLCollectionTree"]
            n_events = int(tree.num_entries)
            guid = None
            if "##Params" in handle:
                try:
                    guid = str(handle["##Params"])
                except Exception:
                    guid = None
            per = int(source["events_per_condition"]) * N_GEOMETRY_CONDITIONS
            block = tree.arrays(
                ["RunNumber", "EventNumber"],
                entry_start=SKIP_BASE,
                entry_stop=SKIP_BASE + per,
                library="np",
            )
        runs = [int(value) for value in block["RunNumber"]]
        events = [int(value) for value in block["EventNumber"]]
        uids = [replica_uid(source["source_id"], run, event) for run, event in zip(runs, events)]
        if len(set(uids)) != len(uids):
            raise PhysicalReplicaProductionError(
                f"allocated slice in {source['source_id']} is not unique under source:run:event"
            )
        if n_events < SKIP_BASE + per:
            raise PhysicalReplicaProductionError(f"{source['source_id']} has only {n_events} events")
        sources.append(
            {
                "source_id": source["source_id"],
                "run_range": source["run_range"],
                "channel": source["channel"],
                "path": str(path),
                "file_guid_or_params": guid,
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "file_n_events": n_events,
                "allocated_event_range": [SKIP_BASE, SKIP_BASE + per - 1],
                "n_unique_events_allocated": len(set(uids)),
                "run_ids_in_allocated_slice": sorted(set(runs)),
                "event_id_min": min(events),
                "event_id_max": max(events),
                "w64_already_saw_this_source": True,
            }
        )
    payload = {
        "schema": "calypso_physical_independent_v1_source_allowlist",
        "utc": _utc(),
        "forbidden_assets_scanned_for_content": False,
        "forbidden_assets_opened": False,
        "sources": sources,
        "n_unique_events_allocated_total": sum(item["n_unique_events_allocated"] for item in sources),
        "source_unseen_ml_evaluation": False,
        "truth_only_alignment_qualification_input": True,
        **locked_qualification_flags(),
    }
    if payload["n_unique_events_allocated_total"] < REQUIRED_INPUT_INDEPENDENT_EVENTS:
        payload["production_complete"] = False
        payload["gap"] = REQUIRED_INPUT_INDEPENDENT_EVENTS - payload["n_unique_events_allocated_total"]
    write_json(output / "source_allowlist.json", payload)
    return payload


def cmd_geometry(output: Path, *, write_payloads: bool) -> dict[str, Any]:
    output = _output(output)
    payload_root = output / "geometry_payloads"
    conditions = []
    for cell in frozen_conditions():
        family_dir = payload_root / cell["family"]
        spec_path = family_dir / "requested_transforms.json"
        write_json(
            spec_path,
            {
                "family": cell["family"],
                "station_transforms": cell["station_transforms"],
                "requested_payload_sha256": cell["requested_payload_sha256"],
            },
        )
        written = None
        if write_payloads:
            written = _write_family_payload(family_dir, cell)
        conditions.append(
            {
                **cell,
                "requested_spec_path": str(spec_path),
                "payload_dir": str(family_dir),
                "calypso_payload": written,
            }
        )
    manifest = {
        "schema": "calypso_physical_independent_v1_geometry_payload_manifest",
        "utc": _utc(),
        "not_wb83_matrix": True,
        "wb83_matrix_sha256_must_not_be_used_as_gate": "ca895e44082d4021fe6a3f9d008cf594e03c6330044466a883ca6355e8ac2720",
        "payloads_written": bool(write_payloads),
        "conditions": conditions,
        **locked_qualification_flags(),
    }
    write_json(output / "geometry_payload_manifest.json", manifest)
    return manifest


def _write_family_payload(family_dir: Path, cell: Mapping[str, Any]) -> dict[str, Any]:
    payload_dir = family_dir / "calypso_payload"
    if (payload_dir / "alignment_payload.json").is_file():
        manifest = json.loads((payload_dir / "alignment_payload.json").read_text(encoding="utf-8"))
    else:
        payload_dir.mkdir(parents=True, exist_ok=True)
        arguments = " ".join(
            "--transform "
            + str(station)
            + ":"
            + ":".join(f"{float(component):.17g}" for component in values)
            for station, values in sorted(cell["station_transforms"].items(), key=lambda item: int(item[0]))
        )
        command = (
            "unset PYTHONPATH LD_LIBRARY_PATH ROOTSYS ROOT_INCLUDE_PATH PYTHONHOME"
            f" Athena_SET_UP AthenaExternals_SET_UP\nsource {shlex.quote(str(SETUP))} calypso\n"
            f"python {shlex.quote(str(PAYLOAD_WRITER))} --output-dir {shlex.quote(str(payload_dir))} {arguments}"
        )
        log_path = family_dir / "payload.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(["bash", "-lc", command], cwd=family_dir, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0 or not (payload_dir / "alignment_payload.json").is_file():
            raise PhysicalReplicaProductionError(f"payload write failed for {cell['family']}: see {log_path}")
        manifest = json.loads((payload_dir / "alignment_payload.json").read_text(encoding="utf-8"))
    requested = {int(station): [float(v) for v in values] for station, values in cell["station_transforms"].items()}
    written = {
        int(key.split(":")[1]): [float(v) for v in values]
        for key, values in manifest["alignment_constants"].items()
        if str(key).startswith("station:")
    }
    if requested != written:
        raise PhysicalReplicaProductionError(f"written payload != requested for {cell['family']}")
    return {
        "sqlite": manifest["sqlite"],
        "pool": manifest["pool"],
        "pool_catalog": manifest["pool_catalog"],
        "sqlite_sha256": sha256_file(Path(manifest["sqlite"])),
        "pool_sha256": sha256_file(Path(manifest["pool"])),
        "requested_equals_written": True,
    }


def cmd_allocate(output: Path) -> dict[str, Any]:
    output = _output(output)
    allowlist = json.loads((output / "source_allowlist.json").read_text(encoding="utf-8"))
    by_source = {row["source_id"]: row for row in allowlist["sources"]}
    plan = allocation_plan()
    materialized = []
    for event in plan["events"]:
        source = by_source[event["source_id"]]
        run_id = int(source["run_ids_in_allocated_slice"][0])
        event_id = int(event["expected_event_id"])
        materialized.append(
            {
                **event,
                "run_id": run_id,
                "event_id": event_id,
                "event_uid": replica_uid(event["source_id"], run_id, event_id),
                "input_xaod": next(item["path"] for item in AUTHORIZED_SOURCES if item["source_id"] == event["source_id"]),
            }
        )
    audit = independence_audit(materialized)
    if audit["independence_violation"] or audit["n_unique"] != REQUIRED_INPUT_INDEPENDENT_EVENTS:
        raise PhysicalReplicaProductionError(f"allocation independence failed: {audit}")
    payload = {
        **plan,
        "utc": _utc(),
        "events": materialized,
        "independence_audit_at_allocation": {"n_unique": audit["n_unique"], "n_violations": audit["n_violations"]},
        **locked_qualification_flags(),
    }
    write_json(output / "event_allocation_manifest.json", payload)
    return payload


def cmd_roundtrip(output: Path, *, nevents: int) -> dict[str, Any]:
    output = _output(output)
    manifest = json.loads((output / "geometry_payload_manifest.json").read_text(encoding="utf-8"))
    identity = next(cell for cell in manifest["conditions"] if cell["family"] == "identity")
    translation = next(cell for cell in manifest["conditions"] if cell["family"] == "identifiable_translation")
    if not identity.get("calypso_payload") or not translation.get("calypso_payload"):
        raise PhysicalReplicaProductionError("geometry payloads are not written; cannot roundtrip")
    source = AUTHORIZED_SOURCES[4]  # 100047, smaller file
    refuse_forbidden_path(source["path"])
    reports = {}
    for label, cell in (("identity", identity), ("translation", translation)):
        work = output / "geometry_roundtrip" / label
        work.mkdir(parents=True, exist_ok=True)
        enhanced = work / "enhanced_tracklets.root"
        payload = cell["calypso_payload"]
        if not _has_ntuple_tree(enhanced):
            if enhanced.exists():
                enhanced.unlink()
            command = (
                "unset PYTHONPATH LD_LIBRARY_PATH ROOTSYS ROOT_INCLUDE_PATH PYTHONHOME"
                f" Athena_SET_UP AthenaExternals_SET_UP\nsource {shlex.quote(str(SETUP))} calypso\n"
                "faser_ntuple_maker.py "
                f"{shlex.quote(source['path'])} --isMC --useIFT --nevents {int(nevents)} --skip-events 0 "
                "--export-tracklets --export-tracklet-propagation --refit-segments "
                f"--tracker-align-sqlite {shlex.quote(payload['sqlite'])} "
                f"--tracker-align-pool-catalog {shlex.quote(payload['pool_catalog'])} "
                f"--outfile {shlex.quote(str(enhanced))}"
            )
            log_path = work / "roundtrip.log"
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(["bash", "-lc", command], cwd=work, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode != 0:
                raise PhysicalReplicaProductionError(f"roundtrip refit failed for {label}: {log_path}")
        else:
            log_path = work / "roundtrip.log"
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        markers = {
            "sqlite_override": "Reading folder /Tracker/Align from sqlite" in log_text,
            "sct_alignment_store": "recorded new CDO SCTAlignmentStore" in log_text or "SCTAlignmentStore" in log_text,
            "acts_alignment_context": "Recorded new FaserActsAlignment" in log_text or "FaserActsAlignment" in log_text,
            "job_success": "Execution succeeded" in log_text,
        }
        reports[label] = {
            "log": str(log_path),
            "markers": markers,
            "requested_equals_written": bool(cell["calypso_payload"]["requested_equals_written"]),
            "sqlite_sha256": cell["calypso_payload"]["sqlite_sha256"],
        }
    passed = all(
        report["requested_equals_written"] and report["markers"]["job_success"] and report["markers"]["sqlite_override"]
        for report in reports.values()
    )
    payload = {
        "schema": "calypso_physical_independent_v1_geometry_roundtrip",
        "utc": _utc(),
        "input_transform_equals_written_payload": all(report["requested_equals_written"] for report in reports.values()),
        "framework_loaded_tracker_align_sqlite": all(report["markers"]["sqlite_override"] for report in reports.values()),
        "acts_saw_alignment_context": all(report["markers"]["acts_alignment_context"] for report in reports.values()),
        "reports": reports,
        "geometry_roundtrip": "PASS" if passed else "FAIL",
        **locked_qualification_flags(),
    }
    write_json(output / "geometry_roundtrip.json", payload)
    return payload


def _load_allocation(output: Path) -> dict[str, Any]:
    return json.loads((output / "event_allocation_manifest.json").read_text(encoding="utf-8"))


def cmd_worker(output: Path, shard_id: str) -> dict[str, Any]:
    output = _output(output)
    allocation = _load_allocation(output)
    shard = next((row for row in allocation["shards"] if row["shard_id"] == shard_id), None)
    if shard is None:
        raise PhysicalReplicaProductionError(f"unknown shard {shard_id}")
    refuse_forbidden_path(shard["input_xaod"])
    geometry = json.loads((output / "geometry_payload_manifest.json").read_text(encoding="utf-8"))
    cell = next(row for row in geometry["conditions"] if row["condition_id"] == shard["condition_id"])
    if not cell.get("calypso_payload"):
        raise PhysicalReplicaProductionError("payload missing; worker refuses to invent geometry")
    work = output / "shards" / shard_id
    work.mkdir(parents=True, exist_ok=True)
    enhanced = work / "enhanced_tracklets.root"
    tracklets = work / "tracklets.root"
    propagations = work / "propagations.root"
    if not _has_ntuple_tree(enhanced):
        if enhanced.exists():
            enhanced.unlink()
        payload = cell["calypso_payload"]
        command = (
            "unset PYTHONPATH LD_LIBRARY_PATH ROOTSYS ROOT_INCLUDE_PATH PYTHONHOME"
            f" Athena_SET_UP AthenaExternals_SET_UP\nsource {shlex.quote(str(SETUP))} calypso\n"
            "faser_ntuple_maker.py "
            f"{shlex.quote(shard['input_xaod'])} --isMC --useIFT "
            f"--nevents {int(shard['nevents'])} --skip-events {int(shard['skip_events'])} "
            "--export-tracklets --export-tracklet-propagation --refit-segments "
            f"--tracker-align-sqlite {shlex.quote(payload['sqlite'])} "
            f"--tracker-align-pool-catalog {shlex.quote(payload['pool_catalog'])} "
            f"--outfile {shlex.quote(str(enhanced))}"
        )
        log_path = work / "refit.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(["bash", "-lc", command], cwd=work, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            _write_shard_failure(work, shard, "reconstruction_failure", f"refit rc={result.returncode}")
            return {"shard_id": shard_id, "status": "failed", "reason": "reconstruction_failure"}
    export_cmd = (
        f"source {shlex.quote(str(SETUP))} ml\ncd {shlex.quote(str(PROJECT_ROOT))}\n"
        f"python scripts/convert_ntuple_tracklets.py {shlex.quote(str(enhanced))} "
        f"--output {shlex.quote(str(tracklets))} --include-truth\n"
        f"python scripts/convert_ntuple_tracklet_propagations.py {shlex.quote(str(enhanced))} "
        f"--output {shlex.quote(str(propagations))}"
    )
    if not (tracklets.is_file() and propagations.is_file()):
        log_path = work / "export.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(["bash", "-lc", export_cmd], cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            _write_shard_failure(work, shard, "other", f"export rc={result.returncode}")
            return {"shard_id": shard_id, "status": "failed", "reason": "other"}
    return _export_shard_events(output, work, shard, cell, tracklets, propagations)


def _write_shard_failure(work: Path, shard: Mapping[str, Any], reason: str, message: str) -> None:
    write_json(
        work / "shard_failure.json",
        {"shard_id": shard["shard_id"], "reject_reason": reason, "message": message, "utc": _utc()},
    )


def _export_shard_events(
    output: Path,
    work: Path,
    shard: Mapping[str, Any],
    cell: Mapping[str, Any],
    tracklets_path: Path,
    propagations_path: Path,
) -> dict[str, Any]:
    snapshot = json.loads((output / "production_snapshot.json").read_text(encoding="utf-8"))
    allocated = {
        int(row["event_id"]): row
        for row in json.loads((output / "event_allocation_manifest.json").read_text(encoding="utf-8"))["events"]
        if row["shard_id"] == shard["shard_id"]
    }
    with uproot.open(tracklets_path) as handle:
        tree = handle["tracklets"]
        arrays = tree.arrays(library="np")
    n = len(arrays["event_id"])
    rejects = empty_reject_counts()
    records = []
    cov_failures = 0
    by_event: dict[int, list[int]] = {}
    for index in range(n):
        event_id = int(arrays["event_id"][index])
        by_event.setdefault(event_id, []).append(index)
    for event_id, alloc in allocated.items():
        rows = by_event.get(event_id, [])
        if not rows:
            rejects["reconstruction_failure"] += 1
            continue
        run_id = int(arrays["run_id"][rows[0]])
        uid = replica_uid(shard["source_id"], run_id, event_id)
        hits = []
        cov_ok = True
        for index in rows:
            row = {name: arrays[name][index] for name in arrays}
            matrix = covariance_matrix(row)
            cov = audit_covariance(matrix)
            if not cov["accepted"]:
                cov_ok = False
                cov_failures += 1
                break
            truth = {
                "particle_id": int(row["truth_particle_id"]),
                "pdg": int(row["truth_pdg"]),
                "match_fraction": float(row["truth_match_fraction"]),
            }
            hits.append(
                {
                    "station_id": int(row["station_id"]),
                    "measurement_id": f"{uid}:{int(row['station_id'])}:{int(row['tracklet_id'])}",
                    "tracklet_id": int(row["tracklet_id"]),
                    "x_mm": float(row["x_mm"]),
                    "y_mm": float(row["y_mm"]),
                    "tx": float(row["tx"]),
                    "ty": float(row["ty"]),
                    "z_mm": float(row["z_mm"]),
                    "q_over_p_per_mev": float(row["q_over_p_per_mev"]) if "q_over_p_per_mev" in arrays else None,
                    "covariance_4x4": matrix,
                    "covariance_contract": cov,
                    "surface": {"z_mm": float(row["z_mm"]), "frame": "aligned_station_local"},
                    "truth_label": truth,
                }
            )
        if not cov_ok:
            rejects["invalid_covariance"] += 1
            continue
        tracks = _group_truth_tracks(uid, hits)
        reason = _admission_reason(tracks, hits)
        if reason is not None:
            rejects[reason] += 1
            continue
        records.append(
            {
                "schema": "calypso_physical_independent_v1_event",
                "event_uid": uid,
                "source_id": shard["source_id"],
                "run_id": run_id,
                "event_id": event_id,
                "xaod_entry_index": alloc["xaod_entry_index"],
                "condition_id": shard["condition_id"],
                "geometry_payload_hash": cell["calypso_payload"]["sqlite_sha256"],
                "requested_payload_sha256": cell["requested_payload_sha256"],
                "field_material_provenance": {
                    "field_map": snapshot["field_map"]["path"],
                    "material_map": snapshot["material_map"]["path"],
                },
                "software_hashes": {
                    "calypso_bundle": snapshot["calypso"].get("diff_sha256"),
                    "alignment_ml_head": snapshot["alignment_ml"]["git_sha"],
                },
                "tracks": tracks,
                "n_hits": len(hits),
            }
        )
    export_path = work / "replicas.jsonl"
    with export_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    summary = {
        "shard_id": shard["shard_id"],
        "condition_id": shard["condition_id"],
        "source_id": shard["source_id"],
        "requested_events": int(shard["nevents"]),
        "processed_events": len(by_event),
        "successful_reconstructions": len(by_event),
        "qualified_exports": len(records),
        "rejected_events": int(shard["nevents"]) - len(records),
        "yield": len(records) / float(shard["nevents"]),
        "reject_reasons": rejects,
        "covariance_failures": cov_failures,
        "export_sha256": sha256_file(export_path) if export_path.is_file() else None,
        "cluster_proc": os.environ.get("CLUSTER_ID", os.environ.get("ClusterId")),
        "host": socket.gethostname(),
        "config_hash": sha256_file(PROJECT_ROOT / "alignment" / "calypso_physical_replica_production.py"),
        "payload_hash": cell["calypso_payload"]["sqlite_sha256"],
        "utc": _utc(),
        **locked_qualification_flags(),
    }
    write_json(work / "shard_summary.json", summary)
    return summary


def _group_truth_tracks(event_uid: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    unlabeled: list[dict[str, Any]] = []
    for hit in hits:
        truth = hit["truth_label"]
        measurement = {key: value for key, value in hit.items() if key != "truth_label"}
        if truth is None or int(truth["particle_id"]) < 0:
            unlabeled.append({"measurement": measurement, "truth": None})
            continue
        grouped.setdefault(int(truth["particle_id"]), []).append({"measurement": measurement, "truth": truth})
    tracks = []
    for particle_id, members in grouped.items():
        tracks.append(
            {
                "track_uid": f"{event_uid}:truth:{particle_id}",
                "truth": members[0]["truth"],
                "hits": [member["measurement"] for member in members],
            }
        )
    if unlabeled:
        tracks.append({"track_uid": f"{event_uid}:unlabeled", "truth": None, "hits": [member["measurement"] for member in unlabeled]})
    return tracks


def _admission_reason(tracks: list[dict[str, Any]], hits: list[dict[str, Any]]) -> str | None:
    complete = []
    for track in tracks:
        truth = track.get("truth")
        if not truth or float(truth["match_fraction"]) < MIN_TRUTH_MATCH_FRACTION:
            continue
        stations = {int(hit["station_id"]) for hit in track["hits"]}
        if stations >= {0, 1, 2, 3}:
            complete.append(track)
    if complete:
        return None
    if not any(track.get("truth") for track in tracks):
        return "missing_truth"
    return "insufficient_stations"


def cmd_qa(output: Path) -> dict[str, Any]:
    output = _output(output)
    allocation = _load_allocation(output)
    snapshot = json.loads((output / "production_snapshot.json").read_text(encoding="utf-8"))
    allowlist = json.loads((output / "source_allowlist.json").read_text(encoding="utf-8"))
    geometry = json.loads((output / "geometry_payload_manifest.json").read_text(encoding="utf-8"))
    roundtrip_path = output / "geometry_roundtrip.json"
    roundtrip = json.loads(roundtrip_path.read_text(encoding="utf-8")) if roundtrip_path.is_file() else {}
    summaries = []
    replicas = []
    failures = []
    for shard in allocation["shards"]:
        work = output / "shards" / shard["shard_id"]
        summary_path = work / "shard_summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summaries.append(summary)
            export_path = work / "replicas.jsonl"
            if export_path.is_file():
                for line in export_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        replicas.append(json.loads(line))
        else:
            failure_path = work / "shard_failure.json"
            if failure_path.is_file():
                failures.append(json.loads(failure_path.read_text(encoding="utf-8")))
            else:
                failures.append({"shard_id": shard["shard_id"], "reject_reason": "other", "message": "shard not produced"})
    audit = independence_audit(replicas) if replicas else {"n_unique": 0, "n_violations": 0, "independence_violation": False, "violations": []}
    by_condition: dict[str, dict[str, Any]] = {}
    for cell in frozen_conditions():
        cid = cell["condition_id"]
        rows = [row for row in replicas if row["condition_id"] == cid]
        shard_rows = [row for row in summaries if row["condition_id"] == cid]
        requested = 400
        qualified = len(rows)
        rejected = sum(int(row.get("rejected_events", 0)) for row in shard_rows)
        reasons = empty_reject_counts()
        for row in shard_rows:
            for key, value in row.get("reject_reasons", {}).items():
                reasons[key] = reasons.get(key, 0) + int(value)
        by_condition[cid] = {
            "requested_events": requested,
            "processed_events": sum(int(row.get("processed_events", 0)) for row in shard_rows),
            "successful_reconstructions": sum(int(row.get("successful_reconstructions", 0)) for row in shard_rows),
            "qualified_exports": qualified,
            "rejected_events": rejected,
            "yield": qualified / float(requested),
            "reject_reasons": reasons,
            "target_replicas": TARGET_REPLICAS_PER_CONDITION,
            "meets_target": qualified >= TARGET_REPLICAS_PER_CONDITION,
        }
    condition_counts_ok = all(block["meets_target"] for block in by_condition.values())
    unique_ok = int(audit["n_unique"]) >= REQUIRED_INPUT_INDEPENDENT_EVENTS and not audit["independence_violation"]
    # Unique qualified events can be < 3200; the gate requires unique physical events
    # among allocated inputs that were exported without reuse.  Corpus qualification
    # requires >= 200 qualified exports per condition, not 3200 qualified events.
    unique_qualified_ok = audit["n_unique"] >= TARGET_REPLICAS_PER_CONDITION * N_GEOMETRY_CONDITIONS and not audit["independence_violation"]
    production_complete = condition_counts_ok and unique_qualified_ok and not failures
    cov_ok = all(
        all(hit["covariance_contract"]["accepted"] for track in record["tracks"] for hit in track["hits"])
        for record in replicas
    ) if replicas else False
    schema_ok = all("event_uid" in record and "tracks" in record for record in replicas) if replicas else False
    truth_ok = all(
        any(track.get("truth") for track in record["tracks"])
        for record in replicas
    ) if replicas else False
    checks = {
        "unique_physical_events": unique_qualified_ok,
        "zero_independence_violations": not bool(audit.get("independence_violation")),
        "zero_forbidden_assets": allowlist.get("forbidden_assets_opened") is False,
        "geometry_roundtrip": roundtrip.get("geometry_roundtrip") == "PASS",
        "software_provenance_complete": bool(snapshot.get("calypso", {}).get("dirty_file_sha256")),
        "field_material_hashes_consistent": bool(snapshot.get("field_map", {}).get("path")),
        "truth_label_completeness": truth_ok,
        "measurement_schema": schema_ok,
        "covariance_contract": cov_ok,
        "condition_counts": condition_counts_ok,
        "production_complete": production_complete,
    }
    qa = qa_gate(checks)
    qa.update(
        {
            "utc": _utc(),
            "n_shards_done": len(summaries),
            "n_shards_expected": len(allocation["shards"]),
            "n_qualified_exports": len(replicas),
            "conditions": by_condition,
            "gap": {
                cid: max(0, TARGET_REPLICAS_PER_CONDITION - block["qualified_exports"])
                for cid, block in by_condition.items()
            },
        }
    )
    if not production_complete:
        qa["physical_replica_production_complete"] = False
        qa["qualification_authorized"] = False
    write_json(output / "corpus_qa.json", qa)
    write_json(output / "independence_audit.json", audit)
    write_json(
        output / "covariance_audit.json",
        {"n_events": len(replicas), "contract_pass": cov_ok, "silent_repair": False},
    )
    write_json(output / "production_failures.json", {"failures": failures, "n": len(failures)})
    write_json(
        output / "physical_replica_manifest.json",
        {
            "provenance": PROVENANCE,
            "n_replicas": len(replicas),
            "event_uids": [row["event_uid"] for row in replicas],
            **locked_qualification_flags(),
        },
    )
    write_json(
        output / "provenance_freeze.json",
        {
            "snapshot_sha256": sha256_payload(snapshot),
            "allowlist_sha256": sha256_payload(allowlist),
            "geometry_sha256": sha256_payload(geometry),
            "qa_sha256": sha256_payload(qa),
            "calypso_physical_corpus_qualified": qa["calypso_physical_corpus_qualified"],
            "physical_replica_production_complete": qa["physical_replica_production_complete"],
            **locked_qualification_flags(),
        },
    )
    return qa


def cmd_submit(output: Path, *, submit: bool, job_flavour: str, memory_mb: int) -> dict[str, Any]:
    output = _output(output)
    for required in (
        "production_snapshot.json",
        "source_allowlist.json",
        "geometry_payload_manifest.json",
        "event_allocation_manifest.json",
    ):
        if not (output / required).is_file():
            raise PhysicalReplicaProductionError(f"missing {required}")
    geometry = json.loads((output / "geometry_payload_manifest.json").read_text(encoding="utf-8"))
    if not all(cell.get("calypso_payload") for cell in geometry["conditions"]):
        raise PhysicalReplicaProductionError("geometry payloads must be written before submit")
    allocation = _load_allocation(output)
    condor = output / "condor"
    logs = condor / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    worker = PROJECT_ROOT / "scripts" / "run_calypso_physical_replica_condor.sh"
    qa_worker = PROJECT_ROOT / "scripts" / "run_calypso_physical_qa_condor.sh"
    dag_lines = []
    names = []
    for shard in allocation["shards"]:
        name = shard["shard_id"].replace(".", "p")
        submit_path = condor / f"{name}.sub"
        submit_path.write_text(
            "\n".join(
                (
                    "universe = vanilla",
                    f"executable = {worker}",
                    f"arguments = {PROJECT_ROOT} {output} {shard['shard_id']}",
                    f"output = {logs}/{name}.$(ClusterId).$(ProcId).out",
                    f"error = {logs}/{name}.$(ClusterId).$(ProcId).err",
                    f"log = {logs}/{name}.$(ClusterId).log",
                    f"request_memory = {int(memory_mb)}",
                    "request_cpus = 1",
                    "request_disk = 4000000",
                    f'+JobFlavour = "{job_flavour}"',
                    "getenv = True",
                    "queue",
                    "",
                )
            ),
            encoding="utf-8",
        )
        dag_lines.append(f"JOB {name} {submit_path}")
        names.append(name)
    qa_sub = condor / "qa.sub"
    qa_sub.write_text(
        "\n".join(
            (
                "universe = vanilla",
                f"executable = {qa_worker}",
                f"arguments = {PROJECT_ROOT} {output}",
                f"output = {logs}/qa.$(ClusterId).$(ProcId).out",
                f"error = {logs}/qa.$(ClusterId).$(ProcId).err",
                f"log = {logs}/qa.$(ClusterId).log",
                "request_memory = 4000",
                "request_cpus = 1",
                f'+JobFlavour = "{job_flavour}"',
                "getenv = True",
                "queue",
                "",
            )
        ),
        encoding="utf-8",
    )
    dag_lines.append(f"JOB QA {qa_sub}")
    dag_lines.append("PARENT " + " ".join(names) + " CHILD QA")
    dag_path = condor / "physical_replicas.dag"
    dag_path.write_text("\n".join(dag_lines) + "\n", encoding="utf-8")
    manifest = {
        "schema": "calypso_physical_independent_v1_condor_dag",
        "utc": _utc(),
        "dag": str(dag_path),
        "n_refit_jobs": len(names),
        "job_flavour": job_flavour,
        "request_memory_mb": memory_mb,
        "submitted": False,
        **locked_qualification_flags(),
    }
    if submit:
        result = subprocess.run(
            [
                "bash",
                "-lc",
                "source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out && "
                f"condor_submit_dag {shlex.quote(str(dag_path))}",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        manifest["condor_submit_output"] = result.stdout
        manifest["condor_submit_returncode"] = result.returncode
        manifest["submitted"] = result.returncode == 0
        if result.returncode != 0:
            write_json(output / "condor_dag_manifest.json", manifest)
            raise PhysicalReplicaProductionError("condor_submit_dag failed:\n" + result.stdout)
    write_json(output / "condor_dag_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("snapshot")
    sub.add_parser("allowlist")
    geo = sub.add_parser("geometry")
    geo.add_argument("--write-payloads", action="store_true")
    sub.add_parser("allocate")
    rt = sub.add_parser("roundtrip")
    rt.add_argument("--nevents", type=int, default=1)
    worker = sub.add_parser("worker")
    worker.add_argument("--shard-id", required=True)
    sub.add_parser("qa")
    submit = sub.add_parser("submit")
    submit.add_argument("--submit", action="store_true")
    submit.add_argument("--job-flavour", default="tomorrow")
    submit.add_argument("--request-memory-mb", type=int, default=6000)
    freeze = sub.add_parser("freeze-preproduction")
    freeze.add_argument("--write-payloads", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = Path(args.output_root).expanduser().resolve()
    if args.command == "snapshot":
        print(json.dumps({"wrote": "production_snapshot.json", "calypso_dirty": cmd_snapshot(output)["calypso"]["dirty"]}, indent=2))
    elif args.command == "allowlist":
        payload = cmd_allowlist(output)
        print(json.dumps({"n_unique": payload["n_unique_events_allocated_total"]}, indent=2))
    elif args.command == "geometry":
        payload = cmd_geometry(output, write_payloads=args.write_payloads)
        print(json.dumps({"n_conditions": len(payload["conditions"]), "payloads_written": payload["payloads_written"]}, indent=2))
    elif args.command == "allocate":
        payload = cmd_allocate(output)
        print(json.dumps({"n_events": payload["n_events"], "n_shards": payload["n_shards"]}, indent=2))
    elif args.command == "roundtrip":
        payload = cmd_roundtrip(output, nevents=args.nevents)
        print(json.dumps({"geometry_roundtrip": payload["geometry_roundtrip"]}, indent=2))
    elif args.command == "worker":
        print(json.dumps(cmd_worker(output, args.shard_id), indent=2))
    elif args.command == "qa":
        payload = cmd_qa(output)
        print(json.dumps(
            {
                "physical_replica_production_complete": payload["physical_replica_production_complete"],
                "calypso_physical_corpus_qualified": payload["calypso_physical_corpus_qualified"],
                "alignment_oracle_qualified_for_physical_FASER": payload["alignment_oracle_qualified_for_physical_FASER"],
            },
            indent=2,
        ))
    elif args.command == "submit":
        payload = cmd_submit(output, submit=args.submit, job_flavour=args.job_flavour, memory_mb=args.request_memory_mb)
        print(json.dumps({"submitted": payload["submitted"], "n_refit_jobs": payload["n_refit_jobs"]}, indent=2))
    elif args.command == "freeze-preproduction":
        cmd_snapshot(output)
        cmd_allowlist(output)
        cmd_geometry(output, write_payloads=args.write_payloads)
        cmd_allocate(output)
        cmd_qa(output)
        print(json.dumps({"output": str(output), "preproduction_frozen": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
