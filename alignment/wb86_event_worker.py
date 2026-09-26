"""WB86 per-event official worker.  Writes technical + sealed files only."""

from __future__ import annotations

import json
import os
import platform
import socket
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alignment.calypso_physical_replica_production import sha256_file
from alignment.physical_common_track_execution import CalypsoActsPhysicalBackend, OFFICIAL_ENGINE
from alignment.wb85_physical_qualification_protocol import write_json
from alignment.wb85b_official_runner import official_runner_contract
from alignment.wb86_official_execution import iterate_official_event, payload_digest, true_payload
from alignment.wb86_physical_qualification import (
    EXPECTED_HASHES,
    OUTPUT_ROOT,
    WB86Error,
    load_catalog_event,
    load_replica_record,
    locked_execution_flags,
    refuse_wb84_wb85b_rewrite,
    sealed_event_path,
    technical_event_dir,
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scratch_root(output_root: Path, index: int) -> Path:
    base = Path(os.environ.get("TMPDIR") or "/tmp")
    path = base / f"wb86_event_{int(index):04d}_{os.getpid()}"
    refuse_wb84_wb85b_rewrite(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _technical_envelope(row: dict[str, Any], *, status: str, error_code: str | None) -> dict[str, Any]:
    return {
        **locked_execution_flags(),
        "schema": "wb86_event_technical_v1",
        "utc": _utc(),
        "index": int(row["index"]),
        "event_uid": row["event_uid"],
        "condition_id": row["condition_id"],
        "technical_status": status,
        "error_code": error_code,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cluster_id": os.environ.get("CLUSTER_ID") or os.environ.get("ClusterId"),
        "proc_id": os.environ.get("PROC_ID") or os.environ.get("ProcId"),
        "engine": OFFICIAL_ENGINE,
        "official_runner": official_runner_contract()["official_runner"],
        "common_track_solver_sha256": EXPECTED_HASHES["common_track_solver_sha256"],
        "official_runner_sha256": EXPECTED_HASHES["official_runner_sha256"],
        "no_scientific_aggregates": True,
    }


def execute_one_event(index: int, output_root: Path | None = None) -> dict[str, Any]:
    root = OUTPUT_ROOT if output_root is None else Path(output_root)
    refuse_wb84_wb85b_rewrite(root)
    planner = CalypsoActsPhysicalBackend(execute=False)
    if planner.execute is not False:
        raise WB86Error("CalypsoActsPhysicalBackend.execute must stay False")
    row = load_catalog_event(int(index), root)
    technical_dir = technical_event_dir(root, int(index))
    technical_path = technical_dir / "technical.json"
    sealed_path = sealed_event_path(root, int(index))
    if technical_path.is_file() and sealed_path.is_file():
        existing = json.loads(technical_path.read_text(encoding="utf-8"))
        if existing.get("technical_status") == "SUCCESS" and existing.get("sealed_sha256") == sha256_file(sealed_path):
            return existing
    technical_dir.mkdir(parents=True, exist_ok=True)
    sealed_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        record = load_replica_record(row)
        work = _scratch_root(root, int(index))
        result = iterate_official_event(
            record,
            family=str(row["family"]),
            survey_mode=str(row["survey_mode"]),
            chart_kind=str(row["chart_kind"]),
            work_dir=work,
        )
        projected = result.get("projected")
        analyzable = bool(
            result["converged"]
            and result["solver_ok"]
            and int(result["n_dropped"]) == 0
            and projected is not None
        )
        sealed = {
            "schema": "wb86_event_sealed_v1",
            "index": int(row["index"]),
            "event_uid": row["event_uid"],
            "stratum_id": row["condition_id"],
            "family": row["family"],
            "survey_mode": row["survey_mode"],
            "enters_primary_global_test": bool(row["enters_primary_global_test"]),
            "analyzable": analyzable,
            "converged": result["converged"],
            "reason": result["reason"],
            "iterations": result["iterations"],
            "solver_ok": result["solver_ok"],
            "solver_status": result["solver_status"],
            "n_dropped": result["n_dropped"],
            "parameter_names": result["parameter_names"],
            "theta_hat": result["theta_hat"],
            "theta_true": result["theta_true"],
            "projected": projected,
            "per_parameter_z": result.get("per_parameter_z") or {},
            "fd_rel_max_observed": result["fd_rel_max_observed"],
            "sign_frame_failure": result["sign_frame_failure"],
            "engine": result["engine"],
        }
        write_json(sealed_path, sealed)
        technical = _technical_envelope(row, status="SUCCESS", error_code=None)
        technical.update(
            {
                "runtime_s": time.time() - started,
                "iterations": result["iterations"],
                "converged": result["converged"],
                "solver_status": result["solver_status"],
                "n_official_refits": result["n_official_refits"],
                "payload_hash": payload_digest(true_payload(str(row["family"]))),
                "input_uid": row["event_uid"],
                "scratch_dir": str(work),
                "sealed_sha256": sha256_file(sealed_path),
                "last_refit_sqlite_sha256": (result.get("last_refit") or {}).get("sqlite_sha256"),
                "last_refit_executor": (result.get("last_refit") or {}).get("executor"),
            }
        )
        write_json(technical_path, technical)
        return technical
    except Exception as error:
        sealed = {
            "schema": "wb86_event_sealed_v1",
            "index": int(row["index"]),
            "event_uid": row["event_uid"],
            "stratum_id": row["condition_id"],
            "family": row["family"],
            "survey_mode": row["survey_mode"],
            "enters_primary_global_test": bool(row["enters_primary_global_test"]),
            "analyzable": False,
            "converged": False,
            "reason": type(error).__name__,
            "iterations": 0,
            "solver_ok": False,
            "solver_status": "error",
            "n_dropped": 0,
            "parameter_names": [],
            "theta_hat": [],
            "theta_true": [],
            "projected": None,
            "per_parameter_z": {},
            "fd_rel_max_observed": 0.0,
            "sign_frame_failure": False,
            "error": str(error),
        }
        write_json(sealed_path, sealed)
        technical = _technical_envelope(row, status="FAIL", error_code=type(error).__name__)
        technical.update(
            {
                "runtime_s": time.time() - started,
                "iterations": 0,
                "input_uid": row["event_uid"],
                "sealed_sha256": sha256_file(sealed_path),
                "traceback": traceback.format_exc()[-4000:],
            }
        )
        write_json(technical_path, technical)
        return technical
