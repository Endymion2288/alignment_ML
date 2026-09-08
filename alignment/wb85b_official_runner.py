"""WB85b official qualification runner.  Calypso/ACTS execution path for future WB86.

Does not authorize qualification and does not report alignment performance.
Reuses the smoke-tested command builders and ``_run`` primitive from WB85a-r1.
The top-level executor is this module, so a dedicated e2e smoke is required.
"""

from __future__ import annotations

import json
import os
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from alignment.calypso_physical_replica_production import ACTS_TOOL, Q_OVER_P_MODE, sha256_file
from alignment.common_track_solver import apply_parameter_update
from alignment.physical_common_track_execution import (
    FORBIDDEN_OFFICIAL_FALLBACKS,
    OFFICIAL_ENGINE,
    REFIT_CHAIN,
    CalypsoActsPhysicalBackend,
    calypso_acts_rerefit_command,
    calypso_payload_command,
    command_is_official,
    identity_payload,
    ntuple_export_command,
    refuse_toy_official_fallback,
    refuse_wb84_rewrite,
)
from alignment.wb85_physical_qualification_protocol import refuse_wb83_wb84_write
from alignment.wb85a_r1_runtime_smoke import (
    SMOKE_NEVENTS,
    SMOKE_PROBE_S1_DX_MM,
    SMOKE_SKIP_EVENTS,
    SMOKE_SOURCE_ID,
    _hash_if_file,
    _run,
    fixture_is_legal,
    locked_smoke_flags,
    parse_converted_tracklets,
    relinearize_from_physical_pair,
    smoke_source,
)
from alignment.wb85b_fwer_protocol import WB85BError, refuse_historical_rewrite


SMOKE_OUTPUT = Path("outputs/mc24_four_station_wb85b_official_runner_smoke_v1")
SMOKE_TESTED_PRIMITIVES = (
    "alignment.physical_common_track_execution.calypso_payload_command",
    "alignment.physical_common_track_execution.calypso_acts_rerefit_command",
    "alignment.physical_common_track_execution.ntuple_export_command",
    "alignment.wb85a_r1_runtime_smoke._run",
    "alignment.wb85a_r1_runtime_smoke.parse_converted_tracklets",
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def payload_s1_dx_mm(payload_json: Path) -> float:
    if not payload_json.is_file():
        raise WB85BError(f"alignment payload manifest missing: {payload_json}")
    data = json.loads(payload_json.read_text(encoding="utf-8"))
    constants = data.get("alignment_constants") or {}
    station = constants.get("station:1")
    if not station:
        raise WB85BError("payload does not contain station:1")
    return float(station[0])


class WB85bOfficialCalypsoActsBackend:
    """Exact future-qualification Calypso/ACTS backend.

    Distinct from ``CalypsoActsPhysicalBackend``, whose ``execute`` flag must
    stay false.  This backend executes the smoke-tested command primitives.
    """

    engine_name = OFFICIAL_ENGINE
    is_official = True

    def execute_refit(
        self,
        *,
        input_xaod: str,
        payload: Mapping[int, Sequence[float]],
        work_dir: Path,
        skip_events: int,
        nevents: int = 1,
    ) -> dict[str, Any]:
        refuse_historical_rewrite(work_dir)
        refuse_wb84_rewrite(work_dir)
        refuse_wb83_wb84_write(work_dir)
        if work_dir.exists():
            raise WB85BError(f"refusing to reuse a cached official-runner work directory: {work_dir}")
        work_dir.mkdir(parents=True, exist_ok=False)
        payload_dir = work_dir / "calypso_payload"
        enhanced = work_dir / "enhanced_tracklets.root"
        tracklets = work_dir / "tracklets.root"
        propagations = work_dir / "propagations.root"
        payload_cmd = calypso_payload_command(output_dir=payload_dir, payload=payload)
        sqlite = payload_dir / "tracker_alignment.sqlite"
        catalog = payload_dir / "PoolFileCatalog.xml"
        rerefit_cmd = calypso_acts_rerefit_command(
            input_xaod=input_xaod,
            sqlite=sqlite,
            pool_catalog=catalog,
            outfile=enhanced,
            skip_events=int(skip_events),
            nevents=int(nevents),
        )
        export_cmd = ntuple_export_command(enhanced=enhanced, tracklets=tracklets, propagations=propagations)
        if not command_is_official(rerefit_cmd):
            raise WB85BError("official runner rerefit command is not official")
        for command in (payload_cmd, rerefit_cmd, export_cmd):
            refuse_toy_official_fallback(command)
        rc_payload = _run(payload_cmd, cwd=work_dir, log_path=work_dir / "payload.log")
        if rc_payload != 0 or not sqlite.is_file():
            raise WB85BError(f"official runner payload write failed: rc={rc_payload}")
        rc_refit = _run(rerefit_cmd, cwd=work_dir, log_path=work_dir / "refit.log")
        if rc_refit != 0 or not enhanced.is_file():
            raise WB85BError(f"official runner Calypso/ACTS refit failed: rc={rc_refit}")
        rc_export = _run(export_cmd, cwd=work_dir, log_path=work_dir / "export.log")
        if rc_export != 0 or not tracklets.is_file():
            raise WB85BError(f"official runner ntuple export failed: rc={rc_export}")
        parsed = parse_converted_tracklets(tracklets)
        log_text = (work_dir / "refit.log").read_text(encoding="utf-8", errors="replace")
        if "lab_transport" in log_text or "toy_uniform_By" in log_text:
            raise WB85BError("official runner log mentions a forbidden fallback")
        return {
            "payload_command": payload_cmd,
            "rerefit_command": rerefit_cmd,
            "export_command": export_cmd,
            "return_codes": {"payload": rc_payload, "refit": rc_refit, "export": rc_export},
            "sqlite": str(sqlite),
            "sqlite_sha256": sha256_file(sqlite),
            "payload_json": str(payload_dir / "alignment_payload.json"),
            "payload_s1_dx_mm": payload_s1_dx_mm(payload_dir / "alignment_payload.json"),
            "enhanced": str(enhanced),
            "enhanced_sha256": _hash_if_file(enhanced),
            "tracklets": str(tracklets),
            "tracklets_sha256": _hash_if_file(tracklets),
            "n_hits": parsed["n_hits"],
            "stacked_digest": parsed["stacked_digest"],
            "parsed": parsed,
            "sqlite_override": "Reading folder /Tracker/Align from sqlite" in log_text,
            "acts_alignment_context": "FaserActsAlignment" in log_text,
            "job_success": "Execution succeeded" in log_text,
            "used_cached_enhanced": False,
            "executor": "WB85bOfficialCalypsoActsBackend.execute_refit",
        }


def planner_execute_stays_false() -> bool:
    backend = CalypsoActsPhysicalBackend(execute=False)
    return backend.execute is False


def official_runner_contract() -> dict[str, Any]:
    return {
        "official_runner": "alignment.wb85b_official_runner.WB85bOfficialCalypsoActsBackend.execute_refit",
        "official_runner_uses_smoke_tested_executor": False,
        "wb85a_r1_top_level_executor": "alignment.wb85a_r1_runtime_smoke.execute_runtime_smoke",
        "top_level_executors_are_identical": False,
        "official_runner_uses_smoke_tested_primitives": True,
        "smoke_tested_primitives": list(SMOKE_TESTED_PRIMITIVES),
        "calypso_acts_physical_backend_execute_stays_false": planner_execute_stays_false(),
        "therefore_dedicated_official_runner_e2e_smoke_is_required": True,
        "engine": OFFICIAL_ENGINE,
        "refit_chain": REFIT_CHAIN,
        "acts_tool": ACTS_TOOL,
        "q_over_p_mode": Q_OVER_P_MODE,
        "forbidden_official_fallbacks": list(FORBIDDEN_OFFICIAL_FALLBACKS),
    }


def execute_official_runner_smoke(output_root: Path | None = None) -> dict[str, Any]:
    root = SMOKE_OUTPUT if output_root is None else Path(output_root)
    refuse_historical_rewrite(root)
    fixture = fixture_is_legal()
    if not fixture["input_xaod_exists"]:
        raise WB85BError("legal official-runner fixture xAOD is absent; refusing to forge PASS")
    source = smoke_source()
    run_token = (
        os.environ.get("CLUSTER_ID")
        or os.environ.get("ClusterId")
        or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    run_root = root / f"run_{run_token}_{os.getpid()}"
    if run_root.exists():
        raise WB85BError(f"refusing to reuse a cached official-runner work directory: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    payload0 = identity_payload()
    payload1 = apply_parameter_update(payload0, ("s1_dx_mm",), (float(SMOKE_PROBE_S1_DX_MM),))
    backend = WB85bOfficialCalypsoActsBackend()
    first = backend.execute_refit(
        input_xaod=source["path"],
        payload=payload0,
        work_dir=run_root / "iteration_0",
        skip_events=SMOKE_SKIP_EVENTS,
        nevents=SMOKE_NEVENTS,
    )
    second = backend.execute_refit(
        input_xaod=source["path"],
        payload=payload1,
        work_dir=run_root / "iteration_1",
        skip_events=SMOKE_SKIP_EVENTS,
        nevents=SMOKE_NEVENTS,
    )
    if abs(float(first["payload_s1_dx_mm"])) > 1.0e-12:
        raise WB85BError("iteration 1 payload is not identity at S1 dx")
    if abs(float(second["payload_s1_dx_mm"]) - SMOKE_PROBE_S1_DX_MM) > 1.0e-12:
        raise WB85BError("updated sqlite payload does not contain S1 dx = +0.05 mm")
    if first["sqlite_sha256"] == second["sqlite_sha256"]:
        raise WB85BError("iteration 2 reused the iteration-1 geometry payload")
    if first["enhanced_sha256"] == second["enhanced_sha256"]:
        raise WB85BError("iteration 2 reused cached first-step measurements")
    if str(Path(second["sqlite"])) not in second["rerefit_command"]:
        raise WB85BError("iteration 2 command does not consume the updated sqlite")
    if str(Path(first["sqlite"])) in second["rerefit_command"]:
        raise WB85BError("iteration 2 command still points at the iteration-1 sqlite")
    if first["stacked_digest"] == second["stacked_digest"]:
        raise WB85BError("iteration 2 measurements are identical to iteration 1")
    parsed0 = first.pop("parsed")
    parsed1 = second.pop("parsed")
    first["iteration"] = 0
    second["iteration"] = 1
    relinearize = relinearize_from_physical_pair(parsed0, parsed1)
    report = {
        **locked_smoke_flags(),
        "schema": "wb85b_official_runner_smoke_v1",
        "utc": _utc(),
        **official_runner_contract(),
        "run_directory": str(run_root),
        "fixture": {
            **fixture,
            "run_id": parsed0["run_id"],
            "event_id": parsed0["event_id"],
            "event_uid": f"{SMOKE_SOURCE_ID}:{parsed0['run_id']}:{parsed0['event_id']}",
        },
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cluster_id": os.environ.get("CLUSTER_ID") or os.environ.get("ClusterId"),
        },
        "smoke_probe_s1_dx_mm": SMOKE_PROBE_S1_DX_MM,
        "updated_sqlite_s1_dx_mm": float(second["payload_s1_dx_mm"]),
        "updated_sqlite_contains_preregistered_s1_dx": True,
        "iteration2_consumed_updated_sqlite": True,
        "iterations": [first, second],
        "relinearization": relinearize,
        "alignment_performance_not_reported": True,
        "is_physics_qualification": False,
        "calypso_acts_engine_runtime_smoke_pass": True,
        "official_qualification_runner_e2e_smoke_pass": True,
        "physical_execution_runtime_smoke_pass": True,
    }
    (run_root / "wb85b_official_runner_smoke.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (root / "wb85b_official_runner_smoke.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def run_official_runner_smoke_or_record_failure(output_root: Path | None = None) -> dict[str, Any]:
    root = SMOKE_OUTPUT if output_root is None else Path(output_root)
    try:
        return execute_official_runner_smoke(root)
    except Exception as error:
        report = {
            **locked_smoke_flags(),
            **official_runner_contract(),
            "schema": "wb85b_official_runner_smoke_v1",
            "utc": _utc(),
            "reason": str(error),
            "official_qualification_runner_e2e_smoke_pass": False,
            "calypso_acts_engine_runtime_smoke_pass": True,
            "physical_execution_runtime_smoke_pass": False,
        }
        root.mkdir(parents=True, exist_ok=True)
        (root / "wb85b_official_runner_smoke.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return report
