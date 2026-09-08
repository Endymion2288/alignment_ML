"""WB85a-r1 official Calypso/ACTS runtime smoke.  Plumbing only.

Does not authorize qualification, does not read WB84 alignment outcomes,
and does not report alignment performance.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from alignment.calypso_physical_replica_production import (
    ACTS_TOOL,
    AUTHORIZED_SOURCES,
    Q_OVER_P_MODE,
    REFIT_TAG,
    SKIP_BASE,
    allocation_plan,
    audit_covariance,
    covariance_matrix,
    refuse_forbidden_path,
    replica_uid,
    sha256_file,
)
from alignment.common_track_solver import (
    SURVEY_FIXED_DZ,
    StationHit,
    apply_parameter_update,
    assemble_track_block,
    solve_common_track,
)
from alignment.physical_common_track_execution import (
    FORBIDDEN_OFFICIAL_FALLBACKS,
    OFFICIAL_ENGINE,
    REFIT_CHAIN,
    calypso_acts_rerefit_command,
    calypso_payload_command,
    command_is_official,
    identity_payload,
    ntuple_export_command,
    refuse_toy_official_fallback,
    refuse_wb84_rewrite,
)
from alignment.wb85_physical_qualification_protocol import refuse_wb83_wb84_write
from alignment.wb85a_protocol_qa import WB85AError


SMOKE_OUTPUT = Path("outputs/mc24_four_station_wb85a_r1_runtime_smoke_v1")
SMOKE_SOURCE_ID = "mc24_100047_00100_00149"
SMOKE_SKIP_EVENTS = 0
SMOKE_NEVENTS = 1
SMOKE_PROBE_S1_DX_MM = 0.05
WB85A_V1_ROOT = Path("outputs/mc24_four_station_wb85a_protocol_qa_v1")


class WB85AR1SmokeError(WB85AError):
    """Fail-closed runtime smoke contract."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def smoke_source() -> dict[str, Any]:
    source = next(item for item in AUTHORIZED_SOURCES if item["source_id"] == SMOKE_SOURCE_ID)
    refuse_forbidden_path(source["path"])
    return source


def fixture_is_legal(*, allocation_path: Path | None = None) -> dict[str, Any]:
    source = smoke_source()
    text = str(source["path"]).lower()
    forbidden = ("00350", "00800", "100116", "100117", "final_blind", "sealed", "overlay_synthetic")
    if any(needle in text for needle in forbidden):
        raise WB85AR1SmokeError(f"smoke fixture path is forbidden: {source['path']}")
    if int(SMOKE_SKIP_EVENTS) >= int(SKIP_BASE):
        raise WB85AR1SmokeError("smoke skip_events must sit below the WB84 allocation skip_base")
    plan = allocation_plan()
    allocated = any(
        row.get("source_id") == SMOKE_SOURCE_ID and int(row.get("xaod_entry_index", -1)) == SMOKE_SKIP_EVENTS
        for row in plan["events"]
    )
    if allocated:
        raise WB85AR1SmokeError("smoke fixture belongs to the WB84 qualification allocation")
    allocation_used = "alignment.calypso_physical_replica_production.allocation_plan"
    if allocation_path is not None:
        path = Path(allocation_path)
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if any(
                row.get("source_id") == SMOKE_SOURCE_ID
                and int(row.get("xaod_entry_index", -1)) == SMOKE_SKIP_EVENTS
                for row in payload.get("events", [])
            ):
                raise WB85AR1SmokeError("smoke fixture belongs to the WB84 qualification allocation")
            allocation_used = f"{allocation_used}; {path}"
    xaod = Path(source["path"])
    return {
        "source_id": SMOKE_SOURCE_ID,
        "input_xaod": source["path"],
        "input_xaod_exists": xaod.is_file(),
        "skip_events": SMOKE_SKIP_EVENTS,
        "nevents": SMOKE_NEVENTS,
        "below_wb84_skip_base": True,
        "in_wb84_qualification_allocation": False,
        "allocation_checked": allocation_used,
        "run_range": source["run_range"],
        "channel": source["channel"],
        "family": source["family"],
        "prohibited_domain": False,
    }


def _run(command: str, *, cwd: Path, log_path: Path) -> int:
    refuse_forbidden_path(cwd)
    refuse_wb84_rewrite(cwd)
    refuse_wb83_wb84_write(cwd)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            ["bash", "-lc", command],
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(result.returncode)


def _hash_if_file(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def parse_converted_tracklets(tracklets_path: Path) -> dict[str, Any]:
    import uproot

    if not tracklets_path.is_file():
        raise WB85AR1SmokeError(f"converted tracklets missing: {tracklets_path}")
    with uproot.open(tracklets_path) as handle:
        tree = handle["tracklets"]
        arrays = tree.arrays(library="np")
    n = len(arrays["event_id"])
    if n <= 0:
        raise WB85AR1SmokeError("converted tracklets contain no rows")
    hits = []
    for index in range(n):
        row = {name: arrays[name][index] for name in arrays}
        matrix = covariance_matrix(row)
        cov = audit_covariance(matrix)
        if not cov["accepted"]:
            continue
        hits.append(
            {
                "station_id": int(row["station_id"]),
                "event_id": int(row["event_id"]),
                "run_id": int(row["run_id"]),
                "x_mm": float(row["x_mm"]),
                "y_mm": float(row["y_mm"]),
                "tx": float(row["tx"]),
                "ty": float(row["ty"]),
                "covariance_4x4": matrix,
                "truth_particle_id": int(row["truth_particle_id"]) if "truth_particle_id" in arrays else -1,
            }
        )
    if not hits:
        raise WB85AR1SmokeError("no SPD physical hits survived covariance admission")
    stacked = np.concatenate(
        [np.asarray([hit["x_mm"], hit["y_mm"], hit["tx"], hit["ty"]], dtype=np.float64) for hit in hits]
    )
    return {
        "n_rows": n,
        "n_hits": len(hits),
        "run_id": hits[0]["run_id"],
        "event_id": hits[0]["event_id"],
        "stations": sorted({hit["station_id"] for hit in hits}),
        "stacked_digest": hashlib.sha256(stacked.tobytes()).hexdigest(),
        "hits": hits,
    }


def _payload_with_s1_dx(delta_mm: float) -> dict[int, tuple[float, ...]]:
    start = identity_payload()
    return apply_parameter_update(start, ("s1_dx_mm",), (float(delta_mm),))


def relinearize_from_physical_pair(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    """Plumbing-only 1-parameter physical FD.  Not an alignment performance claim."""
    by_key = {}
    for hit in first["hits"]:
        key = (int(hit["truth_particle_id"]), int(hit["station_id"]))
        by_key.setdefault(key, {})["first"] = hit
    for hit in second["hits"]:
        key = (int(hit["truth_particle_id"]), int(hit["station_id"]))
        by_key.setdefault(key, {})["second"] = hit
    paired = [pair for pair in by_key.values() if "first" in pair and "second" in pair]
    if len(paired) < 2:
        raise WB85AR1SmokeError("cannot relinearize: fewer than two matched physical hits")
    residual = []
    jacobian = []
    hits = []
    for index, pair in enumerate(paired):
        a = np.asarray(
            [pair["first"]["x_mm"], pair["first"]["y_mm"], pair["first"]["tx"], pair["first"]["ty"]],
            dtype=np.float64,
        )
        b = np.asarray(
            [pair["second"]["x_mm"], pair["second"]["y_mm"], pair["second"]["tx"], pair["second"]["ty"]],
            dtype=np.float64,
        )
        residual.append(a)
        jacobian.append((b - a) / float(SMOKE_PROBE_S1_DX_MM))
        hits.append(
            StationHit(
                track_id=1,
                station=int(pair["first"]["station_id"]),
                measurement_id=f"smoke:{index}:s{pair['first']['station_id']}",
                observed=a,
                covariance=np.asarray(pair["first"]["covariance_4x4"], dtype=np.float64),
            )
        )
    residual_v = np.concatenate(residual)
    jac_global = np.concatenate(jacobian).reshape(-1, 1)
    n_meas = residual_v.size
    jac_xi = np.eye(n_meas, 5, dtype=np.float64)
    if n_meas > 5:
        jac_xi[5:, :] += 0.25 * np.eye(n_meas - 5, 5, dtype=np.float64)
    block = assemble_track_block(hits, jac_xi, jac_global, residual_v)
    solution = solve_common_track((block,), ("s1_dx_mm",), survey_mode=SURVEY_FIXED_DZ, damping=0.0)
    return {
        "n_matched_hits": len(paired),
        "parameter_names": ["s1_dx_mm"],
        "solver_status": solution.status,
        "solver_ok": bool(solution.ok),
        "n_dropped": solution.n_dropped,
        "used_lab_transport": False,
        "used_toy_uniform_By": False,
        "alignment_performance_not_reported": True,
    }


def _write_smoke_json(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_fail_closed_smoke_report(
    output_root: Path,
    *,
    reason: str,
    fixture: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        **locked_smoke_flags(),
        "schema": "wb85a_r1_runtime_smoke_v1",
        "utc": _utc(),
        "reason": reason,
        "fixture": fixture,
        "physical_execution_runtime_smoke_pass": False,
        "physical_execution_ready": False,
        "alignment_performance_not_reported": True,
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cluster_id": os.environ.get("CLUSTER_ID") or os.environ.get("ClusterId"),
        },
    }
    if extra:
        report.update(extra)
    _write_smoke_json(Path(output_root) / "wb85a_r1_runtime_smoke.json", report)
    return report


def execute_runtime_smoke(output_root: Path | None = None) -> dict[str, Any]:
    root = SMOKE_OUTPUT if output_root is None else Path(output_root)
    refuse_forbidden_path(root)
    refuse_wb84_rewrite(root)
    refuse_wb83_wb84_write(root)
    if "wb85a_protocol_qa_v1" in str(root) and "wb85a_r1" not in str(root):
        raise WB85AR1SmokeError("must not write into the historical WB85a artifact root")
    fixture = fixture_is_legal()
    if not fixture["input_xaod_exists"]:
        return write_fail_closed_smoke_report(
            root,
            reason="legal smoke fixture xAOD is absent; refusing to forge PASS",
            fixture=fixture,
        )
    source = smoke_source()
    run_token = (
        os.environ.get("CLUSTER_ID")
        or os.environ.get("ClusterId")
        or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    run_root = root / f"run_{run_token}_{os.getpid()}"
    if run_root.exists():
        raise WB85AR1SmokeError(f"refusing to reuse a cached smoke work directory: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    payload0 = identity_payload()
    payload1 = _payload_with_s1_dx(SMOKE_PROBE_S1_DX_MM)
    if payload0[1][0] == payload1[1][0]:
        raise WB85AR1SmokeError("smoke probe did not change S1 dx")
    iterations = []
    parsed = []
    for index, payload in enumerate((payload0, payload1)):
        work = run_root / f"iteration_{index}"
        if work.exists():
            raise WB85AR1SmokeError(f"refusing to reuse a cached smoke work directory: {work}")
        work.mkdir(parents=True, exist_ok=False)
        payload_dir = work / "calypso_payload"
        enhanced = work / "enhanced_tracklets.root"
        tracklets = work / "tracklets.root"
        propagations = work / "propagations.root"
        payload_cmd = calypso_payload_command(output_dir=payload_dir, payload=payload)
        sqlite = payload_dir / "tracker_alignment.sqlite"
        catalog = payload_dir / "PoolFileCatalog.xml"
        rerefit_cmd = calypso_acts_rerefit_command(
            input_xaod=source["path"],
            sqlite=sqlite,
            pool_catalog=catalog,
            outfile=enhanced,
            skip_events=SMOKE_SKIP_EVENTS,
            nevents=SMOKE_NEVENTS,
        )
        export_cmd = ntuple_export_command(enhanced=enhanced, tracklets=tracklets, propagations=propagations)
        if not command_is_official(rerefit_cmd):
            raise WB85AR1SmokeError("smoke rerefit command is not official")
        refuse_toy_official_fallback(rerefit_cmd)
        refuse_toy_official_fallback(payload_cmd)
        refuse_toy_official_fallback(export_cmd)
        for command in (payload_cmd, rerefit_cmd, export_cmd):
            lowered = command.lower()
            if "lab_transport" in lowered or "toy_uniform" in lowered or "field_y=" in lowered:
                raise WB85AR1SmokeError("silent toy/lab fallback appeared in an official smoke command")
        rc_payload = _run(payload_cmd, cwd=work, log_path=work / "payload.log")
        if rc_payload != 0 or not sqlite.is_file():
            raise WB85AR1SmokeError(f"payload write failed at iteration {index}: rc={rc_payload}")
        rc_refit = _run(rerefit_cmd, cwd=work, log_path=work / "refit.log")
        if rc_refit != 0 or not enhanced.is_file():
            raise WB85AR1SmokeError(f"Calypso/ACTS refit failed at iteration {index}: rc={rc_refit}")
        rc_export = _run(export_cmd, cwd=work, log_path=work / "export.log")
        if rc_export != 0 or not tracklets.is_file():
            raise WB85AR1SmokeError(f"ntuple export failed at iteration {index}: rc={rc_export}")
        parsed_row = parse_converted_tracklets(tracklets)
        parsed.append(parsed_row)
        log_text = (work / "refit.log").read_text(encoding="utf-8", errors="replace")
        if "lab_transport" in log_text or "toy_uniform_By" in log_text:
            raise WB85AR1SmokeError("refit log mentions a forbidden fallback")
        iterations.append(
            {
                "iteration": index,
                "payload_s1_dx_mm": float(payload[1][0]),
                "payload_command": payload_cmd,
                "rerefit_command": rerefit_cmd,
                "export_command": export_cmd,
                "return_codes": {
                    "payload": rc_payload,
                    "refit": rc_refit,
                    "export": rc_export,
                },
                "sqlite": str(sqlite),
                "sqlite_sha256": _hash_if_file(sqlite),
                "enhanced": str(enhanced),
                "enhanced_sha256": _hash_if_file(enhanced),
                "tracklets": str(tracklets),
                "tracklets_sha256": _hash_if_file(tracklets),
                "propagations_sha256": _hash_if_file(propagations),
                "n_hits": parsed_row["n_hits"],
                "stacked_digest": parsed_row["stacked_digest"],
                "sqlite_override": "Reading folder /Tracker/Align from sqlite" in log_text,
                "acts_alignment_context": "FaserActsAlignment" in log_text,
                "job_success": "Execution succeeded" in log_text,
                "used_cached_enhanced": False,
            }
        )
    first, second = iterations
    if first["sqlite_sha256"] == second["sqlite_sha256"]:
        raise WB85AR1SmokeError("iteration 2 reused the iteration-1 geometry payload")
    if first["enhanced_sha256"] == second["enhanced_sha256"]:
        raise WB85AR1SmokeError("iteration 2 reused cached first-step measurements")
    if first["tracklets_sha256"] == second["tracklets_sha256"]:
        raise WB85AR1SmokeError("iteration 2 parsed the same measurement file as iteration 1")
    if first["enhanced"] == second["enhanced"] or first["tracklets"] == second["tracklets"]:
        raise WB85AR1SmokeError("iteration 2 reused a cached measurement path")
    if str(Path(second["sqlite"])) not in second["rerefit_command"]:
        raise WB85AR1SmokeError("iteration 2 command does not load the updated sqlite")
    if str(Path(first["sqlite"])) in second["rerefit_command"]:
        raise WB85AR1SmokeError("iteration 2 command still points at the iteration-1 sqlite")
    if parsed[0]["stacked_digest"] == parsed[1]["stacked_digest"]:
        raise WB85AR1SmokeError("iteration 2 measurements are identical to iteration 1")
    relinearize = relinearize_from_physical_pair(parsed[0], parsed[1])
    event_uid = replica_uid(SMOKE_SOURCE_ID, parsed[0]["run_id"], parsed[0]["event_id"])
    report = {
        **locked_smoke_flags(),
        "schema": "wb85a_r1_runtime_smoke_v1",
        "utc": _utc(),
        "engine": OFFICIAL_ENGINE,
        "refit_chain": REFIT_CHAIN,
        "acts_tool": ACTS_TOOL,
        "q_over_p_mode": Q_OVER_P_MODE,
        "refit_tag": REFIT_TAG,
        "run_directory": str(run_root),
        "fixture": {**fixture, "event_uid": event_uid, "run_id": parsed[0]["run_id"], "event_id": parsed[0]["event_id"]},
        "smoke_probe_s1_dx_mm": SMOKE_PROBE_S1_DX_MM,
        "geometry_update_is_preregistered_smoke_probe": True,
        "geometry_update_is_not_recovered_alignment": True,
        "iterations": iterations,
        "iteration2_used_updated_geometry": True,
        "iteration2_used_new_measurements": True,
        "relinearization": relinearize,
        "forbidden_fallbacks_absent": True,
        "forbidden_official_fallbacks": list(FORBIDDEN_OFFICIAL_FALLBACKS),
        "silent_fallback": False,
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cluster_id": os.environ.get("CLUSTER_ID") or os.environ.get("ClusterId"),
            "calypso_identity": {
                "athena": "24.0.41",
                "geometry_name": "FASERNU-04",
                "global_tag": "OFLCOND-FASER-06",
                "acts_tool": ACTS_TOOL,
                "q_over_p_mode": Q_OVER_P_MODE,
            },
        },
        "physical_execution_runtime_smoke_pass": True,
        "physical_execution_ready": True,
        "is_physics_qualification": False,
    }
    _write_smoke_json(run_root / "wb85a_r1_runtime_smoke.json", report)
    _write_smoke_json(root / "wb85a_r1_runtime_smoke.json", report)
    return report


def run_runtime_smoke_or_record_failure(output_root: Path | None = None) -> dict[str, Any]:
    root = SMOKE_OUTPUT if output_root is None else Path(output_root)
    try:
        return execute_runtime_smoke(root)
    except Exception as error:
        return write_fail_closed_smoke_report(root, reason=str(error))


def locked_smoke_flags() -> dict[str, Any]:
    return {
        "qualification_authorized": False,
        "executable": False,
        "looked_at_physical_alignment_outcomes": False,
        "alignment_oracle_qualified_for_physical_FASER": False,
        "ml_alignment_eval_authorized": False,
        "wb86_automatically_authorized": False,
        "alignment_performance_not_reported": True,
    }
