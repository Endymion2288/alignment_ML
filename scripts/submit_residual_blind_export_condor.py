#!/usr/bin/env python3
"""Submit the workbook-75 residual-blind tracklet export to CERN Condor.

One job per declared input xAOD.  The worker
``scripts/run_residual_blind_export_condor.sh`` runs the validated nominal
chain (persisted SegmentFit -> GhostBusters -> NtupleDumperAlg detailed
tracklets -> canonical converter -> content audit) with no alignment
payload, no physical FD point, and no residual-based selection.  Every job
writes ``job_provenance.json``; failed jobs exit nonzero and are classified
by the report step, never silently skipped.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alignment.operating_protocol_v1_final_closure import project_root, sha256_file
from alignment.physically_distinct_track_coverage_export import (
    EXPORT_JOB_SCHEMA,
    load_export_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER = PROJECT_ROOT / "scripts" / "run_residual_blind_export_condor.sh"


def _require_eos_path(path: Path, *, label: str) -> None:
    if not str(path).startswith("/eos/"):
        raise ValueError(f"EOSSubmit requires {label} to be under /eos: {path}")


def _write_candidate_submit(
    target: Path,
    *,
    jobs_file: Path,
    log_dir: Path,
    candidate_id: str,
    request_memory_mb: int,
    job_flavour: str,
) -> None:
    target.write_text(
        "\n".join(
            (
                "universe = vanilla",
                f"executable = {WORKER}",
                'arguments = "$(input_xaod) $(source_id) $(output_dir) $(project_root)"',
                f"output = {log_dir}/{candidate_id}.$(ClusterId).$(ProcId).out",
                f"error = {log_dir}/{candidate_id}.$(ClusterId).$(ProcId).err",
                f"log = {log_dir}/{candidate_id}.$(ClusterId).log",
                f"request_memory = {request_memory_mb}",
                "request_cpus = 1",
                f'+JobFlavour = "{job_flavour}"',
                "getenv = False",
                "environment = \"EOS_MGM_URL=root://eospublic.cern.ch\"",
                f"queue input_xaod,source_id,output_dir,project_root from {jobs_file}",
                "",
            )
        ),
        encoding="utf-8",
    )


def _submit(submit_file: Path) -> subprocess.CompletedProcess[str]:
    command = [
        "bash",
        "-lc",
        "source /usr/share/Modules/init/bash "
        "&& module load lxbatch/eossubmit "
        "&& myschedd out "
        f"&& condor_submit {shlex.quote(str(submit_file))}",
    ]
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--submit-dir", required=True)
    parser.add_argument("--request-memory-mb", type=int, default=8000)
    parser.add_argument("--job-flavour", default="tomorrow")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.request_memory_mb < 1:
        parser.error("--request-memory-mb must be positive")
    if not WORKER.is_file():
        raise FileNotFoundError(WORKER)
    if not WORKER.stat().st_mode & 0o111:
        raise PermissionError(f"Condor worker must be executable: {WORKER}")

    config_path = Path(args.config).expanduser().resolve()
    config = load_export_config(config_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    submit_dir = Path(args.submit_dir).expanduser().resolve()
    if submit_dir.exists() and any(submit_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty Condor submit directory")
    log_dir = submit_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=False)
    for label, path in (
        ("worker", WORKER),
        ("configuration", config_path),
        ("output directory", output_dir),
        ("submit directory", submit_dir),
        ("log directory", log_dir),
    ):
        _require_eos_path(path, label=label)

    condor_cfg = config.get("condor") or {}
    manifest: dict[str, Any] = {
        "schema_version": EXPORT_JOB_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": None,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "output_dir": str(output_dir),
        "submit_dir": str(submit_dir),
        "worker": str(WORKER),
        "schedd_mode": "eossubmit",
        "job_flavour": str(condor_cfg.get("job_flavour") or args.job_flavour),
        "request_memory_mb": int(condor_cfg.get("request_memory_mb") or args.request_memory_mb),
        "residual_blind": True,
        "alignment_payload_injected": False,
        "physical_fd_point_constructed": False,
        "candidates": [],
        "submitted": False,
    }
    from alignment.cad_survey_nov22 import git_head_sha

    manifest["git_head"] = git_head_sha(PROJECT_ROOT)

    for candidate in config.get("export_candidates") or []:
        cid = str(candidate["id"])
        jobs_file = submit_dir / f"{cid}.jobs.txt"
        lines = []
        for row in candidate.get("inputs") or []:
            source_id = str(row["source_id"])
            input_path = str(row["path"])
            source_out = output_dir / "exports" / cid / source_id
            lines.append(f"{input_path},{source_id},{source_out},{PROJECT_ROOT}")
        jobs_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        submit_file = submit_dir / f"{cid}.sub"
        _write_candidate_submit(
            submit_file,
            jobs_file=jobs_file,
            log_dir=log_dir,
            candidate_id=cid,
            request_memory_mb=int(manifest["request_memory_mb"]),
            job_flavour=str(manifest["job_flavour"]),
        )
        entry: dict[str, Any] = {
            "id": cid,
            "n_jobs": len(lines),
            "jobs_file": str(jobs_file),
            "submit_file": str(submit_file),
            "submitted": False,
        }
        if args.submit:
            result = _submit(submit_file)
            entry["condor_submit_output"] = result.stdout
            entry["condor_submit_returncode"] = result.returncode
            entry["submitted"] = result.returncode == 0
            if result.returncode:
                manifest["candidates"].append(entry)
                (submit_dir / "submission.json").write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                raise RuntimeError("condor_submit failed:\n" + result.stdout)
        manifest["candidates"].append(entry)

    manifest["submitted"] = bool(args.submit) and all(
        row["submitted"] for row in manifest["candidates"]
    )
    (submit_dir / "submission.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "submit_dir": str(submit_dir),
                "candidates": [
                    {"id": row["id"], "n_jobs": row["n_jobs"], "submitted": row["submitted"]}
                    for row in manifest["candidates"]
                ],
                "submitted": bool(manifest["submitted"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
