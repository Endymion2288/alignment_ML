#!/usr/bin/env python3
"""Submit one HTCondor job per frozen WB87 source for Stage 2 batch dumps."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from datasets.three_st_qp_calibration import dump_path_for_source, load_config

PROJECT_ROOT = project_root()
WORKER = PROJECT_ROOT / "scripts" / "run_three_st_qp_calibration_condor.sh"


def _write_submit(target: Path, jobs_file: Path, log_dir: Path, memory_mb: int, flavour: str) -> None:
    target.write_text(
        "\n".join(
            (
                "universe = vanilla",
                f"executable = {WORKER}",
                'arguments = "$(input_xaod) $(source_id) $(output_jsonl) $(event_jsonl) $(nevents) $(project_root) $(campaign)"',
                f"output = {log_dir}/$(source_id).$(ClusterId).$(ProcId).out",
                f"error = {log_dir}/$(source_id).$(ClusterId).$(ProcId).err",
                f"log = {log_dir}/$(source_id).$(ClusterId).log",
                f"request_memory = {memory_mb}",
                "request_cpus = 1",
                f'+JobFlavour = "{flavour}"',
                "getenv = False",
                'environment = "EOS_MGM_URL=root://eospublic.cern.ch"',
                f"queue input_xaod,source_id,output_jsonl,event_jsonl,nevents,project_root,campaign from {jobs_file}",
                "",
            )
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/three_st_qp_calibration_v1.yaml")
    parser.add_argument(
        "--submit-dir", default="outputs/three_st_qp_calibration_v1/condor"
    )
    parser.add_argument("--request-memory-mb", type=int, default=4000)
    parser.add_argument("--job-flavour", default="workday")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    config_path = resolve_under_root(PROJECT_ROOT, args.config)
    config = load_config(config_path)
    submit_dir = resolve_under_root(PROJECT_ROOT, args.submit_dir)
    log_dir = submit_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    campaign = "batch"
    nevents = int(config["batch"]["nevents"])
    jobs = []
    for split in ("batch_construction_sources", "batch_validation_sources"):
        for spec in config["mc_data"][split]:
            output = dump_path_for_source(config, spec["source_id"], "tracks", campaign)
            event_output = dump_path_for_source(
                config, spec["source_id"], "events", campaign
            )
            if output.exists() or event_output.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing batch dump: {output}"
                )
            output.parent.mkdir(parents=True, exist_ok=True)
            xaod = Path(spec["input_xaod"])
            if not xaod.is_file():
                raise FileNotFoundError(f"input xAOD is missing: {xaod}")
            jobs.append(
                f"{spec['input_xaod']} {spec['source_id']} {output} {event_output} "
                f"{nevents} {PROJECT_ROOT} {campaign}"
            )
    jobs_file = submit_dir / "jobs.txt"
    jobs_file.write_text("\n".join(jobs) + "\n", encoding="utf-8")
    submit_file = submit_dir / "three_st_qp_calibration.sub"
    _write_submit(submit_file, jobs_file, log_dir, args.request_memory_mb, args.job_flavour)
    payload = {
        "kind": "condor_submit_plan",
        "task": "YASU-S2",
        "campaign": campaign,
        "n_jobs": len(jobs),
        "nevents": nevents,
        "new_source_campaign": False,
        "geometry_write_allowed": False,
        "truth_used_as_fit_seed": False,
        "config_sha256": sha256_file(config_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "submit_file": str(submit_file),
    }
    (submit_dir / "submit_plan.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(jobs)} jobs to {submit_file}")
    if not args.submit:
        return 0
    command = [
        "bash",
        "-lc",
        "source /usr/share/Modules/init/bash "
        "&& module load lxbatch/eossubmit "
        "&& myschedd out "
        f"&& condor_submit {shlex.quote(str(submit_file))}",
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
    print(result.stdout)
    print(result.stderr)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
