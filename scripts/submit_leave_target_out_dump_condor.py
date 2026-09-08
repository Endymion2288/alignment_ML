#!/usr/bin/env python3
"""Submit one HTCondor job per WB87 source for the independent LTO dump."""

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
from datasets.leave_target_out_state_materialization import (
    lto_dump_path_for_source,
    load_config,
    plugin_library_path,
)

PROJECT_ROOT = project_root()
WORKER = PROJECT_ROOT / "scripts" / "run_leave_target_out_dump_condor.sh"


def _write_submit(target: Path, jobs_file: Path, log_dir: Path, memory_mb: int, flavour: str) -> None:
    target.write_text(
        "\n".join(
            (
                "universe = vanilla",
                f"executable = {WORKER}",
                'arguments = "$(input_xaod) $(source_id) $(output_jsonl) $(nevents) $(project_root)"',
                f"output = {log_dir}/$(source_id).$(ClusterId).$(ProcId).out",
                f"error = {log_dir}/$(source_id).$(ClusterId).$(ProcId).err",
                f"log = {log_dir}/$(source_id).$(ClusterId).log",
                f"request_memory = {memory_mb}",
                "request_cpus = 1",
                f'+JobFlavour = "{flavour}"',
                "getenv = False",
                'environment = "EOS_MGM_URL=root://eospublic.cern.ch"',
                f"queue input_xaod,source_id,output_jsonl,nevents,project_root from {jobs_file}",
                "",
            )
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/leave_target_out_state_materialization_v1.yaml"
    )
    parser.add_argument(
        "--submit-dir", default="outputs/leave_target_out_dump_v1/condor"
    )
    parser.add_argument("--request-memory-mb", type=int, default=4000)
    parser.add_argument("--job-flavour", default="workday")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    config_path = resolve_under_root(PROJECT_ROOT, args.config)
    config = load_config(config_path)
    library = plugin_library_path(config)
    if not library.is_file():
        raise FileNotFoundError(f"Independent LTO helper is absent: {library}")
    submit_dir = resolve_under_root(PROJECT_ROOT, args.submit_dir)
    log_dir = submit_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for split in ("construction_sources", "validation_sources"):
        for spec in config["mc_data"][split]:
            output = lto_dump_path_for_source(config, spec["source_id"])
            output.parent.mkdir(parents=True, exist_ok=True)
            xaod = Path(spec["input_xaod"])
            if not xaod.is_file():
                raise FileNotFoundError(f"input xAOD is missing: {xaod}")
            if output.is_file():
                raise FileExistsError(f"refusing to overwrite LTO dump: {output}")
            jobs.append(
                f"{spec['input_xaod']} {spec['source_id']} {output} "
                f"{int(config['nevents'])} {PROJECT_ROOT}"
            )
    jobs_file = submit_dir / "jobs.txt"
    jobs_file.write_text("\n".join(jobs) + "\n", encoding="utf-8")
    submit_file = submit_dir / "ckf_leave_target_out_dump.sub"
    _write_submit(submit_file, jobs_file, log_dir, args.request_memory_mb, args.job_flavour)
    payload = {
        "kind": "condor_submit_plan",
        "task": "SB-B13",
        "workbook": 109,
        "n_jobs": len(jobs),
        "nevents": int(config["nevents"]),
        "new_source_campaign": False,
        "geometry_write_allowed": False,
        "kalman_fitter_tool_fit_called": False,
        "helper": "CkfLeaveTargetOutDumpAlg",
        "plugin_sha256": sha256_file(library),
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
