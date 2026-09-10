#!/usr/bin/env python3
"""Submit two HTCondor jobs for B14M-S basin diagnosis.  Not 1989."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)

PROJECT_ROOT = project_root()
EOS_WORKER = PROJECT_ROOT / "scripts" / "run_b14m_profile_basin_diagnosis_condor.sh"
AFS_SUBMIT_ROOT = Path(
    "/afs/cern.ch/user/x/xcheng/work/b14ms_basin_condor"
)

JOBS = (
    {
        "part": "evt0_37",
        "source_id": "mc24_100043_00400_00499",
        "input_xaod": (
            "/eos/experiment/faser/data0/sim/mc24/particle_gun/100043/rec/"
            "s0013-r0022/FaserMC-MC24_PG_mumi_fasernu_5mrad_flukaE-100043-00400-00499-s0013-r0022-xAOD.root"
        ),
        "nevents": "50",
        "select_events": "0,37",
        "basin_source": (
            "outputs/leave_target_out_dump_v1/b14m_reopen_smoke/"
            "mc24_100043_00400_00499/ckf_leave_target_out_b14m_reopen.jsonl"
        ),
    },
    {
        "part": "evt86",
        "source_id": "mc24_100048_00000_00049",
        "input_xaod": (
            "/eos/experiment/faser/data0/sim/mc24/particle_gun/100048/rec/"
            "s0013-r0022/FaserMC-MC24_PG_mupl_fasernu_5mrad_flukaE-100048-00000-00049-s0013-r0022-xAOD.root"
        ),
        "nevents": "100",
        "select_events": "86",
        "basin_source": (
            "outputs/leave_target_out_dump_v1/b14m_reopen_smoke/"
            "mc24_100048_00000_00049/ckf_leave_target_out_b14m_reopen.jsonl"
        ),
    },
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submit-dir",
        default="outputs/leave_target_out_dump_v1/b14ms_basin_smoke/condor",
    )
    parser.add_argument("--request-memory-mb", type=int, default=4000)
    parser.add_argument("--job-flavour", default="tomorrow")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    plugin = PROJECT_ROOT / "build" / "leave_target_out_dump" / "libCkfLeaveTargetOutDump.so"
    if not plugin.is_file():
        raise FileNotFoundError(f"Independent LTO helper is absent: {plugin}")
    if not EOS_WORKER.is_file():
        raise FileNotFoundError(f"worker missing: {EOS_WORKER}")
    eos_plan = resolve_under_root(PROJECT_ROOT, args.submit_dir)
    eos_plan.mkdir(parents=True, exist_ok=True)
    afs_dir = AFS_SUBMIT_ROOT
    log_dir = afs_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    worker = afs_dir / "run_b14m_profile_basin_diagnosis_condor.sh"
    worker.write_text(EOS_WORKER.read_text(encoding="utf-8"), encoding="utf-8")
    worker.chmod(0o755)
    lines = []
    outputs = []
    for job in JOBS:
        basin = resolve_under_root(PROJECT_ROOT, job["basin_source"])
        if not basin.is_file():
            raise FileNotFoundError(f"WB129 dump missing: {basin}")
        if not Path(job["input_xaod"]).is_file():
            raise FileNotFoundError(f"xAOD missing: {job['input_xaod']}")
        output = (
            resolve_under_root(
                PROJECT_ROOT, "outputs/leave_target_out_dump_v1/b14ms_basin_smoke"
            )
            / job["source_id"]
            / f"part_{job['part']}.jsonl"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.is_file():
            raise FileExistsError(f"refusing to overwrite {output}")
        outputs.append(str(output))
        lines.append(f"{job['source_id']} {job['part']}")
    jobs_file = afs_dir / "jobs.txt"
    jobs_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    submit_file = afs_dir / "b14ms_basin_diagnosis.sub"
    submit_file.write_text(
        "\n".join(
            (
                "universe = vanilla",
                f"executable = {worker}",
                f"initialdir = {afs_dir}",
                'arguments = "$(source_id) $(part)"',
                f"output = {log_dir}/$(part).$(ClusterId).$(ProcId).out",
                f"error = {log_dir}/$(part).$(ClusterId).$(ProcId).err",
                f"log = {log_dir}/b14ms_basin.$(ClusterId).log",
                f"request_memory = {args.request_memory_mb}",
                "request_cpus = 1",
                f'+JobFlavour = "{args.job_flavour}"',
                "getenv = False",
                f"queue source_id,part from {jobs_file}",
                "",
            )
        ),
        encoding="utf-8",
    )
    payload = {
        "kind": "condor_submit_plan",
        "task": "SB-B14MS",
        "workbook": 130,
        "n_jobs": len(JOBS),
        "not_1989": True,
        "plugin_sha256": sha256_file(plugin),
        "afs_submit_file": str(submit_file),
        "outputs": outputs,
    }
    (eos_plan / "submit_plan.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(submit_file)
    print(f"n_jobs={len(JOBS)} not_1989=true")
    if args.submit:
        subprocess.run(
            ["condor_submit", str(submit_file)],
            check=True,
            cwd=str(afs_dir),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
