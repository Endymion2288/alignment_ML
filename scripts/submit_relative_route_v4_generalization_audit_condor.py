#!/usr/bin/env python3
"""Submit Workbook-71 RelativeRoute V4 train->development generalization audit.

One GPU job.  Read-only mechanism diagnosis: does not train, retune, or open
unused final blind / sealed test.  Interactive lxplus GPUs are not used.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER = PROJECT_ROOT / "scripts" / "run_relative_route_v4_generalization_audit_condor.sh"
GPU_REQUIREMENT = (
    '(TARGET.GPUs_Capability >= 7.5) || regexp("H100|A100|L40|RTX", TARGET.GPUs_DeviceName)'
)


def _write_submit(
    submit_dir: Path, *, job_flavour: str, memory_mb: int, cpus: int, disk_kb: int
) -> Path:
    submit_file = submit_dir / "generalization_audit.sub"
    log_dir = submit_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    submit_file.write_text(
        "\n".join(
            (
                "universe = vanilla",
                f"executable = {WORKER}",
                f"arguments = {PROJECT_ROOT}",
                f"output = {log_dir}/audit.$(ClusterId).$(ProcId).out",
                f"error = {log_dir}/audit.$(ClusterId).$(ProcId).err",
                f"log = {log_dir}/audit.$(ClusterId).log",
                f"request_memory = {int(memory_mb)}",
                f"request_cpus = {int(cpus)}",
                f"request_disk = {int(disk_kb)}",
                "request_gpus = 1",
                f"requirements = {GPU_REQUIREMENT}",
                f'+JobFlavour = "{job_flavour}"',
                "getenv = True",
                "queue",
                "",
            )
        ),
        encoding="utf-8",
    )
    return submit_file


def _condor_submit(submit_file: Path) -> subprocess.CompletedProcess[str]:
    command = [
        "bash",
        "-lc",
        "source /usr/share/Modules/init/bash "
        "&& module load lxbatch/eossubmit "
        "&& myschedd out "
        f"&& condor_submit {shlex.quote(str(submit_file))}",
    ]
    return subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submit-dir",
        default=str(PROJECT_ROOT / "outputs/mc24_four_station_relative_route_v4_generalization_audit_v1/condor"),
    )
    parser.add_argument("--job-flavour", default="tomorrow")
    parser.add_argument("--request-memory-mb", type=int, default=16000)
    parser.add_argument("--request-cpus", type=int, default=4)
    parser.add_argument("--request-disk-kb", type=int, default=2000000)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    submit_dir = Path(args.submit_dir).expanduser().resolve()
    submit_dir.mkdir(parents=True, exist_ok=True)
    submit_file = _write_submit(
        submit_dir,
        job_flavour=args.job_flavour,
        memory_mb=args.request_memory_mb,
        cpus=args.request_cpus,
        disk_kb=args.request_disk_kb,
    )
    print(submit_file)
    if not args.submit:
        return
    result = _condor_submit(submit_file)
    print(result.stdout)
    print(result.stderr)
    (submit_dir / "submit_stdout.txt").write_text(result.stdout, encoding="utf-8")
    (submit_dir / "submit_stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
