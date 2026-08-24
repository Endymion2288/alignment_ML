#!/usr/bin/env python3
"""Submit the workbook-59 GPU training+transfer job.  Does not retune anything."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER = PROJECT_ROOT / "scripts" / "run_four_station_dustbin_aware_condor.sh"
GPU_REQUIREMENT = (
    '(TARGET.GPUs_Capability >= 7.5) || regexp("H100|A100|L40|RTX", TARGET.GPUs_DeviceName)'
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit-dir", default=str(PROJECT_ROOT / "outputs/mc24_four_station_dustbin_aware_route_v1/condor"))
    parser.add_argument("--job-flavour", default="tomorrow")
    parser.add_argument("--request-memory-mb", type=int, default=8000)
    parser.add_argument("--request-cpus", type=int, default=2)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    submit_dir = Path(args.submit_dir).expanduser().resolve()
    submit_dir.mkdir(parents=True, exist_ok=True)
    submit_file = submit_dir / "dustbin_aware_route.sub"
    submit_file.write_text(
        "\n".join(
            (
                "universe = vanilla",
                f"executable = {WORKER}",
                f"arguments = {PROJECT_ROOT}",
                f"output = {submit_dir}/dustbin_aware.$(ClusterId).$(ProcId).out",
                f"error = {submit_dir}/dustbin_aware.$(ClusterId).$(ProcId).err",
                f"log = {submit_dir}/dustbin_aware.$(ClusterId).log",
                f"request_memory = {int(args.request_memory_mb)}",
                f"request_cpus = {int(args.request_cpus)}",
                "request_gpus = 1",
                f"requirements = {GPU_REQUIREMENT}",
                f'+JobFlavour = "{args.job_flavour}"',
                "getenv = True",
                "queue",
                "",
            )
        ),
        encoding="utf-8",
    )
    print(submit_file)
    if not args.submit:
        return
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
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
