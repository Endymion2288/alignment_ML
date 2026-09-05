"""Submit Workbook 73 route-representation extraction jobs (one per family).

Diagnostic-only feature extraction; no production training.  CPU slots are
abundant and sufficient for the small frozen-backbone + head forward pass.
Submits through the EOS-aware schedd so /eos output paths are accepted.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKER = PROJECT_ROOT / "scripts" / "run_route_representation_extract_condor.sh"
LOG_DIR = PROJECT_ROOT / "outputs" / "mc24_four_station_route_representation_domain_audit_v1" / "condor"
CPU_REQUIREMENT = '(TARGET.OpSysAndVer == "AlmaLinux9")'
GPU_REQUIREMENT = (
    '(TARGET.GPUs_Capability >= 7.0) || '
    'regexp("H100|H200|A100|V100|T4|L40|RTX", TARGET.GPUs_DeviceName)'
)
FAMILIES = ("family1", "family2")


def _write_submit(family: str, device: str, job_flavour: str) -> Path:
    log_dir = LOG_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    resource_lines = (
        (f"request_gpus = 1", f"requirements = {GPU_REQUIREMENT}")
        if device == "cuda"
        else (f"requirements = {CPU_REQUIREMENT}",)
    )
    submit_file = LOG_DIR / f"extract_{family}.sub"
    submit_file.write_text(
        "\n".join(
            (
                "universe = vanilla",
                f"executable = {WORKER}",
                f"arguments = {PROJECT_ROOT} {family} {device}",
                f"output = {log_dir}/extract_{family}.$(ClusterId).$(ProcId).out",
                f"error = {log_dir}/extract_{family}.$(ClusterId).$(ProcId).err",
                f"log = {log_dir}/extract_{family}.$(ClusterId).log",
                "request_memory = 24000",
                "request_cpus = 4",
                "request_disk = 4000000",
                *resource_lines,
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
        "bash", "-lc",
        "source /usr/share/Modules/init/bash "
        "&& module load lxbatch/eossubmit "
        "&& myschedd out "
        f"&& condor_submit {shlex.quote(str(submit_file))}",
    ]
    return subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--job-flavour", default="workday")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    for family in FAMILIES:
        sub = _write_submit(family, args.device, args.job_flavour)
        print(f"Prepared submit file: {sub}")
        if not args.submit:
            continue
        result = _condor_submit(sub)
        print(result.stdout.strip())
        if result.returncode != 0:
            print(result.stderr.strip())
            raise SystemExit(f"condor_submit failed for {family}")
    if not args.submit:
        print("Dry-run (no --submit). Re-run with --submit to submit to Condor.")


if __name__ == "__main__":
    main()
