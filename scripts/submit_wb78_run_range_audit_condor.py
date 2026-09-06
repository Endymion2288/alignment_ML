#!/usr/bin/env python3
"""Submit Workbook-78 read-only run-range data-contract audit to Condor (CPU)."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER = PROJECT_ROOT / "scripts" / "run_wb78_run_range_audit_condor.sh"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submit-dir",
        default=str(
            PROJECT_ROOT / "outputs/mc24_four_station_run_range_data_contract_v1/condor"
        ),
    )
    parser.add_argument("--job-flavour", default="tomorrow")
    parser.add_argument("--request-memory-mb", type=int, default=32000)
    parser.add_argument("--request-cpus", type=int, default=8)
    parser.add_argument("--request-disk-kb", type=int, default=2000000)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    submit_dir = Path(args.submit_dir).expanduser().resolve()
    submit_dir.mkdir(parents=True, exist_ok=True)
    log_dir = submit_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = "wb78_run_range_data_contract"
    submit_file = submit_dir / f"{tag}.sub"
    submit_file.write_text(
        "\n".join(
            (
                "universe = vanilla",
                f"executable = {WORKER}",
                f"arguments = {PROJECT_ROOT}",
                f"output = {log_dir}/{tag}.$(ClusterId).$(ProcId).out",
                f"error = {log_dir}/{tag}.$(ClusterId).$(ProcId).err",
                f"log = {log_dir}/{tag}.$(ClusterId).log",
                f"request_memory = {int(args.request_memory_mb)}",
                f"request_cpus = {int(args.request_cpus)}",
                f"request_disk = {int(args.request_disk_kb)}",
                f'+JobFlavour = "{args.job_flavour}"',
                "getenv = True",
                "queue",
                "",
            )
        ),
        encoding="utf-8",
    )
    print(f"Prepared submit file:\n  {submit_file}")
    if not args.submit:
        print("\nDry-run (no --submit). Re-run with --submit to submit to Condor.")
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
        raise SystemExit(f"condor_submit failed for {submit_file}")


if __name__ == "__main__":
    main()
