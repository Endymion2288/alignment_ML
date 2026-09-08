#!/usr/bin/env python3
"""Submit Workbook-81b Arm D calibration folds to Condor (CPU).

Same request as the formal Workbook-79/80 CPU jobs: 8 CPU, 32 GB, flavour
tomorrow.  Do not submit GPU leftovers.  LCG PyTorch 2.11 has no V100
CC 7.0 kernels; H100 slots are VO-blocked for FASER.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER = PROJECT_ROOT / "scripts" / "run_wb81b_calibration_condor.sh"
FOLDS = ("holdout_family1", "holdout_family2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submit-dir",
        default=str(PROJECT_ROOT / "outputs/mc24_four_station_wb81b_calibration_v1/condor"),
    )
    parser.add_argument("--job-flavour", default="tomorrow")
    parser.add_argument("--request-memory-mb", type=int, default=32000)
    parser.add_argument("--request-cpus", type=int, default=8)
    parser.add_argument("--request-disk-kb", type=int, default=2000000)
    parser.add_argument("--fold", choices=FOLDS, action="append")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    submit_dir = Path(args.submit_dir).expanduser().resolve()
    submit_dir.mkdir(parents=True, exist_ok=True)
    log_dir = submit_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    folds = tuple(args.fold) if args.fold else FOLDS
    submit_files = []
    for fold in folds:
        submit_file = submit_dir / f"{fold}.sub"
        submit_file.write_text(
            "\n".join(
                (
                    "universe = vanilla",
                    f"executable = {WORKER}",
                    f"arguments = {PROJECT_ROOT} {fold}",
                    f"output = {log_dir}/{fold}.$(ClusterId).$(ProcId).out",
                    f"error = {log_dir}/{fold}.$(ClusterId).$(ProcId).err",
                    f"log = {log_dir}/{fold}.$(ClusterId).log",
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
        submit_files.append(submit_file)
    print("Prepared submit files:")
    for path in submit_files:
        print(f"  {path}")
    if not args.submit:
        print("\nDry-run (no --submit). Re-run with --submit to submit to Condor.")
        return
    for path in submit_files:
        command = [
            "bash",
            "-lc",
            "source /usr/share/Modules/init/bash "
            "&& module load lxbatch/eossubmit "
            "&& myschedd out "
            f"&& condor_submit {shlex.quote(str(path))}",
        ]
        result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
        print(f"\n=== condor_submit {path.name} ===")
        print(result.stdout)
        print(result.stderr)
        if result.returncode != 0:
            raise SystemExit(f"condor_submit failed for {path}")


if __name__ == "__main__":
    main()
