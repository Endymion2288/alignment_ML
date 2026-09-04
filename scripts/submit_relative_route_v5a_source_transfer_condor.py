#!/usr/bin/env python3
"""Submit Workbook-72 V5A source-transfer CV head-only GPU jobs.

Submits the 2-fold x 2-arm paired training grid (4 jobs):

    fold holdout_family1 (train on family2) x {control, primary}
    fold holdout_family2 (train on family1) x {control, primary}

Each (fold, arm) job re-trains from the same frozen Workbook-64 parent, the
same seed (20260822), and zero-initialization; the only difference between
control and primary is ``route_correction_bound`` (null vs 4.0).  No
development / final-blind / sealed data is touched.  Logs stay under the
project output tree.  Interactive lxplus GPUs are not used for the 30-epoch
loop.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER = PROJECT_ROOT / "scripts" / "run_relative_route_v5a_source_transfer_condor.sh"
GPU_REQUIREMENT = (
    '(TARGET.GPUs_Capability >= 7.5) || regexp("H100|A100|L40|RTX", TARGET.GPUs_DeviceName)'
)
FOLDS = ("holdout_family1", "holdout_family2")
ARMS = ("control", "primary")


def _write_submit(
    submit_dir: Path,
    fold: str,
    arm: str,
    *,
    job_flavour: str,
    memory_mb: int,
    cpus: int,
    disk_kb: int,
) -> Path:
    tag = f"{fold}_{arm}"
    submit_file = submit_dir / f"{tag}.sub"
    log_dir = submit_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    submit_file.write_text(
        "\n".join(
            (
                "universe = vanilla",
                f"executable = {WORKER}",
                f"arguments = {PROJECT_ROOT} {fold} {arm}",
                f"output = {log_dir}/{tag}.$(ClusterId).$(ProcId).out",
                f"error = {log_dir}/{tag}.$(ClusterId).$(ProcId).err",
                f"log = {log_dir}/{tag}.$(ClusterId).log",
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
        default=str(
            PROJECT_ROOT
            / "outputs/mc24_four_station_relative_route_v5a_source_transfer_v1/training/condor"
        ),
    )
    parser.add_argument("--job-flavour", default="tomorrow")
    parser.add_argument("--request-memory-mb", type=int, default=16000)
    parser.add_argument("--request-cpus", type=int, default=4)
    parser.add_argument("--request-disk-kb", type=int, default=2000000)
    parser.add_argument("--fold", choices=FOLDS, action="append")
    parser.add_argument("--arm", choices=ARMS, action="append")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    submit_dir = Path(args.submit_dir).expanduser().resolve()
    submit_dir.mkdir(parents=True, exist_ok=True)

    folds = tuple(args.fold) if args.fold else FOLDS
    arms = tuple(args.arm) if args.arm else ARMS

    submit_files = []
    for fold in folds:
        for arm in arms:
            submit_files.append(
                _write_submit(
                    submit_dir,
                    fold,
                    arm,
                    job_flavour=args.job_flavour,
                    memory_mb=args.request_memory_mb,
                    cpus=args.request_cpus,
                    disk_kb=args.request_disk_kb,
                )
            )

    print("Prepared submit files:")
    for path in submit_files:
        print(f"  {path}")

    if not args.submit:
        print("\nDry-run (no --submit). Re-run with --submit to submit to Condor.")
        return

    for path in submit_files:
        result = _condor_submit(path)
        print(f"\n=== condor_submit {path.name} ===")
        print(result.stdout.strip())
        if result.returncode != 0:
            print(result.stderr.strip())
            raise SystemExit(f"condor_submit failed for {path}")


if __name__ == "__main__":
    main()
