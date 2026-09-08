#!/usr/bin/env python3
"""Submit the WB85b official qualification-runner smoke to CERN Condor."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER = PROJECT_ROOT / "scripts" / "run_wb85b_official_runner_smoke_condor.sh"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/mc24_four_station_wb85b_official_runner_smoke_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--job-flavour", default="tomorrow")
    parser.add_argument("--request-memory-mb", type=int, default=8000)
    parser.add_argument("--request-disk", type=int, default=8000000)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_root).expanduser().resolve()
    submit_dir = output / "condor"
    submit_dir.mkdir(parents=True, exist_ok=True)
    log_dir = submit_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    submit_file = submit_dir / "wb85b_official_runner_smoke.sub"
    submit_file.write_text(
        "\n".join(
            (
                "universe = vanilla",
                f"executable = {WORKER}",
                f"arguments = {PROJECT_ROOT} {output}",
                f"output = {log_dir}/wb85b_runner_smoke.$(ClusterId).$(ProcId).out",
                f"error = {log_dir}/wb85b_runner_smoke.$(ClusterId).$(ProcId).err",
                f"log = {log_dir}/wb85b_runner_smoke.$(ClusterId).log",
                f"request_memory = {int(args.request_memory_mb)}",
                "request_cpus = 1",
                f"request_disk = {int(args.request_disk)}",
                f'+JobFlavour = "{args.job_flavour}"',
                "getenv = True",
                "queue 1",
                "",
            )
        ),
        encoding="utf-8",
    )
    (submit_dir / "submit_manifest.json").write_text(
        json.dumps(
            {
                "schema": "wb85b_official_runner_smoke_condor_v1",
                "utc": datetime.now(timezone.utc).isoformat(),
                "submit_file": str(submit_file),
                "qualification_authorized": False,
                "wb86_automatically_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {submit_file}")
    if not args.submit:
        print("Dry-run (no --submit).")
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
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit(result.returncode)
    (submit_dir / "condor_submit.log").write_text((result.stdout or "") + (result.stderr or ""), encoding="utf-8")


if __name__ == "__main__":
    main()
