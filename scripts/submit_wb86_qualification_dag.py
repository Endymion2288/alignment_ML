#!/usr/bin/env python3
"""Submit the WB86 physical qualification DAG to CERN Condor."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from alignment.wb86_physical_qualification import (
    OUTPUT_ROOT,
    WB84_QUALIFIED_EXPORTS,
    write_prereq_or_fail,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER = PROJECT_ROOT / "scripts" / "run_wb86_qualification_condor.sh"


def _write_submit(
    target: Path,
    *,
    arguments: str,
    log_dir: Path,
    name: str,
    request_memory_mb: int,
    request_disk: int,
    job_flavour: str,
    queue: int = 1,
) -> None:
    lines = [
        "universe = vanilla",
        f"executable = {WORKER}",
        f"arguments = {arguments}",
        f"output = {log_dir}/{name}.$(ClusterId).$(ProcId).out",
        f"error = {log_dir}/{name}.$(ClusterId).$(ProcId).err",
        f"log = {log_dir}/{name}.$(ClusterId).log",
        f"request_memory = {int(request_memory_mb)}",
        "request_cpus = 1",
        f"request_disk = {int(request_disk)}",
        f'+JobFlavour = "{job_flavour}"',
        "getenv = True",
        "environment = \"CLUSTER_ID=$(ClusterId) PROC_ID=$(ProcId)\"",
        "max_retries = 1",
        # Athena/Calypso startup reads the EOS Calypso build.  Materializing
        # all 2394 procs at once livelocks write_station_alignment_payload.
        f"max_materialize = {40 if queue > 1 else 1}",
        f"queue {int(queue)}",
        "",
    ]
    target.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / OUTPUT_ROOT))
    parser.add_argument("--job-flavour", default="tomorrow")
    parser.add_argument("--request-memory-mb", type=int, default=8000)
    parser.add_argument("--request-disk", type=int, default=8000000)
    parser.add_argument("--n-events", type=int, default=WB84_QUALIFIED_EXPORTS)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_root).expanduser().resolve()
    write_prereq_or_fail(output)
    catalog = json.loads((output / "wb86_event_catalog.json").read_text(encoding="utf-8"))
    n_events = int(args.n_events)
    if n_events != int(catalog["n_events"]):
        raise SystemExit(f"n-events={n_events} does not match catalog {catalog['n_events']}")
    submit_dir = output / "condor"
    submit_dir.mkdir(parents=True, exist_ok=True)
    log_dir = submit_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = {
        "SNAPSHOT_VERIFY": ("snapshot", 1, None),
        "CORPUS_VERIFY": ("corpus", 1, None),
        "PHYSICAL_ALIGNMENT_SHARDS": ("event", n_events, "$(Process)"),
        "RESULT_INTEGRITY": ("integrity", 1, None),
        "GLOBAL_SUMMARY": ("summary", 1, None),
        "FROZEN_QUALIFICATION_GATE": ("gate", 1, None),
        "CLOSURE": ("closure", 1, None),
    }
    dag_lines = []
    for name, (stage, queue, extra) in jobs.items():
        submit_file = submit_dir / f"{name.lower()}.sub"
        arguments = f"{PROJECT_ROOT} {output} {stage}"
        if extra is not None:
            arguments = f"{arguments} {extra}"
        _write_submit(
            submit_file,
            arguments=arguments,
            log_dir=log_dir,
            name=name.lower(),
            request_memory_mb=args.request_memory_mb,
            request_disk=args.request_disk,
            job_flavour=args.job_flavour,
            queue=queue,
        )
        dag_lines.append(f"JOB {name} {submit_file}")
    dag_lines.extend(
        [
            "PARENT SNAPSHOT_VERIFY CHILD CORPUS_VERIFY",
            "PARENT CORPUS_VERIFY CHILD PHYSICAL_ALIGNMENT_SHARDS",
            "PARENT PHYSICAL_ALIGNMENT_SHARDS CHILD RESULT_INTEGRITY",
            "PARENT RESULT_INTEGRITY CHILD GLOBAL_SUMMARY",
            "PARENT GLOBAL_SUMMARY CHILD FROZEN_QUALIFICATION_GATE",
            "PARENT FROZEN_QUALIFICATION_GATE CHILD CLOSURE",
            "",
        ]
    )
    dag_path = submit_dir / "wb86.dag"
    dag_path.write_text("\n".join(dag_lines), encoding="utf-8")
    (submit_dir / "submit_manifest.json").write_text(
        json.dumps(
            {
                "schema": "wb86_qualification_dag_v1",
                "utc": datetime.now(timezone.utc).isoformat(),
                "dag": str(dag_path),
                "n_events": n_events,
                "job_flavour": args.job_flavour,
                "qualification_authorized": True,
                "executable": True,
                "ml_alignment_eval_authorized": False,
                "looked_at_physical_alignment_outcomes_before_authorization": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote DAG {dag_path} with {n_events} event jobs")
    if not args.submit:
        print("Dry-run (no --submit).")
        return
    command = [
        "bash",
        "-lc",
        "source /usr/share/Modules/init/bash "
        "&& module load lxbatch/eossubmit "
        "&& myschedd out "
        f"&& condor_submit_dag {shlex.quote(str(dag_path))}",
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit(result.returncode)
    (submit_dir / "condor_submit.log").write_text((result.stdout or "") + (result.stderr or ""), encoding="utf-8")


if __name__ == "__main__":
    main()
