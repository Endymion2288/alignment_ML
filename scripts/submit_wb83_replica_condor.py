#!/usr/bin/env python3
"""Submit the WB83 independent-replica DAG to CERN Condor (CPU).

Freeze the geometry matrix first.  Do not run the 100-replica batch on an
interactive lxplus node.  8-CPU / 32-GB GPU leftovers are not required;
this is a small CPU numpy job (2 CPU / 4 GB, flavour tomorrow).
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER = PROJECT_ROOT / "scripts" / "run_wb83_replica_condor.sh"
SUMMARIZE = PROJECT_ROOT / "scripts" / "run_wb83_summarize_condor.sh"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/mc24_four_station_wb83_truth_only_replicas_v1"


def _write_submit(
    target: Path,
    *,
    executable: Path,
    arguments: str,
    log_dir: Path,
    name: str,
    request_memory_mb: int,
    request_cpus: int,
    job_flavour: str,
) -> None:
    target.write_text(
        "\n".join(
            (
                "universe = vanilla",
                f"executable = {executable}",
                f"arguments = {arguments}",
                f"output = {log_dir}/{name}.$(ClusterId).$(ProcId).out",
                f"error = {log_dir}/{name}.$(ClusterId).$(ProcId).err",
                f"log = {log_dir}/{name}.$(ClusterId).log",
                f"request_memory = {int(request_memory_mb)}",
                f"request_cpus = {int(request_cpus)}",
                "request_disk = 2000000",
                f'+JobFlavour = "{job_flavour}"',
                "getenv = True",
                "queue",
                "",
            )
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--submit-dir", default="")
    parser.add_argument("--job-flavour", default="tomorrow")
    parser.add_argument("--request-memory-mb", type=int, default=4000)
    parser.add_argument("--request-cpus", type=int, default=2)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_root).expanduser().resolve()
    matrix_path = output / "geometry_qualification_matrix.json"
    if not matrix_path.exists():
        raise SystemExit(f"freeze the matrix first: {matrix_path} is absent")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    submit_dir = Path(args.submit_dir).expanduser().resolve() if args.submit_dir else output / "condor"
    submit_dir.mkdir(parents=True, exist_ok=True)
    log_dir = submit_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    dag_lines = []
    worker_names = []
    for cell in matrix["cells"]:
        cell_id = str(cell["cell_id"])
        name = cell_id.replace(".", "p")
        submit_file = submit_dir / f"{name}.sub"
        _write_submit(
            submit_file,
            executable=WORKER,
            arguments=f"{PROJECT_ROOT} {output} {cell_id} 0 {int(matrix['n_target_replicas'])}",
            log_dir=log_dir,
            name=name,
            request_memory_mb=args.request_memory_mb,
            request_cpus=args.request_cpus,
            job_flavour=args.job_flavour,
        )
        dag_lines.append(f"JOB {name} {submit_file}")
        worker_names.append(name)
    summary_sub = submit_dir / "summarize.sub"
    _write_submit(
        summary_sub,
        executable=SUMMARIZE,
        arguments=f"{PROJECT_ROOT} {output}",
        log_dir=log_dir,
        name="summarize",
        request_memory_mb=args.request_memory_mb,
        request_cpus=1,
        job_flavour=args.job_flavour,
    )
    dag_lines.append(f"JOB summarize {summary_sub}")
    dag_lines.append("PARENT " + " ".join(worker_names) + " CHILD summarize")
    dag_path = submit_dir / "wb83.dag"
    dag_path.write_text("\n".join(dag_lines) + "\n", encoding="utf-8")
    print(f"Wrote DAG {dag_path} with {len(worker_names)} replica jobs")
    if not args.submit:
        print("Dry-run (no --submit). Re-run with --submit after freeze.")
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
    print(result.stderr)
    if result.returncode != 0:
        raise SystemExit("condor_submit_dag failed")


if __name__ == "__main__":
    main()
