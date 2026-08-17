#!/usr/bin/env python3
"""Submit one frozen physical refit scan to CERN Condor.

The worker invokes ``run_physical_refit_capture_scan.py`` unchanged.  Thus
every configured point still writes a real /Tracker/Align payload and executes
SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit -> NtupleDumper ->
mode-0 FaserActs propagation.  This helper exists for bounded closure or
diagnostic scans that are not a multi-source training corpus.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scripts.run_physical_refit_capture_scan import _build_plan, _load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER = PROJECT_ROOT / "scripts" / "run_physical_refit_scan_condor.sh"


def _require_eos_path(path: Path, *, label: str) -> None:
    if not str(path).startswith("/eos/"):
        raise ValueError(f"EOSSubmit requires {label} to be under /eos: {path}")


def _write_submit(
    target: Path,
    *,
    config: Path,
    output_dir: Path,
    log_dir: Path,
    request_memory_mb: int,
    job_flavour: str,
    schedd_mode: str,
) -> None:
    if schedd_mode == "eossubmit":
        executable = str(WORKER)
        arguments = f"{config} {output_dir} {PROJECT_ROOT}"
        output = f"{log_dir}/scan.$(ClusterId).$(ProcId).out"
        error = f"{log_dir}/scan.$(ClusterId).$(ProcId).err"
        log = f"{log_dir}/scan.$(ClusterId).log"
    elif schedd_mode == "standard":
        executable = "/bin/bash"
        arguments = f"{WORKER} {config} {output_dir} {PROJECT_ROOT}"
        output = f"{log_dir}/scan.out"
        error = f"{log_dir}/scan.err"
        log = f"{log_dir}/scan.log"
    else:
        raise ValueError(f"unsupported schedd mode: {schedd_mode}")
    target.write_text(
        "\n".join(
            (
                "universe = vanilla",
                f"executable = {executable}",
                f"arguments = {arguments}",
                f"output = {output}",
                f"error = {error}",
                f"log = {log}",
                f"request_memory = {request_memory_mb}",
                "request_cpus = 1",
                f'+JobFlavour = "{job_flavour}"',
                "getenv = True",
                "queue 1",
                "",
            )
        ),
        encoding="utf-8",
    )


def _submit(submit_file: Path, *, schedd_mode: str) -> subprocess.CompletedProcess[str]:
    if schedd_mode == "standard":
        command = ["condor_submit", str(submit_file)]
    elif schedd_mode == "eossubmit":
        command = [
            "bash",
            "-lc",
            "source /usr/share/Modules/init/bash "
            "&& module load lxbatch/eossubmit "
            "&& myschedd out "
            f"&& condor_submit {shlex.quote(str(submit_file))}",
        ]
    else:
        raise ValueError(f"unsupported schedd mode: {schedd_mode}")
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--submit-dir", required=True)
    parser.add_argument("--request-memory-mb", type=int, default=6000)
    parser.add_argument("--job-flavour", default="tomorrow")
    parser.add_argument("--schedd-mode", choices=("eossubmit", "standard"), default="eossubmit")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.request_memory_mb < 1:
        parser.error("--request-memory-mb must be positive")
    if not WORKER.is_file():
        raise FileNotFoundError(WORKER)
    if not WORKER.stat().st_mode & 0o111:
        raise PermissionError(f"Condor worker must be executable: {WORKER}")

    config_path = Path(args.config).expanduser().resolve()
    scan_config = _load_config(config_path)
    plan = _build_plan(scan_config)
    if int(scan_config.get("q_over_p_mode", 0)) != 0:
        raise ValueError("physical scan V1 must use q_over_p_mode=0")
    output_dir = Path(args.output_dir).expanduser().resolve()
    submit_dir = Path(args.submit_dir).expanduser().resolve()
    if submit_dir.exists() and any(submit_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty Condor submit directory")
    log_dir = submit_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=False)
    if args.schedd_mode == "eossubmit":
        for label, path in (
            ("worker", WORKER),
            ("configuration", config_path),
            ("physical output directory", output_dir),
            ("submit directory", submit_dir),
            ("log directory", log_dir),
        ):
            _require_eos_path(path, label=label)
    submit_file = submit_dir / "physical_refit_scan.sub"
    _write_submit(
        submit_file,
        config=config_path,
        output_dir=output_dir,
        log_dir=log_dir,
        request_memory_mb=args.request_memory_mb,
        job_flavour=args.job_flavour,
        schedd_mode=args.schedd_mode,
    )
    manifest: dict[str, Any] = {
        "schema_version": "faser-physical-refit-scan-condor-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "output_dir": str(output_dir),
        "submit_file": str(submit_file),
        "worker": str(WORKER),
        "schedd_mode": args.schedd_mode,
        "scan_mode": str(plan["scan_mode"]),
        "points": [str(point["name"]) for point in plan["points"]],
        "physical_geometry_repropagation": True,
        "q_over_p_mode": 0,
        "submitted": False,
    }
    if args.submit:
        result = _submit(submit_file, schedd_mode=args.schedd_mode)
        manifest["condor_submit_output"] = result.stdout
        manifest["condor_submit_returncode"] = result.returncode
        manifest["submitted"] = result.returncode == 0
        if result.returncode:
            (submit_dir / "submission.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            raise RuntimeError("condor_submit failed:\n" + result.stdout)
    (submit_dir / "submission.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "submit_dir": str(submit_dir),
                "submit_file": str(submit_file),
                "points": len(plan["points"]),
                "submitted": bool(manifest["submitted"]),
                "condor_submit_output": manifest.get("condor_submit_output"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
