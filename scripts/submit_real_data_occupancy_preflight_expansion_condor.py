#!/usr/bin/env python3
"""Submit residual-blind occupancy preflight for monitoring expansion runs.

One Condor job per (run, segment).  Jobs scan official current geometry with
no alignment overlay and do not apply the frozen window rule.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alignment.real_data_occupancy_preflight import format_segment
from alignment.real_data_residual_dq_monitoring_expansion import (
    EXPANSION_RUNS,
    first_wave_segments,
    load_expansion_config,
)
from scripts.submit_physical_curriculum_condor import PROJECT_ROOT, _require_eos_path, _submit


WORKER = PROJECT_ROOT / "scripts" / "run_real_data_occupancy_preflight_expansion.sh"


def _write_submit(
    target: Path,
    *,
    output_root: Path,
    jobs_path: Path,
    log_root: Path,
    request_memory_mb: int,
    job_flavour: str,
    schedd_mode: str,
) -> None:
    if schedd_mode == "eossubmit":
        executable = str(WORKER)
        arguments = f"{output_root} $(run) $(segment) {PROJECT_ROOT}"
        output = f"{log_root}/$(run)_$(segment).$(ClusterId).$(ProcId).out"
        error = f"{log_root}/$(run)_$(segment).$(ClusterId).$(ProcId).err"
        log = f"{log_root}/occupancy_preflight.$(ClusterId).log"
    elif schedd_mode == "standard":
        executable = "/bin/bash"
        arguments = f"{WORKER} {output_root} $(run) $(segment) {PROJECT_ROOT}"
        output = f"{log_root}/$(run)_$(segment).out"
        error = f"{log_root}/$(run)_$(segment).err"
        log = f"{log_root}/$(run)_$(segment).log"
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
                f"queue run,segment from {jobs_path}",
                "",
            )
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--submit-dir", required=True)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "operating_protocol_v1_real_data_residual_dq_monitoring_expansion_v1.yaml"))
    parser.add_argument("--segment", action="append", default=None)
    parser.add_argument("--run", action="append", type=int, default=None)
    parser.add_argument("--request-memory-mb", type=int, default=6000)
    parser.add_argument("--job-flavour", default="tomorrow")
    parser.add_argument("--schedd-mode", choices=("eossubmit", "standard"), default="eossubmit")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    config = load_expansion_config(args.config)
    if args.request_memory_mb < 1:
        parser.error("--request-memory-mb must be positive")
    if not WORKER.is_file():
        raise FileNotFoundError(WORKER)
    if not WORKER.stat().st_mode & 0o111:
        raise PermissionError(f"Condor worker must be executable: {WORKER}")
    runs = list(args.run) if args.run else list(EXPANSION_RUNS)
    unknown = [run for run in runs if run not in EXPANSION_RUNS]
    if unknown:
        raise ValueError(f"runs outside the frozen expansion list: {unknown}")
    segments = [format_segment(item) for item in args.segment] if args.segment else first_wave_segments(config)
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    submit_root = Path(args.submit_dir).expanduser().resolve()
    if submit_root.exists() and any(submit_root.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty Condor submit directory")
    log_root = submit_root / "logs"
    log_root.mkdir(parents=True, exist_ok=False)
    jobs_path = submit_root / "jobs.txt"
    jobs_path.write_text(
        "\n".join(f"{run} {segment}" for run in runs for segment in segments) + "\n",
        encoding="utf-8",
    )
    submit_file = submit_root / "occupancy_preflight.sub"
    if args.schedd_mode == "eossubmit":
        for label, path in (
            ("worker", WORKER),
            ("occupancy output directory", output_root),
            ("submit directory", submit_root),
            ("job file", jobs_path),
            ("log directory", log_root),
            ("submit file", submit_file),
        ):
            _require_eos_path(path, label=label)
    _write_submit(
        submit_file,
        output_root=output_root,
        jobs_path=jobs_path,
        log_root=log_root,
        request_memory_mb=args.request_memory_mb,
        job_flavour=args.job_flavour,
        schedd_mode=args.schedd_mode,
    )
    manifest: dict[str, Any] = {
        "schema_version": "faser-operating-protocol-v1-real-data-occupancy-expansion-condor",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "residual_blind": True,
        "alignment_perturbation": False,
        "v2_scoring": False,
        "role": "monitoring",
        "window_selection_not_applied": True,
        "output_dir": str(output_root),
        "segments": segments,
        "runs": runs,
        "worker": str(WORKER),
        "submit_file": str(submit_file),
        "schedd_mode": args.schedd_mode,
        "submitted": False,
        "geometry_write_allowed": False,
        "station_calibration_mode_available": False,
        "cdx_mode_allowed": False,
    }
    if args.submit:
        result = _submit(submit_file, schedd_mode=args.schedd_mode)
        manifest["condor_submit_output"] = result.stdout
        manifest["submitted"] = result.returncode == 0
        manifest["condor_submit_returncode"] = result.returncode
        (submit_root / "submission.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if result.returncode:
            raise RuntimeError("condor_submit failed:\n" + result.stdout)
    else:
        (submit_root / "submission.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "submit_dir": str(submit_root),
                "submit_file": str(submit_file),
                "runs": runs,
                "segments": segments,
                "n_jobs": len(runs) * len(segments),
                "submitted": bool(manifest["submitted"]),
                "condor_submit_output": manifest.get("condor_submit_output"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
