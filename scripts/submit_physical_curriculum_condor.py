#!/usr/bin/env python3
"""Prepare or submit one Condor worker per bounded physical source.

Workers run the existing real-geometry scan driver, so each source creates a
fresh /Tracker/Align payload and reruns the SCT-cluster -> segment-refit ->
Acts chain at every configured magnitude.  They never update the shared corpus
manifest concurrently; after successful jobs, rerun the corpus builder with
``--prepare-only --resume`` to refresh provenance in one process.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scripts.build_physical_curriculum_corpus import (
    _allowed_splits,
    _load_config,
    _physical_point_completion,
    _validate_sources,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER = PROJECT_ROOT / "scripts" / "run_physical_curriculum_source.sh"


def _complete_source(source_root: Path) -> bool:
    plan_path = source_root / "physical_scan" / "scan_plan.json"
    if not plan_path.is_file():
        return False
    with plan_path.open(encoding="utf-8") as handle:
        plan = json.load(handle)
    points = plan.get("points") if isinstance(plan, Mapping) else None
    if not isinstance(points, list) or not points:
        return False
    for point in points:
        if not isinstance(point, Mapping):
            return False
        root = source_root / "physical_scan" / str(point.get("relative_point_dir", ""))
        station_ids = plan.get("station_ids")
        offsets = point.get("injected_offsets_xy_mm")
        transforms = point.get("injected_station_transforms")
        if not isinstance(station_ids, list) or (
            not isinstance(offsets, Mapping) and not isinstance(transforms, Mapping)
        ):
            return False
        accepted, _ = _physical_point_completion(
            tracklets=root / "refit" / "tracklets.root",
            propagations=root / "refit" / "propagations.root",
            payload_manifest=root / "payload" / "alignment_payload.json",
            content_audit=root / "refit" / "content_audit.json",
            failure=root / "failure.json",
            station_ids=tuple(int(station) for station in station_ids),
            expected_offsets_xy_mm=dict(offsets) if isinstance(offsets, Mapping) else None,
            expected_station_transforms=(
                dict(transforms) if isinstance(transforms, Mapping) else None
            ),
        )
        if not accepted:
            return False
    return True


def _write_submit(
    target: Path,
    *,
    output_root: Path,
    source_ids_path: Path,
    log_root: Path,
    request_memory_mb: int,
    job_flavour: str,
    schedd_mode: str,
) -> None:
    if schedd_mode == "eossubmit":
        # EOSSubmit requires all submit-file paths to live on EOS.  In
        # particular, use the EOS-resident worker directly instead of
        # ``/bin/bash`` and keep one scheduler log per cluster.
        executable = str(WORKER)
        arguments = f"{output_root} $(source_id) {PROJECT_ROOT}"
        output = f"{log_root}/$(source_id).$(ClusterId).$(ProcId).out"
        error = f"{log_root}/$(source_id).$(ClusterId).$(ProcId).err"
        log = f"{log_root}/physical_curriculum.$(ClusterId).log"
    elif schedd_mode == "standard":
        executable = "/bin/bash"
        arguments = f"{WORKER} {output_root} $(source_id) {PROJECT_ROOT}"
        output = f"{log_root}/$(source_id).out"
        error = f"{log_root}/$(source_id).err"
        log = f"{log_root}/$(source_id).log"
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
                f"queue source_id from {source_ids_path}",
                "",
            )
        ),
        encoding="utf-8",
    )


def _require_eos_path(path: Path, *, label: str) -> None:
    if not str(path).startswith("/eos/"):
        raise ValueError(f"EOSSubmit requires {label} to be under /eos: {path}")


def _submit(submit_file: Path, *, schedd_mode: str) -> subprocess.CompletedProcess[str]:
    if schedd_mode == "standard":
        command = ["condor_submit", str(submit_file)]
    elif schedd_mode == "eossubmit":
        # `myschedd out` chooses a user-owned EOSSubmit scheduler.  It must run
        # in the same shell as condor_submit so the scheduler environment is
        # retained.
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
    parser.add_argument("--physical-output-dir", required=True)
    parser.add_argument("--submit-dir", required=True)
    parser.add_argument("--source-id", action="append", default=None)
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--request-memory-mb", type=int, default=6000)
    parser.add_argument("--job-flavour", default="tomorrow")
    parser.add_argument(
        "--schedd-mode",
        choices=("eossubmit", "standard"),
        default="eossubmit",
        help="Use CERN EOSSubmit by default; standard is retained for non-EOS environments.",
    )
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.request_memory_mb < 1:
        parser.error("--request-memory-mb must be positive")
    if not WORKER.is_file():
        raise FileNotFoundError(WORKER)
    if not WORKER.stat().st_mode & 0o111:
        raise PermissionError(f"Condor worker must be executable: {WORKER}")
    config_path = Path(args.config).expanduser().resolve()
    config = _load_config(config_path)
    allowed_splits = _allowed_splits(config)
    if tuple(allowed_splits) != ("train", "validation"):
        raise ValueError("V3 Condor source expansion must allow exactly train and validation")
    if "test" not in {str(value) for value in config.get("forbidden_splits", ())}:
        raise ValueError("V3 Condor source expansion must explicitly forbid test")
    sources = _validate_sources(config, allowed_splits)
    known = {source["source_id"] for source in sources}
    if args.source_id is not None:
        requested = set(args.source_id)
        unknown = requested - known
        if unknown:
            raise ValueError("unknown configured source ID(s): " + ", ".join(sorted(unknown)))
        sources = [source for source in sources if source["source_id"] in requested]
    output_root = Path(args.physical_output_dir).expanduser().resolve()
    selected = [
        source
        for source in sources
        if not args.skip_complete or not _complete_source(output_root / "sources" / source["source_id"])
    ]
    if not selected:
        raise ValueError("no incomplete bounded source remains for Condor submission")
    submit_root = Path(args.submit_dir).expanduser().resolve()
    if submit_root.exists() and any(submit_root.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty Condor submit directory")
    log_root = submit_root / "logs"
    log_root.mkdir(parents=True, exist_ok=False)
    source_ids_path = submit_root / "source_ids.txt"
    source_ids_path.write_text(
        "\n".join(source["source_id"] for source in selected) + "\n", encoding="utf-8"
    )
    submit_file = submit_root / "physical_curriculum.sub"
    if args.schedd_mode == "eossubmit":
        for label, path in (
            ("worker", WORKER),
            ("physical output directory", output_root),
            ("submit directory", submit_root),
            ("source-id file", source_ids_path),
            ("log directory", log_root),
            ("submit file", submit_file),
        ):
            _require_eos_path(path, label=label)
    _write_submit(
        submit_file,
        output_root=output_root,
        source_ids_path=source_ids_path,
        log_root=log_root,
        request_memory_mb=args.request_memory_mb,
        job_flavour=args.job_flavour,
        schedd_mode=args.schedd_mode,
    )
    manifest: dict[str, Any] = {
        "schema_version": "faser-v3-physical-curriculum-condor-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "physical_output_dir": str(output_root),
        "allowed_splits": list(allowed_splits),
        "forbidden_splits": ["test"],
        "source_split_unit": "original_xaod_file",
        "worker": str(WORKER),
        "submit_file": str(submit_file),
        "schedd_mode": args.schedd_mode,
        "source_ids": [source["source_id"] for source in selected],
        "sources_by_split": {
            split: [source["source_id"] for source in selected if source["split"] == split]
            for split in allowed_splits
        },
        "physical_geometry_repropagation": True,
        "q_over_p_mode": 0,
        "shared_manifest_write_policy": "post_job_single_process_refresh_only",
        "submitted": False,
    }
    if args.submit:
        result = _submit(submit_file, schedd_mode=args.schedd_mode)
        manifest["condor_submit_output"] = result.stdout
        manifest["submitted"] = result.returncode == 0
        manifest["condor_submit_returncode"] = result.returncode
        if result.returncode:
            (submit_root / "submission.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            raise RuntimeError("condor_submit failed:\n" + result.stdout)
    (submit_root / "submission.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "submit_dir": str(submit_root),
                "submit_file": str(submit_file),
                "sources": len(selected),
                "submitted": bool(manifest["submitted"]),
                "condor_submit_output": manifest.get("condor_submit_output"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
