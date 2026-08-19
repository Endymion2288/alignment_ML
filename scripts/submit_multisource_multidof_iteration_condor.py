#!/usr/bin/env python3
"""Submit prepared source-disjoint multi-DoF iteration scans to Condor.

The source configurations are written before submission by
``prepare_multisource_multidof_iteration.py``.  This command only dispatches
those immutable configurations; the worker reruns the complete physical
conditions/refit/Acts chain for each selected original xAOD source.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scripts.build_physical_curriculum_corpus import _physical_point_completion
from scripts.submit_physical_curriculum_condor import (
    WORKER,
    _require_eos_path,
    _submit,
    _write_submit,
)


SCHEMA_VERSION = "faser-multisource-physical-alignment-iteration-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"not a multi-source alignment iteration manifest: {path}")
    if payload.get("physical_geometry_repropagation") is not True or payload.get("coordinate_surrogate") is not False:
        raise ValueError("iteration manifest lacks physical geometry contract")
    if int(payload.get("q_over_p_mode", -1)) != 0:
        raise ValueError("iteration manifest is not mode-0")
    if tuple(payload.get("allowed_splits", ())) != ("train", "validation") or "test" not in set(
        payload.get("forbidden_splits", ())
    ):
        raise ValueError("iteration manifest does not seal test data")
    if payload.get("test_data_accessed") is not False:
        raise ValueError("iteration manifest has an invalid test-access declaration")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("iteration manifest has no sources")
    return dict(payload)


def _complete_source(source_root: Path) -> bool:
    plan_path = source_root / "physical_scan" / "scan_plan.json"
    if not plan_path.is_file():
        return False
    with plan_path.open(encoding="utf-8") as handle:
        plan = json.load(handle)
    if not isinstance(plan, Mapping) or int(plan.get("q_over_p_mode", -1)) != 0:
        return False
    station_ids = plan.get("station_ids")
    points = plan.get("points")
    if not isinstance(station_ids, list) or not isinstance(points, list) or not points:
        return False
    for point in points:
        if not isinstance(point, Mapping):
            return False
        transforms = point.get("injected_station_transforms")
        relative = point.get("relative_point_dir")
        if not isinstance(transforms, Mapping) or not isinstance(relative, str):
            return False
        root = source_root / "physical_scan" / relative
        layer_transforms = point.get("injected_layer_transforms")
        completed, _ = _physical_point_completion(
            tracklets=root / "refit" / "tracklets.root",
            propagations=root / "refit" / "propagations.root",
            payload_manifest=root / "payload" / "alignment_payload.json",
            content_audit=root / "refit" / "content_audit.json",
            failure=root / "failure.json",
            station_ids=tuple(int(station) for station in station_ids),
            expected_offsets_xy_mm=None,
            expected_station_transforms=dict(transforms),
            expected_layer_transforms=(
                dict(layer_transforms) if isinstance(layer_transforms, Mapping) else None
            ),
        )
        if not completed:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-manifest", required=True)
    parser.add_argument("--submit-dir", required=True)
    parser.add_argument("--source-id", action="append", default=None)
    parser.add_argument("--split", choices=("train", "validation"), default=None)
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--request-memory-mb", type=int, default=6000)
    parser.add_argument("--job-flavour", default="tomorrow")
    parser.add_argument("--schedd-mode", choices=("eossubmit", "standard"), default="eossubmit")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.request_memory_mb < 1:
        parser.error("--request-memory-mb must be positive")
    manifest_path = Path(args.iteration_manifest).expanduser().resolve()
    manifest = _read_manifest(manifest_path)
    iteration_root = manifest_path.parent
    sources = [dict(item) for item in manifest["sources"] if isinstance(item, Mapping)]
    if len(sources) != len(manifest["sources"]):
        raise ValueError("iteration manifest has invalid source entries")
    known = {str(source.get("source_id", "")) for source in sources}
    requested = None if args.source_id is None else {str(value) for value in args.source_id}
    if requested is not None:
        unknown = requested - known
        if unknown:
            raise ValueError("unknown source ID(s): " + ", ".join(sorted(unknown)))
    selected = []
    for source in sources:
        source_id = str(source.get("source_id", ""))
        split = str(source.get("split", ""))
        if split not in {"train", "validation"}:
            raise ValueError(f"invalid source split for '{source_id}'")
        if requested is not None and source_id not in requested:
            continue
        if args.split is not None and split != args.split:
            continue
        config = Path(str(source.get("physical_scan_config", ""))).expanduser().resolve()
        if not config.is_file() or config.parent != iteration_root / "sources" / source_id:
            raise ValueError(f"source '{source_id}' lacks its immutable physical scan config")
        if args.skip_complete and _complete_source(config.parent):
            continue
        selected.append({"source_id": source_id, "split": split})
    if not selected:
        raise ValueError("no incomplete selected source remains")
    submit_dir = Path(args.submit_dir).expanduser().resolve()
    if submit_dir.exists() and any(submit_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty submit directory: {submit_dir}")
    log_root = submit_dir / "logs"
    log_root.mkdir(parents=True, exist_ok=False)
    source_ids_path = submit_dir / "source_ids.txt"
    source_ids_path.write_text("\n".join(item["source_id"] for item in selected) + "\n", encoding="utf-8")
    submit_file = submit_dir / "physical_multidof_iteration.sub"
    if args.schedd_mode == "eossubmit":
        for label, path in (
            ("worker", WORKER),
            ("iteration root", iteration_root),
            ("submit directory", submit_dir),
            ("source IDs", source_ids_path),
            ("log directory", log_root),
            ("submit file", submit_file),
        ):
            _require_eos_path(path, label=label)
    _write_submit(
        submit_file,
        output_root=iteration_root,
        source_ids_path=source_ids_path,
        log_root=log_root,
        request_memory_mb=int(args.request_memory_mb),
        job_flavour=str(args.job_flavour),
        schedd_mode=str(args.schedd_mode),
    )
    submission: dict[str, object] = {
        "schema_version": "faser-multisource-physical-alignment-iteration-condor-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "iteration_manifest": str(manifest_path),
        "iteration_root": str(iteration_root),
        "allowed_splits": ["train", "validation"],
        "forbidden_splits": ["test"],
        "test_data_accessed": False,
        "source_ids": [item["source_id"] for item in selected],
        "sources_by_split": {
            split: [item["source_id"] for item in selected if item["split"] == split]
            for split in ("train", "validation")
        },
        "physical_geometry_repropagation": True,
        "q_over_p_mode": 0,
        "submit_file": str(submit_file),
        "schedd_mode": str(args.schedd_mode),
        "submitted": False,
    }
    if args.submit:
        result = _submit(submit_file, schedd_mode=str(args.schedd_mode))
        submission["condor_submit_output"] = result.stdout
        submission["condor_submit_returncode"] = int(result.returncode)
        submission["submitted"] = result.returncode == 0
        if result.returncode:
            (submit_dir / "submission.json").write_text(
                json.dumps(submission, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            raise RuntimeError("condor_submit failed:\n" + result.stdout)
    (submit_dir / "submission.json").write_text(
        json.dumps(submission, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "submit_dir": str(submit_dir),
                "sources": len(selected),
                "submitted": bool(submission["submitted"]),
                "test_data_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
