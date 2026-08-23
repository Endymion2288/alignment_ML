#!/usr/bin/env python3
"""Submit Operating Protocol V1 real-data Station Mode physical jobs.

Uses the same Calypso worker as MC, but completion requires the absence of
MC labels.  Does not write the official conditions database.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scripts.build_physical_curriculum_corpus import _physical_point_completion
from scripts.submit_physical_curriculum_condor import WORKER, PROJECT_ROOT, _require_eos_path, _submit, _write_submit


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
        transforms = point.get("injected_station_transforms")
        if not isinstance(station_ids, list) or not isinstance(transforms, Mapping):
            return False
        accepted, _ = _physical_point_completion(
            tracklets=root / "refit" / "tracklets.root",
            propagations=root / "refit" / "propagations.root",
            payload_manifest=root / "payload" / "alignment_payload.json",
            content_audit=root / "refit" / "content_audit.json",
            failure=root / "failure.json",
            station_ids=tuple(int(station) for station in station_ids),
            expected_offsets_xy_mm=None,
            expected_station_transforms=dict(transforms),
            require_mc_labels=False,
        )
        if not accepted:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-output-dir", required=True)
    parser.add_argument("--submit-dir", required=True)
    parser.add_argument("--source-id", action="append", default=None)
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--request-memory-mb", type=int, default=6000)
    parser.add_argument("--job-flavour", default="tomorrow")
    parser.add_argument("--schedd-mode", choices=("eossubmit", "standard"), default="eossubmit")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    output_root = Path(args.physical_output_dir).expanduser().resolve()
    manifest_path = output_root / "iteration_manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("is_mc") is not False:
        raise ValueError("real-data submitter refuses an MC iteration manifest")
    if manifest.get("official_conditions_db_write") is not False:
        raise ValueError("real-data submitter refuses a conditions-write campaign")
    sources = list(manifest["sources"])
    if args.source_id is not None:
        requested = set(args.source_id)
        known = {str(source["source_id"]) for source in sources}
        unknown = requested - known
        if unknown:
            raise ValueError("unknown source ID(s): " + ", ".join(sorted(unknown)))
        sources = [source for source in sources if str(source["source_id"]) in requested]
    selected = [
        source
        for source in sources
        if not args.skip_complete
        or not _complete_source(output_root / "sources" / str(source["source_id"]))
    ]
    if not selected:
        raise ValueError("no incomplete real-data source remains for Condor submission")
    submit_root = Path(args.submit_dir).expanduser().resolve()
    if submit_root.exists() and any(submit_root.iterdir()):
        raise FileExistsError("refusing to overwrite a non-empty Condor submit directory")
    log_root = submit_root / "logs"
    log_root.mkdir(parents=True, exist_ok=False)
    source_ids_path = submit_root / "source_ids.txt"
    source_ids_path.write_text(
        "\n".join(str(source["source_id"]) for source in selected) + "\n", encoding="utf-8"
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
    payload: dict[str, Any] = {
        "schema_version": "faser-operating-protocol-v1-real-data-condor",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "physical_output_dir": str(output_root),
        "official_conditions_db_write": False,
        "is_mc": False,
        "source_ids": [str(source["source_id"]) for source in selected],
        "submitted": False,
        "submit_file": str(submit_file),
        "schedd_mode": args.schedd_mode,
        "worker": str(WORKER),
    }
    if args.submit:
        result = _submit(submit_file, schedd_mode=args.schedd_mode)
        payload["condor_submit_output"] = result.stdout
        payload["submitted"] = result.returncode == 0
        payload["condor_submit_returncode"] = result.returncode
        (submit_root / "submission.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if result.returncode:
            raise RuntimeError("condor_submit failed:\n" + result.stdout)
    else:
        (submit_root / "submission.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({"submit_dir": str(submit_root), "sources": len(selected), "submitted": payload["submitted"]}, indent=2))


if __name__ == "__main__":
    main()
