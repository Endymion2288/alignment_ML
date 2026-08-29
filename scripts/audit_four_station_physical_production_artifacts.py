#!/usr/bin/env python3
"""Artifact-level audit of a four-station physical production bank.

Condor return code 0 is not accepted as production success.  Every expected
payload must exist, be ROOT-readable, match the frozen scan plan, have no
failure.json, and pass four-station / covariance content checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from datasets.root_loader import load_events
from scripts.build_physical_curriculum_corpus import _physical_point_completion
from scripts.merge_four_station_source_diversity_train_corpus import common_scan_contract
from scripts.prepare_multisource_multidof_iteration import iteration_split_mode
from scripts.run_refit_multidof_closure import _json_ready, _read_json


EXPECTED_PAYLOADS = (
    "iteration_00_reference",
    "iteration_00_hard_s3_ry",
    "iteration_00_hard_s3_ry_plus_common",
    "iteration_00_draw_00",
    "iteration_00_draw_00_plus_common",
    "iteration_00_draw_01",
    "iteration_00_draw_01_plus_common",
)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_root(path: Path, *, source_id: str, payload: str, label: str) -> dict[str, object]:
    events = load_events(path, require_mc_labels=True)
    if not events:
        raise ValueError(f"{source_id}/{payload} {label} has no events")
    stations = set()
    uids = []
    for event in events:
        stations.update(int(value) for value in event.station_id)
        uids.append(f"{source_id}:{event.run_id}:{event.event_id}")
    missing = [station for station in (0, 1, 2, 3) if station not in stations]
    if missing:
        raise ValueError(f"{source_id}/{payload} {label} missing stations {missing}")
    return {
        "events": int(len(events)),
        "uids": uids,
        "stations": sorted(stations),
        "path": str(path),
        "ok": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--reference-iteration-manifest", default=None)
    parser.add_argument("--expected-source-ids", action="append", default=None)
    parser.add_argument("--require-split-mode", default=None)
    args = parser.parse_args()
    iteration_path = Path(args.iteration_manifest).expanduser().resolve()
    iteration = _read_json(iteration_path)
    split_mode = iteration_split_mode(iteration, label="iteration manifest")
    if args.require_split_mode and split_mode != args.require_split_mode:
        raise SystemExit(f"iteration split mode {split_mode} != {args.require_split_mode}")
    plan = iteration.get("common_scan_plan")
    if not isinstance(plan, Mapping):
        raise SystemExit("iteration lacks common_scan_plan")
    names = [str(point.get("name")) for point in plan.get("points") or [] if isinstance(point, Mapping)]
    if names != list(EXPECTED_PAYLOADS):
        raise SystemExit(f"payload names drifted: {names}")
    sources = [dict(item) for item in iteration.get("sources") or [] if isinstance(item, Mapping)]
    source_ids = [str(item["source_id"]) for item in sources]
    if args.expected_source_ids and source_ids != list(args.expected_source_ids):
        raise SystemExit(f"source IDs drifted: {source_ids}")
    plan_ok = True
    if args.reference_iteration_manifest:
        reference = _read_json(Path(args.reference_iteration_manifest).expanduser().resolve())
        plan_ok = common_scan_contract(plan) == common_scan_contract(reference["common_scan_plan"])
        if not plan_ok:
            raise SystemExit("common_scan_plan does not match the reference workbook-53 bank")
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for source in sources:
        source_id = str(source["source_id"])
        root = Path(str(source["physical_scan_root"])).expanduser().resolve()
        source_plan = _read_json(root / "scan_plan.json")
        if common_scan_contract(source_plan) != common_scan_contract(plan):
            failures.append(f"{source_id}:scan_plan_mismatch")
        reference_uids = None
        for point in source_plan.get("points") or []:
            if not isinstance(point, Mapping):
                failures.append(f"{source_id}:invalid_point")
                continue
            name = str(point.get("name"))
            relative = str(point.get("relative_point_dir"))
            point_root = root / relative
            failure = point_root / "failure.json"
            tracklets = point_root / "refit" / "tracklets.root"
            propagations = point_root / "refit" / "propagations.root"
            payload_manifest = point_root / "payload" / "alignment_payload.json"
            content_audit = point_root / "refit" / "content_audit.json"
            completed, status = _physical_point_completion(
                tracklets=tracklets,
                propagations=propagations,
                payload_manifest=payload_manifest,
                content_audit=content_audit,
                failure=failure,
                station_ids=tuple(int(station) for station in plan["station_ids"]),
                expected_offsets_xy_mm=None,
                expected_station_transforms=dict(point.get("injected_station_transforms") or {}),
                expected_layer_transforms=(
                    dict(point["injected_layer_transforms"])
                    if isinstance(point.get("injected_layer_transforms"), Mapping)
                    else None
                ),
            )
            root_report = None
            try:
                root_report = _load_root(tracklets, source_id=source_id, payload=name, label="tracklets")
            except Exception as error:  # noqa: BLE001 - audit must record the concrete ROOT failure
                completed = False
                status = f"root_unreadable:{type(error).__name__}:{error}"
            if name == "iteration_00_reference" and root_report is not None:
                reference_uids = list(root_report["uids"])
                if not all(uid.startswith(f"{source_id}:") for uid in reference_uids):
                    completed = False
                    status = "source_uid_prefix_mismatch"
            payload = _read_json(payload_manifest) if payload_manifest.is_file() else {}
            audit = _read_json(content_audit) if content_audit.is_file() else {}
            covariance = audit.get("covariance") if isinstance(audit, Mapping) else {}
            row = {
                "source_id": source_id,
                "payload_id": name,
                "relative_point_dir": relative,
                "completed": bool(completed),
                "status": status,
                "failure_json_present": failure.is_file(),
                "input_xaod": str(source.get("input_xaod")),
                "payload_source_xaod": payload.get("input_xaod") or (payload.get("physical_refit_capture_scan") or {}).get("input_xaod"),
                "events": None if not isinstance(audit, Mapping) else audit.get("events"),
                "tracklets": None if not isinstance(audit, Mapping) else audit.get("tracklets"),
                "station_counts": None if not isinstance(audit, Mapping) else audit.get("station_counts"),
                "positive_definite_rows": None if not isinstance(covariance, Mapping) else covariance.get("positive_definite_rows"),
                "covariance_rows": None if not isinstance(covariance, Mapping) else covariance.get("rows"),
                "q_over_p_mode": source_plan.get("q_over_p_mode"),
                "root": None if root_report is None else {key: root_report[key] for key in ("events", "stations", "ok")},
                "n_reference_uids": None if reference_uids is None else len(reference_uids),
            }
            rows.append(row)
            if not completed:
                failures.append(f"{source_id}:{name}:{status}")
    decision = {
        "iteration_manifest": str(iteration_path),
        "split_mode": split_mode,
        "source_ids": source_ids,
        "n_sources": len(source_ids),
        "n_payloads": len(EXPECTED_PAYLOADS),
        "n_expected_points": len(source_ids) * len(EXPECTED_PAYLOADS),
        "n_audited_points": len(rows),
        "common_scan_plan_matches_reference": plan_ok,
        "payload_names": list(EXPECTED_PAYLOADS),
        "relative_curriculum": iteration.get("alignment_iteration", {}).get("relative_curriculum"),
        "q_over_p_mode": iteration.get("q_over_p_mode"),
        "test_data_accessed": False,
        "failures": failures,
        "passed": not failures and len(rows) == len(source_ids) * len(EXPECTED_PAYLOADS),
        "points": rows,
    }
    output = Path(args.output_json).expanduser().resolve()
    _write_json(output, decision)
    print(json.dumps(_json_ready({key: decision[key] for key in decision if key != "points"}), indent=2))
    if not decision["passed"]:
        raise SystemExit("physical production artifact audit failed")


if __name__ == "__main__":
    main()
