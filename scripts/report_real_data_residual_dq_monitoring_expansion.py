#!/usr/bin/env python3
"""Write expansion residual/DQ monitoring reports.

Reuses the frozen 14973/14974 reference scale and entry-52 alarm
priority.  Does not generate FD probes, run Newton, or write a payload.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from alignment.real_data_residual_dq_monitoring import (
    assemble_monitored_run,
    classify_monitoring_campaign,
    summarize_monitoring_run,
)
from alignment.real_data_residual_dq_monitoring_expansion import (
    SCHEMA_VERSION,
    attach_consecutive_drift_in_time,
    detect_adjacent_alignment_drift,
    frozen_alarm_thresholds,
    interpret_expansion,
    lhc_fill_from_occupancy_root,
    load_expansion_config,
    load_frozen_calibration_reference,
    time_stability_in_real_time,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("efficiency", "purity", "fake", "auc", "average_precision", "ap", "roc")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON mapping: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _walk_forbidden(payload: object, *, where: str) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            lowered = str(key).lower()
            parts = set(lowered.split("_"))
            if any(token == lowered or token in parts for token in FORBIDDEN):
                raise ValueError(f"{where} contains forbidden truth-metric key {key!r}")
            _walk_forbidden(value, where=where)
    elif isinstance(payload, list):
        for item in payload:
            _walk_forbidden(item, where=where)


def _jsonable(payload: object) -> object:
    if isinstance(payload, Mapping):
        return {str(key): _jsonable(value) for key, value in payload.items() if not isinstance(value, np.ndarray)}
    if isinstance(payload, list):
        return [_jsonable(item) for item in payload]
    if isinstance(payload, (np.floating, np.integer)):
        return payload.item()
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    return rows


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle) if any(str(value).strip() for value in row.values())]


def _n_all_pairs(field_candidates: Path) -> tuple[int, int]:
    if not field_candidates.is_file():
        return 0, 0
    import uproot

    with uproot.open(field_candidates) as source:
        tree = source["propagations"]
        arrays = tree.arrays(["run_id", "event_id"], library="np")
        n_candidates = int(np.asarray(arrays["run_id"]).size)
    return n_candidates, n_candidates


def _block_from_association(
    *,
    run: int,
    role: str,
    source_id: str,
    n_events: int,
    n_tracklets: int,
    n_all_pairs: int,
    association_dir: Path,
    lhc_fill: int | None,
    skip_events: int | None,
    reference: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    routes = _load_jsonl(association_dir / "selected_routes.jsonl")
    field_edges = _load_csv(association_dir / "selected_route_field_edge_residuals.csv")
    summary = summarize_monitoring_run(
        run=run,
        role=role,
        source_id=source_id,
        n_events=n_events,
        n_tracklets=n_tracklets,
        n_all_pairs_candidates=n_all_pairs,
        routes=routes,
        field_edges=field_edges,
    )
    summary["lhc_fill"] = lhc_fill
    summary["skip_events"] = skip_events
    summary["expansion_run"] = role == "monitoring" and run not in {14973, 14974, 14975, 14976, 14977}
    return assemble_monitored_run(summary, reference, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "operating_protocol_v1_real_data_residual_dq_monitoring_expansion_v1.yaml"),
    )
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--frozen-windows", default="")
    parser.add_argument("--athena-root", default="")
    args = parser.parse_args()
    config = load_expansion_config(args.config)
    campaign_root = Path(args.campaign_root).expanduser().resolve()
    campaign_root.mkdir(parents=True, exist_ok=True)
    reference = load_frozen_calibration_reference(PROJECT_ROOT / str(config["parent_run_level_report"]))
    parent_report = _read_json(PROJECT_ROOT / str(config["parent_run_level_report"]))
    route_dq = _read_json(PROJECT_ROOT / str(config["self_nulling_root"]) / "route_dq_fullscale_report.json")
    route_dq_by_run = {int(block["run"]): block for block in route_dq.get("blocks") or []}
    parent_by_run = {int(block["run"]): block for block in parent_report.get("blocks") or []}
    scaling_assoc = PROJECT_ROOT / str(config["scaling_root"]) / "association"
    frozen_windows = {}
    if args.frozen_windows:
        frozen_windows = _read_json(Path(args.frozen_windows).expanduser().resolve())
    elif (campaign_root / "occupancy" / "frozen_windows.json").is_file():
        frozen_windows = _read_json(campaign_root / "occupancy" / "frozen_windows.json")
    athena_root = Path(args.athena_root).expanduser().resolve() if args.athena_root else campaign_root / "athena"
    created = datetime.now(timezone.utc).isoformat()
    criteria = frozen_alarm_thresholds(config)
    blocks: list[dict[str, Any]] = []

    for run_key, meta in config["parent_full_segment_sources"].items():
        run = int(run_key)
        parent = parent_by_run[run]
        prior = route_dq_by_run[run]
        occupancy_json = None
        fill = parent.get("lhc_fill")
        skip = None
        if run == 14973:
            occupancy_json = PROJECT_ROOT / "outputs/operating_protocol_v1_real_data_occupancy_preflight_v1/runs/14973/00007/occupancy.root"
            skip = 49500
        elif run == 14974:
            occupancy_json = PROJECT_ROOT / "outputs/operating_protocol_v1_real_data_occupancy_preflight_v1/runs/14974/00005/occupancy.root"
            skip = 74400
        elif run == 14975:
            occupancy_json = PROJECT_ROOT / "outputs/operating_protocol_v1_real_data_occupancy_preflight_v1/runs/14975/00005/occupancy.root"
            skip = 21600
        elif run == 14976:
            occupancy_json = PROJECT_ROOT / "outputs/operating_protocol_v1_real_data_occupancy_preflight_v1/runs/14976/00004/occupancy.root"
            skip = 95500
        elif run == 14977:
            occupancy_json = PROJECT_ROOT / "outputs/operating_protocol_v1_real_data_occupancy_preflight_v1/runs/14977/00005/occupancy.root"
            skip = 135900
        if fill is None and occupancy_json is not None:
            fill = lhc_fill_from_occupancy_root(occupancy_json)
        block = _block_from_association(
            run=run,
            role=str(meta["role"]),
            source_id=str(meta["source_id"]),
            n_events=int(meta["nevents"]),
            n_tracklets=int(prior.get("n_tracklets") or parent.get("n_tracklets") or 0),
            n_all_pairs=int(prior.get("n_all_pairs_candidates") or parent.get("n_all_pairs_candidates") or 0),
            association_dir=scaling_assoc / str(meta["source_id"]),
            lhc_fill=fill,
            skip_events=skip,
            reference=reference,
            config=config,
        )
        blocks.append(block)

    expansion_blocks: list[dict[str, Any]] = []
    expansion_rows = (frozen_windows.get("runs") or {}) if frozen_windows else {}
    identity_root = athena_root / "identity"
    for run_key, row in expansion_rows.items():
        if row.get("window_accepted") is not True:
            continue
        run = int(run_key)
        source_id = None
        iteration = athena_root / "iteration_manifest.json"
        if iteration.is_file():
            for source in _read_json(iteration).get("sources") or []:
                if int(source["run"]) == run:
                    source_id = str(source["source_id"])
                    n_events = int(source["nevents"])
                    break
        if source_id is None:
            continue
        association_dir = athena_root / "association" / source_id
        if not (association_dir / "selected_routes.jsonl").is_file():
            continue
        manifest = identity_root / "manifests" / f"{source_id}.json"
        n_tracklets = 0
        n_all_pairs = 0
        if manifest.is_file():
            raw = _read_json(manifest)
            samples = raw.get("samples") or []
            if samples:
                n_all_pairs, _ = _n_all_pairs(Path(str(samples[0].get("field_candidates") or "")))
                tracklets = Path(str(samples[0].get("physical_tracklets") or ""))
                if tracklets.is_file():
                    import uproot

                    with uproot.open(tracklets) as source:
                        n_tracklets = int(source["tracklets"].num_entries)
        fill = row.get("lhc_fill")
        if fill is None:
            fill = lhc_fill_from_occupancy_root(Path(str(row.get("occupancy_root") or "")))
        block = _block_from_association(
            run=run,
            role="monitoring",
            source_id=source_id,
            n_events=n_events,
            n_tracklets=n_tracklets,
            n_all_pairs=n_all_pairs,
            association_dir=association_dir,
            lhc_fill=None if fill is None else int(fill),
            skip_events=int(row["skip_events"]),
            reference=reference,
            config=config,
        )
        expansion_blocks.append(block)
        blocks.append(block)

    blocks = attach_consecutive_drift_in_time(blocks)
    drift = detect_adjacent_alignment_drift(
        blocks,
        robust_z_drift=float(config["robust_z_drift"]),
        min_adjacent_runs=int(config["min_nonreference_runs_for_repeatable_drift"]),
    )
    stability = time_stability_in_real_time(blocks)
    classification = classify_monitoring_campaign(blocks, drift)
    interpretation = interpret_expansion(expansion_blocks, drift)
    if interpretation["reading"] == "independent_survey_or_external_alignment_follow_up":
        classification["reasons"] = list(classification.get("reasons") or []) + [
            "consecutive_detector_condition_or_alignment_drift_requires_external_follow_up"
        ]
    classification["expansion_interpretation"] = interpretation

    provenance = {
        "schema_version": SCHEMA_VERSION + "-provenance",
        "created_utc": created,
        "parent_decision": config["parent_decision"],
        "frozen_v2_checkpoint_sha256": config["frozen_v2_checkpoint_sha256"],
        "residual_blind_occupancy": True,
        "current_geometry_only": True,
        "fd_probes_generated": False,
        "newton_run": False,
        "do_not_reestimate_reference_scale": True,
        "do_not_reestimate_alarm_thresholds": True,
        "frozen_windows": frozen_windows if frozen_windows else None,
        "geometry_write_allowed": False,
        "station_calibration_mode_available": False,
        "cdx_mode_allowed": False,
    }
    run_level = {
        "schema_version": SCHEMA_VERSION + "-run-level",
        "created_utc": created,
        "frozen_v2_checkpoint_sha256": config["frozen_v2_checkpoint_sha256"],
        "calibration_reference": reference,
        "dq_alarm_criteria": criteria,
        "current_geometry_only": True,
        "fd_probes_generated": False,
        "newton_run": False,
        "geometry_write_allowed": False,
        "station_calibration_mode_available": False,
        "cdx_mode_allowed": False,
        "residual_reduction_is_not_alignment_success": True,
        "blocks": blocks,
    }
    time_report = {
        "schema_version": SCHEMA_VERSION + "-time-stability",
        "created_utc": created,
        "calibration_reference_runs": [14973, 14974],
        **stability,
    }
    alarm = {
        "schema_version": SCHEMA_VERSION + "-alarm-summary",
        "created_utc": created,
        "dq_alarm_criteria": criteria,
        "alignment_drift_candidate": drift,
        "expansion_interpretation": interpretation,
        "run_statuses": {str(block["run"]): block["status"] for block in blocks},
        "geometry_write_allowed": False,
        "station_calibration_mode_available": False,
        "cdx_mode_allowed": False,
        "cannot_convert_to_geometry_correction": True,
    }
    decision = {
        "schema_version": SCHEMA_VERSION + "-monitoring-decision",
        "created_utc": created,
        **classification,
        "dq_alarm_criteria": criteria,
        "official_conditions_db_modified": False,
        "sealed_test_opened": False,
        "retrain_v2": False,
        "conditions_writing_rehearsal_allowed": False,
        "physics_production_validation_allowed": False,
        "next_allowed_step": (
            "Keep real_data_residual_dq_monitoring_only.  "
            "Do not construct a Station calibration mode.  "
            "Do not start C_dx Mode.  "
            + (
                "Consecutive detector-condition or alignment-drift candidates "
                "are an independent survey/external alignment follow-up, not a "
                "self-nulling reopen."
                if interpretation["reading"] == "independent_survey_or_external_alignment_follow_up"
                else "Official current geometry plus frozen V2 remains the "
                "long-term DQ monitoring backbone."
            )
        ),
        "blocks": [
            {
                "run": int(block["run"]),
                "role": block["role"],
                "status": block["status"],
                "selected_routes": block["selected_routes"],
                "lhc_fill": block.get("lhc_fill"),
                "expansion_run": block.get("expansion_run"),
            }
            for block in blocks
        ],
    }

    outputs = (
        (provenance, "expansion_provenance.json"),
        (run_level, "run_level_dq_report.json"),
        (time_report, "time_stability_report.json"),
        (alarm, "alarm_summary.json"),
        (decision, "operating_protocol_monitoring_decision.json"),
    )
    for payload, name in outputs:
        clean = _jsonable(payload)
        _walk_forbidden(clean, where=name)
        _write_json(campaign_root / name, clean)  # type: ignore[arg-type]
    print(
        json.dumps(
            {
                "campaign_root": str(campaign_root),
                "decision": classification["decision"],
                "geometry_write_allowed": False,
                "station_calibration_mode_available": False,
                "cdx_mode_allowed": False,
                "alignment_drift_candidate": classification["alignment_drift_candidate"],
                "expansion_interpretation": interpretation["reading"],
                "run_statuses": {str(block["run"]): block["status"] for block in blocks},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
