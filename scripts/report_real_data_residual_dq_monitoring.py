#!/usr/bin/env python3
"""Write current-geometry residual/DQ monitoring reports.

Reuses frozen-V2 selected routes.  Does not generate FD probes, run
Newton, write a payload, construct a Station calibration mode, or
start C_dx Mode.
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
    SCHEMA_VERSION,
    assemble_monitored_run,
    attach_consecutive_drift,
    build_calibration_reference,
    classify_monitoring_campaign,
    detect_alignment_drift_candidates,
    dq_alarm_criteria,
    load_monitoring_config,
    summarize_monitoring_run,
    time_stability_series,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "operating_protocol_v1_real_data_residual_dq_monitoring_v1.yaml"),
    )
    parser.add_argument("--campaign-root", required=True)
    args = parser.parse_args()
    config = load_monitoring_config(args.config)
    campaign_root = Path(args.campaign_root).expanduser().resolve()
    campaign_root.mkdir(parents=True, exist_ok=True)
    scaling_root = PROJECT_ROOT / str(config["scaling_root"])
    self_nulling_root = PROJECT_ROOT / str(config["self_nulling_root"])
    route_dq = _read_json(self_nulling_root / "route_dq_fullscale_report.json")
    route_dq_by_run = {int(block["run"]): block for block in route_dq.get("blocks") or []}
    association_root = scaling_root / "association"
    created = datetime.now(timezone.utc).isoformat()
    reference_runs = [int(run) for run in config["calibration_reference_runs"]]
    residual_rows_by_run: dict[int, list[dict[str, Any]]] = {}
    utilities_by_run: dict[int, list[float]] = {}
    summaries: list[dict[str, Any]] = []

    for run_key, meta in config["full_segment_sources"].items():
        run = int(run_key)
        role = str(config["blind_roles"][str(run)])
        source_id = str(meta["source_id"])
        association_dir = association_root / source_id
        routes = _load_jsonl(association_dir / "selected_routes.jsonl")
        field_edges = _load_csv(association_dir / "selected_route_field_edge_residuals.csv")
        residual_rows_by_run[run] = field_edges
        utilities_by_run[run] = [
            float(row["utility"]) for row in routes if row.get("utility") not in (None, "")
        ]
        prior = route_dq_by_run[run]
        summaries.append(
            summarize_monitoring_run(
                run=run,
                role=role,
                source_id=source_id,
                n_events=int(meta["nevents"]),
                n_tracklets=int(prior.get("n_tracklets") or 0),
                n_all_pairs_candidates=int(prior.get("n_all_pairs_candidates") or 0),
                routes=routes,
                field_edges=field_edges,
            )
        )

    reference = build_calibration_reference(
        residual_rows_by_run,
        utilities_by_run,
        reference_runs=reference_runs,
    )
    blocks = [assemble_monitored_run(summary, reference, config) for summary in summaries]
    blocks = attach_consecutive_drift(blocks)
    drift = detect_alignment_drift_candidates(
        blocks,
        reference_runs=reference_runs,
        robust_z_drift=float(config["robust_z_drift"]),
        min_nonreference_runs=int(config["min_nonreference_runs_for_repeatable_drift"]),
    )
    stability = time_stability_series(blocks)
    classification = classify_monitoring_campaign(blocks, drift)
    criteria = dq_alarm_criteria(config)

    run_level = {
        "schema_version": SCHEMA_VERSION + "-run-level",
        "created_utc": created,
        "frozen_v2_checkpoint_sha256": config["frozen_v2_checkpoint_sha256"],
        "current_geometry_only": True,
        "fd_probes_generated": False,
        "newton_run": False,
        "geometry_write_allowed": False,
        "station_calibration_mode_available": False,
        "cdx_mode_allowed": False,
        "residual_reduction_is_not_alignment_success": True,
        "calibration_reference": reference,
        "dq_alarm_criteria": criteria,
        "blocks": blocks,
    }
    time_report = {
        "schema_version": SCHEMA_VERSION + "-time-stability",
        "created_utc": created,
        "order": "run_number_as_time_proxy",
        "calibration_reference_runs": reference_runs,
        **stability,
    }
    drift_report = {
        "schema_version": SCHEMA_VERSION + "-alignment-drift-candidate",
        "created_utc": created,
        **drift,
        "dx_ry_cross_level_sensitive": True,
        "dz_track_driven_observable": False,
        "rz_direct_track_residual": False,
        "cannot_convert_to_geometry_correction": True,
    }
    decision = {
        "schema_version": SCHEMA_VERSION + "-monitoring-decision",
        "created_utc": created,
        **classification,
        "dq_alarm_criteria": criteria,
        "dq_versus_alignment": {
            "selected_route_count": "association_dq_not_alignment_success",
            "route_composition": "association_dq_not_alignment_success",
            "event_concentration": "association_or_reconstruction_dq",
            "score_utility_quantiles": "association_dq",
            "isolation_residual_median_iqr_tails": "dq_observable_only",
            "robust_standardized_shift": "dq_versus_14973_14974_reference_not_a_correction",
            "alignment_drift_candidate": "tag_only_never_a_geometry_update",
            "residual_drop": "never_alignment_success",
        },
        "expansion_candidates_not_yet_associated": list(config.get("expansion_candidates_not_yet_associated") or []),
        "expansion_entry_requirements": {
            "residual_blind_occupancy_window": True,
            "current_geometry_athena_only": True,
            "frozen_v2_association": True,
            "do_not_retune_thresholds_or_penalty": True,
            "do_not_generate_fd_probes": True,
            "do_not_run_newton": True,
            "do_not_write_payload": True,
        },
        "official_conditions_db_modified": False,
        "sealed_test_opened": False,
        "retrain_v2": False,
        "same_data_stage_iteration": False,
        "conditions_writing_rehearsal_allowed": False,
        "physics_production_validation_allowed": False,
        "next_allowed_step": (
            "Keep real_data_residual_dq_monitoring_only.  "
            "Do not construct a Station calibration mode.  "
            "Do not start C_dx Mode.  "
            "Do not extract an alignment payload from self-nulling residuals.  "
            "Additional 2024 r0022 runs may enter only as current-geometry "
            "monitoring after residual-blind occupancy and frozen V2 association."
        ),
        "blocks": [
            {
                "run": int(block["run"]),
                "role": block["role"],
                "source_id": block["source_id"],
                "status": block["status"],
                "selected_routes": block["selected_routes"],
            }
            for block in blocks
        ],
    }

    outputs = (
        (run_level, "run_level_dq_report.json"),
        (time_report, "time_stability_report.json"),
        (drift_report, "alignment_drift_candidate_report.json"),
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
                "run_statuses": {
                    str(block["run"]): block["status"] for block in blocks
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
