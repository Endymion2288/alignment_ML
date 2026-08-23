#!/usr/bin/env python3
"""Write full-segment Station Mode self-nulling DQ reports.

Uses existing frozen-V2 selected routes.  Residual numbers are DQ
observables only.  Official conditions are never written.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from alignment.real_data_operating_protocol import ROLE_CALIBRATION, ROLE_HELD_OUT_DQ
from alignment.real_data_station_mode_fullscale import (
    SCHEMA_VERSION,
    assign_next_decision,
    evaluate_frozen_selected_graph_gate,
    evaluate_fullscale_route_dq,
    leakage_report,
    load_frozen_contract,
    load_fullscale_config,
    summarize_residual_observables,
    summarize_selected_routes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("efficiency", "purity", "fake", "auc", "average_precision", "ap", "roc")
STATION_Z_MM = {0: -1860.15, 1: 47.4, 2: 1237.4, 3: 2427.4}


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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
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
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle) if any(str(value).strip() for value in row.values())]


def _physical_concentration(field_candidates: Path) -> dict[str, Any]:
    if not field_candidates.is_file():
        return {
            "n_all_pairs_candidates": 0,
            "physical_all_pairs_candidate_graph_nonempty": False,
            "physical_event_concentration": {
                "max_event_share": 0.0,
                "top2_event_share": 0.0,
                "n_events_with_candidates": 0,
            },
        }
    import uproot

    with uproot.open(field_candidates) as source:
        tree = source["propagations"]
        arrays = tree.arrays(["run_id", "event_id"], library="np")
        if isinstance(arrays, Mapping):
            run_ids = np.asarray(arrays["run_id"])
            event_ids = np.asarray(arrays["event_id"])
        else:
            structured = np.asarray(arrays)
            run_ids = np.asarray(structured["run_id"])
            event_ids = np.asarray(structured["event_id"])
    n_candidates = int(run_ids.size)
    keys = np.stack((run_ids.astype(np.int64), event_ids.astype(np.int64)), axis=1)
    _, counts = np.unique(keys, axis=0, return_counts=True)
    total = float(n_candidates)
    shares = sorted((float(count) / total for count in counts.tolist()), reverse=True) if total else []
    return {
        "n_all_pairs_candidates": n_candidates,
        "physical_all_pairs_candidate_graph_nonempty": n_candidates > 0,
        "physical_event_concentration": {
            "max_event_share": shares[0] if shares else 0.0,
            "top2_event_share": float(sum(shares[:2])),
            "n_events_with_candidates": int(counts.size),
        },
    }


def _dz_percentiles(field_edges: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    values: list[float] = []
    for row in field_edges:
        try:
            source = int(row["source_station_id"])
            target = int(row["target_station_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if source in STATION_Z_MM and target in STATION_Z_MM:
            values.append(float(STATION_Z_MM[target] - STATION_Z_MM[source]))
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"p16": None, "p50": None, "p84": None}
    return {
        "p16": float(np.percentile(array, 16)),
        "p50": float(np.percentile(array, 50)),
        "p84": float(np.percentile(array, 84)),
    }


def _condor_state(campaign_root: Path) -> dict[str, Any]:
    submission = campaign_root / "condor_calibration_fd" / "submission.json"
    if not submission.is_file():
        return {"fd_submitted": False, "submitted": False}
    payload = _read_json(submission)
    return {
        "fd_submitted": bool(payload.get("submitted")),
        "submitted": bool(payload.get("submitted")),
        "source_ids": list(payload.get("source_ids") or []),
        "condor_submit_output": payload.get("condor_submit_output"),
    }


def _newton_state(campaign_root: Path) -> dict[str, Any]:
    newton_root = campaign_root / "newton"
    if not newton_root.is_dir():
        return {"newton_complete": False, "station_dx_by_run": {}, "station_ry_by_run": {}, "capture_passed": None}
    dx: dict[int, float | None] = {}
    ry: dict[int, float | None] = {}
    capture: list[bool] = []
    for path in sorted(newton_root.glob("*/update_summary.json")) + sorted(
        newton_root.glob("*/route_selected_update.json")
    ):
        summary = _read_json(path)
        run = int(summary.get("run") or 0)
        if run <= 0:
            continue
        values = (
            summary.get("parameter_update")
            or summary.get("alignment_parameter_values")
            or summary.get("proposed_next_parameter_values")
            or {}
        )
        if isinstance(values, Mapping):
            dx[run] = None if values.get("ift_dx_mm") is None else float(values["ift_dx_mm"])
            ry[run] = None if values.get("ift_ry_mrad") is None else float(values["ift_ry_mrad"])
        raw_capture = summary.get("capture") or {}
        if raw_capture.get("applicable") is False:
            continue
        if "passed" in raw_capture:
            capture.append(bool(raw_capture.get("passed")))
    return {
        "newton_complete": bool(dx) and all(value is not None for value in dx.values()),
        "station_dx_by_run": dx,
        "station_ry_by_run": ry,
        "capture_passed": None if not capture else all(capture),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "operating_protocol_v1_real_data_station_mode_self_nulling_fullscale_v1.yaml"),
    )
    parser.add_argument("--campaign-root", required=True)
    args = parser.parse_args()
    config = load_fullscale_config(args.config)
    campaign_root = Path(args.campaign_root).expanduser().resolve()
    campaign_root.mkdir(parents=True, exist_ok=True)
    scaling_root = PROJECT_ROOT / str(config["scaling_root"])
    scaling_report = _read_json(PROJECT_ROOT / str(config["scaling_report"]))
    contract = load_frozen_contract(config)
    created = datetime.now(timezone.utc).isoformat()
    identity_root = scaling_root / "identity"
    association_root = scaling_root / "association"
    scale_rows = {
        (int(row["run"]), str(row["scale"])): row
        for row in scaling_report.get("blocks") or []
    }
    blocks: list[dict[str, Any]] = []
    gate_blocks: list[dict[str, Any]] = []
    for run_key, meta in config["full_segment_sources"].items():
        run = int(run_key)
        role = str(config["blind_roles"][str(run)])
        source_id = str(meta["source_id"])
        association_dir = association_root / source_id
        routes = _load_jsonl(association_dir / "selected_routes.jsonl")
        field_edges = _load_csv(association_dir / "selected_route_field_edge_residuals.csv")
        leave_one_out = _load_csv(association_dir / "selected_route_leave_one_out_residuals.csv")
        route_summary = summarize_selected_routes(routes)
        residual = summarize_residual_observables(field_edges, leave_one_out)
        manifest = identity_root / "manifests" / f"{source_id}.json"
        field_path = Path()
        if manifest.is_file():
            raw = _read_json(manifest)
            samples = raw.get("samples") or []
            if samples:
                field_path = Path(str(samples[0].get("field_candidates") or ""))
        physical = _physical_concentration(field_path)
        scale_row = scale_rows.get((run, "full")) or {}
        n_tracklets = int(scale_row.get("n_tracklets") or 0)
        block = {
            "source_id": source_id,
            "run": run,
            "role": role,
            "used_for_verdict": role == ROLE_CALIBRATION,
            "n_events": int(meta["nevents"]),
            "n_tracklets": n_tracklets,
            "n_all_pairs_candidates": physical["n_all_pairs_candidates"],
            "candidate_graph_nonempty": physical["physical_all_pairs_candidate_graph_nonempty"],
            "selected_edges": int(len(field_edges)),
            "selected_routes": route_summary["selected_routes"],
            "complete_four_station_routes": route_summary["complete_four_station_routes"],
            "truth_free_complete_route_fraction": route_summary["truth_free_complete_route_fraction"],
            "route_length_histogram": route_summary["route_length_histogram"],
            "edge_reuse": route_summary["edge_reuse"],
            "event_concentration": route_summary["event_concentration"],
            "route_multiplicity": route_summary["route_multiplicity"],
            "selected_utility_distribution": route_summary["selected_utility_distribution"],
            "residual_observables": residual,
            "values_finite": bool(residual["selected_field_edge"]["values_finite"]),
            "physical_event_concentration": physical["physical_event_concentration"],
            "geometry_write_allowed": False,
            "residual_reduction_is_not_alignment_success": True,
        }
        blocks.append(block)
        field = residual["selected_field_edge"]
        gate_blocks.append(
            {
                "source_id": source_id,
                "run": run,
                "role": role,
                "physical_all_pairs_candidate_graph_nonempty": physical["physical_all_pairs_candidate_graph_nonempty"],
                "selected_edges": int(len(field_edges)),
                "selected_routes": route_summary["selected_routes"],
                "values_finite": bool(residual["selected_field_edge"]["values_finite"]),
                "physical_event_concentration": physical["physical_event_concentration"],
                "per_event_route_distribution": {
                    "max_event_share_of_selected_routes": route_summary["event_concentration"][
                        "max_event_share_of_selected_routes"
                    ]
                },
                "residual_percentiles": {
                    "x_mm": field["residual_x_mm"],
                    "y_mm": field["residual_y_mm"],
                    "tx": field["residual_tx"],
                    "ty": field["residual_ty"],
                },
                "pull_percentiles": {"x": field["pull_x"], "y": field["pull_y"]},
                "chi2_percentiles": field["chi2"],
                "dz_percentiles": _dz_percentiles(field_edges),
            }
        )
    route_dq = evaluate_fullscale_route_dq(blocks, config)
    frozen_gate = evaluate_frozen_selected_graph_gate(gate_blocks, config)
    condor = _condor_state(campaign_root)
    newton = _newton_state(campaign_root)
    station_dx = {int(run): newton["station_dx_by_run"].get(int(run)) for run in (14973, 14974)}
    station_ry = {int(run): newton["station_ry_by_run"].get(int(run)) for run in (14973, 14974)}
    leakage = leakage_report(
        contract,
        station_dx_by_run=station_dx,
        station_ry_by_run=station_ry,
        independent_cdx_evidence=False,
    )
    next_decision = assign_next_decision(
        route_dq_passed=bool(route_dq["passed"] and frozen_gate["passed"]),
        fd_submitted=bool(condor["fd_submitted"]),
        newton_complete=bool(newton["newton_complete"]),
        capture_passed=newton["capture_passed"],
        implied_cdx_exceeds_operating_band=bool(leakage["exceeds_operating_band"]),
        independent_cdx_evidence=False,
        holdout_dq_worsened=False,
    )
    route_report = {
        "schema_version": SCHEMA_VERSION + "-route-dq",
        "created_utc": created,
        "scale": "full_segment_remaining",
        "frozen_v2": str(PROJECT_ROOT / str(config["frozen_v2"])),
        "frozen_v2_checkpoint_sha256": config["frozen_v2_checkpoint_sha256"],
        "do_not_change_thresholds": True,
        "mc_truth_used": False,
        "truth_metrics_omitted": True,
        "route_dq": route_dq,
        "frozen_candidate_graph_gate": frozen_gate,
        "blocks": blocks,
        "geometry_write_allowed": False,
        "residual_reduction_is_not_alignment_success": True,
        "note": (
            "truth_free_complete_route_fraction is the real-data stand-in for the "
            "requested route-quality audit.  MC route labels are absent."
        ),
    }
    leakage_out = {
        "schema_version": SCHEMA_VERSION + "-cdx-leakage",
        "created_utc": created,
        "cdx_fixed_by": "external_geometry",
        "cdx_mode_started": False,
        "cdx_mode_allowed": False,
        "not_a_C_dx_measurement": True,
        "independent_cdx_evidence": False,
        "newton_complete": bool(newton["newton_complete"]),
        **leakage,
        "note": (
            "Implied |C_dx| from frozen A is a contamination diagnostic.  "
            "It is not a C_dx measurement and does not start C_dx Mode.  "
            "Without a self-nulling Newton correction, station dx/ry are unavailable."
        ),
    }
    station_report = {
        "schema_version": SCHEMA_VERSION + "-station-mode",
        "created_utc": created,
        "decision": next_decision["decision"],
        "self_nulling_update": True,
        "station_mode_started": bool(condor["fd_submitted"] or newton["newton_complete"]),
        "calibration_only_geometry_estimation": True,
        "official_conditions_db_write": False,
        "geometry_write_allowed": False,
        "cdx_mode_allowed": False,
        "joint_station_cdx_newton": False,
        "new_layer_or_module_dof": False,
        "mc_truth_used": False,
        "route_dq_passed": bool(route_dq["passed"] and frozen_gate["passed"]),
        "frozen_candidate_graph_gate_passed": bool(frozen_gate["passed"]),
        "fd_submitted": bool(condor["fd_submitted"]),
        "newton_complete": bool(newton["newton_complete"]),
        "capture_passed": newton["capture_passed"],
        "mode_validity_status": "prerequisite_unmet",
        "A_stability": leakage["A_stability"],
        "residual_reduction_is_not_alignment_success": True,
        "dq_versus_alignment": {
            "route_dq_and_residual_observables": "data_quality_only",
            "self_nulling_parameter_update": "estimator_output_not_correctness",
            "residual_drop": "never_alignment_success",
            "implied_C_dx": "contamination_diagnostic_not_a_measurement",
        },
        "blocks": [
            {
                "source_id": block["source_id"],
                "run": block["run"],
                "role": block["role"],
                "selected_routes": block["selected_routes"],
                "complete_four_station_routes": block["complete_four_station_routes"],
                "used_for_verdict": block["used_for_verdict"],
                "geometry_write_allowed": False,
            }
            for block in blocks
        ],
    }
    decision = {
        "schema_version": SCHEMA_VERSION + "-next-decision",
        "created_utc": created,
        **next_decision,
        "official_conditions_db_modified": False,
        "sealed_test_opened": False,
        "retrain_v2": False,
        "reregister_capture_or_A": False,
        "same_data_stage_iteration": False,
        "conditions_writing_rehearsal_allowed": False,
        "physics_production_validation_allowed": False,
        "held_out_dq_used_for_verdict": any(
            block["role"] == ROLE_HELD_OUT_DQ and block["used_for_verdict"] for block in blocks
        ),
        "dq_versus_alignment": station_report["dq_versus_alignment"],
        "A_stability": leakage["A_stability"],
        "next_allowed_step": (
            "Run calibration-only Station Mode finite-difference capture on the "
            "full remaining segments, then self-nulling Newton.  Do not start "
            "C_dx Mode.  Do not write official conditions."
            if next_decision["decision"]
            in {"station_mode_self_nulling_pending", "station_mode_self_nulling_fd_submitted"}
            else "Keep geometry_write_allowed=false and do not start C_dx Mode."
        ),
        "blocks": [
            {"source_id": block["source_id"], "run": block["run"], "role": block["role"]}
            for block in blocks
        ],
    }
    for payload, name in (
        (route_report, "route_dq_fullscale_report.json"),
        (leakage_out, "cdx_leakage_diagnostic_report.json"),
        (station_report, "station_mode_self_nulling_report.json"),
        (decision, "operating_protocol_next_decision.json"),
    ):
        _walk_forbidden(payload, where=name)
        _write_json(campaign_root / name, payload)
    print(
        json.dumps(
            {
                "campaign_root": str(campaign_root),
                "decision": next_decision["decision"],
                "route_dq_passed": bool(route_dq["passed"] and frozen_gate["passed"]),
                "geometry_write_allowed": False,
                "held_out_dq_used_for_verdict": decision["held_out_dq_used_for_verdict"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
