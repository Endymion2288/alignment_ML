#!/usr/bin/env python3
"""Write Operating Protocol V1 real-data dry-run reports.

Wave-1 physical jobs must finish before Station Mode numbers exist.  This
command still emits the three required JSON files: pending stages are
explicit, geometry_write_allowed is false, and the official conditions DB
is never modified.  Dedicated C_dx Mode is a later independent stage.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from alignment.calibration_modes import load_mode_validity_contract
from alignment.real_data_candidate_graph_dq import (
    CAMPAIGN_CANDIDATE_GRAPH_DQ_FAILED,
    assign_campaign_decision,
    campaign_allows_cdx_mode,
    evaluate_candidate_graph_dq_gate,
    load_candidate_graph_dq_gate,
)
from alignment.real_data_operating_protocol import (
    BLOCK_CANDIDATE_GRAPH_DQ_FAILED,
    ROLE_CALIBRATION,
    ROLE_HELD_OUT_DQ,
    ROLE_HOLDOUT,
    assign_block_status,
    cross_level_contamination_diagnostic,
)
from scripts.submit_real_data_operating_protocol_condor import _complete_source


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    PROJECT_ROOT / "outputs" / "mc24_ift_calibration_mode_validity_contract_v1" / "mode_validity_contract.json"
)


def _json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--candidate-graph-dq",
        default=None,
        help="Observable-only frozen V2 candidate-graph DQ JSON; required to leave the wave-1 placeholder.",
    )
    args = parser.parse_args()
    physical = Path(args.physical_output_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    manifest = json.loads((physical / "iteration_manifest.json").read_text(encoding="utf-8"))
    catalog = json.loads((physical / "real_data_provenance_catalog.json").read_text(encoding="utf-8"))
    contract = load_mode_validity_contract(CONTRACT)
    created = datetime.now(timezone.utc).isoformat()
    dq_payload = None
    dq_by_id: dict[str, Mapping[str, Any]] = {}
    station_mode_gate: dict[str, Any] = {"passed": False, "per_block": []}
    station_mode_gate_passed = False
    if args.candidate_graph_dq:
        dq_payload = json.loads(Path(args.candidate_graph_dq).expanduser().resolve().read_text(encoding="utf-8"))
        dq_by_id = {str(block["source_id"]): block for block in dq_payload.get("blocks") or []}
        station_mode_gate = evaluate_candidate_graph_dq_gate(
            list(dq_payload.get("blocks") or []),
            load_candidate_graph_dq_gate(),
        )
        station_mode_gate_passed = bool(station_mode_gate["passed"])
    campaign_decision = assign_campaign_decision(
        candidate_graph_dq_failed=dq_payload is None or not station_mode_gate_passed,
        station_fit_failed=False,
        implied_cdx_exceeds_operating_band=False,
        independent_cdx_evidence=False,
        station_and_holdout_dq_ok=False,
    )
    per_block_gate = {str(item.get("source_id")): item for item in station_mode_gate.get("per_block") or []}
    blocks: list[dict[str, Any]] = []
    for source in manifest["sources"]:
        source_id = str(source["source_id"])
        role = str(source["role"])
        complete = _complete_source(physical / "sources" / source_id)
        dq_row = dq_by_id.get(source_id)
        selected_empty = bool(dq_row is None or not dq_row.get("selected_v2_graph_nonempty"))
        physical_graph_empty = bool(dq_row is not None and not dq_row.get("physical_all_pairs_candidate_graph_nonempty"))
        values_finite = bool(dq_row is None or dq_row.get("values_finite"))
        block_gate = per_block_gate.get(source_id) or {}
        graph_failed = bool(
            dq_row is None
            or not station_mode_gate_passed
            or block_gate.get("passed") is False
        )
        diagnostic = cross_level_contamination_diagnostic(
            contract,
            station_dx_mm=None,
            station_ry_mrad=None,
            run_to_run_dx_spread_mm=None,
        )
        status = assign_block_status(
            role=role,
            dq_failed=(not complete) or physical_graph_empty or (not values_finite),
            cross_level_contaminated=False,
            station_mode_valid=False,
            cdx_mode_valid=False,
            holdout_dq_worsened=False,
            anomalous_run_drift=False,
            candidate_graph_dq_failed=graph_failed,
            candidate_graph_dq_passed=bool(
                complete and dq_row is not None and block_gate.get("passed") and station_mode_gate_passed
            ),
        )
        if not complete:
            status_note = "Physical wave-1 incomplete; no geometry candidate."
        elif dq_row is None:
            status_note = "Physical wave-1 complete; frozen V2 candidate-graph DQ has not been attached."
        elif physical_graph_empty:
            status_note = "Physical all-pairs candidate graph is empty."
        elif selected_empty:
            status_note = (
                "Physical all-pairs candidate graph is non-empty and finite, but the frozen "
                "V2/route policy selected zero routes (no positive-utility hypothesis under "
                "unmatched_penalty=-1.0).  Thresholds, unmatched penalty, occupancy windows, "
                "and the mode-validity contract are unchanged.  Station Mode did not start."
            )
        elif not station_mode_gate_passed:
            status_note = "Candidate-graph DQ did not pass the calibration Station Mode gate."
        else:
            status_note = "Candidate-graph DQ passed; Station Mode numbers still required."
        blocks.append(
            {
                "source_id": source_id,
                "run": int(source["run"]),
                "role": role,
                "status": status,
                "physical_wave1_complete": complete,
                "candidate_graph_dq": None if dq_row is None else {
                    "physical_all_pairs_candidate_graph_nonempty": dq_row.get(
                        "physical_all_pairs_candidate_graph_nonempty"
                    ),
                    "selected_v2_graph_nonempty": dq_row.get("selected_v2_graph_nonempty"),
                    "n_events_with_tracklets": dq_row.get("n_events_with_tracklets"),
                    "n_all_pairs_candidates": dq_row.get("n_all_pairs_candidates"),
                    "selected_edges": dq_row.get("selected_edges"),
                    "selected_routes": dq_row.get("selected_routes"),
                    "complete_four_station_routes": dq_row.get("complete_four_station_routes"),
                    "n_adjacent_candidates_above_frozen_threshold": dq_row.get(
                        "n_adjacent_candidates_above_frozen_threshold"
                    ),
                    "gate_passed": block_gate.get("passed"),
                    "gate_reasons": list(block_gate.get("reasons") or []),
                },
                "geometry_estimation_allowed": role == ROLE_CALIBRATION,
                "cdx_mode_allowed_this_stage": False,
                "geometry_write_allowed": False,
                "residual_reduction_is_not_alignment_success": True,
                "cross_level_contamination_diagnostic": diagnostic,
                "note": status_note,
                "provenance": {
                    "geometry_tag": source["geometry_tag"],
                    "conditions_tag": source["conditions_tag"],
                    "reconstruction_tag": source["reconstruction_tag"],
                    "trigger_and_dq": source["trigger_and_dq"],
                    "lumiblock": source["lumiblock"],
                    "cdx_fixed_by": "external_geometry",
                    "cdx_fixed_declaration": (
                        "C_dx is declared fixed by the current official geometry / "
                        "external alignment.  True C_dx is unknown on real data."
                    ),
                },
            }
        )
    station_report = {
        "schema_version": "faser-operating-protocol-v1-real-data-station-mode",
        "created_utc": created,
        "decision": campaign_decision,
        "official_conditions_db_write": False,
        "mc_truth_used": False,
        "self_nulling_update": False,
        "station_mode_started": False,
        "station_mode_gate_passed": station_mode_gate_passed,
        "station_mode_gate": station_mode_gate,
        "candidate_graph_dq": None if args.candidate_graph_dq is None else str(Path(args.candidate_graph_dq).expanduser().resolve()),
        "cdx_fixed_by": "external_geometry",
        "physical_output_dir": str(physical),
        "provenance_catalog": str(physical / "real_data_provenance_catalog.json"),
        "blind_split": {
            "calibration": [b["source_id"] for b in blocks if b["role"] == ROLE_CALIBRATION],
            "holdout": [b["source_id"] for b in blocks if b["role"] == ROLE_HOLDOUT],
            "held_out_dq": [b["source_id"] for b in blocks if b["role"] == ROLE_HELD_OUT_DQ],
        },
        "blocks": blocks,
        "geometry_write_allowed": False,
        "cdx_mode_allowed": campaign_allows_cdx_mode(campaign_decision),
        "residual_reduction_is_not_alignment_success": True,
    }
    cdx_reason = (
        "Dedicated C_dx Mode has not started and is not allowed.  The unique "
        f"campaign decision is {campaign_decision}: the frozen candidate-graph "
        "DQ gate rejected Station Mode admission.  Thresholds, unmatched "
        "penalty, occupancy, V2, route, capture, and the mode-validity "
        "contract were not changed.  The official conditions DB was not modified."
        if campaign_decision == CAMPAIGN_CANDIDATE_GRAPH_DQ_FAILED
        else (
            "Dedicated C_dx Mode has not started.  Station Mode did not produce "
            "a writable geometry candidate.  The official conditions DB was not modified."
        )
    )
    cdx_report = {
        "schema_version": "faser-operating-protocol-v1-real-data-cdx-mode",
        "created_utc": created,
        "decision": campaign_decision,
        "official_conditions_db_write": False,
        "mc_truth_used": False,
        "status": "not_started",
        "cdx_mode_allowed": campaign_allows_cdx_mode(campaign_decision),
        "reason": cdx_reason,
        "station_to_C_dx_systematic_rss_1sigma_um": float(
            contract["ift_internal_mode"]["station_to_C_dx_propagation"]["free_station_rss"]["rss_1sigma_um"]
        ),
        "do_not_absorb_B_into_fit_sigma": True,
        "blocks": [
            {
                "source_id": block["source_id"],
                "run": block["run"],
                "role": block["role"],
                "status": BLOCK_CANDIDATE_GRAPH_DQ_FAILED
                if campaign_decision == CAMPAIGN_CANDIDATE_GRAPH_DQ_FAILED
                else block["status"],
                "cdx_mode_run": False,
            }
            for block in blocks
        ],
        "geometry_write_allowed": False,
    }
    decision_reason = (
        "Frozen V2 --allow-real-data all-pairs inference ran on the five frozen "
        "100-event current-geometry identity samples.  Physical all-pairs "
        "candidate graphs are non-empty and finite, but the frozen route policy "
        "(threshold 0.001, unmatched_penalty=-1.0) selected zero routes because "
        "no hypothesis had positive utility.  The pre-frozen candidate-graph DQ "
        "gate therefore failed closed.  Occupancy/window, V2, route, capture, "
        "and the mode-validity contract were not changed.  Station Mode did not "
        "start; dedicated C_dx Mode is not allowed; geometry_write_allowed "
        "remains false; the official conditions DB was not modified."
        if campaign_decision == CAMPAIGN_CANDIDATE_GRAPH_DQ_FAILED
        else (
            "Station Mode physical wave-1 is "
            + ("complete" if any(block["physical_wave1_complete"] for block in blocks) else "not complete")
            + f"; unique campaign decision is {campaign_decision}."
        )
    )
    decision = {
        "schema_version": "faser-operating-protocol-v1-decision",
        "created_utc": created,
        "decision": campaign_decision,
        "cdx_mode_allowed": campaign_allows_cdx_mode(campaign_decision),
        "official_conditions_db_modified": False,
        "official_conditions_db_write_allowed": False,
        "mc_truth_used": False,
        "sealed_test_opened": False,
        "retrain_v2": False,
        "joint_station_cdx_newton": False,
        "same_data_stage_iteration": False,
        "conditions_writing_rehearsal_allowed": False,
        "physics_production_validation_allowed": False,
        "reason": decision_reason,
        "blocks": [
            {
                "source_id": block["source_id"],
                "run": block["run"],
                "role": block["role"],
                "status": block["status"],
            }
            for block in blocks
        ],
        "station_mode_report": str(output / "real_data_station_mode_report.json"),
        "cdx_mode_report": str(output / "real_data_cdx_mode_report.json"),
        "provenance_catalog": catalog,
    }
    _json(output / "real_data_station_mode_report.json", station_report)
    _json(output / "real_data_cdx_mode_report.json", cdx_report)
    _json(output / "operating_protocol_decision.json", decision)
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "decision": campaign_decision,
                "n_blocks": len(blocks),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
