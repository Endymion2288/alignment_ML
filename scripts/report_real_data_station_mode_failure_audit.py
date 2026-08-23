#!/usr/bin/env python3
"""Write the frozen-V2 real-data Station Mode failure-audit reports.

Uses the already-captured 12 station FD probes and the frozen current V2
selected routes.  Does not retrain V2, write geometry, start C_dx Mode,
or emit a new C_dx payload.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from alignment.calibration_modes import load_mode_validity_contract
from alignment.real_data_operating_protocol import ROLE_CALIBRATION
from alignment.real_data_station_mode_failure_audit import (
    SCHEMA_VERSION,
    SURVEY_DZ,
    classify_failure,
    contamination_backprojection,
    estimate_statistics_scaling,
    load_failure_audit_config,
    reconstruct_js_identifiability,
    route_station_count,
    scaled_normal_from_js,
)
from alignment.real_data_station_mode_fullscale import summarize_selected_routes


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


def _newton_dir(self_nulling_root: Path, source_id: str) -> Path:
    return self_nulling_root / "newton" / source_id


def _load_newton_js(source_id: str, self_nulling_root: Path) -> dict[str, Any]:
    directory = _newton_dir(self_nulling_root, source_id)
    arrays = np.load(directory / "route_selected_update_arrays.npz")
    update = _read_json(directory / "route_selected_update.json")
    with (directory / "route_selected_update_observations.csv").open(encoding="utf-8", newline="") as handle:
        observations = [dict(row) for row in csv.DictReader(handle)]
    lengths = [route_station_count(row.get("route_origin_signature")) for row in observations]
    names = [str(name) for name in arrays["parameter_names"].tolist()]
    return {
        "source_id": source_id,
        "parameter_names": names,
        "derivative_native": np.asarray(arrays["derivative_native"], dtype=np.float64),
        "covariance": np.asarray(arrays["covariance"], dtype=np.float64),
        "route_lengths": lengths,
        "update": update,
        "n_fd_probes": 12,
        "probe_points": update.get("probe_points"),
    }


def _station_correction(update: Mapping[str, Any]) -> dict[str, float]:
    values = update.get("proposed_next_parameter_values") or update.get("alignment_parameter_values") or {}
    return {str(name): float(value) for name, value in dict(values).items()}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "operating_protocol_v1_real_data_station_mode_failure_audit_v1.yaml"),
    )
    parser.add_argument("--campaign-root", required=True)
    args = parser.parse_args()
    config = load_failure_audit_config(args.config)
    campaign_root = Path(args.campaign_root).expanduser().resolve()
    campaign_root.mkdir(parents=True, exist_ok=True)
    self_nulling_root = PROJECT_ROOT / str(config["self_nulling_root"])
    scaling_root = PROJECT_ROOT / str(config["scaling_root"])
    contract = load_mode_validity_contract(PROJECT_ROOT / str(config["mode_validity_contract"]))
    created = datetime.now(timezone.utc).isoformat()
    a_map = contract["leakage_operator"]["A_native_per_mm_C_dx"]
    target = float(config["target_condition_number"])
    near_abs = float(config["near_degenerate_abs_component"])
    a_threshold = float(config["a_alignment_threshold"])

    calibration_runs = [
        int(run)
        for run, role in config["blind_roles"].items()
        if role == ROLE_CALIBRATION
    ]
    ident_blocks: list[dict[str, Any]] = []
    route_blocks: list[dict[str, Any]] = []
    station_dx: dict[int, float | None] = {}
    station_ry: dict[int, float | None] = {}
    normals: dict[int, dict[str, Any]] = {}
    donor_complete = None
    donor_run = None
    max_five_a_cosine = None

    for run in calibration_runs:
        meta = config["full_segment_sources"][str(run)]
        source_id = str(meta["source_id"])
        packed = _load_newton_js(source_id, self_nulling_root)
        audit = reconstruct_js_identifiability(
            packed["derivative_native"],
            packed["covariance"],
            parameter_names=packed["parameter_names"],
            route_lengths=packed["route_lengths"],
            near_degenerate_abs_component=near_abs,
            a_native_per_mm=a_map,
        )
        native, scaled, names = scaled_normal_from_js(
            packed["derivative_native"],
            packed["covariance"],
            parameter_names=packed["parameter_names"],
        )
        lengths = np.asarray(packed["route_lengths"], dtype=np.int64)
        complete_mask = lengths >= 4
        complete_unit = None
        if np.any(complete_mask):
            _native_c, complete_unit, _ = scaled_normal_from_js(
                packed["derivative_native"],
                packed["covariance"],
                parameter_names=names,
                observation_mask=complete_mask,
            )
            donor_complete = complete_unit
            donor_run = run
        incomplete_mask = lengths < 4
        _native_i, incomplete, _ = scaled_normal_from_js(
            packed["derivative_native"],
            packed["covariance"],
            parameter_names=names,
            observation_mask=incomplete_mask if np.any(incomplete_mask) else None,
        )
        correction = _station_correction(packed["update"])
        station_dx[run] = correction.get("ift_dx_mm")
        station_ry[run] = correction.get("ift_ry_mrad")
        five_cosine = audit["five_dof_excluding_survey_dz"]["weak_direction_cosine_with_A"]
        if five_cosine is not None:
            max_five_a_cosine = five_cosine if max_five_a_cosine is None else max(max_five_a_cosine, five_cosine)
        normals[run] = {
            "names": names,
            "scaled": scaled,
            "incomplete": incomplete,
            "complete_unit": complete_unit,
            "n_complete_observations": int(np.count_nonzero(complete_mask)),
            "n_events": int(meta["nevents"]),
        }
        ident_blocks.append(
            {
                "run": run,
                "role": ROLE_CALIBRATION,
                "source_id": source_id,
                "used_for_verdict": True,
                "n_fd_probes": 12,
                "probe_points": packed["probe_points"],
                "n_observations": audit["n_observations"],
                "route_length_observation_histogram": {
                    str(length): int(np.count_nonzero(lengths == length))
                    for length in sorted(set(lengths.tolist()))
                },
                "identifiability": audit,
                "self_nulling_parameter_update": correction,
                "self_nulling_is_not_correctness": True,
                "newton_condition_number": packed["update"].get("normal_matrix_condition_number"),
                "newton_singular_values": packed["update"].get("data_singular_values"),
            }
        )
        routes = _load_jsonl(scaling_root / "association" / source_id / "selected_routes.jsonl")
        summary = summarize_selected_routes(routes)
        route_blocks.append(
            {
                "run": run,
                "role": ROLE_CALIBRATION,
                "source_id": source_id,
                "n_events": int(meta["nevents"]),
                "used_for_verdict": True,
                "selected_routes": summary["selected_routes"],
                "complete_four_station_routes": summary["complete_four_station_routes"],
                "truth_free_complete_route_fraction": summary["truth_free_complete_route_fraction"],
                "route_length_histogram": summary["route_length_histogram"],
                "edge_reuse": summary["edge_reuse"],
                "event_concentration": summary["event_concentration"],
                "route_multiplicity": summary["route_multiplicity"],
                "selected_graph_nonempty": int(summary["selected_routes"]) > 0,
                "do_not_retrain_v2": True,
            }
        )

    scaling_by_run: dict[str, Any] = {}
    six_recoverable = False
    complete_practical = False
    for run, packed in normals.items():
        unit = packed["complete_unit"] if packed["complete_unit"] is not None else donor_complete
        n_complete_obs = packed["n_complete_observations"]
        n_complete_routes = next(
            int(block["complete_four_station_routes"])
            for block in route_blocks
            if int(block["run"]) == run
        )
        if n_complete_obs <= 0 and donor_complete is not None:
            donor_block = next(block for block in route_blocks if int(block["run"]) == donor_run)
            n_complete_obs = int(normals[int(donor_run)]["n_complete_observations"])
            n_complete_routes = int(donor_block["complete_four_station_routes"])
            n_events = int(donor_block["n_events"])
            donor_used = True
        else:
            n_events = packed["n_events"]
            donor_used = False
        estimate = estimate_statistics_scaling(
            packed["incomplete"],
            unit,
            names=packed["names"],
            n_complete_observations=n_complete_obs,
            n_complete_routes=n_complete_routes,
            n_events=n_events,
            target_condition_number=target,
            practical_complete_routes_max=int(config["practical_complete_routes_max"]),
            practical_events_max=int(config["practical_events_max"]),
        )
        estimate["donor_complete_block_run"] = None if packed["complete_unit"] is not None else donor_run
        estimate["used_donor_complete_block"] = donor_used
        scaling_by_run[str(run)] = estimate
        six = estimate["complete_four_station_scaling"]["six_dof"]
        if six.get("recovers"):
            six_recoverable = True
            if six.get("practical") is not False:
                complete_practical = True

    contamination = contamination_backprojection(
        contract,
        station_dx_by_run=station_dx,
        station_ry_by_run=station_ry,
    )
    selected = [int(block["selected_routes"]) for block in route_blocks]
    complete = [int(block["complete_four_station_routes"]) for block in route_blocks]
    classification = classify_failure(
        selected_routes_calibration=selected,
        complete_four_station_calibration=complete,
        all_pairs_nonempty=True,
        six_dof_recoverable_from_complete_tracks=six_recoverable,
        complete_track_recovery_practical=complete_practical,
        same_topology_recovers_six_dof=False,
        implied_cdx_exceeds_operating_band=bool(contamination["exceeds_operating_band"]),
        weak_direction_cosine_with_A=max_five_a_cosine,
        a_alignment_threshold=a_threshold,
    )
    local_field_dominated = all(
        int((block.get("route_length_histogram") or {}).get("2") or 0)
        + int((block.get("route_length_histogram") or {}).get("3") or 0)
        >= int(block["selected_routes"]) - int(block["complete_four_station_routes"])
        and int(block["complete_four_station_routes"]) <= 2
        for block in route_blocks
    )

    ident_report = {
        "schema_version": SCHEMA_VERSION + "-identifiability",
        "created_utc": created,
        "frozen_v2_checkpoint_sha256": config["frozen_v2_checkpoint_sha256"],
        "do_not_retrain_v2": True,
        "q_over_p_mode": 0,
        "reconstruction": "central_finite_difference_J_s_from_12_station_FD_probes",
        "residual_improvement_is_not_closure": True,
        "residual_reduction_is_not_alignment_success": True,
        "geometry_write_allowed": False,
        "target_condition_number": target,
        "survey_parameter": SURVEY_DZ,
        "blocks": ident_blocks,
        "near_degenerate_summary": {
            str(block["run"]): block["identifiability"]["near_degenerate"]["near_degenerate"]
            for block in ident_blocks
        },
        "note": (
            "J_s is rebuilt from the 12 axial station FD probes already captured "
            "on the frozen selected-route edges.  Rank, condition number, and the "
            "singular spectrum are identifiability diagnostics, not closure."
        ),
    }
    leakage_report = {
        "schema_version": SCHEMA_VERSION + "-cdx-contamination",
        "created_utc": created,
        "frozen_A_source": contract["leakage_operator"]["source"],
        "A_native_per_mm_C_dx": dict(a_map),
        "mc_A_subspace_r2": contract["leakage_operator"].get("subspace_r2"),
        "mc_A_subspace_cosine": contract["leakage_operator"].get("subspace_cosine"),
        "weak_direction_cosine_with_A": {
            str(block["run"]): block["identifiability"]["five_dof_excluding_survey_dz"][
                "weak_direction_cosine_with_A"
            ]
            for block in ident_blocks
        },
        **contamination,
        "note": (
            "Implied |C_dx| is a rejection diagnostic against the frozen "
            "1.5-1.7 um isolation budget.  It is not a C_dx measurement and "
            "must not be inverted into a new C_dx payload."
        ),
    }
    route_report = {
        "schema_version": SCHEMA_VERSION + "-route-statistics",
        "created_utc": created,
        "do_not_retrain_v2": True,
        "current_v2_association_retained": True,
        "truth_metrics_omitted": True,
        "local_field_topology_dominated": bool(local_field_dominated),
        "blocks": route_blocks,
        "comparison": {
            "selected_routes": {str(block["run"]): block["selected_routes"] for block in route_blocks},
            "complete_four_station_routes": {
                str(block["run"]): block["complete_four_station_routes"] for block in route_blocks
            },
            "truth_free_complete_route_fraction": {
                str(block["run"]): block["truth_free_complete_route_fraction"] for block in route_blocks
            },
            "edge_reuse": {str(block["run"]): block["edge_reuse"] for block in route_blocks},
            "event_concentration": {
                str(block["run"]): block["event_concentration"] for block in route_blocks
            },
        },
        "assessment": {
            "complete_four_station_insufficient": bool(sum(complete) <= 2),
            "station_solve_dominated_by_local_field_topology": bool(local_field_dominated),
            "v2_association_failure": False,
            "do_not_retrain_v2": True,
        },
        "residual_reduction_is_not_alignment_success": True,
        "note": (
            "truth_free_complete_route_fraction is the real-data stand-in for "
            "complete-track occupancy.  MC labels are absent."
        ),
    }
    any_six = any(
        bool(item["complete_four_station_scaling"]["six_dof"].get("recovers"))
        for item in scaling_by_run.values()
    )
    decision = {
        "schema_version": SCHEMA_VERSION + "-failure-classification",
        "created_utc": created,
        **classification,
        "classes": {
            "A": "reconstruction_statistics_limitation",
            "B": "physical_nonidentifiability",
            "C": "v2_association_failure",
        },
        "dq_versus_alignment": {
            "route_statistics": "data_quality_only",
            "J_s_rank_condition_spectrum": "identifiability_diagnostic_not_closure",
            "implied_C_dx": "rejection_diagnostic_not_a_measurement",
            "residual_drop": "never_alignment_success",
            "self_nulling_parameter_update": "estimator_output_not_correctness",
        },
        "statistics_scaling_control": {
            "schur_production_estimator_used": False,
            "geometry_write_allowed": False,
            "per_run": scaling_by_run,
            "six_dof_recoverable_from_observed_complete_topology": any_six,
            "station_covariance_can_recover_by_more_complete_tracks": any_six,
            "practical_path_to_geometry_write": False,
        },
        "official_conditions_db_modified": False,
        "sealed_test_opened": False,
        "retrain_v2": False,
        "reregister_capture_or_A": False,
        "same_data_stage_iteration": False,
        "conditions_writing_rehearsal_allowed": False,
        "physics_production_validation_allowed": False,
        "next_allowed_step": (
            "Keep geometry_write_allowed=false.  Do not start C_dx Mode.  "
            "Do not invert implied |C_dx| into a payload.  Do not retune V2.  "
            "Redefine the calibration mode before any production conditions path."
        ),
        "held_out_dq_used_for_verdict": False,
        "blocks": [
            {"run": block["run"], "role": block["role"], "source_id": block["source_id"]}
            for block in route_blocks
        ],
    }

    outputs = (
        (ident_report, "station_identifiability_real_data_audit.json"),
        (leakage_report, "cdx_cross_level_contamination_audit.json"),
        (route_report, "route_statistics_limitation_audit.json"),
        (decision, "operating_protocol_failure_classification.json"),
    )
    for payload, name in outputs:
        _walk_forbidden(payload, where=name)
        _write_json(campaign_root / name, payload)
    print(
        json.dumps(
            {
                "campaign_root": str(campaign_root),
                "unique_class": classification["unique_class"],
                "concurrent_class": classification["concurrent_class"],
                "geometry_write_allowed": False,
                "cdx_mode_allowed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
