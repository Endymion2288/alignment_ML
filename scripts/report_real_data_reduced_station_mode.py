#!/usr/bin/env python3
"""Write the reduced Station calibration mode feasibility reports.

Uses existing 14973/14974 finite-difference J_s and the frozen MC A.
Does not retrain V2, write geometry, start C_dx Mode, or float dz.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from alignment.calibration_modes import load_mode_validity_contract
from alignment.hierarchical_v1 import STATION_SOLVE_PARAMETERS
from alignment.real_data_operating_protocol import ROLE_CALIBRATION, ROLE_HELD_OUT_DQ, ROLE_HOLDOUT
from alignment.real_data_reduced_station_mode import (
    SCHEMA_VERSION,
    SURVEY_DZ,
    TRACK_DRIVEN_PARAMETERS,
    attach_leakage,
    blind_transfer_from_residual_blocks,
    classify_reduced_campaign,
    common_identifiable_subspace,
    evaluate_reduced_mode,
    isolation_axes,
    linearized_transfer_chi2_ratio,
    load_reduced_mode_config,
    predeclared_modes,
    run_to_run_parameter_consistency,
    solve_reduced_self_nulling,
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


def _load_newton(self_nulling_root: Path, source_id: str) -> dict[str, Any]:
    arrays = np.load(self_nulling_root / "newton" / source_id / "route_selected_update_arrays.npz")
    return {
        "parameter_names": [str(name) for name in arrays["parameter_names"].tolist()],
        "derivative_native": np.asarray(arrays["derivative_native"], dtype=np.float64),
        "covariance": np.asarray(arrays["covariance"], dtype=np.float64),
        "residual": np.asarray(arrays["anchor_residual"], dtype=np.float64),
    }


def _public_solve(solve: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in solve.items()
        if key not in {"normal_matrix_native", "normal_matrix_scaled"}
    }


def _evaluate_mode(
    mode_name: str,
    floated: tuple[str, ...],
    banks: Mapping[int, Mapping[str, Any]],
    a_map: Mapping[str, float],
    config: Mapping[str, Any],
    blind_ok: bool,
) -> dict[str, Any]:
    solves: dict[int, dict[str, Any]] = {}
    leakages: dict[int, dict[str, Any]] = {}
    for run, bank in banks.items():
        fit = solve_reduced_self_nulling(
            bank["derivative_native"],
            bank["covariance"],
            bank["residual"],
            all_parameter_names=bank["parameter_names"],
            floated=floated,
        )
        solves[run] = fit
        leakages[run] = attach_leakage(
            fit,
            a_map,
            max_a_projection=float(config["max_a_subspace_projection"]),
        )
    runs = sorted(solves)
    consistency = run_to_run_parameter_consistency(
        solves[runs[0]]["delta"],
        solves[runs[1]]["delta"],
        solves[runs[0]]["sigma"],
        solves[runs[1]]["sigma"],
        floated,
        max_nsigma=float(config["max_run_to_run_nsigma"]),
    )
    transfers = []
    for source, target in ((runs[0], runs[1]), (runs[1], runs[0])):
        item = linearized_transfer_chi2_ratio(
            banks[target]["derivative_native"],
            banks[target]["covariance"],
            banks[target]["residual"],
            all_parameter_names=banks[target]["parameter_names"],
            floated=floated,
            delta=solves[source]["delta"],
        )
        item["from_run"] = source
        item["to_run"] = target
        transfers.append(item)
    admission = evaluate_reduced_mode(
        mode_name=mode_name,
        floated=floated,
        per_run_solve=solves,
        per_run_leakage=leakages,
        consistency=consistency,
        transfers=transfers,
        blind_transfer_ok=blind_ok,
        target_condition=float(config["target_condition_number"]),
        max_condition_rel_spread=float(config["max_condition_rel_spread"]),
        max_transfer_chi2_ratio=float(config["max_transfer_chi2_ratio"]),
    )
    return {
        "admission": admission,
        "per_run": {
            str(run): {
                "solve": _public_solve(solves[run]),
                "leakage": leakages[run],
            }
            for run in runs
        },
        "run_to_run_consistency": consistency,
        "linearized_calibration_transfer": transfers,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "operating_protocol_v1_real_data_reduced_station_mode_feasibility_v1.yaml"),
    )
    parser.add_argument("--campaign-root", required=True)
    args = parser.parse_args()
    config = load_reduced_mode_config(args.config)
    campaign_root = Path(args.campaign_root).expanduser().resolve()
    campaign_root.mkdir(parents=True, exist_ok=True)
    contract = load_mode_validity_contract(PROJECT_ROOT / str(config["mode_validity_contract"]))
    a_map = contract["leakage_operator"]["A_native_per_mm_C_dx"]
    self_nulling_root = PROJECT_ROOT / str(config["self_nulling_root"])
    created = datetime.now(timezone.utc).isoformat()

    banks: dict[int, dict[str, Any]] = {}
    for run, meta in config["full_segment_sources"].items():
        if str(config["blind_roles"][str(run)]) != ROLE_CALIBRATION:
            continue
        banks[int(run)] = _load_newton(self_nulling_root, str(meta["source_id"]))

    route_dq = _read_json(
        PROJECT_ROOT / str(config["self_nulling_root"]) / "route_dq_fullscale_report.json"
    )
    cal_blocks = [block for block in route_dq["blocks"] if block.get("role") == ROLE_CALIBRATION]
    blind_blocks = [
        block for block in route_dq["blocks"] if block.get("role") in {ROLE_HOLDOUT, ROLE_HELD_OUT_DQ}
    ]
    blind = blind_transfer_from_residual_blocks(cal_blocks, blind_blocks)

    modes = predeclared_modes(config)
    evaluated: dict[str, dict[str, Any]] = {}
    for name, floated in modes.items():
        evaluated[name] = _evaluate_mode(name, floated, banks, a_map, config, bool(blind["holdout_transfer_ok"]))

    extra_names = {
        "two_dof_dy_rx": ("ift_dy_mm", "ift_rx_mrad"),
        "two_dof_dy_rz": ("ift_dy_mm", "ift_rz_mrad"),
        "two_dof_rx_rz": ("ift_rx_mrad", "ift_rz_mrad"),
        "one_dof_dy": ("ift_dy_mm",),
        "one_dof_rx": ("ift_rx_mrad",),
        "one_dof_rz": ("ift_rz_mrad",),
    }
    extras: dict[str, dict[str, Any]] = {}
    for name, floated in extra_names.items():
        extras[name] = _evaluate_mode(name, floated, banks, a_map, config, bool(blind["holdout_transfer_ok"]))

    five_by_run: dict[int, dict[str, Any]] = {}
    for run, payload in evaluated["five_dof_no_dz"]["per_run"].items():
        five_by_run[int(run)] = payload["solve"]
    subspace = common_identifiable_subspace(
        five_by_run,
        a_map,
        max_a_projection=float(config["max_a_subspace_projection"]),
        target_condition=float(config["target_condition_number"]),
    )
    subspace["isolation_safe_axes_expected"] = list(isolation_axes())

    admissions = [evaluated[name]["admission"] for name in modes] + [
        extras[name]["admission"] for name in extras
    ]
    classification = classify_reduced_campaign(admissions)

    ident = {
        "schema_version": SCHEMA_VERSION + "-identifiability",
        "created_utc": created,
        "survey_dz_mm": 0.0,
        "track_driven_dz": False,
        "track_driven_parameters": list(TRACK_DRIVEN_PARAMETERS),
        "all_parameter_names_in_J_s": list(STATION_SOLVE_PARAMETERS),
        "predeclared_modes": {
            name: {
                "floated_parameters": list(modes[name]),
                "per_run": {
                    run: {
                        "full_rank": payload["per_run"][run]["solve"]["full_rank"],
                        "rank": payload["per_run"][run]["solve"]["rank"],
                        "condition_number": payload["per_run"][run]["solve"]["condition_number"],
                        "raw_condition_number": payload["per_run"][run]["solve"]["raw_condition_number"],
                        "singular_values": payload["per_run"][run]["solve"]["singular_values"],
                        "sigma": payload["per_run"][run]["solve"]["sigma"],
                        "delta": payload["per_run"][run]["solve"]["delta"],
                        "n_observations": payload["per_run"][run]["solve"]["n_observations"],
                    }
                    for run in payload["per_run"]
                },
                "condition_stability": payload["admission"]["condition_stability"],
                "run_to_run_consistency": payload["run_to_run_consistency"],
            }
            for name, payload in evaluated.items()
        },
        "common_identifiable_subspace": subspace,
        "residual_reduction_is_not_alignment_success": True,
        "geometry_write_allowed": False,
        "note": (
            "J_s comes from the already-captured 12 station FD probes.  "
            "dz is dropped from the solve and fixed at survey 0."
        ),
    }
    leakage = {
        "schema_version": SCHEMA_VERSION + "-leakage",
        "created_utc": created,
        "frozen_A_source": contract["leakage_operator"]["source"],
        "A_native_per_mm_C_dx": dict(a_map),
        "operating_band_um": [1.5, 1.7],
        "cannot_rescue_dx_ry_by_more_events_or_looser_thresholds": True,
        "do_not_emit_cdx_payload": True,
        "new_cdx_payload": None,
        "not_a_C_dx_measurement": True,
        "predeclared_modes": {
            name: {
                "floated_parameters": list(modes[name]),
                "per_run": {
                    run: payload["per_run"][run]["leakage"] for run in payload["per_run"]
                },
                "hard_isolation_reject": payload["admission"]["hard_isolation_reject"],
            }
            for name, payload in evaluated.items()
        },
        "isolation_safe_subset_search": {
            name: {
                "floated_parameters": list(extra_names[name]),
                "hard_isolation_reject": extras[name]["admission"]["hard_isolation_reject"],
                "clearly_orthogonal_to_A": extras[name]["admission"]["clearly_orthogonal_to_A"],
                "per_run": {
                    run: extras[name]["per_run"][run]["leakage"] for run in extras[name]["per_run"]
                },
            }
            for name in extra_names
        },
        "geometry_write_allowed": False,
        "cdx_mode_started": False,
    }
    transfer = {
        "schema_version": SCHEMA_VERSION + "-transfer-dq",
        "created_utc": created,
        "linearized_calibration_transfer": {
            name: evaluated[name]["linearized_calibration_transfer"] for name in modes
        },
        "blind_current_geometry_transfer": blind,
        "note": (
            "14975/14976/14977 have no finite-difference J_s.  Their transfer "
            "DQ is current-geometry residual IQR overlap, not a linearized "
            "update.  Residual drop is never alignment success."
        ),
        "geometry_write_allowed": False,
        "residual_reduction_is_not_alignment_success": True,
    }
    feasibility = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": created,
        "frozen_v2_checkpoint_sha256": config["frozen_v2_checkpoint_sha256"],
        "do_not_retrain_v2": True,
        "survey_dz_mm": 0.0,
        "predeclared_mode_admissions": {name: evaluated[name]["admission"] for name in modes},
        "isolation_safe_subset_admissions": {name: extras[name]["admission"] for name in extras},
        "common_identifiable_subspace": subspace,
        "decision": classification["decision"],
        "v2_candidate_mode": classification["v2_candidate_mode"],
        "geometry_write_allowed": False,
        "residual_reduction_is_not_alignment_success": True,
    }
    decision = {
        "schema_version": SCHEMA_VERSION + "-next-decision",
        "created_utc": created,
        **classification,
        "dq_versus_alignment": {
            "reduced_mode_rank_condition_sigma": "identifiability_diagnostic_not_closure",
            "implied_C_dx": "rejection_diagnostic_not_a_measurement",
            "residual_drop": "never_alignment_success",
            "self_nulling_parameter_update": "estimator_output_not_correctness",
            "blind_transfer_iqr": "data_quality_only",
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
            "Do not write any self-nulling correction.  "
            + (
                "Register the reduced mode as a Real-Data Station Calibration "
                "Mode V2 candidate definition only; remaining station DoF stay "
                "with survey/external alignment."
                if classification["decision"] == "real_data_station_calibration_mode_v2_candidate"
                else "Current real data can only support dedicated residual/DQ "
                "monitoring and cannot produce a station geometry update."
            )
        ),
        "held_out_dq_used_for_verdict": False,
        "blocks": [
            {"run": int(run), "role": str(config["blind_roles"][str(run)]), "source_id": str(meta["source_id"])}
            for run, meta in config["full_segment_sources"].items()
        ],
    }

    outputs = (
        (ident, "reduced_mode_identifiability_audit.json"),
        (leakage, "reduced_mode_leakage_audit.json"),
        (transfer, "reduced_mode_transfer_dq_report.json"),
        (feasibility, "reduced_station_mode_feasibility_report.json"),
        (decision, "operating_protocol_next_decision.json"),
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
                "v2_candidate_mode": classification["v2_candidate_mode"],
                "geometry_write_allowed": False,
                "cdx_mode_allowed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
