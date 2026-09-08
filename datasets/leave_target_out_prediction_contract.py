"""Task B10: leave-target-out prediction contract.

Asks whether a statistically independent prediction state exists
after excluding the current target-station measurements.  Does not
invent Cov(pred,target), retune C/Q, drop 100043/37, or enter V2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from alignment.faseracts_propagated_covariance_validation_v2 import FROZEN_WB81_GATES
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from datasets.acts_transport_tail_analysis import identity_key
from datasets.transport_uncertainty_shape_diagnosis import (
    CASE_E as WB105_CASE,
    inherit_frozen_stage as inherit_through_wb104,
    load_contracted_sample,
)

SCHEMA_VERSION = "leave-target-out-prediction-contract-v1"
DEFAULT_CONFIG = "configs/leave_target_out_prediction_contract_v1.yaml"
TASK = "SB-B10"
WORKBOOK = 106

DECISION_NOT_ESTABLISHED = "leave_target_out_prediction_contract_not_established"
DECISION_ESTABLISHED = "leave_target_out_prediction_contract_established"

DUMP_LTO_FIELDS = (
    "leave_target_out_state",
    "leave_target_out_covariance",
    "lto_input_covariance",
    "excluded_target_station",
)


class LeaveTargetOutError(ValueError):
    """Raised when the B10 contract is illegal."""


def refuse_truth_qoverp() -> None:
    raise LeaveTargetOutError("truth q/p is not a real-data solution")


def refuse_empirical_cross_covariance() -> None:
    raise LeaveTargetOutError("empirical Cov(pred,target) must not be invented")


def refuse_measurement_model_v2() -> None:
    raise LeaveTargetOutError("Measurement Model V2 is not entered in Task B10")


def refuse_raw_ckf() -> None:
    raise LeaveTargetOutError("raw CKF must not re-enter the official B10 input")


def refuse_focus_drop() -> None:
    raise LeaveTargetOutError("focus identity 100043/37 must be retained")


def refuse_tighten_contract() -> None:
    raise LeaveTargetOutError("reconstruction contract must not be tightened")


def refuse_kalmanfitter_as_lto() -> None:
    raise LeaveTargetOutError(
        "KalmanFitterTool.fit cannot be treated as leave-target-out"
    )


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise LeaveTargetOutError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise LeaveTargetOutError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise LeaveTargetOutError(f"workbook must be {WORKBOOK}")
    if str(config.get("official_input_scope")) != "contract_eligible":
        raise LeaveTargetOutError("official input must be contract_eligible")
    for key in (
        "geometry_write_allowed",
        "held_out_accessed",
        "real_data_alignment_authorized",
        "measurement_model_validated",
    ):
        if bool(config.get(key, True)):
            raise LeaveTargetOutError(f"{key} must be false")
    for key in (
        "do_not_rescale_covariance",
        "do_not_use_truth_q_over_p_as_real_data_solution",
        "do_not_invent_empirical_cross_covariance",
        "do_not_use_raw_ckf_as_official_input",
        "do_not_tighten_reconstruction_contract",
        "do_not_drop_focus_identity",
        "do_not_enter_measurement_model_v2",
        "do_not_use_kalmanfittertool_fit_as_leave_target_out",
        "eligibility_independent_of_closure",
    ):
        if bool(config.get(key, False)) is not True:
            raise LeaveTargetOutError(f"{key} must be true")
    gates = config.get("closure_gates", {})
    for key, expected in FROZEN_WB81_GATES.items():
        if gates.get(key) != expected:
            raise LeaveTargetOutError(f"WB81/WB87/WB98 gate must stay frozen: {key}")
    wb103 = yaml.safe_load(
        resolve_under_root(
            project_root(), config["inheritance"]["workbook_103"]["config_path"]
        ).read_text(encoding="utf-8")
    )
    if dict(config["eligibility"]) != dict(wb103["eligibility"]):
        raise LeaveTargetOutError("B10 eligibility must stay identical to WB103")
    if Path(str(config.get("output_root"))).as_posix() == Path(
        str(config.get("dump_root"))
    ).as_posix():
        raise LeaveTargetOutError("B10 must not write into the WB98 dump root")
    return dict(config)


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb104(config)
    spec = config["inheritance"]["workbook_105"]
    decision = json.loads(
        resolve_under_root(project_root(), spec["decision_path"]).read_text(encoding="utf-8")
    )
    digest = sha256_file(resolve_under_root(project_root(), spec["config_path"]))
    if digest != spec["config_sha256"]:
        raise LeaveTargetOutError(f"workbook_105 config hash mismatch: {digest}")
    digest = sha256_file(resolve_under_root(project_root(), spec["decision_path"]))
    if digest != spec["decision_sha256"]:
        raise LeaveTargetOutError(f"workbook_105 decision hash mismatch: {digest}")
    if decision.get("decision") != spec["frozen_decision"]:
        raise LeaveTargetOutError("workbook_105 decision must stay frozen")
    if decision.get("primary_case") != spec["frozen_primary_case"]:
        raise LeaveTargetOutError("workbook_105 primary case must stay frozen")
    if spec["frozen_primary_case"] != WB105_CASE:
        raise LeaveTargetOutError("WB105 primary case token mismatch")
    inherited["workbook_105"] = {
        "decision": spec["frozen_decision"],
        "primary_case": spec["frozen_primary_case"],
        "config_sha256": spec["config_sha256"],
        "decision_sha256": spec["decision_sha256"],
    }
    return inherited


def _calypso_root(config: Mapping[str, Any]) -> Path:
    return Path(str(config["software_provenance"]["calypso_root"]))


def catalog_calypso_interfaces(config: Mapping[str, Any]) -> dict[str, Any]:
    root = _calypso_root(config)
    sources = {}
    for name, spec in config["pinned_calypso_sources"].items():
        path = root / spec["path"]
        digest = sha256_file(path) if path.is_file() else None
        if digest != spec["sha256"]:
            raise LeaveTargetOutError(f"pinned Calypso source drifted: {name}")
        sources[name] = {
            "path": spec["path"],
            "sha256": digest,
            "present": path.is_file(),
        }
    return {
        "calypso_git_sha": config["software_provenance"]["calypso_git_sha"],
        "athena_release": config["software_provenance"]["athena_release"],
        "acts_version": config["software_provenance"]["acts_version"],
        "sources": sources,
        "interfaces": {
            "kalman_fitter_tool_fit": {
                "exists": True,
                "exports_leave_target_out_source_state": False,
                "uses_all_measurements_on_track": True,
                "outlier_finder_cluster_z": -1_000_000.0,
                "excludes_ift_z_lt_minus_100mm": True,
                "includes_stations_1_2_3": True,
                "seed_covariance_scale_default": 100.0,
                "extra_seed_scale_in_fit": 10.0,
                "note": (
                    "KalmanFitterTool.fit always rebuilds measurements from the "
                    "full Trk::Track and marks IFT (z < -100 mm) as outliers.  "
                    "That is leave-source-out, not leave-target-out."
                ),
            },
            "get_unbiased_residual_cluster_z": {
                "exists": True,
                "exports_leave_target_out_source_5x5": False,
                "excludes_typical_target_station": False,
                "note": (
                    "FaserActsOutlierFinder only special-cases cluster_z < -10000.  "
                    "A target-station z such as 47.4 / 1237.4 / 2427.4 is not excluded."
                ),
            },
            "get_unbiased_residual_ift_cluster_list": {
                "exists": True,
                "exports_leave_target_out_source_5x5": False,
                "get_measurements_from_track_replaces_hits": False,
                "note": (
                    "The cluster-list overload appends the supplied clusters and "
                    "then still adds every measurementsOnTrack hit."
                ),
            },
            "create_trk_track_front_state": {
                "exists": True,
                "measurement_tsos": "smoothed",
                "outlier_tsos": "filtered",
                "hole_tsos": "predicted",
                "refit_create_track_adds_fitted_params": False,
            },
            "official_ckf_track_collection": {
                "collection": "CKFTrackCollection",
                "preferred_payload": "KalmanFitterTool.fit result when non-null",
                "fallback_payload": "CKF2 CreateTrkTrackTool track, with optional fittedParams Hole at front",
            },
        },
        "station_level_leave_target_out_export_exists": False,
        "perigee_unbiased_lto_refit_export_exists": False,
    }


def audit_dump_lto_fields(sample: Mapping[str, Any]) -> dict[str, Any]:
    n = 0
    present = {name: 0 for name in DUMP_LTO_FIELDS}
    for row in sample["contracted"]:
        n += 1
        for name in DUMP_LTO_FIELDS:
            if row.get(name) is not None:
                present[name] += 1
    return {
        "n_contracted": n,
        "lto_field_counts": present,
        "lto_states_materialized": False,
        "lto_fields_absent": all(count == 0 for count in present.values()),
        "dump_helper": "CkfActsTransportDumpAlg",
        "dump_source_state": "Trk::Track.trackParameters().front()",
        "note": (
            "The official WB98 dump exports full-track Cin from "
            "trackParameters().front().  It has no leave-target-out state."
        ),
    }


def audit_target_independence(config: Mapping[str, Any], sample: Mapping[str, Any]) -> dict[str, Any]:
    n_official = 0
    n_target_in_tracklets = 0
    n_has_lto = 0
    for event in sample["official_events"]:
        reco = sample["reco_by_identity"].get(identity_key(event)) or {}
        stations = {int(v) for v in reco.get("tracklet_stations") or [] if v is not None}
        target = int(event["target_station"])
        n_official += 1
        if target in stations:
            n_target_in_tracklets += 1
    for row in sample["contracted"]:
        if any(row.get(name) is not None for name in DUMP_LTO_FIELDS):
            n_has_lto += 1
    fraction = float(n_target_in_tracklets / n_official) if n_official else None
    return {
        "n_official_pairs": n_official,
        "n_contracted": len(sample["contracted"]),
        "target_station_in_same_ckf_fit_fraction": fraction,
        "prediction_and_target_measurement_independent": False,
        "lto_states_available": n_has_lto > 0,
        "cov_pred_target_estimated": False,
        "cov_pred_target": None,
        "empirical_cross_covariance_invented": False,
        "official_residual_partner": "truth_not_target_measurement",
        "independence_proven": False,
        "note": (
            "Contracted long tracks have four tracklet stations, so the dumped "
            "Cin already includes the target station.  No leave-target-out state "
            "is present to prove the opposite."
        ),
    }


def build_state_inventory(sample: Mapping[str, Any]) -> dict[str, Any]:
    by_split = {"construction": 0, "validation": 0}
    for row in sample["contracted"]:
        by_split[str(row.get("split"))] = by_split.get(str(row.get("split")), 0) + 1
    return {
        "n_raw": sample["n_raw"],
        "n_ineligible": sample["n_ineligible"],
        "n_contracted": len(sample["contracted"]),
        "n_official_pairs": len(sample["official_events"]),
        "n_lto_states": 0,
        "n_lto_covariances": 0,
        "rows_by_split": by_split,
        "present_sources": list(sample["present_sources"]),
        "missing_sources": list(sample["missing_sources"]),
        "focus_identity_retained": True,
        "official_state_kind": "full_track_ckf_front",
        "leave_target_out_state_kind": None,
    }


def build_covariance_contract(interfaces: Mapping[str, Any], dump: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "official_cin_source": "Trk::Track.trackParameters().front() of CKFTrackCollection",
        "official_cin_is_leave_target_out": False,
        "official_fit_excludes_ift": True,
        "official_fit_includes_target_stations": True,
        "required_lto_contract": {
            "exclude_current_target_station_measurements": True,
            "include_source_station_measurements": True,
            "export_5x5_with_qoverp_cross_terms": True,
            "same_geometry_field_material_conditions": True,
            "truth_qoverp_forbidden_in_fitter": True,
        },
        "available_lto_export": False,
        "kalmanfittertool_fit_satisfies_required_contract": False,
        "get_unbiased_residual_satisfies_required_contract": False,
        "lto_states_materialized": bool(dump.get("lto_states_materialized")),
        "empirical_cross_covariance_used_as_substitute": False,
        "interfaces": interfaces["interfaces"],
    }


def decide_case(
    interfaces: Mapping[str, Any],
    dump: Mapping[str, Any],
    independence: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    established = bool(
        interfaces.get("station_level_leave_target_out_export_exists")
        and dump.get("lto_states_materialized")
        and independence.get("independence_proven")
        and inventory.get("n_lto_states")
    )
    if bool(independence.get("empirical_cross_covariance_invented")):
        refuse_empirical_cross_covariance()
    if established:
        return {
            "verdict": "ESTABLISHED",
            "decision": DECISION_ESTABLISHED,
            "primary_case": DECISION_ESTABLISHED,
            "next_step": "transport_covariance_v4_falsification",
        }
    return {
        "verdict": "NOT_ESTABLISHED",
        "decision": DECISION_NOT_ESTABLISHED,
        "primary_case": DECISION_NOT_ESTABLISHED,
        "secondary_cases": [
            "calypso_has_cluster_unbiased_residual_only",
            "official_fit_is_leave_source_ift_out_not_leave_target_out",
        ],
        "next_step": "independent_leave_target_out_helper_not_kalmanfittertool_fit",
    }


def inventory_and_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    sample = load_contracted_sample(config)
    interfaces = catalog_calypso_interfaces(config)
    dump = audit_dump_lto_fields(sample)
    independence = audit_target_independence(config, sample)
    inventory = build_state_inventory(sample)
    covariance = build_covariance_contract(interfaces, dump)
    mechanism = decide_case(interfaces, dump, independence, inventory)
    return {
        "present_sources": sample["present_sources"],
        "missing_sources": sample["missing_sources"],
        "n_contracted": len(sample["contracted"]),
        "n_official_pairs": len(sample["official_events"]),
        "provenance_hashes": sample["provenance_hashes"],
        "interfaces": interfaces,
        "dump": dump,
        "independence": independence,
        "inventory": inventory,
        "covariance": covariance,
        "mechanism": mechanism,
    }


def decide(inventory: Mapping[str, Any], inherited: Mapping[str, Any]) -> dict[str, Any]:
    mechanism = inventory["mechanism"]
    if bool(inventory["independence"].get("empirical_cross_covariance_invented")):
        refuse_empirical_cross_covariance()
    if bool(inventory["covariance"].get("kalmanfittertool_fit_satisfies_required_contract")):
        refuse_kalmanfitter_as_lto()
    return {
        "verdict": mechanism["verdict"],
        "decision": mechanism["decision"],
        "primary_case": mechanism["primary_case"],
        "secondary_cases": mechanism.get("secondary_cases", []),
        "next_step": mechanism["next_step"],
        "measurement_model_v2_authorized": False,
        "measurement_model_v2_entered": False,
        "geometry_write_allowed": False,
        "official_input_scope": "contract_eligible",
        "raw_ckf_used": False,
        "focus_identity_retained": True,
        "empirical_cross_covariance_invented": False,
        "lto_states_materialized": False,
        "independence_proven": False,
        "inherited_wb105_decision": inherited["workbook_105"]["decision"],
        "inherited_wb105_decision_sha256": inherited["workbook_105"]["decision_sha256"],
        "inherited_wb104_decision_sha256": inherited["workbook_104"]["decision_sha256"],
        "inherited_wb103_contract_sha256": inherited["workbook_103"]["contract_sha256"],
        "wb96_through_wb105_rewritten": False,
    }
