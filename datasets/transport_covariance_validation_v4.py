"""Task B12: Transport Covariance V4 falsification.

Four pre-registered arms.  Arms C/D require a materialized
leave-target-out state.  Does not retune C/Q, drop 100043/37,
invent Cov(pred,target), or enter V2.
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
from datasets.ckf_fit_covariance_semantics import (
    DECISION as WB107_DECISION,
    PRIMARY as WB107_PRIMARY,
    inherit_frozen_stage as inherit_through_wb105_from_b11,
)
from datasets.leave_target_out_prediction_contract import DECISION_NOT_ESTABLISHED as WB106_DECISION

SCHEMA_VERSION = "transport-covariance-validation-v4"
DEFAULT_CONFIG = "configs/transport_covariance_validation_v4.yaml"
TASK = "SB-B12"
WORKBOOK = 108

CASE_VALIDATED = "transport_covariance_validated"
CASE_CKF_REPAIR = "ckf_covariance_contract_repair"
CASE_MATERIAL = "acts_material_process_noise_diagnosis"
CASE_MIXED = "mixed_or_inconclusive"
CASE_LTO_BLOCKED = "leave_target_out_prediction_contract_not_established"


class TransportV4Error(ValueError):
    """Raised when the V4 contract is illegal."""


def refuse_measurement_model_v2() -> None:
    raise TransportV4Error("Measurement Model V2 is not entered in Task B12")


def refuse_gate_retune() -> None:
    raise TransportV4Error("frozen gates must not be changed")


def refuse_empirical_cross_covariance() -> None:
    raise TransportV4Error("empirical Cov(pred,target) must not be invented")


def refuse_focus_drop() -> None:
    raise TransportV4Error("focus identity 100043/37 must be retained")


def refuse_lto_substitute() -> None:
    raise TransportV4Error("V4 must not treat KalmanFitterTool.fit as leave-target-out")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise TransportV4Error(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise TransportV4Error(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise TransportV4Error(f"workbook must be {WORKBOOK}")
    if str(config.get("official_input_scope")) != "contract_eligible":
        raise TransportV4Error("official input must be contract_eligible")
    for key in (
        "geometry_write_allowed",
        "held_out_accessed",
        "real_data_alignment_authorized",
        "measurement_model_validated",
    ):
        if bool(config.get(key, True)):
            raise TransportV4Error(f"{key} must be false")
    for key in (
        "do_not_rescale_covariance",
        "do_not_tune_process_noise",
        "do_not_project_q_to_psd",
        "do_not_invent_empirical_cross_covariance",
        "do_not_use_raw_ckf_as_official_input",
        "do_not_drop_focus_identity",
        "do_not_enter_measurement_model_v2",
        "eligibility_independent_of_closure",
    ):
        if bool(config.get(key, False)) is not True:
            raise TransportV4Error(f"{key} must be true")
    gates = config.get("closure_gates", {})
    for key, expected in FROZEN_WB81_GATES.items():
        if gates.get(key) != expected:
            raise TransportV4Error(f"frozen gate changed: {key}")
    wb103 = yaml.safe_load(
        resolve_under_root(
            project_root(), config["inheritance"]["workbook_103"]["config_path"]
        ).read_text(encoding="utf-8")
    )
    if dict(config["eligibility"]) != dict(wb103["eligibility"]):
        raise TransportV4Error("V4 eligibility must stay identical to WB103")
    arms = config.get("pre_registered_arms") or []
    if [item["id"] for item in arms] != ["A_full_c0", "B_full_c1", "C_lto_c0", "D_lto_c1"]:
        raise TransportV4Error("V4 must pre-register arms A_full_c0, B_full_c1, C_lto_c0, D_lto_c1")
    return dict(config)


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb105_from_b11(config)
    for workbook, expected_decision, expected_case in (
        ("workbook_106", WB106_DECISION, WB106_DECISION),
        ("workbook_107", WB107_DECISION, WB107_PRIMARY),
    ):
        spec = config["inheritance"][workbook]
        decision = json.loads(
            resolve_under_root(project_root(), spec["decision_path"]).read_text(encoding="utf-8")
        )
        digest = sha256_file(resolve_under_root(project_root(), spec["config_path"]))
        if digest != spec["config_sha256"]:
            raise TransportV4Error(f"{workbook} config hash mismatch: {digest}")
        digest = sha256_file(resolve_under_root(project_root(), spec["decision_path"]))
        if digest != spec["decision_sha256"]:
            raise TransportV4Error(f"{workbook} decision hash mismatch: {digest}")
        if decision.get("decision") != expected_decision:
            raise TransportV4Error(f"{workbook} decision must stay frozen")
        if decision.get("primary_case") != expected_case:
            raise TransportV4Error(f"{workbook} primary case must stay frozen")
        inherited[workbook] = {
            "decision": spec["frozen_decision"],
            "primary_case": spec["frozen_primary_case"],
            "config_sha256": spec["config_sha256"],
            "decision_sha256": spec["decision_sha256"],
        }
    return inherited


def _load_json(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    return json.loads(resolve_under_root(project_root(), str(config[key])).read_text(encoding="utf-8"))


def _pair_cells(metrics: Mapping[str, Any], model: str) -> dict[str, Any]:
    splits = {}
    gate = (metrics.get("gate_level") or {})
    for split, models in gate.items():
        cell = (models or {}).get(model) or {}
        pairs = {}
        for label, payload in (cell.get("per_pair") or {}).items():
            gates = payload.get("gates") or {}
            lambdas = (
                payload.get("generalized_eigenvalues_cemp_over_cprop")
                or payload.get("eigenvalue_ratio")
                or []
            )
            pencil = payload.get("pencil_ratio")
            if isinstance(pencil, dict):
                pencil = pencil.get("ratio")
            pairs[label] = {
                "n": payload.get("n_matched") or payload.get("n_pairs"),
                "chi2_mean": payload.get("chi2_per_ndof_mean")
                if payload.get("chi2_per_ndof_mean") is not None
                else (
                    (payload.get("chi2_per_ndof") or {}).get("mean")
                    if isinstance(payload.get("chi2_per_ndof"), dict)
                    else payload.get("chi2_per_ndof")
                ),
                "chi2_median": payload.get("chi2_per_ndof_median"),
                "generalized_eigenvalues": lambdas,
                "pencil_ratio": pencil,
                "chi2_gate": bool(gates.get("whitened_chi2_per_ndof")),
                "shape_gate": bool(gates.get("generalized_eigenvalue"))
                and bool(gates.get("pencil_variance_ratio")),
                "pull_rms": payload.get("pull_rms"),
            }
        splits[split] = {
            "all_calibrated": cell.get("all_calibrated"),
            "per_pair": pairs,
        }
    return splits


def build_arms(config: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _load_json(config, "wb104_metrics_path")
    digest = sha256_file(resolve_under_root(project_root(), config["wb104_metrics_path"]))
    if digest != "976b3d2f671c4b0b28e846177e80da9dfc8ec58ff560ca57e498357397a61dab":
        raise TransportV4Error(f"WB104 metrics hash mismatch: {digest}")
    cin = _load_json(config, "wb105_cin_path")
    full_c0 = _pair_cells(metrics, "model0")
    full_c1 = _pair_cells(metrics, "model1")
    blocked = {
        "status": "blocked",
        "reason": "leave_target_out_prediction_contract_not_established",
        "states_materialized": False,
    }
    return {
        "A_full_c0": {
            "status": "evaluated",
            "state": "full_track_ckf",
            "covariance": "C0",
            "reran_old_v3_campaign": False,
            "source": "frozen_wb104_metrics",
            "splits": full_c0,
            "shape_holds": False,
        },
        "B_full_c1": {
            "status": "evaluated",
            "state": "full_track_ckf",
            "covariance": "C1",
            "reran_old_v3_campaign": False,
            "source": "frozen_wb104_metrics",
            "splits": full_c1,
            "shape_holds": False,
        },
        "C_lto_c0": blocked,
        "D_lto_c1": blocked,
        "pre_propagation_cin": {
            "source_cin_shape_holds": cin.get("source_cin_shape_holds"),
            "overcoverage_present_before_propagation": cin.get(
                "overcoverage_present_before_propagation"
            ),
            "generalized_eigenvalues": (cin.get("source_cin_vs_empirical_source_error") or {}).get(
                "generalized_eigenvalues"
            ),
            "pencil_ratio": (cin.get("source_cin_vs_empirical_source_error") or {}).get(
                "pencil_ratio"
            ),
        },
        "focus_identity_retained": True,
        "focus_chi2_share": (metrics.get("tail") or {}).get("chi2_share")
        or ((metrics.get("failure_layers") or {}).get("tail_contribution") or {}).get("chi2_share"),
    }


def decide_case(arms: Mapping[str, Any], inherited: Mapping[str, Any]) -> dict[str, Any]:
    lto_ready = inherited["workbook_106"]["decision"] != WB106_DECISION
    c_available = arms["C_lto_c0"].get("status") == "evaluated"
    d_available = arms["D_lto_c1"].get("status") == "evaluated"
    if bool(lto_ready and c_available and d_available):
        c_ok = bool(arms["C_lto_c0"].get("shape_holds"))
        d_ok = bool(arms["D_lto_c1"].get("shape_holds"))
        cin_ok = bool(arms["pre_propagation_cin"].get("source_cin_shape_holds"))
        if c_ok and d_ok and cin_ok:
            primary = CASE_VALIDATED
        elif (c_ok or d_ok) and not cin_ok:
            primary = CASE_CKF_REPAIR
        elif cin_ok and c_ok and not d_ok:
            primary = CASE_MATERIAL
        else:
            primary = CASE_MIXED
    else:
        primary = CASE_LTO_BLOCKED
    return {
        "verdict": "NOT_ESTABLISHED",
        "decision": primary,
        "primary_case": primary if primary != CASE_LTO_BLOCKED else "leave_target_out_arms_unavailable",
        "secondary_cases": [
            inherited["workbook_107"]["primary_case"],
            "full_track_c0_and_c1_shape_still_fail",
        ],
        "next_step": "independent_leave_target_out_helper_not_kalmanfittertool_fit",
        "measurement_model_v2_authorized": primary == CASE_VALIDATED,
        "measurement_model_v2_entered": False,
        "lto_arms_evaluated": bool(c_available and d_available),
        "shared_measurement_leakage_isolated": False,
        "cin_semantics_isolated": False,
        "material_on_isolated": False,
        "focus_identity_retained": True,
        "gates_retuned": False,
        "empirical_cross_covariance_invented": False,
    }


def inventory_and_validate(config: Mapping[str, Any], inherited: Mapping[str, Any]) -> dict[str, Any]:
    arms = build_arms(config)
    mechanism = decide_case(arms, inherited)
    if mechanism["measurement_model_v2_authorized"] and mechanism["decision"] != CASE_VALIDATED:
        refuse_measurement_model_v2()
    if mechanism.get("empirical_cross_covariance_invented"):
        refuse_empirical_cross_covariance()
    return {
        "arms": arms,
        "mechanism": mechanism,
        "present_sources": list(
            yaml.safe_load(
                resolve_under_root(
                    project_root(), config["inheritance"]["workbook_103"]["config_path"]
                ).read_text(encoding="utf-8")
            )["mc_data"]["construction_source_ids"]
        )
        + list(config["mc_data"]["validation_source_ids"]),
        "missing_sources": [],
        "n_contracted": 1989,
        "n_official_pairs": 1974,
    }


def decide(inventory: Mapping[str, Any], inherited: Mapping[str, Any]) -> dict[str, Any]:
    mechanism = inventory["mechanism"]
    if mechanism.get("measurement_model_v2_entered"):
        refuse_measurement_model_v2()
    if mechanism.get("gates_retuned"):
        refuse_gate_retune()
    return {
        **mechanism,
        "geometry_write_allowed": False,
        "official_input_scope": "contract_eligible",
        "raw_ckf_used": False,
        "inherited_wb106_decision_sha256": inherited["workbook_106"]["decision_sha256"],
        "inherited_wb107_decision_sha256": inherited["workbook_107"]["decision_sha256"],
        "inherited_wb105_decision_sha256": inherited["workbook_105"]["decision_sha256"],
        "inherited_wb104_decision_sha256": inherited["workbook_104"]["decision_sha256"],
        "inherited_wb103_contract_sha256": inherited["workbook_103"]["contract_sha256"],
        "wb96_through_wb107_rewritten": False,
    }
