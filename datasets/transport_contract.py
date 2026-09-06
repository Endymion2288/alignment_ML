"""T11 transport / covariance chart contract.

Mathematical conversions are tested here.  Physical production C_prop is the
frozen WB87 negative result and is not reopened or rescaled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment.numerical_contract import require_spd

SCHEMA_VERSION = "transport-covariance-contract-v1"
DEFAULT_CONFIG = "configs/transport_covariance_contract_v1.yaml"
STATE_NAMES = ("x_mm", "y_mm", "tx", "ty")
QOP_NATIVE_UNIT = "per_MeV"
MEV_PER_GEV = 1000.0
WB87_DECISION = "faseracts_transport_covariance_not_validated"
WB87_MECHANISM = "q_over_p_uncertainty_semantics"


class TransportContractError(ValueError):
    """Raised when a transport-chart conversion is illegal or a repair is requested."""


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise TransportContractError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != "T11":
        raise TransportContractError("task must be T11")
    if bool(config.get("geometry_write_allowed", True)):
        raise TransportContractError("geometry_write_allowed must be false")
    if bool(config.get("do_not_rescale_covariance", False)) is not True:
        raise TransportContractError("covariance rescale is forbidden")
    if bool(config.get("do_not_use_truth_q_over_p_as_deployment_seed", False)) is not True:
        raise TransportContractError("truth q/p is not a deployment seed")
    return dict(config)


def geometric_jacobian(lever_arm_mm: float) -> np.ndarray:
    """Field-free F: x' = x + L tx, y' = y + L ty, slopes unchanged."""
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 2] = float(lever_arm_mm)
    matrix[1, 3] = float(lever_arm_mm)
    return matrix


def transport_covariance(
    source: np.ndarray,
    jacobian: np.ndarray,
    process_noise: np.ndarray | None = None,
) -> np.ndarray:
    cov = require_spd(source, name="source covariance")
    jac = np.asarray(jacobian, dtype=np.float64)
    if jac.shape != cov.shape:
        raise TransportContractError("F and C must have the same square shape")
    transported = jac @ cov @ jac.T
    if process_noise is not None:
        transported = transported + require_spd(process_noise, name="process noise")
    return require_spd(transported, name="transported covariance")


def propagate_state(state: np.ndarray, lever_arm_mm: float) -> np.ndarray:
    x_mm, y_mm, tx, ty = (float(item) for item in state)
    return np.array(
        [x_mm + float(lever_arm_mm) * tx, y_mm + float(lever_arm_mm) * ty, tx, ty],
        dtype=np.float64,
    )


def numerical_jacobian(state: np.ndarray, lever_arm_mm: float, step: float) -> np.ndarray:
    base = np.asarray(state, dtype=np.float64)
    columns = []
    for index in range(base.size):
        plus = np.array(base, copy=True)
        minus = np.array(base, copy=True)
        plus[index] += step
        minus[index] -= step
        columns.append(
            (propagate_state(plus, lever_arm_mm) - propagate_state(minus, lever_arm_mm))
            / (2.0 * step)
        )
    return np.column_stack(columns)


def q_over_p_per_mev_from_gev(q_over_p_per_gev: float) -> float:
    return float(q_over_p_per_gev) / MEV_PER_GEV


def q_over_p_per_gev_from_mev(q_over_p_per_mev: float) -> float:
    return float(q_over_p_per_mev) * MEV_PER_GEV


def signed_q_over_p(charge: float, p_mev: float) -> float:
    if p_mev == 0.0:
        raise TransportContractError("momentum must be non-zero")
    return float(charge) / float(p_mev)


def slope_from_direction(direction: np.ndarray) -> tuple[float, float]:
    vector = np.asarray(direction, dtype=np.float64).reshape(3)
    if abs(float(vector[2])) < 1.0e-12:
        raise TransportContractError("cannot form slopes when tz ~ 0")
    return float(vector[0] / vector[2]), float(vector[1] / vector[2])


def refuse_covariance_rescale() -> None:
    raise TransportContractError("core-width rescale is not a transport repair")


def inherit_wb87(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = config["workbook_87"]
    decision_path = resolve_under_root(project_root(), spec["decision_path"])
    config_path = resolve_under_root(project_root(), spec["config_path"])
    decision_sha = sha256_file(decision_path)
    config_sha = sha256_file(config_path)
    if decision_sha != spec["decision_sha256"]:
        raise TransportContractError("WB87 decision hash mismatch")
    if config_sha != spec["config_sha256"]:
        raise TransportContractError("WB87 config hash mismatch")
    payload = json.loads(decision_path.read_text(encoding="utf-8"))
    if payload.get("decision") != WB87_DECISION:
        raise TransportContractError("WB87 decision must stay frozen")
    return {
        "kind": "inherited_wb87",
        "decision": payload["decision"],
        "mechanism": payload["failure_classification"]["category"],
        "decision_sha256": decision_sha,
        "config_sha256": config_sha,
        "jacobian_self_consistency_calibrated": bool(
            payload.get("jacobian_self_consistency_calibrated")
        ),
        "process_noise_calibrated": bool(payload.get("process_noise_calibrated")),
        "whitening_closure_recovered": bool(payload.get("whitening_closure_recovered")),
        "measurement_model_validated": False,
        "truth_qoverp_used_as_real_data_solution": False,
        "qoverp_column_deleted_as_final_scheme": False,
        "covariance_tuned_to_chi2": False,
        "physical_transport_fd_rerun": False,
        "eighteen_source_jobs_submitted": False,
    }


def decide(config: Mapping[str, Any], math_pass: bool, inherited: Mapping[str, Any]) -> dict[str, Any]:
    physical_ok = inherited["decision"] != WB87_DECISION
    verdict = "PASS" if math_pass and physical_ok else "FAIL"
    return {
        "kind": "transport_covariance_contract_decision",
        "task": "T11",
        "verdict": verdict,
        "math_conversion_pass": bool(math_pass),
        "physical_transport_pass": False,
        "inherited_decision": inherited["decision"],
        "inherited_mechanism": inherited["mechanism"],
        "contract_established_for_t12": False,
        "geometry_write_allowed": False,
        "held_out_accessed": False,
        "do_not_rescale_covariance": True,
        "do_not_use_truth_q_over_p_as_deployment_seed": True,
        "reason": (
            "mathematical conversions pass, but production C_prop remains the "
            "frozen WB87 negative and is not a validated transport covariance"
        ),
    }
