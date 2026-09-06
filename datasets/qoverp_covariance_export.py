"""Task A2: CKF physical q/p covariance export contract.

Validates a reconstruction-chain 5x5 (estimate + uncertainty + cross terms).
Dummy SegmentFit covariance and truth q/p cannot pass.  Missing dumps fail
closed.  This is not alignment and does not enter Task B by itself.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from alignment.numerical_contract import CovarianceNotSPDError, is_spd, require_spd
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment.segmentfit_covariance_coordinate_contract import (
    curvilinear_uvt,
    exporter_jacobian,
)
from datasets.access_policy import AccessScope, authorize_path
from datasets.qoverp_semantics import (
    DECISION_NOT_ESTABLISHED as WB95_DECISION,
    MECHANISM_NOT_EXPORTED as WB95_MECHANISM,
    SEGMENTFIT_DUMMY_QOVERP_PER_MEV,
    dummy_segmentfit_variance_per_mev2,
)
from datasets.transport_contract import MEV_PER_GEV

SCHEMA_VERSION = "qoverp-covariance-export-v1"
DEFAULT_CONFIG = "configs/qoverp_covariance_export_v1.yaml"
TASK = "SB-A2"
WORKBOOK = 96

NATIVE_STATE = ("loc1_mm", "loc2_mm", "phi", "theta", "q_over_p_per_mev")
DERIVED_STATE = ("x_mm", "y_mm", "tx", "ty", "q_over_p_per_mev")
QOVERP_INDEX = 4

DECISION_ESTABLISHED = "ckf_qoverp_covariance_export_established"
DECISION_NOT_ESTABLISHED = "ckf_qoverp_covariance_export_not_established"
MECHANISM_NOT_MATERIALIZED = "ckf_5x5_export_not_materialized"
MECHANISM_COV_ABSENT = "ckf_covariance_absent_on_trackparameters"
MECHANISM_DUMMY = "ckf_covariance_matches_segmentfit_dummy"
MECHANISM_NOT_SPD = "ckf_exported_covariance_not_spd"
MECHANISM_NO_CROSS = "ckf_qoverp_cross_terms_missing"
MECHANISM_TRUTH = "truth_qoverp_forbidden"


class QoverPExportError(ValueError):
    """Raised when the A2 export contract is illegal."""


def state_definition() -> dict[str, Any]:
    return {
        "native_athena": list(NATIVE_STATE),
        "native_frame": "curvilinear_or_bound_trackparameters",
        "derived_export": list(DERIVED_STATE),
        "derived_frame": "global_cartesian_slopes",
        "q_over_p_native_unit": "per_MeV",
        "q_over_p_signed": True,
        "loc_unit": "mm",
        "angle_unit": "rad",
        "mev_per_gev": MEV_PER_GEV,
        "covariance_dimension": 5,
    }


def provenance_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    spec = config["software_provenance"]
    tags = config["tags"]
    calypso = spec["calypso_git_sha"]
    geometry = f"{tags['geometry']}|{calypso}"
    field = f"{tags['field']}|{calypso}"
    conditions = f"{tags['conditions']}|{tags['database_instance']}|{calypso}"
    return {
        "geometry_hash": hashlib.sha256(geometry.encode("utf-8")).hexdigest(),
        "field_hash": hashlib.sha256(field.encode("utf-8")).hexdigest(),
        "conditions_hash": hashlib.sha256(conditions.encode("utf-8")).hexdigest(),
        "geometry_tag": tags["geometry"],
        "field_tag": tags["field"],
        "conditions_tag": tags["conditions"],
        "calypso_git_sha": calypso,
        "athena_release": spec["athena_release"],
        "acts_version": spec["acts_version"],
    }


def derived_jacobian_5d(
    phi: float,
    theta: float,
    ref_pos: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Map native (loc1, loc2, phi, theta, q/p) to (x, y, tx, ty, q/p)."""
    curv_u, curv_v, _ = curvilinear_uvt(direction)
    jac4 = exporter_jacobian(float(phi), float(theta), ref_pos, curv_u, curv_v)
    jac = np.eye(5, dtype=np.float64)
    jac[:4, :4] = jac4
    return jac


def transport_native_covariance(
    native_cov: np.ndarray,
    phi: float,
    theta: float,
    ref_pos: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    jac = derived_jacobian_5d(phi, theta, ref_pos, direction)
    cov = require_spd(native_cov, name="native CKF covariance")
    return require_spd(jac @ cov @ jac.T, name="derived CKF covariance")


def refuse_truth_qoverp() -> None:
    raise QoverPExportError("truth q/p is not a real-data solution")


def refuse_dummy_segmentfit() -> None:
    raise QoverPExportError("dummy SegmentFit covariance is not a physical prior")


def refuse_synthetic_injection() -> None:
    raise QoverPExportError("synthetic q/p injection is forbidden")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise QoverPExportError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise QoverPExportError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise QoverPExportError(f"workbook must be {WORKBOOK}")
    for key in (
        "geometry_write_allowed",
        "held_out_accessed",
        "real_data_alignment_authorized",
        "measurement_model_validated",
    ):
        if bool(config.get(key, True)):
            raise QoverPExportError(f"{key} must be false")
    for key in (
        "do_not_rescale_covariance",
        "do_not_use_truth_q_over_p_as_real_data_solution",
        "do_not_use_dummy_segmentfit_covariance",
        "do_not_inject_synthetic_qoverp",
        "do_not_enter_alignment",
        "do_not_enter_measurement_model_v2",
        "do_not_enter_task_b_unless_pass",
        "do_not_start_new_source_campaign",
        "unconstrained_tracker_only_stopped",
    ):
        if bool(config.get(key, False)) is not True:
            raise QoverPExportError(f"{key} must be true")
    return dict(config)


def _expect_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise QoverPExportError(f"{label} hash mismatch")


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    spec87 = config["inheritance"]["workbook_87"]
    decision87 = json.loads(
        resolve_under_root(project_root(), spec87["decision_path"]).read_text(encoding="utf-8")
    )
    _expect_sha(
        resolve_under_root(project_root(), spec87["config_path"]),
        spec87["config_sha256"],
        "WB87 config",
    )
    _expect_sha(
        resolve_under_root(project_root(), spec87["decision_path"]),
        spec87["decision_sha256"],
        "WB87 decision",
    )
    if decision87.get("decision") != spec87["frozen_decision"]:
        raise QoverPExportError("WB87 decision must stay frozen")

    spec95 = config["inheritance"]["workbook_95"]
    decision95 = json.loads(
        resolve_under_root(project_root(), spec95["decision_path"]).read_text(encoding="utf-8")
    )
    _expect_sha(
        resolve_under_root(project_root(), spec95["config_path"]),
        spec95["config_sha256"],
        "WB95 config",
    )
    _expect_sha(
        resolve_under_root(project_root(), spec95["decision_path"]),
        spec95["decision_sha256"],
        "WB95 decision",
    )
    if decision95.get("decision") != spec95["frozen_decision"]:
        raise QoverPExportError("WB95 decision must stay frozen")
    if decision95.get("mechanism") != spec95["frozen_mechanism"]:
        raise QoverPExportError("WB95 mechanism must stay frozen")
    return {
        "workbook_87": {
            "decision": decision87["decision"],
            "config_sha256": spec87["config_sha256"],
        },
        "workbook_95": {
            "decision": WB95_DECISION,
            "mechanism": WB95_MECHANISM,
            "config_sha256": spec95["config_sha256"],
            "decision_sha256": spec95["decision_sha256"],
        },
    }


def dump_path_for_source(config: Mapping[str, Any], source_id: str) -> Path:
    root = resolve_under_root(project_root(), str(config["dump_root"]))
    return root / source_id / "ckf_qoverp_covariance.jsonl"


def load_dump_records(path: Path, *, split: str) -> list[dict[str, Any]]:
    authorized = authorize_path(path, AccessScope.DEVELOPMENT_VALIDATION, split=split)
    if not authorized.is_file():
        return []
    rows = []
    for line in authorized.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _as_matrix(payload: Any) -> np.ndarray | None:
    if payload is None:
        return None
    matrix = np.asarray(payload, dtype=np.float64)
    if matrix.shape != (5, 5):
        return None
    return matrix


def _finite_stats(values: np.ndarray) -> dict[str, int | float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    result: dict[str, int | float | None] = {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "min": None,
        "median": None,
        "max": None,
    }
    if finite.size:
        result.update(
            {
                "min": float(np.min(finite)),
                "median": float(np.median(finite)),
                "max": float(np.max(finite)),
            }
        )
    return result


def evaluate_records(
    records: list[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    dummy_q = float(config["acceptance"]["dummy_qoverp_per_mev"])
    dummy_v = float(config["acceptance"]["dummy_variance_per_mev2"])
    tol = float(config["acceptance"]["dummy_match_tolerance"])
    qoverp = []
    sigma = []
    min_eigs = []
    dummy_q_hits = 0
    dummy_v_hits = 0
    truth_hits = 0
    spd_ok = 0
    spd_fail = 0
    not_symmetric = 0
    missing_cov = 0
    cross_nonzero = 0
    used = 0
    corr_sum = np.zeros((5, 5), dtype=np.float64)
    corr_n = 0
    for row in records:
        if bool(row.get("is_truth", False)):
            truth_hits += 1
            continue
        cov = _as_matrix(row.get("native_covariance"))
        state = np.asarray(row.get("native_state"), dtype=np.float64).reshape(-1)
        if state.size != 5 or not np.isfinite(state).all():
            continue
        q_value = float(state[QOVERP_INDEX])
        qoverp.append(q_value)
        if abs(q_value - dummy_q) <= tol:
            dummy_q_hits += 1
        if cov is None or not bool(row.get("has_covariance", False)):
            missing_cov += 1
            continue
        used += 1
        variance = float(cov[QOVERP_INDEX, QOVERP_INDEX])
        sigma.append(float(np.sqrt(variance)) if variance >= 0.0 else np.nan)
        if abs(variance - dummy_v) <= tol:
            dummy_v_hits += 1
        if not np.allclose(cov, cov.T, rtol=1.0e-7, atol=1.0e-12):
            not_symmetric += 1
        eigenvalues = np.linalg.eigvalsh(0.5 * (cov + cov.T))
        min_eigs.append(float(np.min(eigenvalues)))
        if is_spd(cov):
            spd_ok += 1
            diag = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
            if np.all(diag > 0.0):
                corr = cov / np.outer(diag, diag)
                corr_sum += corr
                corr_n += 1
        else:
            spd_fail += 1
        cross = cov[QOVERP_INDEX, :QOVERP_INDEX]
        if np.any(np.abs(cross) > 0.0):
            cross_nonzero += 1
    return {
        "n_records": len(records),
        "n_used_with_covariance": used,
        "n_missing_covariance": missing_cov,
        "n_truth_rejected": truth_hits,
        "n_spd_ok": spd_ok,
        "n_spd_fail": spd_fail,
        "n_not_symmetric": not_symmetric,
        "n_dummy_qoverp": dummy_q_hits,
        "n_dummy_variance": dummy_v_hits,
        "n_nonzero_qoverp_cross_term": cross_nonzero,
        "q_over_p_per_mev": _finite_stats(np.asarray(qoverp, dtype=np.float64)),
        "sigma_q_over_p_per_mev": _finite_stats(np.asarray(sigma, dtype=np.float64)),
        "min_eigenvalue": _finite_stats(np.asarray(min_eigs, dtype=np.float64)),
        "mean_correlation": (corr_sum / corr_n).tolist() if corr_n else None,
        "n_correlation_matrices": corr_n,
    }


def decide(
    construction: Mapping[str, Any],
    validation: Mapping[str, Any],
    *,
    dumps_materialized: bool,
) -> dict[str, Any]:
    min_tracks = 20
    if not dumps_materialized:
        mechanism = MECHANISM_NOT_MATERIALIZED
        verdict = "FAIL"
    elif int(construction.get("n_truth_rejected", 0)) or int(
        validation.get("n_truth_rejected", 0)
    ):
        mechanism = MECHANISM_TRUTH
        verdict = "FAIL"
    elif int(construction.get("n_used_with_covariance", 0)) < min_tracks or int(
        validation.get("n_used_with_covariance", 0)
    ) < min_tracks:
        mechanism = (
            MECHANISM_COV_ABSENT
            if int(construction.get("n_records", 0)) + int(validation.get("n_records", 0))
            else MECHANISM_NOT_MATERIALIZED
        )
        verdict = "FAIL"
    elif int(construction.get("n_spd_fail", 0)) or int(validation.get("n_spd_fail", 0)):
        mechanism = MECHANISM_NOT_SPD
        verdict = "FAIL"
    elif (
        int(construction.get("n_dummy_qoverp", 0))
        == int(construction.get("n_records", 0))
        and int(validation.get("n_dummy_qoverp", 0))
        == int(validation.get("n_records", 0))
        and int(construction.get("n_records", 0)) > 0
    ):
        mechanism = MECHANISM_DUMMY
        verdict = "FAIL"
    elif (
        int(construction.get("n_dummy_variance", 0))
        == int(construction.get("n_used_with_covariance", 0))
        and int(validation.get("n_dummy_variance", 0))
        == int(validation.get("n_used_with_covariance", 0))
        and int(construction.get("n_used_with_covariance", 0)) > 0
    ):
        mechanism = MECHANISM_DUMMY
        verdict = "FAIL"
    elif int(construction.get("n_nonzero_qoverp_cross_term", 0)) == 0 or int(
        validation.get("n_nonzero_qoverp_cross_term", 0)
    ) == 0:
        mechanism = MECHANISM_NO_CROSS
        verdict = "FAIL"
    else:
        mechanism = None
        verdict = "PASS"
    established = verdict == "PASS"
    return {
        "kind": "qoverp_covariance_contract",
        "task": TASK,
        "workbook": WORKBOOK,
        "verdict": verdict,
        "decision": DECISION_ESTABLISHED if established else DECISION_NOT_ESTABLISHED,
        "mechanism": mechanism,
        "state_definition": state_definition(),
        "units": {
            "q_over_p": "per_MeV",
            "q_over_p_variance": "per_MeV2",
            "loc": "mm",
            "angle": "rad",
        },
        "source_collection": "CKFTrackCollection",
        "covariance_dimension": 5,
        "cross_terms": [
            "Cov(loc1,q/p)",
            "Cov(loc2,q/p)",
            "Cov(phi,q/p)",
            "Cov(theta,q/p)",
            "Cov(x,q/p)",
            "Cov(y,q/p)",
            "Cov(tx,q/p)",
            "Cov(ty,q/p)",
        ],
        "contract_established_for_process_noise": established,
        "measurement_model_v2_entered": False,
        "geometry_write_allowed": False,
        "held_out_accessed": False,
        "truth_qoverp_used_as_real_data_solution": False,
        "dummy_segmentfit_used": False,
        "synthetic_qoverp_injected": False,
        "new_source_campaign_started": False,
        "unconstrained_tracker_only_stopped": True,
        "inherited_wb87_decision": "faseracts_transport_covariance_not_validated",
    }


def inventory_sources(config: Mapping[str, Any]) -> dict[str, Any]:
    splits = {}
    materialized = True
    for split, key in (
        ("construction", "construction_sources"),
        ("validation", "validation_sources"),
    ):
        access = "train" if split == "construction" else "validation"
        records: list[dict[str, Any]] = []
        present = []
        missing = []
        for spec in config["mc_data"][key]:
            source_id = spec["source_id"]
            authorize_path(spec["input_xaod"], AccessScope.DEVELOPMENT_VALIDATION, split=access)
            path = dump_path_for_source(config, source_id)
            rows = load_dump_records(path, split=access)
            if rows:
                present.append(source_id)
                records.extend(rows)
            else:
                missing.append(source_id)
                materialized = False
        splits[split] = {
            "n_sources": len(config["mc_data"][key]),
            "present_sources": present,
            "missing_sources": missing,
            **evaluate_records(records, config),
        }
    return {"dumps_materialized": materialized and not any(
        splits[name]["missing_sources"] for name in splits
    ), "splits": splits}
