"""Task B: ACTS MaterialInteractor / process-noise contract.

B1 audits the production transport configuration.  B2 compares

    C0 = F Cin F^T
    C1 = F Cin F^T + Q_ACTS
    C2 = F Cin F^T + Q_Highland   (diagnostic only)

on the frozen WB87 construction/validation split.  Cin is the WB96 CKF 5x5.
Truth q/p, dummy SegmentFit covariance, rescale, and Q-tuning cannot pass.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Mapping

import numpy as np
import yaml

from alignment.faseracts_propagated_covariance_validation_v2 import (
    FROZEN_WB81_GATES,
    highland_theta0,
    process_noise_covariance,
)
from alignment.numerical_contract import is_spd, require_spd
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment.propagated_covariance_closure import (
    ClosureRecords,
    _load_truth_table,
    compute_closure_metrics,
    evaluate_closure_gates,
)
from datasets.access_policy import AccessScope, authorize_path
from datasets.qoverp_covariance_export import (
    DECISION_ESTABLISHED as WB96_DECISION,
    provenance_hashes as wb96_provenance_hashes,
)
from datasets.qoverp_semantics import (
    DECISION_NOT_ESTABLISHED as WB95_DECISION,
    MECHANISM_NOT_EXPORTED as WB95_MECHANISM,
)
from datasets.transport_contract import MEV_PER_GEV

SCHEMA_VERSION = "acts-process-noise-contract-v1"
DEFAULT_CONFIG = "configs/acts_process_noise_contract_v1.yaml"
TASK = "SB-B"
WORKBOOK = 97

DECISION_ESTABLISHED = "acts_process_noise_contract_established"
DECISION_NOT_ESTABLISHED = "acts_process_noise_contract_not_established"
MECHANISM_NOT_MATERIALIZED = "acts_process_noise_not_materialized"
MECHANISM_NOT_WRITTEN = "acts_process_noise_not_written"
MECHANISM_CLOSURE_FAILED = "acts_process_noise_closure_failed"
MECHANISM_CONFIGURATION = "acts_process_noise_configuration_error"
MECHANISM_TRUTH = "truth_qoverp_forbidden"
MECHANISM_DUMMY = "dummy_segmentfit_forbidden"

FAILURE_CONFIGURATION = "acts_process_noise_configuration"
FAILURE_MEASUREMENT_MODEL = "measurement_track_model_insufficient"

MODEL0 = "model0_no_process_noise"
MODEL1 = "model1_acts_process_noise"
MODEL2 = "model2_highland_diagnostic"


class ProcessNoiseContractError(ValueError):
    """Raised when the Task B contract is illegal."""


def refuse_truth_qoverp() -> None:
    raise ProcessNoiseContractError("truth q/p is not a real-data solution")


def refuse_dummy_segmentfit() -> None:
    raise ProcessNoiseContractError("dummy SegmentFit covariance is not a physical prior")


def refuse_covariance_rescale() -> None:
    raise ProcessNoiseContractError("covariance rescale is forbidden")


def refuse_process_noise_tuning() -> None:
    raise ProcessNoiseContractError("process noise must not be adjusted to chi2")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ProcessNoiseContractError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise ProcessNoiseContractError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise ProcessNoiseContractError(f"workbook must be {WORKBOOK}")
    for key in (
        "geometry_write_allowed",
        "held_out_accessed",
        "real_data_alignment_authorized",
        "measurement_model_validated",
    ):
        if bool(config.get(key, True)):
            raise ProcessNoiseContractError(f"{key} must be false")
    for key in (
        "do_not_rescale_covariance",
        "do_not_use_truth_q_over_p_as_real_data_solution",
        "do_not_use_dummy_segmentfit_covariance",
        "do_not_tune_covariance_to_chi2",
        "do_not_tune_process_noise",
        "do_not_enter_alignment",
        "do_not_enter_measurement_model_v2",
        "do_not_start_new_source_campaign",
        "unconstrained_tracker_only_stopped",
    ):
        if bool(config.get(key, False)) is not True:
            raise ProcessNoiseContractError(f"{key} must be true")
    gates = config.get("closure_gates", {})
    for key, expected in FROZEN_WB81_GATES.items():
        if gates.get(key) != expected:
            raise ProcessNoiseContractError(f"WB81/WB87 gate must stay frozen: {key}")
    return dict(config)


def _expect_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise ProcessNoiseContractError(f"{label} hash mismatch")


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = {}
    for name, decision_key, mechanism_key in (
        ("workbook_87", "decision", None),
        ("workbook_95", "decision", "mechanism"),
        ("workbook_96", "decision", None),
    ):
        spec = config["inheritance"][name]
        decision = json.loads(
            resolve_under_root(project_root(), spec["decision_path"]).read_text(
                encoding="utf-8"
            )
        )
        _expect_sha(
            resolve_under_root(project_root(), spec["config_path"]),
            spec["config_sha256"],
            f"{name} config",
        )
        _expect_sha(
            resolve_under_root(project_root(), spec["decision_path"]),
            spec["decision_sha256"],
            f"{name} decision",
        )
        if decision.get("decision") != spec["frozen_decision"]:
            raise ProcessNoiseContractError(f"{name} decision must stay frozen")
        if mechanism_key and decision.get("mechanism") != spec.get("frozen_mechanism"):
            raise ProcessNoiseContractError(f"{name} mechanism must stay frozen")
        inherited[name] = {
            "decision": spec["frozen_decision"],
            "config_sha256": spec["config_sha256"],
            "decision_sha256": spec["decision_sha256"],
        }
        if "frozen_mechanism" in spec:
            inherited[name]["mechanism"] = spec["frozen_mechanism"]
    if inherited["workbook_95"]["decision"] != WB95_DECISION:
        raise ProcessNoiseContractError("WB95 decision token mismatch")
    if inherited["workbook_95"].get("mechanism") != WB95_MECHANISM:
        raise ProcessNoiseContractError("WB95 mechanism token mismatch")
    if inherited["workbook_96"]["decision"] != WB96_DECISION:
        raise ProcessNoiseContractError("WB96 must keep the CKF export established")
    return inherited


def _read_calypso(config: Mapping[str, Any], relative: str) -> str:
    root = Path(str(config["software_provenance"]["calypso_root"]))
    path = root / relative
    if not path.is_file():
        raise ProcessNoiseContractError(f"Calypso source is absent: {relative}")
    return path.read_text(encoding="utf-8", errors="replace")


def audit_process_noise_configuration(config: Mapping[str, Any]) -> dict[str, Any]:
    """B1: source/runtime process-noise provenance.  Does not tune Q."""
    sw = config["software_provenance"]
    tool_h = _read_calypso(config, sw["extrapolation_tool_header"])
    tool_cxx = _read_calypso(config, sw["extrapolation_tool_source"])
    ntuple_cfg = _read_calypso(config, sw["ntuple_dumper_config"])
    ckf2 = _read_calypso(config, sw["ckf2_source"])
    geo_h = _read_calypso(config, sw["tracking_geometry_svc_header"])
    geo_py = _read_calypso(config, sw["tracking_geometry_config"])

    ms_default = bool(
        re.search(
            r'InteractionMultiScatering",\s*false',
            tool_h,
        )
    )
    eloss_default = bool(re.search(r'InteractionEloss",\s*false', tool_h))
    record_default = bool(re.search(r'InteractionRecord",\s*false', tool_h))
    interactor_present = "Acts::MaterialInteractor" in tool_cxx
    ntuple_sets_ms = "InteractionMultiScatering" in ntuple_cfg
    ckf_fitter_ms_on = bool(
        re.search(
            r"makeTrackFitterFunction\([\s\S]{0,240}true,\s*true,",
            ckf2,
        )
    )
    use_map_default_false = 'UseMaterialMap", false' in geo_h
    map_enabled_if_json = "MaterialSource.find(\".json\")" in geo_py.replace("'", '"') or (
        'MaterialSource.find(".json")' in geo_py
    )

    material = config["material_map"]
    material_path = Path(str(material["path"]))
    material_exists = material_path.is_file()
    material_sha = sha256_file(material_path) if material_exists else None
    provenance = wb96_provenance_hashes(
        {
            "software_provenance": sw,
            "tags": config["tags"],
        }
    )
    material_hash = (
        hashlib.sha256(
            f"{material['path']}|{material_sha}|{sw['calypso_git_sha']}".encode("utf-8")
        ).hexdigest()
        if material_sha
        else None
    )
    production_q_written = False
    contract_clear = all(
        (
            interactor_present,
            ms_default,
            eloss_default,
            record_default,
            not ntuple_sets_ms,
            material_exists,
            material_sha is not None,
        )
    )
    return {
        "kind": "acts_process_noise_configuration",
        "task": TASK,
        "workbook": WORKBOOK,
        "acts_version": sw["acts_version"],
        "athena_release": sw["athena_release"],
        "calypso_git_sha": sw["calypso_git_sha"],
        "geometry_hash": provenance["geometry_hash"],
        "field_hash": provenance["field_hash"],
        "conditions_hash": provenance["conditions_hash"],
        "material_map_hash": material_hash,
        "geometry_tag": provenance["geometry_tag"],
        "field_tag": provenance["field_tag"],
        "conditions_tag": provenance["conditions_tag"],
        "transport_options": {
            "field_mode": "FASER",
            "pt_loopers_mev": 300,
            "max_step_size_m": 10,
            "max_steps_default": 4000,
            "ntuple_max_steps": 10000,
            "particle_hypothesis": "Acts::ParticleHypothesis::muon()",
        },
        "process_noise_flags": {
            "material_interactor_in_action_list": interactor_present,
            "production_interaction_multiple_scattering": False,
            "production_interaction_energy_loss": False,
            "production_interaction_record": False,
            "header_defaults_all_false": bool(ms_default and eloss_default and record_default),
            "ntuple_dumper_overrides_interaction_flags": ntuple_sets_ms,
            "production_process_noise_written_to_covariance": production_q_written,
            "ckf2_fitter_multiple_scattering": ckf_fitter_ms_on,
            "ckf2_fitter_energy_loss": ckf_fitter_ms_on,
        },
        "material_map": {
            "path": str(material["path"]),
            "exists": material_exists,
            "file_sha256": material_sha,
            "use_material_map_header_default": False if use_map_default_false else None,
            "enabled_when_material_source_is_json": map_enabled_if_json,
            "used_by_production_ckf_reco": bool(material["used_by_production_ckf_reco"]),
        },
        "contract_clear": contract_clear,
        "production_transport_has_process_noise": False,
        "validation_job_may_enable_flags_without_source_edit": True,
        "highland_is_diagnostic_only": True,
        "geometry_write_allowed": False,
        "measurement_model_v2_entered": False,
    }


def dump_path_for_source(config: Mapping[str, Any], source_id: str) -> Path:
    root = resolve_under_root(project_root(), str(config["dump_root"]))
    return root / source_id / "ckf_acts_process_noise.jsonl"


def load_dump_records(path: Path, *, split: str) -> list[dict[str, Any]]:
    authorized = authorize_path(path, AccessScope.DEVELOPMENT_VALIDATION, split=split)
    if not authorized.is_file():
        return []
    rows = []
    for line in authorized.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _as_matrix(payload: Any, dim: int) -> np.ndarray | None:
    if payload is None:
        return None
    matrix = np.asarray(payload, dtype=np.float64)
    if matrix.shape != (dim, dim):
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


def _truth_residual(
    pred: np.ndarray,
    truth_pos: np.ndarray,
    truth_mom: np.ndarray,
    target_z: float,
    *,
    straight_line: bool,
) -> np.ndarray | None:
    if abs(float(truth_mom[2])) < 1.0e-12:
        return None
    tx_t = float(truth_mom[0] / truth_mom[2])
    ty_t = float(truth_mom[1] / truth_mom[2])
    dz = (float(target_z) - float(truth_pos[2])) if straight_line else 0.0
    residual = np.array(
        [
            float(pred[0]) - (float(truth_pos[0]) + tx_t * dz),
            float(pred[1]) - (float(truth_pos[1]) + ty_t * dz),
            float(pred[2]) - tx_t,
            float(pred[3]) - ty_t,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(residual)):
        return None
    return residual


def _pick_truth(event_truth: Mapping[int, Mapping[str, Any]], target_station: int):
    for barcode, per_station in event_truth.items():
        payload = per_station.get(target_station)
        if payload is None:
            continue
        if np.all(np.isfinite(payload["pos"])) and np.all(np.isfinite(payload["mom"])):
            return barcode, payload
    return None, None


def _augment_chi2(metrics: dict[str, Any], e: np.ndarray, c: np.ndarray) -> dict[str, Any]:
    samples = []
    for residual, cov in zip(e, c):
        try:
            samples.append(float(residual @ np.linalg.solve(cov, residual)) / 4.0)
        except np.linalg.LinAlgError:
            continue
    array = np.asarray(samples, dtype=np.float64)
    metrics = dict(metrics)
    metrics["chi2_per_ndof_mean"] = float(np.mean(array)) if array.size else float("nan")
    metrics["chi2_per_ndof_median"] = (
        float(np.median(array)) if array.size else float("nan")
    )
    metrics["fraction_chi2_per_ndof_above_4"] = (
        float(np.mean(array > 4.0)) if array.size else float("nan")
    )
    return metrics


def _qoverp_bin_name(abs_qoverp: float, bins: Mapping[str, Sequence]) -> str | None:
    value = float(abs_qoverp)
    for name, (lo, hi) in bins.items():
        if float(lo) <= value < float(hi):
            return str(name)
    return None


def highland_on_model0(c0: np.ndarray, lever_arm_mm: float, p_mev: float, x0: float) -> np.ndarray:
    return require_spd(c0 + process_noise_covariance(lever_arm_mm, p_mev, x0), name="C2")


def evaluate_model_records(
    records: list[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    model_key: str,
    truth_tables: Mapping[str, Mapping[tuple[int, int], Any]],
) -> dict[str, Any]:
    pairs = [tuple(item) for item in config["station_pairs"]]
    bins = config["qoverp_bins_abs_per_mev"]
    tr = config["truth_reference"]
    per_pair: dict[str, Any] = {}
    bin_chi2: dict[str, list[float]] = {name: [] for name in bins}
    n_truth = 0
    n_dummy = 0
    n_q_added = 0
    n_success = 0
    for pair in pairs:
        e_list = []
        c_list = []
        lever = []
        ptx = []
        pty = []
        for row in records:
            if bool(row.get("is_truth", False)):
                n_truth += 1
                continue
            if int(row.get("source_station", 0)) != pair[0]:
                continue
            if int(row.get("target_station", -1)) != pair[1]:
                continue
            block = row.get(model_key) or {}
            if not bool(block.get("success", False)):
                continue
            cov = _as_matrix(block.get("covariance_4x4"), 4)
            pred = np.asarray(block.get("state_xy_tx_ty"), dtype=np.float64).reshape(-1)
            if cov is None or pred.size != 4 or not is_spd(cov):
                continue
            table = truth_tables.get(str(row.get("source_id")))
            if table is None:
                continue
            event_truth = table.get((int(row["run_id"]), int(row["event_id"])))
            if not event_truth:
                continue
            _, tgt = _pick_truth(event_truth, pair[1])
            if tgt is None:
                continue
            residual = _truth_residual(
                pred,
                tgt["pos"],
                tgt["mom"],
                float(row["target_z_mm"]),
                straight_line=bool(tr["straight_line_z_correction"]),
            )
            if residual is None:
                continue
            dummy_q = abs(float(row.get("q_over_p_per_mev", 0.0)) - 1.0e-5) <= 1.0e-18
            if dummy_q:
                n_dummy += 1
                continue
            e_list.append(residual)
            c_list.append(cov)
            lever.append(float(row["target_z_mm"]) - float(row["source_z_mm"]))
            ptx.append(float(pred[2]))
            pty.append(float(pred[3]))
            n_success += 1
            if model_key == MODEL1:
                c0 = _as_matrix((row.get(MODEL0) or {}).get("covariance_4x4"), 4)
                if c0 is not None and float(np.linalg.norm(cov - c0)) > float(
                    config["acceptance"]["q_added_relative_floor"]
                ):
                    n_q_added += 1
            abs_q = abs(float(row.get("q_over_p_per_mev", np.nan)))
            name = _qoverp_bin_name(abs_q, bins)
            if name is not None:
                try:
                    bin_chi2[name].append(float(residual @ np.linalg.solve(cov, residual)) / 4.0)
                except np.linalg.LinAlgError:
                    pass
        label = f"({pair[0]},{pair[1]})"
        if len(e_list) < int(config["closure_gates"]["min_pairs_per_pair"]):
            per_pair[label] = {
                "n_pairs": len(e_list),
                "sufficient": False,
                "gate_verdict": {"calibrated": False, "reason": "insufficient_pairs"},
            }
            continue
        recs = ClosureRecords(
            station_pair=pair,
            q_over_p_mode=-1,
            e_prop=np.asarray(e_list, dtype=np.float64),
            c_prop=np.asarray(c_list, dtype=np.float64),
            lever_arm_mm=np.asarray(lever, dtype=np.float64),
            pred_tx=np.asarray(ptx, dtype=np.float64),
            pred_ty=np.asarray(pty, dtype=np.float64),
            n_matched=len(e_list),
            has_covariance=True,
        )
        metrics = _augment_chi2(
            compute_closure_metrics(recs, config),
            recs.e_prop,
            recs.c_prop,
        )
        metrics["gate_verdict"] = evaluate_closure_gates(metrics, config)
        per_pair[label] = metrics
    return {
        "model": model_key,
        "n_success": n_success,
        "n_truth_rejected": n_truth,
        "n_dummy_rejected": n_dummy,
        "n_q_added": n_q_added,
        "per_pair": per_pair,
        "qoverp_bins": {
            name: _finite_stats(np.asarray(values, dtype=np.float64))
            for name, values in bin_chi2.items()
        },
        "all_calibrated": all(
            bool((cell.get("gate_verdict") or {}).get("calibrated"))
            for cell in per_pair.values()
        )
        and len(per_pair) == len(pairs),
    }


def _all_sources(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(config["mc_data"]["construction_sources"]) + list(
        config["mc_data"]["validation_sources"]
    )


def attach_highland_model(records: list[dict[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    x0 = float(config["models"][2]["material_x_over_x0_per_gap"])
    attached = []
    for row in records:
        item = dict(row)
        model0 = item.get(MODEL0) or {}
        cov0 = _as_matrix(model0.get("covariance_4x4"), 4)
        p_mev = float(item.get("p_mev") or 0.0)
        lever = float(item.get("target_z_mm", 0.0)) - float(item.get("source_z_mm", 0.0))
        if cov0 is not None and p_mev > 0.0 and bool(model0.get("success", False)):
            item[MODEL2] = {
                "success": True,
                "state_xy_tx_ty": model0.get("state_xy_tx_ty"),
                "covariance_4x4": highland_on_model0(cov0, lever, p_mev, x0).tolist(),
                "diagnostic_only": True,
            }
        else:
            item[MODEL2] = {"success": False, "diagnostic_only": True}
        attached.append(item)
    return attached


def inventory_sources(config: Mapping[str, Any]) -> dict[str, Any]:
    truth_tables: dict[str, Any] = {}
    splits: dict[str, Any] = {}
    materialized = True
    for split, key, ids_key in (
        ("construction", "construction_sources", "construction_source_ids"),
        ("validation", "validation_sources", "validation_source_ids"),
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
                records.extend(attach_highland_model(rows, config))
                refit = resolve_under_root(
                    project_root(),
                    str(config["mc_data"]["source_root_template"]).format(source_id=source_id),
                )
                truth_tables[source_id] = _load_truth_table(
                    refit / config["mc_data"]["enhanced_file"],
                    config["mc_data"]["enhanced_tree"],
                    config,
                )
            else:
                missing.append(source_id)
                materialized = False
        models = {}
        for model_key in (MODEL0, MODEL1, MODEL2):
            models[model_key] = evaluate_model_records(
                records, config, model_key=model_key, truth_tables=truth_tables
            )
        splits[split] = {
            "n_sources": len(config["mc_data"][ids_key]),
            "present_sources": present,
            "missing_sources": missing,
            "n_records": len(records),
            "models": models,
        }
    return {
        "dumps_materialized": materialized
        and not any(splits[name]["missing_sources"] for name in splits),
        "splits": splits,
    }


def decide(
    configuration: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    construction = inventory["splits"]["construction"]
    validation = inventory["splits"]["validation"]
    model1_c = construction["models"][MODEL1]
    model1_v = validation["models"][MODEL1]
    model0_c = construction["models"][MODEL0]
    model0_v = validation["models"][MODEL0]
    if int(model1_c.get("n_truth_rejected", 0)) or int(model1_v.get("n_truth_rejected", 0)):
        mechanism = MECHANISM_TRUTH
        failure_type = FAILURE_CONFIGURATION
        verdict = "FAIL"
    elif not configuration.get("contract_clear"):
        mechanism = MECHANISM_CONFIGURATION
        failure_type = FAILURE_CONFIGURATION
        verdict = "FAIL"
    elif not inventory.get("dumps_materialized"):
        mechanism = MECHANISM_NOT_MATERIALIZED
        failure_type = FAILURE_CONFIGURATION
        verdict = "FAIL"
    elif int(model1_c.get("n_success", 0)) < 20 or int(model1_v.get("n_success", 0)) < 20:
        mechanism = MECHANISM_NOT_MATERIALIZED
        failure_type = FAILURE_CONFIGURATION
        verdict = "FAIL"
    elif int(model1_c.get("n_dummy_rejected", 0)) == int(model1_c.get("n_success", 0)) + int(
        model1_c.get("n_dummy_rejected", 0)
    ) and int(model1_c.get("n_dummy_rejected", 0)) > 0:
        mechanism = MECHANISM_DUMMY
        failure_type = FAILURE_CONFIGURATION
        verdict = "FAIL"
    elif int(model1_c.get("n_q_added", 0)) == 0 or int(model1_v.get("n_q_added", 0)) == 0:
        mechanism = MECHANISM_NOT_WRITTEN
        failure_type = FAILURE_CONFIGURATION
        verdict = "FAIL"
    elif bool(model1_c.get("all_calibrated")) and bool(model1_v.get("all_calibrated")):
        mechanism = None
        failure_type = None
        verdict = "PASS"
    else:
        mechanism = MECHANISM_CLOSURE_FAILED
        failure_type = FAILURE_MEASUREMENT_MODEL
        verdict = "FAIL"
    established = verdict == "PASS"
    return {
        "kind": "acts_process_noise_contract",
        "task": TASK,
        "workbook": WORKBOOK,
        "verdict": verdict,
        "decision": DECISION_ESTABLISHED if established else DECISION_NOT_ESTABLISHED,
        "mechanism": mechanism,
        "failure_type": failure_type,
        "contract_clear": bool(configuration.get("contract_clear")),
        "production_transport_has_process_noise": False,
        "acts_version": configuration.get("acts_version"),
        "geometry_hash": configuration.get("geometry_hash"),
        "material_map_hash": configuration.get("material_map_hash"),
        "field_hash": configuration.get("field_hash"),
        "conditions_hash": configuration.get("conditions_hash"),
        "transport_options": configuration.get("transport_options"),
        "process_noise_flags": configuration.get("process_noise_flags"),
        "construction": construction,
        "validation": validation,
        "model0_calibrated": bool(model0_c.get("all_calibrated"))
        and bool(model0_v.get("all_calibrated")),
        "model1_calibrated": bool(model1_c.get("all_calibrated"))
        and bool(model1_v.get("all_calibrated")),
        "highland_diagnostic_only": True,
        "highland_promoted": False,
        "covariance_rescaled": False,
        "process_noise_tuned": False,
        "truth_qoverp_used_as_real_data_solution": False,
        "dummy_segmentfit_used": False,
        "contract_established_for_measurement_model_v2": established,
        "measurement_model_v2_entered": False,
        "geometry_write_allowed": False,
        "held_out_accessed": False,
        "unconstrained_tracker_only_stopped": True,
        "inherited_wb87_decision": "faseracts_transport_covariance_not_validated",
        "inherited_wb96_decision": WB96_DECISION,
    }
