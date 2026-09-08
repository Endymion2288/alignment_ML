"""Task B0: C++ ACTS transport dump contract.

Materializes

    C0 = F Cin F^T
    C1 = C0 + Q_ACTS

from FaserActsExtrapolationTool without editing Calypso.  WB97 stays FAIL.
Truth q/p, dummy SegmentFit, rescale, and Q-tuning cannot pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from alignment.faseracts_propagated_covariance_validation_v2 import FROZEN_WB81_GATES
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from datasets.access_policy import AccessScope, authorize_path
from datasets.acts_process_noise_contract import (
    DECISION_ESTABLISHED,
    DECISION_NOT_ESTABLISHED,
    FAILURE_CONFIGURATION,
    FAILURE_MEASUREMENT_MODEL,
    MECHANISM_CLOSURE_FAILED,
    MECHANISM_DUMMY,
    MECHANISM_NOT_MATERIALIZED,
    MECHANISM_NOT_WRITTEN,
    MECHANISM_TRUTH,
    MODEL0,
    MODEL1,
    MODEL2,
    attach_highland_model,
    audit_process_noise_configuration,
    evaluate_model_records,
    load_dump_records,
)
from datasets.qoverp_covariance_export import DECISION_ESTABLISHED as WB96_DECISION
from datasets.qoverp_semantics import (
    DECISION_NOT_ESTABLISHED as WB95_DECISION,
    MECHANISM_NOT_EXPORTED as WB95_MECHANISM,
)

SCHEMA_VERSION = "acts-transport-dump-v1"
DEFAULT_CONFIG = "configs/acts_transport_dump_v1.yaml"
TASK = "SB-B0"
WORKBOOK = 98

REQUIRED_ROW_KEYS = (
    "state_definition",
    "input_covariance",
    "transport_jacobian",
    "output_covariance_no_material",
    "output_covariance_with_material",
    "process_noise",
    "geometry_hash",
    "field_hash",
    "material_hash",
    "conditions_hash",
)

SITUATION_A = "acts_integration_export_contract_not_established"
SITUATION_B = "acts_q_materialized_closure_failed"

WB97_DUMP_FILENAME = "ckf_acts_process_noise.jsonl"


class TransportDumpError(ValueError):
    """Raised when the B0 dump contract is illegal."""


def refuse_truth_qoverp() -> None:
    raise TransportDumpError("truth q/p is not a real-data solution")


def refuse_dummy_segmentfit() -> None:
    raise TransportDumpError("dummy SegmentFit covariance is not a physical prior")


def refuse_covariance_rescale() -> None:
    raise TransportDumpError("covariance rescale is forbidden")


def refuse_process_noise_tuning() -> None:
    raise TransportDumpError("process noise must not be adjusted to chi2")


def plugin_dir(config: Mapping[str, Any] | None = None) -> Path:
    relative = "build/acts_transport_dump"
    if config is not None:
        relative = str(config.get("plugin", {}).get("build_dir", relative))
    return resolve_under_root(project_root(), relative)


def plugin_library_path(config: Mapping[str, Any] | None = None) -> Path:
    name = "libCkfActsTransportDump.so"
    if config is not None:
        name = str(config.get("plugin", {}).get("library", name))
    return plugin_dir(config) / name


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise TransportDumpError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise TransportDumpError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise TransportDumpError(f"workbook must be {WORKBOOK}")
    for key in (
        "geometry_write_allowed",
        "held_out_accessed",
        "real_data_alignment_authorized",
        "measurement_model_validated",
    ):
        if bool(config.get(key, True)):
            raise TransportDumpError(f"{key} must be false")
    for key in (
        "do_not_rescale_covariance",
        "do_not_use_truth_q_over_p_as_real_data_solution",
        "do_not_use_dummy_segmentfit_covariance",
        "do_not_tune_covariance_to_chi2",
        "do_not_tune_process_noise",
        "do_not_enter_alignment",
        "do_not_enter_measurement_model_v2",
        "do_not_start_new_source_campaign",
        "do_not_modify_faseracts_extrapolation_tool_source",
        "do_not_construct_acts_objects_in_python",
        "unconstrained_tracker_only_stopped",
    ):
        if bool(config.get(key, False)) is not True:
            raise TransportDumpError(f"{key} must be true")
    gates = config.get("closure_gates", {})
    for key, expected in FROZEN_WB81_GATES.items():
        if gates.get(key) != expected:
            raise TransportDumpError(f"WB81/WB87 gate must stay frozen: {key}")
    dump_name = Path(str(config.get("dump_filename", "ckf_acts_transport.jsonl"))).name
    if dump_name == WB97_DUMP_FILENAME:
        raise TransportDumpError("B0 dumps must not overwrite the WB97 jsonl name")
    return dict(config)


def _expect_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise TransportDumpError(f"{label} hash mismatch")


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited: dict[str, Any] = {}
    for name, mechanism_key, failure_key in (
        ("workbook_87", None, None),
        ("workbook_95", "mechanism", None),
        ("workbook_96", None, None),
        ("workbook_97", "mechanism", "failure_type"),
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
            raise TransportDumpError(f"{name} decision must stay frozen")
        if mechanism_key and spec.get("frozen_mechanism"):
            if decision.get("mechanism") != spec.get("frozen_mechanism"):
                raise TransportDumpError(f"{name} mechanism must stay frozen")
        if failure_key and spec.get("frozen_failure_type"):
            if decision.get(failure_key) != spec.get("frozen_failure_type"):
                raise TransportDumpError(f"{name} failure_type must stay frozen")
        inherited[name] = {
            "decision": spec["frozen_decision"],
            "config_sha256": spec["config_sha256"],
            "decision_sha256": spec["decision_sha256"],
        }
        if spec.get("frozen_mechanism"):
            inherited[name]["mechanism"] = spec["frozen_mechanism"]
        if spec.get("frozen_failure_type"):
            inherited[name]["failure_type"] = spec["frozen_failure_type"]
    if inherited["workbook_95"]["decision"] != WB95_DECISION:
        raise TransportDumpError("WB95 decision token mismatch")
    if inherited["workbook_95"].get("mechanism") != WB95_MECHANISM:
        raise TransportDumpError("WB95 mechanism token mismatch")
    if inherited["workbook_96"]["decision"] != WB96_DECISION:
        raise TransportDumpError("WB96 must keep the CKF export established")
    if inherited["workbook_97"]["decision"] != DECISION_NOT_ESTABLISHED:
        raise TransportDumpError("WB97 decision must stay not-established")
    return inherited


def dump_path_for_source(config: Mapping[str, Any], source_id: str) -> Path:
    root = resolve_under_root(project_root(), str(config["dump_root"]))
    return root / source_id / str(config.get("dump_filename", "ckf_acts_transport.jsonl"))


def _as_matrix(payload: Any, dim: int) -> np.ndarray | None:
    if payload is None:
        return None
    matrix = np.asarray(payload, dtype=np.float64)
    if matrix.shape != (dim, dim):
        return None
    return matrix


def analyze_process_noise(records: list[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    floor = float(config["acceptance"]["q_added_relative_floor"])
    q_norms: list[float] = []
    c0_norms: list[float] = []
    ratios: list[float] = []
    n_schema = 0
    n_q = 0
    n_scaled = 0
    for row in records:
        if all(key in row for key in REQUIRED_ROW_KEYS):
            n_schema += 1
        if bool(row.get("process_noise_artificial_scale", False)):
            n_scaled += 1
        q_mat = _as_matrix(row.get("process_noise"), 5)
        c0 = _as_matrix(row.get("output_covariance_no_material"), 5)
        if q_mat is None or c0 is None:
            continue
        q_norm = float(np.linalg.norm(q_mat))
        c0_norm = float(np.linalg.norm(c0))
        q_norms.append(q_norm)
        c0_norms.append(c0_norm)
        if c0_norm > 0.0:
            ratios.append(q_norm / c0_norm)
        if q_norm > floor:
            n_q += 1
    visible = n_q > 0 and n_scaled == 0
    return {
        "n_rows": len(records),
        "n_schema_complete": n_schema,
        "n_q_visible": n_q,
        "n_artificial_scale": n_scaled,
        "q_frobenius": {
            "count": len(q_norms),
            "min": float(np.min(q_norms)) if q_norms else None,
            "median": float(np.median(q_norms)) if q_norms else None,
            "max": float(np.max(q_norms)) if q_norms else None,
        },
        "q_over_c0": {
            "count": len(ratios),
            "min": float(np.min(ratios)) if ratios else None,
            "median": float(np.median(ratios)) if ratios else None,
            "max": float(np.max(ratios)) if ratios else None,
        },
        "process_noise_visible": visible,
        "process_noise_artificial_scale": n_scaled > 0,
    }


def _principal_direction_report(per_pair: Mapping[str, Any]) -> dict[str, Any]:
    report = {}
    for label, cell in per_pair.items():
        pencil = cell.get("pencil") or {}
        direction = pencil.get("direction")
        axis = None
        if isinstance(direction, list) and len(direction) >= 4:
            names = ("x", "y", "tx", "ty")
            axis = names[int(np.argmax(np.abs(np.asarray(direction, dtype=np.float64))))]
        report[label] = {
            "direction": direction,
            "principal_axis": axis,
            "variance_ratio": pencil.get("variance_ratio_prop_over_emp"),
        }
    return report


def inventory_sources(config: Mapping[str, Any]) -> dict[str, Any]:
    from alignment.propagated_covariance_closure import _load_truth_table

    truth_tables: dict[str, Any] = {}
    splits: dict[str, Any] = {}
    materialized = True
    q_reports: dict[str, Any] = {}
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
        q_reports[split] = analyze_process_noise(records, config)
        models = {}
        for model_key in (MODEL0, MODEL1, MODEL2):
            models[model_key] = evaluate_model_records(
                records, config, model_key=model_key, truth_tables=truth_tables
            )
            models[model_key]["principal_direction"] = _principal_direction_report(
                models[model_key].get("per_pair") or {}
            )
        splits[split] = {
            "n_sources": len(config["mc_data"][ids_key]),
            "present_sources": present,
            "missing_sources": missing,
            "n_records": len(records),
            "process_noise": q_reports[split],
            "models": models,
        }
    return {
        "dumps_materialized": materialized
        and not any(splits[name]["missing_sources"] for name in splits),
        "splits": splits,
        "dump_root": str(config["dump_root"]),
        "wb97_dump_filename_avoided": True,
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
    q_c = construction.get("process_noise") or {}
    q_v = validation.get("process_noise") or {}
    q_visible = bool(q_c.get("process_noise_visible")) and bool(
        q_v.get("process_noise_visible")
    )
    situation = None
    if int(model1_c.get("n_truth_rejected", 0)) or int(model1_v.get("n_truth_rejected", 0)):
        mechanism = MECHANISM_TRUTH
        failure_type = FAILURE_CONFIGURATION
        verdict = "FAIL"
        situation = SITUATION_A
    elif not configuration.get("contract_clear"):
        mechanism = "acts_process_noise_configuration_error"
        failure_type = FAILURE_CONFIGURATION
        verdict = "FAIL"
        situation = SITUATION_A
    elif not inventory.get("dumps_materialized"):
        mechanism = MECHANISM_NOT_MATERIALIZED
        failure_type = FAILURE_CONFIGURATION
        verdict = "FAIL"
        situation = SITUATION_A
    elif int(model1_c.get("n_success", 0)) < 20 or int(model1_v.get("n_success", 0)) < 20:
        mechanism = MECHANISM_NOT_MATERIALIZED
        failure_type = FAILURE_CONFIGURATION
        verdict = "FAIL"
        situation = SITUATION_A
    elif int(model1_c.get("n_dummy_rejected", 0)) == int(model1_c.get("n_success", 0)) + int(
        model1_c.get("n_dummy_rejected", 0)
    ) and int(model1_c.get("n_dummy_rejected", 0)) > 0:
        mechanism = MECHANISM_DUMMY
        failure_type = FAILURE_CONFIGURATION
        verdict = "FAIL"
        situation = SITUATION_A
    elif (
        int(model1_c.get("n_q_added", 0)) == 0
        or int(model1_v.get("n_q_added", 0)) == 0
        or not q_visible
    ):
        mechanism = MECHANISM_NOT_WRITTEN
        failure_type = FAILURE_CONFIGURATION
        verdict = "FAIL"
        situation = SITUATION_A
    elif bool(model1_c.get("all_calibrated")) and bool(model1_v.get("all_calibrated")):
        mechanism = None
        failure_type = None
        verdict = "PASS"
        situation = None
    else:
        mechanism = MECHANISM_CLOSURE_FAILED
        failure_type = FAILURE_MEASUREMENT_MODEL
        verdict = "FAIL"
        situation = SITUATION_B
    established = verdict == "PASS"
    return {
        "kind": "acts_transport_dump_contract",
        "task": TASK,
        "workbook": WORKBOOK,
        "verdict": verdict,
        "decision": DECISION_ESTABLISHED if established else DECISION_NOT_ESTABLISHED,
        "mechanism": mechanism,
        "failure_type": failure_type,
        "situation": situation,
        "process_noise_visible": q_visible,
        "process_noise_artificial_scale": bool(q_c.get("process_noise_artificial_scale"))
        or bool(q_v.get("process_noise_artificial_scale")),
        "dump_helper": "CkfActsTransportDumpAlg",
        "dump_helper_language": "C++",
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
        "inherited_wb97_decision": DECISION_NOT_ESTABLISHED,
    }
