"""Read-only provenance audit of workbook-69 rank 2/3/4 independent sources.

Does not retune S or rank_tolerance, drop sources, reselect population, or
reopen the stable-core campaign.  Classifies coverage loss versus pipeline
corruption.  Not a confirmatory set.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.cad_survey_nov22 import git_head_sha
from alignment.identifiable_subspace import (
    FROZEN_RANK_TOLERANCE,
    MixedUnitNakedJacobianError,
    refuse_forced_identifiable_rank,
    refuse_rank_threshold_from_spectrum,
    subspace_as_json,
    svd_naked_jacobian,
)
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
    RESIDUAL_DECREASE_LABEL,
    project_root,
    sha256_file,
)
from alignment.tracker_only_identifiable_subspace import (
    load_physical_banks,
    subspace_from_physical_bank,
)
from alignment.true_cluster_local_residual import RESIDUAL_KIND, assert_no_alignment_payload, common_operating_state


SCHEMA_VERSION = "faser-tracklet-independent-failure-provenance-audit-v1"
DEFAULT_CONFIG_RELATIVE = Path("configs") / "tracklet_independent_failure_provenance_audit_v1.yaml"
INHERITED_ENTRY_69 = "cross_source_stable_core_independent_validation_fail"
CLASS_COVERAGE = "normal_physics_or_coverage_loss"
CLASS_CORRUPTION = "pipeline_or_artifact_corruption"
CLASS_MIXED = "coverage_loss_with_localized_artifact_flags"

CONFIG_MUST_BE_FALSE = (
    "geometry_write_allowed",
    "station_calibration_mode_available",
    "cdx_mode_allowed",
    "alignment_payload_from_self_nulling_residuals",
    "survey_is_alignment_input",
)
CONFIG_MUST_BE_TRUE = (
    "frozen",
    "do_not_retrain_v2",
    "do_not_modify_frozen_v2_checkpoint",
    "do_not_modify_pairwise_or_route_policy",
    "do_not_run_full_parameter_newton",
    "do_not_solve_alignment_correction_on_real_data",
    "do_not_write_official_conditions",
    "do_not_write_geometry_pool_or_cool",
    "do_not_open_sealed_test",
    "do_not_svd_naked_mixed_unit_jacobian",
    "do_not_retune_scales_from_singular_values",
    "do_not_retune_rank_threshold_from_spectrum",
    "do_not_drop_failure_sources",
    "do_not_reselect_population",
    "do_not_force_identifiable_rank_five",
    "do_not_use_as_stable_core_confirmation",
    "do_not_merge_into_cluster_local_gate",
    "read_only_audit",
    "survey_is_external_cross_check_only",
)


def load_audit_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve() if path is not None else (
        project_root() / DEFAULT_CONFIG_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"failure-provenance config must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected failure-provenance schema: {source}")
    for key in CONFIG_MUST_BE_TRUE:
        if payload.get(key) is not True:
            raise ValueError(f"failure-provenance config must set {key}=true")
    for key in CONFIG_MUST_BE_FALSE:
        if payload.get(key) is not False:
            raise ValueError(f"failure-provenance config must set {key}=false")
    if payload.get("real_data_operating_mode") != OPERATING_MODE:
        raise ValueError("audit must remain residual_dq_monitoring_only")
    if payload.get("frozen_v2_checkpoint_sha256") != FROZEN_V2_CHECKPOINT_SHA256:
        raise ValueError("audit must not replace the frozen V2 checkpoint SHA256")
    if payload.get("inherited_entry_69_decision") != INHERITED_ENTRY_69:
        raise ValueError("audit must inherit the workbook-69 freeze")
    if payload.get("residual_kind") != RESIDUAL_KIND:
        raise ValueError("audit must keep the residual label")
    if payload.get("residual_decrease_label") != RESIDUAL_DECREASE_LABEL:
        raise ValueError("residual decrease must stay labeled DQ observable")
    if float(payload["rank_tolerance"]) != float(FROZEN_RANK_TOLERANCE):
        refuse_rank_threshold_from_spectrum()
    failures = tuple(str(item) for item in payload["jacobian_corpus"]["failure_source_ids"])
    siblings = tuple(str(item) for item in payload["jacobian_corpus"]["sibling_control_source_ids"])
    if set(failures) & set(siblings):
        raise ValueError("sibling controls overlap failure sources")
    config = dict(payload)
    config["config_path"] = str(source)
    return config


def _operating_state() -> dict[str, Any]:
    state = common_operating_state()
    state.update(
        {
            "survey_is_alignment_input": False,
            "survey_is_external_cross_check_only": True,
            "inherited_entry_69_decision": INHERITED_ENTRY_69,
            "read_only_audit": True,
            "not_a_stable_core_confirmation": True,
            "not_a_cluster_local_gate": True,
        }
    )
    return state


def _loader_config(config: Mapping[str, Any], source_ids: Sequence[str]) -> dict[str, Any]:
    payload = dict(config)
    corpus = dict(config["jacobian_corpus"])
    corpus["source_ids"] = list(source_ids)
    payload["jacobian_corpus"] = corpus
    return payload


def _finite_fraction(array: object) -> float:
    values = np.asarray(array, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.mean(np.isfinite(values)))


def _covariance_conditions(covariance: object) -> list[float]:
    matrices = np.asarray(covariance, dtype=np.float64)
    rows = []
    for matrix in matrices:
        if not np.isfinite(matrix).all():
            rows.append(float("inf"))
            continue
        try:
            rows.append(float(np.linalg.cond(matrix)))
        except np.linalg.LinAlgError:
            rows.append(float("inf"))
    return rows


def _probe_completeness(bank: Mapping[str, Any]) -> dict[str, Any]:
    names = tuple(str(name) for name in bank["names"])
    n_pairs = int(np.asarray(bank["anchor_residual"]).shape[0])
    plus = np.asarray(bank["positive_residual"])
    minus = np.asarray(bank["negative_residual"])
    plus_values = np.asarray(bank["positive_values"], dtype=np.float64)
    minus_values = np.asarray(bank["negative_values"], dtype=np.float64)
    missing = []
    for index, name in enumerate(names):
        if plus.shape[0] <= index or minus.shape[0] <= index:
            missing.append(name)
            continue
        if plus[index].shape[0] != n_pairs or minus[index].shape[0] != n_pairs:
            missing.append(name)
    steps = plus_values - minus_values
    return {
        "n_pairs": n_pairs,
        "n_parameters": int(len(names)),
        "plus_minus_probes_present": not missing and plus.shape[0] == len(names) and minus.shape[0] == len(names),
        "missing_probes": missing,
        "plus_minus_population_aligned": bool(
            plus.shape[1] == n_pairs and minus.shape[1] == n_pairs
        ) if plus.ndim == 3 else False,
        "distinct_plus_minus_steps": bool(np.all(np.abs(steps) > 0.0)),
        "finite_anchor_fraction": _finite_fraction(bank["anchor_residual"]),
        "finite_plus_fraction": _finite_fraction(plus),
        "finite_minus_fraction": _finite_fraction(minus),
        "finite_covariance_fraction": _finite_fraction(bank["covariance"]),
        "held_out_physical_points_loaded": bool(bank.get("held_out_physical_points_loaded")),
        "test_data_accessed": bool(bank.get("test_data_accessed")),
    }


def _column_audit(fit: Any, *, near_zero_relative: float) -> dict[str, Any]:
    derivative = np.asarray(fit.derivative_native, dtype=np.float64)
    # derivative shape [pairs, residual, parameters]
    column_norms = [
        float(np.linalg.norm(derivative[:, :, index]))
        for index in range(derivative.shape[2])
    ]
    maximum = max(column_norms) if column_norms else 0.0
    near_zero = [
        bool(norm <= float(near_zero_relative) * maximum) if maximum > 0.0 else True
        for norm in column_norms
    ]
    return {
        "column_norms": column_norms,
        "near_zero_columns": near_zero,
        "parameter_observability_fraction": [
            float(value) for value in fit.parameter_observability_fraction
        ],
        "normal_matrix_rank": int(fit.normal_matrix_rank),
        "normal_matrix_condition_number": fit.normal_matrix_condition_number,
    }


def _coverage(bank: Mapping[str, Any]) -> dict[str, Any]:
    tx = np.asarray(bank.get("anchor_residual"), dtype=np.float64)
    n_pairs = int(tx.shape[0]) if tx.ndim == 2 else 0
    stations = {
        "source": Counterable(bank.get("source_station_id")),
        "target": Counterable(bank.get("target_station_id")),
    }
    slopes = []
    residual = np.asarray(bank["anchor_residual"], dtype=np.float64)
    if residual.ndim == 2 and residual.shape[1] >= 4:
        rtx = residual[:, 2]
        rty = residual[:, 3]
        slopes = [float(np.hypot(a, b)) for a, b in zip(rtx.tolist(), rty.tolist()) if np.isfinite(a) and np.isfinite(b)]
    return {
        "n_pairs": n_pairs,
        "n_unique_events": int(len({(int(r), int(e)) for r, e in zip(bank["run_id"], bank["event_id"])})),
        "source_station_counts": stations["source"],
        "target_station_counts": stations["target"],
        "residual_abs_slope_proxy_median": float(np.median(slopes)) if slopes else None,
        "residual_abs_slope_proxy_p90": float(np.quantile(slopes, 0.9)) if slopes else None,
        "note": "residual (rtx, rty) is a leftover-slope proxy, not a reconstructed track angle",
    }


def Counterable(values: object) -> dict[str, int]:
    array = np.asarray(values)
    counts: dict[str, int] = {}
    for item in array.tolist():
        key = str(int(item) if isinstance(item, (int, np.integer)) else item)
        counts[key] = counts.get(key, 0) + 1
    return counts


def classify_source(audit: Mapping[str, Any]) -> str:
    flags = list(audit.get("artifact_flags") or [])
    severe = {
        "missing_fd_probes",
        "plus_minus_population_misaligned",
        "non_finite_residuals_or_jacobian",
        "sealed_test_accessed",
        "held_out_points_mixed_into_svd",
        "forced_rank_attempt",
    }
    if any(flag in severe for flag in flags):
        return CLASS_CORRUPTION if not audit.get("low_rank_consistent_with_weak_columns") else CLASS_MIXED
    return CLASS_COVERAGE


def audit_bank(bank: Mapping[str, Any], *, config: Mapping[str, Any]) -> dict[str, Any]:
    rank_tolerance = float(config["rank_tolerance"])
    rcond = float(config["normal_matrix_rcond"])
    refuse_forced_identifiable_rank(
        native_rank=0, requested_rank=None
    )
    completeness = _probe_completeness(bank)
    subspace, extras = subspace_from_physical_bank(
        bank, rank_tolerance=rank_tolerance, rcond=rcond
    )
    fit = extras["fit"]
    columns = _column_audit(fit, near_zero_relative=float(config["near_zero_column_relative"]))
    cov_cond = _covariance_conditions(bank["covariance"])
    finite_cov_cond = [value for value in cov_cond if np.isfinite(value)]
    flags = []
    if not completeness["plus_minus_probes_present"]:
        flags.append("missing_fd_probes")
    if not completeness["plus_minus_population_aligned"]:
        flags.append("plus_minus_population_misaligned")
    if completeness["finite_anchor_fraction"] < 1.0 or completeness["finite_plus_fraction"] < 1.0:
        flags.append("non_finite_residuals_or_jacobian")
    if completeness["test_data_accessed"]:
        flags.append("sealed_test_accessed")
    if completeness["held_out_physical_points_loaded"]:
        flags.append("held_out_points_mixed_into_svd")
    if any(columns["near_zero_columns"]):
        flags.append("near_zero_jacobian_column")
    if finite_cov_cond and max(finite_cov_cond) >= float(config["covariance_condition_warn"]):
        flags.append("ill_conditioned_pair_covariance")
    payload = {
        "source_id": str(bank["source_id"]),
        "split": str(bank["split"]),
        "native_identifiable_rank": int(subspace.identifiable_rank),
        "null_dimension": int(subspace.null_dimension),
        "singular_values": [float(value) for value in subspace.singular_values],
        "truncated_to_five": False,
        "probe_completeness": completeness,
        "columns": columns,
        "coverage": _coverage(bank),
        "covariance_condition_max": max(finite_cov_cond) if finite_cov_cond else None,
        "n_pairs": int(extras["n_pairs"]),
        "subspace": subspace_as_json(subspace),
        "artifact_flags": flags,
        "low_rank_consistent_with_weak_columns": bool(
            int(subspace.identifiable_rank) < 5 and (
                any(columns["near_zero_columns"]) or int(fit.normal_matrix_rank) < 7
            )
        ),
        "not_used_as_stable_core_confirmation": True,
        "not_dropped": True,
    }
    payload["classification"] = classify_source(payload)
    return payload


def build_all_reports(
    config: Mapping[str, Any],
    *,
    banks_by_role: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    created = datetime.now(timezone.utc).isoformat()
    root = project_root()
    failure_ids = list(config["jacobian_corpus"]["failure_source_ids"])
    sibling_ids = list(config["jacobian_corpus"]["sibling_control_source_ids"])
    if banks_by_role is None:
        failure_banks = load_physical_banks(_loader_config(config, failure_ids))
        sibling_banks = load_physical_banks(_loader_config(config, sibling_ids))
    else:
        failure_banks = list(banks_by_role["failure"])
        sibling_banks = list(banks_by_role["sibling"])
    if any(bool(bank.get("test_data_accessed")) for bank in failure_banks + sibling_banks):
        raise ValueError("sealed test was accessed")
    failures = [audit_bank(bank, config=config) for bank in failure_banks]
    siblings = [audit_bank(bank, config=config) for bank in sibling_banks]
    classes = [row["classification"] for row in failures]
    if all(item == CLASS_CORRUPTION for item in classes):
        overall = CLASS_CORRUPTION
    elif all(item == CLASS_COVERAGE for item in classes):
        overall = CLASS_COVERAGE
    else:
        overall = CLASS_MIXED
    reports = {
        "parameter_definition": {
            "schema_version": SCHEMA_VERSION,
            "created_utc": created,
            "git_sha": git_head_sha(root),
            "config_path": config["config_path"],
            "config_sha256": sha256_file(Path(config["config_path"])),
            "operating_state": _operating_state(),
            "inherited_entry_69_decision": INHERITED_ENTRY_69,
            "rank_tolerance": float(config["rank_tolerance"]),
            "scale_matrix_S": dict(config["scale_matrix_S"]),
            "failure_source_ids": failure_ids,
            "sibling_control_source_ids": sibling_ids,
            "assumptions": list(config.get("assumptions", [])),
            "not_a_stable_core_confirmation": True,
            "not_a_cluster_local_gate": True,
        },
        "failure_sources": failures,
        "sibling_controls": siblings,
        "next_stage_decision": {
            "decision": "tracklet_independent_failure_provenance_audit_complete_sources_not_dropped",
            "inherited_entry_69_decision": INHERITED_ENTRY_69,
            "overall_classification": overall,
            "n_failure_sources": int(len(failures)),
            "classifications": {row["source_id"]: row["classification"] for row in failures},
            "native_ranks": {row["source_id"]: row["native_identifiable_rank"] for row in failures},
            "sources_dropped": False,
            "rank_forced_to_five": False,
            "stable_core_criterion_redefined": False,
            "used_as_cluster_local_confirmation": False,
            "geometry_write_allowed": False,
            "if_failed_do_not_chase_by_dropping_sources": True,
        },
    }
    for payload in reports.values():
        if isinstance(payload, Mapping):
            assert_no_alignment_payload(payload)
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, Mapping):
                    assert_no_alignment_payload(item)
    return reports


def refuse_forbidden_operations() -> dict[str, Any]:
    try:
        svd_naked_jacobian(np.eye(3), parameter_units=("mm", "mrad", "mm"))
        naked = False
    except MixedUnitNakedJacobianError:
        naked = True
    return {
        "refused_naked_mixed_unit_svd": naked,
        "geometry_write_allowed": False,
        "sources_dropped": False,
        "opened_sealed_test": False,
        "stable_core_reopened": False,
    }
