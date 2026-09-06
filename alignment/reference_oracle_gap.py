"""T10: paired-target WLS vs zero-target WLS on frozen iteration-01 arrays.

This is a diagnostic.  Zero-target is not a field-aware likelihood.
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

SCHEMA_VERSION = "reference-oracle-gap-v1"
DEFAULT_CONFIG = "configs/reference_oracle_gap_v1.yaml"
DX_DY_GATE_MM = 0.1
RY_GATE_MRAD = 1.0
FORBIDDEN_ZERO_TARGET_KEYS = (
    "target_residual",
    "target_payload",
    "truth_displacement",
)


class ReferenceOracleError(ValueError):
    """Raised when the zero-target estimator is given a reference oracle."""


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ReferenceOracleError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != "T10":
        raise ReferenceOracleError("task must be T10")
    if bool(config.get("geometry_write_allowed", True)):
        raise ReferenceOracleError("geometry_write_allowed must be false")
    if bool(config.get("do_not_reread_sealed_test", False)) is not True:
        raise ReferenceOracleError("sealed test must stay closed")
    return dict(config)


def load_frozen_arrays(relative: str | Path) -> dict[str, Any]:
    path = resolve_under_root(project_root(), relative)
    with np.load(path, allow_pickle=False) as handle:
        payload = {key: handle[key] for key in handle.files}
    payload["path"] = str(path)
    payload["sha256"] = sha256_file(path)
    return payload


def load_frozen_report(relative: str | Path) -> dict[str, Any]:
    path = resolve_under_root(project_root(), relative)
    return json.loads(path.read_text(encoding="utf-8"))


def wls_delta(
    jacobian: np.ndarray,
    covariance: np.ndarray,
    residual: np.ndarray,
) -> np.ndarray:
    """delta = N^{-1} J^T C^{-1} residual on the frozen selection."""
    design = np.asarray(jacobian, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    rhs_residual = np.asarray(residual, dtype=np.float64)
    weighted_residual = np.linalg.solve(cov, rhs_residual[..., None])[..., 0]
    normal = np.einsum("nki,nkj->ij", design, np.linalg.solve(cov, design))
    rhs = np.einsum("nki,nk->i", design, weighted_residual)
    return np.linalg.solve(normal, rhs)


def estimate_paired_target(
    jacobian: np.ndarray,
    covariance: np.ndarray,
    anchor_residual: np.ndarray,
    target_residual: np.ndarray,
) -> np.ndarray:
    return wls_delta(jacobian, covariance, target_residual - anchor_residual)


def estimate_zero_target(
    jacobian: np.ndarray,
    covariance: np.ndarray,
    anchor_residual: np.ndarray,
    **kwargs: Any,
) -> np.ndarray:
    """No-reference estimator.  Target residual / payload / truth are forbidden."""
    extra = set(kwargs) & set(FORBIDDEN_ZERO_TARGET_KEYS)
    if extra:
        raise ReferenceOracleError(
            "zero-target estimator refuses reference oracle fields: "
            + ", ".join(sorted(extra))
        )
    return wls_delta(jacobian, covariance, -np.asarray(anchor_residual, dtype=np.float64))


def saved_paired_delta(report: Mapping[str, Any], names: list[str]) -> np.ndarray:
    by_name = {row["name"]: row for row in report["parameters"]}
    return np.asarray(
        [float(by_name[name]["recovered_local_delta"]) for name in names],
        dtype=np.float64,
    )


def saved_anchor(report: Mapping[str, Any], names: list[str]) -> np.ndarray:
    by_name = {row["name"]: row for row in report["parameters"]}
    return np.asarray([float(by_name[name]["anchor_value"]) for name in names], dtype=np.float64)


def parse_observation_key(raw: str) -> dict[str, Any]:
    key = json.loads(str(raw))
    sample, run_id, event_id, signature, source_station, target_station = key[:6]
    origin = json.loads(signature) if isinstance(signature, str) else signature
    first = origin[0]
    return {
        "sample_id": str(sample),
        "overlay_run_id": int(run_id),
        "overlay_event_id": int(event_id),
        "origin_run_id": int(first["origin_run_id"]),
        "origin_event_id": int(first["origin_event_id"]),
        "source_station_id": int(source_station),
        "target_station_id": int(target_station),
        "source_uid": f"{sample}:{first['origin_run_id']}",
        "event_key": (str(sample), int(first["origin_run_id"]), int(first["origin_event_id"])),
    }


def source_leave_one_out(
    arrays: Mapping[str, Any],
    *,
    residual: np.ndarray,
) -> list[dict[str, Any]]:
    keys = [parse_observation_key(item) for item in arrays["observation_keys"]]
    sources = sorted({row["source_uid"] for row in keys})
    jacobian = np.asarray(arrays["derivative_native"], dtype=np.float64)
    covariance = np.asarray(arrays["covariance"], dtype=np.float64)
    names = [str(name) for name in arrays["parameter_names"].tolist()]
    rows: list[dict[str, Any]] = []
    for source in sources:
        mask = np.array([row["source_uid"] != source for row in keys], dtype=bool)
        if int(mask.sum()) < 3:
            continue
        delta = wls_delta(jacobian[mask], covariance[mask], residual[mask])
        rows.append(
            {
                "source_uid": source,
                "n_held_in": int(mask.sum()),
                "n_held_out": int((~mask).sum()),
                **{name: float(delta[index]) for index, name in enumerate(names)},
            }
        )
    return rows


def gate_vector(delta: np.ndarray, names: list[str]) -> dict[str, Any]:
    gates = {}
    passed = True
    for index, name in enumerate(names):
        value = float(delta[index])
        limit = RY_GATE_MRAD if name.endswith("ry_mrad") else DX_DY_GATE_MM
        ok = abs(value) <= limit
        gates[name] = {"value": value, "limit": limit, "pass": ok}
        passed = passed and ok
    return {"gates": gates, "all_pass": passed}


def evaluate_split(
    arrays: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    split: str,
) -> dict[str, Any]:
    names = [str(name) for name in arrays["parameter_names"].tolist()]
    jacobian = np.asarray(arrays["derivative_native"], dtype=np.float64)
    covariance = np.asarray(arrays["covariance"], dtype=np.float64)
    anchor = np.asarray(arrays["anchor_residual"], dtype=np.float64)
    target = np.asarray(arrays["target_residual"], dtype=np.float64)
    paired = estimate_paired_target(jacobian, covariance, anchor, target)
    zero = estimate_zero_target(jacobian, covariance, anchor)
    reference = wls_delta(jacobian, covariance, target)
    saved = saved_paired_delta(report, names)
    anchors = saved_anchor(report, names)
    remaining_paired = anchors + paired
    remaining_zero = anchors + zero
    paired_match = bool(np.allclose(paired, saved, rtol=1.0e-8, atol=1.0e-8))
    paired_gate = gate_vector(remaining_paired, names)
    zero_gate = gate_vector(remaining_zero, names)
    normal_rank = int(np.linalg.matrix_rank(np.asarray(arrays["normal_matrix_scaled"])))
    return {
        "split": split,
        "n_edges": int(anchor.shape[0]),
        "parameter_names": names,
        "paired_delta": [float(value) for value in paired],
        "zero_delta": [float(value) for value in zero],
        "reference_projection": [float(value) for value in reference],
        "saved_paired_delta": [float(value) for value in saved],
        "anchor_value": [float(value) for value in anchors],
        "remaining_paired": [float(value) for value in remaining_paired],
        "remaining_zero": [float(value) for value in remaining_zero],
        "paired_matches_saved_1e-8": paired_match,
        "paired_remaining_gate": paired_gate,
        "zero_remaining_gate": zero_gate,
        "legacy_operational_rank": normal_rank,
        "source_effects_zero": source_leave_one_out(arrays, residual=-anchor),
        "source_effects_paired": source_leave_one_out(arrays, residual=target - anchor),
        "is_correct_likelihood": False,
        "diagnostic_only": True,
    }


def decide(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    zero_pass = all(item["zero_remaining_gate"]["all_pass"] for item in results.values())
    paired_pass = all(item["paired_remaining_gate"]["all_pass"] for item in results.values())
    saved_ok = all(item["paired_matches_saved_1e-8"] for item in results.values())
    success = bool(zero_pass and paired_pass and saved_ok)
    return {
        "kind": "reference_oracle_gap_decision",
        "task": "T10",
        "verdict": "PASS" if success else "FAIL",
        "paired_reproduced": saved_ok,
        "paired_remaining_pass": paired_pass,
        "zero_remaining_pass": zero_pass,
        "old_wls_is_diagnostic_only": True,
        "old_wls_authorized_for_data_correction": False,
        "zero_target_is_not_correct_likelihood": True,
        "geometry_write_allowed": False,
        "held_out_accessed": False,
        "reason": (
            "zero-target and paired both pass the frozen 0.1 mm / 1 mrad remaining-error gates"
            if success
            else "zero-target remaining error fails the frozen gates or depends on the reference residual"
        ),
    }
