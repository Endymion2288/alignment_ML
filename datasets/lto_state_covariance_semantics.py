"""Task B14: LTO state / covariance semantics before propagation.

Compares official WB107 full-track Cin with the B13 LTO Cin on the
frozen WB103 contracted identities.  Acceptance is the pre-propagation
uncertainty contract, not V4 C/D closure.  Does not rescale, clip,
PSD-project, drop 100043/37, or enter B15 / Measurement Model V2.
"""

from __future__ import annotations

import json
from collections import defaultdict
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
from datasets.acts_process_noise_contract import _pick_truth, _truth_residual
from datasets.acts_transport_diagnosis import _stats
from datasets.acts_transport_dump import _as_matrix
from datasets.acts_transport_tail_analysis import identity_key
from datasets.leave_target_out_state_materialization import (
    DECISION_MATERIALIZED as WB109_DECISION,
    FROZEN_N_CONTRACTED,
    FROZEN_N_INELIGIBLE,
    FROZEN_N_OFFICIAL_PAIRS,
    FROZEN_N_RAW,
    _row_key,
    inherit_frozen_stage as inherit_through_wb108,
    load_lto_dumps,
)
from datasets.transport_uncertainty_shape_diagnosis import (
    STATE_NAMES,
    _shape_against,
    audit_cin_shape,
    load_contracted_sample,
)

SCHEMA_VERSION = "lto-state-covariance-semantics-v1"
DEFAULT_CONFIG = "configs/lto_state_covariance_semantics_v1.yaml"
TASK = "SB-B14"
WORKBOOK = 110

DECISION_ESTABLISHED = "lto_state_covariance_semantics_established"
DECISION_NOT_VALIDATED = "lto_input_covariance_shape_not_validated"
DECISION_MIXED = "mixed_or_inconclusive"
DECISION_BLOCKED = "lto_states_not_available_for_semantics"

FROZEN_WB105_LAMBDAS = (
    0.026795151736981965,
    0.05412438924744891,
    0.15259249246272533,
    11.349200687274577,
)
FROZEN_WB105_PENCIL = 6.798273330334669
MATERIAL_REDUCTION_FACTOR = 2.0


class LtoSemanticsError(ValueError):
    """Raised when the B14 contract is illegal."""


def refuse_truth_qoverp() -> None:
    raise LtoSemanticsError("truth q/p is not a real-data solution")


def refuse_covariance_rescale() -> None:
    raise LtoSemanticsError("covariance rescale is forbidden")


def refuse_q_psd_projection() -> None:
    raise LtoSemanticsError("Q/Cin must not be PSD-projected to pass a gate")


def refuse_empirical_cross_covariance() -> None:
    raise LtoSemanticsError("empirical Cov(pred,target) must not be invented")


def refuse_focus_drop() -> None:
    raise LtoSemanticsError("focus identity 100043/37 must be retained")


def refuse_measurement_model_v2() -> None:
    raise LtoSemanticsError("Measurement Model V2 is not entered in Task B14")


def refuse_b15() -> None:
    raise LtoSemanticsError("Task B15 is not entered until the LTO Cin contract holds")


def refuse_raw_ckf() -> None:
    raise LtoSemanticsError("raw CKF must not re-enter the official B14 input")


def refuse_tighten_contract() -> None:
    raise LtoSemanticsError("reconstruction contract must not be tightened")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise LtoSemanticsError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise LtoSemanticsError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise LtoSemanticsError(f"workbook must be {WORKBOOK}")
    if str(config.get("official_input_scope")) != "contract_eligible":
        raise LtoSemanticsError("official input must be contract_eligible")
    for key in (
        "geometry_write_allowed",
        "held_out_accessed",
        "real_data_alignment_authorized",
        "measurement_model_validated",
    ):
        if bool(config.get(key, True)):
            raise LtoSemanticsError(f"{key} must be false")
    for key in (
        "do_not_rescale_covariance",
        "do_not_use_truth_q_over_p_as_real_data_solution",
        "do_not_invent_empirical_cross_covariance",
        "do_not_use_raw_ckf_as_official_input",
        "do_not_tighten_reconstruction_contract",
        "do_not_drop_focus_identity",
        "do_not_enter_measurement_model_v2",
        "do_not_enter_b15_without_lto_cin_contract",
        "do_not_claim_transport_covariance_validated",
        "do_not_rescale_or_clip_lto_cin",
        "eligibility_independent_of_closure",
        "do_not_select_states_from_closure",
    ):
        if bool(config.get(key, False)) is not True:
            raise LtoSemanticsError(f"{key} must be true")
    gates = config.get("closure_gates", {})
    for key, expected in FROZEN_WB81_GATES.items():
        if gates.get(key) != expected:
            raise LtoSemanticsError(f"WB81/WB87/WB98 gate must stay frozen: {key}")
    wb103 = yaml.safe_load(
        resolve_under_root(
            project_root(), config["inheritance"]["workbook_103"]["config_path"]
        ).read_text(encoding="utf-8")
    )
    if dict(config["eligibility"]) != dict(wb103["eligibility"]):
        raise LtoSemanticsError("B14 eligibility must stay identical to WB103")
    if Path(str(config.get("output_root"))).as_posix() in {
        Path(str(config.get("dump_root"))).as_posix(),
        Path(str(config.get("lto_dump_root"))).as_posix(),
    }:
        raise LtoSemanticsError("B14 must not write into a dump root")
    digest = sha256_file(resolve_under_root(project_root(), config["wb105_cin_path"]))
    if digest != config["wb105_cin_sha256"]:
        raise LtoSemanticsError(f"WB105 Cin artifact hash mismatch: {digest}")
    return dict(config)


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited = inherit_through_wb108(config)
    spec = config["inheritance"]["workbook_109"]
    decision = json.loads(
        resolve_under_root(project_root(), spec["decision_path"]).read_text(encoding="utf-8")
    )
    digest = sha256_file(resolve_under_root(project_root(), spec["config_path"]))
    if digest != spec["config_sha256"]:
        raise LtoSemanticsError(f"workbook_109 config hash mismatch: {digest}")
    digest = sha256_file(resolve_under_root(project_root(), spec["decision_path"]))
    if digest != spec["decision_sha256"]:
        raise LtoSemanticsError(f"workbook_109 decision hash mismatch: {digest}")
    if decision.get("decision") != spec["frozen_decision"]:
        raise LtoSemanticsError("workbook_109 decision must stay frozen")
    if decision.get("primary_case") != spec["frozen_primary_case"]:
        raise LtoSemanticsError("workbook_109 primary case must stay frozen")
    if spec["frozen_decision"] != WB109_DECISION:
        raise LtoSemanticsError("WB109 decision token mismatch")
    if not decision.get("b14_authorized"):
        raise LtoSemanticsError("B14 is not authorized by the frozen WB109 decision")
    inherited["workbook_109"] = {
        "decision": spec["frozen_decision"],
        "primary_case": spec["frozen_primary_case"],
        "config_sha256": spec["config_sha256"],
        "decision_sha256": spec["decision_sha256"],
        "b14_authorized": True,
        "b15_authorized": False,
    }
    return inherited


def _focus_keys(config: Mapping[str, Any]) -> set[tuple[Any, ...]]:
    return {
        (str(item["source_id"]), int(item["run_id"]), int(item["event_id"]))
        for item in config["focus_identities"]
    }


def _sigma_q(matrix: np.ndarray | None) -> float | None:
    if matrix is None or matrix.shape != (5, 5):
        return None
    value = float(matrix[4, 4])
    if not np.isfinite(value) or value < 0.0:
        return None
    return float(np.sqrt(value))


def _correlations(matrix: np.ndarray) -> dict[str, float | None]:
    diag = np.sqrt(np.clip(np.diag(matrix), 0.0, None))
    out: dict[str, float | None] = {}
    if diag[4] <= 0.0:
        return {name: None for name in ("x", "y", "tx", "ty")}
    for index, name in enumerate(("x", "y", "tx", "ty")):
        out[name] = (
            float(matrix[index, 4] / (diag[index] * diag[4])) if diag[index] > 0.0 else None
        )
    return out


def _population(matrices: list[np.ndarray]) -> dict[str, Any]:
    spectra = []
    conditions = []
    sigmas = []
    diagonals: dict[str, list[float]] = {
        name: [] for name in ("x", "y", "tx", "ty", "q_over_p")
    }
    corrs: dict[str, list[float]] = {name: [] for name in ("x", "y", "tx", "ty")}
    for matrix in matrices:
        eig = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        spectra.append([float(v) for v in eig])
        conditions.append(float(np.linalg.cond(matrix)))
        sigma = _sigma_q(matrix)
        if sigma is not None:
            sigmas.append(sigma)
        for index, name in enumerate(("x", "y", "tx", "ty", "q_over_p")):
            diagonals[name].append(float(matrix[index, index]))
        for name, value in _correlations(matrix).items():
            if value is not None:
                corrs[name].append(value)
    return {
        "n": len(matrices),
        "eigen_min": _stats([spec[0] for spec in spectra]),
        "eigen_max": _stats([spec[-1] for spec in spectra]),
        "condition_number": _stats(conditions),
        "sigma_qoverp": _stats(sigmas),
        "diagonal_variance": {name: _stats(values) for name, values in diagonals.items()},
        "qoverp_correlations": {name: _stats(values) for name, values in corrs.items()},
    }


def _shape_from_pairs(
    residuals: list[np.ndarray],
    cins: list[np.ndarray],
) -> dict[str, Any] | None:
    if len(residuals) < 8:
        return None
    array = np.asarray(residuals, dtype=np.float64)
    c_emp = np.cov((array - array.mean(axis=0)).T)
    c_model = np.mean(np.asarray(cins, dtype=np.float64), axis=0)
    return _shape_against(c_emp, c_model)


def _pulls(residuals: list[np.ndarray], cins: list[np.ndarray]) -> dict[str, Any]:
    values: dict[str, list[float]] = {name: [] for name in STATE_NAMES}
    for residual, cov in zip(residuals, cins):
        for index, name in enumerate(STATE_NAMES):
            sigma = float(np.sqrt(max(float(cov[index, index]), 0.0)))
            if sigma > 0.0 and np.isfinite(residual[index]):
                values[name].append(float(residual[index] / sigma))
    return {name: _stats(items) for name, items in values.items()}


def _truth_qoverp(payload: Mapping[str, Any], charge: float | None) -> float | None:
    mom = np.asarray(payload.get("mom"), dtype=np.float64).reshape(-1)
    if mom.size < 3 or not np.all(np.isfinite(mom)):
        return None
    momentum = float(np.linalg.norm(mom))
    if momentum <= 0.0 or charge is None or not np.isfinite(float(charge)):
        return None
    return float(charge) / momentum


def _official_residual(
    config: Mapping[str, Any],
    sample: Mapping[str, Any],
    row: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    cin = _as_matrix(row.get("input_covariance"), 5)
    state = np.asarray(row.get("derived_state"), dtype=np.float64).reshape(-1)
    if cin is None or state.size < 4:
        return None
    table = sample["truth_tables"].get(str(row.get("source_id")))
    event_truth = table.get((int(row["run_id"]), int(row["event_id"]))) if table else None
    if not event_truth:
        return None
    _, src = _pick_truth(event_truth, int(row["source_station"]))
    if src is None:
        return None
    residual = _truth_residual(
        state[:4],
        src["pos"],
        src["mom"],
        float(row.get("source_z_mm", src["pos"][2])),
        straight_line=bool(config["truth_reference"]["straight_line_z_correction"]),
    )
    if residual is None:
        return None
    charge = row.get("charge")
    if charge is None and row.get("q_over_p_per_mev") is not None:
        charge = float(np.sign(float(row["q_over_p_per_mev"])) or 0.0)
    extra = {
        "p_truth_mev": float(np.linalg.norm(src["mom"])),
        "q_over_p_truth": _truth_qoverp(src, None if charge is None else float(charge)),
    }
    return residual, cin, extra


def _lto_residual(
    config: Mapping[str, Any],
    sample: Mapping[str, Any],
    official: Mapping[str, Any],
    lto: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    cin = _as_matrix(lto.get("input_covariance") or lto.get("native_covariance"), 5)
    state = np.asarray(lto.get("derived_state"), dtype=np.float64).reshape(-1)
    if cin is None or state.size < 4:
        return None
    table = sample["truth_tables"].get(str(official.get("source_id")))
    event_truth = table.get((int(official["run_id"]), int(official["event_id"]))) if table else None
    if not event_truth:
        return None
    _, src = _pick_truth(event_truth, int(official["source_station"]))
    if src is None:
        return None
    surface = lto.get("reference_surface") or {}
    z_mm = surface.get("z_mm")
    if z_mm is None:
        pos = lto.get("fitted_position_xyz_mm") or []
        z_mm = pos[2] if len(pos) >= 3 else official.get("source_z_mm", src["pos"][2])
    residual = _truth_residual(
        state[:4],
        src["pos"],
        src["mom"],
        float(z_mm),
        straight_line=bool(config["truth_reference"]["straight_line_z_correction"]),
    )
    if residual is None:
        return None
    charge = lto.get("charge")
    if charge is None and lto.get("q_over_p_per_mev") is not None:
        charge = float(np.sign(float(lto["q_over_p_per_mev"])) or 0.0)
    extra = {
        "p_truth_mev": float(np.linalg.norm(src["mom"])),
        "q_over_p_truth": _truth_qoverp(src, None if charge is None else float(charge)),
    }
    return residual, cin, extra


def _pack_shape(
    residuals: list[np.ndarray],
    cins: list[np.ndarray],
) -> dict[str, Any]:
    shape = _shape_from_pairs(residuals, cins)
    return {
        "n_truth_matched": len(residuals),
        "shape": shape,
        "shape_holds": bool(shape and shape.get("shape_holds")),
        "overwide": bool(shape and shape.get("overwide")),
        "undercovered": bool(
            shape
            and (
                any(float(v) > 4.0 for v in (shape.get("generalized_eigenvalues") or []))
                or (
                    shape.get("pencil_ratio") is not None
                    and float(shape["pencil_ratio"]) < 0.25
                )
            )
        ),
        "pulls": _pulls(residuals, cins),
    }


def audit_lto_shapes(
    config: Mapping[str, Any],
    sample: Mapping[str, Any],
    lto_by_key: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> dict[str, Any]:
    by_split: dict[str, dict[str, list]] = defaultdict(lambda: {"residuals": [], "cins": []})
    by_target: dict[str, dict[str, list]] = defaultdict(lambda: {"residuals": [], "cins": []})
    by_source: dict[str, dict[str, list]] = defaultdict(lambda: {"residuals": [], "cins": []})
    all_residuals: list[np.ndarray] = []
    all_cins: list[np.ndarray] = []
    qop_pulls: list[float] = []
    matrices: list[np.ndarray] = []
    n_success = 0
    n_missing = 0
    for row in sample["contracted"]:
        lto = lto_by_key.get(_row_key(row))
        if lto is None or not bool(lto.get("fit_success")):
            n_missing += 1
            continue
        n_success += 1
        matrix = _as_matrix(lto.get("input_covariance"), 5)
        if matrix is not None:
            matrices.append(matrix)
        matched = _lto_residual(config, sample, row, lto)
        if matched is None:
            continue
        residual, cin, extra = matched
        split = str(row.get("split") or "construction")
        target = str(int(row["target_station"]))
        source = str(row["source_id"])
        for bucket in (by_split[split], by_target[target], by_source[source]):
            bucket["residuals"].append(residual)
            bucket["cins"].append(cin[:4, :4])
        all_residuals.append(residual)
        all_cins.append(cin[:4, :4])
        q_fit = lto.get("q_over_p_per_mev")
        q_truth = extra.get("q_over_p_truth")
        sigma = _sigma_q(cin)
        if (
            q_fit is not None
            and q_truth is not None
            and sigma not in (None, 0.0)
            and np.isfinite(float(q_fit))
            and np.isfinite(float(q_truth))
        ):
            qop_pulls.append(float((float(q_fit) - float(q_truth)) / sigma))
    packed = {
        "all": _pack_shape(all_residuals, all_cins),
        "construction": _pack_shape(
            by_split["construction"]["residuals"], by_split["construction"]["cins"]
        ),
        "validation": _pack_shape(
            by_split["validation"]["residuals"], by_split["validation"]["cins"]
        ),
        "by_target": {
            key: _pack_shape(bucket["residuals"], bucket["cins"])
            for key, bucket in sorted(by_target.items())
        },
        "by_source": {
            key: _pack_shape(bucket["residuals"], bucket["cins"])
            for key, bucket in sorted(by_source.items())
        },
        "population": _population(matrices),
        "qoverp_pulls": _stats(qop_pulls),
        "n_contracted_success": n_success,
        "n_contracted_missing_or_failed": n_missing,
        "truth_used_as_diagnostic_only": True,
    }
    return packed


def compare_official_vs_lto(
    config: Mapping[str, Any],
    sample: Mapping[str, Any],
    lto_by_key: Mapping[tuple[Any, ...], Mapping[str, Any]],
    official_shape: Mapping[str, Any],
    lto_shape: Mapping[str, Any],
) -> dict[str, Any]:
    n_compared = 0
    n_identical = 0
    official_matrices = []
    seen: set[tuple[Any, ...]] = set()
    for row in sample["contracted"]:
        official = _as_matrix(row.get("input_covariance"), 5)
        key = (
            str(row.get("source_id")),
            int(row.get("run_id", -1)),
            int(row.get("event_id", -1)),
            int(row.get("track_index", -1)),
        )
        if official is not None and key not in seen:
            seen.add(key)
            official_matrices.append(official)
        lto = lto_by_key.get(_row_key(row))
        lto_cov = _as_matrix((lto or {}).get("input_covariance"), 5) if lto else None
        if official is None or lto_cov is None:
            continue
        n_compared += 1
        if np.allclose(official, lto_cov, rtol=0.0, atol=0.0):
            n_identical += 1
    frozen = official_shape.get("source_cin_vs_empirical_source_error") or {}
    current = [float(v) for v in (frozen.get("generalized_eigenvalues") or [])]
    lto_all = (lto_shape.get("all") or {}).get("shape") or {}
    lto_lams = [float(v) for v in (lto_all.get("generalized_eigenvalues") or [])]
    lto_pencil = lto_all.get("pencil_ratio")
    min_official = min(FROZEN_WB105_LAMBDAS)
    min_lto = min(lto_lams) if lto_lams else None
    pencil_reduced = bool(
        lto_pencil is not None
        and float(lto_pencil) <= FROZEN_WB105_PENCIL / MATERIAL_REDUCTION_FACTOR
    )
    lambda_increased = bool(
        min_lto is not None and min_lto >= MATERIAL_REDUCTION_FACTOR * min_official
    )
    return {
        "n_compared": n_compared,
        "n_bit_identical": n_identical,
        "official_WB107_Cin_reused": bool(n_compared and n_identical > 0),
        "official_population": _population(official_matrices),
        "this_run_official_lambdas": current,
        "frozen_wb105_lambdas": list(FROZEN_WB105_LAMBDAS),
        "frozen_wb105_pencil": FROZEN_WB105_PENCIL,
        "lto_lambdas": lto_lams,
        "lto_pencil": lto_pencil,
        "overcoverage_disappeared": bool((lto_shape.get("all") or {}).get("shape_holds")),
        "overcoverage_materially_reduced": bool(lambda_increased or pencil_reduced),
        "lto_still_overwide": bool((lto_shape.get("all") or {}).get("overwide")),
        "lto_still_undercovered": bool((lto_shape.get("all") or {}).get("undercovered")),
        "shared_measurement_leakage_isolated": None,
        "truth_used_as_diagnostic_only": True,
    }


def audit_focus(
    config: Mapping[str, Any],
    sample: Mapping[str, Any],
    lto_by_key: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> dict[str, Any]:
    focus = _focus_keys(config)
    rows = []
    for official in sample["contracted"]:
        if identity_key(official) not in focus:
            continue
        lto = lto_by_key.get(_row_key(official))
        official_cov = _as_matrix(official.get("input_covariance"), 5)
        lto_cov = _as_matrix((lto or {}).get("input_covariance"), 5) if lto else None
        matched = None
        if lto is not None:
            matched = _lto_residual(config, sample, official, lto)
        extra = matched[2] if matched else {}
        p_truth = extra.get("p_truth_mev")
        q_lto = None if lto is None else lto.get("q_over_p_per_mev")
        p_lto = None if not q_lto else abs(1.0 / float(q_lto))
        q_off = official.get("q_over_p_per_mev")
        p_off = None if not q_off else abs(1.0 / float(q_off))
        def _ratio(reco: float | None) -> float | None:
            if reco is None or p_truth in (None, 0.0):
                return None
            return float(abs(np.log10(max(reco, 1.0e-30) / max(float(p_truth), 1.0e-30))))

        rows.append(
            {
                "source_id": official.get("source_id"),
                "run_id": official.get("run_id"),
                "event_id": official.get("event_id"),
                "target_station": official.get("target_station"),
                "official_q_over_p_per_mev": q_off,
                "official_p_mev": p_off,
                "official_sigma_qoverp": _sigma_q(official_cov),
                "lto_fit_success": None if lto is None else lto.get("fit_success"),
                "lto_q_over_p_per_mev": q_lto,
                "lto_p_mev": p_lto,
                "lto_sigma_qoverp": _sigma_q(lto_cov),
                "lto_chi2": None if lto is None else lto.get("chi2"),
                "lto_n_measurements_in_fit": None if lto is None else lto.get(
                    "n_measurements_in_fit"
                ),
                "p_truth_mev": p_truth,
                "q_over_p_truth": extra.get("q_over_p_truth"),
                "official_abs_log10_p_ratio": _ratio(p_off),
                "lto_abs_log10_p_ratio": _ratio(p_lto),
            }
        )
    q_lto_values = [
        float(row["lto_q_over_p_per_mev"])
        for row in rows
        if row.get("lto_q_over_p_per_mev") is not None
    ]
    official_q = [
        float(row["official_q_over_p_per_mev"])
        for row in rows
        if row.get("official_q_over_p_per_mev") is not None
    ]
    identical_lto = bool(q_lto_values and max(q_lto_values) - min(q_lto_values) == 0.0)
    same_as_official = bool(
        q_lto_values
        and official_q
        and abs(q_lto_values[0] - official_q[0]) / max(abs(official_q[0]), 1.0e-30) < 1.0e-6
    )
    seedlike = bool(
        rows
        and all(
            row.get("lto_sigma_qoverp") is not None
            and abs(float(row["lto_sigma_qoverp"]) - 1.0e-3) / 1.0e-3 < 1.0e-3
            for row in rows
        )
    )
    remains = bool(rows) and (
        identical_lto
        or same_as_official
        or any(
            (row.get("lto_abs_log10_p_ratio") or 0.0) >= 0.3
            for row in rows
        )
    )
    return {
        "retained": bool(rows),
        "n_rows": len(rows),
        "targets": rows,
        "lto_qoverp_identical_across_targets": identical_lto,
        "lto_qoverp_matches_official": same_as_official,
        "lto_qoverp_seedlike": seedlike,
        "momentum_anomaly_remains": remains,
        "downweighted": False,
        "deleted": False,
        "truth_qoverp_used_in_fit": False,
        "truth_used_as_diagnostic_only": True,
    }


def decide_case(inventory: Mapping[str, Any]) -> dict[str, Any]:
    dumps = inventory.get("dumps") or {}
    lto = inventory.get("lto_shape") or {}
    comparison = inventory.get("comparison") or {}
    focus = inventory.get("focus") or {}
    denom = inventory.get("denominator") or {}
    construction = (lto.get("construction") or {}).get("shape_holds")
    validation = (lto.get("validation") or {}).get("shape_holds")
    if not dumps.get("dumps_present"):
        primary = DECISION_BLOCKED
        verdict = "BLOCKED"
        next_step = "materialize_lto_states_before_semantics"
    elif denom.get("frozen_denominator_holds") is False:
        primary = "wb103_denominator_redefined"
        verdict = "FAIL"
        next_step = "restore_frozen_wb103_contracted_denominator"
    elif comparison.get("official_WB107_Cin_reused"):
        primary = "official_wb107_cin_reused"
        verdict = "FAIL"
        next_step = "keep_independent_lto_covariance"
    elif construction and validation:
        primary = DECISION_ESTABLISHED
        verdict = "PASS"
        next_step = "transport_covariance_v4_lto_arms"
    elif (lto.get("all") or {}).get("shape_holds"):
        primary = DECISION_MIXED
        verdict = "MIXED"
        next_step = "do_not_enter_b15_without_split_contract"
    elif comparison.get("lto_still_overwide") or comparison.get("lto_still_undercovered"):
        primary = DECISION_NOT_VALIDATED
        verdict = "FAIL"
        next_step = "do_not_enter_b15_lto_cin_not_validated"
    else:
        primary = DECISION_MIXED
        verdict = "MIXED"
        next_step = "do_not_enter_b15_without_cin_contract"
    if inventory.get("empirical_cross_covariance_invented"):
        refuse_empirical_cross_covariance()
    if inventory.get("covariance_rescaled"):
        refuse_covariance_rescale()
    leakage_isolated = bool(
        comparison.get("overcoverage_disappeared")
        or (
            comparison.get("overcoverage_materially_reduced")
            and not comparison.get("lto_still_overwide")
        )
    )
    return {
        "verdict": verdict,
        "decision": (
            DECISION_ESTABLISHED
            if verdict == "PASS"
            else DECISION_BLOCKED
            if verdict == "BLOCKED"
            else DECISION_NOT_VALIDATED
            if verdict == "FAIL"
            else DECISION_MIXED
        ),
        "primary_case": primary,
        "next_step": next_step,
        "lto_cin_contract_established": verdict == "PASS",
        "overcoverage_disappeared": bool(comparison.get("overcoverage_disappeared")),
        "overcoverage_materially_reduced": bool(
            comparison.get("overcoverage_materially_reduced")
        ),
        "shared_measurement_leakage_is_shape_source": leakage_isolated,
        "lto_still_overwide": bool(comparison.get("lto_still_overwide")),
        "lto_still_undercovered": bool(comparison.get("lto_still_undercovered")),
        "focus_momentum_anomaly_remains": bool(focus.get("momentum_anomaly_remains")),
        "official_WB107_Cin_reused": bool(comparison.get("official_WB107_Cin_reused")),
        "transport_covariance_validated": False,
        "measurement_model_v2_authorized": False,
        "measurement_model_v2_entered": False,
        "b15_authorized": verdict == "PASS",
        "focus_identity_retained": bool(focus.get("retained", True)),
        "truth_used_as_diagnostic_only": True,
        "covariance_rescaled": False,
        "q_psd_projected": False,
        "empirical_cross_covariance_invented": False,
    }


def inventory_and_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    dumps = load_lto_dumps(config)
    sample = load_contracted_sample(config)
    if not any(identity_key(row) in _focus_keys(config) for row in sample["contracted"]):
        refuse_focus_drop()
    frozen_ok = bool(
        int(sample["n_raw"]) == FROZEN_N_RAW
        and int(sample["n_ineligible"]) == FROZEN_N_INELIGIBLE
        and len(sample["contracted"]) == FROZEN_N_CONTRACTED
        and len(sample["official_events"]) == FROZEN_N_OFFICIAL_PAIRS
    )
    lto_by_key = {_row_key(row): row for row in dumps["rows"]}
    official_shape = audit_cin_shape(config, sample)
    lto_shape = audit_lto_shapes(config, sample, lto_by_key)
    comparison = compare_official_vs_lto(
        config, sample, lto_by_key, official_shape, lto_shape
    )
    if comparison["overcoverage_disappeared"] or comparison["overcoverage_materially_reduced"]:
        comparison["shared_measurement_leakage_isolated"] = bool(
            comparison["overcoverage_disappeared"]
            or comparison["overcoverage_materially_reduced"]
        )
    focus = audit_focus(config, sample, lto_by_key)
    return {
        "dumps": {
            "present_sources": dumps["present_sources"],
            "missing_sources": dumps["missing_sources"],
            "dumps_present": dumps["dumps_present"],
            "n_lto_rows": len(dumps["rows"]),
        },
        "denominator": {
            "n_raw": sample["n_raw"],
            "n_ineligible": sample["n_ineligible"],
            "n_contracted": len(sample["contracted"]),
            "n_official_pairs": len(sample["official_events"]),
            "frozen_denominator_holds": frozen_ok,
        },
        "official_shape": official_shape,
        "lto_shape": lto_shape,
        "comparison": comparison,
        "focus": focus,
        "present_sources": sample["present_sources"],
        "missing_sources": dumps["missing_sources"],
        "n_contracted": len(sample["contracted"]),
        "n_official_pairs": len(sample["official_events"]),
        "n_raw": sample["n_raw"],
        "n_ineligible": sample["n_ineligible"],
        "empirical_cross_covariance_invented": False,
        "covariance_rescaled": False,
        "provenance_hashes": sample["provenance_hashes"],
    }


def decide(inventory: Mapping[str, Any], inherited: Mapping[str, Any]) -> dict[str, Any]:
    mechanism = decide_case(inventory)
    if mechanism["measurement_model_v2_entered"]:
        refuse_measurement_model_v2()
    if mechanism["b15_authorized"] and not mechanism["lto_cin_contract_established"]:
        refuse_b15()
    return {
        **mechanism,
        "geometry_write_allowed": False,
        "official_input_scope": "contract_eligible",
        "raw_ckf_used": False,
        "inherited_wb109_decision_sha256": inherited["workbook_109"]["decision_sha256"],
        "inherited_wb108_decision_sha256": inherited["workbook_108"]["decision_sha256"],
        "inherited_wb107_decision_sha256": inherited["workbook_107"]["decision_sha256"],
        "inherited_wb105_decision_sha256": inherited["workbook_105"]["decision_sha256"],
        "inherited_wb103_contract_sha256": inherited["workbook_103"]["contract_sha256"],
        "wb96_through_wb109_rewritten": False,
    }
