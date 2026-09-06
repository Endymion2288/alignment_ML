"""Workbook 87: FaserActs Propagated Covariance Validation V2.

Stage B after WB86.  Tests whether

    C_target = J_transport C_source_WB86 J_transport^T + process_noise

describes

    e_target = propagated fitted state - truth target state

on the frozen WB83/86 source-disjoint MC.  Not an alignment task.
WB81 mode 3 is not inherited as the answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
)
from alignment import full_curvilinear_uvt_repair_validation as fcv
from alignment import segmentfit_getstate_repair_validation as sgrv
from alignment.propagated_covariance_closure import (
    ClosureRecords,
    _load_truth_table,
    compute_closure_metrics,
    evaluate_closure_gates,
)
from alignment.source_tracklet_covariance_closure import (
    _TRACKLET_BRANCHES,
    _source_refit_dir,
)
from datasets.propagation_loader import load_propagation_records
from datasets.schema import covariance_from_columns

SCHEMA_VERSION = "faseracts-propagated-covariance-validation-v2"
WORKBOOK = 87
DEFAULT_CONFIG = "configs/faseracts_propagated_covariance_validation_v2.yaml"

DECISION_VALIDATED = "faseracts_propagated_covariance_validated"
DECISION_NOT_VALIDATED = "faseracts_transport_covariance_not_validated"

MECHANISM_MISSING_PROCESS_NOISE = "missing_process_noise"
MECHANISM_QOP_SEMANTICS = "q_over_p_uncertainty_semantics"
MECHANISM_MATERIAL_MISMATCH = "material_model_mismatch"
MECHANISM_JACOBIAN_ERROR = "transport_jacobian_error"

FROZEN_WB81_GATES = {
    "whitened_chi2_per_ndof_max": 4.0,
    "cov_z_eigenvalue_min": 0.25,
    "cov_z_eigenvalue_max": 4.0,
    "generalized_eigenvalue_min": 0.25,
    "generalized_eigenvalue_max": 4.0,
    "pencil_variance_ratio_min": 0.25,
    "pencil_variance_ratio_max": 4.0,
    "min_pairs_per_pair": 20,
}

MODE_PRODUCTION = 0
MODE_EXISTING_QOP = 1
MODE_CORRECTED_QOP = 2
MODE_PROCESS_NOISE = 3


class ConfigError(ValueError):
    """Raised when the WB87 config or its frozen inheritance is inconsistent."""


def _expect_false(config: Mapping[str, Any], key: str) -> None:
    if bool(config.get(key, False)) is not False:
        raise ConfigError(f"frozen flag must be false: {key}")


def _expect_true(config: Mapping[str, Any], key: str) -> None:
    if bool(config.get(key, False)) is not True:
        raise ConfigError(f"frozen prohibition must be true: {key}")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the WB87 config and verify the frozen WB81–WB86 inheritance."""
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ConfigError(f"WB87 config must be a mapping: {config_path}")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {SCHEMA_VERSION}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise ConfigError(f"workbook must be {WORKBOOK}")

    for key in (
        "geometry_write_allowed",
        "official_conditions_write_allowed",
        "real_data_candidate_alignment_authorized",
        "real_data_alignment_authorized",
        "measurement_model_validated",
        "held_out_accessed",
        "external_constraint_ingest_authorized",
    ):
        _expect_false(config, key)
    for key in (
        "do_not_open_held_out",
        "do_not_read_real_data_residuals",
        "do_not_modify_faseracts_extrapolation_tool",
        "do_not_tune_covariance_to_chi2",
        "do_not_promote_mode3_to_production",
        "do_not_use_truth_qoverp_for_real_data_solution",
        "do_not_delete_qoverp_column_as_final_scheme",
        "do_not_add_scale_factors",
        "do_not_enter_alignment",
        "do_not_enter_measurement_model_v2",
        "do_not_write_geometry_or_conditions_payload",
    ):
        _expect_true(config, key)

    gates = config.get("closure_gates", {})
    for key, expected in FROZEN_WB81_GATES.items():
        if gates.get(key) != expected:
            raise ConfigError(f"WB81 Stage-2 gate must stay frozen: {key}")

    inheritance = config.get("inheritance", {})

    def _parent(*args, **kwargs):
        try:
            sgrv._verify_parent_inheritance(*args, **kwargs)
        except sgrv.ConfigError as exc:
            raise ConfigError(str(exc)) from exc

    _parent(
        inheritance,
        "workbook_81",
        decision_filename="propagated_covariance_decision.json",
        require_mechanism=True,
    )
    _parent(
        inheritance,
        "workbook_82",
        decision_filename="wide_ty_mc_support_decision.json",
        require_mechanism=False,
    )
    _parent(
        inheritance,
        "workbook_83",
        decision_filename="propagated_covariance_upstream_repair_decision.json",
        require_mechanism=True,
    )
    _parent(
        inheritance,
        "workbook_84",
        decision_filename="segmentfit_covariance_coordinate_contract_decision.json",
        require_mechanism=True,
    )
    _parent(
        inheritance,
        "workbook_85",
        decision_filename="covariance_repair_decision.json",
        require_mechanism=False,
    )
    _parent(
        inheritance,
        "workbook_86",
        decision_filename="covariance_repair_decision.json",
        require_mechanism=False,
    )
    wb86_root = resolve_under_root(
        project_root(), str(inheritance["workbook_86_output_root"])
    )
    wb86 = json.loads(
        (wb86_root / "covariance_repair_decision.json").read_text(encoding="utf-8")
    )
    if wb86.get("decision") != str(inheritance["workbook_86_frozen_decision"]):
        raise ConfigError("WB86 frozen decision mismatch")
    if bool(wb86.get("propagated_covariance_validation_authorized", False)) is not True:
        raise ConfigError("WB86 must authorize propagated-covariance validation")

    mc = config["mc_data"]
    if set(mc["construction_source_ids"]) & set(mc["validation_source_ids"]):
        raise ConfigError("MC construction/validation sources are not disjoint")
    reserved = str(mc.get("reserved_conditional_j_support_source_id") or "")
    if reserved and reserved in set(mc["construction_source_ids"]) | set(
        mc["validation_source_ids"]
    ):
        raise ConfigError("reserved conditional-J source must stay out of this split")
    return dict(config)


# ---------------------------------------------------------------------------
# Transport / process-noise primitives
# ---------------------------------------------------------------------------


def geometric_transport_jacobian(lever_arm_mm: float) -> np.ndarray:
    """Field-free 4D Jacobian ``(x,y,tx,ty)_src -> (x,y,tx,ty)_tgt``."""
    j = np.eye(4)
    j[0, 2] = float(lever_arm_mm)
    j[1, 3] = float(lever_arm_mm)
    return j


def highland_theta0(p_mev: float, x_over_x0: float) -> float:
    """Highland projected RMS plane angle (PDG).  Not chi2-tuned."""
    if p_mev <= 0.0 or x_over_x0 <= 0.0:
        return 0.0
    return (13.6 / float(p_mev)) * np.sqrt(x_over_x0) * (
        1.0 + 0.038 * np.log(x_over_x0)
    )


def process_noise_covariance(
    lever_arm_mm: float, p_mev: float, x_over_x0: float
) -> np.ndarray:
    """Uniform-path Highland process noise in ``[x,y,tx,ty]``.

    ``var(angle) = θ0²``, ``var(pos) = θ0² L² / 3``, ``cov(pos,angle) = θ0² L / 2``.
    Applied identically to the ``x-tx`` and ``y-ty`` blocks.
    """
    theta0 = highland_theta0(p_mev, x_over_x0)
    q = np.zeros((4, 4))
    var_a = theta0 * theta0
    L = float(lever_arm_mm)
    var_p = var_a * L * L / 3.0
    cov_pa = var_a * L / 2.0
    for pos, slp in ((0, 2), (1, 3)):
        q[pos, pos] = var_p
        q[slp, slp] = var_a
        q[pos, slp] = q[slp, pos] = cov_pa
    return q


def _spd_clip(c: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(0.5 * (c + c.T))
    w = np.maximum(w, 1e-18)
    return (v * w) @ v.T


# ---------------------------------------------------------------------------
# Per-pair joined records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageBRecords:
    """Joined source + propagation records for one station pair / split."""

    station_pair: tuple[int, int]
    e_target: np.ndarray
    e_transported_source: np.ndarray
    c_production: np.ndarray
    c_existing_qop: np.ndarray
    c_corrected_qop: np.ndarray
    c_process_noise: np.ndarray
    lever_arm_mm: np.ndarray
    pred_tx: np.ndarray
    pred_ty: np.ndarray
    abs_q_over_p: np.ndarray
    n_matched: int

    @property
    def size(self) -> int:
        return int(self.e_target.shape[0])

    def as_closure(self, covariance: np.ndarray, *, transported_source: bool = False) -> ClosureRecords:
        return ClosureRecords(
            station_pair=self.station_pair,
            q_over_p_mode=-1,
            e_prop=self.e_transported_source if transported_source else self.e_target,
            c_prop=covariance,
            lever_arm_mm=self.lever_arm_mm,
            pred_tx=self.pred_tx,
            pred_ty=self.pred_ty,
            n_matched=self.n_matched,
            has_covariance=True,
        )


def _source_tracklet_map(tracklets_path: Path, tree: str, source_station: int) -> dict:
    import uproot

    with uproot.open(tracklets_path) as handle:
        arrays = handle[tree].arrays(list(_TRACKLET_BRANCHES), library="np")
    covariance = covariance_from_columns(arrays)
    out: dict[tuple[int, int, int], dict[str, Any]] = {}
    for i in range(len(arrays["run_id"])):
        if int(arrays["station_id"][i]) != int(source_station):
            continue
        key = (
            int(arrays["run_id"][i]),
            int(arrays["event_id"][i]),
            int(arrays["truth_particle_id"][i]),
        )
        out[key] = {
            "x": float(arrays["x_mm"][i]),
            "y": float(arrays["y_mm"][i]),
            "z": float(arrays["z_mm"][i]),
            "tx": float(arrays["tx"][i]),
            "ty": float(arrays["ty"][i]),
            "q_over_p": float(arrays["q_over_p_per_mev"][i]),
            "truth_match_fraction": float(arrays["truth_match_fraction"][i]),
            "c_exported": np.asarray(covariance[i], dtype=float),
        }
    return out


def _target_residual(
    pred: np.ndarray,
    truth_pos: np.ndarray,
    truth_mom: np.ndarray,
    target_z: float,
    *,
    straight_line: bool,
) -> np.ndarray | None:
    if abs(truth_mom[2]) < 1e-12:
        return None
    tx_t = truth_mom[0] / truth_mom[2]
    ty_t = truth_mom[1] / truth_mom[2]
    dz = (target_z - truth_pos[2]) if straight_line else 0.0
    e = np.array(
        [
            pred[0] - (truth_pos[0] + tx_t * dz),
            pred[1] - (truth_pos[1] + ty_t * dz),
            pred[2] - tx_t,
            pred[3] - ty_t,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(e)):
        return None
    return e


def _source_residual(
    src: Mapping[str, Any],
    truth_pos: np.ndarray,
    truth_mom: np.ndarray,
    *,
    straight_line: bool,
) -> np.ndarray | None:
    if abs(truth_mom[2]) < 1e-12:
        return None
    tx_t = truth_mom[0] / truth_mom[2]
    ty_t = truth_mom[1] / truth_mom[2]
    dz = (float(src["z"]) - truth_pos[2]) if straight_line else 0.0
    e = np.array(
        [
            src["x"] - (truth_pos[0] + tx_t * dz),
            src["y"] - (truth_pos[1] + ty_t * dz),
            src["tx"] - tx_t,
            src["ty"] - ty_t,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(e)):
        return None
    return e


def build_stage_b_records(
    config: Mapping[str, Any],
    source_id: str,
    station_pair: tuple[int, int],
) -> StageBRecords:
    """Join production propagations to WB86-repaired source covariances."""
    mc = config["mc_data"]
    tr = config["truth_reference"]
    refit = _source_refit_dir(config, source_id)
    props = load_propagation_records(refit / mc["propagations_file"])
    truth = _load_truth_table(refit / mc["enhanced_file"], mc["enhanced_tree"], config)
    src_map = _source_tracklet_map(
        refit / mc["tracklets_file"], mc["tracklets_tree"], station_pair[0]
    )
    mode0 = int(mc["production_q_over_p_mode"])
    mode3 = int(mc["ntuple_mode_no_qoverp_cov"])
    x0 = float(config["part_b"]["material_x_over_x0_per_gap"])
    acc = float(tr["physical_acceptance_abs_tx_ty_max"])
    min_tmf = float(tr["min_truth_match_fraction"])
    straight = bool(tr["straight_line_z_correction"])

    c_mode3: dict[tuple[int, int, int], np.ndarray] = {}
    sel3 = (
        (props.q_over_p_mode == mode3)
        & (props.source_station_id == station_pair[0])
        & (props.target_station_id == station_pair[1])
        & props.success
        & props.has_covariance
    )
    for i in np.where(sel3)[0]:
        key = (
            int(props.run_id[i]),
            int(props.event_id[i]),
            int(props.truth_particle_id[i]),
        )
        c_mode3[key] = np.asarray(props.covariance[i], dtype=float)

    sel0 = (
        (props.q_over_p_mode == mode0)
        & (props.source_station_id == station_pair[0])
        & (props.target_station_id == station_pair[1])
        & props.success
        & props.has_covariance
    )
    e_t: list[np.ndarray] = []
    e_js: list[np.ndarray] = []
    c0: list[np.ndarray] = []
    c1: list[np.ndarray] = []
    c2: list[np.ndarray] = []
    c3: list[np.ndarray] = []
    lever: list[float] = []
    ptx: list[float] = []
    pty: list[float] = []
    aqp: list[float] = []
    n_matched = 0
    for i in np.where(sel0)[0]:
        key = (
            int(props.run_id[i]),
            int(props.event_id[i]),
            int(props.truth_particle_id[i]),
        )
        src = src_map.get(key)
        if src is None or src["truth_match_fraction"] < min_tmf:
            continue
        if abs(src["tx"]) > acc or abs(src["ty"]) > acc:
            continue
        per = (truth.get((key[0], key[1])) or {}).get(key[2])
        if per is None or station_pair[0] not in per or station_pair[1] not in per:
            continue
        tgt = per[station_pair[1]]
        struth = per[station_pair[0]]
        if not (
            np.all(np.isfinite(tgt["pos"]))
            and np.all(np.isfinite(tgt["mom"]))
            and np.all(np.isfinite(struth["pos"]))
        ):
            continue
        target_z = float(props.target_z_mm[i])
        e_target = _target_residual(
            props.prediction[i], tgt["pos"], tgt["mom"], target_z, straight_line=straight
        )
        e_source = _source_residual(
            src, struth["pos"], struth["mom"], straight_line=straight
        )
        if e_target is None or e_source is None:
            continue
        c_rep = fcv.repair_exported_covariance_full(
            src["c_exported"], src["tx"], src["ty"]
        )
        if c_rep is None or not np.all(np.isfinite(c_rep)):
            continue
        L = target_z - float(src["z"])
        j = geometric_transport_jacobian(L)
        c_transport = _spd_clip(j @ c_rep @ j.T)
        c_prod = np.asarray(props.covariance[i], dtype=float)
        if not np.all(np.isfinite(c_prod)):
            continue
        c_no_qop = c_mode3.get(key)
        if c_no_qop is None or not np.all(np.isfinite(c_no_qop)):
            delta_qop = np.zeros((4, 4))
        else:
            delta_qop = _spd_clip(c_prod - c_no_qop)
        p_mev = (
            1.0 / abs(src["q_over_p"]) if abs(src["q_over_p"]) > 1e-12 else np.inf
        )
        q_ms = process_noise_covariance(abs(L), p_mev, x0)
        n_matched += 1
        e_t.append(e_target)
        e_js.append(j @ e_source)
        c0.append(c_prod)
        c1.append(_spd_clip(c_transport + delta_qop))
        c2.append(c_transport)
        c3.append(_spd_clip(c_transport + q_ms))
        lever.append(abs(target_z - float(struth["pos"][2])))
        ptx.append(float(props.prediction[i, 2]))
        pty.append(float(props.prediction[i, 3]))
        aqp.append(abs(src["q_over_p"]))
    if not e_t:
        empty = np.empty((0, 4))
        empty_c = np.empty((0, 4, 4))
        return StageBRecords(
            station_pair=station_pair,
            e_target=empty,
            e_transported_source=empty,
            c_production=empty_c,
            c_existing_qop=empty_c,
            c_corrected_qop=empty_c,
            c_process_noise=empty_c,
            lever_arm_mm=np.empty(0),
            pred_tx=np.empty(0),
            pred_ty=np.empty(0),
            abs_q_over_p=np.empty(0),
            n_matched=0,
        )
    return StageBRecords(
        station_pair=station_pair,
        e_target=np.asarray(e_t),
        e_transported_source=np.asarray(e_js),
        c_production=np.asarray(c0),
        c_existing_qop=np.asarray(c1),
        c_corrected_qop=np.asarray(c2),
        c_process_noise=np.asarray(c3),
        lever_arm_mm=np.asarray(lever),
        pred_tx=np.asarray(ptx),
        pred_ty=np.asarray(pty),
        abs_q_over_p=np.asarray(aqp),
        n_matched=n_matched,
    )


def _stack_stage_b(parts: list[StageBRecords], pair: tuple[int, int]) -> StageBRecords:
    parts = [p for p in parts if p.size > 0]
    if not parts:
        empty = np.empty((0, 4))
        empty_c = np.empty((0, 4, 4))
        return StageBRecords(
            station_pair=pair,
            e_target=empty,
            e_transported_source=empty,
            c_production=empty_c,
            c_existing_qop=empty_c,
            c_corrected_qop=empty_c,
            c_process_noise=empty_c,
            lever_arm_mm=np.empty(0),
            pred_tx=np.empty(0),
            pred_ty=np.empty(0),
            abs_q_over_p=np.empty(0),
            n_matched=0,
        )
    return StageBRecords(
        station_pair=pair,
        e_target=np.concatenate([p.e_target for p in parts]),
        e_transported_source=np.concatenate([p.e_transported_source for p in parts]),
        c_production=np.concatenate([p.c_production for p in parts]),
        c_existing_qop=np.concatenate([p.c_existing_qop for p in parts]),
        c_corrected_qop=np.concatenate([p.c_corrected_qop for p in parts]),
        c_process_noise=np.concatenate([p.c_process_noise for p in parts]),
        lever_arm_mm=np.concatenate([p.lever_arm_mm for p in parts]),
        pred_tx=np.concatenate([p.pred_tx for p in parts]),
        pred_ty=np.concatenate([p.pred_ty for p in parts]),
        abs_q_over_p=np.concatenate([p.abs_q_over_p for p in parts]),
        n_matched=int(sum(p.n_matched for p in parts)),
    )


def _chi2_per_ndof_samples(e: np.ndarray, c: np.ndarray) -> np.ndarray:
    out: list[float] = []
    for residual, cov in zip(e, c):
        try:
            out.append(float(residual @ np.linalg.solve(cov, residual)) / 4.0)
        except np.linalg.LinAlgError:
            continue
    return np.asarray(out, dtype=float)


def _augment_metrics(
    metrics: dict[str, Any], e: np.ndarray, c: np.ndarray
) -> dict[str, Any]:
    samples = _chi2_per_ndof_samples(e, c)
    metrics = dict(metrics)
    metrics["chi2_per_ndof_median"] = (
        float(np.median(samples)) if samples.size else float("nan")
    )
    metrics["fraction_chi2_per_ndof_above_4"] = (
        float(np.mean(samples > 4.0)) if samples.size else float("nan")
    )
    return metrics


def _evaluate_arm(
    recs: StageBRecords,
    config: Mapping[str, Any],
    covariance: np.ndarray,
    *,
    transported_source: bool = False,
) -> dict[str, Any]:
    closure = recs.as_closure(covariance, transported_source=transported_source)
    metrics = compute_closure_metrics(closure, config)
    e = recs.e_transported_source if transported_source else recs.e_target
    metrics = _augment_metrics(metrics, e, covariance)
    metrics["gate_verdict"] = evaluate_closure_gates(metrics, config)
    return metrics


def _mode_covariances(recs: StageBRecords) -> dict[int, np.ndarray]:
    return {
        MODE_PRODUCTION: recs.c_production,
        MODE_EXISTING_QOP: recs.c_existing_qop,
        MODE_CORRECTED_QOP: recs.c_corrected_qop,
        MODE_PROCESS_NOISE: recs.c_process_noise,
    }


def run_split(config: Mapping[str, Any], split: str) -> dict[str, Any]:
    """Run all four modes plus Jacobian self-consistency for one split."""
    mc = config["mc_data"]
    pairs = [tuple(p) for p in mc["station_pairs"]]
    per_pair: dict[str, Any] = {}
    for pair in pairs:
        parts = [
            build_stage_b_records(config, sid, pair) for sid in mc[f"{split}_source_ids"]
        ]
        stacked = _stack_stage_b(parts, pair)
        modes = {}
        for mode, cov in _mode_covariances(stacked).items():
            modes[str(mode)] = _evaluate_arm(stacked, config, cov)
        jacobian = _evaluate_arm(
            stacked, config, stacked.c_corrected_qop, transported_source=True
        )
        leftover = stacked.e_target - stacked.e_transported_source
        leftover_rms = leftover.std(axis=0).tolist() if stacked.size else [None] * 4
        per_pair[f"({pair[0]},{pair[1]})"] = {
            "n_pairs": stacked.size,
            "n_matched": stacked.n_matched,
            "modes": modes,
            "jacobian_self_consistency": jacobian,
            "leftover_e_target_minus_J_e_source_rms": leftover_rms,
        }
    return {
        "split": split,
        "source_ids": list(mc[f"{split}_source_ids"]),
        "per_pair": per_pair,
    }


def _all_calibrated(split_out: Mapping[str, Any], mode: int) -> bool:
    return all(
        cell["modes"][str(mode)]["gate_verdict"]["calibrated"]
        for cell in split_out["per_pair"].values()
    )


def _all_jacobian_calibrated(split_out: Mapping[str, Any]) -> bool:
    return all(
        cell["jacobian_self_consistency"]["gate_verdict"]["calibrated"]
        for cell in split_out["per_pair"].values()
    )


def _mode_stat(split_out: Mapping[str, Any], mode: int, key: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for label, cell in split_out["per_pair"].items():
        metrics = cell["modes"][str(mode)]
        if key == "pencil":
            pencil = metrics.get("pencil") or {}
            out[label] = float(pencil.get("variance_ratio_prop_over_emp", np.nan))
        elif key == "gen_max":
            gen = metrics.get("generalized_eigenvalues_cemp_over_cprop") or []
            out[label] = float(np.max(gen)) if gen else float("nan")
        else:
            out[label] = float(metrics.get(key, np.nan))
    return out


def classify_mechanism(
    construction: Mapping[str, Any], validation: Mapping[str, Any]
) -> dict[str, Any]:
    """Classify why the primary 4D transport model fails, if it fails."""
    jac_ok = _all_jacobian_calibrated(construction) and _all_jacobian_calibrated(
        validation
    )
    mode0_pencil = {
        **_mode_stat(construction, MODE_PRODUCTION, "pencil"),
        **_mode_stat(validation, MODE_PRODUCTION, "pencil"),
    }
    mode2_gen = {
        **_mode_stat(construction, MODE_CORRECTED_QOP, "gen_max"),
        **_mode_stat(validation, MODE_CORRECTED_QOP, "gen_max"),
    }
    mode2_chi2 = {
        **_mode_stat(construction, MODE_CORRECTED_QOP, "chi2_per_ndof"),
        **_mode_stat(validation, MODE_CORRECTED_QOP, "chi2_per_ndof"),
    }
    mode3_chi2 = {
        **_mode_stat(construction, MODE_PROCESS_NOISE, "chi2_per_ndof"),
        **_mode_stat(validation, MODE_PROCESS_NOISE, "chi2_per_ndof"),
    }
    pencil_over = any(np.isfinite(v) and v > 4.0 for v in mode0_pencil.values())
    under = any(np.isfinite(v) and v > 4.0 for v in mode2_gen.values()) or any(
        np.isfinite(v) and v > 4.0 for v in mode2_chi2.values()
    )
    highland_helps = False
    if mode2_chi2 and mode3_chi2:
        ratios = []
        for key, a in mode2_chi2.items():
            b = mode3_chi2.get(key, np.nan)
            if np.isfinite(a) and np.isfinite(b) and b > 0:
                ratios.append(a / b)
        highland_helps = bool(ratios) and float(np.median(ratios)) > 1.2

    if not jac_ok:
        category = MECHANISM_JACOBIAN_ERROR
        detail = (
            "Geometric J does not transport the WB86 source residual onto itself "
            "(J e_source vs J C J^T fails the frozen WB81 gates)."
        )
    elif pencil_over and under:
        category = MECHANISM_QOP_SEMANTICS
        detail = (
            "Dummy SegmentFit q/p covariance is not a physical momentum "
            "uncertainty (production pencil still over-estimates), but dropping "
            "that column is not the final scheme: the 4D transport of the "
            "validated source still under-estimates e_target because a physical "
            "q/p is required for magnetic transport and is not supplied."
        )
    elif under and highland_helps and not _all_calibrated(construction, MODE_PROCESS_NOISE):
        category = MECHANISM_MATERIAL_MISMATCH
        detail = (
            "Highland process noise moves chi2 in the right direction but does "
            "not restore closure; the material / field budget is not the "
            "production ACTS model."
        )
    elif under:
        category = MECHANISM_MISSING_PROCESS_NOISE
        detail = (
            "Validated source + deterministic J C J^T under-estimates e_target; "
            "production ACTS process noise is disabled and the pre-registered "
            "Highland term is not sufficient."
        )
    else:
        category = MECHANISM_MISSING_PROCESS_NOISE
        detail = "Primary model failed the gates; residual mechanism not isolated."
    return {
        "category": category,
        "detail": detail,
        "jacobian_self_consistency_calibrated": jac_ok,
        "production_pencil_overestimate": pencil_over,
        "mode2_underestimate": under,
        "highland_helps_but_insufficient": highland_helps,
        "production_pencil_ratio": mode0_pencil,
        "mode2_chi2_per_ndof": mode2_chi2,
        "mode3_chi2_per_ndof": mode3_chi2,
        "qoverp_dummy_is_physical": False,
        "deleting_qoverp_column_is_final_scheme": False,
    }


def qoverp_audit_report(
    construction: Mapping[str, Any], validation: Mapping[str, Any]
) -> dict[str, Any]:
    """With vs without dummy q/p; dummy is not promoted to a final scheme."""
    return {
        "kind": "qoverp_audit",
        "with_dummy_qoverp": {
            "mode": MODE_PRODUCTION,
            "construction_pencil": _mode_stat(construction, MODE_PRODUCTION, "pencil"),
            "validation_pencil": _mode_stat(validation, MODE_PRODUCTION, "pencil"),
            "construction_chi2": _mode_stat(construction, MODE_PRODUCTION, "chi2_per_ndof"),
            "validation_chi2": _mode_stat(validation, MODE_PRODUCTION, "chi2_per_ndof"),
        },
        "without_dummy_qoverp": {
            "mode": MODE_CORRECTED_QOP,
            "construction_pencil": _mode_stat(construction, MODE_CORRECTED_QOP, "pencil"),
            "validation_pencil": _mode_stat(validation, MODE_CORRECTED_QOP, "pencil"),
            "construction_chi2": _mode_stat(construction, MODE_CORRECTED_QOP, "chi2_per_ndof"),
            "validation_chi2": _mode_stat(validation, MODE_CORRECTED_QOP, "chi2_per_ndof"),
        },
        "existing_qoverp_plus_validated_source": {
            "mode": MODE_EXISTING_QOP,
            "construction_chi2": _mode_stat(construction, MODE_EXISTING_QOP, "chi2_per_ndof"),
            "validation_chi2": _mode_stat(validation, MODE_EXISTING_QOP, "chi2_per_ndof"),
        },
        "conclusion": {
            "dummy_qoverp_covariance_is_physical": False,
            "dummy_qoverp_is_unconstrained_segmentfit_input": True,
            "physical_qoverp_required_for_magnetic_transport": True,
            "deleting_qoverp_column_is_final_scheme": False,
        },
        "diagnostic_only": True,
    }


def run_campaign(config: Mapping[str, Any]) -> dict[str, Any]:
    construction = run_split(config, "construction")
    validation = run_split(config, "validation")
    jac_ok = _all_jacobian_calibrated(construction) and _all_jacobian_calibrated(
        validation
    )
    mode2_ok = _all_calibrated(construction, MODE_CORRECTED_QOP) and _all_calibrated(
        validation, MODE_CORRECTED_QOP
    )
    mode3_ok = _all_calibrated(construction, MODE_PROCESS_NOISE) and _all_calibrated(
        validation, MODE_PROCESS_NOISE
    )
    recovered = bool(mode2_ok or (jac_ok and mode3_ok))
    agreed = bool(
        _all_calibrated(construction, MODE_CORRECTED_QOP)
        == _all_calibrated(validation, MODE_CORRECTED_QOP)
        and _all_calibrated(construction, MODE_PROCESS_NOISE)
        == _all_calibrated(validation, MODE_PROCESS_NOISE)
    )
    return {
        "part_a_deterministic_transport": {
            "kind": "part_a_deterministic_transport",
            "jacobian": "geometric_lever_arm",
            "process_noise": False,
            "construction": construction,
            "validation": validation,
        },
        "part_b_process_noise": {
            "kind": "part_b_process_noise",
            "model": config["part_b"]["model"],
            "material_x_over_x0_per_gap": config["part_b"]["material_x_over_x0_per_gap"],
            "momentum_source": config["part_b"]["momentum_source"],
            "chi2_tuned": False,
            "construction_mode3_calibrated": _all_calibrated(
                construction, MODE_PROCESS_NOISE
            ),
            "validation_mode3_calibrated": _all_calibrated(
                validation, MODE_PROCESS_NOISE
            ),
        },
        "mode_comparison": {
            "kind": "mode_comparison",
            "arms": ["production", "existing_qop", "corrected_qop", "process_noise"],
            "construction_all_calibrated": {
                "0": _all_calibrated(construction, 0),
                "1": _all_calibrated(construction, 1),
                "2": _all_calibrated(construction, 2),
                "3": _all_calibrated(construction, 3),
            },
            "validation_all_calibrated": {
                "0": _all_calibrated(validation, 0),
                "1": _all_calibrated(validation, 1),
                "2": _all_calibrated(validation, 2),
                "3": _all_calibrated(validation, 3),
            },
            "jacobian_self_consistency_calibrated": {
                "construction": _all_jacobian_calibrated(construction),
                "validation": _all_jacobian_calibrated(validation),
            },
        },
        "qoverp_audit": qoverp_audit_report(construction, validation),
        "validation": {
            "construction_primary_calibrated": _all_calibrated(
                construction, MODE_CORRECTED_QOP
            ),
            "validation_primary_calibrated": _all_calibrated(
                validation, MODE_CORRECTED_QOP
            ),
            "jacobian_self_consistency_calibrated": jac_ok,
            "process_noise_calibrated": mode3_ok,
            "source_disjoint_agreement": agreed,
            "whitening_closure_recovered": recovered,
        },
    }


def decide(
    config: Mapping[str, Any],
    campaign: Mapping[str, Any],
) -> dict[str, Any]:
    validation = campaign["validation"]
    recovered = bool(validation.get("whitening_closure_recovered", False))
    agreed = bool(validation.get("source_disjoint_agreement", False))
    decision = DECISION_VALIDATED if recovered and agreed else DECISION_NOT_VALIDATED
    construction = campaign["part_a_deterministic_transport"]["construction"]
    validation_split = campaign["part_a_deterministic_transport"]["validation"]
    mechanism = None
    if decision == DECISION_NOT_VALIDATED:
        mechanism = classify_mechanism(construction, validation_split)
    return {
        "kind": "propagated_covariance_decision",
        "decision": decision,
        "whitening_closure_recovered": recovered,
        "source_disjoint_agreement": agreed,
        "construction_primary_calibrated": bool(
            validation["construction_primary_calibrated"]
        ),
        "validation_primary_calibrated": bool(
            validation["validation_primary_calibrated"]
        ),
        "jacobian_self_consistency_calibrated": bool(
            validation["jacobian_self_consistency_calibrated"]
        ),
        "process_noise_calibrated": bool(validation["process_noise_calibrated"]),
        "failure_classification": mechanism,
        "measurement_model_validated": False,
        "measurement_model_v2_discussion_allowed": bool(decision == DECISION_VALIDATED),
        "measurement_model_v2_entered": False,
        "geometry_write_allowed": False,
        "real_data_alignment_authorized": False,
        "held_out_accessed": False,
        "stage_b_entered": True,
        "frozen_v2_alignment_authorized": False,
        "qoverp_column_deleted_as_final_scheme": False,
        "truth_qoverp_used_as_real_data_solution": False,
        "covariance_tuned_to_chi2": False,
    }
