"""Yasu-S2E: split curvature-information limit from covariance failure.

Reuses WB119 dumps.  Does not flip S2, does not open S3, does not drop
sign-flips, tails, or non-SPD tracks.  Clean-subsample numbers are
conditional diagnostics only.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import yaml

from alignment.numerical_contract import is_spd
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from datasets.access_policy import AccessScope, authorize_path
from datasets.three_st_qp_calibration import (
    P_TRUTH_EDGES_MEV,
    P_TRUTH_LABELS,
    QOVERP_INDEX,
    SLOPE_EDGES,
    SOURCE_COLLECTION_NAME,
    TX_LABELS,
    TY_LABELS,
    access_split,
    assign_bin,
    dump_path_for_source,
    evaluate_events,
    iter_jsonl,
    source_list_key,
)
from datasets.three_st_qp_failure_diagnosis import (
    CHI2_EDGES,
    CHI2_LABELS,
    MISSING_LABELS,
    N_MOT_EDGES,
    N_MOT_LABELS,
    SIGNIFICANCE_EDGES,
    SIGNIFICANCE_LABELS,
    verify_pinned_calypso_sources,
)

SCHEMA_VERSION = "three-st-qp-root-cause-audit-v1"
DEFAULT_CONFIG = "configs/three_st_qp_root_cause_audit_v1.yaml"
TASK = "YASU-S2E"
WORKBOOK = 121

DECISION_CONTRACT = "three_st_qp_root_cause_contract_established"
DECISION_RECORDED = "three_st_qp_root_cause_recorded"
DECISION_NOT = "three_st_qp_root_cause_not_established"

STATUS_SUPPORTED = "supported"
STATUS_REJECTED = "rejected"
STATUS_UNRESOLVED = "unresolved"

MECHANISM_A = "intrinsic_curvature_information_limit"
MECHANISM_B = "topology_amplified_information_loss"
MECHANISM_C = "covariance_basis_unit_provenance_bug"
MECHANISM_D = "covariance_model_calibration_failure"

LOG_COND_EDGES = (0.0, 10.0, 20.0, 30.0, 40.0, 1.0e6)
LOG_COND_LABELS = ("cond_lt_10", "cond_10_20", "cond_20_30", "cond_30_40", "cond_ge_40")
CORR_EDGES = (0.0, 0.5, 0.9, 0.99, 1.01)
CORR_LABELS = ("corr_lt_0p5", "corr_0p5_0p9", "corr_0p9_0p99", "corr_sat")
MIN_EIG_LABELS = ("min_eig_le_0", "min_eig_lt_1e-16", "min_eig_1e-16_1e-8", "min_eig_ge_1e-8")
LOW_SIG_LABELS = ("sig_lt_0p5", "sig_0p5_to_1")
HIGH_SIG_LABELS = ("sig_3_to_5", "sig_5_to_10", "sig_ge_10")
SCALE_FACTORS = {
    "sigma_times_1e3": 1.0e-3,
    "sigma_times_1e-3": 1.0e3,
    "sigma_times_sqrt_1e3": 1.0 / math.sqrt(1.0e3),
    "sigma_times_1_over_sqrt_1e3": math.sqrt(1.0e3),
}


class ThreeStQpRootCauseError(ValueError):
    """Raised when the S2E contract is illegal."""


def refuse_flip_s2() -> None:
    raise ThreeStQpRootCauseError("S2E cannot set three_st_qp_trusted_observable true")


def refuse_residual_conditional() -> None:
    raise ThreeStQpRootCauseError("E[r_IFT|q/p] is Stage 3; S2E does not open it")


def refuse_drop_nspd() -> None:
    raise ThreeStQpRootCauseError("non-SPD tracks stay in the sample; they are a contribution")


def conversion_chain() -> dict[str, Any]:
    return {
        "acts_qoverp_unit": "per_GeV",
        "trk_qoverp_unit": "per_MeV",
        "acts_1_MeV": 1.0e-3,
        "parameter_scale": "q_trk = q_acts * 1_MeV",
        "covariance_scale": (
            "C_i4 *= 1_MeV then C_4i *= 1_MeV; C_44 therefore *= 1e-6"
        ),
        "comment_in_source_is_wrong": "CreateTrkTrackTool says 'to GeV' but converts to MeV",
        "bound_5x5_stuffed_into_curvilinear": True,
        "time_row_dropped": True,
        "qoverp_is_basis_invariant": True,
        "ckf2_adds_fitted_params_as_front_hole": True,
        "ckf2_then_kalman_refit": True,
        "refit_uses_create_track_without_fitted_params": True,
        "refit_inflates_input_covariance_times_10": True,
        "refit_failure_keeps_ckf_track": True,
        "persisted_front_is_first_tsos_after_create_track_order": True,
    }


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ThreeStQpRootCauseError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise ThreeStQpRootCauseError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise ThreeStQpRootCauseError("workbook must be 121")
    if bool(config.get("three_st_qp_trusted_observable", True)):
        raise ThreeStQpRootCauseError("three_st_qp_trusted_observable must stay false")
    if bool(config.get("residual_conditional_authorized", True)):
        raise ThreeStQpRootCauseError("residual_conditional_authorized must stay false")
    if bool(config.get("do_not_submit_reconstruction_dump", False)) is not True:
        raise ThreeStQpRootCauseError("do_not_submit_reconstruction_dump must be true")
    if len(config.get("focus_identities") or []) != 6:
        raise ThreeStQpRootCauseError("focus_identities must stay the six WB119 smoke rows")
    return dict(config)


def _expect_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise ThreeStQpRootCauseError(f"{label} hash mismatch: {digest}")


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited: dict[str, Any] = {}
    frozen_root = Path(str(config.get("frozen_artifact_root") or project_root()))
    for workbook in ("workbook_117", "workbook_118", "workbook_119", "workbook_120"):
        spec = config["inheritance"][workbook]
        root = project_root() if spec.get("artifact_root") == "local" else frozen_root
        decision = json.loads(
            resolve_under_root(root, spec["decision_path"]).read_text(encoding="utf-8")
        )
        _expect_sha(
            resolve_under_root(project_root(), spec["config_path"]),
            spec["config_sha256"],
            f"{workbook} config",
        )
        _expect_sha(
            resolve_under_root(root, spec["decision_path"]),
            spec["decision_sha256"],
            f"{workbook} decision",
        )
        if decision.get("decision") != spec["frozen_decision"]:
            raise ThreeStQpRootCauseError(f"{workbook} decision must stay frozen")
        inherited[workbook] = {
            "decision": decision["decision"],
            "config_sha256": spec["config_sha256"],
            "decision_sha256": spec["decision_sha256"],
        }
    if inherited["workbook_119"]["decision"] != "three_st_qp_calibration_not_established":
        raise ThreeStQpRootCauseError("WB119 must remain not_established")
    if inherited["workbook_120"]["decision"] != "three_st_qp_failure_diagnosis_recorded":
        raise ThreeStQpRootCauseError("WB120 diagnosis must remain recorded")
    return inherited


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _empty_bin() -> dict[str, float | int]:
    return {
        "n": 0,
        "n_flip": 0,
        "n_cov68": 0,
        "n_cov95": 0,
        "sum_pull": 0.0,
        "sumsq_pull": 0.0,
        "sum_sigma": 0.0,
        "sum_sig_truth": 0.0,
        "sum_sig_fit": 0.0,
        "n_tail": 0,
        "sumsq_tail": 0.0,
        "n_sym_fail": 0,
        "sum_schur": 0.0,
        "n_schur": 0,
        "n_schur_le_0": 0,
        "n_extreme": 0,
        "sumsq_extreme": 0.0,
        "n_flip_extreme": 0,
    }


def _add(bin_acc: dict[str, Any], row: Mapping[str, Any]) -> None:
    bin_acc["n"] += 1
    if row["flip"]:
        bin_acc["n_flip"] += 1
    if row["cov68"]:
        bin_acc["n_cov68"] += 1
    if row["cov95"]:
        bin_acc["n_cov95"] += 1
    bin_acc["sum_pull"] += row["pull"]
    bin_acc["sumsq_pull"] += row["pull"] * row["pull"]
    bin_acc["sum_sigma"] += row["sigma"]
    bin_acc["sum_sig_truth"] += row["sig_truth"]
    bin_acc["sum_sig_fit"] += row["sig_fit"]
    if row["tail"]:
        bin_acc["n_tail"] += 1
        bin_acc["sumsq_tail"] += row["pull"] * row["pull"]
    if row["sym_fail"]:
        bin_acc["n_sym_fail"] += 1
    if row["schur"] is not None:
        bin_acc["sum_schur"] += row["schur"]
        bin_acc["n_schur"] += 1
        if row["schur"] <= 0.0:
            bin_acc["n_schur_le_0"] += 1
    if row["extreme"]:
        bin_acc["n_extreme"] += 1
        bin_acc["sumsq_extreme"] += row["pull"] * row["pull"]
        if row["flip"]:
            bin_acc["n_flip_extreme"] += 1


def _merge_bin(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(left)
    for key, value in right.items():
        out[key] = left.get(key, 0) + value
    return out


def _summarize(bin_acc: Mapping[str, Any]) -> dict[str, Any]:
    n = int(bin_acc["n"])
    if n <= 0:
        return {
            "n": 0,
            "flip_rate": None,
            "coverage_68": None,
            "coverage_95": None,
            "mean_pull": None,
            "rms_pull": None,
            "mean_sigma": None,
            "mean_sig_truth": None,
            "mean_sig_fit": None,
            "n_tail": 0,
            "tail_sumsq_fraction": None,
            "mean_schur": None,
            "n_schur_le_0": 0,
            "n_extreme": 0,
            "extreme_sumsq_fraction": None,
        }
    return {
        "n": n,
        "n_flip": int(bin_acc["n_flip"]),
        "flip_rate": float(bin_acc["n_flip"]) / n,
        "coverage_68": float(bin_acc["n_cov68"]) / n,
        "coverage_95": float(bin_acc["n_cov95"]) / n,
        "mean_pull": float(bin_acc["sum_pull"]) / n,
        "rms_pull": math.sqrt(float(bin_acc["sumsq_pull"]) / n),
        "mean_sigma": float(bin_acc["sum_sigma"]) / n,
        "mean_sig_truth": float(bin_acc["sum_sig_truth"]) / n,
        "mean_sig_fit": float(bin_acc["sum_sig_fit"]) / n,
        "n_tail": int(bin_acc["n_tail"]),
        "tail_sumsq_fraction": (
            float(bin_acc["sumsq_tail"]) / float(bin_acc["sumsq_pull"])
            if float(bin_acc["sumsq_pull"]) > 0.0
            else None
        ),
        "mean_schur": (
            float(bin_acc["sum_schur"]) / float(bin_acc["n_schur"])
            if int(bin_acc["n_schur"])
            else None
        ),
        "n_schur_le_0": int(bin_acc["n_schur_le_0"]),
        "n_extreme": int(bin_acc["n_extreme"]),
        "extreme_sumsq_fraction": (
            float(bin_acc["sumsq_extreme"]) / float(bin_acc["sumsq_pull"])
            if float(bin_acc["sumsq_pull"]) > 0.0
            else None
        ),
    }


def empty_accumulator() -> dict[str, Any]:
    return {
        "n_records": 0,
        "n_primary": 0,
        "n_flip": 0,
        "n_truth_as_solution": 0,
        "n_nspd": 0,
        "n_min_eig_le_0": 0,
        "n_clean": 0,
        "n_dirty": 0,
        "all": _empty_bin(),
        "clean": _empty_bin(),
        "dirty": _empty_bin(),
        "nspd": _empty_bin(),
        "spd": _empty_bin(),
        "tail": _empty_bin(),
        "extreme": _empty_bin(),
        "given_p": {label: _empty_bin() for label in P_TRUTH_LABELS},
        "clean_p": {label: _empty_bin() for label in P_TRUTH_LABELS},
        "dirty_p": {label: _empty_bin() for label in P_TRUTH_LABELS},
        "given_sig_truth": {label: _empty_bin() for label in SIGNIFICANCE_LABELS},
        "clean_sig_truth": {label: _empty_bin() for label in SIGNIFICANCE_LABELS},
        "given_sig_fit": {label: _empty_bin() for label in SIGNIFICANCE_LABELS},
        "clean_sig_fit": {label: _empty_bin() for label in SIGNIFICANCE_LABELS},
        "given_tx": {label: _empty_bin() for label in TX_LABELS},
        "given_ty": {label: _empty_bin() for label in TY_LABELS},
        "given_cond": {label: _empty_bin() for label in LOG_COND_LABELS},
        "given_corr": {label: _empty_bin() for label in CORR_LABELS},
        "given_n_mot": {label: _empty_bin() for label in N_MOT_LABELS},
        "given_missing": {label: _empty_bin() for label in MISSING_LABELS},
        "given_chi2ndof": {},
        "given_min_eig": {label: _empty_bin() for label in MIN_EIG_LABELS},
        "clean_n_mot": {label: _empty_bin() for label in N_MOT_LABELS},
        "clean_missing": {label: _empty_bin() for label in MISSING_LABELS},
        "scale": {name: {"n_cov68": 0, "sumsq": 0.0, "n": 0} for name in SCALE_FACTORS},
        "p_x_sig_clean": {},
        "focus_hits": [],
    }


def merge_accumulators(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    out = empty_accumulator()
    for key in (
        "n_records", "n_primary", "n_flip", "n_truth_as_solution",
        "n_nspd", "n_min_eig_le_0", "n_clean", "n_dirty",
    ):
        out[key] = left[key] + right[key]
    for key in ("all", "clean", "dirty", "nspd", "spd", "tail", "extreme"):
        out[key] = _merge_bin(left[key], right[key])
    for family in (
        "given_p", "clean_p", "dirty_p", "given_sig_truth", "clean_sig_truth",
        "given_sig_fit", "clean_sig_fit", "given_tx", "given_ty", "given_cond",
        "given_corr", "given_n_mot", "given_missing", "given_chi2ndof",
        "given_min_eig", "clean_n_mot", "clean_missing",
    ):
        keys = set(left[family]) | set(right[family])
        out[family] = {
            key: _merge_bin(left[family].get(key, _empty_bin()), right[family].get(key, _empty_bin()))
            for key in keys
        }
    for name in SCALE_FACTORS:
        out["scale"][name] = {
            "n_cov68": left["scale"][name]["n_cov68"] + right["scale"][name]["n_cov68"],
            "sumsq": left["scale"][name]["sumsq"] + right["scale"][name]["sumsq"],
            "n": left["scale"][name]["n"] + right["scale"][name]["n"],
        }
    keys = set(left["p_x_sig_clean"]) | set(right["p_x_sig_clean"])
    out["p_x_sig_clean"] = {
        key: _merge_bin(
            left["p_x_sig_clean"].get(key, _empty_bin()),
            right["p_x_sig_clean"].get(key, _empty_bin()),
        )
        for key in keys
    }
    out["focus_hits"] = list(left["focus_hits"]) + list(right["focus_hits"])
    return out


def _missing_complete(stations: Sequence[int]) -> bool:
    have = {int(s) for s in stations if int(s) in (1, 2, 3)}
    return have == {1, 2, 3}


def _missing_label(stations: Sequence[int]) -> str:
    have = {int(s) for s in stations if int(s) in (1, 2, 3)}
    missing = {1, 2, 3} - have
    if not missing:
        return "complete_s1s2s3"
    if len(missing) >= 2:
        return "missing_two_or_more"
    return f"missing_s{next(iter(missing))}"


def _min_eig_label(min_eig: float | None) -> str:
    if min_eig is None or min_eig <= 0.0:
        return "min_eig_le_0"
    if min_eig < 1.0e-16:
        return "min_eig_lt_1e-16"
    if min_eig < 1.0e-8:
        return "min_eig_1e-16_1e-8"
    return "min_eig_ge_1e-8"


def _cov_features(
    cov_payload: Any,
) -> tuple[bool, bool, float | None, float, float | None, float | None]:
    spd = False
    sym_fail = True
    log_cond = None
    abs_corr_max = 0.0
    schur = None
    min_eig = None
    if cov_payload is None:
        return spd, sym_fail, log_cond, abs_corr_max, schur, min_eig
    cov = np.asarray(cov_payload, dtype=np.float64)
    if cov.shape != (5, 5) or not np.isfinite(cov).all():
        return spd, True, log_cond, abs_corr_max, schur, min_eig
    sym_fail = not np.allclose(cov, cov.T, rtol=1.0e-7, atol=1.0e-12)
    spd = bool(is_spd(cov))
    diag = np.diag(cov)
    if diag[QOVERP_INDEX] > 0.0:
        for idx in range(4):
            if diag[idx] > 0.0:
                corr = abs(
                    float(cov[idx, QOVERP_INDEX])
                    / math.sqrt(float(diag[idx] * diag[QOVERP_INDEX]))
                )
                if corr > abs_corr_max:
                    abs_corr_max = corr
    try:
        eig = np.linalg.eigvalsh(0.5 * (cov + cov.T))
        if np.isfinite(eig).all():
            min_eig = float(np.min(eig))
            if min_eig > 0.0:
                log_cond = math.log(float(np.max(eig)) / min_eig)
    except np.linalg.LinAlgError:
        pass
    block = cov[:4, :4]
    try:
        if np.linalg.det(block) != 0.0:
            inv = np.linalg.inv(block)
            cross = cov[4, :4]
            schur = float(cov[4, 4] - cross @ inv @ cross)
    except np.linalg.LinAlgError:
        schur = None
    return spd, sym_fail, log_cond, abs_corr_max, schur, min_eig


def process_track(row: Mapping[str, Any], acc: dict[str, Any], config: Mapping[str, Any]) -> None:
    if str(row.get("collection")) != SOURCE_COLLECTION_NAME:
        return
    if str(row.get("kind", "track")) != "track":
        return
    acc["n_records"] += 1
    if bool(row.get("truth_used_as_fit_seed")) or bool(row.get("truth_used_as_solution")):
        acc["n_truth_as_solution"] += 1
    q_fit = _finite(row.get("q_over_p_fit_per_mev"))
    q_truth = _finite(row.get("q_over_p_truth_s1_per_mev"))
    sigma = _finite(row.get("sigma_q_over_p_per_mev"))
    if (
        q_fit is None
        or q_truth is None
        or sigma is None
        or sigma <= 0.0
        or not bool(row.get("truth_matched"))
        or not bool(row.get("truth_reference_available"))
    ):
        return
    gates = config["gates"]
    clean_spec = config["clean_subsample"]
    delta = q_fit - q_truth
    pull = delta / sigma
    flip = _sign(q_fit) == 0 or _sign(q_truth) == 0 or _sign(q_fit) != _sign(q_truth)
    sig_truth = abs(q_truth) / sigma
    sig_fit = abs(q_fit) / sigma
    p_truth = _finite(row.get("p_truth_s1_mev"))
    tx = _finite(row.get("tx_truth")) or _finite(row.get("tx"))
    ty = _finite(row.get("ty_truth")) or _finite(row.get("ty"))
    stations = [int(s) for s in (row.get("measurements_on_track_stations") or [])]
    n_mot = int(row.get("n_mot") or 0)
    n_out = int(row.get("n_outlier_hits") or 0)
    chi2 = _finite(row.get("chi2"))
    ndof = _finite(row.get("ndof"))
    chi2ndof = None if ndof is None or ndof <= 0.0 or chi2 is None else chi2 / ndof
    spd, sym_fail, log_cond, abs_corr_max, schur, min_eig = _cov_features(
        row.get("native_covariance")
    )
    complete = _missing_complete(stations)
    missing = _missing_label(stations)
    if ndof is None or ndof <= 0.0:
        chi_label = "ndof_zero"
    else:
        chi_label = assign_bin(float(chi2 or 0.0) / ndof, CHI2_EDGES, CHI2_LABELS)
    clean = (
        complete
        and n_mot == int(clean_spec["require_n_mot"])
        and ndof is not None
        and ndof > 0.0
        and n_out == int(clean_spec["require_n_outlier"])
        and chi2ndof is not None
        and chi2ndof < float(clean_spec["max_chi2_ndof"])
        and spd
        and log_cond is not None
        and min_eig is not None
        and min_eig > 0.0
    )
    payload = {
        "flip": flip,
        "pull": pull,
        "sigma": sigma,
        "sig_truth": sig_truth,
        "sig_fit": sig_fit,
        "cov68": abs(pull) < float(gates["coverage_z68"]),
        "cov95": abs(pull) < float(gates["coverage_z95"]),
        "tail": abs(pull) >= float(gates["tail_abs_z"]),
        "extreme": abs(pull) >= float(gates["extreme_abs_z"]),
        "sym_fail": sym_fail,
        "schur": schur,
    }
    acc["n_primary"] += 1
    if flip:
        acc["n_flip"] += 1
    if min_eig is None or min_eig <= 0.0:
        acc["n_min_eig_le_0"] += 1
    _add(acc["all"], payload)
    if clean:
        acc["n_clean"] += 1
        _add(acc["clean"], payload)
    else:
        acc["n_dirty"] += 1
        _add(acc["dirty"], payload)
    if spd:
        _add(acc["spd"], payload)
    else:
        acc["n_nspd"] += 1
        _add(acc["nspd"], payload)
    if payload["tail"]:
        _add(acc["tail"], payload)
    if payload["extreme"]:
        _add(acc["extreme"], payload)

    def _put(family: str, label: str) -> None:
        acc[family].setdefault(label, _empty_bin())
        _add(acc[family][label], payload)

    if p_truth is not None:
        p_bin = assign_bin(p_truth, P_TRUTH_EDGES_MEV, P_TRUTH_LABELS)
        _put("given_p", p_bin)
        _put("clean_p" if clean else "dirty_p", p_bin)
    sig_t = assign_bin(sig_truth, SIGNIFICANCE_EDGES, SIGNIFICANCE_LABELS)
    _put("given_sig_truth", sig_t)
    if clean:
        _put("clean_sig_truth", sig_t)
        if p_truth is not None:
            key = f"{assign_bin(p_truth, P_TRUTH_EDGES_MEV, P_TRUTH_LABELS)}|{sig_t}"
            acc["p_x_sig_clean"].setdefault(key, _empty_bin())
            _add(acc["p_x_sig_clean"][key], payload)
    sig_f = assign_bin(sig_fit, SIGNIFICANCE_EDGES, SIGNIFICANCE_LABELS)
    _put("given_sig_fit", sig_f)
    if clean:
        _put("clean_sig_fit", sig_f)
    if tx is not None:
        _put("given_tx", assign_bin(tx, SLOPE_EDGES, TX_LABELS))
    if ty is not None:
        _put("given_ty", assign_bin(ty, SLOPE_EDGES, TY_LABELS))
    if log_cond is not None:
        _put("given_cond", assign_bin(log_cond, LOG_COND_EDGES, LOG_COND_LABELS))
    _put("given_corr", assign_bin(abs_corr_max, CORR_EDGES, CORR_LABELS))
    _put("given_n_mot", assign_bin(float(n_mot), N_MOT_EDGES, N_MOT_LABELS))
    _put("given_missing", missing)
    _put("given_chi2ndof", chi_label)
    _put("given_min_eig", _min_eig_label(min_eig))
    if clean:
        _put("clean_n_mot", assign_bin(float(n_mot), N_MOT_EDGES, N_MOT_LABELS))
        _put("clean_missing", missing)
    for name, factor in SCALE_FACTORS.items():
        z = pull * factor
        acc["scale"][name]["n"] += 1
        acc["scale"][name]["sumsq"] += z * z
        if abs(z) < float(gates["coverage_z68"]):
            acc["scale"][name]["n_cov68"] += 1
    if (
        str(row.get("campaign")) == "smoke"
        and any(
            str(item["source_id"]) == str(row.get("source_id"))
            and int(item["run_id"]) == int(row.get("run_id") or 0)
            and int(item["event_id"]) == int(row.get("event_id") or 0)
            and int(item["track_index"]) == int(row.get("track_index") or 0)
            for item in config["focus_identities"]
        )
    ):
        acc["focus_hits"].append(
            {
                "source_id": row.get("source_id"),
                "event_id": row.get("event_id"),
                "pull": pull,
                "flip": flip,
                "sig_truth": sig_truth,
                "sig_fit": sig_fit,
                "clean": clean,
                "spd": spd,
                "n_mot": n_mot,
                "min_eig": min_eig,
                "missing": missing,
            }
        )


def evaluate_tracks(records: Iterable[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    acc = empty_accumulator()
    n = 0
    for row in records:
        n += 1
        if n % 250000 == 0:
            print(f"three_st_qp_root_cause: evaluated {n} rows", file=sys.stderr, flush=True)
        process_track(row, acc, config)
    return acc


def inventory_split(config: Mapping[str, Any], split: str, campaign: str) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    dumps_present = True
    key = source_list_key(campaign, split)
    specs = []
    for spec in config["mc_data"][key]:
        xaod = authorize_path(
            spec["input_xaod"], AccessScope.DEVELOPMENT_VALIDATION, split=access_split(split)
        )
        track_path = dump_path_for_source(config, spec["source_id"], "tracks", campaign)
        event_path = dump_path_for_source(config, spec["source_id"], "events", campaign)
        if not track_path.is_file() or not event_path.is_file():
            dumps_present = False
        specs.append((spec, xaod, track_path, event_path))
    event_counts: list[int] = []

    def _event_iter() -> Iterator[dict[str, Any]]:
        for _spec, _xaod, _track_path, event_path in specs:
            n_event_rows = 0
            for row in iter_jsonl(event_path, split=split):
                n_event_rows += 1
                yield row
            event_counts.append(n_event_rows)

    def _track_iter() -> Iterator[dict[str, Any]]:
        for index, (spec, xaod, track_path, _event_path) in enumerate(specs):
            n_track_rows = 0
            for row in iter_jsonl(track_path, split=split):
                n_track_rows += 1
                yield row
            sources.append(
                {
                    "source_id": spec["source_id"],
                    "input_xaod": str(xaod),
                    "n_track_rows": n_track_rows,
                    "n_event_rows": event_counts[index] if index < len(event_counts) else 0,
                }
            )

    events = evaluate_events(_event_iter())
    tracks = evaluate_tracks(_track_iter(), config)
    return {
        "split": split,
        "campaign": campaign,
        "sources": sources,
        "dumps_present": dumps_present and bool(sources),
        "events": events,
        "tracks": tracks,
    }


def _monotonic_decreasing(summaries: Sequence[Mapping[str, Any]], max_inversion: float) -> bool:
    rates = [item["flip_rate"] for item in summaries if item.get("flip_rate") is not None and item.get("n", 0) >= 50]
    if len(rates) < 3:
        return False
    inversions = 0.0
    for left, right in zip(rates, rates[1:]):
        if right > left + max_inversion:
            inversions += 1
    return inversions <= 1


def evaluate_abcd(acc: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    gates = config["gates"]
    all_s = _summarize(acc["all"])
    clean_s = _summarize(acc["clean"])
    dirty_s = _summarize(acc["dirty"])
    nspd_s = _summarize(acc["nspd"])
    spd_s = _summarize(acc["spd"])
    clean_p = {k: _summarize(v) for k, v in acc["clean_p"].items()}
    dirty_p = {k: _summarize(v) for k, v in acc["dirty_p"].items()}
    clean_sig_t = {k: _summarize(v) for k, v in acc["clean_sig_truth"].items()}
    clean_sig_f = {k: _summarize(v) for k, v in acc["clean_sig_fit"].items()}
    high = clean_p.get("p_ge_2000gev") or {}
    low = clean_p.get("p_lt_200gev") or {}
    high_dirty = dirty_p.get("p_ge_2000gev") or {}
    a_applicable = int(clean_s["n"]) >= int(gates["min_n_mechanism_verdict"]) and int(
        high.get("n") or 0
    ) >= int(gates["min_n_high_momentum"])
    flip_gap = None
    if high.get("flip_rate") is not None and low.get("flip_rate") is not None:
        flip_gap = float(high["flip_rate"]) - float(low["flip_rate"])
    ordered_truth = [_summarize(acc["clean_sig_truth"][lab]) for lab in SIGNIFICANCE_LABELS]
    ordered_fit = [_summarize(acc["clean_sig_fit"][lab]) for lab in SIGNIFICANCE_LABELS]
    mono_truth = _monotonic_decreasing(ordered_truth, float(gates["monotonic_max_inversion"]))
    mono_fit = _monotonic_decreasing(ordered_fit, float(gates["monotonic_max_inversion"]))
    low_sig = clean_sig_t.get("sig_lt_0p5") or {}
    high_sig = clean_sig_t.get("sig_ge_10") or {}
    sig_gap = None
    if low_sig.get("flip_rate") is not None and high_sig.get("flip_rate") is not None:
        sig_gap = float(low_sig["flip_rate"]) - float(high_sig["flip_rate"])
    a_support = bool(
        a_applicable
        and flip_gap is not None
        and flip_gap >= float(gates["flip_rate_high_minus_low_min"])
        and (mono_truth or mono_fit)
        and sig_gap is not None
        and sig_gap >= float(gates["flip_rate_low_sig_minus_high_sig_min"])
    )
    a_reject = bool(a_applicable and flip_gap is not None and flip_gap < 0.0)
    a_status = STATUS_SUPPORTED if a_support else STATUS_REJECTED if a_reject else STATUS_UNRESOLVED

    dirty_gap = None
    if dirty_s.get("flip_rate") is not None and clean_s.get("flip_rate") is not None:
        dirty_gap = float(dirty_s["flip_rate"]) - float(clean_s["flip_rate"])
    high_topo_gap = None
    if high_dirty.get("flip_rate") is not None and high.get("flip_rate") is not None:
        high_topo_gap = float(high_dirty["flip_rate"]) - float(high["flip_rate"])
    b_support = bool(
        dirty_gap is not None
        and dirty_gap >= float(gates["dirty_minus_clean_flip_min"])
        and int(dirty_s["n"]) >= int(gates["min_n_stratum"])
        and int(clean_s["n"]) >= int(gates["min_n_stratum"])
    )
    b_reject = bool(dirty_gap is not None and dirty_gap < 0.05 and a_applicable)
    b_status = STATUS_SUPPORTED if b_support else STATUS_REJECTED if b_reject else STATUS_UNRESOLVED

    scale_reports = {}
    unit_rescue = False
    for name, item in acc["scale"].items():
        n = int(item["n"])
        rms = math.sqrt(float(item["sumsq"]) / n) if n else None
        cov68 = float(item["n_cov68"]) / n if n else None
        rescued = bool(
            rms is not None
            and cov68 is not None
            and float(gates["scale_rescue_rms_min"]) <= rms <= float(gates["scale_rescue_rms_max"])
            and float(gates["scale_rescue_cov68_min"]) <= cov68 <= float(gates["scale_rescue_cov68_max"])
        )
        scale_reports[name] = {"rms_pull": rms, "coverage_68": cov68, "rescues": rescued}
        if rescued:
            unit_rescue = True
    c_status = STATUS_SUPPORTED if unit_rescue else STATUS_REJECTED

    d_applicable = int(clean_s["n"]) >= int(gates["min_n_mechanism_verdict"])
    clean_miscal = bool(
        clean_s.get("rms_pull") is not None
        and (
            float(clean_s["rms_pull"]) > float(gates["pull_rms_calibrated_max"])
            or float(clean_s["coverage_68"] or 0.0) < float(gates["coverage_68_calibrated_min"])
            or float(clean_s["coverage_95"] or 0.0) < float(gates["coverage_95_calibrated_min"])
        )
    )
    tail_frac = clean_s.get("tail_sumsq_fraction")
    d_support = bool(
        d_applicable
        and not unit_rescue
        and clean_miscal
        and tail_frac is not None
        and float(tail_frac) >= float(gates["tail_sumsq_fraction_min"])
    )
    d_reject = bool(d_applicable and not clean_miscal)
    d_status = STATUS_SUPPORTED if d_support else STATUS_REJECTED if d_reject else STATUS_UNRESOLVED

    def _contrib(part: Mapping[str, Any]) -> dict[str, Any]:
        all_sumsq = float(acc["all"]["sumsq_pull"])
        all_out68 = float(acc["all"]["n"] - acc["all"]["n_cov68"])
        return {
            "n": int(part["n"]),
            "n_flip": int(part["n_flip"]),
            "sumsq_fraction": (
                float(part["sumsq_pull"]) / all_sumsq if all_sumsq > 0.0 else None
            ),
            "flip_fraction": (
                float(part["n_flip"]) / float(acc["n_flip"]) if acc["n_flip"] else None
            ),
            "out68_fraction": (
                float(part["n"] - part["n_cov68"]) / all_out68 if all_out68 else None
            ),
            "deleted": False,
        }

    def _pooled(family: Mapping[str, Any], labels: Sequence[str]) -> dict[str, Any]:
        pooled_bin = _empty_bin()
        for label in labels:
            pooled_bin = _merge_bin(pooled_bin, family.get(label, _empty_bin()))
        return _summarize(pooled_bin)

    nspd_contrib = _contrib(acc["nspd"])
    tail_contrib = _contrib(acc["tail"])
    extreme_contrib = _contrib(acc["extreme"])
    sig_fit_low = _pooled(acc["given_sig_fit"], LOW_SIG_LABELS)
    sig_fit_high = _pooled(acc["given_sig_fit"], HIGH_SIG_LABELS)
    sig_truth_low = _pooled(acc["given_sig_truth"], LOW_SIG_LABELS)
    sig_truth_high = _pooled(acc["given_sig_truth"], HIGH_SIG_LABELS)
    clean_fit_low = _pooled(acc["clean_sig_fit"], LOW_SIG_LABELS)
    clean_fit_high = _pooled(acc["clean_sig_fit"], HIGH_SIG_LABELS)
    clean_truth_low = _pooled(acc["clean_sig_truth"], LOW_SIG_LABELS)
    clean_truth_high = _pooled(acc["clean_sig_truth"], HIGH_SIG_LABELS)
    unidentifiable = {
        "measurement_sig_fit_lt_1": sig_fit_low,
        "measurement_sig_fit_ge_3": sig_fit_high,
        "truth_sig_lt_1": sig_truth_low,
        "truth_sig_ge_3": sig_truth_high,
        "clean_measurement_sig_fit_lt_1": clean_fit_low,
        "clean_measurement_sig_fit_ge_3": clean_fit_high,
        "clean_truth_sig_lt_1": clean_truth_low,
        "clean_truth_sig_ge_3": clean_truth_high,
        "flip_share_in_sig_fit_lt_1": (
            float(sig_fit_low.get("n_flip") or 0) / float(acc["n_flip"])
            if acc["n_flip"]
            else None
        ),
        "flip_mostly_where_qp_unresolvable": bool(
            sig_fit_low.get("flip_rate") is not None
            and sig_fit_high.get("flip_rate") is not None
            and int(sig_fit_low.get("n") or 0) >= int(gates["min_n_stratum"])
            and float(sig_fit_low["flip_rate"])
            >= float(sig_fit_high["flip_rate"]) + 0.10
        ),
    }
    info_limit_remains_if_cov_fixed = bool(a_status == STATUS_SUPPORTED)

    return {
        "A": {
            "name": MECHANISM_A,
            "status": a_status,
            "applicable": a_applicable,
            "clean_high_minus_low_flip": flip_gap,
            "clean_high_p_flip": high.get("flip_rate"),
            "clean_low_p_flip": low.get("flip_rate"),
            "clean_high_p_n": high.get("n"),
            "monotonic_truth_significance": mono_truth,
            "monotonic_fit_significance": mono_fit,
            "clean_low_minus_high_sig_flip": sig_gap,
            "clean_n": clean_s["n"],
        },
        "B": {
            "name": MECHANISM_B,
            "status": b_status,
            "dirty_minus_clean_flip": dirty_gap,
            "high_p_dirty_minus_clean_flip": high_topo_gap,
            "dirty_flip": dirty_s.get("flip_rate"),
            "clean_flip": clean_s.get("flip_rate"),
        },
        "C": {
            "name": MECHANISM_C,
            "status": c_status,
            "global_unit_scale_rescues": unit_rescue,
            "scale_tests": scale_reports,
            "source_chain": conversion_chain(),
            "qoverp_variance_is_basis_invariant": True,
            "remaining_unresolved_without_new_dump": [
                "whether this track is CKF-only or KF-refit",
                "Acts-native 6x6 before ConvertActsTrackParameterToATLAS",
            ],
        },
        "D": {
            "name": MECHANISM_D,
            "status": d_status,
            "clean_rms_pull": clean_s.get("rms_pull"),
            "clean_coverage_68": clean_s.get("coverage_68"),
            "clean_coverage_95": clean_s.get("coverage_95"),
            "clean_tail_sumsq_fraction": tail_frac,
            "clean_miscalibrated": clean_miscal,
        },
        "information_limit_remains_if_covariance_fixed": info_limit_remains_if_cov_fixed,
        "overall": all_s,
        "clean": clean_s,
        "dirty": dirty_s,
        "spd": spd_s,
        "nspd": nspd_s,
        "nspd_contribution": nspd_contrib,
        "tail_contribution": tail_contrib,
        "extreme_contribution": extreme_contrib,
        "unidentifiable": unidentifiable,
        "clean_p": clean_p,
        "dirty_p": dirty_p,
        "clean_sig_truth": clean_sig_t,
        "clean_sig_fit": clean_sig_f,
        "given_p": {k: _summarize(v) for k, v in acc["given_p"].items()},
        "given_tx": {k: _summarize(v) for k, v in acc["given_tx"].items()},
        "given_ty": {k: _summarize(v) for k, v in acc["given_ty"].items()},
        "given_cond": {k: _summarize(v) for k, v in acc["given_cond"].items()},
        "given_corr": {k: _summarize(v) for k, v in acc["given_corr"].items()},
        "given_n_mot": {k: _summarize(v) for k, v in acc["given_n_mot"].items()},
        "given_missing": {k: _summarize(v) for k, v in acc["given_missing"].items()},
        "given_chi2ndof": {k: _summarize(v) for k, v in acc["given_chi2ndof"].items()},
        "given_min_eig": {k: _summarize(v) for k, v in acc["given_min_eig"].items()},
        "p_x_sig_clean": {k: _summarize(v) for k, v in acc["p_x_sig_clean"].items()},
        "tail": _summarize(acc["tail"]),
        "extreme": _summarize(acc["extreme"]),
        "focus_hits": list(acc["focus_hits"]),
        "n_primary": int(acc["n_primary"]),
        "n_records": int(acc["n_records"]),
        "n_flip": int(acc["n_flip"]),
        "n_clean": int(acc["n_clean"]),
        "n_nspd": int(acc["n_nspd"]),
        "n_min_eig_le_0": int(acc["n_min_eig_le_0"]),
        "n_truth_as_solution": int(acc["n_truth_as_solution"]),
    }


def diagnostic_export_contract() -> dict[str, Any]:
    return {
        "authorized": False,
        "reason_not_authorized": (
            "Existing dumps plus the pinned CreateTrkTrack/CKF2/KalmanFitter "
            "source chain already decide A/B/C-unit/D.  Acts-native covariance "
            "would only label CKF-vs-refit, which does not change those decisions."
        ),
        "if_later_opened_would_need": [
            "acts_bound_covariance_6x6_before_unit_scale",
            "trk_curvilinear_covariance_5x5_after_unit_scale",
            "kf_refit_succeeded",
            "front_surface_type_and_z",
            "same focus identities, <= 20 events, no truth in the fit",
        ],
        "must_not": [
            "change seed, fitter, hit selection, covariance scale, or geometry",
            "replace fit q/p with truth",
            "flip WB119 or authorize S3",
        ],
    }


def _denominator_ok(acc: Mapping[str, Any], campaign: str, config: Mapping[str, Any]) -> bool:
    if campaign == "batch":
        frozen = config["frozen_wb119_batch_denominator"]
        return (
            int(acc["n_primary"]) == int(frozen["n_primary"])
            and int(acc["n_flip"]) == int(frozen["n_sign_flip"])
            and int(acc["n_records"]) == int(frozen["n_without_ift_tracks"])
        )
    frozen = config["frozen_wb119_smoke_denominator"]
    return int(acc["n_primary"]) == int(frozen["n_primary"]) and int(acc["n_flip"]) == int(
        frozen["n_sign_flip"]
    )


def decide(
    construction: Mapping[str, Any],
    validation: Mapping[str, Any],
    pins: Mapping[str, Any],
    inherited: Mapping[str, Any],
    *,
    dumps_materialized: bool,
    campaign: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    mechanism = None
    contract = "FAIL"
    pooled = merge_accumulators(construction["tracks"], validation["tracks"])
    if inherited.get("workbook_119", {}).get("decision") != "three_st_qp_calibration_not_established":
        mechanism = "s2_not_inherited_as_fail"
    elif inherited.get("workbook_120", {}).get("decision") != "three_st_qp_failure_diagnosis_recorded":
        mechanism = "s2d_not_inherited"
    elif not pins.get("all_match", False):
        mechanism = "pinned_calypso_source_hash_mismatch"
    elif not dumps_materialized:
        mechanism = "calibration_dump_not_materialized"
    elif int(pooled["n_truth_as_solution"]):
        mechanism = "truth_used_as_fit_or_solution"
    elif not _denominator_ok(pooled, campaign, config):
        mechanism = "wb119_denominator_drift"
    else:
        contract = "PASS"
    diagnosis = evaluate_abcd(pooled, config)
    if contract != "PASS":
        decision = DECISION_NOT
        verdict = "FAIL"
        diagnosis_verdict = "FAIL"
    elif int(diagnosis["n_primary"]) < int(config["gates"]["min_n_mechanism_verdict"]):
        decision = DECISION_CONTRACT
        verdict = "PASS"
        diagnosis_verdict = "INCONCLUSIVE"
        mechanism = None
    else:
        decision = DECISION_RECORDED
        verdict = "PASS"
        diagnosis_verdict = "RECORDED"
        mechanism = "structured_abcd"
    return {
        "kind": "three_st_qp_root_cause_contract",
        "task": TASK,
        "workbook": WORKBOOK,
        "campaign": campaign,
        "verdict": verdict,
        "decision": decision,
        "mechanism": mechanism,
        "contract_verdict": contract,
        "diagnosis_verdict": diagnosis_verdict,
        "A": diagnosis["A"]["status"] if contract == "PASS" else None,
        "B": diagnosis["B"]["status"] if contract == "PASS" else None,
        "C": diagnosis["C"]["status"] if contract == "PASS" else None,
        "D": diagnosis["D"]["status"] if contract == "PASS" else None,
        "information_limit_remains_if_covariance_fixed": (
            diagnosis["information_limit_remains_if_covariance_fixed"]
            if contract == "PASS"
            else None
        ),
        "three_st_qp_trusted_observable": False,
        "residual_conditional_authorized": False,
        "s2_flipped_to_pass": False,
        "sign_flip_tracks_dropped": False,
        "nspd_tracks_dropped": False,
        "new_reconstruction_dump_authorized": False,
        "diagnostic_export_contract": diagnostic_export_contract(),
        "conversion_chain": conversion_chain(),
        "diagnosis": diagnosis,
        "construction": {
            "n_records": construction["tracks"]["n_records"],
            "n_primary": construction["tracks"]["n_primary"],
            "n_flip": construction["tracks"]["n_flip"],
        },
        "validation": {
            "n_records": validation["tracks"]["n_records"],
            "n_primary": validation["tracks"]["n_primary"],
            "n_flip": validation["tracks"]["n_flip"],
        },
    }
