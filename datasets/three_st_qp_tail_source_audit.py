"""Yasu-S2G: extreme-tail and split/source instability audit.

Official means stay on the untrimmed primary sample:
    μ_fit  = E[(q/p)_fit]
    μ_bias = E[Δ(q/p)]
Influence numbers ("mean after removing class X") are diagnostics only.
Matched / reweighted comparisons are not a new calibration population.
Does not flip S2 or open S3.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import yaml

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
)
from datasets.three_st_qp_root_cause_audit import CORR_EDGES, CORR_LABELS, LOG_COND_EDGES, LOG_COND_LABELS

SCHEMA_VERSION = "three-st-qp-tail-source-audit-v1"
DEFAULT_CONFIG = "configs/three_st_qp_tail_source_audit_v1.yaml"
TASK = "YASU-S2G"
WORKBOOK = 123

DECISION_CONTRACT = "three_st_qp_tail_source_contract_established"
DECISION_RECORDED = "three_st_qp_tail_source_recorded"
DECISION_NOT = "three_st_qp_tail_source_not_established"

STATUS_SUPPORTED = "supported"
STATUS_REJECTED = "rejected"
STATUS_UNRESOLVED = "unresolved"

MECH_CATASTROPHIC = "tail_from_sparse_catastrophic_fits"
MECH_BROAD = "broad_reconstruction_shift"
MECH_SOURCE = "source_specific_reconstruction_response"
MECH_COMPOSITION = "population/composition_explains_split"
MECH_REFIT = "refit_provenance_suspected"
MECH_MIXED = "mixed/inconclusive"

REQUIRED_INHERITANCE = (
    "workbook_117",
    "workbook_118",
    "workbook_119",
    "workbook_120",
    "workbook_121",
    "workbook_122",
)

SIGMA_EDGES = (0.0, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e12)
SIGMA_LABELS = (
    "sigma_lt_1e-7",
    "sigma_1e-7_1e-6",
    "sigma_1e-6_1e-5",
    "sigma_1e-5_1e-4",
    "sigma_ge_1e-4",
)
C44_EDGES = (0.0, 1.0e-14, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e12)
C44_LABELS = (
    "c44_lt_1e-14",
    "c44_1e-14_1e-12",
    "c44_1e-12_1e-10",
    "c44_1e-10_1e-8",
    "c44_ge_1e-8",
)
NDOF_EDGES = (0.0, 1.0, 9.0, 13.0, 16.0, 1.0e6)
NDOF_LABELS = ("ndof_le_0", "ndof_1_8", "ndof_9_12", "ndof_13_15", "ndof_ge_16")
FRONT_Z_LABELS = (
    "front_upstream",
    "front_near_s1",
    "front_s1_s2",
    "front_near_s2",
    "front_s2_s3",
    "front_near_s3",
    "front_other",
    "front_unknown",
)
USUAL_FRONT = "front_near_s1"
BIN_FAMILIES = (
    "given_source",
    "given_charge",
    "given_p",
    "given_tx",
    "given_ty",
    "given_n_mot",
    "given_missing",
    "given_ndof",
    "given_chi2ndof",
    "given_spd",
    "given_c44",
    "given_sigma",
    "given_corr",
    "given_cond",
    "given_front",
)


class ThreeStQpTailSourceError(ValueError):
    """Raised when the S2G contract is illegal."""


def refuse_flip_s2() -> None:
    raise ThreeStQpTailSourceError("S2G cannot set three_st_qp_trusted_observable true")


def refuse_residual_conditional() -> None:
    raise ThreeStQpTailSourceError("E[r_IFT|q/p] is Stage 3; S2G does not open it")


def refuse_trim_official_mean() -> None:
    raise ThreeStQpTailSourceError(
        "official μ is untrimmed; do not delete, trim, or winsorize"
    )


def refuse_reweight_as_calibration() -> None:
    raise ThreeStQpTailSourceError(
        "matched/reweighted numbers are diagnostic only, not a calibration population"
    )


def refuse_drop_tails() -> None:
    raise ThreeStQpTailSourceError("extreme tails stay in the sample; they are a contribution")


def official_quantities() -> dict[str, str]:
    return {
        "mu_fit": "E[(q/p)_fit] on the untrimmed primary sample",
        "mu_truth": "E[(q/p)_truth] on the same sample",
        "mu_bias": "E[Δ(q/p)] = E[(q/p)_fit - (q/p)_truth]",
        "identity": "μ_fit = μ_truth + μ_bias",
        "sum_delta": "Σ Δ(q/p); class share is contributive",
        "sum_fit": "Σ (q/p)_fit; class share is contributive",
        "sumsq_pull": "Σ z²; class share of pull RMS²",
        "sign_flip": "count of sign(q_fit) ≠ sign(q_truth)",
        "coverage_failure": "count of |z| ≥ 1",
        "tail_z10": "|z| ≥ 10",
        "tail_z100": "|z| ≥ 100",
        "tail_0p1": "top 0.1% |Δ(q/p)|",
        "tail_0p01": "top 0.01% |Δ(q/p)|",
        "influence": "remaining mean after removing a class; diagnostic only",
        "matched": "same charge, (p,tx,ty), topology; diagnostic only",
    }


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ThreeStQpTailSourceError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise ThreeStQpTailSourceError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise ThreeStQpTailSourceError("workbook must be 123")
    if bool(config.get("three_st_qp_trusted_observable", True)):
        raise ThreeStQpTailSourceError("three_st_qp_trusted_observable must stay false")
    if bool(config.get("residual_conditional_authorized", True)):
        raise ThreeStQpTailSourceError("residual_conditional_authorized must stay false")
    if bool(config.get("do_not_submit_reconstruction_dump", False)) is not True:
        raise ThreeStQpTailSourceError("do_not_submit_reconstruction_dump must be true")
    if bool(config.get("do_not_trim_or_winsorize_official_mean", False)) is not True:
        raise ThreeStQpTailSourceError("official mean must stay untrimmed")
    if bool(config.get("do_not_treat_reweight_as_calibration_population", False)) is not True:
        raise ThreeStQpTailSourceError("reweight must stay diagnostic")
    if bool(config.get("do_not_treat_influence_as_official_mean", False)) is not True:
        raise ThreeStQpTailSourceError("influence must stay diagnostic")
    if len(config.get("focus_identities") or []) != 6:
        raise ThreeStQpTailSourceError("focus_identities must stay the six WB119 smoke rows")
    return dict(config)


def _expect_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise ThreeStQpTailSourceError(f"{label} hash mismatch: {digest}")


def inherit_frozen_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    inherited: dict[str, Any] = {}
    frozen_root = Path(str(config.get("frozen_artifact_root") or project_root()))
    for workbook in REQUIRED_INHERITANCE:
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
            raise ThreeStQpTailSourceError(f"{workbook} decision must stay frozen")
        inherited[workbook] = {
            "decision": decision["decision"],
            "config_sha256": spec["config_sha256"],
            "decision_sha256": spec["decision_sha256"],
        }
    if inherited["workbook_119"]["decision"] != "three_st_qp_calibration_not_established":
        raise ThreeStQpTailSourceError("WB119 must remain not_established")
    if inherited["workbook_122"]["decision"] != "three_st_qp_mean_decomposition_recorded":
        raise ThreeStQpTailSourceError("WB122 mean decomposition must remain recorded")
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


def _same_sign(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return False
    return _sign(left) != 0 and _sign(left) == _sign(right)


def _ratio(numer: float | None, denom: float | None) -> float | None:
    if numer is None or denom is None or denom == 0.0:
        return None
    return float(numer) / float(denom)


def _empty_bin() -> dict[str, float | int]:
    return {
        "n": 0,
        "n_plus": 0,
        "n_minus": 0,
        "n_flip": 0,
        "n_out68": 0,
        "n_out95": 0,
        "n_z10": 0,
        "n_z100": 0,
        "n_delta_neg": 0,
        "sum_fit": 0.0,
        "sum_truth": 0.0,
        "sum_delta": 0.0,
        "sum_abs_fit": 0.0,
        "sum_abs_delta": 0.0,
        "sumsq_pull": 0.0,
    }


def _add(bin_acc: dict[str, Any], row: Mapping[str, Any]) -> None:
    bin_acc["n"] += 1
    if row["charge"] > 0.0:
        bin_acc["n_plus"] += 1
    elif row["charge"] < 0.0:
        bin_acc["n_minus"] += 1
    if row["flip"]:
        bin_acc["n_flip"] += 1
    if row["out68"]:
        bin_acc["n_out68"] += 1
    if row["out95"]:
        bin_acc["n_out95"] += 1
    if row["z10"]:
        bin_acc["n_z10"] += 1
    if row["z100"]:
        bin_acc["n_z100"] += 1
    if row["delta"] < 0.0:
        bin_acc["n_delta_neg"] += 1
    bin_acc["sum_fit"] += row["q_fit"]
    bin_acc["sum_truth"] += row["q_truth"]
    bin_acc["sum_delta"] += row["delta"]
    bin_acc["sum_abs_fit"] += abs(row["q_fit"])
    bin_acc["sum_abs_delta"] += abs(row["delta"])
    bin_acc["sumsq_pull"] += row["pull"] * row["pull"]


def _merge_bin(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(left)
    for key, value in right.items():
        out[key] = left.get(key, 0) + value
    return out


def _mean_pack(bin_acc: Mapping[str, Any]) -> dict[str, Any]:
    n = int(bin_acc["n"])
    if n <= 0:
        return {
            "n": 0,
            "n_plus": 0,
            "n_minus": 0,
            "n_flip": 0,
            "n_out68": 0,
            "n_out95": 0,
            "n_z10": 0,
            "n_z100": 0,
            "pi_plus": None,
            "pi_minus": None,
            "mu_fit": None,
            "mu_truth": None,
            "mu_bias": None,
            "identity_residual": None,
            "flip_rate": None,
            "coverage_68": None,
            "coverage_95": None,
            "rms_pull": None,
            "frac_delta_neg": None,
        }
    mu_fit = float(bin_acc["sum_fit"]) / n
    mu_truth = float(bin_acc["sum_truth"]) / n
    mu_bias = float(bin_acc["sum_delta"]) / n
    return {
        "n": n,
        "n_plus": int(bin_acc["n_plus"]),
        "n_minus": int(bin_acc["n_minus"]),
        "n_flip": int(bin_acc["n_flip"]),
        "n_out68": int(bin_acc["n_out68"]),
        "n_out95": int(bin_acc["n_out95"]),
        "n_z10": int(bin_acc["n_z10"]),
        "n_z100": int(bin_acc["n_z100"]),
        "pi_plus": float(bin_acc["n_plus"]) / n,
        "pi_minus": float(bin_acc["n_minus"]) / n,
        "mu_fit": mu_fit,
        "mu_truth": mu_truth,
        "mu_bias": mu_bias,
        "identity_residual": mu_fit - mu_truth - mu_bias,
        "flip_rate": float(bin_acc["n_flip"]) / n,
        "coverage_68": 1.0 - float(bin_acc["n_out68"]) / n,
        "coverage_95": 1.0 - float(bin_acc["n_out95"]) / n,
        "rms_pull": math.sqrt(float(bin_acc["sumsq_pull"]) / n),
        "frac_delta_neg": float(bin_acc["n_delta_neg"]) / n,
    }


def _influence(total: Mapping[str, Any], part: Mapping[str, Any]) -> dict[str, Any]:
    n_all = int(total["n"])
    n_part = int(part["n"])
    n_rem = n_all - n_part
    base = {
        "diagnostic_only": True,
        "official_mean_redefined": False,
        "calibration_population": False,
        "n_removed": n_part,
        "remaining_n": n_rem,
    }
    if n_all <= 0 or n_rem <= 0:
        return {
            **base,
            "applicable": False,
            "remaining_mu_fit": None,
            "remaining_mu_truth": None,
            "remaining_mu_bias": None,
            "fraction_of_original_mu_bias": None,
            "fraction_of_original_mu_fit": None,
        }
    rem_fit = (float(total["sum_fit"]) - float(part["sum_fit"])) / n_rem
    rem_truth = (float(total["sum_truth"]) - float(part["sum_truth"])) / n_rem
    rem_bias = (float(total["sum_delta"]) - float(part["sum_delta"])) / n_rem
    mu_bias = float(total["sum_delta"]) / n_all
    mu_fit = float(total["sum_fit"]) / n_all
    return {
        **base,
        "applicable": True,
        "remaining_mu_fit": rem_fit,
        "remaining_mu_truth": rem_truth,
        "remaining_mu_bias": rem_bias,
        "fraction_of_original_mu_bias": _ratio(rem_bias, mu_bias),
        "fraction_of_original_mu_fit": _ratio(rem_fit, mu_fit),
    }


def _contrib(part: Mapping[str, Any], total: Mapping[str, Any]) -> dict[str, Any]:
    pack = _mean_pack(part)
    n_all = int(total["n"])
    return {
        **pack,
        "share_n": _ratio(part["n"], n_all),
        "share_sum_delta": _ratio(part["sum_delta"], total["sum_delta"]),
        "share_sum_fit": _ratio(part["sum_fit"], total["sum_fit"]),
        "share_sum_abs_delta": _ratio(part["sum_abs_delta"], total["sum_abs_delta"]),
        "share_sumsq_pull": _ratio(part["sumsq_pull"], total["sumsq_pull"]),
        "share_flip": _ratio(part["n_flip"], total["n_flip"]),
        "share_out68": _ratio(part["n_out68"], total["n_out68"]),
        "influence": _influence(total, part),
        "official_mean_redefined": False,
    }


def _family_report(family: Mapping[str, Mapping[str, Any]], total: Mapping[str, Any]) -> dict[str, Any]:
    return {label: _contrib(bin_acc, total) for label, bin_acc in sorted(family.items())}


def empty_accumulator() -> dict[str, Any]:
    return {
        "n_records": 0,
        "n_primary": 0,
        "n_flip": 0,
        "n_truth_as_solution": 0,
        "n_clean": 0,
        "n_dirty": 0,
        "n_nspd": 0,
        "all": _empty_bin(),
        "plus": _empty_bin(),
        "minus": _empty_bin(),
        "clean": _empty_bin(),
        "dirty": _empty_bin(),
        "spd": _empty_bin(),
        "nspd": _empty_bin(),
        "z10": _empty_bin(),
        "z100": _empty_bin(),
        "unusual_front": _empty_bin(),
        "given_source": {},
        "given_charge": {"plus": _empty_bin(), "minus": _empty_bin()},
        "given_p": {label: _empty_bin() for label in P_TRUTH_LABELS},
        "given_tx": {label: _empty_bin() for label in TX_LABELS},
        "given_ty": {label: _empty_bin() for label in TY_LABELS},
        "given_n_mot": {label: _empty_bin() for label in N_MOT_LABELS},
        "given_missing": {label: _empty_bin() for label in MISSING_LABELS},
        "given_ndof": {label: _empty_bin() for label in NDOF_LABELS},
        "given_chi2ndof": {},
        "given_spd": {"spd": _empty_bin(), "nspd": _empty_bin(), "cov_missing": _empty_bin()},
        "given_c44": {label: _empty_bin() for label in C44_LABELS},
        "given_sigma": {label: _empty_bin() for label in SIGMA_LABELS},
        "given_corr": {label: _empty_bin() for label in CORR_LABELS},
        "given_cond": {label: _empty_bin() for label in LOG_COND_LABELS},
        "given_front": {label: _empty_bin() for label in FRONT_Z_LABELS},
        "matched": {},
        "q_fit": [],
        "q_truth": [],
        "delta": [],
        "pull": [],
        "source_id": [],
        "front": [],
        "focus_hits": [],
    }


def merge_accumulators(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    out = empty_accumulator()
    for key in (
        "n_records",
        "n_primary",
        "n_flip",
        "n_truth_as_solution",
        "n_clean",
        "n_dirty",
        "n_nspd",
    ):
        out[key] = left[key] + right[key]
    for key in ("all", "plus", "minus", "clean", "dirty", "spd", "nspd", "z10", "z100", "unusual_front"):
        out[key] = _merge_bin(left[key], right[key])
    for family in (*BIN_FAMILIES, "matched"):
        keys = set(left[family]) | set(right[family])
        out[family] = {
            key: _merge_bin(left[family].get(key, _empty_bin()), right[family].get(key, _empty_bin()))
            for key in keys
        }
    for key in ("q_fit", "q_truth", "delta", "pull", "source_id", "front", "focus_hits"):
        out[key] = list(left[key]) + list(right[key])
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


def _front_z_label(z_mm: float | None, config: Mapping[str, Any]) -> str:
    if z_mm is None:
        return "front_unknown"
    station_z = config.get("station_z_mm") or {}
    s1 = float(station_z.get(1, station_z.get("1", 47.4)))
    s2 = float(station_z.get(2, station_z.get("2", 1237.4)))
    s3 = float(station_z.get(3, station_z.get("3", 2427.4)))
    near_s1 = float(config["binning"]["front_near_s1_abs_mm"])
    near_other = float(config["binning"]["front_near_other_abs_mm"])
    upstream_max = float(config["binning"]["front_upstream_max_mm"])
    if abs(z_mm - s1) <= near_s1:
        return "front_near_s1"
    if z_mm < upstream_max:
        return "front_upstream"
    if abs(z_mm - s2) <= near_other:
        return "front_near_s2"
    if abs(z_mm - s3) <= near_other:
        return "front_near_s3"
    if s1 + near_s1 < z_mm < s2 - near_other:
        return "front_s1_s2"
    if s2 + near_other < z_mm < s3 - near_other:
        return "front_s2_s3"
    return "front_other"


def _cov_features(
    cov_payload: Any, sigma: float
) -> tuple[bool | None, float | None, float, float | None, float]:
    c44 = sigma * sigma
    if cov_payload is None:
        return None, None, 0.0, None, c44
    cov = np.asarray(cov_payload, dtype=np.float64)
    if cov.shape != (5, 5) or not np.isfinite(cov).all():
        return None, None, 0.0, None, c44
    if float(cov[QOVERP_INDEX, QOVERP_INDEX]) > 0.0:
        c44 = float(cov[QOVERP_INDEX, QOVERP_INDEX])
    abs_corr_max = 0.0
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
    spd = None
    log_cond = None
    min_eig = None
    try:
        eig = np.linalg.eigvalsh(0.5 * (cov + cov.T))
        if np.isfinite(eig).all():
            min_eig = float(np.min(eig))
            if min_eig > 0.0:
                spd = True
                log_cond = math.log(float(np.max(eig)) / min_eig)
            else:
                spd = False
    except np.linalg.LinAlgError:
        spd = False
    return spd, log_cond, abs_corr_max, min_eig, c44


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
    charge = _finite(row.get("truth_charge"))
    if charge is None:
        charge = float(_sign(q_truth))
    delta = q_fit - q_truth
    pull = delta / sigma
    flip = _sign(q_fit) == 0 or _sign(q_truth) == 0 or _sign(q_fit) != _sign(q_truth)
    p_truth = _finite(row.get("p_truth_s1_mev"))
    tx = _finite(row.get("tx_truth")) or _finite(row.get("tx"))
    ty = _finite(row.get("ty_truth")) or _finite(row.get("ty"))
    stations = [int(s) for s in (row.get("measurements_on_track_stations") or [])]
    n_mot = int(row.get("n_mot") or 0)
    n_out = int(row.get("n_outlier_hits") or 0)
    chi2 = _finite(row.get("chi2"))
    ndof = _finite(row.get("ndof"))
    chi2ndof = None if ndof is None or ndof <= 0.0 or chi2 is None else chi2 / ndof
    complete = _missing_complete(stations)
    missing = _missing_label(stations)
    if ndof is None or ndof <= 0.0:
        chi_label = "ndof_zero"
        ndof_label = "ndof_le_0"
    else:
        chi_label = assign_bin(float(chi2 or 0.0) / ndof, CHI2_EDGES, CHI2_LABELS)
        ndof_label = assign_bin(float(ndof), NDOF_EDGES, NDOF_LABELS)
    clean = (
        complete
        and n_mot == int(clean_spec["require_n_mot"])
        and ndof is not None
        and ndof > 0.0
        and n_out == int(clean_spec["require_n_outlier"])
        and chi2ndof is not None
        and chi2ndof < float(clean_spec["max_chi2_ndof"])
    )
    spd, log_cond, abs_corr_max, _min_eig, c44 = _cov_features(row.get("native_covariance"), sigma)
    front = _front_z_label(_finite(row.get("z_mm")), config)
    z10 = abs(pull) >= float(gates["tail_abs_z"])
    z100 = abs(pull) >= float(gates["extreme_abs_z"])
    payload = {
        "q_fit": q_fit,
        "q_truth": q_truth,
        "delta": delta,
        "pull": pull,
        "charge": charge,
        "flip": flip,
        "out68": abs(pull) >= float(gates["coverage_z68"]),
        "out95": abs(pull) >= float(gates["coverage_z95"]),
        "z10": z10,
        "z100": z100,
    }
    acc["n_primary"] += 1
    if flip:
        acc["n_flip"] += 1
    _add(acc["all"], payload)
    if charge > 0.0:
        _add(acc["plus"], payload)
        _add(acc["given_charge"]["plus"], payload)
    elif charge < 0.0:
        _add(acc["minus"], payload)
        _add(acc["given_charge"]["minus"], payload)
    if clean:
        acc["n_clean"] += 1
        _add(acc["clean"], payload)
    else:
        acc["n_dirty"] += 1
        _add(acc["dirty"], payload)
    if spd is True:
        _add(acc["spd"], payload)
        _add(acc["given_spd"]["spd"], payload)
    elif spd is False:
        acc["n_nspd"] += 1
        _add(acc["nspd"], payload)
        _add(acc["given_spd"]["nspd"], payload)
    else:
        _add(acc["given_spd"]["cov_missing"], payload)
    if z10:
        _add(acc["z10"], payload)
    if z100:
        _add(acc["z100"], payload)
    if front != USUAL_FRONT and front != "front_unknown":
        _add(acc["unusual_front"], payload)

    def _put(family: str, label: str) -> None:
        acc[family].setdefault(label, _empty_bin())
        _add(acc[family][label], payload)

    if p_truth is not None:
        p_bin = assign_bin(p_truth, P_TRUTH_EDGES_MEV, P_TRUTH_LABELS)
        _put("given_p", p_bin)
    else:
        p_bin = "unavailable"
    tx_bin = assign_bin(tx, SLOPE_EDGES, TX_LABELS) if tx is not None else "unavailable"
    ty_bin = assign_bin(ty, SLOPE_EDGES, TY_LABELS) if ty is not None else "unavailable"
    if tx is not None:
        _put("given_tx", tx_bin)
    if ty is not None:
        _put("given_ty", ty_bin)
    _put("given_source", str(row.get("source_id") or "unknown"))
    _put("given_n_mot", assign_bin(float(n_mot), N_MOT_EDGES, N_MOT_LABELS))
    _put("given_missing", missing)
    _put("given_ndof", ndof_label)
    _put("given_chi2ndof", chi_label)
    _put("given_sigma", assign_bin(sigma, SIGMA_EDGES, SIGMA_LABELS))
    _put("given_c44", assign_bin(c44, C44_EDGES, C44_LABELS))
    _put("given_corr", assign_bin(abs_corr_max, CORR_EDGES, CORR_LABELS))
    if log_cond is not None:
        _put("given_cond", assign_bin(log_cond, LOG_COND_EDGES, LOG_COND_LABELS))
    _put("given_front", front)
    if p_bin != "unavailable" and tx_bin != "unavailable" and ty_bin != "unavailable" and charge != 0.0:
        charge_lab = "plus" if charge > 0.0 else "minus"
        topo = "clean" if clean else "dirty"
        _put("matched", f"{charge_lab}|{p_bin}|{tx_bin}|{ty_bin}|{topo}")
    acc["q_fit"].append(q_fit)
    acc["q_truth"].append(q_truth)
    acc["delta"].append(delta)
    acc["pull"].append(pull)
    acc["source_id"].append(str(row.get("source_id") or "unknown"))
    acc["front"].append(front)
    if str(row.get("campaign")) == "smoke" and any(
        str(item["source_id"]) == str(row.get("source_id"))
        and int(item["run_id"]) == int(row.get("run_id") or 0)
        and int(item["event_id"]) == int(row.get("event_id") or 0)
        and int(item["track_index"]) == int(row.get("track_index") or 0)
        for item in config["focus_identities"]
    ):
        acc["focus_hits"].append(
            {
                "source_id": row.get("source_id"),
                "event_id": row.get("event_id"),
                "q_fit": q_fit,
                "q_truth": q_truth,
                "delta": delta,
                "pull": pull,
                "front": front,
                "clean": clean,
                "flip": flip,
                "z10": z10,
            }
        )


def evaluate_tracks(records: Iterable[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    acc = empty_accumulator()
    n = 0
    for row in records:
        n += 1
        if n % 250000 == 0:
            print(f"three_st_qp_tail_source_audit: evaluated {n} rows", file=sys.stderr, flush=True)
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


def _quantiles(values: Sequence[float], probabilities: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {f"q{int(round(p * 100)):02d}": None for p in probabilities}
    array = np.asarray(values, dtype=np.float64)
    qs = np.quantile(array, list(probabilities))
    out: dict[str, float | None] = {}
    for prob, value in zip(probabilities, qs):
        out[f"q{int(round(float(prob) * 100)):02d}"] = float(value)
    out["mean"] = float(np.mean(array))
    out["median"] = float(np.median(array))
    return out


def _rank_tail(
    acc: Mapping[str, Any],
    total: Mapping[str, Any],
    *,
    fraction: float,
) -> dict[str, Any]:
    n = len(acc["delta"])
    if n <= 0:
        return {"n": 0, "k": 0, "fraction": fraction, "rank_by": "abs_delta", **_contrib(_empty_bin(), total)}
    k = max(1, int(math.ceil(fraction * n)))
    order = np.argsort(np.abs(np.asarray(acc["delta"], dtype=np.float64)))[::-1]
    top = order[:k]
    part = _empty_bin()
    source_share: dict[str, int] = {}
    front_share: dict[str, int] = {}
    q_fit = acc["q_fit"]
    q_truth = acc["q_truth"]
    delta = acc["delta"]
    pull = acc["pull"]
    source_id = acc["source_id"]
    front = acc["front"]
    for idx in top:
        z = float(pull[idx])
        payload = {
            "q_fit": float(q_fit[idx]),
            "q_truth": float(q_truth[idx]),
            "delta": float(delta[idx]),
            "pull": z,
            "charge": float(_sign(float(q_truth[idx]))),
            "flip": _sign(float(q_fit[idx])) != _sign(float(q_truth[idx])),
            "out68": abs(z) >= 1.0,
            "out95": abs(z) >= 1.96,
            "z10": abs(z) >= 10.0,
            "z100": abs(z) >= 100.0,
        }
        _add(part, payload)
        sid = source_id[idx]
        source_share[sid] = source_share.get(sid, 0) + 1
        flab = front[idx]
        front_share[flab] = front_share.get(flab, 0) + 1
    report = _contrib(part, total)
    report.update(
        {
            "k": k,
            "fraction": fraction,
            "rank_by": "abs_delta",
            "source_counts": source_share,
            "front_counts": front_share,
        }
    )
    return report


def _catastrophic_flag(influence: Mapping[str, Any], gates: Mapping[str, Any]) -> bool | None:
    if not influence.get("applicable"):
        return None
    remain = influence.get("remaining_mu_bias")
    frac = influence.get("fraction_of_original_mu_bias")
    if remain is None:
        return None
    if abs(float(remain)) <= float(gates["catastrophic_abs_remain_max"]):
        return True
    if frac is None:
        return None
    return abs(float(frac)) < float(gates["catastrophic_remain_fraction_max"])


def _broad_flag(influence: Mapping[str, Any], mu_bias: float | None, gates: Mapping[str, Any]) -> bool | None:
    if not influence.get("applicable"):
        return None
    remain = influence.get("remaining_mu_bias")
    frac = influence.get("fraction_of_original_mu_bias")
    if remain is None or frac is None or mu_bias is None:
        return None
    return _same_sign(remain, mu_bias) and abs(float(frac)) >= float(gates["broad_remain_fraction_min"])


def _status_from_bools(supported: bool, rejected: bool) -> str:
    if supported and not rejected:
        return STATUS_SUPPORTED
    if rejected and not supported:
        return STATUS_REJECTED
    return STATUS_UNRESOLVED


def _ab_verdict(tails: Mapping[str, Any], overall: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    gates = config["gates"]
    mu_bias = overall.get("mu_bias")
    z10 = tails["z_ge_10"]["influence"]
    top = tails["top_0.001_abs_delta"]["influence"]
    cat_flags = [_catastrophic_flag(z10, gates), _catastrophic_flag(top, gates)]
    broad_flags = [_broad_flag(z10, mu_bias, gates), _broad_flag(top, mu_bias, gates)]
    known_cat = [flag for flag in cat_flags if flag is not None]
    known_broad = [flag for flag in broad_flags if flag is not None]
    cat_supported = bool(known_cat) and all(known_cat)
    broad_supported = bool(known_broad) and all(known_broad)
    if cat_supported and broad_supported:
        cat_supported = False
        broad_supported = False
    cat_rejected = bool(known_cat) and not any(known_cat)
    broad_rejected = bool(known_broad) and not any(known_broad)
    return {
        MECH_CATASTROPHIC: {
            "status": _status_from_bools(cat_supported, cat_rejected),
            "z10_catastrophic": cat_flags[0],
            "top_0p1_catastrophic": cat_flags[1],
            "z10_remain_fraction": z10.get("fraction_of_original_mu_bias"),
            "top_0p1_remain_fraction": top.get("fraction_of_original_mu_bias"),
        },
        MECH_BROAD: {
            "status": _status_from_bools(broad_supported, broad_rejected),
            "z10_broad": broad_flags[0],
            "top_0p1_broad": broad_flags[1],
            "frac_delta_neg": overall.get("frac_delta_neg"),
        },
    }


def _source_majority_charge(pack: Mapping[str, Any]) -> str | None:
    n_plus = int(pack.get("n_plus") or 0)
    n_minus = int(pack.get("n_minus") or 0)
    if n_plus > n_minus and n_plus > 0:
        return "plus"
    if n_minus > n_plus and n_minus > 0:
        return "minus"
    return None


def _source_verdict(given_source: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    gates = config["gates"]
    min_n = int(gates["min_n_stratum"])
    floor = float(gates["abs_mean_floor"])
    rel = float(gates["source_bias_rel_range_min"])
    groups: dict[str, list[tuple[str, float, int]]] = {"plus": [], "minus": []}
    for source_id, pack in given_source.items():
        if int(pack.get("n") or 0) < min_n or pack.get("mu_bias") is None:
            continue
        charge = _source_majority_charge(pack)
        if charge is None:
            continue
        groups[charge].append((source_id, float(pack["mu_bias"]), int(pack["n"])))
    disagree = False
    rel_range = None
    pairs: list[dict[str, Any]] = []
    for charge, items in groups.items():
        if len(items) < 2:
            continue
        biases = [item[1] for item in items]
        span = max(biases) - min(biases)
        scale = max(max(abs(v) for v in biases), floor)
        this_rel = span / scale
        rel_range = this_rel if rel_range is None else max(rel_range, this_rel)
        signs = {_sign(v) for v in biases if _sign(v) != 0}
        if len(signs) > 1:
            disagree = True
        if this_rel >= rel or len(signs) > 1:
            disagree = True
        pairs.append(
            {
                "charge": charge,
                "n_sources": len(items),
                "mu_bias_min": min(biases),
                "mu_bias_max": max(biases),
                "rel_range": this_rel,
                "sign_disagree": len(signs) > 1,
                "sources": [item[0] for item in items],
            }
        )
    supported = bool(pairs) and (
        disagree or (rel_range is not None and rel_range >= rel)
    )
    rejected = bool(pairs) and not supported
    return {
        "status": _status_from_bools(supported, rejected),
        "rel_range": rel_range,
        "pairs": pairs,
        "n_comparable_groups": len(pairs),
    }


def evaluate_audit(acc: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    total = acc["all"]
    overall = _mean_pack(total)
    tails = {
        "z_ge_10": _contrib(acc["z10"], total),
        "z_ge_100": _contrib(acc["z100"], total),
        "top_0.001_abs_delta": _rank_tail(acc, total, fraction=0.001),
        "top_0.0001_abs_delta": _rank_tail(acc, total, fraction=0.0001),
    }
    families = {name: _family_report(acc[name], total) for name in BIN_FAMILIES}
    ab = _ab_verdict(tails, overall, config)
    source = _source_verdict(families["given_source"], config)
    return {
        "overall": overall,
        "plus": _mean_pack(acc["plus"]),
        "minus": _mean_pack(acc["minus"]),
        "clean": _contrib(acc["clean"], total),
        "dirty": _contrib(acc["dirty"], total),
        "spd": _contrib(acc["spd"], total),
        "nspd": _contrib(acc["nspd"], total),
        "unusual_front": _contrib(acc["unusual_front"], total),
        "tails": tails,
        "quantiles": {
            "q_fit": _quantiles(acc["q_fit"], config["gates"]["quantile_probabilities"]),
            "q_truth": _quantiles(acc["q_truth"], config["gates"]["quantile_probabilities"]),
            "delta": _quantiles(acc["delta"], config["gates"]["quantile_probabilities"]),
            "pull": _quantiles(acc["pull"], config["gates"]["quantile_probabilities"]),
        },
        **families,
        "ab_verdict": ab,
        "source_verdict": source,
        "focus_hits": list(acc["focus_hits"]),
        "n_primary": int(acc["n_primary"]),
        "n_records": int(acc["n_records"]),
        "n_flip": int(acc["n_flip"]),
        "n_clean": int(acc["n_clean"]),
        "n_nspd": int(acc["n_nspd"]),
        "n_truth_as_solution": int(acc["n_truth_as_solution"]),
        "official_mean_trimmed": False,
        "reweight_used_as_calibration": False,
        "influence_used_as_official_mean": False,
    }


def _matched_comparison(
    construction: Mapping[str, Any],
    validation: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = config["gates"]
    min_n = int(gates["min_n_stratum"])
    raw_c = _mean_pack(construction["all"])
    raw_v = _mean_pack(validation["all"])
    raw_gap = None
    if raw_c["mu_bias"] is not None and raw_v["mu_bias"] is not None:
        raw_gap = float(raw_v["mu_bias"]) - float(raw_c["mu_bias"])
    cells = set(construction["matched"]) | set(validation["matched"])
    num = 0.0
    den = 0.0
    n_used = 0
    n_dropped = 0
    n_sign_disagree = 0
    used: list[dict[str, Any]] = []
    for cell in sorted(cells):
        left = construction["matched"].get(cell, _empty_bin())
        right = validation["matched"].get(cell, _empty_bin())
        n_c = int(left["n"])
        n_v = int(right["n"])
        if n_c < min_n or n_v < min_n:
            n_dropped += 1
            continue
        mu_c = float(left["sum_delta"]) / n_c
        mu_v = float(right["sum_delta"]) / n_v
        weight = float(min(n_c, n_v))
        gap = mu_v - mu_c
        num += weight * gap
        den += weight
        n_used += 1
        if _sign(mu_c) != 0 and _sign(mu_v) != 0 and _sign(mu_c) != _sign(mu_v):
            n_sign_disagree += 1
        used.append(
            {
                "cell": cell,
                "n_construction": n_c,
                "n_validation": n_v,
                "mu_bias_construction": mu_c,
                "mu_bias_validation": mu_v,
                "gap": gap,
            }
        )
    matched_gap = num / den if den > 0.0 else None
    floor = float(gates["composition_raw_gap_floor"])
    if matched_gap is None or raw_gap is None or abs(raw_gap) < floor:
        status = STATUS_UNRESOLVED
    elif abs(matched_gap) <= float(gates["composition_matched_gap_frac_max"]) * abs(raw_gap):
        status = STATUS_SUPPORTED
    elif abs(matched_gap) >= float(gates["composition_reject_gap_frac_min"]) * abs(raw_gap):
        status = STATUS_REJECTED
    else:
        status = STATUS_UNRESOLVED
    plus_c = _mean_pack(construction["plus"])
    plus_v = _mean_pack(validation["plus"])
    minus_c = _mean_pack(construction["minus"])
    minus_v = _mean_pack(validation["minus"])
    charge_gaps = []
    if plus_c["mu_bias"] is not None and plus_v["mu_bias"] is not None:
        charge_gaps.append(float(plus_v["mu_bias"]) - float(plus_c["mu_bias"]))
    if minus_c["mu_bias"] is not None and minus_v["mu_bias"] is not None:
        charge_gaps.append(float(minus_v["mu_bias"]) - float(minus_c["mu_bias"]))
    return {
        "diagnostic_only": True,
        "calibration_population": False,
        "status": status,
        "raw_gap": raw_gap,
        "matched_gap": matched_gap,
        "matched_over_raw": _ratio(matched_gap, raw_gap),
        "n_cells_used": n_used,
        "n_cells_dropped": n_dropped,
        "n_cells_sign_disagree": n_sign_disagree,
        "charge_only_mean_gap": float(np.mean(charge_gaps)) if charge_gaps else None,
        "cells": used[:40],
    }


def _refit_verdict(report: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    gates = config["gates"]
    thresh = float(gates["refit_proxy_share_min"])
    unusual = report["unusual_front"]
    z10 = report["tails"]["z_ge_10"]
    top = report["tails"]["top_0.001_abs_delta"]
    shares = {
        "unusual_share_sum_delta": unusual.get("share_sum_delta"),
        "unusual_share_sumsq_pull": unusual.get("share_sumsq_pull"),
        "unusual_share_n": unusual.get("share_n"),
        "z10_share_sum_delta": z10.get("share_sum_delta"),
        "top_0p1_unusual_front_fraction": None,
    }
    top_front = top.get("front_counts") or {}
    top_k = int(top.get("k") or 0)
    unusual_top = sum(count for label, count in top_front.items() if label not in {USUAL_FRONT, "front_unknown"})
    if top_k > 0:
        shares["top_0p1_unusual_front_fraction"] = unusual_top / top_k
    candidates = [
        abs(float(shares["unusual_share_sum_delta"]))
        if shares["unusual_share_sum_delta"] is not None
        else None,
        shares["unusual_share_sumsq_pull"],
        shares["top_0p1_unusual_front_fraction"],
    ]
    known = [value for value in candidates if value is not None]
    supported = bool(known) and any(float(value) >= thresh for value in known)
    rejected = bool(known) and all(float(value) < thresh for value in known)
    return {
        "status": _status_from_bools(supported, rejected),
        "existing_fields_distinguish_ckf_vs_refit": False,
        "proxy": "front_z_not_near_s1",
        "shares": shares,
        "near_s1": report["given_front"].get(USUAL_FRONT, {}),
    }


def _repeatable_bias(construction: Mapping[str, Any], validation: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    floor = float(config["gates"]["repeatable_abs_bias_min"])
    mu_c = construction["overall"].get("mu_bias")
    mu_v = validation["overall"].get("mu_bias")
    if mu_c is None or mu_v is None:
        return {"repeatable": False, "same_sign": False, "reason": "missing_mu_bias"}
    same = _same_sign(mu_c, mu_v)
    strong = abs(float(mu_c)) >= floor and abs(float(mu_v)) >= floor
    return {
        "repeatable": bool(same and strong),
        "same_sign": same,
        "both_above_floor": strong,
        "construction_mu_bias": mu_c,
        "validation_mu_bias": mu_v,
    }


def diagnostic_export_contract(
    *,
    authorized: bool,
    reason: str,
    shares: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "authorized": authorized,
        "max_events": 20,
        "reason": reason,
        "existing_fields_distinguish_ckf_vs_refit": False,
        "proxy_shares": dict(shares),
        "required_fields": [
            "kf_refit_succeeded",
            "q_over_p_before_refit",
            "q_over_p_after_refit",
            "covariance_before_refit",
            "covariance_after_refit",
            "front_surface_type",
            "front_surface_z",
            "track_state_provenance",
        ],
        "must_not": [
            "change fitter, seed, hit selection, geometry, or covariance scale",
            "use truth to change the fit",
            "flip WB119 or authorize S3",
            "trim or redefine the official mean",
        ],
        "reuse_focus_identities_plus_extreme_tail_rows": True,
    }


def _identity_ok(pack: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    residual = pack.get("identity_residual")
    if residual is None:
        return False
    return abs(float(residual)) <= float(config["gates"]["identity_abs_max"])


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


def _pick_official(verdicts: Mapping[str, Mapping[str, Any]]) -> str:
    supported = [name for name, item in verdicts.items() if item.get("status") == STATUS_SUPPORTED]
    if len(supported) == 1:
        return supported[0]
    return MECH_MIXED


def decide(
    construction: Mapping[str, Any],
    validation: Mapping[str, Any],
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
    elif inherited.get("workbook_122", {}).get("decision") != "three_st_qp_mean_decomposition_recorded":
        mechanism = "s2f_not_inherited"
    elif not dumps_materialized:
        mechanism = "calibration_dump_not_materialized"
    elif int(pooled["n_truth_as_solution"]):
        mechanism = "truth_used_as_fit_or_solution"
    elif not _denominator_ok(pooled, campaign, config):
        mechanism = "wb119_denominator_drift"
    else:
        contract = "PASS"
    construction_rep = evaluate_audit(construction["tracks"], config)
    validation_rep = evaluate_audit(validation["tracks"], config)
    pooled_rep = evaluate_audit(pooled, config)
    if contract == "PASS" and not (
        _identity_ok(construction_rep["overall"], config)
        and _identity_ok(validation_rep["overall"], config)
        and _identity_ok(pooled_rep["overall"], config)
    ):
        contract = "FAIL"
        mechanism = "mean_identity_failed"
    matched = _matched_comparison(construction["tracks"], validation["tracks"], config)
    refit = _refit_verdict(validation_rep, config)
    repeatable = _repeatable_bias(construction_rep, validation_rep, config)
    dump_share = refit["shares"]
    dump_thresh = float(config["gates"]["dump_share_min"])
    dump_known = [
        value
        for value in (
            dump_share.get("unusual_share_sum_delta"),
            dump_share.get("unusual_share_sumsq_pull"),
            dump_share.get("top_0p1_unusual_front_fraction"),
        )
        if value is not None
    ]
    dump_could_explain = bool(dump_known) and any(abs(float(value)) >= dump_thresh for value in dump_known)
    dump_authorized = bool(
        dump_could_explain and refit["existing_fields_distinguish_ckf_vs_refit"] is False
    )
    if dump_authorized:
        dump_reason = (
            "unusual front-z proxy holds a major share of validation bias or "
            "tails, and existing fields cannot tag CKF-only vs KalmanFitter refit"
        )
    elif dump_could_explain:
        dump_reason = "proxy share is large but the existing-field gap was not proven"
    else:
        dump_reason = (
            "existing fields cannot tag CKF-only vs refit, but no current proxy "
            "shows that the tag would own the validation bias or tail share"
        )
    export = diagnostic_export_contract(
        authorized=dump_authorized,
        reason=dump_reason,
        shares=dump_share,
    )
    verdicts = {
        MECH_CATASTROPHIC: validation_rep["ab_verdict"][MECH_CATASTROPHIC],
        MECH_BROAD: validation_rep["ab_verdict"][MECH_BROAD],
        MECH_SOURCE: pooled_rep["source_verdict"],
        MECH_COMPOSITION: {"status": matched["status"], "matched_over_raw": matched.get("matched_over_raw")},
        MECH_REFIT: refit,
    }
    if contract != "PASS":
        decision = DECISION_NOT
        verdict = "FAIL"
        diagnosis_verdict = "FAIL"
        official_mechanism = None
    elif int(pooled_rep["n_primary"]) < int(config["gates"]["min_n_mechanism_verdict"]):
        decision = DECISION_CONTRACT
        verdict = "PASS"
        diagnosis_verdict = "INCONCLUSIVE"
        official_mechanism = None
        mechanism = None
    else:
        decision = DECISION_RECORDED
        verdict = "PASS"
        diagnosis_verdict = "RECORDED"
        official_mechanism = _pick_official(verdicts)
        mechanism = official_mechanism
    return {
        "kind": "three_st_qp_tail_source_contract",
        "task": TASK,
        "workbook": WORKBOOK,
        "campaign": campaign,
        "verdict": verdict,
        "decision": decision,
        "mechanism": mechanism,
        "contract_verdict": contract,
        "diagnosis_verdict": diagnosis_verdict,
        "official_mechanism": official_mechanism,
        "structured_verdicts": {
            name: {key: value for key, value in item.items() if key != "near_s1"}
            for name, item in verdicts.items()
        },
        "repeatable_reconstruction_bias": repeatable,
        "matched_comparison": matched,
        "diagnostic_export_contract": export,
        "three_st_qp_trusted_observable": False,
        "residual_conditional_authorized": False,
        "s2_flipped_to_pass": False,
        "sign_flip_tracks_dropped": False,
        "official_mean_trimmed": False,
        "reweight_used_as_calibration": False,
        "influence_used_as_official_mean": False,
        "new_reconstruction_dump_authorized": dump_authorized,
        "cannot_claim_ift_residual_or_weak_mode": True,
        "construction": {
            "n_records": construction["tracks"]["n_records"],
            "n_primary": construction["tracks"]["n_primary"],
            "n_flip": construction["tracks"]["n_flip"],
            "report": construction_rep,
        },
        "validation": {
            "n_records": validation["tracks"]["n_records"],
            "n_primary": validation["tracks"]["n_primary"],
            "n_flip": validation["tracks"]["n_flip"],
            "report": validation_rep,
        },
        "pooled": pooled_rep,
        "official_quantities": official_quantities(),
    }
