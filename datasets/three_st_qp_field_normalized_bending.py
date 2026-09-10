"""Yasu-S2K: field-integral-normalized 3ST bending response closure.

Promotes the frozen WB125/WB126 YZ bending_raw to a provisional
qp_bending_proxy using a fit-independent S1→S2→S3 measurement-chord
integral of the pinned FaserFieldTable_v2.  Fitted q/p never enters
construction.  Truth q/p is used only after construction, and only as a
source-disjoint MC reference.  No free scale is fitted to force slope=1.
Does not flip WB119, open Stage 3, or mark the proxy as trusted momentum.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from datasets.faser_field_table import (
    GEV_TO_MEV,
    K_GEV_PER_TM,
    FaserFieldTable,
    FaserFieldTableError,
    lock_field_unit_sign_contract,
    qp_bending_proxy_per_mev,
    unit_sign_source_contract,
)
from datasets.three_st_qp_calibration import (
    SOURCE_COLLECTION_NAME,
    iter_jsonl,
    verify_pinned_calypso_sources,
)
from datasets.three_st_qp_measurement_bending import (
    THREE_STATION_IDS,
    centroids_from_row,
    construct_bending,
)
from datasets.three_st_qp_measurement_bending_batch import (
    assign_bin,
    decorate_track,
    slim_track_row,
    _split_of,
    _truth_charge,
)

SCHEMA_VERSION = "three-st-qp-field-normalized-bending-v1"
DEFAULT_CONFIG = "configs/three_st_qp_field_normalized_bending_v1.yaml"
TASK = "YASU-S2K"
WORKBOOK = 127

DECISION_CONTRACT = "three_st_qp_field_normalized_bending_contract_established"
DECISION_RECORDED = "three_st_qp_field_normalized_bending_recorded"
DECISION_NOT = "three_st_qp_field_normalized_bending_not_established"

STATUS_SUPPORTED = "supported"
STATUS_REJECTED = "rejected"
STATUS_UNRESOLVED = "unresolved"

VERDICT_FIELD = "field_unit_sign_contract"
VERDICT_RESPONSE = "field_normalized_bending_response"
VERDICT_TRANSFER = "cross_split_transferability"
VERDICT_HIGH_P = "high_p_proxy_information"
VERDICT_SELECT = "bending_availability_selection"

REQUIRED_INHERITANCE = (
    "workbook_117",
    "workbook_118",
    "workbook_119",
    "workbook_120",
    "workbook_121",
    "workbook_122",
    "workbook_123",
    "workbook_124",
    "workbook_125",
    "workbook_126",
)

N_MOT_LABELS = ("n_mot_0_5", "n_mot_6_11", "n_mot_12_17", "n_mot_18", "n_mot_gt_18")


class ThreeStQpFieldNormalizedBendingError(ValueError):
    """Raised when the S2K contract is illegal."""


def refuse_flip_s2() -> None:
    raise ThreeStQpFieldNormalizedBendingError(
        "S2K cannot set three_st_qp_trusted_observable true"
    )


def refuse_residual_conditional() -> None:
    raise ThreeStQpFieldNormalizedBendingError(
        "E[r_IFT|.] is Stage 3; S2K does not open it"
    )


def refuse_fitted_qp() -> None:
    raise ThreeStQpFieldNormalizedBendingError(
        "fitted q/p must not enter qp_bending_proxy construction"
    )


def refuse_truth_in_construction() -> None:
    raise ThreeStQpFieldNormalizedBendingError(
        "truth q/p must not enter qp_bending_proxy construction"
    )


def refuse_empirical_scale() -> None:
    raise ThreeStQpFieldNormalizedBendingError(
        "S2K must not fit a free scale from the same truth batch"
    )


def refuse_seed_jacobian() -> None:
    raise ThreeStQpFieldNormalizedBendingError(
        "1.13 T m, 0.55 T, and handwritten 0.3 are not an official Jacobian"
    )


def refuse_invent_sigma() -> None:
    raise ThreeStQpFieldNormalizedBendingError(
        "S2K must not invent sigma_bending or claim pull/coverage"
    )


def refuse_two_station_surrogate() -> None:
    raise ThreeStQpFieldNormalizedBendingError(
        "S2K must not fill missing 3ST bending with a two-station surrogate"
    )


def conversion_chain() -> dict[str, Any]:
    return {
        "calypso_git_sha": "40892527e9c65409afd2378a2abfc25ddbddac03",
        "inherited_bending_raw": "workbook_125/126 YZ atan(ty12)-atan(ty23), unchanged",
        "path": "piecewise_linear S1->S2->S3 measurement centroids",
        "integral": "I = int (Bx dz - Bz dx) along that path, B in tesla, dl in metres",
        "k_gev_per_tm": K_GEV_PER_TM,
        "k_kind": "pdg_ec_in_GeV_per_T_m",
        "qp_per_gev": "-bending_raw / (k * I)",
        "qp_per_mev": "qp_per_gev / 1000",
        "dipole_scale": 1.0,
        "dipole_scale_kind": "FaserFieldCacheCondAlg default UseDipoScale",
        "forbidden_jacobian": "1.13 Tm, 0.55 T, handwritten 0.3",
        "truth_role": "source_disjoint_mc_calibration_reference_after_construction",
        "empirical_scale": False,
        "point_estimator_only": True,
        "trusted_momentum": False,
        "source_contract": unit_sign_source_contract(),
    }


def official_quantities() -> dict[str, str]:
    return {
        "bending_raw": "frozen WB125 YZ observable; construction input",
        "I_yz_tm": "per-track path integral int(Bx dz - Bz dx) in tesla*metre",
        "qp_bending_proxy_per_mev": "provisional field-normalized curvature; not trusted momentum",
        "q_over_p_truth_s1_per_mev": "source-disjoint MC reference only; after construction",
        "q_over_p_fit_per_mev": "comparison only; never a construction input",
        "sigma_bending": "absent; stereo/space-point Jacobian not in the dump",
        "bending_available": "whether frozen 3ST centroids construct bending_raw",
    }


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ThreeStQpFieldNormalizedBendingError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise ThreeStQpFieldNormalizedBendingError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise ThreeStQpFieldNormalizedBendingError("workbook must be 127")
    if bool(config.get("three_st_qp_trusted_observable", True)):
        raise ThreeStQpFieldNormalizedBendingError("three_st_qp_trusted_observable must stay false")
    if bool(config.get("residual_conditional_authorized", True)):
        raise ThreeStQpFieldNormalizedBendingError("residual_conditional_authorized must stay false")
    if bool(config.get("official_qp_like_jacobian_authorized", True)):
        raise ThreeStQpFieldNormalizedBendingError("official_qp_like_jacobian_authorized must stay false")
    if bool(config.get("measurement_uncertainty_propagated", True)):
        raise ThreeStQpFieldNormalizedBendingError("do not claim a propagated sigma_bending")
    if str(config.get("observable", {}).get("name")) != "qp_bending_proxy":
        raise ThreeStQpFieldNormalizedBendingError("primary constructed observable must be qp_bending_proxy")
    if str(config.get("bending_plane")) != "YZ":
        raise ThreeStQpFieldNormalizedBendingError("bending_plane must stay YZ")
    if float(config.get("conversion", {}).get("k_gev_per_tm")) != K_GEV_PER_TM:
        raise ThreeStQpFieldNormalizedBendingError("k_gev_per_tm must stay the PDG 0.299792458")
    if bool(config.get("conversion", {}).get("fit_free_scale_from_truth", True)):
        refuse_empirical_scale()
    if str(config.get("path", {}).get("name")) != "s1_s2_s3_measurement_chords":
        raise ThreeStQpFieldNormalizedBendingError("path must stay s1_s2_s3_measurement_chords")
    return dict(config)


def _expect_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise ThreeStQpFieldNormalizedBendingError(f"{label} hash mismatch: {digest}")


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
            raise ThreeStQpFieldNormalizedBendingError(f"{workbook} decision must stay frozen")
        inherited[workbook] = {
            "decision": decision["decision"],
            "config_sha256": spec["config_sha256"],
            "decision_sha256": spec["decision_sha256"],
        }
    if inherited["workbook_119"]["decision"] != "three_st_qp_calibration_not_established":
        raise ThreeStQpFieldNormalizedBendingError("WB119 must remain not_established")
    if inherited["workbook_125"]["decision"] != "three_st_qp_measurement_bending_contract_established":
        raise ThreeStQpFieldNormalizedBendingError("WB125 bending contract must remain established")
    if inherited["workbook_126"]["decision"] != "three_st_qp_measurement_bending_batch_recorded":
        raise ThreeStQpFieldNormalizedBendingError("WB126 batch must remain recorded")
    pins = verify_pinned_calypso_sources(config)
    if not pins.get("all_match"):
        raise ThreeStQpFieldNormalizedBendingError("pinned Calypso sources must match config hashes")
    inherited["pinned_calypso_sources"] = pins
    return inherited


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _sign(value: float | None) -> int:
    if value is None:
        return 0
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 3:
        return None
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.size != right.size or left.size == 0:
        return None
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    rx = np.argsort(np.argsort(left)).astype(np.float64)
    ry = np.argsort(np.argsort(right)).astype(np.float64)
    value = float(np.corrcoef(rx, ry)[0, 1])
    return value if np.isfinite(value) else None


def _ols_with_intercept(x: Sequence[float], y: Sequence[float]) -> dict[str, float | None]:
    left = np.asarray(list(x), dtype=np.float64)
    right = np.asarray(list(y), dtype=np.float64)
    if left.size < 2 or left.size != right.size or float(np.std(left)) == 0.0:
        return {"slope": None, "intercept": None, "rms_x": None, "n": int(left.size)}
    slope, intercept = np.polyfit(left, right, 1)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "rms_x": float(np.std(left)),
        "n": int(left.size),
    }


def _n_mot_label(n_mot: int) -> str:
    if n_mot <= 5:
        return N_MOT_LABELS[0]
    if n_mot <= 11:
        return N_MOT_LABELS[1]
    if n_mot <= 17:
        return N_MOT_LABELS[2]
    if n_mot == 18:
        return N_MOT_LABELS[3]
    return N_MOT_LABELS[4]


def _truth_tx_ty_s1(row: Mapping[str, Any]) -> tuple[float | None, float | None]:
    stations = row.get("truth_station_momenta")
    if isinstance(stations, Sequence):
        for item in stations:
            if isinstance(item, Mapping) and int(item.get("station", -1)) == 1:
                return _finite(item.get("tx")), _finite(item.get("ty"))
    return None, None


def construct_qp_bending_proxy(
    bending_raw: float,
    integral_yz_tm: float,
    config: Mapping[str, Any],
    *,
    q_over_p_fit: Any = None,
    q_over_p_truth: Any = None,
    empirical_scale: Any = None,
) -> dict[str, Any]:
    if q_over_p_fit is not None:
        refuse_fitted_qp()
    if q_over_p_truth is not None:
        refuse_truth_in_construction()
    if empirical_scale is not None:
        refuse_empirical_scale()
    min_abs = float(config["gates"]["min_abs_integral_tm"])
    if not np.isfinite(integral_yz_tm) or abs(float(integral_yz_tm)) < min_abs:
        return {
            "qp_bending_proxy_per_mev": None,
            "I_yz_tm": float(integral_yz_tm) if np.isfinite(integral_yz_tm) else None,
            "constructable_proxy": False,
            "reason": "integral_below_floor",
            "used_fitted_q_over_p": False,
            "used_truth_q_over_p": False,
            "empirical_scale_applied": False,
            "k_gev_per_tm": K_GEV_PER_TM,
        }
    proxy = qp_bending_proxy_per_mev(float(bending_raw), float(integral_yz_tm))
    return {
        "qp_bending_proxy_per_mev": proxy,
        "I_yz_tm": float(integral_yz_tm),
        "constructable_proxy": True,
        "reason": None,
        "used_fitted_q_over_p": False,
        "used_truth_q_over_p": False,
        "empirical_scale_applied": False,
        "k_gev_per_tm": K_GEV_PER_TM,
        "mev_per_gev": GEV_TO_MEV,
    }


def load_official_field(config: Mapping[str, Any]) -> FaserFieldTable:
    field_cfg = config.get("field") or {}
    path = field_cfg.get("resolved_table_path")
    scale = float(field_cfg.get("dipole_scale", 1.0))
    if scale != 1.0:
        raise ThreeStQpFieldNormalizedBendingError(
            "dipole_scale must stay the Calypso default 1.0 in this stage"
        )
    expected = field_cfg.get("resolved_table_sha256")
    if expected and path:
        digest = sha256_file(Path(str(path)))
        if digest != expected:
            raise ThreeStQpFieldNormalizedBendingError(
                f"pinned field table hash mismatch: {digest}"
            )
    return FaserFieldTable.from_root(path, dipole_scale=scale)


def _empty_proxy_acc() -> dict[str, Any]:
    return {
        "n": 0,
        "n_constructable_bending": 0,
        "n_constructable_proxy": 0,
        "n_with_truth": 0,
        "n_sign": 0,
        "n_sign_agree": 0,
        "n_proxy_flip": 0,
        "charge_odd_sum": 0.0,
        "charge_odd_n": 0,
        "charge_even_sum": 0.0,
        "charge_even_n": 0,
        "residual_odd_sum": 0.0,
        "residual_even_sum": 0.0,
        "residual_n": 0,
        "proxy": [],
        "truth": [],
        "p_bins": defaultdict(lambda: {
            "n_sign": 0,
            "n_agree": 0,
            "n_truth": 0,
            "residual_sum": 0.0,
            "proxy": [],
            "truth": [],
        }),
        "tx_bins": defaultdict(lambda: {"n_sign": 0, "n_agree": 0, "n_truth": 0, "residual_sum": 0.0}),
        "ty_bins": defaultdict(lambda: {"n_sign": 0, "n_agree": 0, "n_truth": 0, "residual_sum": 0.0}),
    }


def _observe_proxy(acc: dict[str, Any], row: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    acc["n"] += 1
    if not row.get("constructable") or _finite(row.get("bending_raw")) is None:
        return
    acc["n_constructable_bending"] += 1
    proxy = _finite(row.get("qp_bending_proxy_per_mev"))
    if proxy is None or not row.get("constructable_proxy"):
        return
    acc["n_constructable_proxy"] += 1
    charge = _truth_charge(row)
    qp_truth = _finite(row.get("q_over_p_truth_s1_per_mev"))
    p_truth = _finite(row.get("p_truth_s1_mev"))
    t_x = _finite(row.get("t_x_chord"))
    t_y = _finite(row.get("t_y_chord"))
    if charge is None:
        return
    acc["n_with_truth"] += 1
    acc["charge_odd_sum"] += proxy * _sign(charge)
    acc["charge_odd_n"] += 1
    acc["charge_even_sum"] += proxy
    acc["charge_even_n"] += 1
    if qp_truth is not None:
        residual = proxy - qp_truth
        acc["residual_odd_sum"] += residual * _sign(charge)
        acc["residual_even_sum"] += residual
        acc["residual_n"] += 1
        acc["proxy"].append(proxy)
        acc["truth"].append(qp_truth)
    bins = config["binning"]
    p_bin = assign_bin(p_truth, bins["p_truth_edges_mev"], bins["p_truth_labels"])
    tx_bin = assign_bin(t_x, bins["tx_edges"], bins["tx_labels"])
    ty_bin = assign_bin(t_y, bins["ty_edges"], bins["ty_labels"])
    if _sign(proxy) != 0 and _sign(charge) != 0:
        acc["n_sign"] += 1
        agree = _sign(proxy) == _sign(charge)
        if agree:
            acc["n_sign_agree"] += 1
        else:
            acc["n_proxy_flip"] += 1
        acc["p_bins"][p_bin]["n_sign"] += 1
        acc["p_bins"][p_bin]["n_agree"] += int(agree)
        acc["tx_bins"][tx_bin]["n_sign"] += 1
        acc["tx_bins"][tx_bin]["n_agree"] += int(agree)
        acc["ty_bins"][ty_bin]["n_sign"] += 1
        acc["ty_bins"][ty_bin]["n_agree"] += int(agree)
    if qp_truth is not None:
        residual = proxy - qp_truth
        acc["p_bins"][p_bin]["n_truth"] += 1
        acc["p_bins"][p_bin]["residual_sum"] += residual
        acc["p_bins"][p_bin]["proxy"].append(proxy)
        acc["p_bins"][p_bin]["truth"].append(qp_truth)
        acc["tx_bins"][tx_bin]["n_truth"] += 1
        acc["tx_bins"][tx_bin]["residual_sum"] += residual
        acc["ty_bins"][ty_bin]["n_truth"] += 1
        acc["ty_bins"][ty_bin]["residual_sum"] += residual


def _finalize_proxy(acc: dict[str, Any]) -> dict[str, Any]:
    def rate(num: int, den: int) -> float | None:
        return (num / den) if den else None

    fit = _ols_with_intercept(acc["truth"], acc["proxy"])
    p_bins = {}
    for label, item in acc["p_bins"].items():
        ols = _ols_with_intercept(item["truth"], item["proxy"])
        p_bins[label] = {
            "n_sign": item["n_sign"],
            "sign_agree": rate(item["n_agree"], item["n_sign"]),
            "n_truth": item["n_truth"],
            "mean_residual": (item["residual_sum"] / item["n_truth"]) if item["n_truth"] else None,
            "slope": ols["slope"],
            "intercept": ols["intercept"],
        }
    return {
        "n": acc["n"],
        "n_constructable_bending": acc["n_constructable_bending"],
        "n_constructable_proxy": acc["n_constructable_proxy"],
        "n_with_truth": acc["n_with_truth"],
        "n_sign": acc["n_sign"],
        "n_sign_agree": acc["n_sign_agree"],
        "sign_agree": rate(acc["n_sign_agree"], acc["n_sign"]),
        "n_proxy_flip": acc["n_proxy_flip"],
        "p_proxy_flip": rate(acc["n_proxy_flip"], acc["n_sign"]),
        "spearman_proxy_vs_qp_truth": _spearman(acc["proxy"], acc["truth"]),
        "ols_proxy_vs_qp_truth": fit,
        "charge_odd_mean": (acc["charge_odd_sum"] / acc["charge_odd_n"]) if acc["charge_odd_n"] else None,
        "charge_even_mean": (acc["charge_even_sum"] / acc["charge_even_n"]) if acc["charge_even_n"] else None,
        "residual_charge_odd_mean": (
            acc["residual_odd_sum"] / acc["residual_n"] if acc["residual_n"] else None
        ),
        "residual_charge_even_mean": (
            acc["residual_even_sum"] / acc["residual_n"] if acc["residual_n"] else None
        ),
        "p_bins": p_bins,
        "tx_bins": {
            label: {
                "n_sign": item["n_sign"],
                "sign_agree": rate(item["n_agree"], item["n_sign"]),
                "mean_residual": (item["residual_sum"] / item["n_truth"]) if item["n_truth"] else None,
            }
            for label, item in acc["tx_bins"].items()
        },
        "ty_bins": {
            label: {
                "n_sign": item["n_sign"],
                "sign_agree": rate(item["n_agree"], item["n_sign"]),
                "mean_residual": (item["residual_sum"] / item["n_truth"]) if item["n_truth"] else None,
            }
            for label, item in acc["ty_bins"].items()
        },
        "sigma_b_used": False,
        "point_estimator_only": True,
        "empirical_scale_applied": False,
    }


def _empty_missing() -> dict[str, Any]:
    return {
        "n": 0,
        "n_fit": 0,
        "n_fit_flip": 0,
        "n_fit_flip_no_bending": 0,
        "n_fit_ok_no_bending": 0,
        "n_no_bending": 0,
        "n_bending": 0,
        "station_patterns": defaultdict(int),
        "n_mot": defaultdict(int),
        "p_bins": defaultdict(int),
        "sources": defaultdict(int),
        "truth_tx_ty_available": 0,
        "truth_tx_ty_unavailable": 0,
        "reasons": defaultdict(int),
    }


def _observe_missing(
    acc: dict[str, Any],
    row: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    fit_flip_only: bool,
) -> None:
    acc["n"] += 1
    qp_fit = _finite(row.get("q_over_p_fit_per_mev"))
    qp_truth = _finite(row.get("q_over_p_truth_s1_per_mev"))
    fit_ok = qp_fit is not None and qp_truth is not None and _sign(qp_fit) != 0 and _sign(qp_truth) != 0
    is_flip = bool(fit_ok and _sign(qp_fit) != _sign(qp_truth))
    if fit_flip_only and not is_flip:
        return
    if fit_ok:
        acc["n_fit"] += 1
        if is_flip:
            acc["n_fit_flip"] += 1
    available = bool(row.get("constructable") and _finite(row.get("bending_raw")) is not None)
    if available:
        acc["n_bending"] += 1
        return
    acc["n_no_bending"] += 1
    if is_flip:
        acc["n_fit_flip_no_bending"] += 1
    elif fit_ok:
        acc["n_fit_ok_no_bending"] += 1
    if fit_flip_only and not is_flip:
        return
    stations = tuple(
        sorted({int(station) for station in (row.get("measurements_on_track_stations") or []) if station is not None})
    )
    acc["station_patterns"][str(stations)] += 1
    acc["n_mot"][_n_mot_label(int(row.get("n_mot") or 0))] += 1
    p_bin = assign_bin(
        _finite(row.get("p_truth_s1_mev")),
        config["binning"]["p_truth_edges_mev"],
        config["binning"]["p_truth_labels"],
    )
    acc["p_bins"][p_bin] += 1
    acc["sources"][str(row.get("source_id") or "unknown")] += 1
    tx, ty = _truth_tx_ty_s1(row)
    if tx is None or ty is None:
        acc["truth_tx_ty_unavailable"] += 1
    else:
        acc["truth_tx_ty_available"] += 1
    mot = set(stations)
    if not {1, 2, 3}.issubset(mot):
        acc["reasons"]["incomplete_mot_stations"] += 1
    elif not row.get("has_three_station_centroids"):
        acc["reasons"]["centroids_missing_despite_stations"] += 1
    else:
        acc["reasons"]["construct_bending_failed"] += 1


def _finalize_missing(acc: dict[str, Any]) -> dict[str, Any]:
    n_fit = int(acc["n_fit"])
    n_flip = int(acc["n_fit_flip"])
    n_ok = n_fit - n_flip
    p_unavail_flip = (acc["n_fit_flip_no_bending"] / n_flip) if n_flip else None
    p_unavail_ok = (acc["n_fit_ok_no_bending"] / n_ok) if n_ok else None
    gap = None
    if p_unavail_flip is not None and p_unavail_ok is not None:
        gap = abs(p_unavail_flip - p_unavail_ok)
    return {
        "n": acc["n"],
        "n_fit": n_fit,
        "n_fit_flip": n_flip,
        "n_fit_flip_no_bending": int(acc["n_fit_flip_no_bending"]),
        "n_fit_ok_no_bending": int(acc["n_fit_ok_no_bending"]),
        "n_no_bending": int(acc["n_no_bending"]),
        "n_bending": int(acc["n_bending"]),
        "p_unavailable_given_fit_flip": p_unavail_flip,
        "p_unavailable_given_fit_ok": p_unavail_ok,
        "availability_rate_gap": gap,
        "station_patterns": dict(sorted(acc["station_patterns"].items(), key=lambda kv: -kv[1])),
        "n_mot": dict(acc["n_mot"]),
        "p_bins": dict(acc["p_bins"]),
        "sources": dict(acc["sources"]),
        "truth_tx_ty_available": int(acc["truth_tx_ty_available"]),
        "truth_tx_ty_unavailable": int(acc["truth_tx_ty_unavailable"]),
        "reasons": dict(acc["reasons"]),
        "two_station_surrogate_used": False,
        "classified_as_measurement_failure": False,
        "classified_as_reconstruction_failure": False,
    }


def match_residual_stability(
    cells: Mapping[tuple[str, ...], Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = config["gates"]
    min_cell = int(gates["min_matched_cell"])
    floor = float(gates["abs_mean_floor"])
    weight = 0.0
    weighted_c = 0.0
    weighted_v = 0.0
    matched_c = 0.0
    matched_v = 0.0
    n_cells = 0
    n_tracks_c = 0
    n_tracks_v = 0
    for item in cells.values():
        n_c = int(item.get("n_construction") or 0)
        n_v = int(item.get("n_validation") or 0)
        n_tracks_c += n_c
        n_tracks_v += n_v
        if n_c < min_cell or n_v < min_cell:
            continue
        mean_c = item["sum_construction"] / n_c
        mean_v = item["sum_validation"] / n_v
        w = float(min(n_c, n_v))
        weighted_c += mean_c * w
        weighted_v += mean_v * w
        weight += w
        matched_c += n_c
        matched_v += n_v
        n_cells += 1
    total = n_tracks_c + n_tracks_v
    matched_frac = ((matched_c + matched_v) / total) if total else None
    if weight <= 0.0:
        return {
            "n_matched_cells": 0,
            "matched_fraction": matched_frac,
            "construction_mean_residual": None,
            "validation_mean_residual": None,
            "rel_range": None,
            "sign_disagree": None,
            "enough": False,
        }
    mean_c = weighted_c / weight
    mean_v = weighted_v / weight
    rel = abs(mean_c - mean_v) / max(abs(mean_c), abs(mean_v), floor)
    return {
        "n_matched_cells": n_cells,
        "matched_fraction": matched_frac,
        "n_matched_construction": int(matched_c),
        "n_matched_validation": int(matched_v),
        "construction_mean_residual": mean_c,
        "validation_mean_residual": mean_v,
        "rel_range": rel,
        "sign_disagree": _sign(mean_c) != 0 and _sign(mean_v) != 0 and _sign(mean_c) != _sign(mean_v),
        "enough": (matched_frac or 0.0) >= float(gates["min_matched_fraction"]),
    }


def _apply_integrals(
    pending: list[dict[str, Any]],
    integrals: np.ndarray,
    config: Mapping[str, Any],
    groups: Mapping[str, dict[str, Any]],
    match_cells: dict[tuple[str, ...], dict[str, float]],
    source_stats: dict[str, dict[str, Any]],
) -> None:
    for row, integral in zip(pending, integrals):
        built = construct_qp_bending_proxy(float(row["bending_raw"]), float(integral), config)
        row.update(built)
        _observe_proxy(groups["all"], row, config)
        if row.get("clean_18hit"):
            _observe_proxy(groups["clean_18hit"], row, config)
        if row.get("complete_three_st"):
            _observe_proxy(groups["complete_three_st"], row, config)
        if row.get("dirty"):
            _observe_proxy(groups["dirty"], row, config)
        split = _split_of(str(row.get("source_id") or ""))
        if row.get("clean_18hit"):
            _observe_proxy(groups[split], row, config)
        src = source_stats[str(row.get("source_id") or "unknown")]
        proxy = _finite(row.get("qp_bending_proxy_per_mev"))
        charge = _truth_charge(row)
        qp_truth = _finite(row.get("q_over_p_truth_s1_per_mev"))
        src["n"] += 1
        if proxy is not None and charge is not None and row.get("constructable_proxy"):
            if _sign(proxy) != 0 and _sign(charge) != 0:
                src["n_sign"] += 1
                src["n_sign_agree"] += int(_sign(proxy) == _sign(charge))
            if qp_truth is not None:
                src["proxy"].append(proxy)
                src["truth"].append(qp_truth)
        if not (row.get("clean_18hit") and row.get("constructable_proxy") and proxy is not None and qp_truth is not None):
            continue
        if charge is None:
            continue
        bins = config["binning"]
        key = (
            "plus" if _sign(charge) > 0 else "minus",
            assign_bin(_finite(row.get("p_truth_s1_mev")), bins["p_truth_edges_mev"], bins["p_truth_labels"]),
            assign_bin(_finite(row.get("t_x_chord")), bins["tx_edges"], bins["tx_labels"]),
            assign_bin(_finite(row.get("t_y_chord")), bins["ty_edges"], bins["ty_labels"]),
            "clean_18hit",
        )
        residual = proxy - qp_truth
        if split == "construction":
            match_cells[key]["n_construction"] += 1
            match_cells[key]["sum_construction"] += residual
        elif split == "validation":
            match_cells[key]["n_validation"] += 1
            match_cells[key]["sum_validation"] += residual


def evaluate_rows(
    rows: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    field: FaserFieldTable | None,
) -> dict[str, Any]:
    groups = {
        "all": _empty_proxy_acc(),
        "clean_18hit": _empty_proxy_acc(),
        "complete_three_st": _empty_proxy_acc(),
        "dirty": _empty_proxy_acc(),
        "construction": _empty_proxy_acc(),
        "validation": _empty_proxy_acc(),
        "unknown": _empty_proxy_acc(),
    }
    match_cells: dict[tuple[str, ...], dict[str, float]] = defaultdict(
        lambda: {"n_construction": 0.0, "n_validation": 0.0, "sum_construction": 0.0, "sum_validation": 0.0}
    )
    source_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "n_sign": 0, "n_sign_agree": 0, "proxy": [], "truth": []}
    )
    missing_all = _empty_missing()
    missing_flip = _empty_missing()
    n_tracks = 0
    n_steps = int(config["path"]["n_steps_per_segment"])
    batch_size = int(config.get("sample_rules", {}).get("integral_batch_size", 4096))
    pending: list[dict[str, Any]] = []
    starts12: list[list[float]] = []
    ends12: list[list[float]] = []
    starts23: list[list[float]] = []
    ends23: list[list[float]] = []

    def flush() -> None:
        if not pending:
            return
        if field is None:
            raise ThreeStQpFieldNormalizedBendingError("field table is required to construct the proxy")
        i12 = field.integrate_yz_response_many(np.asarray(starts12), np.asarray(ends12), n_steps=n_steps)
        i23 = field.integrate_yz_response_many(np.asarray(starts23), np.asarray(ends23), n_steps=n_steps)
        _apply_integrals(pending, i12 + i23, config, groups, match_cells, source_stats)
        pending.clear()
        starts12.clear()
        ends12.clear()
        starts23.clear()
        ends23.clear()

    for raw in rows:
        slim = slim_track_row(raw) if "station_centroids" in raw or "mot_hits" in raw else dict(raw)
        row = decorate_track(slim, config)
        n_tracks += 1
        _observe_missing(missing_all, row, config, fit_flip_only=False)
        _observe_missing(missing_flip, row, config, fit_flip_only=True)
        if not (row.get("constructable") and _finite(row.get("bending_raw")) is not None):
            continue
        centroids = centroids_from_row(slim) or centroids_from_row(row)
        if centroids is None:
            continue
        pending.append(row)
        starts12.append([centroids[1]["x"], centroids[1]["y"], centroids[1]["z"]])
        ends12.append([centroids[2]["x"], centroids[2]["y"], centroids[2]["z"]])
        starts23.append([centroids[2]["x"], centroids[2]["y"], centroids[2]["z"]])
        ends23.append([centroids[3]["x"], centroids[3]["y"], centroids[3]["z"]])
        if len(pending) >= batch_size:
            flush()
    flush()
    clean = _finalize_proxy(groups["clean_18hit"])
    construction = _finalize_proxy(groups["construction"])
    validation = _finalize_proxy(groups["validation"])
    sources_out = {}
    for source_id, item in source_stats.items():
        ols = _ols_with_intercept(item["truth"], item["proxy"])
        n_sign = int(item["n_sign"])
        sources_out[source_id] = {
            "n": int(item["n"]),
            "n_sign": n_sign,
            "sign_agree": (item["n_sign_agree"] / n_sign) if n_sign else None,
            "ols_proxy_vs_qp_truth": ols,
            "spearman": _spearman(item["proxy"], item["truth"]),
        }
    return {
        "n_tracks": n_tracks,
        "n_clean_18hit": clean["n"],
        "n_complete_three_st": groups["complete_three_st"]["n"],
        "n_dirty": groups["dirty"]["n"],
        "n_constructable_bending": groups["all"]["n_constructable_bending"],
        "n_constructable_proxy": groups["all"]["n_constructable_proxy"],
        "clean_18hit": clean,
        "complete_three_st": _finalize_proxy(groups["complete_three_st"]),
        "dirty": _finalize_proxy(groups["dirty"]),
        "all": _finalize_proxy(groups["all"]),
        "construction": construction,
        "validation": validation,
        "source_stability_matched": match_residual_stability(match_cells, config),
        "by_source": sources_out,
        "missingness_all": _finalize_missing(missing_all),
        "missingness_fit_flip": _finalize_missing(missing_flip),
        "empirical_scale_applied": False,
        "used_fitted_q_over_p": False,
        "k_gev_per_tm": K_GEV_PER_TM,
    }


def evaluate_verdicts(
    report: Mapping[str, Any],
    field_contract: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = config["gates"]
    field_status = str(field_contract.get("status") or STATUS_UNRESOLVED)
    if field_contract.get("supported") is True:
        field_status = STATUS_SUPPORTED
    elif field_contract.get("supported") is False:
        field_status = STATUS_REJECTED
    clean = report.get("clean_18hit") or {}
    construction = report.get("construction") or {}
    validation = report.get("validation") or {}
    n_clean = int(report.get("n_clean_18hit") or 0)
    enough = n_clean >= int(gates["min_clean_contract"])
    enough_batch = n_clean >= int(gates["min_clean_recorded"])
    response = STATUS_UNRESOLVED
    fit = (construction.get("ols_proxy_vs_qp_truth") or clean.get("ols_proxy_vs_qp_truth") or {})
    slope = fit.get("slope")
    intercept = fit.get("intercept")
    rms_x = fit.get("rms_x")
    sign_agree = construction.get("sign_agree") if construction.get("n_sign") else clean.get("sign_agree")
    spearman = construction.get("spearman_proxy_vs_qp_truth") or clean.get("spearman_proxy_vs_qp_truth")
    if field_status != STATUS_SUPPORTED:
        response = STATUS_UNRESOLVED
    elif enough and slope is not None and sign_agree is not None and spearman is not None:
        intercept_ok = True
        if intercept is not None and rms_x:
            intercept_ok = abs(float(intercept)) / float(rms_x) <= float(gates["intercept_over_rms_max"])
        if (
            abs(float(slope) - 1.0) <= float(gates["slope_abs_dev_from_unity"])
            and intercept_ok
            and float(sign_agree) >= float(gates["sign_agree_present"])
            and abs(float(spearman)) >= float(gates["spearman_abs_present"])
            and not bool(report.get("empirical_scale_applied"))
        ):
            response = STATUS_SUPPORTED
        else:
            response = STATUS_REJECTED
    transfer = STATUS_UNRESOLVED
    matched = report.get("source_stability_matched") or {}
    slope_c = (construction.get("ols_proxy_vs_qp_truth") or {}).get("slope")
    slope_v = (validation.get("ols_proxy_vs_qp_truth") or {}).get("slope")
    sign_c = construction.get("sign_agree")
    sign_v = validation.get("sign_agree")
    if (
        matched.get("enough")
        and matched.get("rel_range") is not None
        and slope_c is not None
        and slope_v is not None
        and sign_c is not None
        and sign_v is not None
    ):
        floor = float(gates["abs_mean_floor"])
        slope_rel = abs(float(slope_c) - float(slope_v)) / max(abs(float(slope_c)), abs(float(slope_v)), floor)
        if (
            not bool(matched.get("sign_disagree"))
            and float(matched["rel_range"]) < float(gates["source_rel_range_matched"])
            and slope_rel < float(gates["slope_rel_range_construction_validation"])
            and float(sign_c) >= float(gates["sign_agree_present"])
            and float(sign_v) >= float(gates["sign_agree_present"])
        ):
            transfer = STATUS_SUPPORTED
        else:
            transfer = STATUS_REJECTED
    p_bins = clean.get("p_bins") or {}
    high = p_bins.get(str(config["binning"]["high_p_limit_label"])) or {}
    low = p_bins.get(str(config["binning"]["low_p_label"])) or {}
    high_p = STATUS_UNRESOLVED
    high_n = int(high.get("n_sign") or 0)
    low_n = int(low.get("n_sign") or 0)
    if (
        (enough_batch or enough)
        and high_n >= int(gates["min_high_p_2tev"])
        and low_n >= int(gates["min_low_p"])
        and high.get("sign_agree") is not None
        and low.get("sign_agree") is not None
    ):
        if (
            float(high["sign_agree"]) >= float(gates["high_p_raw_fidelity_min"])
            and float(low["sign_agree"]) - float(high["sign_agree"]) < float(gates["high_p_sign_drop"])
        ):
            high_p = STATUS_SUPPORTED
        else:
            high_p = STATUS_REJECTED
    missing = report.get("missingness_all") or {}
    selection = STATUS_UNRESOLVED
    n_flip_missing = int(missing.get("n_fit_flip_no_bending") or 0)
    gap = missing.get("availability_rate_gap")
    if n_flip_missing >= int(gates["min_fit_flip_no_bending"]) and gap is not None:
        if float(gap) >= float(gates["selection_rate_gap"]):
            selection = STATUS_SUPPORTED
        else:
            selection = STATUS_REJECTED
    key = (
        field_status == STATUS_SUPPORTED
        and response == STATUS_SUPPORTED
        and transfer == STATUS_SUPPORTED
    )
    return {
        VERDICT_FIELD: field_status,
        VERDICT_RESPONSE: response,
        VERDICT_TRANSFER: transfer,
        VERDICT_HIGH_P: high_p,
        VERDICT_SELECT: selection,
        "ckf_independent_physically_scaled_transferable_3st_curvature_proxy": key,
        "stage_independent_pass": key,
        "empirical_scale_applied": False,
    }


def dump_path_for_source(
    config: Mapping[str, Any],
    source_id: str,
    campaign: str,
) -> Path:
    filename = str(config["dump_filename"])
    return resolve_under_root(project_root(), str(config["dump_root"])) / campaign / source_id / filename


def _campaign_source_specs(config: Mapping[str, Any], campaign: str) -> list[tuple[str, dict[str, Any], Path]]:
    specs: list[tuple[str, dict[str, Any], Path]] = []
    if campaign == "smoke" and bool(config.get("smoke", {}).get("reuse_wb125_contract")):
        root = resolve_under_root(project_root(), str(config["smoke"]["wb125_dump_root"]))
        for split, key in (("construction", "construction_sources"), ("validation", "validation_sources")):
            for spec in config["mc_data"][key]:
                specs.append((split, spec, root / spec["source_id"] / str(config["dump_filename"])))
        return specs
    source_key = (
        (("construction", "batch_construction_sources"), ("validation", "batch_validation_sources"))
        if campaign == "batch"
        else (("construction", "construction_sources"), ("validation", "validation_sources"))
    )
    for split, key in source_key:
        for spec in config["mc_data"][key]:
            specs.append((split, spec, dump_path_for_source(config, spec["source_id"], campaign)))
    return specs


def iter_campaign_tracks(config: Mapping[str, Any], campaign: str):
    for split, spec, path in _campaign_source_specs(config, campaign):
        for row in iter_jsonl(path, split=split):
            if row.get("kind") != "track":
                continue
            if campaign != "smoke" and row.get("collection") not in (None, SOURCE_COLLECTION_NAME):
                continue
            yield slim_track_row(row)


def campaign_source_presence(config: Mapping[str, Any], campaign: str) -> dict[str, bool]:
    return {spec["source_id"]: path.is_file() for _split, spec, path in _campaign_source_specs(config, campaign)}


def verify_frozen_dumps(config: Mapping[str, Any], campaign: str) -> dict[str, Any]:
    report: dict[str, Any] = {"all_match": True, "files": {}}
    if campaign != "batch":
        report["skipped"] = True
        return report
    expected = config.get("frozen_wb126_dumps") or {}
    root = resolve_under_root(project_root(), str(config["dump_root"]))
    check_sha = bool((config.get("sample_rules") or {}).get("verify_dump_sha"))
    for source_id, spec in expected.items():
        path = root / "batch" / source_id / str(config["dump_filename"])
        present = path.is_file()
        item = {
            "path": str(path),
            "present": present,
            "expected": spec.get("sha256"),
            "sha_verified": False,
        }
        if not present:
            report["all_match"] = False
        elif check_sha and spec.get("sha256"):
            digest = sha256_file(path)
            item["sha256"] = digest
            item["match"] = digest == spec["sha256"]
            item["sha_verified"] = True
            if not item["match"]:
                report["all_match"] = False
        report["files"][source_id] = item
    report["sha_checked"] = check_sha
    return report


def inventory_campaign(
    config: Mapping[str, Any],
    campaign: str,
    field: FaserFieldTable | None,
) -> dict[str, Any]:
    sources_present = campaign_source_presence(config, campaign)
    acc = evaluate_rows(iter_campaign_tracks(config, campaign), config, field)
    acc["sources_present"] = sources_present
    acc["dumps_present"] = bool(sources_present) and all(sources_present.values())
    acc["campaign"] = campaign
    return acc


def decide(
    report: Mapping[str, Any],
    inherited: Mapping[str, Any],
    field_contract: Mapping[str, Any],
    *,
    dumps_materialized: bool,
    campaign: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if bool(config.get("three_st_qp_trusted_observable")):
        refuse_flip_s2()
    if bool(config.get("residual_conditional_authorized")):
        refuse_residual_conditional()
    verdicts = evaluate_verdicts(report, field_contract, config)
    n_clean = int(report.get("n_clean_18hit") or 0)
    dumps_ok = bool(dumps_materialized)
    pins_ok = bool((inherited.get("pinned_calypso_sources") or {}).get("all_match", True))
    inherited_ok = inherited.get("workbook_119", {}).get("decision") == "three_st_qp_calibration_not_established"
    if not dumps_ok or not pins_ok or not inherited_ok:
        decision = DECISION_NOT
        verdict = "FAIL"
        diagnosis = "FAIL"
    elif campaign == "batch" and n_clean >= int(config["gates"]["min_clean_recorded"]):
        decision = DECISION_RECORDED
        verdict = "PASS"
        diagnosis = "RECORDED"
    else:
        decision = DECISION_CONTRACT
        verdict = "PASS"
        diagnosis = "INCONCLUSIVE" if n_clean < int(config["gates"]["min_clean_contract"]) else "RECORDED"
    missing = report.get("missingness_all") or {}
    expected_missing = int(config["gates"]["wb126_n_fit_flip_no_bending_expected"])
    observed_missing = int(missing.get("n_fit_flip_no_bending") or 0)
    missing_quantified = (
        campaign == "batch"
        and abs(observed_missing - expected_missing)
        <= float(config["gates"]["wb119_n_fit_flip_rel_tol"]) * expected_missing
    )
    return {
        "kind": "three_st_qp_field_normalized_bending_contract",
        "task": TASK,
        "workbook": WORKBOOK,
        "campaign": campaign,
        "verdict": verdict,
        "diagnosis_verdict": diagnosis,
        "decision": decision,
        "verdicts": verdicts,
        "field_unit_sign_contract": field_contract,
        "n_tracks": report.get("n_tracks"),
        "n_clean_18hit": n_clean,
        "n_complete_three_st": report.get("n_complete_three_st"),
        "n_dirty": report.get("n_dirty"),
        "n_constructable_bending": report.get("n_constructable_bending"),
        "n_constructable_proxy": report.get("n_constructable_proxy"),
        "clean_18hit": report.get("clean_18hit"),
        "complete_three_st": report.get("complete_three_st"),
        "dirty": report.get("dirty"),
        "all": report.get("all"),
        "construction": report.get("construction"),
        "validation": report.get("validation"),
        "source_stability_matched": report.get("source_stability_matched"),
        "by_source": report.get("by_source"),
        "missingness_all": missing,
        "missingness_fit_flip": report.get("missingness_fit_flip"),
        "missingness_quantified": missing_quantified,
        "three_st_qp_trusted_observable": False,
        "residual_conditional_authorized": False,
        "s2_flipped_to_pass": False,
        "official_qp_like_jacobian_authorized": False,
        "measurement_uncertainty_propagated": False,
        "sigma_b_invented": False,
        "pull_coverage_claimed": False,
        "trusted_momentum": False,
        "e_r_ift_given_qp_computed": False,
        "e_r_ift_given_bending_computed": False,
        "empirical_scale_applied": False,
        "two_station_surrogate_used": False,
        "dumps_materialized": dumps_ok,
        "inherited_workbook_119": inherited.get("workbook_119", {}).get("decision"),
        "inherited_workbook_125": inherited.get("workbook_125", {}).get("decision"),
        "inherited_workbook_126": inherited.get("workbook_126", {}).get("decision"),
        "conversion_chain": conversion_chain(),
        "official_quantities": official_quantities(),
    }
