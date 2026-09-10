"""Yasu-S2F: decompose the 3ST q/p population mean.

Official means use the untrimmed primary sample:
    μ_fit   = E[(q/p)_fit]
    μ_truth = E[(q/p)_truth]
    μ_bias  = E[Δ(q/p)]
and the machine identity μ_fit = μ_truth + μ_bias.

Reweighting is a diagnostic counterfactual only.  Clean/dirty and
tail ranks are reported; they never redefine the official mean.
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
    ABS_QOVERP_EDGES,
    ABS_QOVERP_LABELS,
    P_TRUTH_EDGES_MEV,
    P_TRUTH_LABELS,
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

SCHEMA_VERSION = "three-st-qp-mean-decomposition-v1"
DEFAULT_CONFIG = "configs/three_st_qp_mean_decomposition_v1.yaml"
TASK = "YASU-S2F"
WORKBOOK = 122

DECISION_CONTRACT = "three_st_qp_mean_decomposition_contract_established"
DECISION_RECORDED = "three_st_qp_mean_decomposition_recorded"
DECISION_NOT = "three_st_qp_mean_decomposition_not_established"

MECH_POPULATION = "population_mean_dominates"
MECH_RECONSTRUCTION = "reconstruction_bias_dominates"
MECH_TAIL = "tail_dominated_mean"
MECH_MIXED = "mixed/inconclusive"

REQUIRED_INHERITANCE = (
    "workbook_117",
    "workbook_118",
    "workbook_119",
    "workbook_120",
    "workbook_121",
)


class ThreeStQpMeanDecompositionError(ValueError):
    """Raised when the S2F contract is illegal."""


def refuse_flip_s2() -> None:
    raise ThreeStQpMeanDecompositionError(
        "S2F cannot set three_st_qp_trusted_observable true"
    )


def refuse_residual_conditional() -> None:
    raise ThreeStQpMeanDecompositionError(
        "E[r_IFT|q/p] is Stage 3; S2F does not open it"
    )


def refuse_trim_official_mean() -> None:
    raise ThreeStQpMeanDecompositionError(
        "official μ is untrimmed; do not delete, trim, or winsorize"
    )


def refuse_reweight_as_calibration() -> None:
    raise ThreeStQpMeanDecompositionError(
        "reweighting is a diagnostic counterfactual, not a calibration population"
    )


def official_quantities() -> dict[str, str]:
    return {
        "mu_fit": "E[(q/p)_fit] on the untrimmed primary sample",
        "mu_truth": "E[(q/p)_truth] on the same sample",
        "mu_bias": "E[Δ(q/p)] = E[(q/p)_fit - (q/p)_truth]",
        "identity": "μ_fit = μ_truth + μ_bias",
        "charge_composition": "π± and π± μ± contributions",
        "charge_balanced": "diagnostic 0.5 μ+ + 0.5 μ−",
        "kinematics_matched": "diagnostic: both charges reweighted to pooled (p,tx,ty)",
        "tail_0p1": "contribution of top 0.1% |Δ| to the official mean",
        "tail_0p01": "contribution of top 0.01% |Δ| to the official mean",
    }


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ThreeStQpMeanDecompositionError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("task") != TASK:
        raise ThreeStQpMeanDecompositionError(f"task must be {TASK}")
    if int(config.get("workbook", -1)) != WORKBOOK:
        raise ThreeStQpMeanDecompositionError("workbook must be 122")
    if bool(config.get("three_st_qp_trusted_observable", True)):
        raise ThreeStQpMeanDecompositionError("three_st_qp_trusted_observable must stay false")
    if bool(config.get("residual_conditional_authorized", True)):
        raise ThreeStQpMeanDecompositionError("residual_conditional_authorized must stay false")
    if bool(config.get("do_not_submit_reconstruction_dump", False)) is not True:
        raise ThreeStQpMeanDecompositionError("do_not_submit_reconstruction_dump must be true")
    if bool(config.get("do_not_trim_or_winsorize_official_mean", False)) is not True:
        raise ThreeStQpMeanDecompositionError("official mean must stay untrimmed")
    if len(config.get("focus_identities") or []) != 6:
        raise ThreeStQpMeanDecompositionError("focus_identities must stay the six WB119 smoke rows")
    return dict(config)


def _expect_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise ThreeStQpMeanDecompositionError(f"{label} hash mismatch: {digest}")


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
            raise ThreeStQpMeanDecompositionError(f"{workbook} decision must stay frozen")
        inherited[workbook] = {
            "decision": decision["decision"],
            "config_sha256": spec["config_sha256"],
            "decision_sha256": spec["decision_sha256"],
        }
    if inherited["workbook_119"]["decision"] != "three_st_qp_calibration_not_established":
        raise ThreeStQpMeanDecompositionError("WB119 must remain not_established")
    if inherited["workbook_121"]["decision"] != "three_st_qp_root_cause_recorded":
        raise ThreeStQpMeanDecompositionError("WB121 root-cause must remain recorded")
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


def _empty_bin() -> dict[str, float | int]:
    return {
        "n": 0,
        "n_plus": 0,
        "n_minus": 0,
        "n_flip": 0,
        "sum_fit": 0.0,
        "sum_truth": 0.0,
        "sum_delta": 0.0,
        "sum_abs_fit": 0.0,
        "sum_abs_delta": 0.0,
    }


def _add(bin_acc: dict[str, Any], row: Mapping[str, Any]) -> None:
    bin_acc["n"] += 1
    if row["charge"] > 0.0:
        bin_acc["n_plus"] += 1
    elif row["charge"] < 0.0:
        bin_acc["n_minus"] += 1
    if row["flip"]:
        bin_acc["n_flip"] += 1
    bin_acc["sum_fit"] += row["q_fit"]
    bin_acc["sum_truth"] += row["q_truth"]
    bin_acc["sum_delta"] += row["delta"]
    bin_acc["sum_abs_fit"] += abs(row["q_fit"])
    bin_acc["sum_abs_delta"] += abs(row["delta"])


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
            "pi_plus": None,
            "pi_minus": None,
            "mu_fit": None,
            "mu_truth": None,
            "mu_bias": None,
            "identity_residual": None,
            "flip_rate": None,
        }
    mu_fit = float(bin_acc["sum_fit"]) / n
    mu_truth = float(bin_acc["sum_truth"]) / n
    mu_bias = float(bin_acc["sum_delta"]) / n
    return {
        "n": n,
        "n_plus": int(bin_acc["n_plus"]),
        "n_minus": int(bin_acc["n_minus"]),
        "n_flip": int(bin_acc["n_flip"]),
        "pi_plus": float(bin_acc["n_plus"]) / n,
        "pi_minus": float(bin_acc["n_minus"]) / n,
        "mu_fit": mu_fit,
        "mu_truth": mu_truth,
        "mu_bias": mu_bias,
        "identity_residual": mu_fit - mu_truth - mu_bias,
        "flip_rate": float(bin_acc["n_flip"]) / n,
    }


def empty_accumulator() -> dict[str, Any]:
    return {
        "n_records": 0,
        "n_primary": 0,
        "n_flip": 0,
        "n_truth_as_solution": 0,
        "n_clean": 0,
        "n_dirty": 0,
        "all": _empty_bin(),
        "plus": _empty_bin(),
        "minus": _empty_bin(),
        "clean": _empty_bin(),
        "dirty": _empty_bin(),
        "given_p": {label: _empty_bin() for label in P_TRUTH_LABELS},
        "given_q": {label: _empty_bin() for label in ABS_QOVERP_LABELS},
        "given_tx": {label: _empty_bin() for label in TX_LABELS},
        "given_ty": {label: _empty_bin() for label in TY_LABELS},
        "given_source": {},
        "kin_plus": {},
        "kin_minus": {},
        "q_fit": [],
        "q_truth": [],
        "delta": [],
        "focus_hits": [],
    }


def merge_accumulators(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    out = empty_accumulator()
    for key in ("n_records", "n_primary", "n_flip", "n_truth_as_solution", "n_clean", "n_dirty"):
        out[key] = left[key] + right[key]
    for key in ("all", "plus", "minus", "clean", "dirty"):
        out[key] = _merge_bin(left[key], right[key])
    for family in ("given_p", "given_q", "given_tx", "given_ty", "given_source", "kin_plus", "kin_minus"):
        keys = set(left[family]) | set(right[family])
        out[family] = {
            key: _merge_bin(left[family].get(key, _empty_bin()), right[family].get(key, _empty_bin()))
            for key in keys
        }
    out["q_fit"] = list(left["q_fit"]) + list(right["q_fit"])
    out["q_truth"] = list(left["q_truth"]) + list(right["q_truth"])
    out["delta"] = list(left["delta"]) + list(right["delta"])
    out["focus_hits"] = list(left["focus_hits"]) + list(right["focus_hits"])
    return out


def _missing_complete(stations: Sequence[int]) -> bool:
    have = {int(s) for s in stations if int(s) in (1, 2, 3)}
    return have == {1, 2, 3}


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
    if (
        q_fit is None
        or q_truth is None
        or not bool(row.get("truth_matched"))
        or not bool(row.get("truth_reference_available"))
    ):
        return
    charge = _finite(row.get("truth_charge"))
    if charge is None:
        charge = float(_sign(q_truth))
    p_truth = _finite(row.get("p_truth_s1_mev"))
    tx = _finite(row.get("tx_truth")) or _finite(row.get("tx"))
    ty = _finite(row.get("ty_truth")) or _finite(row.get("ty"))
    stations = [int(s) for s in (row.get("measurements_on_track_stations") or [])]
    n_mot = int(row.get("n_mot") or 0)
    n_out = int(row.get("n_outlier_hits") or 0)
    chi2 = _finite(row.get("chi2"))
    ndof = _finite(row.get("ndof"))
    chi2ndof = None if ndof is None or ndof <= 0.0 or chi2 is None else chi2 / ndof
    clean_spec = config["clean_subsample"]
    clean = (
        _missing_complete(stations)
        and n_mot == int(clean_spec["require_n_mot"])
        and ndof is not None
        and ndof > 0.0
        and n_out == int(clean_spec["require_n_outlier"])
        and chi2ndof is not None
        and chi2ndof < float(clean_spec["max_chi2_ndof"])
    )
    flip = _sign(q_fit) == 0 or _sign(q_truth) == 0 or _sign(q_fit) != _sign(q_truth)
    payload = {
        "q_fit": q_fit,
        "q_truth": q_truth,
        "delta": q_fit - q_truth,
        "charge": charge,
        "flip": flip,
    }
    acc["n_primary"] += 1
    if flip:
        acc["n_flip"] += 1
    _add(acc["all"], payload)
    if charge > 0.0:
        _add(acc["plus"], payload)
    elif charge < 0.0:
        _add(acc["minus"], payload)
    if clean:
        acc["n_clean"] += 1
        _add(acc["clean"], payload)
    else:
        acc["n_dirty"] += 1
        _add(acc["dirty"], payload)

    def _put(family: str, label: str) -> None:
        acc[family].setdefault(label, _empty_bin())
        _add(acc[family][label], payload)

    if p_truth is not None:
        _put("given_p", assign_bin(p_truth, P_TRUTH_EDGES_MEV, P_TRUTH_LABELS))
    _put("given_q", assign_bin(abs(q_truth), ABS_QOVERP_EDGES, ABS_QOVERP_LABELS))
    if tx is not None:
        _put("given_tx", assign_bin(tx, SLOPE_EDGES, TX_LABELS))
    if ty is not None:
        _put("given_ty", assign_bin(ty, SLOPE_EDGES, TY_LABELS))
    _put("given_source", str(row.get("source_id") or "unknown"))
    if p_truth is not None and tx is not None and ty is not None:
        kin = (
            f"{assign_bin(p_truth, P_TRUTH_EDGES_MEV, P_TRUTH_LABELS)}|"
            f"{assign_bin(tx, SLOPE_EDGES, TX_LABELS)}|"
            f"{assign_bin(ty, SLOPE_EDGES, TY_LABELS)}"
        )
        family = "kin_plus" if charge > 0.0 else "kin_minus" if charge < 0.0 else None
        if family:
            _put(family, kin)
    acc["q_fit"].append(q_fit)
    acc["q_truth"].append(q_truth)
    acc["delta"].append(q_fit - q_truth)
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
                "delta": q_fit - q_truth,
                "clean": clean,
                "flip": flip,
            }
        )


def evaluate_tracks(records: Iterable[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    acc = empty_accumulator()
    n = 0
    for row in records:
        n += 1
        if n % 250000 == 0:
            print(f"three_st_qp_mean_decomposition: evaluated {n} rows", file=sys.stderr, flush=True)
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


def _tail_report(
    q_fit: Sequence[float],
    q_truth: Sequence[float],
    delta: Sequence[float],
    *,
    fraction: float,
    rank_by: str,
) -> dict[str, Any]:
    n = len(q_fit)
    if n <= 0:
        return {"n": 0, "k": 0, "fraction": fraction, "rank_by": rank_by}
    k = max(1, int(math.ceil(fraction * n)))
    if rank_by == "abs_delta":
        order = np.argsort(np.abs(np.asarray(delta, dtype=np.float64)))[::-1]
    else:
        order = np.argsort(np.abs(np.asarray(q_fit, dtype=np.float64)))[::-1]
    top = order[:k]
    fit = np.asarray(q_fit, dtype=np.float64)
    truth = np.asarray(q_truth, dtype=np.float64)
    dlt = np.asarray(delta, dtype=np.float64)
    sum_all_fit = float(np.sum(fit))
    sum_top_fit = float(np.sum(fit[top]))
    sum_top_truth = float(np.sum(truth[top]))
    sum_top_delta = float(np.sum(dlt[top]))
    return {
        "n": n,
        "k": k,
        "fraction": fraction,
        "rank_by": rank_by,
        "mu_fit_contribution": sum_top_fit / n,
        "mu_truth_contribution": sum_top_truth / n,
        "mu_bias_contribution": sum_top_delta / n,
        "share_of_sum_fit": (sum_top_fit / sum_all_fit) if sum_all_fit != 0.0 else None,
        "official_mean_redefined": False,
    }


def _charge_balanced(plus: Mapping[str, Any], minus: Mapping[str, Any]) -> dict[str, Any]:
    plus_s = _mean_pack(plus)
    minus_s = _mean_pack(minus)
    if plus_s["n"] <= 0 or minus_s["n"] <= 0:
        return {
            "applicable": False,
            "diagnostic_only": True,
            "calibration_population": False,
        }
    return {
        "applicable": True,
        "diagnostic_only": True,
        "calibration_population": False,
        "mu_fit": 0.5 * float(plus_s["mu_fit"]) + 0.5 * float(minus_s["mu_fit"]),
        "mu_truth": 0.5 * float(plus_s["mu_truth"]) + 0.5 * float(minus_s["mu_truth"]),
        "mu_bias": 0.5 * float(plus_s["mu_bias"]) + 0.5 * float(minus_s["mu_bias"]),
        "n_plus": plus_s["n"],
        "n_minus": minus_s["n"],
    }


def _kinematics_matched(
    kin_plus: Mapping[str, Mapping[str, Any]],
    kin_minus: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    cells = set(kin_plus) | set(kin_minus)
    num_fit = num_truth = num_bias = 0.0
    den = 0.0
    n_used = 0
    n_dropped = 0
    for cell in cells:
        plus = kin_plus.get(cell, _empty_bin())
        minus = kin_minus.get(cell, _empty_bin())
        n_p = int(plus["n"])
        n_m = int(minus["n"])
        if n_p <= 0 or n_m <= 0:
            n_dropped += 1
            continue
        weight = float(n_p + n_m)
        mu_fit_c = 0.5 * (float(plus["sum_fit"]) / n_p + float(minus["sum_fit"]) / n_m)
        mu_truth_c = 0.5 * (float(plus["sum_truth"]) / n_p + float(minus["sum_truth"]) / n_m)
        mu_bias_c = 0.5 * (float(plus["sum_delta"]) / n_p + float(minus["sum_delta"]) / n_m)
        num_fit += weight * mu_fit_c
        num_truth += weight * mu_truth_c
        num_bias += weight * mu_bias_c
        den += weight
        n_used += 1
    if den <= 0.0:
        return {
            "applicable": False,
            "diagnostic_only": True,
            "calibration_population": False,
            "n_cells_used": n_used,
            "n_cells_dropped": n_dropped,
        }
    return {
        "applicable": True,
        "diagnostic_only": True,
        "calibration_population": False,
        "mu_fit": num_fit / den,
        "mu_truth": num_truth / den,
        "mu_bias": num_bias / den,
        "n_cells_used": n_used,
        "n_cells_dropped": n_dropped,
    }


def _classify(pack: Mapping[str, Any], tail_share: float | None, config: Mapping[str, Any]) -> dict[str, Any]:
    gates = config["gates"]
    mu_f = pack.get("mu_fit")
    mu_t = pack.get("mu_truth")
    mu_b = pack.get("mu_bias")
    n = int(pack.get("n") or 0)
    if n < int(gates["min_n_mechanism_verdict"]) or mu_f is None or mu_t is None or mu_b is None:
        return {
            "mechanism": None,
            "applicable": False,
            "flags": {"population": False, "reconstruction": False, "tail": False},
        }
    scale = max(abs(float(mu_f)), float(gates["abs_mean_floor"]))
    truth_frac = abs(float(mu_t)) / scale
    bias_frac = abs(float(mu_b)) / scale
    population = bool(
        _same_sign(mu_f, mu_t)
        and truth_frac >= float(gates["population_truth_fraction_min"])
        and bias_frac < float(gates["population_bias_fraction_max"])
    )
    reconstruction = bool(
        _same_sign(mu_f, mu_b)
        and truth_frac < float(gates["reconstruction_truth_fraction_max"])
        and bias_frac >= float(gates["reconstruction_bias_fraction_min"])
    )
    tail = bool(
        tail_share is not None and abs(float(tail_share)) >= float(gates["tail_mean_fraction_min"])
    )
    if tail:
        mechanism = MECH_TAIL
    elif population:
        mechanism = MECH_POPULATION
    elif reconstruction:
        mechanism = MECH_RECONSTRUCTION
    else:
        mechanism = MECH_MIXED
    return {
        "mechanism": mechanism,
        "applicable": True,
        "flags": {"population": population, "reconstruction": reconstruction, "tail": tail},
        "truth_fraction_of_fit": truth_frac,
        "bias_fraction_of_fit": bias_frac,
        "tail_share_of_sum_fit": tail_share,
    }


def evaluate_decomposition(acc: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    gates = config["gates"]
    overall = _mean_pack(acc["all"])
    plus = _mean_pack(acc["plus"])
    minus = _mean_pack(acc["minus"])
    clean = _mean_pack(acc["clean"])
    dirty = _mean_pack(acc["dirty"])
    quantiles = {
        "q_fit": _quantiles(acc["q_fit"], gates["quantile_probabilities"]),
        "q_truth": _quantiles(acc["q_truth"], gates["quantile_probabilities"]),
        "delta": _quantiles(acc["delta"], gates["quantile_probabilities"]),
    }
    tails = {}
    for fraction in gates["tail_fractions"]:
        label = f"top_{fraction}"
        tails[f"{label}_abs_delta"] = _tail_report(
            acc["q_fit"], acc["q_truth"], acc["delta"], fraction=float(fraction), rank_by="abs_delta"
        )
        tails[f"{label}_abs_fit"] = _tail_report(
            acc["q_fit"], acc["q_truth"], acc["delta"], fraction=float(fraction), rank_by="abs_fit"
        )
    primary_tail = tails["top_0.001_abs_delta"]
    classification = _classify(overall, primary_tail.get("share_of_sum_fit"), config)
    charge_balanced = _charge_balanced(acc["plus"], acc["minus"])
    kinematics = _kinematics_matched(acc["kin_plus"], acc["kin_minus"])
    if plus["n"] and minus["n"] and overall["n"]:
        composition = {
            "pi_plus": overall["pi_plus"],
            "pi_minus": overall["pi_minus"],
            "plus_contribution_to_mu_fit": overall["pi_plus"] * plus["mu_fit"],
            "minus_contribution_to_mu_fit": overall["pi_minus"] * minus["mu_fit"],
            "plus_contribution_to_mu_truth": overall["pi_plus"] * plus["mu_truth"],
            "minus_contribution_to_mu_truth": overall["pi_minus"] * minus["mu_truth"],
            "population_vs_balanced_truth": (
                overall["mu_truth"] - charge_balanced["mu_truth"]
                if charge_balanced.get("applicable")
                else None
            ),
        }
    else:
        composition = {"pi_plus": overall.get("pi_plus"), "pi_minus": overall.get("pi_minus")}
    return {
        "overall": overall,
        "plus": plus,
        "minus": minus,
        "clean": clean,
        "dirty": dirty,
        "given_p": {k: _mean_pack(v) for k, v in acc["given_p"].items()},
        "given_q": {k: _mean_pack(v) for k, v in acc["given_q"].items()},
        "given_tx": {k: _mean_pack(v) for k, v in acc["given_tx"].items()},
        "given_ty": {k: _mean_pack(v) for k, v in acc["given_ty"].items()},
        "given_source": {k: _mean_pack(v) for k, v in acc["given_source"].items()},
        "quantiles": quantiles,
        "tails": tails,
        "charge_composition": composition,
        "charge_balanced": charge_balanced,
        "kinematics_matched": kinematics,
        "classification": classification,
        "focus_hits": list(acc["focus_hits"]),
        "n_primary": int(acc["n_primary"]),
        "n_records": int(acc["n_records"]),
        "n_flip": int(acc["n_flip"]),
        "n_clean": int(acc["n_clean"]),
        "n_truth_as_solution": int(acc["n_truth_as_solution"]),
        "official_mean_trimmed": False,
        "reweight_used_as_calibration": False,
    }


def _direction_consistent(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    mech = left.get("classification", {}).get("mechanism")
    if mech is None or mech != right.get("classification", {}).get("mechanism"):
        return False
    if mech == MECH_MIXED:
        return False
    lo = left["overall"]
    ro = right["overall"]
    if mech == MECH_POPULATION:
        return _same_sign(lo.get("mu_truth"), ro.get("mu_truth"))
    if mech == MECH_RECONSTRUCTION:
        return _same_sign(lo.get("mu_bias"), ro.get("mu_bias"))
    if mech == MECH_TAIL:
        return True
    return False


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
    elif inherited.get("workbook_121", {}).get("decision") != "three_st_qp_root_cause_recorded":
        mechanism = "s2e_not_inherited"
    elif not dumps_materialized:
        mechanism = "calibration_dump_not_materialized"
    elif int(pooled["n_truth_as_solution"]):
        mechanism = "truth_used_as_fit_or_solution"
    elif not _denominator_ok(pooled, campaign, config):
        mechanism = "wb119_denominator_drift"
    else:
        contract = "PASS"
    construction_rep = evaluate_decomposition(construction["tracks"], config)
    validation_rep = evaluate_decomposition(validation["tracks"], config)
    pooled_rep = evaluate_decomposition(pooled, config)
    if contract == "PASS" and not (
        _identity_ok(construction_rep["overall"], config)
        and _identity_ok(validation_rep["overall"], config)
        and _identity_ok(pooled_rep["overall"], config)
    ):
        contract = "FAIL"
        mechanism = "mean_identity_failed"
    if contract != "PASS":
        decision = DECISION_NOT
        verdict = "FAIL"
        diagnosis_verdict = "FAIL"
        stable = False
        official_mechanism = None
    elif int(pooled_rep["n_primary"]) < int(config["gates"]["min_n_mechanism_verdict"]):
        decision = DECISION_CONTRACT
        verdict = "PASS"
        diagnosis_verdict = "INCONCLUSIVE"
        mechanism = None
        stable = False
        official_mechanism = None
    else:
        decision = DECISION_RECORDED
        verdict = "PASS"
        diagnosis_verdict = "RECORDED"
        stable = _direction_consistent(construction_rep, validation_rep)
        official_mechanism = (
            construction_rep["classification"]["mechanism"] if stable else MECH_MIXED
        )
        mechanism = official_mechanism
    return {
        "kind": "three_st_qp_mean_decomposition_contract",
        "task": TASK,
        "workbook": WORKBOOK,
        "campaign": campaign,
        "verdict": verdict,
        "decision": decision,
        "mechanism": mechanism,
        "contract_verdict": contract,
        "diagnosis_verdict": diagnosis_verdict,
        "official_mechanism": official_mechanism,
        "train_validation_direction_consistent": stable if contract == "PASS" else False,
        "three_st_qp_trusted_observable": False,
        "residual_conditional_authorized": False,
        "s2_flipped_to_pass": False,
        "sign_flip_tracks_dropped": False,
        "official_mean_trimmed": False,
        "reweight_used_as_calibration": False,
        "new_reconstruction_dump_authorized": False,
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
