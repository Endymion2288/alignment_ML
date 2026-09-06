#!/usr/bin/env python3
"""Workbook 76: read-only failure-mechanism attribution.

Why did the same Physical Pair-Relative representation pass two-fold
source-transfer (Workbook 74) but destroy truth on frozen development after
six-source training (Workbook 75)?

Does not train, does not change architecture / loss / B, and never opens
Final Blind (00800_00849) or sealed test.  Development (00350_00399) is
read only for attribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scipy.stats import ks_2samp

from scripts.audit_relative_route_v4_generalization import _load_split
from scripts.evaluate_physical_pair_relative_route_v1_source_transfer import (
    BOUND,
    SATURATION_FRACTION_OF_BOUND,
    _predict_primary_r7,
)
from scripts.evaluate_relative_route_v4_development import (
    _score_arm,
    cd_counts,
    frozen_packing_config,
    refuse_forbidden_paths,
)
from scripts.evaluate_relative_route_v5a_source_transfer import (
    _catastrophic_and_new_c,
    _delta_summary,
    _evaluate_arm_single_pass,
    _load_arm_flexible,
    _metrics_from_routemetrics,
)
from scripts.run_frozen_association_backbone import _sha256
from scripts.run_refit_multidof_closure import _json_ready
from training.geometry_aware_transformer import EDGE_FEATURE_NAMES
from training.route_aware_transformer import (
    _forward_route_batch,
    _make_route_batch,
    load_relative_route_v4_head_only_artifact,
)
from training.source_diversity_audit import (
    RESERVED_BLIND_SOURCES,
    SOURCE_CHARGE,
    UNUSED_RESERVE_SOURCES,
    is_sealed_source,
    source_dsid,
)

W64_SHA = "0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236"
WB75_SHA = "b961cefdf499f648487f05bac381e2bc02d413769419c9f041961fa072d94f02"
W64_ROOT = Path("outputs/mc24_four_station_source_diversity_v1/checkpoint")
WB75_CKPT = Path(
    "outputs/mc24_four_station_physical_pair_relative_route_v1_six_source/training/primary/checkpoint_last.pt"
)
WB74_F1_CKPT = Path(
    "outputs/mc24_four_station_physical_pair_relative_route_v1_source_transfer/training/holdout_family1/primary/checkpoint_last.pt"
)
WB74_F2_CKPT = Path(
    "outputs/mc24_four_station_physical_pair_relative_route_v1_source_transfer/training/holdout_family2/primary/checkpoint_last.pt"
)
TRAIN_MANIFEST = Path(
    "outputs/mc24_four_station_source_diversity_train_v1/overlay_synthetic_v1/synthetic_corpus_manifest.json"
)
DEV_MANIFEST = Path(
    "outputs/mc24_four_station_source_diversity_blind_v1/overlay_synthetic_v1/synthetic_corpus_manifest.json"
)
OUT = Path("outputs/mc24_four_station_physical_pair_relative_route_v1_failure_audit")
RPHYS_NAMES = (
    [f"e01_{n}" for n in EDGE_FEATURE_NAMES]
    + [f"e12_{n}" for n in EDGE_FEATURE_NAMES]
    + [f"e23_{n}" for n in EDGE_FEATURE_NAMES]
    + ["L01", "L12", "L23"]
)
THR = SATURATION_FRACTION_OF_BOUND * BOUND


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _uid_source_map(samples) -> dict[tuple[int, int], str]:
    mapping: dict[tuple[int, int], str] = {}
    for sample in samples:
        for uid in sample.source_event_uids:
            parts = str(uid).split(":")
            if len(parts) != 3:
                continue
            source, run_s, event_s = parts
            mapping[(int(run_s), int(event_s))] = str(source)
    return mapping


def _event_source(graph, uid_map: dict[tuple[int, int], str]) -> str:
    key = (int(graph.event.run_id), int(graph.event.event_id))
    if key in uid_map:
        return uid_map[key]
    # Fallback: unique constituent if the sample is source-pure.
    ids = tuple(graph.sample.source_ids)
    if len(ids) == 1:
        return ids[0]
    return str(graph.sample.source_id)


def _charge_from_source(source: str) -> str | None:
    try:
        return SOURCE_CHARGE.get(source_dsid(source))
    except ValueError:
        return None


def _charge_from_pdg(event, node_indices: np.ndarray) -> str | None:
    pdg = np.asarray(event.truth_pdg, dtype=np.int32)[np.asarray(node_indices, dtype=np.int64)]
    uniq = {int(v) for v in pdg.tolist()}
    if uniq == {13}:
        return "mu_minus"
    if uniq == {-13}:
        return "mu_plus"
    return None


def _capture_rphys(wrapper, bundle, tables, node_std, edge_std, device, batch_size):
    holder: dict[str, np.ndarray] = {}
    hook = wrapper.trainable.route_encoder.register_forward_pre_hook(
        lambda module, inputs: holder.__setitem__("x", inputs[0].detach().cpu().numpy())
    )
    details: dict[int, np.ndarray] = {}
    try:
        with __import__("torch").no_grad():
            for start in range(0, len(bundle.graphs), batch_size):
                graphs = bundle.graphs[start : start + batch_size]
                batch = _make_route_batch(graphs, node_std, edge_std, device, tables)
                _forward_route_batch(wrapper, batch)
                x = holder["x"].astype(np.float64)
                roff = 0
                for g in graphs:
                    stop = roff + tables[id(g)].size
                    details[id(g)] = x[roff:stop]
                    roff = stop
    finally:
        hook.remove()
    return details


def _row_key(row: Mapping[str, Any]):
    return (
        str(row["payload_id"]),
        int(row["run_id"]),
        int(row["event_id"]),
        tuple(int(e["index"]) for e in row["endpoints"]),
    )


def _truth_index_from_table(table, endpoints: tuple[int, ...]) -> int | None:
    nodes = np.asarray(table.node_indices, dtype=np.int64)
    target = np.asarray(endpoints, dtype=np.int64)
    hits = np.where((nodes == target).all(axis=1))[0]
    if hits.size != 1:
        return None
    return int(hits[0])


def _summarize_array(values: np.ndarray, bound: float | None = None) -> dict[str, Any]:
    return _delta_summary(np.asarray(values, dtype=np.float64), bound)


def _ks_table(a: np.ndarray, b: np.ndarray, names: list[str]) -> list[dict[str, Any]]:
    rows = []
    if a.size == 0 or b.size == 0:
        return rows
    for i, name in enumerate(names):
        stat, p = ks_2samp(a[:, i], b[:, i])
        rows.append(
            {
                "name": name,
                "ks_statistic": float(stat),
                "p_value": float(p),
                "train_mean": float(a[:, i].mean()),
                "dev_mean": float(b[:, i].mean()),
                "train_std": float(a[:, i].std()),
                "dev_std": float(b[:, i].std()),
                "mean_shift_over_train_std": float((b[:, i].mean() - a[:, i].mean()) / max(a[:, i].std(), 1e-12)),
            }
        )
    rows.sort(key=lambda r: -r["ks_statistic"])
    return rows


def _nn_distance(query: np.ndarray, ref: np.ndarray, max_ref: int = 2000) -> np.ndarray:
    if query.size == 0 or ref.size == 0:
        return np.empty(0, dtype=np.float64)
    rng = np.random.default_rng(20260906)
    if ref.shape[0] > max_ref:
        ref = ref[rng.choice(ref.shape[0], size=max_ref, replace=False)]
    dmin = np.full(query.shape[0], np.inf, dtype=np.float64)
    chunk = 256
    for start in range(0, query.shape[0], chunk):
        q = query[start : start + chunk]
        # (Q, R) squared Euclidean
        q2 = np.sum(q * q, axis=1, keepdims=True)
        r2 = np.sum(ref * ref, axis=1, keepdims=True).T
        dist2 = np.maximum(q2 + r2 - 2.0 * q.dot(ref.T), 0.0)
        dmin[start : start + chunk] = np.sqrt(dist2.min(axis=1))
    return dmin


def checkpoint_contract(device: str) -> dict[str, Any]:
    observed = _file_sha(WB75_CKPT)
    if observed != WB75_SHA:
        raise SystemExit(f"WB75 checkpoint sha {observed} != frozen {WB75_SHA}")
    wrapper, artifact = load_relative_route_v4_head_only_artifact(WB75_CKPT, device=device)
    cfg = artifact.model_config
    import torch

    try:
        raw = torch.load(WB75_CKPT, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(WB75_CKPT, map_location="cpu")
    parent = str(raw.get("frozen_workbook64_sha256") or raw.get("training_summary", {}).get("frozen_workbook64_sha256") or "")
    mode = str(cfg.route_representation_mode)
    bound = cfg.route_correction_bound
    has_node_modules = any(
        hasattr(wrapper.trainable, name)
        for name in ("route_query", "route_node_key", "route_edge_projection", "route_pair_embedding")
    )
    return {
        "checkpoint_path": str(WB75_CKPT),
        "checkpoint_sha256": observed,
        "expected_checkpoint_sha256": WB75_SHA,
        "sha_ok": observed == WB75_SHA,
        "route_representation_mode": mode,
        "mode_ok": mode == "physical_pair_relative",
        "route_correction_bound": bound,
        "bound_ok": float(bound) == 4.0,
        "no_absolute_node_latent_modules": not has_node_modules,
        "route_input_width": int(wrapper.trainable.route_encoder[0].in_features),
        "frozen_workbook64_sha256_in_payload": parent,
        "parent_ok": parent == W64_SHA or parent == "",
        "expected_parent": W64_SHA,
        "edge_feature_names": list(artifact.edge_feature_names),
    }


def _score_named(name, loaded, bundle, tables, device, batch_size, config, capture_rphys: bool):
    raw, calibrated, route_maps = _score_arm(loaded, bundle, device, batch_size)
    rows, metrics = _evaluate_arm_single_pass(bundle, calibrated, route_maps, config)
    details = None
    rphys = None
    if loaded["arm"] != "w64":
        details = _predict_primary_r7(
            loaded["model"],
            bundle,
            tables,
            loaded["artifact"].node_standardizer,
            loaded["artifact"].edge_standardizer,
            device,
            batch_size,
        )
        if capture_rphys:
            rphys = _capture_rphys(
                loaded["model"],
                bundle,
                tables,
                loaded["artifact"].node_standardizer,
                loaded["artifact"].edge_standardizer,
                device,
                batch_size,
            )
    return {
        "name": name,
        "rows": rows,
        "metrics": _metrics_from_routemetrics(metrics),
        "cd": cd_counts(rows),
        "details": details,
        "rphys": rphys,
        "checkpoint_sha256": loaded["checkpoint_sha256"],
    }


def _attach_truth_records(bundle, tables, uid_map, scored, w64_rows) -> list[dict[str, Any]]:
    w64_index = {_row_key(r): r for r in w64_rows}
    rows_by_event: dict[tuple[str, int, int], list] = defaultdict(list)
    for row in scored["rows"]:
        rows_by_event[(str(row["payload_id"]), int(row["run_id"]), int(row["event_id"]))].append(row)
    records = []
    details = scored["details"] or {}
    rphys = scored["rphys"] or {}
    for g in bundle.graphs:
        table = tables[id(g)]
        source = _event_source(g, uid_map)
        try:
            dsid = source_dsid(source)
        except ValueError:
            dsid = None
        src_charge = _charge_from_source(source) if dsid else None
        det = details.get(id(g))
        rp = rphys.get(id(g))
        event_rows = rows_by_event.get((str(g.sample.payload_id), int(g.event.run_id), int(g.event.event_id)), [])
        for row in event_rows:
            endpoints = tuple(int(e["index"]) for e in row["endpoints"])
            idx = _truth_index_from_table(table, endpoints)
            if idx is None:
                continue
            pdg_charge = _charge_from_pdg(g.event, np.asarray(endpoints))
            w64 = w64_index.get(_row_key(row))
            u = row.get("complete_truth_route_utility")
            rec = {
                "payload_id": str(row["payload_id"]),
                "run_id": int(row["run_id"]),
                "event_id": int(row["event_id"]),
                "source": source,
                "dsid": dsid,
                "source_charge": src_charge,
                "pdg_charge": pdg_charge,
                "charge_agree": bool(src_charge and pdg_charge and src_charge == pdg_charge),
                "selected": bool(row.get("selected")),
                "loss_stage": str(row.get("loss_stage")),
                "U_complete": None if u is None else float(u),
                "w64_selected": None if w64 is None else bool(w64.get("selected")),
                "w64_loss_stage": None if w64 is None else str(w64.get("loss_stage")),
                "w64_U": None
                if w64 is None or w64.get("complete_truth_route_utility") is None
                else float(w64["complete_truth_route_utility"]),
            }
            if det is not None:
                rec["raw_delta"] = float(det["raw_delta"][idx])
                rec["bounded_delta"] = float(det["bounded_delta"][idx])
                rec["L_edge"] = float(det["L_edge"][idx])
                rec["L_corrected"] = float(det["L_corrected"][idx])
            if rp is not None:
                rec["R_phys"] = np.asarray(rp[idx], dtype=np.float64)
                rec["L01"] = float(rp[idx, 33])
                rec["L12"] = float(rp[idx, 34])
                rec["L23"] = float(rp[idx, 35])
            rec["catastrophic"] = bool(
                rec.get("w64_selected") and rec.get("U_complete") is not None and rec["U_complete"] <= 0.0
            )
            records.append(rec)
    return records


def _cohort(records: list[dict[str, Any]], pred) -> list[dict[str, Any]]:
    return [r for r in records if pred(r)]


def _record_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"n_routes": 0, "n_events": 0}
    events = {(r["payload_id"], r["run_id"], r["event_id"]) for r in records}
    sources = Counter(r["source"] for r in records)
    dsids = Counter(r["dsid"] for r in records)
    charges = Counter(r["source_charge"] for r in records)
    c_count = sum(1 for r in records if str(r["loss_stage"]) == "utility_nonpositive" or (r.get("U_complete") is not None and r["U_complete"] <= 0))
    cat = sum(1 for r in records if r.get("catastrophic"))
    deltas = np.asarray([r["bounded_delta"] for r in records if "bounded_delta" in r], dtype=np.float64)
    return {
        "n_routes": int(len(records)),
        "n_events": int(len(events)),
        "C_like": int(c_count),
        "catastrophic": int(cat),
        "sources": dict(sources),
        "dsids": dict(dsids),
        "source_charge": dict(charges),
        "bounded_delta": _summarize_array(deltas, BOUND) if deltas.size else {"count": 0},
        "frac_sat_neg": float(np.mean(deltas <= -THR)) if deltas.size else None,
        "frac_neg": float(np.mean(deltas < 0.0)) if deltas.size else None,
    }


def _margin_bin(w64_U) -> str:
    if w64_U is None:
        return "unknown"
    u = float(w64_U)
    if u <= 0.0:
        return "already_nonpositive"
    if u <= 1.0:
        return "near_dustbin"
    if u <= 2.0:
        return "low_margin"
    return "high_margin"


def analyze(train_wb75, train_w64, dev_wb75, dev_w64, dev_f1, dev_f2) -> dict[str, Any]:
    train_truth = [r for r in train_wb75 if "bounded_delta" in r]
    dev_truth = [r for r in dev_wb75 if "bounded_delta" in r]
    f1_truth = [r for r in dev_f1 if "bounded_delta" in r]
    f2_truth = [r for r in dev_f2 if "bounded_delta" in r]

    def arr(recs, key):
        return np.asarray([r[key] for r in recs if key in r], dtype=np.float64)

    train_rp = np.stack([r["R_phys"] for r in train_truth if "R_phys" in r]) if train_truth else np.empty((0, 36))
    dev_rp = np.stack([r["R_phys"] for r in dev_truth if "R_phys" in r]) if dev_truth else np.empty((0, 36))
    ks = _ks_table(train_rp, dev_rp, list(RPHYS_NAMES)) if train_rp.size and dev_rp.size else []
    nn = _nn_distance(dev_rp, train_rp) if train_rp.size and dev_rp.size else np.empty(0)
    for rec, dist in zip([r for r in dev_truth if "R_phys" in r], nn):
        rec["nn_dist_to_train_truth_R_phys"] = float(dist)

    destroyed = [r for r in dev_truth if r.get("catastrophic")]
    survived = [r for r in dev_truth if r.get("w64_selected") and not r.get("catastrophic")]
    sat_neg = [r for r in dev_truth if r.get("bounded_delta", 0) <= -THR]

    # Per-source train vs implied family on development.
    by_dsid = defaultdict(list)
    for r in train_truth:
        by_dsid[str(r["dsid"])].append(r)
    by_dsid_dev = defaultdict(list)
    for r in dev_truth:
        by_dsid_dev[str(r["dsid"])].append(r)

    # Station-pair: L01/L12/L23 of destroyed vs survived.
    def edge_block(recs):
        if not recs:
            return {}
        return {
            "L01": _summarize_array(arr(recs, "L01")),
            "L12": _summarize_array(arr(recs, "L12")),
            "L23": _summarize_array(arr(recs, "L23")),
        }

    # Margin strata on development using W64 U.
    margin = defaultdict(list)
    for r in dev_truth:
        margin[_margin_bin(r.get("w64_U"))].append(r)
    fragment = [r for r in dev_truth if r.get("w64_loss_stage") == "packing_competition"]

    # Charge: only where source-code map and event PDG agree.
    agreed = [r for r in dev_truth if r.get("charge_agree")]
    charge_split = {
        "agreement_rate": (len(agreed) / len(dev_truth)) if dev_truth else None,
        "n_disagreement": int(sum(1 for r in dev_truth if r.get("source_charge") and r.get("pdg_charge") and not r.get("charge_agree"))),
        "by_source_charge": {ch: _record_stats([r for r in agreed if r["source_charge"] == ch]) for ch in ("mu_minus", "mu_plus")},
    }

    # Extrapolation: delta vs nn distance / L_edge.
    def corr(x, y):
        if x.size < 3 or y.size < 3:
            return None
        if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    nn_arr = arr(dev_truth, "nn_dist_to_train_truth_R_phys")
    dlt = arr(dev_truth, "bounded_delta")
    ledge = arr(dev_truth, "L_edge")
    nn_destroyed = arr(destroyed, "nn_dist_to_train_truth_R_phys")
    nn_survived = arr(survived, "nn_dist_to_train_truth_R_phys")

    # Loss alignment: fake vs truth saturation needs fake arrays from details — passed separately later.
    return {
        "train_truth": _record_stats(train_truth),
        "dev_truth": _record_stats(dev_truth),
        "wb74_fold1_on_dev_truth": _record_stats(f1_truth),
        "wb74_fold2_on_dev_truth": _record_stats(f2_truth),
        "failure_cohort_catastrophic": _record_stats(destroyed),
        "survived_w64_selected": _record_stats(survived),
        "sat_neg_truth_dev": _record_stats(sat_neg),
        "train_truth_by_dsid": {k: _record_stats(v) for k, v in sorted(by_dsid.items())},
        "dev_truth_by_dsid": {k: _record_stats(v) for k, v in sorted(by_dsid_dev.items())},
        "station_pair_logits_destroyed": edge_block(destroyed),
        "station_pair_logits_survived": edge_block(survived),
        "station_pair_logits_train_truth": edge_block(train_truth),
        "margin_strata_dev": {k: _record_stats(v) for k, v in sorted(margin.items())},
        "w64_fragment_competition_dev": _record_stats(fragment),
        "charge": charge_split,
        "rphys_shift_ks_top": ks[:12],
        "rphys_shift_ks_all": ks,
        "nn_distance_dev_to_train_truth": _summarize_array(nn_arr),
        "nn_distance_destroyed": _summarize_array(nn_destroyed),
        "nn_distance_survived": _summarize_array(nn_survived),
        "corr_delta_vs_nn": corr(nn_arr, dlt),
        "corr_delta_vs_L_edge": corr(ledge, dlt),
        "ood_implies_destruction": {
            "nn_q90_threshold": float(np.quantile(nn_arr, 0.90)) if nn_arr.size else None,
            "frac_destroyed_in_nn_top10pct": (
                float(np.mean([bool(r.get("catastrophic")) for r, dist in zip([x for x in dev_truth if "nn_dist_to_train_truth_R_phys" in x], nn_arr) if dist >= np.quantile(nn_arr, 0.90)]))
                if nn_arr.size
                else None
            ),
            "frac_destroyed_overall_among_w64_selected": (
                (len(destroyed) / max(1, len(destroyed) + len(survived)))
            ),
        },
    }


def _fake_truth_sat(scored, bundle, tables) -> dict[str, Any]:
    details = scored["details"]
    if details is None:
        return {}
    truth_d, fake_d = [], []
    for g in bundle.graphs:
        det = details.get(id(g))
        if det is None:
            continue
        lab = np.asarray(tables[id(g)].labels, dtype=bool)
        d = np.asarray(det["bounded_delta"], dtype=np.float64)
        truth_d.append(d[lab])
        fake_d.append(d[~lab])
    t = np.concatenate(truth_d) if truth_d else np.empty(0)
    f = np.concatenate(fake_d) if fake_d else np.empty(0)
    return {
        "truth": _summarize_array(t, BOUND),
        "fake": _summarize_array(f, BOUND),
        "truth_frac_sat_neg": float(np.mean(t <= -THR)) if t.size else None,
        "fake_frac_sat_neg": float(np.mean(f <= -THR)) if f.size else None,
    }


def decide(analysis: dict[str, Any], sat: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    train_sat = (analysis["train_truth"] or {}).get("frac_sat_neg")
    dev_sat = (analysis["dev_truth"] or {}).get("frac_sat_neg")
    f1_sat = (analysis["wb74_fold1_on_dev_truth"] or {}).get("frac_sat_neg")
    f1_cat = (analysis["wb74_fold1_on_dev_truth"] or {}).get("catastrophic")
    f1_n = (analysis["wb74_fold1_on_dev_truth"] or {}).get("n_routes") or 0
    ks_top = analysis.get("rphys_shift_ks_top") or []
    max_ks = max((row["ks_statistic"] for row in ks_top), default=0.0)
    nn_d = (analysis.get("nn_distance_destroyed") or {}).get("median")
    nn_s = (analysis.get("nn_distance_survived") or {}).get("median")
    nn_gap = None if nn_d is None or nn_s is None else float(nn_d) - float(nn_s)
    fake_sat_train = (sat.get("wb75_train") or {}).get("fake_frac_sat_neg")
    truth_sat_train = (sat.get("wb75_train") or {}).get("truth_frac_sat_neg")

    # Pre-registered attribution rules (frozen before looking at numbers in code;
    # the inequalities are the decision procedure).
    mixture_or_objective = bool(f1_n and (f1_sat is not None) and f1_sat <= 0.01 and (dev_sat or 0) > 0.03)
    in_sample_destruction = bool(train_sat is not None and train_sat > 0.01)
    strong_rphys_shift = bool(max_ks >= 0.25)
    ood_drives_delta = bool(nn_gap is not None and nn_gap > 0.0 and (analysis.get("corr_delta_vs_nn") or 0) < -0.2)
    loss_fake_vs_truth = bool(
        fake_sat_train is not None and truth_sat_train is not None and fake_sat_train >= 0.9 and truth_sat_train <= 0.02
        and (dev_sat or 0) > 0.03
    )

    hypotheses = {
        "data_boundary_shift": {
            "status": "supported" if strong_rphys_shift or ood_drives_delta else "not_primary",
            "evidence": {
                "max_ks": max_ks,
                "nn_median_gap_destroyed_minus_survived": nn_gap,
                "corr_delta_vs_nn": analysis.get("corr_delta_vs_nn"),
            },
        },
        "source_mixture_conflict": {
            "status": "supported" if mixture_or_objective and not (f1_sat and f1_sat > 0.03) else "not_primary",
            "evidence": {
                "wb74_family2_only_head_on_dev_sat_neg": f1_sat,
                "wb75_six_source_head_on_dev_sat_neg": dev_sat,
                "wb74_fold1_catastrophic": f1_cat,
            },
        },
        "nonlinear_extrapolation": {
            "status": "supported" if ood_drives_delta and (dev_sat or 0) > (train_sat or 0) + 0.02 else "not_primary",
            "evidence": {
                "train_truth_sat_neg": train_sat,
                "dev_truth_sat_neg": dev_sat,
                "corr_delta_vs_nn": analysis.get("corr_delta_vs_nn"),
            },
        },
        "loss_solver_mismatch": {
            "status": "supported" if in_sample_destruction or (fake_sat_train is not None and fake_sat_train >= 0.9) else "not_primary",
            "evidence": {
                "train_truth_sat_neg": train_sat,
                "train_fake_sat_neg": fake_sat_train,
                "dev_truth_sat_neg": dev_sat,
                "in_sample_truth_destruction": in_sample_destruction,
            },
        },
    }

    # Single required decision.
    if in_sample_destruction or (mixture_or_objective and not strong_rphys_shift):
        decision = "A"
        label = "Representation valid, training objective failed"
    elif strong_rphys_shift and (f1_sat or 0) > 0.03:
        decision = "C"
        label = "Data contract mismatch"
    elif ood_drives_delta and not mixture_or_objective:
        decision = "C"
        label = "Data contract mismatch"
    else:
        # Default: six-source objective / mixture over a still-valid representation.
        if max_ks < 0.25 and (f1_sat is None or f1_sat <= 0.02):
            decision = "A"
            label = "Representation valid, training objective failed"
        else:
            decision = "B"
            label = "Representation insufficient"

    return {
        "hypotheses": hypotheses,
        "decision_case": decision,
        "decision_label": label,
        "rules_fired": {
            "mixture_or_objective": mixture_or_objective,
            "in_sample_destruction": in_sample_destruction,
            "strong_rphys_shift": strong_rphys_shift,
            "ood_drives_delta": ood_drives_delta,
            "loss_fake_vs_truth": loss_fake_vs_truth,
        },
        "checkpoint_contract_ok": bool(contract.get("sha_ok") and contract.get("mode_ok") and contract.get("bound_ok")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-events-per-payload", type=int, default=None)
    parser.add_argument("--output-dir", default=str(OUT))
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    paths = [W64_ROOT, WB75_CKPT, WB74_F1_CKPT, WB74_F2_CKPT, TRAIN_MANIFEST, DEV_MANIFEST, output]
    refuse_forbidden_paths(paths)
    for p in paths:
        if any(n in str(p) for n in UNUSED_RESERVE_SOURCES) or "00800_00849" in str(p):
            raise SystemExit(f"refusing Final Blind path: {p}")

    print("=== Workbook 76 failure-mechanism audit ===", flush=True)
    print(f"hostname {socket.gethostname()}", flush=True)
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT, text=True
    )
    print(f"git_commit {git_commit} dirty={bool(dirty.strip())}", flush=True)

    contract = checkpoint_contract(args.device)
    print(f"checkpoint contract mode={contract['route_representation_mode']} sha_ok={contract['sha_ok']}", flush=True)
    _write_json(output / "checkpoint_contract.json", contract)

    config = frozen_packing_config()
    w64 = _load_arm_flexible("w64", w64_root=W64_ROOT, head_checkpoint=None, device=args.device)
    wb75 = _load_arm_flexible("primary", w64_root=W64_ROOT, head_checkpoint=WB75_CKPT, device=args.device)
    f1 = _load_arm_flexible("primary", w64_root=W64_ROOT, head_checkpoint=WB74_F1_CKPT, device=args.device)
    f2 = _load_arm_flexible("primary", w64_root=W64_ROOT, head_checkpoint=WB74_F2_CKPT, device=args.device)
    for label, loaded in (("wb75", wb75), ("wb74f1", f1), ("wb74f2", f2)):
        mode = str(getattr(loaded["artifact"].model_config, "route_representation_mode", ""))
        if mode != "physical_pair_relative":
            raise SystemExit(f"{label} is not physical_pair_relative: {mode}")
    if w64["checkpoint_sha256"] != W64_SHA:
        raise SystemExit("W64 sha mismatch")

    print("--- load TRAIN six-source ---", flush=True)
    train_samples, train_bundle, train_tables, _ = _load_split(str(TRAIN_MANIFEST), "train", args.max_events_per_payload)
    train_const = {str(s) for sample in train_samples for s in sample.source_ids}
    if train_const & set(RESERVED_BLIND_SOURCES) or train_const & set(UNUSED_RESERVE_SOURCES):
        raise SystemExit("train corpus leaked development/final-blind")
    if any(is_sealed_source(s) for s in train_const):
        raise SystemExit("train corpus leaked sealed")
    train_uid = _uid_source_map(train_samples)
    print(f"train graphs={len(train_bundle.graphs)} sources={sorted(train_const)}", flush=True)

    train_w64_s = _score_named("w64_train", w64, train_bundle, train_tables, args.device, args.batch_size, config, False)
    print(f"  w64_train C={train_w64_s['cd']['C']} D={train_w64_s['cd']['D']}", flush=True)
    train_wb75_s = _score_named("wb75_train", wb75, train_bundle, train_tables, args.device, args.batch_size, config, True)
    print(f"  wb75_train C={train_wb75_s['cd']['C']} D={train_wb75_s['cd']['D']}", flush=True)

    print("--- load DEVELOPMENT 00350_00399 ---", flush=True)
    dev_samples, dev_bundle, dev_tables, _ = _load_split(str(DEV_MANIFEST), "validation", args.max_events_per_payload)
    dev_const = {str(s) for sample in dev_samples for s in sample.source_ids}
    if dev_const != set(RESERVED_BLIND_SOURCES):
        raise SystemExit(f"development sources {sorted(dev_const)} != reserved pair")
    if dev_const & set(UNUSED_RESERVE_SOURCES):
        raise SystemExit("Final Blind leaked")
    dev_uid = _uid_source_map(dev_samples)
    print(f"dev graphs={len(dev_bundle.graphs)} sources={sorted(dev_const)}", flush=True)

    dev_w64_s = _score_named("w64_dev", w64, dev_bundle, dev_tables, args.device, args.batch_size, config, False)
    print(f"  w64_dev C={dev_w64_s['cd']['C']} D={dev_w64_s['cd']['D']}", flush=True)
    dev_wb75_s = _score_named("wb75_dev", wb75, dev_bundle, dev_tables, args.device, args.batch_size, config, True)
    print(f"  wb75_dev C={dev_wb75_s['cd']['C']} D={dev_wb75_s['cd']['D']}", flush=True)
    dev_f1_s = _score_named("wb74f1_dev", f1, dev_bundle, dev_tables, args.device, args.batch_size, config, True)
    print(f"  wb74f1_dev C={dev_f1_s['cd']['C']} D={dev_f1_s['cd']['D']}", flush=True)
    dev_f2_s = _score_named("wb74f2_dev", f2, dev_bundle, dev_tables, args.device, args.batch_size, config, True)
    print(f"  wb74f2_dev C={dev_f2_s['cd']['C']} D={dev_f2_s['cd']['D']}", flush=True)

    train_recs = _attach_truth_records(train_bundle, train_tables, train_uid, train_wb75_s, train_w64_s["rows"])
    dev_recs = _attach_truth_records(dev_bundle, dev_tables, dev_uid, dev_wb75_s, dev_w64_s["rows"])
    f1_recs = _attach_truth_records(dev_bundle, dev_tables, dev_uid, dev_f1_s, dev_w64_s["rows"])
    f2_recs = _attach_truth_records(dev_bundle, dev_tables, dev_uid, dev_f2_s, dev_w64_s["rows"])

    ctd_train = _catastrophic_and_new_c(train_w64_s["rows"], train_wb75_s["rows"])
    ctd_dev = _catastrophic_and_new_c(dev_w64_s["rows"], dev_wb75_s["rows"])
    ctd_f1 = _catastrophic_and_new_c(dev_w64_s["rows"], dev_f1_s["rows"])
    ctd_f2 = _catastrophic_and_new_c(dev_w64_s["rows"], dev_f2_s["rows"])

    sat = {
        "wb75_train": _fake_truth_sat(train_wb75_s, train_bundle, train_tables),
        "wb75_dev": _fake_truth_sat(dev_wb75_s, dev_bundle, dev_tables),
        "wb74f1_dev": _fake_truth_sat(dev_f1_s, dev_bundle, dev_tables),
        "wb74f2_dev": _fake_truth_sat(dev_f2_s, dev_bundle, dev_tables),
    }
    analysis = analyze(train_recs, train_w64_s["rows"], dev_recs, dev_w64_s["rows"], f1_recs, f2_recs)
    decision = decide(analysis, sat, contract)

    arm_table = {
        "w64_train": {"cd": train_w64_s["cd"], "metrics": train_w64_s["metrics"], "sha": train_w64_s["checkpoint_sha256"]},
        "wb75_train": {"cd": train_wb75_s["cd"], "metrics": train_wb75_s["metrics"], "sha": train_wb75_s["checkpoint_sha256"], "ctd": ctd_train},
        "w64_dev": {"cd": dev_w64_s["cd"], "metrics": dev_w64_s["metrics"], "sha": dev_w64_s["checkpoint_sha256"]},
        "wb75_dev": {"cd": dev_wb75_s["cd"], "metrics": dev_wb75_s["metrics"], "sha": dev_wb75_s["checkpoint_sha256"], "ctd": ctd_dev},
        "wb74_fold1_family2trained_dev": {"cd": dev_f1_s["cd"], "metrics": dev_f1_s["metrics"], "sha": dev_f1_s["checkpoint_sha256"], "ctd": ctd_f1},
        "wb74_fold2_family1trained_dev": {"cd": dev_f2_s["cd"], "metrics": dev_f2_s["metrics"], "sha": dev_f2_s["checkpoint_sha256"], "ctd": ctd_f2},
    }

    # Compact npz for destroyed development truth (no full fake dump).
    dest = [r for r in dev_recs if r.get("catastrophic") and "R_phys" in r]
    if dest:
        np.savez_compressed(
            output / "failure_cohort_catastrophic.npz",
            R_phys=np.stack([r["R_phys"] for r in dest]),
            bounded_delta=np.asarray([r["bounded_delta"] for r in dest]),
            raw_delta=np.asarray([r["raw_delta"] for r in dest]),
            L_edge=np.asarray([r["L_edge"] for r in dest]),
            L_corrected=np.asarray([r["L_corrected"] for r in dest]),
            L01=np.asarray([r["L01"] for r in dest]),
            L12=np.asarray([r["L12"] for r in dest]),
            L23=np.asarray([r["L23"] for r in dest]),
            nn=np.asarray([r.get("nn_dist_to_train_truth_R_phys", np.nan) for r in dest]),
        )

    summary = {
        "workbook": 76,
        "git_commit": git_commit,
        "git_dirty": bool(dirty.strip()),
        "hostname": socket.gethostname(),
        "checkpoint_contract": contract,
        "arms": arm_table,
        "saturation": sat,
        "analysis": analysis,
        "decision": decision,
        "continue_to_15d_relative_wls": False,
        "new_final_blind_content_accessed": False,
        "sealed_test_accessed": False,
        "retrained": False,
        "architecture_changed": False,
        "loss_changed": False,
        "bound_changed": False,
    }
    _write_json(output / "failure_mechanism_summary.json", summary)
    _write_json(output / "decision.json", {
        "workbook": 76,
        "decision_case": decision["decision_case"],
        "decision_label": decision["decision_label"],
        "hypotheses": decision["hypotheses"],
        "continue_to_15d_relative_wls": False,
        "final_blind_eval_authorized": False,
        "new_final_blind_content_accessed": False,
        "sealed_test_accessed": False,
        "retrained": False,
    })
    print(json.dumps(_json_ready({
        "decision": decision["decision_case"],
        "label": decision["decision_label"],
        "wb75_train_C": train_wb75_s["cd"]["C"],
        "wb75_dev_C": dev_wb75_s["cd"]["C"],
        "wb74f1_dev_C": dev_f1_s["cd"]["C"],
        "wb74f2_dev_C": dev_f2_s["cd"]["C"],
        "train_truth_sat_neg": analysis["train_truth"].get("frac_sat_neg"),
        "dev_truth_sat_neg": analysis["dev_truth"].get("frac_sat_neg"),
        "f1_dev_sat_neg": analysis["wb74_fold1_on_dev_truth"].get("frac_sat_neg"),
    }), indent=2, sort_keys=True), flush=True)
    print("=== Workbook 76 audit done ===", flush=True)


if __name__ == "__main__":
    main()
