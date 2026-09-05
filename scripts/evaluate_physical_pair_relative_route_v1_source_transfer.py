#!/usr/bin/env python3
"""Workbook 74: Physical Pair-Relative Route Encoder V1 train-side source-transfer evaluation.

For each leave-one-DSID-pair-family-out fold (training/source_transfer_cv.py),
this loads the fold's retrained Arm-1 bounded-absolute Control and Arm-2
physical-pair-relative bounded Primary checkpoints plus the frozen Workbook-64
baseline, runs inference + the frozen unit-capacity solver on the HELD-OUT
family's regenerated synthetic corpus, and computes the pre-registered
Workbook-74 source-transfer gate:

    C / D / selected / efficiency / purity / fake
    selected->C and D->C transitions (Mechanism C)
    catastrophic_truth_destruction (selected by W64 -> U_truth <= 0)
    Primary representation diagnostics: raw_delta / bounded_delta / L_edge /
        L_corrected, truth FNR proxy, truth/fake delta distributions,
        fraction truth delta < 0, fraction truth near -B

    Gate A (relative improvement vs bounded absolute control), both folds:
        C_primary < C_control
        catastrophic_primary < catastrophic_control
        new_C_primary < new_C_control
        selected_primary >= selected_control
    Gate B (transfer safety), each fold:
        C_primary / truth_count <= 0.01
        catastrophic_primary / W64_selected_truth <= 0.01
        efficiency_primary >= efficiency_W64 - 0.01
    Gate C (quality guardrail), each fold:
        purity_primary >= purity_W64 - 0.01
        fake_rate_primary <= fake_rate_W64 + 0.01
    Final CV PASS = Fold1 PASS AND Fold2 PASS.

Read-only.  It never opens development (00350_00399) for any selection, and
never opens final-blind (00800_00849) or sealed test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from scripts.audit_relative_route_v4_generalization import (
    _load_split,
    _predict_route_details,
)
from scripts.evaluate_relative_route_v4_development import (
    _assert_identity_platt,
    _filter_route_maps,
    _filter_sets,
    _load_v2,
    _read_json,
    _score_arm,
    cd_counts,
    frozen_packing_config,
)
from scripts.evaluate_relative_route_v5a_source_transfer import (
    _catastrophic_and_new_c,
    _delta_summary,
    _evaluate_arm_single_pass,
    _load_arm_flexible,
    _metrics_from_routemetrics,
    _transition_counts,
)
from scripts.run_frozen_association_backbone import _sha256
from scripts.run_refit_multidof_closure import _json_ready
from training.route_aware_transformer import (
    _forward_route_batch,
    _make_route_batch,
    load_relative_route_v4_head_only_artifact,
)
from training.source_transfer_cv import (
    FAMILY1_DSID_PAIR,
    FAMILY2_DSID_PAIR,
    folds,
    validate_fold_sources,
)

WORKBOOK64_SHA256 = "0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236"
W64_ROOT = "outputs/mc24_four_station_source_diversity_v1/checkpoint"
TRAINING_ROOT = "outputs/mc24_four_station_physical_pair_relative_route_v1_source_transfer/training"
CORPUS_ROOT = "outputs/mc24_four_station_relative_route_v5a_source_transfer_v1/corpora"
OUTPUT_DIR = "outputs/mc24_four_station_physical_pair_relative_route_v1_source_transfer/evaluation"

FAMILY_CORPUS_DIR = {
    FAMILY1_DSID_PAIR: "family1",
    FAMILY2_DSID_PAIR: "family2",
}

# Gate thresholds (pre-registered, Workbook 74 section 21).
GATE_B_MAX_C_FRAC = 0.01
GATE_B_MAX_CATASTROPHIC_FRAC = 0.01
GATE_B_MAX_EFF_DROP = 0.01
GATE_C_MAX_PURITY_DROP = 0.01
GATE_C_MAX_FAKE_RATE_INCREASE = 0.01
SATURATION_FRACTION_OF_BOUND = 0.99
BOUND = 4.0


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _predict_primary_r7(wrapper, bundle, tables, node_std, edge_std, device, batch_size):
    """Capture per-route raw_delta, bounded_delta, L_edge, L_corrected for the Primary.

    raw_delta is the pre-bound route_score output (captured via a forward hook);
    bounded_delta is output.delta_route_logits; L_corrected is output.route_logits
    (= L_edge_W64 + bounded_delta); L_edge = L_corrected - bounded_delta.
    """
    wrapper.eval()
    raw_cap: dict[str, np.ndarray] = {}
    hook = wrapper.trainable.route_score.register_forward_hook(
        lambda module, inputs, output: raw_cap.__setitem__(
            "raw", output.detach().cpu().numpy().reshape(-1)
        )
    )
    details: dict[int, dict[str, np.ndarray]] = {}
    try:
        with torch.no_grad():
            for start in range(0, len(bundle.graphs), batch_size):
                graphs = bundle.graphs[start : start + batch_size]
                batch = _make_route_batch(graphs, node_std, edge_std, device, tables)
                output = _forward_route_batch(wrapper, batch)
                route_logits = output.route_logits.detach().cpu().numpy().astype(np.float64)
                bounded = output.delta_route_logits.detach().cpu().numpy().astype(np.float64)
                raw_delta = raw_cap["raw"].astype(np.float64)
                roff = 0
                for g in graphs:
                    table = tables[id(g)]
                    stop = roff + table.size
                    lc = route_logits[roff:stop]
                    bd = bounded[roff:stop]
                    details[id(g)] = {
                        "raw_delta": raw_delta[roff:stop],
                        "bounded_delta": bd,
                        "L_corrected": lc,
                        "L_edge": lc - bd,
                    }
                    roff = stop
    finally:
        hook.remove()
    return details


def _primary_representation_diagnostics(primary_details, bundle, tables) -> dict[str, Any]:
    """Workbook-74 section 19: Primary held-out route correction diagnostics."""
    labels_by_graph = {id(g): tables[id(g)].labels for g in bundle.graphs}
    truth = {"raw_delta": [], "bounded_delta": [], "L_edge": [], "L_corrected": []}
    fake = {"raw_delta": [], "bounded_delta": [], "L_edge": [], "L_corrected": []}
    for g in bundle.graphs:
        det = primary_details.get(id(g))
        if det is None:
            continue
        lab = np.asarray(labels_by_graph[id(g)], dtype=bool)
        for key in truth:
            truth[key].append(np.asarray(det[key], dtype=np.float64)[lab])
            fake[key].append(np.asarray(det[key], dtype=np.float64)[~lab])

    def _cat(parts):
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)

    truth = {k: _cat(v) for k, v in truth.items()}
    fake = {k: _cat(v) for k, v in fake.items()}

    thr = SATURATION_FRACTION_OF_BOUND * BOUND
    truth_bounded = truth["bounded_delta"]
    diagnostics = {
        "bound": BOUND,
        "truth": {k: _delta_summary(truth[k], BOUND) for k in truth},
        "fake": {k: _delta_summary(fake[k], BOUND) for k in fake},
        "truth_frac_delta_negative": float(np.mean(truth_bounded < 0.0)) if truth_bounded.size else None,
        "truth_frac_near_neg_bound": float(np.mean(truth_bounded <= -thr)) if truth_bounded.size else None,
        "truth_frac_near_pos_bound": float(np.mean(truth_bounded >= thr)) if truth_bounded.size else None,
    }
    return diagnostics


def _gate_a_fold(cd_c, cd_p, ctd) -> dict[str, Any]:
    """Gate A (relative improvement vs bounded absolute control) for one fold."""
    c_control = int(cd_c["C"])
    c_primary = int(cd_p["C"])
    cat_control = int(ctd["control_vs_w64"]["catastrophic_truth_destruction"])
    cat_primary = int(ctd["primary_vs_w64"]["catastrophic_truth_destruction"])
    newc_control = int(ctd["control_vs_w64"]["new_C_from_selected"]) + int(ctd["control_vs_w64"]["new_C_from_D"])
    newc_primary = int(ctd["primary_vs_w64"]["new_C_from_selected"]) + int(ctd["primary_vs_w64"]["new_C_from_D"])
    sel_control = int(cd_c["selected"])
    sel_primary = int(cd_p["selected"])
    criteria = {
        "C_primary_lt_control": bool(c_primary < c_control),
        "catastrophic_primary_lt_control": bool(cat_primary < cat_control),
        "new_C_primary_lt_control": bool(newc_primary < newc_control),
        "selected_primary_ge_control": bool(sel_primary >= sel_control),
    }
    return {
        "pass": bool(all(criteria.values())),
        "criteria": criteria,
        "values": {
            "C": {"control": c_control, "primary": c_primary},
            "catastrophic": {"control": cat_control, "primary": cat_primary},
            "new_C": {"control": newc_control, "primary": newc_primary},
            "selected": {"control": sel_control, "primary": sel_primary},
        },
    }


def _gate_b_fold(cd_w, cd_p, metrics_w, metrics_p, ctd, w64_selected_truth) -> dict[str, Any]:
    """Gate B (transfer safety) for one fold."""
    truth_count = int(metrics_w["complete_truth_chains"])
    c_primary = int(cd_p["C"])
    cat_primary = int(ctd["primary_vs_w64"]["catastrophic_truth_destruction"])
    eff_w = metrics_w["efficiency"]
    eff_p = metrics_p["efficiency"]
    c_frac = (c_primary / truth_count) if truth_count else None
    cat_frac = (cat_primary / w64_selected_truth) if w64_selected_truth else None
    eff_drop = (float(eff_w) - float(eff_p)) if (eff_w is not None and eff_p is not None) else None
    criteria = {
        "C_primary_frac_le_1pct": bool(c_frac is not None and c_frac <= GATE_B_MAX_C_FRAC),
        "catastrophic_frac_le_1pct": bool(cat_frac is not None and cat_frac <= GATE_B_MAX_CATASTROPHIC_FRAC),
        "efficiency_primary_ge_w64_minus_1pct": bool(eff_drop is not None and eff_drop <= GATE_B_MAX_EFF_DROP),
    }
    return {
        "pass": bool(all(criteria.values())),
        "criteria": criteria,
        "values": {
            "truth_count": truth_count,
            "w64_selected_truth": int(w64_selected_truth),
            "C_primary": c_primary,
            "C_primary_frac": c_frac,
            "catastrophic_primary": cat_primary,
            "catastrophic_primary_frac": cat_frac,
            "efficiency_w64": eff_w,
            "efficiency_primary": eff_p,
            "efficiency_drop": eff_drop,
        },
    }


def _gate_c_fold(metrics_w, metrics_p) -> dict[str, Any]:
    """Gate C (quality guardrail) for one fold."""
    pur_w = metrics_w["purity"]
    pur_p = metrics_p["purity"]
    sel_w = int(metrics_w["selected_routes"])
    sel_p = int(metrics_p["selected_routes"])
    fake_w = int(metrics_w["fake_selected_routes"])
    fake_p = int(metrics_p["fake_selected_routes"])
    fake_rate_w = (fake_w / sel_w) if sel_w else None
    fake_rate_p = (fake_p / sel_p) if sel_p else None
    purity_drop = (float(pur_w) - float(pur_p)) if (pur_w is not None and pur_p is not None) else None
    fake_rate_increase = (fake_rate_p - fake_rate_w) if (fake_rate_w is not None and fake_rate_p is not None) else None
    criteria = {
        "purity_primary_ge_w64_minus_1pct": bool(purity_drop is not None and purity_drop <= GATE_C_MAX_PURITY_DROP),
        "fake_rate_increase_le_1pct": bool(fake_rate_increase is not None and fake_rate_increase <= GATE_C_MAX_FAKE_RATE_INCREASE),
    }
    return {
        "pass": bool(all(criteria.values())),
        "criteria": criteria,
        "values": {
            "purity_w64": pur_w,
            "purity_primary": pur_p,
            "purity_drop": purity_drop,
            "fake_rate_w64": fake_rate_w,
            "fake_rate_primary": fake_rate_p,
            "fake_rate_increase": fake_rate_increase,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", default=CORPUS_ROOT)
    parser.add_argument("--training-root", default=TRAINING_ROOT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--w64-root", default=W64_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-events-per-payload", type=int, default=None)
    args = parser.parse_args()

    corpus_root = Path(args.corpus_root).expanduser().resolve()
    training_root = Path(args.training_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = args.device

    print("=== Workbook 74 Physical Pair-Relative source-transfer evaluation ===", flush=True)
    print(f"hostname {socket.gethostname()}", flush=True)
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    git_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True)
    print(f"git_commit {git_commit} dirty={bool(git_status.strip())}", flush=True)

    config = frozen_packing_config()
    w64_root = Path(args.w64_root).expanduser().resolve()

    fold_results: dict[str, Any] = {}
    for fold in folds():
        holdout_family = str(fold["holdout_family"])
        train_sources = [str(s) for s in fold["train_sources"]]
        holdout_sources = [str(s) for s in fold["holdout_sources"]]
        validate_fold_sources(train_sources, holdout_sources)

        fold_name = f"holdout_{FAMILY_CORPUS_DIR[holdout_family]}"
        eval_manifest = corpus_root / FAMILY_CORPUS_DIR[holdout_family] / "overlay_synthetic_v1" / "synthetic_corpus_manifest.json"
        control_ckpt = training_root / fold_name / "control" / "checkpoint_last.pt"
        primary_ckpt = training_root / fold_name / "primary" / "checkpoint_last.pt"
        for p in (eval_manifest, control_ckpt, primary_ckpt):
            if not p.exists():
                raise SystemExit(f"missing required input: {p}")

        print(f"\n--- fold {fold_name}: holdout={holdout_family} ---", flush=True)
        samples, bundle, tables, namespace = _load_split(
            str(eval_manifest), "train", args.max_events_per_payload
        )
        constituents = {str(src) for sample in samples for src in sample.source_ids}
        if constituents != set(holdout_sources):
            raise SystemExit(
                f"held-out corpus sources {sorted(constituents)} != fold holdout {sorted(holdout_sources)}"
            )
        print(f"  held-out graphs={len(bundle.graphs)} routes={sum(t.size for t in tables.values())}", flush=True)

        arms = {
            "w64": _load_arm_flexible("w64", w64_root=w64_root, head_checkpoint=None, device=device),
            "control": _load_arm_flexible("control", w64_root=w64_root, head_checkpoint=control_ckpt, device=device),
            "primary": _load_arm_flexible("primary", w64_root=w64_root, head_checkpoint=primary_ckpt, device=device),
        }
        for name, loaded in arms.items():
            print(f"  loaded {name} sha={loaded['checkpoint_sha256'][:12]}", flush=True)

        arm_rows: dict[str, list] = {}
        arm_metrics: dict[str, Any] = {}
        arm_details: dict[str, Any] = {}
        for name, loaded in arms.items():
            raw, calibrated, route_maps = _score_arm(loaded, bundle, device, args.batch_size)
            rows, metrics = _evaluate_arm_single_pass(bundle, calibrated, route_maps, config)
            details = _predict_route_details(
                loaded["model"], bundle, tables,
                loaded["artifact"].node_standardizer, loaded["artifact"].edge_standardizer,
                device, args.batch_size,
            )
            arm_rows[name] = rows
            arm_details[name] = details
            cd = cd_counts(rows)
            arm_metrics[name] = {
                "checkpoint_sha256": loaded["checkpoint_sha256"],
                "cd_pooled": cd,
                "route_metrics": _metrics_from_routemetrics(metrics),
            }
            print(
                f"  {name}: C={cd['C']} D={cd['D']} selected={cd['selected']} "
                f"eff={arm_metrics[name]['route_metrics']['efficiency']}",
                flush=True,
            )

        # --- Primary representation diagnostics (Workbook 74 section 19) ---
        primary_r7 = _predict_primary_r7(
            arms["primary"]["model"], bundle, tables,
            arms["primary"]["artifact"].node_standardizer, arms["primary"]["artifact"].edge_standardizer,
            device, args.batch_size,
        )
        primary_diag = _primary_representation_diagnostics(primary_r7, bundle, tables)

        # --- delta distributions (control / primary bounded delta) ---
        labels_by_graph = {id(g): tables[id(g)].labels for g in bundle.graphs}
        delta_dist: dict[str, Any] = {}
        for name in ("control", "primary"):
            truth_d, fake_d = [], []
            for g in bundle.graphs:
                det = arm_details[name].get(id(g))
                if det is None or det.get("delta") is None:
                    continue
                d = np.asarray(det["delta"], dtype=np.float64)
                lab = np.asarray(labels_by_graph[id(g)], dtype=bool)
                truth_d.append(d[lab])
                fake_d.append(d[~lab])
            truth_arr = np.concatenate(truth_d) if truth_d else np.empty(0)
            fake_arr = np.concatenate(fake_d) if fake_d else np.empty(0)
            delta_dist[name] = {
                "bound": BOUND,
                "truth": _delta_summary(truth_arr, BOUND),
                "fake": _delta_summary(fake_arr, BOUND),
            }

        # --- transitions + catastrophic truth destruction + new C ---
        transitions = {
            "w64_to_control": _transition_counts(arm_rows["w64"], arm_rows["control"]),
            "w64_to_primary": _transition_counts(arm_rows["w64"], arm_rows["primary"]),
            "control_to_primary": _transition_counts(arm_rows["control"], arm_rows["primary"]),
        }
        ctd = {
            "control_vs_w64": _catastrophic_and_new_c(arm_rows["w64"], arm_rows["control"]),
            "primary_vs_w64": _catastrophic_and_new_c(arm_rows["w64"], arm_rows["primary"]),
            "primary_vs_control": _catastrophic_and_new_c(arm_rows["control"], arm_rows["primary"]),
        }

        # W64-selected truth count (for Gate B catastrophic fraction + truth FNR proxy).
        w64_selected_truth = sum(1 for r in arm_rows["w64"] if r.get("selected"))
        # Truth FNR proxy: W64-selected truth routes pushed to U<=0 by the Primary.
        truth_fnr_proxy = (
            ctd["primary_vs_w64"]["catastrophic_truth_destruction"] / w64_selected_truth
            if w64_selected_truth else None
        )

        cd_w = arm_metrics["w64"]["cd_pooled"]
        cd_c = arm_metrics["control"]["cd_pooled"]
        cd_p = arm_metrics["primary"]["cd_pooled"]
        rm_w = arm_metrics["w64"]["route_metrics"]
        rm_c = arm_metrics["control"]["route_metrics"]
        rm_p = arm_metrics["primary"]["route_metrics"]

        gate_a = _gate_a_fold(cd_c, cd_p, ctd)
        gate_b = _gate_b_fold(cd_w, cd_p, rm_w, rm_p, ctd, w64_selected_truth)
        gate_c = _gate_c_fold(rm_w, rm_p)
        fold_pass = bool(gate_a["pass"] and gate_b["pass"] and gate_c["pass"])

        fold_results[fold_name] = {
            "holdout_family": holdout_family,
            "holdout_sources": holdout_sources,
            "train_sources": train_sources,
            "eval_corpus": str(eval_manifest),
            "eval_corpus_sha256": _sha256(eval_manifest),
            "n_events": int(len(bundle.graphs)),
            "arms": arm_metrics,
            "w64_selected_truth": int(w64_selected_truth),
            "truth_fnr_proxy_primary": truth_fnr_proxy,
            "delta_distributions": delta_dist,
            "primary_representation_diagnostics": primary_diag,
            "transitions": transitions,
            "catastrophic_and_new_C": ctd,
            "gate_A": gate_a,
            "gate_B": gate_b,
            "gate_C": gate_c,
            "fold_pass": fold_pass,
        }
        _write_json(output / fold_name / "evaluation.json", fold_results[fold_name])
        print(
            f"  fold {fold_name}: gateA={gate_a['pass']} gateB={gate_b['pass']} "
            f"gateC={gate_c['pass']} fold_pass={fold_pass}",
            flush=True,
        )

    # --- aggregate source-transfer gate ---
    gate_pass = bool(all(fr["fold_pass"] for fr in fold_results.values()))
    summary = {
        "workbook": 74,
        "git_commit": git_commit,
        "git_dirty": bool(git_status.strip()),
        "hostname": socket.gethostname(),
        "folds": fold_results,
        "source_transfer_gate": gate_pass,
        "gate_thresholds": {
            "gate_B_max_C_frac": GATE_B_MAX_C_FRAC,
            "gate_B_max_catastrophic_frac": GATE_B_MAX_CATASTROPHIC_FRAC,
            "gate_B_max_efficiency_drop": GATE_B_MAX_EFF_DROP,
            "gate_C_max_purity_drop": GATE_C_MAX_PURITY_DROP,
            "gate_C_max_fake_rate_increase": GATE_C_MAX_FAKE_RATE_INCREASE,
        },
        "continue_to_15d_relative_wls": False,
        "training_authorized": False,
        "final_blind_eval_authorized": False,
        "development_used_for_selection": False,
        "new_final_blind_content_accessed": False,
        "sealed_test_accessed": False,
    }
    _write_json(output / "source_transfer_summary.json", summary)
    print("\n=== Workbook 74 source-transfer gate ===", flush=True)
    print(json.dumps(_json_ready({"source_transfer_gate": gate_pass,
                                  "folds": {k: v["fold_pass"] for k, v in fold_results.items()}}),
                     indent=2, sort_keys=True), flush=True)
    print("=== evaluation done ===", flush=True)


if __name__ == "__main__":
    main()
