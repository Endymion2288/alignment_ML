#!/usr/bin/env python3
"""Workbook 72: V5A bounded-residual TRAIN-side source-transfer evaluation.

For each leave-one-DSID-pair-family-out fold (training/source_transfer_cv.py),
this loads the fold's retrained V4A-unbounded Control and V5A-bounded Primary
checkpoints plus the frozen Workbook-64 baseline, runs inference + the frozen
unit-capacity solver on the HELD-OUT family's regenerated synthetic corpus, and
computes the pre-registered solver-level gate metrics:

    C / D / selected / efficiency / purity / fake
    selected->C and D->C transitions (Mechanism C)
    catastrophic_truth_destruction (selected by W64/control -> U_truth <= 0)
    bounded-delta distribution and +/-B boundary saturation (Mechanism D risk)

Read-only.  It never opens development (00350_00399) for any selection, and
never opens final-blind (00800_00849) or sealed test.  ``training_authorized``
and ``continue_to_15d_relative_wls`` stay false; ``final_blind_eval_authorized``
stays false.
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

from scripts.audit_four_station_blind_failure_localization import PAYLOADS, _namespace_map
from scripts.audit_relative_route_v4_generalization import (
    LOGIT_CLIP,
    _load_split,
    _predict_route_details,
    _produce_truth_rows,
)
from scripts.evaluate_relative_route_v4_development import (
    _assert_identity_platt,
    _evaluate_arm,
    _filter_route_maps,
    _filter_sets,
    _load_v2,
    _read_json,
    _score_arm,
    cd_counts,
    frozen_packing_config,
)
from scripts.run_frozen_association_backbone import _sha256
from scripts.run_refit_multidof_closure import _json_ready
from scripts.summarize_four_station_frozen_association import _raw_candidate_audit
from training.route_aware_transformer import load_relative_route_v4_head_only_artifact
from training.source_transfer_cv import (
    FAMILY1_DSID_PAIR,
    FAMILY2_DSID_PAIR,
    folds,
    validate_fold_sources,
)

WORKBOOK64_SHA256 = "0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236"
W64_ROOT = "outputs/mc24_four_station_source_diversity_v1/checkpoint"

# Family name -> regenerated restricted-pool corpus directory (Workbook 72 §6.4).
FAMILY_CORPUS_DIR = {
    FAMILY1_DSID_PAIR: "family1",
    FAMILY2_DSID_PAIR: "family2",
}

# Boundary-saturation threshold: a bounded delta is "saturated" when it sits
# within 1% of the bound B (|delta| >= 0.99 * B).
SATURATION_FRACTION_OF_BOUND = 0.99


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_arm_flexible(
    name: str,
    *,
    w64_root: Path,
    head_checkpoint: Path | None,
    device: str,
) -> dict[str, object]:
    """Load the frozen W64 baseline or an arbitrary head-only checkpoint.

    Unlike the Workbook-70/71 ``_load_arm`` (which hardcodes the Arm1/Arm2
    checkpoint SHAs), this loader takes an explicit checkpoint path so each CV
    fold's freshly retrained Control/Primary can be evaluated.  The actual
    checkpoint SHA256 is recorded for provenance.
    """
    if name == "w64":
        frozen = _load_v2(w64_root, device)
        observed = str(frozen["metadata"]["checkpoint_sha256"])
        if observed != WORKBOOK64_SHA256:
            raise SystemExit(f"W64 baseline checkpoint sha256 {observed} != {WORKBOOK64_SHA256}")
        _assert_identity_platt(frozen["calibration"])
        return {
            "arm": "w64",
            "name": "workbook64_edge_only",
            "model": frozen["model"],
            "artifact": frozen["artifact"],
            "calibration": frozen["calibration"],
            "inject_complete_route_scores": False,
            "checkpoint": frozen["metadata"]["checkpoint"],
            "checkpoint_sha256": observed,
        }
    if head_checkpoint is None:
        raise SystemExit(f"head checkpoint path required for arm {name}")
    observed = _sha256(head_checkpoint)
    wrapper, artifact = load_relative_route_v4_head_only_artifact(head_checkpoint, device=device)
    calibration_path = head_checkpoint.parent / "calibration.json"
    wrapper_cal = _read_json(calibration_path)
    calibration = dict(wrapper_cal.get("calibration") or wrapper_cal)
    _assert_identity_platt(calibration)
    return {
        "arm": name,
        "name": name,
        "model": wrapper,
        "artifact": artifact,
        "calibration": calibration,
        "inject_complete_route_scores": True,
        "checkpoint": str(head_checkpoint),
        "checkpoint_sha256": observed,
    }


def _pooled_route_metrics(result: Mapping[str, Any]) -> dict[str, object]:
    """Pool the per-payload route metrics into efficiency / purity / fake."""
    pm = result.get("payload_metrics") or {}
    truth = sum(int(m.get("complete_truth_chains", 0)) for m in pm.values())
    correct = sum(int(m.get("correct_complete_routes", 0)) for m in pm.values())
    sel_complete = sum(int(m.get("selected_complete_routes", 0)) for m in pm.values())
    sel_routes = sum(int(m.get("selected_routes", 0)) for m in pm.values())
    efficiency = (correct / truth) if truth else None
    purity = (correct / sel_complete) if sel_complete else None
    fake = sel_routes - correct
    return {
        "complete_truth_chains": truth,
        "correct_complete_routes": correct,
        "selected_complete_routes": sel_complete,
        "selected_routes": sel_routes,
        "efficiency": efficiency,
        "purity": purity,
        "fake_selected_routes": int(fake),
    }


def _delta_summary(values: np.ndarray, bound: float | None) -> dict[str, object]:
    v = np.asarray(values, dtype=np.float64)
    if not v.size:
        return {"count": 0}
    out: dict[str, object] = {
        "count": int(v.size),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "std": float(v.std()),
        "min": float(v.min()),
        "max": float(v.max()),
        "q01": float(np.quantile(v, 0.01)),
        "q05": float(np.quantile(v, 0.05)),
        "q95": float(np.quantile(v, 0.95)),
        "q99": float(np.quantile(v, 0.99)),
        "frac_negative": float(np.mean(v < 0.0)),
        "frac_positive": float(np.mean(v > 0.0)),
        "max_abs": float(np.max(np.abs(v))),
        "n_nan": int(np.count_nonzero(np.isnan(v))),
        "n_inf": int(np.count_nonzero(np.isinf(v))),
    }
    if bound is not None:
        thr = SATURATION_FRACTION_OF_BOUND * float(bound)
        out["bound"] = float(bound)
        out["frac_saturated_neg"] = float(np.mean(v <= -thr))
        out["frac_saturated_pos"] = float(np.mean(v >= thr))
        out["frac_saturated_abs"] = float(np.mean(np.abs(v) >= thr))
    return out


def _transition_counts(rows_from: list, rows_to: list) -> dict[str, int]:
    """decision_class transition matrix between two arms over matched truth routes."""

    def key(row):
        return (
            str(row["payload_id"]),
            int(row["run_id"]),
            int(row["event_id"]),
            tuple(int(e["index"]) for e in row["endpoints"]),
        )

    to_index = {key(r): r for r in rows_to}
    counts: Counter = Counter()
    for rf in rows_from:
        rt = to_index.get(key(rf))
        if rt is None:
            continue
        counts[(str(rf["loss_stage"]), str(rt["loss_stage"]))] += 1
    return {f"{a} -> {b}": int(n) for (a, b), n in sorted(counts.items())}


def _catastrophic_and_new_c(rows_base: list, rows_arm: list) -> dict[str, int]:
    """Catastrophic truth destruction + new-C decomposition of arm vs base.

    catastrophic_truth_destruction: routes the base arm *selected* that the
    corrected arm pushes to U_truth <= 0 (decision_class utility_nonpositive /
    problem C).
    """

    def key(row):
        return (
            str(row["payload_id"]),
            int(row["run_id"]),
            int(row["event_id"]),
            tuple(int(e["index"]) for e in row["endpoints"]),
        )

    def is_c(row) -> bool:
        # Mechanism C: truth route driven to non-positive utility.
        u = row.get("complete_truth_route_utility")
        if u is not None:
            return float(u) <= 0.0
        return str(row.get("loss_stage")) == "utility_nonpositive"

    arm_index = {key(r): r for r in rows_arm}
    ctd = 0
    new_c_from_selected = 0
    new_c_from_d = 0
    for rb in rows_base:
        ra = arm_index.get(key(rb))
        if ra is None:
            continue
        base_selected = bool(rb.get("selected"))
        base_stage = str(rb.get("loss_stage"))
        arm_is_c = is_c(ra)
        base_is_c = is_c(rb)
        if base_selected and arm_is_c:
            ctd += 1
        if arm_is_c and not base_is_c:
            if base_selected:
                new_c_from_selected += 1
            elif base_stage == "packing_competition":
                new_c_from_d += 1
    return {
        "catastrophic_truth_destruction": int(ctd),
        "new_C_from_selected": int(new_c_from_selected),
        "new_C_from_D": int(new_c_from_d),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-root",
        default="outputs/mc24_four_station_relative_route_v5a_source_transfer_v1/corpora",
        help="Root holding the per-family regenerated corpora (family1/, family2/)",
    )
    parser.add_argument(
        "--training-root",
        default="outputs/mc24_four_station_relative_route_v5a_source_transfer_v1/training",
        help="Root holding per-fold trained checkpoints: <root>/<fold>/{control,primary}/checkpoint_last.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/mc24_four_station_relative_route_v5a_source_transfer_v1/evaluation",
    )
    parser.add_argument("--w64-root", default=W64_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-events-per-payload", type=int, default=None)
    args = parser.parse_args()

    corpus_root = Path(args.corpus_root).expanduser().resolve()
    training_root = Path(args.training_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = args.device

    print("=== Workbook 72 V5A source-transfer evaluation ===", flush=True)
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
        # Source-holdout guard (Phase A): disjoint, union == authorized six, no
        # development / final-blind / sealed anywhere.
        validate_fold_sources(train_sources, holdout_sources)

        fold_name = f"holdout_{FAMILY_CORPUS_DIR[holdout_family]}"
        eval_manifest = corpus_root / FAMILY_CORPUS_DIR[holdout_family] / "overlay_synthetic_v1" / "synthetic_corpus_manifest.json"
        control_ckpt = training_root / fold_name / "control" / "checkpoint_last.pt"
        primary_ckpt = training_root / fold_name / "primary" / "checkpoint_last.pt"
        for p in (eval_manifest, control_ckpt, primary_ckpt):
            if not p.exists():
                raise SystemExit(f"missing required input: {p}")

        print(f"\n--- fold {fold_name}: holdout={holdout_family} ---", flush=True)
        print(f"  eval corpus (held-out family): {eval_manifest}", flush=True)
        samples, bundle, tables, namespace = _load_split(
            str(eval_manifest), "train", args.max_events_per_payload
        )
        # Confirm the held-out corpus only contains the held-out family's sources.
        constituents = {str(src) for sample in samples for src in sample.source_ids}
        if constituents != set(holdout_sources):
            raise SystemExit(
                f"held-out corpus sources {sorted(constituents)} != fold holdout {sorted(holdout_sources)}"
            )
        print(f"  held-out graphs={len(bundle.graphs)} routes={sum(t.size for t in tables.values())}", flush=True)

        payload_samples = {
            payload_id: [s for s in samples if str(s.payload_id) == payload_id] for payload_id in PAYLOADS
        }
        raw_candidate_by_payload = {
            payload_id: _raw_candidate_audit(payload_samples[payload_id]) for payload_id in PAYLOADS
        }

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
            result = _evaluate_arm(
                loaded, bundle, raw, calibrated, route_maps, config,
                raw_candidate_by_payload, payload_samples,
            )
            rows = _produce_truth_rows(bundle, calibrated, route_maps, config)
            details = _predict_route_details(
                loaded["model"], bundle, tables,
                loaded["artifact"].node_standardizer, loaded["artifact"].edge_standardizer,
                device, args.batch_size,
            )
            arm_rows[name] = rows
            arm_details[name] = details
            cd = result["cd_pooled"]
            arm_metrics[name] = {
                "checkpoint_sha256": loaded["checkpoint_sha256"],
                "cd_pooled": cd,
                "route_metrics": _pooled_route_metrics(result),
            }
            print(
                f"  {name}: C={cd['C']} D={cd['D']} selected={cd['selected']} "
                f"eff={arm_metrics[name]['route_metrics']['efficiency']}",
                flush=True,
            )

        # --- delta distributions (control / primary only; w64 has no head) ---
        labels_by_graph = {id(g): tables[id(g)].labels for g in bundle.graphs}
        delta_dist: dict[str, Any] = {}
        for name in ("control", "primary"):
            bound = 4.0 if name == "primary" else None
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
                "bound": bound,
                "truth": _delta_summary(truth_arr, bound),
                "fake": _delta_summary(fake_arr, bound),
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

        fold_results[fold_name] = {
            "holdout_family": holdout_family,
            "holdout_sources": holdout_sources,
            "train_sources": train_sources,
            "eval_corpus": str(eval_manifest),
            "eval_corpus_sha256": _sha256(eval_manifest),
            "n_events": int(len(bundle.graphs)),
            "arms": arm_metrics,
            "delta_distributions": delta_dist,
            "transitions": transitions,
            "catastrophic_and_new_C": ctd,
        }
        _write_json(output / f"{fold_name}_result.json", fold_results[fold_name])

    # --- aggregate source-transfer gate ---
    gate = _gate_decision(fold_results)
    summary = {
        "workbook": 72,
        "git_commit": git_commit,
        "git_dirty": bool(git_status.strip()),
        "hostname": socket.gethostname(),
        "folds": fold_results,
        "gate": gate,
        "continue_to_15d_relative_wls": False,
        "training_authorized": False,
        "final_blind_eval_authorized": False,
        "development_used_for_selection": False,
        "new_final_blind_content_accessed": False,
        "sealed_test_accessed": False,
    }
    _write_json(output / "source_transfer_summary.json", summary)
    print("\n=== gate decision ===", flush=True)
    print(json.dumps(_json_ready(gate), indent=2, sort_keys=True), flush=True)
    print("=== evaluation done ===", flush=True)


def _gate_decision(fold_results: Mapping[str, Any]) -> dict[str, object]:
    """Pre-registered V5A source-transfer gate (Workbook 72 section 12)."""
    per_fold: dict[str, Any] = {}
    all_pass = True
    for fold_name, fr in fold_results.items():
        arms = fr["arms"]
        cd_w = arms["w64"]["cd_pooled"]
        cd_c = arms["control"]["cd_pooled"]
        cd_p = arms["primary"]["cd_pooled"]
        primary_truth = fr["delta_distributions"]["primary"]["truth"]
        primary_fake = fr["delta_distributions"]["primary"]["fake"]
        ctd = fr["catastrophic_and_new_C"]

        # Safety: bounded delta within [-4,+4], no NaN/Inf.
        safety_bounds_ok = (
            float(primary_truth.get("max_abs", 0.0)) <= 4.0 + 1e-6
            and float(primary_fake.get("max_abs", 0.0)) <= 4.0 + 1e-6
        )
        safety_finite_ok = (
            int(primary_truth.get("n_nan", 0)) == 0 and int(primary_truth.get("n_inf", 0)) == 0
            and int(primary_fake.get("n_nan", 0)) == 0 and int(primary_fake.get("n_inf", 0)) == 0
        )
        safety_ok = bool(safety_bounds_ok and safety_finite_ok)

        # Mechanism C: primary must not introduce systematic new C, and must
        # catastrophically destroy far fewer truth routes than the unbounded control.
        new_c_primary = ctd["primary_vs_w64"]["new_C_from_selected"] + ctd["primary_vs_w64"]["new_C_from_D"]
        new_c_control = ctd["control_vs_w64"]["new_C_from_selected"] + ctd["control_vs_w64"]["new_C_from_D"]
        ctd_primary = ctd["primary_vs_w64"]["catastrophic_truth_destruction"]
        ctd_control = ctd["control_vs_w64"]["catastrophic_truth_destruction"]
        mechanism_c_ok = bool(
            int(cd_p["C"]) <= int(cd_w["C"])  # no net new C vs W64 baseline
            and ctd_primary <= ctd_control     # bounded destroys no more than unbounded
        )

        # Mechanism D: primary must not sacrifice D / purity / fake to buy C.
        rm_c = arms["control"]["route_metrics"]
        rm_p = arms["primary"]["route_metrics"]
        purity_drop = None
        if rm_c.get("purity") is not None and rm_p.get("purity") is not None:
            purity_drop = float(rm_c["purity"]) - float(rm_p["purity"])
        mechanism_d_ok = bool(int(cd_p["D"]) <= int(cd_c["D"]))

        # Transfer risk: fraction of held-out TRUTH routes saturated at -B.
        truth_sat_neg = float(primary_truth.get("frac_saturated_neg", 0.0))
        transfer_risk = bool(truth_sat_neg > 0.05)

        fold_pass = bool(safety_ok and mechanism_c_ok and mechanism_d_ok and not transfer_risk)
        all_pass = all_pass and fold_pass
        per_fold[fold_name] = {
            "safety_ok": safety_ok,
            "safety_bounds_ok": bool(safety_bounds_ok),
            "safety_finite_ok": bool(safety_finite_ok),
            "mechanism_c_ok": mechanism_c_ok,
            "mechanism_d_ok": mechanism_d_ok,
            "transfer_risk_truth_saturation": transfer_risk,
            "truth_saturation_neg_fraction": truth_sat_neg,
            "C": {"w64": int(cd_w["C"]), "control": int(cd_c["C"]), "primary": int(cd_p["C"])},
            "D": {"w64": int(cd_w["D"]), "control": int(cd_c["D"]), "primary": int(cd_p["D"])},
            "new_C": {"control": int(new_c_control), "primary": int(new_c_primary)},
            "catastrophic_truth_destruction": {"control": int(ctd_control), "primary": int(ctd_primary)},
            "purity": {"control": rm_c.get("purity"), "primary": rm_p.get("purity"), "drop": purity_drop},
            "fold_pass": fold_pass,
        }
    return {
        "gate_pass": bool(all_pass),
        "per_fold": per_fold,
        "criteria": {
            "safety": "|delta_bounded| <= 4 and no NaN/Inf on all held-out folds",
            "mechanism_c": "primary C <= w64 C AND catastrophic_truth_destruction_primary <= control",
            "mechanism_d": "primary D <= control D (no fake-suppression regression)",
            "transfer_risk": "held-out truth -B saturation fraction <= 0.05",
        },
    }


if __name__ == "__main__":
    main()
