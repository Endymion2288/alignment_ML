#!/usr/bin/env python3
"""Workbook 75 Stage 2: frozen development evaluation of the six-source Primary.

Loads the Workbook-75 six-source Physical Pair-Relative checkpoint plus the
frozen Workbook-64 baseline and evaluates both on the reserved development
pair only:

    mc24_100047_00350_00399
    mc24_100048_00350_00399

Reports efficiency / purity / fake rate / C / D / catastrophic truth
destruction / U_truth<=0 / fragment winner (packing_competition = D) /
delta distributions, then applies the pre-registered development gate
(Workbook-74 Gate B safety + Gate C quality, frozen before this eval):

    C_primary / truth_count <= 0.01
    catastrophic_primary / W64_selected_truth <= 0.01
    efficiency_primary >= efficiency_W64 - 0.01
    purity_primary >= purity_W64 - 0.01
    fake_rate_primary <= fake_rate_W64 + 0.01

Read-only.  Never opens Final Blind (00800_00849) or sealed test.  Does not
retune architecture / loss / B / solver / OP after seeing development.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_relative_route_v4_generalization import _load_split
from scripts.evaluate_physical_pair_relative_route_v1_source_transfer import (
    BOUND,
    GATE_B_MAX_C_FRAC,
    GATE_B_MAX_CATASTROPHIC_FRAC,
    GATE_B_MAX_EFF_DROP,
    GATE_C_MAX_FAKE_RATE_INCREASE,
    GATE_C_MAX_PURITY_DROP,
    _predict_primary_r7,
    _primary_representation_diagnostics,
)
from scripts.evaluate_relative_route_v4_development import (
    _assert_identity_platt,
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
    _transition_counts,
)
from scripts.run_frozen_association_backbone import _sha256
from scripts.run_refit_multidof_closure import _json_ready
from training.source_diversity_audit import RESERVED_BLIND_SOURCES, UNUSED_RESERVE_SOURCES

W64_ROOT = "outputs/mc24_four_station_source_diversity_v1/checkpoint"
DEV_MANIFEST = (
    "outputs/mc24_four_station_source_diversity_blind_v1/"
    "overlay_synthetic_v1/synthetic_corpus_manifest.json"
)
PRIMARY_CKPT = (
    "outputs/mc24_four_station_physical_pair_relative_route_v1_six_source/"
    "training/primary/checkpoint_last.pt"
)
OUTPUT_DIR = (
    "outputs/mc24_four_station_physical_pair_relative_route_v1_six_source/development_eval"
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _development_gates(cd_w, cd_p, rm_w, rm_p, ctd, w64_selected_truth) -> dict[str, Any]:
    truth_count = int(rm_w["complete_truth_chains"])
    c_primary = int(cd_p["C"])
    c_w64 = int(cd_w["C"])
    d_primary = int(cd_p["D"])
    d_w64 = int(cd_w["D"])
    cat_primary = int(ctd["primary_vs_w64"]["catastrophic_truth_destruction"])
    c_frac = (c_primary / truth_count) if truth_count else None
    cat_frac = (cat_primary / w64_selected_truth) if w64_selected_truth else None
    eff_w, eff_p = rm_w["efficiency"], rm_p["efficiency"]
    eff_drop = (float(eff_w) - float(eff_p)) if (eff_w is not None and eff_p is not None) else None
    pur_w, pur_p = rm_w["purity"], rm_p["purity"]
    purity_drop = (float(pur_w) - float(pur_p)) if (pur_w is not None and pur_p is not None) else None
    sel_w = int(rm_w["selected_routes"])
    sel_p = int(rm_p["selected_routes"])
    fake_w = int(rm_w["fake_selected_routes"])
    fake_p = int(rm_p["fake_selected_routes"])
    fake_rate_w = (fake_w / sel_w) if sel_w else None
    fake_rate_p = (fake_p / sel_p) if sel_p else None
    fake_rate_increase = (
        (fake_rate_p - fake_rate_w) if (fake_rate_w is not None and fake_rate_p is not None) else None
    )
    safety = {
        "C_primary_frac_le_1pct": bool(c_frac is not None and c_frac <= GATE_B_MAX_C_FRAC),
        "catastrophic_frac_le_1pct": bool(cat_frac is not None and cat_frac <= GATE_B_MAX_CATASTROPHIC_FRAC),
        "efficiency_primary_ge_w64_minus_1pct": bool(eff_drop is not None and eff_drop <= GATE_B_MAX_EFF_DROP),
    }
    quality = {
        "purity_primary_ge_w64_minus_1pct": bool(purity_drop is not None and purity_drop <= GATE_C_MAX_PURITY_DROP),
        "fake_rate_increase_le_1pct": bool(
            fake_rate_increase is not None and fake_rate_increase <= GATE_C_MAX_FAKE_RATE_INCREASE
        ),
    }
    cd_vs_w64 = {
        "C_primary_le_w64": bool(c_primary <= c_w64),
        "D_primary_le_w64": bool(d_primary <= d_w64),
        "C_primary": c_primary,
        "C_w64": c_w64,
        "D_primary": d_primary,
        "D_w64": d_w64,
    }
    return {
        "safety": {"pass": bool(all(safety.values())), "criteria": safety},
        "quality": {"pass": bool(all(quality.values())), "criteria": quality},
        "cd_vs_w64": cd_vs_w64,
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
            "purity_w64": pur_w,
            "purity_primary": pur_p,
            "purity_drop": purity_drop,
            "fake_rate_w64": fake_rate_w,
            "fake_rate_primary": fake_rate_p,
            "fake_rate_increase": fake_rate_increase,
        },
        "pass": bool(all(safety.values()) and all(quality.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-manifest", default=DEV_MANIFEST)
    parser.add_argument("--w64-root", default=W64_ROOT)
    parser.add_argument("--primary-checkpoint", default=PRIMARY_CKPT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-events-per-payload", type=int, default=None)
    args = parser.parse_args()

    manifest_path = Path(args.synthetic_manifest).expanduser().resolve()
    w64_root = Path(args.w64_root).expanduser().resolve()
    primary_ckpt = Path(args.primary_checkpoint).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    refuse_forbidden_paths([manifest_path, w64_root, primary_ckpt, output])
    if any(n in str(p) for p in (manifest_path, primary_ckpt, output) for n in UNUSED_RESERVE_SOURCES):
        raise SystemExit("refusing Final Blind path")
    if not primary_ckpt.exists():
        raise SystemExit(f"missing primary checkpoint: {primary_ckpt}")
    output.mkdir(parents=True, exist_ok=True)

    print("=== Workbook 75 frozen development evaluation ===", flush=True)
    print(f"hostname {socket.gethostname()}", flush=True)
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    git_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT, text=True
    )
    print(f"git_commit {git_commit} dirty={bool(git_status.strip())}", flush=True)

    samples, bundle, tables, _ns = _load_split(
        str(manifest_path), "validation", args.max_events_per_payload
    )
    constituents = {str(src) for sample in samples for src in sample.source_ids}
    if constituents != set(RESERVED_BLIND_SOURCES):
        raise SystemExit(f"development sources {sorted(constituents)} != reserved pair")
    if constituents & set(UNUSED_RESERVE_SOURCES):
        raise SystemExit("Final Blind leaked into development evaluation")
    print(f"development graphs={len(bundle.graphs)} routes={sum(t.size for t in tables.values())}", flush=True)

    config = frozen_packing_config()
    arms = {
        "w64": _load_arm_flexible("w64", w64_root=w64_root, head_checkpoint=None, device=args.device),
        "primary": _load_arm_flexible(
            "primary", w64_root=w64_root, head_checkpoint=primary_ckpt, device=args.device
        ),
    }
    mode = str(getattr(arms["primary"]["artifact"].model_config, "route_representation_mode", ""))
    if mode != "physical_pair_relative":
        raise SystemExit(f"primary checkpoint is not physical_pair_relative: {mode}")

    arm_rows: dict[str, list] = {}
    arm_metrics: dict[str, Any] = {}
    arm_details: dict[str, Any] = {}
    from scripts.audit_relative_route_v4_generalization import _predict_route_details

    for name, loaded in arms.items():
        raw, calibrated, route_maps = _score_arm(loaded, bundle, args.device, args.batch_size)
        rows, metrics = _evaluate_arm_single_pass(bundle, calibrated, route_maps, config)
        details = _predict_route_details(
            loaded["model"],
            bundle,
            tables,
            loaded["artifact"].node_standardizer,
            loaded["artifact"].edge_standardizer,
            args.device,
            args.batch_size,
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

    primary_r7 = _predict_primary_r7(
        arms["primary"]["model"],
        bundle,
        tables,
        arms["primary"]["artifact"].node_standardizer,
        arms["primary"]["artifact"].edge_standardizer,
        args.device,
        args.batch_size,
    )
    primary_diag = _primary_representation_diagnostics(primary_r7, bundle, tables)
    labels_by_graph = {id(g): tables[id(g)].labels for g in bundle.graphs}
    truth_d, fake_d = [], []
    for g in bundle.graphs:
        det = arm_details["primary"].get(id(g))
        if det is None or det.get("delta") is None:
            continue
        d = np.asarray(det["delta"], dtype=np.float64)
        lab = np.asarray(labels_by_graph[id(g)], dtype=bool)
        truth_d.append(d[lab])
        fake_d.append(d[~lab])
    truth_arr = np.concatenate(truth_d) if truth_d else np.empty(0)
    fake_arr = np.concatenate(fake_d) if fake_d else np.empty(0)
    delta_dist = {"bound": BOUND, "truth": _delta_summary(truth_arr, BOUND), "fake": _delta_summary(fake_arr, BOUND)}

    transitions = {"w64_to_primary": _transition_counts(arm_rows["w64"], arm_rows["primary"])}
    ctd = {"primary_vs_w64": _catastrophic_and_new_c(arm_rows["w64"], arm_rows["primary"])}
    w64_selected_truth = sum(1 for r in arm_rows["w64"] if r.get("selected"))
    gates = _development_gates(
        arm_metrics["w64"]["cd_pooled"],
        arm_metrics["primary"]["cd_pooled"],
        arm_metrics["w64"]["route_metrics"],
        arm_metrics["primary"]["route_metrics"],
        ctd,
        w64_selected_truth,
    )
    gate_pass = bool(gates["pass"])
    report = {
        "workbook": 75,
        "git_commit": git_commit,
        "git_dirty": bool(git_status.strip()),
        "hostname": socket.gethostname(),
        "development_sources": sorted(constituents),
        "eval_corpus": str(manifest_path),
        "eval_corpus_sha256": _sha256(manifest_path),
        "n_events": int(len(bundle.graphs)),
        "arms": arm_metrics,
        "w64_selected_truth": int(w64_selected_truth),
        "truth_fnr_proxy_primary": (
            ctd["primary_vs_w64"]["catastrophic_truth_destruction"] / w64_selected_truth
            if w64_selected_truth
            else None
        ),
        "delta_distributions": {"primary": delta_dist},
        "primary_representation_diagnostics": primary_diag,
        "transitions": transitions,
        "catastrophic_and_new_C": ctd,
        "development_gate": gates,
        "development_gate_pass": gate_pass,
        "continue_to_15d_relative_wls": False,
        "final_blind_eval_authorized": False,
        "new_final_blind_content_accessed": False,
        "sealed_test_accessed": False,
        "development_used_for_architecture_change": False,
    }
    _write_json(output / "development_evaluation.json", report)
    print(json.dumps(_json_ready({
        "development_gate": gate_pass,
        "safety": gates["safety"]["pass"],
        "quality": gates["quality"]["pass"],
        "cd_vs_w64": gates["cd_vs_w64"],
    }), indent=2, sort_keys=True), flush=True)
    print("=== development evaluation done ===", flush=True)


if __name__ == "__main__":
    main()
