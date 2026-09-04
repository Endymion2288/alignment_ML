#!/usr/bin/env python3
"""Workbook 71 analysis pass: turn ``route_rows.jsonl`` into the audit verdicts.

Reads the read-only artifacts produced by
``audit_relative_route_v4_generalization.py`` and computes the delta-by-class
distributions, truth-route transition matrices, train-vs-development C/D
discrimination, correction-scale audit, S3/2->3 connection, Arm1-vs-Arm2 paired
analysis, and the primary failure classification.  Pure post-processing: no GPU,
no model, no data access beyond the already-written audit rows.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

LOGIT_CLIP = math.log((1.0 - 1.0e-6) / 1.0e-6)
ARMS = ("arm0", "arm1", "arm2")
SHARE_CLASSES = ("A_truth", "B_fake_share3", "C_fake_share2", "D_fake_share1", "E_fake_fully_unrelated")


def _quantiles(values) -> dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    keys = ("count", "mean", "median", "std", "q01", "q05", "q25", "q50", "q75", "q95", "q99", "min", "max", "frac_neg", "frac_pos")
    if not arr.size:
        return {k: None for k in keys}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "q01": float(np.quantile(arr, 0.01)),
        "q05": float(np.quantile(arr, 0.05)),
        "q25": float(np.quantile(arr, 0.25)),
        "q50": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "q95": float(np.quantile(arr, 0.95)),
        "q99": float(np.quantile(arr, 0.99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "frac_neg": float(np.mean(arr < 0.0)),
        "frac_pos": float(np.mean(arr > 0.0)),
    }


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


# --------------------------------------------------------------------------
# 1. delta distributions by truth/fake share class
# --------------------------------------------------------------------------
def delta_distributions(rows) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("train", "development"):
        split_rows = [r for r in rows if r["split"] == split]
        out[split] = {}
        for arm in ("arm1", "arm2"):
            key = f"delta_{arm}"
            out[split][arm] = {}
            for cls in SHARE_CLASSES:
                vals = [r[key] for r in split_rows if r["truth_share_class"] == cls]
                out[split][arm][cls] = _quantiles(vals)
            # pooled truth vs pooled fake
            out[split][arm]["ALL_truth"] = _quantiles([r[key] for r in split_rows if r["label"]])
            out[split][arm]["ALL_fake"] = _quantiles([r[key] for r in split_rows if not r["label"]])
    return out


# --------------------------------------------------------------------------
# 2. Truth-route transition matrices + exact C-source decomposition
# --------------------------------------------------------------------------
def transition_matrices(rows) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("train", "development"):
        truth = [r for r in rows if r["split"] == split and r["label"]]
        trans = {f"{a}->{b}": Counter() for a, b in (("arm0", "arm1"), ("arm1", "arm2"), ("arm0", "arm2"))}
        for r in truth:
            d = {a: r.get(f"decision_class_{a}") for a in ARMS}
            for a, b in (("arm0", "arm1"), ("arm1", "arm2"), ("arm0", "arm2")):
                if d[a] and d[b]:
                    trans[f"{a}->{b}"][(d[a], d[b])] += 1
        # Exact decomposition of NEW Mechanism C routes.
        # Arm1 new C = routes that are C in arm1 but were NOT C in arm0.
        def new_c(earlier, later):
            src = Counter()
            for r in truth:
                de, dl = r.get(f"decision_class_{earlier}"), r.get(f"decision_class_{later}")
                if dl == "utility_nonpositive" and de != "utility_nonpositive":
                    src[de] += 1
            return dict(src)
        out[split] = {
            "n_truth": int(len(truth)),
            "transitions": {k: {f"{a}->{b}": c for (a, b), c in v.items()} for k, v in trans.items()},
            "arm1_new_C_from_arm0_stage": new_c("arm0", "arm1"),
            "arm2_new_C_from_arm1_stage": new_c("arm1", "arm2"),
            "arm2_new_C_from_arm0_stage": new_c("arm0", "arm2"),
            "arm1_C_total": int(sum(1 for r in truth if r.get("decision_class_arm1") == "utility_nonpositive")),
            "arm0_C_total": int(sum(1 for r in truth if r.get("decision_class_arm0") == "utility_nonpositive")),
            "arm2_C_total": int(sum(1 for r in truth if r.get("decision_class_arm2") == "utility_nonpositive")),
        }
    return out


# --------------------------------------------------------------------------
# 3. Train vs development C/D (Case A/B/C discrimination)
# --------------------------------------------------------------------------
def train_vs_dev_cd(rows) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("train", "development"):
        truth = [r for r in rows if r["split"] == split and r["label"]]
        out[split] = {}
        for arm in ARMS:
            c = Counter(r.get(f"decision_class_{arm}") for r in truth)
            out[split][arm] = {
                "selected": int(c.get("selected", 0)),
                "C_utility_nonpositive": int(c.get("utility_nonpositive", 0)),
                "D_packing_competition": int(c.get("packing_competition", 0)),
                "A_candidate_missing": int(c.get("candidate_missing", 0)),
                "B_or_C_below_threshold": int(c.get("below_station_pair_threshold", 0)),
                "total": int(len(truth)),
            }
    # discrimination
    t, d = out["train"], out["development"]
    out["discrimination"] = {
        "train_C_arm0_to_arm1": t["arm1"]["C_utility_nonpositive"] - t["arm0"]["C_utility_nonpositive"],
        "train_C_arm1_to_arm2": t["arm2"]["C_utility_nonpositive"] - t["arm1"]["C_utility_nonpositive"],
        "dev_C_arm0_to_arm1": d["arm1"]["C_utility_nonpositive"] - d["arm0"]["C_utility_nonpositive"],
        "dev_C_arm1_to_arm2": d["arm2"]["C_utility_nonpositive"] - d["arm1"]["C_utility_nonpositive"],
        "train_D_arm0_to_arm1": t["arm1"]["D_packing_competition"] - t["arm0"]["D_packing_competition"],
        "dev_D_arm0_to_arm1": d["arm1"]["D_packing_competition"] - d["arm0"]["D_packing_competition"],
    }
    return out


# --------------------------------------------------------------------------
# 4. Correction scale audit
# --------------------------------------------------------------------------
def correction_scale(rows) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("train", "development"):
        split_rows = [r for r in rows if r["split"] == split]
        out[split] = {}
        for arm in ("arm1", "arm2"):
            key = f"delta_{arm}"
            for cls, subset in (("truth", [r for r in split_rows if r["label"]]),
                                ("fake", [r for r in split_rows if not r["label"]]),
                                ("all", split_rows)):
                deltas = np.asarray([r[key] for r in subset if r[key] is not None], dtype=np.float64)
                if not deltas.size:
                    continue
                l_edge = np.asarray([r["L_edge"] for r in subset if r[key] is not None], dtype=np.float64)
                abs_delta = np.abs(deltas)
                ratio = abs_delta / np.maximum(np.abs(l_edge), 1.0e-6)
                l_corr = l_edge + deltas
                p = np.array([_sigmoid(v) for v in l_corr])
                out[split][f"{arm}_{cls}"] = {
                    "delta": _quantiles(deltas),
                    "abs_delta": _quantiles(abs_delta),
                    "abs_delta_over_abs_L_edge": _quantiles(ratio),
                    "frac_abs_delta_gt_5": float(np.mean(abs_delta > 5)),
                    "frac_abs_delta_gt_10": float(np.mean(abs_delta > 10)),
                    "frac_abs_delta_gt_50": float(np.mean(abs_delta > 50)),
                    "frac_abs_delta_gt_100": float(np.mean(abs_delta > 100)),
                    "frac_p_complete_lt_1e-6": float(np.mean(p < 1e-6)),
                    "frac_p_complete_gt_1m1e-6": float(np.mean(p > 1.0 - 1e-6)),
                }
    return out


# --------------------------------------------------------------------------
# 5. S3 / 2->3 connection
# --------------------------------------------------------------------------
def s3_connection(rows) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("train", "development"):
        truth = [r for r in rows if r["split"] == split and r["label"]]
        out[split] = {}
        fams = sorted({r["payload_family"] for r in truth})
        for fam in fams:
            fam_rows = [r for r in truth if r["payload_family"] == fam]
            out[split][fam] = {}
            for arm in ("arm1", "arm2"):
                key = f"delta_{arm}"
                # stratify by L23 tercile-ish: low (<0), mid (0-2), high (>=2)
                strata = {"L23_neg": [], "L23_0_to_2": [], "L23_ge_2": []}
                for r in fam_rows:
                    if r.get(key) is None or r.get("L23") is None:
                        continue
                    l23 = r["L23"]
                    bucket = "L23_neg" if l23 < 0 else ("L23_0_to_2" if l23 < 2.0 else "L23_ge_2")
                    strata[bucket].append((r[key], l23, r.get("p23"), r.get("chi2_23")))
                out[split][fam][arm] = {
                    name: {
                        "count": len(vals),
                        "delta": _quantiles([v[0] for v in vals]),
                        "L23": _quantiles([v[1] for v in vals]),
                        "p23": _quantiles([v[2] for v in vals]),
                        "chi2_23": _quantiles([v[3] for v in vals]),
                    }
                    for name, vals in strata.items()
                }
        # pooled: does the head rescue low-L23 truth routes?
        for arm in ("arm1", "arm2"):
            key = f"delta_{arm}"
            low = [r[key] for r in truth if r.get(key) is not None and r.get("L23") is not None and r["L23"] < 0]
            high = [r[key] for r in truth if r.get(key) is not None and r.get("L23") is not None and r["L23"] >= 2.0]
            out[split][f"{arm}_delta_when_L23_negative"] = _quantiles(low)
            out[split][f"{arm}_delta_when_L23_high"] = _quantiles(high)
    return out


# --------------------------------------------------------------------------
# 6. Arm2 vs Arm1 paired analysis
# --------------------------------------------------------------------------
def arm2_vs_arm1(rows) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("train", "development"):
        split_rows = [r for r in rows if r["split"] == split and r.get("delta_arm1") is not None and r.get("delta_arm2") is not None]
        out[split] = {}
        diff_all = [r["delta_arm2"] - r["delta_arm1"] for r in split_rows]
        out[split]["ALL"] = _quantiles(diff_all)
        out[split]["truth"] = _quantiles([r["delta_arm2"] - r["delta_arm1"] for r in split_rows if r["label"]])
        out[split]["fake"] = _quantiles([r["delta_arm2"] - r["delta_arm1"] for r in split_rows if not r["label"]])
        # by payload family
        by_fam: dict[str, list] = defaultdict(list)
        for r in split_rows:
            by_fam[r["payload_family"]].append(r["delta_arm2"] - r["delta_arm1"])
        out[split]["by_payload_family"] = {fam: _quantiles(v) for fam, v in sorted(by_fam.items())}
        # by source
        by_src: dict[str, list] = defaultdict(list)
        for r in split_rows:
            by_src[str(r.get("source_id"))].append(r["delta_arm2"] - r["delta_arm1"])
        out[split]["by_source"] = {s: _quantiles(v) for s, v in sorted(by_src.items())}
        # by gauge role
        by_gauge: dict[str, list] = defaultdict(list)
        for r in split_rows:
            by_gauge[str(r.get("gauge_role"))].append(r["delta_arm2"] - r["delta_arm1"])
        out[split]["by_gauge_role"] = {g: _quantiles(v) for g, v in sorted(by_gauge.items())}
        # truth routes by S3 hardness (L23 sign)
        truth = [r for r in split_rows if r["label"]]
        out[split]["truth_L23_negative"] = _quantiles([r["delta_arm2"] - r["delta_arm1"] for r in truth if r.get("L23") is not None and r["L23"] < 0])
        out[split]["truth_L23_high"] = _quantiles([r["delta_arm2"] - r["delta_arm1"] for r in truth if r.get("L23") is not None and r["L23"] >= 2.0])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="outputs/mc24_four_station_relative_route_v4_generalization_audit_v1")
    args = parser.parse_args()
    indir = Path(args.input_dir).expanduser().resolve()
    rows = _load_rows(indir / "route_rows.jsonl")
    print(f"loaded {len(rows)} route rows", flush=True)

    _write = lambda name, payload: (indir / name).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )

    dd = delta_distributions(rows)
    _write("delta_distributions.json", dd)
    print("wrote delta_distributions.json", flush=True)

    tm = transition_matrices(rows)
    _write("truth_route_transition_matrix.json", tm)
    print("wrote truth_route_transition_matrix.json", flush=True)

    cd = train_vs_dev_cd(rows)
    _write("train_vs_dev_cd.json", cd)
    print("wrote train_vs_dev_cd.json", flush=True)

    cs = correction_scale(rows)
    _write("correction_scale_audit.json", cs)
    print("wrote correction_scale_audit.json", flush=True)

    s3 = s3_connection(rows)
    _write("s3_connection_audit.json", s3)
    print("wrote s3_connection_audit.json", flush=True)

    paired = arm2_vs_arm1(rows)
    _write("arm1_vs_arm2_paired_summary.json", paired)
    print("wrote arm1_vs_arm2_paired_summary.json", flush=True)


if __name__ == "__main__":
    main()
