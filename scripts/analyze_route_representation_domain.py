"""Workbook 73: route-representation source-domain shift localization analysis.

Diagnostic-only.  Consumes the per-route representation dump produced by
``scripts/extract_route_representations.py`` (family1 / family2 source-pure
corpora) and runs three pre-registered analysis classes plus the catastrophic
cohort and fold-asymmetry studies:

  1. Direct distribution shift (SMD / median / quantile / Wasserstein / MMD).
  2. Source-family predictability probe (L2 logistic, truth-only + fake-only).
  3. Truth/fake cross-family transfer probe (L2 logistic, both directions).

Levels probed: R0, R1, R2, R3 (full/abs/qual), R4, R5, R6, R_phys.  R7 is the
known-failure output endpoint and is reported for distribution shift only.

Frozen probe contract (chosen once, not tuned on results):
  * training set = per-truth 1:1:1:1 sampling (1 truth, 1 share3, 1 share2,
    1 share1/unrelated fake); if a fake class is short, N = min class size.
  * preprocessing = StandardScaler fit on the TRAIN family sample only.
  * classifier = LogisticRegression(C=1.0, class_weight='balanced', lbfgs).
  * decision threshold = 0.5 for the truth false-negative rate.
These probes are diagnostic_only and not checkpoint candidates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from scipy.stats import wasserstein_distance

SHARE_CLASS_NAMES = ("A_truth", "B_fake_share3", "C_fake_share2", "D_fake_share1", "E_fake_fully_unrelated")
NODE_ABS_IDX = (0, 1, 2, 3, 4)
NODE_QUAL_IDX = tuple(range(5, 17))
EDGE_W = 11
NODE_W = 17


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def _load_family(extract_dir: Path, family: str) -> dict[str, np.ndarray]:
    lo = np.load(extract_dir / f"{family}_lowdim.npz")
    hi = np.load(extract_dir / f"{family}_highdim.npz")
    # join high-dim onto low-dim by uid
    lo_uid = lo["uid"]
    hi_uid = hi["uid"]
    lo_index = {int(u): i for i, u in enumerate(lo_uid)}
    hi_pos = np.array([lo_index[int(u)] for u in hi_uid], dtype=np.int64)
    out: dict[str, np.ndarray] = {}
    # subsample (high-dim present) route set, aligned
    out["sub_label"] = lo["label"][hi_pos]
    out["sub_share_class"] = lo["share_class"][hi_pos]
    out["sub_payload"] = lo["payload_idx"][hi_pos]
    out["sub_R0"] = lo["R0"][hi_pos]
    out["sub_R1"] = lo["R1"][hi_pos]
    out["sub_R2logits"] = lo["R2logits"][hi_pos]
    out["sub_R3"] = lo["R3"][hi_pos]
    out["sub_R7"] = lo["R7"][hi_pos]
    out["sub_R_phys"] = lo["R_phys"][hi_pos]
    out["sub_R2latent"] = hi["R2latent"]
    out["sub_R4"] = hi["R4"]
    out["sub_R5"] = hi["R5"]
    out["sub_R6"] = hi["R6"]
    # full low-dim set (for distribution shift on physical levels)
    out["full_label"] = lo["label"]
    out["full_share_class"] = lo["share_class"]
    out["full_payload"] = lo["payload_idx"]
    out["full_R0"] = lo["R0"]
    out["full_R1"] = lo["R1"]
    out["full_R3"] = lo["R3"]
    out["full_R_phys"] = lo["R_phys"]
    out["full_R7"] = lo["R7"]
    out["full_R2logits"] = lo["R2logits"]
    return out


def _r3_split(R3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R = R3.reshape(R3.shape[0], 4, NODE_W)
    abs_ = R[:, :, list(NODE_ABS_IDX)].reshape(R3.shape[0], -1)
    qual = R[:, :, list(NODE_QUAL_IDX)].reshape(R3.shape[0], -1)
    return abs_, qual


def build_levels(fam: dict[str, np.ndarray], *, prefix: str) -> dict[str, np.ndarray]:
    """Return {level_name: feature matrix} over the family's subsample routes."""
    R3_abs, R3_qual = _r3_split(fam[f"{prefix}_R3"])
    levels = {
        "R0": fam[f"{prefix}_R0"],
        "R1": fam[f"{prefix}_R1"],
        "R2": np.concatenate([fam["sub_R2latent"], fam["sub_R2logits"]], axis=1) if prefix == "sub" else fam[f"{prefix}_R2logits"],
        "R3": fam[f"{prefix}_R3"],
        "R3_abs": R3_abs,
        "R3_qual": R3_qual,
        "R_phys": fam[f"{prefix}_R_phys"],
    }
    if prefix == "sub":
        levels["R4"] = fam["sub_R4"]
        levels["R5"] = fam["sub_R5"]
        levels["R6"] = fam["sub_R6"]
    return levels


# --------------------------------------------------------------------------- #
# frozen probe contract
# --------------------------------------------------------------------------- #
def _balanced_train_index(label: np.ndarray, share_class: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per-truth 1:1:1:1 sampling.  N = min available class size (fallback)."""
    truth_idx = np.where(label == 1)[0]
    share3 = np.where((label == 0) & (share_class == 1))[0]
    share2 = np.where((label == 0) & (share_class == 2))[0]
    share1unrel = np.where((label == 0) & ((share_class == 3) | (share_class == 4)))[0]
    pools = [truth_idx, share3, share2, share1unrel]
    n = min(len(p) for p in pools)
    if n == 0:
        raise ValueError("a share-class pool is empty; cannot build balanced probe set")
    chosen = [rng.choice(p, size=n, replace=False) for p in pools]
    return np.concatenate(chosen)


def _fit_probe(Xtr: np.ndarray, ytr: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs")
    clf.fit(scaler.transform(Xtr), ytr)
    return scaler, clf


def _probe_metrics(scaler: StandardScaler, clf: LogisticRegression, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    p = clf.predict_proba(scaler.transform(X))[:, 1]
    pred = (p > 0.5).astype(int)
    out: dict[str, float] = {}
    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    out["balanced_acc"] = float(balanced_accuracy_score(y, pred))
    truth = y == 1
    out["truth_fnr"] = float((pred[truth] == 0).mean()) if truth.any() else float("nan")
    out["n"] = int(y.size)
    out["n_truth"] = int(truth.sum())
    return out


# --------------------------------------------------------------------------- #
# analysis 3: cross-family truth/fake transfer probe (MOST IMPORTANT)
# --------------------------------------------------------------------------- #
def cross_family_transfer(
    levels1: dict[str, np.ndarray], lab1: np.ndarray, sc1: np.ndarray,
    levels2: dict[str, np.ndarray], lab2: np.ndarray, sc2: np.ndarray,
    *, seed: int = 20260905, in_family_holdout: float = 0.3,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    results: dict[str, object] = {}
    for level in levels1:
        X1, X2 = levels1[level], levels2[level]
        res: dict[str, object] = {}
        for direction, (Xtr_f, ytr_f, sctr_f, Xte_f, yte_f) in {
            "family1_to_family2": (X1, lab1, sc1, X2, lab2),
            "family2_to_family1": (X2, lab2, sc2, X1, lab1),
        }.items():
            # stratified held-out in-family test split (keeps a clean truth test set)
            n = ytr_f.size
            perm = rng.permutation(n)
            n_hold = int(in_family_holdout * n)
            hold_idx = perm[:n_hold]
            train_pool = perm[n_hold:]
            tr_idx = train_pool[_balanced_train_index(ytr_f[train_pool], sctr_f[train_pool], rng)]
            scaler, clf = _fit_probe(Xtr_f[tr_idx], ytr_f[tr_idx])
            in_fam = _probe_metrics(scaler, clf, Xtr_f[hold_idx], ytr_f[hold_idx])
            cross = _probe_metrics(scaler, clf, Xte_f, yte_f)
            res[direction] = {
                "in_family": in_fam,
                "cross_family": cross,
                "delta_auc": in_fam["auc"] - cross["auc"],
                "delta_balanced_acc": in_fam["balanced_acc"] - cross["balanced_acc"],
                "cross_truth_fnr": cross["truth_fnr"],
                "in_family_truth_fnr": in_fam["truth_fnr"],
                "n_train_per_class": int(tr_idx.size // 4),
            }
        # asymmetry of the two directions
        res["asymmetry_cross_truth_fnr"] = (
            res["family2_to_family1"]["cross_truth_fnr"] - res["family1_to_family2"]["cross_truth_fnr"]
        )
        results[level] = res
    return results


# --------------------------------------------------------------------------- #
# analysis 2: source-family predictability probe
# --------------------------------------------------------------------------- #
def source_predictability(
    levels1: dict[str, np.ndarray], lab1: np.ndarray,
    levels2: dict[str, np.ndarray], lab2: np.ndarray,
    *, seed: int = 20260905, max_per_class: int = 20000,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    results: dict[str, object] = {}
    for level in levels1:
        res: dict[str, object] = {}
        for subset, (m1, m2) in {
            "truth_only": (lab1 == 1, lab2 == 1),
            "fake_only": (lab1 == 0, lab2 == 0),
        }.items():
            X = np.concatenate([levels1[level][m1], levels2[level][m2]], axis=0)
            y = np.concatenate([np.zeros(m1.sum(), int), np.ones(m2.sum(), int)])  # 0=family1, 1=family2
            # balance the two families
            n1, n2 = int((y == 0).sum()), int((y == 1).sum())
            n = min(n1, n2, max_per_class)
            i1 = rng.choice(np.where(y == 0)[0], size=n, replace=False)
            i2 = rng.choice(np.where(y == 1)[0], size=n, replace=False)
            idx = np.concatenate([i1, i2])
            rng.shuffle(idx)
            Xb, yb = X[idx], y[idx]
            # train/test split
            split = int(0.7 * n)
            tr = np.concatenate([np.where(yb == 0)[0][:split], np.where(yb == 1)[0][:split]])
            te = np.concatenate([np.where(yb == 0)[0][split:], np.where(yb == 1)[0][split:]])
            scaler, clf = _fit_probe(Xb[tr], yb[tr])
            met = _probe_metrics(scaler, clf, Xb[te], yb[te])
            res[subset] = {"auc": met["auc"], "balanced_acc": met["balanced_acc"], "n_test": met["n"]}
        results[level] = res
    return results


# --------------------------------------------------------------------------- #
# analysis 1: direct distribution shift
# --------------------------------------------------------------------------- #
def _smd(a: np.ndarray, b: np.ndarray) -> float:
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    pooled = np.sqrt((va + vb) / 2.0)
    return float((ma - mb) / pooled) if pooled > 0 else 0.0


def distribution_shift(
    fam1: dict[str, np.ndarray], fam2: dict[str, np.ndarray], *, max_mmd: int = 1000, seed: int = 20260905,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    results: dict[str, object] = {}
    # low-dim physical levels use the FULL pooled distribution
    low_levels = {"R0": ("full_R0",), "R1": ("full_R1",), "R3": ("full_R3",), "R_phys": ("full_R_phys",)}
    for level, (key,) in low_levels.items():
        X1, X2 = fam1[key], fam2[key]
        per_class: dict[str, object] = {}
        for ci, cname in enumerate(SHARE_CLASS_NAMES):
            m1 = fam1["full_share_class"] == ci
            m2 = fam2["full_share_class"] == ci
            if m1.sum() < 20 or m2.sum() < 20:
                continue
            A, B = X1[m1], X2[m2]
            smd = np.array([_smd(A[:, j], B[:, j]) for j in range(A.shape[1])])
            wd = np.array([wasserstein_distance(A[:, j], B[:, j]) for j in range(A.shape[1])])
            per_class[cname] = {
                "n1": int(m1.sum()), "n2": int(m2.sum()),
                "smd_abs_max": float(np.abs(smd).max()),
                "smd_abs_mean": float(np.abs(smd).mean()),
                "wasserstein_max": float(wd.max()),
                "wasserstein_mean": float(wd.mean()),
            }
        results[level] = {"per_share_class": per_class}
    # high-dim latents: MMD on a subsample (truth + fake separately)
    def _mmd(X: np.ndarray, Y: np.ndarray) -> float:
        n = min(len(X), len(Y), max_mmd)
        X = X[rng.choice(len(X), n, replace=False)]
        Y = Y[rng.choice(len(Y), n, replace=False)]
        # median heuristic bandwidth on a subsample
        Z = np.concatenate([X, Y])
        sub = Z[rng.choice(len(Z), min(500, len(Z)), replace=False)]
        d2 = np.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=-1)
        sigma2 = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
        def k(A, B):
            return np.exp(-np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1) / (2 * sigma2))
        return float(k(X, X).mean() + k(Y, Y).mean() - 2 * k(X, Y).mean())

    for level in ("R4", "R5", "R6", "R2latent"):
        X1, X2 = fam1[f"sub_{level}"], fam2[f"sub_{level}"]
        per_class = {}
        for ci, cname in enumerate(SHARE_CLASS_NAMES):
            m1 = fam1["sub_share_class"] == ci
            m2 = fam2["sub_share_class"] == ci
            if m1.sum() < 50 or m2.sum() < 50:
                continue
            per_class[cname] = {"mmd": _mmd(X1[m1], X2[m2]), "n1": int(m1.sum()), "n2": int(m2.sum())}
        results[level] = {"per_share_class_mmd": per_class}
    return results


# --------------------------------------------------------------------------- #
# analysis 4: fold asymmetry (family support / coverage)
# --------------------------------------------------------------------------- #
def _nn_distance(query: np.ndarray, ref: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Squared-distance nearest-neighbour distance, chunked to bound memory."""
    out = np.empty(len(query), dtype=np.float64)
    for i in range(0, len(query), chunk):
        q = query[i:i + chunk]
        d2 = np.sum((q[:, None, :] - ref[None, :, :]) ** 2, axis=-1)
        out[i:i + chunk] = np.sqrt(d2.min(axis=1))
    return out


def fold_asymmetry(
    levels1: dict[str, np.ndarray], lab1: np.ndarray,
    levels2: dict[str, np.ndarray], lab2: np.ndarray,
    *, seed: int = 20260905, max_ref: int = 2000, max_query: int = 1500,
) -> dict[str, object]:
    """Is Family1 truth outside Family2 training truth support (and vice versa)?"""
    rng = np.random.default_rng(seed)
    results: dict[str, object] = {}
    for level in levels1:
        X1t = levels1[level][lab1 == 1]
        X2t = levels2[level][lab2 == 1]
        if len(X1t) < 10 or len(X2t) < 10:
            continue
        scaler = StandardScaler().fit(np.concatenate([X1t, X2t], axis=0))
        Z1, Z2 = scaler.transform(X1t), scaler.transform(X2t)
        res: dict[str, object] = {}
        for name, (query, ref) in {
            "family1truth_to_family2": (Z1, Z2),
            "family2truth_to_family1": (Z2, Z1),
        }.items():
            ref_sub = ref[rng.choice(len(ref), min(len(ref), max_ref), replace=False)]
            q_sub = query[rng.choice(len(query), min(len(query), max_query), replace=False)]
            nn = _nn_distance(q_sub, ref_sub)
            mu = ref_sub.mean(axis=0)
            cov = np.cov(ref_sub.T) + 1e-3 * np.eye(ref_sub.shape[1])
            inv = np.linalg.pinv(cov)
            diff = q_sub - mu
            maha = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv, diff))
            res[name] = {
                "nn_dist_median": float(np.median(nn)),
                "nn_dist_q90": float(np.quantile(nn, 0.9)),
                "mahalanobis_median": float(np.median(maha)),
                "mahalanobis_q90": float(np.quantile(maha, 0.9)),
            }
        results[level] = {
            **res,
            "nn_asymmetry": res["family1truth_to_family2"]["nn_dist_median"] - res["family2truth_to_family1"]["nn_dist_median"],
        }
    return results


# --------------------------------------------------------------------------- #
# analysis 5: catastrophic cohort (fold1 = family1 eval, fold1-primary head)
# --------------------------------------------------------------------------- #
def catastrophic_cohort(
    fam1: dict[str, np.ndarray], levels1: dict[str, np.ndarray],
) -> dict[str, object]:
    """Doomed = W64-viable truth (L_edge>4) that V5A pushes to U<=0 (L_corr<=4).

    Uses the dustbin boundary U = L_corrected - 4 (unmatched_penalty=-1, 4
    stations).  Validated against the Workbook-72 solver count (215)."""
    lab = fam1["sub_label"]
    R7 = fam1["sub_R7"]
    L_edge, L_corr = R7[:, 2], R7[:, 3]
    truth = lab == 1
    w64_viable = truth & (L_edge > 4.0)
    doomed = w64_viable & (L_corr <= 4.0)
    surviving = w64_viable & (L_corr > 4.0)
    out: dict[str, object] = {
        "n_truth": int(truth.sum()),
        "n_w64_viable_truth": int(w64_viable.sum()),
        "n_doomed": int(doomed.sum()),
        "n_surviving": int(surviving.sum()),
        "note": "doomed = W64-viable truth (L_edge>4) with V5A L_corrected<=4 (U<=0)",
    }
    per_level: dict[str, object] = {}
    for level, X in levels1.items():
        if doomed.sum() < 10 or surviving.sum() < 10:
            continue
        A, B = X[doomed], X[surviving]
        smd = np.array([_smd(A[:, j], B[:, j]) for j in range(A.shape[1])])
        per_level[level] = {
            "smd_abs_max": float(np.abs(smd).max()),
            "smd_abs_mean": float(np.abs(smd).mean()),
            "n_doomed": int(doomed.sum()), "n_surviving": int(surviving.sum()),
        }
    out["per_level_doomed_vs_surviving"] = per_level
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-dir", default="outputs/mc24_four_station_route_representation_domain_audit_v1/extract")
    parser.add_argument("--output-dir", default="outputs/mc24_four_station_route_representation_domain_audit_v1")
    args = parser.parse_args()
    extract_dir = Path(args.extract_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fam1 = _load_family(extract_dir, "family1")
    fam2 = _load_family(extract_dir, "family2")
    levels1 = build_levels(fam1, prefix="sub")
    levels2 = build_levels(fam2, prefix="sub")
    lab1, sc1 = fam1["sub_label"], fam1["sub_share_class"]
    lab2, sc2 = fam2["sub_label"], fam2["sub_share_class"]

    def _dump(name: str, payload: object) -> None:
        (out_dir / name).write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
        print(f"wrote {name}")

    print("=== distribution shift ===")
    _dump("distribution_shift_summary.json", distribution_shift(fam1, fam2))
    print("=== source predictability probe ===")
    _dump("source_probe_summary.json", source_predictability(levels1, lab1, levels2, lab2))
    print("=== cross-family truth/fake transfer probe ===")
    _dump("cross_family_truth_fake_probe.json", cross_family_transfer(levels1, lab1, sc1, levels2, lab2, sc2))
    print("=== fold asymmetry ===")
    _dump("fold_asymmetry_summary.json", fold_asymmetry(levels1, lab1, levels2, lab2))
    print("=== catastrophic cohort (fold1) ===")
    _dump("catastrophic_cohort_summary.json", catastrophic_cohort(fam1, levels1))


if __name__ == "__main__":
    main()
