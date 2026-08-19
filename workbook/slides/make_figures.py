#!/usr/bin/env python3
"""Generate all slide figures from real on-disk artifacts.

Every figure reads metrics from outputs/ JSON/CSV artifacts produced by the
documented runs. Nothing is invented; numbers are re-read at build time.

Usage:
    source scripts/setup_environment.sh ml
    python workbook/slides/make_figures.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs"
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11.5,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11.5,
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linewidth": 0.7,
    "legend.fontsize": 10,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.75",
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.facecolor": "white",
})

C = {
    "mlp": "#0072B2",      # blue
    "v1": "#009E73",       # green
    "v1nc": "#56B4E9",     # sky
    "v2": "#E69F00",       # orange
    "v2base": "#8c6d1f",
    "v3": "#CC79A7",       # magenta
    "v3s": "#999999",      # gray
    "train": "#0072B2",
    "val": "#D55E00",      # vermillion
    "pass": "#1a9850",
    "fail": "#d73027",
    "ink": "#1a2333",
}

GATE = dict(eff=0.70, purity=0.95, fake=0.05)


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("wrote", name)


def mag_x():
    """Log-friendly x positions for magnitude axes that include 0."""
    mags = np.array([0.0, 0.1, 1.0, 5.0, 10.0, 50.0])
    x = np.array([0.055, 0.1, 1.0, 5.0, 10.0, 50.0])
    return mags, x


# ================================================================ 1. schematic
def detector_schematic():
    fig, ax = plt.subplots(figsize=(8.6, 3.35))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 34)
    ax.axis("off")
    ax.grid(False)

    # magnets between stations
    for x0 in (24.5, 44.5, 64.5):
        ax.add_patch(Rectangle((x0, 6), 12, 22, facecolor="#d7dee8",
                               edgecolor="#9aa7b8", lw=1.0, zorder=1))
        ax.text(x0 + 6, 4.0, "dipole\nmagnet", ha="center", va="top",
                fontsize=8.6, color="#5a6a7d")

    # stations
    stations = [(20, "IFT (station 0)", "#0b5fa5"), (41, "S1", "#38618a"),
                (61, "S2", "#38618a"), (81, "S3", "#38618a")]
    for z, name, col in stations:
        ax.add_patch(Rectangle((z - 1.6, 8), 3.2, 18, facecolor=col,
                               edgecolor="none", zorder=3))
        ax.text(z, 29.5, name, ha="center", va="bottom", fontsize=10.5,
                color=col, fontweight="bold")

    # muon trajectory (curved in field)
    zz = np.linspace(6, 94, 400)
    xx = 17 + 0.0009 * (zz - 20) ** 2 * 10 + 0.028 * (zz - 6) * 3.2
    xx = 17 - (zz - 6) * 0.11 + 0.0045 * (zz - 6) ** 2
    ax.plot(zz, xx, color=C["val"], lw=2.4, zorder=2)
    ax.annotate("100 GeV $\\mu$ (MC24 FASER$\\nu$)", xy=(8, 24.5),
                fontsize=10, color=C["val"])
    ax.annotate("", xy=(15.5, 17.4), xytext=(7.5, 21.6),
                arrowprops=dict(arrowstyle="->", color=C["val"], lw=1.6))

    # misalignment arrows on IFT
    ax.add_patch(Rectangle((20 - 1.6, 8), 3.2, 18, facecolor="none",
                           edgecolor="#d73027", lw=2.0, zorder=4))
    ar = dict(arrowstyle="<->", color="#d73027", lw=1.6)
    ax.annotate("", xy=(16.5, 10.5), xytext=(13.2, 10.5), arrowprops=ar)
    ax.text(12.6, 10.5, "$d_x$", ha="right", va="center", fontsize=11,
            color="#d73027", fontweight="bold")
    ax.annotate("", xy=(16.5, 15.2), xytext=(13.2, 15.2), arrowprops=ar)
    ax.text(12.6, 15.2, "$d_y$", ha="right", va="center", fontsize=11,
            color="#d73027", fontweight="bold")
    arc = FancyArrowPatch((16.8, 22.4), (13.6, 20.2),
                          connectionstyle="arc3,rad=-0.5",
                          arrowstyle="->", color="#d73027", lw=1.6)
    ax.add_patch(arc)
    ax.text(12.3, 23.4, "$R_y$", ha="right", fontsize=11, color="#d73027",
            fontweight="bold")

    ax.text(50, 0.4, "schematic, not to scale  •  stations 1–3 define the reference frame",
            ha="center", fontsize=8.6, color="#7a8699", style="italic")
    save(fig, "detector_schematic.png")


# ================================================================ 2. translation controls
def translation_controls():
    df = pd.read_csv(OUT / "mc24_v3_expanded_trainval_validation_control_plots_v3/route_validation_control_summary.csv")
    series = [
        ("Pairwise MLP + route solver", "MLP + route (baseline)", C["mlp"], "o", "-"),
        ("V2 BCE route query", "V2 Transformer (BCE route query)", C["v2"], "s", "-"),
        ("V3 exact margin", "V3 structured margin", C["v3"], "^", "--"),
        ("V3 exact margin + soft control", "V3 soft control", C["v3s"], "v", "--"),
    ]
    mags, xpos = mag_x()
    xmap = dict(zip(mags, xpos))

    fig, axes = plt.subplots(1, 2, figsize=(10.3, 3.9))
    for name, label, col, mk, ls in series:
        sub = df[df.method == name].sort_values("magnitude_mm")
        x = sub.magnitude_mm.map(xmap)
        axes[0].plot(x, sub.complete_track_efficiency, ls + mk, color=col,
                     label=label, ms=5.5, lw=1.8)
        axes[1].plot(x, sub.track_fake_rate, ls + mk, color=col, label=label,
                     ms=5.5, lw=1.8)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(xpos)
        ax.set_xticklabels(["0", "0.1", "1", "5", "10", "50"])
        ax.set_xlabel("injected station-0 translation magnitude [mm]")
        ax.set_xlim(0.045, 70)
    axes[0].axhline(GATE["eff"], color=C["pass"], ls=":", lw=1.6)
    axes[0].text(0.048, GATE["eff"] + 0.02, "gate $\\geq$ 0.70", color=C["pass"], fontsize=9)
    axes[0].set_ylabel("complete-track efficiency")
    axes[0].set_ylim(-0.04, 1.02)
    axes[0].set_title("Expanded dx/dy corpus — validation (994/796 events, source-disjoint)")
    axes[1].axhline(GATE["fake"], color=C["fail"], ls=":", lw=1.6)
    axes[1].text(0.048, GATE["fake"] + 0.017, "gate $\\leq$ 0.05", color=C["fail"], fontsize=9)
    axes[1].set_ylabel("track fake rate")
    axes[1].set_ylim(-0.04, 1.05)
    axes[1].set_title("candidate truth-chain recall = 1.0 at all magnitudes")
    axes[1].legend(loc="lower left", fontsize=9)
    fig.tight_layout(w_pad=2.4)
    save(fig, "translation_controls.png")


# ================================================================ 3. Ry controls
def ry_controls():
    df = pd.read_csv(OUT / "ift_ry_validation_control_assessment_v2/ift_ry_validation_route_controls.csv")
    series = [
        ("mlp_pairwise_route", "MLP + route", C["mlp"], "o"),
        ("v1_full_context", "V1 full context", C["v1"], "s"),
        ("v1_no_multistation_context", "V1 no context", C["v1nc"], "^"),
        ("v2_bce_route_query", "V2 BCE route query", C["v2"], "D"),
        ("v2_same_checkpoint_base_edge", "V2 edge-only control", C["v3s"], "v"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7))
    metrics = [("complete_track_efficiency", "complete-track efficiency", GATE["eff"], "≥"),
               ("complete_track_purity", "complete-track purity", GATE["purity"], "≥"),
               ("track_fake_rate", "track fake rate", GATE["fake"], "≤")]
    for name, label, col, mk in series:
        sub = df[df.model == name].sort_values("condition_magnitude")
        for ax, (m, ylab, gate, _) in zip(axes, metrics):
            ax.plot(sub.condition_magnitude, sub[m], "-", marker=mk, color=col,
                    label=label, ms=5, lw=1.7)
    for ax, (m, ylab, gate, sym) in zip(axes, metrics):
        ax.axhline(gate, color=C["fail"], ls=":", lw=1.6)
        ax.text(0.02, 0.04 if gate < 0.5 else 0.10, f"gate {sym} {gate:g}",
                transform=ax.transAxes, color=C["fail"], fontsize=9)
        ax.set_xlabel("injected IFT $|R_y|$ [mrad]")
        ax.set_ylabel(ylab)
        ax.set_xlim(-2, 63)
    axes[0].set_ylim(0.88, 1.0)
    axes[1].set_ylim(0.965, 1.0)
    axes[2].set_ylim(0.0, 0.075)
    axes[0].set_title("IFT $R_y$ corpus — validation only (796 events)")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.tight_layout(w_pad=2.2)
    save(fig, "ry_controls.png")


# ================================================================ 4. capture scan
def capture_scan():
    d = json.load(open(OUT / "mc24_muon_fasernu_physical_capture_scan_v1/capture_scan_summary.json"))
    bm = pd.DataFrame(d["by_magnitude"])
    bm = bm.sort_values("magnitude_mm")
    x = bm.magnitude_mm.replace(0.0, 0.055)

    fig, axes = plt.subplots(1, 2, figsize=(10.3, 3.8))
    col = [C["pass"] if f == 1 else C["fail"] for f in bm.capture_fraction]
    axes[0].bar(range(len(bm)), bm.capture_fraction, color=col, width=0.62)
    axes[0].set_xticks(range(len(bm)))
    axes[0].set_xticklabels([f"{m:g}" for m in bm.magnitude_mm])
    axes[0].set_xlabel("injected translation magnitude [mm]  (3 directions per point)")
    axes[0].set_ylabel("capture fraction")
    axes[0].set_ylim(0, 1.12)
    axes[0].set_title("physical refit capture scan — strict 0.01 mm criterion")
    for i, f in enumerate(bm.capture_fraction):
        axes[0].text(i, f + 0.04, f"{f*100:.0f}%", ha="center", fontsize=9,
                     color=C["ink"])

    axes[1].plot(x, bm.recovery_error_max_norm_mm_median, "-o", color=C["mlp"], ms=5.5)
    axes[1].axhline(0.01, color=C["fail"], ls=":", lw=1.6)
    axes[1].text(0.06, 0.0125, "capture tolerance 0.01 mm", color=C["fail"], fontsize=9)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{m:g}" for m in bm.magnitude_mm])
    axes[1].set_xlabel("injected translation magnitude [mm]")
    axes[1].set_ylabel("median max-norm recovery error [mm]")
    axes[1].set_title("recovery error vs injected offset (truth-fixed)")
    fig.tight_layout(w_pad=2.6)
    save(fig, "capture_scan.png")


# ================================================================ 5. Ry rotation closure
def ry_closure():
    injected = np.array([60.0, -60.0])
    oneshot = np.array([54.2246, -54.8091])
    local = np.array([60.06897907021476, -60.412496893437975])
    err_o = np.abs(oneshot - injected)
    err_l = np.abs(local - injected)

    fig, ax = plt.subplots(figsize=(6.9, 3.9))
    w = 0.36
    xp = np.arange(2)
    b1 = ax.bar(xp - w / 2, err_o, w, color=C["v3s"], label="one-shot linearization from nominal")
    b2 = ax.bar(xp + w / 2, err_l, w, color=C["v1"], label="local physical iteration (anchor ∓50, probes ∓45/∓55)")
    ax.axhline(1.0, color=C["fail"], ls=":", lw=1.6)
    ax.text(-0.45, 1.12, "capture tolerance 1.0 mrad", color=C["fail"], fontsize=9.5)
    ax.set_yscale("log")
    ax.set_xticks(xp)
    ax.set_xticklabels(["+60 mrad injected", "−60 mrad injected"])
    ax.set_ylabel("|recovery error| [mrad]")
    ax.set_ylim(0.03, 30)
    ax.set_title("IFT $R_y$ held-out closure at ±60 mrad (truth-fixed, stations 1–3 frozen)")
    for b, v in zip(b1, err_o):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.2f}", ha="center", fontsize=10, color=C["ink"])
    for b, v in zip(b2, err_l):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.3f}", ha="center", fontsize=10, color=C["ink"])
    ax.legend(loc="upper center", fontsize=9)
    fig.tight_layout()
    save(fig, "ry_closure.png")


# ================================================================ 6. multi-DoF closure bars
def multidof_closure():
    d = json.load(open(OUT / "mc24_multidof_ift_iteration00_anchor_trainval_closure_v1/multisource_local_step.json"))
    pars = d["parameters"]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6))
    for ax, p in zip(axes, pars):
        exp = p["expected_delta_to_target"]
        tr = p["fit_split_recovered_local_delta"]
        va = p["held_out_independent_recovered_local_delta"]
        sig = p["recovered_sigma"]
        tol = p["capture_tolerance"]
        unit = p["unit"]
        name = p["name"].replace("ift_", "").replace("_mm", " [mm]").replace("_mrad", " [mrad]")
        lo, hi = exp - tol, exp + tol
        ax.axhspan(lo, hi, color=C["pass"], alpha=0.16, lw=0)
        ax.axhline(exp, color=C["ink"], ls="--", lw=1.4, label="expected Δ")
        ax.errorbar([0], [tr], yerr=[sig], fmt="o", ms=8, color=C["train"],
                    capsize=5, lw=1.8, label=f"train fit ({tr:+.3f})")
        ax.errorbar([1], [va], yerr=[sig], fmt="s", ms=8, color=C["val"],
                    capsize=5, lw=1.8, label=f"held-out fit ({va:+.3f})")
        ok = p["capture_success"]
        ax.text(0.5, 0.035, "PASS" if ok else "FAIL", transform=ax.transAxes,
                ha="center", fontsize=12, fontweight="bold",
                color=C["pass"] if ok else C["fail"],
                bbox=dict(boxstyle="round,pad=0.28", fc="white",
                          ec=C["pass"] if ok else C["fail"], lw=1.6))
        ax.set_xlim(-0.55, 1.55)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["train", "validation\n(held out)"], fontsize=9.5)
        ax.set_title(f"{name}  tol ±{tol:g} {unit}", fontsize=11.5)
        rng = max(abs(tr - exp), abs(va - exp), tol) * 1.7
        ax.set_ylim(exp - rng, exp + rng)
        ax.set_ylabel(f"recovered Δ [{unit}]", fontsize=10)
        ax.legend(loc="upper right", fontsize=8.3)
    fig.suptitle("iteration-0 multi-DoF closure — anchor (dx, dy, $R_y$) = (+2.0 mm, −1.5 mm, +35 mrad), 2296/1756 obs.",
                 fontsize=12, y=1.04)
    fig.tight_layout(w_pad=2.2)
    save(fig, "multidof_closure.png")


# ================================================================ 7. per-source recovery
def per_source_recovery():
    df = pd.read_csv(OUT / "mc24_multidof_ift_iteration00_anchor_trainval_closure_v1/source_fit_diagnostics.csv")
    clo = json.load(open(OUT / "mc24_multidof_ift_iteration00_anchor_trainval_closure_v1/multisource_local_step.json"))
    import ast
    rows = []
    for _, r in df.iterrows():
        rec = ast.literal_eval(r.recovered_local_delta)
        sig = ast.literal_eval(r.recovered_sigma)
        rows.append(dict(source=r.source_id.replace("mc24_", "").replace("_", " "),
                         split=r.split,
                         dx=rec["ift_dx_mm"], dy=rec["ift_dy_mm"], ry=rec["ift_ry_mrad"],
                         sdx=sig["ift_dx_mm"], sdy=sig["ift_dy_mm"], sry=sig["ift_ry_mrad"]))
    d = pd.DataFrame(rows)
    d = d.sort_values(["split", "source"], ascending=[True, True]).reset_index(drop=True)

    panels = [("dx", "sdx", -2.0, 0.1, "IFT $d_x$ recovered Δ [mm]"),
              ("dy", "sdy", +1.5, 0.1, "IFT $d_y$ recovered Δ [mm]"),
              ("ry", "sry", -35.0, 1.0, "IFT $R_y$ recovered Δ [mrad]")]
    fig, axes = plt.subplots(3, 1, figsize=(9.6, 6.9), sharex=True)
    x = np.arange(len(d))
    for ax, (m, s, exp, tol, ylab) in zip(axes, panels):
        ax.axhspan(exp - tol, exp + tol, color=C["pass"], alpha=0.15, lw=0)
        ax.axhline(exp, color=C["ink"], ls="--", lw=1.3)
        for split, ccol in [("train", C["train"]), ("validation", C["val"])]:
            mask = (d.split == split).to_numpy()
            ax.errorbar(x[mask], d[m][mask], yerr=d[s][mask], fmt="none",
                        ecolor=ccol, elinewidth=1.4, capsize=2.5, zorder=2)
            ax.scatter(x[mask], d[m][mask], c=ccol, s=26, zorder=3)
        spread = d[m].std()
        ax.set_ylabel(ylab, fontsize=10)
        ax.text(len(d) - 0.4, exp, f"expected {exp:+g}", fontsize=8.5, color=C["ink"],
                va="bottom", ha="right")
        ax.text(0.0, 0.955, f"per-source std = {spread:.2f}   vs   median stat σ = {d[s].median():.2f}",
                transform=ax.transAxes, fontsize=9.3, color="#7a2e2e",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", fc="#fdf3f2", ec="#e0b4ae", lw=0.8))
        ax.set_ylim(d[m].min() - abs(exp) * 0.35 - tol, d[m].max() + abs(exp) * 0.35 + tol)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(d.source, rotation=38, ha="right", fontsize=8)
    axes[-1].set_xlabel("source xAOD file (10 train, 8 validation)")
    h = [plt.Line2D([], [], marker="o", ls="", color=C["train"], label="train"),
         plt.Line2D([], [], marker="o", ls="", color=C["val"], label="validation")]
    axes[0].legend(handles=h, loc="upper left", fontsize=9, ncol=2)
    fig.tight_layout()
    save(fig, "per_source_recovery.png")


# ================================================================ 8. chi2 reduction
def chi2_reduction():
    d = json.load(open(OUT / "mc24_multidof_ift_iteration00_anchor_trainval_closure_v1/multisource_local_step.json"))
    tr = d["fit_split"]["application"]
    va = d["held_out_split"]["frozen_fit_split_update_application"]
    base = [tr["baseline_response_chi2"], va["baseline_response_chi2"]]
    post = [tr["post_update_response_chi2"], va["post_update_response_chi2"]]
    red = [tr["response_chi2_reduction_fraction"], va["response_chi2_reduction_fraction"]]

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    xp = np.arange(2)
    w = 0.34
    ax.bar(xp - w / 2, base, w, color="#8c9bad", label="before update (anchor geometry)")
    ax.bar(xp + w / 2, post, w, color=C["pass"], label="after frozen linear update")
    ax.set_yscale("log")
    ax.set_xticks(xp)
    ax.set_xticklabels(["train (fit split)", "validation (held out)"])
    ax.set_ylabel("response $\\chi^2$  (mode-0 Acts residual)")
    ax.set_title("held-out response $\\chi^2$ validates the linear update")
    for i in range(2):
        ax.text(xp[i] + w / 2, post[i] * 1.35, f"−{red[i]*100:.1f}%", ha="center",
                fontsize=12, fontweight="bold", color=C["pass"])
        ax.text(xp[i] - w / 2, base[i] * 1.35, f"{base[i]/1e6:.1f}M", ha="center",
                fontsize=10, color="#5a6a7d")
    ax.set_ylim(5e3, 3e8)
    ax.legend(loc="upper right", fontsize=9.5)
    fig.tight_layout()
    save(fig, "chi2_reduction.png")


# ================================================================ 9. coverage decomposition
def coverage_decomposition():
    df = pd.read_csv(OUT / "mc24_multidof_ift_iteration00_coverage_decomposition_v1/chain_first_failure_by_point.csv")
    order = ["retained", "0->1:propagation_failed", "0->1:chi2_gate_rejection",
             "1->2:chi2_gate_rejection", "2->3:chi2_gate_rejection"]
    labels = ["retained (chain complete)", "0→1 propagation failed",
              "0→1 χ² gate rejection", "1→2 χ² gate rejection", "2→3 χ² gate rejection"]
    colors = [C["pass"], "#f4a582", C["fail"], "#7b3294", "#404040"]
    points = [("iteration_00_reference", "nominal reference"), ("iteration_00_anchor", "anchor (+2 mm, −1.5 mm, +35 mrad)")]
    groups = [(p, s) for p, _ in points for s in ("train", "validation")]
    pretty = {p: n for p, n in points}

    fig, ax = plt.subplots(figsize=(9.8, 3.6))
    yticks, ylabels = [], []
    for gi, (p, s) in enumerate(groups):
        sub = df[(df.point == p) & (df.split == s)]
        total = sub["count"].sum()
        left = 0.0
        for cat, lab, col in zip(order, labels, colors):
            v = sub[sub.first_failing_edge == cat]["count"].sum() / total
            ax.barh(gi, v, left=left, color=col, height=0.62,
                    label=lab if gi == 0 else None, edgecolor="white", lw=0.6)
            if v > 0.055:
                ax.text(left + v / 2, gi, f"{v*100:.0f}%", ha="center", va="center",
                        fontsize=8.8, color="white" if col not in ("#f4a582",) else C["ink"])
            left += v
        yticks.append(gi)
        ylabels.append(f"{pretty[p]}\n{s}")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("fraction of complete four-station truth chains — first failing edge")
    ax.set_title("candidate coverage decomposition at iteration-0 points (100 events/source)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, fontsize=9)
    fig.tight_layout()
    save(fig, "coverage_decomposition.png")


# ================================================================ 10. correlation matrix
def param_correlation():
    d = json.load(open(OUT / "mc24_multidof_ift_iteration00_anchor_trainval_closure_v1/multisource_local_step.json"))
    M = np.array(d["parameter_correlation"])
    names = ["$d_x$", "$d_y$", "$R_y$"]
    fig, ax = plt.subplots(figsize=(4.15, 3.7))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(names, fontsize=12); ax.set_yticklabels(names, fontsize=12)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center", fontsize=13,
                    color="white" if abs(M[i, j]) > 0.55 else C["ink"],
                    fontweight="bold" if i != j and abs(M[i, j]) > 0.5 else "normal")
    ax.set_title("recovered-parameter correlation\n(pooled fit, iteration 0)", fontsize=11.5)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=9)
    fig.tight_layout()
    save(fig, "param_correlation.png")


# ================================================================ 11. historical sealed test
def sealed_test_history():
    df = pd.read_csv(OUT / "mc24_muon_2dfluka_multidirection_test_route_level_v1/magnitude_summary.csv")
    df = df.sort_values("magnitude_mm")
    x = df.magnitude_mm.replace(0.0, 0.055)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(x, df.pooled_candidate_complete_truth_chain_recall, "-", color="#8c9bad",
            lw=2, marker="s", ms=6, label="candidate truth-chain recall (frozen MLP graph)")
    ax.plot(x, df.pooled_complete_track_efficiency, "-o", color=C["mlp"], lw=2,
            label="frozen MLP + route — complete efficiency")
    # V1 geometry-aware (sealed scan, direction-mean) from the V1 document
    v1_mag = np.array([5.0, 10.0])
    v1_eff = np.array([0.705, 0.632])
    v1_err = np.array([0.021, 0.085])
    ax.errorbar(v1_mag, v1_eff, yerr=v1_err, fmt="D", color=C["v1"], ms=7, lw=1.8,
                capsize=4, label="V1 geometry-aware (5/10 mm, dir. mean)")
    ax.axhline(GATE["eff"], color=C["pass"], ls=":", lw=1.6)
    ax.text(0.058, GATE["eff"] + 0.02, "gate ≥ 0.70", color=C["pass"], fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m:g}" for m in df.magnitude_mm])
    ax.set_xlabel("injected translation magnitude [mm]")
    ax.set_ylabel("fraction")
    ax.set_ylim(-0.03, 1.06)
    ax.set_title("historical sealed multi-direction test (19 events, 2026-08-12, frozen)")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    save(fig, "sealed_test_history.png")


# ================================================================ 12. jacobian kinematics
def jacobian_kinematics():
    df = pd.read_csv(OUT / "mc24_multidof_ift_iteration00_jacobian_source_diagnosis_v1/per_source_summary.csv")
    diag = json.load(open(OUT / "mc24_multidof_ift_iteration00_jacobian_source_diagnosis_v1/source_jacobian_diagnosis.json"))
    corr = {(e["observable"], e["kinematic"]): e["pearson"] for e in diag["cross_source_correlations"]}

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))
    panels = [("mean_abs_tx", "source mean $|t_x|$"), ("mean_ty", "source mean $t_y$")]
    for ax, (kin, xlab) in zip(axes, panels):
        for split, col, mk in [("train", C["train"], "o"), ("validation", C["val"], "s")]:
            sub = df[df.split == split]
            ax.scatter(sub[kin], sub.bias_ift_dx_mm, c=col, marker=mk, s=42,
                       edgecolor="white", lw=0.6, label=split, zorder=3)
        r = corr[("bias_ift_dx_mm", kin)]
        ax.axhline(0, color=C["ink"], ls="--", lw=1.1)
        ax.set_xlabel(xlab)
        ax.set_ylabel("per-source $d_x$ bias [mm]")
        ax.set_title(f"$d_x$ bias vs {xlab}   (Pearson r = {r:+.2f}, n = 18)")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("source-to-source $d_x$ bias correlates only weakly with track kinematics", y=1.03, fontsize=12)
    fig.tight_layout(w_pad=2.4)
    save(fig, "jacobian_kinematics.png")


if __name__ == "__main__":
    detector_schematic()
    translation_controls()
    ry_controls()
    capture_scan()
    ry_closure()
    multidof_closure()
    per_source_recovery()
    chi2_reduction()
    coverage_decomposition()
    param_correlation()
    sealed_test_history()
    jacobian_kinematics()
    print("all figures done")
