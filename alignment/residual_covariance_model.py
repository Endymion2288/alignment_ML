"""Workbook-80 residual covariance model reconstruction & validation.

This module rebuilds and validates the *residual covariance* half of the
measurement model.  It never solves for an alignment correction, never opens
held-out data, and never promotes a diagnostic covariance to a weight matrix.

Scientific context (frozen by Workbook 79): the combined 4x4 covariance
``C_combined = C_propagated_source + C_target`` assumes source/target
independence (``evaluation/field_propagation.py``).  WB79 proved the frozen
covariance is numerically near-singular (median condition ~2.3e12), that
99.74% of the zero-candidate chi2 sits in the smallest covariance eigenmode,
and that whitening the de-meaned residuals leaves chi2/ndof = 1058.46.  It also
proved the near-singular direction is *alignment-blind* (full and diagonal
covariances give a bit-identical information matrix and beta), so the large
gamma is NOT caused solely by the off-diagonal correlation and diagonalizing
the covariance does NOT fix alignment.

The candidate covariance models here are pre-registered from first-principles
physics (multiple-scattering process noise over the lever arm; an uncorrelated
missing-noise floor; pure variance inflation as a contrast).  Any nuisance
scale is estimated ONLY by cross-fitting (14973 derive -> 14974 validate and
the reverse); no candidate is chosen because it makes gamma smaller, the rank
more stable, the condition number nicer, or the chi2 drop more.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from alignment import gauge_fixed_real_data_diagnostic as wb78
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events

OBSERVABLE_NAMES = wb78.OBSERVABLE_NAMES
STATION_PAIRS = wb78.STATION_PAIRS
N_OBS = 4


# ---------------------------------------------------------------------------
# Enriched per-pair calibration records (read-only; never modifies the bank)
# ---------------------------------------------------------------------------


def load_calibration_pairs_enriched(
    config: Mapping[str, Any], bank: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Join the frozen calibration bank to per-pair propagated-source and
    target covariances, lever arm and kinematics.

    The frozen bank (from ``wb78.build_real_data_bank``) stores only the
    combined covariance.  This loader re-reads the same frozen identity files
    and extracts, per bank pair (matched by identity): ``C_prop`` (propagated
    source covariance), ``C_target`` (target tracklet fit covariance), the
    lever arm ``L = z_target - z_source`` and the residual-blind kinematics
    ``(pred_tx, pred_ty)``.  It verifies ``C_combined == C_prop + C_target``
    and that the residuals match the frozen bank bit-exactly, so the enriched
    view is guaranteed consistent with the frozen bank.
    """
    wb78_config = config["wb78_config"]
    # Per-source lookup: pair identity -> enriched record.
    enriched: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for source in wb78_config["real_data_population"]["sources"]:
        if str(source["role"]) != "calibration":
            continue
        roots = wb78._source_roots(wb78_config, source)
        events = load_events(roots["identity"] / "synthetic_tracklets.root", require_mc_labels=False)
        records = load_propagation_records(roots["identity"] / "field_candidates.root")
        tracklet_index: dict[tuple[int, int, int], tuple[Any, int]] = {}
        for event in events:
            for row, tid in enumerate(event.tracklet_id):
                tracklet_index[(event.run_id, event.event_id, int(tid))] = (event, row)
        for row in range(records.size):
            if int(records.q_over_p_mode[row]) != 0:
                continue
            if not records.success[row] or not records.has_covariance[row]:
                continue
            skey = (
                int(records.run_id[row]),
                int(records.event_id[row]),
                int(records.source_tracklet_id[row]),
            )
            tkey = (
                int(records.run_id[row]),
                int(records.event_id[row]),
                int(records.target_tracklet_id[row]),
            )
            if skey not in tracklet_index or tkey not in tracklet_index:
                continue
            sevent, srow = tracklet_index[skey]
            tevent, trow = tracklet_index[tkey]
            pair_key = (
                int(records.run_id[row]),
                int(records.event_id[row]),
                int(records.source_tracklet_id[row]),
                int(records.target_tracklet_id[row]),
            )
            enriched[pair_key] = {
                "c_prop": np.asarray(records.covariance[row], dtype=np.float64),
                "c_target": np.asarray(tevent.covariance[trow], dtype=np.float64),
                "lever_arm_mm": float(records.target_z_mm[row] - sevent.z_mm[srow]),
                "pred_tx": float(records.prediction[row, 2]),
                "pred_ty": float(records.prediction[row, 3]),
            }

    arrays = wb78.concatenate_banks(bank)
    n_pairs = arrays["residual"].shape[0]
    c_prop = np.zeros((n_pairs, N_OBS, N_OBS))
    c_target = np.zeros((n_pairs, N_OBS, N_OBS))
    lever_arm = np.zeros(n_pairs)
    pred_tx = np.zeros(n_pairs)
    pred_ty = np.zeros(n_pairs)
    max_comb_diff = 0.0
    for i in range(n_pairs):
        key = (
            int(arrays["run_id_arr"][i]),
            int(arrays["event_id"][i]),
            int(arrays["source_tracklet_id"][i]),
            int(arrays["target_tracklet_id"][i]),
        )
        rec = enriched.get(key)
        if rec is None:
            raise ValueError(f"bank pair missing enriched record: {key}")
        c_prop[i] = rec["c_prop"]
        c_target[i] = rec["c_target"]
        lever_arm[i] = rec["lever_arm_mm"]
        pred_tx[i] = rec["pred_tx"]
        pred_ty[i] = rec["pred_ty"]
        max_comb_diff = max(
            max_comb_diff,
            float(np.max(np.abs(arrays["covariance"][i] - (rec["c_prop"] + rec["c_target"])))),
        )
    return {
        "residual": np.asarray(arrays["residual"], dtype=np.float64),
        "c_combined": np.asarray(arrays["covariance"], dtype=np.float64),
        "c_prop": c_prop,
        "c_target": c_target,
        "lever_arm_mm": lever_arm,
        "pred_tx": pred_tx,
        "pred_ty": pred_ty,
        "target_station_id": np.asarray(arrays["target_station_id"], dtype=np.int64),
        "run_id": np.asarray(arrays["run_id_arr"], dtype=np.int64),
        "combined_equals_prop_plus_target_max_abs_diff": max_comb_diff,
    }


# ---------------------------------------------------------------------------
# Stage: covariance semantics audit (read-only code/provenance trace)
# ---------------------------------------------------------------------------


def covariance_semantics_audit(
    config: Mapping[str, Any], enriched: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Produce the covariance semantics audit answering the 12 pre-registered
    questions with formula / source location / ordering / units / assumption /
    verdict, plus an empirical characterization.  Read-only; no pair dropped.
    """
    c_combined = enriched["c_combined"]
    c_prop = enriched["c_prop"]
    c_target = enriched["c_target"]
    residual = enriched["residual"]
    lever = enriched["lever_arm_mm"]
    target_station = enriched["target_station_id"]
    n_pairs = residual.shape[0]

    # Empirical characterization.
    def _eig(c):
        w, v = np.linalg.eigh(c)
        return w, v

    cond_comb = np.zeros(n_pairs)
    cond_prop = np.zeros(n_pairs)
    cond_tgt = np.zeros(n_pairs)
    min_eig_comb = np.zeros(n_pairs)
    smallest_dir = np.zeros((n_pairs, N_OBS))
    for i in range(n_pairs):
        wc, vc = _eig(c_combined[i])
        wp, _ = _eig(c_prop[i])
        wt, _ = _eig(c_target[i])
        cond_comb[i] = wc[-1] / max(wc[0], 1.0e-300)
        cond_prop[i] = wp[-1] / max(wp[0], 1.0e-300)
        cond_tgt[i] = wt[-1] / max(wt[0], 1.0e-300)
        min_eig_comb[i] = wc[0]
        smallest_dir[i] = np.abs(vc[:, 0])

    # Marginal (diagonal) per-pair pull RMS: sqrt(mean((r_m/sigma_m)^2)) per
    # observable, matching the WB79 definition (per-pair normalisation, then
    # RMS).  This is the meaningful marginal-calibration metric; a plain
    # residual_rms/median_sigma ratio is misleading here because the per-pair
    # sigma distribution is heavy-tailed.  Also report the raw RMS and median
    # sigma for provenance.
    marginal = {}
    for m in range(N_OBS):
        sigma_m = np.sqrt(np.maximum(c_combined[:, m, m], 1.0e-300))
        pull_rms = float(np.sqrt(np.mean((residual[:, m] / sigma_m) ** 2)))
        rms_res = float(np.sqrt(np.mean(residual[:, m] ** 2)))
        sig = float(np.sqrt(np.median(c_combined[:, m, m])))
        marginal[OBSERVABLE_NAMES[m]] = {
            "pull_rms": pull_rms,
            "residual_rms": rms_res,
            "median_covariance_sigma": sig,
        }

    # Mean correlation matrix (model) vs empirical residual correlation.
    corrs = []
    for i in range(n_pairs):
        d = np.sqrt(np.diag(c_combined[i]))
        corrs.append(c_combined[i] / np.outer(d, d))
    model_corr = np.mean(corrs, axis=0)
    dec = residual - residual.mean(axis=0)
    emp_cov = np.cov(dec, rowvar=False)
    d = np.sqrt(np.diag(emp_cov))
    emp_corr = emp_cov / np.outer(d, d)

    per_pair_type = {}
    for pair in STATION_PAIRS:
        mask = target_station == pair[1]
        if not np.any(mask):
            continue
        per_pair_type[f"0->{pair[1]}"] = {
            "n_pairs": int(mask.sum()),
            "lever_arm_mm_median": float(np.median(lever[mask])),
            "lever_arm_mm_min": float(np.min(lever[mask])),
            "lever_arm_mm_max": float(np.max(lever[mask])),
            "combined_condition_median": float(np.median(cond_comb[mask])),
            "prop_condition_median": float(np.median(cond_prop[mask])),
            "target_condition_median": float(np.median(cond_tgt[mask])),
            "combined_min_eigenvalue_median": float(np.median(min_eig_comb[mask])),
            "smallest_eigenvector_mean_abs": [
                float(v) for v in np.mean(smallest_dir[mask], axis=0)
            ],
        }

    questions = {
        "source_tracklet_state_covariance_origin": {
            "answer": "Local tracklet FIT covariance of the station-0 (IFT) source "
            "tracklet, read from the `tracklets` tree `cov_*` branches.",
            "source_location": "datasets/root_loader.py (EventTracklets.covariance), "
            "datasets/schema.py COVARIANCE_FIELDS",
            "verdict": "resolved",
        },
        "propagated_state_covariance_transformation": {
            "answer": "Field-aware propagated source covariance, computed UPSTREAM by "
            "the external Calypso/ACTS extrapolation (FaserActsExtrapolationTool) and "
            "read from the `propagations` tree `pred_cov_*` branches.  This repo only "
            "reads it.  Empirically near-singular (median condition ~1e15, min "
            "eigenvalue ~1e-10), i.e. a deterministic pencil-beam transport of the "
            "source fit covariance.  Whether the external tool adds multiple-scattering "
            "process noise is NOT visible from this repo.",
            "source_location": "datasets/propagation_loader.py (PropagationRecords."
            "covariance from pred_cov_*); external exporter faser_ntuple_maker.py",
            "verdict": "unresolved_external_provenance (multiple-scattering process "
            "noise inclusion not confirmable from this repo)",
        },
        "target_tracklet_covariance_origin": {
            "answer": "Local tracklet FIT covariance of the station-j target tracklet, "
            "from the `tracklets` tree `cov_*` branches.",
            "source_location": "datasets/root_loader.py (EventTracklets.covariance)",
            "verdict": "resolved",
        },
        "residual_covariance_formula": {
            "answer": "C_combined = C_propagated_source + C_target (plain sum).",
            "source_location": "evaluation/field_propagation.py:238 "
            "(combined_covariance = prediction_covariance + target_covariance)",
            "verdict": "resolved",
        },
        "source_target_independence_assumption": {
            "answer": "ASSUMED independent: the sum C_prop + C_target contains NO "
            "cross-covariance term Cov(propagated_source, target).",
            "source_location": "evaluation/field_propagation.py:131-132 (docstring), :238",
            "verdict": "resolved (assumption present); validity questionable (H_cov_2)",
        },
        "shared_track_common_hits_common_fit": {
            "answer": "Source and target tracklets are truth-matched to the SAME "
            "particle but are SEPARATE local fits to different station hits.  They do "
            "not share a common track fit in this tracklet-pair observable model.",
            "source_location": "evaluation/field_propagation.py (truth match, separate "
            "source/target tracklet states)",
            "verdict": "resolved (structurally separate fits); residual common-mode "
            "(e.g. multiple scattering) is a process-noise question, not a shared-fit one",
        },
        "missing_cross_covariance_term": {
            "answer": "The formula omits -2 Cov(target, propagated_source).  A missing "
            "POSITIVE cross-covariance would make the predicted residual SMALLER than "
            "the sum (reducing chi2); the observed chi2 is instead enormous, so a "
            "missing cross-covariance does NOT explain the inflation.",
            "source_location": "evaluation/field_propagation.py:238",
            "verdict": "resolved (present omission, but wrong sign to explain large chi2)",
        },
        "local_global_surface_jacobian_units_ordering": {
            "answer": "State order is [x_mm, y_mm, tx, ty] consistently across the "
            "tracklets tree, propagations tree and covariance fields.  Units: x,y in "
            "mm; tx,ty dimensionless slopes; covariance entries mixed (mm^2, mm, "
            "dimensionless) matching that order.",
            "source_location": "datasets/schema.py (STATE_FIELDS, COVARIANCE_FIELDS, "
            "covariance_from_columns)",
            "verdict": "resolved",
        },
        "x_tx_y_ty_correlation_physical_origin": {
            "answer": "Long-lever-arm propagation x(z) ~ x0 + tx*dz produces a strong "
            "x<->tx (and y<->ty) correlation in the propagated covariance.  Empirically "
            "the model mean corr(x,tx) ~ 0.92-0.999 while the empirical residual "
            "corr(x,tx) ~ 0.64: the model over-states the correlation.",
            "source_location": "geometry/propagation.py (matrix @ cov @ matrix.T for the "
            "straight-line reference); external field-aware propagation for pred_cov_*",
            "verdict": "resolved (physical origin identified; magnitude mis-modelled)",
        },
        "long_lever_arm_near_rank1_correlation": {
            "answer": "YES.  Lever arms are ~1.2-4.3 m; the deterministic propagated "
            "covariance is near rank-1 (pencil beam), and the smallest-eigenvalue "
            "direction of C_combined lies in the SLOPE (tx,ty) subspace.",
            "source_location": "empirical (this audit); evaluation/field_propagation.py",
            "verdict": "resolved",
        },
        "inverse_method_direct_inv_vs_stable_factorization": {
            "answer": "The real-data alignment solve uses np.linalg.inv(C) directly "
            "(gauge_fixed_real_data_diagnostic.py _information_and_rhs / applicability "
            "audit), whereas the MC bank build (physical_jacobian.py "
            "_inverse_covariances) uses a stable Cholesky factorization.  For condition "
            "~1e14 the direct inverse is numerically unreliable.",
            "source_location": "alignment/gauge_fixed_real_data_diagnostic.py:669,1028; "
            "alignment/physical_jacobian.py:89-105",
            "verdict": "resolved (inconsistent; direct inv used in the real-data solve)",
        },
        "covariance_type_fit_vs_prediction_vs_mixed": {
            "answer": "MIXED: C_propagated_source is a (deterministic) prediction/"
            "transport of the source fit covariance; C_target is a target fit "
            "covariance.  The sum mixes a transported-fit and a local-fit uncertainty "
            "and (see H_cov_5) likely omits process noise over the lever arm.",
            "source_location": "evaluation/field_propagation.py:238",
            "verdict": "resolved (mixed); adequacy is the subject of the model rebuild",
        },
    }

    return {
        "kind": "covariance_semantics_audit",
        "n_pairs": n_pairs,
        "questions": questions,
        "empirical": {
            "combined_condition_median": float(np.median(cond_comb)),
            "combined_condition_p95": float(np.percentile(cond_comb, 95.0)),
            "prop_condition_median": float(np.median(cond_prop)),
            "target_condition_median": float(np.median(cond_tgt)),
            "combined_min_eigenvalue_median": float(np.median(min_eig_comb)),
            "marginal_residual_vs_covariance": marginal,
            "model_mean_correlation": model_corr.tolist(),
            "empirical_residual_correlation": emp_corr.tolist(),
            "per_pair_type": per_pair_type,
        },
        "combined_equals_prop_plus_target_max_abs_diff": float(
            enriched["combined_equals_prop_plus_target_max_abs_diff"]
        ),
        "diagnostic_only": True,
        "alignment_authorized": False,
    }


# ---------------------------------------------------------------------------
# Candidate covariance models
# ---------------------------------------------------------------------------


def _ms_leverarm_structure(lever_arm_mm: np.ndarray) -> np.ndarray:
    """Multiple-scattering random-walk process-noise structure G_MS(L) per pair.

    State order [x, y, tx, ty]; couples (x,tx) and (y,ty) with
    [[L^2/3, L/2],[L/2, 1]].  Returns (n, 4, 4); the overall scale theta^2 is
    the cross-fitted nuisance.
    """
    n = lever_arm_mm.shape[0]
    L = np.asarray(lever_arm_mm, dtype=np.float64)
    G = np.zeros((n, N_OBS, N_OBS))
    G[:, 0, 0] = L * L / 3.0
    G[:, 0, 2] = G[:, 2, 0] = L / 2.0
    G[:, 2, 2] = 1.0
    G[:, 1, 1] = L * L / 3.0
    G[:, 1, 3] = G[:, 3, 1] = L / 2.0
    G[:, 3, 3] = 1.0
    return G


def build_candidate_covariance(
    kind: str,
    c_combined: np.ndarray,
    lever_arm_mm: np.ndarray,
    scale: float,
    pairtype_marginal_floor: np.ndarray | None = None,
) -> np.ndarray:
    """Build a candidate covariance C_model for each pair.

    kind:
      frozen_baseline        -> C_combined (no nuisance)
      ms_leverarm_process_noise -> C_combined + scale * G_MS(L)
      scaled_diagonal_floor  -> C_combined + scale * diag(pairtype_marginal_floor)
      variance_inflation     -> scale * C_combined
    """
    n = c_combined.shape[0]
    if kind == "frozen_baseline":
        return np.asarray(c_combined, dtype=np.float64).copy()
    if kind == "ms_leverarm_process_noise":
        return c_combined + float(scale) * _ms_leverarm_structure(lever_arm_mm)
    if kind == "scaled_diagonal_floor":
        if pairtype_marginal_floor is None:
            raise ValueError("scaled_diagonal_floor requires pairtype_marginal_floor")
        floor = np.zeros((n, N_OBS, N_OBS))
        for i in range(n):
            floor[i] = np.diag(pairtype_marginal_floor)
        return c_combined + float(scale) * floor
    if kind == "variance_inflation":
        return float(scale) * c_combined
    raise ValueError(f"unknown candidate covariance kind {kind}")


def _logdet_and_solve_eigh(covariance: np.ndarray, vector: np.ndarray) -> tuple[float, np.ndarray]:
    """Numerically stable log|C| and C^{-1} vector via eigendecomposition.

    Eigenvalues are clamped at a tiny absolute floor (1e-30) purely to avoid a
    NaN from an exactly-zero/negative numerical eigenvalue; the floor is far
    below any physical scale (min physical eigenvalue ~1e-10), so it does NOT
    regularize the model.  This is numerical method (A): a stable evaluation of
    the SAME covariance, never a modification of C.
    """
    w, v = np.linalg.eigh(covariance)
    w_clamped = np.maximum(w, 1.0e-300)
    logdet = float(np.sum(np.log(w_clamped)))
    solved = v @ ((v.T @ vector) / w_clamped)
    return logdet, solved


def _gaussian_nll(covariance: np.ndarray, residuals: np.ndarray) -> float:
    """Gaussian negative log-likelihood of `residuals` under per-pair `covariance`."""
    total = 0.0
    for i in range(residuals.shape[0]):
        logdet, solved = _logdet_and_solve_eigh(covariance[i], residuals[i])
        total += logdet + float(residuals[i] @ solved)
    return 0.5 * total


def _demean_cells(
    residual: np.ndarray, run_id: np.ndarray, target_station: np.ndarray
) -> np.ndarray:
    """Remove the per-(run, station-pair-type) coherent mean (diagnostic-only)."""
    cells = {}
    centered = residual.copy()
    for i in range(residual.shape[0]):
        cells.setdefault((int(run_id[i]), int(target_station[i])), []).append(i)
    for rows in cells.values():
        idx = np.asarray(rows, dtype=np.intp)
        centered[idx] = residual[idx] - residual[idx].mean(axis=0)
    return centered


def _pairtype_marginal_floor(c_combined: np.ndarray) -> np.ndarray:
    """Median marginal variances (diagonal) across a set of pairs."""
    return np.median(np.diagonal(c_combined, axis1=1, axis2=2), axis=0)


def _estimate_scale_nll(
    kind: str,
    c_combined: np.ndarray,
    lever_arm_mm: np.ndarray,
    demeaned: np.ndarray,
    pairtype_marginal_floor: np.ndarray | None,
) -> dict[str, Any]:
    """Estimate the single nuisance scale for a candidate by Gaussian NLL on the
    DERIVATION run's de-meaned residuals (deterministic log-grid + refine)."""
    if kind == "frozen_baseline":
        return {"scale": 1.0, "nll": _gaussian_nll(c_combined, demeaned), "n_nuisance": 0}

    def nll_at(log_scale: float) -> float:
        cov = build_candidate_covariance(
            kind, c_combined, lever_arm_mm, math.exp(log_scale), pairtype_marginal_floor
        )
        return _gaussian_nll(cov, demeaned)

    # Deterministic log-grid scan then local golden refine.
    grid = np.linspace(math.log(1.0e-12), math.log(1.0e6), 241)
    values = np.array([nll_at(g) for g in grid])
    best = int(np.argmin(values))
    lo = grid[max(best - 1, 0)]
    hi = grid[min(best + 1, grid.size - 1)]
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c1 = b - gr * (b - a)
    c2 = a + gr * (b - a)
    f1, f2 = nll_at(c1), nll_at(c2)
    for _ in range(80):
        if f1 > f2:
            a = c1
            c1, f1 = c2, f2
            c2 = a + gr * (b - a)
            f2 = nll_at(c2)
        else:
            b = c2
            c2, f2 = c1, f1
            c1 = b - gr * (b - a)
            f1 = nll_at(c1)
    scale = math.exp(0.5 * (a + b))
    return {"scale": float(scale), "nll": float(nll_at(0.5 * (a + b))), "n_nuisance": 1}


def _whiten_eigh(covariance: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """z = C^{-1/2} vector via eigendecomposition (stable, method A)."""
    n = covariance.shape[0]
    out = np.zeros((n, vector.shape[1]))
    for i in range(n):
        w, v = np.linalg.eigh(covariance[i])
        w = np.maximum(w, 1.0e-300)
        out[i] = (v.T @ vector[i]) / np.sqrt(w)
    return out


def _generalized_eigenvalue_rms_log(c1: np.ndarray, c2: np.ndarray) -> float:
    """Basis-independent RMS log-deviation of generalized eigenvalues of (c1, c2)."""
    w1, v1 = np.linalg.eigh(c1)
    half = (v1 * np.sqrt(np.maximum(w1, 1.0e-300))) @ v1.T
    half_inv = np.linalg.inv(half)
    generalized = np.linalg.eigvalsh(half_inv @ c2 @ half_inv)
    return float(np.sqrt(np.mean(np.log(np.maximum(generalized, 1.0e-300)) ** 2)))


def _validation_metrics(
    c_model: np.ndarray,
    demeaned_val: np.ndarray,
    val_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Confirmatory whitening metrics on the VALIDATION run (scale frozen)."""
    n_pairs, n_obs = demeaned_val.shape
    z = _whiten_eigh(c_model, demeaned_val)
    chi2 = float(np.sum(z ** 2))
    # ndof subtracts the per-cell means estimated on the validation run.
    ndof = int(val_spec["ndof"])
    chi2_per_ndof = chi2 / max(ndof, 1)
    cov_z = np.cov(z, rowvar=False)
    cov_z_eig = np.linalg.eigvalsh(cov_z)
    # Positive definiteness + factorization stability of the model.
    spd_ok = True
    max_inv_resid = 0.0
    for i in range(n_pairs):
        w = np.linalg.eigvalsh(c_model[i])
        if w[0] <= 0.0:
            spd_ok = False
        inv = np.linalg.inv(c_model[i])
        max_inv_resid = max(
            max_inv_resid, float(np.max(np.abs(c_model[i] @ inv - np.eye(N_OBS))))
        )
    return {
        "whitened_chi2_per_ndof": chi2_per_ndof,
        "cov_z_eigenvalues": [float(v) for v in cov_z_eig],
        "cov_z_eigenvalue_min": float(cov_z_eig[0]),
        "cov_z_eigenvalue_max": float(cov_z_eig[-1]),
        "positive_definite": bool(spd_ok),
        "factorization_max_inverse_residual": max_inv_resid,
    }


def derive_pairtype_scales(
    kind: str,
    c_combined: np.ndarray,
    lever_arm_mm: np.ndarray,
    residual: np.ndarray,
    run_id: np.ndarray,
    target_station: np.ndarray,
    derive_run: int,
    min_pairs: int,
) -> dict[str, dict[str, Any]]:
    """Derive per-pair-type nuisance scales + marginal floors from ONE run's
    de-meaned residuals (the derivation run only).  Returns {label: {...}}."""
    out: dict[str, dict[str, Any]] = {}
    for pair in STATION_PAIRS:
        station = pair[1]
        label = f"0->{station}"
        dmask = (run_id == derive_run) & (target_station == station)
        if dmask.sum() < min_pairs:
            out[label] = {"sufficient": False, "n_derive": int(dmask.sum())}
            continue
        demean_d = _demean_cells(residual[dmask], run_id[dmask], target_station[dmask])
        floor = _pairtype_marginal_floor(c_combined[dmask])
        est = _estimate_scale_nll(kind, c_combined[dmask], lever_arm_mm[dmask], demean_d, floor)
        out[label] = {
            "sufficient": True,
            "n_derive": int(dmask.sum()),
            "scale": float(est["scale"]),
            "derivation_nll": float(est["nll"]),
            "marginal_floor": floor,
        }
    return out


def assemble_model_covariance(
    kind: str,
    c_combined: np.ndarray,
    lever_arm_mm: np.ndarray,
    target_station: np.ndarray,
    scales_by_pairtype: Mapping[str, Mapping[str, Any]],
) -> np.ndarray:
    """Assemble the per-pair model covariance for the given pairs using the
    pre-derived per-pair-type scales (and marginal floors)."""
    n = c_combined.shape[0]
    out = np.zeros_like(c_combined)
    for pair in STATION_PAIRS:
        station = pair[1]
        label = f"0->{station}"
        mask = target_station == station
        if not np.any(mask):
            continue
        info = scales_by_pairtype.get(label, {})
        scale = float(info.get("scale", 1.0))
        floor = info.get("marginal_floor")
        out[mask] = build_candidate_covariance(
            kind, c_combined[mask], lever_arm_mm[mask], scale, floor
        )
    return out


def build_crossfit_model_covariance(
    config: Mapping[str, Any], enriched: Mapping[str, np.ndarray], kind: str
) -> np.ndarray:
    """Build the per-pair model covariance for ALL calibration pairs, where each
    run's pairs use the nuisance scale derived from the OTHER run (cross-fit, no
    same-run self-proof).  Diagnostic-only; never enters an alignment solve in
    Workbook 80.
    """
    spec = config["covariance_model"]
    min_pairs = int(spec["min_pairs_for_estimation"])
    residual = enriched["residual"]
    c_combined = enriched["c_combined"]
    lever = enriched["lever_arm_mm"]
    target_station = enriched["target_station_id"]
    run_id = enriched["run_id"]
    runs = sorted(set(int(v) for v in run_id))
    out = np.zeros_like(c_combined)
    for run in runs:
        others = [r for r in runs if r != run]
        if len(others) != 1:
            raise ValueError("cross-fit model covariance requires exactly two runs")
        scales = derive_pairtype_scales(
            kind, c_combined, lever, residual, run_id, target_station, others[0], min_pairs
        )
        rmask = run_id == run
        out[rmask] = assemble_model_covariance(
            kind, c_combined[rmask], lever[rmask], target_station[rmask], scales
        )
    return out


def cross_fit_covariance_validation(
    config: Mapping[str, Any], enriched: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Cross-fit (14973<->14974) validation of the pre-registered candidate
    covariance models.  No candidate enters an alignment solve; the empirical
    covariance is never promoted to a weight matrix.
    """
    spec = config["covariance_model"]
    val_spec = config["covariance_validation"]
    residual = enriched["residual"]
    c_combined = enriched["c_combined"]
    lever = enriched["lever_arm_mm"]
    target_station = enriched["target_station_id"]
    run_id = enriched["run_id"]
    min_pairs = int(spec["min_pairs_for_estimation"])

    folds = [(int(f["derive"]), int(f["validate"])) for f in spec["cross_fit_folds"]]
    candidates = [c for c in spec["candidates"]]

    results: dict[str, Any] = {}
    for cand in candidates:
        name = str(cand["name"])
        kind = str(cand["kind"])
        fold_reports = {}
        fold_pass = []
        for derive_run, validate_run in folds:
            per_pairtype = {}
            fold_ok = True
            for pair in STATION_PAIRS:
                station = pair[1]
                label = f"0->{station}"
                dmask = (run_id == derive_run) & (target_station == station)
                vmask = (run_id == validate_run) & (target_station == station)
                if dmask.sum() < min_pairs or vmask.sum() < min_pairs:
                    per_pairtype[label] = {
                        "n_derive": int(dmask.sum()),
                        "n_validate": int(vmask.sum()),
                        "sufficient": False,
                    }
                    continue
                # De-mean each run's pairs by their own (run, pair-type) cell mean.
                demean_d = _demean_cells(
                    residual[dmask], run_id[dmask], target_station[dmask]
                )
                demean_v = _demean_cells(
                    residual[vmask], run_id[vmask], target_station[vmask]
                )
                floor = _pairtype_marginal_floor(c_combined[dmask])
                est = _estimate_scale_nll(
                    kind, c_combined[dmask], lever[dmask], demean_d, floor
                )
                scale = float(est["scale"])
                c_model_val = build_candidate_covariance(
                    kind, c_combined[vmask], lever[vmask], scale, floor
                )
                # ndof: validation pairs*obs minus one 4-vector mean per cell.
                n_cells = len(
                    set(zip(run_id[vmask].tolist(), target_station[vmask].tolist()))
                )
                ndof = int(vmask.sum()) * N_OBS - n_cells * N_OBS
                metrics = _validation_metrics(
                    c_model_val, demean_v, {"ndof": ndof}
                )
                # Gates.
                gates = {
                    "whitened_chi2_within_gate": bool(
                        metrics["whitened_chi2_per_ndof"]
                        <= float(val_spec["whitened_chi2_per_ndof_max"])
                    ),
                    "cov_z_eigenvalues_within_gate": bool(
                        metrics["cov_z_eigenvalue_min"] >= float(val_spec["cov_z_eigenvalue_min"])
                        and metrics["cov_z_eigenvalue_max"] <= float(val_spec["cov_z_eigenvalue_max"])
                    ),
                    "positive_definite": bool(metrics["positive_definite"]),
                    "factorization_stable": bool(
                        metrics["factorization_max_inverse_residual"]
                        <= float(val_spec["factorization_max_inverse_residual"])
                    )
                    if kind != "frozen_baseline"
                    else None,  # frozen near-singular baseline is documented, not gated
                }
                gated = [v for v in gates.values() if v is not None]
                pair_ok = bool(all(gated))
                fold_ok = fold_ok and pair_ok
                per_pairtype[label] = {
                    "n_derive": int(dmask.sum()),
                    "n_validate": int(vmask.sum()),
                    "sufficient": True,
                    "derived_scale": scale,
                    "derivation_nll": float(est["nll"]),
                    **metrics,
                    "gates": gates,
                    "within_gate": pair_ok,
                }
            fold_reports[f"derive{derive_run}_validate{validate_run}"] = {
                "per_pair_type": per_pairtype,
                "fold_pass": bool(fold_ok),
            }
            fold_pass.append(bool(fold_ok))

        # Cross-run reproducibility of the derived scale (report) + empirical
        # covariance generalized-eigenvalue deviation (gate, inherit WB79).
        scales = {}
        for pair in STATION_PAIRS:
            label = f"0->{pair[1]}"
            s = []
            for derive_run, validate_run in folds:
                rep = fold_reports[f"derive{derive_run}_validate{validate_run}"][
                    "per_pair_type"
                ][label]
                if rep.get("sufficient"):
                    s.append(rep["derived_scale"])
            if len(s) == 2 and all(v > 0 for v in s):
                scales[label] = {
                    "scale_fold_A": float(s[0]),
                    "scale_fold_B": float(s[1]),
                    "scale_log_ratio": float(abs(math.log(s[0] / s[1]))),
                }
        results[name] = {
            "kind": kind,
            "n_nuisance": int(cand.get("n_nuisance", 0)),
            "folds": fold_reports,
            "cross_run_scale_reproducibility": scales,
            "validated_both_folds": bool(all(fold_pass)) if fold_pass else False,
        }

    # covariance_model_validated: at least one physically-motivated candidate
    # (not the frozen baseline) validates on both folds.
    physical = [
        name
        for name, rep in results.items()
        if rep["kind"] != "frozen_baseline" and rep["validated_both_folds"]
    ]
    return {
        "kind": "covariance_model_cross_fit_validation",
        "candidates": results,
        "validated_physical_candidates": physical,
        "covariance_model_validated": bool(len(physical) > 0),
        "cross_fit_folds": [list(f) for f in folds],
        "no_candidate_entered_alignment_solve": True,
        "empirical_covariance_not_promoted_to_weight": True,
        "diagnostic_only": True,
        "alignment_authorized": False,
    }


# ---------------------------------------------------------------------------
# Stage: numerical inversion audit (method A vs model change B)
# ---------------------------------------------------------------------------


def numerical_inversion_audit(
    config: Mapping[str, Any], enriched: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Compare numerical methods for applying C^{-1} on the SAME frozen
    covariance.  Strictly separates (A) stable evaluation of the same inverse
    from (B) modifying C (a statistical-model change, NOT done here)."""
    c_combined = enriched["c_combined"]
    residual = enriched["residual"]
    n_pairs = residual.shape[0]

    def chi2_direct(c, r):
        return float(r @ np.linalg.inv(c) @ r)

    def chi2_cholesky(c, r):
        L = np.linalg.cholesky(c)
        z = np.linalg.solve(L, r)
        return float(z @ z)

    def chi2_eigh(c, r):
        w, v = np.linalg.eigh(c)
        w = np.maximum(w, 1.0e-300)
        proj = v.T @ r
        return float(np.sum(proj ** 2 / w))

    def chi2_svd(c, r):
        # Hermitian pseudo-inverse with no relative cutoff (pure; for a strictly
        # PD matrix this equals the inverse).
        w, v = np.linalg.eigh(c)
        inv = (v * (1.0 / np.maximum(w, 1.0e-300))) @ v.T
        return float(r @ inv @ r)

    methods = {
        "direct_inv": chi2_direct,
        "cholesky_solve": chi2_cholesky,
        "eigendecomposition": chi2_eigh,
        "svd_pseudoinverse": chi2_svd,
    }
    totals = {name: 0.0 for name in methods}
    failures = {name: 0 for name in methods}
    max_inv_resid = {name: 0.0 for name in methods}
    for i in range(n_pairs):
        c = c_combined[i]
        r = residual[i]
        eye = np.eye(N_OBS)
        ref = None
        for name, fn in methods.items():
            try:
                totals[name] += fn(c, r)
                if name == "direct_inv":
                    inv = np.linalg.inv(c)
                elif name == "cholesky_solve":
                    L = np.linalg.cholesky(c)
                    inv = np.linalg.solve(L, eye)
                    inv = inv.T @ inv
                elif name == "eigendecomposition":
                    w, v = np.linalg.eigh(c)
                    inv = (v * (1.0 / np.maximum(w, 1.0e-300))) @ v.T
                else:
                    w, v = np.linalg.eigh(c)
                    inv = (v * (1.0 / np.maximum(w, 1.0e-300))) @ v.T
                max_inv_resid[name] = max(
                    max_inv_resid[name], float(np.max(np.abs(c @ inv - eye)))
                )
            except np.linalg.LinAlgError:
                failures[name] += 1
    ref_total = totals["eigendecomposition"]
    rel_diff = {
        name: abs(totals[name] - ref_total) / max(abs(ref_total), 1.0e-300)
        for name in methods
    }
    return {
        "kind": "numerical_inversion_audit",
        "n_pairs": n_pairs,
        "total_chi2_by_method": {k: float(v) for k, v in totals.items()},
        "relative_difference_vs_eigendecomposition": {k: float(v) for k, v in rel_diff.items()},
        "factorization_failures": failures,
        "max_inverse_residual_CCinv_minus_I": {k: float(v) for k, v in max_inv_resid.items()},
        "note": "All methods evaluate the SAME frozen covariance (numerical method A). "
        "No eigenvalue floor/cap/pseudoinverse cutoff is applied to C itself; such a "
        "modification would be a statistical-model change (B) requiring separate "
        "pre-registration.",
        "diagnostic_only": True,
        "alignment_authorized": False,
    }
