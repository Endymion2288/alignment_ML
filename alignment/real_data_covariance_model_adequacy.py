"""Workbook-79 real-data residual/covariance model adequacy V1.

Single scientific question: is the Workbook-78 out-of-support candidate
(|gamma|max = 2.46, 16x the frozen 0.15 linear envelope) caused by

* H1 - a genuine large nonlinear geometry displacement (the
  residual/covariance statistical model itself is adequate), or
* H2 - a covariance/statistical-model mismatch (the combined 4x4 covariance's
  correlation structure mis-weights near-singular eigen-directions and
  inflates chi2 and the inferred gamma), or
* H3 - an observable/Jacobian-transfer mismatch (station-pair mean MC
  Jacobian transfer error on the real collision topology)?

Until this is answered, real-data nonlinear alignment is forbidden.

This campaign is model-adequacy ONLY.  It produces no correction, opens no
held-out data, writes no geometry, and authorizes no alignment:

``geometry_write_allowed = false``, ``official_conditions_write_allowed =
false``, ``real_data_candidate_alignment_authorized = false``,
``external_constraint_ingest_authorized = false``, ``held_out_accessed =
false``.

The frozen Workbook-78 candidate artifact is used only as an
``out_of_support_diagnostic_candidate``; it is never a correction and is never
scaled, clipped, or applied.

Frozen inputs (all SHA-pinned, inherited from Workbook 78):

* Calibration population = runs 14973 + 14974 ONLY (frozen split SHA
  ``c9c35979...``); held-out (14975/14976, 7 monitoring runs, 14977
  report-only) remains completely closed.
* Tracker information / gauge / transfer contracts from the frozen WB78
  config, reused verbatim (loaded and SHA-verified, never duplicated).
* The WB78 calibration baseline (bank SHA, chi2, gamma, beta, eigenvalues,
  rank, condition, bootstrap) which Stage-0 must reproduce exactly before any
  audit quantity is computed.

All diagnostic counterfactual covariance models (diagonal / condition-capped /
unit) are ``diagnostic_only`` and ``alignment_authorized = false``; they never
enter the production solver and never produce a deployable candidate.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment import gauge_fixed_real_data_diagnostic as wb78
from alignment.cad_survey_nov22 import git_head_sha
from alignment.identifiable_subspace import IdentifiableSubspace
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from datasets.propagation_loader import load_propagation_records

SCHEMA_VERSION = "faser-real-data-residual-covariance-model-adequacy-v1"
DEFAULT_CONFIG_RELATIVE = Path("configs/real_data_residual_covariance_model_adequacy_v1.yaml")

OBSERVABLE_NAMES = wb78.OBSERVABLE_NAMES
STATION_PAIRS = wb78.STATION_PAIRS

DECISION_PASS = "real_data_model_adequacy_pass_nonlinear_response_preregistration_allowed"
DECISION_INCONCLUSIVE = "real_data_model_adequacy_inconclusive"
DECISION_MULTIPLE = "real_data_statistical_model_multiple_failures"
DECISION_COVARIANCE = "real_data_covariance_model_not_adequate"
DECISION_TRANSFER = "real_data_jacobian_transfer_support_not_adequate"
DECISION_CROSS_RUN = "real_data_calibration_information_not_cross_run_stable"

CONFIG_MUST_BE_TRUE = (
    "do_not_open_held_out",
    "do_not_select_events_by_residual",
    "do_not_drop_outlier_pairs",
    "do_not_switch_gauge",
    "do_not_repick_split",
    "do_not_tune_S_or_rank_tolerance",
    "do_not_redefine_identifiable_basis_on_real_data",
    "do_not_use_external_evidence_as_prior",
    "do_not_generate_fd_probes",
    "do_not_run_newton",
    "do_not_write_payload",
    "do_not_promote_counterfactual_covariance_to_alignment",
    "do_not_promote_empirical_covariance_to_weight",
    "do_not_scale_or_clip_wb78_candidate",
    "do_not_reopen_identifiability_rescue",
    "do_not_start_temporary_reconstruction_validation",
)
CONFIG_MUST_BE_FALSE = (
    "geometry_write_allowed",
    "official_conditions_write_allowed",
    "real_data_candidate_alignment_authorized",
    "external_constraint_ingest_authorized",
    "held_out_accessed",
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG_RELATIVE))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    for key in CONFIG_MUST_BE_FALSE:
        if config.get(key) is not False:
            raise ValueError(f"config must set {key}: false")
    for key in CONFIG_MUST_BE_TRUE:
        if config.get(key) is not True:
            raise ValueError(f"config must set {key}: true")
    if not list(config["eligible_external_physical_constraints"]) == []:
        raise ValueError("external constraint branch remains closed: eligibility list must be empty")
    if str(config["output_parameter_naming"]) != "out_of_support_diagnostic_candidate":
        raise ValueError("WB78 candidate may only be named out_of_support_diagnostic_candidate")

    inheritance = config["inheritance"]
    wb78_config_path = resolve_under_root(project_root(), str(inheritance["workbook_78_config"]))
    if sha256_file(wb78_config_path) != str(inheritance["workbook_78_config_sha256"]):
        raise ValueError("workbook-78 config SHA256 mismatch")
    wb78_root = resolve_under_root(project_root(), str(inheritance["workbook_78_output_root"]))
    for name, expected in inheritance["workbook_78_artifact_sha256"].items():
        actual = sha256_file(wb78_root / name)
        if actual != str(expected):
            raise ValueError(f"workbook-78 artifact SHA256 mismatch for {name}")

    # Load the frozen WB78 config and reuse its contracts verbatim.
    wb78_config = wb78.load_config(wb78_config_path)
    config["wb78_config"] = wb78_config
    config["config_path"] = str(config_path)
    return config


# ---------------------------------------------------------------------------
# Stage 0: reproduce the frozen WB78 calibration baseline (hard stop)
# ---------------------------------------------------------------------------


def reproduce_baseline(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild the WB78 calibration bank + one-shot solve and verify the frozen
    baseline numbers exactly.  Any mismatch is a hard provenance failure.

    Returns the rebuilt objects needed by every downstream audit so that no
    audit ever rebuilds the bank differently.
    """
    wb78_config = config["wb78_config"]
    baseline = config["inheritance"]["baseline"]
    rtol = float(config["inheritance"]["reproduce_rtol"])
    atol = float(config["inheritance"]["reproduce_atol"])

    banks, pooled, subspace, extras, regression = wb78.load_tracker_information(wb78_config)
    if not regression["pass"]:
        raise ValueError("tracker information regression failed; WB79 refused")
    transfer = wb78.build_transfer_model(pooled, extras)
    gauges = wb78.build_gauge_candidates(wb78_config, subspace)
    bank = wb78.build_real_data_bank(wb78_config, roles=wb78_config["split"]["calibration_roles"])
    if not bank["csv_reproduction"]["bit_exact"]:
        raise ValueError("bank/CSV reproduction mismatch; WB79 refused")

    # Bank identity must match the frozen calibration bank SHA.
    bank_sha = wb78.bank_sha256(bank)
    bank_sha_ok = bank_sha == str(config["inheritance"]["calibration_bank_sha256"])

    audit = wb78.applicability_audit(wb78_config, bank, transfer, pooled)
    candidate = wb78.solve_candidate(wb78_config, subspace, transfer, bank, gauges)

    arrays = wb78.concatenate_banks(bank)
    chi2_zero = float(audit["weighted_residual_norm_zero_candidate"])

    def _close(a: float, b: float) -> bool:
        return bool(np.isclose(a, b, rtol=rtol, atol=atol))

    checks = {
        "bank_sha256": bank_sha_ok,
        "n_pairs": int(arrays["residual"].shape[0]) == int(baseline["n_pairs"]),
        "chi2_zero": _close(chi2_zero, float(baseline["chi2_zero_candidate"])),
        "informed_rank": int(candidate["informed_rank"]) == int(baseline["informed_rank"]),
        "restricted_condition": _close(
            float(candidate["restricted_condition"]), float(baseline["restricted_condition"])
        ),
        "gamma": bool(
            np.allclose(
                np.asarray(candidate["gamma_hat_informed_coordinates"], dtype=np.float64),
                np.asarray(baseline["gamma_hat_informed_coordinates"], dtype=np.float64),
                rtol=rtol,
                atol=atol,
            )
        ),
        "beta": bool(
            np.allclose(
                np.asarray(candidate["beta_hat_identifiable_projection"], dtype=np.float64),
                np.asarray(baseline["beta_hat_identifiable_projection"], dtype=np.float64),
                rtol=rtol,
                atol=atol,
            )
        ),
        "information_eigenvalues": bool(
            np.allclose(
                np.asarray(candidate["information_eigenvalues"], dtype=np.float64),
                np.asarray(baseline["information_eigenvalues"], dtype=np.float64),
                rtol=rtol,
                atol=atol,
            )
        ),
        "bootstrap_rank_changes": int(candidate["bootstrap"]["rank_changes"])
        == int(baseline["bootstrap_rank_changes"]),
        "bootstrap_max_direction_angle_deg": _close(
            float(candidate["bootstrap"]["max_direction_angle_deg"]),
            float(baseline["bootstrap_max_direction_angle_deg"]),
        ),
    }
    per_run_ok = True
    for run, expected in baseline["per_run_chi2_zero"].items():
        actual = float(audit["per_run"][str(run)]["chi2_zero_candidate"])
        per_run_ok = per_run_ok and _close(actual, float(expected))
        per_run_ok = per_run_ok and int(audit["per_run"][str(run)]["n_pairs"]) == int(
            baseline["per_run_n_pairs"][run]
        )
    checks["per_run_chi2"] = per_run_ok

    report = {
        "kind": "wb78_baseline_reproduction",
        "bank_sha256": bank_sha,
        "chi2_zero": chi2_zero,
        "candidate_status": candidate.get("status"),
        "within_linear_envelope": bool(candidate.get("within_linear_envelope", False)),
        "checks": checks,
        "pass": bool(all(checks.values())),
        "note": "WB78 candidate reproduced only as an out_of_support_diagnostic_candidate; "
        "never scaled, clipped, or applied",
    }
    return {
        "report": report,
        "bank": bank,
        "pooled": pooled,
        "subspace": subspace,
        "extras": extras,
        "transfer": transfer,
        "gauges": gauges,
        "audit": audit,
        "candidate": candidate,
    }


# ---------------------------------------------------------------------------
# Shared linear-algebra helpers
# ---------------------------------------------------------------------------


def _whiten(covariance: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """z = L^{-1} vector per pair via Cholesky (covariance assumed SPD)."""
    n = covariance.shape[0]
    out = np.zeros((n, vector.shape[1]), dtype=np.float64)
    for i in range(n):
        chol = np.linalg.cholesky(covariance[i])
        out[i] = np.linalg.solve(chol, vector[i])
    return out


def _cell_labels(run_id: np.ndarray, target_station_id: np.ndarray) -> list[tuple[int, int]]:
    return [(int(run_id[i]), int(target_station_id[i])) for i in range(run_id.size)]


# ---------------------------------------------------------------------------
# Stage 1: covariance eigenstructure + whitening audit (H2 primary)
# ---------------------------------------------------------------------------


def covariance_eigen_audit(
    config: Mapping[str, Any],
    bank: Mapping[str, Any],
) -> dict[str, Any]:
    """Frozen-covariance eigenstructure and whitening audit on the calibration
    subset.  Report-first; nothing here selects or drops pairs.
    """
    arrays = wb78.concatenate_banks(bank)
    residual = np.asarray(arrays["residual"], dtype=np.float64)
    covariance = np.asarray(arrays["covariance"], dtype=np.float64)
    target_station = np.asarray(arrays["target_station_id"], dtype=np.int64)
    run_id = np.asarray(arrays["run_id_arr"], dtype=np.int64)
    n_pairs, n_obs = residual.shape

    # Per-pair covariance eigenstructure.
    eigvals = np.zeros((n_pairs, n_obs))
    eigvecs = np.zeros((n_pairs, n_obs, n_obs))
    condition = np.zeros(n_pairs)
    min_eigval = np.zeros(n_pairs)
    for i in range(n_pairs):
        w, v = np.linalg.eigh(covariance[i])
        eigvals[i] = w
        eigvecs[i] = v
        condition[i] = w[-1] / max(w[0], 1.0e-300)
        min_eigval[i] = w[0]

    # chi2 decomposition by covariance eigenmode: r = sum_a (r.v_a) v_a,
    # chi2_i = sum_a (r.v_a)^2 / lambda_a.
    proj = np.einsum("ni,nia->na", residual, eigvecs)  # (n, obs) projection coeffs
    mahalanobis_by_mode = proj**2 / np.maximum(eigvals, 1.0e-300)
    chi2_total = float(np.sum(mahalanobis_by_mode))
    chi2_by_mode = mahalanobis_by_mode.sum(axis=0)  # ascending eigenvalue order
    fraction_smallest_mode = float(chi2_by_mode[0] / max(chi2_total, 1.0e-300))

    # Marginal pulls.
    marginal_pull = residual / np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))

    # Whitening with coherent-mean removal (per run x pair cell).
    cells = _cell_labels(run_id, target_station)
    unique_cells = sorted(set(cells))
    centered = residual.copy()
    cell_counts: dict[tuple[int, int], int] = {}
    for cell in unique_cells:
        rows = np.array([i for i, c in enumerate(cells) if c == cell], dtype=np.intp)
        cell_counts[cell] = int(rows.size)
        centered[rows] = residual[rows] - residual[rows].mean(axis=0)
    z = _whiten(covariance, centered)
    chi2_resid = float(np.sum(z**2))
    n_means = sum(1 for c in unique_cells if cell_counts[c] > 0)
    ndof = n_pairs * n_obs - n_means * n_obs
    chi2_resid_per_ndof = chi2_resid / max(ndof, 1)

    # Empirical covariance of the whitened residual (after mean removal).
    cov_z = np.cov(z, rowvar=False)
    cov_z_eigvals = np.linalg.eigvalsh(cov_z)
    cov_z_condition = float(cov_z_eigvals[-1] / max(cov_z_eigvals[0], 1.0e-300))

    gate = float(config["covariance_adequacy"]["whitened_chi2_per_ndof_max"])

    # Per (run, pair-type) summaries.
    per_cell = {}
    for cell in unique_cells:
        rows = np.array([i for i, c in enumerate(cells) if c == cell], dtype=np.intp)
        per_cell[f"run{cell[0]}_0->{cell[1]}"] = {
            "n_pairs": int(rows.size),
            "chi2": float(np.sum(mahalanobis_by_mode[rows])),
            "median_condition": float(np.median(condition[rows])),
            "median_min_eigenvalue": float(np.median(min_eigval[rows])),
        }

    return {
        "kind": "covariance_eigenstructure_whitening_audit",
        "n_pairs": n_pairs,
        "chi2_total": chi2_total,
        "chi2_per_ndof_zero_candidate": chi2_total / (n_pairs * n_obs),
        "chi2_by_covariance_eigenmode_ascending": [float(v) for v in chi2_by_mode],
        "fraction_chi2_from_smallest_eigenmode": fraction_smallest_mode,
        "covariance_condition_median": float(np.median(condition)),
        "covariance_condition_p95": float(np.percentile(condition, 95.0)),
        "covariance_min_eigenvalue_median": float(np.median(min_eigval)),
        "marginal_pull_rms_per_observable": {
            OBSERVABLE_NAMES[m]: float(np.sqrt(np.mean(marginal_pull[:, m] ** 2)))
            for m in range(n_obs)
        },
        "whitened_chi2_resid_after_mean_removal": chi2_resid,
        "whitened_ndof": int(ndof),
        "whitened_chi2_resid_per_ndof": chi2_resid_per_ndof,
        "whitened_residual_covariance_eigenvalues": [float(v) for v in cov_z_eigvals],
        "whitened_residual_covariance_condition": cov_z_condition,
        "whitened_chi2_per_ndof_gate": gate,
        "whitened_chi2_within_gate": bool(chi2_resid_per_ndof <= gate),
        "per_cell": per_cell,
        "no_pair_dropped": True,
        "diagnostic_only": True,
        "alignment_authorized": False,
    }


# ---------------------------------------------------------------------------
# Stage 2: empirical covariance cross-check (14973 <-> 14974)
# ---------------------------------------------------------------------------


def _empirical_covariance(residual: np.ndarray) -> np.ndarray:
    centered = residual - residual.mean(axis=0)
    return np.cov(centered, rowvar=False)


def empirical_covariance_crosscheck(
    config: Mapping[str, Any],
    bank: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-run empirical residual covariance structure audit.

    Compares the empirical residual covariance between runs 14973 and 14974
    per station-pair type (0,1) and (0,2) (sufficient statistics), and to the
    propagated combined covariance.  This is a model-adequacy audit only; the
    empirical covariance is never promoted to a weight matrix.
    """
    arrays = wb78.concatenate_banks(bank)
    residual = np.asarray(arrays["residual"], dtype=np.float64)
    covariance = np.asarray(arrays["covariance"], dtype=np.float64)
    target_station = np.asarray(arrays["target_station_id"], dtype=np.int64)
    run_id = np.asarray(arrays["run_id_arr"], dtype=np.int64)
    min_pairs = int(config["covariance_adequacy"]["min_pairs_for_empirical_covariance"])
    rms_log_gate = float(
        config["covariance_adequacy"]["cross_run_generalized_eigenvalue_max_rms_log"]
    )

    runs = sorted(set(int(v) for v in run_id))
    per_pair_type = {}
    reproducible = True
    for pair in STATION_PAIRS:
        station = pair[1]
        label = f"0->{station}"
        cell = {}
        emp = {}
        for run in runs:
            rows = np.flatnonzero((target_station == station) & (run_id == run))
            if rows.size < min_pairs:
                cell[str(run)] = {"n_pairs": int(rows.size), "sufficient": False}
                continue
            emp_cov = _empirical_covariance(residual[rows])
            prop_cov = covariance[rows].mean(axis=0)
            e_emp, v_emp = np.linalg.eigh(emp_cov)
            e_prop = np.linalg.eigvalsh(prop_cov)
            emp[run] = (e_emp, v_emp)
            cell[str(run)] = {
                "n_pairs": int(rows.size),
                "sufficient": True,
                "empirical_eigenvalues": [float(v) for v in e_emp],
                "propagated_eigenvalues": [float(v) for v in e_prop],
                "empirical_marginal_variance": [
                    float(emp_cov[m, m]) for m in range(emp_cov.shape[0])
                ],
                "propagated_marginal_variance": [
                    float(prop_cov[m, m]) for m in range(prop_cov.shape[0])
                ],
            }
        if len(emp) == 2:
            r1, r2 = runs[0], runs[1]
            e1, v1 = emp[r1]
            e2, v2 = emp[r2]
            # Primary basis-independent metric: generalized-eigenvalue RMS
            # log-deviation.  Whiten run-2's empirical covariance with run-1's
            # and measure the RMS deviation of the generalized eigenvalues from
            # 1 (robust to near-degenerate spectra).
            c1 = _empirical_covariance(residual[(target_station == station) & (run_id == r1)])
            c2 = _empirical_covariance(residual[(target_station == station) & (run_id == r2)])
            w1, v1c = np.linalg.eigh(c1)
            half = (v1c * np.sqrt(np.maximum(w1, 1.0e-300))) @ v1c.T
            half_inv = np.linalg.inv(half)
            generalized = np.linalg.eigvalsh(half_inv @ c2 @ half_inv)
            rms_log = float(np.sqrt(np.mean(np.log(np.maximum(generalized, 1.0e-300)) ** 2)))
            # Report-only context: principal eigenvector angle + log-ratios.
            cosine = float(abs(np.dot(v1[:, -1], v2[:, -1])))
            angle = math.degrees(math.acos(min(max(cosine, -1.0), 1.0)))
            log_ratio = float(
                np.max(np.abs(np.log(np.maximum(e1, 1.0e-300) / np.maximum(e2, 1.0e-300))))
            )
            cell["cross_run"] = {
                "generalized_eigenvalue_rms_log": rms_log,
                "generalized_eigenvalues": [float(v) for v in generalized],
                "principal_eigenvector_angle_deg": angle,
                "eigenvalue_max_log_ratio": log_ratio,
                "within_gate": bool(rms_log <= rms_log_gate),
            }
            reproducible = reproducible and rms_log <= rms_log_gate
        per_pair_type[label] = cell

    return {
        "kind": "empirical_covariance_crosscheck",
        "runs": [int(r) for r in runs],
        "generalized_eigenvalue_rms_log_gate": rms_log_gate,
        "per_pair_type": per_pair_type,
        "cross_run_reproducible": bool(reproducible),
        "empirical_covariance_not_promoted_to_weight": True,
        "diagnostic_only": True,
        "alignment_authorized": False,
    }


# ---------------------------------------------------------------------------
# Stage 3: alignment-score contribution decomposition (report-only)
# ---------------------------------------------------------------------------


def score_contribution_decomposition(
    config: Mapping[str, Any],
    subspace: IdentifiableSubspace,
    transfer: Mapping[str, Any],
    bank: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Decompose the informed-direction score b = J^T W r by run, station-pair,
    observable, and covariance eigenmode.  Answers whether the large gamma is
    driven coherently by many pairs or by a few near-singular directions.
    Report-only; no pair is dropped.
    """
    arrays = wb78.concatenate_banks(bank)
    residual = np.asarray(arrays["residual"], dtype=np.float64)
    covariance = np.asarray(arrays["covariance"], dtype=np.float64)
    target_station = np.asarray(arrays["target_station_id"], dtype=np.int64)
    run_id = np.asarray(arrays["run_id_arr"], dtype=np.int64)
    design = wb78._pair_design_rows(transfer, target_station, subspace)
    basis = np.asarray(
        [col for col in candidate["informed_basis_columns"]], dtype=np.float64
    ).T  # (5, rank)
    rank = basis.shape[1]
    n_pairs = residual.shape[0]

    # Per-pair, per-informed-direction score contribution c_ik, with
    # decompositions.
    by_run = {str(r): np.zeros(rank) for r in sorted(set(int(v) for v in run_id))}
    by_pair = {f"0->{p[1]}": np.zeros(rank) for p in STATION_PAIRS}
    by_observable = {name: np.zeros(rank) for name in OBSERVABLE_NAMES}
    by_eigenmode = np.zeros((residual.shape[1], rank))  # ascending eigenvalue
    abs_contrib = np.zeros((n_pairs, rank))
    signed_contrib = np.zeros((n_pairs, rank))

    for i in range(n_pairs):
        cov = covariance[i]
        weight = np.linalg.inv(cov)
        wr = weight @ residual[i]  # (4,)
        e_vals, e_vecs = np.linalg.eigh(cov)
        proj = residual[i] @ e_vecs  # (4,) projection coefficients
        for k in range(rank):
            au = design[i] @ basis[:, k]  # (4,) design row onto direction k
            c = float(au @ wr)
            signed_contrib[i, k] = c
            abs_contrib[i, k] = abs(c)
            by_run[str(int(run_id[i]))][k] += c
            by_pair[f"0->{int(target_station[i])}"][k] += c
            for m in range(residual.shape[1]):
                by_observable[OBSERVABLE_NAMES[m]][k] += au[m] * wr[m]
            for a in range(residual.shape[1]):
                by_eigenmode[a, k] += (proj[a] / max(e_vals[a], 1.0e-300)) * float(
                    au @ e_vecs[:, a]
                )

    # Cumulative concentration: fraction of total |score| from top X% pairs.
    concentration = {}
    for k in range(rank):
        order = np.argsort(-abs_contrib[:, k])
        total = float(np.sum(abs_contrib[:, k]))
        sorted_abs = abs_contrib[order, k]
        cum = np.cumsum(sorted_abs)
        entry = {}
        for frac in (0.01, 0.05, 0.10, 0.50):
            count = max(1, int(math.ceil(frac * n_pairs)))
            entry[f"top_{int(frac*100)}pct_abs_fraction"] = float(
                cum[count - 1] / max(total, 1.0e-300)
            )
        concentration[f"informed_direction_{k}"] = entry

    return {
        "kind": "alignment_score_contribution_decomposition",
        "informed_rank": int(rank),
        "score_by_run": {k: [float(v) for v in val] for k, val in by_run.items()},
        "score_by_station_pair": {k: [float(v) for v in val] for k, val in by_pair.items()},
        "score_by_observable": {k: [float(v) for v in val] for k, val in by_observable.items()},
        "score_by_covariance_eigenmode_ascending": [
            [float(v) for v in by_eigenmode[a]] for a in range(residual.shape[1])
        ],
        "concentration": concentration,
        "no_pair_dropped": True,
        "diagnostic_only": True,
        "alignment_authorized": False,
    }


# ---------------------------------------------------------------------------
# Stage 4 + 5: bootstrap-instability decomposition and covariance
# counterfactuals (diagnostic-only)
# ---------------------------------------------------------------------------


def _counterfactual_covariance(
    covariance: np.ndarray, kind: str, condition_cap: float
) -> np.ndarray:
    n = covariance.shape[0]
    out = np.zeros_like(covariance)
    for i in range(n):
        cov = covariance[i]
        if kind == "full":
            out[i] = cov
        elif kind == "diagonal":
            out[i] = np.diag(np.diagonal(cov))
        elif kind == "unit":
            out[i] = np.eye(cov.shape[0])
        elif kind == "condition_capped":
            e_vals, e_vecs = np.linalg.eigh(cov)
            floor = e_vals[-1] / float(condition_cap)
            out[i] = (e_vecs * np.maximum(e_vals, floor)) @ e_vecs.T
        else:
            raise ValueError(f"unknown counterfactual covariance kind {kind}")
    return out


def _solve_with_covariance(
    config: Mapping[str, Any],
    subspace: IdentifiableSubspace,
    transfer: Mapping[str, Any],
    residual: np.ndarray,
    covariance: np.ndarray,
    target_station: np.ndarray,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    design = wb78._pair_design_rows(transfer, target_station, subspace)
    information, rhs = wb78._information_and_rhs(design, covariance, residual)
    solved = wb78._solve_informed(
        information, rhs, rank_tolerance=float(config["wb78_config"]["solver"]["rank_tolerance"])
    )
    # chi2 concentration under this covariance.
    chi2 = 0.0
    chi2_by_mode = np.zeros(residual.shape[1])
    for i in range(residual.shape[0]):
        w = np.linalg.inv(covariance[i])
        chi2 += float(residual[i] @ w @ residual[i])
        e_vals, e_vecs = np.linalg.eigh(covariance[i])
        proj = residual[i] @ e_vecs
        chi2_by_mode += proj**2 / np.maximum(e_vals, 1.0e-300)
    # Bootstrap stability.
    rng = np.random.default_rng(bootstrap_seed)
    n_pairs = residual.shape[0]
    rank_changes = 0
    angles: list[float] = []
    beta0 = solved["beta"]
    for _ in range(bootstrap_replicates):
        rows = []
        for pair in STATION_PAIRS:
            available = np.flatnonzero(target_station == pair[1])
            if available.size:
                rows.append(rng.choice(available, size=available.size, replace=True))
        sample = np.concatenate(rows)
        b_design = wb78._pair_design_rows(transfer, target_station[sample], subspace)
        b_info, b_rhs = wb78._information_and_rhs(
            b_design, covariance[sample], residual[sample]
        )
        b_solved = wb78._solve_informed(
            b_info, b_rhs, rank_tolerance=float(config["wb78_config"]["solver"]["rank_tolerance"])
        )
        if b_solved["rank"] != solved["rank"]:
            rank_changes += 1
            continue
        cosine = float(
            np.dot(b_solved["beta"], beta0)
            / max(np.linalg.norm(b_solved["beta"]) * np.linalg.norm(beta0), 1.0e-300)
        )
        angles.append(math.degrees(math.acos(min(max(cosine, -1.0), 1.0))))
    return {
        "informed_rank": int(solved["rank"]),
        "information_eigenvalues": [float(v) for v in solved["eigenvalues"]],
        "gamma": [float(v) for v in solved["gamma"]],
        "beta": [float(v) for v in solved["beta"]],
        "max_abs_gamma": float(np.max(np.abs(solved["gamma"]))) if solved["rank"] else 0.0,
        "chi2": chi2,
        "chi2_fraction_smallest_eigenmode": float(chi2_by_mode[0] / max(chi2, 1.0e-300)),
        "bootstrap_rank_changes": int(rank_changes),
        "bootstrap_max_direction_angle_deg": max(angles) if angles else math.inf,
    }


def covariance_counterfactuals(
    config: Mapping[str, Any],
    subspace: IdentifiableSubspace,
    transfer: Mapping[str, Any],
    bank: Mapping[str, Any],
) -> dict[str, Any]:
    """Diagnostic-only covariance counterfactuals (A/B/C/D).  None of these
    enter the production solver or produce a deployable candidate; they exist
    only to localize the cause of the large gamma.
    """
    arrays = wb78.concatenate_banks(bank)
    residual = np.asarray(arrays["residual"], dtype=np.float64)
    covariance = np.asarray(arrays["covariance"], dtype=np.float64)
    target_station = np.asarray(arrays["target_station_id"], dtype=np.int64)
    spec = config["covariance_counterfactuals"]
    cap = float(spec["condition_cap"])
    n_boot = int(spec["bootstrap_replicates"])
    seed = int(spec["bootstrap_seed"])

    baseline_beta = None
    results = {}
    for item in spec["counterfactuals"]:
        name = str(item["name"])
        kind = str(item["kind"])
        cov_cf = _counterfactual_covariance(covariance, kind, cap)
        solved = _solve_with_covariance(
            config,
            subspace,
            transfer,
            residual,
            cov_cf,
            target_station,
            bootstrap_replicates=n_boot,
            bootstrap_seed=seed,
        )
        beta = np.asarray(solved["beta"], dtype=np.float64)
        if kind == "full":
            baseline_beta = beta
        angle_to_baseline = None
        if baseline_beta is not None and np.linalg.norm(beta) > 0 and np.linalg.norm(baseline_beta) > 0:
            cosine = float(
                np.dot(beta, baseline_beta)
                / (np.linalg.norm(beta) * np.linalg.norm(baseline_beta))
            )
            angle_to_baseline = math.degrees(math.acos(min(max(cosine, -1.0), 1.0)))
        results[name] = {
            **solved,
            "beta_direction_angle_to_full_baseline_deg": angle_to_baseline,
        }

    return {
        "kind": "covariance_counterfactuals",
        "condition_cap": cap,
        "results": results,
        "diagnostic_only": True,
        "alignment_authorized": False,
        "counterfactuals_not_promoted_to_alignment_model": True,
        "note": "Sensitivity of the conclusion to covariance treatment is model-mismatch "
        "evidence, not authorization to pick a best W",
    }


def bootstrap_instability_decomposition(
    config: Mapping[str, Any],
    subspace: IdentifiableSubspace,
    transfer: Mapping[str, Any],
    bank: Mapping[str, Any],
) -> dict[str, Any]:
    """Decompose the WB78 bootstrap instability: rank distribution, eigenvalue
    distribution around the rank cut, mode persistence, the controlling pair
    type, and whether the instability is covariance-weight driven (via the
    diagonal-covariance diagnostic counterfactual).  Report-only.
    """
    arrays = wb78.concatenate_banks(bank)
    residual = np.asarray(arrays["residual"], dtype=np.float64)
    covariance = np.asarray(arrays["covariance"], dtype=np.float64)
    target_station = np.asarray(arrays["target_station_id"], dtype=np.int64)
    wb78_config = config["wb78_config"]
    rank_tolerance = float(wb78_config["solver"]["rank_tolerance"])
    n_boot = int(wb78_config["solver"]["bootstrap_replicates"])
    seed = int(wb78_config["solver"]["bootstrap_seed"])

    design = wb78._pair_design_rows(transfer, target_station, subspace)
    information, _rhs = wb78._information_and_rhs(design, covariance, residual)
    eigvals0, basis0, rank0 = wb78._informed_basis(information, rank_tolerance=rank_tolerance)
    rank_cut = rank_tolerance * eigvals0[-1]

    # Per-pair-type contribution to each information eigenvalue.
    pair_type_info: dict[str, np.ndarray] = {}
    for pair in STATION_PAIRS:
        rows = np.flatnonzero(target_station == pair[1])
        if rows.size == 0:
            continue
        info = np.zeros_like(information)
        for i in rows:
            w = np.linalg.inv(covariance[i])
            a = design[i]
            info += a.T @ w @ a
        pair_type_info[f"0->{pair[1]}"] = np.linalg.eigvalsh(info)

    rng = np.random.default_rng(seed)
    rank_distribution: dict[int, int] = {}
    fragile_eigval_samples: list[float] = []
    for _ in range(n_boot):
        rows = []
        for pair in STATION_PAIRS:
            available = np.flatnonzero(target_station == pair[1])
            if available.size:
                rows.append(rng.choice(available, size=available.size, replace=True))
        sample = np.concatenate(rows)
        b_design = wb78._pair_design_rows(transfer, target_station[sample], subspace)
        b_info, _b_rhs = wb78._information_and_rhs(
            b_design, covariance[sample], residual[sample]
        )
        b_eigvals, _b_basis, b_rank = wb78._informed_basis(b_info, rank_tolerance=rank_tolerance)
        rank_distribution[int(b_rank)] = rank_distribution.get(int(b_rank), 0) + 1
        # The eigenvalue nearest the rank cut (the fragile mode).
        fragile_eigval_samples.append(float(b_eigvals[-rank0]) if b_eigvals.size >= rank0 else 0.0)

    # Diagonal-covariance counterfactual bootstrap (diagnostic only).
    cov_diag = _counterfactual_covariance(covariance, "diagonal", 1.0e3)
    diag_solved = _solve_with_covariance(
        config,
        subspace,
        transfer,
        residual,
        cov_diag,
        target_station,
        bootstrap_replicates=n_boot,
        bootstrap_seed=seed,
    )

    return {
        "kind": "bootstrap_instability_decomposition",
        "full_rank": int(rank0),
        "rank_cut_eigenvalue": float(rank_cut),
        "information_eigenvalues": [float(v) for v in eigvals0],
        "bootstrap_rank_distribution": {str(k): int(v) for k, v in sorted(rank_distribution.items())},
        "fragile_mode_eigenvalue_min": float(np.min(fragile_eigval_samples)),
        "fragile_mode_eigenvalue_max": float(np.max(fragile_eigval_samples)),
        "pair_type_information_eigenvalues": {
            k: [float(v) for v in val] for k, val in pair_type_info.items()
        },
        "diagonal_covariance_counterfactual": diag_solved,
        "diagonal_counterfactual_is_diagnostic_only": True,
        "diagnostic_only": True,
        "alignment_authorized": False,
    }


# ---------------------------------------------------------------------------
# Stage 6: observable/Jacobian-transfer support audit (H3)
# ---------------------------------------------------------------------------


def _real_pair_kinematics(
    config: Mapping[str, Any], bank: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Extract (pred_tx, pred_ty) per calibration bank pair from the frozen
    propagation records (residual-blind kinematics only)."""
    wb78_config = config["wb78_config"]
    tx = []
    ty = []
    target_station = []
    for item in bank["banks"]:
        source = next(
            s
            for s in wb78_config["real_data_population"]["sources"]
            if str(s["source_id"]) == str(item["source_id"])
        )
        roots = wb78._source_roots(wb78_config, source)
        records = load_propagation_records(roots["identity"] / "field_candidates.root")
        mask = (records.q_over_p_mode == 0) & records.success
        index = {}
        for row in np.flatnonzero(mask):
            key = (
                int(records.run_id[row]),
                int(records.event_id[row]),
                int(records.source_tracklet_id[row]),
                int(records.target_tracklet_id[row]),
            )
            index[key] = row
        for k in range(item["residual"].shape[0]):
            key = (
                int(item["run_id_arr"][k]),
                int(item["event_id"][k]),
                int(item["source_tracklet_id"][k]),
                int(item["target_tracklet_id"][k]),
            )
            row = index.get(key)
            if row is None:
                raise ValueError(f"bank pair missing kinematic record: {key}")
            tx.append(float(records.prediction[row, 2]))
            ty.append(float(records.prediction[row, 3]))
            target_station.append(int(item["target_station_id"][k]))
    return {
        "pred_tx": np.asarray(tx, dtype=np.float64),
        "pred_ty": np.asarray(ty, dtype=np.float64),
        "target_station_id": np.asarray(target_station, dtype=np.int64),
    }


def _mc_support_cloud(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """MC Jacobian-bank kinematic support cloud (pred_tx, pred_ty) per station
    pair, from the frozen reference propagations of the WB68 corpus."""
    wb78_config = config["wb78_config"]
    corpus = wb78_config["tracker_information"]["jacobian_corpus"]
    manifest_path = resolve_under_root(project_root(), str(corpus["iteration_manifest"]))
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = set(str(s) for s in corpus["source_ids"])
    clouds: dict[str, dict[str, list[float]]] = {
        f"0->{p[1]}": {"tx": [], "ty": []} for p in STATION_PAIRS
    }
    for entry in manifest["sources"]:
        source_id = str(entry["source_id"])
        if source_id not in selected:
            continue
        scan_root = Path(str(entry["physical_scan_root"]))
        ref = scan_root / "points" / "iteration_00_reference" / "refit" / "propagations.root"
        if not ref.is_file():
            continue
        records = load_propagation_records(ref)
        mask = (records.q_over_p_mode == 0) & records.success & (records.source_station_id == 0)
        for pair in STATION_PAIRS:
            station = pair[1]
            sel = mask & (records.target_station_id == station)
            clouds[f"0->{station}"]["tx"].extend(
                records.prediction[sel, 2].astype(np.float64).tolist()
            )
            clouds[f"0->{station}"]["ty"].extend(
                records.prediction[sel, 3].astype(np.float64).tolist()
            )
    return {
        label: {
            "pred_tx": np.asarray(vals["tx"], dtype=np.float64),
            "pred_ty": np.asarray(vals["ty"], dtype=np.float64),
        }
        for label, vals in clouds.items()
    }


def transfer_support_audit(
    config: Mapping[str, Any],
    transfer: Mapping[str, Any],
    pooled: Mapping[str, Any],
    extras: Mapping[str, Any],
    bank: Mapping[str, Any],
    *,
    real_kinematics: Mapping[str, np.ndarray] | None = None,
    mc_cloud: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """H3: MC per-pair Jacobian dispersion (frozen) + real-vs-MC kinematic
    support overlap.  Residual-blind; no track selection by residual.

    ``real_kinematics`` / ``mc_cloud`` may be injected (used by tests); when
    None they are extracted from the frozen propagation records.
    """
    spec = config["transfer_support"]
    jacobian = np.asarray(extras["fit"].derivative_native, dtype=np.float64)
    source_station = np.asarray(pooled["source_station_id"], dtype=np.int64)
    target_station = np.asarray(pooled["target_station_id"], dtype=np.int64)

    # (a) MC per-pair J dispersion within each station pair.
    j_dispersion = {}
    j_ok = True
    for pair in STATION_PAIRS:
        label = f"{pair[0]}->{pair[1]}"
        mask = (source_station == pair[0]) & (target_station == pair[1])
        block = jacobian[mask]
        mean = transfer["mean_jacobian"][label]
        mean_norm = np.linalg.norm(mean)
        rel = np.linalg.norm(block - mean, axis=(1, 2)) / max(mean_norm, 1.0e-300)
        rms = float(np.sqrt(np.mean(rel**2)))
        j_dispersion[label] = {
            "n_mc_pairs": int(mask.sum()),
            "rms_relative_deviation": rms,
            "p95_relative_deviation": float(np.percentile(rel, 95.0)),
            "max_relative_deviation": float(np.max(rel)),
        }
        j_ok = j_ok and rms <= float(spec["mc_per_pair_jacobian_max_rms_relative"])

    # (b) Real-vs-MC kinematic support overlap in (pred_tx, pred_ty).
    real_kin = real_kinematics if real_kinematics is not None else _real_pair_kinematics(config, bank)
    if mc_cloud is None:
        mc_cloud = _mc_support_cloud(config)
    maha2_gate = float(spec["mahalanobis2_99_2dof"])
    frac_gate = float(spec["min_fraction_within_support"])
    support = {}
    support_ok = True
    for pair in STATION_PAIRS:
        label = f"{pair[0]}->{pair[1]}"
        cloud_tx = mc_cloud[label]["pred_tx"]
        cloud_ty = mc_cloud[label]["pred_ty"]
        cloud = np.column_stack([cloud_tx, cloud_ty])
        cloud = cloud[np.isfinite(cloud).all(axis=1)]
        rows = np.flatnonzero(real_kin["target_station_id"] == pair[1])
        real = np.column_stack([real_kin["pred_tx"][rows], real_kin["pred_ty"][rows]])
        real = real[np.isfinite(real).all(axis=1)]
        if cloud.shape[0] < 10 or real.shape[0] == 0:
            support[label] = {"n_real": int(real.shape[0]), "n_mc": int(cloud.shape[0]), "sufficient": False}
            continue
        center = cloud.mean(axis=0)
        cov = np.cov(cloud, rowvar=False)
        cov += np.eye(2) * (1.0e-6 * float(np.mean(np.diagonal(cov))) + 1.0e-12)
        inv = np.linalg.inv(cov)
        diff = real - center
        maha2 = np.einsum("ni,ij,nj->n", diff, inv, diff)
        within = maha2 <= maha2_gate
        fraction = float(np.mean(within))
        sufficient = label in ("0->1", "0->2")
        support[label] = {
            "n_real": int(real.shape[0]),
            "n_mc": int(cloud.shape[0]),
            "sufficient": True,
            "fraction_within_support": fraction,
            "gated": sufficient,
            "within_gate": (fraction >= frac_gate) if sufficient else None,
            "real_tx_p95_abs": float(np.percentile(np.abs(real[:, 0]), 95.0)),
            "mc_tx_p95_abs": float(np.percentile(np.abs(cloud[:, 0]), 95.0)),
            "real_ty_p95_abs": float(np.percentile(np.abs(real[:, 1]), 95.0)),
            "mc_ty_p95_abs": float(np.percentile(np.abs(cloud[:, 1]), 95.0)),
        }
        if sufficient:
            support_ok = support_ok and fraction >= frac_gate

    return {
        "kind": "jacobian_transfer_support_audit",
        "mc_jacobian_dispersion": j_dispersion,
        "mc_jacobian_dispersion_within_gate": bool(j_ok),
        "mc_jacobian_rms_gate": float(spec["mc_per_pair_jacobian_max_rms_relative"]),
        "kinematic_support": support,
        "kinematic_support_within_gate": bool(support_ok),
        "mahalanobis2_gate": maha2_gate,
        "min_fraction_gate": frac_gate,
        "kinematic_variables": list(spec["kinematic_variables"]),
        "pass": bool(j_ok and support_ok),
        "residual_blind": True,
        "diagnostic_only": True,
        "alignment_authorized": False,
    }


# ---------------------------------------------------------------------------
# Stage 7: calibration cross-run transportability (report-only solves)
# ---------------------------------------------------------------------------


def cross_run_transportability(
    config: Mapping[str, Any],
    subspace: IdentifiableSubspace,
    transfer: Mapping[str, Any],
    bank: Mapping[str, Any],
) -> dict[str, Any]:
    """Report-only one-shot diagnostic solves on 14973-only and 14974-only.
    These are NOT corrections and never open held-out data.  Compares rank,
    informed-subspace direction, and gamma direction between runs.
    """
    wb78_config = config["wb78_config"]
    rank_tolerance = float(wb78_config["solver"]["rank_tolerance"])
    spec = config["cross_run_transportability"]
    arrays = wb78.concatenate_banks(bank)
    residual = np.asarray(arrays["residual"], dtype=np.float64)
    covariance = np.asarray(arrays["covariance"], dtype=np.float64)
    target_station = np.asarray(arrays["target_station_id"], dtype=np.int64)
    run_id = np.asarray(arrays["run_id_arr"], dtype=np.int64)

    per_run = {}
    betas = {}
    bases = {}
    ranks = {}
    for run in sorted(set(int(v) for v in run_id)):
        rows = np.flatnonzero(run_id == run)
        design = wb78._pair_design_rows(transfer, target_station[rows], subspace)
        information, rhs = wb78._information_and_rhs(
            design, covariance[rows], residual[rows]
        )
        solved = wb78._solve_informed(information, rhs, rank_tolerance=rank_tolerance)
        ranks[run] = int(solved["rank"])
        betas[run] = np.asarray(solved["beta"], dtype=np.float64)
        bases[run] = np.asarray(solved["basis"], dtype=np.float64)
        per_run[str(run)] = {
            "n_pairs": int(rows.size),
            "informed_rank": int(solved["rank"]),
            "information_eigenvalues": [float(v) for v in solved["eigenvalues"]],
            "gamma": [float(v) for v in solved["gamma"]],
            "max_abs_gamma": float(np.max(np.abs(solved["gamma"]))) if solved["rank"] else 0.0,
        }

    runs = sorted(ranks)
    comparison = {}
    ok = True
    if len(runs) == 2:
        r1, r2 = runs[0], runs[1]
        rank_equal = ranks[r1] == ranks[r2]
        # Dominant informed eigenvector angle.
        dir_angle = None
        if ranks[r1] > 0 and ranks[r2] > 0:
            v1 = bases[r1][:, -1]
            v2 = bases[r2][:, -1]
            cosine = float(abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
            dir_angle = math.degrees(math.acos(min(max(cosine, -1.0), 1.0)))
        # Gamma direction angle.
        gamma_angle = None
        b1, b2 = betas[r1], betas[r2]
        if np.linalg.norm(b1) > 0 and np.linalg.norm(b2) > 0:
            cosine = float(abs(np.dot(b1, b2) / (np.linalg.norm(b1) * np.linalg.norm(b2))))
            gamma_angle = math.degrees(math.acos(min(max(cosine, -1.0), 1.0)))
        # Informed-subspace projector Frobenius distance (report-only).
        proj_dist = None
        if bases[r1].size and bases[r2].size and bases[r1].shape[1] == bases[r2].shape[1]:
            p1 = bases[r1] @ bases[r1].T
            p2 = bases[r2] @ bases[r2].T
            proj_dist = float(np.linalg.norm(p1 - p2, "fro"))
        comparison = {
            "rank_equal": bool(rank_equal),
            "dominant_direction_angle_deg": dir_angle,
            "gamma_direction_angle_deg": gamma_angle,
            "informed_subspace_projector_frobenius": proj_dist,
        }
        ok = rank_equal
        if spec.get("require_equal_rank"):
            ok = ok and rank_equal
        if dir_angle is not None:
            ok = ok and dir_angle <= float(spec["direction_max_angle_deg"])
        if gamma_angle is not None:
            ok = ok and gamma_angle <= float(spec["gamma_direction_max_angle_deg"])

    return {
        "kind": "calibration_cross_run_transportability",
        "per_run": per_run,
        "comparison": comparison,
        "direction_gate_deg": float(spec["direction_max_angle_deg"]),
        "gamma_direction_gate_deg": float(spec["gamma_direction_max_angle_deg"]),
        "pass": bool(ok),
        "report_only_not_a_correction": True,
        "diagnostic_only": True,
        "alignment_authorized": False,
    }


# ---------------------------------------------------------------------------
# Decision tree
# ---------------------------------------------------------------------------


def decide_campaign(
    *,
    reproduction: Mapping[str, Any],
    covariance_audit: Mapping[str, Any] | None,
    crosscheck: Mapping[str, Any] | None,
    transfer_support: Mapping[str, Any] | None,
    cross_run: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Pre-registered decision tree.  Exactly one terminal string is frozen."""
    if not bool(reproduction["pass"]):
        decision = DECISION_INCONCLUSIVE
        failed: list[str] = []
    else:
        failed = []
        cov_ok = bool(covariance_audit["whitened_chi2_within_gate"]) and bool(
            crosscheck["cross_run_reproducible"]
        )
        if not cov_ok:
            failed.append("covariance_adequacy")
        if not bool(transfer_support["pass"]):
            failed.append("transfer_support")
        if not bool(cross_run["pass"]):
            failed.append("cross_run_transportability")

        component_map = {
            "covariance_adequacy": DECISION_COVARIANCE,
            "transfer_support": DECISION_TRANSFER,
            "cross_run_transportability": DECISION_CROSS_RUN,
        }
        if len(failed) == 0:
            decision = DECISION_PASS
        elif len(failed) == 1:
            decision = component_map[failed[0]]
        else:
            decision = DECISION_MULTIPLE

    return {
        "kind": "model_adequacy_decision",
        "decision": decision,
        "failed_components": failed,
        "nonlinear_response_preregistration_allowed": decision == DECISION_PASS,
        "geometry_write_allowed": False,
        "official_conditions_write_allowed": False,
        "real_data_candidate_alignment_authorized": False,
        "external_constraint_ingest_authorized": False,
        "held_out_accessed": False,
        "eligible_external_physical_constraints": [],
        "if_fail_continue": "residual_dq_monitoring_only",
        "next_stage_if_pass": (
            "calibration_only_nonlinear_trust_region_reconstruction_response_feasibility_v1_"
            "requires_separate_preregistration"
        ),
        "official_cool_pool_write_remains_closed": True,
    }
