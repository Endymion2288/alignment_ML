"""Workbook-80 conditional Jacobian-transfer model (MC->real).

The Workbook-79 transfer-support audit showed the station-pair mean Jacobian is
a reasonable summary on MC (per-pair dispersion 0.244/0.078/0.111 for
(0,1)/(0,2)/(0,3)) but the REAL (0,1) calibration pairs extend beyond the MC
kinematic support (86.34% < 90% gate), especially in ``ty`` (real ~3x wider
than MC).  A conditional model ``J(pair_type, tx, ty, ...)`` can improve
INTERPOLATION inside the MC support but cannot manufacture support where the MC
bank has none; out-of-support real pairs are never extrapolated into a solve.

This module develops a conditional Jacobian model on a source-disjoint MC
split (construction vs validation), validates it on held-out MC sources, then
applies the frozen model to the real calibration kinematics in a strictly
residual-blind way to measure support coverage.  It never reads a real-data
residual to choose features or bins, never uses nearest-neighbour J, and never
produces an alignment solve.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from alignment import gauge_fixed_real_data_diagnostic as wb78
from alignment.tracker_only_identifiable_subspace import (
    _fit_bank,
    load_fd_only_bank,
)
from alignment.operating_protocol_v1_final_closure import project_root, resolve_under_root
from datasets.propagation_loader import load_propagation_records

OBSERVABLE_NAMES = wb78.OBSERVABLE_NAMES
STATION_PAIRS = wb78.STATION_PAIRS
N_OBS = 4
N_PARAM = 7


# ---------------------------------------------------------------------------
# Per-source MC Jacobian bank with residual-blind kinematics
# ---------------------------------------------------------------------------


def _manifest_entries_by_source(config: Mapping[str, Any]) -> dict[str, Path]:
    """Map source_id -> physical_scan_root from the frozen iteration manifest."""
    corpus = config["wb78_config"]["tracker_information"]["jacobian_corpus"]
    manifest_path = resolve_under_root(project_root(), str(corpus["iteration_manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        str(entry["source_id"]): Path(str(entry["physical_scan_root"])).expanduser().resolve()
        for entry in manifest["sources"]
    }


def _reference_propagations_path(scan_root: Path, anchor_point: str) -> Path:
    return scan_root / "points" / anchor_point / "refit" / "propagations.root"


def load_mc_source_jacobian_bank(
    config: Mapping[str, Any], source_id: str
) -> dict[str, Any]:
    """Load one MC source's per-pair native Jacobian + residual-blind kinematics.

    Uses the frozen FD bank chain (``load_fd_only_bank`` + ``_fit_bank``) so the
    per-pair Jacobian is identical to the one pooled into the WB78 transfer
    model.  Kinematics (pred_tx, pred_ty) come from the source's reference
    propagations, joined by pair identity.  No real-data residual is touched.
    """
    corpus = config["wb78_config"]["tracker_information"]["jacobian_corpus"]
    scan_roots = _manifest_entries_by_source(config)
    if source_id not in scan_roots:
        raise ValueError(f"MC source {source_id} not in iteration manifest")
    anchor_point = str(corpus["anchor_point"])
    # Rebuild the manifest entry dict expected by load_fd_only_bank.
    entry = {"source_id": source_id, "physical_scan_root": str(scan_roots[source_id])}
    bank = load_fd_only_bank(
        entry,
        anchor_point=anchor_point,
        min_truth_match_fraction=float(corpus["min_truth_match_fraction"]),
        only_parameters=tuple(str(p) for p in corpus["parameter_names"]),
    )
    fit = _fit_bank(bank, rcond=float(config["wb78_config"]["tracker_information"]["rcond"]))
    jacobian = np.asarray(fit.derivative_native, dtype=np.float64)  # (n, 4, 7)

    # Kinematics from the reference propagations, joined by pair identity.
    ref = _reference_propagations_path(scan_roots[source_id], anchor_point)
    records = load_propagation_records(ref)
    kin: dict[tuple[int, int, int, int], tuple[float, float]] = {}
    mask = (records.q_over_p_mode == 0) & records.success
    for row in np.flatnonzero(mask):
        key = (
            int(records.run_id[row]),
            int(records.event_id[row]),
            int(records.source_tracklet_id[row]),
            int(records.target_tracklet_id[row]),
        )
        kin[key] = (float(records.prediction[row, 2]), float(records.prediction[row, 3]))
    n = jacobian.shape[0]
    tx = np.full(n, np.nan)
    ty = np.full(n, np.nan)
    for i in range(n):
        key = (
            int(bank["run_id"][i]),
            int(bank["event_id"][i]),
            int(bank["source_tracklet_id"][i]),
            int(bank["target_tracklet_id"][i]),
        )
        if key in kin:
            tx[i], ty[i] = kin[key]
    return {
        "source_id": source_id,
        "jacobian": jacobian,
        "pred_tx": tx,
        "pred_ty": ty,
        "source_station_id": np.asarray(bank["source_station_id"], dtype=np.int64),
        "target_station_id": np.asarray(bank["target_station_id"], dtype=np.int64),
        "anchor_residual": np.asarray(bank["anchor_residual"], dtype=np.float64),
        "covariance": np.asarray(bank["covariance"], dtype=np.float64),
    }


def _stack_sources(banks: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    out: dict[str, list[np.ndarray]] = {}
    for bank in banks:
        for key, value in bank.items():
            if isinstance(value, np.ndarray):
                out.setdefault(key, []).append(value)
    return {key: np.concatenate(values) for key, values in out.items()}


# ---------------------------------------------------------------------------
# Conditional Jacobian models (fit on construction sources only)
# ---------------------------------------------------------------------------


def fit_conditional_jacobian_model(
    kind: str,
    construction: Mapping[str, np.ndarray],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit a conditional Jacobian model on the CONSTRUCTION MC sources only."""
    J = construction["jacobian"]
    tx = construction["pred_tx"]
    ty = construction["pred_ty"]
    ss = construction["source_station_id"]
    tsd = construction["target_station_id"]
    model: dict[str, Any] = {"kind": kind, "per_pair": {}}
    for pair in STATION_PAIRS:
        label = f"{pair[0]}->{pair[1]}"
        mask = (ss == pair[0]) & (tsd == pair[1]) & np.isfinite(tx) & np.isfinite(ty)
        block = J[mask]
        btx = tx[mask]
        bty = ty[mask]
        if block.shape[0] == 0:
            raise ValueError(f"no construction MC pairs for station pair {label}")
        entry: dict[str, Any] = {"n_construction": int(block.shape[0])}
        if kind == "station_pair_mean":
            entry["mean"] = block.mean(axis=0)
        elif kind == "kinematic_binned_mean":
            tx_edges = np.asarray(spec["tx_bin_edges"], dtype=np.float64)
            ty_edges = np.asarray(spec["ty_bin_edges"], dtype=np.float64)
            min_per_bin = int(spec["min_pairs_per_bin"])
            global_mean = block.mean(axis=0)
            bins: dict[str, Any] = {}
            for a in range(tx_edges.size - 1):
                for b in range(ty_edges.size - 1):
                    sel = (
                        (btx >= tx_edges[a])
                        & (btx < tx_edges[a + 1])
                        & (bty >= ty_edges[b])
                        & (bty < ty_edges[b + 1])
                    )
                    if sel.sum() >= min_per_bin:
                        bins[f"{a},{b}"] = block[sel].mean(axis=0)
            entry["mean"] = global_mean
            entry["bins"] = bins
            entry["tx_edges"] = tx_edges
            entry["ty_edges"] = ty_edges
        elif kind == "linear_kinematic_regression":
            # Per element J[o,p] = a + b_tx*tx + b_ty*ty (least squares).
            design = np.column_stack([np.ones_like(btx), btx, bty])
            coef, *_ = np.linalg.lstsq(design, block.reshape(block.shape[0], -1), rcond=None)
            entry["coef"] = coef.reshape(3, N_OBS, N_PARAM)  # [const, tx, ty]
            entry["mean"] = block.mean(axis=0)
        else:
            raise ValueError(f"unknown conditional jacobian kind {kind}")
        model["per_pair"][label] = entry
    return model


def predict_jacobian(
    model: Mapping[str, Any], label: str, tx: np.ndarray, ty: np.ndarray
) -> np.ndarray:
    """Predict per-pair Jacobian (n, 4, 7) for kinematics (tx, ty)."""
    kind = model["kind"]
    entry = model["per_pair"][label]
    tx = np.atleast_1d(np.asarray(tx, dtype=np.float64))
    ty = np.atleast_1d(np.asarray(ty, dtype=np.float64))
    n = tx.shape[0]
    if kind == "station_pair_mean":
        return np.repeat(entry["mean"][None, :, :], n, axis=0)
    if kind == "kinematic_binned_mean":
        out = np.repeat(entry["mean"][None, :, :], n, axis=0)
        tx_edges = entry["tx_edges"]
        ty_edges = entry["ty_edges"]
        for i in range(n):
            a = int(np.searchsorted(tx_edges, tx[i], side="right") - 1)
            b = int(np.searchsorted(ty_edges, ty[i], side="right") - 1)
            key = f"{a},{b}"
            if key in entry["bins"]:
                out[i] = entry["bins"][key]
        return out
    if kind == "linear_kinematic_regression":
        coef = entry["coef"]  # (3, 4, 7)
        out = np.empty((n, N_OBS, N_PARAM))
        for i in range(n):
            out[i] = coef[0] + coef[1] * tx[i] + coef[2] * ty[i]
        return out
    raise ValueError(f"unknown conditional jacobian kind {kind}")


# ---------------------------------------------------------------------------
# MC source-disjoint validation
# ---------------------------------------------------------------------------


def _relative_frobenius(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    num = np.linalg.norm(pred - actual, axis=(1, 2))
    den = np.maximum(np.linalg.norm(actual, axis=(1, 2)), 1.0e-300)
    return num / den


def mc_jacobian_model_validation(
    config: Mapping[str, Any],
    *,
    construction_banks: Sequence[Mapping[str, Any]] | None = None,
    validation_banks: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Source-disjoint MC validation of each candidate conditional-J model.

    Fits on the construction sources, evaluates the per-pair Jacobian
    prediction error, identifiable-direction injection recovery and null
    leakage on the held-out validation sources.  No real data is used.
    """
    spec = config["jacobian_transfer"]
    val_spec = spec["mc_validation"]
    subspace = config["subspace"]
    s_mat = np.diag(np.asarray(subspace.parameter_scales, dtype=np.float64))
    v_id = np.asarray(subspace.v_id, dtype=np.float64)
    v_null = np.asarray(subspace.v_null, dtype=np.float64)

    if construction_banks is None:
        construction_banks = [
            load_mc_source_jacobian_bank(config, s) for s in spec["mc_construction_source_ids"]
        ]
    if validation_banks is None:
        validation_banks = [
            load_mc_source_jacobian_bank(config, s) for s in spec["mc_validation_source_ids"]
        ]
    construction = _stack_sources(construction_banks)
    validation = _stack_sources(validation_banks)

    rng = np.random.default_rng(int(spec.get("validation_seed", 20260905)))
    n_inject = int(spec.get("injection_replicates", 64))
    amp = float(spec.get("injection_amplitude_scaled", 0.10))

    results: dict[str, Any] = {}
    for model_spec in spec["models"]:
        name = str(model_spec["name"])
        kind = str(model_spec["kind"])
        model = fit_conditional_jacobian_model(kind, construction, model_spec)
        per_pair = {}
        all_rel: list[np.ndarray] = []
        inject_rel: list[float] = []
        null_leak: list[float] = []
        for pair in STATION_PAIRS:
            label = f"{pair[0]}->{pair[1]}"
            mask = (
                (validation["source_station_id"] == pair[0])
                & (validation["target_station_id"] == pair[1])
                & np.isfinite(validation["pred_tx"])
                & np.isfinite(validation["pred_ty"])
            )
            if mask.sum() < int(val_spec["min_validation_pairs_per_pairtype"]):
                per_pair[label] = {"n_validation": int(mask.sum()), "sufficient": False}
                continue
            actual = validation["jacobian"][mask]
            pred = predict_jacobian(
                model, label, validation["pred_tx"][mask], validation["pred_ty"][mask]
            )
            rel = _relative_frobenius(pred, actual)
            all_rel.append(rel)
            # Identifiable-direction injection recovery + null leakage.
            for _ in range(n_inject):
                u = rng.normal(size=v_id.shape[1])
                u /= max(np.linalg.norm(u), 1.0e-300)
                dtheta_id = s_mat @ v_id @ (u * amp)
                actual_resp = actual @ dtheta_id
                pred_resp = pred @ dtheta_id
                denom = np.maximum(np.linalg.norm(actual_resp, axis=1), 1.0e-300)
                inject_rel.append(
                    float(np.median(np.linalg.norm(pred_resp - actual_resp, axis=1) / denom))
                )
                if v_null.shape[1] > 0:
                    un = rng.normal(size=v_null.shape[1])
                    un /= max(np.linalg.norm(un), 1.0e-300)
                    dtheta_null = s_mat @ v_null @ (un * amp)
                    pred_null = pred @ dtheta_null
                    actual_id_norm = np.maximum(
                        np.linalg.norm(actual @ dtheta_id, axis=1), 1.0e-300
                    )
                    null_leak.append(
                        float(np.median(np.linalg.norm(pred_null, axis=1) / actual_id_norm))
                    )
            per_pair[label] = {
                "n_validation": int(mask.sum()),
                "sufficient": True,
                "frobenius_relative_rms": float(np.sqrt(np.mean(rel ** 2))),
                "frobenius_relative_median": float(np.median(rel)),
                "frobenius_relative_p95": float(np.percentile(rel, 95.0)),
            }
        rel_all = np.concatenate(all_rel) if all_rel else np.array([np.nan])
        frob_rms = float(np.sqrt(np.mean(rel_all ** 2)))
        inject = float(np.median(inject_rel)) if inject_rel else math.inf
        leak = float(np.median(null_leak)) if null_leak else 0.0
        gates = {
            "frobenius_within_gate": bool(
                frob_rms <= float(val_spec["max_frobenius_relative_error"])
            ),
            "injection_recovery_within_gate": bool(
                inject <= float(val_spec["injection_recovery_max_relative"])
            ),
            "null_leakage_within_gate": bool(
                leak <= float(val_spec["null_leakage_max_relative"])
            ),
        }
        results[name] = {
            "kind": kind,
            "per_pair_type": per_pair,
            "frobenius_relative_rms": frob_rms,
            "injection_recovery_median_relative": inject,
            "null_leakage_median_relative": leak,
            "gates": gates,
            "mc_validated": bool(all(gates.values())),
        }

    validated = [name for name, rep in results.items() if rep["mc_validated"]]
    return {
        "kind": "mc_jacobian_model_source_disjoint_validation",
        "construction_source_ids": list(spec["mc_construction_source_ids"]),
        "validation_source_ids": list(spec["mc_validation_source_ids"]),
        "models": results,
        "mc_validated_models": validated,
        "source_disjoint": True,
        "residual_blind_features": list(spec["features"]),
        "diagnostic_only": True,
        "alignment_authorized": False,
    }


# ---------------------------------------------------------------------------
# Real-data residual-blind support validation (frozen model applied to real)
# ---------------------------------------------------------------------------


def real_jacobian_support_validation(
    config: Mapping[str, Any],
    model: Mapping[str, Any],
    real_kinematics: Mapping[str, np.ndarray],
    mc_construction_kinematics: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Apply the frozen conditional-J model's applicability domain to the real
    calibration kinematics (residual-blind) and measure support coverage.

    A real pair is in-domain if its (tx, ty) lies within the MC construction
    99% Mahalanobis envelope for that station pair.  Out-of-support pairs are
    NEVER extrapolated into a solve; substantial out-of-support freezes
    ``jacobian_transfer_model_not_validated``.
    """
    spec = config["jacobian_transfer"]["real_support"]
    maha2_gate = float(spec["mahalanobis2_99_2dof"])
    frac_gate = float(spec["min_fraction_within_support"])
    support = {}
    support_ok = True
    for pair in STATION_PAIRS:
        label = f"{pair[0]}->{pair[1]}"
        cloud = np.column_stack(
            [mc_construction_kinematics[label]["pred_tx"], mc_construction_kinematics[label]["pred_ty"]]
        )
        cloud = cloud[np.isfinite(cloud).all(axis=1)]
        rows = np.flatnonzero(real_kinematics["target_station_id"] == pair[1])
        real = np.column_stack([real_kinematics["pred_tx"][rows], real_kinematics["pred_ty"][rows]])
        real = real[np.isfinite(real).all(axis=1)]
        if cloud.shape[0] < 10 or real.shape[0] == 0:
            support[label] = {
                "n_real": int(real.shape[0]),
                "n_mc": int(cloud.shape[0]),
                "sufficient": False,
            }
            continue
        center = cloud.mean(axis=0)
        cov = np.cov(cloud, rowvar=False)
        cov += np.eye(2) * (1.0e-6 * float(np.mean(np.diagonal(cov))) + 1.0e-12)
        inv = np.linalg.inv(cov)
        diff = real - center
        maha2 = np.einsum("ni,ij,nj->n", diff, inv, diff)
        fraction = float(np.mean(maha2 <= maha2_gate))
        gated = label in ("0->1", "0->2")
        within = bool(fraction >= frac_gate) if gated else None
        support[label] = {
            "n_real": int(real.shape[0]),
            "n_mc": int(cloud.shape[0]),
            "sufficient": True,
            "fraction_within_support": fraction,
            "gated": gated,
            "within_gate": within,
            "real_tx_p95_abs": float(np.percentile(np.abs(real[:, 0]), 95.0)),
            "mc_tx_p95_abs": float(np.percentile(np.abs(cloud[:, 0]), 95.0)),
            "real_ty_p95_abs": float(np.percentile(np.abs(real[:, 1]), 95.0)),
            "mc_ty_p95_abs": float(np.percentile(np.abs(cloud[:, 1]), 95.0)),
        }
        if gated:
            support_ok = support_ok and bool(within)
    return {
        "kind": "real_jacobian_transfer_support_validation",
        "model_kind": model["kind"],
        "kinematic_support": support,
        "kinematic_support_within_gate": bool(support_ok),
        "min_fraction_gate": frac_gate,
        "mahalanobis2_gate": maha2_gate,
        "residual_blind": True,
        "out_of_support_never_extrapolated_into_solve": True,
        "diagnostic_only": True,
        "alignment_authorized": False,
    }
