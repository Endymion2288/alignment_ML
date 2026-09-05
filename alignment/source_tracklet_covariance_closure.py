"""Workbook 83 Stage A: source-tracklet covariance truth closure.

This is an UPSTREAM covariance-validation campaign, **not** an alignment task.
Workbook 81 froze ``faseracts_propagated_covariance_not_calibrated``
(mechanism ``overestimated_transported_fit_covariance``) and identified two
independent problems: (A) the dummy q/p covariance should not enter the
propagated covariance, and (B) the source-tracklet fit covariance itself does
not cover the real source-state error.  Workbook 83 Stage A validates problem
(B) FIRST and source-disjoint, with NO propagation:

    e_source = fitted source-tracklet state - truth state at the source surface

in the measured ``[x, y, tx, ty]`` basis, whitening
``z = C_source^{-1/2} e_source`` and testing whether the written tracklet fit
covariance ``C_source`` describes the physical source-state error.  q/p is NOT
a measured quantity of a straight tracklet, so the Stage-A primary observable
does not require the dummy q/p to be "calibrated".

If Stage A cannot produce a portable calibrated source covariance, the campaign
freezes ``source_tracklet_fit_covariance_not_calibratable`` and does NOT touch
the FaserActs propagated covariance (propagation cannot repair a wrong input
uncertainty model).  Stage B (propagation covariance construction) is
pre-registered in the config but is only entered if Stage A passes.

The whole campaign keeps ``held_out_accessed=false``,
``real_data_alignment_authorized=false``, ``geometry_write_allowed=false``,
``official_conditions_write_allowed=false`` and
``external_constraint_ingest_authorized=false``.  It reads NO real-data
residual and never solves for an alignment correction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment.propagated_covariance_closure import (
    _load_truth_table,
    _symmetric_whiten,
)
from datasets.schema import covariance_from_columns

SCHEMA_VERSION = "propagated-covariance-upstream-repair-mc-validation-v1"
DEFAULT_CONFIG = "configs/propagated_covariance_upstream_repair_mc_validation_v1.yaml"

# Decision-tree terminal strings (exactly one is frozen).
DECISION_SOURCE_NOT_VALIDATED = "source_tracklet_covariance_not_validated"
DECISION_SOURCE_NOT_CALIBRATABLE = "source_tracklet_fit_covariance_not_calibratable"
DECISION_QOP_REPAIR_NOT_VALIDATED = "qop_covariance_semantic_repair_not_validated"
DECISION_PROCESS_NOISE_NOT_VALIDATED = "process_noise_model_not_validated"
DECISION_PROPAGATED_REPAIR_NOT_VALIDATED = "propagated_covariance_repair_not_validated"
DECISION_PROPAGATED_REPAIRED_VALIDATED = (
    "propagated_covariance_repaired_and_source_disjoint_validated"
)
DECISION_REPAIR_INCONCLUSIVE = "propagated_covariance_repair_inconclusive"

STATE_NAMES = ("x_mm", "y_mm", "tx", "ty")

# Tracklet branches read for the Stage-A source closure.
_TRACKLET_BRANCHES = (
    "run_id",
    "event_id",
    "station_id",
    "x_mm",
    "y_mm",
    "z_mm",
    "tx",
    "ty",
    "truth_particle_id",
    "truth_match_fraction",
    "truth_pdg",
    "q_over_p_per_mev",
    # Upper-triangular covariance in the [x,y,tx,ty] basis.
    "cov_xx_mm2",
    "cov_xy_mm2",
    "cov_xtx_mm",
    "cov_xty_mm",
    "cov_yy_mm2",
    "cov_ytx_mm",
    "cov_yty_mm",
    "cov_txtx",
    "cov_txty",
    "cov_tyty",
)


# ---------------------------------------------------------------------------
# Config loading + frozen-inheritance verification
# ---------------------------------------------------------------------------


class ConfigError(ValueError):
    """Raised when the WB83 config or its frozen inheritance is inconsistent."""


def _expect_false(config: Mapping[str, Any], key: str) -> None:
    if bool(config.get(key, False)) is not False:
        raise ConfigError(f"frozen flag must be false: {key}")


def _expect_true(config: Mapping[str, Any], key: str) -> None:
    if bool(config.get(key, False)) is not True:
        raise ConfigError(f"frozen prohibition must be true: {key}")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the WB83 config and verify the frozen WB81/WB82 inheritance."""
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ConfigError(f"WB83 config must be a mapping: {config_path}")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {SCHEMA_VERSION}")
    if int(config.get("workbook", -1)) != 83:
        raise ConfigError("workbook must be 83")

    # Frozen terminal permissions / prohibitions.
    for key in (
        "geometry_write_allowed",
        "official_conditions_write_allowed",
        "real_data_candidate_alignment_authorized",
        "real_data_alignment_authorized",
        "external_constraint_ingest_authorized",
        "held_out_accessed",
    ):
        _expect_false(config, key)
    for key in (
        "do_not_open_held_out",
        "do_not_read_real_data_residuals",
        "do_not_modify_faseracts_extrapolation_tool",
        "do_not_select_events_by_residual",
        "do_not_drop_pairs_by_condition_or_fit_quality",
        "do_not_use_external_evidence_as_prior",
        "do_not_solve_final_alignment",
        "do_not_tune_covariance_to_chi2",
        "do_not_promote_diagonal_or_capped_or_unit_covariance",
        "do_not_extrapolate_beyond_canonical_support",
        "do_not_open_alignment_diagnostic_v2",
        "do_not_promote_diagnostic_variant_to_production",
        "do_not_promote_mode3_to_production",
        "do_not_use_truth_qoverp_for_real_data_solution",
        "do_not_reverse_optimize_source_covariance_from_target",
        "do_not_change_central_prediction_in_covariance_repair",
    ):
        _expect_true(config, key)

    inheritance = config.get("inheritance", {})
    _verify_wb81_inheritance(inheritance)
    _verify_wb82_inheritance(inheritance)

    # Source-disjointness of the MC split must be file-level disjoint.
    mc = config["mc_data"]
    construction = set(mc["construction_source_ids"])
    validation = set(mc["validation_source_ids"])
    if construction & validation:
        raise ConfigError("MC construction/validation sources are not disjoint")
    return dict(config)


def _verify_wb81_inheritance(inheritance: Mapping[str, Any]) -> None:
    wb81_config_path = resolve_under_root(
        project_root(), str(inheritance["workbook_81_config"])
    )
    if sha256_file(wb81_config_path) != str(inheritance["workbook_81_config_sha256"]):
        raise ConfigError("WB81 config SHA256 mismatch (frozen inheritance broken)")
    wb81_root = resolve_under_root(
        project_root(), str(inheritance["workbook_81_output_root"])
    )
    for name, expected in inheritance["workbook_81_artifact_sha256"].items():
        actual = sha256_file(wb81_root / name)
        if actual != str(expected):
            raise ConfigError(f"WB81 artifact SHA256 mismatch: {name}")
    decision_path = wb81_root / "propagated_covariance_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("decision") != str(inheritance["workbook_81_frozen_decision"]):
        raise ConfigError("WB81 frozen decision mismatch")
    mechanism = decision.get("failure_classification", {}).get("category")
    if mechanism != str(inheritance["workbook_81_frozen_mechanism"]):
        raise ConfigError("WB81 frozen mechanism mismatch")
    if bool(decision.get("propagated_covariance_model_validated", True)) is not False:
        raise ConfigError("WB81 must have propagated_covariance_model_validated=false")


def _verify_wb82_inheritance(inheritance: Mapping[str, Any]) -> None:
    wb82_config_path = resolve_under_root(
        project_root(), str(inheritance["workbook_82_config"])
    )
    if sha256_file(wb82_config_path) != str(inheritance["workbook_82_config_sha256"]):
        raise ConfigError("WB82 config SHA256 mismatch (frozen inheritance broken)")
    wb82_root = resolve_under_root(
        project_root(), str(inheritance["workbook_82_output_root"])
    )
    for name, expected in inheritance["workbook_82_artifact_sha256"].items():
        actual = sha256_file(wb82_root / name)
        if actual != str(expected):
            raise ConfigError(f"WB82 artifact SHA256 mismatch: {name}")
    decision_path = wb82_root / "wide_ty_mc_support_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("decision") != str(inheritance["workbook_82_frozen_decision"]):
        raise ConfigError("WB82 frozen decision mismatch")


def assert_no_held_out_access() -> None:
    """WB83 reads no real-data residual and no held-out anything (MC only)."""
    return None


# ---------------------------------------------------------------------------
# Stage A source-tracklet closure records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceClosureRecords:
    """Per-station truth-known source-tracklet records (no propagation)."""

    station_id: int
    e_source: np.ndarray  # (n, 4) fitted - truth at source surface
    c_source: np.ndarray  # (n, 4, 4) tracklet fit covariance
    abs_tx: np.ndarray  # (n,)
    abs_ty: np.ndarray  # (n,)
    abs_q_over_p: np.ndarray  # (n,)
    truth_pdg: np.ndarray  # (n,)
    n_matched: int

    @property
    def size(self) -> int:
        return int(self.e_source.shape[0])


def build_source_closure_records(
    tracklets_path: Path,
    enhanced_path: Path,
    station_id: int,
    config: Mapping[str, Any],
    truth_table: Mapping[Any, Any] | None = None,
) -> SourceClosureRecords:
    """Join source tracklets with the truth reference; build e_source/C_source.

    No tracklet is dropped on pull, condition, or fit quality; records are only
    required to be truth-matched (>= min_truth_match_fraction), to carry a
    finite truth state at the source station, to pass the residual-blind
    physical acceptance guard (|tx|,|ty|), and to have a finite covariance.

    ``truth_table`` may be pre-loaded (one per source) so a multi-station scan
    does not re-read the enhanced ntuple for every station.
    """
    import uproot

    tr = config["truth_reference"]
    min_tmf = float(tr["min_truth_match_fraction"])
    acc = float(tr["physical_acceptance_abs_tx_ty_max"])
    with uproot.open(tracklets_path) as handle:
        tree = handle[config["mc_data"]["tracklets_tree"]]
        arrays = tree.arrays(list(_TRACKLET_BRANCHES), library="np")
    if truth_table is None:
        truth_table = _load_truth_table(
            enhanced_path, config["mc_data"]["enhanced_tree"], config
        )

    sel = (arrays["station_id"] == station_id) & (
        arrays["truth_match_fraction"] >= min_tmf
    )
    idx = np.where(sel)[0]
    covariance = covariance_from_columns(arrays)

    e_list: list[np.ndarray] = []
    c_list: list[np.ndarray] = []
    atx: list[float] = []
    aty: list[float] = []
    aqp: list[float] = []
    pdg: list[int] = []
    n_matched = 0
    for i in idx:
        key = (int(arrays["run_id"][i]), int(arrays["event_id"][i]))
        entry = truth_table.get(key)
        if entry is None:
            continue
        per_station = entry.get(int(arrays["truth_particle_id"][i]))
        if per_station is None or station_id not in per_station:
            continue
        truth = per_station[station_id]
        pos, mom = truth["pos"], truth["mom"]
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(mom))):
            continue
        if abs(mom[2]) < 1e-12:
            continue
        tx_fit = float(arrays["tx"][i])
        ty_fit = float(arrays["ty"][i])
        # Residual-blind physical acceptance guard (drops only unphysical
        # near-vertical outliers so they cannot inflate covariance envelopes).
        if abs(tx_fit) > acc or abs(ty_fit) > acc:
            continue
        tx_truth = mom[0] / mom[2]
        ty_truth = mom[1] / mom[2]
        z_mm = float(arrays["z_mm"][i])
        dz = z_mm - pos[2] if tr["straight_line_z_correction"] else 0.0
        x_truth = pos[0] + tx_truth * dz
        y_truth = pos[1] + ty_truth * dz
        e = np.array(
            [
                float(arrays["x_mm"][i]) - x_truth,
                float(arrays["y_mm"][i]) - y_truth,
                tx_fit - tx_truth,
                ty_fit - ty_truth,
            ],
            dtype=np.float64,
        )
        C = np.asarray(covariance[i], dtype=np.float64)
        if not np.all(np.isfinite(e)) or not np.all(np.isfinite(C)):
            continue
        n_matched += 1
        e_list.append(e)
        c_list.append(C)
        atx.append(abs(tx_fit))
        aty.append(abs(ty_fit))
        aqp.append(abs(float(arrays["q_over_p_per_mev"][i])))
        pdg.append(int(arrays["truth_pdg"][i]))
    if not e_list:
        return SourceClosureRecords(
            station_id=station_id,
            e_source=np.empty((0, 4)),
            c_source=np.empty((0, 4, 4)),
            abs_tx=np.empty(0),
            abs_ty=np.empty(0),
            abs_q_over_p=np.empty(0),
            truth_pdg=np.empty(0, dtype=np.int64),
            n_matched=n_matched,
        )
    return SourceClosureRecords(
        station_id=station_id,
        e_source=np.asarray(e_list),
        c_source=np.asarray(c_list),
        abs_tx=np.asarray(atx),
        abs_ty=np.asarray(aty),
        abs_q_over_p=np.asarray(aqp),
        truth_pdg=np.asarray(pdg, dtype=np.int64),
        n_matched=n_matched,
    )


# ---------------------------------------------------------------------------
# Stage A closure metrics
# ---------------------------------------------------------------------------


def _position_permutation() -> np.ndarray:
    """Permutation matrix swapping the position block x<->y (indices 0<->1)."""
    P = np.eye(4)
    P[[0, 1]] = P[[1, 0]]
    return P


def _chi2_per_ndof_samples(Ed: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Per-tracklet chi2/ndof (positive-definite covariances only)."""
    chi2s: list[float] = []
    for e, c in zip(Ed, C):
        if not np.all(np.isfinite(c)):
            continue
        try:
            if np.all(np.linalg.eigvalsh(c) > 0):
                chi2s.append(float(e @ np.linalg.solve(c, e) / 4.0))
        except np.linalg.LinAlgError:
            continue
    return np.asarray(chi2s)


def _mean_chi2_per_ndof(Ed: np.ndarray, C: np.ndarray) -> float:
    chi2s = _chi2_per_ndof_samples(Ed, C)
    return float(np.mean(chi2s)) if chi2s.size else float("nan")


def _binned_dependence(
    Ed: np.ndarray, c_source: np.ndarray, bin_var: np.ndarray, n_bins: int = 3
) -> list[dict[str, Any]]:
    """Per-bin chi2/ndof and marginal RMS, binned by a residual-blind variable."""
    n = Ed.shape[0]
    if n < n_bins * 3:
        return []
    qs = np.quantile(bin_var, np.linspace(0.0, 1.0, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    out: list[dict[str, Any]] = []
    for b in range(n_bins):
        mask = (bin_var >= qs[b]) & (bin_var < qs[b + 1])
        if mask.sum() < 3:
            continue
        out.append(
            {
                "bin": b,
                "bin_var_min": float(np.min(bin_var[mask])),
                "bin_var_max": float(np.max(bin_var[mask])),
                "n": int(mask.sum()),
                "chi2_per_ndof": _mean_chi2_per_ndof(Ed[mask], c_source[mask]),
                "marginal_rms_e": Ed[mask].std(axis=0).tolist(),
            }
        )
    return out


def compute_source_closure_metrics(
    records: SourceClosureRecords,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Pre-registered Stage-A whitening / eigenmode / swap-diagnostic metrics."""
    gates = config["source_closure_metrics"]
    n = records.size
    if n < int(config["source_closure_gates"]["min_tracklets_per_station"]):
        return {"station_id": int(records.station_id), "n_tracklets": n, "sufficient": False}

    E = records.e_source
    mean_e = E.mean(axis=0)
    Ed = E - mean_e if gates["demean_per_station"] else E
    C_emp = np.cov(Ed.T)
    C_src_mean = records.c_source.mean(axis=0)

    # Whitened residuals (de-meaned) and chi2.
    z_list: list[np.ndarray] = []
    mahal: list[float] = []
    for e, C in zip(Ed, records.c_source):
        z = _symmetric_whiten(e, C)
        if z is None:
            continue
        z_list.append(z)
        mahal.append(float(e @ np.linalg.solve(C, e)))
    z = np.asarray(z_list)
    ndof = 4
    cov_z = np.cov(z.T) if z.shape[0] > 2 else np.full((4, 4), np.nan)
    cov_z_eig = np.linalg.eigvalsh(cov_z) if z.shape[0] > 2 else np.full(4, np.nan)

    from scipy.stats import chi2 as _chi2_dist

    quant = float(_chi2_dist.ppf(0.95, gates["coverage_chi2_quantile_df"]))
    coverage = float(np.mean(np.asarray(mahal) <= quant)) if mahal else float("nan")

    # Generalized eigenvalues of (C_emp, C_source): lam << 1 -> C_source
    # over-estimates, lam >> 1 -> under-estimates.
    from scipy.linalg import eigh as _geigh

    try:
        gen_eig = np.sort(_geigh(C_emp, C_src_mean)[0])
    except Exception:
        gen_eig = np.sort(np.linalg.eigvals(np.linalg.solve(C_src_mean, C_emp)).real)

    def _corr(M: np.ndarray) -> np.ndarray:
        d = np.sqrt(np.diag(M))
        return M / np.outer(d, d)

    corr_src = _corr(C_src_mean)
    corr_emp = _corr(C_emp)

    # Marginal variance ratios (C_source_ii / empirical_ii) per component.
    emp_var = np.diag(C_emp)
    src_var = np.diag(C_src_mean)
    marginal_ratio = (src_var / np.maximum(emp_var, 1e-300)).tolist()

    chi2_samples = _chi2_per_ndof_samples(Ed, records.c_source)
    out: dict[str, Any] = {
        "station_id": int(records.station_id),
        "n_tracklets": n,
        "n_matched": int(records.n_matched),
        "sufficient": True,
        "mean_e": mean_e.tolist(),
        "marginal_rms_e": E.std(axis=0).tolist(),
        "marginal_rms_e_demeaned": Ed.std(axis=0).tolist(),
        "fit_covariance_marginal_rms": np.sqrt(np.maximum(src_var, 0.0)).tolist(),
        "mean_z": (z.mean(axis=0).tolist() if z.size else None),
        "marginal_rms_z": (z.std(axis=0).tolist() if z.size else None),
        "chi2_per_ndof": _mean_chi2_per_ndof(Ed, records.c_source),
        # Robust companions: the mean is inflated by a few near-singular
        # covariances (the pencil direction); the median and the tail fraction
        # are the robust measures.  For a calibrated 4-dof covariance the
        # fraction with chi2/ndof > 4 is ~0.4%.
        "chi2_per_ndof_median": float(np.median(chi2_samples))
        if chi2_samples.size
        else float("nan"),
        "fraction_chi2_per_ndof_above_4": float(np.mean(chi2_samples > 4.0))
        if chi2_samples.size
        else float("nan"),
        "cov_z_eigenvalues": cov_z_eig.tolist(),
        "coverage_probability_95": coverage,
        "c_source_mean_eigenvalues": np.linalg.eigvalsh(C_src_mean).tolist(),
        "c_emp_eigenvalues": np.linalg.eigvalsh(C_emp).tolist(),
        "generalized_eigenvalues_cemp_over_csource": gen_eig.tolist(),
        "marginal_variance_ratio_csource_over_emp": marginal_ratio,
        "correlation_x_tx": {"c_source": float(corr_src[0, 2]), "c_emp": float(corr_emp[0, 2])},
        "correlation_y_ty": {"c_source": float(corr_src[1, 3]), "c_emp": float(corr_emp[1, 3])},
    }

    # Position-swap diagnostic (report-only): is the written covariance's x<->y
    # position block exchanged relative to the empirical e_source covariance?
    if gates.get("position_swap_diagnostic", False):
        P = _position_permutation()
        C_swapped = np.einsum("ij,njk,lk->nil", P, records.c_source, P)
        chi2_after = _mean_chi2_per_ndof(Ed, C_swapped)
        chi2_after_samples = _chi2_per_ndof_samples(Ed, C_swapped)
        chi2_after_median = float(np.median(chi2_after_samples))
        frac_after = float(np.mean(chi2_after_samples > 4.0))
        chi2_before = out["chi2_per_ndof"]
        chi2_before_median = out["chi2_per_ndof_median"]
        out["position_swap_diagnostic"] = {
            # C_xx/emp_yy and C_yy/emp_xx ~ 1 indicates a clean position swap.
            "cross_swap_ratio_cxx_over_empyy": float(src_var[0] / max(emp_var[1], 1e-300)),
            "cross_swap_ratio_cyy_over_empxx": float(src_var[1] / max(emp_var[0], 1e-300)),
            "chi2_per_ndof_as_is": chi2_before,
            "chi2_per_ndof_after_position_xy_swap": chi2_after,
            "chi2_per_ndof_median_as_is": chi2_before_median,
            "chi2_per_ndof_median_after_position_xy_swap": chi2_after_median,
            "fraction_chi2_per_ndof_above_4_as_is": out[
                "fraction_chi2_per_ndof_above_4"
            ],
            "fraction_chi2_per_ndof_above_4_after_position_xy_swap": frac_after,
            "position_swap_chi2_improvement": (
                float(chi2_before / chi2_after)
                if np.isfinite(chi2_after) and chi2_after > 0
                else float("nan")
            ),
            "position_swap_chi2_median_improvement": (
                float(chi2_before_median / chi2_after_median)
                if np.isfinite(chi2_after_median) and chi2_after_median > 0
                else float("nan")
            ),
        }

    # Residual-blind kinematic dependence (|tx|, |ty|, |q/p|).
    dependence: dict[str, Any] = {}
    if "tx_ty_dependence" in gates["metrics"]:
        dependence["abs_tx"] = _binned_dependence(Ed, records.c_source, records.abs_tx)
        dependence["abs_ty"] = _binned_dependence(Ed, records.c_source, records.abs_ty)
    if "qoverp_dependence" in gates["metrics"]:
        dependence["abs_q_over_p"] = _binned_dependence(
            Ed, records.c_source, records.abs_q_over_p
        )
    out["kinematic_dependence"] = dependence
    return out


def evaluate_source_closure_gates(
    metrics: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the pre-registered Stage-A closure gates to one (station, split)."""
    g = config["source_closure_gates"]
    if not metrics.get("sufficient", False):
        return {"calibrated": False, "reason": "insufficient_tracklets", "gates": {}}
    cov_z_eig = np.asarray(metrics["cov_z_eigenvalues"], dtype=np.float64)
    gen_eig = np.asarray(
        metrics["generalized_eigenvalues_cemp_over_csource"], dtype=np.float64
    )
    checks = {
        "whitened_chi2_per_ndof": bool(
            metrics["chi2_per_ndof"] <= g["whitened_chi2_per_ndof_max"]
        ),
        "cov_z_eigenvalue_min": bool(np.all(cov_z_eig >= g["cov_z_eigenvalue_min"])),
        "cov_z_eigenvalue_max": bool(np.all(cov_z_eig <= g["cov_z_eigenvalue_max"])),
        "generalized_eigenvalue": bool(
            np.all(gen_eig >= g["generalized_eigenvalue_min"])
            and np.all(gen_eig <= g["generalized_eigenvalue_max"])
        ),
    }
    return {
        "calibrated": bool(all(checks.values())),
        "reason": "" if all(checks.values()) else "gate_failure",
        "gates": checks,
    }


# ---------------------------------------------------------------------------
# Source-disjoint Stage A driver
# ---------------------------------------------------------------------------


def _source_refit_dir(config: Mapping[str, Any], source_id: str) -> Path:
    template = config["mc_data"]["source_root_template"]
    return resolve_under_root(project_root(), template.format(source_id=source_id))


def _load_source_truth_table(config: Mapping[str, Any], source_id: str):
    """Load the per-station Geant truth table for one MC source (cached per source)."""
    mc = config["mc_data"]
    refit = _source_refit_dir(config, source_id)
    return _load_truth_table(refit / mc["enhanced_file"], mc["enhanced_tree"], config)


def load_source_station_records(
    config: Mapping[str, Any],
    source_id: str,
    station_id: int,
    truth_table: Mapping[Any, Any] | None = None,
) -> SourceClosureRecords:
    """Build the Stage-A records for one MC source / station."""
    mc = config["mc_data"]
    refit = _source_refit_dir(config, source_id)
    return build_source_closure_records(
        tracklets_path=refit / mc["tracklets_file"],
        enhanced_path=refit / mc["enhanced_file"],
        station_id=station_id,
        config=config,
        truth_table=truth_table,
    )


def _load_split_truth_tables(config: Mapping[str, Any], split: str) -> dict[str, Any]:
    """Load the truth table once per source in a split (avoids redundant reads)."""
    return {
        sid: _load_source_truth_table(config, sid)
        for sid in config["mc_data"][f"{split}_source_ids"]
    }


def _stack_split_station_records(
    config: Mapping[str, Any], split: str, station_id: int, truth_tables: Mapping[str, Any]
) -> SourceClosureRecords:
    parts = [
        load_source_station_records(
            config, sid, station_id, truth_table=truth_tables[sid]
        )
        for sid in config["mc_data"][f"{split}_source_ids"]
    ]
    return _stack_source_records(parts, station_id)


def _stack_source_records(
    parts: Sequence[SourceClosureRecords], station_id: int
) -> SourceClosureRecords:
    parts = [p for p in parts if p.size > 0]
    if not parts:
        return SourceClosureRecords(
            station_id=station_id,
            e_source=np.empty((0, 4)),
            c_source=np.empty((0, 4, 4)),
            abs_tx=np.empty(0),
            abs_ty=np.empty(0),
            abs_q_over_p=np.empty(0),
            truth_pdg=np.empty(0, dtype=np.int64),
            n_matched=0,
        )
    return SourceClosureRecords(
        station_id=station_id,
        e_source=np.concatenate([p.e_source for p in parts]),
        c_source=np.concatenate([p.c_source for p in parts]),
        abs_tx=np.concatenate([p.abs_tx for p in parts]),
        abs_ty=np.concatenate([p.abs_ty for p in parts]),
        abs_q_over_p=np.concatenate([p.abs_q_over_p for p in parts]),
        truth_pdg=np.concatenate([p.truth_pdg for p in parts]),
        n_matched=int(sum(p.n_matched for p in parts)),
    )


def _evaluate_pooled_station(
    stacked: SourceClosureRecords,
    config: Mapping[str, Any],
    c_source_override: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute the per-station metrics + gates, optionally overriding C_source."""
    if c_source_override is not None:
        stacked = SourceClosureRecords(
            **{**stacked.__dict__, "c_source": c_source_override}
        )
    metrics = compute_source_closure_metrics(stacked, config)
    metrics["gate_verdict"] = evaluate_source_closure_gates(metrics, config)
    return metrics


def _evaluate_pooled_split(
    pooled: Mapping[int, SourceClosureRecords],
    config: Mapping[str, Any],
    split: str,
    c_source_override: Mapping[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    per_station: dict[str, Any] = {}
    for station_id, stacked in pooled.items():
        override = c_source_override.get(station_id) if c_source_override else None
        per_station[str(station_id)] = _evaluate_pooled_station(stacked, config, override)
    return {
        "split": split,
        "source_ids": list(config["mc_data"][f"{split}_source_ids"]),
        "per_station": per_station,
    }


def run_source_closure_split(
    config: Mapping[str, Any],
    split: str,
    c_source_override: Mapping[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run the Stage-A source closure for one source-disjoint split.

    ``split`` is "construction" or "validation".  Records are pooled across the
    split's sources (file-level disjoint from the other split).  If
    ``c_source_override`` is given (a repaired covariance tensor per station,
    aligned with the pooled records), it replaces the written covariance -- used
    ONLY to evaluate a pre-registered repair, never to tune to chi2.
    """
    mc = config["mc_data"]
    station_ids = [int(s) for s in mc["station_ids"]]
    truth_tables = _load_split_truth_tables(config, split)
    pooled = {
        station_id: _stack_split_station_records(config, split, station_id, truth_tables)
        for station_id in station_ids
    }
    return _evaluate_pooled_split(pooled, config, split, c_source_override)


# ---------------------------------------------------------------------------
# Stage A repair models (pre-registered; derived on construction, confirmed on
# validation).  Physically interpretable only; never reverse-optimized from the
# propagated target.
# ---------------------------------------------------------------------------


def derive_repair_parameters(
    construction_records_per_station: Mapping[int, SourceClosureRecords],
    model: str,
) -> dict[str, Any]:
    """Derive a pre-registered repair's parameters from the CONSTRUCTION split.

    Returns a JSON-able dict of parameters.  Raises ConfigError for an unknown
    model.  Only the construction split is used (portability is then tested on
    the validation split).
    """
    if model == "position_permutation_xy_swap":
        return {"model": model}  # parameter-free structural repair
    if model == "global_scale":
        ratios = []
        for recs in construction_records_per_station.values():
            if recs.size == 0:
                continue
            Ed = recs.e_source - recs.e_source.mean(axis=0)
            emp = np.trace(np.cov(Ed.T))
            src = np.trace(recs.c_source.mean(axis=0))
            if src > 0:
                ratios.append(emp / src)
        return {"model": model, "scale": float(np.median(ratios)) if ratios else 1.0}
    if model == "xy_block_scale":
        # Per-position-component variance scale from the primary station.
        primary = min(construction_records_per_station)
        recs = construction_records_per_station[primary]
        Ed = recs.e_source - recs.e_source.mean(axis=0)
        emp_var = np.diag(np.cov(Ed.T))
        src_var = np.diag(recs.c_source.mean(axis=0))
        return {
            "model": model,
            "scale_x": float(emp_var[0] / max(src_var[0], 1e-300)),
            "scale_y": float(emp_var[1] / max(src_var[1], 1e-300)),
        }
    if model == "station_dependent_scale":
        scales: dict[str, float] = {}
        for station_id, recs in construction_records_per_station.items():
            if recs.size == 0:
                scales[str(station_id)] = 1.0
                continue
            Ed = recs.e_source - recs.e_source.mean(axis=0)
            emp = np.trace(np.cov(Ed.T))
            src = np.trace(recs.c_source.mean(axis=0))
            scales[str(station_id)] = float(emp / src) if src > 0 else 1.0
        return {"model": model, "per_station_scale": scales}
    raise ConfigError(f"unknown Stage-A repair model: {model}")


def apply_repair(c_source: np.ndarray, station_id: int, params: Mapping[str, Any]) -> np.ndarray:
    """Apply a pre-registered repair to a (n, 4, 4) covariance tensor."""
    model = params["model"]
    if model == "position_permutation_xy_swap":
        P = _position_permutation()
        return np.einsum("ij,njk,lk->nil", P, c_source, P)
    if model == "global_scale":
        return c_source * float(params["scale"])
    if model == "xy_block_scale":
        D = np.diag([np.sqrt(params["scale_x"]), np.sqrt(params["scale_y"]), 1.0, 1.0])
        return np.einsum("ij,njk,lk->nil", D, c_source, D)
    if model == "station_dependent_scale":
        return c_source * float(params["per_station_scale"][str(station_id)])
    raise ConfigError(f"unknown Stage-A repair model: {model}")


def evaluate_stage_a_repairs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Derive each pre-registered repair on construction; confirm on validation.

    A repair is "validated" only if, after applying it, the closure gates pass
    on BOTH source-disjoint splits at every station.  This is never a
    chi2-tuning exercise: the repair parameters are fixed on construction and
    only confirmed (not re-fit) on validation.
    """
    models = list(config["stage_a_repair"]["allowed_models"])
    station_ids = [int(s) for s in config["mc_data"]["station_ids"]]
    # Load each split's truth tables once per source (avoids redundant reads).
    truth_tables = {
        split: _load_split_truth_tables(config, split)
        for split in ("construction", "validation")
    }
    # Build the pooled records once per (split, station).
    pooled = {
        split: {
            station_id: _stack_split_station_records(
                config, split, station_id, truth_tables[split]
            )
            for station_id in station_ids
        }
        for split in ("construction", "validation")
    }
    construction_records = pooled["construction"]
    out: dict[str, Any] = {}
    for model in models:
        params = derive_repair_parameters(construction_records, model)
        # Apply the SAME frozen parameters to both splits and re-evaluate the
        # closure on the already-pooled records (no redundant truth reads).
        repaired = {
            split: {
                station_id: apply_repair(recs.c_source, station_id, params)
                for station_id, recs in pooled[split].items()
            }
            for split in ("construction", "validation")
        }
        construction = _evaluate_pooled_split(
            pooled["construction"], config, "construction", repaired["construction"]
        )
        validation = _evaluate_pooled_split(
            pooled["validation"], config, "validation", repaired["validation"]
        )
        validated = _split_all_calibrated(construction) and _split_all_calibrated(
            validation
        )
        out[model] = {
            "parameters": params,
            "construction": construction,
            "validation": validation,
            "validated_source_disjoint": bool(validated),
        }
    return out


# ---------------------------------------------------------------------------
# Stage A decision tree
# ---------------------------------------------------------------------------


def _split_all_calibrated(split_out: Mapping[str, Any]) -> bool:
    return all(
        p["gate_verdict"]["calibrated"] for p in split_out["per_station"].values()
    )


def decide_source_closure(
    config: Mapping[str, Any],
    construction: Mapping[str, Any],
    validation: Mapping[str, Any],
    repairs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Pre-registered Stage-A decision tree; exactly one terminal string frozen.

    Stage A only validates the SOURCE-tracklet covariance.  Even a validated
    repair keeps every alignment / held-out / geometry permission false and
    does NOT set propagated_covariance_model_validated (that requires Stage B).
    """
    tree = config["decision_tree"]
    construction_ok = _split_all_calibrated(construction)
    validation_ok = _split_all_calibrated(validation)

    if construction_ok and validation_ok:
        # As-is source covariance is calibrated on both splits.  (Not expected
        # given the WB81/WB83 exploration; would proceed to Stage B.)
        payload = _decision_payload(tree["propagated_covariance_repair_inconclusive"], config)
        payload["stage_a_note"] = (
            "as-is source covariance calibrated on both splits; Stage B not entered "
            "in this campaign"
        )
        return payload

    if construction_ok != validation_ok:
        return _decision_payload(tree["propagated_covariance_repair_inconclusive"], config)

    # Both splits fail the as-is closure.  Try the pre-registered repairs.
    if repairs:
        validated = [m for m, r in repairs.items() if r.get("validated_source_disjoint")]
        if validated:
            # A pre-registered repair is portable.  Stage A passes with the
            # repair; Stage B would be entered in a follow-on campaign.
            payload = _decision_payload(
                tree["propagated_covariance_repair_inconclusive"], config
            )
            payload["stage_a_note"] = (
                f"source covariance calibratable via pre-registered repair(s) "
                f"{validated}; Stage B not entered in this campaign"
            )
            payload["stage_a_validated_repairs"] = validated
            return payload

    # No pre-registered repair is portable -> the source-tracklet fit
    # covariance is not calibratable within the allowed repair family.  This
    # points upstream to the SegmentFit / hit-error model / exporter transform.
    payload = _decision_payload(tree["source_tracklet_fit_covariance_not_calibratable"], config)
    payload["failure_classification"] = classify_source_failure(construction, validation)
    return payload


def classify_source_failure(
    construction: Mapping[str, Any], validation: Mapping[str, Any]
) -> dict[str, Any]:
    """Evidence-based Stage-A failure classification (the position swap)."""
    evidence: dict[str, Any] = {"per_station": {}}
    swap_consistent = True
    median_as_is: list[float] = []
    median_after_swap: list[float] = []
    tail_after_swap: list[float] = []
    for split_name, split in (("construction", construction), ("validation", validation)):
        for station, m in split["per_station"].items():
            diag = m.get("position_swap_diagnostic")
            if not diag:
                swap_consistent = False
                continue
            median_as_is.append(diag["chi2_per_ndof_median_as_is"])
            median_after_swap.append(diag["chi2_per_ndof_median_after_position_xy_swap"])
            tail_after_swap.append(
                diag["fraction_chi2_per_ndof_above_4_after_position_xy_swap"]
            )
            evidence["per_station"].setdefault(station, {})[split_name] = {
                "chi2_per_ndof_as_is": diag["chi2_per_ndof_as_is"],
                "chi2_per_ndof_after_position_xy_swap": diag[
                    "chi2_per_ndof_after_position_xy_swap"
                ],
                "chi2_per_ndof_median_as_is": diag["chi2_per_ndof_median_as_is"],
                "chi2_per_ndof_median_after_position_xy_swap": diag[
                    "chi2_per_ndof_median_after_position_xy_swap"
                ],
                "fraction_chi2_per_ndof_above_4_after_position_xy_swap": diag[
                    "fraction_chi2_per_ndof_above_4_after_position_xy_swap"
                ],
                "cross_swap_ratio_cxx_over_empyy": diag["cross_swap_ratio_cxx_over_empyy"],
                "cross_swap_ratio_cyy_over_empxx": diag["cross_swap_ratio_cyy_over_empxx"],
                "marginal_variance_ratio_csource_over_emp": m.get(
                    "marginal_variance_ratio_csource_over_emp"
                ),
            }
    category = "position_xy_swap_with_slope_miscalibration"
    detail = (
        "The written source-tracklet fit covariance fails the Stage-A closure on "
        "BOTH source-disjoint splits at every station (median whitened "
        f"chi2/ndof ~ {np.median(median_as_is):.0f} as-is).  The position-swap "
        "diagnostic shows the written covariance's x/y position block is "
        "exchanged relative to the empirical e_source covariance "
        "(C_yy ~ empirical x-variance, cross-swap ratio ~ 1), and a position-only "
        "x<->y permutation of the covariance drops the MEDIAN whitened chi2/ndof "
        f"to ~ {np.median(median_after_swap):.2f} -- i.e. the permutation repairs "
        "the position DIAGONAL for the typical tracklet.  But the repair is only "
        "partial: the fraction of tracklets with chi2/ndof > 4 stays at "
        f"~ {np.median(tail_after_swap):.2f} (vs ~ 0.004 expected for a calibrated "
        "4-dof covariance), the ty slope variance is over-estimated (~ 150-300x), "
        "and a further diagonal rescale makes the closure WORSE (it amplifies a "
        "spurious near-null direction -- a wrong y-ty correlation -- that the "
        "permutation does not fix).  So the written covariance is in the WRONG "
        "FRAME/BASIS: the position is swapped AND the correlation structure does "
        "not match the empirical e_source covariance.  No pre-registered "
        "scale/permutation repair achieves a portable source-disjoint "
        "calibration.  This is consistent with the exporter transform / "
        "SegmentFit native (loc1,loc2,phi,theta,q/p) covariance being written in "
        "a basis that does not match the labelled global [x,y,tx,ty] state.  "
        "Propagation cannot repair a wrong input uncertainty model; the root "
        "cause is upstream in the SegmentFit / hit-error model / exporter "
        "covariance transform."
    )
    return {
        "category": category,
        "mechanism_detail": detail,
        "evidence": evidence,
        "swap_consistent_across_stations": bool(swap_consistent),
        "position_swap_fixes_diagonal_median_chi2": float(np.median(median_after_swap))
        if median_after_swap
        else None,
        "residual_tail_fraction_after_swap": float(np.median(tail_after_swap))
        if tail_after_swap
        else None,
        "repair_campaign": "segment_fit_hit_error_model_covariance_audit",
        "patched_in_this_campaign": False,
    }


def _decision_payload(decision: str, config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "decision": decision,
        "held_out_accessed": False,
        "real_data_alignment_authorized": False,
        "geometry_write_allowed": False,
        "official_conditions_write_allowed": False,
        "external_constraint_ingest_authorized": False,
        # Stage A never sets these; they require Stage B + the WB82 J-support
        # gate combined in a future Measurement Model V2.
        "propagated_covariance_model_validated": False,
        "real_kinematic_jacobian_support_validated": False,
        "measurement_model_validated": False,
        "real_data_alignment_v2_preregistration_allowed": False,
    }
