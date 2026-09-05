"""Workbook-79 real-data residual/covariance model adequacy tests.

Covers the pre-registered config freeze (flags, WB78 inheritance SHAs, output
naming), the structural held-out closure, the exact reproduction of the WB78
calibration baseline, the covariance eigenstructure/whitening adequacy gate,
the 14973<->14974 empirical covariance cross-check, the deterministic
alignment-score contribution decomposition, the bootstrap-instability
decomposition, the diagnostic-only covariance counterfactuals (which can never
enter the production solver), the Jacobian-transfer support audit, the
calibration cross-run transportability audit, and every decision-tree branch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from alignment.identifiable_subspace import FROZEN_RANK_TOLERANCE, IdentifiableSubspace, frozen_scales_for
from alignment.real_data_covariance_model_adequacy import (
    DECISION_COVARIANCE,
    DECISION_CROSS_RUN,
    DECISION_INCONCLUSIVE,
    DECISION_MULTIPLE,
    DECISION_PASS,
    DECISION_TRANSFER,
    SCHEMA_VERSION,
    bootstrap_instability_decomposition,
    covariance_counterfactuals,
    covariance_eigen_audit,
    cross_run_transportability,
    decide_campaign,
    empirical_covariance_crosscheck,
    load_config,
    reproduce_baseline,
    score_contribution_decomposition,
    transfer_support_audit,
    _counterfactual_covariance,
)
from alignment.gauge_fixed_real_data_diagnostic import PARAMETER_NAMES

CONFIG_PATH = "configs/real_data_residual_covariance_model_adequacy_v1.yaml"


def _config():
    return load_config(CONFIG_PATH)


def _tampered_config(tmp_path: Path, mutate) -> str:
    raw = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "tampered.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(path)


def _synthetic_subspace() -> IdentifiableSubspace:
    names = PARAMETER_NAMES
    scales = frozen_scales_for(names)
    mixed = np.zeros(7)
    mixed[[0, 6]] = 1.0 / np.sqrt(2.0)
    v_null = np.column_stack([np.eye(7)[:, 2], mixed])
    id_columns = [np.eye(7)[:, i] for i in (1, 3, 4, 5)]
    anti = np.zeros(7)
    anti[0] = 1.0 / np.sqrt(2.0)
    anti[6] = -1.0 / np.sqrt(2.0)
    v_id = np.column_stack(id_columns + [anti])
    v = np.column_stack([v_id, v_null])
    return IdentifiableSubspace(
        parameter_names=tuple(names),
        parameter_units=("mm", "mm", "mm", "mrad", "mrad", "mrad", "mm"),
        parameter_scales=scales,
        rank_tolerance=float(FROZEN_RANK_TOLERANCE),
        singular_values=np.asarray([100.0, 80.0, 60.0, 40.0, 20.0, 0.0, 0.0]),
        identifiable_rank=5,
        u=np.zeros((5, 5)),
        v=v,
        v_id=v_id,
        v_null=v_null,
        projector_id=v_id @ v_id.T,
        mode_signs=tuple([1] * 7),
        n_observations=20,
        n_parameters=7,
    )


def _synthetic_transfer(rng: np.random.Generator) -> dict:
    means = {}
    for pair in ((0, 1), (0, 2), (0, 3)):
        means[f"{pair[0]}->{pair[1]}"] = rng.normal(scale=0.5, size=(4, 7))
    return {"kind": "station_pair_mean_native_jacobian", "mean_jacobian": means}


def _spd_from_eigen(rng: np.random.Generator, eigvals: tuple[float, ...]) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    return (q * np.asarray(eigvals)) @ q.T


def _synthetic_bank(
    rng: np.random.Generator,
    *,
    n_pairs_per_run: int = 60,
    covariance_kind: str = "well_conditioned",
    noise_scale: float = 1.0,
) -> dict:
    """Two-run synthetic calibration bank with a controllable covariance model.

    covariance_kind="well_conditioned": residuals drawn from the same
    well-conditioned covariance used as the weight (model adequate).
    covariance_kind="near_singular_mismatch": weight covariance is near
    singular but residuals are drawn isotropically (model mismatch -> the
    near-singular direction is violated).
    """
    residuals = []
    covariances = []
    targets = []
    runs = []
    events = []
    for run_index, run in enumerate((14973, 14974)):
        for i in range(n_pairs_per_run):
            pair = ((0, 1), (0, 2), (0, 3))[i % 3]
            if covariance_kind == "well_conditioned":
                # Anisotropic but well-conditioned (condition ~ 8) so that the
                # covariance eigenstructure is well-defined and a rotation of
                # one run's residuals is detectable.
                cov = _spd_from_eigen(rng, (0.5, 1.0, 2.0, 4.0))
                true_cov = cov
            else:
                cov = _spd_from_eigen(rng, (1.0e-4, 1.0, 1.0, 1.0))  # near-singular weight
                true_cov = np.eye(4)  # isotropic truth -> mismatch
            r = rng.multivariate_normal(np.zeros(4), true_cov) * noise_scale
            residuals.append(r)
            covariances.append(cov)
            targets.append(pair[1])
            runs.append(run)
            events.append(run_index * n_pairs_per_run + i)
    n = len(residuals)
    return {
        "kind": "real_data_anchor_pair_bank",
        "roles": ["calibration"],
        "banks": [
            {
                "source_id": "synthetic",
                "run_id": 0,
                "role": "calibration",
                "n_routes": n,
                "n_anchor_pair_keys": n,
                "n_pairs_missing_propagation": 0,
                "residual": np.asarray(residuals),
                "covariance": np.asarray(covariances),
                "run_id_arr": np.asarray(runs, dtype=np.int64),
                "event_id": np.asarray(events, dtype=np.int64),
                "route_index": np.asarray(events, dtype=np.int64),
                "source_tracklet_id": np.zeros(n, dtype=np.int64),
                "target_tracklet_id": np.arange(n, dtype=np.int64),
                "target_station_id": np.asarray(targets, dtype=np.int64),
            }
        ],
        "csv_reproduction": {"rows_checked": 0, "max_abs_diff": 0.0, "bit_exact": True},
    }


# ---------------------------------------------------------------------------
# Config freeze
# ---------------------------------------------------------------------------


def test_config_loads_and_verifies_inheritance():
    config = _config()
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["geometry_write_allowed"] is False
    assert config["official_conditions_write_allowed"] is False
    assert config["real_data_candidate_alignment_authorized"] is False
    assert config["external_constraint_ingest_authorized"] is False
    assert config["held_out_accessed"] is False
    assert config["eligible_external_physical_constraints"] == []
    assert config["output_parameter_naming"] == "out_of_support_diagnostic_candidate"
    assert config["population"]["calibration_runs"] == [14973, 14974]
    assert config["population"]["held_out_closed"] is True
    # WB78 config is loaded and reused verbatim.
    assert config["wb78_config"]["gauges"]["primary"] == "minimum_norm_scaled_gauge"


def test_config_rejects_geometry_write(tmp_path):
    path = _tampered_config(tmp_path, lambda c: c.__setitem__("geometry_write_allowed", True))
    with pytest.raises(ValueError, match="geometry_write_allowed"):
        load_config(path)


def test_config_rejects_held_out_access(tmp_path):
    path = _tampered_config(tmp_path, lambda c: c.__setitem__("held_out_accessed", True))
    with pytest.raises(ValueError, match="held_out_accessed"):
        load_config(path)


def test_config_rejects_counterfactual_promotion(tmp_path):
    path = _tampered_config(
        tmp_path,
        lambda c: c.__setitem__("do_not_promote_counterfactual_covariance_to_alignment", False),
    )
    with pytest.raises(ValueError, match="do_not_promote_counterfactual_covariance"):
        load_config(path)


def test_config_rejects_empirical_weight_promotion(tmp_path):
    path = _tampered_config(
        tmp_path, lambda c: c.__setitem__("do_not_promote_empirical_covariance_to_weight", False)
    )
    with pytest.raises(ValueError, match="do_not_promote_empirical_covariance"):
        load_config(path)


def test_config_rejects_candidate_scaling(tmp_path):
    path = _tampered_config(
        tmp_path, lambda c: c.__setitem__("do_not_scale_or_clip_wb78_candidate", False)
    )
    with pytest.raises(ValueError, match="do_not_scale_or_clip_wb78_candidate"):
        load_config(path)


def test_config_rejects_wb78_config_sha_tampering(tmp_path):
    path = _tampered_config(
        tmp_path,
        lambda c: c["inheritance"].__setitem__("workbook_78_config_sha256", "0" * 64),
    )
    with pytest.raises(ValueError, match="workbook-78 config SHA256 mismatch"):
        load_config(path)


def test_config_rejects_output_naming_change(tmp_path):
    path = _tampered_config(
        tmp_path, lambda c: c.__setitem__("output_parameter_naming", "measured_station_position")
    )
    with pytest.raises(ValueError, match="out_of_support_diagnostic_candidate"):
        load_config(path)


# ---------------------------------------------------------------------------
# Stage 0: exact reproduction of the WB78 calibration baseline (real data)
# ---------------------------------------------------------------------------


def test_reproduce_wb78_baseline_exact():
    config = _config()
    result = reproduce_baseline(config)
    report = result["report"]
    assert report["pass"] is True
    assert report["bank_sha256"] == config["inheritance"]["calibration_bank_sha256"]
    assert report["chi2_zero"] == pytest.approx(
        config["inheritance"]["baseline"]["chi2_zero_candidate"], rel=1e-9
    )
    # The reproduced candidate is only an out-of-support diagnostic candidate.
    assert report["within_linear_envelope"] is False
    assert all(report["checks"].values())


# ---------------------------------------------------------------------------
# Stage 1: covariance eigenstructure / whitening adequacy gate
# ---------------------------------------------------------------------------


def test_covariance_audit_adequate_model_within_gate():
    config = _config()
    rng = np.random.default_rng(1)
    bank = _synthetic_bank(rng, covariance_kind="well_conditioned")
    report = covariance_eigen_audit(config, bank)
    assert report["no_pair_dropped"] is True
    assert report["diagnostic_only"] is True
    assert report["alignment_authorized"] is False
    # Adequate model: whitened chi2/ndof ~ 1, well within the gate.
    assert report["whitened_chi2_resid_per_ndof"] <= report["whitened_chi2_per_ndof_gate"]
    assert report["whitened_chi2_within_gate"] is True


def test_covariance_audit_mismatch_model_exceeds_gate():
    config = _config()
    rng = np.random.default_rng(2)
    bank = _synthetic_bank(rng, covariance_kind="near_singular_mismatch")
    report = covariance_eigen_audit(config, bank)
    # Near-singular weight violated by isotropic truth -> huge whitened chi2.
    assert report["whitened_chi2_resid_per_ndof"] > report["whitened_chi2_per_ndof_gate"]
    assert report["whitened_chi2_within_gate"] is False
    # chi2 is concentrated in the smallest covariance eigenmode.
    assert report["fraction_chi2_from_smallest_eigenmode"] > 0.5


# ---------------------------------------------------------------------------
# Stage 2: empirical covariance cross-check (14973 <-> 14974)
# ---------------------------------------------------------------------------


def test_empirical_crosscheck_reproducible_runs():
    config = _config()
    rng = np.random.default_rng(3)
    bank = _synthetic_bank(rng, covariance_kind="well_conditioned", n_pairs_per_run=120)
    report = empirical_covariance_crosscheck(config, bank)
    assert report["empirical_covariance_not_promoted_to_weight"] is True
    assert report["diagnostic_only"] is True
    assert report["cross_run_reproducible"] is True
    cell = report["per_pair_type"]["0->1"]["cross_run"]
    assert cell["within_gate"] is True
    assert cell["generalized_eigenvalue_rms_log"] <= report["generalized_eigenvalue_rms_log_gate"]


def test_empirical_crosscheck_detects_nonreproducible_runs():
    config = _config()
    rng = np.random.default_rng(4)
    # Build a bank where the two runs have different covariance structure.
    bank = _synthetic_bank(rng, covariance_kind="well_conditioned", n_pairs_per_run=150)
    res = bank["banks"][0]["residual"]
    run_arr = bank["banks"][0]["run_id_arr"]
    # Scale run 14974's x-residuals by a large factor: a decisive change to the
    # empirical covariance structure that the generalized-eigenvalue metric
    # must flag as non-reproducible.
    mask = run_arr == 14974
    res[mask, 0] *= 6.0
    report = empirical_covariance_crosscheck(config, bank)
    assert report["cross_run_reproducible"] is False


# ---------------------------------------------------------------------------
# Stage 3: score contribution decomposition (deterministic, no pair dropped)
# ---------------------------------------------------------------------------


def test_score_decomposition_sums_to_total_and_deterministic():
    config = _config()
    subspace = _synthetic_subspace()
    rng = np.random.default_rng(5)
    transfer = _synthetic_transfer(rng)
    bank = _synthetic_bank(rng, covariance_kind="well_conditioned")
    # Build a real candidate on this bank via the WB78 solver path.
    from alignment import gauge_fixed_real_data_diagnostic as wb78

    gauges = wb78.build_gauge_candidates(config["wb78_config"], subspace)
    candidate = wb78.solve_candidate(config["wb78_config"], subspace, transfer, bank, gauges)
    if candidate.get("status") != "solved":
        pytest.skip("synthetic bank produced rank-zero information")
    report = score_contribution_decomposition(config, subspace, transfer, bank, candidate)
    assert report["no_pair_dropped"] is True
    assert report["diagnostic_only"] is True
    rank = report["informed_rank"]
    # Run decomposition sums to the same total as the sum of station-pair and
    # observable decompositions (all are the same score, differently grouped).
    for k in range(rank):
        by_run = sum(report["score_by_run"][r][k] for r in report["score_by_run"])
        by_pair = sum(report["score_by_station_pair"][p][k] for p in report["score_by_station_pair"])
        by_obs = sum(report["score_by_observable"][o][k] for o in report["score_by_observable"])
        assert by_run == pytest.approx(by_pair, rel=1e-6)
        assert by_run == pytest.approx(by_obs, rel=1e-6)
    # Deterministic.
    again = score_contribution_decomposition(config, subspace, transfer, bank, candidate)
    assert report["score_by_run"] == again["score_by_run"]


# ---------------------------------------------------------------------------
# Stage 5: covariance counterfactuals (diagnostic-only, never production)
# ---------------------------------------------------------------------------


def test_counterfactuals_marked_diagnostic_only_and_reduce_chi2_concentration():
    config = _config()
    subspace = _synthetic_subspace()
    rng = np.random.default_rng(6)
    transfer = _synthetic_transfer(rng)
    bank = _synthetic_bank(rng, covariance_kind="near_singular_mismatch", n_pairs_per_run=40)
    report = covariance_counterfactuals(config, subspace, transfer, bank)
    assert report["diagnostic_only"] is True
    assert report["alignment_authorized"] is False
    assert report["counterfactuals_not_promoted_to_alignment_model"] is True
    full = report["results"]["full_frozen"]
    diag = report["results"]["diagonal_marginal"]
    # The near-singular full covariance concentrates chi2 in the smallest mode;
    # the diagonal counterfactual removes that concentration.
    assert full["chi2_fraction_smallest_eigenmode"] > diag["chi2_fraction_smallest_eigenmode"]
    # All four counterfactuals present.
    assert set(report["results"]) == {
        "full_frozen",
        "diagonal_marginal",
        "condition_capped",
        "unit_weight",
    }


def test_counterfactual_covariance_kinds():
    rng = np.random.default_rng(7)
    cov = np.array([_spd_from_eigen(rng, (1.0e-4, 1.0, 1.0, 1.0))])
    diag = _counterfactual_covariance(cov, "diagonal", 1.0e3)[0]
    assert np.allclose(diag, np.diag(np.diagonal(cov[0])))
    unit = _counterfactual_covariance(cov, "unit", 1.0e3)[0]
    assert np.allclose(unit, np.eye(4))
    capped = _counterfactual_covariance(cov, "condition_capped", 1.0e3)[0]
    e = np.linalg.eigvalsh(capped)
    assert e[-1] / e[0] <= 1.0e3 * 1.001


# ---------------------------------------------------------------------------
# Stage 4: bootstrap-instability decomposition
# ---------------------------------------------------------------------------


def test_bootstrap_decomposition_reports_rank_distribution():
    config = _config()
    subspace = _synthetic_subspace()
    rng = np.random.default_rng(8)
    transfer = _synthetic_transfer(rng)
    bank = _synthetic_bank(rng, covariance_kind="well_conditioned", n_pairs_per_run=40)
    report = bootstrap_instability_decomposition(config, subspace, transfer, bank)
    assert report["diagnostic_only"] is True
    assert "bootstrap_rank_distribution" in report
    assert "diagonal_covariance_counterfactual" in report
    assert report["diagonal_counterfactual_is_diagnostic_only"] is True
    total = sum(int(v) for v in report["bootstrap_rank_distribution"].values())
    assert total == int(config["wb78_config"]["solver"]["bootstrap_replicates"])


# ---------------------------------------------------------------------------
# Stage 6: transfer support audit (injected kinematics)
# ---------------------------------------------------------------------------


def _pooled_extras_for_transfer(rng, transfer, n_mc=200, j_spread=0.05):
    source_station = np.asarray([0] * n_mc)
    target_station = np.asarray([1, 2, 3] * (n_mc // 3) + [1] * (n_mc % 3))
    jacobian = np.zeros((n_mc, 4, 7))
    for i in range(n_mc):
        mean = transfer["mean_jacobian"][f"0->{target_station[i]}"]
        jacobian[i] = mean + rng.normal(scale=j_spread, size=(4, 7))
    pooled = {"source_station_id": source_station, "target_station_id": target_station}

    class _Fit:
        derivative_native = jacobian

    return pooled, {"fit": _Fit(), "n_pairs": n_mc}


def _kinematics_cloud(rng, n, tx_scale, ty_scale, station):
    return {
        "pred_tx": rng.normal(0.0, tx_scale, n),
        "pred_ty": rng.normal(0.0, ty_scale, n),
        "target_station_id": np.full(n, station, dtype=np.int64),
    }


def test_transfer_support_passes_when_j_compact_and_support_overlaps():
    config = _config()
    rng = np.random.default_rng(9)
    transfer = _synthetic_transfer(rng)
    pooled, extras = _pooled_extras_for_transfer(rng, transfer, j_spread=0.01)
    bank = _synthetic_bank(rng, covariance_kind="well_conditioned")
    # Real kinematics overlap the MC cloud.
    real_kin = {
        "pred_tx": rng.normal(0, 0.02, 60),
        "pred_ty": rng.normal(0, 0.02, 60),
        "target_station_id": np.asarray([1, 2, 3] * 20, dtype=np.int64),
    }
    mc_cloud = {
        label: {"pred_tx": rng.normal(0, 0.03, 300), "pred_ty": rng.normal(0, 0.03, 300)}
        for label in ("0->1", "0->2", "0->3")
    }
    report = transfer_support_audit(
        config, transfer, pooled, extras, bank, real_kinematics=real_kin, mc_cloud=mc_cloud
    )
    assert report["residual_blind"] is True
    assert report["diagnostic_only"] is True
    assert report["mc_jacobian_dispersion_within_gate"] is True
    assert report["kinematic_support_within_gate"] is True
    assert report["pass"] is True


def test_transfer_support_fails_when_real_outside_mc_envelope():
    config = _config()
    rng = np.random.default_rng(10)
    transfer = _synthetic_transfer(rng)
    pooled, extras = _pooled_extras_for_transfer(rng, transfer, j_spread=0.01)
    bank = _synthetic_bank(rng, covariance_kind="well_conditioned")
    # Real kinematics far outside the tight MC cloud.
    real_kin = {
        "pred_tx": rng.normal(5.0, 0.02, 60),
        "pred_ty": rng.normal(5.0, 0.02, 60),
        "target_station_id": np.asarray([1, 2, 3] * 20, dtype=np.int64),
    }
    mc_cloud = {
        label: {"pred_tx": rng.normal(0, 0.01, 300), "pred_ty": rng.normal(0, 0.01, 300)}
        for label in ("0->1", "0->2", "0->3")
    }
    report = transfer_support_audit(
        config, transfer, pooled, extras, bank, real_kinematics=real_kin, mc_cloud=mc_cloud
    )
    assert report["kinematic_support_within_gate"] is False
    assert report["pass"] is False


# ---------------------------------------------------------------------------
# Stage 7: cross-run transportability
# ---------------------------------------------------------------------------


def test_cross_run_transportability_reports_both_runs():
    config = _config()
    subspace = _synthetic_subspace()
    rng = np.random.default_rng(11)
    transfer = _synthetic_transfer(rng)
    bank = _synthetic_bank(rng, covariance_kind="well_conditioned", n_pairs_per_run=90)
    report = cross_run_transportability(config, subspace, transfer, bank)
    assert report["report_only_not_a_correction"] is True
    assert report["diagnostic_only"] is True
    assert set(report["per_run"]) == {"14973", "14974"}
    assert "rank_equal" in report["comparison"]


# ---------------------------------------------------------------------------
# Decision tree branches
# ---------------------------------------------------------------------------


def _decision_inputs():
    reproduction = {"pass": True}
    covariance_audit = {"whitened_chi2_within_gate": True}
    crosscheck = {"cross_run_reproducible": True}
    transfer = {"pass": True}
    cross_run = {"pass": True}
    return reproduction, covariance_audit, crosscheck, transfer, cross_run


def test_decision_tree_pass_branch():
    reproduction, cov, cross, transfer, cross_run = _decision_inputs()
    decision = decide_campaign(
        reproduction=reproduction, covariance_audit=cov, crosscheck=cross,
        transfer_support=transfer, cross_run=cross_run,
    )
    assert decision["decision"] == DECISION_PASS
    assert decision["nonlinear_response_preregistration_allowed"] is True
    assert decision["geometry_write_allowed"] is False
    assert decision["held_out_accessed"] is False


def test_decision_tree_reproduction_failure_inconclusive():
    _r, cov, cross, transfer, cross_run = _decision_inputs()
    decision = decide_campaign(
        reproduction={"pass": False}, covariance_audit=cov, crosscheck=cross,
        transfer_support=transfer, cross_run=cross_run,
    )
    assert decision["decision"] == DECISION_INCONCLUSIVE
    assert decision["nonlinear_response_preregistration_allowed"] is False


def test_decision_tree_single_component_failures():
    reproduction, cov, cross, transfer, cross_run = _decision_inputs()
    assert decide_campaign(
        reproduction=reproduction, covariance_audit={"whitened_chi2_within_gate": False},
        crosscheck=cross, transfer_support=transfer, cross_run=cross_run,
    )["decision"] == DECISION_COVARIANCE
    assert decide_campaign(
        reproduction=reproduction, covariance_audit=cov,
        crosscheck={"cross_run_reproducible": False}, transfer_support=transfer, cross_run=cross_run,
    )["decision"] == DECISION_COVARIANCE
    assert decide_campaign(
        reproduction=reproduction, covariance_audit=cov, crosscheck=cross,
        transfer_support={"pass": False}, cross_run=cross_run,
    )["decision"] == DECISION_TRANSFER
    assert decide_campaign(
        reproduction=reproduction, covariance_audit=cov, crosscheck=cross,
        transfer_support=transfer, cross_run={"pass": False},
    )["decision"] == DECISION_CROSS_RUN


def test_decision_tree_multiple_failures():
    reproduction, cov, cross, transfer, cross_run = _decision_inputs()
    decision = decide_campaign(
        reproduction=reproduction, covariance_audit={"whitened_chi2_within_gate": False},
        crosscheck={"cross_run_reproducible": False}, transfer_support={"pass": False},
        cross_run={"pass": False},
    )
    assert decision["decision"] == DECISION_MULTIPLE
    assert set(decision["failed_components"]) == {
        "covariance_adequacy",
        "transfer_support",
        "cross_run_transportability",
    }
    assert decision["if_fail_continue"] == "residual_dq_monitoring_only"


# ---------------------------------------------------------------------------
# Structural held-out closure
# ---------------------------------------------------------------------------


def test_decision_tree_has_no_held_out_input():
    # decide_campaign takes no held-out argument at all: the campaign is
    # structurally incapable of opening held-out data.
    import inspect

    params = set(inspect.signature(decide_campaign).parameters)
    assert "held_out" not in params
    assert "held_out_bank" not in params
