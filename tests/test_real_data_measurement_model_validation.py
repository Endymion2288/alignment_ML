"""Workbook-80 real-data measurement-model reconstruction & cross-run validation tests.

Covers the pre-registered config freeze (flags, WB79 inheritance SHAs, output
naming, prohibitions), the structural held-out closure (code-level guard), the
exact reproduction of the WB79 covariance eigensystem/whitening baseline, the
covariance semantics audit (combined == propagated + target), the candidate
covariance cross-fit validation with derivation/validation run separation (no
same-run self-proof), the numerical-inversion audit (stable evaluation of the
SAME inverse vs. a statistical-model change), the MC source-disjoint
conditional-Jacobian validation (residual-blind features), the residual-blind
real-data transfer-support audit, and every decision-tree branch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from alignment.identifiable_subspace import FROZEN_RANK_TOLERANCE, IdentifiableSubspace, frozen_scales_for
from alignment import residual_covariance_model as cov_model
from alignment import conditional_jacobian_transfer as jtran
from alignment import real_data_measurement_model_validation as wb80
from alignment.real_data_measurement_model_validation import (
    CALIBRATION_RUN_IDS,
    DECISION_COV_NOT_VALIDATED,
    DECISION_CROSS_RUN_NOT_STABLE,
    DECISION_INCONCLUSIVE,
    DECISION_JAC_NOT_VALIDATED,
    DECISION_MULTIPLE_NOT_VALIDATED,
    DECISION_VALIDATED,
    HELD_OUT_RUN_IDS,
    SCHEMA_VERSION,
    assert_no_held_out_access,
    decide_campaign,
    load_config,
    reproduce_baseline,
)
from alignment.gauge_fixed_real_data_diagnostic import PARAMETER_NAMES

CONFIG_PATH = "configs/real_data_measurement_model_reconstruction_validation_v1.yaml"


@pytest.fixture(scope="module")
def real_config():
    """Load the real WB80 config once per module (loads the frozen subspace)."""
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


def _spd_from_eigen(rng: np.random.Generator, eigvals: tuple[float, ...]) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    return (q * np.asarray(eigvals)) @ q.T


# ---------------------------------------------------------------------------
# Synthetic enriched calibration records (covariance tests)
# ---------------------------------------------------------------------------


def _synthetic_enriched(
    rng: np.random.Generator,
    *,
    n_per_run_per_pair: int = 120,
    true_kind: str = "ms_leverarm",
    mismatch: bool = False,
) -> dict:
    """Two-run, two-pair-type synthetic enriched records with a known truth.

    true_kind="ms_leverarm": residuals drawn from C_combined + s_true*G_MS(L),
    so the C1 candidate is the true model and must validate.
    mismatch=True: C_combined is diagonal while the truth carries a strong x-y
    correlation that NONE of the pre-registered candidates (which only keep
    C_combined's structure, add an x-tx/y-ty random-walk term, add a diagonal
    floor, or inflate) can reproduce -> no candidate validates.
    """
    residuals, c_comb, c_prop, c_tgt, lever, ptx, pty, tstat, runs = ([] for _ in range(9))
    cell_mean = np.array([0.02, -0.03, 1.0e-3, -1.5e-3])
    for run in (14973, 14974):
        for pair in ((0, 1), (0, 2)):
            station = pair[1]
            L = 1907.55 if station == 1 else 2415.0
            if mismatch:
                base = np.diag([1.0, 1.0, 1.0, 1.0])  # diagonal weight: no x-y correlation
                true_cov = np.eye(4)
                true_cov[0, 1] = true_cov[1, 0] = 0.9  # x-y correlation no candidate can make
            else:
                base = _spd_from_eigen(rng, (0.5, 1.0, 2.0, 4.0))
                g_ms = cov_model._ms_leverarm_structure(np.asarray([L]))[0]
                true_cov = base + 2.0e-6 * g_ms  # C1 is the true model
            for _ in range(n_per_run_per_pair):
                r = rng.multivariate_normal(cell_mean, true_cov)
                residuals.append(r)
                c_comb.append(base)
                # Split base into prop+target (provenance shape; sum == combined).
                c_prop.append(base * 0.9)
                c_tgt.append(base * 0.1)
                lever.append(L)
                ptx.append(rng.normal(0.0, 0.01))
                pty.append(rng.normal(0.0, 0.005))
                tstat.append(station)
                runs.append(run)
    n = len(residuals)
    c_comb = np.asarray(c_comb)
    return {
        "residual": np.asarray(residuals),
        "c_combined": c_comb,
        "c_prop": np.asarray(c_prop),
        "c_target": np.asarray(c_tgt),
        "lever_arm_mm": np.asarray(lever),
        "pred_tx": np.asarray(ptx),
        "pred_ty": np.asarray(pty),
        "target_station_id": np.asarray(tstat, dtype=np.int64),
        "run_id": np.asarray(runs, dtype=np.int64),
        "combined_equals_prop_plus_target_max_abs_diff": 0.0,
    }


def _cov_config() -> dict:
    """Minimal config for the synthetic covariance cross-fit (no subspace load)."""
    raw = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    return {
        "covariance_model": raw["covariance_model"],
        "covariance_validation": raw["covariance_validation"],
    }


# ---------------------------------------------------------------------------
# Config freeze
# ---------------------------------------------------------------------------


def test_config_loads_and_freezes_flags(real_config):
    config = real_config
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["geometry_write_allowed"] is False
    assert config["official_conditions_write_allowed"] is False
    assert config["real_data_candidate_alignment_authorized"] is False
    assert config["external_constraint_ingest_authorized"] is False
    assert config["held_out_accessed"] is False
    assert config["eligible_external_physical_constraints"] == []
    assert config["population"]["calibration_runs"] == [14973, 14974]
    # WB79/WB78 contracts inherited verbatim.
    assert config["wb78_config"]["gauges"]["primary"] == "minimum_norm_scaled_gauge"
    # Frozen identifiable subspace injected (never re-derived on real data).
    assert config["subspace"].identifiable_rank == 5


def test_config_rejects_geometry_write(tmp_path):
    path = _tampered_config(tmp_path, lambda c: c.__setitem__("geometry_write_allowed", True))
    with pytest.raises(ValueError, match="geometry_write_allowed"):
        load_config(path)


def test_config_rejects_held_out_access_flag(tmp_path):
    path = _tampered_config(tmp_path, lambda c: c.__setitem__("held_out_accessed", True))
    with pytest.raises(ValueError, match="held_out_accessed"):
        load_config(path)


def test_config_rejects_external_constraint_ingest(tmp_path):
    path = _tampered_config(
        tmp_path, lambda c: c.__setitem__("external_constraint_ingest_authorized", True)
    )
    with pytest.raises(ValueError, match="external_constraint_ingest_authorized"):
        load_config(path)


def test_config_rejects_candidate_alignment(tmp_path):
    path = _tampered_config(
        tmp_path, lambda c: c.__setitem__("real_data_candidate_alignment_authorized", True)
    )
    with pytest.raises(ValueError, match="real_data_candidate_alignment_authorized"):
        load_config(path)


def test_config_rejects_wb79_config_sha_tampering(tmp_path):
    path = _tampered_config(
        tmp_path,
        lambda c: c["inheritance"].__setitem__("workbook_79_config_sha256", "0" * 64),
    )
    with pytest.raises(ValueError, match="workbook-79 config SHA256 mismatch"):
        load_config(path)


def test_config_rejects_diagonal_covariance_promotion(tmp_path):
    path = _tampered_config(
        tmp_path,
        lambda c: c.__setitem__("do_not_promote_diagonal_or_capped_or_unit_covariance", False),
    )
    with pytest.raises(ValueError, match="do_not_promote_diagonal_or_capped_or_unit_covariance"):
        load_config(path)


# ---------------------------------------------------------------------------
# Structural held-out closure (code-level guard)
# ---------------------------------------------------------------------------


def test_held_out_guard_refuses_every_held_out_run():
    for run in sorted(HELD_OUT_RUN_IDS):
        with pytest.raises(ValueError, match="held-out run access refused"):
            assert_no_held_out_access(np.asarray([run]))
    # A mix containing any held-out run also fails.
    with pytest.raises(ValueError):
        assert_no_held_out_access(np.asarray([14973, 14975]))


def test_held_out_guard_allows_calibration_runs():
    assert_no_held_out_access(np.asarray(sorted(CALIBRATION_RUN_IDS)))
    assert_no_held_out_access(np.asarray([14973, 14974, 14973]))


def test_decision_tree_has_no_held_out_input():
    import inspect

    params = set(inspect.signature(decide_campaign).parameters)
    assert "held_out" not in params
    assert "held_out_bank" not in params


# ---------------------------------------------------------------------------
# Stage 0: exact reproduction of the WB79 baseline (real data, slow)
# ---------------------------------------------------------------------------


def test_reproduce_wb79_baseline_exact(real_config):
    result = reproduce_baseline(real_config)
    report = result["report"]
    assert report["pass"] is True
    assert report["bank_sha256"] == real_config["inheritance"]["calibration_bank_sha256"]
    base = real_config["inheritance"]["wb79_baseline"]
    assert report["chi2_total"] == pytest.approx(base["chi2_total"], rel=1e-6)
    assert report["whitened_chi2_resid_per_ndof"] == pytest.approx(
        base["whitened_chi2_resid_per_ndof"], rel=1e-6
    )
    # All six pre-registered baseline checks (incl. n_pairs, condition median,
    # smallest-eigenmode chi2 fraction) reproduce the WB79 frozen values.
    assert all(report["checks"].values())
    assert report["held_out_accessed"] is False


# ---------------------------------------------------------------------------
# Covariance semantics audit
# ---------------------------------------------------------------------------


def test_covariance_semantics_audit_answers_12_questions():
    rng = np.random.default_rng(101)
    enriched = _synthetic_enriched(rng, n_per_run_per_pair=40)
    # semantics audit reads no config section; pass an empty mapping.
    report = cov_model.covariance_semantics_audit({}, enriched)
    assert report["n_pairs"] == enriched["residual"].shape[0]
    assert report["combined_equals_prop_plus_target_max_abs_diff"] == pytest.approx(0.0)
    assert len(report["questions"]) == 12
    # The residual-covariance formula question must be resolved as a plain sum.
    formula = report["questions"]["residual_covariance_formula"]
    assert "C_propagated_source + C_target" in formula["answer"]
    assert report["diagnostic_only"] is True
    assert report["alignment_authorized"] is False


# ---------------------------------------------------------------------------
# Candidate covariance structure
# ---------------------------------------------------------------------------


def test_ms_leverarm_structure_is_random_walk():
    L = np.asarray([1907.55])
    g = cov_model._ms_leverarm_structure(L)[0]
    # [[L^2/3, L/2],[L/2,1]] couples (x,tx) and (y,ty); no x-y or tx-ty coupling.
    assert g[0, 0] == pytest.approx(L[0] ** 2 / 3.0)
    assert g[0, 2] == pytest.approx(L[0] / 2.0)
    assert g[2, 2] == pytest.approx(1.0)
    assert g[1, 1] == pytest.approx(L[0] ** 2 / 3.0)
    assert g[1, 3] == pytest.approx(L[0] / 2.0)
    assert g[0, 1] == 0.0 and g[2, 3] == 0.0


def test_build_candidate_covariance_spd_and_kinds():
    rng = np.random.default_rng(102)
    base = _spd_from_eigen(rng, (0.5, 1.0, 2.0, 4.0))
    c_comb = np.array([base, base])
    lever = np.asarray([1907.55, 1907.55])
    floor = np.asarray([0.5, 1.0, 2.0, 4.0])
    frozen = cov_model.build_candidate_covariance("frozen_baseline", c_comb, lever, 1.0)
    assert np.allclose(frozen, c_comb)
    ms = cov_model.build_candidate_covariance("ms_leverarm_process_noise", c_comb, lever, 1.0e-6)
    assert np.all(np.linalg.eigvalsh(ms[0]) > 0.0)
    diag = cov_model.build_candidate_covariance(
        "scaled_diagonal_floor", c_comb, lever, 1.0e-3, floor
    )
    assert np.allclose(diag[0], base + 1.0e-3 * np.diag(floor))
    infl = cov_model.build_candidate_covariance("variance_inflation", c_comb, lever, 3.0)
    assert np.allclose(infl[0], 3.0 * base)
    with pytest.raises(ValueError, match="unknown candidate covariance kind"):
        cov_model.build_candidate_covariance("not_a_kind", c_comb, lever, 1.0)


# ---------------------------------------------------------------------------
# Cross-fit covariance validation (derivation/validation separation)
# ---------------------------------------------------------------------------


def test_cross_fit_recovers_true_ms_model():
    config = _cov_config()
    rng = np.random.default_rng(103)
    enriched = _synthetic_enriched(rng, n_per_run_per_pair=120, true_kind="ms_leverarm")
    report = cov_model.cross_fit_covariance_validation(config, enriched)
    c1 = report["candidates"]["C1_ms_leverarm"]
    # The true model is C1: it must validate on both cross-fit folds.
    assert c1["validated_both_folds"] is True
    assert "C1_ms_leverarm" in report["validated_physical_candidates"]
    assert report["covariance_model_validated"] is True
    # Never enters an alignment solve; empirical covariance never promoted.
    assert report["no_candidate_entered_alignment_solve"] is True
    assert report["empirical_covariance_not_promoted_to_weight"] is True
    assert report["alignment_authorized"] is False


def test_cross_fit_detects_structural_mismatch():
    config = _cov_config()
    rng = np.random.default_rng(104)
    enriched = _synthetic_enriched(rng, n_per_run_per_pair=120, mismatch=True)
    report = cov_model.cross_fit_covariance_validation(config, enriched)
    # No physically-motivated candidate can reproduce an isotropic truth from a
    # near-singular base: the campaign must NOT validate the covariance model.
    assert report["covariance_model_validated"] is False
    assert report["validated_physical_candidates"] == []


def test_cross_fit_is_deterministic():
    config = _cov_config()
    rng = np.random.default_rng(105)
    enriched = _synthetic_enriched(rng, n_per_run_per_pair=80)
    r1 = cov_model.cross_fit_covariance_validation(config, enriched)
    r2 = cov_model.cross_fit_covariance_validation(config, enriched)
    s1 = r1["candidates"]["C1_ms_leverarm"]["cross_run_scale_reproducibility"]
    s2 = r2["candidates"]["C1_ms_leverarm"]["cross_run_scale_reproducibility"]
    assert s1 == s2


def test_cross_fit_never_self_proves_same_run():
    """Cross-fit validates each run with the scale derived from the OTHER run."""
    config = _cov_config()
    rng = np.random.default_rng(106)
    enriched = _synthetic_enriched(rng, n_per_run_per_pair=80)
    c_model = cov_model.build_crossfit_model_covariance(
        config, enriched, "ms_leverarm_process_noise"
    )
    # For each run, the model covariance must equal the assembly using the scale
    # derived from the OTHER run (no same-run derivation -> no self-proof).
    for run, other in ((14973, 14974), (14974, 14973)):
        scales = cov_model.derive_pairtype_scales(
            "ms_leverarm_process_noise",
            enriched["c_combined"],
            enriched["lever_arm_mm"],
            enriched["residual"],
            enriched["run_id"],
            enriched["target_station_id"],
            other,
            30,
        )
        mask = enriched["run_id"] == run
        expect = cov_model.assemble_model_covariance(
            "ms_leverarm_process_noise",
            enriched["c_combined"][mask],
            enriched["lever_arm_mm"][mask],
            enriched["target_station_id"][mask],
            scales,
        )
        assert np.allclose(c_model[mask], expect)


def test_cross_fit_requires_exactly_two_runs():
    config = _cov_config()
    rng = np.random.default_rng(107)
    enriched = _synthetic_enriched(rng, n_per_run_per_pair=40)
    # Collapse to a single run: cross-fit must refuse (cannot self-prove).
    n = enriched["run_id"].shape[0]
    mask = enriched["run_id"] == 14973
    single = {
        k: (v[mask] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n else v)
        for k, v in enriched.items()
    }
    with pytest.raises(ValueError, match="exactly two runs"):
        cov_model.build_crossfit_model_covariance(config, single, "ms_leverarm_process_noise")


# ---------------------------------------------------------------------------
# Numerical inversion audit (stable evaluation of the SAME inverse)
# ---------------------------------------------------------------------------


def test_numerical_inversion_methods_agree_on_same_covariance():
    rng = np.random.default_rng(108)
    enriched = _synthetic_enriched(rng, n_per_run_per_pair=40)
    report = cov_model.numerical_inversion_audit({}, enriched)
    # All methods evaluate the SAME inverse; total chi2 must agree closely.
    totals = report["total_chi2_by_method"]
    ref = totals["eigendecomposition"]
    for name, value in totals.items():
        assert value == pytest.approx(ref, rel=1e-6)
    # No factorization failures on well-conditioned synthetic covariance.
    assert all(v == 0 for v in report["factorization_failures"].values())
    # The audit is explicitly a numerical-implementation comparison (method A),
    # NOT a statistical-model change (B): the note must state that no
    # eigenvalue floor/cap is applied to C itself.
    assert "SAME frozen covariance" in report["note"]
    assert "statistical-model change" in report["note"]
    assert report["diagnostic_only"] is True
    assert report["alignment_authorized"] is False


# ---------------------------------------------------------------------------
# MC source-disjoint conditional-Jacobian validation
# ---------------------------------------------------------------------------


def _synthetic_mc_bank(rng, source_index, n_pairs=120, j_spread=0.02):
    """One synthetic MC source bank with a kinematic-dependent Jacobian."""
    target = np.asarray([1, 2, 3] * (n_pairs // 3) + [1] * (n_pairs % 3), dtype=np.int64)
    source = np.zeros(n_pairs, dtype=np.int64)
    tx = rng.normal(0.0, 0.02, n_pairs)
    ty = rng.normal(0.0, 0.01, n_pairs)
    jacobian = np.zeros((n_pairs, 4, 7))
    for i in range(n_pairs):
        label_mean = np.full((4, 7), 0.1 * target[i])
        # Smooth kinematic dependence (same for all sources) + per-source noise.
        label_mean += 0.5 * tx[i] + 0.3 * ty[i]
        jacobian[i] = label_mean + rng.normal(scale=j_spread, size=(4, 7))
    return {
        "jacobian": jacobian,
        "pred_tx": tx,
        "pred_ty": ty,
        "source_station_id": source,
        "target_station_id": target,
    }


def _jac_config() -> dict:
    raw = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    return {"jacobian_transfer": raw["jacobian_transfer"], "subspace": _synthetic_subspace()}


def test_jacobian_model_source_disjoint_and_validated():
    config = _jac_config()
    rng = np.random.default_rng(109)
    construction = [_synthetic_mc_bank(rng, i) for i in range(3)]
    validation = [_synthetic_mc_bank(rng, 100 + i) for i in range(4)]
    report = jtran.mc_jacobian_model_validation(
        config, construction_banks=construction, validation_banks=validation
    )
    # Construction and validation source sets are disjoint and reported.
    assert report["source_disjoint"] is True
    # The smooth kinematic model is recoverable: at least the regression model
    # must pass the MC source-disjoint gates.
    assert len(report["mc_validated_models"]) >= 1
    # Features are residual-blind kinematics only.
    assert set(report["residual_blind_features"]) <= {"pred_tx", "pred_ty"}
    assert "residual" not in " ".join(report["residual_blind_features"]).lower()
    assert report["alignment_authorized"] is False


def test_jacobian_features_are_residual_blind():
    import inspect

    # fit_conditional_jacobian_model takes only (kind, construction, spec): the
    # construction bank carries Jacobians + kinematics, never alignment residuals.
    params = set(inspect.signature(jtran.fit_conditional_jacobian_model).parameters)
    assert "residual" not in params
    assert "residuals" not in params
    # The construction features are residual-blind kinematics.
    config = _jac_config()
    features = config["jacobian_transfer"]["features"]
    assert "residual" not in " ".join(features).lower()


# ---------------------------------------------------------------------------
# Real-data transfer-support audit (residual-blind)
# ---------------------------------------------------------------------------


def _mc_construction_kinematics(rng, tx_scale=0.03, ty_scale=0.008, n=300):
    return {
        label: {
            "pred_tx": rng.normal(0.0, tx_scale, n),
            "pred_ty": rng.normal(0.0, ty_scale, n),
        }
        for label in ("0->1", "0->2", "0->3")
    }


def test_real_support_passes_when_real_within_mc_envelope():
    config = _jac_config()
    rng = np.random.default_rng(110)
    mc_kin = _mc_construction_kinematics(rng)
    n = 90
    real_kin = {
        "pred_tx": rng.normal(0.0, 0.03, n),
        "pred_ty": rng.normal(0.0, 0.008, n),
        "target_station_id": np.asarray([1, 2, 3] * 30, dtype=np.int64),
    }
    model = {"kind": "station_pair_mean"}
    report = jtran.real_jacobian_support_validation(config, model, real_kin, mc_kin)
    assert report["residual_blind"] is True
    assert report["out_of_support_never_extrapolated_into_solve"] is True
    assert report["kinematic_support_within_gate"] is True
    assert report["alignment_authorized"] is False


def test_real_support_fails_when_real_outside_mc_envelope():
    config = _jac_config()
    rng = np.random.default_rng(111)
    mc_kin = _mc_construction_kinematics(rng, tx_scale=0.01, ty_scale=0.005)
    n = 90
    # Real kinematics far outside the tight MC envelope.
    real_kin = {
        "pred_tx": rng.normal(0.4, 0.01, n),
        "pred_ty": rng.normal(0.3, 0.01, n),
        "target_station_id": np.asarray([1, 2, 3] * 30, dtype=np.int64),
    }
    model = {"kind": "station_pair_mean"}
    report = jtran.real_jacobian_support_validation(config, model, real_kin, mc_kin)
    assert report["kinematic_support_within_gate"] is False


def test_real_support_audit_reads_no_residual():
    import inspect

    params = set(inspect.signature(jtran.real_jacobian_support_validation).parameters)
    assert "residual" not in params
    assert "residuals" not in params


# ---------------------------------------------------------------------------
# Decision tree (all branches)
# ---------------------------------------------------------------------------


def _decision_inputs():
    reproduction = {"pass": True}
    cov = {"covariance_model_validated": True}
    jac = {"jacobian_transfer_model_validated": True}
    cross = {"pass": True}
    return reproduction, cov, jac, cross


def test_decision_tree_validated_branch():
    reproduction, cov, jac, cross = _decision_inputs()
    decision = decide_campaign(
        reproduction=reproduction, covariance_validation=cov,
        jacobian_validation=jac, cross_run_info=cross,
    )
    assert decision["decision"] == DECISION_VALIDATED
    assert decision["real_data_alignment_v2_preregistration_allowed"] is True
    # Even when validated, WB80 authorizes only a NEW workbook, never a write.
    assert decision["geometry_write_allowed"] is False
    assert decision["official_conditions_write_allowed"] is False
    assert decision["real_data_candidate_alignment_authorized"] is False
    assert decision["held_out_accessed"] is False


def test_decision_tree_reproduction_failure_inconclusive():
    _r, cov, jac, cross = _decision_inputs()
    decision = decide_campaign(
        reproduction={"pass": False}, covariance_validation=cov,
        jacobian_validation=jac, cross_run_info=cross,
    )
    assert decision["decision"] == DECISION_INCONCLUSIVE
    assert decision["real_data_alignment_v2_preregistration_allowed"] is False


def test_decision_tree_single_component_failures():
    reproduction, cov, jac, cross = _decision_inputs()
    assert decide_campaign(
        reproduction=reproduction, covariance_validation={"covariance_model_validated": False},
        jacobian_validation=jac, cross_run_info=cross,
    )["decision"] == DECISION_COV_NOT_VALIDATED
    assert decide_campaign(
        reproduction=reproduction, covariance_validation=cov,
        jacobian_validation={"jacobian_transfer_model_validated": False}, cross_run_info=cross,
    )["decision"] == DECISION_JAC_NOT_VALIDATED


def test_decision_tree_multiple_failures():
    reproduction, _c, _j, cross = _decision_inputs()
    decision = decide_campaign(
        reproduction=reproduction, covariance_validation={"covariance_model_validated": False},
        jacobian_validation={"jacobian_transfer_model_validated": False}, cross_run_info=cross,
    )
    assert decision["decision"] == DECISION_MULTIPLE_NOT_VALIDATED
    assert decision["real_data_alignment_v2_preregistration_allowed"] is False
    assert decision["if_fail_continue"] == "residual_dq_monitoring_only"


def test_decision_tree_cross_run_not_stable():
    reproduction, cov, jac, _x = _decision_inputs()
    decision = decide_campaign(
        reproduction=reproduction, covariance_validation=cov,
        jacobian_validation=jac, cross_run_info={"pass": False},
    )
    assert decision["decision"] == DECISION_CROSS_RUN_NOT_STABLE
    assert decision["real_data_alignment_v2_preregistration_allowed"] is False


def test_decision_tree_cross_run_skipped_when_not_validated():
    reproduction, _c, _j, _x = _decision_inputs()
    # When a component fails, the cross-run check is skipped (None) and must not
    # raise; the decision is driven by the failed component.
    decision = decide_campaign(
        reproduction=reproduction, covariance_validation={"covariance_model_validated": False},
        jacobian_validation={"jacobian_transfer_model_validated": False}, cross_run_info=None,
    )
    assert decision["decision"] == DECISION_MULTIPLE_NOT_VALIDATED
    assert decision["cross_run_information_stable"] is None
