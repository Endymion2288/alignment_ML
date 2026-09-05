"""Workbook-82 Wide-ty Real-Support-Matched MC Coverage Feasibility V1 tests.

Covers the pre-registered config freeze (flags, WB81 inheritance SHAs,
sealed-source exclusion, prohibitions), the structural held-out guard, the
residual-blind coverage metrics (99% Mahalanobis-envelope support fraction,
density bin-occupancy, quadrant coverage) on synthetic clouds, the
particle-species/momentum audit, the per-candidate gate evaluation, and every
decision-tree branch.  No alignment residual, Jacobian/FD derivative, singular
value, rank, or final-correction performance is used anywhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from alignment import wide_ty_mc_support_feasibility as wb82

CONFIG_PATH = "configs/wide_ty_real_support_matched_mc_coverage_feasibility_v1.yaml"


@pytest.fixture(scope="module")
def real_config():
    return wb82.load_config(CONFIG_PATH)


def _tampered_config(tmp_path: Path, mutate) -> str:
    raw = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "tampered.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(path)


# Minimal self-contained config for the metric/audit/decision unit tests (no
# dependence on the frozen WB81 artifacts).
MINI_CONFIG = {
    "coverage": {
        "mahalanobis2_99_2dof": 9.210,
        "min_fraction_within_support": 0.90,
        "min_truth_match_fraction": 0.99,
        "physical_acceptance_abs_tx_ty_max": 0.2,
        "density_bin_widths": [0.01, 0.02],
        "primary_density_bin_width": 0.01,
        "min_density_bin_occupancy_fraction": 0.90,
        "min_candidate_pairs_per_pair_type": 30,
        "min_candidate_pairs_absolute": 10,
        "min_independent_sources": 2,
        "quantile_levels": [50, 90, 95, 99],
    },
    "particle_domain_audit": {
        "expected_abs_pdg": [13],
        "momentum_gev_min": 10.0,
        "momentum_gev_max": 5000.0,
        "min_momentum_fraction_in_range": 0.50,
    },
    "decision_tree": {
        "validated": "existing_mc_real_wide_ty_support_validated",
        "angular_only_particle_mismatch": "existing_mc_angular_support_only_particle_domain_mismatch",
        "insufficient": "existing_mc_insufficient_real_wide_ty_support",
        "inconclusive": "wide_ty_mc_coverage_validation_inconclusive",
    },
}


# ---------------------------------------------------------------------------
# Config freeze + frozen inheritance
# ---------------------------------------------------------------------------


def test_config_loads_with_correct_schema(real_config):
    assert real_config["schema_version"] == wb82.SCHEMA_VERSION
    assert int(real_config["workbook"]) == 82


@pytest.mark.parametrize(
    "flag",
    (
        "geometry_write_allowed",
        "official_conditions_write_allowed",
        "real_data_candidate_alignment_authorized",
        "external_constraint_ingest_authorized",
        "held_out_accessed",
    ),
)
def test_frozen_permission_flags_must_be_false(tmp_path, flag):
    bad = _tampered_config(tmp_path, lambda c: c.__setitem__(flag, True))
    with pytest.raises(ValueError):
        wb82.load_config(bad)


@pytest.mark.parametrize(
    "flag",
    (
        "do_not_open_held_out",
        "do_not_open_sealed_test",
        "do_not_read_alignment_residual",
        "do_not_use_jacobian_or_fd_derivative_for_admission",
        "do_not_select_candidate_by_coverage_result",
        "residual_blind",
    ),
)
def test_frozen_prohibitions_must_be_true(tmp_path, flag):
    bad = _tampered_config(tmp_path, lambda c: c.__setitem__(flag, False))
    with pytest.raises(ValueError):
        wb82.load_config(bad)


def test_wb81_inheritance_sha_tamper_fails(tmp_path):
    bad = _tampered_config(
        tmp_path,
        lambda c: c["inheritance"].__setitem__("workbook_81_config_sha256", "0" * 64),
    )
    with pytest.raises(ValueError):
        wb82.load_config(bad)


def test_sealed_source_in_candidate_fails(tmp_path):
    def mutate(c):
        c["candidates"][0]["source_ids"] = ["mc24_100116_00030_00039"]  # sealed

    bad = _tampered_config(tmp_path, mutate)
    with pytest.raises(ValueError):
        wb82.load_config(bad)


def test_real_config_excludes_sealed_and_test(real_config):
    all_ids = {sid for cand in real_config["candidates"] for sid in cand["source_ids"]}
    assert "mc24_100116_00030_00039" not in all_ids
    assert "mc24_100117_00030_00039" not in all_ids
    assert "mc24_00020_00024" not in all_ids  # historical 100012 MC test split


def test_held_out_guard_raises_on_held_out_run():
    wb82.assert_no_held_out_access(np.array([14973, 14974]))  # calibration ok
    with pytest.raises(ValueError):
        wb82.assert_no_held_out_access(np.array([14975]))


# ---------------------------------------------------------------------------
# Coverage metrics on synthetic clouds
# ---------------------------------------------------------------------------


def _covering_cloud(real: np.ndarray, n=400, seed=0) -> np.ndarray:
    """Candidate cloud sampled to cover the real cloud's support."""
    rng = np.random.default_rng(seed)
    center = real.mean(axis=0)
    cov = np.cov(real, rowvar=False) * 1.5
    return rng.multivariate_normal(center, cov, size=n)


def test_mahalanobis_envelope_high_when_covering():
    rng = np.random.default_rng(1)
    real = rng.multivariate_normal([0.0, 0.0], [[4e-4, 0.0], [0.0, 1e-4]], size=200)
    cand = _covering_cloud(real)
    frac = wb82._mahalanobis_envelope_fraction(cand, real, 9.210)
    assert frac is not None and frac >= 0.90


def test_mahalanobis_envelope_low_when_not_covering():
    rng = np.random.default_rng(2)
    # Real cloud far wider in ty than the candidate.
    real = rng.multivariate_normal([0.0, 0.0], [[4e-4, 0.0], [0.0, 4e-4]], size=200)
    cand = rng.multivariate_normal([0.0, 0.0], [[4e-4, 0.0], [0.0, 1e-5]], size=400)
    frac = wb82._mahalanobis_envelope_fraction(cand, real, 9.210)
    assert frac is not None and frac < 0.90


def test_bin_occupancy_fraction():
    real = np.array([[0.0, 0.0], [0.005, 0.005], [0.5, 0.5]])  # last is far away
    cand = np.array([[0.001, 0.001], [0.004, 0.004], [0.002, 0.0]])
    frac = wb82._bin_occupancy_fraction(cand, real, 0.01)
    # two of the three real points share a 0.01 bin with a candidate point
    assert frac == pytest.approx(2.0 / 3.0)


def test_quadrant_coverage():
    real = np.array([[0.01, 0.01], [-0.01, 0.01], [0.01, -0.01], [-0.01, -0.01]])
    cand = np.array([[0.01, 0.01], [-0.01, 0.01], [0.01, -0.01]])  # missing (-,-)
    out = wb82._quadrant_coverage(cand, real)
    assert out["real_quadrants_populated"] == 4
    assert out["quadrants_covered"] == 3
    assert out["fraction"] == pytest.approx(0.75)


def test_compute_support_coverage_keys():
    rng = np.random.default_rng(3)
    real = rng.multivariate_normal([0.0, 0.0], [[4e-4, 0.0], [0.0, 1e-4]], size=100)
    cand = _covering_cloud(real)
    out = wb82.compute_support_coverage(real, cand, MINI_CONFIG)
    assert out["sufficient"] is True
    assert 0.0 <= out["fraction_within_mahalanobis_99"] <= 1.0
    assert set(out["density_bin_occupancy_fraction"]) == {"0.01", "0.02"}
    assert "real_ty_abs_quantiles" in out and "candidate_ty_abs_quantiles" in out


def test_compute_support_coverage_insufficient_when_few_candidates():
    real = np.random.default_rng(4).normal(0.0, 0.01, size=(50, 2))
    cand = np.random.default_rng(5).normal(0.0, 0.01, size=(3, 2))  # < min absolute
    out = wb82.compute_support_coverage(real, cand, MINI_CONFIG)
    assert out["sufficient"] is False


# ---------------------------------------------------------------------------
# Particle-domain audit
# ---------------------------------------------------------------------------


def _audit_cloud(pdg, qop_per_mev, n=100):
    return {
        "truth_pdg": np.full(n, pdg, dtype=np.int64),
        "truth_q_over_p_per_mev": np.full(n, qop_per_mev, dtype=np.float64),
        "source_z_mm": np.full(n, 1000.0),
        "lever_arm_mm": np.full(n, 2000.0),
    }


def test_audit_muon_compatible():
    # 100 GeV muon: |q/p| = 1/(1e5 MeV) = 1e-5 /MeV
    out = wb82.audit_particle_domain(_audit_cloud(-13, 1.0e-5), MINI_CONFIG)
    assert out["dominant_abs_pdg"] == 13
    assert out["species_compatible"] is True
    assert out["momentum_compatible"] is True
    assert out["particle_domain_compatible"] is True
    assert out["report_only"] is True


def test_audit_pion_species_mismatch():
    out = wb82.audit_particle_domain(_audit_cloud(211, 1.0e-5), MINI_CONFIG)
    assert out["dominant_abs_pdg"] == 211
    assert out["species_compatible"] is False
    assert out["particle_domain_compatible"] is False


def test_audit_low_momentum_mismatch():
    # 1 GeV muon: |q/p| = 1e-3 /MeV -> below the 10 GeV floor
    out = wb82.audit_particle_domain(_audit_cloud(13, 1.0e-3), MINI_CONFIG)
    assert out["species_compatible"] is True
    assert out["momentum_compatible"] is False
    assert out["particle_domain_compatible"] is False


# ---------------------------------------------------------------------------
# Gate evaluation + decision tree (all branches)
# ---------------------------------------------------------------------------


def _coverage_block(maha, density, n_cand=100, n_src=5):
    return {
        "sufficient": True,
        "fraction_within_mahalanobis_99": maha,
        "density_bin_occupancy_fraction": {"0.01": density, "0.02": density},
        "n_candidate": n_cand,
        "n_sources_with_pairs": n_src,
        "n_real": 200,
    }


def _audit_block(compatible=True):
    return {"particle_domain_compatible": compatible, "report_only": True}


def _candidate_result(name, maha, density, compatible=True, n_cand=100, n_src=5):
    cand = {"name": name}
    cov = {
        "0->1": _coverage_block(maha, density, n_cand, n_src),
        "0->2": _coverage_block(maha, density, n_cand, n_src),
        "0->3": {"sufficient": False},
    }
    audit = {"0->1": _audit_block(compatible), "0->2": _audit_block(compatible)}
    return wb82.evaluate_candidate(cand, cov, audit, MINI_CONFIG)


def test_evaluate_candidate_pass():
    res = _candidate_result("good", 0.95, 0.95)
    assert res["coverage_gates_pass"] is True
    assert res["particle_domain_compatible"] is True
    assert res["per_pair"]["0->1"]["within_gate"] is True
    assert res["per_pair"]["0->3"]["report_only"] is True


def test_evaluate_candidate_fails_on_mahalanobis():
    res = _candidate_result("bad", 0.80, 0.95)
    assert res["coverage_gates_pass"] is False


def test_evaluate_candidate_fails_on_density():
    # Wide-but-sparse: Mahalanobis passes but density occupancy fails.
    res = _candidate_result("sparse", 0.95, 0.70)
    assert res["coverage_gates_pass"] is False
    assert res["per_pair"]["0->1"]["mahalanobis_ok"] is True
    assert res["per_pair"]["0->1"]["density_ok"] is False


def test_evaluate_candidate_fails_on_statistics():
    res = _candidate_result("small", 0.95, 0.95, n_cand=20)  # < min 30
    assert res["coverage_gates_pass"] is False
    assert res["any_statistics_fail"] is True


def test_decision_validated():
    results = [_candidate_result("good", 0.95, 0.95, compatible=True)]
    out = wb82.decide(results, MINI_CONFIG)
    assert out["decision"] == "existing_mc_real_wide_ty_support_validated"
    assert out["existing_mc_real_wide_ty_support_validated"] is True
    assert out["real_kinematic_jacobian_support_validated"] is True
    assert out["conditional_j_retrain_permitted"] is True
    assert out["new_mc_generation_required"] is False
    # Even on success, WB82 never authorises alignment / measurement model.
    assert out["measurement_model_validated"] is False
    assert out["real_data_alignment_authorized"] is False
    assert out["held_out_accessed"] is False


def test_decision_angular_only_particle_mismatch():
    results = [_candidate_result("pion", 0.95, 0.95, compatible=False)]
    out = wb82.decide(results, MINI_CONFIG)
    assert out["decision"] == "existing_mc_angular_support_only_particle_domain_mismatch"
    assert out["existing_mc_real_wide_ty_support_validated"] is False
    assert out["conditional_j_retrain_permitted"] is False


def test_decision_insufficient():
    results = [_candidate_result("narrow", 0.80, 0.80)]
    out = wb82.decide(results, MINI_CONFIG)
    assert out["decision"] == "existing_mc_insufficient_real_wide_ty_support"
    assert out["new_mc_generation_required"] is True


def test_decision_inconclusive():
    # All candidates statistically insufficient (cannot even evaluate coverage).
    cand = {"name": "tiny"}
    cov = {"0->1": {"sufficient": False}, "0->2": {"sufficient": False}, "0->3": {"sufficient": False}}
    audit = {"0->1": _audit_block(), "0->2": _audit_block()}
    res = wb82.evaluate_candidate(cand, cov, audit, MINI_CONFIG)
    out = wb82.decide([res], MINI_CONFIG)
    assert out["decision"] == "wide_ty_mc_coverage_validation_inconclusive"
    assert out["existing_mc_real_wide_ty_support_validated"] is False
