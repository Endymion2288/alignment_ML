"""Workbook-81 FaserActsExtrapolation propagated-covariance provenance & closure tests.

Covers the pre-registered config freeze (flags, WB80 inheritance SHAs, MC
source-disjointness, prohibitions), the structural held-out / real-data /
geometry-write guards, the exact [x,y,tx,ty] parameter ordering and covariance
basis/unit, the exact C_prop extraction + deterministic truth-target identity
from synthetic ROOT ntuples, the no-event-dropping contract, the deterministic
symmetric whitening, the no-covariance (mode-2) control path, the
diagnostic-only variant contract, and every decision-tree branch including the
evidence-based failure classification.
"""

from __future__ import annotations

from pathlib import Path

import awkward as ak
import numpy as np
import pytest
import uproot
import yaml

from alignment import propagated_covariance_closure as pcc
from alignment.propagated_covariance_closure import (
    DECISION_INCONCLUSIVE,
    DECISION_NOT_CALIBRATED,
    DECISION_PROVENANCE_UNRESOLVED,
    DECISION_TRUTH_CLOSURE_UNAVAILABLE,
    DECISION_VALIDATED_CANONICAL,
    MECHANISM_OVERESTIMATED_TRANSPORT,
    ConfigError,
)

CONFIG_PATH = "configs/faseracts_propagated_covariance_provenance_closure_v1.yaml"


@pytest.fixture(scope="module")
def real_config():
    return pcc.load_config(CONFIG_PATH)


def _tampered_config(tmp_path: Path, mutate) -> str:
    raw = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "tampered.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Config freeze + frozen inheritance
# ---------------------------------------------------------------------------


def test_config_loads_with_correct_schema(real_config):
    assert real_config["schema_version"] == "faseracts-propagated-covariance-provenance-closure-v1"
    assert int(real_config["workbook"]) == 81


@pytest.mark.parametrize(
    "flag",
    (
        "geometry_write_allowed",
        "official_conditions_write_allowed",
        "real_data_candidate_alignment_authorized",
        "real_data_alignment_authorized",
        "external_constraint_ingest_authorized",
        "held_out_accessed",
    ),
)
def test_frozen_permission_flags_must_be_false(tmp_path, flag):
    bad = _tampered_config(tmp_path, lambda c: c.__setitem__(flag, True))
    with pytest.raises(ConfigError):
        pcc.load_config(bad)


@pytest.mark.parametrize(
    "flag",
    (
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
    ),
)
def test_frozen_prohibitions_must_be_true(tmp_path, flag):
    bad = _tampered_config(tmp_path, lambda c: c.__setitem__(flag, False))
    with pytest.raises(ConfigError):
        pcc.load_config(bad)


def test_wb80_inheritance_sha_tamper_fails(tmp_path):
    bad = _tampered_config(
        tmp_path,
        lambda c: c["inheritance"].__setitem__("workbook_80_config_sha256", "0" * 64),
    )
    with pytest.raises(ConfigError):
        pcc.load_config(bad)


def test_mc_split_must_be_source_disjoint(tmp_path, real_config):
    overlap = real_config["mc_data"]["construction_source_ids"][0]

    def mutate(c):
        c["mc_data"]["validation_source_ids"] = [overlap]

    bad = _tampered_config(tmp_path, mutate)
    with pytest.raises(ConfigError):
        pcc.load_config(bad)


def test_real_config_mc_split_is_disjoint(real_config):
    mc = real_config["mc_data"]
    assert not (
        set(mc["construction_source_ids"]) & set(mc["validation_source_ids"])
    )


def test_held_out_guard_is_structural_noop():
    # WB81 exposes no real-data entry point; the guard must never raise.
    assert pcc.assert_no_held_out_access() is None


# ---------------------------------------------------------------------------
# Provenance audit
# ---------------------------------------------------------------------------


def test_provenance_audit_closes_all_15_questions(real_config):
    prov = pcc.provenance_audit(real_config)
    assert prov["n_questions"] == 15
    assert prov["n_unresolved"] == 0
    assert prov["provenance_closed"] is True
    verdicts = {a["verdict"] for a in prov["questions"].values()}
    assert verdicts <= {"resolved_from_source", "resolved_from_runtime_config"}
    # The decisive finding: deterministic transport with no stochastic noise.
    assert "NO actual stochastic" in prov["questions"][
        "deterministic_transport_without_stochastic_noise"
    ]["answer"]


# ---------------------------------------------------------------------------
# Symmetric whitening (deterministic)
# ---------------------------------------------------------------------------


def test_symmetric_whiten_deterministic():
    C = np.array([[4.0, 1.0], [1.0, 9.0]])
    e = np.array([2.0, 3.0])
    z1 = pcc._symmetric_whiten(e, C)
    z2 = pcc._symmetric_whiten(e, C)
    assert z1 is not None and z2 is not None
    np.testing.assert_allclose(z1, z2)
    # Whitened residual satisfies z.z == e^T C^-1 e (Mahalanobis).
    np.testing.assert_allclose(z1 @ z1, e @ np.linalg.solve(C, e), rtol=1e-10)


def test_symmetric_whiten_non_positive_definite_returns_none():
    C = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3, -1
    assert pcc._symmetric_whiten(np.array([1.0, 1.0]), C) is None


# ---------------------------------------------------------------------------
# Closure records / metrics on synthetic data
# ---------------------------------------------------------------------------


def _make_records(e_prop: np.ndarray, c_prop: np.ndarray, has_covariance=True):
    n = e_prop.shape[0]
    return pcc.ClosureRecords(
        station_pair=(0, 1),
        q_over_p_mode=0,
        e_prop=e_prop,
        c_prop=c_prop,
        lever_arm_mm=np.full(n, 1000.0),
        pred_tx=np.full(n, 0.01),
        pred_ty=np.full(n, 0.01),
        n_matched=n,
        has_covariance=has_covariance,
    )


def _calibrated_records(n=200, seed=0):
    """e_prop drawn from N(0, C0); c_prop set to the empirical covariance."""
    rng = np.random.default_rng(seed)
    A = np.array(
        [
            [2.0, 0.3, 0.0, 0.0],
            [0.3, 1.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.05],
            [0.0, 0.0, 0.05, 0.4],
        ]
    )
    C0 = A @ A.T
    E = rng.multivariate_normal(np.zeros(4), C0, size=n)
    E = E - E.mean(axis=0)
    C_emp = np.cov(E.T)
    c_prop = np.repeat(C_emp[None, :, :], n, axis=0)
    return _make_records(E, c_prop)


def test_metrics_calibrated_when_cprop_matches_empirical(real_config):
    metrics = pcc.compute_closure_metrics(_calibrated_records(), real_config)
    assert metrics["sufficient"] is True
    assert metrics["has_covariance"] is True
    # cov_z ~ identity, chi2/ndof ~ 1, pencil ratio ~ 1, generalized eig ~ 1.
    np.testing.assert_allclose(metrics["cov_z_eigenvalues"], np.ones(4), atol=0.6)
    assert abs(metrics["chi2_per_ndof"] - 1.0) < 0.5
    assert 0.5 < metrics["pencil"]["variance_ratio_prop_over_emp"] < 2.0
    verdict = pcc.evaluate_closure_gates(metrics, real_config)
    assert verdict["calibrated"] is True


def test_metrics_overestimated_cprop_fails_gates(real_config):
    recs = _calibrated_records()
    recs = pcc.ClosureRecords(
        **{**recs.__dict__, "c_prop": recs.c_prop * 1000.0}
    )
    metrics = pcc.compute_closure_metrics(recs, real_config)
    # C_prop 1000x too large -> pencil ratio huge, generalized eig tiny.
    assert metrics["pencil"]["variance_ratio_prop_over_emp"] > 100.0
    assert max(metrics["generalized_eigenvalues_cemp_over_cprop"]) < 0.01
    verdict = pcc.evaluate_closure_gates(metrics, real_config)
    assert verdict["calibrated"] is False


def test_metrics_insufficient_pairs(real_config):
    recs = _make_records(np.zeros((3, 4)), np.repeat(np.eye(4)[None], 3, axis=0))
    metrics = pcc.compute_closure_metrics(recs, real_config)
    assert metrics["sufficient"] is False
    verdict = pcc.evaluate_closure_gates(metrics, real_config)
    assert verdict["calibrated"] is False
    assert verdict["reason"] == "insufficient_pairs"


def test_no_covariance_control_reports_only_eprop(real_config):
    n = 50
    rng = np.random.default_rng(1)
    E = rng.normal(0.0, 2.0, size=(n, 4))
    c_nan = np.full((n, 4, 4), np.nan)
    recs = _make_records(E, c_nan, has_covariance=False)
    metrics = pcc.compute_closure_metrics(recs, real_config)
    assert metrics["has_covariance"] is False
    assert metrics["chi2_per_ndof"] is None
    assert metrics["pencil"] is None
    assert metrics["marginal_rms_e_demeaned"] is not None
    verdict = pcc.evaluate_closure_gates(metrics, real_config)
    assert verdict["reason"] == "no_covariance_control"


def test_kinematic_dependence_binned(real_config):
    metrics = pcc.compute_closure_metrics(_calibrated_records(), real_config)
    dep = metrics["kinematic_dependence"]
    assert "lever_arm" in dep and "abs_tx" in dep and "abs_ty" in dep
    # Every record falls in exactly one bin (no dropping).
    assert sum(b["n"] for b in dep["lever_arm"]) == metrics["n_pairs"]


# ---------------------------------------------------------------------------
# Exact C_prop extraction + truth identity from synthetic ROOT ntuples
# ---------------------------------------------------------------------------


def _write_synthetic_mc(refit: Path) -> dict:
    """Write a one-event/one-track synthetic propagations.root + enhanced ntuple.

    Truth at station 1 (target): pos=(10,20,999), mom giving tx=0.005, ty=0.003.
    target_z_mm=1000 -> dz=+1 straight-line correction.  Station 0 z=500 (lever).
    """
    refit.mkdir(parents=True, exist_ok=True)
    # --- flat propagations tree ---
    cov = {
        "cov_xx_mm2": 1.0, "cov_xy_mm2": 0.1, "cov_xtx_mm": 0.01, "cov_xty_mm": 0.0,
        "cov_yy_mm2": 4.0, "cov_ytx_mm": 0.0, "cov_yty_mm": 0.02,
        "cov_txtx": 1e-6, "cov_txty": 0.0, "cov_tyty": 4e-6,
    }
    prop = {
        "run_id": np.asarray([1], dtype=np.int32),
        "event_id": np.asarray([100], dtype=np.int64),
        "source_tracklet_id": np.asarray([5], dtype=np.int32),
        "target_tracklet_id": np.asarray([6], dtype=np.int32),
        "source_station_id": np.asarray([0], dtype=np.int16),
        "target_station_id": np.asarray([1], dtype=np.int16),
        "truth_particle_id": np.asarray([10001], dtype=np.int64),
        "target_z_mm": np.asarray([1000.0], dtype=np.float64),
        "pred_x_mm": np.asarray([10.1], dtype=np.float64),
        "pred_y_mm": np.asarray([20.2], dtype=np.float64),
        "pred_tx": np.asarray([0.0051], dtype=np.float64),
        "pred_ty": np.asarray([0.0031], dtype=np.float64),
        "success": np.asarray([True], dtype=bool),
        "has_covariance": np.asarray([True], dtype=bool),
        "q_over_p_mode": np.asarray([0], dtype=np.int8),
    }
    for k, v in cov.items():
        prop[f"pred_{k}"] = np.asarray([v], dtype=np.float64)
    with uproot.recreate(refit / "propagations.root") as f:
        f["propagations"] = prop

    # --- enhanced ntuple (TTree with jagged per-track truth, matching production) ---
    px, py, pz = 0.5, 0.3, 100.0  # tx=0.005, ty=0.003
    st_pos = {0: (0.0, 0.0, 500.0), 1: (10.0, 20.0, 999.0),
              2: (0.0, 0.0, 2000.0), 3: (0.0, 0.0, 3000.0)}
    enh_types: dict = {"run": np.int32, "eventID": np.int32, "truth_barcode": "var * uint64"}
    enh_data: dict = {
        "run": np.asarray([1], dtype=np.int32),
        "eventID": np.asarray([100], dtype=np.int32),
        "truth_barcode": ak.Array([[10001]]),
    }
    for s in range(4):
        x, y, z = st_pos[s]
        for coord, val in (("x", x), ("y", y), ("z", z)):
            enh_types[f"truth_st{s}_{coord}"] = "var * float64"
            enh_data[f"truth_st{s}_{coord}"] = ak.Array([[val]])
        for mom, val in (("px", px), ("py", py), ("pz", pz)):
            enh_types[f"truth_st{s}_{mom}"] = "var * float64"
            enh_data[f"truth_st{s}_{mom}"] = ak.Array([[val]])
    with uproot.recreate(refit / "enhanced_tracklets.root") as f:
        tree = f.mktree("nt", enh_types)
        tree.extend(enh_data)
    return cov


def test_build_closure_records_exact_extraction_and_ordering(tmp_path, real_config):
    cov = _write_synthetic_mc(tmp_path)
    recs = pcc.build_closure_records(
        propagations_path=tmp_path / "propagations.root",
        enhanced_path=tmp_path / "enhanced_tracklets.root",
        station_pair=(0, 1),
        q_over_p_mode=0,
        config=real_config,
    )
    assert recs.size == 1 and recs.n_matched == 1
    # Truth target state: straight-line correction dz = 1000 - 999 = +1 mm.
    # x_truth = 10 + 0.005*1 = 10.005 ; y_truth = 20 + 0.003*1 = 20.003.
    # e_prop = pred - truth (ordering [x, y, tx, ty]).
    np.testing.assert_allclose(
        recs.e_prop[0],
        [10.1 - 10.005, 20.2 - 20.003, 0.0051 - 0.005, 0.0031 - 0.003],
        atol=1e-9,
    )
    # Exact C_prop reproduction in the [x,y,tx,ty] basis (units mm / rad).
    expected_C = np.array(
        [
            [cov["cov_xx_mm2"], cov["cov_xy_mm2"], cov["cov_xtx_mm"], cov["cov_xty_mm"]],
            [cov["cov_xy_mm2"], cov["cov_yy_mm2"], cov["cov_ytx_mm"], cov["cov_yty_mm"]],
            [cov["cov_xtx_mm"], cov["cov_ytx_mm"], cov["cov_txtx"], cov["cov_txty"]],
            [cov["cov_xty_mm"], cov["cov_yty_mm"], cov["cov_txty"], cov["cov_tyty"]],
        ]
    )
    np.testing.assert_allclose(recs.c_prop[0], expected_C, atol=1e-12)
    # Lever arm from the source-station truth z.
    assert recs.lever_arm_mm[0] == pytest.approx(500.0)


def test_build_closure_records_is_deterministic(tmp_path, real_config):
    _write_synthetic_mc(tmp_path)
    kwargs = dict(
        propagations_path=tmp_path / "propagations.root",
        enhanced_path=tmp_path / "enhanced_tracklets.root",
        station_pair=(0, 1),
        q_over_p_mode=0,
        config=real_config,
    )
    r1 = pcc.build_closure_records(**kwargs)
    r2 = pcc.build_closure_records(**kwargs)
    np.testing.assert_array_equal(r1.e_prop, r2.e_prop)
    np.testing.assert_array_equal(r1.c_prop, r2.c_prop)


def test_truth_reference_z_correction_toggle(tmp_path, real_config):
    _write_synthetic_mc(tmp_path)
    cfg = dict(real_config)
    cfg["truth_reference"] = {**real_config["truth_reference"], "straight_line_z_correction": False}
    recs = pcc.build_closure_records(
        propagations_path=tmp_path / "propagations.root",
        enhanced_path=tmp_path / "enhanced_tracklets.root",
        station_pair=(0, 1),
        q_over_p_mode=0,
        config=cfg,
    )
    # Without the correction the truth position is the raw station-1 pos.
    np.testing.assert_allclose(recs.e_prop[0][:2], [10.1 - 10.0, 20.2 - 20.0], atol=1e-9)


# ---------------------------------------------------------------------------
# Diagnostic variants contract
# ---------------------------------------------------------------------------


def test_diagnostic_variants_are_diagnostic_only(real_config):
    variants = real_config["diagnostic_variants"]
    assert variants["diagnostic_only"] is True
    assert variants["alignment_authorized"] is False
    # A diagnostic variant can never be promoted to a production covariance.
    assert real_config["do_not_promote_diagnostic_variant_to_production"] is True
    roles = {m["role"] for m in variants["modes"]}
    assert "production" in roles and "reference_floor" in roles


# ---------------------------------------------------------------------------
# Decision tree: every branch
# ---------------------------------------------------------------------------


def _prov(closed=True):
    return {"provenance_closed": closed}


def _split(calibrated):
    return {
        "per_pair": {
            "(0,1)": {"gate_verdict": {"calibrated": calibrated}},
            "(0,2)": {"gate_verdict": {"calibrated": calibrated}},
            "(0,3)": {"gate_verdict": {"calibrated": calibrated}},
        }
    }


def test_decision_provenance_unresolved(real_config):
    d = pcc.decide(real_config, _prov(False), _split(True), _split(True), True)
    assert d["decision"] == DECISION_PROVENANCE_UNRESOLVED
    assert d["propagated_covariance_model_validated"] is False
    assert d["real_data_alignment_authorized"] is False


def test_decision_truth_closure_unavailable(real_config):
    d = pcc.decide(real_config, _prov(True), _split(True), _split(True), False)
    assert d["decision"] == DECISION_TRUTH_CLOSURE_UNAVAILABLE
    assert d["held_out_accessed"] is False


def test_decision_validated_canonical_keeps_alignment_forbidden(real_config):
    d = pcc.decide(real_config, _prov(True), _split(True), _split(True), True)
    assert d["decision"] == DECISION_VALIDATED_CANONICAL
    assert d["propagated_covariance_model_validated"] is True
    # Even when validated, the independent J-support gate keeps alignment off.
    assert d["real_kinematic_jacobian_support_validated"] is False
    assert d["measurement_model_validated"] is False
    assert d["real_data_alignment_authorized"] is False
    assert d["real_data_alignment_v2_preregistration_allowed"] is False


def test_decision_inconclusive(real_config):
    d = pcc.decide(real_config, _prov(True), _split(True), _split(False), True)
    assert d["decision"] == DECISION_INCONCLUSIVE
    assert d["propagated_covariance_model_validated"] is False


def test_decision_not_calibrated_with_classification(real_config):
    variants = {
        "modes": {
            "3": {
                "splits": {
                    "construction": {
                        "per_pair": {
                            "(0,1)": {"pencil": {"variance_ratio_prop_over_emp": 0.1}},
                            "(0,2)": {"pencil": {"variance_ratio_prop_over_emp": 0.1}},
                            "(0,3)": {"pencil": {"variance_ratio_prop_over_emp": 0.1}},
                        }
                    }
                }
            }
        }
    }
    construction = {
        "per_pair": {
            "(0,1)": {"gate_verdict": {"calibrated": False},
                      "pencil": {"variance_ratio_prop_over_emp": 300.0}},
            "(0,2)": {"gate_verdict": {"calibrated": False},
                      "pencil": {"variance_ratio_prop_over_emp": 1200.0}},
            "(0,3)": {"gate_verdict": {"calibrated": False},
                      "pencil": {"variance_ratio_prop_over_emp": 2400.0}},
        }
    }
    validation = {
        "per_pair": {
            "(0,1)": {"gate_verdict": {"calibrated": False},
                      "pencil": {"variance_ratio_prop_over_emp": 2500.0}},
            "(0,2)": {"gate_verdict": {"calibrated": False},
                      "pencil": {"variance_ratio_prop_over_emp": 2200.0}},
            "(0,3)": {"gate_verdict": {"calibrated": False},
                      "pencil": {"variance_ratio_prop_over_emp": 3900.0}},
        }
    }
    d = pcc.decide(
        real_config, _prov(True), construction, validation, True, variants=variants
    )
    assert d["decision"] == DECISION_NOT_CALIBRATED
    fc = d["failure_classification"]
    assert fc["category"] == MECHANISM_OVERESTIMATED_TRANSPORT
    assert fc["patched_in_this_campaign"] is False
    assert fc["repair_campaign"] == "propagated_covariance_upstream_repair_mc_validation_v1"
    assert d["real_data_alignment_authorized"] is False
