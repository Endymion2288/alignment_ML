"""Workbook-83 Stage A source-tracklet covariance truth closure tests.

Covers the pre-registered config freeze (flags, WB81/WB82 inheritance SHAs, MC
source-disjointness, prohibitions), the structural held-out / real-data /
geometry-write guards, the exact [x,y,tx,ty] e_source / C_source extraction from
synthetic ROOT ntuples (including the straight-line z-correction and the
truth-match / physical-acceptance filters), the no-tracklet-dropping contract,
the deterministic symmetric whitening, the position-swap diagnostic, the
pre-registered repair derivation/application, and every Stage-A decision-tree
branch.
"""

from __future__ import annotations

from pathlib import Path

import awkward as ak
import numpy as np
import pytest
import uproot
import yaml

from alignment import source_tracklet_covariance_closure as stcc
from alignment.source_tracklet_covariance_closure import (
    DECISION_REPAIR_INCONCLUSIVE,
    DECISION_SOURCE_NOT_CALIBRATABLE,
    ConfigError,
)

CONFIG_PATH = "configs/propagated_covariance_upstream_repair_mc_validation_v1.yaml"


@pytest.fixture(scope="module")
def real_config():
    return stcc.load_config(CONFIG_PATH)


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
    assert real_config["schema_version"] == "propagated-covariance-upstream-repair-mc-validation-v1"
    assert int(real_config["workbook"]) == 83


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
        stcc.load_config(bad)


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
        "do_not_promote_mode3_to_production",
        "do_not_use_truth_qoverp_for_real_data_solution",
        "do_not_reverse_optimize_source_covariance_from_target",
        "do_not_change_central_prediction_in_covariance_repair",
    ),
)
def test_frozen_prohibitions_must_be_true(tmp_path, flag):
    bad = _tampered_config(tmp_path, lambda c: c.__setitem__(flag, False))
    with pytest.raises(ConfigError):
        stcc.load_config(bad)


def test_wb81_inheritance_sha_tamper_fails(tmp_path):
    bad = _tampered_config(
        tmp_path,
        lambda c: c["inheritance"].__setitem__("workbook_81_config_sha256", "0" * 64),
    )
    with pytest.raises(ConfigError):
        stcc.load_config(bad)


def test_wb82_inheritance_sha_tamper_fails(tmp_path):
    bad = _tampered_config(
        tmp_path,
        lambda c: c["inheritance"].__setitem__("workbook_82_config_sha256", "0" * 64),
    )
    with pytest.raises(ConfigError):
        stcc.load_config(bad)


def test_wb81_frozen_mechanism_tamper_fails(tmp_path):
    bad = _tampered_config(
        tmp_path,
        lambda c: c["inheritance"].__setitem__("workbook_81_frozen_mechanism", "wrong"),
    )
    with pytest.raises(ConfigError):
        stcc.load_config(bad)


def test_mc_split_must_be_source_disjoint(tmp_path, real_config):
    overlap = real_config["mc_data"]["construction_source_ids"][0]

    def mutate(c):
        c["mc_data"]["validation_source_ids"] = [overlap]

    bad = _tampered_config(tmp_path, mutate)
    with pytest.raises(ConfigError):
        stcc.load_config(bad)


def test_real_config_mc_split_is_disjoint(real_config):
    mc = real_config["mc_data"]
    assert not (
        set(mc["construction_source_ids"]) & set(mc["validation_source_ids"])
    )


def test_held_out_guard_is_structural_noop():
    assert stcc.assert_no_held_out_access() is None


# ---------------------------------------------------------------------------
# Closure records / metrics on synthetic data
# ---------------------------------------------------------------------------


def _make_records(e_source: np.ndarray, c_source: np.ndarray):
    n = e_source.shape[0]
    return stcc.SourceClosureRecords(
        station_id=0,
        e_source=e_source,
        c_source=c_source,
        abs_tx=np.full(n, 0.01),
        abs_ty=np.full(n, 0.01),
        abs_q_over_p=np.full(n, 1e-6),
        truth_pdg=np.full(n, 13),
        n_matched=n,
    )


def _calibrated_records(n=300, seed=0):
    """e_source drawn from N(0, C0); c_source set to the empirical covariance."""
    rng = np.random.default_rng(seed)
    A = np.array(
        [
            [0.5, 0.01, 0.0, 0.0],
            [0.01, 0.02, 0.0, 0.0],
            [0.0, 0.0, 0.02, 0.001],
            [0.0, 0.0, 0.001, 0.0005],
        ]
    )
    C0 = A @ A.T
    E = rng.multivariate_normal(np.zeros(4), C0, size=n)
    E = E - E.mean(axis=0)
    C_emp = np.cov(E.T)
    c_source = np.repeat(C_emp[None, :, :], n, axis=0)
    return _make_records(E, c_source)


def test_metrics_calibrated_when_csource_matches_empirical(real_config):
    metrics = stcc.compute_source_closure_metrics(_calibrated_records(), real_config)
    assert metrics["sufficient"] is True
    np.testing.assert_allclose(metrics["cov_z_eigenvalues"], np.ones(4), atol=0.6)
    assert abs(metrics["chi2_per_ndof"] - 1.0) < 0.5
    verdict = stcc.evaluate_source_closure_gates(metrics, real_config)
    assert verdict["calibrated"] is True


def test_metrics_overestimated_csource_fails_gates(real_config):
    recs = _calibrated_records()
    recs = stcc.SourceClosureRecords(
        **{**recs.__dict__, "c_source": recs.c_source * 1000.0}
    )
    metrics = stcc.compute_source_closure_metrics(recs, real_config)
    assert max(metrics["generalized_eigenvalues_cemp_over_csource"]) < 0.01
    verdict = stcc.evaluate_source_closure_gates(metrics, real_config)
    assert verdict["calibrated"] is False


def test_metrics_insufficient_tracklets(real_config):
    recs = _make_records(np.zeros((3, 4)), np.repeat(np.eye(4)[None], 3, axis=0))
    metrics = stcc.compute_source_closure_metrics(recs, real_config)
    assert metrics["sufficient"] is False
    verdict = stcc.evaluate_source_closure_gates(metrics, real_config)
    assert verdict["calibrated"] is False
    assert verdict["reason"] == "insufficient_tracklets"


def test_position_swap_diagnostic_detects_swap(real_config):
    """A covariance with x<->y swapped relative to the empirical must be flagged."""
    rng = np.random.default_rng(7)
    n = 400
    # Empirical: x imprecise (0.5), y precise (0.02).
    E = np.zeros((n, 4))
    E[:, 0] = rng.normal(0.0, 0.5, n)
    E[:, 1] = rng.normal(0.0, 0.02, n)
    E[:, 2] = rng.normal(0.0, 0.02, n)
    E[:, 3] = rng.normal(0.0, 0.0005, n)
    E = E - E.mean(axis=0)
    # Written covariance: x/y position swapped (claims x precise, y imprecise).
    C = np.zeros((4, 4))
    C[0, 0] = 0.02 ** 2  # cov_xx = empirical y variance
    C[1, 1] = 0.5 ** 2   # cov_yy = empirical x variance
    C[2, 2] = 0.02 ** 2
    C[3, 3] = 0.0005 ** 2
    c_source = np.repeat(C[None, :, :], n, axis=0)
    recs = _make_records(E, c_source)
    metrics = stcc.compute_source_closure_metrics(recs, real_config)
    diag = metrics["position_swap_diagnostic"]
    # Cross-swap ratios ~ 1 confirm the position swap.
    assert diag["cross_swap_ratio_cxx_over_empyy"] == pytest.approx(1.0, rel=0.2)
    assert diag["cross_swap_ratio_cyy_over_empxx"] == pytest.approx(1.0, rel=0.2)
    # The swap must dramatically improve the chi2.
    assert diag["position_swap_chi2_improvement"] > 100.0
    assert diag["chi2_per_ndof_after_position_xy_swap"] < diag["chi2_per_ndof_as_is"]


def test_kinematic_dependence_binned(real_config):
    metrics = stcc.compute_source_closure_metrics(_calibrated_records(), real_config)
    dep = metrics["kinematic_dependence"]
    assert "abs_tx" in dep and "abs_ty" in dep and "abs_q_over_p" in dep
    assert sum(b["n"] for b in dep["abs_tx"]) == metrics["n_tracklets"]


# ---------------------------------------------------------------------------
# Exact e_source / C_source extraction from synthetic ROOT ntuples
# ---------------------------------------------------------------------------


def _write_synthetic_mc(refit: Path) -> dict:
    """Write a one-tracklet synthetic tracklets.root + enhanced ntuple.

    Station 0 truth: pos=(1, 2, 499), mom giving tx=0.005, ty=0.003.
    tracklet z_mm=500 -> dz=+1 straight-line correction.
    """
    refit.mkdir(parents=True, exist_ok=True)
    cov = {
        "cov_xx_mm2": 1.0, "cov_xy_mm2": 0.1, "cov_xtx_mm": 0.01, "cov_xty_mm": 0.0,
        "cov_yy_mm2": 4.0, "cov_ytx_mm": 0.0, "cov_yty_mm": 0.02,
        "cov_txtx": 1e-6, "cov_txty": 0.0, "cov_tyty": 4e-6,
    }
    trk = {
        "run_id": np.asarray([1], dtype=np.int32),
        "event_id": np.asarray([100], dtype=np.int64),
        "station_id": np.asarray([0], dtype=np.int16),
        "tracklet_id": np.asarray([5], dtype=np.int32),
        "x_mm": np.asarray([1.1], dtype=np.float64),
        "y_mm": np.asarray([2.2], dtype=np.float64),
        "z_mm": np.asarray([500.0], dtype=np.float64),
        "tx": np.asarray([0.0051], dtype=np.float64),
        "ty": np.asarray([0.0031], dtype=np.float64),
        "truth_particle_id": np.asarray([10001], dtype=np.int64),
        "truth_match_fraction": np.asarray([1.0], dtype=np.float64),
        "truth_pdg": np.asarray([13], dtype=np.int32),
        "q_over_p_per_mev": np.asarray([1e-6], dtype=np.float64),
    }
    for k, v in cov.items():
        trk[k] = np.asarray([v], dtype=np.float64)
    with uproot.recreate(refit / "tracklets.root") as f:
        f["tracklets"] = trk

    px, py, pz = 0.5, 0.3, 100.0  # tx=0.005, ty=0.003
    st_pos = {0: (1.0, 2.0, 499.0), 1: (0.0, 0.0, 1000.0),
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


def test_build_source_records_exact_extraction_and_ordering(tmp_path, real_config):
    cov = _write_synthetic_mc(tmp_path)
    recs = stcc.build_source_closure_records(
        tracklets_path=tmp_path / "tracklets.root",
        enhanced_path=tmp_path / "enhanced_tracklets.root",
        station_id=0,
        config=real_config,
    )
    assert recs.size == 1 and recs.n_matched == 1
    # Truth source state: straight-line correction dz = 500 - 499 = +1 mm.
    # x_truth = 1 + 0.005*1 = 1.005 ; y_truth = 2 + 0.003*1 = 2.003.
    np.testing.assert_allclose(
        recs.e_source[0],
        [1.1 - 1.005, 2.2 - 2.003, 0.0051 - 0.005, 0.0031 - 0.003],
        atol=1e-9,
    )
    expected_C = np.array(
        [
            [cov["cov_xx_mm2"], cov["cov_xy_mm2"], cov["cov_xtx_mm"], cov["cov_xty_mm"]],
            [cov["cov_xy_mm2"], cov["cov_yy_mm2"], cov["cov_ytx_mm"], cov["cov_yty_mm"]],
            [cov["cov_xtx_mm"], cov["cov_ytx_mm"], cov["cov_txtx"], cov["cov_txty"]],
            [cov["cov_xty_mm"], cov["cov_yty_mm"], cov["cov_txty"], cov["cov_tyty"]],
        ]
    )
    np.testing.assert_allclose(recs.c_source[0], expected_C, atol=1e-12)


def test_build_source_records_truth_match_filter(tmp_path, real_config):
    _write_synthetic_mc(tmp_path)
    # Raise the truth-match threshold above the tracklet's tmf=1.0 -> dropped.
    cfg = dict(real_config)
    cfg["truth_reference"] = {**real_config["truth_reference"], "min_truth_match_fraction": 1.01}
    recs = stcc.build_source_closure_records(
        tracklets_path=tmp_path / "tracklets.root",
        enhanced_path=tmp_path / "enhanced_tracklets.root",
        station_id=0,
        config=cfg,
    )
    assert recs.size == 0


def test_build_source_records_physical_acceptance_filter(tmp_path, real_config):
    _write_synthetic_mc(tmp_path)
    # Tighten the acceptance below the tracklet's |tx|=0.0051 -> dropped.
    cfg = dict(real_config)
    cfg["truth_reference"] = {
        **real_config["truth_reference"],
        "physical_acceptance_abs_tx_ty_max": 0.001,
    }
    recs = stcc.build_source_closure_records(
        tracklets_path=tmp_path / "tracklets.root",
        enhanced_path=tmp_path / "enhanced_tracklets.root",
        station_id=0,
        config=cfg,
    )
    assert recs.size == 0


def test_build_source_records_is_deterministic(tmp_path, real_config):
    _write_synthetic_mc(tmp_path)
    kwargs = dict(
        tracklets_path=tmp_path / "tracklets.root",
        enhanced_path=tmp_path / "enhanced_tracklets.root",
        station_id=0,
        config=real_config,
    )
    r1 = stcc.build_source_closure_records(**kwargs)
    r2 = stcc.build_source_closure_records(**kwargs)
    np.testing.assert_array_equal(r1.e_source, r2.e_source)
    np.testing.assert_array_equal(r1.c_source, r2.c_source)


# ---------------------------------------------------------------------------
# Repair derivation / application
# ---------------------------------------------------------------------------


def test_position_permutation_repair_application(real_config):
    recs = _calibrated_records()
    P = stcc._position_permutation()
    params = {"model": "position_permutation_xy_swap"}
    repaired = stcc.apply_repair(recs.c_source, 0, params)
    np.testing.assert_allclose(repaired[0], P @ recs.c_source[0] @ P, atol=1e-12)


def test_global_scale_repair_derivation_uses_construction_only(real_config):
    recs = _calibrated_records()
    params = stcc.derive_repair_parameters({0: recs}, "global_scale")
    assert params["model"] == "global_scale"
    # For a calibrated covariance the trace ratio is ~ 1.
    assert params["scale"] == pytest.approx(1.0, rel=0.2)


def test_unknown_repair_model_raises(real_config):
    with pytest.raises(ConfigError):
        stcc.derive_repair_parameters({0: _calibrated_records()}, "nonexistent_model")
    with pytest.raises(ConfigError):
        stcc.apply_repair(np.eye(4)[None], 0, {"model": "nonexistent_model"})


# ---------------------------------------------------------------------------
# Decision tree: every Stage-A branch
# ---------------------------------------------------------------------------


def _station_split(calibrated, with_swap_diag=True):
    diag = (
        {
            "chi2_per_ndof_as_is": 800.0,
            "chi2_per_ndof_after_position_xy_swap": 40.0,
            "chi2_per_ndof_median_as_is": 410.0,
            "chi2_per_ndof_median_after_position_xy_swap": 1.3,
            "fraction_chi2_per_ndof_above_4_after_position_xy_swap": 0.34,
            "cross_swap_ratio_cxx_over_empyy": 1.0,
            "cross_swap_ratio_cyy_over_empxx": 1.0,
        }
        if with_swap_diag
        else None
    )
    station = {
        "gate_verdict": {"calibrated": calibrated},
        "position_swap_diagnostic": diag,
        "marginal_variance_ratio_csource_over_emp": [0.0, 3500.0, 0.84, 91.0],
    }
    return {"per_station": {str(s): station for s in (0, 1, 2, 3)}}


def test_decision_both_splits_fail_no_repair_is_not_calibratable(real_config):
    d = stcc.decide_source_closure(
        real_config, _station_split(False), _station_split(False), repairs={}
    )
    assert d["decision"] == DECISION_SOURCE_NOT_CALIBRATABLE
    assert d["propagated_covariance_model_validated"] is False
    assert d["real_data_alignment_authorized"] is False
    assert d["held_out_accessed"] is False
    fc = d["failure_classification"]
    assert fc["category"] == "position_xy_swap_with_slope_miscalibration"
    assert fc["patched_in_this_campaign"] is False


def test_decision_split_disagreement_is_inconclusive(real_config):
    d = stcc.decide_source_closure(
        real_config, _station_split(True), _station_split(False), repairs={}
    )
    assert d["decision"] == DECISION_REPAIR_INCONCLUSIVE
    assert d["propagated_covariance_model_validated"] is False


def test_decision_validated_repair_stays_inconclusive_not_production(real_config):
    repairs = {"position_permutation_xy_swap": {"validated_source_disjoint": True}}
    d = stcc.decide_source_closure(
        real_config, _station_split(False), _station_split(False), repairs=repairs
    )
    # A portable repair is noted but Stage B is not entered in this campaign.
    assert d["decision"] == DECISION_REPAIR_INCONCLUSIVE
    assert "position_permutation_xy_swap" in d["stage_a_validated_repairs"]
    # Even a validated repair never enables alignment / held-out / production.
    assert d["propagated_covariance_model_validated"] is False
    assert d["real_data_alignment_authorized"] is False
    assert d["real_data_alignment_v2_preregistration_allowed"] is False


def test_decision_payload_keeps_all_permissions_false(real_config):
    d = stcc.decide_source_closure(
        real_config, _station_split(False), _station_split(False), repairs={}
    )
    for key in (
        "held_out_accessed",
        "real_data_alignment_authorized",
        "geometry_write_allowed",
        "official_conditions_write_allowed",
        "external_constraint_ingest_authorized",
        "propagated_covariance_model_validated",
        "measurement_model_validated",
        "real_data_alignment_v2_preregistration_allowed",
    ):
        assert d[key] is False
