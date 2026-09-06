"""Tests for Workbook 84: SegmentFit Covariance Coordinate Contract Audit V1.

All judgments are based on a code audit and a synthetic covariance closure, NOT on
alignment residuals or real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from alignment import segmentfit_covariance_coordinate_contract as sccc
from alignment.segmentfit_covariance_coordinate_contract import ConfigError

CONFIG_PATH = "configs/segmentfit_covariance_coordinate_contract_audit_v1.yaml"


@pytest.fixture(scope="module")
def real_config():
    return sccc.load_config(CONFIG_PATH)


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
    assert (
        real_config["schema_version"]
        == "segmentfit-covariance-coordinate-contract-audit-v1"
    )
    assert int(real_config["workbook"]) == 84


@pytest.mark.parametrize(
    "flag",
    (
        "geometry_write_allowed",
        "official_conditions_write_allowed",
        "real_data_candidate_alignment_authorized",
        "real_data_alignment_authorized",
        "measurement_model_validated",
        "held_out_accessed",
    ),
)
def test_frozen_permission_flags_must_be_false(tmp_path, flag):
    bad = _tampered_config(tmp_path, lambda c: c.__setitem__(flag, True))
    with pytest.raises(ConfigError):
        sccc.load_config(bad)


@pytest.mark.parametrize(
    "flag",
    (
        "do_not_modify_covariance",
        "do_not_add_scale_factors",
        "do_not_tune_parameters_to_chi2",
        "do_not_enter_alignment",
        "do_not_read_real_data_residuals",
        "do_not_write_geometry_or_conditions_payload",
    ),
)
def test_frozen_prohibitions_must_be_true(tmp_path, flag):
    bad = _tampered_config(tmp_path, lambda c: c.__setitem__(flag, False))
    with pytest.raises(ConfigError):
        sccc.load_config(bad)


@pytest.mark.parametrize("parent", ("workbook_81", "workbook_82", "workbook_83"))
def test_inheritance_sha_tamper_fails(tmp_path, parent):
    def mutate(c):
        c["inheritance"][f"{parent}_config_sha256"] = "0" * 64

    bad = _tampered_config(tmp_path, mutate)
    with pytest.raises(ConfigError):
        sccc.load_config(bad)


@pytest.mark.parametrize("parent", ("workbook_81", "workbook_82", "workbook_83"))
def test_inheritance_artifact_sha_tamper_fails(tmp_path, parent):
    def mutate(c):
        arts = c["inheritance"][f"{parent}_artifact_sha256"]
        first = next(iter(arts))
        arts[first] = "0" * 64

    bad = _tampered_config(tmp_path, mutate)
    with pytest.raises(ConfigError):
        sccc.load_config(bad)


def test_wb83_frozen_decision_tamper_fails(tmp_path):
    bad = _tampered_config(
        tmp_path,
        lambda c: c["inheritance"].__setitem__(
            "workbook_83_frozen_decision", "wrong"
        ),
    )
    with pytest.raises(ConfigError):
        sccc.load_config(bad)


def test_wb83_frozen_mechanism_tamper_fails(tmp_path):
    bad = _tampered_config(
        tmp_path,
        lambda c: c["inheritance"].__setitem__(
            "workbook_83_frozen_mechanism", "wrong"
        ),
    )
    with pytest.raises(ConfigError):
        sccc.load_config(bad)


# ---------------------------------------------------------------------------
# Curvilinear frame (reimplements Athena CurvilinearUVT)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tx,ty",
    [(0.0, 0.0006), (0.01, 0.0006), (0.02, 0.0004), (0.05, 0.001), (-0.03, -0.0005)],
)
def test_curvilinear_frame_maps_loc1_to_minus_y_loc2_to_plus_x(tx, ty):
    """For FASER beam tracks (|t.z| >= 0.99), loc1 -> -y and loc2 -> +x."""
    direction = np.array([tx, ty, 1.0])
    curv_u, curv_v, curv_t = sccc.curvilinear_uvt(direction)
    # beam tracks are along z
    assert abs(float(curv_t.dot(np.array([0.0, 0.0, 1.0])))) >= 0.99
    # loc1 (curvU) ~ -global y
    assert curv_u[1] < -0.9
    assert abs(curv_u[0]) < 0.1
    # loc2 (curvV) ~ +global x
    assert curv_v[0] > 0.9
    assert abs(curv_v[1]) < 0.1


def test_curvilinear_frame_orthonormal():
    direction = np.array([0.02, 0.0004, 1.0])
    curv_u, curv_v, curv_t = sccc.curvilinear_uvt(direction)
    assert np.isclose(np.linalg.norm(curv_u), 1.0)
    assert np.isclose(np.linalg.norm(curv_v), 1.0)
    assert np.isclose(float(curv_u.dot(curv_v)), 0.0, atol=1e-9)
    assert np.isclose(float(curv_u.dot(curv_t)), 0.0, atol=1e-9)
    assert np.isclose(float(curv_v.dot(curv_t)), 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# SegmentFit GetState analytic Jacobian (the sign error)
# ---------------------------------------------------------------------------


def test_segment_fit_jacobian_has_dphi_dtx_sign_error():
    """The code-as-written Jacobian uses d phi/d tx = +ty/r^2 (should be -ty/r^2)."""
    tx, ty = 0.02, 0.0004
    r2 = tx * tx + ty * ty
    j_code = sccc.segment_fit_jacobian(tx, ty, sign_error=True)
    j_corr = sccc.segment_fit_jacobian(tx, ty, sign_error=False)
    # code-as-written
    assert np.isclose(j_code[2, 2], +ty / r2)
    # mathematically correct for phi = atan2(ty, tx)
    assert np.isclose(j_corr[2, 2], -ty / r2)
    # d phi/d ty is unaffected
    assert np.isclose(j_code[2, 3], tx / r2)
    assert np.isclose(j_corr[2, 3], tx / r2)


def test_segment_fit_jacobian_matches_numerical_dphi_dtx_when_corrected():
    """The corrected analytic d phi/d tx matches a numerical derivative of atan2."""
    tx, ty = 0.02, 0.0004
    eps = 1e-7
    num = (np.arctan2(ty, tx + eps) - np.arctan2(ty, tx - eps)) / (2 * eps)
    j_corr = sccc.segment_fit_jacobian(tx, ty, sign_error=False)
    assert np.isclose(j_corr[2, 2], num, rtol=1e-3)


# ---------------------------------------------------------------------------
# Synthetic covariance injection test (Cases A-E)
# ---------------------------------------------------------------------------


def _rep_frame(tx=0.02, ty=0.0004):
    phi = float(np.arctan2(ty, tx))
    theta = float(np.arctan(np.sqrt(tx * tx + ty * ty)))
    direction = np.array([tx, ty, 1.0])
    curv_u, curv_v, _ = sccc.curvilinear_uvt(direction)
    ref_pos = np.array([0.0, 0.0, 0.0])
    return phi, theta, ref_pos, curv_u, curv_v


def test_synthetic_case_a_loc1_lands_on_y():
    phi, theta, ref_pos, curv_u, curv_v = _rep_frame()
    cases = sccc.synthetic_injection_test(phi, theta, ref_pos, curv_u, curv_v)
    assert cases["A_loc1"]["dominant_output"] == "y"
    assert np.isclose(cases["A_loc1"]["output_variance"]["y"], 1.0, rtol=1e-3)
    assert np.isclose(cases["A_loc1"]["output_variance"]["x"], 0.0, atol=1e-6)


def test_synthetic_case_b_loc2_lands_on_x():
    phi, theta, ref_pos, curv_u, curv_v = _rep_frame()
    cases = sccc.synthetic_injection_test(phi, theta, ref_pos, curv_u, curv_v)
    assert cases["B_loc2"]["dominant_output"] == "x"
    assert np.isclose(cases["B_loc2"]["output_variance"]["x"], 1.0, rtol=1e-3)
    assert np.isclose(cases["B_loc2"]["output_variance"]["y"], 0.0, atol=1e-6)


def test_synthetic_case_e_qoverp_is_dropped():
    phi, theta, ref_pos, curv_u, curv_v = _rep_frame()
    cases = sccc.synthetic_injection_test(phi, theta, ref_pos, curv_u, curv_v)
    assert cases["E_qoverp"]["dominant_output"] is None
    assert all(v == 0.0 for v in cases["E_qoverp"]["output_variance"].values())


def test_synthetic_case_c_phi_matches_analytic_slope_jacobian():
    """Exporter d(tx, ty)/d(phi) = (-ty, tx).  This is NOT a mapping error."""
    tx, ty = 0.02, 0.0004
    phi, theta, ref_pos, curv_u, curv_v = _rep_frame(tx, ty)
    cases = sccc.synthetic_injection_test(phi, theta, ref_pos, curv_u, curv_v)
    assert np.isclose(cases["C_phi"]["jacobian_column"]["tx"], -ty, rtol=1e-3)
    assert np.isclose(cases["C_phi"]["jacobian_column"]["ty"], tx, rtol=1e-3)
    assert np.isclose(cases["C_phi"]["jacobian_column"]["x"], 0.0, atol=1e-6)
    assert np.isclose(cases["C_phi"]["jacobian_column"]["y"], 0.0, atol=1e-6)


def test_synthetic_case_d_theta_matches_analytic_slope_jacobian():
    """Exporter d(tx, ty)/d(theta) = (1+r^2) * (tx, ty) / r."""
    tx, ty = 0.02, 0.0004
    r = np.sqrt(tx * tx + ty * ty)
    scale = (1.0 + r * r) / r
    phi, theta, ref_pos, curv_u, curv_v = _rep_frame(tx, ty)
    cases = sccc.synthetic_injection_test(phi, theta, ref_pos, curv_u, curv_v)
    assert np.isclose(cases["D_theta"]["jacobian_column"]["tx"], scale * tx, rtol=1e-3)
    assert np.isclose(cases["D_theta"]["jacobian_column"]["ty"], scale * ty, rtol=1e-3)


def test_transform_matrices_record_sign_error_and_exporter_swap():
    ref_pos = np.array([0.0, 0.0, 0.0])
    mats = sccc.transform_matrices(0.02, 0.0004, ref_pos)
    r2 = 0.02**2 + 0.0004**2
    assert np.isclose(mats["dphi_dtx_code"], +0.0004 / r2)
    assert np.isclose(mats["dphi_dtx_corrected"], -0.0004 / r2)
    j2 = np.array(mats["j_exporter"])
    # loc1 column -> -y; loc2 column -> +x
    assert j2[1, 0] < -0.9
    assert j2[0, 1] > 0.9


# ---------------------------------------------------------------------------
# Full transform: the position swap
# ---------------------------------------------------------------------------


def test_full_transform_swaps_position():
    """A fit covariance with var(x) >> var(y) is exported as cov_yy >> cov_xx."""
    phi, theta, ref_pos, curv_u, curv_v = _rep_frame()
    fit_cov = np.diag([0.25, 1e-4, 4e-4, 1.6e-7])  # var(x) large, var(y) small
    c_global, _, _ = sccc.full_transform(
        fit_cov, 0.02, 0.0004, ref_pos, sign_error=True
    )
    # the swap: exported cov_xx ~ var(y), exported cov_yy ~ var(x)
    assert c_global[0, 0] < c_global[1, 1]
    assert np.isclose(c_global[0, 0], 1e-4, rtol=1e-2)
    assert np.isclose(c_global[1, 1], 0.25, rtol=1e-2)


def test_full_transform_round_trip_recovers_fit_covariance():
    """Inverting the actual transform recovers the injected fit covariance."""
    phi, theta, ref_pos, curv_u, curv_v = _rep_frame()
    fit_cov = np.diag([0.25, 1e-4, 4e-4, 1.6e-7])
    c_global, _, _ = sccc.full_transform(
        fit_cov, 0.02, 0.0004, ref_pos, sign_error=True
    )
    # reconstruct via the actual transform
    pop = {
        "tx": np.array([0.02]),
        "ty": np.array([0.0004]),
        "cov_xx_mm2": np.array([c_global[0, 0]]),
        "cov_xy_mm2": np.array([c_global[0, 1]]),
        "cov_xtx_mm": np.array([c_global[0, 2]]),
        "cov_xty_mm": np.array([c_global[0, 3]]),
        "cov_yy_mm2": np.array([c_global[1, 1]]),
        "cov_ytx_mm": np.array([c_global[1, 2]]),
        "cov_yty_mm": np.array([c_global[1, 3]]),
        "cov_txtx": np.array([c_global[2, 2]]),
        "cov_txty": np.array([c_global[2, 3]]),
        "cov_tyty": np.array([c_global[3, 3]]),
    }
    c_fit, _ = sccc.reconstruct_fit_covariance(pop, 0, ref_pos, sign_error=True)
    assert c_fit is not None
    assert np.allclose(c_fit, fit_cov, rtol=1e-3, atol=1e-12)


def test_sign_error_causes_direction_dependent_cov_tyty():
    """The sign error couples var(tx) into var(ty) in a direction-dependent way."""
    ref_pos = np.array([0.0, 0.0, 0.0])
    # direction-independent fit covariance (stereo geometry: var_ty = var_tx / alpha^2)
    results = {}
    for tx in (0.001, 0.01, 0.03):
        ty = 0.0006
        var_tx = 4e-4
        fit_cov = np.diag([0.25, 1e-4, var_tx, var_tx * 0.0004])
        cg_code, _, _ = sccc.full_transform(fit_cov, tx, ty, ref_pos, sign_error=True)
        cg_corr, _, _ = sccc.full_transform(fit_cov, tx, ty, ref_pos, sign_error=False)
        results[tx] = (cg_code[3, 3], cg_corr[3, 3])
    # the corrected Jacobian gives a direction-independent cov_tyty
    corr_vals = np.array([v[1] for v in results.values()])
    assert np.allclose(corr_vals, corr_vals[0], rtol=0.2)
    # the code-as-written Jacobian gives a direction-dependent cov_tyty
    code_vals = np.array([v[0] for v in results.values()])
    assert code_vals.max() / code_vals.min() > 2.0
