"""T11 mathematical transport contract.  No Stage B rerun, no rescale."""

from __future__ import annotations

import numpy as np
import pytest

from alignment.faseracts_propagated_covariance_validation_v2 import (
    geometric_transport_jacobian,
)
from datasets.transport_contract import (
    MEV_PER_GEV,
    WB87_DECISION,
    TransportContractError,
    geometric_jacobian,
    inherit_wb87,
    load_config,
    numerical_jacobian,
    q_over_p_per_gev_from_mev,
    q_over_p_per_mev_from_gev,
    refuse_covariance_rescale,
    signed_q_over_p,
    slope_from_direction,
    transport_covariance,
)


def test_config_freezes_wb87_and_forbids_rescale():
    config = load_config()
    assert config["do_not_rescale_covariance"] is True
    assert config["do_not_use_truth_q_over_p_as_deployment_seed"] is True
    assert config["eighteen_source_jobs_submitted"] is False
    inherited = inherit_wb87(config)
    assert inherited["decision"] == WB87_DECISION
    assert inherited["physical_transport_fd_rerun"] is False


def test_diagonal_and_nondiagonal_transport():
    lever = 2000.0
    jac = geometric_jacobian(lever)
    assert np.allclose(jac, geometric_transport_jacobian(lever))
    diag = np.diag([4.0, 9.0, 1.0e-6, 4.0e-6])
    out = transport_covariance(diag, jac)
    assert out[0, 0] == pytest.approx(4.0 + (lever**2) * 1.0e-6)
    assert out[1, 1] == pytest.approx(9.0 + (lever**2) * 4.0e-6)
    dense = np.array(
        [
            [4.0, 0.2, 1.0e-4, 0.0],
            [0.2, 9.0, 0.0, 2.0e-4],
            [1.0e-4, 0.0, 1.0e-6, 1.0e-8],
            [0.0, 2.0e-4, 1.0e-8, 4.0e-6],
        ]
    )
    dense = 0.5 * (dense + dense.T)
    transported = transport_covariance(dense, jac)
    expected = jac @ dense @ jac.T
    rel = np.linalg.norm(transported - expected) / np.linalg.norm(expected)
    assert rel <= 1.0e-6


def test_nonzero_slopes_and_signed_q_over_p():
    state = np.array([10.0, -4.0, 0.012, -0.008], dtype=np.float64)
    lever = 1500.0
    analytic = geometric_jacobian(lever)
    numeric = numerical_jacobian(state, lever, 1.0e-4)
    rel = np.linalg.norm(numeric - analytic) / np.linalg.norm(analytic)
    assert rel <= 1.0e-6
    assert signed_q_over_p(1.0, 1.0e5) > 0.0
    assert signed_q_over_p(-1.0, 1.0e5) < 0.0
    assert signed_q_over_p(-1.0, 1.0e5) == pytest.approx(-1.0 / 1.0e5)


def test_q_over_p_mev_gev_units():
    per_gev = 0.01
    per_mev = q_over_p_per_mev_from_gev(per_gev)
    assert per_mev == pytest.approx(per_gev / MEV_PER_GEV)
    assert q_over_p_per_gev_from_mev(per_mev) == pytest.approx(per_gev)


def test_angular_to_slope():
    tx, ty = slope_from_direction(np.array([0.03, -0.04, 1.0]))
    assert tx == pytest.approx(0.03)
    assert ty == pytest.approx(-0.04)


def test_refuse_rescale_and_non_spd_source():
    with pytest.raises(TransportContractError, match="rescale"):
        refuse_covariance_rescale()
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(Exception):
        transport_covariance(bad, np.eye(2))
