"""Tests for Workbook 86: Full CurvilinearUVT Branch Covariance Contract Repair."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from alignment import full_curvilinear_uvt_repair_validation as fcv
from alignment import segmentfit_covariance_coordinate_contract as sccc
from alignment import segmentfit_getstate_repair_validation as sgrv
from alignment.full_curvilinear_uvt_repair_validation import (
    ConfigError,
    DECISION_NOT_CLOSED,
    DECISION_VALIDATED,
)
from alignment.segmentfit_getstate_repair_validation import FROZEN_WB83_GATES

CONFIG_PATH = "configs/full_curvilinear_uvt_covariance_contract_repair_validation_v1.yaml"
WB83_CONFIG_PATH = "configs/propagated_covariance_upstream_repair_mc_validation_v1.yaml"


@pytest.fixture(scope="module")
def real_config():
    return fcv.load_config(CONFIG_PATH)


def _tampered_config(tmp_path: Path, mutate) -> str:
    raw = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "tampered.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(path)


def test_config_loads_with_correct_schema(real_config):
    assert real_config["schema_version"] == (
        "full-curvilinear-uvt-covariance-contract-repair-validation-v1"
    )
    assert int(real_config["workbook"]) == 86


def test_gates_identical_to_frozen_wb83(real_config):
    wb83 = yaml.safe_load(Path(WB83_CONFIG_PATH).read_text(encoding="utf-8"))
    assert real_config["source_closure_gates"] == wb83["source_closure_gates"]
    assert real_config["source_closure_gates"] == FROZEN_WB83_GATES


@pytest.mark.parametrize(
    "flag",
    (
        "geometry_write_allowed",
        "real_data_alignment_authorized",
        "measurement_model_validated",
        "held_out_accessed",
    ),
)
def test_frozen_permission_flags_must_be_false(tmp_path, flag):
    bad = _tampered_config(tmp_path, lambda c: c.__setitem__(flag, True))
    with pytest.raises(ConfigError):
        fcv.load_config(bad)


@pytest.mark.parametrize(
    "flag",
    (
        "do_not_add_scale_factors",
        "do_not_tune_parameters_to_chi2",
        "do_not_reject_by_angle",
        "do_not_drop_outliers",
        "do_not_enter_stage_b",
        "do_not_enter_faseracts_propagation",
    ),
)
def test_frozen_prohibitions_must_be_true(tmp_path, flag):
    bad = _tampered_config(tmp_path, lambda c: c.__setitem__(flag, False))
    with pytest.raises(ConfigError):
        fcv.load_config(bad)


def test_gate_tamper_fails(tmp_path):
    bad = _tampered_config(
        tmp_path,
        lambda c: c["source_closure_gates"].__setitem__("whitened_chi2_per_ndof_max", 99.0),
    )
    with pytest.raises(ConfigError):
        fcv.load_config(bad)


@pytest.mark.parametrize("parent", ("workbook_81", "workbook_85"))
def test_inheritance_sha_tamper_fails(tmp_path, parent):
    def mutate(c):
        c["inheritance"][f"{parent}_config_sha256"] = "0" * 64

    bad = _tampered_config(tmp_path, mutate)
    with pytest.raises(ConfigError):
        fcv.load_config(bad)


def test_wb85_remaining_structure_tamper_fails(tmp_path):
    bad = _tampered_config(
        tmp_path,
        lambda c: c["inheritance"].__setitem__("workbook_85_frozen_mechanism", "wrong"),
    )
    with pytest.raises(ConfigError):
        fcv.load_config(bad)


def test_full_slot_map_is_near_beam_map_on_beam_branch():
    u, v, t = sccc.curvilinear_uvt(np.array([0.02, 0.0004, 1.0]))
    assert abs(float(t[2])) >= 0.99
    p_full = fcv.full_curvilinear_slot_map(u, v)
    # beam-track leading structure: loc1 ≈ -y, loc2 ≈ +x
    assert p_full[0, 1] < -0.99
    assert p_full[1, 0] > 0.99
    assert abs(p_full[0, 0]) < 0.05
    assert abs(p_full[1, 1]) < 0.05


def test_full_slot_map_differs_on_large_angle_branch():
    # |t.z| = 1/sqrt(1+0.2^2) ≈ 0.980 < 0.99
    u, v, t = sccc.curvilinear_uvt(np.array([0.2, 0.0, 1.0]))
    assert abs(float(t[2])) < 0.99
    p_full = fcv.full_curvilinear_slot_map(u, v)
    p_beam = sgrv.curvilinear_slot_map()
    assert not np.allclose(p_full[:2, :2], p_beam[:2, :2], atol=1e-3)


def test_full_slot_map_inverts_curvilinear_xy():
    u, v, _ = sccc.curvilinear_uvt(np.array([0.18, 0.05, 1.0]))
    p = fcv.full_curvilinear_slot_map(u, v)
    xy = np.array([1.5, -0.7])
    loc = p[:2, :2] @ xy
    recovered = loc[0] * u[:2] + loc[1] * v[:2]
    assert np.allclose(recovered, xy, atol=1e-9)


def test_full_repair_recovers_fit_on_large_angle_where_beam_map_fails():
    ref = np.zeros(3)
    tx, ty = 0.18, 0.04  # |t.z| < 0.99
    fit = np.diag([0.25, 1e-4, 4e-4, 1.6e-7])
    c_buggy, _, _ = sccc.full_transform(fit, tx, ty, ref, sign_error=True)
    beam = sgrv.repair_exported_covariance(c_buggy, tx, ty, ref_pos=ref)
    full = fcv.repair_exported_covariance_full(c_buggy, tx, ty, ref_pos=ref)
    assert beam is not None and full is not None
    # beam approximation leaves a large C_yy error on this branch
    assert abs(beam[1, 1] / fit[1, 1] - 1.0) > 0.2
    assert np.isclose(full[0, 0], fit[0, 0], rtol=5e-2)
    assert np.isclose(full[1, 1], fit[1, 1], rtol=5e-2)
    assert np.isclose(full[3, 3], fit[3, 3], rtol=5e-2)


def test_per_track_diagnostic_reports_all_three_arms():
    recs = sgrv.RepairRecords(
        station_id=0,
        e_source=np.zeros((1, 4)),
        c_source=np.array([np.diag([0.2, 0.1, 1e-4, 1e-6])]),
        tx=np.array([0.18]),
        ty=np.array([0.04]),
        abs_tx=np.array([0.18]),
        abs_ty=np.array([0.04]),
        abs_q_over_p=np.array([1e-5]),
        truth_pdg=np.array([13]),
        n_matched=1,
    )
    c_map = {
        "baseline": recs.c_source,
        "wb85_beam": recs.c_source * 0.5,
        "wb86_full": np.array([np.diag([0.2, 1e-4, 1e-4, 1e-7])]),
    }
    dump = fcv.per_track_covariance_diagnostic(recs, c_map)
    assert len(dump) == 1
    assert dump[0]["abs_t_dot_z"] < 0.99
    assert dump[0]["Cyy"]["baseline"] == pytest.approx(0.1)
    assert dump[0]["Cyy"]["wb86_full"] == pytest.approx(1e-4)


def test_full_repair_recovers_fit_on_beam_track():
    ref = np.zeros(3)
    tx, ty = 0.02, 0.0004
    fit = np.diag([0.25, 1e-4, 4e-4, 1.6e-7])
    c_buggy, _, _ = sccc.full_transform(fit, tx, ty, ref, sign_error=True)
    full = fcv.repair_exported_covariance_full(c_buggy, tx, ty, ref_pos=ref)
    assert full is not None
    assert np.allclose(np.diag(full), np.diag(fit), rtol=2e-2)


def test_decide_validated_keeps_alignment_closed():
    validation = {
        "construction": {
            "wb86_full_all_stations_calibrated": True,
            "wb85_beam_all_stations_calibrated": False,
        },
        "validation": {
            "wb86_full_all_stations_calibrated": True,
            "wb85_beam_all_stations_calibrated": False,
        },
        "source_disjoint_agreement": True,
        "whitening_closure_recovered": True,
    }
    d = fcv.decide({}, validation)
    assert d["decision"] == DECISION_VALIDATED
    assert d["measurement_model_validated"] is False
    assert d["geometry_write_allowed"] is False
    assert d["real_data_alignment_authorized"] is False
    assert d["propagated_covariance_validation_authorized"] is True
    assert d["stage_b_entered"] is False


def test_decide_not_closed_when_full_repair_fails():
    validation = {
        "construction": {
            "wb86_full_all_stations_calibrated": False,
            "wb85_beam_all_stations_calibrated": False,
        },
        "validation": {
            "wb86_full_all_stations_calibrated": False,
            "wb85_beam_all_stations_calibrated": False,
        },
        "source_disjoint_agreement": True,
        "whitening_closure_recovered": False,
    }
    d = fcv.decide({}, validation)
    assert d["decision"] == DECISION_NOT_CLOSED
    assert d["propagated_covariance_validation_authorized"] is False
    assert d["measurement_model_v2_discussion_allowed"] is False
