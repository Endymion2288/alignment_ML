"""Tests for Workbook 85: GetState Covariance Transform Repair Validation V1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from alignment import segmentfit_getstate_repair_validation as sgrv
from alignment import source_tracklet_covariance_closure as stcc
from alignment.segmentfit_getstate_repair_validation import (
    ConfigError,
    DECISION_INSUFFICIENT,
    DECISION_REPAIRED_VALIDATED,
    FROZEN_WB83_GATES,
)

CONFIG_PATH = "configs/segmentfit_getstate_covariance_transform_repair_validation_v1.yaml"
WB83_CONFIG_PATH = "configs/propagated_covariance_upstream_repair_mc_validation_v1.yaml"


@pytest.fixture(scope="module")
def real_config():
    return sgrv.load_config(CONFIG_PATH)


def _tampered_config(tmp_path: Path, mutate) -> str:
    raw = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "tampered.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(path)


def test_config_loads_with_correct_schema(real_config):
    assert real_config["schema_version"] == (
        "segmentfit-getstate-covariance-transform-repair-validation-v1"
    )
    assert int(real_config["workbook"]) == 85


def test_gates_identical_to_frozen_wb83(real_config):
    wb83 = yaml.safe_load(Path(WB83_CONFIG_PATH).read_text(encoding="utf-8"))
    assert real_config["source_closure_gates"] == wb83["source_closure_gates"]
    assert real_config["source_closure_gates"] == FROZEN_WB83_GATES


def test_mc_split_identical_to_wb83_and_disjoint(real_config):
    wb83 = yaml.safe_load(Path(WB83_CONFIG_PATH).read_text(encoding="utf-8"))
    assert (
        real_config["mc_data"]["construction_source_ids"]
        == wb83["mc_data"]["construction_source_ids"]
    )
    assert (
        real_config["mc_data"]["validation_source_ids"]
        == wb83["mc_data"]["validation_source_ids"]
    )
    assert not (
        set(real_config["mc_data"]["construction_source_ids"])
        & set(real_config["mc_data"]["validation_source_ids"])
    )


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
        sgrv.load_config(bad)


@pytest.mark.parametrize(
    "flag",
    (
        "do_not_add_scale_factors",
        "do_not_tune_parameters_to_chi2",
        "do_not_enter_alignment",
        "do_not_read_real_data_residuals",
        "do_not_write_geometry_or_conditions_payload",
        "do_not_enter_faseracts_propagation",
        "do_not_enter_stage_b",
        "do_not_change_central_prediction_in_covariance_repair",
    ),
)
def test_frozen_prohibitions_must_be_true(tmp_path, flag):
    bad = _tampered_config(tmp_path, lambda c: c.__setitem__(flag, False))
    with pytest.raises(ConfigError):
        sgrv.load_config(bad)


def test_gate_tamper_fails(tmp_path):
    bad = _tampered_config(
        tmp_path,
        lambda c: c["source_closure_gates"].__setitem__("whitened_chi2_per_ndof_max", 99.0),
    )
    with pytest.raises(ConfigError):
        sgrv.load_config(bad)


@pytest.mark.parametrize("parent", ("workbook_81", "workbook_82", "workbook_83", "workbook_84"))
def test_inheritance_sha_tamper_fails(tmp_path, parent):
    def mutate(c):
        c["inheritance"][f"{parent}_config_sha256"] = "0" * 64

    bad = _tampered_config(tmp_path, mutate)
    with pytest.raises(ConfigError):
        sgrv.load_config(bad)


def test_wb84_frozen_decision_tamper_fails(tmp_path):
    bad = _tampered_config(
        tmp_path,
        lambda c: c["inheritance"].__setitem__("workbook_84_frozen_decision", "wrong"),
    )
    with pytest.raises(ConfigError):
        sgrv.load_config(bad)


def test_curvilinear_slot_map_is_minus_y_plus_x():
    p = sgrv.curvilinear_slot_map()
    # loc1 = -y, loc2 = +x
    assert np.allclose(p @ np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0, 0.0]))
    assert np.allclose(p @ np.array([0.0, 1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0, 0.0]))


def test_repair_recovers_injected_fit_covariance():
    """A known C_fit, exported through the buggy transform, is recovered by the repair."""
    from alignment import segmentfit_covariance_coordinate_contract as sccc

    ref = np.zeros(3)
    tx, ty = 0.02, 0.0004
    fit = np.diag([0.25, 1e-4, 4e-4, 1.6e-7])
    c_buggy, _, _ = sccc.full_transform(fit, tx, ty, ref, sign_error=True)
    # buggy export swaps position
    assert c_buggy[0, 0] < c_buggy[1, 1]
    repaired = sgrv.repair_exported_covariance(c_buggy, tx, ty, ref_pos=ref)
    assert repaired is not None
    assert np.isclose(repaired[0, 0], fit[0, 0], rtol=1e-2)
    assert np.isclose(repaired[1, 1], fit[1, 1], rtol=1e-2)
    assert np.isclose(repaired[2, 2], fit[2, 2], rtol=2e-2)
    assert np.isclose(repaired[3, 3], fit[3, 3], rtol=2e-2)
    assert repaired[0, 0] > repaired[1, 1]


def test_repair_introduces_no_scale_factor():
    """The repair is a similarity J C J^T of a recovered C_fit; no extra scale."""
    from alignment import segmentfit_covariance_coordinate_contract as sccc

    ref = np.zeros(3)
    tx, ty = 0.01, 0.0006
    fit = np.diag([0.25, 1e-4, 4e-4, 1.6e-7])
    c_buggy, _, _ = sccc.full_transform(fit, tx, ty, ref, sign_error=True)
    repaired = sgrv.repair_exported_covariance(c_buggy, tx, ty, ref_pos=ref)
    # re-applying the repair is idempotent up to inversion noise: second pass
    # would treat the already-repaired matrix as if it were buggy, so we only
    # check there is no global scale vs the known fit diagonal.
    ratio = np.diag(repaired) / np.diag(fit)
    assert np.allclose(ratio, 1.0, atol=0.05)


def test_decide_insufficient_when_repair_fails_gates():
    validation = {
        "construction": {
            "repaired_all_stations_calibrated": False,
            "baseline_all_stations_calibrated": False,
        },
        "validation": {
            "repaired_all_stations_calibrated": False,
            "baseline_all_stations_calibrated": False,
        },
        "source_disjoint_agreement": True,
        "whitening_closure_recovered": False,
    }
    d = sgrv.decide({}, validation)
    assert d["decision"] == DECISION_INSUFFICIENT
    assert d["measurement_model_validated"] is False
    assert d["measurement_model_v2_discussion_allowed"] is False


def test_decide_validated_keeps_alignment_closed():
    validation = {
        "construction": {
            "repaired_all_stations_calibrated": True,
            "baseline_all_stations_calibrated": False,
        },
        "validation": {
            "repaired_all_stations_calibrated": True,
            "baseline_all_stations_calibrated": False,
        },
        "source_disjoint_agreement": True,
        "whitening_closure_recovered": True,
    }
    d = sgrv.decide({}, validation)
    assert d["decision"] == DECISION_REPAIRED_VALIDATED
    assert d["measurement_model_validated"] is False
    assert d["geometry_write_allowed"] is False
    assert d["real_data_alignment_authorized"] is False
    assert d["frozen_v2_alignment_authorized"] is False
    assert d["measurement_model_v2_discussion_allowed"] is True


def test_wb83_evaluate_gates_helper_still_used():
    """WB85 must use the frozen WB83 gate function, not a reimplemented one."""
    assert sgrv.evaluate_source_closure_gates is stcc.evaluate_source_closure_gates
    assert sgrv.compute_source_closure_metrics is stcc.compute_source_closure_metrics
