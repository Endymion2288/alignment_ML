"""Tests for Workbook 87: FaserActs Propagated Covariance Validation V2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from alignment import faseracts_propagated_covariance_validation_v2 as wb87
from alignment.faseracts_propagated_covariance_validation_v2 import (
    ConfigError,
    DECISION_NOT_VALIDATED,
    DECISION_VALIDATED,
    FROZEN_WB81_GATES,
    MECHANISM_JACOBIAN_ERROR,
    MECHANISM_QOP_SEMANTICS,
)
from alignment.propagated_covariance_closure import ClosureRecords

CONFIG_PATH = "configs/faseracts_propagated_covariance_validation_v2.yaml"
WB81_CONFIG_PATH = "configs/faseracts_propagated_covariance_provenance_closure_v1.yaml"


@pytest.fixture(scope="module")
def real_config():
    return wb87.load_config(CONFIG_PATH)


def _tampered_config(tmp_path: Path, mutate) -> str:
    raw = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "tampered.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(path)


def test_config_loads_with_correct_schema(real_config):
    assert real_config["schema_version"] == "faseracts-propagated-covariance-validation-v2"
    assert int(real_config["workbook"]) == 87


def test_gates_identical_to_frozen_wb81(real_config):
    wb81 = yaml.safe_load(Path(WB81_CONFIG_PATH).read_text(encoding="utf-8"))
    assert real_config["closure_gates"] == wb81["closure_gates"]
    assert real_config["closure_gates"] == FROZEN_WB81_GATES


def test_same_source_split_as_wb83(real_config):
    wb83 = yaml.safe_load(
        Path("configs/propagated_covariance_upstream_repair_mc_validation_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert real_config["mc_data"]["construction_source_ids"] == wb83["mc_data"][
        "construction_source_ids"
    ]
    assert real_config["mc_data"]["validation_source_ids"] == wb83["mc_data"][
        "validation_source_ids"
    ]


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
        wb87.load_config(bad)


@pytest.mark.parametrize(
    "flag",
    (
        "do_not_tune_covariance_to_chi2",
        "do_not_delete_qoverp_column_as_final_scheme",
        "do_not_use_truth_qoverp_for_real_data_solution",
        "do_not_enter_alignment",
        "do_not_enter_measurement_model_v2",
        "do_not_promote_mode3_to_production",
    ),
)
def test_frozen_prohibitions_must_be_true(tmp_path, flag):
    bad = _tampered_config(tmp_path, lambda c: c.__setitem__(flag, False))
    with pytest.raises(ConfigError):
        wb87.load_config(bad)


def test_gate_tamper_fails(tmp_path):
    bad = _tampered_config(
        tmp_path,
        lambda c: c["closure_gates"].__setitem__("whitened_chi2_per_ndof_max", 99.0),
    )
    with pytest.raises(ConfigError):
        wb87.load_config(bad)


@pytest.mark.parametrize("parent", ("workbook_81", "workbook_86"))
def test_inheritance_sha_tamper_fails(tmp_path, parent):
    def mutate(c):
        c["inheritance"][f"{parent}_config_sha256"] = "0" * 64

    bad = _tampered_config(tmp_path, mutate)
    with pytest.raises(ConfigError):
        wb87.load_config(bad)


def test_reserved_floor_muon_stays_out_of_split(real_config):
    reserved = real_config["mc_data"]["reserved_conditional_j_support_source_id"]
    assert reserved == "floor_muon_100120"
    used = set(real_config["mc_data"]["construction_source_ids"]) | set(
        real_config["mc_data"]["validation_source_ids"]
    )
    assert reserved not in used


def test_geometric_jacobian_is_lever_arm():
    j = wb87.geometric_transport_jacobian(1906.0)
    assert j[0, 2] == pytest.approx(1906.0)
    assert j[1, 3] == pytest.approx(1906.0)
    assert np.allclose(np.diag(j), 1.0)


def test_highland_scales_as_one_over_p():
    t_soft = wb87.highland_theta0(1.0e4, 0.05)
    t_hard = wb87.highland_theta0(1.0e6, 0.05)
    assert t_soft / t_hard == pytest.approx(100.0, rel=1e-6)


def test_process_noise_is_spd_and_not_chi2_tuned():
    q = wb87.process_noise_covariance(1906.0, 1.0e5, 0.05)
    w = np.linalg.eigvalsh(q)
    assert np.all(w >= -1e-18)
    # First-principles Highland at 100 GeV / 5% X0 is << 1 mm^2 in y.
    assert q[1, 1] < 1.0


def test_wb81_mode3_is_not_the_validated_answer(real_config):
    assert real_config["modes"][2]["role"] == "part_a_primary"
    assert real_config["part_b"]["model"] == "highland_multiple_scattering"
    assert real_config["qoverp_audit"]["do_not_delete_qoverp_column_as_final_scheme"] is True


def _empty_split(calibrated: bool, *, jacobian: bool = True, pencil: float = 0.1, gen_max: float = 20.0, chi2: float = 50.0):
    def cell():
        return {
            "modes": {
                "0": {
                    "chi2_per_ndof": chi2,
                    "pencil": {"variance_ratio_prop_over_emp": pencil},
                    "generalized_eigenvalues_cemp_over_cprop": [0.01, 0.02, 0.03, gen_max],
                    "gate_verdict": {"calibrated": calibrated},
                },
                "1": {
                    "chi2_per_ndof": chi2,
                    "pencil": {"variance_ratio_prop_over_emp": pencil},
                    "generalized_eigenvalues_cemp_over_cprop": [0.01, 0.02, 0.03, gen_max],
                    "gate_verdict": {"calibrated": calibrated},
                },
                "2": {
                    "chi2_per_ndof": chi2,
                    "pencil": {"variance_ratio_prop_over_emp": 0.05},
                    "generalized_eigenvalues_cemp_over_cprop": [0.01, 0.02, 0.03, gen_max],
                    "gate_verdict": {"calibrated": calibrated},
                },
                "3": {
                    "chi2_per_ndof": chi2 * 0.9,
                    "pencil": {"variance_ratio_prop_over_emp": 0.05},
                    "generalized_eigenvalues_cemp_over_cprop": [0.01, 0.02, 0.03, gen_max],
                    "gate_verdict": {"calibrated": False},
                },
            },
            "jacobian_self_consistency": {"gate_verdict": {"calibrated": jacobian}},
        }

    return {
        "per_pair": {"(0,1)": cell(), "(0,2)": cell(), "(0,3)": cell()},
    }


def test_decide_validated_keeps_alignment_closed():
    split = _empty_split(True, jacobian=True, pencil=1.0, gen_max=1.0, chi2=1.0)
    # Force mode 2/3 calibrated true in the helper above only for mode gate_verdict;
    # decide() reads campaign["validation"] not the per-mode flags.
    campaign = {
        "validation": {
            "whitening_closure_recovered": True,
            "source_disjoint_agreement": True,
            "construction_primary_calibrated": True,
            "validation_primary_calibrated": True,
            "jacobian_self_consistency_calibrated": True,
            "process_noise_calibrated": True,
        },
        "part_a_deterministic_transport": {
            "construction": split,
            "validation": split,
        },
    }
    d = wb87.decide({}, campaign)
    assert d["decision"] == DECISION_VALIDATED
    assert d["measurement_model_validated"] is False
    assert d["measurement_model_v2_discussion_allowed"] is True
    assert d["measurement_model_v2_entered"] is False
    assert d["geometry_write_allowed"] is False
    assert d["real_data_alignment_authorized"] is False
    assert d["qoverp_column_deleted_as_final_scheme"] is False
    assert d["stage_b_entered"] is True


def test_decide_not_validated_classifies_qoverp_semantics():
    split = _empty_split(False, jacobian=True, pencil=300.0, gen_max=50.0, chi2=1.0e4)
    campaign = {
        "validation": {
            "whitening_closure_recovered": False,
            "source_disjoint_agreement": True,
            "construction_primary_calibrated": False,
            "validation_primary_calibrated": False,
            "jacobian_self_consistency_calibrated": True,
            "process_noise_calibrated": False,
        },
        "part_a_deterministic_transport": {
            "construction": split,
            "validation": split,
        },
    }
    d = wb87.decide({}, campaign)
    assert d["decision"] == DECISION_NOT_VALIDATED
    assert d["failure_classification"]["category"] == MECHANISM_QOP_SEMANTICS
    assert d["failure_classification"]["deleting_qoverp_column_is_final_scheme"] is False
    assert d["measurement_model_v2_discussion_allowed"] is False


def test_jacobian_failure_classified_separately():
    split = _empty_split(False, jacobian=False, pencil=300.0, gen_max=50.0, chi2=1.0e4)
    mech = wb87.classify_mechanism(split, split)
    assert mech["category"] == MECHANISM_JACOBIAN_ERROR


def test_augment_metrics_adds_median_and_tail():
    rng = np.random.default_rng(0)
    e = rng.normal(size=(40, 4))
    c = np.repeat(np.eye(4)[None, :, :], 40, axis=0)
    metrics = wb87._augment_metrics({"chi2_per_ndof": 1.0}, e, c)
    assert metrics["chi2_per_ndof_median"] < 2.0
    assert 0.0 <= metrics["fraction_chi2_per_ndof_above_4"] < 0.2


def test_as_closure_can_select_transported_source_residual():
    recs = wb87.StageBRecords(
        station_pair=(0, 1),
        e_target=np.ones((2, 4)),
        e_transported_source=2.0 * np.ones((2, 4)),
        c_production=np.repeat(np.eye(4)[None, :, :], 2, axis=0),
        c_existing_qop=np.repeat(np.eye(4)[None, :, :], 2, axis=0),
        c_corrected_qop=np.repeat(np.eye(4)[None, :, :], 2, axis=0),
        c_process_noise=np.repeat(np.eye(4)[None, :, :], 2, axis=0),
        lever_arm_mm=np.array([1906.0, 1906.0]),
        pred_tx=np.zeros(2),
        pred_ty=np.zeros(2),
        abs_q_over_p=np.array([1e-5, 1e-5]),
        n_matched=2,
    )
    closure = recs.as_closure(recs.c_corrected_qop, transported_source=True)
    assert isinstance(closure, ClosureRecords)
    assert np.allclose(closure.e_prop, 2.0)
