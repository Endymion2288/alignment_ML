"""Workbook-81b: linear residual, frozen loss, fake hard gate.  No Condor."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from models.explicit_route_energy import FEATURE_DIM
from models.physics_constrained_calibration import MODEL_CONTRACT, PhysicsConstrainedCalibrator
from training.wb81_calibration_contract import (
    ACCEPT_EFF_GAIN,
    BIN_WEIGHTS,
    DELTA_MAX,
    EXPECTED_FLIPPABLE,
    calibration_acceptable,
    fake_hard_gate,
    refuse_wb81a_path,
    verify_wb81a_checksum_file,
)
from training.wb81b_calibration import fold_decision


def test_linear_zero_init_is_identity_and_bound():
    assert MODEL_CONTRACT == "physics_constrained_calibration_d_v1"
    assert FEATURE_DIM == 54
    model = PhysicsConstrainedCalibrator()
    assert type(model.g).__name__ == "Linear"
    features = torch.randn(5, FEATURE_DIM)
    delta = model(features)
    assert delta.shape == (5,)
    assert torch.allclose(delta, torch.zeros(5), atol=1.0e-7)
    with torch.no_grad():
        model.g.weight.fill_(10.0)
        model.g.bias.fill_(10.0)
    saturated = model(features)
    assert float(torch.max(torch.abs(saturated)).detach()) <= DELTA_MAX + 1.0e-6


def test_loss_weights_and_accept_rule_are_frozen():
    assert BIN_WEIGHTS == {"negative": 2.0, "near_zero": 1.0, "comfortable": 0.0}
    assert ACCEPT_EFF_GAIN == 0.01
    assert DELTA_MAX == 0.25
    assert fake_hard_gate(0.013, 0.013) is True
    assert fake_hard_gate(0.014, 0.013) is False
    assert fake_hard_gate(None, 0.013) is False
    assert calibration_acceptable(0.86, 0.013, 0.84, 0.013) is True
    assert calibration_acceptable(0.845, 0.013, 0.84, 0.013) is False
    assert calibration_acceptable(0.86, 0.014, 0.84, 0.013) is False


def test_fold_decision_restores_zero_when_no_admission():
    rejected = fold_decision(
        identity_passed=True,
        admitted=False,
        holdout_a={"complete_track_efficiency": 0.84, "complete_fake_rate": 0.01},
        holdout_d={"complete_track_efficiency": 0.84, "complete_fake_rate": 0.01},
    )
    assert rejected["calibration_unacceptable_under_zero_fake_slack"] is True
    assert rejected["restore"] == "DeltaU = 0"
    assert rejected["default_system"] == "frozen_W64_raw_energy_plus_exact_solver"
    assert rejected["physics_constrained_calibration_supported"] is False
    accepted = fold_decision(
        identity_passed=True,
        admitted=True,
        holdout_a={"complete_track_efficiency": 0.84, "complete_fake_rate": 0.02},
        holdout_d={"complete_track_efficiency": 0.86, "complete_fake_rate": 0.02},
    )
    assert accepted["calibration_acceptable"] is True
    assert accepted["physics_constrained_calibration_supported"] is False
    no_inc = fold_decision(
        identity_passed=True,
        admitted=True,
        holdout_a={"complete_track_efficiency": 0.84, "complete_fake_rate": 0.02},
        holdout_d={"complete_track_efficiency": 0.841, "complete_fake_rate": 0.02},
    )
    assert no_inc["calibration_has_no_accepted_increment"] is True


def test_wb81a_checksum_and_forbidden_paths():
    assert EXPECTED_FLIPPABLE["family1_ds100043_100044"] == 919
    assert EXPECTED_FLIPPABLE["family2_ds100047_100048"] == 458
    report = verify_wb81a_checksum_file(
        "outputs/mc24_four_station_wb81a_calibration_foundation_v1/family1_ds100043_100044/checksums.json",
        "family1_ds100043_100044",
    )
    assert report["passed"] is True
    with pytest.raises(ValueError, match="refuses"):
        refuse_wb81a_path("outputs/mc24_four_station_explicit_route_energy_v1/holdout_family1/arm_b_checkpoint.json")
    with pytest.raises(ValueError, match="refuses"):
        refuse_wb81a_path("mc24_100047_00350_00399/x.root")


def test_train_script_does_not_import_b_c_or_lai():
    text = Path("scripts/train_eval_wb81b_calibration.py").read_text(encoding="utf-8")
    model_text = Path("models/physics_constrained_calibration.py").read_text(encoding="utf-8")
    loss_text = Path("training/wb81b_calibration.py").read_text(encoding="utf-8")
    combined = text + model_text + loss_text
    assert "ExplicitRouteEnergyScorer" not in text
    assert "HybridRouteEnergy" not in text
    assert "loss_augmented_structured_hinge" not in combined
    assert "arm_b_checkpoint" not in text
    assert "arm_c_checkpoint" not in text
    assert "sigmoid(" not in combined
    assert "clip(" not in combined
    assert "logit(" not in combined
    assert "_route_hypotheses" not in combined
    assert "n_stations" not in model_text
    assert FEATURE_DIM == 54
