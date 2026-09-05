"""Workbook-78 gauge-fixed real-data alignment diagnostic tests.

Covers the pre-registered config freeze (primary gauge, split rule, gates),
the deterministic residual-blind split freeze, the real-data bank builder's
bit-exact reproduction of the frozen DQ CSV edges, gauge-equivalent
representatives predicting identical observables, the one-shot
informed-subspace solver on synthetic banks, the MC-control machinery,
candidate-artifact reproducibility, the structural held-out freeze ordering,
and every pre-registered decision-tree branch.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from alignment.gauge_fixed_real_data_diagnostic import (
    DECISION_GAUGE_INVARIANCE_FAILED,
    DECISION_INCONCLUSIVE,
    DECISION_NOT_IMPROVED,
    DECISION_NOT_STABLE,
    DECISION_OUT_OF_SUPPORT,
    DECISION_PASS,
    DECISION_RANK_ZERO,
    DECISION_TRACKER_REGRESSION_FAILED,
    DECISION_TRANSFER_FAILED,
    PARAMETER_NAMES,
    SCHEMA_VERSION,
    _canonical_sha256,
    applicability_audit,
    build_gauge_candidates,
    build_real_data_bank,
    build_transfer_model,
    decide_campaign,
    evaluate_held_out,
    freeze_candidate_artifact,
    freeze_split,
    gauge_representatives,
    load_config,
    load_frozen_csv_edges,
    mc_control,
    predict_delta_for_pairs,
    solve_candidate,
)
from alignment.identifiable_subspace import FROZEN_RANK_TOLERANCE, IdentifiableSubspace, frozen_scales_for
from scripts.report_gauge_fixed_real_data_alignment_diagnostic import _stage_evaluate

CONFIG_PATH = "configs/gauge_fixed_real_data_alignment_diagnostic_v1.yaml"


def _config():
    return load_config(CONFIG_PATH)


def _tampered_config(tmp_path: Path, mutate) -> str:
    raw = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "tampered.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(path)


def _synthetic_subspace() -> IdentifiableSubspace:
    """7D subspace, 2D tilted null (WB77 test convention)."""
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


def _synthetic_transfer(rng: np.random.Generator) -> dict:
    means = {}
    for pair in ((0, 1), (0, 2), (0, 3)):
        means[f"{pair[0]}->{pair[1]}"] = rng.normal(scale=0.5, size=(4, 7))
    return {"kind": "station_pair_mean_native_jacobian", "mean_jacobian": means}


def _synthetic_bank(
    rng: np.random.Generator,
    subspace: IdentifiableSubspace,
    transfer: dict,
    *,
    n_pairs: int = 60,
    beta_inj: np.ndarray | None = None,
    noise: float = 0.05,
) -> dict:
    """Synthetic calibration bank through the transfer model itself."""
    scales = np.asarray(subspace.parameter_scales, dtype=np.float64)
    v_id = np.asarray(subspace.v_id, dtype=np.float64)
    residuals = []
    covariances = []
    targets = []
    for i in range(n_pairs):
        pair = ((0, 1), (0, 2), (0, 3))[i % 3]
        mean = transfer["mean_jacobian"][f"{pair[0]}->{pair[1]}"]
        cov = np.diag(rng.uniform(0.5, 1.5, size=4))
        r = rng.normal(scale=noise, size=4)
        if beta_inj is not None:
            r = r + mean @ (np.diag(scales) @ v_id @ beta_inj)
        residuals.append(r)
        covariances.append(cov)
        targets.append(pair[1])
    return {
        "kind": "real_data_anchor_pair_bank",
        "roles": ["calibration"],
        "banks": [
            {
                "source_id": "synthetic",
                "run_id": 1,
                "role": "calibration",
                "n_routes": n_pairs,
                "n_anchor_pair_keys": n_pairs,
                "n_pairs_missing_propagation": 0,
                "residual": np.asarray(residuals),
                "covariance": np.asarray(covariances),
                "run_id_arr": np.ones(n_pairs, dtype=np.int64),
                "event_id": np.arange(n_pairs, dtype=np.int64),
                "route_index": np.arange(n_pairs, dtype=np.int64),
                "source_tracklet_id": np.zeros(n_pairs, dtype=np.int64),
                "target_tracklet_id": np.arange(n_pairs, dtype=np.int64),
                "target_station_id": np.asarray(targets, dtype=np.int64),
            }
        ],
        "csv_reproduction": {"rows_checked": 0, "max_abs_diff": 0.0, "bit_exact": True},
    }


# ---------------------------------------------------------------------------
# Config freeze
# ---------------------------------------------------------------------------


def test_config_loads_and_verifies_inheritance():
    config = _config()
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["gauges"]["primary"] == "minimum_norm_scaled_gauge"
    assert config["gauges"]["secondary_control"] == "named_parameter_zero_gauge"
    assert config["gauges"]["report_only"] == "minimum_norm_native_gauge"
    assert config["geometry_write_allowed"] is False
    assert config["official_conditions_write_allowed"] is False
    assert config["external_constraint_ingest_authorized"] is False
    assert config["eligible_external_physical_constraints"] == []
    assert config["output_parameter_naming"] == "reconstruction_gauge_representative"


def test_primary_gauge_frozen_rejects_switch(tmp_path):
    path = _tampered_config(tmp_path, lambda c: c["gauges"].__setitem__("primary", "named_parameter_zero_gauge"))
    with pytest.raises(ValueError, match="primary gauge is frozen"):
        load_config(path)


def test_config_rejects_geometry_write(tmp_path):
    path = _tampered_config(tmp_path, lambda c: c.__setitem__("geometry_write_allowed", True))
    with pytest.raises(ValueError, match="geometry_write_allowed"):
        load_config(path)


def test_config_rejects_official_conditions_write(tmp_path):
    path = _tampered_config(
        tmp_path, lambda c: c.__setitem__("official_conditions_write_allowed", True)
    )
    with pytest.raises(ValueError, match="official_conditions_write_allowed"):
        load_config(path)


def test_config_rejects_external_prior_ingestion(tmp_path):
    path = _tampered_config(
        tmp_path, lambda c: c.__setitem__("external_constraint_ingest_authorized", True)
    )
    with pytest.raises(ValueError, match="external_constraint_ingest_authorized"):
        load_config(path)
    path = _tampered_config(
        tmp_path,
        lambda c: c.__setitem__("eligible_external_physical_constraints", ["nov22_cdx"]),
    )
    with pytest.raises(ValueError, match="external constraint branch remains closed"):
        load_config(path)


def test_config_rejects_rank_tolerance_change(tmp_path):
    path = _tampered_config(
        tmp_path, lambda c: c["tracker_information"].__setitem__("rank_tolerance", 0.02)
    )
    with pytest.raises(ValueError, match="rank_tolerance"):
        load_config(path)


# ---------------------------------------------------------------------------
# Split freeze (deterministic, residual-blind, provenance-hard-failing)
# ---------------------------------------------------------------------------


def test_split_freeze_deterministic_and_covers_frozen_population():
    config = _config()
    first = freeze_split(config)
    second = freeze_split(config)
    assert first["split_sha256"] == second["split_sha256"]
    assert first["residual_blind"] is True
    roles = {entry["source_id"]: entry["role"] for entry in first["entries"]}
    assert roles["data24_r14973_00007_skip49500_n84988"] == "calibration"
    assert roles["data24_r14974_00005_skip74400_n60886"] == "calibration"
    assert roles["data24_r14975_00005_skip21600_n111781"] == "held_out"
    assert roles["data24_r14976_00004_skip95500_n40562"] == "held_out"
    assert roles["data24_r14977_00005_skip135900_n01687"] == "held_out_report_only"
    calibration = [
        entry for entry in first["entries"] if entry["role"] == "calibration"
    ]
    assert sum(e["n_selected_routes"] for e in calibration) == 230


def test_split_freeze_rejects_provenance_tampering(tmp_path):
    def mutate(config):
        config["real_data_population"]["sources"][0]["expected_selected_routes"] = 999

    path = _tampered_config(tmp_path, mutate)
    config = load_config(path)
    with pytest.raises(ValueError, match="frozen population mismatch"):
        freeze_split(config)


def test_split_freeze_rejects_mc_control_composition_drift(tmp_path):
    def mutate(config):
        config["mc_control"]["calibration_composition"]["0->1"] = 1

    path = _tampered_config(tmp_path, mutate)
    config = load_config(path)
    with pytest.raises(ValueError, match="does not match the frozen split composition"):
        freeze_split(config)


# ---------------------------------------------------------------------------
# Real-data bank builder: bit-exact reproduction of the frozen DQ CSV edges
# ---------------------------------------------------------------------------


def test_bank_builder_reproduces_frozen_csv_bit_exact():
    config = _config()
    bank = build_real_data_bank(config, roles=config["split"]["calibration_roles"])
    assert bank["csv_reproduction"]["bit_exact"] is True
    assert bank["csv_reproduction"]["rows_checked"] == 390
    assert bank["csv_reproduction"]["max_abs_diff"] == 0.0
    totals = {"0->1": 0, "0->2": 0, "0->3": 0}
    for item in bank["banks"]:
        # Every frozen route anchor pair either is rebuilt or is accounted
        # for by the frozen propagation-acceptance chain.
        assert item["residual"].shape[0] == (
            item["n_anchor_pair_keys"] - item["n_pairs_missing_propagation"]
        )
        for station in (1, 2, 3):
            totals[f"0->{station}"] += int(
                np.count_nonzero(item["target_station_id"] == station)
            )
    assert totals["0->1"] == 227
    assert totals["0->3"] == 2
    missing = sum(item["n_pairs_missing_propagation"] for item in bank["banks"])
    assert totals["0->2"] == 156 - missing


# ---------------------------------------------------------------------------
# Gauge representatives: gauge-equivalent predictions
# ---------------------------------------------------------------------------


def test_gauge_representatives_differ_but_predict_same_observable():
    subspace = _synthetic_subspace()
    gauges = build_gauge_candidates(_config(), subspace)
    rng = np.random.default_rng(3)
    beta = rng.normal(size=5)
    representatives = gauge_representatives(beta, subspace, gauges)
    thetas = [rep["theta_native"] for rep in representatives.values()]
    # Gauge-dependent absolute components differ across legal gauges.
    assert max(float(np.max(np.abs(a - b))) for a in thetas for b in thetas) > 1.0e-6
    for gauge_id, rep in representatives.items():
        assert rep["gauge_residual_max_abs"] <= 1.0e-9
        np.testing.assert_allclose(rep["identifiable_projection_beta"], beta, atol=1e-12)
    transfer = _synthetic_transfer(rng)
    targets = np.asarray([1, 2, 3, 1, 2], dtype=np.int64)
    predictions = {
        gauge_id: predict_delta_for_pairs(
            transfer,
            np.zeros(targets.size, dtype=np.int64),
            targets,
            subspace,
            rep["identifiable_projection_beta"],
        )
        for gauge_id, rep in representatives.items()
    }
    ids = sorted(predictions)
    for left in range(len(ids)):
        for right in range(left + 1, len(ids)):
            np.testing.assert_allclose(predictions[ids[left]], predictions[ids[right]], atol=1e-12)


def test_non_ift_pairs_have_exactly_zero_prediction():
    subspace = _synthetic_subspace()
    rng = np.random.default_rng(4)
    transfer = _synthetic_transfer(rng)
    beta = rng.normal(size=5)
    source = np.asarray([1, 2, 0, 0], dtype=np.int64)
    target = np.asarray([2, 3, 1, 2], dtype=np.int64)
    delta = predict_delta_for_pairs(transfer, source, target, subspace, beta)
    assert np.all(delta[:2] == 0.0)
    assert np.any(delta[2:] != 0.0)


# ---------------------------------------------------------------------------
# One-shot solver on synthetic banks
# ---------------------------------------------------------------------------


def test_solve_candidate_recovers_synthetic_injection():
    from alignment.gauge_fixed_real_data_diagnostic import (
        _information_and_rhs,
        _informed_basis,
        _pair_design_rows,
        concatenate_banks,
    )

    config = _config()
    subspace = _synthetic_subspace()
    rng = np.random.default_rng(11)
    transfer = _synthetic_transfer(rng)
    gauges = build_gauge_candidates(config, subspace)
    # Determine the informed subspace of the synthetic bank first, then
    # inject along its strongest direction (mirrors the campaign logic).
    pilot = _synthetic_bank(rng, subspace, transfer, n_pairs=90, noise=0.01)
    arrays = concatenate_banks(pilot)
    design = _pair_design_rows(transfer, arrays["target_station_id"], subspace)
    information, _rhs = _information_and_rhs(design, arrays["covariance"], arrays["residual"])
    _eigvals, basis, rank = _informed_basis(information, rank_tolerance=0.01)
    assert rank >= 2
    beta_inj = 0.10 * basis[:, -1] / np.linalg.norm(basis[:, -1])
    bank = _synthetic_bank(rng, subspace, transfer, n_pairs=90, beta_inj=beta_inj, noise=0.01)
    candidate = solve_candidate(config, subspace, transfer, bank, gauges)
    assert candidate["status"] == "solved"
    assert candidate["informed_rank"] == rank
    beta_hat = np.asarray(candidate["beta_hat_identifiable_projection"])
    proj_hat = basis.T @ beta_hat
    proj_inj = basis.T @ beta_inj
    np.testing.assert_allclose(proj_hat, proj_inj, atol=2e-2)
    assert candidate["within_linear_envelope"] is True
    assert candidate["gauge_residual_max_abs"] <= 1.0e-9
    assert candidate["bootstrap"]["stable"] is True
    primary = candidate["reconstruction_gauge_representative"]["minimum_norm_scaled_gauge"]
    np.testing.assert_allclose(
        np.asarray(primary["null_coordinates_mu"]), np.zeros(2), atol=1e-12
    )


def test_solve_candidate_detects_rank_zero_information():
    config = _config()
    subspace = _synthetic_subspace()
    rng = np.random.default_rng(13)
    transfer = _synthetic_transfer(rng)
    for key in transfer["mean_jacobian"]:
        transfer["mean_jacobian"][key] = np.zeros((4, 7))
    gauges = build_gauge_candidates(config, subspace)
    bank = _synthetic_bank(rng, subspace, transfer, n_pairs=10)
    candidate = solve_candidate(config, subspace, transfer, bank, gauges)
    assert candidate["status"] == DECISION_RANK_ZERO
    assert candidate["informed_rank"] == 0


def test_candidate_artifact_reproducible():
    config = _config()
    subspace = _synthetic_subspace()
    rng = np.random.default_rng(17)
    transfer = _synthetic_transfer(rng)
    gauges = build_gauge_candidates(config, subspace)
    bank = _synthetic_bank(rng, subspace, transfer, n_pairs=30, noise=0.05)
    split = {"split_sha256": "0" * 64}
    audit = {"kind": "pre_fit_applicability_audit", "hard_fail": False}
    candidate = solve_candidate(config, subspace, transfer, bank, gauges)
    control = {"null_improvement_floor": 0.0}
    first = freeze_candidate_artifact(
        config, split=split, bank=bank, audit=audit, candidate=candidate,
        subspace=subspace, gauges=gauges, mc_control_report=control,
    )
    second = freeze_candidate_artifact(
        config, split=split, bank=bank, audit=audit, candidate=candidate,
        subspace=subspace, gauges=gauges, mc_control_report=control,
    )
    assert _canonical_sha256(first) == _canonical_sha256(second)
    assert first["frozen_before_held_out_access"] is True


# ---------------------------------------------------------------------------
# Structural held-out freeze ordering (driver stage gates)
# ---------------------------------------------------------------------------


def test_held_out_inaccessible_before_candidate_freeze(tmp_path):
    path = _tampered_config(tmp_path, lambda c: c.__setitem__("output_root", str(tmp_path / "out")))
    config = load_config(path)
    (tmp_path / "out").mkdir()
    with pytest.raises(RuntimeError, match="no frozen candidate artifact"):
        _stage_evaluate(config)


def test_held_out_refuses_sha_mismatch(tmp_path):
    path = _tampered_config(tmp_path, lambda c: c.__setitem__("output_root", str(tmp_path / "out")))
    config = load_config(path)
    out = tmp_path / "out"
    out.mkdir()
    artifact = {
        "kind": "gauge_fixed_candidate_diagnostic",
        "frozen_before_held_out_access": True,
        "candidate": {"status": "solved"},
        "candidate_artifact_sha256": "0" * 64,
    }
    (out / "candidate_artifact.json").write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        _stage_evaluate(config)


# ---------------------------------------------------------------------------
# Decision tree branches
# ---------------------------------------------------------------------------


def _decision_inputs():
    regression = {"pass": True}
    control = {"pass": True, "null_improvement_floor": 1.0}
    audit = {"hard_fail": False}
    candidate = {
        "status": "solved",
        "within_linear_envelope": True,
        "solver_within_condition_gate": True,
        "bootstrap": {"stable": True},
    }
    gates = {
        "A_held_out_weighted_residual_improvement_beyond_null_floor": True,
        "B_run_level_consistency": True,
        "C_no_catastrophic_cell_degradation": True,
        "D_gauge_invariant_held_out_prediction": True,
        "E_identifiable_projection_bootstrap_stable": True,
        "F_no_new_systematic_shift_in_dq_slices": True,
        "G_correction_within_linear_envelope": True,
    }
    held_out = {"gates": gates, "pass": True}
    return regression, control, audit, candidate, held_out


def test_decision_tree_pass_branch():
    regression, control, audit, candidate, held_out = _decision_inputs()
    decision = decide_campaign(
        regression=regression, mc_control_report=control, audit=audit,
        candidate=candidate, held_out=held_out,
    )
    assert decision["decision"] == DECISION_PASS
    assert decision["geometry_write_allowed"] is False
    assert decision["official_conditions_write_allowed"] is False
    assert decision["external_constraint_ingest_authorized"] is False


def test_decision_tree_failure_branches():
    regression, control, audit, candidate, held_out = _decision_inputs()

    bad_regression = {"pass": False}
    assert decide_campaign(
        regression=bad_regression, mc_control_report=control, audit=audit,
        candidate=candidate, held_out=held_out,
    )["decision"] == DECISION_TRACKER_REGRESSION_FAILED

    assert decide_campaign(
        regression=regression, mc_control_report={"pass": False}, audit=audit,
        candidate=candidate, held_out=held_out,
    )["decision"] == DECISION_TRANSFER_FAILED

    assert decide_campaign(
        regression=regression, mc_control_report=control,
        audit={"hard_fail": True}, candidate=candidate, held_out=held_out,
    )["decision"] == DECISION_INCONCLUSIVE

    rank_zero = dict(candidate, status=DECISION_RANK_ZERO)
    assert decide_campaign(
        regression=regression, mc_control_report=control, audit=audit,
        candidate=rank_zero, held_out=held_out,
    )["decision"] == DECISION_RANK_ZERO

    out_of_support = dict(candidate, within_linear_envelope=False)
    assert decide_campaign(
        regression=regression, mc_control_report=control, audit=audit,
        candidate=out_of_support, held_out=held_out,
    )["decision"] == DECISION_OUT_OF_SUPPORT

    unstable = dict(candidate, bootstrap={"stable": False})
    assert decide_campaign(
        regression=regression, mc_control_report=control, audit=audit,
        candidate=unstable, held_out=held_out,
    )["decision"] == DECISION_NOT_STABLE

    bad_invariance = dict(held_out["gates"], D_gauge_invariant_held_out_prediction=False)
    assert decide_campaign(
        regression=regression, mc_control_report=control, audit=audit,
        candidate=candidate, held_out={"gates": bad_invariance, "pass": False},
    )["decision"] == DECISION_GAUGE_INVARIANCE_FAILED

    not_improved = dict(held_out["gates"], A_held_out_weighted_residual_improvement_beyond_null_floor=False)
    assert decide_campaign(
        regression=regression, mc_control_report=control, audit=audit,
        candidate=candidate, held_out={"gates": not_improved, "pass": False},
    )["decision"] == DECISION_NOT_IMPROVED


# ---------------------------------------------------------------------------
# MC-control machinery on synthetic information
# ---------------------------------------------------------------------------


def test_mc_control_passes_on_transfer_consistent_synthetic_information():
    config = _config()
    subspace = _synthetic_subspace()
    rng = np.random.default_rng(23)
    transfer = _synthetic_transfer(rng)
    gauges = build_gauge_candidates(config, subspace)
    # Synthetic MC bank: per-pair Jacobians fluctuate mildly around the
    # transfer means; residuals are pure noise (null hypothesis).
    n_pairs = 400
    source_station = np.asarray([0] * n_pairs)
    target_station = np.asarray([1, 2, 3] * (n_pairs // 3) + [1] * (n_pairs % 3))
    jacobian = np.zeros((n_pairs, 4, 7))
    covariance = np.zeros((n_pairs, 4, 4))
    residual = rng.normal(scale=0.05, size=(n_pairs, 4))
    for i in range(n_pairs):
        mean = transfer["mean_jacobian"][f"0->{target_station[i]}"]
        jacobian[i] = mean + rng.normal(scale=0.01, size=(4, 7))
        covariance[i] = np.diag(rng.uniform(0.5, 1.5, size=4))
    pooled = {
        "source_station_id": source_station,
        "target_station_id": target_station,
        "covariance": covariance,
        "anchor_residual": residual,
    }

    class _Fit:
        derivative_native = jacobian

    extras = {"fit": _Fit(), "n_pairs": n_pairs}
    synthetic_transfer = build_transfer_model(pooled, extras)
    control = copy.deepcopy(config["mc_control"])
    control["calibration_composition"] = {"0->1": 40, "0->2": 40, "0->3": 40}
    control["held_out_composition"] = {"0->1": 40, "0->2": 40, "0->3": 40}
    control["null_ensemble_replicates"] = 20
    config = dict(config, mc_control=control)
    report = mc_control(config, pooled, subspace, extras, synthetic_transfer, gauges)
    assert report["null_improvement_floor"] >= 0.0
    for scenario in report["scenarios"]:
        assert scenario["informed_rank"] > 0
        assert scenario["gauge_invariance_max_relative_diff"] <= 1.0e-8
        for injection in scenario["injection"]:
            if injection["direction"] == "informed":
                assert injection["max_relative_deviation"] <= 0.35
