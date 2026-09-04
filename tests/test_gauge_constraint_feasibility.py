"""Workbook-77 gauge-constrained / external-constraint feasibility tests.

Covers the pre-registered config freeze, the candidate gauge constraint
matrices, the KKT constrained solver, the gauge-invariant observable
closure on synthetic tracker information, the external-constraint
eligibility rule, and the campaign decision branches.  All closure tests
use synthetic information matrices; the real campaign rebuilds the frozen
workbook-68 tracker information with regression.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from alignment.gauge_constraint_feasibility import (
    DECISION_CLOSURE_FAILED,
    DECISION_EXTERNAL_CANDIDATE_FOUND,
    DECISION_GAUGE_CONTRACT_INVALID,
    DECISION_GAUGE_FEASIBILITY_CLOSED,
    DECISION_SOLVER_NOT_WELL_POSED,
    DECISION_TRACKER_REGRESSION_FAILED,
    PARAMETER_NAMES,
    SCHEMA_VERSION,
    build_gauge_candidates,
    decide_campaign,
    external_eligibility_table,
    gauge_invariant_closure,
    load_config,
    solve_gauge_fixed_kkt,
)
from alignment.identifiable_subspace import (
    FROZEN_RANK_TOLERANCE,
    IdentifiableSubspace,
    frozen_scales_for,
)
from alignment.module_level_residual_poc import json_ready
from scripts.report_gauge_constraint_external_constraint_feasibility import (
    _strip_private,
)

CONFIG_PATH = "configs/gauge_constraint_external_constraint_feasibility_v1.yaml"


def _config():
    return load_config(CONFIG_PATH)


def _synthetic_subspace(tilted=True) -> IdentifiableSubspace:
    """7D subspace with a 2D null space.

    With ``tilted=True`` (default) the null space is
    ``span(e_dz, (e_dx + e_Cdx)/sqrt(2))`` - mimicking the frozen
    workbook-68 structure where one null direction is a tilted dx-C_dx
    combination, so the named-parameter and minimum-norm gauges genuinely
    differ.  With ``tilted=False`` the null space is the axis-aligned
    ``span(e_rx, e_ry)`` used to probe invalid gauge detection.
    """
    names = PARAMETER_NAMES
    scales = frozen_scales_for(names)
    if tilted:
        mixed = np.zeros(7)
        mixed[[0, 6]] = 1.0 / np.sqrt(2.0)
        v_null = np.column_stack([np.eye(7)[:, 2], mixed])
        id_columns = [np.eye(7)[:, i] for i in (1, 3, 4, 5)]
        anti = np.zeros(7)
        anti[0] = 1.0 / np.sqrt(2.0)
        anti[6] = -1.0 / np.sqrt(2.0)
        v_id = np.column_stack(id_columns + [anti])
    else:
        v_null = np.eye(7)[:, [3, 4]]
        v_id = np.eye(7)[:, [0, 1, 2, 5, 6]]
    v = np.column_stack([v_id, v_null])
    singular = np.asarray([100.0, 80.0, 60.0, 40.0, 20.0, 0.0, 0.0])
    return IdentifiableSubspace(
        parameter_names=tuple(names),
        parameter_units=("mm", "mm", "mm", "mrad", "mrad", "mrad", "mm"),
        parameter_scales=scales,
        rank_tolerance=float(FROZEN_RANK_TOLERANCE),
        singular_values=singular,
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


def _synthetic_extras(subspace: IdentifiableSubspace, n_pairs=30, seed=7):
    """A = U S V^T with exact rank 5 on the synthetic identifiable basis."""
    rng = np.random.default_rng(seed)
    u_left = np.linalg.qr(rng.normal(size=(4 * n_pairs, 5)))[0]
    spectrum = np.asarray([100.0, 80.0, 60.0, 40.0, 20.0])
    a = u_left @ np.diag(spectrum) @ subspace.v_id.T
    return {"weighted_matrix": a, "n_pairs": n_pairs}


# ---------------------------------------------------------------------------
# Config freeze
# ---------------------------------------------------------------------------


def test_config_loads_and_verifies_inheritance():
    config = _config()
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["frozen"] is True
    assert tuple(config["parameter_space"]["parameter_names"]) == PARAMETER_NAMES
    assert config["closure"]["full_truth_recovery_is_not_a_gate"] is True
    decision = config["decision_contract"]
    assert decision["real_data_candidate_alignment_authorized"] is False
    assert decision["geometry_write_allowed"] is False
    assert decision["official_conditions_write_allowed"] is False
    assert decision["external_constraint_ingest_authorized"] is False


def test_config_rejects_retuned_rank_tolerance(tmp_path):
    import yaml

    payload = yaml.safe_load(Path(CONFIG_PATH).read_text())
    payload["parameter_space"]["rank_tolerance"] = 0.02
    path = tmp_path / "retuned.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="rank_tolerance"):
        load_config(path)


def test_config_rejects_missing_prohibition(tmp_path):
    import yaml

    payload = yaml.safe_load(Path(CONFIG_PATH).read_text())
    payload["do_not_lower_min_pairs_post_hoc"] = False
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="do_not_lower_min_pairs_post_hoc"):
        load_config(path)


# ---------------------------------------------------------------------------
# Gauge candidates
# ---------------------------------------------------------------------------


def test_named_parameter_zero_gauge_matrix():
    config = _config()
    subspace = _synthetic_subspace()
    candidates = build_gauge_candidates(config, subspace)
    named = next(c for c in candidates if c["id"] == "named_parameter_zero_gauge")
    g = np.asarray(named["_g_native"])
    assert g.shape == (2, 7)
    assert g[0, PARAMETER_NAMES.index("ift_dz_mm")] == 1.0
    assert g[1, PARAMETER_NAMES.index("C_dx")] == 1.0
    assert named["valid"] is True
    assert named["remaining_dof"] == 5
    assert named["changes_tracker_observable_prediction"] is False


def test_minimum_norm_gauges_orthogonal_to_null():
    config = _config()
    subspace = _synthetic_subspace()
    candidates = build_gauge_candidates(config, subspace)
    scales = np.asarray(subspace.parameter_scales)
    v_null = np.asarray(subspace.v_null)
    scaled = next(c for c in candidates if c["id"] == "minimum_norm_scaled_gauge")
    native = next(c for c in candidates if c["id"] == "minimum_norm_native_gauge")
    # Scaled minimum-norm: G u = V_null^T u with u = S^{-1} theta.
    g_scaled = np.asarray(scaled["_g_native"]) @ np.diag(scales)
    assert np.allclose(g_scaled, v_null.T)
    # Native minimum-norm: theta orthogonal to S V_null.
    g_native = np.asarray(native["_g_native"])
    assert np.allclose(g_native, (np.diag(scales) @ v_null).T)
    # The two minimum-norm gauges differ because S is not uniform.
    assert not np.allclose(g_scaled, g_native @ np.diag(scales))
    assert scaled["valid"] and native["valid"]


def test_gauge_candidate_invalid_when_constraint_misses_null():
    config = _config()
    # Null space is span(e_rx, e_ry): fixing dz and C_dx then constrains
    # only identifiable directions and cannot complement the null space.
    subspace = _synthetic_subspace(tilted=False)
    candidates = build_gauge_candidates(config, subspace)
    named = next(c for c in candidates if c["id"] == "named_parameter_zero_gauge")
    assert named["valid"] is False


# ---------------------------------------------------------------------------
# KKT solver
# ---------------------------------------------------------------------------


def test_kkt_solver_satisfies_constraint_and_recovers_identifiable_projection():
    subspace = _synthetic_subspace()
    extras = _synthetic_extras(subspace)
    a = extras["weighted_matrix"]
    rng = np.random.default_rng(11)
    beta = rng.normal(size=5) * 0.1  # identifiable coordinates
    alpha = rng.normal(size=2) * 0.1  # null coordinates (unobservable)
    u_star = subspace.v_id @ beta + subspace.v_null @ alpha
    r_w = a @ u_star + rng.normal(size=a.shape[0]) * 0.01

    g_native = np.zeros((2, 7))
    g_native[0, 2] = 1.0
    g_native[1, 6] = 1.0
    g_scaled = g_native @ np.diag(subspace.parameter_scales)
    result = solve_gauge_fixed_kkt(a, r_w, g_scaled)
    assert result["kkt_full_rank"] is True
    assert result["gauge_residual_max_abs"] < 1e-12
    assert result["solve_residual_relative"] < 1e-10
    # The identifiable projection is recovered up to the injected noise
    # (sigma=0.01 in weighted space, smallest singular value 20); the
    # gauge-fixed native parameters are exactly zero.
    assert np.allclose(subspace.v_id.T @ result["u_hat"], beta, atol=2e-3)
    theta_hat = np.diag(subspace.parameter_scales) @ result["u_hat"]
    assert abs(theta_hat[2]) < 1e-12
    assert abs(theta_hat[6]) < 1e-12


# ---------------------------------------------------------------------------
# Gauge-invariant observable closure
# ---------------------------------------------------------------------------


def test_closure_passes_on_synthetic_information():
    config = _config()
    subspace = _synthetic_subspace()
    extras = _synthetic_extras(subspace)
    candidates = build_gauge_candidates(config, subspace)
    report = gauge_invariant_closure(config, subspace, extras, candidates)
    assert report["pass"] is True, report["checks"]
    assert report["kkt_full_rank_everywhere"] is True
    # Gauge-dependent null coordinates must differ across gauges (expected).
    assert report["gauge_dependent_components_differ_as_expected"] is True
    assert report["full_truth_recovery_is_not_a_gate"] is True
    # JSON-serializable through the report driver path.
    json.dumps(json_ready(_strip_private(report)))


def test_closure_detects_gauge_violation():
    config = _config()
    subspace = _synthetic_subspace()
    extras = _synthetic_extras(subspace)
    candidates = build_gauge_candidates(config, subspace)
    # Tamper: replace one gauge matrix with a non-complementing constraint.
    broken = []
    for candidate in candidates:
        item = dict(candidate)
        if item["id"] == "minimum_norm_scaled_gauge":
            bad = np.zeros((2, 7))
            bad[0, 0] = 1.0  # dx = 0: identifiable direction
            bad[1, 1] = 1.0  # dy = 0: identifiable direction
            item["_g_native"] = bad
        broken.append(item)
    report = gauge_invariant_closure(config, subspace, extras, broken)
    assert report["checks"]["kkt_numerically_stable"] is False
    assert report["pass"] is False


# ---------------------------------------------------------------------------
# External-constraint eligibility
# ---------------------------------------------------------------------------


def test_eligibility_table_mechanical_rule():
    config = _config()
    table = external_eligibility_table(config)
    assert table["eligible_external_physical_constraints"] == []
    assert table["matches_preregistered_expectation"] is True
    assert table["no_ingestable_external_physical_constraint_available"] is True
    assert table["consistent_with_frozen_slot_artifacts"] is True
    assert table["fisher_or_posterior_ingest_ran"] is False
    by_id = {row["id"]: row for row in table["rows"]}
    # ift_C_dx has a validated mapping and independence, but no covariance
    # and no valid IOV: exactly two gates fail.
    cdx = by_id["ift_C_dx"]
    assert cdx["gates"]["parameter_mapping_validated"] is True
    assert cdx["gates"]["measurement_covariance_has_independent_provenance"] is False
    assert cdx["gates"]["measurement_year_conditions_iov_identified"] is False
    assert cdx["gates"]["independent_of_tracks_and_conditions"] is True
    assert cdx["eligible_physical_constraint"] is False
    # Conditions constants are not independent.
    assert by_id["tracker_align_conditions_station0_ry"]["gates"][
        "independent_of_tracks_and_conditions"
    ] is False
    json.dumps(json_ready(_strip_private(table)))


def test_eligibility_rule_would_admit_a_real_measurement():
    config = _config()
    row = {
        "id": "hypothetical_future_survey_ry",
        "value": 1.0,
        "unit": "mrad",
        "real_measurement_sigma_or_covariance": 0.5,
        "coordinate_frame": "survey_frame",
        "pivot_rotation_convention": "faser_origin",
        "calypso_parameter_mapping": "validated_as_station_ry",
        "measurement_date": "2026-01-01",
        "iov_mechanical_stability_provenance": "valid_iov:2024_conditions_with_stability_evidence",
        "independent_of_tracks_and_conditions": True,
        "evidence_source": "synthetic_test",
    }
    import copy

    config2 = copy.deepcopy(dict(config))
    config2["external_constraint_eligibility"]["audit_rows"] = [row]
    table = external_eligibility_table(config2)
    assert table["eligible_external_physical_constraints"] == ["hypothetical_future_survey_ry"]
    assert table["no_ingestable_external_physical_constraint_available"] is False


# ---------------------------------------------------------------------------
# Decision branches
# ---------------------------------------------------------------------------


def _passing_reports():
    config = _config()
    subspace = _synthetic_subspace()
    extras = _synthetic_extras(subspace)
    candidates = build_gauge_candidates(config, subspace)
    closure = gauge_invariant_closure(config, subspace, extras, candidates)
    regression = {"pass": True}
    eligibility = {"eligible_external_physical_constraints": []}
    return candidates, closure, regression, eligibility


def test_decision_gauge_feasibility_closed():
    candidates, closure, regression, eligibility = _passing_reports()
    decision = decide_campaign(
        regression=regression,
        candidates=candidates,
        closure_report=closure,
        eligibility=eligibility,
    )
    assert decision["decision"] == DECISION_GAUGE_FEASIBILITY_CLOSED
    assert decision["gauge_constraint_contract_valid"] is True
    assert decision["gauge_fixed_solver_well_posed"] is True
    assert decision["gauge_invariant_observable_closure"] is True
    assert decision["external_constraint_ingest_authorized"] is False
    assert decision["real_data_candidate_alignment_authorized"] is False
    assert decision["geometry_write_allowed"] is False
    assert decision["official_conditions_write_allowed"] is False
    json.dumps(json_ready(_strip_private(decision)))


def test_decision_regression_failure_short_circuits():
    candidates, closure, _, eligibility = _passing_reports()
    decision = decide_campaign(
        regression={"pass": False},
        candidates=candidates,
        closure_report=closure,
        eligibility=eligibility,
    )
    assert decision["decision"] == DECISION_TRACKER_REGRESSION_FAILED


def test_decision_contract_invalid():
    _, closure, regression, eligibility = _passing_reports()
    bad = [{"valid": False}]
    decision = decide_campaign(
        regression=regression,
        candidates=bad,
        closure_report=closure,
        eligibility=eligibility,
    )
    assert decision["decision"] == DECISION_GAUGE_CONTRACT_INVALID


def test_decision_closure_failed():
    candidates, closure, regression, eligibility = _passing_reports()
    broken_closure = dict(closure)
    broken_closure["pass"] = False
    broken_checks = dict(closure["checks"])
    broken_checks["observable_prediction_gauge_invariant"] = False
    broken_closure["checks"] = broken_checks
    decision = decide_campaign(
        regression=regression,
        candidates=candidates,
        closure_report=broken_closure,
        eligibility=eligibility,
    )
    assert decision["decision"] == DECISION_CLOSURE_FAILED


def test_decision_external_candidate_found():
    candidates, closure, regression, _ = _passing_reports()
    decision = decide_campaign(
        regression=regression,
        candidates=candidates,
        closure_report=closure,
        eligibility={"eligible_external_physical_constraints": ["future_survey"]},
    )
    assert decision["decision"] == DECISION_EXTERNAL_CANDIDATE_FOUND
    # Even then, this stage never authorizes ingest or real-data alignment.
    assert decision["external_constraint_ingest_authorized"] is False
    assert decision["real_data_candidate_alignment_authorized"] is False
