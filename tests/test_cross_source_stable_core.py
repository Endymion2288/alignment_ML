from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alignment.cross_source_stable_core import (
    DECISION_HYPOTHESIS_FAIL,
    DECISION_INDEPENDENT_FAIL,
    DECISION_INDEPENDENT_PASS,
    INHERITED_V1_DECISION,
    SCHEMA_VERSION,
    build_all_reports,
    decide_campaign,
    independent_core_gate,
    load_campaign_config,
    source_bootstrap_core_stability,
)
from alignment.identifiable_subspace import (
    FROZEN_RANK_TOLERANCE,
    ForcedRankError,
    consensus_operator,
    core_contained_in_projector,
    identifiable_subspace_from_core,
    quadratic_persistence,
    refuse_forced_identifiable_rank,
    select_stable_core,
    signed_eigh_descending,
)
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
)
from alignment.tracker_only_identifiable_subspace import DECISION_BASIS_UNSTABLE
from alignment.true_cluster_local_residual import assert_no_alignment_payload


def _synthetic_derivative():
    derivative = np.zeros((6, 4, 3), dtype=np.float64)
    derivative[:, 0, 0] = 1.0
    derivative[:, 1, 1] = 0.5
    derivative[:, 0, 2] = 1.0
    return derivative


def _synthetic_bank(derivative, *, source_id="s0", split="train", scales=(5.0, 60.0, 0.12)):
    pairs, _dim, parameters = derivative.shape
    covariance = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], pairs, axis=0)
    nominal = np.zeros((pairs, 4), dtype=np.float64)
    step = 1.0
    positive = np.zeros((parameters, pairs, 4), dtype=np.float64)
    negative = np.zeros((parameters, pairs, 4), dtype=np.float64)
    for index in range(parameters):
        positive[index] = nominal + step * derivative[:, :, index]
        negative[index] = nominal - step * derivative[:, :, index]
    names = ("ift_dx_mm", "ift_ry_mrad", "C_dx")[:parameters]
    run_offset = 1000 * (sum(ord(char) for char in source_id) + 1)
    return {
        "source_id": source_id,
        "split": split,
        "specs": (),
        "names": names,
        "scales": np.asarray(scales[:parameters], dtype=np.float64),
        "anchor_values": np.zeros(parameters, dtype=np.float64),
        "reference_values": np.zeros(parameters, dtype=np.float64),
        "target_names": (),
        "target_values": {},
        "target_residuals": {},
        "positive_values": np.full(parameters, step, dtype=np.float64),
        "negative_values": np.full(parameters, -step, dtype=np.float64),
        "anchor_residual": nominal,
        "positive_residual": positive,
        "negative_residual": negative,
        "reference_residual": np.array(nominal, copy=True),
        "covariance": covariance,
        "run_id": np.arange(pairs, dtype=np.int64) + run_offset,
        "event_id": np.arange(pairs, dtype=np.int64),
        "truth_particle_id": np.ones(pairs, dtype=np.int64),
        "source_station_id": np.zeros(pairs, dtype=np.int64),
        "target_station_id": np.ones(pairs, dtype=np.int64),
        "layer_weights": np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
        "overlap": {"pairs": int(pairs)},
        "test_data_accessed": False,
    }


def _config():
    return load_campaign_config(
        Path("configs/cross_source_stable_core_identifiable_subspace_v1.yaml")
    )


def test_config_inherits_v1_failure_and_forbids_retune():
    config = _config()
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["inherited_v1_decision"] == INHERITED_V1_DECISION == DECISION_BASIS_UNSTABLE
    assert config["real_data_operating_mode"] == OPERATING_MODE
    assert config["frozen_v2_checkpoint_sha256"] == FROZEN_V2_CHECKPOINT_SHA256
    assert config["geometry_write_allowed"] is False
    assert config["rank_tolerance"] == pytest.approx(FROZEN_RANK_TOLERANCE)
    assert config["scale_matrix_S"]["ift_dx_mm"] == pytest.approx(5.0)
    assert config["scale_matrix_S"]["ift_ry_mrad"] == pytest.approx(60.0)
    assert config["scale_matrix_S"]["C_dx"] == pytest.approx(0.12)
    assert config["do_not_force_identifiable_rank_five"] is True
    assert config["do_not_truncate_rank_six_source"] is True
    assert config["do_not_drop_v1_failure_source"] is True
    assert config["hypothesis_sources_are_not_confirmatory_evidence"] is True
    assert config["algorithm"]["forced_core_dimension"] is None
    assert "mc24_100047_00150_00199" in config["jacobian_corpus"]["hypothesis_source_ids"]
    assert "mc24_100047_00150_00199" not in config["jacobian_corpus"]["independent_validation_source_ids"]
    overlap = set(config["jacobian_corpus"]["hypothesis_source_ids"]) & set(
        config["jacobian_corpus"]["independent_validation_source_ids"]
    )
    assert not overlap
    assert config["do_not_open_sealed_test"] is True
    assert config["do_not_remove_dz_to_improve_this_svd"] is True
    assert config["do_not_merge_cluster_local_repair_into_stable_core_gate"] is True
    assert config["unknown_association_frozen_v2_closure_opened"] is False


def test_refuses_to_truncate_rank_six_projector():
    with pytest.raises(ForcedRankError):
        refuse_forced_identifiable_rank(native_rank=6, requested_rank=5)


def test_consensus_operator_is_equal_weight_average_of_projectors():
    a = np.diag([1.0, 1.0, 0.0])
    b = np.diag([1.0, 1.0, 1.0])
    consensus = consensus_operator([a, b])
    assert np.allclose(consensus, np.diag([1.0, 1.0, 0.5]))
    values, vectors, signs = signed_eigh_descending(consensus)
    assert values[0] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(0.5)
    assert all(int(sign) in (-1, 1) for sign in signs)
    leading = int(np.argmax(np.abs(vectors[:, 0])))
    assert vectors[leading, 0] > 0.0


def test_select_stable_core_keeps_persistent_prefix_and_isolates_marginal():
    shared = np.eye(3)[:, :2]
    p_shared = shared @ shared.T
    extra = np.eye(3)[:, :3]
    p_extra = extra @ extra.T
    consensus = consensus_operator([p_shared, p_shared, p_extra])
    values, vectors, _signs = signed_eigh_descending(consensus)
    selected = select_stable_core(
        values,
        vectors,
        [p_shared, p_shared, p_extra],
        min_consensus_eigenvalue=0.70,
        min_mode_persistence=0.85,
        min_source_support_fraction=0.80,
    )
    assert selected["core_dimension"] == 2
    assert selected["labels"][2] != "core"
    assert selected["forced_core_dimension"] is None
    assert selected["not_forced_to_rank_five"] is True
    persist = quadratic_persistence(vectors[:, 2], p_shared)
    assert persist == pytest.approx(0.0, abs=1.0e-12)


def test_core_contained_in_higher_rank_projector_has_small_missing_norm():
    core = np.eye(3)[:, :2]
    full = np.eye(3)
    contained = core_contained_in_projector(core, full)
    assert contained["min_mode_persistence"] == pytest.approx(1.0)
    assert contained["max_principal_angle_deg"] == pytest.approx(0.0, abs=1.0e-8)
    assert contained["missing_projector_frobenius"] == pytest.approx(0.0, abs=1.0e-8)


def test_independent_gate_does_not_use_hypothesis_sources_as_confirmation():
    core = np.eye(3)[:, :2]
    projector = np.eye(3)
    rows = [
        {
            "source_id": "unseen_a",
            "split": "train",
            "identifiable_rank": 3,
            "projector": projector,
        }
    ]
    hypothesis = {"v_core": core}
    result = independent_core_gate(
        rows,
        hypothesis,
        min_mode_persistence=0.85,
        min_source_support_fraction=0.80,
        max_principal_angle_deg=15.0,
        max_missing_frobenius=0.75,
    )
    assert result["pass"] is True
    assert result["hypothesis_sources_are_not_confirmatory_evidence"] is True
    assert result["sealed_test_not_opened"] is True


def test_decide_campaign_never_authorizes_v2_or_geometry():
    failed = decide_campaign(
        hypothesis_core_dimension=4,
        hypothesis_stable=False,
        independent={"pass": True},
        three_arm=None,
    )
    assert failed["decision"] == DECISION_HYPOTHESIS_FAIL
    assert failed["inherited_v1_decision"] == DECISION_BASIS_UNSTABLE
    assert failed["independent_validation_pass"] is True
    assert failed["three_arm_authorized"] is False
    assert failed["authorize_frozen_v2_unknown_association_closure"] is False
    assert failed["geometry_write_allowed"] is False
    assert failed["if_failed_do_not_chase_by_retuning"] is True
    independent_fail = decide_campaign(
        hypothesis_core_dimension=4,
        hypothesis_stable=True,
        independent={"pass": False},
        three_arm=None,
    )
    assert independent_fail["decision"] == DECISION_INDEPENDENT_FAIL
    assert independent_fail["three_arm_authorized"] is False
    passed = decide_campaign(
        hypothesis_core_dimension=4,
        hypothesis_stable=True,
        independent={"pass": True},
        three_arm={"pass": True, "null_injection_leakage_gate": True, "mixed_injection_projected_closure": True},
    )
    assert passed["decision"] == DECISION_INDEPENDENT_PASS
    assert passed["three_arm_authorized"] is True
    assert passed["authorize_frozen_v2_unknown_association_closure"] is False
    assert passed["used_v1_pooled_v_id"] is False


def test_build_all_reports_on_synthetic_disjoint_hypothesis_and_validation():
    derivative = _synthetic_derivative()
    hypothesis = [
        _synthetic_bank(derivative, source_id="hyp_a", split="train"),
        _synthetic_bank(derivative * 1.01, source_id="hyp_b", split="train"),
        _synthetic_bank(derivative, source_id="mc24_100047_00150_00199", split="validation"),
    ]
    independent = [
        _synthetic_bank(derivative * 0.99, source_id="val_a", split="validation"),
        _synthetic_bank(derivative * 1.02, source_id="val_b", split="train"),
    ]
    config = _config()
    config["injections"]["n_replicates"] = 3
    config["bootstrap"]["n_source_replicates"] = 8
    config["bootstrap"]["n_event_replicates_per_source"] = 3
    config["bootstrap"]["min_pairs"] = 2
    reports = build_all_reports(
        config,
        banks_by_role={"hypothesis": hypothesis, "independent": independent},
    )
    assert_no_alignment_payload(reports["next_stage_decision"])
    assert reports["hypothesis_core"]["core_dimension"] == 2
    assert reports["hypothesis_core"]["not_forced_to_rank_five"] is True
    assert reports["independent_validation"]["pass"] is True
    assert reports["three_arm_closure"]["used_v1_pooled_v_id"] is False
    assert reports["three_arm_closure"]["solver_basis"] == "frozen_v_core"
    assert reports["next_stage_decision"]["decision"] == DECISION_INDEPENDENT_PASS
    assert reports["next_stage_decision"]["authorize_frozen_v2_unknown_association_closure"] is False
    assert reports["parameter_definition"]["operating_state"]["inherited_v1_decision"] == DECISION_BASIS_UNSTABLE
    assert reports["source_bootstrap"]["kind"] == "source_with_replacement_consensus"
    assert reports["event_bootstrap"]["kind"] == "event_bootstrap_core_persistence"
    assert reports["source_bootstrap"]["n_replicates_requested"] == int(config["bootstrap"]["n_source_replicates"])


def test_source_bootstrap_compares_consensus_core_not_signed_vector_elements():
    derivative = _synthetic_derivative()
    banks = [
        _synthetic_bank(derivative, source_id="boot_a", split="train"),
        _synthetic_bank(derivative * 1.01, source_id="boot_b", split="train"),
        _synthetic_bank(derivative, source_id="boot_c", split="validation"),
    ]
    from alignment.cross_source_stable_core import source_projectors, build_core_from_rows

    rows = source_projectors(banks, rank_tolerance=0.01, rcond=1.0e-10)
    core = build_core_from_rows(
        rows,
        min_consensus_eigenvalue=0.70,
        min_mode_persistence=0.85,
        min_source_support_fraction=0.80,
    )
    boot = source_bootstrap_core_stability(
        rows,
        core,
        n_replicates=8,
        seed=20260902,
        min_consensus_eigenvalue=0.70,
        min_mode_persistence=0.85,
        min_source_support_fraction=0.80,
        max_principal_angle_deg=20.0,
        max_projector_frobenius=1.2,
    )
    assert boot["stable"] is True
    assert boot["n_replicates_kept"] == 8
    assert all(row["compares_subspace_not_signed_vector_elements"] for row in boot["comparisons"])


def test_core_solver_wrap_keeps_orthogonal_as_gauge_not_measurement():
    names = ("ift_dx_mm", "ift_ry_mrad", "C_dx")
    scales = np.array([5.0, 60.0, 0.12])
    core = np.eye(3)[:, :2]
    orthogonal = np.eye(3)[:, 2:]
    space = identifiable_subspace_from_core(
        parameter_names=names,
        parameter_units=("mm", "mrad", "mm"),
        parameter_scales=scales,
        v_core=core,
        v_orthogonal=orthogonal,
    )
    assert space.identifiable_rank == 2
    assert space.null_dimension == 1
    from alignment.identifiable_subspace import inject_identifiable, inject_null, native_to_scaled

    mixed = inject_identifiable(space, [0.1, -0.05]) + inject_null(space, [0.2])
    hat = native_to_scaled(mixed, scales)
    # The wrap itself is a basis; gauge zero is enforced by the restricted solver, not by this constructor.
    assert space.v_null.shape == (3, 1)
    assert not np.allclose(hat, native_to_scaled(inject_identifiable(space, [0.1, -0.05]), scales))
