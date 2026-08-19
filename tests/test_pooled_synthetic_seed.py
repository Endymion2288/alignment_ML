from scripts.materialize_pooled_curriculum_synthetics import (
    _namespace_base,
    _stable_seed,
    _synthetic_run_id,
)


def test_direction_controlled_overlay_seed_is_shared_only_within_magnitude():
    first = _stable_seed(
        20260812,
        "test",
        "mag_5_test_00",
        5.0,
        "magnitude_shared_across_direction_trials",
    )
    second = _stable_seed(
        20260812,
        "test",
        "mag_5_test_02",
        5.0,
        "magnitude_shared_across_direction_trials",
    )
    other_magnitude = _stable_seed(
        20260812,
        "test",
        "mag_10_test_00",
        10.0,
        "magnitude_shared_across_direction_trials",
    )

    assert first == second
    assert first != other_magnitude


def test_legacy_payload_seed_scope_remains_payload_specific():
    first = _stable_seed(1, "test", "mag_5_test_00", 5.0, "payload")
    second = _stable_seed(1, "test", "mag_5_test_01", 5.0, "payload")

    assert first != second


def test_alignment_iteration_overlay_seed_is_shared_across_physical_probes():
    anchor = _stable_seed(
        20260815,
        "validation",
        "iteration_00_anchor",
        "ift_dx_dy_ry_joint_l2",
        1.0,
        "alignment_iteration_shared_across_payloads",
    )
    derivative_probe = _stable_seed(
        20260815,
        "validation",
        "iteration_00_fd_ift_ry_mrad_p",
        "ift_dx_dy_ry_joint_l2",
        2.0,
        "alignment_iteration_shared_across_payloads",
    )
    other_split = _stable_seed(
        20260815,
        "train",
        "iteration_00_fd_ift_ry_mrad_p",
        "ift_dx_dy_ry_joint_l2",
        2.0,
        "alignment_iteration_shared_across_payloads",
    )

    assert anchor == derivative_probe
    assert anchor != other_split


def test_legacy_magnitude_identity_reproduces_v3_production_seed():
    # The v3 expanded trainval overlay was materialized before the condition
    # axis entered the seed identity; the legacy magnitude-only identity must
    # keep reproducing its recorded per-sample seeds.
    legacy = _stable_seed(
        20260813,
        "train",
        "mag_0_train_00",
        0.0,
        "magnitude_shared_across_direction_trials",
    )
    assert legacy == 876273998
    explicit_axis = _stable_seed(
        20260813,
        "train",
        "mag_0_train_00",
        "translation_xy_mm",
        0.0,
        "magnitude_shared_across_direction_trials",
    )
    assert explicit_axis != legacy


def test_alignment_iteration_namespace_is_shared_across_payloads():
    scope = "alignment_iteration_shared_across_payloads"
    anchor = _namespace_base(0, 2, scope)
    probe = _namespace_base(7, 2, scope)
    assert anchor == probe
    assert anchor != _namespace_base(0, 3, scope)
    # The default payload scope keeps payload-specific namespaces.
    assert _namespace_base(0, 2, "payload") != _namespace_base(7, 2, "payload")


def test_alignment_iteration_synthetic_run_id_is_shared_across_payloads():
    scope = "alignment_iteration_shared_across_payloads"
    assert _synthetic_run_id(996000, 0, scope) == _synthetic_run_id(996000, 7, scope)
    assert _synthetic_run_id(996000, 0, "payload") != _synthetic_run_id(996000, 7, "payload")
