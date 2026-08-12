from scripts.materialize_pooled_curriculum_synthetics import _stable_seed


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
