from __future__ import annotations

import numpy as np
import pytest

from alignment.gauge_equivalence import (
    CANONICAL_INTERNAL_BASIS,
    IFT_PLANE_Z_MM,
    additive_element_six,
    additive_elements_match,
    audit_outer_contrast_equivalence,
    composed_elements_match,
    conjugate_to_plane_z,
    detector_element_global_delta,
    frozen_station_reference_layer_payload,
    outer_contrast_payload,
    reference_layer_additive_equivalent,
    six_vector_with_component,
    transform_point,
)
from alignment.layer_hierarchy import (
    CANONICAL_INTERNAL_BASIS,
    IFT_LAYER_IDS,
    expand_gauged_parameters,
    project_to_gauge,
    reduce_gauge,
)


def _rx_specs():
    specs = []
    for component in ("dx_mm", "dy_mm", "rx_mrad", "ry_mrad", "rz_mrad"):
        unit = "mm" if component.endswith("mm") else "mrad"
        specs.append(
            {
                "name": f"ift_{component}",
                "scope": "station",
                "station_id": 0,
                "component": component,
                "unit": unit,
            }
        )
    for layer in IFT_LAYER_IDS:
        specs.append(
            {
                "name": f"ift_layer{layer}_rx_mrad",
                "scope": "layer",
                "station_id": 0,
                "layer_id": layer,
                "component": "rx_mrad",
                "unit": "mrad",
            }
        )
    return specs


def test_analytic_map_is_subtract_c_from_layers_and_add_c_to_station():
    contrast = 0.7
    physical = outer_contrast_payload("rx_mrad", contrast)
    compensated = reference_layer_additive_equivalent("rx_mrad", contrast)
    frozen = frozen_station_reference_layer_payload("rx_mrad", contrast)
    assert physical["station"] == pytest.approx((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert physical["layers"][0][3] == pytest.approx(0.0007)
    assert physical["layers"][1][3] == pytest.approx(0.0)
    assert physical["layers"][2][3] == pytest.approx(-0.0007)
    assert compensated["station"][3] == pytest.approx(0.0007)
    assert compensated["layers"][0][3] == pytest.approx(0.0)
    assert compensated["layers"][1][3] == pytest.approx(-0.0007)
    assert compensated["layers"][2][3] == pytest.approx(-0.0014)
    assert frozen["station"][3] == pytest.approx(0.0)
    assert frozen["layers"][2][3] == pytest.approx(-0.0014)
    assert additive_elements_match(physical, compensated)
    assert not additive_elements_match(physical, frozen)
    # Frozen-station reference-layer cannot represent [+C, 0, -C] at all.
    for layer in IFT_LAYER_IDS:
        element = additive_element_six(frozen["station"], frozen["layers"][layer])
        assert element[3] == pytest.approx(frozen["layers"][layer][3])
    assert additive_element_six(frozen["station"], frozen["layers"][0])[3] == pytest.approx(0.0)


def test_project_to_gauge_is_the_same_additive_map_when_station_is_free():
    specs = _rx_specs()
    values = {spec["name"]: 0.0 for spec in specs}
    values["ift_layer0_rx_mrad"] = 0.7
    values["ift_layer2_rx_mrad"] = -0.7
    reduction = reduce_gauge(specs, choice="reference_layer", reference_layer=0)
    projected = project_to_gauge(values, reduction, specs)
    full = expand_gauged_parameters(projected, reduction, specs)
    assert projected["ift_rx_mrad"] == pytest.approx(0.7)
    assert full["ift_layer0_rx_mrad"] == pytest.approx(0.0)
    assert full["ift_layer1_rx_mrad"] == pytest.approx(-0.7)
    assert full["ift_layer2_rx_mrad"] == pytest.approx(-1.4)
    contrast = reduce_gauge(specs, choice="outer_contrast")
    assert project_to_gauge(values, contrast, specs)["C_rx"] == pytest.approx(0.7)


def test_dx_calypso_elements_match_only_after_station_compensation():
    physical = outer_contrast_payload("dx_mm", 0.12)
    compensated = reference_layer_additive_equivalent("dx_mm", 0.12)
    frozen = frozen_station_reference_layer_payload("dx_mm", 0.12)
    assert composed_elements_match(physical, compensated)
    assert not composed_elements_match(physical, frozen)
    report = audit_outer_contrast_equivalence("dx_mm", 0.12)
    assert report["additive_station_plus_layer"]["contrast_equals_compensated_reference_layer"] is True
    assert report["additive_station_plus_layer"]["contrast_equals_frozen_station_reference_layer"] is False
    assert report["calypso_detector_element_global_delta"]["contrast_equals_compensated_reference_layer"] is True
    assert report["conclusion"]["canonical_internal_basis"] == CANONICAL_INTERNAL_BASIS


def test_rx_calypso_elements_do_not_match_even_with_station_compensation():
    physical = outer_contrast_payload("rx_mrad", 0.7)
    compensated = reference_layer_additive_equivalent("rx_mrad", 0.7)
    frozen = frozen_station_reference_layer_payload("rx_mrad", 0.7)
    assert additive_elements_match(physical, compensated)
    assert not composed_elements_match(physical, compensated)
    assert not composed_elements_match(physical, frozen)
    report = audit_outer_contrast_equivalence("rx_mrad", 0.7)
    assert report["calypso_detector_element_global_delta"]["pivot_or_conjugation_limits_rotation_gauge"] is True
    # Station rx about the origin versus conjugated layer rx about plane z
    # differs by a dy lever arm of order C * z ~ 1.3 mm.
    layer0 = report["calypso_detector_element_global_delta"]["compensated_mismatch"]["layer_0"]
    assert abs(layer0["translation_difference_mm"][1]) > 1.0
    assert layer0["matrices_equal"] is False


def test_plane_z_conjugation_leaves_pure_translation_invariant():
    translation = six_vector_with_component("dx_mm", 0.12)
    from alignment.gauge_equivalence import calypso_alignment_matrix

    raw = calypso_alignment_matrix(*translation)
    conjugated = conjugate_to_plane_z(raw, IFT_PLANE_Z_MM[0])
    assert np.allclose(raw, conjugated)


def test_composed_rx_rotates_about_plane_origin_not_global_origin():
    layer = six_vector_with_component("rx_mrad", 0.7)
    plane_z = float(IFT_PLANE_Z_MM[0])
    delta = detector_element_global_delta((0.0,) * 6, layer, plane_z)
    origin = transform_point(delta, (0.0, 0.0, plane_z))
    # A rotation about the plane origin must leave the plane origin fixed.
    assert origin == pytest.approx((0.0, 0.0, plane_z), abs=1.0e-9)
    station_only = detector_element_global_delta(layer, (0.0,) * 6, plane_z)
    station_image = transform_point(station_only, (0.0, 0.0, plane_z))
    # Station rx about the global origin moves the plane origin in y.
    assert abs(float(station_image[1])) > 1.0
