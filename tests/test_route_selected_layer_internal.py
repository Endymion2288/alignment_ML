from __future__ import annotations

import numpy as np
import pytest

from alignment.layer_hierarchy import reduce_gauge
from scripts.run_route_selected_multidof_update import (
    _dx_internals,
    _gauged_finite_difference_fit,
    _point_parameter_values,
    _select_parameter_specs,
    _six_vectors_close,
)


def _layer_dx_specs():
    return [
        {
            "name": f"ift_layer{layer}_dx_mm",
            "scope": "layer",
            "station_id": 0,
            "layer_id": layer,
            "component": "dx_mm",
            "unit": "mm",
            "severity_scale": 5.0,
        }
        for layer in (0, 1, 2)
    ]


def test_select_parameter_specs_keeps_requested_order_and_rejects_unknown():
    specs = _layer_dx_specs() + [
        {
            "name": "ift_dx_mm",
            "scope": "station",
            "station_id": 0,
            "component": "dx_mm",
            "unit": "mm",
            "severity_scale": 5.0,
        }
    ]
    selected = _select_parameter_specs(
        specs, ("ift_layer0_dx_mm", "ift_layer1_dx_mm", "ift_layer2_dx_mm")
    )
    assert [spec["name"] for spec in selected] == [
        "ift_layer0_dx_mm",
        "ift_layer1_dx_mm",
        "ift_layer2_dx_mm",
    ]
    with pytest.raises(ValueError, match="unknown --only-parameters"):
        _select_parameter_specs(specs, ("ift_layer0_dy_mm",))
    with pytest.raises(ValueError, match="duplicated"):
        _select_parameter_specs(specs, ("ift_layer0_dx_mm", "ift_layer0_dx_mm"))


def test_point_parameter_values_read_layer_payload_not_station_slot():
    specs = _layer_dx_specs()
    point = {
        "name": "iteration_00_closure_relative_dx",
        "injected_station_transforms": {station: [0.0] * 6 for station in range(4)},
        "injected_layer_transforms": {
            "0": {
                "0": [0.12, 0.0, 0.0, 0.0, 0.0, 0.0],
                "1": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "2": [-0.12, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        },
    }
    values = _point_parameter_values(specs, point)
    assert values["ift_layer0_dx_mm"] == pytest.approx(0.12)
    assert values["ift_layer1_dx_mm"] == pytest.approx(0.0)
    assert values["ift_layer2_dx_mm"] == pytest.approx(-0.12)


def test_gauged_finite_difference_recovers_outer_relative_dx_not_labels():
    specs = _layer_dx_specs()
    names = [spec["name"] for spec in specs]
    rng = np.random.default_rng(0)
    pairs = 24
    template = rng.normal(size=(pairs, 4))
    template /= np.linalg.norm(template)
    weak = rng.normal(size=(pairs, 4)) * 0.02
    identity = np.eye(4)
    covariance = np.repeat(identity[None, :, :], pairs, axis=0)
    anchor = np.zeros((pairs, 4))
    step = 0.2
    # Layer0 and layer2 are antisymmetric; layer1 is weakly visible.
    effects = [template, weak, -template]
    positive = np.stack([anchor + step * effect for effect in effects])
    negative = np.stack([anchor - step * effect for effect in effects])
    expected = np.asarray([0.12, 0.0, -0.12])
    target = anchor + sum(value * effect for value, effect in zip(expected, effects))
    reduction = reduce_gauge(specs, choice="sum_to_zero")
    _unconstrained, gauged = _gauged_finite_difference_fit(
        anchor_residual=anchor,
        positive_residual=positive,
        negative_residual=negative,
        target_residual=target,
        covariance=covariance,
        names=names,
        positive_values=np.full(3, step),
        negative_values=np.full(3, -step),
        scales=np.full(3, 5.0),
        prior=None,
        rcond=1.0e-10,
        reduction=reduction,
    )
    from alignment.layer_hierarchy import expand_gauged_parameters

    recovered_full = expand_gauged_parameters(
        {name: float(gauged.recovered_parameters[index]) for index, name in enumerate(gauged.parameter_names)},
        reduction,
        specs,
    )
    internals = _dx_internals(recovered_full, specs)
    expected_split = _dx_internals(dict(zip(names, expected)), specs)
    assert gauged.full_rank
    assert internals["outer_relative_layer0_minus_layer2_mm"] == pytest.approx(
        expected_split["outer_relative_layer0_minus_layer2_mm"], abs=1.0e-3
    )
    assert internals["layer_internal_dx_mm"]["layer_0"] == pytest.approx(0.12, abs=5.0e-3)
    assert internals["layer_internal_dx_mm"]["layer_2"] == pytest.approx(-0.12, abs=5.0e-3)
    assert internals["station_common"]["dx_mm"] == pytest.approx(0.0)


def test_station_payload_stays_fixed_when_only_layers_move():
    left = {str(station): [0.0] * 6 for station in range(4)}
    right = {str(station): [0.0] * 6 for station in range(4)}
    assert _six_vectors_close(left, right)
    right["0"] = [0.12, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert not _six_vectors_close(left, right)
