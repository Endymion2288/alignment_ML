from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from alignment.layer_hierarchy import (
    FREE_COMPONENTS,
    IFT_LAYER_IDS,
    calypso_layer_key,
    expand_gauged_parameters,
    hierarchy_index,
    parse_calypso_layer_key,
    project_to_gauge,
    reduce_gauge,
    split_common_and_internal,
)
from alignment.payload import load_station_alignment_payload, load_station_rigid_alignment_payload
from alignment.physical_jacobian import (
    parameter_values_from_payload,
    parameter_values_from_station_transforms,
    payload_transforms_with_parameter_values,
    station_transforms_with_parameter_values,
)
from scripts.prepare_layer_identifiability_pilot import compile_layer_identifiability_pilot
from scripts.run_layer_linear_internal_closure import physics_verdict
from scripts.run_physical_refit_capture_scan import _baseline_refit_dir, _build_plan
from scripts.write_station_alignment_payload import parse_layer_transform, parse_transform


def _hierarchy_specs():
    specs = []
    for component, unit, step, scale in (
        ("dx_mm", "mm", 0.5, 5.0),
        ("dy_mm", "mm", 0.5, 5.0),
        ("rx_mrad", "mrad", 10.0, 60.0),
        ("ry_mrad", "mrad", 10.0, 60.0),
        ("rz_mrad", "mrad", 10.0, 60.0),
    ):
        specs.append(
            {
                "name": f"ift_{component}",
                "scope": "station",
                "station_id": 0,
                "component": component,
                "unit": unit,
                "finite_difference_step": step,
                "severity_scale": scale,
            }
        )
        for layer in IFT_LAYER_IDS:
            specs.append(
                {
                    "name": f"ift_layer{layer}_{component}",
                    "scope": "layer",
                    "station_id": 0,
                    "layer_id": layer,
                    "component": component,
                    "unit": unit,
                    "finite_difference_step": 0.2 if unit == "mm" else 2.0,
                    "severity_scale": scale,
                }
            )
    return specs


def _identity_point(name="nominal", role="nominal"):
    return {
        "name": name,
        "point_role": role,
        "direction_trial": name,
        "station_transforms": {station: [0.0] * 6 for station in range(4)},
        "layer_transforms": {0: {layer: [0.0] * 6 for layer in IFT_LAYER_IDS}},
    }


def _fd_point(spec, sign):
    suffix = "p" if sign > 0 else "m"
    values = {item["name"]: 0.0 for item in _hierarchy_specs()}
    values[spec["name"]] = sign * float(spec["finite_difference_step"])
    stations, layers = payload_transforms_with_parameter_values(
        _hierarchy_specs(),
        {station: [0.0] * 6 for station in range(4)},
        {0: {layer: [0.0] * 6 for layer in IFT_LAYER_IDS}},
        values,
    )
    return {
        "name": f"fd_{spec['name']}_{suffix}",
        "point_role": "finite_difference_positive" if sign > 0 else "finite_difference_negative",
        "direction_trial": f"fd_{spec['name']}_{suffix}",
        "finite_difference_for": spec["name"],
        "probe_sign": "positive" if sign > 0 else "negative",
        "finite_difference_anchor": "nominal",
        "alignment_parameter_values": values,
        "station_transforms": stations,
        "layer_transforms": layers,
    }


def _hierarchy_config(points=None):
    specs = _hierarchy_specs()
    if points is None:
        points = [_identity_point()]
        for spec in specs:
            points.append(_fd_point(spec, 1.0))
            points.append(_fd_point(spec, -1.0))
    return {
        "scan_mode": "ift_layer_hierarchy",
        "station_ids": [0, 1, 2, 3],
        "reference_station_ids": [1, 2, 3],
        "movable_station_ids": [0],
        "movable_layer_ids": [0, 1, 2],
        "q_over_p_mode": 0,
        "condition_axis": "ift_station_layer_hierarchy_l2",
        "alignment_parameter_specs": specs,
        "rigid_points": points,
    }


def _manifest(tmp_path, constants):
    sqlite = tmp_path / "tracker_alignment.sqlite"
    pool = tmp_path / "tracker_alignment.pool.root"
    catalog = tmp_path / "PoolFileCatalog.xml"
    for path in (sqlite, pool, catalog):
        path.write_bytes(b"")
    manifest = tmp_path / "alignment_payload.json"
    manifest.write_text(
        json.dumps(
            {
                "folder": "/Tracker/Align",
                "station_transform_convention": {
                    "frame": "global",
                    "components": "[dx_mm, dy_mm, dz_mm, rx_rad, ry_rad, rz_rad]",
                },
                "alignment_constants": constants,
                "sqlite": str(sqlite),
                "pool": str(pool),
                "pool_catalog": str(catalog),
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_calypso_layer_keys_match_tracker_align_db_tool():
    assert calypso_layer_key(0, 0) == "00"
    assert calypso_layer_key(0, 2) == "02"
    assert parse_calypso_layer_key("12") == (1, 2)
    with pytest.raises(ValueError):
        calypso_layer_key(0, 3)
    with pytest.raises(ValueError):
        parse_calypso_layer_key("000")
    with pytest.raises(ValueError):
        parse_calypso_layer_key("station:0")


def test_parse_layer_transform_uses_station_layer_mm_rad():
    station, layer, transform = parse_layer_transform("0:1:0.2:-0.1:0:0:0.002:0")
    assert (station, layer) == (0, 1)
    assert transform == (0.2, -0.1, 0.0, 0.0, 0.002, 0.0)
    with pytest.raises(Exception):
        parse_layer_transform("0:0.2:0:0:0:0:0")
    station, transform = parse_transform("0:1:0:0:0:0:0")
    assert station == 0


def test_payload_loader_reads_layer_keys_and_v1_rejects_them(tmp_path):
    manifest = _manifest(
        tmp_path,
        {
            "station:0": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            "00": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            "01": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "02": [-0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    )
    rigid = load_station_rigid_alignment_payload(manifest)
    assert rigid.transform_for_station(0)[0] == 0.1
    assert rigid.transform_for_layer(0, 0)[0] == 0.2
    assert rigid.transform_for_layer(0, 2)[0] == -0.2
    with pytest.raises(ValueError, match="layer transforms"):
        load_station_alignment_payload(manifest)


def test_station_helpers_refuse_to_map_layer_scope_onto_station_payload():
    specs = _hierarchy_specs()
    stations = {0: [0.0] * 6}
    with pytest.raises(ValueError, match="not station-scoped"):
        parameter_values_from_station_transforms(specs, stations)
    with pytest.raises(ValueError, match="not station-scoped"):
        station_transforms_with_parameter_values(specs, stations, {spec["name"]: 0.0 for spec in specs})


def test_payload_helpers_round_trip_station_and_layer_values():
    specs = _hierarchy_specs()
    values = {spec["name"]: 0.0 for spec in specs}
    values["ift_dx_mm"] = 0.15
    values["ift_layer0_dx_mm"] = 0.12
    values["ift_layer2_dx_mm"] = -0.12
    values["ift_layer0_ry_mrad"] = 1.2
    stations, layers = payload_transforms_with_parameter_values(
        specs,
        {station: [0.0] * 6 for station in range(4)},
        {0: {layer: [0.0] * 6 for layer in IFT_LAYER_IDS}},
        values,
    )
    assert stations["0"][0] == pytest.approx(0.15)
    assert layers["0"]["0"][0] == pytest.approx(0.12)
    assert layers["0"]["2"][0] == pytest.approx(-0.12)
    assert layers["0"]["0"][4] == pytest.approx(0.0012)
    recovered = parameter_values_from_payload(specs, stations, layers)
    assert recovered["ift_layer0_ry_mrad"] == pytest.approx(1.2)


def test_sum_to_zero_and_reference_layer_gauges_agree_on_internal_deformation():
    specs = _hierarchy_specs()
    values = {spec["name"]: 0.0 for spec in specs}
    values["ift_dx_mm"] = 0.15
    values["ift_layer0_dx_mm"] = 0.27
    values["ift_layer1_dx_mm"] = 0.15
    values["ift_layer2_dx_mm"] = 0.03
    # Internals relative to mean 0.15: +0.12, 0, -0.12. Common = station + mean = 0.30.
    sum_to_zero = reduce_gauge(specs, choice="sum_to_zero", dropped_layer=2)
    reference = reduce_gauge(specs, choice="reference_layer", reference_layer=0)
    projected_a = project_to_gauge(values, sum_to_zero, specs)
    projected_b = project_to_gauge(values, reference, specs)
    full_a = expand_gauged_parameters(projected_a, sum_to_zero, specs)
    full_b = expand_gauged_parameters(projected_b, reference, specs)
    split_a = split_common_and_internal(full_a, specs)
    split_b = split_common_and_internal(full_b, specs)
    assert split_a["layer_internal"]["layer_0"]["dx_mm"] == pytest.approx(0.12)
    assert split_a["layer_internal"]["layer_2"]["dx_mm"] == pytest.approx(-0.12)
    assert split_b["layer_internal"]["layer_0"]["dx_mm"] == pytest.approx(0.12)
    assert split_b["layer_internal"]["layer_2"]["dx_mm"] == pytest.approx(-0.12)
    assert split_a["total_common_station_plus_layer_mean"]["dx_mm"] == pytest.approx(0.30)
    assert split_b["total_common_station_plus_layer_mean"]["dx_mm"] == pytest.approx(0.30)
    index = hierarchy_index(specs)
    reduced_shape = sum_to_zero.apply_derivative(np.zeros((2, 4, len(index.names)))).shape
    assert reduced_shape == (2, 4, 15)
    # Equal-weight layer common mode is orthogonal to the sum-to-zero columns.
    common = np.zeros(len(index.names))
    for layer in IFT_LAYER_IDS:
        common[index.layer_by_component["dx_mm"][layer]] = 1.0
    layer_columns = [
        sum_to_zero.column_transform[:, i]
        for i, name in enumerate(sum_to_zero.names)
        if "layer" in name and name.endswith("dx_mm")
    ]
    for column in layer_columns:
        assert abs(float(common @ column)) < 1.0e-12


def test_gauge_allows_station_only_components_when_layers_are_excluded():
    specs = [spec for spec in _hierarchy_specs() if spec["component"] not in {"dy_mm", "rz_mrad"} or spec["scope"] == "station"]
    names = [spec["name"] for spec in specs]
    assert "ift_layer0_dy_mm" not in names
    assert "ift_dy_mm" in names
    index = hierarchy_index(specs)
    assert index.layer_by_component["dy_mm"] == {}
    assert set(index.layer_by_component["dx_mm"]) == set(IFT_LAYER_IDS)
    reduction = reduce_gauge(specs, choice="sum_to_zero", dropped_layer=2)
    assert "ift_dy_mm" in reduction.names
    assert "ift_layer2_dx_mm" in reduction.dropped_names
    assert all("dy_mm" not in name or name == "ift_dy_mm" for name in reduction.names)
    values = {name: 0.0 for name in names}
    values["ift_dy_mm"] = 0.4
    values["ift_layer0_dx_mm"] = 0.12
    values["ift_layer2_dx_mm"] = -0.12
    split = split_common_and_internal(values, specs)
    assert split["station_common"]["dy_mm"] == pytest.approx(0.4)
    assert split["layer_internal"]["layer_0"]["dy_mm"] == pytest.approx(0.0)
    assert split["layer_internal"]["layer_0"]["dx_mm"] == pytest.approx(0.12)


def test_layer_only_gauge_keeps_internal_dx_when_station_is_frozen():
    specs = [
        spec
        for spec in _hierarchy_specs()
        if spec["scope"] == "layer" and spec["component"] in {"dx_mm", "rx_mrad", "ry_mrad"}
    ]
    values = {spec["name"]: 0.0 for spec in specs}
    values["ift_layer0_dx_mm"] = 0.12
    values["ift_layer2_dx_mm"] = -0.12
    reduction = reduce_gauge(specs, choice="sum_to_zero", dropped_layer=2)
    assert "ift_dx_mm" not in reduction.names
    assert len(reduction.names) == 6
    projected = project_to_gauge(values, reduction, specs)
    full = expand_gauged_parameters(projected, reduction, specs)
    split = split_common_and_internal(full, specs)
    assert split["layer_internal"]["layer_0"]["dx_mm"] == pytest.approx(0.12)
    assert split["layer_internal"]["layer_2"]["dx_mm"] == pytest.approx(-0.12)
    assert split["station_common"]["dx_mm"] == pytest.approx(0.0)


def test_layer_only_gauge_keeps_internal_dx_when_station_is_frozen():
    specs = [spec for spec in _hierarchy_specs() if spec["scope"] == "layer" and spec["component"] in {"dx_mm", "rx_mrad", "ry_mrad"}]
    values = {spec["name"]: 0.0 for spec in specs}
    values["ift_layer0_dx_mm"] = 0.12
    values["ift_layer2_dx_mm"] = -0.12
    reduction = reduce_gauge(specs, choice="sum_to_zero", dropped_layer=2)
    assert "ift_dx_mm" not in reduction.names
    assert len(reduction.names) == 6
    projected = project_to_gauge(values, reduction, specs)
    full = expand_gauged_parameters(projected, reduction, specs)
    split = split_common_and_internal(full, specs)
    assert split["layer_internal"]["layer_0"]["dx_mm"] == pytest.approx(0.12)
    assert split["layer_internal"]["layer_2"]["dx_mm"] == pytest.approx(-0.12)
    assert split["station_common"]["dx_mm"] == pytest.approx(0.0)


def test_station_rigid_plan_still_rejects_layer_scope():
    config = {
        "scan_mode": "station_rigid_multidof",
        "station_ids": [0, 1, 2, 3],
        "reference_station_ids": [1, 2, 3],
        "movable_station_ids": [0],
        "q_over_p_mode": 0,
        "alignment_parameter_specs": [
            {
                "name": "ift_layer0_dx_mm",
                "scope": "layer",
                "station_id": 0,
                "layer_id": 0,
                "component": "dx_mm",
                "unit": "mm",
                "finite_difference_step": 0.2,
                "severity_scale": 5.0,
            }
        ],
        "rigid_points": [_identity_point()],
    }
    with pytest.raises(ValueError, match="station scope only"):
        _build_plan(config)


def test_hierarchy_plan_requires_joint_station_and_layer_probes_and_forbids_dz():
    plan = _build_plan(_hierarchy_config())
    assert plan["scan_mode"] == "ift_layer_hierarchy"
    assert len(plan["alignment_parameter_specs"]) == 20
    assert len(plan["points"]) == 41
    names = {point["name"] for point in plan["points"]}
    assert "fd_ift_layer0_dx_mm_p" in names
    assert "fd_ift_dx_mm_m" in names
    bad = _hierarchy_config()
    bad["alignment_parameter_specs"].append(
        {
            "name": "ift_layer0_dz_mm",
            "scope": "layer",
            "station_id": 0,
            "layer_id": 0,
            "component": "dz_mm",
            "unit": "mm",
            "finite_difference_step": 1.0,
            "severity_scale": 5.0,
        }
    )
    with pytest.raises(ValueError, match="does not admit dz"):
        _build_plan(bad)


def test_compile_layer_pilot_builds_reference_fd_and_held_out_internal_points():
    root = Path(__file__).resolve().parents[1]
    template = yaml.safe_load(
        (root / "configs" / "physical_refit_ift_layer_identifiability_pilot.yaml").read_text()
    )
    current = {
        spec["name"]: 0.0 for spec in template["physical_refit_capture_scan"]["alignment_parameter_specs"]
    }
    compiled, contract = compile_layer_identifiability_pilot(template, iteration=0, current_values=current)
    points = compiled["physical_refit_capture_scan"]["rigid_points"]
    assert len(points) == 43
    assert contract["anchor_point"] == "iteration_00_reference"
    assert "iteration_00_closure_internal" in contract["held_out_closure_points"]
    plan = _build_plan(compiled["physical_refit_capture_scan"])
    assert plan["scan_mode"] == "ift_layer_hierarchy"
    internal = next(point for point in points if point["name"] == "iteration_00_closure_internal")
    assert internal["station_transforms"]["0"][0] == pytest.approx(0.0)
    assert internal["layer_transforms"]["0"]["0"][0] == pytest.approx(0.12)
    assert internal["layer_transforms"]["0"]["2"][0] == pytest.approx(-0.12)
    mixed = next(point for point in points if point["name"] == "iteration_00_closure_mixed")
    assert mixed["station_transforms"]["0"][0] == pytest.approx(0.15)
    with pytest.raises(ValueError, match="all-zero"):
        compile_layer_identifiability_pilot(template, iteration=0, current_values={**current, "ift_dx_mm": 0.1})


def _linear_heldout_template():
    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load(
        (root / "configs" / "physical_refit_ift_layer_linear_internal_heldout.yaml").read_text()
    )


def test_compile_linear_internal_heldout_skips_fd_and_keeps_station_zero():
    template = _linear_heldout_template()
    current = {
        spec["name"]: 0.0 for spec in template["physical_refit_capture_scan"]["alignment_parameter_specs"]
    }
    compiled, contract = compile_layer_identifiability_pilot(
        template, iteration=0, current_values=current, include_finite_differences=False
    )
    scan = compiled["physical_refit_capture_scan"]
    points = scan["rigid_points"]
    assert len(points) == 2
    assert scan["held_out_only"] is True
    assert contract["held_out_only"] is True
    assert contract["include_finite_differences"] is False
    assert contract["reference_point"] is None
    plan = _build_plan(scan)
    assert plan["held_out_only"] is True
    assert len(plan["points"]) == 2
    assert all(float(point["condition_magnitude"]) > 0.0 for point in plan["points"])
    baseline = _baseline_refit_dir(plan, Path("/tmp/heldout"), run_alignment_closure=False)
    assert baseline.name == "refit"
    assert baseline.parent.name == "_unused_held_out_only_baseline"
    with pytest.raises(ValueError, match="lacks a zero-magnitude baseline"):
        _baseline_refit_dir(plan, Path("/tmp/heldout"), run_alignment_closure=True)
    dx = next(point for point in points if point["name"] == "iteration_00_closure_relative_dx")
    assert dx["station_transforms"]["0"] == pytest.approx([0.0] * 6)
    assert dx["layer_transforms"]["0"]["0"][0] == pytest.approx(0.12)
    assert dx["layer_transforms"]["0"]["1"][0] == pytest.approx(0.0)
    assert dx["layer_transforms"]["0"]["2"][0] == pytest.approx(-0.12)
    ry = next(point for point in points if point["name"] == "iteration_00_closure_relative_ry")
    assert ry["layer_transforms"]["0"]["0"][4] == pytest.approx(0.0007)
    assert ry["layer_transforms"]["0"]["2"][4] == pytest.approx(-0.0007)


def test_compile_linear_internal_heldout_rejects_dy_station_rx_and_mixed_rotation():
    template = _linear_heldout_template()
    current = {
        spec["name"]: 0.0 for spec in template["physical_refit_capture_scan"]["alignment_parameter_specs"]
    }

    def compile_with(held_out):
        payload = yaml.safe_load(yaml.safe_dump(template))
        payload["physical_refit_capture_scan"]["held_out_closure_points"] = held_out
        return compile_layer_identifiability_pilot(
            payload, iteration=0, current_values=current, include_finite_differences=False
        )

    with pytest.raises(ValueError, match="unusable layer dy_mm"):
        compile_with([{"name": "bad_dy", "alignment_parameter_values": {"ift_layer0_dy_mm": 0.08, "ift_layer2_dy_mm": -0.08}}])
    with pytest.raises(ValueError, match="station correction at zero"):
        compile_with(
            [{"name": "bad_station", "alignment_parameter_values": {"ift_dx_mm": 0.15, "ift_layer0_dx_mm": 0.12, "ift_layer2_dx_mm": -0.12}}]
        )
    with pytest.raises(ValueError, match="must not inject layer rx"):
        compile_with(
            [{"name": "bad_rx", "alignment_parameter_values": {"ift_layer0_rx_mrad": 0.7, "ift_layer2_rx_mrad": -0.7}}]
        )
    with pytest.raises(ValueError, match="must not mix dx and rotation"):
        compile_with(
            [
                {
                    "name": "bad_mix",
                    "alignment_parameter_values": {
                        "ift_layer0_dx_mm": 0.12,
                        "ift_layer2_dx_mm": -0.12,
                        "ift_layer0_ry_mrad": 0.7,
                        "ift_layer2_ry_mrad": -0.7,
                    },
                }
            ]
        )


def _rank(full_rank=True, condition=80.0, dimension=2):
    return {
        "sum_to_zero": {
            "full_rank": full_rank,
            "normal_matrix_rank": dimension if full_rank else dimension - 1,
            "dimension": dimension,
            "normal_matrix_condition_number": condition,
        },
        "reference_layer": {
            "full_rank": full_rank,
            "normal_matrix_rank": dimension if full_rank else dimension - 1,
            "dimension": dimension,
            "normal_matrix_condition_number": condition,
        },
    }


def test_physics_verdict_compares_gauge_invariant_internals_not_parameter_labels():
    expected = {"layer_0": 0.12, "layer_1": 0.0, "layer_2": -0.12}
    recovered = {
        "sum_to_zero": {"layer_0": 0.118, "layer_1": 0.002, "layer_2": -0.120},
        "reference_layer": {"layer_0": 0.119, "layer_1": 0.001, "layer_2": -0.120},
    }
    sources = {
        "mc24_a": {
            "sum_to_zero": {"layer_0": 0.117, "layer_1": 0.001, "layer_2": -0.118},
            "reference_layer": {"layer_0": 0.118, "layer_1": 0.0, "layer_2": -0.118},
        },
        "mc24_b": {
            "sum_to_zero": {"layer_0": 0.121, "layer_1": -0.001, "layer_2": -0.120},
            "reference_layer": {"layer_0": 0.122, "layer_1": 0.0, "layer_2": -0.122},
        },
    }
    leakage = {
        "sum_to_zero": {component: 0.0 for component in FREE_COMPONENTS},
        "reference_layer": {component: 0.0 for component in FREE_COMPONENTS},
    }
    verdict = physics_verdict(
        component="dx_mm",
        expected_internals=expected,
        recovered_by_gauge=recovered,
        source_internals=sources,
        station_leakage_by_gauge=leakage,
        ranks=_rank(),
        residual_ratios={"sum_to_zero": 0.12, "reference_layer": 0.11},
        max_condition_number=1.0e4,
    )
    assert verdict["passed"] is True
    assert verdict["gauges_agree"] is True
    assert verdict["physics_recovered"] is True
    assert verdict["no_station_common_mode_leakage"] is True


def test_physics_verdict_uses_outer_relative_stability_and_station_common_not_gauge_mean():
    expected = {"layer_0": 0.12, "layer_1": 0.0, "layer_2": -0.12}
    recovered = {
        "sum_to_zero": {"layer_0": 0.116, "layer_1": 0.008, "layer_2": -0.124},
        "reference_layer": {"layer_0": 0.123, "layer_1": -0.004, "layer_2": -0.119},
    }
    sources = {
        "mc24_a": {
            "sum_to_zero": {"layer_0": 0.110, "layer_1": 0.010, "layer_2": -0.120},
            "reference_layer": {"layer_0": 0.110, "layer_1": 0.010, "layer_2": -0.120},
        },
        "mc24_b": {
            "sum_to_zero": {"layer_0": 0.072, "layer_1": 0.108, "layer_2": -0.179},
            "reference_layer": {"layer_0": 0.072, "layer_1": 0.108, "layer_2": -0.179},
        },
        "mc24_c": {
            "sum_to_zero": {"layer_0": 0.120, "layer_1": 0.0, "layer_2": -0.120},
            "reference_layer": {"layer_0": 0.120, "layer_1": 0.0, "layer_2": -0.120},
        },
    }
    leakage = {
        "sum_to_zero": {component: 0.0 for component in FREE_COMPONENTS},
        "reference_layer": {component: 0.0 for component in FREE_COMPONENTS},
    }
    verdict = physics_verdict(
        component="dx_mm",
        expected_internals=expected,
        recovered_by_gauge=recovered,
        source_internals=sources,
        station_leakage_by_gauge=leakage,
        ranks=_rank(condition=37.0),
        residual_ratios={"sum_to_zero": 0.037, "reference_layer": 0.035},
        max_condition_number=1.0e4,
    )
    assert verdict["passed"] is True
    assert verdict["source_stable"] is True
    assert verdict["no_station_common_mode_leakage"] is True
    assert verdict["source_stability"]["outer_relative_layer0_minus_layer2"]["abs_spread"] == pytest.approx(0.021)


def test_physics_verdict_rejects_station_leakage_and_gauge_disagreement():
    expected = {"layer_0": 0.12, "layer_1": 0.0, "layer_2": -0.12}
    recovered = {
        "sum_to_zero": {"layer_0": 0.12, "layer_1": 0.0, "layer_2": -0.12},
        "reference_layer": {"layer_0": 0.01, "layer_1": 0.0, "layer_2": -0.01},
    }
    sources = {
        "mc24_a": recovered,
        "mc24_b": recovered,
    }
    leakage = {
        "sum_to_zero": {**{component: 0.0 for component in FREE_COMPONENTS}, "dx_mm": 0.15},
        "reference_layer": {**{component: 0.0 for component in FREE_COMPONENTS}, "dx_mm": 0.15},
    }
    verdict = physics_verdict(
        component="dx_mm",
        expected_internals=expected,
        recovered_by_gauge=recovered,
        source_internals=sources,
        station_leakage_by_gauge=leakage,
        ranks=_rank(condition=80.0),
        residual_ratios={"sum_to_zero": 0.12, "reference_layer": 0.11},
        max_condition_number=1.0e4,
    )
    assert verdict["passed"] is False
    assert verdict["gauges_agree"] is False
    assert verdict["no_station_common_mode_leakage"] is False
