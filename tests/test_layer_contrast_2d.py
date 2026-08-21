from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from alignment.capture_criteria import (
    evaluate_parameter_capture,
    load_capture_criteria,
    overall_framework_capture,
)
from alignment.contrast_sampling import (
    CONTRAST_ENVELOPE,
    contrast_radius,
    draw_joint_contrast,
    inside_contrast_envelope,
    sample_joint_contrast_points,
)
from alignment.layer_contrast_capture import SCHEMA_VERSION as CONTRAST_CAPTURE_SCHEMA
from alignment.layer_hierarchy import contrast_layer_six_vectors, spec_scope
from alignment.physical_jacobian import (
    parameter_values_from_payload,
    payload_transforms_with_parameter_values,
)
from scripts.prepare_layer_identifiability_pilot import compile_layer_identifiability_pilot
from scripts.run_physical_refit_capture_scan import _build_plan


def _contrast_specs():
    return [
        {
            "name": "C_dx",
            "scope": "contrast",
            "station_id": 0,
            "component": "dx_mm",
            "unit": "mm",
            "finite_difference_step": 0.10,
            "severity_scale": 0.12,
        },
        {
            "name": "C_rx",
            "scope": "contrast",
            "station_id": 0,
            "component": "rx_mrad",
            "unit": "mrad",
            "finite_difference_step": 0.50,
            "severity_scale": 0.70,
        },
    ]


def _zero_payload():
    stations = {str(station): [0.0] * 6 for station in range(4)}
    layers = {"0": {str(layer): [0.0] * 6 for layer in range(3)}}
    return stations, layers


def test_joint_contrast_sampling_is_seed_stable_and_inside_envelope():
    first = sample_joint_contrast_points(
        seed=20260820,
        start_radius=0.65,
        linear_radii=(0.32, 0.48),
        stress_radii=(0.96,),
        held_out_radii=(0.72,),
    )
    second = sample_joint_contrast_points(
        seed=20260820,
        start_radius=0.65,
        linear_radii=(0.32, 0.48),
        stress_radii=(0.96,),
        held_out_radii=(0.72,),
    )
    assert first["start"] == second["start"]
    assert math.isclose(contrast_radius(first["start"]), 0.65, rel_tol=0.0, abs_tol=1.0e-12)
    for group in ("linear", "stress", "held_out"):
        for item in first[group]:
            values = item["alignment_parameter_values"]
            assert inside_contrast_envelope(values)
            assert abs(values["C_dx"]) > 0.0
            assert abs(values["C_rx"]) > 0.0
            assert abs(values["C_dx"]) <= CONTRAST_ENVELOPE["C_dx"] + 1.0e-12
            assert abs(values["C_rx"]) <= CONTRAST_ENVELOPE["C_rx"] + 1.0e-12
    held = first["held_out"][0]["alignment_parameter_values"]
    start = first["start"]
    assert not math.isclose(held["C_dx"], start["C_dx"], rel_tol=0.0, abs_tol=1.0e-12)
    assert first["held_out"][0]["reserved_from_fd_and_operating_point"] is True


def test_sampling_rejects_radius_outside_envelope():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="exceeds"):
        draw_joint_contrast(rng, radius=1.01)


def test_contrast_payload_is_L0_plus_L2_minus_station_zero():
    specs = _contrast_specs()
    stations, layers = _zero_payload()
    values = {"C_dx": 0.12, "C_rx": 0.70}
    out_stations, out_layers = payload_transforms_with_parameter_values(specs, stations, layers, values)
    assert out_stations["0"] == pytest.approx([0.0] * 6)
    assert out_stations["1"] == pytest.approx([0.0] * 6)
    expected = contrast_layer_six_vectors(0.12, 0.70)
    assert out_layers["0"]["0"] == pytest.approx(expected["0"])
    assert out_layers["0"]["1"] == pytest.approx(expected["1"])
    assert out_layers["0"]["2"] == pytest.approx(expected["2"])
    readout = parameter_values_from_payload(specs, out_stations, out_layers)
    assert readout["C_dx"] == pytest.approx(0.12)
    assert readout["C_rx"] == pytest.approx(0.70)


def _contrast_template(held_out):
    return {
        "physical_refit_capture_scan": {
            "scan_mode": "ift_layer_hierarchy",
            "q_over_p_mode": 0,
            "station_ids": [0, 1, 2, 3],
            "reference_station_ids": [1, 2, 3],
            "movable_station_ids": [0],
            "movable_layer_ids": [0, 1, 2],
            "condition_axis": "ift_layer_outer_contrast_2d_l2",
            "alignment_parameter_specs": _contrast_specs(),
            "held_out_closure_points": held_out,
        }
    }


def test_compile_contrast_2d_bank_uses_four_fd_and_mixed_heldout():
    template = _contrast_template(
        [
            {"name": "start", "alignment_parameter_values": {"C_dx": 0.06, "C_rx": 0.35}},
            {"name": "heldout_00", "alignment_parameter_values": {"C_dx": -0.10, "C_rx": 0.65}},
        ]
    )
    compiled, contract = compile_layer_identifiability_pilot(
        template, iteration=0, current_values={"C_dx": 0.0, "C_rx": 0.0}
    )
    scan = compiled["physical_refit_capture_scan"]
    points = scan["rigid_points"]
    names = [point["name"] for point in points]
    assert "iteration_00_reference" in names
    assert "iteration_00_fd_C_dx_p" in names
    assert "iteration_00_fd_C_dx_m" in names
    assert "iteration_00_fd_C_rx_p" in names
    assert "iteration_00_fd_C_rx_m" in names
    assert "iteration_00_start" in names
    assert "iteration_00_heldout_00" in names
    assert contract["method"] == "physical_ift_layer_contrast_2d_curriculum"
    assert contract["contrast_parameters"] == ["C_dx", "C_rx"]
    fd_dx = next(point for point in points if point["name"] == "iteration_00_fd_C_dx_p")
    assert fd_dx["layer_transforms"]["0"]["0"][0] == pytest.approx(0.10)
    assert fd_dx["layer_transforms"]["0"]["2"][0] == pytest.approx(-0.10)
    assert fd_dx["layer_transforms"]["0"]["1"] == pytest.approx([0.0] * 6)
    assert fd_dx["station_transforms"]["0"] == pytest.approx([0.0] * 6)
    start = next(point for point in points if point["name"] == "iteration_00_start")
    assert start["layer_transforms"]["0"]["0"][0] == pytest.approx(0.06)
    assert start["layer_transforms"]["0"]["0"][3] == pytest.approx(0.00035)
    assert start["layer_transforms"]["0"]["2"][0] == pytest.approx(-0.06)
    assert start["layer_transforms"]["0"]["2"][3] == pytest.approx(-0.00035)
    plan = _build_plan(scan)
    assert plan["scan_mode"] == "ift_layer_hierarchy"
    assert [spec["name"] for spec in plan["alignment_parameter_specs"]] == ["C_dx", "C_rx"]
    assert all(spec_scope(spec) == "contrast" for spec in plan["alignment_parameter_specs"])


def test_compile_rejects_envelope_violation_and_ry():
    with pytest.raises(ValueError, match="inside"):
        compile_layer_identifiability_pilot(
            _contrast_template(
                [{"name": "too_big", "alignment_parameter_values": {"C_dx": 0.13, "C_rx": 0.10}}]
            ),
            iteration=0,
            current_values={"C_dx": 0.0, "C_rx": 0.0},
        )
    template = _contrast_template([])
    template["physical_refit_capture_scan"]["alignment_parameter_specs"].append(
        {
            "name": "C_ry",
            "scope": "contrast",
            "station_id": 0,
            "component": "ry_mrad",
            "unit": "mrad",
            "finite_difference_step": 0.4,
            "severity_scale": 0.7,
        }
    )
    with pytest.raises(ValueError, match="C_dx or exactly C_dx\\+C_rx"):
        compile_layer_identifiability_pilot(
            template, iteration=0, current_values={"C_dx": 0.0, "C_rx": 0.0, "C_ry": 0.0}
        )


def test_plan_builder_accepts_compiled_contrast_only_specs():
    root = Path(__file__).resolve().parents[1]
    contrast = yaml.safe_load((root / "configs" / "physical_refit_ift_layer_contrast_2d_iteration.yaml").read_text())
    compiled, _ = compile_layer_identifiability_pilot(
        contrast,
        iteration=0,
        current_values={"C_dx": 0.0, "C_rx": 0.0},
    )
    plan = _build_plan(compiled["physical_refit_capture_scan"])
    assert [spec["name"] for spec in plan["alignment_parameter_specs"]] == ["C_dx", "C_rx"]
    assert all(spec["scope"] == "contrast" for spec in plan["alignment_parameter_specs"])
    fd = [point for point in plan["points"] if point.get("finite_difference_for")]
    assert len(fd) == 4


def test_contrast_capture_dual_gate_and_schema_dispatch(tmp_path):
    criteria = {
        "schema_version": CONTRAST_CAPTURE_SCHEMA,
        "validation_used_in_registration": False,
        "test_data_accessed": False,
        "per_parameter": {
            "C_dx": {
                "role": "free",
                "engineering_tolerance": 0.03,
                "registered_sigma": 0.007,
                "statistical_k": 3.0,
                "coverage_k": 3.0,
            },
            "C_rx": {
                "role": "free",
                "engineering_tolerance": 0.15,
                "registered_sigma": 0.15,
                "statistical_k": 3.0,
                "coverage_k": 3.0,
            },
        },
    }
    path = tmp_path / "capture_criteria.json"
    path.write_text(__import__("json").dumps(criteria), encoding="utf-8")
    loaded = load_capture_criteria(path)
    assert loaded["schema_version"] == CONTRAST_CAPTURE_SCHEMA
    dx = evaluate_parameter_capture(name="C_dx", error=0.018, fit_sigma=0.005, criteria=loaded)
    rx = evaluate_parameter_capture(name="C_rx", error=0.02, fit_sigma=0.15, criteria=loaded)
    assert dx["statistical_capture"] is True
    assert dx["coverage_capture"] is False
    aggregate = overall_framework_capture([dx, rx], survey_parameters=[])
    assert aggregate["framework_capture_success"] is False
    assert aggregate["survey_parameters_excluded_from_capture"] == []


def test_compile_cdx_only_transfer_keeps_station_nominal_and_envelope():
    root = Path(__file__).resolve().parents[1]
    template = yaml.safe_load(
        (root / "configs" / "physical_refit_ift_internal_cdx_transfer.yaml").read_text()
    )
    compiled, contract = compile_layer_identifiability_pilot(
        template, iteration=0, current_values={"C_dx": 0.0}
    )
    points = compiled["physical_refit_capture_scan"]["rigid_points"]
    names = [point["name"] for point in points]
    assert names == [
        "iteration_00_reference",
        "iteration_00_fd_C_dx_p",
        "iteration_00_fd_C_dx_m",
        "iteration_00_start",
        "iteration_00_heldout_00",
    ]
    assert contract["method"] == "physical_ift_internal_cdx_transfer"
    assert contract["contrast_parameters"] == ["C_dx"]
    assert "C_rx" in contract["forbidden_components"]
    start = next(point for point in points if point["name"] == "iteration_00_start")
    assert start["station_transforms"]["0"] == pytest.approx([0.0] * 6)
    assert start["layer_transforms"]["0"]["0"][0] == pytest.approx(0.12)
    assert start["layer_transforms"]["0"]["1"][0] == pytest.approx(0.0)
    assert start["layer_transforms"]["0"]["2"][0] == pytest.approx(-0.12)
    held = next(point for point in points if point["name"] == "iteration_00_heldout_00")
    assert held["layer_transforms"]["0"]["0"][0] == pytest.approx(-0.12)
    assert held["layer_transforms"]["0"]["2"][0] == pytest.approx(0.12)
    too_big = yaml.safe_load(
        (root / "configs" / "physical_refit_ift_internal_cdx_transfer.yaml").read_text()
    )
    too_big["physical_refit_capture_scan"]["held_out_closure_points"] = [
        {"name": "too_big", "alignment_parameter_values": {"C_dx": 0.13}}
    ]
    with pytest.raises(ValueError, match="inside"):
        compile_layer_identifiability_pilot(too_big, iteration=0, current_values={"C_dx": 0.0})
    plan = _build_plan(compiled["physical_refit_capture_scan"])
    assert plan["scan_mode"] == "ift_layer_hierarchy"
    assert [spec["name"] for spec in plan["alignment_parameter_specs"]] == ["C_dx"]
