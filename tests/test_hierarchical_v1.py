from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from alignment.five_dof_sampling import SURVEY_PARAMETER, five_dof_severity
from alignment.hierarchical_v1 import (
    C_DX,
    HIERARCHICAL_V1_PARAMETERS,
    LAYER_LEVEL,
    STATION_LEVEL,
    naive_proposed_next_zeros_unfloated,
    remaining_after_cdx_step,
    remaining_after_level_step,
    remaining_after_station_step,
    station_absorption_of_cdx,
    station_payload_stability,
)
from alignment.hierarchical_v1_sampling import sample_hierarchical_v1_points
from scripts.prepare_hierarchical_v1_iteration import compile_hierarchical_v1
from scripts.prepare_hierarchical_v1_remaining_iteration import _held_out_from_remaining
from scripts.run_physical_refit_capture_scan import _build_plan


ROOT = Path(__file__).resolve().parents[1]


def _template():
    return yaml.safe_load((ROOT / "configs" / "physical_refit_ift_hierarchical_v1_iteration.yaml").read_text())


def test_hierarchical_v1_sampling_is_seed_stable_and_excites_both_levels():
    first = sample_hierarchical_v1_points(seed=20260821)
    second = sample_hierarchical_v1_points(seed=20260821)
    assert first["start"] == second["start"]
    assert first["held_out"] == second["held_out"]
    assert first["start"][SURVEY_PARAMETER] == 0.0
    assert first["held_out"][SURVEY_PARAMETER] == 0.0
    assert math.isclose(five_dof_severity(first["start"]), 0.12, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(five_dof_severity(first["held_out"]), 0.14, rel_tol=0.0, abs_tol=1.0e-12)
    assert abs(first["start"][C_DX]) <= 0.12 + 1.0e-12
    assert abs(first["held_out"][C_DX]) <= 0.12 + 1.0e-12
    assert first["start"][C_DX] * first["held_out"][C_DX] < 0.0
    assert abs(first["start"][C_DX]) >= 0.40 * 0.12 - 1.0e-12
    other = sample_hierarchical_v1_points(seed=20260819)
    assert other["start"] != first["start"]


def test_remaining_after_station_step_keeps_true_cdx_and_writes_dz_zero():
    injected = {
        "ift_dx_mm": 0.40,
        "ift_dy_mm": -0.20,
        "ift_dz_mm": 0.0,
        "ift_rx_mrad": 3.0,
        "ift_ry_mrad": -1.5,
        "ift_rz_mrad": 2.0,
        "C_dx": 0.08,
    }
    recovered = {
        "ift_dx_mm": 0.39,
        "ift_dy_mm": -0.19,
        "ift_dz_mm": 0.7,
        "ift_rx_mrad": 2.9,
        "ift_ry_mrad": -1.4,
        "ift_rz_mrad": 1.9,
    }
    remaining = remaining_after_station_step(injected=injected, recovered_station=recovered)
    assert remaining[C_DX] == pytest.approx(0.08)
    assert remaining["ift_dx_mm"] == pytest.approx(0.01)
    assert remaining[SURVEY_PARAMETER] == 0.0
    naive = naive_proposed_next_zeros_unfloated(
        injected=injected, recovered=recovered, floated_level=STATION_LEVEL
    )
    assert naive[C_DX] == pytest.approx(0.0)
    assert remaining[C_DX] != pytest.approx(naive[C_DX])


def test_remaining_after_cdx_step_freezes_closed_station():
    injected = {
        "ift_dx_mm": 0.01,
        "ift_dy_mm": -0.01,
        "ift_dz_mm": 0.0,
        "ift_rx_mrad": 0.1,
        "ift_ry_mrad": -0.1,
        "ift_rz_mrad": 0.1,
        "C_dx": 0.08,
    }
    remaining = remaining_after_cdx_step(injected=injected, recovered_cdx=0.079)
    assert remaining[C_DX] == pytest.approx(0.001)
    assert remaining["ift_dx_mm"] == pytest.approx(injected["ift_dx_mm"])
    assert remaining["ift_ry_mrad"] == pytest.approx(injected["ift_ry_mrad"])
    naive = naive_proposed_next_zeros_unfloated(
        injected=injected, recovered={"C_dx": 0.079}, floated_level=LAYER_LEVEL
    )
    assert naive["ift_dx_mm"] == pytest.approx(0.0)
    assert remaining["ift_dx_mm"] != pytest.approx(naive["ift_dx_mm"])


def test_cross_level_leakage_helpers():
    injected = {
        "ift_dx_mm": 0.40,
        "ift_dy_mm": 0.0,
        "ift_dz_mm": 0.0,
        "ift_rx_mrad": 0.0,
        "ift_ry_mrad": 0.0,
        "ift_rz_mrad": 0.0,
        "C_dx": 0.08,
    }
    recovered = {
        "ift_dx_mm": 0.48,
        "ift_dy_mm": 0.0,
        "ift_rx_mrad": 0.0,
        "ift_ry_mrad": 0.2,
        "ift_rz_mrad": 0.0,
    }
    leak = station_absorption_of_cdx(injected=injected, recovered_station=recovered)
    assert leak["ift_dx_mm_error"] == pytest.approx(0.08)
    assert leak["dx_error_over_injected_C_dx"] == pytest.approx(1.0)
    remaining = remaining_after_level_step(
        injected=injected, recovered={"C_dx": 0.079}, floated_level=LAYER_LEVEL
    )
    stable = station_payload_stability(injected, remaining)
    assert stable["max_abs_station_delta"] == pytest.approx(0.0)


def test_compile_and_plan_builder_accept_hierarchical_v1_joint_points():
    template = _template()
    sampled = sample_hierarchical_v1_points(seed=20260821)
    template["physical_refit_capture_scan"]["held_out_closure_points"] = [
        {"name": "start", "alignment_parameter_values": sampled["start"]},
        {"name": "heldout_00", "alignment_parameter_values": sampled["held_out"]},
    ]
    current = {name: 0.0 for name in HIERARCHICAL_V1_PARAMETERS}
    compiled, contract = compile_hierarchical_v1(template, iteration=0, current_values=current)
    points = compiled["physical_refit_capture_scan"]["rigid_points"]
    assert contract["joint_station_cdx_newton"] is False
    assert len(points) == 1 + 2 * 7 + 2
    names = {point["name"] for point in points}
    assert "iteration_00_reference" in names
    assert "iteration_00_fd_ift_dx_mm_p" in names
    assert "iteration_00_fd_ift_dz_mm_m" in names
    assert "iteration_00_fd_C_dx_p" in names
    assert "iteration_00_start" in names
    assert "iteration_00_heldout_00" in names
    start = next(point for point in points if point["name"] == "iteration_00_start")
    assert start["station_transforms"]["0"][2] == pytest.approx(0.0)
    assert start["layer_transforms"]["0"]["0"][0] == pytest.approx(sampled["start"][C_DX])
    assert start["layer_transforms"]["0"]["2"][0] == pytest.approx(-sampled["start"][C_DX])
    assert start["layer_transforms"]["0"]["1"] == [0.0] * 6
    plan = _build_plan(compiled["physical_refit_capture_scan"])
    assert plan["fit_basis"] == "hierarchical_v1"
    assert plan["scan_mode"] == "ift_layer_hierarchy"
    assert len(plan["points"]) == 17


def test_plan_builder_still_rejects_mixed_contrast_without_hierarchical_v1():
    template = _template()
    scan = template["physical_refit_capture_scan"]
    scan["fit_basis"] = "outer_contrast"
    scan["alignment_parameter_specs"] = [
        spec for spec in scan["alignment_parameter_specs"] if spec["name"] != "ift_dz_mm"
    ]
    scan["rigid_points"] = [
        {
            "name": "nominal",
            "point_role": "nominal",
            "direction_trial": "nominal",
            "station_transforms": {station: [0.0] * 6 for station in range(4)},
            "layer_transforms": {0: {layer: [0.0] * 6 for layer in range(3)}},
            "alignment_parameter_values": {name: 0.0 for name in HIERARCHICAL_V1_PARAMETERS},
        }
    ]
    with pytest.raises(ValueError, match="cannot be mixed"):
        _build_plan(scan)


def test_plan_builder_still_rejects_layer_dz():
    template = _template()
    scan = template["physical_refit_capture_scan"]
    scan["alignment_parameter_specs"].append(
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
    scan["rigid_points"] = [
        {
            "name": "nominal",
            "point_role": "nominal",
            "direction_trial": "nominal",
            "station_transforms": {station: [0.0] * 6 for station in range(4)},
            "layer_transforms": {0: {layer: [0.0] * 6 for layer in range(3)}},
        }
    ]
    with pytest.raises(ValueError, match="does not admit dz"):
        _build_plan(scan)


def test_remaining_bank_compile_skips_fd_and_may_be_near_zero_station():
    template = _template()
    remaining = remaining_after_station_step(
        injected={
            "ift_dx_mm": 0.40,
            "ift_dy_mm": -0.10,
            "ift_dz_mm": 0.0,
            "ift_rx_mrad": 2.0,
            "ift_ry_mrad": -1.0,
            "ift_rz_mrad": 1.0,
            "C_dx": 0.08,
        },
        recovered_station={
            "ift_dx_mm": 0.399,
            "ift_dy_mm": -0.099,
            "ift_dz_mm": 0.2,
            "ift_rx_mrad": 1.99,
            "ift_ry_mrad": -0.99,
            "ift_rz_mrad": 0.99,
        },
    )
    template["physical_refit_capture_scan"]["held_out_only"] = True
    template["physical_refit_capture_scan"]["held_out_closure_points"] = [
        {
            "name": "start",
            "alignment_parameter_values": remaining,
            "allow_identically_zero": False,
        }
    ]
    current = {name: 0.0 for name in HIERARCHICAL_V1_PARAMETERS}
    compiled, contract = compile_hierarchical_v1(
        template,
        iteration=0,
        current_values=current,
        include_finite_differences=False,
    )
    points = compiled["physical_refit_capture_scan"]["rigid_points"]
    assert contract["held_out_only"] is True
    assert len(points) == 1
    assert points[0]["alignment_parameter_values"][C_DX] == pytest.approx(0.08)
    plan = _build_plan(compiled["physical_refit_capture_scan"])
    assert plan["held_out_only"] is True
    assert len(plan["points"]) == 1


def test_remaining_json_reconstructs_block_step_and_rejects_joint_newton_values():
    injected = {
        "ift_dx_mm": 0.40,
        "ift_dy_mm": 0.0,
        "ift_dz_mm": 0.0,
        "ift_rx_mrad": 0.0,
        "ift_ry_mrad": 0.0,
        "ift_rz_mrad": 0.0,
        "C_dx": 0.08,
    }
    payload = {
        "points": [
            {
                "name": "start",
                "floated_level": STATION_LEVEL,
                "injected": injected,
                "recovered": {
                    "ift_dx_mm": 0.39,
                    "ift_dy_mm": 0.0,
                    "ift_rx_mrad": 0.0,
                    "ift_ry_mrad": 0.0,
                    "ift_rz_mrad": 0.0,
                },
            }
        ]
    }
    entries = _held_out_from_remaining(payload)
    assert entries[0]["alignment_parameter_values"][C_DX] == pytest.approx(0.08)
    assert entries[0]["alignment_parameter_values"]["ift_dx_mm"] == pytest.approx(0.01)


def test_leakage_operator_transfer_is_axial_fd_only():
    template = yaml.safe_load(
        (ROOT / "configs" / "physical_refit_leakage_operator_transfer.yaml").read_text()
    )
    current = {name: 0.0 for name in HIERARCHICAL_V1_PARAMETERS}
    compiled, contract = compile_hierarchical_v1(template, iteration=0, current_values=current)
    points = compiled["physical_refit_capture_scan"]["rigid_points"]
    assert contract["held_out_closure_points"] == []
    assert len(points) == 1 + 2 * len(HIERARCHICAL_V1_PARAMETERS)
    joint = []
    for point in points:
        values = point["alignment_parameter_values"]
        station = any(abs(float(values[name])) > 1.0e-15 for name in HIERARCHICAL_V1_PARAMETERS if name != C_DX)
        cdx = abs(float(values[C_DX])) > 1.0e-15
        if station and cdx:
            joint.append(point["name"])
        if point["name"] != "iteration_00_reference":
            assert point.get("finite_difference_for")
    assert joint == []
    reference = next(point for point in points if point["name"] == "iteration_00_reference")
    assert reference["station_transforms"]["0"] == pytest.approx([0.0] * 6)
    assert reference["layer_transforms"]["0"]["0"] == pytest.approx([0.0] * 6)
