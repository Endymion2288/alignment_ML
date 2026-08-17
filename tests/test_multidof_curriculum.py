from __future__ import annotations

import copy

import pytest

from scripts.build_physical_curriculum_corpus import _joint_rigid_points, _source_scan_config
from scripts.prepare_multidof_alignment_iteration import compile_iteration
from scripts.run_physical_refit_capture_scan import _build_plan


def _config():
    return {
        "refit": {
            "nevents": 10,
            "station_ids": [0, 1, 2, 3],
            "q_over_p_mode": 0,
            "min_truth_match_fraction": 0.99,
            "chi2_gate": 25.0,
            "refinement_iterations": 1,
        },
        "joint_rigid_curriculum": {
            "scan_mode": "station_rigid_multidof",
            "condition_axis": "ift_dx_dy_ry_joint_l2",
            "seed": 17,
            "reference_station_ids": [1, 2, 3],
            "movable_station_ids": [0],
            "alignment_parameter_specs": [
                {
                    "name": "ift_dx_mm",
                    "station_id": 0,
                    "component": "dx_mm",
                    "finite_difference_step": 0.5,
                    "severity_scale": 5.0,
                },
                {
                    "name": "ift_dy_mm",
                    "station_id": 0,
                    "component": "dy_mm",
                    "finite_difference_step": 0.5,
                    "severity_scale": 5.0,
                },
                {
                    "name": "ift_ry_mrad",
                    "station_id": 0,
                    "component": "ry_mrad",
                    "finite_difference_step": 10.0,
                    "severity_scale": 60.0,
                },
            ],
            "curriculum_stages": [
                {
                    "name": "joint",
                    "points_per_split": 1,
                    "ranges": {
                        "ift_dx_mm": [-1.0, 1.0],
                        "ift_dy_mm": [-1.0, 1.0],
                        "ift_ry_mrad": [-20.0, 20.0],
                    },
                }
            ],
            "held_out_closure_stages": [
                {
                    "name": "closure",
                    "points_per_split": 1,
                    "ranges": {
                        "ift_dx_mm": [-2.0, 2.0],
                        "ift_dy_mm": [-2.0, 2.0],
                        "ift_ry_mrad": [-40.0, 40.0],
                    },
                }
            ],
        },
    }


def test_joint_multidof_curriculum_is_split_deterministic_and_builds_physical_probe_plan():
    config = _config()
    points = _joint_rigid_points(config, ("train", "validation"))
    repeated = _joint_rigid_points(config, ("train", "validation"))
    assert points == repeated
    assert points["train"] != points["validation"]
    scan = _source_scan_config(
        config,
        {"source_id": "source", "split": "train", "input_xaod": "/tmp/source.root"},
        {},
        points,
    )["physical_refit_capture_scan"]
    plan = _build_plan(scan)

    assert scan["run_alignment_closure"] is False
    assert plan["scan_mode"] == "station_rigid_multidof"
    assert plan["reference_station_ids"] == [1, 2, 3]
    assert [point["name"] for point in plan["points"][:7]] == [
        "joint_nominal",
        "fd_ift_dx_mm_p",
        "fd_ift_dx_mm_m",
        "fd_ift_dy_mm_p",
        "fd_ift_dy_mm_m",
        "fd_ift_ry_mrad_p",
        "fd_ift_ry_mrad_m",
    ]
    assert plan["points"][6]["injected_station_transforms"]["0"][4] == -0.01
    assert all(
        plan["points"][7]["alignment_parameter_values"][name] != 0.0
        for name in ("ift_dx_mm", "ift_dy_mm", "ift_ry_mrad")
    )


def test_multidof_plan_rejects_a_payload_that_moves_the_downstream_reference():
    config = _config()
    points = _joint_rigid_points(config, ("train", "validation"))
    scan = _source_scan_config(
        config,
        {"source_id": "source", "split": "train", "input_xaod": "/tmp/source.root"},
        {},
        points,
    )["physical_refit_capture_scan"]
    scan["rigid_points"][0]["station_transforms"][1][0] = 1.0

    with pytest.raises(ValueError, match="moves reference station 1"):
        _build_plan(scan)


def test_multidof_plan_accepts_explicit_iteration_centred_finite_differences():
    config = _config()
    points = _joint_rigid_points(config, ("train",))
    scan = _source_scan_config(
        config,
        {"source_id": "source", "split": "train", "input_xaod": "/tmp/source.root"},
        {},
        points,
    )["physical_refit_capture_scan"]
    nominal = copy.deepcopy(scan["rigid_points"][0])
    anchor_values = {"ift_dx_mm": 2.0, "ift_dy_mm": -1.5, "ift_ry_mrad": 35.0}

    def point(name, values, role, parameter=None, sign=None):
        transform = [values["ift_dx_mm"], values["ift_dy_mm"], 0.0, 0.0, values["ift_ry_mrad"] / 1.0e3, 0.0]
        result = {
            "name": name,
            "point_role": role,
            "direction_trial": name,
            "alignment_parameter_values": dict(values),
            "station_transforms": {0: transform, 1: [0.0] * 6, 2: [0.0] * 6, 3: [0.0] * 6},
        }
        if parameter is not None:
            result.update(
                {
                    "finite_difference_for": parameter,
                    "probe_sign": sign,
                    "finite_difference_anchor": "iteration_anchor",
                }
            )
        return result

    iteration_points = [nominal, point("iteration_anchor", anchor_values, "iteration_anchor")]
    steps = {"ift_dx_mm": 0.5, "ift_dy_mm": 0.5, "ift_ry_mrad": 10.0}
    for parameter, step in steps.items():
        for sign, suffix in ((1.0, "p"), (-1.0, "m")):
            values = dict(anchor_values)
            values[parameter] += sign * step
            iteration_points.append(
                point(
                    f"iteration_fd_{parameter}_{suffix}",
                    values,
                    "finite_difference_positive" if sign > 0.0 else "finite_difference_negative",
                    parameter,
                    "positive" if sign > 0.0 else "negative",
                )
            )
    scan["rigid_points"] = iteration_points

    plan = _build_plan(scan)

    probes = [item for item in plan["points"] if item["finite_difference_anchor"] is not None]
    assert len(probes) == 6
    assert {item["finite_difference_anchor"] for item in probes} == {"iteration_anchor"}


def test_iteration_compiler_writes_anchor_centred_physical_probe_bank():
    config = _config()
    points = _joint_rigid_points(config, ("train",))
    scan = _source_scan_config(
        config,
        {"source_id": "source", "split": "train", "input_xaod": "/tmp/source.root"},
        {},
        points,
    )

    compiled, contract = compile_iteration(
        scan,
        iteration=3,
        current_values={"ift_dx_mm": 2.0, "ift_dy_mm": -1.5, "ift_ry_mrad": 35.0},
    )
    plan = _build_plan(compiled["physical_refit_capture_scan"])

    assert contract["anchor_point"] == "iteration_03_anchor"
    assert [item["name"] for item in plan["points"][:2]] == [
        "iteration_03_reference",
        "iteration_03_anchor",
    ]
    anchor = plan["points"][1]
    assert anchor["alignment_parameter_values"] == {
        "ift_dx_mm": 2.0,
        "ift_dy_mm": -1.5,
        "ift_ry_mrad": 35.0,
    }
    assert anchor["injected_station_transforms"]["0"] == [2.0, -1.5, 0.0, 0.0, 0.035, 0.0]
