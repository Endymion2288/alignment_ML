from __future__ import annotations

import json

import pytest

from alignment.sequential_contrast import remaining_after_block_step
from scripts.collect_sequential_contrast_remaining import _remaining_from_update
from scripts.prepare_contrast_remaining_iteration import _held_out_from_remaining
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


def test_remaining_after_block_step_keeps_unfloated_geometry():
    injected = {"C_dx": 0.04033375082751082, "C_rx": 0.38944604413854195}
    remaining = remaining_after_block_step(
        injected=injected, recovered_floated=0.0398, floated="C_dx"
    )
    assert remaining["C_dx"] == pytest.approx(injected["C_dx"] - 0.0398)
    assert remaining["C_rx"] == pytest.approx(injected["C_rx"])
    swapped = remaining_after_block_step(
        injected=injected, recovered_floated=0.311, floated="C_rx"
    )
    assert swapped["C_rx"] == pytest.approx(injected["C_rx"] - 0.311)
    assert swapped["C_dx"] == pytest.approx(injected["C_dx"])


def test_remaining_is_not_the_1d_newton_chart_that_zeros_the_unfloated_contrast():
    injected = {"C_dx": 0.04, "C_rx": 0.389}
    naive_proposed_next = {"C_dx": 0.0398, "C_rx": 0.0}
    remaining = remaining_after_block_step(
        injected=injected, recovered_floated=naive_proposed_next["C_dx"], floated="C_dx"
    )
    assert remaining["C_rx"] != pytest.approx(naive_proposed_next["C_rx"])
    assert remaining["C_rx"] == pytest.approx(0.389)


def test_remaining_rejects_unknown_or_incomplete_inputs():
    with pytest.raises(ValueError, match="floated contrast"):
        remaining_after_block_step(
            injected={"C_dx": 0.04, "C_rx": 0.389}, recovered_floated=0.01, floated="C_ry"
        )
    with pytest.raises(ValueError, match="missing"):
        remaining_after_block_step(injected={"C_dx": 0.04}, recovered_floated=0.01, floated="C_dx")


def test_compile_identically_zero_heldout_requires_explicit_flag():
    with pytest.raises(ValueError, match="identically zero"):
        compile_layer_identifiability_pilot(
            _contrast_template(
                [{"name": "nominal_again", "alignment_parameter_values": {"C_dx": 0.0, "C_rx": 0.0}}]
            ),
            iteration=0,
            current_values={"C_dx": 0.0, "C_rx": 0.0},
            include_finite_differences=False,
        )
    compiled, contract = compile_layer_identifiability_pilot(
        _contrast_template(
            [
                {
                    "name": "after_both",
                    "alignment_parameter_values": {"C_dx": 0.0, "C_rx": 0.0},
                    "allow_identically_zero": True,
                }
            ]
        ),
        iteration=0,
        current_values={"C_dx": 0.0, "C_rx": 0.0},
        include_finite_differences=False,
    )
    assert contract["held_out_only"] is True
    names = [point["name"] for point in compiled["physical_refit_capture_scan"]["rigid_points"]]
    assert names == ["iteration_00_after_both"]


def test_compile_held_out_only_remaining_skips_fd_and_keeps_fixed_contrast():
    template = _contrast_template(
        [
            {
                "name": "start_after_C_dx",
                "alignment_parameter_values": {"C_dx": 0.0012, "C_rx": 0.38944604413854195},
            }
        ]
    )
    template["physical_refit_capture_scan"]["held_out_only"] = True
    compiled, contract = compile_layer_identifiability_pilot(
        template,
        iteration=0,
        current_values={"C_dx": 0.0, "C_rx": 0.0},
        include_finite_differences=False,
    )
    scan = compiled["physical_refit_capture_scan"]
    names = [point["name"] for point in scan["rigid_points"]]
    assert "iteration_00_reference" not in names
    assert not any("fd_C_" in name for name in names)
    assert "iteration_00_start_after_C_dx" in names
    assert contract["held_out_only"] is True
    point = next(item for item in scan["rigid_points"] if item["name"] == "iteration_00_start_after_C_dx")
    assert point["station_transforms"]["0"] == pytest.approx([0.0] * 6)
    assert point["layer_transforms"]["0"]["0"][0] == pytest.approx(0.0012)
    assert point["layer_transforms"]["0"]["2"][0] == pytest.approx(-0.0012)
    assert point["layer_transforms"]["0"]["0"][3] == pytest.approx(0.00038944604413854195)
    assert point["layer_transforms"]["0"]["2"][3] == pytest.approx(-0.00038944604413854195)
    assert point["layer_transforms"]["0"]["1"] == pytest.approx([0.0] * 6)
    plan = _build_plan(scan)
    assert plan["held_out_only"] is True
    assert all(point["finite_difference_for"] is None for point in plan["points"])


def test_held_out_from_remaining_uses_block_step_not_proposed_next():
    payload = {
        "points": [
            {
                "name": "start_after_C_dx",
                "floated": "C_dx",
                "injected": {"C_dx": 0.04, "C_rx": 0.389},
                "recovered_floated": 0.0395,
            }
        ]
    }
    entries = _held_out_from_remaining(payload)
    assert entries[0]["alignment_parameter_values"]["C_dx"] == pytest.approx(0.0005)
    assert entries[0]["alignment_parameter_values"]["C_rx"] == pytest.approx(0.389)


def test_collect_remaining_rejects_joint_2d_update(tmp_path):
    path = tmp_path / "route_selected_update.json"
    path.write_text(
        json.dumps(
            {
                "q_over_p_mode": 0,
                "test_opened": False,
                "only_parameters": ["C_dx", "C_rx"],
                "target_point": "iteration_00_start",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sequential 1-D"):
        _remaining_from_update(path, name="start_after_C_dx", split="train")


def test_uid_tracklets_fall_back_on_held_out_only_remaining(tmp_path):
    from scripts.build_6dof_pilot_physical_corpus import _uid_tracklets_path

    scan_root = tmp_path / "scan"
    plan = {
        "held_out_only": True,
        "points": [
            {
                "name": "iteration_00_start_after_C_dx",
                "condition_magnitude": 0.56,
                "relative_point_dir": "points/iteration_00_start_after_C_dx",
            }
        ],
    }
    path = _uid_tracklets_path(scan_root, plan)
    assert path == scan_root / "points/iteration_00_start_after_C_dx/refit/tracklets.root"
    with pytest.raises(ValueError, match="zero-magnitude"):
        _uid_tracklets_path(scan_root, {"held_out_only": False, "points": plan["points"]})
    nominal = {
        "held_out_only": False,
        "points": [
            {
                "name": "iteration_00_reference",
                "condition_magnitude": 0.0,
                "relative_point_dir": "points/iteration_00_reference",
            },
            {
                "name": "iteration_00_start",
                "condition_magnitude": 0.65,
                "relative_point_dir": "points/iteration_00_start",
            },
        ],
    }
    assert _uid_tracklets_path(scan_root, nominal) == scan_root / "points/iteration_00_reference/refit/tracklets.root"
