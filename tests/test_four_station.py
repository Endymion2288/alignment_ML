from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from alignment.four_station import (
    FORMULATION,
    FREE_COMPONENTS,
    GAUGE_COMMON_MODE,
    GAUGE_REFERENCE_STATION,
    GAUGE_UNCONSTRAINED_FULL,
    IDENTITY_SIX,
    N_RELATIVE_FREE_PARAMETERS,
    STATION_IDS,
    add_common_six_vector,
    apply_additive_common_mode_constraint,
    apply_left_common_mode_constraint,
    apply_reference_station_gauge,
    capture_relative_tables,
    classify_scaled_singular_vector,
    compose_six_vectors,
    default_parameter_specs,
    finite_rotation_additive_is_not_automatically_a_gauge,
    formulation_contract,
    free_parameter_names,
    identity_station_transforms,
    invert_six_vector,
    draw_relative_native,
    left_multiply_all,
    native_values_from_station_transforms,
    relative_family_payloads,
    RELATIVE_CURRICULUM_KIND,
    s0_gauge_transforms_from_relative_native,
    zero_parameter_values,
    matrix_to_six_vector,
    parameter_name,
    parse_parameter_name,
    relative_alignment_table,
    relative_free_parameter_names,
    relatives_agree,
    require_four_station_route_selected_contract,
    six_vector_to_matrix,
    station_transforms_from_native_values,
    survey_parameter_names,
)
from alignment.physical_jacobian import station_transforms_with_parameter_values
from scripts.prepare_four_station_identifiability_pilot import compile_four_station_identifiability_pilot
from scripts.run_physical_refit_capture_scan import _build_plan
from scripts.write_station_alignment_payload import parse_transform


def _legacy_scan():
    from scripts.build_physical_curriculum_corpus import _joint_rigid_points, _source_scan_config

    config = {
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
    points = _joint_rigid_points(config, ("train",))
    return _source_scan_config(
        config,
        {"source_id": "source", "split": "train", "input_xaod": "/tmp/source.root"},
        {},
        points,
    )["physical_refit_capture_scan"]


def _four_station_scan(*, extra_points=None):
    specs = [
        {
            "name": parameter_name(station, component),
            "station_id": station,
            "component": component,
            "unit": "mm" if component.endswith("_mm") else "mrad",
            "finite_difference_step": 0.5 if component.endswith("_mm") else 10.0,
            "severity_scale": 5.0 if component.endswith("_mm") else 60.0,
        }
        for station in STATION_IDS
        for component in ("dx_mm", "dy_mm")
    ]
    names = [str(spec["name"]) for spec in specs]
    zeros = {name: 0.0 for name in names}
    identity = {station: list(IDENTITY_SIX) for station in STATION_IDS}
    points = [
        {
            "name": "nominal",
            "point_role": "nominal",
            "alignment_parameter_values": dict(zeros),
            "station_transforms": copy.deepcopy(identity),
        }
    ]
    for spec in specs:
        name = str(spec["name"])
        step = float(spec["finite_difference_step"])
        station = int(spec["station_id"])
        index = 0 if spec["component"] == "dx_mm" else 1
        for sign, suffix, role in ((1.0, "p", "finite_difference_positive"), (-1.0, "m", "finite_difference_negative")):
            transforms = copy.deepcopy(identity)
            transforms[station][index] = sign * step
            values = dict(zeros)
            values[name] = sign * step
            points.append(
                {
                    "name": f"fd_{name}_{suffix}",
                    "point_role": role,
                    "alignment_parameter_values": values,
                    "station_transforms": transforms,
                    "finite_difference_for": name,
                    "probe_sign": "positive" if sign > 0.0 else "negative",
                }
            )
    if extra_points:
        points.extend(extra_points)
    return {
        "scan_mode": "station_rigid_multidof",
        "alignment_formulation": FORMULATION,
        "gauge": GAUGE_UNCONSTRAINED_FULL,
        "station_ids": list(STATION_IDS),
        "reference_station_ids": [],
        "movable_station_ids": list(STATION_IDS),
        "condition_axis": "four_station_identifiability_l2",
        "q_over_p_mode": 0,
        "alignment_parameter_specs": specs,
        "rigid_points": points,
    }


def test_parameter_schema_has_twenty_free_and_four_survey_coordinates():
    specs = default_parameter_specs(include_survey_dz=True)
    assert len(specs) == 24
    assert len(free_parameter_names()) == 20
    assert survey_parameter_names() == ("s0_dz_mm", "s1_dz_mm", "s2_dz_mm", "s3_dz_mm")
    assert {spec["station_id"] for spec in specs} == set(STATION_IDS)
    assert all(spec["scope"] == "station" for spec in specs)
    free = [spec for spec in specs if spec["role"] == "free"]
    survey = [spec for spec in specs if spec["role"] == "survey_constrained"]
    assert len(free) == 20
    assert len(survey) == 4
    assert parse_parameter_name("s3_ry_mrad") == (3, "ry_mrad")


def test_payload_se3_round_trip_and_true_left_gauge_preserves_relatives():
    original = (1.25, -0.5, 0.4, 0.03, -0.02, 0.01)
    assert np.allclose(matrix_to_six_vector(six_vector_to_matrix(original)), original, atol=1.0e-12)
    base = {
        0: (0.2, -0.1, 0.0, 0.01, 0.0, 0.0),
        1: (0.0, 0.3, 0.0, 0.0, -0.015, 0.0),
        2: (-0.4, 0.0, 0.0, 0.0, 0.0, 0.02),
        3: (0.1, 0.2, 0.3, 0.0, 0.0, 0.0),
    }
    common = (0.5, -0.25, 0.1, 0.02, -0.01, 0.015)
    assert relatives_agree(base, left_multiply_all(base, common))
    composed = compose_six_vectors(common, invert_six_vector(common))
    assert np.allclose(composed, IDENTITY_SIX, atol=1.0e-12)


def test_additive_translation_is_a_gauge_but_finite_additive_rotation_is_not():
    base = identity_station_transforms()
    base["1"] = [0.3, -0.2, 0.0, 0.0, 0.0, 0.0]
    base["3"] = [0.0, 0.4, 0.0, 0.0, 0.0, 0.0]
    shifted = add_common_six_vector(base, (0.7, -0.2, 0.15, 0.0, 0.0, 0.0))
    gauged = left_multiply_all(base, (0.7, -0.2, 0.15, 0.0, 0.0, 0.0))
    assert relatives_agree(shifted, gauged)
    assert finite_rotation_additive_is_not_automatically_a_gauge()


def test_reference_and_common_mode_charts_agree_on_relative_geometry():
    injected = {
        0: (0.20, -0.10, 0.0, 0.002, 0.0, 0.0),
        1: (0.35, 0.00, 0.0, 0.0, -0.004, 0.0),
        2: (0.10, 0.15, 0.0, 0.0, 0.0, 0.003),
        3: (0.00, 0.05, 0.0, -0.001, 0.0, 0.0),
    }
    by_s0 = apply_reference_station_gauge(injected, reference_station=0)
    by_s3 = apply_reference_station_gauge(injected, reference_station=3)
    common = apply_left_common_mode_constraint(injected)
    additive = apply_additive_common_mode_constraint(injected)
    assert np.allclose(by_s0["0"], IDENTITY_SIX, atol=1.0e-12)
    assert np.allclose(by_s3["3"], IDENTITY_SIX, atol=1.0e-12)
    assert relatives_agree(injected, by_s0)
    assert relatives_agree(injected, by_s3)
    assert relatives_agree(by_s0, by_s3)
    assert relatives_agree(injected, common)
    assert not relatives_agree(injected, additive, atol=1.0e-8)
    table = relative_alignment_table(injected)
    assert set(table) == {"0_1", "1_2", "2_3", "0_3", "0_2", "1_3"}


def test_fixing_a_station_without_rewriting_others_is_not_called_a_gauge():
    injected = {
        0: (0.20, 0.0, 0.0, 0.0, 0.0, 0.0),
        1: (0.35, 0.0, 0.0, 0.0, 0.0, 0.0),
        2: (0.10, 0.0, 0.0, 0.0, 0.0, 0.0),
        3: (0.00, 0.0, 0.0, 0.0, 0.0, 0.0),
    }
    forced_s0_identity = dict(injected)
    forced_s0_identity[0] = IDENTITY_SIX
    assert not relatives_agree(injected, forced_s0_identity, atol=1.0e-9)


def test_legacy_multidof_plan_still_rejects_a_moved_downstream_reference():
    scan = _legacy_scan()
    assert "alignment_formulation" not in scan
    plan = _build_plan(scan)
    assert plan["alignment_formulation"] == "legacy_ift_vs_fixed_downstream"
    assert plan["reference_station_ids"] == [1, 2, 3]
    assert plan["movable_station_ids"] == [0]
    scan["rigid_points"][0]["station_transforms"][2][1] = 0.4
    with pytest.raises(ValueError, match="moves reference station 2"):
        _build_plan(scan)


def test_four_station_plan_writes_independent_payloads_and_rejects_silent_s0_reference():
    scan = _four_station_scan()
    held = {
        "name": "closure_relative",
        "point_role": "held_out_closure",
        "alignment_parameter_values": {
            **{name: 0.0 for name in [spec["name"] for spec in scan["alignment_parameter_specs"]]},
            "s1_dx_mm": 0.30,
            "s2_dy_mm": -0.20,
            "s3_dx_mm": 0.10,
        },
        "station_transforms": {
            0: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1: [0.30, 0.0, 0.0, 0.0, 0.0, 0.0],
            2: [0.0, -0.20, 0.0, 0.0, 0.0, 0.0],
            3: [0.10, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    }
    scan["rigid_points"].append(held)
    plan = _build_plan(scan)
    assert plan["alignment_formulation"] == FORMULATION
    assert plan["gauge"] == GAUGE_UNCONSTRAINED_FULL
    assert plan["reference_station_ids"] == []
    assert plan["movable_station_ids"] == [0, 1, 2, 3]
    closure = next(point for point in plan["points"] if point["name"] == "closure_relative")
    assert closure["injected_station_transforms"]["1"][0] == 0.30
    assert closure["injected_station_transforms"]["2"][1] == -0.20
    assert closure["injected_station_transforms"]["3"][0] == 0.10
    assert closure["injected_station_transforms"]["0"] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    scan["reference_station_ids"] = [0]
    with pytest.raises(ValueError, match="forbids reference_station_ids"):
        _build_plan(scan)


def test_four_station_compiler_linearizes_at_nominal_and_probes_every_station():
    specs = default_parameter_specs(include_survey_dz=True)
    template = {
        "physical_refit_capture_scan": {
            "scan_mode": "station_rigid_multidof",
            "alignment_formulation": FORMULATION,
            "gauge": GAUGE_UNCONSTRAINED_FULL,
            "station_ids": list(STATION_IDS),
            "reference_station_ids": [],
            "movable_station_ids": list(STATION_IDS),
            "condition_axis": "four_station_identifiability_l2",
            "q_over_p_mode": 0,
            "alignment_parameter_specs": specs,
            "rigid_points": [
                {
                    "name": "nominal",
                    "point_role": "nominal",
                    "station_transforms": identity_station_transforms(),
                }
            ],
            "held_out_closure_points": [
                {
                    "name": "closure_relative",
                    "alignment_parameter_values": {"s1_dx_mm": 0.30, "s2_dy_mm": -0.20, "s3_ry_mrad": 5.0},
                }
            ],
        }
    }
    current = {str(spec["name"]): 0.0 for spec in specs}
    compiled, contract = compile_four_station_identifiability_pilot(
        template, iteration=0, current_values=current
    )
    plan = _build_plan(compiled["physical_refit_capture_scan"])
    assert contract["n_free_parameters"] == 20
    assert contract["no_station_is_assumed_correct"] is True
    assert plan["points"][0]["name"] == "iteration_00_reference"
    names = [point["name"] for point in plan["points"]]
    assert "iteration_00_fd_s0_dx_mm_p" in names
    assert "iteration_00_fd_s3_rz_mrad_m" in names
    assert "iteration_00_fd_s2_dz_mm_p" in names
    assert len(plan["points"]) == 1 + 2 * 24 + 1
    closure = next(point for point in plan["points"] if point["name"] == "iteration_00_closure_relative")
    assert closure["injected_station_transforms"]["1"][0] == pytest.approx(0.30)
    assert closure["injected_station_transforms"]["2"][1] == pytest.approx(-0.20)
    assert closure["injected_station_transforms"]["3"][4] == pytest.approx(0.005)
    assert closure["injected_station_transforms"]["0"] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_four_station_parameter_update_writes_independent_station_slots():
    specs = (
        {"name": "s0_dx_mm", "station_id": 0, "component": "dx_mm"},
        {"name": "s2_ry_mrad", "station_id": 2, "component": "ry_mrad"},
        {"name": "s3_dz_mm", "station_id": 3, "component": "dz_mm"},
    )
    transforms = station_transforms_with_parameter_values(
        specs,
        identity_station_transforms(),
        {"s0_dx_mm": 1.5, "s2_ry_mrad": 20.0, "s3_dz_mm": -0.8},
    )
    assert transforms["0"] == [1.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert transforms["1"] == list(IDENTITY_SIX)
    assert transforms["2"] == [0.0, 0.0, 0.0, 0.0, 0.02, 0.0]
    assert transforms["3"] == [0.0, 0.0, -0.8, 0.0, 0.0, 0.0]


def test_payload_parser_accepts_independent_station_transforms():
    assert parse_transform("0:0:0:0:0:0:0")[0] == 0
    assert parse_transform("1:0.3:-0.2:0:0:0:0")[1][0] == 0.3
    assert parse_transform("2:0:0:0:0:0.01:0")[1][4] == 0.01
    assert parse_transform("3:0:0:1.5:0:0:0")[1][2] == 1.5


def test_reference_station_gauge_contract_does_not_privilege_s0():
    s3 = formulation_contract(gauge=GAUGE_REFERENCE_STATION, reference_station=3)
    assert s3["reference_station_ids"] == [3]
    assert s3["movable_station_ids"] == [0, 1, 2]
    common = formulation_contract(gauge=GAUGE_COMMON_MODE)
    assert common["reference_station_ids"] == []
    assert common["movable_station_ids"] == [0, 1, 2, 3]


def test_singular_vector_labels_common_and_long_baseline_modes():
    names = [parameter_name(station, component) for station in STATION_IDS for component in FREE_COMPONENTS]
    common_dx = np.zeros(len(names))
    for station in STATION_IDS:
        common_dx[names.index(parameter_name(station, "dx_mm"))] = 0.5
    labels = classify_scaled_singular_vector(names, common_dx)
    assert "global_common_dx_mm" in labels["labels"]
    long_ry = np.zeros(len(names))
    long_ry[names.index("s0_ry_mrad")] = 0.7
    long_ry[names.index("s3_ry_mrad")] = -0.7
    labels = classify_scaled_singular_vector(names, long_ry)
    assert "long_baseline_ry_mrad" in labels["labels"]


def test_relative_free_names_are_fifteen_and_do_not_privilege_s0():
    s0 = relative_free_parameter_names(gauge=GAUGE_REFERENCE_STATION, reference_station=0)
    s3 = relative_free_parameter_names(gauge=GAUGE_REFERENCE_STATION, reference_station=3)
    assert len(s0) == N_RELATIVE_FREE_PARAMETERS == 15
    assert len(s3) == 15
    assert all(not name.startswith("s0_") for name in s0)
    assert all(not name.startswith("s3_") for name in s3)
    assert "s0_dx_mm" in s3
    assert "s3_ry_mrad" in s0
    assert "s0_dz_mm" not in s0
    common = relative_free_parameter_names(gauge=GAUGE_COMMON_MODE)
    assert common == free_parameter_names()
    with pytest.raises(ValueError, match="not a column-reduced"):
        relative_free_parameter_names(gauge=GAUGE_UNCONSTRAINED_FULL)


def test_relative_delta_t_capture_is_gauge_invariant():
    injected = {
        0: (0.25, 0.0, 0.0, 0.0, 0.0, 0.0),
        1: (0.55, 0.0, 0.0, 0.0, 0.0, 0.0),
        2: (0.25, -0.20, 0.0, 0.0, 0.0, 0.0),
        3: (0.25, 0.0, 0.0, 0.0, 0.005, 0.0),
    }
    recovered = {
        "s1_dx_mm": 0.30,
        "s2_dy_mm": -0.20,
        "s3_ry_mrad": 5.0,
    }
    payload = station_transforms_from_native_values(recovered)
    tolerances = {
        "dx_mm": 0.50,
        "dy_mm": 0.45,
        "dz_mm": 1.00,
        "rx_mrad": 0.50,
        "ry_mrad": 0.50,
        "rz_mrad": 4.0,
    }
    assert capture_relative_tables(injected, payload, tolerances)["success"] is True
    assert capture_relative_tables(
        injected, apply_reference_station_gauge(injected, reference_station=3), tolerances
    )["success"] is True
    wrong = dict(payload)
    wrong["3"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert capture_relative_tables(injected, wrong, tolerances)["success"] is False


def test_subset_bank_keeps_aligned_columns():
    from scripts.audit_6dof_identifiability import _subset_bank

    bank = {
        "names": ("s0_dx_mm", "s1_dx_mm", "s2_dy_mm"),
        "specs": (
            {"name": "s0_dx_mm"},
            {"name": "s1_dx_mm"},
            {"name": "s2_dy_mm"},
        ),
        "scales": np.asarray([5.0, 5.0, 5.0]),
        "anchor_values": np.asarray([0.0, 0.0, 0.0]),
        "reference_values": np.asarray([0.1, 0.2, -0.3]),
        "positive_values": np.asarray([0.5, 0.5, 0.5]),
        "negative_values": np.asarray([-0.5, -0.5, -0.5]),
        "positive_residual": np.zeros((3, 2, 4)),
        "negative_residual": np.zeros((3, 2, 4)),
        "anchor_residual": np.zeros((2, 4)),
    }
    bank["positive_residual"][1] = 1.0
    subset = _subset_bank(bank, ("s1_dx_mm", "s2_dy_mm"))
    assert subset["names"] == ("s1_dx_mm", "s2_dy_mm")
    assert subset["reference_values"].tolist() == [0.2, -0.3]
    assert subset["positive_residual"].shape == (2, 2, 4)
    assert float(subset["positive_residual"][0, 0, 0]) == 1.0
    with pytest.raises(ValueError, match="unknown or duplicated"):
        _subset_bank(bank, ("s1_dx_mm", "s1_dx_mm"))


def test_relative_closure_operating_point_rejects_unconstrained_20d():
    from scripts.run_four_station_relative_closure import _load_operating_point

    payload = _load_operating_point(
        Path("configs/physical_refit_four_station_relative_closure.yaml")
    )
    assert payload["unconstrained_20d_admitted"] is False
    assert payload["n_admitted_free_parameters"] == 15
    assert set(payload["relative_delta_t_capture_tolerance"]) == {
        "dx_mm",
        "dy_mm",
        "dz_mm",
        "rx_mrad",
        "ry_mrad",
        "rz_mrad",
    }


def test_four_station_route_selected_contract_keeps_all_stations_and_15d_solve():
    plan = {
        "alignment_formulation": FORMULATION,
        "q_over_p_mode": 0,
        "movable_station_ids": [0, 1, 2, 3],
        "reference_station_ids": [],
    }
    s0 = relative_free_parameter_names(gauge=GAUGE_REFERENCE_STATION, reference_station=0)
    assert require_four_station_route_selected_contract(
        plan, only_parameters=s0, observation_statistics="physical_edge_deduplicated"
    ) == s0
    shuffled = tuple(reversed(s0))
    assert require_four_station_route_selected_contract(
        plan, only_parameters=shuffled, observation_statistics="physical_edge_deduplicated"
    ) == s0
    with pytest.raises(ValueError, match="forbids unconstrained"):
        require_four_station_route_selected_contract(
            plan, only_parameters=None, observation_statistics="physical_edge_deduplicated"
        )
    with pytest.raises(ValueError, match="physical_edge_deduplicated"):
        require_four_station_route_selected_contract(
            plan, only_parameters=s0, observation_statistics="replica_weighted"
        )
    with pytest.raises(ValueError, match="silently discard"):
        require_four_station_route_selected_contract(
            {**plan, "movable_station_ids": [0]},
            only_parameters=s0,
            observation_statistics="physical_edge_deduplicated",
        )
    with pytest.raises(ValueError, match="15-DoF"):
        require_four_station_route_selected_contract(
            plan,
            only_parameters=free_parameter_names(),
            observation_statistics="physical_edge_deduplicated",
        )


def test_relative_sample_is_15d_s0_chart_then_left_se3_not_20d_then_gauge():
    rng = np.random.default_rng(20260821)
    names = relative_free_parameter_names(gauge=GAUGE_REFERENCE_STATION, reference_station=0)
    relative_native = draw_relative_native(rng, translation_mm=0.5, rotation_mrad=5.0)
    assert set(relative_native) == set(names)
    assert all(not name.startswith("s0_") for name in relative_native)
    relative = s0_gauge_transforms_from_relative_native(relative_native)
    assert relative["0"] == list(IDENTITY_SIX)
    common = (0.25, -0.10, 0.0, 0.0, 0.002, 0.0)
    relative, control = relative_family_payloads(relative_native, common)
    assert relatives_agree(relative, control)
    assert control["0"][0] == pytest.approx(0.25)
    assert not np.allclose(control["3"], relative["3"])
    native = native_values_from_station_transforms(relative)
    assert native["s0_dx_mm"] == pytest.approx(0.0)
    assert native["s3_dz_mm"] == pytest.approx(0.0)


def test_relative_curriculum_compiles_without_finite_difference_probes():
    from scripts.config_loader import load_yaml_with_base
    from scripts.prepare_four_station_relative_curriculum import compile_four_station_relative_curriculum

    template = load_yaml_with_base(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "physical_refit_four_station_relative_association_curriculum.yaml"
    )
    specs = template["physical_refit_capture_scan"]["alignment_parameter_specs"]
    compiled, contract = compile_four_station_relative_curriculum(
        template,
        iteration=0,
        current_values=zero_parameter_values(specs),
    )
    scan = compiled["physical_refit_capture_scan"]
    assert scan["relative_curriculum"] == RELATIVE_CURRICULUM_KIND
    assert scan["require_central_finite_difference_probes"] is False
    plan = _build_plan(scan)
    names = [point["name"] for point in plan["points"]]
    assert names[0] == "iteration_00_reference"
    assert "iteration_00_hard_s3_ry" in names
    assert "iteration_00_hard_s3_ry_plus_common" in names
    assert "iteration_00_draw_00" in names
    assert "iteration_00_draw_00_plus_common" in names
    assert "iteration_00_draw_01_plus_common" in names
    assert len(plan["points"]) == 7
    assert all("fd_" not in name for name in names)
    hard = next(point for point in plan["points"] if point["name"] == "iteration_00_hard_s3_ry")
    twin = next(point for point in plan["points"] if point["name"] == "iteration_00_hard_s3_ry_plus_common")
    assert hard["relative_family"] == "hard_s3_ry"
    assert twin["gauge_role"] == "left_se3_control"
    assert hard["injected_station_transforms"]["0"] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert hard["injected_station_transforms"]["1"][0] == pytest.approx(0.30)
    assert hard["injected_station_transforms"]["3"][4] == pytest.approx(0.005)
    assert relatives_agree(hard["injected_station_transforms"], twin["injected_station_transforms"])
    assert twin["injected_station_transforms"]["0"] != [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert contract["reference_point"] == "iteration_00_reference"
    assert contract["sampling_reference_station"] == 0


def test_skipping_finite_difference_probes_outside_relative_curriculum_is_rejected():
    scan = _four_station_scan()
    scan["require_central_finite_difference_probes"] = False
    with pytest.raises(ValueError, match="outside the 15-DoF relative curriculum"):
        _build_plan(scan)


def test_relative_sampling_refuses_to_leave_the_local_linearization_region():
    from scripts.config_loader import load_yaml_with_base
    from scripts.prepare_four_station_relative_curriculum import compile_four_station_relative_curriculum

    template = load_yaml_with_base(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "physical_refit_four_station_relative_association_curriculum.yaml"
    )
    template["physical_refit_capture_scan"]["relative_sampling"]["relative_bounds"]["rotation_mrad"] = 8.0
    specs = template["physical_refit_capture_scan"]["alignment_parameter_specs"]
    with pytest.raises(ValueError, match="local region"):
        compile_four_station_relative_curriculum(
            template,
            iteration=0,
            current_values=zero_parameter_values(specs),
        )


def test_matched_association_evaluator_groups_gauge_twins_and_refuses_workbook_52_payloads():
    from scripts.evaluate_four_station_matched_association import (
        FORBIDDEN_PAYLOADS,
        _families_from_plan,
        _gates,
        _gauge_audit,
    )

    families = _families_from_plan(
        {
            "points": [
                {"name": "iteration_00_reference", "relative_family": "nominal", "gauge_role": "identity"},
                {"name": "iteration_00_hard_s3_ry", "relative_family": "hard_s3_ry", "gauge_role": "s0_sampling_chart"},
                {
                    "name": "iteration_00_hard_s3_ry_plus_common",
                    "relative_family": "hard_s3_ry",
                    "gauge_role": "left_se3_control",
                },
            ]
        }
    )
    assert families == {"hard_s3_ry": ("iteration_00_hard_s3_ry", "iteration_00_hard_s3_ry_plus_common")}
    mapped = _gates(
        {
            "raw_complete_truth_chain_recall_min": 0.9,
            "adjacent_truth_edge_recall_min": 0.9,
            "complete_track_efficiency_drop_vs_nominal_max": 0.1,
            "complete_track_purity_drop_vs_nominal_max": 0.05,
            "track_fake_rate_increase_vs_nominal_max": 0.05,
        }
    )
    assert mapped["vs_nominal_efficiency_drop_max"] == 0.1
    assert "iteration_00_closure_relative" in FORBIDDEN_PAYLOADS
    reports = {
        "iteration_00_hard_s3_ry": {
            "selected_route": {
                "complete_track_efficiency": 0.80,
                "complete_track_purity": 0.96,
                "track_fake_rate": 0.03,
            },
            "by_adjacent_station_pair": {"2->3": {"association": {"association_efficiency": 0.70}}},
        },
        "iteration_00_hard_s3_ry_plus_common": {
            "selected_route": {
                "complete_track_efficiency": 0.81,
                "complete_track_purity": 0.95,
                "track_fake_rate": 0.04,
            },
            "by_adjacent_station_pair": {"2->3": {"association": {"association_efficiency": 0.72}}},
        },
    }
    audit = _gauge_audit(
        reports,
        families,
        {
            "complete_track_efficiency_twin_abs_diff_max": 0.05,
            "complete_track_purity_twin_abs_diff_max": 0.05,
            "track_fake_rate_twin_abs_diff_max": 0.05,
            "adjacent_23_efficiency_twin_abs_diff_max": 0.08,
        },
    )
    assert audit["ok"] is True
