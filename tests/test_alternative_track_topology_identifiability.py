from __future__ import annotations

from pathlib import Path

import pytest

from alignment.alternative_track_topology_feasibility import (
    DECISION_NONE,
    DECISION_PORTABLE,
    GO_NO_GO_QUESTION,
    SCHEMA_VERSION,
    TopologyFeatures,
    assign_categories,
    concentrated_bootstrap,
    decide_next_stage,
    evaluate_category_portability,
    fd_steps,
    load_feasibility_config,
    matches_category,
    r0022_shallow_dominated,
)
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
)
from alignment.true_cluster_local_residual import assert_no_alignment_payload


def _config():
    return load_feasibility_config(
        Path("configs/alternative_track_topology_identifiability_feasibility_v1.yaml")
    )


def _features(**overrides) -> TopologyFeatures:
    payload = dict(
        run_id=14973,
        event_id=1,
        route_index=0,
        n_stations=3,
        stations=(0, 1, 2),
        has_ift=True,
        is_complete_four_station=False,
        lever_arm_mm=3100.0,
        slope=0.0002,
        local_ift_slope=0.02,
        incident_angle_rad=0.0002,
        ift_layers=(0, 1, 2),
        n_ift_modules=6,
        jacobian_eligible=True,
    )
    payload.update(overrides)
    return TopologyFeatures(**payload)


def test_config_freezes_protocol_and_physics_not_entry58_tertiles():
    config = _config()
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["real_data_operating_mode"] == OPERATING_MODE
    assert config["do_not_select_events_from_residual_or_cosine"] is True
    assert config["do_not_invent_new_cosine_cut"] is True
    assert config["do_not_enter_full_module_identifiability_map"] is True
    assert config["frozen_v2_checkpoint_sha256"] == FROZEN_V2_CHECKPOINT_SHA256
    physics = config["physics_scales"]
    assert physics["wide_angle_min_slope"] == pytest.approx(0.002)
    assert physics["ip_like_max_slope"] == pytest.approx(0.0005)
    assert physics["wide_angle_min_slope"] != pytest.approx(0.0015438179210822344)
    assert physics["slope_definition"] == "global_route_from_endpoint_tracklet_xyz"
    assert physics["slope_not_local_segmentfit_tx_ty"] is True
    steps = fd_steps(config)
    assert steps["translation_step_mm"] == pytest.approx(0.010)
    assert steps["rotation_step_rad"] == pytest.approx(5.0e-5)
    ids = [row["id"] for row in config["predeclared_categories"]]
    assert ids[0] == "ge3_station_with_ift"
    assert "wide_incident_angle" in ids
    unlock = {row["id"]: row["can_unlock_module_map"] for row in config["predeclared_categories"]}
    assert unlock["ge3_station_with_ift"] is False
    assert unlock["ip_like_forward"] is False
    assert unlock["wide_incident_angle"] is True


def test_global_route_slope_uses_endpoint_xyz_not_local_tx():
    from alignment.alternative_track_topology_feasibility import _global_slope_from_points

    slope = _global_slope_from_points([[0.0, 0.0, -1860.15], [4.287, 0.0, 2427.4]])
    assert slope == pytest.approx(4.287 / (2427.4 + 1860.15), rel=1.0e-6)

    config = _config()
    categories = config["predeclared_categories"]
    ip_like = _features(slope=0.0002)
    wide = _features(slope=0.003)
    four = _features(
        n_stations=4,
        stations=(0, 1, 2, 3),
        is_complete_four_station=True,
        slope=0.001,
        lever_arm_mm=4287.0,
    )
    no_ift = _features(has_ift=False, stations=(1, 2, 3), jacobian_eligible=False)
    assert "ip_like_forward" in assign_categories(ip_like, categories)
    assert "wide_incident_angle" not in assign_categories(ip_like, categories)
    assert "wide_incident_angle" in assign_categories(wide, categories)
    assert "ip_like_forward" not in assign_categories(wide, categories)
    assert "complete_four_station" in assign_categories(four, categories)
    assert "long_lever_arm" in assign_categories(four, categories)
    assert "ge3_station_with_ift" not in assign_categories(no_ift, categories)
    wide_spec = next(row for row in categories if row["id"] == "wide_incident_angle")
    assert matches_category(_features(slope=0.002), wide_spec) is True
    assert matches_category(_features(slope=0.0019), wide_spec) is False


def test_concentrated_bootstrap_and_portability_gate():
    yes_boot = {
        "event_bootstrap": {
            "station_ry_vs_C_dx": {
                "median": 0.35,
                "width_95": 0.08,
                "interval_95": [0.30, 0.40],
                "interval_68": [0.32, 0.38],
            },
            "rank": {"median": 3.0},
        }
    }
    conc = concentrated_bootstrap(0.34, yes_boot, 0.995)
    assert conc["concentrated"] is True
    spec = {"id": "wide_incident_angle", "can_unlock_module_map": True, "role": "high_angular_lever_arm"}
    portable = evaluate_category_portability(
        spec=spec,
        reference={
            "insufficient": False,
            "n_routes": 20,
            "point": {"station_ry_vs_C_dx": 0.34, "rank": 3},
            "bootstrap": yes_boot,
        },
        transfer={
            "insufficient": False,
            "n_routes": 18,
            "point": {"station_ry_vs_C_dx": 0.36, "rank": 3},
            "bootstrap": {
                "event_bootstrap": {"station_ry_vs_C_dx": {"interval_68": [0.31, 0.41]}}
            },
        },
        proxy_ry=0.995,
        min_routes=8,
    )
    assert portable["portable"] is True
    baseline = evaluate_category_portability(
        spec={"id": "ge3_station_with_ift", "can_unlock_module_map": False},
        reference={
            "insufficient": False,
            "n_routes": 80,
            "point": {"station_ry_vs_C_dx": 0.34, "rank": 3},
            "bootstrap": yes_boot,
        },
        transfer={
            "insufficient": False,
            "n_routes": 70,
            "point": {"station_ry_vs_C_dx": 0.36, "rank": 3},
            "bootstrap": yes_boot,
        },
        proxy_ry=0.995,
        min_routes=8,
    )
    assert baseline["portable"] is False


def test_decision_requires_predeclared_portable_topology():
    yes = decide_next_stage(
        evaluations=[{"category": "wide_incident_angle", "portable": True}],
        inventory_summary={"r0022_still_shallow_dominated": False},
        alternative_survey={"independent_survey_available": False},
    )
    assert yes["answer"] == "Yes"
    assert yes["decision"] == DECISION_PORTABLE
    assert yes["go_to_full_module_identifiability_map"] is True
    assert yes["still_no_new_network"] is True
    no = decide_next_stage(
        evaluations=[{"category": "wide_incident_angle", "portable": False}],
        inventory_summary={"r0022_still_shallow_dominated": True},
        alternative_survey={
            "independent_survey_available": False,
            "dedicated_cosmic_or_halo_stream_in_rec_tree": False,
        },
    )
    assert no["answer"] == "No"
    assert no["decision"] == DECISION_NONE
    assert no["go_to_full_module_identifiability_map"] is False
    assert no["go_no_go_question"] == GO_NO_GO_QUESTION
    assert "Do not stack more collision-like events" in no["reason"]
    assert_no_alignment_payload(no)
    inventories = [
        {
            "selected": {
                "n_jacobian_eligible": 80,
                "jacobian_eligible_category_counts": {"wide_incident_angle": 20},
            }
        }
    ]
    assert r0022_shallow_dominated(inventories, {}) is True
    inventories[0]["selected"]["jacobian_eligible_category_counts"]["wide_incident_angle"] = 50
    assert r0022_shallow_dominated(inventories, {}) is False
