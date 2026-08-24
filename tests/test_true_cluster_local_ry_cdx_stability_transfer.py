from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
)
from alignment.true_cluster_local_stability_transfer import (
    DECISION_CANDIDATE,
    DECISION_NOT_TRANSFERABLE,
    GO_NO_GO_QUESTION,
    SCHEMA_VERSION,
    decide_next_stage,
    fd_steps,
    intervals_overlap,
    load_audit_config,
    parameter_subspace_transfer,
    principal_angles,
    stacked_rows,
    summarize_distribution,
    summarize_jacobian,
)
from alignment.true_cluster_local_residual import assert_no_alignment_payload


def _config():
    return load_audit_config(Path("configs/true_cluster_local_ry_cdx_stability_transfer_v1.yaml"))


def test_config_keeps_frozen_protocol_and_forbids_new_cut():
    config = _config()
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["real_data_operating_mode"] == OPERATING_MODE
    assert config["geometry_write_allowed"] is False
    assert config["do_not_retrain_v2"] is True
    assert config["do_not_enter_full_module_identifiability_map"] is True
    assert config["do_not_invent_new_cosine_cut"] is True
    assert config["do_not_use_tracklet_intercept_as_cluster_residual"] is True
    assert config["frozen_v2_checkpoint_sha256"] == FROZEN_V2_CHECKPOINT_SHA256
    assert config["runs"]["reference"]["run"] == 14973
    assert config["runs"]["transfer"]["run"] == 14974
    assert config["runs"]["transfer"]["participates_in_method_selection"] is False
    for spec in config["runs"]["read_only"]:
        assert spec["participates_in_method_selection"] is False
        assert spec["run"] in (14975, 14976)
    steps = fd_steps(config)
    assert steps["translation_step_mm"] == pytest.approx(0.010)
    assert steps["rotation_step_rad"] == pytest.approx(5.0e-5)


def test_percentile_interval_and_overlap():
    values = [0.50, 0.51, 0.52, 0.53, 0.54]
    summary = summarize_distribution(values, reference=0.53)
    assert summary["median"] == pytest.approx(0.52)
    assert summary["reference_inside_95"] is True
    assert intervals_overlap([0.40, 0.60], [0.55, 0.80]) is True
    assert intervals_overlap([0.40, 0.50], [0.60, 0.80]) is False


def test_jacobian_summary_and_identical_subspace_angles():
    rng = np.random.default_rng(0)
    dx = np.ones(30)
    cdx = np.array([1.0, 0.0, -1.0] * 10)
    ry = rng.normal(0.0, 1.0, size=30)
    jacobian = np.column_stack([dx, ry, cdx])
    summary = summarize_jacobian(jacobian)
    assert summary["station_dx_vs_C_dx"] < 0.05
    assert summary["rank"] >= 2
    stacked = stacked_rows(jacobian, [list(range(10)), list(range(10, 20))])
    assert stacked.shape[0] == 20
    angles = principal_angles(np.eye(3), np.eye(3))
    assert max(angles) == pytest.approx(0.0, abs=1.0e-12)
    transfer = parameter_subspace_transfer(summary, summary)
    assert transfer["comparable"] is True
    assert transfer["max_weak_principal_angle_deg"] == pytest.approx(0.0, abs=1.0e-5)


def test_decision_requires_stable_14973_and_14974_overlap():
    yes = decide_next_stage(
        reference_point={"station_ry_vs_C_dx": 0.531},
        bootstrap={
            "event_bootstrap": {
                "station_ry_vs_C_dx": {
                    "median": 0.53,
                    "width_95": 0.08,
                    "interval_95": [0.48, 0.56],
                    "interval_68": [0.50, 0.55],
                }
            },
            "random_half_split": {"station_ry_vs_C_dx": {"width_95": 0.10}},
        },
        coverage={"largest_ry_cdx_shifts_when_leaving_a_group_out": [{"abs_delta_ry_cdx": 0.04}]},
        transfer_point={"station_ry_vs_C_dx": 0.54},
        transfer_bootstrap={
            "event_bootstrap": {"station_ry_vs_C_dx": {"interval_68": [0.51, 0.58]}}
        },
        coverage_matched={"possible": True, "station_ry_vs_C_dx": {"median": 0.53}},
        proxy_ry=0.995,
    )
    assert yes["answer"] == "Yes"
    assert yes["decision"] == DECISION_CANDIDATE
    assert yes["go_to_full_module_identifiability_map"] is True
    assert yes["still_no_new_network"] is True
    assert yes["go_no_go_question"] == GO_NO_GO_QUESTION

    no = decide_next_stage(
        reference_point={"station_ry_vs_C_dx": 0.531},
        bootstrap={
            "event_bootstrap": {
                "station_ry_vs_C_dx": {
                    "median": 0.62,
                    "width_95": 0.55,
                    "interval_95": [0.35, 0.90],
                    "interval_68": [0.42, 0.78],
                }
            },
            "random_half_split": {"station_ry_vs_C_dx": {"width_95": 0.40}},
        },
        coverage={"largest_ry_cdx_shifts_when_leaving_a_group_out": [{"abs_delta_ry_cdx": 0.30}]},
        transfer_point={"station_ry_vs_C_dx": 0.91},
        transfer_bootstrap={
            "event_bootstrap": {"station_ry_vs_C_dx": {"interval_68": [0.85, 0.96]}}
        },
        coverage_matched={"possible": True, "station_ry_vs_C_dx": {"median": 0.88}},
        proxy_ry=0.995,
    )
    assert no["answer"] == "No"
    assert no["decision"] == DECISION_NOT_TRANSFERABLE
    assert no["go_to_full_module_identifiability_map"] is False
    assert no["do_not_sink_ml_to_more_complex_cluster_architecture"] is True
    assert no["read_only_runs_did_not_select_the_method"] is True
    assert_no_alignment_payload(no)
    with pytest.raises(ValueError, match="geometry write"):
        assert_no_alignment_payload({**no, "geometry_write_allowed": True})
