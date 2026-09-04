from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alignment.five_dof_sampling import FREE_PARAMETERS
from alignment.identifiable_subspace import FROZEN_RANK_TOLERANCE
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
)
from alignment.physically_distinct_track_coverage import (
    DECISION_EXPORT,
    DECISION_INSUFFICIENT,
    DECISION_OPEN_FD,
    FilenameGuessError,
    GO_NO_GO_QUESTION,
    INHERITED_ENTRY_59,
    INHERITED_ENTRY_60,
    INHERITED_ENTRY_73,
    ResidualBlindStageError,
    SCHEMA_VERSION,
    SealedTestAccessError,
    admit_candidate,
    build_all_reports,
    decide_next_stage,
    histogram_intersection,
    load_inventory_config,
    outside_envelope_fraction,
    pool_population,
    refuse_fd_identifiability_this_stage,
    refuse_filename_guess,
    refuse_forbidden_operations,
    refuse_sealed_source,
    summarize_events,
)
from alignment.true_cluster_local_residual import assert_no_alignment_payload
from datasets.root_loader import EventTracklets


def _config():
    return load_inventory_config(
        Path("configs/physically_distinct_track_coverage_identifiability_feasibility_v1.yaml")
    )


def _event(stations, tx, ty, *, event_id=1, pdg=13) -> EventTracklets:
    size = len(stations)
    state = np.zeros((size, 4), dtype=np.float64)
    state[:, 2] = tx
    state[:, 3] = ty
    return EventTracklets(
        run_id=1,
        event_id=int(event_id),
        station_id=np.asarray(stations, dtype=np.int16),
        tracklet_id=np.arange(size, dtype=np.int32),
        z_mm=np.asarray([-1860.15, 47.4, 1237.4, 2427.4][:size], dtype=np.float64),
        state=state,
        covariance=np.tile(np.eye(4, dtype=np.float64), (size, 1, 1)),
        chi2=np.ones(size, dtype=np.float64),
        ndof=np.ones(size, dtype=np.float64),
        n_hit=np.full(size, 6, dtype=np.int16),
        hit_pattern=np.full(size, 0b111, dtype=np.uint64),
        truth_particle_id=np.arange(10, 10 + size, dtype=np.int64),
        truth_pdg=np.full(size, int(pdg), dtype=np.int32),
        truth_match_fraction=np.ones(size, dtype=np.float64),
    )


def _source_row(source_id, events, config):
    summary = summarize_events(
        events,
        source_id=source_id,
        physics=config["physics_scales"],
        phase_space=config["phase_space"],
    )
    summary["split"] = "train"
    return summary


def test_config_freezes_protocol_and_forbids_rank_rescue():
    config = _config()
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["stage"] == "residual_blind_coverage_inventory"
    assert config["real_data_operating_mode"] == OPERATING_MODE
    assert config["frozen_v2_checkpoint_sha256"] == FROZEN_V2_CHECKPOINT_SHA256
    assert config["do_not_construct_fd_identifiability_this_stage"] is True
    assert config["do_not_compute_svd_or_rank_this_stage"] is True
    assert config["do_not_select_events_from_residual_or_cosine"] is True
    assert config["do_not_restack_2024_r0022_collision_like"] is True
    assert config["do_not_restack_canonical_5mrad_unused_files"] is True
    assert config["do_not_open_sealed_test"] is True
    assert config["do_not_delete_rz_or_further_dof"] is True
    assert config["do_not_lower_statistical_gates"] is True
    assert config["filename_5mrad_is_not_phase_space_proof"] is True
    assert config["three_arm_authorized"] is False
    assert config["frozen_v2_alignment_loop_authorized"] is False
    assert config["real_data_correction_authorized"] is False
    assert config["geometry_write_allowed"] is False
    assert config["inherited_entry_59_decision"] == INHERITED_ENTRY_59
    assert config["inherited_entry_60_decision"] == INHERITED_ENTRY_60
    assert config["inherited_entry_73_decision"] == INHERITED_ENTRY_73
    contract = config["rigid_station_five_dof_contract"]
    assert tuple(contract["parameter_names"]) == FREE_PARAMETERS
    assert contract["rank_tolerance"] == pytest.approx(FROZEN_RANK_TOLERANCE)
    assert contract["scale_matrix_S"]["ift_dx_mm"] == pytest.approx(5.0)
    assert contract["scale_matrix_S"]["ift_rz_mrad"] == pytest.approx(60.0)
    assert contract["do_not_reconstruct_this_stage"] is True
    gates = config["coverage_gates"]
    assert gates["min_events"] == 200
    assert gates["min_complete_four_station_events"] == 80
    assert gates["do_not_lower_gates"] is True
    assert gates["collision_like_metadata_cannot_admit"] is True
    ids = [row["id"] for row in config["predeclared_candidates"]]
    assert "mc24_100012_100gev_gaussian_theta" in ids
    assert "real_2024_r0022_wide_angle_selected_routes" in ids
    sealed = config["sealed_test"]["forbidden_source_ids"]
    assert "mc24_100116_00030_00039" in sealed
    assert "mc24_100117_00030_00039" in sealed


def test_summarize_events_is_residual_blind_and_counts_four_station_coverage():
    config = _config()
    events = [
        _event([0, 1, 2, 3], 0.01, 0.0, event_id=1),
        _event([0, 1], 0.02, 0.01, event_id=2),
        _event([1, 2, 3], 0.03, 0.0, event_id=3),
    ]
    summary = summarize_events(
        events,
        source_id="s0",
        physics=config["physics_scales"],
        phase_space=config["phase_space"],
    )
    assert summary["residual_blind"] is True
    assert summary["slope_is_not_leftover_residual_rtx_rty"] is True
    assert summary["n_events"] == 3
    assert summary["n_ift_events"] == 2
    assert summary["n_complete_four_station_events"] == 1
    assert summary["station_event_counts"]["0"] == 2
    assert summary["tx"]["finite_count"] == 3
    assert "residual" not in summary
    assert "rank" not in summary
    assert_no_alignment_payload(summary)


def test_phase_space_overlap_and_gates_do_not_lower():
    config = _config()
    gates = config["coverage_gates"]
    canonical_events = [_event([0, 1, 2, 3], 0.001 * index, 0.0, event_id=index) for index in range(12)]
    wide_events = [_event([0, 1, 2, 3], 0.20, 0.05, event_id=index) for index in range(12)]
    canonical = pool_population(
        [_source_row("c0", canonical_events, config), _source_row("c1", canonical_events, config)],
        population_id="canonical",
        config=config,
    )
    wide = pool_population(
        [_source_row("w0", wide_events, config), _source_row("w1", wide_events, config)],
        population_id="wide",
        config=config,
    )
    intersection = histogram_intersection(canonical["_histogram"], wide["_histogram"])
    outside = outside_envelope_fraction(wide["_tx"], wide["_ty"], canonical["quantile_box"])
    assert intersection is not None
    assert outside is not None
    assert outside > gates["min_outside_canonical_envelope_fraction"]
    insufficient = admit_candidate(
        {
            "id": "tiny_distinct",
            "metadata_class": "mc_particle_gun_const_100gev_gaussian_theta",
            "physically_distinct_hypothesis": True,
            "same_production_as_canonical": False,
        },
        pooled=wide,
        overlap={
            "outside_canonical_quantile_box_fraction": outside,
            "histogram_intersection": intersection,
        },
        gates=gates,
    )
    assert insufficient["admitted_to_separate_fd_campaign"] is False
    assert insufficient["gates_not_lowered"] is True
    assert "n_events" in insufficient["reason"]


def test_collision_like_and_same_production_cannot_admit():
    config = _config()
    gates = config["coverage_gates"]
    pooled = {
        "n_events": 500,
        "n_ift_events": 500,
        "n_complete_four_station_events": 400,
        "n_sources": 4,
        "coverage_measured": True,
        "stations_present": [0, 1, 2, 3],
        "wide_local_slope_fraction": 0.9,
    }
    overlap = {
        "outside_canonical_quantile_box_fraction": 0.8,
        "histogram_intersection": 0.1,
    }
    collision = admit_candidate(
        {
            "id": "r0022",
            "metadata_class": "real_r0022_collision_like",
            "physically_distinct_hypothesis": False,
            "do_not_restack": True,
        },
        pooled=pooled,
        overlap=overlap,
        gates=gates,
    )
    assert collision["admitted_to_separate_fd_campaign"] is False
    assert collision["authorize_residual_blind_export"] is False
    same = admit_candidate(
        {
            "id": "unused_100043",
            "metadata_class": "canonical_5mrad_flukaE_particle_gun",
            "physically_distinct_hypothesis": False,
            "same_production_as_canonical": True,
        },
        pooled=pooled,
        overlap=overlap,
        gates=gates,
    )
    assert same["admitted_to_separate_fd_campaign"] is False


def test_metadata_distinct_unmeasured_xaod_can_authorize_export_not_fd():
    config = _config()
    gates = config["coverage_gates"]
    pooled = {
        "n_events": 0,
        "n_ift_events": 0,
        "n_complete_four_station_events": 0,
        "n_sources": 0,
        "coverage_measured": False,
        "stations_present": [],
        "wide_local_slope_fraction": None,
    }
    admission = admit_candidate(
        {
            "id": "floor",
            "metadata_class": "mc_particle_gun_floor_origin",
            "physically_distinct_hypothesis": True,
            "same_production_as_canonical": False,
        },
        pooled=pooled,
        overlap={"outside_canonical_quantile_box_fraction": None, "histogram_intersection": None},
        gates=gates,
        xaod={
            "exists": True,
            "n_events": 100000,
            "collection_keys_of_interest": {"SegmentFit": True, "SCT_ClusterContainer": True},
        },
    )
    assert admission["admitted_to_separate_fd_campaign"] is False
    assert admission["authorize_residual_blind_export"] is True
    assert admission["verdict"] == "insufficient_authorize_residual_blind_export"
    empty_fluka = admit_candidate(
        {
            "id": "fluka",
            "metadata_class": "mc_fluka_full_shower",
            "physically_distinct_hypothesis": True,
            "known_empty_segmentfit": True,
        },
        pooled=pooled,
        overlap={"outside_canonical_quantile_box_fraction": None, "histogram_intersection": None},
        gates=gates,
        xaod={
            "exists": True,
            "n_events": 200000,
            "collection_keys_of_interest": {"SegmentFit": True},
        },
    )
    assert empty_fluka["authorize_residual_blind_export"] is False


def test_decision_labels_and_forbidden_operations():
    no = decide_next_stage(
        [{"admitted_to_separate_fd_campaign": False, "authorize_residual_blind_export": False}]
    )
    assert no["decision"] == DECISION_INSUFFICIENT
    assert no["freeze_current_track_coverage_insufficient_for_unconstrained_rigid_station_5dof"] is True
    assert no["open_gauge_constrained_or_external_constraint_branch"] is True
    assert no["three_arm_authorized"] is False
    assert no["geometry_write_allowed"] is False
    assert no["go_no_go_question"] == GO_NO_GO_QUESTION
    assert_no_alignment_payload(no)
    export = decide_next_stage(
        [
            {
                "id": "floor",
                "admitted_to_separate_fd_campaign": False,
                "authorize_residual_blind_export": True,
            }
        ]
    )
    assert export["decision"] == DECISION_EXPORT
    assert export["freeze_current_track_coverage_insufficient_for_unconstrained_rigid_station_5dof"] is False
    yes = decide_next_stage(
        [{"id": "wide", "admitted_to_separate_fd_campaign": True, "authorize_residual_blind_export": False}]
    )
    assert yes["decision"] == DECISION_OPEN_FD
    assert yes["fd_identifiability_executed"] is False
    with pytest.raises(ResidualBlindStageError):
        refuse_fd_identifiability_this_stage()
    with pytest.raises(SealedTestAccessError):
        refuse_sealed_source("mc24_100116_00030_00039", ["mc24_100116_00030_00039"])
    with pytest.raises(FilenameGuessError):
        refuse_filename_guess("5mrad")
    refused = refuse_forbidden_operations()
    assert refused["refused_fd_identifiability_this_stage"] is True
    assert refused["refused_sealed_test"] is True
    assert refused["refused_filename_guess"] is True
    with pytest.raises(ResidualBlindStageError):
        decide_next_stage([], fd_executed=True)


def test_build_all_reports_with_injected_summaries_does_not_svd():
    config = _config()
    canonical_events = [_event([0, 1, 2, 3], 0.002, 0.0, event_id=index) for index in range(5)]
    summaries = {
        "canonical": [_source_row("c0", canonical_events, config)],
        "mc24_100012_100gev_gaussian_theta": [
            _source_row("g0", canonical_events, config)
        ],
        "mc24_100116_117_2d_fluka_nonsealed": [],
        "mc24_100049_050_calonu_5mrad": [],
        "mc24_100120_muon_floor": [],
        "mc24_100123_124_year_labeled_muon": [],
        "mc24_100130_kshort_end_fasernu": [],
        "mc24_fluka_210010_full_shower": [],
        "real_2024_cosmic_beam_mode": [],
        "real_2023_cosmic_like": [],
        "real_backward_alps_ift_collision": [],
        "real_testbeam_2021": [],
        "real_2024_r0022_wide_angle_selected_routes": [],
    }
    reports = build_all_reports(
        config, source_summaries=summaries, probe_unmeasured_xaod=False
    )
    assert reports["next_stage_decision"]["fd_identifiability_executed"] is False
    assert reports["next_stage_decision"]["svd_or_rank_computed"] is False
    assert reports["next_stage_decision"]["geometry_write_allowed"] is False
    assert "identifiable_rank" not in reports["admission"]
    assert_no_alignment_payload(reports["next_stage_decision"])
    assert reports["canonical_coverage"]["pooled"]["n_events"] == 5
    r0022 = next(
        row
        for row in reports["admission"]["admissions"]
        if row["id"] == "real_2024_r0022_wide_angle_selected_routes"
    )
    assert r0022["admitted_to_separate_fd_campaign"] is False
