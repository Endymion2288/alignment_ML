from __future__ import annotations

import numpy as np
import pytest

from datasets.root_loader import EventTracklets
from training.source_diversity_audit import (
    CURRENT_TRAIN_SOURCES,
    FAILURE_PAYLOAD,
    PROPOSED_NEW_TRAIN_SOURCES,
    RESERVED_BLIND_SOURCES,
    attach_route_phase_space,
    assert_sources_allowed,
    charge_from_pdg,
    coverage_report,
    fraction_inside_quantiles,
    quantile,
    recommend_diversity_next,
    select_failure_core,
    summarize_identity_events,
    truth_edge_logit,
)


def _event() -> EventTracklets:
    return EventTracklets(
        run_id=1,
        event_id=2,
        station_id=np.arange(4, dtype=np.int16),
        tracklet_id=np.arange(4, dtype=np.int32),
        z_mm=np.arange(4, dtype=np.float64),
        state=np.asarray(
            [
                [1.0, 2.0, 0.01, 0.02],
                [3.0, 4.0, 0.03, 0.04],
                [5.0, 6.0, 0.05, 0.06],
                [7.0, 8.0, 0.07, 0.08],
            ],
            dtype=np.float64,
        ),
        covariance=np.tile(np.eye(4, dtype=np.float64), (4, 1, 1)),
        chi2=np.ones(4, dtype=np.float64),
        ndof=np.ones(4, dtype=np.float64),
        n_hit=np.full(4, 3, dtype=np.int16),
        hit_pattern=np.full(4, 0b111, dtype=np.uint64),
        truth_particle_id=np.ones(4, dtype=np.int64),
        truth_pdg=np.full(4, 13, dtype=np.int32),
        origin_run_id=np.full(4, 9000000000, dtype=np.int64),
        origin_event_id=np.full(4, 20, dtype=np.int64),
        origin_tracklet_id=np.arange(4, dtype=np.int64),
    )


def _row(**kwargs):
    payload = {
        "payload_id": FAILURE_PAYLOAD,
        "run_id": 1,
        "event_id": 2,
        "truth_id": 1,
        "selected": True,
        "u_truth": 1.2,
        "u_best_solver_fragment": 0.2,
        "production_margin": 1.0,
        "margin_bin": "easy",
        "short_station_bin": "easy",
        "n_complete_truth_in_event": 2,
        "production_fragment_winner": False,
        "origin_signature": (((9000000000, 20, 0), (9000000000, 20, 1), (9000000000, 20, 2), (9000000000, 20, 3)),),
        "endpoints": [
            {"station": 0, "index": 0},
            {"station": 1, "index": 1},
            {"station": 2, "index": 2},
            {"station": 3, "index": 3},
        ],
        "edges": [
            {"calibrated_logit": 1.5},
            {"calibrated_logit": 1.1},
            {"calibrated_logit": 0.4},
        ],
    }
    payload.update(kwargs)
    return payload


def test_charge_and_reserved_source_guard():
    assert charge_from_pdg(13) == "mu_minus"
    assert charge_from_pdg(-13) == "mu_plus"
    assert_sources_allowed(["mc24_100043_00200_00299"])
    with pytest.raises(ValueError, match="sealed"):
        assert_sources_allowed(["mc24_100116_00000_00099"])
    with pytest.raises(ValueError, match="reserved"):
        assert_sources_allowed([RESERVED_BLIND_SOURCES[0]])
    with pytest.raises(ValueError, match="outlier"):
        assert_sources_allowed(["mc24_100047_00150_00199"])


def test_attach_keeps_s3_state_and_2to3_logit():
    attached = attach_route_phase_space(
        _row(),
        _event(),
        source_by_namespaced_run={9000000000: "mc24_100043_00200_00299"},
    )
    assert attached["charge"] == "mu_minus"
    assert attached["origin_source_id"] == "mc24_100043_00200_00299"
    assert attached["logit_2to3"] == pytest.approx(0.4)
    assert attached["s3_state"]["x_mm"] == pytest.approx(7.0)
    assert attached["station_occupancy"]["s3"] == 1
    assert "edges" not in attached
    assert truth_edge_logit(_row(), (2, 3)) == pytest.approx(0.4)


def test_failure_core_keeps_utility_drop_and_near_boundary():
    chart = _row(payload_id="iteration_00_draw_01", u_truth=1.4, production_margin=1.2)
    twin = _row(u_truth=0.8, production_margin=0.4, selected=True)
    easy = _row(truth_id=2, origin_signature=(((1, 2, 3),),), u_truth=1.5, production_margin=1.4)
    core = select_failure_core([chart, twin, easy])
    assert {row["truth_id"] for row in core} == {1}
    assert core[0]["truth_utility_drop"] is True


def test_quantile_coverage_and_identity_summary():
    assert quantile([0.0, 1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.0)
    assert fraction_inside_quantiles([0.1, 0.2], [0.0, 1.0]) == 1.0
    assert fraction_inside_quantiles([5.0], [0.0, 1.0]) == 0.0
    summary = summarize_identity_events([_event()], source_id="mc24_100043_00200_00299")
    assert summary["n_events"] == 1
    assert summary["charge"]["mu_minus"] == 1
    assert summary["kinematics"]["s3_x_mm"]["median"] == pytest.approx(7.0)


def _covered_row(payload, **kwargs):
    row = _row(payload_id=payload, **kwargs)
    row["track_state_by_station"] = {
        "0": {"x_mm": 1.0, "y_mm": 2.0, "tx": 0.01, "ty": 0.02},
        "1": {"x_mm": 3.0, "y_mm": 4.0, "tx": 0.03, "ty": 0.04},
        "2": {"x_mm": 5.0, "y_mm": 6.0, "tx": 0.05, "ty": 0.06},
        "3": {"x_mm": 7.0, "y_mm": 8.0, "tx": 0.07, "ty": 0.08},
    }
    row["station_occupancy"] = {"s0": 3, "s1": 3, "s2": 3, "s3": 3}
    row["charge"] = "mu_minus"
    row["logit_2to3"] = kwargs.get("logit_2to3", 0.8)
    return row


def test_recommend_authorizes_when_transfer_hard_rate_shifts():
    train = [
        _covered_row("iteration_00_draw_01", u_truth=1.5, production_margin=1.4, logit_2to3=1.0),
        _covered_row(FAILURE_PAYLOAD, u_truth=1.4, production_margin=1.3, logit_2to3=0.9),
    ]
    transfer = [
        _covered_row("iteration_00_draw_01", u_truth=1.5, production_margin=1.4, logit_2to3=1.0),
        _covered_row(
            FAILURE_PAYLOAD,
            u_truth=0.2,
            production_margin=-0.1,
            logit_2to3=-0.2,
            production_fragment_winner=True,
            selected=False,
        ),
    ]
    report = coverage_report(train, transfer)
    decision = recommend_diversity_next(report)
    assert decision["authorize_new_training_sources"] is True
    assert decision["coverage_class"] == "source_characteristic_undercoverage"
    assert decision["freeze_workbook62_objective"] is True
    assert decision["workbook56_transfer_is_final_gate"] is False
    assert decision["proposed_new_train_sources"] == list(PROPOSED_NEW_TRAIN_SOURCES)
    assert RESERVED_BLIND_SOURCES[0] in decision["reserved_blind_sources"]


def test_recommend_covers_when_train_matches_failure_region():
    train = [
        _covered_row("iteration_00_draw_01", u_truth=1.2, production_margin=0.4, logit_2to3=0.3),
        _covered_row(
            FAILURE_PAYLOAD,
            u_truth=0.8,
            production_margin=0.3,
            logit_2to3=0.25,
            production_fragment_winner=False,
        ),
    ]
    transfer = [
        _covered_row("iteration_00_draw_01", u_truth=1.2, production_margin=0.4, logit_2to3=0.3),
        _covered_row(FAILURE_PAYLOAD, u_truth=0.75, production_margin=0.28, logit_2to3=0.24),
    ]
    decision = recommend_diversity_next(coverage_report(train, transfer))
    assert decision["authorize_new_training_sources"] is False
    assert decision["coverage_class"] == "current_train_covers_failure_region"
    assert set(decision["proposed_new_train_sources"]) == set(CURRENT_TRAIN_SOURCES)
    assert decision["next_step"] == "discuss_architecture_level_relative_gauge_equivariant_representation"


def test_phase_space_undercoverage_when_s3_is_outside_train():
    train = [_covered_row(FAILURE_PAYLOAD, production_margin=1.2, logit_2to3=0.8)]
    far = _covered_row(
        FAILURE_PAYLOAD,
        production_margin=1.1,
        logit_2to3=0.7,
        selected=False,
    )
    far["track_state_by_station"]["3"] = {"x_mm": 80.0, "y_mm": 90.0, "tx": 0.4, "ty": 0.5}
    decision = recommend_diversity_next(coverage_report(train, [far]))
    assert decision["coverage_class"] == "source_phase_space_undercoverage"
    assert decision["authorize_new_training_sources"] is True
