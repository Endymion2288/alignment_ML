from __future__ import annotations

from pathlib import Path

from alignment.real_data_occupancy_preflight import (
    contiguous_windows,
    load_window_rule,
    select_window_for_segment,
    window_minima,
)
from alignment.real_data_residual_dq_monitoring import (
    DECISION_MONITORING_ONLY,
    STATUS_DETECTOR_CONDITION,
    STATUS_INSUFFICIENT,
    STATUS_NOMINAL,
    assemble_monitored_run,
    campaign_allows_cdx_mode,
    campaign_allows_geometry_write,
    campaign_allows_station_calibration_mode,
    classify_monitoring_campaign,
    summarize_monitoring_run,
)
from alignment.real_data_residual_dq_monitoring_expansion import (
    EXPANSION_RUNS,
    ROLE_MONITORING,
    assert_reference_not_reestimated,
    detect_adjacent_alignment_drift,
    first_wave_segments,
    interpret_expansion,
    load_expansion_config,
    load_frozen_calibration_reference,
    occupancy_time_key,
    order_blocks_by_real_time,
    plan_full_remaining_monitoring,
)
from alignment.real_data_residual_dq_monitoring import (
    build_calibration_reference,
    load_monitoring_config,
)


def _parent_config():
    return load_monitoring_config(
        Path("configs/operating_protocol_v1_real_data_residual_dq_monitoring_v1.yaml")
    )


def _route(run: int, event_id: int, stations: list[int], utility: float = 0.1) -> dict:
    return {
        "run_id": run,
        "event_id": event_id,
        "endpoint_stations": stations,
        "is_complete_four_station_route": len(stations) == 4,
        "utility": utility,
    }


def _edge(
    run: int,
    event_id: int,
    *,
    residual_y: float,
    residual_ty: float,
    residual_x: float = 0.0,
    residual_tx: float = 0.0,
    score: float = 0.8,
    target_station_id: int = 1,
) -> dict:
    return {
        "run_id": run,
        "event_id": event_id,
        "residual_y_mm": residual_y,
        "residual_ty": residual_ty,
        "residual_x_mm": residual_x,
        "residual_tx": residual_tx,
        "score": score,
        "target_station_id": target_station_id,
    }


def _reference_rows() -> dict[int, list[dict]]:
    rows = {}
    for run in (14973, 14974):
        rows[run] = [
            _edge(run, event, residual_y=-4.0 + 0.1 * (event % 5), residual_ty=-0.002 + 0.0002 * (event % 5))
            for event in range(40)
        ]
    return rows

# tests/ is not a package; keep a local occupancy table helper.
def _table(*, n=250, populated_at=0, n_tracklets=4):
    import numpy as np

    clusters = np.zeros(n, dtype=np.int32)
    tracklets = np.zeros(n, dtype=np.int32)
    four = np.zeros(n, dtype=np.int32)
    station_c = np.zeros((n, 4), dtype=np.int32)
    station_t = np.zeros((n, 4), dtype=np.int32)
    start = populated_at
    stop = populated_at + 100
    clusters[start:stop] = 20
    tracklets[start:stop] = n_tracklets
    four[start:stop] = 1
    station_c[start:stop] = 5
    station_t[start:stop] = 1
    return {
        "n_sct_clusters": clusters,
        "n_refit_tracklets": tracklets,
        "four_station_refit_multiplicity": four,
        "n_clusters_station": station_c,
        "n_refit_station": station_t,
        "run": np.full(n, 14973, dtype=np.int32),
        "lumi_block": np.ones(n, dtype=np.uint32),
        "stable_beams": np.ones(n, dtype=np.int32),
    }


def _config():
    return load_expansion_config(
        Path("configs/operating_protocol_v1_real_data_residual_dq_monitoring_expansion_v1.yaml")
    )


def test_expansion_keeps_frozen_decision_and_alarms():
    config = _config()
    parent = _parent_config()
    assert config["parent_decision"] == DECISION_MONITORING_ONLY
    assert config["geometry_write_allowed"] is False
    assert config["station_calibration_mode_available"] is False
    assert config["cdx_mode_allowed"] is False
    assert config["do_not_reestimate_reference_scale"] is True
    assert config["do_not_repick_window_from_residual"] is True
    assert config["min_selected_routes_for_alignment_dq"] == parent["min_selected_routes_for_alignment_dq"]
    assert config["robust_z_detector_condition"] == parent["robust_z_detector_condition"]
    assert config["robust_z_drift"] == parent["robust_z_drift"]
    assert list(config["expansion_runs"]) == list(EXPANSION_RUNS)
    assert campaign_allows_cdx_mode(DECISION_MONITORING_ONLY) is False
    assert campaign_allows_geometry_write(DECISION_MONITORING_ONLY) is False
    assert campaign_allows_station_calibration_mode(DECISION_MONITORING_ONLY) is False


def test_frozen_parent_reference_is_not_reestimated():
    loaded = load_frozen_calibration_reference(
        Path("outputs/operating_protocol_v1_real_data_residual_dq_monitoring_v1/run_level_dq_report.json")
    )
    assert loaded["runs"] == [14973, 14974]
    assert loaded["reestimated"] is False
    assert loaded["used_for_geometry"] is False
    assert_reference_not_reestimated(loaded, loaded)
    mutated = {
        "channels": {
            name: {**channel, "robust_scale": 0.5 * float(channel["robust_scale"])}
            for name, channel in loaded["channels"].items()
        }
    }
    try:
        assert_reference_not_reestimated(mutated, loaded)
    except ValueError as exc:
        assert "re-estimation" in str(exc)
    else:
        raise AssertionError("expected reference re-estimation rejection")


def test_expansion_occupancy_uses_frozen_first_window_and_monitoring_role():
    rule = load_window_rule(Path("configs/operating_protocol_v1_real_data_occupancy_window_rule.yaml"))
    table = _table(n=300, populated_at=0, n_tracklets=4)
    table["n_refit_tracklets"][200:300] = 40
    windows = contiguous_windows(table, nevents=100, skip_start=0, skip_stride=100)
    chosen = select_window_for_segment(
        run=14971,
        segment="00000",
        windows=windows,
        rule=rule,
        role=ROLE_MONITORING,
    )
    assert chosen["window_accepted"] is True
    assert chosen["skip_events"] == 0
    assert chosen["role"] == ROLE_MONITORING
    assert window_minima(rule)["total_tracklets"] == 32
    richer = [window for window in windows if window["skip_events"] == 200][0]
    assert richer["total_tracklets"] > chosen["occupancy_summary"]["total_tracklets"]
    assert first_wave_segments(_config()) == [f"{index:05d}" for index in range(8)]


def test_full_remaining_segment_is_used_after_occupancy_start():
    planned = plan_full_remaining_monitoring(
        run=14971,
        segment="00004",
        skip_events=12000,
        n_segment_events=80000,
    )
    assert planned["nevents"] == 68000
    assert planned["is_full_segment"] is True
    assert planned["needs_athena"] is True
    assert planned["fd_probes_generated"] is False
    assert "n68000" in planned["source_id"]


def test_real_time_order_uses_fill_not_run_number():
    later_run_earlier_fill = {"run": 15007, "lhc_fill": 9500, "skip_events": 0}
    earlier_run_later_fill = {"run": 14971, "lhc_fill": 9600, "skip_events": 0}
    ordered = order_blocks_by_real_time([earlier_run_later_fill, later_run_earlier_fill])
    assert [block["run"] for block in ordered] == [15007, 14971]
    assert occupancy_time_key(lhc_fill=9500, run=15007, skip_events=0) < occupancy_time_key(
        lhc_fill=9600, run=14971, skip_events=0
    )
    from alignment.real_data_residual_dq_monitoring_expansion import time_stability_in_real_time

    series = time_stability_in_real_time(
        [
            {
                "run": 14971,
                "lhc_fill": 9600,
                "skip_events": 0,
                "status": "nominal_monitoring",
                "isolation_residual_observables": {"dy": {"median": 1.0}, "rx": {"median": 0.0}},
            },
            {
                "run": 15007,
                "lhc_fill": 9500,
                "skip_events": 0,
                "status": "nominal_monitoring",
                "isolation_residual_observables": {"dy": {"median": 0.0}, "rx": {"median": 0.0}},
            },
        ]
    )
    assert [row["run"] for row in series["series"]] == [15007, 14971]
    assert series["order"] == "lhc_fill_then_run_then_skip_events"


def test_adjacent_sufficient_drift_is_candidate_only():
    config = _parent_config()
    reference = build_calibration_reference(_reference_rows(), reference_runs=(14973, 14974))
    blocks = []
    for run, residual_y in ((14980, 20.0), (14981, 21.0)):
        routes = [_route(run, event, [0, 1, 2], 0.1) for event in range(15)]
        edges = [_edge(run, event, residual_y=residual_y, residual_ty=-0.002) for event in range(15)]
        summary = summarize_monitoring_run(
            run=run,
            role="monitoring",
            source_id=f"exp_{run}",
            n_events=10000,
            n_tracklets=8000,
            n_all_pairs_candidates=200,
            routes=routes,
            field_edges=edges,
        )
        summary["lhc_fill"] = 9600
        summary["skip_events"] = 0
        blocks.append(assemble_monitored_run(summary, reference, config))
    assert all(block["status"] == STATUS_DETECTOR_CONDITION for block in blocks)
    drift = detect_adjacent_alignment_drift(blocks, robust_z_drift=3.0, min_adjacent_runs=2)
    assert drift["alignment_drift_candidate"] is True
    assert "dy" in drift["channels"]
    assert drift["cannot_convert_to_geometry_correction"] is True
    assert drift["follow_up_if_true"] == "independent_survey_or_external_alignment_not_self_nulling"
    reading = interpret_expansion(blocks, drift)
    assert reading["reading"] == "independent_survey_or_external_alignment_follow_up"
    assert reading["reopen_self_nulling_calibration"] is False
    decision = classify_monitoring_campaign(blocks, drift)
    assert decision["decision"] == DECISION_MONITORING_ONLY
    assert decision["geometry_write_allowed"] is False
    assert decision["station_calibration_mode_available"] is False
    assert decision["cdx_mode_allowed"] is False


def test_insufficient_and_nominal_majority_keep_monitoring_backbone():
    config = _parent_config()
    reference = build_calibration_reference(_reference_rows(), reference_runs=(14973, 14974))
    routes = [_route(14985, event, [0, 1], 0.1) for event in range(20)]
    edges = [_edge(14985, event, residual_y=-4.0, residual_ty=-0.002) for event in range(20)]
    nominal = assemble_monitored_run(
        summarize_monitoring_run(
            run=14985,
            role="monitoring",
            source_id="ok",
            n_events=10000,
            n_tracklets=8000,
            n_all_pairs_candidates=200,
            routes=routes,
            field_edges=edges,
        ),
        reference,
        config,
    )
    low = assemble_monitored_run(
        summarize_monitoring_run(
            run=14989,
            role="monitoring",
            source_id="low",
            n_events=1000,
            n_tracklets=800,
            n_all_pairs_candidates=100,
            routes=[_route(14989, 1, [0, 1], 0.05)],
            field_edges=[_edge(14989, 1, residual_y=30.0, residual_ty=0.2)],
        ),
        reference,
        config,
    )
    assert nominal["status"] == STATUS_NOMINAL
    assert low["status"] == STATUS_INSUFFICIENT
    drift = detect_adjacent_alignment_drift([nominal, low], robust_z_drift=3.0, min_adjacent_runs=2)
    assert drift["alignment_drift_candidate"] is False
    reading = interpret_expansion([nominal, low], drift)
    assert reading["reading"] == "official_geometry_and_frozen_v2_can_carry_long_term_dq_monitoring"
