from __future__ import annotations

from pathlib import Path

from alignment.calibration_modes import OPERATING_BAND_UM
from alignment.five_dof_sampling import SURVEY_PARAMETER
from alignment.real_data_residual_dq_monitoring import (
    DECISION_MONITORING_ONLY,
    STATUS_ASSOCIATION_DEGRADATION,
    STATUS_DETECTOR_CONDITION,
    STATUS_INSUFFICIENT,
    STATUS_NOMINAL,
    assemble_monitored_run,
    build_calibration_reference,
    campaign_allows_cdx_mode,
    campaign_allows_geometry_write,
    campaign_allows_station_calibration_mode,
    classify_monitoring_campaign,
    detect_alignment_drift_candidates,
    load_monitoring_config,
    summarize_monitoring_run,
)


def _config():
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


def test_config_freezes_write_modes_and_forbids_payload():
    config = _config()
    assert config["geometry_write_allowed"] is False
    assert config["station_calibration_mode_available"] is False
    assert config["cdx_mode_allowed"] is False
    assert config["do_not_generate_fd_probes"] is True
    assert config["do_not_run_newton"] is True
    assert config["do_not_write_payload"] is True
    assert config["cannot_convert_drift_to_geometry"] is True
    assert config["dz_track_driven_observable"] is False
    assert config["rz_direct_track_residual"] is False
    assert config["track_driven_dz"] is False
    assert campaign_allows_cdx_mode(DECISION_MONITORING_ONLY) is False
    assert campaign_allows_geometry_write(DECISION_MONITORING_ONLY) is False
    assert campaign_allows_station_calibration_mode(DECISION_MONITORING_ONLY) is False
    assert list(config["calibration_reference_runs"]) == [14973, 14974]


def test_low_statistics_run_is_insufficient_not_alignment():
    config = _config()
    reference = build_calibration_reference(_reference_rows(), reference_runs=(14973, 14974))
    summary = summarize_monitoring_run(
        run=14977,
        role="monitoring",
        source_id="data24_r14977_lowstat",
        n_events=1687,
        n_tracklets=1163,
        n_all_pairs_candidates=1329,
        routes=[
            _route(14977, 1, [0, 1], 0.05),
            _route(14977, 2, [1, 2], 0.04),
        ],
        field_edges=[
            _edge(14977, 1, residual_y=20.0, residual_ty=0.05),
            _edge(14977, 2, residual_y=-20.0, residual_ty=-0.05),
        ],
    )
    block = assemble_monitored_run(summary, reference, config)
    assert block["selected_routes"] == 2
    assert block["status"] == STATUS_INSUFFICIENT
    assert block["not_an_alignment_anomaly"] is True
    assert block["geometry_write_allowed"] is False
    assert block["dz_track_driven_observable"] is False
    assert block["rz_direct_track_residual"] is False
    assert SURVEY_PARAMETER == "ift_dz_mm"


def test_empty_selected_with_nonempty_candidates_is_association_degradation():
    config = _config()
    reference = build_calibration_reference(_reference_rows(), reference_runs=(14973, 14974))
    summary = summarize_monitoring_run(
        run=14980,
        role="monitoring",
        source_id="empty_selected",
        n_events=10000,
        n_tracklets=8000,
        n_all_pairs_candidates=200,
        routes=[],
        field_edges=[],
    )
    block = assemble_monitored_run(summary, reference, config)
    assert block["status"] == STATUS_ASSOCIATION_DEGRADATION
    assert block["not_an_alignment_anomaly"] is False


def test_event_concentration_is_association_degradation():
    config = _config()
    reference = build_calibration_reference(_reference_rows(), reference_runs=(14973, 14974))
    routes = [_route(14975, 7, [0, 1, 2], 0.1) for _ in range(12)]
    edges = [_edge(14975, 7, residual_y=-4.0, residual_ty=-0.002) for _ in range(12)]
    summary = summarize_monitoring_run(
        run=14975,
        role="monitoring",
        source_id="concentrated",
        n_events=10000,
        n_tracklets=8000,
        n_all_pairs_candidates=200,
        routes=routes,
        field_edges=edges,
    )
    block = assemble_monitored_run(summary, reference, config)
    assert block["event_concentration"]["max_event_share_of_selected_routes"] == 1.0
    assert block["status"] == STATUS_ASSOCIATION_DEGRADATION


def test_isolation_shift_is_detector_condition_not_a_correction():
    config = _config()
    reference = build_calibration_reference(_reference_rows(), reference_runs=(14973, 14974))
    routes = [_route(14975, event, [0, 1, 2], 0.1) for event in range(20)]
    edges = [_edge(14975, event, residual_y=20.0, residual_ty=-0.002) for event in range(20)]
    summary = summarize_monitoring_run(
        run=14975,
        role="monitoring",
        source_id="shifted_dy",
        n_events=10000,
        n_tracklets=8000,
        n_all_pairs_candidates=200,
        routes=routes,
        field_edges=edges,
    )
    block = assemble_monitored_run(summary, reference, config)
    assert block["status"] == STATUS_DETECTOR_CONDITION
    assert block["not_a_geometry_correction"] is True
    assert block["cross_level_sensitive_residual_observables"]["dx"]["cross_level_sensitive"] is True
    assert block["cross_level_sensitive_residual_observables"]["ry"]["cross_level_sensitive"] is True
    assert block["isolation_residual_observables"]["dy"]["cross_level_sensitive"] is False
    assert block["isolation_residual_observables"]["dy"]["not_a_geometry_correction"] is True


def test_nominal_compatible_run_stays_nominal():
    config = _config()
    reference = build_calibration_reference(_reference_rows(), reference_runs=(14973, 14974))
    routes = [_route(14975, event, [0, 1] if event % 2 else [0, 1, 2], 0.1) for event in range(20)]
    edges = [
        _edge(14975, event, residual_y=-4.0 + 0.05 * (event % 3), residual_ty=-0.002)
        for event in range(20)
    ]
    summary = summarize_monitoring_run(
        run=14975,
        role="monitoring",
        source_id="compatible",
        n_events=10000,
        n_tracklets=8000,
        n_all_pairs_candidates=200,
        routes=routes,
        field_edges=edges,
    )
    block = assemble_monitored_run(summary, reference, config)
    assert block["status"] == STATUS_NOMINAL
    assert block["route_composition"]["2_station"] == 10
    assert block["route_composition"]["3_station"] == 10
    assert "1" in block["station_residual_observables"]


def test_repeatable_isolation_drift_is_candidate_only():
    config = _config()
    reference = build_calibration_reference(_reference_rows(), reference_runs=(14973, 14974))
    blocks = []
    for run in (14975, 14976):
        routes = [_route(run, event, [0, 1, 2], 0.1) for event in range(15)]
        edges = [_edge(run, event, residual_y=12.0, residual_ty=-0.002) for event in range(15)]
        summary = summarize_monitoring_run(
            run=run,
            role="monitoring",
            source_id=f"drift_{run}",
            n_events=10000,
            n_tracklets=8000,
            n_all_pairs_candidates=200,
            routes=routes,
            field_edges=edges,
        )
        blocks.append(assemble_monitored_run(summary, reference, config))
    drift = detect_alignment_drift_candidates(
        blocks,
        reference_runs=(14973, 14974),
        robust_z_drift=float(config["robust_z_drift"]),
        min_nonreference_runs=int(config["min_nonreference_runs_for_repeatable_drift"]),
    )
    assert drift["alignment_drift_candidate"] is True
    assert "dy" in drift["channels"]
    assert drift["cannot_convert_to_geometry_correction"] is True
    assert drift["geometry_write_allowed"] is False
    assert drift["station_calibration_mode_available"] is False
    assert drift["cdx_mode_allowed"] is False
    decision = classify_monitoring_campaign(blocks, drift)
    assert decision["decision"] == DECISION_MONITORING_ONLY
    assert decision["geometry_write_allowed"] is False
    assert decision["station_calibration_mode_available"] is False
    assert decision["cdx_mode_allowed"] is False
    assert decision["cannot_convert_drift_to_geometry"] is True
    assert decision["do_not_extract_alignment_payload_from_self_nulling"] is True
    assert decision["geometry_reopen_prerequisites"]["required_any_of"][0][
        "independent_C_dx_within_operating_band_um"
    ] == list(OPERATING_BAND_UM)


def test_insufficient_run_cannot_create_drift_candidate():
    config = _config()
    reference = build_calibration_reference(_reference_rows(), reference_runs=(14973, 14974))
    blocks = []
    for run in (14975, 14977):
        routes = [_route(run, event, [0, 1], 0.1) for event in range(2)]
        edges = [_edge(run, event, residual_y=30.0, residual_ty=0.2) for event in range(2)]
        summary = summarize_monitoring_run(
            run=run,
            role="monitoring",
            source_id=f"low_{run}",
            n_events=1000,
            n_tracklets=800,
            n_all_pairs_candidates=100,
            routes=routes,
            field_edges=edges,
        )
        blocks.append(assemble_monitored_run(summary, reference, config))
    drift = detect_alignment_drift_candidates(
        blocks,
        reference_runs=(14973, 14974),
        robust_z_drift=float(config["robust_z_drift"]),
        min_nonreference_runs=int(config["min_nonreference_runs_for_repeatable_drift"]),
    )
    assert all(block["status"] == STATUS_INSUFFICIENT for block in blocks)
    assert drift["alignment_drift_candidate"] is False


def test_classify_rejects_residual_success_and_payload_extraction():
    try:
        classify_monitoring_campaign([], {}, residual_used_as_success=True)
    except ValueError as exc:
        assert "residual" in str(exc)
    else:
        raise AssertionError("expected residual-success rejection")
    try:
        classify_monitoring_campaign([], {}, payload_extracted=True)
    except ValueError as exc:
        assert "payload" in str(exc)
    else:
        raise AssertionError("expected payload-extraction rejection")
