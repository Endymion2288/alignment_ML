"""Workbook-81a: identity, bound, and path guards.  No training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from baselines.field_chi2_matching import FieldCandidate
from datasets.physical_curriculum import CurriculumSample
from datasets.root_loader import EventTracklets
from models.route_energy import canonical_route_energy
from training.curriculum_mlp import CandidateSet
from training.explicit_route_energy import (
    account_event,
    energy_table_from_utilities,
    enumerate_contiguous_physical_routes,
    w64_route_utilities,
)
from training.geometry_aware_transformer import ALL_STATION_PAIRS, build_transformer_graph_bundle
from training.wb81_calibration_contract import (
    DELTA_MAX,
    apply_calibrated_utilities,
    bounded_physics_delta,
    calibrated_energy_table,
    checksum_holdout,
    compare_identity,
    contract_source_guards,
    margin_bin,
    refuse_wb81a_path,
    selected_mask,
    zero_theta_delta,
)


def _sample() -> CurriculumSample:
    return CurriculumSample(
        source_id="route_train",
        source_ids=("route_train",),
        split="train",
        payload_id="route_nominal",
        magnitude_mm=0.0,
        direction_trial="unit",
        injected_offsets_xy_mm={},
        source_event_uids=("route_train:1",),
        physical_event_uids=("route_train:1",),
        physical_tracklets=Path("/tmp/route_tracklets.root"),
        physical_propagations=Path("/tmp/route_prop.root"),
        physical_payload_manifest=Path("/tmp/route_payload.json"),
        synthetic_tracklets=Path("/tmp/route_synthetic.root"),
        field_candidates=Path("/tmp/route_candidates.root"),
    )


def _event() -> EventTracklets:
    return EventTracklets(
        run_id=1,
        event_id=2,
        station_id=np.repeat(np.arange(4, dtype=np.int16), 2),
        tracklet_id=np.arange(8, dtype=np.int32),
        z_mm=np.repeat(np.arange(4, dtype=np.float64), 2),
        state=np.arange(32, dtype=np.float64).reshape(8, 4),
        covariance=np.tile(np.eye(4, dtype=np.float64), (8, 1, 1)),
        chi2=np.ones(8, dtype=np.float64),
        ndof=np.ones(8, dtype=np.float64),
        n_hit=np.full(8, 3, dtype=np.int16),
        hit_pattern=np.full(8, 0b111111, dtype=np.uint64),
        truth_particle_id=np.tile(np.asarray([10, 20], dtype=np.int64), 4),
        truth_pdg=np.full(8, 13, dtype=np.int32),
        truth_match_fraction=np.ones(8, dtype=np.float64),
        synthetic_role=np.zeros(8, dtype=np.int8),
    )


def _candidate_sets() -> list[CandidateSet]:
    event = _event()
    sample = _sample()
    result: list[CandidateSet] = []
    for source_station, target_station in ALL_STATION_PAIRS:
        candidates = []
        labels = []
        for source in event.indices_for_station(source_station):
            for target in event.indices_for_station(target_station):
                truth = bool(event.truth_particle_id[source] == event.truth_particle_id[target])
                candidates.append(
                    FieldCandidate(
                        source_index=int(source),
                        target_index=int(target),
                        source_station=source_station,
                        target_station=target_station,
                        chi2=1.0 if truth else 10.0,
                        residual=np.asarray([1.0, -2.0, 0.1, -0.2]),
                        pull=np.asarray([1.0, -2.0, 0.1, -0.2]),
                        combined_covariance=np.eye(4, dtype=np.float64),
                    )
                )
                labels.append(truth)
        result.append(
            CandidateSet(
                sample=sample,
                event=event,
                station_pair=(source_station, target_station),
                candidates=tuple(candidates),
                features=np.zeros((len(candidates), 1), dtype=np.float64),
                labels=np.asarray(labels, dtype=bool),
            )
        )
    return result


def test_delta_max_is_frozen_and_bound_is_length_agnostic():
    assert DELTA_MAX == 0.25
    values = bounded_physics_delta([-100.0, 0.0, 100.0])
    assert values.shape == (3,)
    assert float(np.max(np.abs(values))) <= DELTA_MAX + 1.0e-12
    assert values[1] == pytest.approx(0.0)
    mixed = apply_calibrated_utilities([1.0, 2.0, 3.0], bounded_physics_delta([8.0, 8.0, 8.0]))
    assert mixed[0] == pytest.approx(1.0 + DELTA_MAX * np.tanh(8.0))
    with pytest.raises(ValueError, match="exceeds frozen bound"):
        apply_calibrated_utilities([0.0], [0.26])


def test_theta_zero_is_identity_on_selected_and_metrics():
    bundle = build_transformer_graph_bundle(_candidate_sets(), context_mode="full_event")
    graph = bundle.graphs[0]
    routes = enumerate_contiguous_physical_routes(graph)
    logits = {}
    for edge_index in range(graph.score_edge_source.size):
        logits[(int(graph.score_owner[edge_index]), int(graph.score_row[edge_index]))] = 8.0
    u_w64 = w64_route_utilities(graph, routes, logits, unmatched_penalty=-1.0)
    completes = [utility for route, utility in zip(routes, u_w64) if len(route.stations) == 4]
    assert completes[0] == pytest.approx(canonical_route_energy((8.0, 8.0, 8.0), unmatched_penalty=-1.0))
    delta = zero_theta_delta(len(routes))
    assert np.array_equal(u_w64, apply_calibrated_utilities(u_w64, delta))
    table_a = energy_table_from_utilities(routes, u_w64, unmatched_penalty=-1.0)
    table_new = calibrated_energy_table(routes, u_w64, delta, unmatched_penalty=-1.0)
    assert selected_mask(table_a) == selected_mask(table_new)
    assert all(record.correction == 0.0 for record in table_new.records)
    metrics_a = account_event(graph, routes, table_a, bundle.adjacent_sets).as_dict()
    metrics_new = account_event(graph, routes, table_new, bundle.adjacent_sets).as_dict()
    report = compare_identity(selected_mask(table_a), selected_mask(table_new), metrics_a, metrics_new)
    assert report["passed"] is True


def test_identity_compare_treats_missing_rates_as_equal():
    empty = {
        "complete_track_efficiency": None,
        "complete_fake_rate": None,
        "complete_track_purity": None,
        "all_route_purity": 1.0,
    }
    report = compare_identity((0,), (0,), empty, empty)
    assert report["passed"] is True


def test_identity_compare_fails_when_selection_changes():
    report = compare_identity(
        (0, 1),
        (0,),
        {"complete_track_efficiency": 0.8, "complete_fake_rate": 0.01, "complete_track_purity": 0.99, "all_route_purity": 0.99},
        {"complete_track_efficiency": 0.8, "complete_fake_rate": 0.01, "complete_track_purity": 0.99, "all_route_purity": 0.99},
    )
    assert report["passed"] is False
    assert report["selected_identical"] is False


def test_margin_bins_and_holdout_checksum():
    assert margin_bin(-0.1) == "negative"
    assert margin_bin(0.0) == "near_zero"
    assert margin_bin(0.99) == "near_zero"
    assert margin_bin(1.0) == "comfortable"
    ok = checksum_holdout(
        "family1_ds100043_100044",
        {
            "n_scored_events": 3360,
            "complete_truth_chains": 6789,
            "t_neg": 1071,
            "t_near": 5718,
            "t_comf": 0,
            "f_a": 77,
            "correct_complete_routes": 5718,
            "fake_complete_routes": 77,
            "selected_complete_routes": 5795,
            "selected_fragment_routes": 0,
        },
    )
    assert ok["passed"] is True
    bad = checksum_holdout("family1_ds100043_100044", {"n_scored_events": 1})
    assert bad["passed"] is False


def test_contract_guards_and_forbidden_paths():
    guards = contract_source_guards()
    assert all(guards.values())
    with pytest.raises(ValueError, match="refuses"):
        refuse_wb81a_path("outputs/mc24_100047_00350_00399/x.root")
    with pytest.raises(ValueError, match="refuses"):
        refuse_wb81a_path("/eos/data/mc24_100047_00800_00849/x.root")
    with pytest.raises(ValueError, match="refuses"):
        refuse_wb81a_path("mc24_100116_00000_00049")
    with pytest.raises(ValueError, match="refuses"):
        refuse_wb81a_path("outputs/mc24_four_station_explicit_route_energy_v1/holdout_family1/arm_b_checkpoint.json")
    with pytest.raises(ValueError, match="refuses"):
        refuse_wb81a_path("outputs/mc24_four_station_wb80_failure_mechanism_v1/holdout_family1/arm_c_checkpoint.json")


def test_materialize_script_does_not_load_b_or_c():
    text = Path("scripts/materialize_wb81a_boundary.py").read_text(encoding="utf-8")
    assert "arm_b_checkpoint" not in text
    assert "arm_c_checkpoint" not in text
    assert "HybridRouteEnergy" not in text
    assert "ExplicitRouteEnergyScorer" not in text
    assert "train_arm" not in text
    assert "Adam" not in text
