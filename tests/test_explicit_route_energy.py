"""Workbook-79 Arm B contract: raw energy, truth-free 2/3/4 routes, no clip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from baselines.field_chi2_matching import FieldCandidate
from datasets.physical_curriculum import CurriculumSample
from datasets.root_loader import EventTracklets
from models.explicit_route_energy import (
    FEATURE_DIM,
    HIDDEN_WIDTH,
    ExplicitRouteEnergyScorer,
    raw_route_utilities,
    route_feature_names,
)
from evaluation.route_accounting import ACCOUNTING_VERSION
from models.route_energy import assign_from_energy_table, canonical_route_energy
from training.curriculum_mlp import CandidateSet
from training.explicit_route_energy import (
    account_event,
    energy_table_from_utilities,
    enumerate_contiguous_physical_routes,
    refuse_forbidden_experiment_path,
    route_feature_matrix,
    target_mask,
    w64_route_utilities,
    _adjacent_pair_ids,
)
from training.geometry_aware_transformer import ALL_STATION_PAIRS, build_transformer_graph_bundle
from training.route_aware_transformer import _adjacent_pair_ids as w64_adjacent_pair_ids


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


def test_feature_contract_is_54_raw_physical_channels():
    names = route_feature_names()
    assert len(names) == FEATURE_DIM == 54
    assert names[0] == "e01_residual_x_mm"
    assert names[-1] == "missing_station_3"
    assert "route_length" in names
    assert HIDDEN_WIDTH <= 128


def test_scorer_emits_raw_energy_without_a_sigmoid():
    torch.manual_seed(7)
    model = ExplicitRouteEnergyScorer()
    features = torch.zeros((3, FEATURE_DIM), dtype=torch.float32)
    cores = model(features)
    assert cores.shape == (3,)
    assert torch.isfinite(cores).all()
    utilities = raw_route_utilities(cores, [4, 3, 2], unmatched_penalty=-1.0)
    assert float((utilities[0] - cores[0]).detach()) == pytest.approx(-4.0)


def test_adjacent_score_pair_ids_match_the_w64_graph_contract():
    assert _adjacent_pair_ids() == w64_adjacent_pair_ids()
    assert _adjacent_pair_ids() == (
        ALL_STATION_PAIRS.index((0, 1)),
        ALL_STATION_PAIRS.index((1, 2)),
        ALL_STATION_PAIRS.index((2, 3)),
    )


def test_contiguous_enumeration_is_truth_free_and_includes_fragments():
    bundle = build_transformer_graph_bundle(_candidate_sets(), context_mode="full_event")
    graph = bundle.graphs[0]
    routes = enumerate_contiguous_physical_routes(graph)
    lengths = {len(route.stations) for route in routes}
    assert lengths == {2, 3, 4}
    assert any(route.stations == (1, 2) for route in routes)
    assert any(route.stations == (2, 3) for route in routes)
    features = route_feature_matrix(graph, routes)
    assert features.shape[1] == FEATURE_DIM
    assert np.isfinite(features).all()
    names = route_feature_names()
    e12 = names.index("e12_residual_x_mm")
    pair_routes = [row for route, row in zip(routes, features) if route.stations == (1, 2)]
    assert pair_routes
    assert pair_routes[0][e12] != 0.0
    object.__setattr__(graph.event, "truth_particle_id", np.asarray([99, 98, 99, 98, 99, 98, 99, 98]))
    rerun = enumerate_contiguous_physical_routes(graph)
    assert [route.node_indices for route in routes] == [route.node_indices for route in rerun]


def test_arm_a_utility_is_raw_logit_sum_plus_penalty():
    bundle = build_transformer_graph_bundle(_candidate_sets(), context_mode="full_event")
    graph = bundle.graphs[0]
    routes = enumerate_contiguous_physical_routes(graph)
    logits = {}
    for edge_index in range(graph.score_edge_source.size):
        logits[(int(graph.score_owner[edge_index]), int(graph.score_row[edge_index]))] = 8.0
    utilities = w64_route_utilities(graph, routes, logits, unmatched_penalty=-1.0)
    completes = [utility for route, utility in zip(routes, utilities) if len(route.stations) == 4]
    prefixes = [utility for route, utility in zip(routes, utilities) if route.stations == (0, 1, 2)]
    assert completes
    assert prefixes
    assert completes[0] == pytest.approx(canonical_route_energy((8.0, 8.0, 8.0), unmatched_penalty=-1.0))
    assert completes[0] == pytest.approx(20.0)
    assert prefixes[0] == pytest.approx(13.0)
    table = energy_table_from_utilities(routes, utilities, unmatched_penalty=-1.0)
    selected = assign_from_energy_table(table).selected
    assert int(np.count_nonzero(selected)) >= 1


def test_target_mask_keeps_only_complete_truth():
    bundle = build_transformer_graph_bundle(_candidate_sets(), context_mode="full_event")
    routes = enumerate_contiguous_physical_routes(bundle.graphs[0])
    mask = target_mask(routes)
    assert int(np.count_nonzero(mask)) == 2
    assert all(len(route.stations) == 4 for route, flag in zip(routes, mask) if flag)


def test_account_event_uses_route_accounting_v2():
    bundle = build_transformer_graph_bundle(_candidate_sets(), context_mode="full_event")
    graph = bundle.graphs[0]
    routes = enumerate_contiguous_physical_routes(graph)
    utilities = np.where(target_mask(routes), 20.0, -10.0)
    table = energy_table_from_utilities(routes, utilities, unmatched_penalty=-1.0)
    accounting = account_event(graph, routes, table, bundle.adjacent_sets)
    assert accounting.metric_version == ACCOUNTING_VERSION
    assert accounting.correct_complete_routes == 2
    assert accounting.fake_complete_routes == 0


def test_refuses_development_and_blind_paths():
    with pytest.raises(ValueError, match="refuses"):
        refuse_forbidden_experiment_path("outputs/mc24_100047_00350_00399/x.root")
    with pytest.raises(ValueError, match="refuses"):
        refuse_forbidden_experiment_path("/eos/data/mc24_100047_00800_00849/x.root")
    with pytest.raises(ValueError, match="refuses"):
        refuse_forbidden_experiment_path("mc24_100116_00000_00049")
