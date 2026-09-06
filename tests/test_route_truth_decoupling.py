"""Truth IDs must not change physical candidates, features, or route chains."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from baselines.field_chi2_matching import FieldCandidate
from datasets.physical_curriculum import CurriculumSample
from datasets.root_loader import EventTracklets
from training.curriculum_mlp import CandidateSet
from training.geometry_aware_transformer import (
    ALL_STATION_PAIRS,
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    build_transformer_graph_bundle,
)
from training.route_aware_transformer import (
    attach_route_candidate_labels,
    enumerate_complete_route_candidates,
    enumerate_physical_complete_route_chains,
)


def _sample() -> CurriculumSample:
    return CurriculumSample(
        source_id="truth_free",
        source_ids=("truth_free",),
        split="train",
        payload_id="truth_free_nominal",
        magnitude_mm=0.0,
        direction_trial="unit",
        injected_offsets_xy_mm={},
        source_event_uids=("truth_free:1",),
        physical_event_uids=("truth_free:1",),
        physical_tracklets=Path("/tmp/truth_free_tracklets.root"),
        physical_propagations=Path("/tmp/truth_free_prop.root"),
        physical_payload_manifest=Path("/tmp/truth_free_payload.json"),
        synthetic_tracklets=Path("/tmp/truth_free_synthetic.root"),
        field_candidates=Path("/tmp/truth_free_candidates.root"),
    )


def _event(truth: np.ndarray | None) -> EventTracklets:
    kwargs = dict(
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
        synthetic_role=np.zeros(8, dtype=np.int8),
    )
    if truth is None:
        return EventTracklets(**kwargs)
    return EventTracklets(
        **kwargs,
        truth_particle_id=np.asarray(truth, dtype=np.int64),
        truth_pdg=np.full(8, 13, dtype=np.int32),
        truth_match_fraction=np.ones(8, dtype=np.float64),
    )


def _candidate_sets(event: EventTracklets) -> list[CandidateSet]:
    sample = _sample()
    result: list[CandidateSet] = []
    for source_station, target_station in ALL_STATION_PAIRS:
        candidates = []
        labels = []
        for source in event.indices_for_station(source_station):
            for target in event.indices_for_station(target_station):
                candidates.append(
                    FieldCandidate(
                        source_index=int(source),
                        target_index=int(target),
                        source_station=source_station,
                        target_station=target_station,
                        chi2=1.0,
                        residual=np.asarray([1.0, -2.0, 0.1, -0.2]),
                        pull=np.asarray([1.0, -2.0, 0.1, -0.2]),
                        combined_covariance=np.eye(4, dtype=np.float64),
                    )
                )
                if event.truth_particle_id is None:
                    labels.append(False)
                else:
                    labels.append(bool(event.truth_particle_id[source] == event.truth_particle_id[target]))
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


def _graph(truth: np.ndarray | None):
    bundle = build_transformer_graph_bundle(_candidate_sets(_event(truth)), context_mode="full_event")
    return bundle.graphs[0]


def test_node_and_edge_feature_names_do_not_include_truth():
    forbidden = ("truth", "particle", "pdg", "label", "fake")
    for name in (*NODE_FEATURE_NAMES, *EDGE_FEATURE_NAMES):
        lowered = name.lower()
        assert all(token not in lowered for token in forbidden), name


def test_missing_truth_does_not_change_physical_route_chains():
    labeled = _graph(np.tile(np.asarray([10, 20], dtype=np.int64), 4))
    unlabeled = _graph(None)
    labeled_nodes, labeled_edges = enumerate_physical_complete_route_chains(labeled)
    unlabeled_nodes, unlabeled_edges = enumerate_physical_complete_route_chains(unlabeled)
    np.testing.assert_array_equal(labeled_nodes, unlabeled_nodes)
    np.testing.assert_array_equal(labeled_edges, unlabeled_edges)
    table = enumerate_complete_route_candidates(unlabeled)
    np.testing.assert_array_equal(table.node_indices, unlabeled_nodes)
    assert not np.any(table.labels)
    assert not np.any(table.fake_endpoint)


def test_truth_id_permutation_does_not_change_chains_or_consistency_labels():
    original = np.tile(np.asarray([10, 20], dtype=np.int64), 4)
    remapped = np.tile(np.asarray([110, 220], dtype=np.int64), 4)
    first = enumerate_complete_route_candidates(_graph(original))
    second = enumerate_complete_route_candidates(_graph(remapped))
    np.testing.assert_array_equal(first.node_indices, second.node_indices)
    np.testing.assert_array_equal(first.score_edge_indices, second.score_edge_indices)
    np.testing.assert_array_equal(first.labels, second.labels)
    assert int(np.count_nonzero(first.labels)) == 2


def test_truth_permutation_does_not_change_physical_features():
    original = _graph(np.tile(np.asarray([10, 20], dtype=np.int64), 4))
    remapped = _graph(np.tile(np.asarray([110, 220], dtype=np.int64), 4))
    np.testing.assert_allclose(original.node_features, remapped.node_features)
    np.testing.assert_allclose(original.score_edge_features, remapped.score_edge_features)
    np.testing.assert_array_equal(original.score_edge_source, remapped.score_edge_source)
    np.testing.assert_array_equal(original.score_edge_destination, remapped.score_edge_destination)


def test_attach_labels_is_the_only_truth_dependent_step():
    graph = _graph(np.tile(np.asarray([10, 20], dtype=np.int64), 4))
    nodes, edges = enumerate_physical_complete_route_chains(graph)
    labels, fake, hard = attach_route_candidate_labels(graph, nodes, edges)
    blank = replace(graph, event=replace(graph.event, truth_particle_id=None, truth_pdg=None, truth_match_fraction=None))
    blank_labels, blank_fake, blank_hard = attach_route_candidate_labels(blank, nodes, edges)
    assert int(np.count_nonzero(labels)) == 2
    assert not np.any(blank_labels)
    assert not np.any(blank_fake)
    np.testing.assert_array_equal(hard, blank_hard)
    with pytest.raises(ValueError, match="align"):
        attach_route_candidate_labels(graph, nodes, edges[:0])
