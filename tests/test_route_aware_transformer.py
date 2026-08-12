from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from baselines.field_chi2_matching import FieldCandidate
from datasets.physical_curriculum import CurriculumSample
from datasets.root_loader import EventTracklets
from models.route_transformer import RouteAwareSparseTransformer, RouteAwareTransformerConfig
from training.curriculum_mlp import CandidateSet
from training.geometry_aware_transformer import ALL_STATION_PAIRS, build_transformer_graph_bundle, fit_graph_standardizers
from training.route_aware_transformer import (
    RouteAwareTrainingConfig,
    RouteAwareTransformerArtifact,
    _forward_route_batch,
    _make_route_batch,
    _route_competition_loss,
    _route_loss_components,
    enumerate_complete_route_candidates,
    load_route_aware_transformer_artifact,
    predict_route_aware_scores,
    route_query_score_maps_by_event,
    save_route_aware_transformer_artifact,
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


def _model() -> RouteAwareSparseTransformer:
    return RouteAwareSparseTransformer(
        RouteAwareTransformerConfig(
            node_feature_dim=17,
            edge_feature_dim=11,
            dropout=0.0,
            route_dropout=0.0,
        )
    )


def test_complete_route_table_uses_only_existing_adjacent_candidate_edges():
    bundle = build_transformer_graph_bundle(_candidate_sets(), context_mode="full_event")
    table = enumerate_complete_route_candidates(bundle.graphs[0])
    assert table.node_indices.shape == (16, 4)
    assert table.score_edge_indices.shape == (16, 3)
    assert int(np.count_nonzero(table.labels)) == 2
    assert not np.any(table.fake_endpoint)
    assert not np.any(table.hard_negative)


def test_route_query_starts_as_a_zero_edge_correction_and_trains_losses():
    torch.manual_seed(23)
    bundle = build_transformer_graph_bundle(_candidate_sets(), context_mode="full_event")
    node_standardizer, edge_standardizer = fit_graph_standardizers(bundle)
    model = _model()
    batch = _make_route_batch(bundle.graphs, node_standardizer, edge_standardizer, torch.device("cpu"))
    output = _forward_route_batch(model, batch)
    assert output.route_logits.shape == (16,)
    assert torch.all(output.route_edge_counts > 0.0)
    torch.testing.assert_close(output.edge_logits, output.base_edge_logits)
    losses = _route_loss_components(
        output,
        batch,
        RouteAwareTrainingConfig(device="cpu", focal_gamma=0.0),
        edge_positive_weight=1.0,
    )
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert model.route_score.weight.grad is not None


def test_endpoint_competition_rewards_the_truth_route_over_a_shared_endpoint_fake():
    route_nodes = torch.as_tensor([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=torch.long)
    labels = torch.as_tensor([1.0, 0.0])
    truth_wins = _route_competition_loss(
        torch.as_tensor([4.0, -4.0]), labels, route_nodes, node_count=7
    )
    fake_wins = _route_competition_loss(
        torch.as_tensor([-4.0, 4.0]), labels, route_nodes, node_count=7
    )
    assert truth_wins < fake_wins


def test_route_aware_artifact_round_trip_preserves_physical_edge_scores(tmp_path: Path):
    torch.manual_seed(29)
    bundle = build_transformer_graph_bundle(_candidate_sets(), context_mode="full_event")
    node_standardizer, edge_standardizer = fit_graph_standardizers(bundle)
    model = _model().eval()
    from training.geometry_aware_transformer import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES

    artifact = RouteAwareTransformerArtifact(
        node_feature_names=NODE_FEATURE_NAMES,
        edge_feature_names=EDGE_FEATURE_NAMES,
        all_station_pairs=ALL_STATION_PAIRS,
        output_station_pairs=((0, 1), (1, 2), (2, 3)),
        node_standardizer=node_standardizer,
        edge_standardizer=edge_standardizer,
        model_config=model.config,
        context_mode="full_event",
        training_summary={},
    )
    before = predict_route_aware_scores(
        model, bundle, node_standardizer, edge_standardizer, device="cpu", batch_size=1
    )
    assert len(before.route_score_sets) == 1
    score_maps = route_query_score_maps_by_event(before.route_score_sets, (before.route_scores,))
    assert len(score_maps) == 1
    assert len(next(iter(score_maps.values()))) == 16
    checkpoint = tmp_path / "route_v2.pt"
    save_route_aware_transformer_artifact(checkpoint, model, artifact)
    loaded, loaded_artifact = load_route_aware_transformer_artifact(checkpoint, device="cpu")
    after = predict_route_aware_scores(
        loaded,
        bundle,
        loaded_artifact.node_standardizer,
        loaded_artifact.edge_standardizer,
        device="cpu",
        batch_size=1,
    )
    for left, right in zip(before.edge_scores, after.edge_scores):
        np.testing.assert_allclose(left, right)
