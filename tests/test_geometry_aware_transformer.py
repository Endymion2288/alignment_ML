from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from baselines.field_chi2_matching import FieldCandidate
from baselines.mlp_pair_classifier import FeatureStandardizer
from datasets.physical_curriculum import CurriculumSample
from datasets.root_loader import EventTracklets
from models.geometric_attention import SparseGeometricAttention
from models.transformer import GeometryAwareSparseTransformer, SparseTransformerConfig
from training.curriculum_mlp import CandidateSet
from training.geometry_aware_transformer import (
    ALL_STATION_PAIRS,
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    CurriculumStage,
    TransformerArtifact,
    TransformerTrainingConfig,
    apply_frozen_transformer_calibration,
    build_transformer_graph_bundle,
    calibrate_transformer_scores,
    fit_graph_standardizers,
    geometric_edge_features,
    load_transformer_artifact,
    predict_local_transformer_scores,
    predict_transformer_scores,
    save_transformer_artifact,
    train_transformer_v1,
)
from training.transformer_route_selection import select_route_operating_point


def _sample(split: str = "train") -> CurriculumSample:
    return CurriculumSample(
        source_id=f"{split}_source",
        source_ids=(f"{split}_source",),
        split=split,
        payload_id=f"mag_0_{split}_00",
        magnitude_mm=0.0,
        direction_trial=f"{split}_00",
        injected_offsets_xy_mm={},
        source_event_uids=(f"{split}_source:1",),
        physical_event_uids=(f"{split}_source:1",),
        physical_tracklets=Path("/tmp/physical_tracklets.root"),
        physical_propagations=Path("/tmp/physical_propagations.root"),
        physical_payload_manifest=Path("/tmp/physical_payload.json"),
        synthetic_tracklets=Path("/tmp/synthetic_tracklets.root"),
        field_candidates=Path("/tmp/field_candidates.root"),
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
    sample = _sample()
    event = _event()
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


def test_sparse_attention_has_no_implicit_all_pairs_messages():
    torch.manual_seed(3)
    attention = SparseGeometricAttention(
        d_model=8,
        nhead=2,
        edge_feature_dim=3,
        num_station_pairs=1,
        use_geometric_bias=False,
        use_chi2_physics_term=False,
    ).eval()
    nodes = torch.randn(3, 8)
    altered = nodes.clone()
    altered[2] += 1000.0
    common = (
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([1, 0], dtype=torch.long),
        torch.zeros((2, 3)),
        torch.ones(2),
        torch.zeros(2, dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
    )
    first = attention(nodes, *common)
    second = attention(altered, *common)
    torch.testing.assert_close(first[:2], second[:2])
    torch.testing.assert_close(first[2], torch.zeros(8))
    torch.testing.assert_close(second[2], torch.zeros(8))


def test_trace_depth_four_matches_forward_and_depth_zero_uses_no_messages():
    torch.manual_seed(17)
    bundle = build_transformer_graph_bundle(_candidate_sets(), context_mode="full_event")
    node_standardizer, edge_standardizer = fit_graph_standardizers(bundle)
    config = SparseTransformerConfig(
        node_feature_dim=len(NODE_FEATURE_NAMES),
        edge_feature_dim=len(EDGE_FEATURE_NAMES),
        dropout=0.0,
        use_local_edge_residual=True,
    )
    model = GeometryAwareSparseTransformer(config).eval()
    graph = bundle.graphs[0]
    from training.geometry_aware_transformer import _make_batch

    batch = _make_batch((graph,), node_standardizer, edge_standardizer, torch.device("cpu"))
    forward = model(
        batch.node_features,
        batch.station_ids,
        batch.message_edge_source,
        batch.message_edge_destination,
        batch.message_edge_features,
        batch.message_edge_chi2,
        batch.message_edge_station_pair,
        batch.message_edge_direction,
        batch.score_edge_source,
        batch.score_edge_destination,
        batch.score_edge_features,
        batch.score_edge_station_pair,
    )
    traced, trace = model.forward_with_trace(
        batch.node_features,
        batch.station_ids,
        batch.message_edge_source,
        batch.message_edge_destination,
        batch.message_edge_features,
        batch.message_edge_chi2,
        batch.message_edge_station_pair,
        batch.message_edge_direction,
        batch.score_edge_source,
        batch.score_edge_destination,
        batch.score_edge_features,
        batch.score_edge_station_pair,
    )
    torch.testing.assert_close(traced, forward)
    assert len(trace["node_states_by_depth"]) == 5
    assert len(trace["logits_by_depth"]) == 5
    assert len(trace["attention_by_layer"]) == 4

    _, zero_trace = model.forward_with_trace(
        batch.node_features,
        batch.station_ids,
        batch.message_edge_source[:0],
        batch.message_edge_destination[:0],
        batch.message_edge_features[:0],
        batch.message_edge_chi2[:0],
        batch.message_edge_station_pair[:0],
        batch.message_edge_direction[:0],
        batch.score_edge_source,
        batch.score_edge_destination,
        batch.score_edge_features,
        batch.score_edge_station_pair,
        maximum_layers=0,
    )
    torch.testing.assert_close(
        zero_trace["logits_by_depth"][0], trace["logits_by_depth"][0]
    )


def test_graph_context_ablation_keeps_same_output_candidate_rows():
    candidate_sets = _candidate_sets()
    full = build_transformer_graph_bundle(candidate_sets, context_mode="full_event")
    pair = build_transformer_graph_bundle(candidate_sets, context_mode="station_pair")

    assert len(full.graphs) == 1
    assert len(pair.graphs) == 3
    assert sum(graph.score_labels.size for graph in full.graphs) == 12
    assert sum(graph.score_labels.size for graph in pair.graphs) == 12
    assert full.graphs[0].message_edge_source.size == 48
    assert sum(graph.message_edge_source.size for graph in pair.graphs) == 24
    np.testing.assert_array_equal(
        geometric_edge_features(_event(), candidate_sets[0].candidates[0]),
        np.asarray([1.0, -2.0, 0.1, -0.2, 1.0, -2.0, 0.1, -0.2, np.log(2.0), 0.0, 1.0]),
    )
    reverse = geometric_edge_features(_event(), candidate_sets[0].candidates[0], reverse=True)
    np.testing.assert_array_equal(reverse[:8], -geometric_edge_features(_event(), candidate_sets[0].candidates[0])[:8])
    assert reverse[-1] == -1.0


def test_transformer_artifact_round_trip_preserves_scores(tmp_path: Path):
    torch.manual_seed(5)
    bundle = build_transformer_graph_bundle(_candidate_sets(), context_mode="full_event")
    node_standardizer, edge_standardizer = fit_graph_standardizers(bundle)
    config = SparseTransformerConfig(
        node_feature_dim=len(NODE_FEATURE_NAMES),
        edge_feature_dim=len(EDGE_FEATURE_NAMES),
        dropout=0.0,
        use_local_edge_residual=True,
    )
    model = GeometryAwareSparseTransformer(config).eval()
    artifact = TransformerArtifact(
        node_feature_names=NODE_FEATURE_NAMES,
        edge_feature_names=EDGE_FEATURE_NAMES,
        all_station_pairs=ALL_STATION_PAIRS,
        output_station_pairs=((0, 1), (1, 2), (2, 3)),
        node_standardizer=node_standardizer,
        edge_standardizer=edge_standardizer,
        model_config=config,
        context_mode="full_event",
        training_summary={},
    )
    before = predict_transformer_scores(
        model, bundle, node_standardizer, edge_standardizer, device="cpu", batch_size=1
    )
    local = predict_local_transformer_scores(
        model, bundle, node_standardizer, edge_standardizer, device="cpu", batch_size=1
    )
    # The V1b/V1c residual decoder starts with a zero context terminal layer.
    # Before joint training its full output is therefore exactly the local score.
    for full, local_only in zip(before, local):
        np.testing.assert_allclose(full, local_only)
    path = tmp_path / "transformer.pt"
    save_transformer_artifact(path, model, artifact)
    loaded_model, loaded_artifact = load_transformer_artifact(path, device="cpu")
    after = predict_transformer_scores(
        loaded_model,
        bundle,
        loaded_artifact.node_standardizer,
        loaded_artifact.edge_standardizer,
        device="cpu",
        batch_size=1,
    )
    for left, right in zip(before, after):
        np.testing.assert_allclose(left, right)


def test_route_threshold_selection_is_validation_graph_only():
    bundle = build_transformer_graph_bundle(_candidate_sets(), context_mode="full_event")
    scores = [np.where(candidate_set.labels, 0.9, 0.1) for candidate_set in bundle.adjacent_sets]
    result = select_route_operating_point(
        bundle.adjacent_sets,
        scores,
        threshold_grid=(0.05, 0.5, 0.95),
        unmatched_penalties=(0.0,),
        initial_thresholds={(0, 1): 0.05, (1, 2): 0.05, (2, 3): 0.05},
        maximum_sweeps=1,
        capture_criteria={
            "minimum_complete_track_efficiency": 0.5,
            "minimum_complete_track_purity": 0.5,
            "maximum_track_fake_rate": 0.5,
        },
        calibration_bins=5,
    )

    assert result.capture_by_magnitude == {0.0: True}
    assert set(result.thresholds) == {(0, 1), (1, 2), (2, 3)}
    assert result.search_rows


def test_local_pretraining_precedes_joint_context_phase():
    bundle = build_transformer_graph_bundle(_candidate_sets(), context_mode="full_event")
    model, artifact, history = train_transformer_v1(
        bundle,
        bundle,
        SparseTransformerConfig(
            node_feature_dim=len(NODE_FEATURE_NAMES),
            edge_feature_dim=len(EDGE_FEATURE_NAMES),
            dropout=0.0,
            use_local_edge_residual=True,
        ),
        TransformerTrainingConfig(
            batch_size=1,
            learning_rate=1.0e-3,
            local_pretrain_learning_rate=1.0e-3,
            weight_decay=0.0,
            seed=19,
            device="cpu",
            early_stopping_patience=1,
            focal_gamma=0.0,
        ),
        stages=(CurriculumStage("joint", 0.0, 1),),
        local_pretrain_stages=(CurriculumStage("local", 0.0, 1),),
        calibration_bins=5,
    )

    assert artifact.training_summary["local_edge_pretraining"]["enabled"] is True
    assert artifact.training_summary["local_edge_pretraining"]["uses_message_edges"] is False
    assert any(row["phase"] == "local_edge_pretrain" for row in history)
    assert any(row["phase"] == "sparse_context_joint" for row in history)
    assert model.use_local_edge_residual is True


def test_station_pair_platt_calibration_round_trip_is_monotonic_and_frozen():
    bundle = build_transformer_graph_bundle(_candidate_sets(), context_mode="full_event")
    # Deliberately overconfident class-weighted-style scores: the calibration
    # needs an intercept as well as a positive temperature-like slope.
    raw = [np.where(candidate_set.labels, 0.95, 0.70) for candidate_set in bundle.adjacent_sets]
    calibrated, payload = calibrate_transformer_scores(
        bundle.adjacent_sets,
        raw,
        calibration_bins=5,
        scope="station_pair",
        method="platt",
    )
    frozen = apply_frozen_transformer_calibration(bundle.adjacent_sets, raw, payload)

    assert payload["method"] == "platt"
    assert payload["scope"] == "station_pair"
    for first, second in zip(calibrated, frozen):
        np.testing.assert_allclose(first, second)
        assert np.all((first >= 0.0) & (first <= 1.0))
        assert first[np.argmax(raw[0])] > first[np.argmin(raw[0])]
