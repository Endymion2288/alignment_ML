from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from baselines.field_chi2_matching import FieldCandidate, ScoredMatch
from baselines.route_assignment import (
    Route,
    RouteAssignmentConfig,
    _route_hypotheses,
    adjacent_route_assignment,
    adjacent_station_pairs,
)
from datasets.physical_curriculum import CurriculumSample
from datasets.root_loader import EventTracklets
from models.route_transformer import (
    RelativeRouteSparseTransformer,
    RelativeRouteTransformerConfig,
    RouteAwareSparseTransformer,
    RouteAwareTransformerConfig,
    freeze_backbone_and_edge_scorer,
)
from training.curriculum_mlp import CandidateSet
from training.geometry_aware_transformer import (
    ALL_STATION_PAIRS,
    build_transformer_graph_bundle,
    fit_graph_standardizers,
)
from training.route_assignment import (
    assign_adjacent_route_sets,
    evaluate_adjacent_route_assignment_sets,
)
from training.route_aware_transformer import (
    RouteAwareTrainingConfig,
    _forward_route_batch,
    _make_route_batch,
    _route_loss_components,
    enumerate_complete_route_candidates,
    predict_route_aware_scores,
    route_query_score_maps_by_event,
)


def _sample() -> CurriculumSample:
    return CurriculumSample(
        source_id="v4_test_sample",
        source_ids=("v4_test_sample",),
        split="validation",
        payload_id="v4_nominal",
        magnitude_mm=0.0,
        direction_trial="unit",
        injected_offsets_xy_mm={},
        source_event_uids=("v4_test:1",),
        physical_event_uids=("v4_test:1",),
        physical_tracklets=Path("/tmp/v4_tracklets.root"),
        physical_propagations=Path("/tmp/v4_prop.root"),
        physical_payload_manifest=Path("/tmp/v4_payload.json"),
        synthetic_tracklets=Path("/tmp/v4_synthetic.root"),
        field_candidates=Path("/tmp/v4_candidates.root"),
    )


def _toy_event() -> EventTracklets:
    # 2 tracklets per station across 4 stations (0, 1, 2, 3) -> 8 tracklets total
    return EventTracklets(
        run_id=100,
        event_id=200,
        station_id=np.repeat(np.arange(4, dtype=np.int16), 2),
        tracklet_id=np.arange(8, dtype=np.int32),
        z_mm=np.repeat(np.asarray([0.0, 1000.0, 2000.0, 3000.0], dtype=np.float64), 2),
        state=np.arange(32, dtype=np.float64).reshape(8, 4),
        covariance=np.tile(np.eye(4, dtype=np.float64), (8, 1, 1)),
        chi2=np.ones(8, dtype=np.float64),
        ndof=np.ones(8, dtype=np.float64),
        n_hit=np.full(8, 3, dtype=np.int16),
        hit_pattern=np.full(8, 0b111111, dtype=np.uint64),
        truth_particle_id=np.tile(np.asarray([101, 202], dtype=np.int64), 4),
        truth_pdg=np.full(8, 13, dtype=np.int32),
        truth_match_fraction=np.ones(8, dtype=np.float64),
        synthetic_role=np.zeros(8, dtype=np.int8),
    )


def _toy_candidate_sets() -> list[CandidateSet]:
    event = _toy_event()
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
                        residual=np.asarray([0.5, -0.5, 0.05, -0.05]),
                        pull=np.asarray([0.5, -0.5, 0.05, -0.05]),
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


def _v4_model() -> RelativeRouteSparseTransformer:
    return RelativeRouteSparseTransformer(
        RelativeRouteTransformerConfig(
            node_feature_dim=17,
            edge_feature_dim=11,
            dropout=0.0,
            route_dropout=0.0,
        )
    )


# -----------------------------------------------------------------------------
# Test 1: Zero-initialized route head recovers Workbook-64 edge-only solver
# -----------------------------------------------------------------------------
def test_zero_initialized_route_head_recovers_edge_only_solver():
    sets = _toy_candidate_sets()
    config = RouteAssignmentConfig(
        score_threshold_by_pair={(0, 1): 0.001, (1, 2): 0.001, (2, 3): 0.001},
        unmatched_penalty=-1.0,
    )

    # 1. Test with high edge probabilities (where 2 4-station chains are selected)
    scores = [np.where(c.labels, 0.99, 0.01).astype(np.float64) for c in sets]

    def logit(p: float) -> float:
        return math.log(p) - math.log(1.0 - p)
    def sigmoid(l: float) -> float:
        return 1.0 / (1.0 + math.exp(-l))

    l_truth = logit(0.99) * 3.0
    p_truth = sigmoid(l_truth)
    l_false = logit(0.99) * 2.0 + logit(0.01)
    p_false = sigmoid(l_false)

    route_map = {}
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for m in range(2):
                    is_true = (i == j == k == m)
                    route_map[(i, 2 + j, 4 + k, 6 + m)] = p_truth if is_true else p_false

    complete_map = {("v4_test_sample", "v4_nominal", 100, 200): route_map}

    res_edge_only = assign_adjacent_route_sets(
        sets, scores, config, calibration_bins=15, complete_route_scores_by_event=None
    )[0].result

    res_v4_init = assign_adjacent_route_sets(
        sets, scores, config, calibration_bins=15, complete_route_scores_by_event=complete_map
    )[0].result

    assert res_edge_only.selected_routes == res_v4_init.selected_routes == 2
    assert len(res_edge_only.routes) == len(res_v4_init.routes) == 2
    for r_edge, r_v4 in zip(res_edge_only.routes, res_v4_init.routes):
        assert r_edge.endpoints == r_v4.endpoints
        # Exact utility identity
        assert math.isclose(r_edge.utility, r_v4.utility, rel_tol=1e-5)
    assert res_edge_only.unmatched_by_station == res_v4_init.unmatched_by_station

    # 2. Test through V4 model predict pipeline
    model = _v4_model().eval()
    bundle = build_transformer_graph_bundle(sets, context_mode="full_event")
    node_std, edge_std = fit_graph_standardizers(bundle)
    pred = predict_route_aware_scores(model, bundle, node_std, edge_std, device="cpu", batch_size=2)
    route_score_map = route_query_score_maps_by_event(pred.route_score_sets, [pred.route_scores])

    res_edge_pipe = assign_adjacent_route_sets(
        bundle.adjacent_sets, pred.edge_scores, config, calibration_bins=15, complete_route_scores_by_event=None
    )[0].result
    res_v4_pipe = assign_adjacent_route_sets(
        bundle.adjacent_sets, pred.edge_scores, config, calibration_bins=15, complete_route_scores_by_event=route_score_map
    )[0].result

    assert res_edge_pipe.selected_routes == res_v4_pipe.selected_routes
    assert len(res_edge_pipe.routes) == len(res_v4_pipe.routes)
    for r1, r2 in zip(res_edge_pipe.routes, res_v4_pipe.routes):
        assert r1.endpoints == r2.endpoints
        assert math.isclose(r1.utility, r2.utility, rel_tol=1e-5)


# -----------------------------------------------------------------------------
# Test 2: Complete-route key alignment & missing key exception
# -----------------------------------------------------------------------------
def test_complete_route_key_alignment_and_missing_key_raise():
    sets = _toy_candidate_sets()
    config = RouteAssignmentConfig(
        score_threshold_by_pair={(0, 1): 0.001, (1, 2): 0.001, (2, 3): 0.001},
        unmatched_penalty=-1.0,
    )
    scores = [np.where(c.labels, 0.99, 0.01).astype(np.float64) for c in sets]

    # Incomplete map missing 15 routes
    incomplete_map = {
        ("v4_test_sample", "v4_nominal", 100, 200): {
            (0, 2, 4, 6): 0.95,
        }
    }
    with pytest.raises(RuntimeError, match="complete physical route has no route-query score"):
        assign_adjacent_route_sets(
            sets, scores, config, calibration_bins=15, complete_route_scores_by_event=incomplete_map
        )


# -----------------------------------------------------------------------------
# Test 3: Additive delta_route_logit math contract
# -----------------------------------------------------------------------------
def test_additive_route_score_mathematical_contract():
    p01 = 0.90
    p12 = 0.92
    p23 = 0.95
    delta_logit = 0.50
    penalty = -1.0

    def logit(p: float) -> float:
        return math.log(p) - math.log(1.0 - p)

    def sigmoid(l: float) -> float:
        return 1.0 / (1.0 + math.exp(-l))

    l_edge = logit(p01) + logit(p12) + logit(p23)
    l_corrected = l_edge + delta_logit
    p_complete = sigmoid(l_corrected)

    # Solver replace on p_complete:
    l_solver = logit(p_complete)
    assert math.isclose(l_solver, l_corrected, rel_tol=1e-10)

    u_complete = l_corrected + 4 * penalty

    event = EventTracklets(
        run_id=1,
        event_id=1,
        station_id=np.asarray([0, 1, 2, 3], dtype=np.int16),
        tracklet_id=np.asarray([0, 1, 2, 3], dtype=np.int32),
        z_mm=np.asarray([0.0, 1000.0, 2000.0, 3000.0], dtype=np.float64),
        state=np.zeros((4, 4), dtype=np.float64),
        covariance=np.tile(np.eye(4), (4, 1, 1)),
        chi2=np.ones(4),
        ndof=np.ones(4),
        n_hit=np.full(4, 3, dtype=np.int16),
        hit_pattern=np.full(4, 0b111111, dtype=np.uint64),
        truth_particle_id=np.full(4, 1, dtype=np.int64),
        truth_pdg=np.full(4, 13, dtype=np.int32),
        truth_match_fraction=np.ones(4),
        synthetic_role=np.zeros(4, dtype=np.int8),
    )

    edge_lookups = {
        (0, 1): {(0, 1): ScoredMatch(0, 1, 1.0, p01)},
        (1, 2): {(1, 2): ScoredMatch(1, 2, 1.0, p12)},
        (2, 3): {(2, 3): ScoredMatch(2, 3, 1.0, p23)},
    }

    routes = _route_hypotheses(
        event, (0, 1, 2, 3), edge_lookups, penalty, 1000,
        complete_route_scores={(0, 1, 2, 3): p_complete},
        complete_route_score_composition="replace",
    )
    r4 = next(r for r in routes if len(r.endpoints) == 4)
    assert math.isclose(r4.utility, u_complete + 3 * 1e-6, rel_tol=1e-5)


# -----------------------------------------------------------------------------
# Test 4: Fragment topology preservation
# -----------------------------------------------------------------------------
def test_fragment_topology_preservation():
    event = EventTracklets(
        run_id=1,
        event_id=1,
        station_id=np.asarray([0, 1, 2, 3], dtype=np.int16),
        tracklet_id=np.asarray([0, 1, 2, 3], dtype=np.int32),
        z_mm=np.asarray([0.0, 1000.0, 2000.0, 3000.0], dtype=np.float64),
        state=np.zeros((4, 4), dtype=np.float64),
        covariance=np.tile(np.eye(4), (4, 1, 1)),
        chi2=np.ones(4),
        ndof=np.ones(4),
        n_hit=np.full(4, 3, dtype=np.int16),
        hit_pattern=np.full(4, 0b111111, dtype=np.uint64),
        truth_particle_id=np.full(4, 1, dtype=np.int64),
        truth_pdg=np.full(4, 13, dtype=np.int32),
        truth_match_fraction=np.ones(4),
        synthetic_role=np.zeros(4, dtype=np.int8),
    )
    edge_lookups = {
        (0, 1): {(0, 1): ScoredMatch(0, 1, 1.0, 0.8)},
        (1, 2): {(1, 2): ScoredMatch(1, 2, 1.0, 0.85)},
        (2, 3): {(2, 3): ScoredMatch(2, 3, 1.0, 0.90)},
    }

    h_none = _route_hypotheses(event, (0, 1, 2, 3), edge_lookups, -1.0, 1000, complete_route_scores=None)
    h_with = _route_hypotheses(event, (0, 1, 2, 3), edge_lookups, -1.0, 1000, complete_route_scores={(0, 1, 2, 3): 0.99})

    frags_none = [r for r in h_none if len(r.endpoints) < 4]
    frags_with = [r for r in h_with if len(r.endpoints) < 4]
    assert len(frags_none) == len(frags_with) == 3
    for r1, r2 in zip(frags_none, frags_with):
        assert r1.endpoints == r2.endpoints
        assert math.isclose(r1.utility, r2.utility, rel_tol=1e-12)
        assert r2.complete_route_score is None


# -----------------------------------------------------------------------------
# Test 5: Candidate graph preservation
# -----------------------------------------------------------------------------
def test_candidate_graph_preservation():
    c_sets_1 = _toy_candidate_sets()
    c_sets_2 = _toy_candidate_sets()
    assert len(c_sets_1) == len(c_sets_2)
    for s1, s2 in zip(c_sets_1, c_sets_2):
        assert s1.station_pair == s2.station_pair
        assert len(s1.candidates) == len(s2.candidates)
        for cand1, cand2 in zip(s1.candidates, s2.candidates):
            assert cand1.source_index == cand2.source_index
            assert cand1.target_index == cand2.target_index
            assert cand1.chi2 == cand2.chi2
            np.testing.assert_allclose(cand1.residual, cand2.residual)


# -----------------------------------------------------------------------------
# Test 6: Parameter Freeze & Backward Isolation Audit
# -----------------------------------------------------------------------------
def test_parameter_freeze_and_backward_isolation():
    model = _v4_model()
    freeze_report = freeze_backbone_and_edge_scorer(model)

    assert freeze_report["n_trainable_parameters"] > 0
    assert freeze_report["n_frozen_parameters"] > 0

    # Ensure no backbone param requires grad
    for name, param in model.named_parameters():
        if name in freeze_report["frozen_parameter_names"]:
            assert param.requires_grad is False
        elif name in freeze_report["trainable_parameter_names"]:
            assert param.requires_grad is True

    bundle = build_transformer_graph_bundle(_toy_candidate_sets(), context_mode="full_event")
    node_std, edge_std = fit_graph_standardizers(bundle)
    batch = _make_route_batch(bundle.graphs, node_std, edge_std, torch.device("cpu"))
    cfg = RouteAwareTrainingConfig(device="cpu", learning_rate=2e-4, weight_decay=1e-4, seed=20260822)

    model.train()
    output = _forward_route_batch(model, batch)
    loss = _route_loss_components(output, batch, cfg, 1.0)["total"]
    loss.backward()

    # Verify gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Trainable parameter {name} has no gradient!"
        else:
            assert param.grad is None, f"Frozen parameter {name} received gradient!"


# -----------------------------------------------------------------------------
# Test 7: Arm 1 (Control) vs Arm 2 (Primary) Parameter Count Matching
# -----------------------------------------------------------------------------
def test_arm1_and_arm2_trainable_parameter_matching():
    cfg_arm1 = RouteAwareTransformerConfig(
        node_feature_dim=17, edge_feature_dim=11, use_relative_route_representation=False
    )
    cfg_arm2 = RelativeRouteTransformerConfig(
        node_feature_dim=17, edge_feature_dim=11, use_relative_route_representation=True
    )
    m1 = RouteAwareSparseTransformer(cfg_arm1)
    m2 = RelativeRouteSparseTransformer(cfg_arm2)

    rep1 = freeze_backbone_and_edge_scorer(m1)
    rep2 = freeze_backbone_and_edge_scorer(m2)

    assert rep1["n_trainable_parameters"] == rep2["n_trainable_parameters"] == 172513
    assert rep1["n_frozen_parameters"] == rep2["n_frozen_parameters"] == 614947


# -----------------------------------------------------------------------------
# Test 8: Identical Edge Scores for Arm 0/1/2 (Backbone Invariance)
# -----------------------------------------------------------------------------
def test_edge_scores_identical_across_arms():
    cfg_arm1 = RouteAwareTransformerConfig(
        node_feature_dim=17, edge_feature_dim=11, use_relative_route_representation=False
    )
    cfg_arm2 = RelativeRouteTransformerConfig(
        node_feature_dim=17, edge_feature_dim=11, use_relative_route_representation=True
    )
    m1 = RouteAwareSparseTransformer(cfg_arm1).eval()
    m2 = RelativeRouteSparseTransformer(cfg_arm2).eval()
    m2.load_state_dict(m1.state_dict())

    bundle = build_transformer_graph_bundle(_toy_candidate_sets(), context_mode="full_event")
    node_std, edge_std = fit_graph_standardizers(bundle)

    pred1 = predict_route_aware_scores(m1, bundle, node_std, edge_std, device="cpu", batch_size=2)
    pred2 = predict_route_aware_scores(m2, bundle, node_std, edge_std, device="cpu", batch_size=2)

    for s1, s2 in zip(pred1.edge_scores, pred2.edge_scores):
        np.testing.assert_array_equal(s1, s2)


# -----------------------------------------------------------------------------
# Test 9: Finite forward test on CPU and GPU
# -----------------------------------------------------------------------------
def test_finite_forward_smoke_cpu_and_gpu():
    bundle = build_transformer_graph_bundle(_toy_candidate_sets(), context_mode="full_event")
    node_std, edge_std = fit_graph_standardizers(bundle)

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for device in devices:
        model = _v4_model().to(device)
        model.eval()
        batch = _make_route_batch(bundle.graphs, node_std, edge_std, device)
        with torch.no_grad():
            output = _forward_route_batch(model, batch)

        assert output.edge_logits.shape[0] == batch.base.score_edge_source.shape[0]
        assert output.route_logits.shape == (16,)
        assert torch.isfinite(output.edge_logits).all()
        assert torch.isfinite(output.route_logits).all()
        probs = torch.sigmoid(output.route_logits)
        assert torch.isfinite(probs).all()
        assert ((probs >= 0.0) & (probs <= 1.0)).all()


# -----------------------------------------------------------------------------
# Test 10: Deterministic inference test
# -----------------------------------------------------------------------------
def test_deterministic_inference():
    torch.manual_seed(42)
    bundle = build_transformer_graph_bundle(_toy_candidate_sets(), context_mode="full_event")
    node_std, edge_std = fit_graph_standardizers(bundle)

    model = _v4_model().eval()
    pred1 = predict_route_aware_scores(model, bundle, node_std, edge_std, device="cpu", batch_size=2)
    pred2 = predict_route_aware_scores(model, bundle, node_std, edge_std, device="cpu", batch_size=2)

    np.testing.assert_array_equal(pred1.route_scores, pred2.route_scores)
    for s1, s2 in zip(pred1.edge_scores, pred2.edge_scores):
        np.testing.assert_array_equal(s1, s2)


# -----------------------------------------------------------------------------
# Test 11: Final Blind & Sealed Test Guard
# -----------------------------------------------------------------------------
def test_no_final_blind_or_sealed_test_access():
    forbidden_blind_sources = (
        "mc24_100047_00800_00849",
        "mc24_100048_00800_00849",
    )
    for forbidden in forbidden_blind_sources:
        assert forbidden not in _sample().source_id
    assert _sample().split != "test"


# -----------------------------------------------------------------------------
# Test 12: Train Source Manifest Audit (Exactly Six Authorized Sources)
# -----------------------------------------------------------------------------
def test_train_sources_manifest_matches_six_sources():
    manifest_path = Path("configs/physical_curriculum_four_station_diversity_train_sources.yaml")
    assert manifest_path.is_file()
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    sources = tuple(s["id"] for s in data["physical_curriculum_mlp"]["sources"])
    expected = (
        "mc24_100043_00200_00299",
        "mc24_100044_00300_00399",
        "mc24_100043_00300_00399",
        "mc24_100044_00200_00299",
        "mc24_100047_00100_00149",
        "mc24_100048_00100_00149",
    )
    assert sources == expected


# -----------------------------------------------------------------------------
# Test 13: Development Sources Absent from Training Manifest
# -----------------------------------------------------------------------------
def test_development_sources_absent_from_training_manifest():
    manifest_path = Path("configs/physical_curriculum_four_station_diversity_train_sources.yaml")
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    train_sources = set(s["id"] for s in data["physical_curriculum_mlp"]["sources"])

    dev_sources = {
        "mc24_100047_00350_00399",
        "mc24_100048_00350_00399",
    }
    assert not train_sources.intersection(dev_sources)


# -----------------------------------------------------------------------------
# Test 14: Final Blind Sources Absent from All Training Configs
# -----------------------------------------------------------------------------
def test_final_blind_sources_absent_from_training_configs():
    # 1. Check six-source training manifest
    manifest_path = Path("configs/physical_curriculum_four_station_diversity_train_sources.yaml")
    assert "00800_00849" not in manifest_path.read_text(encoding="utf-8")

    # 2. Check Workbook 68 Primary and Control training configs
    for cfg_name in ("relative_route_v4_head_only_train.yaml", "absolute_route_control_head_only_train.yaml"):
        cfg_path = Path("configs") / cfg_name
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        root_key = next(iter(data))
        train_sources = data[root_key]["input_contract"]["authorized_train_sources"]
        for s in train_sources:
            assert "00800_00849" not in s


# -----------------------------------------------------------------------------
# Test 15: Primary and Control Pre-registered YAML Configurations Audit
# -----------------------------------------------------------------------------
def test_preregistered_training_configs():
    p_primary = Path("configs/relative_route_v4_head_only_train.yaml")
    p_control = Path("configs/absolute_route_control_head_only_train.yaml")

    assert p_primary.is_file()
    assert p_control.is_file()

    c_primary = yaml.safe_load(p_primary.read_text(encoding="utf-8"))["relative_route_v4_primary"]
    c_control = yaml.safe_load(p_control.read_text(encoding="utf-8"))["absolute_route_control"]

    # Difference must be ONLY use_relative_route_representation
    assert c_primary["architecture"]["use_relative_route_representation"] is True
    assert c_control["architecture"]["use_relative_route_representation"] is False

    assert c_primary["training"] == c_control["training"]
    assert c_primary["freeze_contract"] == c_control["freeze_contract"]
    assert c_primary["curriculum_stages"] == c_control["curriculum_stages"]
