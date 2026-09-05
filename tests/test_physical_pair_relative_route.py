"""Workbook-74 Phase A audit for the Physical Pair-Relative Route Encoder V1.

The Physical Pair-Relative Route Encoder (``route_representation_mode =
"physical_pair_relative"``) removes the route head's dependence on the
absolute/global node latent ``s0..s3`` and consumes only the Workbook-73
``R_phys`` tensor:

    R_phys = [ edge01 physical observables (11),
               edge12 physical observables (11),
               edge23 physical observables (11),
               L01, L12, L23 (frozen Workbook-64 production edge logits) ]
           = 36 dimensions

with the frozen bounded residual ``delta = B * tanh(raw_delta / B)``, B = 4.0.

These tests must all pass before any Workbook-74 training is authorized
(``training_authorized = false`` otherwise).  They never open development,
final-blind, or sealed-test data.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from baselines.field_chi2_matching import FieldCandidate
from datasets.physical_curriculum import CurriculumSample
from datasets.root_loader import EventTracklets
from models.route_transformer import (
    RelativeRouteV4Inference,
    RouteAwareSparseTransformer,
    RouteAwareTransformerConfig,
    freeze_backbone_and_edge_scorer,
)
from training.curriculum_mlp import CandidateSet
from training.geometry_aware_transformer import (
    ADJACENT_STATION_PAIRS,
    ALL_STATION_PAIRS,
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    build_transformer_graph_bundle,
    fit_graph_standardizers,
)
from training.route_aware_transformer import (
    RouteAwareTransformerArtifact,
    _forward_route_batch,
    _make_route_batch,
    load_relative_route_v4_head_only_artifact,
    save_relative_route_v4_head_only_artifact,
)
from training.source_transfer_cv import (
    FAMILY1_DSID_PAIR,
    FAMILY2_DSID_PAIR,
    V5A_SOURCE_FAMILIES,
    assert_source_family_partition,
    folds,
    validate_fold_sources,
)

BOUND = 4.0
EDGE_W = len(EDGE_FEATURE_NAMES)          # 11
RPHYS_W = 3 * EDGE_W + 3                  # 36


# -----------------------------------------------------------------------------
# Toy fixtures (same construction as the V4/V5A Phase A tests)
# -----------------------------------------------------------------------------
def _sample() -> CurriculumSample:
    return CurriculumSample(
        source_id="wb74_test_sample",
        source_ids=("wb74_test_sample",),
        split="validation",
        payload_id="wb74_nominal",
        magnitude_mm=0.0,
        direction_trial="unit",
        injected_offsets_xy_mm={},
        source_event_uids=("wb74_test:1",),
        physical_event_uids=("wb74_test:1",),
        physical_tracklets=Path("/tmp/wb74_tracklets.root"),
        physical_propagations=Path("/tmp/wb74_prop.root"),
        physical_payload_manifest=Path("/tmp/wb74_payload.json"),
        synthetic_tracklets=Path("/tmp/wb74_synthetic.root"),
        field_candidates=Path("/tmp/wb74_candidates.root"),
    )


def _toy_event() -> EventTracklets:
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


def _phys_rel_config(bound: float | None) -> RouteAwareTransformerConfig:
    return RouteAwareTransformerConfig(
        node_feature_dim=17,
        edge_feature_dim=11,
        dropout=0.0,
        route_dropout=0.0,
        use_relative_route_representation=False,
        use_additive_route_correction=True,
        route_correction_bound=bound,
        route_representation_mode="physical_pair_relative",
    )


def _absolute_config(bound: float | None) -> RouteAwareTransformerConfig:
    return RouteAwareTransformerConfig(
        node_feature_dim=17,
        edge_feature_dim=11,
        dropout=0.0,
        route_dropout=0.0,
        use_relative_route_representation=False,
        use_additive_route_correction=True,
        route_correction_bound=bound,
        route_representation_mode="absolute",
    )


def _historical_v2_model() -> RouteAwareSparseTransformer:
    return RouteAwareSparseTransformer(
        RouteAwareTransformerConfig(
            node_feature_dim=17,
            edge_feature_dim=11,
            dropout=0.0,
            route_dropout=0.0,
            use_additive_route_correction=False,
        )
    )


def _copy_frozen_non_head(src: RouteAwareSparseTransformer, dst: RouteAwareSparseTransformer) -> None:
    prefixes = (
        "route_query",
        "route_node_key",
        "route_edge_projection",
        "route_pair_embedding",
        "route_encoder",
        "route_score",
    )
    src_state = src.state_dict()
    dst_state = dst.state_dict()
    for key, value in src_state.items():
        is_head = any(key == prefix or key.startswith(f"{prefix}.") for prefix in prefixes)
        if not is_head and key in dst_state and dst_state[key].shape == value.shape:
            dst_state[key] = value
    dst.load_state_dict(dst_state)


def _wrapper(bound: float | None, *, mode: str, zero_init: bool = True) -> RelativeRouteV4Inference:
    frozen = _historical_v2_model()
    cfg = _phys_rel_config(bound) if mode == "physical_pair_relative" else _absolute_config(bound)
    trainable = RouteAwareSparseTransformer(cfg)
    _copy_frozen_non_head(frozen, trainable)
    if zero_init:
        torch.nn.init.zeros_(trainable.route_score.weight)
        torch.nn.init.zeros_(trainable.route_score.bias)
    freeze_backbone_and_edge_scorer(trainable)
    return RelativeRouteV4Inference(frozen, trainable)


def _batch_and_stds():
    bundle = build_transformer_graph_bundle(_toy_candidate_sets(), context_mode="full_event")
    node_std, edge_std = fit_graph_standardizers(bundle)
    batch = _make_route_batch(bundle.graphs, node_std, edge_std, torch.device("cpu"))
    return bundle, node_std, edge_std, batch


def _capture_route_input(trainable: RouteAwareSparseTransformer):
    holder: dict[str, torch.Tensor] = {}
    hook = trainable.route_encoder.register_forward_pre_hook(
        lambda module, inputs: holder.__setitem__("route_input", inputs[0].detach())
    )
    return holder, hook


# -----------------------------------------------------------------------------
# Representation tests (Workbook 74 section 14, items 1-5)
# -----------------------------------------------------------------------------
def test_no_absolute_node_latent_modules():
    """Item 1/2: physical_pair_relative builds no node-latent route modules."""
    model = RouteAwareSparseTransformer(_phys_rel_config(BOUND))
    # The absolute node latent route representation is impossible to consume:
    # the modules that build it (pooled query, s0..s3 key, learned edge
    # projection, pair embedding) are simply absent.
    for absent in ("route_query", "route_node_key", "route_edge_projection", "route_pair_embedding"):
        assert not hasattr(model, absent), f"physical_pair_relative unexpectedly builds {absent}"
    # Only the route encoder + route score head remain trainable.
    freeze = freeze_backbone_and_edge_scorer(model)
    assert freeze["n_trainable_parameters"] == 21377
    assert freeze["n_frozen_parameters"] == 614947


def test_route_input_width_is_36():
    """Item 3/4: route input width is exactly 3*edge_feature_dim + 3 = 36."""
    model = RouteAwareSparseTransformer(_phys_rel_config(BOUND))
    assert model.route_encoder[0].in_features == RPHYS_W == 36
    # Manifest-level width contract: 3 adjacent edges x 11 physical observables
    # + 3 frozen edge logits.
    assert EDGE_W == 11
    assert RPHYS_W == 3 * 11 + 3


def test_route_input_is_rphys_exactly():
    """Item 3: route input == [physical edge observables (33), frozen W64
    production edge logits L01/L12/L23 (3)] -- nothing else."""
    wrapper = _wrapper(BOUND, mode="physical_pair_relative", zero_init=False)
    with torch.no_grad():
        torch.nn.init.ones_(wrapper.trainable.route_score.weight)
        torch.nn.init.ones_(wrapper.trainable.route_score.bias)
    bundle, node_std, edge_std, batch = _batch_and_stds()
    holder, hook = _capture_route_input(wrapper.trainable)
    wrapper.eval()
    try:
        with torch.no_grad():
            out = _forward_route_batch(wrapper, batch)
            frozen_out = _forward_route_batch(wrapper.frozen_workbook64, batch)
    finally:
        hook.remove()
    route_input = holder["route_input"]
    r = route_input.shape[0]
    assert route_input.shape[1] == RPHYS_W
    rsei = batch.route_score_edge_indices
    # First 33 dims: the standardized physical pair-relative edge observables
    # (the exact features entering the frozen edge scorer), in 01/12/23 order.
    expected_obs = batch.base.score_edge_features[rsei].reshape(r, 3 * EDGE_W)
    torch.testing.assert_close(route_input[:, : 3 * EDGE_W], expected_obs, atol=1e-6, rtol=1e-6)
    # Last 3 dims: the frozen Workbook-64 production edge logits L01/L12/L23.
    expected_logits = frozen_out.edge_logits[rsei]
    torch.testing.assert_close(route_input[:, 3 * EDGE_W :], expected_logits, atol=1e-6, rtol=1e-6)
    assert int(out.delta_route_logits.numel()) == r


def test_route_input_excludes_node_latent():
    """Item 1/2: the physical edge-observable block of the route input is
    bitwise independent of the node features (no absolute node latent leaks)."""
    wrapper = _wrapper(BOUND, mode="physical_pair_relative", zero_init=True)
    bundle, node_std, edge_std, batch = _batch_and_stds()
    holder, hook = _capture_route_input(wrapper.trainable)
    wrapper.eval()
    try:
        with torch.no_grad():
            _forward_route_batch(wrapper, batch)
            input_a = holder["route_input"].clone()
            # Perturb ONLY the node features; the physical edge observables and
            # the candidate graph are unchanged.
            perturbed = _make_route_batch(
                bundle.graphs, node_std, edge_std, torch.device("cpu")
            )
            perturbed.base.node_features.mul_(3.7).add_(1.1)
            _forward_route_batch(wrapper, perturbed)
            input_b = holder["route_input"].clone()
    finally:
        hook.remove()
    # The first 33 dims (physical pair-relative edge observables) are identical
    # under a node-feature perturbation -> they carry no node-latent content.
    torch.testing.assert_close(
        input_a[:, : 3 * EDGE_W], input_b[:, : 3 * EDGE_W], atol=0.0, rtol=0.0
    )


def test_edge_order_is_01_12_23():
    """Item 5: route score-edge columns follow the adjacent order 0->1,1->2,2->3."""
    bundle, node_std, edge_std, batch = _batch_and_stds()
    pair_ids = batch.base.score_edge_station_pair  # global score-edge -> pair id
    adjacent_pair_ids = {
        pair: idx for idx, pair in enumerate(ALL_STATION_PAIRS) if pair in ADJACENT_STATION_PAIRS
    }
    expected_order = [adjacent_pair_ids[pair] for pair in ADJACENT_STATION_PAIRS]
    rsei = batch.route_score_edge_indices
    assert rsei.shape[1] == 3
    for col, expected_pair_id in enumerate(expected_order):
        col_pair_ids = pair_ids[rsei[:, col]]
        assert bool((col_pair_ids == expected_pair_id).all()), (
            f"route score-edge column {col} is not station pair {ADJACENT_STATION_PAIRS[col]}"
        )


# -----------------------------------------------------------------------------
# Frozen physics tests (items 6-10)
# -----------------------------------------------------------------------------
def test_edge_logits_from_frozen_w64_production_scorer():
    """Item 6: the L01/L12/L23 route-input logits are the frozen Workbook-64
    production edge logits, not a re-estimate by the new route network."""
    wrapper = _wrapper(BOUND, mode="physical_pair_relative", zero_init=False)
    with torch.no_grad():
        torch.nn.init.ones_(wrapper.trainable.route_score.weight)
        torch.nn.init.ones_(wrapper.trainable.route_score.bias)
    bundle, node_std, edge_std, batch = _batch_and_stds()
    holder, hook = _capture_route_input(wrapper.trainable)
    wrapper.eval()
    try:
        with torch.no_grad():
            _forward_route_batch(wrapper, batch)
            frozen_out = _forward_route_batch(wrapper.frozen_workbook64, batch)
    finally:
        hook.remove()
    route_input = holder["route_input"]
    rsei = batch.route_score_edge_indices
    # The route-input logits equal the frozen production edges exactly ...
    torch.testing.assert_close(
        route_input[:, 3 * EDGE_W :], frozen_out.edge_logits[rsei], atol=1e-6, rtol=1e-6
    )
    # ... and they carry no gradient (frozen input feature, not a trainable path).
    assert not route_input.requires_grad


def test_candidate_graph_and_production_edges_unchanged():
    """Items 7/8/10: the candidate graph and the production adjacent edges are
    identical to the frozen Workbook-64 baseline (the head only emits delta)."""
    wrapper = _wrapper(BOUND, mode="physical_pair_relative", zero_init=False)
    with torch.no_grad():
        torch.nn.init.normal_(wrapper.trainable.route_score.weight, std=0.5)
        torch.nn.init.normal_(wrapper.trainable.route_score.bias, std=0.5)
    bundle, node_std, edge_std, batch = _batch_and_stds()
    wrapper.eval()
    with torch.no_grad():
        wrap_out = _forward_route_batch(wrapper, batch)
        frozen_out = _forward_route_batch(wrapper.frozen_workbook64, batch)
    # Non-trivial delta, yet the production edges are the frozen W64 edges.
    assert float(wrap_out.delta_route_logits.abs().max()) > 0.0
    torch.testing.assert_close(wrap_out.edge_logits, frozen_out.edge_logits, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        wrap_out.base_edge_logits, frozen_out.base_edge_logits, atol=0.0, rtol=0.0
    )


def test_unmatched_penalty_and_solver_contract_unchanged():
    """Items 9/10: the solver consumes L_corrected = L_edge_W64 + delta_bounded
    with the production unmatched penalty; the head does not alter the solver."""
    wrapper = _wrapper(BOUND, mode="physical_pair_relative", zero_init=False)
    with torch.no_grad():
        torch.nn.init.ones_(wrapper.trainable.route_score.weight)
        torch.nn.init.ones_(wrapper.trainable.route_score.bias)
    bundle, node_std, edge_std, batch = _batch_and_stds()
    wrapper.eval()
    with torch.no_grad():
        wrap_out = _forward_route_batch(wrapper, batch)
        frozen_out = _forward_route_batch(wrapper.frozen_workbook64, batch)
    delta = wrap_out.delta_route_logits
    assert float(delta.abs().max()) <= BOUND + 1e-6
    production_edge_sum = frozen_out.edge_logits[batch.route_score_edge_indices].sum(dim=-1)
    torch.testing.assert_close(wrap_out.route_logits, production_edge_sum + delta, atol=1e-6, rtol=1e-6)


# -----------------------------------------------------------------------------
# Bound tests (items 11-14)
# -----------------------------------------------------------------------------
def test_bounded_delta_always_within_bound():
    """Item 11: delta_bounded is always within [-4, +4]."""
    model = RouteAwareSparseTransformer(_phys_rel_config(BOUND))
    with torch.no_grad():
        model.route_score.weight.mul_(1.0e6)
        model.route_score.bias.fill_(3.0e5)
    bundle, node_std, edge_std, batch = _batch_and_stds()
    model.eval()
    with torch.no_grad():
        out = _forward_route_batch(model, batch)
    delta = out.delta_route_logits
    assert delta is not None and int(delta.numel()) > 0
    assert torch.isfinite(delta).all()
    assert float(delta.abs().max()) <= BOUND + 1e-6


def test_zero_init_delta_zero_and_l_corrected_equals_edge():
    """Items 12/13: zero-init gives delta == 0 and L_corrected == L_edge_W64."""
    wrapper = _wrapper(BOUND, mode="physical_pair_relative", zero_init=True)
    bundle, node_std, edge_std, batch = _batch_and_stds()
    wrapper.eval()
    with torch.no_grad():
        wrap_out = _forward_route_batch(wrapper, batch)
        frozen_out = _forward_route_batch(wrapper.frozen_workbook64, batch)
    assert float(wrap_out.delta_route_logits.abs().max()) == 0.0
    production_edge_sum = frozen_out.edge_logits[batch.route_score_edge_indices].sum(dim=-1)
    torch.testing.assert_close(wrap_out.route_logits, production_edge_sum, atol=1e-6, rtol=1e-6)


def test_derivative_at_origin_is_one():
    """Item 12 (slope): d delta_bounded / d raw_delta at the origin == 1."""
    raw = torch.zeros(8, dtype=torch.float64, requires_grad=True)
    delta = BOUND * torch.tanh(raw / BOUND)
    grad = torch.autograd.grad(delta.sum(), raw)[0]
    torch.testing.assert_close(grad, torch.ones_like(grad), atol=1e-12, rtol=1e-12)


def test_save_load_preserves_mode_and_bound(tmp_path: Path):
    """Item 14: save/load preserves route_representation_mode + B + forward."""
    wrapper = _wrapper(BOUND, mode="physical_pair_relative", zero_init=False)
    with torch.no_grad():
        torch.nn.init.ones_(wrapper.trainable.route_score.weight)
        torch.nn.init.ones_(wrapper.trainable.route_score.bias)
    bundle, node_std, edge_std, batch = _batch_and_stds()
    artifact = RouteAwareTransformerArtifact(
        node_feature_names=NODE_FEATURE_NAMES,
        edge_feature_names=EDGE_FEATURE_NAMES,
        all_station_pairs=ALL_STATION_PAIRS,
        output_station_pairs=ADJACENT_STATION_PAIRS,
        node_standardizer=node_std,
        edge_standardizer=edge_std,
        model_config=wrapper.trainable.config,
        context_mode="full_event",
        training_summary={"epochs": 0},
    )
    path = tmp_path / "physical_pair_relative_head_only.pt"
    save_relative_route_v4_head_only_artifact(
        path, wrapper.frozen_workbook64, wrapper.trainable, artifact, frozen_workbook64_sha256="test"
    )
    loaded, _ = load_relative_route_v4_head_only_artifact(path, device="cpu")
    assert loaded.trainable.config.route_representation_mode == "physical_pair_relative"
    assert loaded.trainable.config.route_correction_bound == BOUND
    # The reloaded head rebuilds the physical-pair-relative architecture.
    assert not hasattr(loaded.trainable, "route_query")
    assert loaded.trainable.route_encoder[0].in_features == RPHYS_W
    wrapper.eval()
    loaded.eval()
    with torch.no_grad():
        original = _forward_route_batch(wrapper, batch)
        restored = _forward_route_batch(loaded, batch)
    torch.testing.assert_close(original.route_logits, restored.route_logits, atol=0.0, rtol=0.0)
    assert float(restored.delta_route_logits.abs().max()) <= BOUND + 1e-6


# -----------------------------------------------------------------------------
# Data guard tests (items 15-18)
# -----------------------------------------------------------------------------
def test_fold_train_and_holdout_disjoint():
    """Item 15: each fold's train sources are disjoint from its held-out family."""
    assert_source_family_partition()
    all_folds = folds()
    assert len(all_folds) == 2
    for fold in all_folds:
        validate_fold_sources(fold["train_sources"], fold["holdout_sources"])
        assert not (set(fold["train_sources"]) & set(fold["holdout_sources"]))
        leaked = tuple(fold["train_sources"]) + (fold["holdout_sources"][0],)
        with pytest.raises(ValueError, match="leaked into training"):
            validate_fold_sources(leaked, fold["holdout_sources"])


def test_development_final_blind_sealed_forbidden():
    """Items 16/17/18: development, Final Blind and sealed sources are rejected."""
    fam1 = V5A_SOURCE_FAMILIES[FAMILY1_DSID_PAIR]
    holdout = ("mc24_100048_00100_00149",)
    for forbidden in (
        ("mc24_100047_00350_00399",),  # development
        ("mc24_100048_00350_00399",),  # development
        ("mc24_100047_00800_00849",),  # Final Blind
        ("mc24_100048_00800_00849",),  # Final Blind
        ("mc24_100116_00000_00049",),  # sealed test
    ):
        with pytest.raises(ValueError, match="development/final-blind/sealed"):
            validate_fold_sources(tuple(fam1) + forbidden, holdout)


# -----------------------------------------------------------------------------
# Numerical tests (items 19-22)
# -----------------------------------------------------------------------------
def test_cpu_finite_forward():
    """Item 19: CPU forward is finite for the physical_pair_relative head."""
    wrapper = _wrapper(BOUND, mode="physical_pair_relative", zero_init=False)
    with torch.no_grad():
        torch.nn.init.normal_(wrapper.trainable.route_score.weight, std=0.5)
    bundle, node_std, edge_std, batch = _batch_and_stds()
    wrapper.eval()
    with torch.no_grad():
        out = _forward_route_batch(wrapper, batch)
    assert torch.isfinite(out.route_logits).all()
    assert torch.isfinite(out.delta_route_logits).all()
    assert torch.isfinite(out.edge_logits).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_finite_forward():
    """Item 20: GPU forward is finite for the physical_pair_relative head."""
    device = torch.device("cuda")
    wrapper = _wrapper(BOUND, mode="physical_pair_relative", zero_init=False).to(device)
    bundle = build_transformer_graph_bundle(_toy_candidate_sets(), context_mode="full_event")
    node_std, edge_std = fit_graph_standardizers(bundle)
    batch = _make_route_batch(bundle.graphs, node_std, edge_std, device)
    wrapper.eval()
    with torch.no_grad():
        out = _forward_route_batch(wrapper, batch)
    assert torch.isfinite(out.route_logits).all()
    assert torch.isfinite(out.delta_route_logits).all()


def test_backward_gradient_enters_only_route_head():
    """Item 21: backward gradient enters only the declared route encoder/head."""
    wrapper = _wrapper(BOUND, mode="physical_pair_relative", zero_init=True)
    bundle, node_std, edge_std, batch = _batch_and_stds()
    wrapper.train()
    out = _forward_route_batch(wrapper, batch)
    out.route_logits.sum().backward()
    declared_head = ("route_encoder", "route_score")
    for name, param in wrapper.trainable.named_parameters():
        is_head = any(name == p or name.startswith(f"{p}.") for p in declared_head)
        if is_head:
            assert param.grad is not None, f"trainable head param {name} received no gradient"
        else:
            assert param.grad is None or float(param.grad.abs().max()) == 0.0, (
                f"non-head trainable param {name} received a gradient"
            )


def test_frozen_w64_zero_gradient_leakage():
    """Item 22: frozen Workbook-64 parameters receive exactly zero gradient."""
    wrapper = _wrapper(BOUND, mode="physical_pair_relative", zero_init=True)
    bundle, node_std, edge_std, batch = _batch_and_stds()
    wrapper.train()
    out = _forward_route_batch(wrapper, batch)
    out.route_logits.sum().backward()
    for name, param in wrapper.frozen_workbook64.named_parameters():
        assert param.grad is None, f"frozen W64 param {name} received a gradient"
    # Trainable head must receive gradient (bound does not block it).
    assert wrapper.trainable.route_score.weight.grad is not None


# -----------------------------------------------------------------------------
# Paired-contract test: Control (absolute) vs Primary (physical_pair_relative)
# differ only in the route representation mode.
# -----------------------------------------------------------------------------
def test_control_and_primary_differ_only_in_representation_mode():
    control = RouteAwareSparseTransformer(_absolute_config(BOUND))
    primary = RouteAwareSparseTransformer(_phys_rel_config(BOUND))
    assert control.config.route_representation_mode == "absolute"
    assert primary.config.route_representation_mode == "physical_pair_relative"
    # Same bound, same additive correction, same hidden width/depth/dropout.
    assert control.config.route_correction_bound == primary.config.route_correction_bound == BOUND
    assert control.config.use_additive_route_correction == primary.config.use_additive_route_correction
    assert control.config.route_hidden_dim == primary.config.route_hidden_dim
    # Same frozen backbone + edge path; only the trainable head input width differs.
    assert control.route_encoder[0].in_features == 1075
    assert primary.route_encoder[0].in_features == RPHYS_W
    assert control.route_encoder[0].out_features == primary.route_encoder[0].out_features
    fc = freeze_backbone_and_edge_scorer(control)
    fp = freeze_backbone_and_edge_scorer(primary)
    assert fc["n_frozen_parameters"] == fp["n_frozen_parameters"] == 614947
    assert fc["n_trainable_parameters"] == 172513
    assert fp["n_trainable_parameters"] == 21377
