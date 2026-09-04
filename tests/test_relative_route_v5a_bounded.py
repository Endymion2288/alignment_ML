"""Workbook-72 Phase A audit for the V5A bounded residual head.

V5A changes exactly one thing relative to V4A (Absolute unbounded control):

    delta_bounded = B * tanh(raw_delta / B)   with B = 4.0  (solver-semantic)
    L_corrected   = L_edge_W64 + delta_bounded

These tests must all pass before any V5A training is authorized
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
    V5A_SOURCE_FAMILIES,
    assert_source_family_partition,
    folds,
    validate_fold_sources,
)

BOUND = 4.0


# -----------------------------------------------------------------------------
# Toy fixtures (same construction as the V4 Phase A tests)
# -----------------------------------------------------------------------------
def _sample() -> CurriculumSample:
    return CurriculumSample(
        source_id="v5a_test_sample",
        source_ids=("v5a_test_sample",),
        split="validation",
        payload_id="v5a_nominal",
        magnitude_mm=0.0,
        direction_trial="unit",
        injected_offsets_xy_mm={},
        source_event_uids=("v5a_test:1",),
        physical_event_uids=("v5a_test:1",),
        physical_tracklets=Path("/tmp/v5a_tracklets.root"),
        physical_propagations=Path("/tmp/v5a_prop.root"),
        physical_payload_manifest=Path("/tmp/v5a_payload.json"),
        synthetic_tracklets=Path("/tmp/v5a_synthetic.root"),
        field_candidates=Path("/tmp/v5a_candidates.root"),
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


def _absolute_config(bound: float | None) -> RouteAwareTransformerConfig:
    return RouteAwareTransformerConfig(
        node_feature_dim=17,
        edge_feature_dim=11,
        dropout=0.0,
        route_dropout=0.0,
        use_relative_route_representation=False,
        use_additive_route_correction=True,
        route_correction_bound=bound,
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


def _v5a_wrapper(bound: float | None, *, zero_init: bool = True) -> RelativeRouteV4Inference:
    frozen = _historical_v2_model()
    trainable = RouteAwareSparseTransformer(_absolute_config(bound))
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


# -----------------------------------------------------------------------------
# Test 1: delta_bounded is always within [-4, +4]
# -----------------------------------------------------------------------------
def test_bounded_delta_always_within_bound():
    model = RouteAwareSparseTransformer(_absolute_config(BOUND))
    # Drive the head with extreme raw outputs by scaling route_score weights.
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


# -----------------------------------------------------------------------------
# Test 2 + 3: zero-init gives delta == 0 and L_corrected == L_edge_W64
# -----------------------------------------------------------------------------
def test_zero_init_delta_zero_and_l_corrected_equals_edge():
    wrapper = _v5a_wrapper(BOUND, zero_init=True)
    bundle, node_std, edge_std, batch = _batch_and_stds()
    wrapper.eval()
    with torch.no_grad():
        wrap_out = _forward_route_batch(wrapper, batch)
        frozen_out = _forward_route_batch(wrapper.frozen_workbook64, batch)
    assert float(wrap_out.delta_route_logits.abs().max()) == 0.0
    production_edge_sum = frozen_out.edge_logits[batch.route_score_edge_indices].sum(dim=-1)
    torch.testing.assert_close(wrap_out.route_logits, production_edge_sum, atol=1e-6, rtol=1e-6)


# -----------------------------------------------------------------------------
# Test 4: derivative of delta_bounded wrt raw_delta at the origin == 1
# -----------------------------------------------------------------------------
def test_derivative_at_origin_is_one():
    raw = torch.zeros(8, dtype=torch.float64, requires_grad=True)
    delta = BOUND * torch.tanh(raw / BOUND)
    grad = torch.autograd.grad(delta.sum(), raw)[0]
    torch.testing.assert_close(grad, torch.ones_like(grad), atol=1e-12, rtol=1e-12)


# -----------------------------------------------------------------------------
# Test 5: frozen W64 parameters receive no gradient under the bounded head
# -----------------------------------------------------------------------------
def test_frozen_workbook64_parameters_get_no_gradient():
    wrapper = _v5a_wrapper(BOUND, zero_init=True)
    bundle, node_std, edge_std, batch = _batch_and_stds()
    wrapper.train()
    out = _forward_route_batch(wrapper, batch)
    loss = out.route_logits.sum()
    loss.backward()
    for name, param in wrapper.frozen_workbook64.named_parameters():
        assert param.grad is None, f"frozen W64 param {name} received a gradient"
    # Trainable head must receive gradient (bound does not block it).
    assert wrapper.trainable.route_score.weight.grad is not None


# -----------------------------------------------------------------------------
# Test 6: frozen edge scores unchanged by the bounded head
# -----------------------------------------------------------------------------
def test_frozen_edge_scores_unchanged():
    wrapper = _v5a_wrapper(BOUND, zero_init=False)
    # Non-trivial head so delta != 0; edges must still equal frozen W64.
    with torch.no_grad():
        torch.nn.init.ones_(wrapper.trainable.route_score.weight)
        torch.nn.init.ones_(wrapper.trainable.route_score.bias)
    bundle, node_std, edge_std, batch = _batch_and_stds()
    wrapper.eval()
    with torch.no_grad():
        wrap_out = _forward_route_batch(wrapper, batch)
        frozen_out = _forward_route_batch(wrapper.frozen_workbook64, batch)
    assert float(wrap_out.delta_route_logits.abs().max()) > 0.0
    torch.testing.assert_close(wrap_out.edge_logits, frozen_out.edge_logits, atol=0.0, rtol=0.0)


# -----------------------------------------------------------------------------
# Test 7: production solver consumes bounded L_corrected = L_edge_W64 + delta_bounded
# -----------------------------------------------------------------------------
def test_solver_consumes_bounded_l_corrected():
    wrapper = _v5a_wrapper(BOUND, zero_init=False)
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
    # route_logits consumed by the solver == L_edge_W64 + delta_bounded exactly.
    torch.testing.assert_close(wrap_out.route_logits, production_edge_sum + delta, atol=1e-6, rtol=1e-6)


# -----------------------------------------------------------------------------
# Test 8: save/load preserves the bound contract
# -----------------------------------------------------------------------------
def test_save_load_preserves_bound(tmp_path: Path):
    wrapper = _v5a_wrapper(BOUND, zero_init=False)
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
    path = tmp_path / "v5a_head_only.pt"
    save_relative_route_v4_head_only_artifact(
        path, wrapper.frozen_workbook64, wrapper.trainable, artifact, frozen_workbook64_sha256="test"
    )
    loaded, _ = load_relative_route_v4_head_only_artifact(path, device="cpu")
    # Bound survives the round trip.
    assert loaded.trainable.config.route_correction_bound == BOUND
    wrapper.eval()
    loaded.eval()
    with torch.no_grad():
        original = _forward_route_batch(wrapper, batch)
        restored = _forward_route_batch(loaded, batch)
    torch.testing.assert_close(original.route_logits, restored.route_logits, atol=0.0, rtol=0.0)
    assert float(restored.delta_route_logits.abs().max()) <= BOUND + 1e-6


# -----------------------------------------------------------------------------
# Test 9: source-holdout guard forbids held-out source entering training
# -----------------------------------------------------------------------------
def test_source_holdout_guard_forbids_leakage():
    assert_source_family_partition()
    all_folds = folds()
    assert len(all_folds) == 2
    for fold in all_folds:
        # A valid fold passes.
        validate_fold_sources(fold["train_sources"], fold["holdout_sources"])
        # Moving one held-out source into training must raise.
        leaked = tuple(fold["train_sources"]) + (fold["holdout_sources"][0],)
        with pytest.raises(ValueError, match="leaked into training"):
            validate_fold_sources(leaked, fold["holdout_sources"])


# -----------------------------------------------------------------------------
# Test 10: development sources forbidden from the CV
# -----------------------------------------------------------------------------
def test_development_sources_forbidden():
    dev = ("mc24_100047_00350_00399", "mc24_100048_00350_00399")
    fam1 = V5A_SOURCE_FAMILIES["family1_ds100043_100044"]
    with pytest.raises(ValueError, match="development/final-blind/sealed"):
        validate_fold_sources(tuple(fam1) + dev[:1], ("mc24_100048_00100_00149",))


# -----------------------------------------------------------------------------
# Test 11: Final Blind sources forbidden from the CV
# -----------------------------------------------------------------------------
def test_final_blind_forbidden():
    blind = ("mc24_100047_00800_00849", "mc24_100048_00800_00849")
    fam1 = V5A_SOURCE_FAMILIES["family1_ds100043_100044"]
    with pytest.raises(ValueError, match="development/final-blind/sealed"):
        validate_fold_sources(tuple(fam1) + blind[:1], ("mc24_100048_00100_00149",))


# -----------------------------------------------------------------------------
# Test 12: sealed-test sources forbidden from the CV
# -----------------------------------------------------------------------------
def test_sealed_test_forbidden():
    sealed = ("mc24_100116_00000_00049",)
    fam1 = V5A_SOURCE_FAMILIES["family1_ds100043_100044"]
    with pytest.raises(ValueError, match="development/final-blind/sealed"):
        validate_fold_sources(tuple(fam1) + sealed, ("mc24_100048_00100_00149",))


# -----------------------------------------------------------------------------
# Test 13: Control (unbounded) vs Primary (bounded) differ ONLY in the bound
# -----------------------------------------------------------------------------
def test_control_and_primary_differ_only_in_bound():
    control = RouteAwareSparseTransformer(_absolute_config(None))
    primary = RouteAwareSparseTransformer(_absolute_config(BOUND))
    assert control.config.route_correction_bound is None
    assert primary.config.route_correction_bound == BOUND
    # Identical parameter shapes (same head architecture).
    assert {k: tuple(v.shape) for k, v in control.state_dict().items()} == {
        k: tuple(v.shape) for k, v in primary.state_dict().items()
    }
    # Unbounded control leaves large raw deltas unbounded; bounded primary does not.
    for m in (control, primary):
        with torch.no_grad():
            m.route_score.weight.mul_(1.0e6)
            m.route_score.bias.fill_(3.0e5)
    bundle, node_std, edge_std, batch = _batch_and_stds()
    control.eval()
    primary.eval()
    with torch.no_grad():
        d_control = _forward_route_batch(control, batch).delta_route_logits
        d_primary = _forward_route_batch(primary, batch).delta_route_logits
    assert float(d_control.abs().max()) > BOUND
    assert float(d_primary.abs().max()) <= BOUND + 1e-6


# -----------------------------------------------------------------------------
# Test 14: V5A preregistered configs differ only in route_correction_bound
# -----------------------------------------------------------------------------
def test_v5a_preregistered_configs_differ_only_in_bound():
    control_path = Path("configs/v5a_absolute_unbounded_control_source_transfer.yaml")
    primary_path = Path("configs/v5a_absolute_bounded_primary_source_transfer.yaml")
    assert control_path.is_file() and primary_path.is_file()
    c_control = yaml.safe_load(control_path.read_text(encoding="utf-8"))
    c_primary = yaml.safe_load(primary_path.read_text(encoding="utf-8"))
    a_control = c_control["v5a_absolute_unbounded_control"]["architecture"]
    a_primary = c_primary["v5a_absolute_bounded_primary"]["architecture"]
    assert a_control.get("route_correction_bound") is None
    assert a_primary.get("route_correction_bound") == BOUND
    assert a_control.get("use_relative_route_representation") is False
    assert a_primary.get("use_relative_route_representation") is False
    # Everything else in the architecture is identical.
    for key in a_control:
        if key == "route_correction_bound":
            continue
        assert a_control[key] == a_primary[key], f"architecture mismatch on {key}"
