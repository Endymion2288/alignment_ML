from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from baselines.route_assignment import solve_unit_capacity_route_packing
from baselines.field_chi2_matching import FieldCandidate
from datasets.physical_curriculum import CurriculumSample
from datasets.root_loader import EventTracklets
from models.route_transformer import RouteAwareSparseTransformer, RouteAwareTransformerConfig
from training.curriculum_mlp import CandidateSet
from training.geometry_aware_transformer import (
    ALL_STATION_PAIRS,
    CurriculumStage,
    build_transformer_graph_bundle,
    fit_graph_standardizers,
)
from training.route_aware_transformer import _forward_route_batch, _make_route_batch
from training.structured_assignment import (
    StructuredAssignmentArtifact,
    StructuredAssignmentTrainingConfig,
    build_structured_assignment_target,
    load_structured_assignment_artifact,
    _optional_soft_assignment_loss,
    route_assignment_utilities,
    save_structured_assignment_artifact,
    soft_unit_capacity_assignment_surrogate,
    structured_margin_loss,
    train_structured_assignment_v3,
)
import training.structured_assignment as structured_assignment
from training.route_aware_transformer import RouteCandidateTable


def _table() -> RouteCandidateTable:
    # Row 0 is truth.  Row 1 is a mixed truth route sharing its IFT endpoint;
    # row 2 is an independent fake-ended route; rows 3/4 are duplicate truth
    # alternatives that share endpoints with one another.
    return RouteCandidateTable(
        node_indices=np.asarray(
            [[0, 1, 2, 3], [0, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14], [11, 15, 16, 17]],
            dtype=np.int64,
        ),
        score_edge_indices=np.zeros((5, 3), dtype=np.int64),
        labels=np.asarray([True, False, False, True, True]),
        fake_endpoint=np.asarray([False, False, True, False, False]),
        hard_negative=np.asarray([False, True, True, False, False]),
    )


def test_generic_unit_capacity_solver_returns_the_best_feasible_route_set():
    result = solve_unit_capacity_route_packing(
        [(0, 1), (0, 2), (3, 4), (4, 5)], np.asarray([3.0, 2.0, 4.0, 3.5])
    )
    np.testing.assert_array_equal(result.selected, np.asarray([True, False, True, False]))
    assert result.objective == pytest.approx(7.0)
    assert np.all(result.endpoint_loads <= 1)


def test_generic_solver_decomposes_disconnected_route_conflicts_exactly(monkeypatch):
    """Independent endpoint components must not be merged into one MILP."""
    import baselines.route_assignment as route_assignment

    original_milp = route_assignment.milp
    widths: list[int] = []

    def counted_milp(*args, **kwargs):
        objective = kwargs.get("c", args[0] if args else None)
        widths.append(int(np.asarray(objective).size))
        return original_milp(*args, **kwargs)

    monkeypatch.setattr(route_assignment, "milp", counted_milp)
    routes = [
        (0, 1),
        (0, 2),
        (2, 3),
        (10, 11),
        (10, 12),
        (12, 13),
        (20, 21),
    ]
    # Endpoint 2 (and likewise 12) appears at two tuple positions.  This is
    # deliberately not a station-ordered V3 component, so it exercises the
    # generic MILP fallback for each disconnected component.
    values = np.asarray([3.0, 2.0, 2.5, 4.0, 3.0, 3.5, 1.0])
    result = solve_unit_capacity_route_packing(routes, values)

    np.testing.assert_array_equal(
        result.selected, np.asarray([True, False, True, True, False, True, True])
    )
    assert result.objective == pytest.approx(14.0)
    # The two conflicting components are solved separately; the isolated
    # positive route is selected without asking SciPy to solve it.
    assert widths == [3, 3]


def test_generic_solver_selects_all_conflict_free_positive_routes_without_milp(monkeypatch):
    import baselines.route_assignment as route_assignment

    def unexpected_milp(*args, **kwargs):  # pragma: no cover - assertion is the test
        raise AssertionError("conflict-free routes should not invoke milp")

    monkeypatch.setattr(route_assignment, "milp", unexpected_milp)
    result = solve_unit_capacity_route_packing(
        [(0, 1), (2, 3), (4, 5)], np.asarray([1.0, 2.0, 3.0])
    )

    np.testing.assert_array_equal(result.selected, np.asarray([True, True, True]))
    assert result.objective == pytest.approx(6.0)


def test_generic_solver_uses_exact_ordered_hyperedge_dynamic_programme(monkeypatch):
    """The V3 four-station tuple fast path remains exact and solver-free."""
    import baselines.route_assignment as route_assignment

    def unexpected_milp(*args, **kwargs):  # pragma: no cover - assertion is the test
        raise AssertionError("ordered four-station component should use exact dynamic programming")

    monkeypatch.setattr(route_assignment, "milp", unexpected_milp)
    routes = [
        (0, 10, 20, 30),
        (0, 11, 21, 31),
        (1, 10, 22, 32),
        (1, 12, 23, 33),
    ]
    values = np.asarray([5.0, 4.0, 4.5, 3.0])
    result = solve_unit_capacity_route_packing(routes, values)

    # (0, 11, 21, 31) and (1, 10, 22, 32) are endpoint-disjoint and beat
    # every feasible choice containing the first route.
    np.testing.assert_array_equal(result.selected, np.asarray([False, True, True, False]))
    assert result.objective == pytest.approx(8.5)


def test_ordered_hyperedge_dynamic_programme_matches_brute_force_optimum():
    routes = [
        (0, 10, 20, 30),
        (0, 11, 21, 31),
        (1, 10, 22, 32),
        (1, 12, 23, 33),
        (2, 13, 20, 34),
        (2, 14, 24, 35),
    ]
    values = np.asarray([2.0, 4.0, 3.5, 3.0, 4.5, 1.0])
    result = solve_unit_capacity_route_packing(routes, values)
    brute_force = -np.inf
    for mask in range(1 << len(routes)):
        selected = [index for index in range(len(routes)) if mask & (1 << index)]
        endpoints = [endpoint for index in selected for endpoint in routes[index]]
        if len(endpoints) != len(set(endpoints)):
            continue
        brute_force = max(brute_force, float(values[selected].sum()))

    assert result.objective == pytest.approx(brute_force)
    assert np.all(result.endpoint_loads <= 1)


def test_structured_target_keeps_one_truth_route_per_conflicting_endpoint():
    target = build_structured_assignment_target(_table())
    assert target.truth_route_count == 2
    assert bool(target.endpoint_conflict[1])
    assert bool(target.fake_endpoint[2])
    assert int(np.count_nonzero(target.duplicate_truth)) == 1
    assert not np.any(target.mixed_truth & target.fake_endpoint)


def test_structured_hinge_penalizes_a_fake_competitor_and_backpropagates():
    table = _table()
    target = build_structured_assignment_target(table)
    utilities = torch.tensor([-3.0, 2.0, 2.0, 3.0, -3.0], requires_grad=True)
    config = StructuredAssignmentTrainingConfig(device="cpu", margin=1.0)
    result = structured_margin_loss(utilities, [table], [target], config)
    assert float(result["loss"].detach()) > 0.0
    assert int(result["competitor_mixed_truth_routes"]) >= 1
    result["loss"].backward()
    assert utilities.grad is not None
    assert float(utilities.grad[0]) < 0.0
    assert float(utilities.grad[1]) > 0.0


def test_soft_capacity_surrogate_is_finite_and_differentiable():
    table = _table()
    target = build_structured_assignment_target(table)
    utilities = torch.tensor([0.2, 1.3, 0.8, 0.4, 0.3], requires_grad=True)
    loss = soft_unit_capacity_assignment_surrogate(
        utilities,
        [table],
        [target],
        StructuredAssignmentTrainingConfig(device="cpu", soft_assignment_iterations=3),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert utilities.grad is not None


def test_primary_v3_does_not_materialize_zero_weight_soft_surrogate(monkeypatch):
    """The exact-margin primary objective must not pay for an unused control."""
    called = False

    def unexpected_soft(*args, **kwargs):  # pragma: no cover - assertion is the test
        nonlocal called
        called = True
        raise AssertionError("zero-weight soft surrogate should not be evaluated")

    monkeypatch.setattr(structured_assignment, "soft_unit_capacity_assignment_surrogate", unexpected_soft)
    utilities = torch.tensor([-1.0, 0.5, 0.3, 0.2, 0.1], requires_grad=True)
    table = _table()
    target = build_structured_assignment_target(table)
    result = _optional_soft_assignment_loss(
        utilities,
        [table],
        [target],
        StructuredAssignmentTrainingConfig(device="cpu", soft_assignment_weight=0.0),
    )
    result.backward()
    assert not called
    assert utilities.grad is not None


def _sample() -> CurriculumSample:
    return CurriculumSample(
        source_id="structured_train",
        source_ids=("structured_train",),
        split="train",
        payload_id="structured_nominal",
        magnitude_mm=0.0,
        direction_trial="unit",
        injected_offsets_xy_mm={},
        source_event_uids=("structured_train:1",),
        physical_event_uids=("structured_train:1",),
        physical_tracklets=Path("/tmp/structured_tracklets.root"),
        physical_propagations=Path("/tmp/structured_prop.root"),
        physical_payload_manifest=Path("/tmp/structured_payload.json"),
        synthetic_tracklets=Path("/tmp/structured_synthetic.root"),
        field_candidates=Path("/tmp/structured_candidates.root"),
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


def test_v3_route_utility_and_checkpoint_round_trip(tmp_path: Path):
    torch.manual_seed(37)
    bundle = build_transformer_graph_bundle(_candidate_sets(), context_mode="full_event")
    node_standardizer, edge_standardizer = fit_graph_standardizers(bundle)
    model = RouteAwareSparseTransformer(
        RouteAwareTransformerConfig(
            node_feature_dim=17, edge_feature_dim=11, dropout=0.0, route_dropout=0.0
        )
    ).eval()
    batch = _make_route_batch(bundle.graphs, node_standardizer, edge_standardizer, torch.device("cpu"))
    output = _forward_route_batch(model, batch)
    utilities = route_assignment_utilities(output, batch)
    assert utilities.shape == output.route_logits.shape
    from training.geometry_aware_transformer import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES

    artifact = StructuredAssignmentArtifact(
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
    checkpoint = tmp_path / "structured_v3.pt"
    save_structured_assignment_artifact(checkpoint, model, artifact)
    loaded, loaded_artifact = load_structured_assignment_artifact(checkpoint, device="cpu")
    loaded_batch = _make_route_batch(
        bundle.graphs,
        loaded_artifact.node_standardizer,
        loaded_artifact.edge_standardizer,
        torch.device("cpu"),
    )
    with torch.no_grad():
        loaded_utilities = route_assignment_utilities(
            _forward_route_batch(loaded, loaded_batch), loaded_batch
        )
    torch.testing.assert_close(utilities, loaded_utilities)


def test_structured_trainer_smoke_uses_the_margin_objective_only():
    torch.manual_seed(41)
    bundle = build_transformer_graph_bundle(_candidate_sets(), context_mode="full_event")
    model, artifact, history = train_structured_assignment_v3(
        bundle,
        bundle,
        RouteAwareTransformerConfig(
            node_feature_dim=17, edge_feature_dim=11, dropout=0.0, route_dropout=0.0
        ),
        StructuredAssignmentTrainingConfig(
            device="cpu", batch_size=1, early_stopping_patience=1, soft_assignment_weight=0.0
        ),
        (CurriculumStage(name="smoke", maximum_magnitude_mm=0.0, epochs=1),),
    )
    assert history
    assert artifact.training_summary["loss"]["independent_edge_bce_used"] is False
    assert artifact.training_summary["loss"]["independent_route_bce_used"] is False
    assert isinstance(model, RouteAwareSparseTransformer)
