from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from baselines.field_chi2_matching import FieldCandidate
from datasets.physical_curriculum import CurriculumSample
from datasets.root_loader import EventTracklets
from evaluation.pairwise_metrics import apply_platt_scaling
from training.curriculum_mlp import CandidateSet
from training.geometry_aware_transformer import ALL_STATION_PAIRS, build_transformer_graph_bundle
from training.gauge_consistent_route import (
    GaugeConsistentAuxConfig,
    gauge_twin_consistency_loss,
    identity_calibration_payload,
    normalize_aux_weights,
    packing_route_competition_loss,
    packing_utility,
    pair_gauge_twin_graphs,
    payload_gauge_role,
)
from training.route_operating_audit import solver_log_odds


def _sample(payload_id: str = "iteration_00_draw_00") -> CurriculumSample:
    return CurriculumSample(
        source_id="route_train",
        source_ids=("route_train",),
        split="train",
        payload_id=payload_id,
        magnitude_mm=1.0,
        direction_trial=payload_id,
        injected_offsets_xy_mm={},
        source_event_uids=("route_train:1",),
        physical_event_uids=("route_train:1",),
        physical_tracklets=Path("/tmp/route_tracklets.root"),
        physical_propagations=Path("/tmp/route_prop.root"),
        physical_payload_manifest=Path("/tmp/route_payload.json"),
        synthetic_tracklets=Path("/tmp/route_synthetic.root"),
        field_candidates=Path("/tmp/route_candidates.root"),
        condition_axis="four_station_relative_l2",
        condition_magnitude=1.0,
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
        origin_run_id=np.full(8, 10, dtype=np.int64),
        origin_event_id=np.full(8, 20, dtype=np.int64),
        origin_tracklet_id=np.asarray([1, 2, 1, 2, 1, 2, 1, 2], dtype=np.int64),
    )


def _candidate_sets(sample: CurriculumSample, event: EventTracklets) -> list[CandidateSet]:
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


def _graph(payload_id: str = "iteration_00_draw_00"):
    bundle = build_transformer_graph_bundle(
        _candidate_sets(_sample(payload_id), _event()), context_mode="full_event"
    )
    return bundle.graphs[0]


def _edge_row(graph, source: int, target: int) -> int:
    for row, (left, right) in enumerate(zip(graph.score_edge_source, graph.score_edge_destination)):
        if int(left) == source and int(right) == target:
            return int(row)
    raise KeyError((source, target))


def test_iteration_split_mode_accepts_transfer_validation_only():
    from scripts.prepare_multisource_multidof_iteration import iteration_split_mode

    assert (
        iteration_split_mode(
            {
                "allowed_splits": ["validation"],
                "forbidden_splits": ["train", "test"],
                "transfer_validation_only": True,
                "test_data_accessed": False,
            },
            label="transfer",
        )
        == "transfer_validation"
    )


def test_iteration_split_mode_accepts_train_only_and_reserved_blind():
    from scripts.prepare_multisource_multidof_iteration import iteration_split_mode

    assert (
        iteration_split_mode(
            {
                "allowed_splits": ["train"],
                "forbidden_splits": ["validation", "test"],
                "test_data_accessed": False,
            },
            label="train-only",
        )
        == "train_only"
    )
    assert (
        iteration_split_mode(
            {
                "allowed_splits": ["validation"],
                "forbidden_splits": ["train", "test"],
                "reserved_blind_validation_only": True,
                "test_data_accessed": False,
            },
            label="blind",
        )
        == "reserved_blind_validation"
    )


def test_payload_gauge_role_pairs_chart_and_left_se3_twin():
    assert payload_gauge_role("iteration_00_draw_00") == ("iteration_00_draw_00", "s0_sampling_chart")
    assert payload_gauge_role("iteration_00_draw_00_plus_common") == (
        "iteration_00_draw_00",
        "left_se3_control",
    )
    assert payload_gauge_role("iteration_00_reference") is None


def test_packing_utility_matches_solver_log_odds_convention():
    logits = torch.as_tensor([2.0, -1.0, 0.5])
    probabilities = torch.sigmoid(logits).tolist()
    expected = sum(solver_log_odds(value) for value in probabilities) + 4 * (-1.0)
    torch.testing.assert_close(packing_utility(logits, -1.0, 4), torch.as_tensor(expected))


def test_identity_platt_is_a_true_identity_on_probabilities():
    values = np.asarray([0.001, 0.2, 0.8, 0.999])
    payload = identity_calibration_payload()
    for parameters in payload["platt_by_station_pair"].values():
        calibrated = apply_platt_scaling(values, float(parameters["slope"]), float(parameters["intercept"]))
        np.testing.assert_allclose(calibrated, values, rtol=1.0e-6, atol=1.0e-6)


def test_three_station_suffix_beats_weak_complete_truth_and_margin_recovers():
    graph = _graph()
    logits = torch.full((graph.score_labels.size,), -8.0)
    for source, target in ((0, 2), (2, 4), (4, 6), (1, 3), (3, 5), (5, 7)):
        logits[_edge_row(graph, source, target)] = 4.0
    logits[_edge_row(graph, 0, 2)] = 0.0
    loss = packing_route_competition_loss(
        graph, logits, threshold=0.001, unmatched_penalty=-1.0, margin=1.0
    )
    assert float(loss) > 0.5
    recovered = logits.clone()
    recovered[_edge_row(graph, 0, 2)] = 3.0
    recovered_loss = packing_route_competition_loss(
        graph, recovered, threshold=0.001, unmatched_penalty=-1.0, margin=1.0
    )
    assert float(recovered_loss) < 1.0e-6


def test_max_reduction_does_not_average_an_easy_truth_route_with_a_hard_one():
    from training.gauge_consistent_route import dustbin_aware_route_margin_loss

    graph = _graph()
    logits = torch.full((graph.score_labels.size,), -8.0)
    for source, target in ((0, 2), (2, 4), (4, 6)):
        logits[_edge_row(graph, source, target)] = 4.0
    for source, target in ((1, 3), (3, 5), (5, 7)):
        logits[_edge_row(graph, source, target)] = 0.5
    mean_loss = float(
        dustbin_aware_route_margin_loss(
            graph, logits, threshold=0.001, unmatched_penalty=-1.0, margin=1.0, reduction="mean"
        )
    )
    max_loss = float(
        dustbin_aware_route_margin_loss(
            graph, logits, threshold=0.001, unmatched_penalty=-1.0, margin=1.0, reduction="max"
        )
    )
    assert max_loss > mean_loss + 1.0e-6
    assert max_loss == pytest.approx(2.0 * mean_loss)
    with pytest.raises(ValueError, match="must be 'mean' or 'max'"):
        packing_route_competition_loss(
            graph, logits, threshold=0.001, unmatched_penalty=-1.0, margin=1.0, reduction="top_k"
        )


def test_copy_frozen_workbook59_aux_weights_rejects_a_different_checkpoint():
    from scripts.train_four_station_gauge_consistent_v2 import _copy_frozen_aux_weights

    payload = {
        "objective": {
            "loss_weight_algorithm": {
                "method": "copy_frozen_workbook59_aux_weights",
                "source_checkpoint_sha256": "0" * 64,
                "gauge_twin_consistency_weight": 1.0,
                "packing_route_competition_weight": 0.07,
                "dustbin_aware_route_margin_weight": 0.05,
            }
        }
    }
    with pytest.raises(ValueError, match="workbook-59"):
        _copy_frozen_aux_weights(payload)
    copied = _copy_frozen_aux_weights(
        {
            "objective": {
                "loss_weight_algorithm": {
                    "method": "copy_frozen_workbook59_aux_weights",
                    "source_checkpoint_sha256": "6565d028e01e561105d4ce467d10d5c82e5a2435daad6a69c47a07920defb5dc",
                    "gauge_twin_consistency_weight": 1.0,
                    "packing_route_competition_weight": 0.07061055340401011,
                    "dustbin_aware_route_margin_weight": 0.05,
                }
            }
        }
    )
    assert copied == {
        "gauge_twin_consistency_weight": 1.0,
        "packing_route_competition_weight": 0.07061055340401011,
        "dustbin_aware_route_margin_weight": 0.05,
    }
    assert _copy_frozen_aux_weights({"objective": {"loss_weight_algorithm": {"method": "train_only_deterministic_mean_scale"}}}) is None


def test_gauge_consistency_matches_origin_edges_and_ignores_unpaired_fakes():
    chart = _graph("iteration_00_draw_00")
    twin = _graph("iteration_00_draw_00_plus_common")
    pairs, leftovers = pair_gauge_twin_graphs([chart, twin])
    assert leftovers == []
    assert len(pairs) == 1
    chart_logits = torch.zeros(chart.score_labels.size)
    twin_logits = torch.zeros(twin.score_labels.size)
    torch.testing.assert_close(
        gauge_twin_consistency_loss(chart, twin, chart_logits, twin_logits),
        torch.as_tensor(0.0),
    )
    shifted = chart_logits.clone()
    shifted[_edge_row(chart, 0, 2)] = 1.5
    loss = gauge_twin_consistency_loss(chart, twin, shifted, twin_logits)
    assert float(loss) > 0.0
    chart_only = EventTracklets(
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
        origin_run_id=np.full(8, 10, dtype=np.int64),
        origin_event_id=np.full(8, 20, dtype=np.int64),
        origin_tracklet_id=np.asarray([1, 99, 1, 2, 1, 2, 1, 2], dtype=np.int64),
    )
    unpaired_chart = build_transformer_graph_bundle(
        _candidate_sets(_sample("iteration_00_draw_00"), chart_only), context_mode="full_event"
    ).graphs[0]
    equal = torch.zeros(unpaired_chart.score_labels.size)
    twin_equal = torch.zeros(twin.score_labels.size)
    equal[_edge_row(unpaired_chart, 0, 3)] = 6.0
    matched = gauge_twin_consistency_loss(unpaired_chart, twin, equal, twin_equal)
    assert torch.isfinite(matched)


def test_normalize_aux_weights_is_deterministic_and_clipped():
    config = GaugeConsistentAuxConfig()
    weights = normalize_aux_weights(2.0, 0.1, 4.0, config)
    assert weights["gauge_twin_consistency_weight"] == pytest.approx(20.0)
    assert weights["packing_route_competition_weight"] == pytest.approx(0.5)
    tiny = normalize_aux_weights(1.0, 1.0e-12, 1.0e-12, config)
    assert tiny["gauge_twin_consistency_weight"] == 20.0


def test_transfer_sources_are_absent_from_historical_contracts():
    root = Path(__file__).resolve().parents[1]
    needles = ("mc24_100047_00300_00349", "mc24_100048_00300_00349")
    historical = [
        root / "configs" / "physical_curriculum_four_station_relative_association_sources.yaml",
        root / "configs" / "physical_curriculum_four_station_identifiability_sources.yaml",
        root / "configs" / "physical_curriculum_v3_expanded_trainval.yaml",
        root / "configs" / "physical_curriculum_calibration_modes_large_stats_sources.yaml",
        root / "configs" / "calibration_modes_large_stats_transfer_v1.yaml",
    ]
    for path in historical:
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            assert needle not in text, path
    transfer = (root / "configs" / "physical_curriculum_four_station_relative_transfer_sources.yaml").read_text(
        encoding="utf-8"
    )
    for needle in needles:
        assert needle in transfer


def test_transfer_curriculum_seed_differs_from_retraining_bank():
    from alignment.four_station import zero_parameter_values
    from scripts.config_loader import load_yaml_with_base
    from scripts.prepare_four_station_relative_curriculum import compile_four_station_relative_curriculum

    root = Path(__file__).resolve().parents[1]
    train = load_yaml_with_base(root / "configs" / "physical_refit_four_station_relative_association_curriculum.yaml")
    transfer = load_yaml_with_base(root / "configs" / "physical_refit_four_station_relative_transfer_curriculum.yaml")
    specs = train["physical_refit_capture_scan"]["alignment_parameter_specs"]
    zeros = zero_parameter_values(specs)
    train_compiled, _ = compile_four_station_relative_curriculum(train, iteration=0, current_values=zeros)
    transfer_compiled, transfer_contract = compile_four_station_relative_curriculum(
        transfer, iteration=0, current_values=zeros
    )
    assert (
        train["physical_refit_capture_scan"]["relative_sampling"]["seed"]
        != transfer["physical_refit_capture_scan"]["relative_sampling"]["seed"]
    )
    train_draw = next(
        point
        for point in train_compiled["physical_refit_capture_scan"]["rigid_points"]
        if point["name"] == "iteration_00_draw_00"
    )
    transfer_draw = next(
        point
        for point in transfer_compiled["physical_refit_capture_scan"]["rigid_points"]
        if point["name"] == "iteration_00_draw_00"
    )
    assert transfer_draw["relative_native_values"] != train_draw["relative_native_values"]
    assert transfer_contract["relative_curriculum"] == "15d_gauge_then_left_se3"
    assert transfer_compiled["physical_refit_capture_scan"]["require_central_finite_difference_probes"] is False
