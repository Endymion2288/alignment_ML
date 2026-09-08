"""Workbook-80 hybrid features and pre-registered reading rules.  No training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from baselines.field_chi2_matching import FieldCandidate
from datasets.physical_curriculum import CurriculumSample
from datasets.root_loader import EventTracklets
from models.hybrid_route_energy import (
    FEATURE_VERSION,
    HYBRID_FEATURE_DIM,
    MODEL_CONTRACT,
    HybridRouteEnergyScorer,
    hybrid_feature_names,
)
from models.explicit_route_energy import FEATURE_DIM, raw_route_utilities
from models.route_energy import canonical_route_energy
from training.curriculum_mlp import CandidateSet
from training.explicit_route_energy import (
    enumerate_contiguous_physical_routes,
    refuse_forbidden_experiment_path,
    w64_edge_logit_matrix,
    w64_route_utilities,
)
from training.geometry_aware_transformer import ALL_STATION_PAIRS, build_transformer_graph_bundle
from training.wb80_failure_mechanism import (
    event_assignment_masks,
    hybrid_feature_matrix,
    interpret_ablation,
    selected_route_rows,
    stratify_holdout,
    truth_margin_rows,
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


def test_hybrid_contract_is_not_wp4_arm_c():
    names = hybrid_feature_names()
    assert MODEL_CONTRACT == "hybrid_route_energy_c_v1"
    assert FEATURE_VERSION == "raw_physical_plus_w64_logits_v1"
    assert len(names) == HYBRID_FEATURE_DIM == FEATURE_DIM + 5
    assert names[-5:] == (
        "w64_logit_e01",
        "w64_logit_e12",
        "w64_logit_e23",
        "w64_logit_sum",
        "w64_n_edges",
    )
    model = HybridRouteEnergyScorer()
    cores = model(torch.zeros((2, HYBRID_FEATURE_DIM), dtype=torch.float32))
    assert cores.shape == (2,)
    utilities = raw_route_utilities(cores, [4, 2], unmatched_penalty=-1.0)
    assert float((utilities[0] - cores[0]).detach()) == pytest.approx(-4.0)


def test_missing_hops_are_zero_and_arm_a_is_raw_logit_sum():
    bundle = build_transformer_graph_bundle(_candidate_sets(), context_mode="full_event")
    graph = bundle.graphs[0]
    routes = enumerate_contiguous_physical_routes(graph)
    logits = {}
    for edge_index in range(graph.score_edge_source.size):
        logits[(int(graph.score_owner[edge_index]), int(graph.score_row[edge_index]))] = 8.0
    matrix = w64_edge_logit_matrix(graph, routes, logits)
    hybrid = hybrid_feature_matrix(graph, routes, logits)
    assert hybrid.shape[1] == HYBRID_FEATURE_DIM
    pair_routes = [row for route, row in zip(routes, matrix) if route.stations == (1, 2)]
    assert pair_routes
    assert pair_routes[0][0] == pytest.approx(0.0)
    assert pair_routes[0][1] == pytest.approx(8.0)
    assert pair_routes[0][2] == pytest.approx(0.0)
    utilities = w64_route_utilities(graph, routes, logits, unmatched_penalty=-1.0)
    completes = [utility for route, utility in zip(routes, utilities) if len(route.stations) == 4]
    assert completes[0] == pytest.approx(canonical_route_energy((8.0, 8.0, 8.0), unmatched_penalty=-1.0))
    hop_count_col = hybrid_feature_names().index("w64_n_edges")
    pair_hybrid = [row for route, row in zip(routes, hybrid) if route.stations == (1, 2)]
    assert pair_hybrid[0][hop_count_col] == pytest.approx(1.0)


def test_refuses_development_and_blind_paths():
    with pytest.raises(ValueError, match="refuses"):
        refuse_forbidden_experiment_path("outputs/mc24_100047_00350_00399/x.root")
    with pytest.raises(ValueError, match="refuses"):
        refuse_forbidden_experiment_path("/eos/data/mc24_100047_00800_00849/x.root")
    with pytest.raises(ValueError, match="refuses"):
        refuse_forbidden_experiment_path("mc24_100116_00000_00049")


def test_stratify_and_interpret_on_synthetic_rows():
    length_rows = [
        {
            "n_stations": 4,
            "truth_consistent": True,
            "selected": {"A": True, "B": True, "C": True},
            "near_dustbin": {"A": False, "B": True, "C": False},
        },
        {
            "n_stations": 2,
            "truth_consistent": False,
            "selected": {"A": False, "B": True, "C": False},
            "near_dustbin": {"A": False, "B": True, "C": False},
        },
    ]
    truth_rows = [
        {
            "w64_margin_bin": "comfortable",
            "selected_A": True,
            "selected_B": False,
            "selected_C": True,
            "inclusion_gap_A": 3.0,
            "inclusion_gap_B": -0.5,
            "inclusion_gap_C": 2.5,
        }
    ]
    utilities = {"A": [10.0, 8.0, 9.0], "B": [1.0, 0.0, 0.5], "C": [10.0, 8.0, 8.5]}
    audit = stratify_holdout(length_rows, truth_rows, utilities)
    assert audit["by_length"]["B"][2]["fake_selected"] == 1
    assert audit["by_length"]["A"][4]["truth_consistent_selected"] == 1
    assert audit["by_w64_margin"]["comfortable"]["selected_C"] == 1
    assert audit["complete_truth_energy_correlation"]["spearman_A_C"] == pytest.approx(1.0)

    reading = interpret_ablation(
        {
            "A": {"complete_track_efficiency": 0.84, "complete_fake_rate": 0.02},
            "B": {"complete_track_efficiency": 0.75, "complete_fake_rate": 0.15},
            "C": {"complete_track_efficiency": 0.84, "complete_fake_rate": 0.02},
        },
        {
            "A": {"complete_track_efficiency": 0.90, "complete_fake_rate": 0.01},
            "B": {"complete_track_efficiency": 0.80, "complete_fake_rate": 0.10},
            "C": {"complete_track_efficiency": 0.90, "complete_fake_rate": 0.01},
        },
        {"spearman_A_B": 0.20},
    )
    assert reading["w64_superiority_from_edge_information"] is True
    assert reading["physical_features_lack_information"] is True
    assert reading["hybrid_worth_entering"] is False
    assert reading["hybrid_not_justified_beyond_w64_sum"] is True


def test_assignment_masks_and_truth_rows_use_the_same_solver():
    bundle = build_transformer_graph_bundle(_candidate_sets(), context_mode="full_event")
    graph = bundle.graphs[0]
    routes = enumerate_contiguous_physical_routes(graph)
    logits = {}
    for edge_index in range(graph.score_edge_source.size):
        logits[(int(graph.score_owner[edge_index]), int(graph.score_row[edge_index]))] = 8.0
    utilities = {
        "A": w64_route_utilities(graph, routes, logits, unmatched_penalty=-1.0),
        "B": w64_route_utilities(graph, routes, logits, unmatched_penalty=-1.0) - 20.0,
        "C": w64_route_utilities(graph, routes, logits, unmatched_penalty=-1.0),
    }
    selected = event_assignment_masks(routes, utilities, unmatched_penalty=-1.0)
    rows = selected_route_rows(routes, utilities, selected)
    truth = truth_margin_rows(routes, utilities, selected, unmatched_penalty=-1.0)
    assert rows
    assert truth
    assert all(row["selected_A"] == row["selected_C"] for row in truth)
