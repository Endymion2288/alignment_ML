from __future__ import annotations

import torch

from training.dustbin_aware_route_margin import (
    PACKING_MARGIN,
    attach_dustbin_aware_margin,
    audit_workbook56_packing_loss_source,
    classify_feasibility_family,
    dustbin_aware_strongest,
    recommend_next_objective,
    required_margin_to_dustbin,
    required_margin_to_fragment,
    workbook56_miner_strongest,
)
from test_gauge_consistent_route import _edge_row, _graph
from training.gauge_consistent_route import (
    GaugeConsistentAuxConfig,
    dustbin_aware_route_margin_loss,
    normalize_dustbin_aware_aux_weights,
    packing_route_competition_loss,
    scale_aux_weight,
)
from training.route_operating_audit import DUSTBIN_UTILITY


def test_workbook56_miner_keeps_negative_fragment_and_skips_dustbin():
    assert workbook56_miner_strongest([-8.0, -2.0]) == -2.0
    assert dustbin_aware_strongest([-8.0, -2.0]) == DUSTBIN_UTILITY
    assert workbook56_miner_strongest([]) == DUSTBIN_UTILITY
    assert dustbin_aware_strongest([]) == DUSTBIN_UTILITY


def test_required_margin_formulas_match_pre_registered_definition():
    assert required_margin_to_dustbin(-0.96, 1.0) == 1.96
    assert abs(required_margin_to_fragment(-0.96, -0.87, 1.0) - 1.09) < 1.0e-12


def test_ranking_correct_but_below_dustbin_is_scale_family():
    assert (
        classify_feasibility_family(
            score_retained=True,
            candidate_retained=True,
            u_truth=-0.7,
            u_best_fragment=-1.2,
            margin=1.0,
        )
        == "truth_gt_fragment_lt_dustbin"
    )


def test_truth_below_fragment_is_fragment_family_even_when_both_negative():
    assert (
        classify_feasibility_family(
            score_retained=True,
            candidate_retained=True,
            u_truth=-1.2,
            u_best_fragment=-0.4,
            margin=1.0,
        )
        == "truth_lt_fragment"
    )


def test_miner_satisfied_below_dustbin_is_objective_solver_mismatch():
    # U_truth=-0.5 beats U_frag=-2.0 by more than margin 1.0, but still < 0.
    assert workbook56_miner_strongest([-2.0]) + 1.0 <= -0.5
    assert (
        classify_feasibility_family(
            score_retained=True,
            candidate_retained=True,
            u_truth=-0.5,
            u_best_fragment=-2.0,
            margin=1.0,
        )
        == "training_margin_satisfied_inference_fails"
    )


def test_attach_records_exact_required_margin_fields():
    row = attach_dustbin_aware_margin(
        {
            "u_truth": -0.70,
            "u_best_competitor": -1.20,
            "u_dustbin": 0.0,
            "score_retained": True,
            "candidate_retained": True,
            "selected": False,
        },
        margin=1.0,
    )
    assert abs(row["required_margin_to_dustbin"] - 1.70) < 1.0e-12
    assert abs(row["required_margin_to_fragment"] - 0.50) < 1.0e-12
    assert row["miner_understates_production_gap"] is True
    assert row["feasibility_family"] == "truth_gt_fragment_lt_dustbin"


def test_packing_loss_source_omits_max_with_dustbin():
    audit = audit_workbook56_packing_loss_source()
    assert audit["uses_max_of_feasible_rival_utilities"] is True
    assert audit["explicitly_takes_max_rival_and_dustbin"] is False
    assert audit["omits_dustbin_when_negative_fragment_exists"] is True
    assert audit["registered_margin"] == PACKING_MARGIN


def test_real_packing_loss_understates_dustbin_aware_gap_on_negative_utilities():
    graph = _graph()
    logits = torch.full((graph.score_labels.size,), -8.0)
    for source, target in ((0, 2), (2, 4), (4, 6), (1, 3), (3, 5), (5, 7)):
        logits[_edge_row(graph, source, target)] = 1.2
    current = float(
        packing_route_competition_loss(
            graph, logits, threshold=0.001, unmatched_penalty=-1.0, margin=1.0
        )
    )
    # Complete truth: 3*1.2 - 4 = -0.4.  Dustbin-aware gap is 1.4.
    # Own 2-station rivals exist, so the workbook-56 miner never sees dustbin 0.
    assert current < 1.4 - 1.0e-6
    assert current > 0.0


def test_dustbin_aware_loss_includes_dustbin_when_fragments_are_negative():
    graph = _graph()
    logits = torch.full((graph.score_labels.size,), -8.0)
    for source, target in ((0, 2), (2, 4), (4, 6), (1, 3), (3, 5), (5, 7)):
        logits[_edge_row(graph, source, target)] = 1.2
    original = float(
        packing_route_competition_loss(
            graph, logits, threshold=0.001, unmatched_penalty=-1.0, margin=1.0
        )
    )
    aware = float(
        dustbin_aware_route_margin_loss(
            graph, logits, threshold=0.001, unmatched_penalty=-1.0, margin=1.0
        )
    )
    # U_truth = 3*1.2 - 4 = -0.4; production gap is 1.4.
    assert original < aware
    assert abs(aware - 1.4) < 1.0e-5


def test_zero_mean_gauge_weight_uses_fallback_not_clip_ceiling():
    config = GaugeConsistentAuxConfig(
        enable_dustbin_aware_route_margin=True,
        zero_mean_fallback_weight=1.0,
        zero_mean_threshold=1.0e-6,
    )
    assert scale_aux_weight(0.24, 0.0, config) == 1.0
    weights = normalize_dustbin_aware_aux_weights(0.24, 0.0, 3.44, 1.73, config)
    assert weights["gauge_twin_consistency_weight"] == 1.0
    assert weights["dustbin_aware_route_margin_weight"] < 20.0
    assert weights["dustbin_aware_route_margin_weight"] >= 0.05


def test_dustbin_scale_majority_recommends_dustbin_aware_objective():
    decision = recommend_next_objective(
        {
            "complete_truth_chains": 10,
            "u_truth_fraction_nonpositive": 1.0,
            "feasibility_family_counts": {
                "truth_gt_fragment_lt_dustbin": 7,
                "truth_lt_fragment": 3,
                "training_margin_satisfied_inference_fails": 0,
                "production_margin_satisfied": 0,
                "below_threshold_or_missing_candidate": 0,
            },
        }
    )
    assert decision["continue_to_15d_relative_wls"] is False
    assert decision["new_checkpoint_authorized"] is False
    assert decision["next_step"] == "design_dustbin_aware_route_margin_objective"
    assert decision["stay_on_route_competition_objective_diagnosis"] is True
