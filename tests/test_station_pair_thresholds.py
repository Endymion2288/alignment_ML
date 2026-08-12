from __future__ import annotations

from training.station_pair_thresholds import (
    coordinate_descent_thresholds,
    exact_pareto_thresholds,
    quality_rank,
)


def _evaluation(efficiency: float, fake_rate: float, purity: float) -> dict[str, object]:
    return {
        "association": {
            "association_efficiency": efficiency,
            "inclusive_fake_rate": fake_rate,
            "inclusive_association_purity": purity,
        }
    }


def test_quality_rank_prefers_constraint_satisfying_point():
    passing = quality_rank(_evaluation(0.5, 0.04, 0.96), 0.05, 0.95)
    failing = quality_rank(_evaluation(0.9, 0.08, 0.92), 0.05, 0.95)

    assert passing > failing


def test_coordinate_descent_selects_best_feasible_threshold():
    def evaluator(thresholds: dict[tuple[int, int], float]) -> dict[str, object]:
        threshold = thresholds[(0, 1)]
        if threshold == 0.1:
            return _evaluation(0.55, 0.03, 0.97)
        if threshold == 0.2:
            return _evaluation(0.70, 0.04, 0.96)
        return _evaluation(0.90, 0.08, 0.92)

    result = coordinate_descent_thresholds(
        station_pairs=((0, 1),),
        threshold_grid=(0.1, 0.2, 0.5),
        initial_thresholds={(0, 1): 0.1},
        evaluator=evaluator,
        maximum_inclusive_fake_rate=0.05,
        minimum_inclusive_purity=0.95,
        maximum_sweeps=2,
    )

    assert result.thresholds == {(0, 1): 0.2}
    assert result.evaluation["association"]["association_efficiency"] == 0.70


def test_exact_pareto_selects_global_feasible_threshold_vector():
    def evaluator(pair: tuple[int, int], threshold: float) -> dict[str, object]:
        counts = {
            ((0, 1), 0.1): (10, 10, 9),
            ((0, 1), 0.9): (10, 5, 5),
            ((1, 2), 0.1): (10, 10, 10),
            ((1, 2), 0.9): (10, 5, 5),
        }
        possible, predicted, correct = counts[(pair, threshold)]
        return {
            "association": {
                "possible_matches": possible,
                "predicted_matches": predicted,
                "correct_matches": correct,
            }
        }

    result = exact_pareto_thresholds(
        station_pairs=((0, 1), (1, 2)),
        threshold_grid=(0.1, 0.9),
        evaluator=evaluator,
        maximum_inclusive_fake_rate=0.05,
        minimum_inclusive_purity=0.95,
    )

    assert result.thresholds == {(0, 1): 0.1, (1, 2): 0.1}
    assert result.evaluation["association"]["association_efficiency"] == 0.95
    assert result.evaluation["association"]["inclusive_fake_rate"] == 0.05
