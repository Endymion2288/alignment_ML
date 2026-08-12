"""Deterministic candidate-score and calibration metrics for pair baselines.

The implementation is deliberately NumPy-only so reported AUC, PR and
calibration values do not acquire an undeclared sklearn dependency.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


def _scores_and_labels(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    target = np.asarray(labels, dtype=bool)
    if values.ndim != 1 or target.ndim != 1 or values.shape != target.shape:
        raise ValueError("scores and labels must be aligned one-dimensional arrays")
    if not values.size or not np.isfinite(values).all():
        raise ValueError("scores must be non-empty and finite")
    return values, target


def binary_roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Rank-based ROC AUC with average ranks for tied scores."""
    values, target = _scores_and_labels(scores, labels)
    positives = int(np.count_nonzero(target))
    negatives = int(target.size - positives)
    if not positives or not negatives:
        return None
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    positive_rank_sum = float(np.sum(ranks[target]))
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def binary_average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Average precision, integrating the precision-recall step function."""
    values, target = _scores_and_labels(scores, labels)
    positives = int(np.count_nonzero(target))
    if not positives:
        return None
    order = np.argsort(-values, kind="mergesort")
    sorted_values = values[order]
    sorted_target = target[order]
    true_positive = 0
    seen = 0
    previous_recall = 0.0
    area = 0.0
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        group = sorted_target[start:end]
        true_positive += int(np.count_nonzero(group))
        seen += int(group.size)
        recall = true_positive / positives
        precision = true_positive / seen
        area += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return float(area)


def probability_to_logit(probabilities: np.ndarray) -> np.ndarray:
    """Return finite logits from finite probabilities without endpoint overflow."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("probabilities must be a non-empty finite one-dimensional array")
    clipped = np.clip(values, 1.0e-7, 1.0 - 1.0e-7)
    return np.log(clipped) - np.log1p(-clipped)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    clipped = np.clip(values, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    return sigmoid(probability_to_logit(probabilities) / temperature)


def apply_platt_scaling(probabilities: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Apply a monotonic validation-fitted Platt calibration map.

    The association networks use class-weighted losses because true physical
    edges are sparse.  A temperature alone cannot compensate the resulting
    prior-logit shift.  A positive-slope affine map in logit space can, while
    preserving the edge ranking used by AUC/PR and introducing no event-level
    or truth-dependent feature at inference time.
    """
    if not np.isfinite(slope) or slope <= 0.0 or not np.isfinite(intercept):
        raise ValueError("Platt slope must be finite and positive and intercept finite")
    return sigmoid(float(slope) * probability_to_logit(probabilities) + float(intercept))


def binary_calibration(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int = 15,
) -> dict[str, float | int | None]:
    """Report Brier/NLL and equal-width expected calibration error."""
    values, target = _scores_and_labels(probabilities, labels)
    if bins < 1:
        raise ValueError("bins must be positive")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("calibration requires probabilities in [0, 1]")
    truth = target.astype(np.float64)
    clipped = np.clip(values, 1.0e-7, 1.0 - 1.0e-7)
    brier = float(np.mean((values - truth) ** 2))
    nll = float(-np.mean(truth * np.log(clipped) + (1.0 - truth) * np.log1p(-clipped)))
    indices = np.minimum((values * bins).astype(np.int64), bins - 1)
    ece = 0.0
    occupied = 0
    for index in range(bins):
        selected = indices == index
        count = int(np.count_nonzero(selected))
        if not count:
            continue
        occupied += 1
        confidence = float(np.mean(values[selected]))
        frequency = float(np.mean(truth[selected]))
        ece += count / values.size * abs(confidence - frequency)
    return {
        "rows": int(values.size),
        "positive_rows": int(np.count_nonzero(target)),
        "roc_auc": binary_roc_auc(values, target),
        "average_precision": binary_average_precision(values, target),
        "brier": brier,
        "negative_log_likelihood": nll,
        "expected_calibration_error": float(ece),
        "calibration_bins": int(bins),
        "occupied_calibration_bins": occupied,
    }


def fit_temperature(
    probabilities: np.ndarray,
    labels: np.ndarray,
    minimum: float = 0.05,
    maximum: float = 20.0,
    grid_size: int = 1001,
) -> float:
    """Fit one validation-only scalar temperature by a deterministic grid scan."""
    values, target = _scores_and_labels(probabilities, labels)
    if not np.any(target) or np.all(target):
        return 1.0
    if minimum <= 0.0 or maximum <= minimum or grid_size < 3:
        raise ValueError("invalid temperature-grid bounds")
    logits = probability_to_logit(values)
    temperatures = np.exp(np.linspace(np.log(minimum), np.log(maximum), grid_size))
    truth = target.astype(np.float64)
    losses = np.empty(temperatures.size, dtype=np.float64)
    for row, temperature in enumerate(temperatures):
        calibrated = sigmoid(logits / temperature)
        clipped = np.clip(calibrated, 1.0e-7, 1.0 - 1.0e-7)
        losses[row] = -np.mean(truth * np.log(clipped) + (1.0 - truth) * np.log1p(-clipped))
    return float(temperatures[int(np.argmin(losses))])


def fit_platt_scaling(
    probabilities: np.ndarray,
    labels: np.ndarray,
    minimum_slope: float = 1.0e-4,
    maximum_slope: float = 100.0,
) -> tuple[float, float]:
    """Fit positive-slope Platt scaling by validation negative log likelihood.

    This implementation intentionally relies only on SciPy, which is already
    required by the established unit-capacity route solver.  Parameterizing
    the slope as ``exp(log_slope)`` makes the transform strictly monotonic and
    avoids a calibration fit that reverses candidate ranking on a small split.
    """
    values, target = _scores_and_labels(probabilities, labels)
    if not np.any(target) or np.all(target):
        return 1.0, 0.0
    if (
        minimum_slope <= 0.0
        or maximum_slope <= minimum_slope
        or not np.isfinite(minimum_slope)
        or not np.isfinite(maximum_slope)
    ):
        raise ValueError("Platt slope bounds must be finite positive values")
    try:
        from scipy.optimize import minimize
    except ImportError as error:  # pragma: no cover - route solver already requires SciPy
        raise RuntimeError("SciPy is required for validation Platt calibration") from error

    logits = probability_to_logit(values)
    truth = target.astype(np.float64)
    prior = float(np.clip(np.mean(truth), 1.0e-7, 1.0 - 1.0e-7))
    # Match the observed positive prior at the initial unit slope, which is a
    # well-conditioned starting point for strongly class-weighted logits.
    initial_intercept = float(np.log(prior) - np.log1p(-prior) - np.mean(logits))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        slope = float(np.exp(parameters[0]))
        predicted_logit = slope * logits + float(parameters[1])
        # logaddexp is stable for the saturated scores produced by focal/BCE
        # training and lets the calibration be fit without ad-hoc clipping.
        loss = float(np.mean(np.logaddexp(0.0, predicted_logit) - truth * predicted_logit))
        probability = sigmoid(predicted_logit)
        residual = probability - truth
        gradient = np.asarray(
            [float(np.mean(residual * logits) * slope), float(np.mean(residual))],
            dtype=np.float64,
        )
        return loss, gradient

    result = minimize(
        fun=lambda parameters: objective(parameters)[0],
        x0=np.asarray([0.0, initial_intercept], dtype=np.float64),
        jac=lambda parameters: objective(parameters)[1],
        method="L-BFGS-B",
        bounds=(
            (float(np.log(minimum_slope)), float(np.log(maximum_slope))),
            (-40.0, 40.0),
        ),
    )
    if not result.success or result.x.shape != (2,) or not np.isfinite(result.x).all():
        raise RuntimeError(f"validation Platt calibration failed: {result.message}")
    slope = float(np.exp(result.x[0]))
    intercept = float(result.x[1])
    return slope, intercept


def calibration_report(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int = 15,
) -> tuple[np.ndarray, Mapping[str, object]]:
    """Fit temperature on validation scores and return calibrated-score evidence."""
    temperature = fit_temperature(probabilities, labels)
    calibrated = apply_temperature(probabilities, temperature)
    return calibrated, {
        "method": "temperature",
        "temperature": temperature,
        "before": binary_calibration(probabilities, labels, bins=bins),
        "after": binary_calibration(calibrated, labels, bins=bins),
        "fit_split": "validation_only",
    }


def platt_calibration_report(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int = 15,
) -> tuple[np.ndarray, Mapping[str, object]]:
    """Fit validation-only Platt scaling and return its calibration evidence."""
    slope, intercept = fit_platt_scaling(probabilities, labels)
    calibrated = apply_platt_scaling(probabilities, slope, intercept)
    return calibrated, {
        "method": "platt",
        "slope": slope,
        "intercept": intercept,
        "before": binary_calibration(probabilities, labels, bins=bins),
        "after": binary_calibration(calibrated, labels, bins=bins),
        "fit_split": "validation_only",
    }
