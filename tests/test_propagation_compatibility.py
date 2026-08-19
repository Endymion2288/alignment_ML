"""Unit tests for the train-frozen propagation-compatibility models."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alignment.propagation_compatibility import (
    MODELS,
    fit_pair_models,
    from_json,
    score_edges,
    to_json,
)


def _edges(rng: np.random.Generator, count: int, scale: float = 1.0):
    covariances = np.tile(np.eye(4)[None], (count, 1, 1)) * rng.uniform(0.5, 2.0, (count, 1, 1))
    residuals = np.asarray(
        [rng.normal(size=4) * np.sqrt(scale) * np.sqrt(np.diag(c)) for c in covariances]
    )
    return residuals, covariances


def test_gaussian_score_is_chi2_plus_logdet() -> None:
    rng = np.random.default_rng(3)
    residuals, covariances = _edges(rng, 30)
    models = fit_pair_models("0->1", residuals, covariances)
    residual, covariance = residuals[0], covariances[0]
    chi2 = float(residual @ np.linalg.solve(covariance, residual))
    logdet = float(np.linalg.slogdet(covariance)[1])
    assert models.score("gaussian", residual, covariance) == pytest.approx(
        chi2 + logdet + 4 * math.log(2 * math.pi)
    )


def test_student_t_approaches_gaussian_at_large_nu() -> None:
    rng = np.random.default_rng(4)
    residuals, covariances = _edges(rng, 30)
    models = fit_pair_models("0->1", residuals, covariances)
    big_nu = type(models)(
        station_pair="0->1", student_t_nu=1.0e6, mixture_s_core=models.mixture_s_core,
        mixture_s_tail=models.mixture_s_tail, mixture_f_tail=models.mixture_f_tail,
        huber_k2=models.huber_k2, tukey_c2=models.tukey_c2,
        veto_thresholds=models.veto_thresholds, train_truth_count=models.train_truth_count,
    )
    gaussian = models.score("gaussian", residuals[0], covariances[0])
    student = big_nu.score("student_t", residuals[0], covariances[0])
    assert student == pytest.approx(gaussian, rel=1e-3)


def test_mixture_em_recovers_scales() -> None:
    rng = np.random.default_rng(5)
    count = 4000
    covariances = np.tile(np.eye(4)[None], (count, 1, 1))
    tail = rng.uniform(size=count) < 0.1
    scales = np.where(tail, 6.0, 0.7)
    residuals = np.asarray(
        [rng.normal(size=4) * np.sqrt(s) for s in scales]
    )
    models = fit_pair_models("0->1", residuals, covariances)
    assert models.mixture_s_core == pytest.approx(0.7, rel=0.15)
    assert models.mixture_s_tail == pytest.approx(6.0, rel=0.25)
    assert models.mixture_f_tail == pytest.approx(0.1, abs=0.04)


def test_veto_threshold_respects_train_quantile() -> None:
    rng = np.random.default_rng(6)
    residuals, covariances = _edges(rng, 500)
    models = fit_pair_models("0->1", residuals, covariances, veto_quantile=0.99)
    for model in MODELS:
        scores = np.asarray(
            [models.score(model, r, c) for r, c in zip(residuals, covariances)]
        )
        retention = float(np.mean(scores <= models.veto_thresholds[model]))
        assert retention == pytest.approx(0.99, abs=0.01)


def test_json_round_trip_and_unknown_pair_vetoed() -> None:
    rng = np.random.default_rng(7)
    residuals, covariances = _edges(rng, 60)
    models = {"0->1": fit_pair_models("0->1", residuals, covariances)}
    restored = from_json(to_json(models))
    assert restored["0->1"].student_t_nu == models["0->1"].student_t_nu
    scores = score_edges(
        restored, "gaussian", ["0->1", "1->2"], residuals[:2], covariances[:2]
    )
    assert np.isfinite(scores[0])
    assert scores[1] == np.inf


def test_fit_requires_minimum_statistics() -> None:
    rng = np.random.default_rng(8)
    residuals, covariances = _edges(rng, 5)
    with pytest.raises(ValueError, match="too few train truth edges"):
        fit_pair_models("0->1", residuals, covariances)
