"""Train-frozen propagation-compatibility likelihood models.

The mode-0 combined covariance makes the plain Mahalanobis chi2 unable to
separate truth-edge heavy tails from wrong edges (a mis-associated edge with
~150 mm residuals scores chi2 ~ 12.8).  This module implements auditable
alternative statistical models of the residual r under the per-edge combined
covariance C, all fitted on train truth edges ONLY and frozen to JSON:

- ``gaussian``: baseline, score = -2 log N(r; 0, C) = chi2 + log|C| + const.
- ``student_t``: multivariate Student-t with per-pair degrees of freedom nu
  fitted by grid MLE on train truth edges; score = -2 log t_nu(r; 0, C).
- ``mixture_core_tail``: two-component scaled-Gaussian mixture
  p = (1-f) N(r; 0, s_core C) + f N(r; 0, s_tail C) with per-pair
  (s_core, s_tail, f) fitted by EM on train truth edges.
- ``huber`` / ``tukey``: robust rho of sqrt(chi2) with the robustness scale
  frozen at a train-truth chi2 quantile per pair; score = 2 rho.

Scores are compatibility scores only: they never modify the covariance used
by the alignment WLS solve, and they never change the candidate endpoint set
or the V2 route scores.  A veto threshold, when used, is the train-truth
score quantile frozen per station pair.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "faser-propagation-compatibility-v1"
MODELS = ("gaussian", "student_t", "mixture_core_tail", "huber", "tukey")
NU_GRID = (3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 20.0, 40.0, 200.0)
DIM = 4


def _logdet(covariance: np.ndarray) -> float:
    sign, value = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("combined covariance is not positive definite")
    return float(value)


def _chi2(residual: np.ndarray, covariance: np.ndarray) -> float:
    return float(residual @ np.linalg.solve(covariance, residual))


def _gaussian_nll(chi2: float, logdet: float) -> float:
    return chi2 + logdet + DIM * math.log(2.0 * math.pi)


def _student_t_nll(chi2: float, logdet: float, nu: float) -> float:
    return (
        (nu + DIM) * math.log1p(chi2 / nu)
        + logdet
        + DIM * math.log(nu * math.pi)
        + 2.0 * math.lgamma(nu / 2.0)
        - 2.0 * math.lgamma((nu + DIM) / 2.0)
    )


def _mixture_nll(chi2: float, logdet: float, s_core: float, s_tail: float, f_tail: float) -> float:
    def _component(scale: float) -> float:
        return -0.5 * (chi2 / scale + DIM * math.log(scale) + logdet + DIM * math.log(2.0 * math.pi))

    core = _component(s_core) + math.log1p(-f_tail)
    tail = _component(s_tail) + math.log(f_tail)
    return -2.0 * float(np.logaddexp(core, tail))


def _huber_score(chi2: float, k2: float) -> float:
    root = math.sqrt(chi2)
    k = math.sqrt(k2)
    if root <= k:
        return chi2
    return 2.0 * k * root - k2


def _tukey_score(chi2: float, c2: float) -> float:
    if chi2 >= c2:
        return c2 / 3.0
    return (c2 / 3.0) * (1.0 - (1.0 - chi2 / c2) ** 3)


@dataclass(frozen=True)
class PairModels:
    """Frozen per-station-pair model parameters and veto thresholds."""

    station_pair: str
    student_t_nu: float
    mixture_s_core: float
    mixture_s_tail: float
    mixture_f_tail: float
    huber_k2: float
    tukey_c2: float
    veto_thresholds: dict[str, float]
    train_truth_count: int

    def score(self, model: str, residual: np.ndarray, covariance: np.ndarray) -> float:
        chi2 = _chi2(residual, covariance)
        logdet = _logdet(covariance)
        if model == "gaussian":
            return _gaussian_nll(chi2, logdet)
        if model == "student_t":
            return _student_t_nll(chi2, logdet, self.student_t_nu)
        if model == "mixture_core_tail":
            return _mixture_nll(
                chi2, logdet, self.mixture_s_core, self.mixture_s_tail, self.mixture_f_tail
            )
        if model == "huber":
            return _huber_score(chi2, self.huber_k2)
        if model == "tukey":
            return _tukey_score(chi2, self.tukey_c2)
        raise ValueError(f"unknown compatibility model '{model}'")


def _fit_student_t_nu(chi2: np.ndarray) -> float:
    best_nu, best_nll = NU_GRID[0], np.inf
    for nu in NU_GRID:
        nll = float(np.sum((nu + DIM) * np.log1p(chi2 / nu))) + chi2.size * (
            DIM * math.log(nu * math.pi)
            + 2.0 * math.lgamma(nu / 2.0)
            - 2.0 * math.lgamma((nu + DIM) / 2.0)
        )
        if nll < best_nll:
            best_nu, best_nll = nu, nll
    return float(best_nu)


def _fit_mixture(chi2: np.ndarray) -> tuple[float, float, float]:
    """EM for p(chi2) = (1-f) s_core chi2_4(chi2/s_core) + f s_tail chi2_4(...)."""
    s_core, s_tail, f_tail = 0.5, 8.0, 0.05

    def _log_component(values: np.ndarray, scale: float) -> np.ndarray:
        # -2 log N(r;0,sC) up to the shared logdet/const; chi2_4 density in values.
        return -0.5 * (values / scale + DIM * math.log(scale))

    for _iteration in range(200):
        core = _log_component(chi2, s_core) + math.log1p(-f_tail)
        tail = _log_component(chi2, s_tail) + math.log(f_tail)
        norm = np.logaddexp(core, tail)
        w_tail = np.exp(tail - norm)
        f_new = float(np.mean(w_tail))
        s_core_new = float(np.sum((1.0 - w_tail) * chi2) / (DIM * np.sum(1.0 - w_tail)))
        s_tail_new = float(np.sum(w_tail * chi2) / (DIM * max(np.sum(w_tail), 1.0e-12)))
        if (
            abs(s_core_new - s_core) < 1e-10
            and abs(s_tail_new - s_tail) < 1e-10
            and abs(f_new - f_tail) < 1e-10
        ):
            s_core, s_tail, f_tail = s_core_new, s_tail_new, f_new
            break
        s_core, s_tail, f_tail = s_core_new, s_tail_new, f_new
    if s_tail <= s_core:
        s_tail = 4.0 * s_core
    return s_core, s_tail, f_tail


def fit_pair_models(
    station_pair: str,
    truth_residuals: np.ndarray,
    truth_covariances: np.ndarray,
    *,
    veto_quantile: float = 0.995,
) -> PairModels:
    """Fit every model on one station pair's train truth edges and freeze."""
    if truth_residuals.shape[0] < 20:
        raise ValueError(f"station pair {station_pair} has too few train truth edges")
    chi2 = np.asarray(
        [_chi2(r, c) for r, c in zip(truth_residuals, truth_covariances)], dtype=np.float64
    )
    nu = _fit_student_t_nu(chi2)
    s_core, s_tail, f_tail = _fit_mixture(chi2)
    huber_k2 = float(np.quantile(chi2, 0.95))
    tukey_c2 = float(np.quantile(chi2, 0.99))

    provisional = PairModels(
        station_pair=station_pair,
        student_t_nu=nu,
        mixture_s_core=s_core,
        mixture_s_tail=s_tail,
        mixture_f_tail=f_tail,
        huber_k2=huber_k2,
        tukey_c2=tukey_c2,
        veto_thresholds={},
        train_truth_count=int(chi2.size),
    )
    thresholds = {}
    for model in MODELS:
        scores = np.asarray(
            [provisional.score(model, r, c) for r, c in zip(truth_residuals, truth_covariances)]
        )
        thresholds[model] = float(np.quantile(scores, veto_quantile))
    return PairModels(
        station_pair=station_pair,
        student_t_nu=nu,
        mixture_s_core=s_core,
        mixture_s_tail=s_tail,
        mixture_f_tail=f_tail,
        huber_k2=huber_k2,
        tukey_c2=tukey_c2,
        veto_thresholds=thresholds,
        train_truth_count=int(chi2.size),
    )


def to_json(models: Mapping[str, PairModels]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "models": MODELS,
        "pairs": {
            pair: {
                "student_t_nu": model.student_t_nu,
                "mixture_s_core": model.mixture_s_core,
                "mixture_s_tail": model.mixture_s_tail,
                "mixture_f_tail": model.mixture_f_tail,
                "huber_k2": model.huber_k2,
                "tukey_c2": model.tukey_c2,
                "veto_thresholds": dict(model.veto_thresholds),
                "train_truth_count": model.train_truth_count,
            }
            for pair, model in sorted(models.items())
        },
        "provenance": {
            "fit_data": "train split truth edges only",
            "validation_usage": "single frozen transfer evaluation",
            "test_data_accessed": False,
        },
    }


def from_json(payload: Mapping[str, Any]) -> dict[str, PairModels]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("compatibility model schema mismatch")
    models = {}
    for pair, data in payload["pairs"].items():
        models[str(pair)] = PairModels(
            station_pair=str(pair),
            student_t_nu=float(data["student_t_nu"]),
            mixture_s_core=float(data["mixture_s_core"]),
            mixture_s_tail=float(data["mixture_s_tail"]),
            mixture_f_tail=float(data["mixture_f_tail"]),
            huber_k2=float(data["huber_k2"]),
            tukey_c2=float(data["tukey_c2"]),
            veto_thresholds={str(k): float(v) for k, v in data["veto_thresholds"].items()},
            train_truth_count=int(data["train_truth_count"]),
        )
    return models


def save(models: Mapping[str, PairModels], path: str | Path) -> None:
    target = Path(path)
    target.write_text(json.dumps(to_json(models), indent=2) + "\n", encoding="utf-8")


def load(path: str | Path) -> dict[str, PairModels]:
    return from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def score_edges(
    models: Mapping[str, PairModels],
    model: str,
    station_pairs: Sequence[str],
    residuals: np.ndarray,
    covariances: np.ndarray,
) -> np.ndarray:
    """Score edges; pairs absent from the frozen models score +inf (vetoed)."""
    scores = np.full(residuals.shape[0], np.inf)
    for index, pair in enumerate(station_pairs):
        pair_models = models.get(str(pair))
        if pair_models is None:
            continue
        scores[index] = pair_models.score(
            model, np.asarray(residuals[index]), np.asarray(covariances[index])
        )
    return scores
