"""Small supervised pair classifier used before any Transformer study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PairClassifierConfig:
    """Deliberately low-capacity MLP and deterministic training settings."""

    hidden_dim: int = 64
    epochs: int = 150
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    seed: int = 20260810
    device: str = "auto"


@dataclass(frozen=True)
class FeatureStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "FeatureStandardizer":
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or not values.shape[0] or not np.isfinite(values).all():
            raise ValueError("training features must be a non-empty finite matrix")
        mean = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
        scale = np.where(scale > 1.0e-12, scale, 1.0)
        return cls(mean=mean, scale=scale)

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean.size or not np.isfinite(values).all():
            raise ValueError("features do not match the fitted standardizer")
        return ((values - self.mean) / self.scale).astype(np.float32)


@dataclass(frozen=True)
class PairClassifierArtifact:
    """Serialized model contract required for deterministic inference."""

    feature_names: tuple[str, ...]
    station_pairs: tuple[tuple[int, int], ...]
    threshold: float
    standardizer: FeatureStandardizer
    config: PairClassifierConfig
    training_summary: dict[str, float | int]


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment-level failure
        raise RuntimeError("PyTorch is required for the MLP pair-classifier baseline") from error
    return torch


def _device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("a CUDA device was requested but CUDA is unavailable")
    return device


def _model(torch: Any, input_dim: int, hidden_dim: int) -> Any:
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, 1),
    )


def _validate_supervision(features: np.ndarray, labels: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=bool)
    if values.ndim != 2 or values.shape[0] != target.size or not values.shape[0]:
        raise ValueError(f"{name} features and labels are inconsistent or empty")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} features contain non-finite values")
    if not np.any(target) or np.all(target):
        raise ValueError(f"{name} labels must include positive and negative pairs")
    return values, target


def select_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Choose a fixed dustbin threshold from validation candidate F1."""
    values = np.asarray(scores, dtype=np.float64)
    target = np.asarray(labels, dtype=bool)
    if values.shape != target.shape or not values.size or not np.isfinite(values).all():
        raise ValueError("validation scores and labels must be aligned finite arrays")
    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        predicted = values >= threshold
        true_positive = int(np.count_nonzero(predicted & target))
        false_positive = int(np.count_nonzero(predicted & ~target))
        false_negative = int(np.count_nonzero(~predicted & target))
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = float(2 * true_positive / denominator) if denominator else 0.0
        if f1 > best_f1 or (np.isclose(f1, best_f1) and threshold > best_threshold):
            best_threshold = float(threshold)
            best_f1 = f1
    return best_threshold, best_f1


def train_pair_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    feature_names: Sequence[str],
    station_pairs: Sequence[tuple[int, int]],
    config: PairClassifierConfig,
    standardizer: FeatureStandardizer | None = None,
    initial_state_dict: Mapping[str, Any] | None = None,
    sampling_weights: np.ndarray | None = None,
) -> tuple[Any, PairClassifierArtifact]:
    """Fit the low-capacity MLP and return the model plus inference contract.

    ``standardizer`` and ``initial_state_dict`` make curriculum learning
    explicit: every stage can use the same training-only feature transform
    while continuing the weights from the preceding physical-payload stage.
    Existing callers omit both arguments and retain the original behaviour.
    """
    train_values, train_target = _validate_supervision(train_features, train_labels, "training")
    validation_values, validation_target = _validate_supervision(
        validation_features, validation_labels, "validation"
    )
    if train_values.shape[1] != validation_values.shape[1]:
        raise ValueError("training and validation feature widths differ")
    if len(feature_names) != train_values.shape[1]:
        raise ValueError("feature_names does not match feature width")
    if config.hidden_dim < 1 or config.epochs < 1 or config.batch_size < 1:
        raise ValueError("hidden_dim, epochs, and batch_size must be positive")
    if config.learning_rate <= 0.0 or config.weight_decay < 0.0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")

    torch = _torch()
    device = _device(torch, config.device)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if standardizer is None:
        standardizer = FeatureStandardizer.fit(train_values)
    elif (
        standardizer.mean.shape != (train_values.shape[1],)
        or standardizer.scale.shape != (train_values.shape[1],)
    ):
        raise ValueError("curriculum standardizer does not match feature width")
    train_tensor = torch.from_numpy(standardizer.transform(train_values))
    train_label_tensor = torch.from_numpy(train_target.astype(np.float32))
    validation_tensor = torch.from_numpy(standardizer.transform(validation_values))
    validation_label_tensor = torch.from_numpy(validation_target.astype(np.float32))
    model = _model(torch, input_dim=train_values.shape[1], hidden_dim=config.hidden_dim).to(device)
    if initial_state_dict is not None:
        model.load_state_dict(dict(initial_state_dict))
    positives = float(np.count_nonzero(train_target))
    negatives = float(train_target.size - positives)
    positive_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    sampling_tensor = None
    if sampling_weights is not None:
        raw_weights = np.asarray(sampling_weights, dtype=np.float64)
        if raw_weights.shape != (train_target.size,) or not np.isfinite(raw_weights).all():
            raise ValueError("sampling_weights must align with finite training rows")
        if np.any(raw_weights <= 0.0):
            raise ValueError("sampling_weights must be strictly positive")
        sampling_tensor = torch.from_numpy(raw_weights.astype(np.float64))
    last_loss = float("nan")
    for _ in range(config.epochs):
        model.train()
        order = (
            torch.multinomial(
                sampling_tensor,
                num_samples=train_tensor.shape[0],
                replacement=True,
                generator=generator,
            )
            if sampling_tensor is not None
            else torch.randperm(train_tensor.shape[0], generator=generator)
        )
        total_loss = 0.0
        total_rows = 0
        for start in range(0, int(order.numel()), config.batch_size):
            batch = order[start : start + config.batch_size]
            batch_features = train_tensor[batch].to(device)
            batch_labels = train_label_tensor[batch].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features).squeeze(1)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            rows = int(batch.numel())
            total_loss += float(loss.detach().cpu()) * rows
            total_rows += rows
        last_loss = total_loss / max(total_rows, 1)
    model.eval()
    with torch.no_grad():
        validation_probability = torch.sigmoid(model(validation_tensor.to(device)).squeeze(1))
    validation_scores = validation_probability.detach().cpu().numpy().astype(np.float64)
    threshold, validation_f1 = select_f1_threshold(validation_scores, validation_target)
    summary: dict[str, float | int] = {
        "training_rows": int(train_target.size),
        "training_positive_rows": int(np.count_nonzero(train_target)),
        "validation_rows": int(validation_target.size),
        "validation_positive_rows": int(np.count_nonzero(validation_target)),
        "training_final_loss": float(last_loss),
        "validation_candidate_f1": float(validation_f1),
        "weighted_training_sampling": int(sampling_tensor is not None),
        "device": str(device),
    }
    artifact = PairClassifierArtifact(
        feature_names=tuple(feature_names),
        station_pairs=tuple((int(source), int(target)) for source, target in station_pairs),
        threshold=threshold,
        standardizer=standardizer,
        config=config,
        training_summary=summary,
    )
    return model, artifact


def predict_pair_classifier(model: Any, artifact: PairClassifierArtifact, features: np.ndarray) -> np.ndarray:
    """Return deterministic sigmoid scores for a fitted MLP artifact."""
    torch = _torch()
    values = artifact.standardizer.transform(features)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(values).to(device)
        probability = torch.sigmoid(model(tensor).squeeze(1))
    return probability.detach().cpu().numpy().astype(np.float64)


def save_pair_classifier(
    path: str | Path,
    model: Any,
    artifact: PairClassifierArtifact,
) -> None:
    """Save only model state and explicit inference metadata."""
    torch = _torch()
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_names": artifact.feature_names,
            "station_pairs": artifact.station_pairs,
            "threshold": artifact.threshold,
            "standardizer_mean": artifact.standardizer.mean,
            "standardizer_scale": artifact.standardizer.scale,
            "config": asdict(artifact.config),
            "training_summary": artifact.training_summary,
        },
        destination,
    )


def load_pair_classifier(path: str | Path, device: str = "auto") -> tuple[Any, PairClassifierArtifact]:
    """Load a pair classifier saved by :func:`save_pair_classifier`."""
    torch = _torch()
    source = Path(path).expanduser().resolve()
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with older LCG PyTorch builds.
        payload = torch.load(source, map_location="cpu")
    config_values = dict(payload["config"])
    config_values["device"] = device
    config = PairClassifierConfig(**config_values)
    model = _model(torch, len(payload["feature_names"]), config.hidden_dim)
    model.load_state_dict(payload["model_state_dict"])
    model.to(_device(torch, device))
    artifact = PairClassifierArtifact(
        feature_names=tuple(payload["feature_names"]),
        station_pairs=tuple(tuple(pair) for pair in payload["station_pairs"]),
        threshold=float(payload["threshold"]),
        standardizer=FeatureStandardizer(
            mean=np.asarray(payload["standardizer_mean"], dtype=np.float64),
            scale=np.asarray(payload["standardizer_scale"], dtype=np.float64),
        ),
        config=config,
        training_summary=dict(payload["training_summary"]),
    )
    return model, artifact
