"""Physical-payload curriculum helpers for the low-capacity pairwise MLP."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from baselines.field_chi2_matching import (
    FieldCandidate,
    candidate_feature_matrix,
    candidate_labels,
    greedy_score_match,
    pair_feature_names,
)
from baselines.mlp_pair_classifier import (
    FeatureStandardizer,
    PairClassifierArtifact,
    PairClassifierConfig,
    predict_pair_classifier,
    train_pair_classifier,
)
from datasets.physical_curriculum import CurriculumSample
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import EventTracklets, load_events
from evaluation.metrics import AssociationMetrics, assess_event_matches
from evaluation.pairwise_metrics import binary_calibration
from baselines.field_chi2_matching import build_field_candidates


@dataclass(frozen=True)
class CandidateSet:
    """All candidates for one synthetic event and ordered station pair."""

    sample: CurriculumSample
    event: EventTracklets
    station_pair: tuple[int, int]
    candidates: tuple[FieldCandidate, ...]
    features: np.ndarray
    labels: np.ndarray


def build_candidate_sets(
    samples: Sequence[CurriculumSample],
    station_pairs: Sequence[tuple[int, int]],
    chi2_gate: float | None,
    target_z_tolerance_mm: float = 1.0e-6,
    feature_set: str = "residual_v1",
    max_events_per_sample: int | None = None,
) -> list[CandidateSet]:
    """Load only same-payload synthetic/Acts pairs and build physical candidates."""
    pairs = tuple((int(source), int(target)) for source, target in station_pairs)
    if not pairs or any(source >= target for source, target in pairs):
        raise ValueError("station_pairs must contain forward station pairs")
    if len(set(pairs)) != len(pairs):
        raise ValueError("station_pairs contains duplicates")
    if max_events_per_sample is not None and max_events_per_sample < 1:
        raise ValueError("max_events_per_sample must be positive when supplied")
    result: list[CandidateSet] = []
    for sample in samples:
        events = load_events(
            sample.synthetic_tracklets,
            require_mc_labels=True,
            max_events=max_events_per_sample,
        )
        records = load_propagation_records(sample.field_candidates)
        for event in events:
            for source_station, target_station in pairs:
                candidates = tuple(
                    build_field_candidates(
                        event,
                        records,
                        source_station=source_station,
                        target_station=target_station,
                        chi2_gate=chi2_gate,
                        q_over_p_mode=0,
                        target_z_tolerance_mm=target_z_tolerance_mm,
                    )
                )
                labels = candidate_labels(event, candidates)
                result.append(
                    CandidateSet(
                        sample=sample,
                        event=event,
                        station_pair=(source_station, target_station),
                        candidates=candidates,
                        features=candidate_feature_matrix(
                            event, candidates, pairs, feature_set=feature_set
                        ),
                        labels=labels,
                    )
                )
    return result


def candidate_chi2_mask(candidate_set: CandidateSet, chi2_gate: float | None) -> np.ndarray:
    """Return the physical candidate-gate mask for one already-built set.

    Candidate construction always starts from the real mode-0 Acts records.
    Applying the gate after construction makes a gate scan inexpensive while
    preserving the identical event, endpoints, covariance and truth-free
    candidate definition used by every operating point.
    """
    if chi2_gate is None:
        return np.ones(candidate_set.labels.size, dtype=bool)
    if not np.isfinite(chi2_gate) or chi2_gate <= 0.0:
        raise ValueError("chi2_gate must be positive when supplied")
    return np.asarray(
        [float(candidate.chi2) <= float(chi2_gate) for candidate in candidate_set.candidates],
        dtype=bool,
    )


def filter_candidate_sets(
    sets: Sequence[CandidateSet], chi2_gate: float | None
) -> list[CandidateSet]:
    """Apply one physical chi2 gate without rebuilding or changing events."""
    filtered: list[CandidateSet] = []
    for candidate_set in sets:
        mask = candidate_chi2_mask(candidate_set, chi2_gate)
        filtered.append(
            replace(
                candidate_set,
                candidates=tuple(
                    candidate
                    for candidate, keep in zip(candidate_set.candidates, mask)
                    if bool(keep)
                ),
                features=candidate_set.features[mask],
                labels=candidate_set.labels[mask],
            )
        )
    return filtered


def filter_candidate_scores(
    sets: Sequence[CandidateSet],
    scores: Sequence[np.ndarray],
    chi2_gate: float | None,
) -> list[np.ndarray]:
    """Filter scores with exactly the same mask as :func:`filter_candidate_sets`."""
    if len(sets) != len(scores):
        raise ValueError("candidate sets and scores are not aligned")
    result: list[np.ndarray] = []
    for candidate_set, raw_scores in zip(sets, scores):
        values = np.asarray(raw_scores, dtype=np.float64)
        if values.shape != candidate_set.labels.shape:
            raise ValueError("score shape does not match a candidate set")
        result.append(values[candidate_chi2_mask(candidate_set, chi2_gate)])
    return result


def concatenate_candidate_sets(sets: Sequence[CandidateSet]) -> tuple[np.ndarray, np.ndarray]:
    """Return finite feature/label matrices while safely skipping empty events."""
    nonempty = [candidate_set for candidate_set in sets if candidate_set.labels.size]
    if not nonempty:
        raise ValueError("no candidates remain after the chi2 gate")
    return (
        np.concatenate([candidate_set.features for candidate_set in nonempty], axis=0),
        np.concatenate([candidate_set.labels for candidate_set in nonempty], axis=0),
    )


def payload_balanced_sampling_weights(sets: Sequence[CandidateSet]) -> np.ndarray:
    """Give every physical source/payload sample equal draw probability.

    Candidate multiplicity grows with the gate width and fake overlay content.
    Without this weight, a high-multiplicity payload would be sampled more
    often than a low-multiplicity physical payload during curriculum training.
    """
    nonempty = [candidate_set for candidate_set in sets if candidate_set.labels.size]
    if not nonempty:
        raise ValueError("no candidates are available for payload-balanced sampling")
    counts: dict[tuple[str, str], int] = {}
    for candidate_set in nonempty:
        key = (candidate_set.sample.source_id, candidate_set.sample.payload_id)
        counts[key] = counts.get(key, 0) + int(candidate_set.labels.size)
    weights = [
        np.full(
            candidate_set.labels.size,
            1.0 / counts[(candidate_set.sample.source_id, candidate_set.sample.payload_id)],
            dtype=np.float64,
        )
        for candidate_set in nonempty
    ]
    merged = np.concatenate(weights)
    # The absolute scale does not alter multinomial draws, but a unit mean
    # makes saved diagnostics easier to inspect.
    return merged / float(np.mean(merged))


def score_candidate_sets(
    model: Any,
    artifact: PairClassifierArtifact,
    sets: Sequence[CandidateSet],
) -> list[np.ndarray]:
    """Score sets in one model batch while preserving event/source boundaries."""
    nonempty_rows = [index for index, candidate_set in enumerate(sets) if candidate_set.labels.size]
    result = [np.empty(0, dtype=np.float64) for _ in sets]
    if not nonempty_rows:
        return result
    features = np.concatenate([sets[index].features for index in nonempty_rows], axis=0)
    merged = predict_pair_classifier(model, artifact, features)
    start = 0
    for index in nonempty_rows:
        size = int(sets[index].labels.size)
        result[index] = merged[start : start + size]
        start += size
    if start != merged.size:
        raise RuntimeError("batched MLP score reconstruction lost candidate rows")
    return result


def _candidate_metric_block(labels: list[np.ndarray], scores: list[np.ndarray], bins: int) -> dict[str, object]:
    rows = int(sum(values.size for values in labels))
    if not rows:
        return {
            "candidate_rows": 0,
            "positive_candidate_rows": 0,
            "roc_auc": None,
            "average_precision": None,
            "brier": None,
            "negative_log_likelihood": None,
            "expected_calibration_error": None,
            "calibration_bins": bins,
            "occupied_calibration_bins": 0,
        }
    merged_labels = np.concatenate(labels)
    merged_scores = np.concatenate(scores)
    return {
        "candidate_rows": rows,
        "positive_candidate_rows": int(np.count_nonzero(merged_labels)),
        **binary_calibration(merged_scores, merged_labels, bins=bins),
    }


def evaluate_scored_candidate_sets(
    sets: Sequence[CandidateSet],
    scores: Sequence[np.ndarray],
    score_threshold: float,
    calibration_bins: int,
) -> dict[str, object]:
    """Evaluate candidate discrimination and greedy one-to-one associations."""
    if len(sets) != len(scores):
        raise ValueError("candidate sets and scores are not aligned")
    aggregate = AssociationMetrics()
    association_by_pair: dict[tuple[int, int], AssociationMetrics] = {}
    labels_by_pair: dict[tuple[int, int], list[np.ndarray]] = {}
    scores_by_pair: dict[tuple[int, int], list[np.ndarray]] = {}
    possible_by_pair: dict[tuple[int, int], int] = {}
    positive_by_pair: dict[tuple[int, int], int] = {}
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    for candidate_set, values in zip(sets, scores):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != candidate_set.labels.shape:
            raise ValueError("score shape does not match a candidate set")
        pair = candidate_set.station_pair
        pair_metrics = association_by_pair.setdefault(pair, AssociationMetrics())
        labels_by_pair.setdefault(pair, []).append(candidate_set.labels)
        scores_by_pair.setdefault(pair, []).append(values)
        possible_by_pair.setdefault(pair, 0)
        positive_by_pair.setdefault(pair, 0)
        # The no-match evaluation supplies the truth-pair denominator even
        # when the candidate gate removed every physical row.
        available, _ = assess_event_matches(
            candidate_set.event, [], source_station=pair[0], target_station=pair[1]
        )
        possible_by_pair[pair] += available.possible_matches
        positive_by_pair[pair] += int(np.count_nonzero(candidate_set.labels))
        matches = greedy_score_match(candidate_set.candidates, values, score_threshold)
        event_metrics, _ = assess_event_matches(
            candidate_set.event, matches, source_station=pair[0], target_station=pair[1]
        )
        aggregate.add(event_metrics)
        pair_metrics.add(event_metrics)
        all_labels.append(candidate_set.labels)
        all_scores.append(values)

    def decorate(pair: tuple[int, int], metrics: AssociationMetrics) -> dict[str, object]:
        candidate = _candidate_metric_block(
            labels_by_pair[pair], scores_by_pair[pair], calibration_bins
        )
        possible = possible_by_pair[pair]
        positives = positive_by_pair[pair]
        candidate["candidate_truth_recall"] = (
            None if not possible else float(positives / possible)
        )
        candidate["truth_pairs_with_unique_known_endpoints"] = possible
        return {"candidate": candidate, "association": metrics.as_dict()}

    by_pair = {
        f"{pair[0]}->{pair[1]}": decorate(pair, metrics)
        for pair, metrics in sorted(association_by_pair.items())
    }
    all_candidate = _candidate_metric_block(all_labels, all_scores, calibration_bins)
    all_possible = int(sum(possible_by_pair.values()))
    all_positive = int(sum(positive_by_pair.values()))
    all_candidate["candidate_truth_recall"] = (
        None if not all_possible else float(all_positive / all_possible)
    )
    all_candidate["truth_pairs_with_unique_known_endpoints"] = all_possible
    return {
        "candidate": all_candidate,
        "association": aggregate.as_dict(),
        "by_station_pair": by_pair,
    }


def threshold_scan(
    sets: Sequence[CandidateSet],
    scores: Sequence[np.ndarray],
    thresholds: Sequence[float],
    calibration_bins: int,
) -> list[dict[str, object]]:
    """Measure threshold-dependent one-to-one performance on validation only."""
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        value = float(threshold)
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("score thresholds must be finite values in [0, 1]")
        evaluation = evaluate_scored_candidate_sets(sets, scores, value, calibration_bins)
        association = evaluation["association"]
        candidate = evaluation["candidate"]
        rows.append(
            {
                "score_threshold": value,
                "association_efficiency": association["association_efficiency"],
                "inclusive_association_purity": association["inclusive_association_purity"],
                "inclusive_fake_rate": association["inclusive_fake_rate"],
                "predicted_matches": association["predicted_matches"],
                "correct_matches": association["correct_matches"],
                "candidate_truth_recall": candidate["candidate_truth_recall"],
            }
        )
    return rows


def adaptive_threshold_grid(
    scores: Sequence[np.ndarray],
    configured_thresholds: Sequence[float],
    quantiles: int = 101,
) -> list[float]:
    """Augment a declared threshold grid with validation-score quantiles.

    Temperature scaling can compress otherwise useful scores into a narrow
    interval.  A coarse fixed grid would then miss the high-score tail and
    falsely report that a requested fake-rate operating point is unavailable.
    This function uses validation scores only and keeps every configured
    threshold, so it changes scan resolution rather than the data split or
    decision rule.
    """
    if quantiles < 3:
        raise ValueError("adaptive threshold scan needs at least three quantiles")
    base = np.asarray([float(value) for value in configured_thresholds], dtype=np.float64)
    if base.ndim != 1 or not base.size or not np.isfinite(base).all() or np.any((base < 0.0) | (base > 1.0)):
        raise ValueError("configured thresholds must be finite values in [0, 1]")
    parts = [np.asarray(values, dtype=np.float64) for values in scores if values.size]
    if not parts:
        return sorted({float(value) for value in base})
    merged = np.concatenate(parts)
    if not np.isfinite(merged).all() or np.any((merged < 0.0) | (merged > 1.0)):
        raise ValueError("adaptive threshold scores must be finite probabilities in [0, 1]")
    values = np.quantile(merged, np.linspace(0.0, 1.0, quantiles))
    # A threshold just above the highest validation score explicitly captures
    # the no-match/dustbin point when the maximum is below one.
    maximum = float(np.max(merged))
    if maximum < 1.0:
        values = np.concatenate([values, np.asarray([np.nextafter(maximum, 1.0)])])
    return sorted({float(value) for value in np.concatenate([base, values])})


def choose_operating_threshold(
    rows: Sequence[Mapping[str, object]],
    constraint: str,
    target: float,
) -> dict[str, object] | None:
    """Choose max-efficiency validation threshold under one declared constraint."""
    if constraint not in {"inclusive_fake_rate", "inclusive_association_purity"}:
        raise ValueError("unsupported operating-point constraint")
    if not np.isfinite(target) or target < 0.0 or target > 1.0:
        raise ValueError("operating-point target must be in [0, 1]")
    passing: list[Mapping[str, object]] = []
    for row in rows:
        efficiency = row.get("association_efficiency")
        value = row.get(constraint)
        predicted = row.get("predicted_matches")
        if efficiency is None or value is None or not int(predicted or 0):
            continue
        if constraint == "inclusive_fake_rate" and float(value) <= target:
            passing.append(row)
        if constraint == "inclusive_association_purity" and float(value) >= target:
            passing.append(row)
    if not passing:
        return None
    selected = max(
        passing,
        key=lambda row: (float(row["association_efficiency"]), float(row["score_threshold"])),
    )
    return dict(selected)


def train_curriculum_pair_classifier(
    train_sets: Sequence[CandidateSet],
    validation_sets: Sequence[CandidateSet],
    station_pairs: Sequence[tuple[int, int]],
    stages: Sequence[Mapping[str, object]],
    config: PairClassifierConfig,
    feature_set: str = "residual_v1",
) -> tuple[Any, PairClassifierArtifact, list[dict[str, object]]]:
    """Continue one MLP across progressively larger physical payload ranges."""
    all_train_features, _ = concatenate_candidate_sets(train_sets)
    standardizer = FeatureStandardizer.fit(all_train_features)
    validation_features, validation_labels = concatenate_candidate_sets(validation_sets)
    state_dict: Mapping[str, Any] | None = None
    model: Any | None = None
    artifact: PairClassifierArtifact | None = None
    summary: list[dict[str, object]] = []
    previous_limit = -np.inf
    for stage_index, stage in enumerate(stages):
        name = str(stage["name"])
        if "maximum_condition_magnitude" in stage:
            maximum = float(stage["maximum_condition_magnitude"])
        else:
            # Legacy translation configurations retain this key.
            maximum = float(stage["maximum_magnitude_mm"])
        if maximum < previous_limit:
            raise ValueError("curriculum stages must have non-decreasing magnitudes")
        previous_limit = maximum
        selected = [
            candidate_set
            for candidate_set in train_sets
            if candidate_set.sample.curriculum_magnitude <= maximum
        ]
        features, labels = concatenate_candidate_sets(selected)
        sampling_weights = payload_balanced_sampling_weights(selected)
        stage_config = replace(
            config,
            epochs=int(stage["epochs"]),
            seed=int(config.seed) + stage_index,
        )
        model, artifact = train_pair_classifier(
            features,
            labels,
            validation_features,
            validation_labels,
            feature_names=pair_feature_names(station_pairs, feature_set=feature_set),
            station_pairs=station_pairs,
            config=stage_config,
            standardizer=standardizer,
            initial_state_dict=state_dict,
            sampling_weights=sampling_weights,
        )
        state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        summary.append(
            {
                "name": name,
                "maximum_condition_magnitude": maximum,
                "epochs": stage_config.epochs,
                "training_candidate_rows": int(labels.size),
                "training_positive_candidate_rows": int(np.count_nonzero(labels)),
                "payload_balanced_sampling": True,
                "payload_ids": sorted({candidate_set.sample.payload_id for candidate_set in selected}),
                "source_ids": sorted({candidate_set.sample.source_id for candidate_set in selected}),
                **artifact.training_summary,
            }
        )
    if model is None or artifact is None:
        raise ValueError("curriculum requires at least one stage")
    return model, artifact, summary


def station_pair_feature_view(
    candidate_sets: Sequence[CandidateSet],
    station_pair: tuple[int, int],
    all_station_pairs: Sequence[tuple[int, int]],
    feature_set: str = "residual_v1",
) -> list[CandidateSet]:
    """Select one station pair and remove irrelevant pair-indicator columns.

    The shared MLP receives a six-way station-pair one-hot suffix.  A
    station-pair-specialised MLP instead keeps the common physical features
    plus its own single indicator, so it has the same feature contract at
    train and inference time without exposing provenance or truth labels.
    """
    pair = (int(station_pair[0]), int(station_pair[1]))
    pairs = tuple((int(source), int(target)) for source, target in all_station_pairs)
    try:
        pair_column = pairs.index(pair)
    except ValueError as error:
        raise ValueError("station_pair is absent from all_station_pairs") from error
    common_width = len(pair_feature_names(pairs, feature_set=feature_set)) - len(pairs)
    result: list[CandidateSet] = []
    for candidate_set in candidate_sets:
        if candidate_set.station_pair != pair:
            continue
        if candidate_set.features.ndim != 2 or candidate_set.features.shape[1] != len(
            pair_feature_names(pairs, feature_set=feature_set)
        ):
            raise ValueError("candidate feature width does not match all_station_pairs")
        own_indicator = candidate_set.features[:, common_width + pair_column : common_width + pair_column + 1]
        result.append(
            replace(
                candidate_set,
                features=np.concatenate([candidate_set.features[:, :common_width], own_indicator], axis=1),
            )
        )
    if not result:
        raise ValueError(f"no candidate sets are available for station pair {pair}")
    return result


def train_station_pair_ensemble(
    train_sets: Sequence[CandidateSet],
    validation_sets: Sequence[CandidateSet],
    station_pairs: Sequence[tuple[int, int]],
    stages: Sequence[Mapping[str, object]],
    config: PairClassifierConfig,
    feature_set: str = "residual_v1",
) -> tuple[
    dict[tuple[int, int], Any],
    dict[tuple[int, int], PairClassifierArtifact],
    dict[tuple[int, int], list[dict[str, object]]],
]:
    """Train independent low-capacity MLPs for each physical station pair."""
    models: dict[tuple[int, int], Any] = {}
    artifacts: dict[tuple[int, int], PairClassifierArtifact] = {}
    summaries: dict[tuple[int, int], list[dict[str, object]]] = {}
    pairs = tuple((int(source), int(target)) for source, target in station_pairs)
    for pair_index, pair in enumerate(pairs):
        pair_train = station_pair_feature_view(
            train_sets, pair, pairs, feature_set=feature_set
        )
        pair_validation = station_pair_feature_view(
            validation_sets, pair, pairs, feature_set=feature_set
        )
        pair_config = replace(config, seed=int(config.seed) + 10_000 * pair_index)
        model, artifact, summary = train_curriculum_pair_classifier(
            pair_train,
            pair_validation,
            station_pairs=(pair,),
            stages=stages,
            config=pair_config,
            feature_set=feature_set,
        )
        models[pair] = model
        artifacts[pair] = artifact
        summaries[pair] = summary
    return models, artifacts, summaries


def score_station_pair_ensemble(
    models: Mapping[tuple[int, int], Any],
    artifacts: Mapping[tuple[int, int], PairClassifierArtifact],
    sets: Sequence[CandidateSet],
    station_pairs: Sequence[tuple[int, int]],
    feature_set: str = "residual_v1",
) -> list[np.ndarray]:
    """Score each candidate set with its corresponding specialised MLP."""
    pairs = tuple((int(source), int(target)) for source, target in station_pairs)
    result = [np.empty(0, dtype=np.float64) for _ in sets]
    for pair in pairs:
        try:
            model = models[pair]
            artifact = artifacts[pair]
        except KeyError as error:
            raise ValueError(f"missing specialised MLP for station pair {pair}") from error
        indices = [index for index, candidate_set in enumerate(sets) if candidate_set.station_pair == pair]
        if not indices:
            continue
        views = station_pair_feature_view(
            [sets[index] for index in indices], pair, pairs, feature_set=feature_set
        )
        values = score_candidate_sets(model, artifact, views)
        if len(values) != len(indices):  # pragma: no cover - score helper invariant
            raise RuntimeError("station-pair ensemble score reconstruction lost candidate sets")
        for index, score in zip(indices, values):
            result[index] = score
    if any(values.size != candidate_set.labels.size for candidate_set, values in zip(sets, result)):
        raise RuntimeError("station-pair ensemble score shape does not match candidate rows")
    return result
