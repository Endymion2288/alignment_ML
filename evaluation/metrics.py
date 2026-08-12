"""Association metrics with explicit handling of ambiguous truth labels."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from baselines.chi2_matching import Match
from datasets.root_loader import EventTracklets


@dataclass
class AssociationMetrics:
    possible_matches: int = 0
    predicted_matches: int = 0
    scored_predicted_matches: int = 0
    correct_matches: int = 0
    incorrect_scored_matches: int = 0
    unscorable_predicted_matches: int = 0
    ambiguous_truth_labels: int = 0
    events_without_truth: int = 0

    def add(self, other: "AssociationMetrics") -> None:
        self.possible_matches += other.possible_matches
        self.predicted_matches += other.predicted_matches
        self.scored_predicted_matches += other.scored_predicted_matches
        self.correct_matches += other.correct_matches
        self.incorrect_scored_matches += other.incorrect_scored_matches
        self.unscorable_predicted_matches += other.unscorable_predicted_matches
        self.ambiguous_truth_labels += other.ambiguous_truth_labels
        self.events_without_truth += other.events_without_truth

    @property
    def efficiency(self) -> float | None:
        if not self.possible_matches:
            return None
        return self.correct_matches / self.possible_matches

    @property
    def purity(self) -> float | None:
        if not self.scored_predicted_matches:
            return None
        return self.correct_matches / self.scored_predicted_matches

    @property
    def fake_rate(self) -> float | None:
        if not self.scored_predicted_matches:
            return None
        return self.incorrect_scored_matches / self.scored_predicted_matches

    @property
    def inclusive_purity(self) -> float | None:
        """Purity with fake/unknown endpoints counted as false associations."""
        if not self.predicted_matches:
            return None
        return self.correct_matches / self.predicted_matches

    @property
    def inclusive_fake_rate(self) -> float | None:
        """False-association fraction including synthetic fake/background rows."""
        if not self.predicted_matches:
            return None
        return (self.predicted_matches - self.correct_matches) / self.predicted_matches

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "possible_matches": self.possible_matches,
            "predicted_matches": self.predicted_matches,
            "scored_predicted_matches": self.scored_predicted_matches,
            "correct_matches": self.correct_matches,
            "incorrect_scored_matches": self.incorrect_scored_matches,
            "unscorable_predicted_matches": self.unscorable_predicted_matches,
            "ambiguous_truth_labels": self.ambiguous_truth_labels,
            "events_without_truth": self.events_without_truth,
            "association_efficiency": self.efficiency,
            "association_purity": self.purity,
            "fake_rate": self.fake_rate,
            "inclusive_association_purity": self.inclusive_purity,
            "inclusive_fake_rate": self.inclusive_fake_rate,
        }


@dataclass
class UnmatchedEndpointMetrics:
    """Evaluation-only accounting for dustbin decisions.

    ``missing_truth`` is restricted to known, unique truth labels whose
    counterpart is absent (rather than ambiguous) in the other station.
    Synthetic fake rows have truth ID ``-1`` and are reported separately.
    """

    source_missing_truth_endpoints: int = 0
    source_correctly_unmatched_missing_truth: int = 0
    target_missing_truth_endpoints: int = 0
    target_correctly_unmatched_missing_truth: int = 0
    source_fake_endpoints: int = 0
    source_correctly_unmatched_fakes: int = 0
    target_fake_endpoints: int = 0
    target_correctly_unmatched_fakes: int = 0
    ambiguous_truth_endpoints: int = 0
    events_without_truth: int = 0

    def add(self, other: "UnmatchedEndpointMetrics") -> None:
        self.source_missing_truth_endpoints += other.source_missing_truth_endpoints
        self.source_correctly_unmatched_missing_truth += other.source_correctly_unmatched_missing_truth
        self.target_missing_truth_endpoints += other.target_missing_truth_endpoints
        self.target_correctly_unmatched_missing_truth += other.target_correctly_unmatched_missing_truth
        self.source_fake_endpoints += other.source_fake_endpoints
        self.source_correctly_unmatched_fakes += other.source_correctly_unmatched_fakes
        self.target_fake_endpoints += other.target_fake_endpoints
        self.target_correctly_unmatched_fakes += other.target_correctly_unmatched_fakes
        self.ambiguous_truth_endpoints += other.ambiguous_truth_endpoints
        self.events_without_truth += other.events_without_truth

    @property
    def missing_truth_endpoints(self) -> int:
        return self.source_missing_truth_endpoints + self.target_missing_truth_endpoints

    @property
    def correctly_unmatched_missing_truth(self) -> int:
        return (
            self.source_correctly_unmatched_missing_truth
            + self.target_correctly_unmatched_missing_truth
        )

    @property
    def missing_truth_unmatched_recall(self) -> float | None:
        if not self.missing_truth_endpoints:
            return None
        return self.correctly_unmatched_missing_truth / self.missing_truth_endpoints

    @property
    def fake_endpoints(self) -> int:
        return self.source_fake_endpoints + self.target_fake_endpoints

    @property
    def correctly_unmatched_fakes(self) -> int:
        return self.source_correctly_unmatched_fakes + self.target_correctly_unmatched_fakes

    @property
    def fake_unmatched_recall(self) -> float | None:
        if not self.fake_endpoints:
            return None
        return self.correctly_unmatched_fakes / self.fake_endpoints

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "source_missing_truth_endpoints": self.source_missing_truth_endpoints,
            "source_correctly_unmatched_missing_truth": self.source_correctly_unmatched_missing_truth,
            "target_missing_truth_endpoints": self.target_missing_truth_endpoints,
            "target_correctly_unmatched_missing_truth": self.target_correctly_unmatched_missing_truth,
            "missing_truth_endpoints": self.missing_truth_endpoints,
            "correctly_unmatched_missing_truth": self.correctly_unmatched_missing_truth,
            "missing_truth_unmatched_recall": self.missing_truth_unmatched_recall,
            "source_fake_endpoints": self.source_fake_endpoints,
            "source_correctly_unmatched_fakes": self.source_correctly_unmatched_fakes,
            "target_fake_endpoints": self.target_fake_endpoints,
            "target_correctly_unmatched_fakes": self.target_correctly_unmatched_fakes,
            "fake_endpoints": self.fake_endpoints,
            "correctly_unmatched_fakes": self.correctly_unmatched_fakes,
            "fake_unmatched_recall": self.fake_unmatched_recall,
            "ambiguous_truth_endpoints": self.ambiguous_truth_endpoints,
            "events_without_truth": self.events_without_truth,
        }


def _unique_truth_indices(
    event: EventTracklets,
    station: int,
) -> tuple[dict[int, int], set[int]]:
    indices = event.indices_for_station(station)
    if event.truth_particle_id is None:
        return {}, set()
    truth_ids = event.truth_particle_id[indices]
    counts = Counter(int(truth_id) for truth_id in truth_ids if truth_id >= 0)
    unique = {
        int(truth_id): int(index)
        for index, truth_id in zip(indices, truth_ids)
        if truth_id >= 0 and counts[int(truth_id)] == 1
    }
    ambiguous = {truth_id for truth_id, count in counts.items() if count > 1}
    return unique, ambiguous


@dataclass(frozen=True)
class TruthMatchAssessment:
    """Truth-label interpretation for one predicted association."""

    source_truth_particle_id: int | None
    target_truth_particle_id: int | None
    relation: str
    is_scorable: bool
    is_correct: bool


def assess_event_matches(
    event: EventTracklets,
    matches: Iterable[Match],
    source_station: int,
    target_station: int,
) -> tuple[AssociationMetrics, list[TruthMatchAssessment]]:
    """Score matches while keeping unknown and ambiguous labels separate.

    Efficiency is evaluated on truth particle IDs that occur exactly once in
    both station collections. Purity and fake rate use only predictions whose
    two endpoint labels are each unique and known. This prevents an ambiguous
    reconstruction label from being reported as a physics fake.
    """
    metrics = AssociationMetrics()
    match_list = list(matches)
    if event.truth_particle_id is None:
        metrics.events_without_truth = 1
        metrics.predicted_matches = len(match_list)
        metrics.unscorable_predicted_matches = len(match_list)
        return metrics, [
            TruthMatchAssessment(
                source_truth_particle_id=None,
                target_truth_particle_id=None,
                relation="unavailable",
                is_scorable=False,
                is_correct=False,
            )
            for _ in match_list
        ]

    source_truth, source_ambiguous = _unique_truth_indices(event, source_station)
    target_truth, target_ambiguous = _unique_truth_indices(event, target_station)
    eligible = set(source_truth).intersection(target_truth)
    metrics.possible_matches = len(eligible)
    metrics.ambiguous_truth_labels = len(source_ambiguous.union(target_ambiguous))

    assessments: list[TruthMatchAssessment] = []
    for match in match_list:
        metrics.predicted_matches += 1
        source_truth_id = int(event.truth_particle_id[match.source_index])
        target_truth_id = int(event.truth_particle_id[match.target_index])

        if source_truth_id < 0 or target_truth_id < 0:
            assessment = TruthMatchAssessment(
                source_truth_particle_id=source_truth_id,
                target_truth_particle_id=target_truth_id,
                relation="unknown",
                is_scorable=False,
                is_correct=False,
            )
        elif source_truth_id in source_ambiguous or target_truth_id in target_ambiguous:
            assessment = TruthMatchAssessment(
                source_truth_particle_id=source_truth_id,
                target_truth_particle_id=target_truth_id,
                relation="ambiguous",
                is_scorable=False,
                is_correct=False,
            )
        else:
            is_correct = (
                source_truth_id == target_truth_id and source_truth_id in eligible
            )
            assessment = TruthMatchAssessment(
                source_truth_particle_id=source_truth_id,
                target_truth_particle_id=target_truth_id,
                relation="correct" if is_correct else "incorrect",
                is_scorable=True,
                is_correct=is_correct,
            )

        if assessment.is_scorable:
            metrics.scored_predicted_matches += 1
            if assessment.is_correct:
                metrics.correct_matches += 1
            else:
                metrics.incorrect_scored_matches += 1
        else:
            metrics.unscorable_predicted_matches += 1
        assessments.append(assessment)
    return metrics, assessments


def evaluate_event_matches(
    event: EventTracklets,
    matches: Iterable[Match],
    source_station: int,
    target_station: int,
) -> AssociationMetrics:
    """Evaluate associations using only unambiguous, known truth labels."""
    metrics, _ = assess_event_matches(
        event,
        matches,
        source_station=source_station,
        target_station=target_station,
    )
    return metrics


def assess_event_unmatched_endpoints(
    event: EventTracklets,
    matches: Iterable[Match],
    source_station: int,
    target_station: int,
) -> UnmatchedEndpointMetrics:
    """Measure whether true-missing and fake endpoints reached a dustbin.

    This evaluation is intentionally independent of association efficiency:
    a reconstructed true endpoint that has no counterpart due to synthetic
    missingness should be left unmatched, while a counterpart that exists but
    receives a wrong association is accounted for by ``AssociationMetrics``.
    """
    metrics = UnmatchedEndpointMetrics()
    if event.truth_particle_id is None:
        metrics.events_without_truth = 1
        return metrics
    match_list = list(matches)
    matched_sources = {int(match.source_index) for match in match_list}
    matched_targets = {int(match.target_index) for match in match_list}
    source_unique, source_ambiguous = _unique_truth_indices(event, source_station)
    target_unique, target_ambiguous = _unique_truth_indices(event, target_station)

    def inspect(
        station: int,
        own_unique: dict[int, int],
        own_ambiguous: set[int],
        opposite_unique: dict[int, int],
        opposite_ambiguous: set[int],
        matched: set[int],
        source_side: bool,
    ) -> None:
        for index in event.indices_for_station(station):
            row = int(index)
            truth_id = int(event.truth_particle_id[row])
            if truth_id < 0:
                if source_side:
                    metrics.source_fake_endpoints += 1
                    metrics.source_correctly_unmatched_fakes += int(row not in matched)
                else:
                    metrics.target_fake_endpoints += 1
                    metrics.target_correctly_unmatched_fakes += int(row not in matched)
                continue
            if truth_id in own_ambiguous or truth_id in opposite_ambiguous:
                metrics.ambiguous_truth_endpoints += 1
                continue
            # A label that is not unique in this station cannot define a
            # meaningful endpoint-level missing target.
            if own_unique.get(truth_id) != row or truth_id in opposite_unique:
                continue
            if source_side:
                metrics.source_missing_truth_endpoints += 1
                metrics.source_correctly_unmatched_missing_truth += int(row not in matched)
            else:
                metrics.target_missing_truth_endpoints += 1
                metrics.target_correctly_unmatched_missing_truth += int(row not in matched)

    inspect(
        source_station,
        source_unique,
        source_ambiguous,
        target_unique,
        target_ambiguous,
        matched_sources,
        True,
    )
    inspect(
        target_station,
        target_unique,
        target_ambiguous,
        source_unique,
        source_ambiguous,
        matched_targets,
        False,
    )
    return metrics
