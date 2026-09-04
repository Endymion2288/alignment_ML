"""Workbook-72 V5A train-side source-transfer cross-validation partitions.

The six authorized train sources are **not** six independent physics
productions.  xAOD provenance (``physical_corpus_manifest.json`` /
``configs/physical_curriculum_four_station_diversity_train_sources.yaml``)
shows they are run-range partitions of four DSIDs, which form two
charge-symmetric DSID pairs with empirically distinct source characteristics
(Workbook 63 measured the ``100047/100048`` family as a stable, distinct
source characteristic versus the ``100043/100044`` current-train family):

* Family 1 (DSID pair 1, "current train"): DSIDs 100043 (mu-) + 100044 (mu+)
* Family 2 (DSID pair 2, "second DSID pair" / failure family): 100047 + 100048

Two run-range partitions of the same DSID (e.g. ``100043_00200_00299`` and
``100043_00300_00399``) share the same generator string and DSID and differ
only in event block, so they are the *same* physics production and must stay
in the same group.  This is why a 6-fold leave-one-source-out is invalid
(same-DSID partition leakage) and a 4-fold leave-one-DSID-out is weaker
(charge-partner process leakage).  The only scheme in which the held-out
group is a genuinely unseen physics production is 2-fold leave-one-family-out.

This module only *defines* and *guards* the partition.  It never opens
development / final-blind / sealed data.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from training.source_diversity_audit import (
    AUTHORIZED_SIX_TRAIN_SOURCES,
    DEVELOPMENT_SOURCES,
    RESERVED_BLIND_SOURCES,
    UNUSED_RESERVE_SOURCES,
    is_sealed_source,
)

# Evidence-based charge-symmetric DSID-pair families (Workbook 72 section 6).
FAMILY1_DSID_PAIR = "family1_ds100043_100044"
FAMILY2_DSID_PAIR = "family2_ds100047_100048"

V5A_SOURCE_FAMILIES: dict[str, tuple[str, ...]] = {
    FAMILY1_DSID_PAIR: (
        "mc24_100043_00200_00299",
        "mc24_100043_00300_00399",
        "mc24_100044_00200_00299",
        "mc24_100044_00300_00399",
    ),
    FAMILY2_DSID_PAIR: (
        "mc24_100047_00100_00149",
        "mc24_100048_00100_00149",
    ),
}

# Sources that must never enter this CV (development / final blind / sealed).
_FORBIDDEN_CV_SOURCES = (
    set(DEVELOPMENT_SOURCES)
    | set(RESERVED_BLIND_SOURCES)
    | set(UNUSED_RESERVE_SOURCES)
)


def source_dsid(source_id: str) -> str:
    parts = str(source_id).split("_")
    if len(parts) < 2:
        raise ValueError(f"unrecognized source id: {source_id}")
    return parts[1]


def family_for_source(source_id: str) -> str:
    """Return the DSID-pair family name for an authorized train source."""
    sid = str(source_id)
    for family, members in V5A_SOURCE_FAMILIES.items():
        if sid in members:
            return family
    raise ValueError(f"source {sid} is not an authorized V5A train source")


def assert_source_family_partition() -> None:
    """The two families must partition the six authorized train sources."""
    union = tuple(src for members in V5A_SOURCE_FAMILIES.values() for src in members)
    if set(union) != set(AUTHORIZED_SIX_TRAIN_SOURCES):
        raise ValueError(
            "V5A source families do not partition the six authorized train sources: "
            + ", ".join(sorted(union))
        )
    if len(union) != len(set(union)):
        raise ValueError("V5A source families contain a duplicate source")


def folds() -> tuple[dict[str, object], ...]:
    """Two-fold leave-one-DSID-pair-out cross-validation."""
    assert_source_family_partition()
    result = []
    for holdout_family in (FAMILY1_DSID_PAIR, FAMILY2_DSID_PAIR):
        train_families = tuple(f for f in V5A_SOURCE_FAMILIES if f != holdout_family)
        train_sources = tuple(
            src for family in train_families for src in V5A_SOURCE_FAMILIES[family]
        )
        result.append(
            {
                "holdout_family": holdout_family,
                "holdout_sources": V5A_SOURCE_FAMILIES[holdout_family],
                "train_families": train_families,
                "train_sources": train_sources,
            }
        )
    return tuple(result)


def validate_fold_sources(
    train_sources: Sequence[str],
    holdout_sources: Sequence[str],
) -> None:
    """Guard a single fold's train/holdout source partition.

    Enforces: train and holdout are disjoint; their union is exactly the six
    authorized train sources; no development / final-blind / sealed source
    appears anywhere.
    """
    train = {str(s) for s in train_sources}
    holdout = {str(s) for s in holdout_sources}
    # Absolute prohibitions first: development / final-blind / sealed sources
    # must never appear in either side of the partition, regardless of whether
    # the remaining sources still union to the authorized six.
    forbidden = (train | holdout) & _FORBIDDEN_CV_SOURCES
    sealed = {s for s in (train | holdout) if is_sealed_source(s)}
    illegal = forbidden | sealed
    if illegal:
        raise ValueError(
            "source-holdout guard: development/final-blind/sealed source present: "
            + ", ".join(sorted(illegal))
        )
    overlap = train & holdout
    if overlap:
        raise ValueError(
            "source-holdout guard: held-out source(s) leaked into training: "
            + ", ".join(sorted(overlap))
        )
    if train | holdout != set(AUTHORIZED_SIX_TRAIN_SOURCES):
        raise ValueError(
            "source-holdout guard: train ∪ holdout must equal the six authorized "
            "train sources. Got train="
            + ", ".join(sorted(train))
            + " holdout="
            + ", ".join(sorted(holdout))
        )


def filter_samples_to_sources(
    samples: Sequence,
    allowed_sources: Iterable[str],
) -> list:
    """Keep only curriculum samples whose constituent sources are all allowed.

    Used as a defensive guard on the *sample* level.  The actual per-event
    source restriction for the pooled synthetic corpus is done by regenerating
    the corpus with a restricted physical source pool (see Workbook 72 §6.4);
    this guard is a secondary check.
    """
    allowed = {str(s) for s in allowed_sources}
    kept = []
    for sample in samples:
        constituents = {str(s) for s in getattr(sample, "source_ids", ()) or ()}
        if not constituents:
            constituents = {str(getattr(sample, "source_id", ""))}
        if constituents and constituents <= allowed:
            kept.append(sample)
    return kept


__all__ = [
    "FAMILY1_DSID_PAIR",
    "FAMILY2_DSID_PAIR",
    "V5A_SOURCE_FAMILIES",
    "family_for_source",
    "source_dsid",
    "assert_source_family_partition",
    "folds",
    "validate_fold_sources",
    "filter_samples_to_sources",
]
