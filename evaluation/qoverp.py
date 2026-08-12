"""Consistency audit for optional local-tracklet q/p export fields."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from datasets.root_loader import EventTracklets


def _finite_stats(values: np.ndarray) -> dict[str, int | float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    result: dict[str, int | float | None] = {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "min": None,
        "median": None,
        "max": None,
        "mean": None,
    }
    if finite.size:
        result.update(
            {
                "min": float(np.min(finite)),
                "median": float(np.median(finite)),
                "max": float(np.max(finite)),
                "mean": float(np.mean(finite)),
            }
        )
    return result


def _station_summary(
    station_id: np.ndarray,
    native: np.ndarray,
    from_momentum: np.ndarray,
    variance: np.ndarray | None,
    tolerance: float,
    absolute_floor: float,
) -> dict[str, object]:
    difference = native - from_momentum
    denominator = np.maximum(np.abs(from_momentum), absolute_floor)
    relative_difference = np.abs(difference) / denominator
    finite_pair = np.isfinite(native) & np.isfinite(from_momentum)
    nonzero_native = np.abs(native) > absolute_floor
    sign_consistent = np.signbit(native) == np.signbit(from_momentum)
    agreement = finite_pair & (relative_difference <= tolerance) & sign_consistent
    covariance_valid = (
        np.isfinite(variance) & (variance >= 0.0)
        if variance is not None
        else np.zeros(native.size, dtype=bool)
    )

    per_station: dict[str, object] = {}
    for station in sorted(set(map(int, station_id))):
        mask = station_id == station
        per_station[str(station)] = {
            "tracklets": int(np.count_nonzero(mask)),
            "finite_q_over_p_pairs": int(np.count_nonzero(finite_pair[mask])),
            "agreement_within_tolerance": int(np.count_nonzero(agreement[mask])),
            "nonzero_native_q_over_p": int(np.count_nonzero(nonzero_native[mask])),
            "valid_q_over_p_variance": int(np.count_nonzero(covariance_valid[mask])),
            "q_over_p_per_mev": _finite_stats(native[mask]),
            "q_over_p_from_momentum_per_mev": _finite_stats(from_momentum[mask]),
            "absolute_difference_per_mev": _finite_stats(difference[mask]),
            "relative_difference": _finite_stats(relative_difference[mask]),
            "q_over_p_variance_per_mev2": (
                _finite_stats(variance[mask]) if variance is not None else None
            ),
        }

    return {
        "tracklets": int(native.size),
        "finite_q_over_p_pairs": int(np.count_nonzero(finite_pair)),
        "agreement_within_tolerance": int(np.count_nonzero(agreement)),
        "nonzero_native_q_over_p": int(np.count_nonzero(nonzero_native)),
        "sign_consistent_pairs": int(np.count_nonzero(finite_pair & sign_consistent)),
        "valid_q_over_p_variance": int(np.count_nonzero(covariance_valid)),
        "q_over_p_per_mev": _finite_stats(native),
        "q_over_p_from_momentum_per_mev": _finite_stats(from_momentum),
        "absolute_difference_per_mev": _finite_stats(difference),
        "relative_difference": _finite_stats(relative_difference),
        "q_over_p_variance_per_mev2": (
            _finite_stats(variance) if variance is not None else None
        ),
        "by_station": per_station,
    }


def _truth_summary(
    station_id: np.ndarray,
    native: np.ndarray,
    variance: np.ndarray | None,
    truth: np.ndarray,
    absolute_floor: float,
) -> dict[str, object]:
    """Compare reconstructed q/p to independently exported MC truth q/p."""
    finite = np.isfinite(native) & np.isfinite(truth)
    difference = native - truth
    denominator = np.maximum(np.abs(truth), absolute_floor)
    relative_difference = difference / denominator
    sign_consistent = np.signbit(native) == np.signbit(truth)
    if variance is None:
        sigma = None
        relative_sigma = None
        pull = None
        covariance_valid = np.zeros(native.size, dtype=bool)
    else:
        covariance_valid = np.isfinite(variance) & (variance >= 0.0)
        sigma = np.full(native.size, np.nan, dtype=np.float64)
        sigma[covariance_valid] = np.sqrt(variance[covariance_valid])
        relative_sigma = sigma / denominator
        pull = difference / sigma

    def summarize(mask: np.ndarray) -> dict[str, object]:
        result: dict[str, object] = {
            "tracklets": int(np.count_nonzero(mask)),
            "finite_truth_pairs": int(np.count_nonzero(finite[mask])),
            "sign_consistent_pairs": int(np.count_nonzero(finite[mask] & sign_consistent[mask])),
            "q_over_p_truth_per_mev": _finite_stats(truth[mask]),
            "difference_per_mev": _finite_stats(difference[mask]),
            "relative_difference": _finite_stats(relative_difference[mask]),
            "q_over_p_sigma_per_mev": _finite_stats(sigma[mask]) if sigma is not None else None,
            "relative_sigma_to_truth": (
                _finite_stats(relative_sigma[mask]) if relative_sigma is not None else None
            ),
            "truth_pull": _finite_stats(pull[mask]) if pull is not None else None,
        }
        return result

    per_station = {
        str(station): summarize(station_id == station)
        for station in sorted(set(map(int, station_id)))
    }
    finite_count = int(np.count_nonzero(finite))
    sign_count = int(np.count_nonzero(finite & sign_consistent))
    median_relative_sigma = (
        _finite_stats(relative_sigma)["median"] if relative_sigma is not None else None
    )
    return {
        "available": True,
        "comparison": "native reconstructed TrackParameters q/p versus MC truth charge/p at the same station",
        "units": "1/MeV",
        "finite_truth_pairs": finite_count,
        "sign_consistent_pairs": sign_count,
        "sign_consistency_fraction": (
            float(sign_count / finite_count) if finite_count else None
        ),
        "median_relative_sigma_to_truth": median_relative_sigma,
        "overall": summarize(np.ones(native.size, dtype=bool)),
        "by_station": per_station,
    }


def audit_q_over_p(
    events: Iterable[EventTracklets],
    relative_tolerance: float = 1.0e-6,
    absolute_floor_per_mev: float = 1.0e-15,
) -> dict[str, object]:
    """Audit native q/p against charge divided by local momentum magnitude.

    The comparison establishes consistency of the exporter convention; both
    quantities come from the same reconstructed track parameters and therefore
    it is not an independent physics validation against truth momentum.
    """
    if relative_tolerance < 0.0:
        raise ValueError("relative_tolerance must be non-negative")
    if absolute_floor_per_mev <= 0.0:
        raise ValueError("absolute_floor_per_mev must be positive")

    event_list = list(events)
    if not event_list:
        return {
            "available": False,
            "reason": "no_events",
            "relative_tolerance": relative_tolerance,
            "absolute_floor_per_mev": absolute_floor_per_mev,
        }

    native_values: list[np.ndarray] = []
    momentum_values: list[np.ndarray] = []
    variance_values: list[np.ndarray] = []
    station_values: list[np.ndarray] = []
    truth_values: list[np.ndarray] = []
    has_variance = True
    has_truth = True
    for event in event_list:
        if (
            event.q_over_p_per_mev is None
            or event.q_over_p_from_momentum_per_mev is None
        ):
            return {
                "available": False,
                "reason": "q_over_p_fields_absent",
                "relative_tolerance": relative_tolerance,
                "absolute_floor_per_mev": absolute_floor_per_mev,
            }
        native_values.append(np.asarray(event.q_over_p_per_mev, dtype=np.float64))
        momentum_values.append(
            np.asarray(event.q_over_p_from_momentum_per_mev, dtype=np.float64)
        )
        station_values.append(np.asarray(event.station_id, dtype=np.int16))
        if event.q_over_p_variance_per_mev2 is None:
            has_variance = False
        else:
            variance_values.append(
                np.asarray(event.q_over_p_variance_per_mev2, dtype=np.float64)
            )
        if event.truth_q_over_p_per_mev is None:
            has_truth = False
        else:
            truth_values.append(np.asarray(event.truth_q_over_p_per_mev, dtype=np.float64))

    native = np.concatenate(native_values)
    from_momentum = np.concatenate(momentum_values)
    station_id = np.concatenate(station_values)
    variance = np.concatenate(variance_values) if has_variance else None
    summary = _station_summary(
        station_id,
        native,
        from_momentum,
        variance,
        tolerance=relative_tolerance,
        absolute_floor=absolute_floor_per_mev,
    )
    finite_pairs = int(summary["finite_q_over_p_pairs"])
    agreement = int(summary["agreement_within_tolerance"])
    truth_summary = (
        _truth_summary(
            station_id,
            native,
            variance,
            np.concatenate(truth_values),
            absolute_floor_per_mev,
        )
        if has_truth
        else {
            "available": False,
            "reason": "truth_q_over_p_field_absent",
            "comparison": "not independently validated against MC truth",
        }
    )
    sign_fraction = truth_summary.get("sign_consistency_fraction")
    median_relative_sigma = truth_summary.get("median_relative_sigma_to_truth")
    usable = bool(
        finite_pairs > 0
        and agreement == finite_pairs
        and truth_summary["available"]
        and sign_fraction == 1.0
        and median_relative_sigma is not None
        and float(median_relative_sigma) <= 1.0
    )
    return {
        "available": True,
        "comparison": (
            "native TrackParameters q/p versus charge divided by reconstructed "
            "momentum magnitude"
        ),
        "units": "1/MeV",
        "relative_tolerance": relative_tolerance,
        "absolute_floor_per_mev": absolute_floor_per_mev,
        "exporter_self_consistent": finite_pairs > 0 and agreement == finite_pairs,
        "usable_as_local_q_over_p_measurement": usable,
        "recommended_model_role": (
            "local_tracklet_feature" if usable else "global_track_latent_parameter_or_MC_audit_only"
        ),
        "q_over_p_covariance_branch_available": variance is not None,
        "truth_comparison": truth_summary,
        **summary,
    }
