"""Pre-registered low-capacity operating-layer control (frozen V2 logits).

This is not an unmatched-penalty scan and not a Transformer architecture
change.  The only learned object is a 6-parameter station-pair Platt map
fitted on train adjacent labels.  Packing constants are copied from the
already frozen historical V2 operating point.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

from baselines.route_assignment import RouteAssignmentConfig, adjacent_station_pairs


def tag_train_only_calibration(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Rewrite a calibration report so it cannot be mistaken for validation Platt."""
    tagged = copy.deepcopy(dict(payload))
    tagged["fit_split"] = "train_only"
    by_pair = tagged.get("by_station_pair")
    if isinstance(by_pair, Mapping):
        tagged["by_station_pair"] = {
            str(pair): (
                {**dict(report), "fit_split": "train_only"}
                if isinstance(report, Mapping)
                else report
            )
            for pair, report in by_pair.items()
        }
    return tagged


def historical_packing_config(
    *,
    unmatched_penalty: float,
    pair_thresholds: Mapping[str, float],
    maximum_hypotheses: int,
) -> RouteAssignmentConfig:
    pairs = adjacent_station_pairs((0, 1, 2, 3))
    thresholds = {}
    for source, target in pairs:
        value = float(pair_thresholds[f"{source}->{target}"])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"historical packing threshold for {source}->{target} is invalid")
        thresholds[(source, target)] = value
    if not math.isfinite(float(unmatched_penalty)):
        raise ValueError("historical unmatched penalty must be finite")
    return RouteAssignmentConfig(
        score_threshold_by_pair=thresholds,
        unmatched_penalty=float(unmatched_penalty),
        station_path=(0, 1, 2, 3),
        maximum_hypotheses=int(maximum_hypotheses),
    )


def nominal_quality_ok(
    route: Mapping[str, Any],
    *,
    maximum_track_fake_rate: float,
    minimum_complete_track_purity: float,
) -> dict[str, object]:
    fake = route.get("track_fake_rate")
    purity = route.get("complete_track_purity")
    fake_ok = fake is not None and float(fake) <= float(maximum_track_fake_rate)
    purity_ok = purity is not None and float(purity) >= float(minimum_complete_track_purity)
    return {
        "ok": bool(fake_ok and purity_ok),
        "track_fake_rate": None if fake is None else float(fake),
        "complete_track_purity": None if purity is None else float(purity),
        "maximum_track_fake_rate": float(maximum_track_fake_rate),
        "minimum_complete_track_purity": float(minimum_complete_track_purity),
    }
