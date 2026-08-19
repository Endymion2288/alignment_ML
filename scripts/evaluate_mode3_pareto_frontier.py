#!/usr/bin/env python3
"""Consolidate the matched mode-0/mode-3 route operating-point Pareto scan.

Each arm directory holds one frozen-backbone anchor output per scanned
operating point (``penalty_<tag>/iteration_01_anchor`` and
``thrscale_<scale>/iteration_01_anchor``), produced by
``run_frozen_association_backbone.py`` with inference-time operating-point
overrides.  Checkpoint, feature standardizers, calibration, candidate graph
and the V2 route architecture are identical across points; only the route
unmatched penalty (and in the probe points the threshold scale) varies.

The script reports, per arm, the full (efficiency, purity, fake, missing
recovery, source-wise spread) frontier and the two predeclared matched
comparisons:

1. metrics interpolated onto a shared complete-track-efficiency grid covering
   the mode-0 frozen operating point and the 0.70-0.95 range;
2. the maximum efficiency each arm reaches under fake <= 0.05 and under
   purity >= 0.95, interpolated at the constraint boundary.

All inputs are validation-split only; the sealed test split is never read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.evaluate_mode3_matched_retraining import _source_wise_routes

EFFICIENCY_GRID = (0.70, 0.7426, 0.75, 0.80, 0.85, 0.90, 0.95)
FAKE_CEILING = 0.05
PURITY_FLOOR = 0.95


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _point_metrics(anchor_dir: Path) -> dict[str, Any]:
    summary = _read_json(anchor_dir / "association_summary.json")
    route = summary.get("truth_labelled_mc_evaluation", {}).get("route", {})
    metadata = summary.get("frozen_backbone", {})
    override = metadata.get("operating_point_override", {})
    source_wise = _source_wise_routes(anchor_dir)
    spread = None
    src_min = _float(source_wise.get("min_truth_consistent_fraction"))
    src_max = _float(source_wise.get("max_truth_consistent_fraction"))
    if src_min is not None and src_max is not None:
        spread = src_max - src_min
    return {
        "unmatched_penalty": _float(override.get("unmatched_penalty", metadata.get("unmatched_penalty"))),
        "threshold_scale": _float(override.get("threshold_scale", 1.0)),
        "complete_track_efficiency": _float(route.get("complete_track_efficiency")),
        "complete_track_purity": _float(route.get("complete_track_purity")),
        "track_fake_rate": _float(route.get("track_fake_rate")),
        "missing_station_recovery": _float(route.get("missing_station_recovery")),
        "truth_consistent_routes": _float(route.get("truth_consistent_routes")),
        "selected_routes": _float(route.get("selected_routes")),
        "source_wise_min": src_min,
        "source_wise_max": src_max,
        "source_wise_spread": spread,
    }


def _penalty_frontier(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Penalty-sweep points sorted by efficiency (threshold probes excluded)."""
    frontier = [p for p in points if p["threshold_scale"] == 1.0]
    return sorted(frontier, key=lambda p: p["complete_track_efficiency"])


def _interpolate(frontier: list[dict[str, Any]], key: str, efficiency: float) -> float | None:
    """Piecewise-linear interpolation of ``key`` at ``efficiency``."""
    usable = [
        (p["complete_track_efficiency"], p[key])
        for p in frontier
        if p["complete_track_efficiency"] is not None and p[key] is not None
    ]
    if len(usable) < 2:
        return None
    xs = [point[0] for point in usable]
    ys = [point[1] for point in usable]
    if efficiency < xs[0] or efficiency > xs[-1]:
        return None
    for left in range(len(xs) - 1):
        if xs[left] <= efficiency <= xs[left + 1]:
            span = xs[left + 1] - xs[left]
            weight = 0.0 if span == 0.0 else (efficiency - xs[left]) / span
            return ys[left] * (1.0 - weight) + ys[left + 1] * weight
    return ys[-1]


def _constrained_max_efficiency(
    frontier: list[dict[str, Any]], constraint_key: str, *, at_most: bool, bound: float
) -> dict[str, float | None]:
    """Maximum efficiency with constraint_key <= bound (at_most) or >= bound."""
    feasible = []
    for index in range(len(frontier) - 1):
        a, b = frontier[index], frontier[index + 1]
        va, vb = a[constraint_key], b[constraint_key]
        if va is None or vb is None:
            continue
        if (va - bound) * (vb - bound) <= 0.0 and va != vb:
            weight = (bound - va) / (vb - va)
            eff = a["complete_track_efficiency"] * (1.0 - weight) + b["complete_track_efficiency"] * weight
            feasible.append((eff, bound))
    for p in frontier:
        value = p[constraint_key]
        if value is None:
            continue
        ok = value <= bound if at_most else value >= bound
        if ok:
            feasible.append((p["complete_track_efficiency"], value))
    if not feasible:
        return {"max_efficiency": None, "constraint_value": None}
    best = max(feasible, key=lambda item: item[0])
    return {"max_efficiency": best[0], "constraint_value": best[1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, help="name:scan_root")
    parser.add_argument("--reference-efficiency", type=float, default=0.7426)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    arms: dict[str, Any] = {}
    for spec in args.arm:
        name, root = spec.split(":", 1)
        root_path = Path(root)
        points = []
        for point_dir in sorted(root_path.iterdir()):
            anchor_dir = point_dir / "iteration_01_anchor"
            if not (anchor_dir / "association_summary.json").is_file():
                continue
            metrics = _point_metrics(anchor_dir)
            metrics["point"] = point_dir.name
            points.append(metrics)
        frontier = _penalty_frontier(points)
        arms[name] = {
            "points": points,
            "penalty_frontier": frontier,
            "matched_efficiency_grid": {
                f"{eff:.4f}": {
                    key: _interpolate(frontier, key, eff)
                    for key in (
                        "complete_track_purity",
                        "track_fake_rate",
                        "missing_station_recovery",
                        "source_wise_spread",
                        "source_wise_min",
                    )
                }
                for eff in EFFICIENCY_GRID
            },
            "max_efficiency_fake_ceiling": _constrained_max_efficiency(
                frontier, "track_fake_rate", at_most=True, bound=FAKE_CEILING
            ),
            "max_efficiency_purity_floor": _constrained_max_efficiency(
                frontier, "complete_track_purity", at_most=False, bound=PURITY_FLOOR
            ),
        }

    comparison = {"arms": arms, "constraints": {"fake_ceiling": FAKE_CEILING, "purity_floor": PURITY_FLOOR}}
    names = sorted(arms)
    if len(names) == 2:
        deltas = {}
        for eff, cells in arms[names[0]]["matched_efficiency_grid"].items():
            other = arms[names[1]]["matched_efficiency_grid"].get(eff, {})
            deltas[eff] = {
                key: (None if cells.get(key) is None or other.get(key) is None else other[key] - cells[key])
                for key in cells
            }
        comparison["matched_efficiency_delta"] = {
            "direction": f"{names[1]} minus {names[0]}",
            "grid": deltas,
        }

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
