#!/usr/bin/env python3
"""Run field-chi2 and MLP unknown-association scans on physical refit points.

Each synthetic sample is an overlay of the independently refitted local
tracklets from one real alignment payload.  Its candidate predictions are
copied only from FaserActs mode-0 records exported under that same payload;
the script never shifts coordinates or reuses residuals from another point.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from baselines.field_chi2_matching import build_field_candidates, candidate_labels
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events
from datasets.synthetic_field_propagation import write_synthetic_field_candidate_root
from datasets.synthetic_overlay import write_synthetic_multitrack_root


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "synthetic_unknown_association_muon.yaml"


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle)
    if not isinstance(supplied, Mapping):
        raise ValueError("unknown-association configuration must be a mapping")
    config = supplied.get("synthetic_unknown_association", supplied)
    if not isinstance(config, Mapping):
        raise ValueError("synthetic_unknown_association must be a mapping")
    return dict(config)


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("# command\n")
        handle.write(" ".join(shlex.quote(value) for value in command))
        handle.write("\n\n# output\n")
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def _write_yaml(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=True)


def _prepare_point(
    physical_root: Path,
    output_root: Path,
    point: Mapping[str, object],
    synthetic_config: Mapping[str, object],
    candidate_config: Mapping[str, object],
    resume: bool,
) -> dict[str, object]:
    physical_dir = physical_root / str(point["relative_point_dir"])
    source_tracklets = physical_dir / "refit" / "tracklets.root"
    source_propagations = physical_dir / "refit" / "propagations.root"
    payload_manifest = physical_dir / "payload" / "alignment_payload.json"
    if not source_tracklets.is_file() or not source_propagations.is_file() or not payload_manifest.is_file():
        raise FileNotFoundError(f"physical point is incomplete: {physical_dir}")
    point_dir = output_root / "points" / str(point["name"])
    synthetic_dir = point_dir / "synthetic"
    synthetic_tracklets = synthetic_dir / "tracklets.root"
    field_candidates = synthetic_dir / "field_candidates.root"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    if not (resume and synthetic_tracklets.is_file()):
        summary = write_synthetic_multitrack_root(
            load_events(source_tracklets, require_mc_labels=True),
            synthetic_tracklets,
            output_events=int(synthetic_config["events"]),
            tracks_per_event=int(synthetic_config["tracks_per_event"]),
            station_ids=tuple(int(station) for station in synthetic_config["stations"]),
            missing_tracklet_probability=float(synthetic_config["missing_tracklet_probability"]),
            fake_mean_per_station=float(synthetic_config["fake_mean_per_station"]),
            minimum_truth_match_fraction=float(synthetic_config["minimum_truth_match_fraction"]),
            seed=int(synthetic_config["seed"]),
            synthetic_run_id=int(synthetic_config["synthetic_run_id"]),
        )
        (synthetic_dir / "overlay_summary.json").write_text(
            json.dumps(summary.__dict__, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if not (resume and field_candidates.is_file()):
        summary = write_synthetic_field_candidate_root(
            synthetic_tracklets=synthetic_tracklets,
            source_propagations=source_propagations,
            destination=field_candidates,
            q_over_p_mode=int(candidate_config["q_over_p_mode"]),
            target_z_tolerance_mm=float(candidate_config["target_z_tolerance_mm"]),
        )
        (synthetic_dir / "field_candidate_summary.json").write_text(
            json.dumps(summary.__dict__, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    _write_yaml(
        synthetic_dir / "resolved_config.yaml",
        {
            "physical_point": str(physical_dir),
            "physical_payload_manifest": str(payload_manifest),
            "source_tracklets": str(source_tracklets),
            "source_propagations": str(source_propagations),
            "synthetic_tracklets": str(synthetic_tracklets),
            "field_candidates": str(field_candidates),
            "synthetic_multitrack": dict(synthetic_config),
            "field_candidate_export": dict(candidate_config),
        },
    )
    return {
        "point_dir": point_dir,
        "synthetic_tracklets": synthetic_tracklets,
        "field_candidates": field_candidates,
    }


def _nominal_gate_audit(tracklets: Path, candidates: Path) -> dict[str, object]:
    """Audit the fixed field-chi2 gate before displaced-point evaluation."""
    events = load_events(tracklets, require_mc_labels=True)
    records = load_propagation_records(candidates)
    truth_chi2: list[float] = []
    all_chi2: list[float] = []
    for event in events:
        stations = sorted(set(int(station) for station in event.station_id.tolist()))
        for source in stations:
            for target in stations:
                if source >= target:
                    continue
                pairs = build_field_candidates(event, records, source, target, chi2_gate=None)
                labels = candidate_labels(event, pairs)
                all_chi2.extend(candidate.chi2 for candidate in pairs)
                truth_chi2.extend(
                    candidate.chi2 for candidate, is_truth in zip(pairs, labels) if is_truth
                )
    if not truth_chi2:
        raise ValueError("nominal synthetic sample has no truth candidate pairs")
    return {
        "all_candidate_rows": len(all_chi2),
        "truth_candidate_rows": len(truth_chi2),
        "truth_chi2_max": float(np.max(truth_chi2)),
        "truth_chi2_p99": float(np.percentile(truth_chi2, 99.0)),
        "all_chi2_p99": float(np.percentile(all_chi2, 99.0)),
    }


def _summarise(output_root: Path, points: list[Mapping[str, object]]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for point in points:
        point_dir = output_root / "points" / str(point["name"])
        for algorithm in ("field_chi2", "mlp"):
            metrics_path = point_dir / algorithm / "metrics.json"
            if not metrics_path.is_file():
                raise FileNotFoundError(f"baseline output is missing: {metrics_path}")
            with metrics_path.open(encoding="utf-8") as handle:
                metrics = json.load(handle)
            overall = metrics["overall"]
            rows.append(
                {
                    "point_name": str(point["name"]),
                    "magnitude_mm": float(point["magnitude_mm"]),
                    "direction_trial": str(point["direction_trial"]),
                    "algorithm": algorithm,
                    "candidate_rows": int(overall["candidate_rows"]),
                    "positive_candidate_rows": int(overall["positive_candidate_rows"]),
                    "possible_matches": int(overall["possible_matches"]),
                    "predicted_matches": int(overall["predicted_matches"]),
                    "correct_matches": int(overall["correct_matches"]),
                    "association_efficiency": overall["association_efficiency"],
                    "inclusive_association_purity": overall["inclusive_association_purity"],
                    "inclusive_fake_rate": overall["inclusive_fake_rate"],
                    "scorable_association_purity": overall["association_purity"],
                    "scorable_fake_rate": overall["fake_rate"],
                }
            )
    grouped: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["algorithm"]), float(row["magnitude_mm"]))].append(row)
    by_magnitude: list[dict[str, object]] = []
    metric_names = (
        "association_efficiency",
        "inclusive_association_purity",
        "inclusive_fake_rate",
    )
    for (algorithm, magnitude), group in sorted(grouped.items()):
        entry: dict[str, object] = {
            "algorithm": algorithm,
            "magnitude_mm": magnitude,
            "trials": len(group),
        }
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in group if row[metric] is not None])
            entry[f"{metric}_mean"] = float(np.mean(values)) if values.size else None
            entry[f"{metric}_min"] = float(np.min(values)) if values.size else None
            entry[f"{metric}_max"] = float(np.max(values)) if values.size else None
        by_magnitude.append(entry)
    return {"points": rows, "by_magnitude": by_magnitude}


def _write_plot(path: Path, summary: Mapping[str, object]) -> None:
    records = list(summary["by_magnitude"])
    figure, axes = plt.subplots(3, 1, figsize=(7.2, 8.5), sharex=True, constrained_layout=True)
    definitions = (
        ("association_efficiency", "association efficiency"),
        ("inclusive_association_purity", "association purity (inclusive)"),
        ("inclusive_fake_rate", "fake rate (inclusive)"),
    )
    for algorithm, color in (("field_chi2", "tab:blue"), ("mlp", "tab:orange")):
        subset = [row for row in records if row["algorithm"] == algorithm]
        x_values = np.asarray([row["magnitude_mm"] for row in subset], dtype=np.float64)
        for axis, (metric, label) in zip(axes, definitions):
            y_values = np.asarray([row[f"{metric}_mean"] for row in subset], dtype=np.float64)
            lower = np.asarray([row[f"{metric}_min"] for row in subset], dtype=np.float64)
            upper = np.asarray([row[f"{metric}_max"] for row in subset], dtype=np.float64)
            axis.plot(x_values, y_values, marker="o", color=color, label=algorithm)
            axis.fill_between(x_values, lower, upper, color=color, alpha=0.15)
            axis.set_ylabel(label)
            axis.set_ylim(-0.05, 1.05)
            axis.grid(True, alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("injected station-offset magnitude [mm]")
    if any(float(row["magnitude_mm"]) > 0.0 for row in records):
        axes[-1].set_xscale("symlog", linthresh=0.1)
        axes[-1].set_xlim(0.0, float(max(row["magnitude_mm"] for row in records)) * 1.08)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-scan-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-points", type=int, default=None)
    args = parser.parse_args()
    physical_root = Path(args.physical_scan_root).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    config = _load_config(config_path)
    if int(config["field_candidate_export"]["q_over_p_mode"]) != 0:
        raise ValueError("physical synthetic association scan is restricted to q_over_p_mode=0")
    plan_path = physical_root / "scan_plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"physical scan plan is unavailable: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    all_points = list(plan["points"])
    missing_closures = [
        point["name"]
        for point in all_points
        if not (physical_root / str(point["relative_point_dir"]) / "closure" / "closure.json").is_file()
    ]
    if missing_closures:
        raise RuntimeError(
            "fixed-truth physical scan must complete before unknown association; missing "
            + ", ".join(missing_closures)
        )
    points = all_points if args.max_points is None else all_points[: args.max_points]
    if args.max_points is not None and args.max_points < 1:
        parser.error("--max-points must be positive")
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise FileExistsError("output already exists; use --resume only with the same scan plan")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "config_source": str(config_path),
                "physical_scan_root": str(physical_root),
                "output_root": str(output_root),
                "config": config,
                "physical_geometry_repropagation": True,
                "q_over_p_mode": 0,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    prepared: dict[str, dict[str, object]] = {}
    for point in points:
        prepared[str(point["name"])] = _prepare_point(
            physical_root,
            output_root,
            point,
            config["synthetic_multitrack"],
            config["field_candidate_export"],
            resume=args.resume,
        )

    zero_point = next(point for point in all_points if float(point["magnitude_mm"]) == 0.0)
    if str(zero_point["name"]) not in prepared:
        raise RuntimeError("the zero-offset point must be included to train the nominal MLP")
    nominal = prepared[str(zero_point["name"])]
    training_dir = output_root / "mlp_nominal"
    training_config = dict(config["mlp_pair_classifier"])
    training_config["q_over_p_mode"] = 0
    training_config_path = output_root / "mlp_training_config.yaml"
    _write_yaml(training_config_path, {"mlp_pair_classifier": training_config})
    checkpoint = training_dir / "mlp_pair_classifier.pt"
    if not (args.resume and checkpoint.is_file()):
        _run(
            [
                sys.executable,
                "scripts/train_mlp_pair_classifier.py",
                "--tracklets",
                str(nominal["synthetic_tracklets"]),
                "--propagations",
                str(nominal["field_candidates"]),
                "--output-dir",
                str(training_dir),
                "--config",
                str(training_config_path),
            ],
            output_root / "logs" / "train_mlp.log",
        )
    training_metrics_path = training_dir / "metrics.json"
    if not training_metrics_path.is_file():
        raise FileNotFoundError(f"nominal MLP training metrics are missing: {training_metrics_path}")
    training_metrics = json.loads(training_metrics_path.read_text(encoding="utf-8"))
    heldout_event_ids = training_metrics.get("validation_event_ids")
    if not isinstance(heldout_event_ids, list) or not heldout_event_ids:
        raise ValueError("nominal MLP training did not record held-out validation event IDs")
    heldout_event_file = output_root / "heldout_event_ids.json"
    heldout_event_file.write_text(
        json.dumps(
            {
                "event_ids": [int(event_id) for event_id in heldout_event_ids],
                "selection": "nominal MLP validation split; shared by all algorithms and payloads",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    gate_audit = _nominal_gate_audit(nominal["synthetic_tracklets"], nominal["field_candidates"])
    gate_audit["fixed_field_chi2_gate"] = float(config["field_chi2"]["chi2_gate"])
    gate_audit["all_nominal_truth_candidates_within_gate"] = bool(
        gate_audit["truth_chi2_max"] <= gate_audit["fixed_field_chi2_gate"]
    )
    (output_root / "nominal_field_chi2_gate_audit.json").write_text(
        json.dumps(gate_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    evaluation_config = {
        "q_over_p_mode": 0,
        "field_chi2_gate": float(config["field_chi2"]["chi2_gate"]),
        "mlp_candidate_chi2_gate": training_config["candidate_chi2_gate"],
        "mlp_score_threshold": None,
        "target_z_tolerance_mm": float(config["field_candidate_export"]["target_z_tolerance_mm"]),
        "device": str(training_config["device"]),
    }
    evaluation_config_path = output_root / "evaluation_config.yaml"
    _write_yaml(evaluation_config_path, {"unknown_association_baseline": evaluation_config})
    for point in points:
        item = prepared[str(point["name"])]
        point_dir = Path(item["point_dir"])
        for algorithm in ("field_chi2", "mlp"):
            metrics_path = point_dir / algorithm / "metrics.json"
            if args.resume and metrics_path.is_file():
                continue
            command = [
                sys.executable,
                "scripts/run_unknown_association_baseline.py",
                "--algorithm",
                algorithm,
                "--tracklets",
                str(item["synthetic_tracklets"]),
                "--propagations",
                str(item["field_candidates"]),
                "--output-dir",
                str(point_dir / algorithm),
                "--config",
                str(evaluation_config_path),
                "--event-id-file",
                str(heldout_event_file),
            ]
            if algorithm == "mlp":
                command.extend(["--checkpoint", str(checkpoint)])
            _run(command, point_dir / "logs" / f"{algorithm}.log")
    summary = _summarise(output_root, points)
    summary.update(
        {
            "method": "physical_payload_synthetic_unknown_association_scan",
            "physical_scan_root": str(physical_root),
            "q_over_p_mode": 0,
            "nominal_field_chi2_gate_audit": gate_audit,
            "evaluation_event_split": "nominal MLP held-out validation events shared by all payloads and algorithms",
            "heldout_event_ids": [int(event_id) for event_id in heldout_event_ids],
        }
    )
    (output_root / "unknown_association_scan_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    fields = sorted({field for row in summary["points"] for field in row})
    with (output_root / "unknown_association_scan_points.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary["points"])
    _write_plot(output_root / "unknown_association_scan.png", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
