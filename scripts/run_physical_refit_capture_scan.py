#!/usr/bin/env python3
"""Run a real-conditions, cluster-to-segment physical capture-range scan.

Every planned point writes a fresh /Tracker/Align SQLite/POOL payload and
reruns Calypso's SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit ->
NtupleDumper chain before field-aware mode-0 propagation is evaluated.  The
only reused input is the original persisted MC xAOD; coordinate-level shifts
and cached residuals are deliberately not part of this workflow.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
SETUP_SCRIPT = PROJECT_ROOT / "scripts" / "setup_environment.sh"
PAYLOAD_WRITER = PROJECT_ROOT / "scripts" / "write_station_alignment_payload.py"
SUMMARIZER = PROJECT_ROOT / "scripts" / "summarize_physical_refit_capture_scan.py"


def _quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def _slug(value: float) -> str:
    text = format(value, ".8g")
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle)
    if not isinstance(supplied, Mapping):
        raise ValueError("scan configuration must be a YAML mapping")
    config = supplied.get("physical_refit_capture_scan", supplied)
    if not isinstance(config, Mapping):
        raise ValueError("physical_refit_capture_scan must be a YAML mapping")
    return dict(config)


def _build_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    stations = tuple(sorted({int(station) for station in config["station_ids"]}))
    reference = int(config["reference_station"])
    if len(stations) < 2 or reference not in stations:
        raise ValueError("station_ids must contain the reference and at least one movable station")
    if any(station not in (0, 1, 2, 3) for station in stations):
        raise ValueError("the current conditions payload supports only stations 0 through 3")
    magnitudes = [float(value) for value in config["magnitudes_mm"]]
    if not magnitudes or any(not math.isfinite(value) or value < 0.0 for value in magnitudes):
        raise ValueError("magnitudes_mm must be a non-empty list of finite non-negative values")
    if magnitudes != sorted(magnitudes) or len(set(magnitudes)) != len(magnitudes):
        raise ValueError("magnitudes_mm must be strictly increasing")
    supplied_directions = config["direction_trials"]
    if not isinstance(supplied_directions, list) or not supplied_directions:
        raise ValueError("direction_trials must be a non-empty list")
    movable = tuple(station for station in stations if station != reference)
    directions: list[dict[str, Any]] = []
    names: set[str] = set()
    for direction in supplied_directions:
        if not isinstance(direction, Mapping):
            raise ValueError("every direction trial must be a mapping")
        name = str(direction["name"])
        if not name.replace("_", "").isalnum() or name in names:
            raise ValueError("direction trial names must be unique alphanumeric identifiers")
        names.add(name)
        raw_offsets = direction["unit_offsets_xy"]
        if not isinstance(raw_offsets, Mapping):
            raise ValueError("unit_offsets_xy must map station IDs to [dx, dy]")
        offsets = {int(station): tuple(float(value) for value in offset) for station, offset in raw_offsets.items()}
        if set(offsets) != set(movable):
            raise ValueError("each direction must specify exactly the non-reference stations")
        for station, offset in offsets.items():
            if len(offset) != 2 or not all(math.isfinite(value) for value in offset):
                raise ValueError(f"invalid unit offset for station {station}")
            if not math.isclose(math.hypot(*offset), 1.0, rel_tol=0.0, abs_tol=1.0e-8):
                raise ValueError(f"unit offset for station {station} is not normalized")
        directions.append({"name": name, "unit_offsets_xy": offsets})

    points: list[dict[str, Any]] = []
    point_index = 0
    for magnitude in magnitudes:
        # A zero injection is direction-independent, so it is one physical
        # baseline refit rather than redundant repeats.
        active_directions = directions[:1] if magnitude == 0.0 else directions
        for direction in active_directions:
            injected = {str(reference): [0.0, 0.0]}
            for station in movable:
                unit = direction["unit_offsets_xy"][station]
                injected[str(station)] = [magnitude * unit[0], magnitude * unit[1]]
            name = f"mag_{_slug(magnitude)}_{direction['name']}"
            points.append(
                {
                    "index": point_index,
                    "name": name,
                    "relative_point_dir": str(Path("points") / name),
                    "magnitude_mm": magnitude,
                    "direction_trial": direction["name"],
                    "injected_offsets_xy_mm": injected,
                }
            )
            point_index += 1
    return {
        "method": "physical_refit_capture_range_scan",
        "station_ids": list(stations),
        "reference_station": reference,
        "q_over_p_mode": int(config["q_over_p_mode"]),
        "points": points,
    }


def _run_shell(
    command: str,
    log_path: Path,
    dry_run: bool,
    cwd: Path = WORKSPACE_ROOT,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"[dry-run] {command}")
        return
    with log_path.open("w", encoding="utf-8") as log:
        log.write("# command\n")
        log.write(command)
        log.write("\n\n# output\n")
        result = subprocess.run(
            ["bash", "-lc", command],
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def _calypso_command(command: str) -> str:
    # The driver is often launched from LCG_110_cuda for its YAML/plotting
    # dependencies.  Do not let that Python 3.13 stack leak into Athena's
    # Python 3.9/LCG_104d process.
    clean_environment = "unset PYTHONPATH LD_LIBRARY_PATH ROOTSYS ROOT_INCLUDE_PATH PYTHONHOME"
    return f"{clean_environment}\nsource {_quote(SETUP_SCRIPT)} calypso\n{command}"


def _ml_command(command: str) -> str:
    return f"source {_quote(SETUP_SCRIPT)} ml\ncd {_quote(PROJECT_ROOT)}\n{command}"


def _write_failure(point_dir: Path, phase: str, error: Exception) -> None:
    payload = {
        "status": "failed",
        "phase": phase,
        "message": str(error),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (point_dir / "failure.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _is_complete(path: Path) -> bool:
    return path.is_file()


def _run_point(
    point: Mapping[str, Any],
    config: Mapping[str, Any],
    scan_root: Path,
    baseline_refit: Path,
    dry_run: bool,
    resume: bool,
) -> bool:
    point_dir = scan_root / str(point["relative_point_dir"])
    point_dir.mkdir(parents=True, exist_ok=True)
    payload_dir = point_dir / "payload"
    refit_dir = point_dir / "refit"
    closure_dir = point_dir / "closure"
    manifest = payload_dir / "alignment_payload.json"
    enhanced = refit_dir / "enhanced_tracklets.root"
    tracklets = refit_dir / "tracklets.root"
    propagations = refit_dir / "propagations.root"
    audit = refit_dir / "content_audit.json"
    closure = closure_dir / "closure.json"

    try:
        if not _is_complete(manifest):
            phase = "payload"
            offsets = point["injected_offsets_xy_mm"]
            arguments = " ".join(
                f"--offset {int(station)}:{float(values[0]):.17g}:{float(values[1]):.17g}"
                for station, values in sorted(offsets.items(), key=lambda item: int(item[0]))
            )
            _run_shell(
                _calypso_command(
                    f"python {_quote(PAYLOAD_WRITER)} --output-dir {_quote(payload_dir)} {arguments}"
                ),
                point_dir / "logs" / "payload.log",
                dry_run,
                cwd=point_dir,
            )
        if not _is_complete(enhanced):
            phase = "segment_refit"
            refit_dir.mkdir(parents=True, exist_ok=True)
            input_xaod = Path(str(config["input_xaod"])).expanduser().resolve()
            command = " ".join(
                [
                    "faser_ntuple_maker.py",
                    _quote(input_xaod),
                    "--isMC",
                    "--useIFT",
                    f"--nevents {int(config['nevents'])}",
                    "--export-tracklets",
                    "--export-tracklet-propagation",
                    "--refit-segments",
                    f"--tracker-align-sqlite {_quote(payload_dir / 'tracker_alignment.sqlite')}",
                    f"--tracker-align-pool-catalog {_quote(payload_dir / 'PoolFileCatalog.xml')}",
                    f"--outfile {_quote(enhanced)}",
                ]
            )
            _run_shell(
                _calypso_command(command), point_dir / "logs" / "refit.log", dry_run, cwd=point_dir
            )
        if not (_is_complete(tracklets) and _is_complete(propagations) and _is_complete(audit)):
            phase = "export_and_propagation"
            command = "\n".join(
                [
                    " ".join(
                        [
                            "python scripts/convert_ntuple_tracklets.py",
                            _quote(enhanced),
                            "--output",
                            _quote(tracklets),
                            "--include-truth",
                        ]
                    ),
                    " ".join(
                        [
                            "python scripts/convert_ntuple_tracklet_propagations.py",
                            _quote(enhanced),
                            "--output",
                            _quote(propagations),
                        ]
                    ),
                    " ".join(
                        [
                            "python scripts/audit_tracklets.py",
                            _quote(tracklets),
                            "--output",
                            _quote(audit),
                            "--require-mc-labels",
                        ]
                    ),
                ]
            )
            _run_shell(_ml_command(command), point_dir / "logs" / "export.log", dry_run)
        if bool(config.get("run_alignment_closure", True)) and not _is_complete(closure):
            phase = "alignment_closure"
            if not dry_run and not (
                _is_complete(baseline_refit / "tracklets.root")
                and _is_complete(baseline_refit / "propagations.root")
            ):
                raise RuntimeError("the zero-offset physical baseline has not completed")
            command = " ".join(
                [
                    "python scripts/run_refit_alignment_closure.py",
                    "--nominal-tracklets",
                    _quote(baseline_refit / "tracklets.root"),
                    "--nominal-propagations",
                    _quote(baseline_refit / "propagations.root"),
                    "--displaced-tracklets",
                    _quote(tracklets),
                    "--displaced-propagations",
                    _quote(propagations),
                    "--payload-manifest",
                    _quote(manifest),
                    "--output-dir",
                    _quote(closure_dir),
                    "--reference-station",
                    str(int(config["reference_station"])),
                    "--q-over-p-mode",
                    str(int(config["q_over_p_mode"])),
                    "--min-truth-match-fraction",
                    str(float(config["min_truth_match_fraction"])),
                    "--refinement-iterations",
                    str(int(config["refinement_iterations"])),
                    "--chi2-gate",
                    str(float(config["chi2_gate"])),
                ]
                + (
                    []
                    if config.get("prior_sigma_mm") is None
                    else ["--prior-sigma-mm", str(float(config["prior_sigma_mm"]))]
                )
            )
            _run_shell(_ml_command(command), point_dir / "logs" / "closure.log", dry_run)
    except Exception as error:
        if not dry_run:
            _write_failure(point_dir, phase=locals().get("phase", "unknown"), error=error)
        print(f"[failed] {point['name']}: {error}", file=sys.stderr)
        return False
    if not dry_run:
        failure = point_dir / "failure.json"
        if failure.exists():
            failure.unlink()
    print(f"[complete] {point['name']}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "physical_refit_capture_scan_muon.yaml"),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-points", type=int, default=None)
    args = parser.parse_args()
    if args.max_points is not None and args.max_points < 1:
        parser.error("--max-points must be positive")

    config_path = Path(args.config).expanduser().resolve()
    config = _load_config(config_path)
    if int(config.get("q_over_p_mode", -1)) != 0:
        raise ValueError("physical V1 capture scans must use q_over_p_mode=0")
    if not Path(str(config["input_xaod"])).expanduser().is_file():
        raise FileNotFoundError(f"input xAOD is unavailable: {config['input_xaod']}")
    plan = _build_plan(config)
    scan_root = Path(args.output_dir).expanduser().resolve()
    if scan_root.exists() and any(scan_root.iterdir()) and not args.resume:
        raise FileExistsError("scan output already exists; use --resume only for the identical plan")
    scan_root.mkdir(parents=True, exist_ok=True)
    plan_path = scan_root / "scan_plan.json"
    if plan_path.is_file():
        existing_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing_plan != plan:
            raise ValueError("existing scan plan differs from the requested configuration")
    else:
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (scan_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                **config,
                "config_source": str(config_path),
                "output_dir": str(scan_root),
                "physical_geometry_repropagation": True,
                "refit_chain": (
                    "persisted SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit -> "
                    "NtupleDumper -> FaserActsExtrapolationTool"
                ),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    zero_point = next(point for point in plan["points"] if float(point["magnitude_mm"]) == 0.0)
    baseline_refit = scan_root / str(zero_point["relative_point_dir"]) / "refit"
    points = plan["points"]
    if args.max_points is not None:
        points = points[: args.max_points]
    completed = 0
    for point in points:
        completed += int(
            _run_point(
                point,
                config,
                scan_root,
                baseline_refit,
                dry_run=args.dry_run,
                resume=args.resume,
            )
        )
    if not args.dry_run and bool(config.get("run_alignment_closure", True)):
        summary_command = " ".join(
            [
                "python scripts/summarize_physical_refit_capture_scan.py",
                "--scan-root",
                _quote(scan_root),
                "--scan-plan",
                _quote(plan_path),
                "--capture-tolerance-mm",
                str(float(config["capture_tolerance_mm"])),
            ]
        )
        _run_shell(_ml_command(summary_command), scan_root / "summary.log", dry_run=False)
    elif not args.dry_run:
        (scan_root / "physical_refit_summary.json").write_text(
            json.dumps(
                {
                    "method": "physical_payload_refit_corpus",
                    "physical_geometry_repropagation": True,
                    "run_alignment_closure": False,
                    "points_requested": len(points),
                    "points_completed": completed,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"physical scan points completed: {completed}/{len(points)}")


if __name__ == "__main__":
    main()
