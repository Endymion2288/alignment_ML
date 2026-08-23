#!/usr/bin/env python3
"""Create source-disjoint, physically refitted payload banks for MLP training.

The driver never alters canonical tracklet coordinates.  It writes a real
/Tracker/Align payload for every bank point and delegates the actual work to
``run_physical_refit_capture_scan.py``, which reruns SCT clusters through the
segment refit and Acts export chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.payload import (
    load_station_alignment_payload,
    load_station_rigid_alignment_payload,
)
from datasets.physical_curriculum import PHYSICAL_CORPUS_SCHEMA, SPLITS
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events
from scripts.config_loader import load_yaml_with_base
from scripts.run_physical_refit_capture_scan import _build_plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCAN_DRIVER = PROJECT_ROOT / "scripts" / "run_physical_refit_capture_scan.py"


def _load_config(path: Path) -> dict[str, Any]:
    supplied = load_yaml_with_base(path)
    config = supplied.get("physical_curriculum_mlp", supplied)
    if not isinstance(config, Mapping):
        raise ValueError("physical_curriculum_mlp must be a YAML mapping")
    return dict(config)


def _normalised_direction(rng: np.random.Generator) -> list[float]:
    values = rng.normal(size=2)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("failed to draw a non-zero random alignment direction")
    return [float(values[0] / norm), float(values[1] / norm)]


def _allowed_splits(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Resolve an explicit corpus read/build boundary before source paths.

    V3 architecture work must not even resolve a sealed test source.  Legacy
    configurations retain the historical all-split behavior when this field is
    absent, while a new train/validation-only configuration has a hard
    boundary at this first parsing step.
    """
    raw_allowed = config.get("allowed_splits", SPLITS)
    if not isinstance(raw_allowed, (list, tuple)):
        raise ValueError("allowed_splits must be a list of split names")
    allowed = tuple(split for split in SPLITS if split in {str(value) for value in raw_allowed})
    if not allowed:
        raise ValueError("allowed_splits must contain at least one known split")
    unknown = {str(value) for value in raw_allowed} - set(SPLITS)
    if unknown:
        raise ValueError("allowed_splits contains unknown split(s): " + ", ".join(sorted(unknown)))
    raw_forbidden = config.get("forbidden_splits", ())
    if not isinstance(raw_forbidden, (list, tuple)):
        raise ValueError("forbidden_splits must be a list of split names")
    forbidden = {str(value) for value in raw_forbidden}
    unknown_forbidden = forbidden - set(SPLITS)
    if unknown_forbidden:
        raise ValueError("forbidden_splits contains unknown split(s): " + ", ".join(sorted(unknown_forbidden)))
    overlap = set(allowed) & forbidden
    if overlap:
        raise ValueError("a split cannot be both allowed and forbidden: " + ", ".join(sorted(overlap)))
    return allowed


def _rotation_curriculum(config: Mapping[str, Any]) -> dict[str, object] | None:
    """Read an explicit IFT R_y curriculum without looking at legacy offsets."""
    supplied = config.get("rotation_curriculum")
    if supplied is None:
        return None
    if not isinstance(supplied, Mapping):
        raise ValueError("rotation_curriculum must be a mapping when supplied")
    if str(supplied.get("scan_mode", "")) != "ift_ry_rotation":
        raise ValueError("rotation_curriculum.scan_mode must be 'ift_ry_rotation'")
    if str(supplied.get("condition_axis", "")) != "ift_ry_mrad":
        raise ValueError("rotation_curriculum.condition_axis must be 'ift_ry_mrad'")
    points = supplied.get("rotation_points")
    if not isinstance(points, list) or not points:
        raise ValueError("rotation_curriculum.rotation_points must be a non-empty list")
    return dict(supplied)


_RIGID_COMPONENTS: dict[str, tuple[int, float]] = {
    "dx_mm": (0, 1.0),
    "dy_mm": (1, 1.0),
    "dz_mm": (2, 1.0),
    "rx_mrad": (3, 1.0e-3),
    "ry_mrad": (4, 1.0e-3),
    "rz_mrad": (5, 1.0e-3),
}


def _joint_rigid_curriculum(config: Mapping[str, Any]) -> dict[str, object] | None:
    """Read a deterministic joint-rigid physical curriculum declaration.

    Point sampling itself happens before individual source scan configs are
    written.  Consequently every source in a split receives the *same*
    auditable real payload bank and can still be pooled without mixing
    geometry conditions, while train and validation receive independently
    seeded joint payloads.
    """
    supplied = config.get("joint_rigid_curriculum")
    if supplied is None:
        return None
    if not isinstance(supplied, Mapping):
        raise ValueError("joint_rigid_curriculum must be a mapping when supplied")
    if str(supplied.get("scan_mode", "")) != "station_rigid_multidof":
        raise ValueError("joint_rigid_curriculum.scan_mode must be 'station_rigid_multidof'")
    if not str(supplied.get("condition_axis", "")):
        raise ValueError("joint_rigid_curriculum.condition_axis must be non-empty")
    if not isinstance(supplied.get("alignment_parameter_specs"), list):
        raise ValueError("joint_rigid_curriculum.alignment_parameter_specs must be a list")
    for field in ("reference_station_ids", "movable_station_ids", "curriculum_stages"):
        if not isinstance(supplied.get(field), (list, tuple)) or not supplied.get(field):
            raise ValueError(f"joint_rigid_curriculum.{field} must be a non-empty list")
    try:
        seed = int(supplied["seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("joint_rigid_curriculum requires integer seed") from error
    if seed < 0:
        raise ValueError("joint_rigid_curriculum.seed must be non-negative")
    return dict(supplied)


def _joint_specs(settings: Mapping[str, object]) -> list[dict[str, object]]:
    raw_specs = settings.get("alignment_parameter_specs")
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("joint rigid curriculum has no alignment parameter specs")
    result: list[dict[str, object]] = []
    names: set[str] = set()
    for raw in raw_specs:
        if not isinstance(raw, Mapping):
            raise ValueError("joint rigid parameter spec must be a mapping")
        name = str(raw.get("name", ""))
        component = str(raw.get("component", ""))
        if not name or not name.replace("_", "").isalnum() or name in names:
            raise ValueError("joint rigid parameter names must be unique alphanumeric identifiers")
        if component not in _RIGID_COMPONENTS:
            raise ValueError(f"joint rigid parameter '{name}' has unsupported component '{component}'")
        try:
            station = int(raw["station_id"])
            step = float(raw["finite_difference_step"])
            severity_scale = float(raw["severity_scale"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"joint rigid parameter '{name}' is incomplete") from error
        if not math.isfinite(step) or step <= 0.0 or not math.isfinite(severity_scale) or severity_scale <= 0.0:
            raise ValueError(f"joint rigid parameter '{name}' has invalid finite-difference/severity scale")
        names.add(name)
        result.append(
            {
                "name": name,
                "station_id": station,
                "component": component,
                "finite_difference_step": step,
                "severity_scale": severity_scale,
                **dict(raw),
            }
        )
    return result


def _joint_point_seed(seed: int, split: str) -> int:
    digest = hashlib.sha256(f"{seed}:faser-joint-rigid:{split}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _joint_severity(values: Mapping[str, float], specs: Sequence[Mapping[str, object]]) -> float:
    return float(
        math.sqrt(
            sum(
                (float(values[str(spec["name"])]) / float(spec["severity_scale"])) ** 2
                for spec in specs
            )
        )
    )


def _joint_station_transforms(
    values: Mapping[str, float],
    specs: Sequence[Mapping[str, object]],
    station_ids: Sequence[int],
) -> dict[int, list[float]]:
    transforms = {int(station): [0.0] * 6 for station in station_ids}
    for spec in specs:
        station = int(spec["station_id"])
        component = str(spec["component"])
        index, payload_scale = _RIGID_COMPONENTS[component]
        transforms[station][index] = float(values[str(spec["name"])]) * payload_scale
    return transforms


def _joint_ranges(
    stage: Mapping[str, object], specs: Sequence[Mapping[str, object]]
) -> dict[str, tuple[float, float]]:
    supplied = stage.get("ranges")
    if not isinstance(supplied, Mapping):
        raise ValueError(f"joint curriculum stage '{stage.get('name')}' requires ranges")
    names = {str(spec["name"]) for spec in specs}
    if {str(key) for key in supplied} != names:
        raise ValueError(
            f"joint curriculum stage '{stage.get('name')}' ranges must specify exactly {sorted(names)}"
        )
    result: dict[str, tuple[float, float]] = {}
    for name in names:
        raw_range = supplied[name]
        if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
            raise ValueError(f"joint curriculum stage '{stage.get('name')}' range '{name}' must be [min, max]")
        lower, upper = (float(raw_range[0]), float(raw_range[1]))
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError(f"joint curriculum stage '{stage.get('name')}' has invalid range '{name}'")
        result[name] = (lower, upper)
    return result


def _joint_rigid_points(
    config: Mapping[str, Any], allowed_splits: tuple[str, ...]
) -> dict[str, list[dict[str, object]]]:
    """Materialize frozen, split-specific joint payload point definitions."""
    settings = _joint_rigid_curriculum(config)
    if settings is None:
        return {}
    specs = _joint_specs(settings)
    refit = config.get("refit")
    if not isinstance(refit, Mapping):
        raise ValueError("joint rigid curriculum requires refit settings")
    station_ids = [int(value) for value in refit["station_ids"]]
    reference = {int(value) for value in settings["reference_station_ids"]}
    movable = {int(value) for value in settings["movable_station_ids"]}
    if reference.intersection(movable) or reference.union(movable) != set(station_ids):
        raise ValueError("joint rigid reference/movable station sets must cover refit.station_ids exactly")
    if any(int(spec["station_id"]) not in movable for spec in specs):
        raise ValueError("joint rigid parameter cannot alter a reference station")

    def point(
        *,
        name: str,
        values: Mapping[str, float],
        role: str,
        direction_trial: str,
        sampling_seed: int | None = None,
        curriculum_stage: str | None = None,
        finite_difference_for: str | None = None,
        probe_sign: str | None = None,
    ) -> dict[str, object]:
        severity = _joint_severity(values, specs)
        result: dict[str, object] = {
            "name": name,
            "direction_trial": direction_trial,
            "point_role": role,
            "condition_value": severity,
            "condition_magnitude": severity,
            "alignment_parameter_values": {key: float(value) for key, value in values.items()},
            "station_transforms": _joint_station_transforms(values, specs, station_ids),
        }
        if sampling_seed is not None:
            result["sampling_seed"] = int(sampling_seed)
        if curriculum_stage is not None:
            result["curriculum_stage"] = curriculum_stage
        if finite_difference_for is not None:
            result["finite_difference_for"] = finite_difference_for
            result["probe_sign"] = probe_sign
        return result

    zero = {str(spec["name"]): 0.0 for spec in specs}
    result: dict[str, list[dict[str, object]]] = {}
    for split in allowed_splits:
        seed = _joint_point_seed(int(settings["seed"]), split)
        rng = np.random.default_rng(seed)
        points: list[dict[str, object]] = [
            point(
                name="joint_nominal",
                values=zero,
                role="nominal",
                direction_trial="nominal",
                sampling_seed=seed,
            )
        ]
        for spec in specs:
            parameter = str(spec["name"])
            step = float(spec["finite_difference_step"])
            for sign, suffix, role in (
                (1.0, "p", "finite_difference_positive"),
                (-1.0, "m", "finite_difference_negative"),
            ):
                values = dict(zero)
                values[parameter] = sign * step
                points.append(
                    point(
                        name=f"fd_{parameter}_{suffix}",
                        values=values,
                        role=role,
                        direction_trial=f"fd_{parameter}_{suffix}",
                        sampling_seed=seed,
                        finite_difference_for=parameter,
                        probe_sign="positive" if sign > 0.0 else "negative",
                    )
                )
        stage_groups = (
            ("curriculum_stages", "curriculum"),
            ("held_out_closure_stages", "held_out_closure"),
        )
        names: set[str] = {str(item["name"]) for item in points}
        for field, role in stage_groups:
            raw_stages = settings.get(field, ())
            if not isinstance(raw_stages, (list, tuple)):
                raise ValueError(f"joint_rigid_curriculum.{field} must be a list when supplied")
            for raw_stage in raw_stages:
                if not isinstance(raw_stage, Mapping):
                    raise ValueError(f"joint_rigid_curriculum.{field} entries must be mappings")
                stage_name = str(raw_stage.get("name", ""))
                if not stage_name or not stage_name.replace("_", "").isalnum():
                    raise ValueError(f"joint rigid {field} stage names must be alphanumeric identifiers")
                try:
                    count = int(raw_stage["points_per_split"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"joint rigid stage '{stage_name}' requires points_per_split") from error
                if count < 1:
                    raise ValueError(f"joint rigid stage '{stage_name}' points_per_split must be positive")
                ranges = _joint_ranges(raw_stage, specs)
                for index in range(count):
                    values = {
                        parameter: float(rng.uniform(lower, upper))
                        for parameter, (lower, upper) in ranges.items()
                    }
                    if all(math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1.0e-15) for value in values.values()):
                        raise ValueError(
                            f"joint rigid stage '{stage_name}' generated an all-zero non-nominal transform"
                        )
                    prefix = "joint" if role == "curriculum" else "closure"
                    name = f"{prefix}_{stage_name}_{index:02d}"
                    if name in names:
                        raise ValueError(f"joint rigid stage generated duplicate point '{name}'")
                    names.add(name)
                    points.append(
                        point(
                            name=name,
                            values=values,
                            role=role,
                            direction_trial=f"{stage_name}_{index:02d}",
                            sampling_seed=seed,
                            curriculum_stage=stage_name,
                        )
                    )
        result[split] = points
    return result


def _payload_directions(
    config: Mapping[str, Any], allowed_splits: tuple[str, ...]
) -> dict[str, list[dict[str, object]]]:
    bank = config["payload_bank"]
    if not isinstance(bank, Mapping):
        raise ValueError("payload_bank must be a mapping")
    counts = bank["direction_trials_per_split"]
    if not isinstance(counts, Mapping):
        raise ValueError("direction_trials_per_split must be a mapping")
    stations = tuple(sorted(int(value) for value in config["refit"]["station_ids"]))
    reference = int(config["refit"]["reference_station"])
    movable = tuple(station for station in stations if station != reference)
    if not movable:
        raise ValueError("at least one non-reference station is required")
    rng = np.random.default_rng(int(bank["seed"]))
    result: dict[str, list[dict[str, object]]] = {}
    for split in allowed_splits:
        count = int(counts.get(split, 0))
        if count < 1:
            raise ValueError(f"payload_bank requires at least one direction for {split}")
        trials: list[dict[str, object]] = []
        for index in range(count):
            trials.append(
                {
                    "name": f"{split}_{index:02d}",
                    "unit_offsets_xy": {
                        int(station): _normalised_direction(rng) for station in movable
                    },
                }
            )
        result[split] = trials
    return result


def _validate_sources(
    config: Mapping[str, Any], allowed_splits: tuple[str, ...]
) -> list[dict[str, str]]:
    supplied = config.get("sources")
    if not isinstance(supplied, list) or not supplied:
        raise ValueError("sources must be a non-empty list")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    present: set[str] = set()
    for raw in supplied:
        if not isinstance(raw, Mapping):
            raise ValueError("each source must be a mapping")
        source_id = str(raw.get("id", ""))
        split = str(raw.get("split", ""))
        if not source_id or not source_id.replace("_", "").isalnum() or source_id in seen:
            raise ValueError("source IDs must be unique alphanumeric identifiers")
        if split not in SPLITS:
            raise ValueError(f"source '{source_id}' has invalid split '{split}'")
        # Reject excluded rows before inspecting their paths.  This prevents a
        # sealed test xAOD from being resolved or stat'ed by a V3 run.
        if split not in allowed_splits:
            raise ValueError(
                f"source '{source_id}' belongs to excluded split '{split}'; "
                "remove it from this bounded corpus configuration"
            )
        input_xaod = Path(str(raw.get("input_xaod", ""))).expanduser().resolve()
        if not input_xaod.is_file():
            raise FileNotFoundError(f"source xAOD is unavailable: {input_xaod}")
        seen.add(source_id)
        present.add(split)
        result.append({"source_id": source_id, "split": split, "input_xaod": str(input_xaod)})
    missing = set(allowed_splits) - present
    if missing:
        raise ValueError("sources omit split(s): " + ", ".join(sorted(missing)))
    return result


def _source_scan_config(
    config: Mapping[str, Any],
    source: Mapping[str, str],
    directions: Mapping[str, list[dict[str, object]]],
    joint_points: Mapping[str, list[dict[str, object]]] | None = None,
) -> dict[str, object]:
    refit = dict(config["refit"])
    rotation = _rotation_curriculum(config)
    joint = _joint_rigid_curriculum(config)
    if rotation is not None and joint is not None:
        raise ValueError("a physical curriculum cannot declare both rotation_curriculum and joint_rigid_curriculum")
    if joint is not None:
        if joint_points is None or source["split"] not in joint_points:
            raise ValueError("joint rigid curriculum has no frozen points for source split")
        scan = {
            "scan_mode": "station_rigid_multidof",
            "input_xaod": source["input_xaod"],
            "nevents": int(refit["nevents"]),
            "station_ids": [int(value) for value in refit["station_ids"]],
            "reference_station_ids": [int(value) for value in joint["reference_station_ids"]],
            "movable_station_ids": [int(value) for value in joint["movable_station_ids"]],
            "condition_axis": str(joint["condition_axis"]),
            "q_over_p_mode": int(refit.get("q_over_p_mode", 0)),
            "min_truth_match_fraction": float(refit.get("min_truth_match_fraction", 0.99)),
            "chi2_gate": float(refit.get("chi2_gate", 25.0)),
            "refinement_iterations": int(refit.get("refinement_iterations", 1)),
            "alignment_parameter_specs": list(joint["alignment_parameter_specs"]),
            "rigid_points": list(joint_points[source["split"]]),
            # All closure calls must be driven by actual finite-difference
            # refits once the physical point bank has passed audit.
            "run_alignment_closure": False,
        }
        return {"physical_refit_capture_scan": scan}
    if rotation is not None:
        raw_reference = rotation.get("reference_station_ids")
        raw_movable = rotation.get("movable_station_ids")
        if not isinstance(raw_reference, (list, tuple)) or not isinstance(raw_movable, (list, tuple)):
            raise ValueError("rotation_curriculum requires reference_station_ids and movable_station_ids")
        scan = {
            "scan_mode": "ift_ry_rotation",
            "input_xaod": source["input_xaod"],
            "nevents": int(refit["nevents"]),
            "station_ids": [int(value) for value in refit["station_ids"]],
            "reference_station_ids": [int(value) for value in raw_reference],
            "movable_station_ids": [int(value) for value in raw_movable],
            "condition_axis": "ift_ry_mrad",
            "q_over_p_mode": int(refit.get("q_over_p_mode", 0)),
            "min_truth_match_fraction": float(refit.get("min_truth_match_fraction", 0.99)),
            "chi2_gate": float(refit.get("chi2_gate", 25.0)),
            "refinement_iterations": int(refit.get("refinement_iterations", 1)),
            "rotation_points": list(rotation["rotation_points"]),
            # R_y closure uses physical finite-difference response points and
            # is explicitly invoked only after the bank passes surface audit.
            "run_alignment_closure": False,
        }
        return {"physical_refit_capture_scan": scan}
    bank = dict(config["payload_bank"])
    scan = {
        "input_xaod": source["input_xaod"],
        "nevents": int(refit["nevents"]),
        "station_ids": [int(value) for value in refit["station_ids"]],
        "reference_station": int(refit["reference_station"]),
        "q_over_p_mode": int(refit.get("q_over_p_mode", 0)),
        "min_truth_match_fraction": float(refit.get("min_truth_match_fraction", 0.99)),
        "chi2_gate": float(refit.get("chi2_gate", 25.0)),
        "refinement_iterations": int(refit.get("refinement_iterations", 1)),
        "prior_sigma_mm": refit.get("prior_sigma_mm"),
        "capture_tolerance_mm": float(refit.get("capture_tolerance_mm", 0.01)),
        "magnitudes_mm": [float(value) for value in bank["magnitudes_mm"]],
        "direction_trials": directions[source["split"]],
        # Fixed-truth closure was already established.  The curriculum corpus
        # requires the expensive physical refit/export, not a duplicate WLS
        # closure at every training payload.
        "run_alignment_closure": False,
    }
    return {"physical_refit_capture_scan": scan}


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_event_uids(source_id: str, tracklets: Path) -> list[str]:
    if not tracklets.is_file():
        return []
    events = load_events(tracklets, require_mc_labels=True)
    return [f"{source_id}:{event.run_id}:{event.event_id}" for event in events]


def _mode0_propagation_completion(
    propagations: Path,
    station_ids: Sequence[int],
) -> tuple[bool, str]:
    """Require finite successful mode-0 Acts coverage for every graph pair."""
    try:
        records = load_propagation_records(propagations)
    except Exception as error:
        return False, f"invalid_propagations:{type(error).__name__}"
    if records.q_over_p_mode is None:  # pragma: no cover - loader supplies legacy zero mode
        return False, "propagations_missing_q_over_p_mode"
    accepted = (
        (np.asarray(records.q_over_p_mode, dtype=np.int8) == 0)
        & np.asarray(records.success, dtype=bool)
        & np.asarray(records.has_covariance, dtype=bool)
        & np.isfinite(records.prediction).all(axis=1)
        & np.isfinite(records.covariance).all(axis=(1, 2))
    )
    if not np.any(accepted):
        return False, "propagations_missing_successful_mode0_records"
    stations = tuple(sorted(int(station) for station in station_ids))
    expected_pairs = {
        (source, target)
        for source_index, source in enumerate(stations)
        for target in stations[source_index + 1 :]
    }
    observed = {
        (int(source), int(target))
        for source, target in zip(
            records.source_station_id[accepted], records.target_station_id[accepted]
        )
    }
    missing = sorted(expected_pairs - observed)
    if missing:
        return False, "propagations_missing_mode0_pair:" + ",".join(
            f"{source}->{target}" for source, target in missing
        )
    return True, "accepted"


def _payload_matches_injection(
    payload_manifest: Path,
    expected_offsets_xy_mm: Mapping[str, object] | None,
    station_ids: Sequence[int],
    expected_station_transforms: Mapping[str, object] | None = None,
    expected_layer_transforms: Mapping[str, object] | None = None,
) -> tuple[bool, str]:
    """Verify that the persisted /Tracker/Align payload is the planned point."""
    if expected_station_transforms is not None:
        try:
            payload = load_station_rigid_alignment_payload(payload_manifest)
        except Exception as error:
            return False, f"invalid_alignment_payload:{type(error).__name__}"
        for station in station_ids:
            raw_expected = expected_station_transforms.get(str(int(station)))
            if not isinstance(raw_expected, (list, tuple)) or len(raw_expected) != 6:
                return False, f"planned_payload_missing_station_transform:{int(station)}"
            try:
                expected = np.asarray(raw_expected, dtype=np.float64)
            except (TypeError, ValueError):
                return False, f"planned_payload_invalid_station_transform:{int(station)}"
            actual = np.asarray(payload.transform_for_station(int(station)), dtype=np.float64)
            if not np.isfinite(expected).all() or not np.allclose(
                actual, expected, rtol=0.0, atol=1.0e-12
            ):
                return False, f"alignment_payload_transform_mismatch:{int(station)}"
        if expected_layer_transforms is not None:
            if not isinstance(expected_layer_transforms, Mapping):
                return False, "planned_payload_invalid_layer_transforms"
            for raw_station, raw_layers in expected_layer_transforms.items():
                if not isinstance(raw_layers, Mapping):
                    return False, f"planned_payload_invalid_layer_station:{raw_station}"
                station = int(raw_station)
                for raw_layer, raw_expected in raw_layers.items():
                    if not isinstance(raw_expected, (list, tuple)) or len(raw_expected) != 6:
                        return False, f"planned_payload_missing_layer_transform:{station}:{raw_layer}"
                    try:
                        expected = np.asarray(raw_expected, dtype=np.float64)
                    except (TypeError, ValueError):
                        return False, f"planned_payload_invalid_layer_transform:{station}:{raw_layer}"
                    actual = np.asarray(
                        payload.transform_for_layer(station, int(raw_layer)), dtype=np.float64
                    )
                    if not np.isfinite(expected).all() or not np.allclose(
                        actual, expected, rtol=0.0, atol=1.0e-12
                    ):
                        return False, f"alignment_payload_layer_transform_mismatch:{station}:{raw_layer}"
        return True, "accepted"
    if expected_offsets_xy_mm is None:
        return False, "planned_payload_missing_injection_definition"
    try:
        payload = load_station_alignment_payload(payload_manifest)
    except Exception as error:
        return False, f"invalid_alignment_payload:{type(error).__name__}"
    for station in station_ids:
        raw_expected = expected_offsets_xy_mm.get(str(int(station)))
        if not isinstance(raw_expected, (list, tuple)) or len(raw_expected) != 2:
            return False, f"planned_payload_missing_station:{int(station)}"
        try:
            expected = np.asarray(raw_expected, dtype=np.float64)
        except (TypeError, ValueError):
            return False, f"planned_payload_invalid_station:{int(station)}"
        actual = np.asarray(payload.offset_for_station(int(station)), dtype=np.float64)
        if not np.isfinite(expected).all() or not np.allclose(actual, expected, rtol=0.0, atol=1.0e-12):
            return False, f"alignment_payload_offset_mismatch:{int(station)}"
    return True, "accepted"


def _physical_point_completion(
    *,
    tracklets: Path,
    propagations: Path,
    payload_manifest: Path,
    content_audit: Path,
    failure: Path,
    station_ids: Sequence[int],
    expected_offsets_xy_mm: Mapping[str, object] | None,
    expected_station_transforms: Mapping[str, object] | None = None,
    expected_layer_transforms: Mapping[str, object] | None = None,
    require_mc_labels: bool = True,
) -> tuple[bool, str]:
    """Accept only a physically complete and audited refit/Acts point.

    File existence alone is insufficient: a partial exporter output could
    otherwise be pooled into a synthetic training sample.  The lightweight
    audit checks below deliberately use the existing ``audit_tracklets.py``
    sidecar instead of recomputing coordinates or re-running reconstruction.
    Propagation schema/content is validated again by the pooled materializer
    when it loads the canonical ROOT asset.
    """
    required = {
        "payload_manifest": payload_manifest,
        "tracklets": tracklets,
        "propagations": propagations,
        "content_audit": content_audit,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return False, "missing:" + ",".join(missing)
    if failure.exists():
        return False, "failure_marker_present"
    if expected_station_transforms is None:
        # Keep the legacy positional call contract intact for translation
        # corpus tooling and its external audit hooks.
        payload_accepted, payload_status = _payload_matches_injection(
            payload_manifest, expected_offsets_xy_mm, station_ids
        )
    else:
        payload_accepted, payload_status = _payload_matches_injection(
            payload_manifest,
            expected_offsets_xy_mm,
            station_ids,
            expected_station_transforms=expected_station_transforms,
            expected_layer_transforms=expected_layer_transforms,
        )
    if not payload_accepted:
        return False, payload_status
    try:
        with content_audit.open(encoding="utf-8") as handle:
            audit = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        return False, f"invalid_content_audit:{type(error).__name__}"
    if not isinstance(audit, Mapping):
        return False, "invalid_content_audit:root_not_mapping"
    if require_mc_labels:
        if audit.get("has_mc_labels") is not True:
            return False, "content_audit_missing_mc_labels"
    elif audit.get("has_mc_labels") is True:
        return False, "real_data_content_audit_has_mc_labels"
    try:
        audited_tracklets = Path(str(audit["input"])).expanduser().resolve()
    except (KeyError, TypeError, ValueError):
        return False, "content_audit_missing_input"
    if audited_tracklets != tracklets.resolve():
        return False, "content_audit_input_mismatch"
    try:
        events = int(audit["events"])
        tracklet_count = int(audit["tracklets"])
        covariance = audit["covariance"]
        if not isinstance(covariance, Mapping):
            return False, "content_audit_invalid_covariance"
        covariance_rows = int(covariance["rows"])
        positive_definite_rows = int(covariance["positive_definite_rows"])
    except (KeyError, TypeError, ValueError):
        return False, "content_audit_invalid_counts"
    if events < 1 or tracklet_count < 1:
        return False, "content_audit_empty"
    if covariance_rows != tracklet_count or positive_definite_rows != tracklet_count:
        return False, "content_audit_covariance_not_positive_definite"
    counts = audit.get("station_counts")
    if not isinstance(counts, Mapping):
        return False, "content_audit_missing_station_counts"
    missing_stations = [
        int(station)
        for station in station_ids
        if int(counts.get(str(int(station)), 0)) < 1
    ]
    if missing_stations:
        return False, "content_audit_missing_station:" + ",".join(map(str, missing_stations))
    propagation_accepted, propagation_status = _mode0_propagation_completion(
        propagations, station_ids
    )
    if not propagation_accepted:
        return False, propagation_status
    return True, "accepted"


def _manifest(
    output_root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    sources: list[dict[str, str]],
    directions: Mapping[str, list[dict[str, object]]],
    joint_points: Mapping[str, list[dict[str, object]]] | None = None,
) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for source in sources:
        source_root = output_root / "sources" / source["source_id"]
        source_config_path = source_root / "physical_scan_config.yaml"
        scan_root = source_root / "physical_scan"
        if source_config_path.is_file():
            with source_config_path.open(encoding="utf-8") as handle:
                source_scan = yaml.safe_load(handle)
        else:
            source_scan = _source_scan_config(config, source, directions, joint_points)
        scan_section = source_scan["physical_refit_capture_scan"]
        plan = _build_plan(scan_section)
        points: list[dict[str, object]] = []
        for point in plan["points"]:
            point_root = scan_root / str(point["relative_point_dir"])
            tracklets = point_root / "refit" / "tracklets.root"
            propagations = point_root / "refit" / "propagations.root"
            payload_manifest = point_root / "payload" / "alignment_payload.json"
            content_audit = point_root / "refit" / "content_audit.json"
            completed, completion_status = _physical_point_completion(
                tracklets=tracklets,
                propagations=propagations,
                payload_manifest=payload_manifest,
                content_audit=content_audit,
                failure=point_root / "failure.json",
                station_ids=tuple(int(value) for value in scan_section["station_ids"]),
                expected_offsets_xy_mm=(
                    dict(point["injected_offsets_xy_mm"])
                    if "injected_offsets_xy_mm" in point
                    else None
                ),
                expected_station_transforms=(
                    dict(point["injected_station_transforms"])
                    if "injected_station_transforms" in point
                    else None
                ),
            )
            points.append(
                {
                    **point,
                    "payload_id": str(point["name"]),
                    "physical_tracklets": str(tracklets),
                    "physical_propagations": str(propagations),
                    "physical_payload_manifest": str(payload_manifest),
                    "physical_content_audit": str(content_audit),
                    "completion_status": completion_status,
                    "completed": completed,
                }
            )
        if str(plan["scan_mode"]) == "translation_xy":
            zero_name = f"mag_0_{directions[source['split']][0]['name']}"
        else:
            zero_point = next(
                point for point in plan["points"] if float(point["condition_magnitude"]) == 0.0
            )
            zero_name = str(zero_point["name"])
        zero_tracklets = scan_root / "points" / zero_name / "refit" / "tracklets.root"
        entries.append(
            {
                **source,
                "physical_scan_config": str(source_config_path),
                "physical_scan_root": str(scan_root),
                "source_event_uids": _source_event_uids(source["source_id"], zero_tracklets),
                "points": points,
            }
        )
    return {
        "schema_version": PHYSICAL_CORPUS_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_source": str(config_path),
        "output_root": str(output_root),
        "source_split_unit": "original_xAOD_file",
        "allowed_splits": list(_allowed_splits(config)),
        "forbidden_splits": [
            split for split in SPLITS if split not in set(_allowed_splits(config))
        ],
        "source_event_uid_convention": "source_id:run_id:event_id",
        "physical_geometry_repropagation": True,
        "refit_chain": (
            "persisted SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit -> "
            "NtupleDumper -> FaserActsExtrapolationTool"
        ),
        "q_over_p_mode": int(config["refit"].get("q_over_p_mode", 0)),
        "payload_bank": (
            {
                **dict(config["payload_bank"]),
                "generated_direction_trials": directions,
            }
            if _rotation_curriculum(config) is None and _joint_rigid_curriculum(config) is None
            else {
                "mode": (
                    "station_rigid_multidof"
                    if _joint_rigid_curriculum(config) is not None
                    else "ift_ry_rotation"
                ),
                **dict(
                    _joint_rigid_curriculum(config)
                    if _joint_rigid_curriculum(config) is not None
                    else (_rotation_curriculum(config) or {})
                ),
                **(
                    {"generated_points_by_split": dict(joint_points or {})}
                    if _joint_rigid_curriculum(config) is not None
                    else {}
                ),
            }
        ),
        "condition_axis": (
            "translation_xy_mm"
            if _rotation_curriculum(config) is None and _joint_rigid_curriculum(config) is None
            else str(
                (_joint_rigid_curriculum(config) or _rotation_curriculum(config) or {})[
                    "condition_axis"
                ]
            )
        ),
        "sources": entries,
    }


def _run_source(source_root: Path, max_points: int | None, resume: bool) -> None:
    config_path = source_root / "physical_scan_config.yaml"
    scan_root = source_root / "physical_scan"
    command = [
        sys.executable,
        str(SCAN_DRIVER),
        "--config",
        str(config_path),
        "--output-dir",
        str(scan_root),
    ]
    if resume or scan_root.exists():
        command.append("--resume")
    if max_points is not None:
        command.extend(["--max-points", str(max_points)])
    log_path = source_root / "physical_scan_driver.log"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("# command\n" + " ".join(command) + "\n\n# output\n")
        result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "physical_curriculum_mlp_muon.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--source-id",
        action="append",
        default=None,
        help="Process only this configured source ID; repeat for a controlled shard",
    )
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--max-points-per-source", type=int, default=None)
    args = parser.parse_args()
    if args.max_sources is not None and args.max_sources < 1:
        parser.error("--max-sources must be positive")
    if args.max_points_per_source is not None and args.max_points_per_source < 1:
        parser.error("--max-points-per-source must be positive")
    config_path = Path(args.config).expanduser().resolve()
    config = _load_config(config_path)
    if int(config["refit"].get("q_over_p_mode", 0)) != 0:
        raise ValueError("physical curriculum V1 must use q_over_p_mode=0")
    allowed_splits = _allowed_splits(config)
    sources = _validate_sources(config, allowed_splits)
    if _rotation_curriculum(config) is not None and _joint_rigid_curriculum(config) is not None:
        raise ValueError("a physical curriculum cannot declare both rotation_curriculum and joint_rigid_curriculum")
    directions = (
        {}
        if _rotation_curriculum(config) is not None or _joint_rigid_curriculum(config) is not None
        else _payload_directions(config, allowed_splits)
    )
    joint_points = _joint_rigid_points(config, allowed_splits)
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for source in sources:
        source_root = output_root / "sources" / source["source_id"]
        source_root.mkdir(parents=True, exist_ok=True)
        source_config = _source_scan_config(config, source, directions, joint_points)
        config_destination = source_root / "physical_scan_config.yaml"
        if config_destination.is_file():
            existing = yaml.safe_load(config_destination.read_text(encoding="utf-8"))
            if existing != source_config:
                raise ValueError(f"existing source config differs for {source['source_id']}")
        else:
            config_destination.write_text(yaml.safe_dump(source_config, sort_keys=True), encoding="utf-8")
    manifest_path = output_root / "physical_corpus_manifest.json"
    _write_json(manifest_path, _manifest(output_root, config_path, config, sources, directions, joint_points))
    if args.prepare_only:
        print(json.dumps({"manifest": str(manifest_path), "status": "prepared"}, indent=2))
        return
    if args.source_id:
        requested = set(args.source_id)
        known = {source["source_id"] for source in sources}
        unknown = requested - known
        if unknown:
            parser.error("unknown --source-id: " + ", ".join(sorted(unknown)))
        selected = [source for source in sources if source["source_id"] in requested]
    else:
        selected = sources if args.max_sources is None else sources[: args.max_sources]
    for source in selected:
        source_root = output_root / "sources" / source["source_id"]
        _run_source(source_root, args.max_points_per_source, resume=args.resume)
        refreshed = _manifest(output_root, config_path, config, sources, directions, joint_points)
        _write_json(manifest_path, refreshed)
        if args.max_points_per_source is None:
            source_entry = next(
                entry for entry in refreshed["sources"] if entry["source_id"] == source["source_id"]
            )
            incomplete = [point["payload_id"] for point in source_entry["points"] if not point["completed"]]
            if incomplete:
                raise RuntimeError(
                    f"physical corpus source '{source['source_id']}' has incomplete payloads: "
                    + ", ".join(incomplete)
                )
    print(json.dumps({"manifest": str(manifest_path), "sources_processed": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
