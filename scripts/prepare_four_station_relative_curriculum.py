#!/usr/bin/env python3
"""Compile a minimal 15-DoF relative four-station association curriculum.

Relative configurations are sampled in an S0-identity *sampling chart*, then
a common left SE(3) element is applied as an explicit gauge-control twin.
The physical payload still writes all four stations; S0 is not a true
reference.  Finite-difference probes are not generated.  Sealed test is
never resolved.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from alignment.four_station import (
    FORMULATION,
    GAUGE_REFERENCE_STATION,
    GAUGE_UNCONSTRAINED_FULL,
    IDENTITY_SIX,
    RELATIVE_CURRICULUM_KIND,
    STATION_IDS,
    default_parameter_specs,
    draw_common_left_se3,
    draw_relative_native,
    formulation_contract,
    native_values_from_station_transforms,
    relative_alignment_table,
    relative_family_payloads,
    relative_free_parameter_names,
    relatives_agree,
    zero_parameter_values,
)
from scripts.config_loader import load_yaml_with_base
from scripts.prepare_four_station_identifiability_pilot import _parameter_specs, _point
from scripts.prepare_multidof_alignment_iteration import _safe_name
from scripts.prepare_multisource_multidof_iteration import prepare_iteration


def _read_template(path: Path) -> dict[str, Any]:
    payload = load_yaml_with_base(path)
    if not isinstance(payload, Mapping):
        raise ValueError("relative curriculum template must be a YAML mapping")
    return dict(payload)


def _bounds(raw: Mapping[str, object], *, label: str) -> tuple[float, float]:
    translation = float(raw["translation_mm"])
    rotation = float(raw["rotation_mrad"])
    if translation < 0.0 or rotation < 0.0 or not math.isfinite(translation) or not math.isfinite(rotation):
        raise ValueError(f"{label} bounds must be finite and non-negative")
    return translation, rotation


def _complete_relative(raw: Mapping[str, object]) -> dict[str, float]:
    names = relative_free_parameter_names(gauge=GAUGE_REFERENCE_STATION, reference_station=0)
    values = {name: 0.0 for name in names}
    unknown = set(str(key) for key in raw) - set(names)
    if unknown:
        raise ValueError("hard relative family has unknown names: " + ", ".join(sorted(unknown)))
    for name, value in raw.items():
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"hard relative '{name}' must be finite")
        values[str(name)] = number
    if all(math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1.0e-15) for value in values.values()):
        raise ValueError("hard relative family must be a non-zero 15-DoF sample")
    return values


def _family_points(
    *,
    prefix: str,
    family: str,
    relative_native: Mapping[str, float],
    common: Sequence[float],
    specs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    relative, control = relative_family_payloads(relative_native, common)
    if not relatives_agree(relative, control):
        raise RuntimeError(f"family '{family}' gauge-control does not preserve ΔT_ij")
    relatives = relative_alignment_table(relative)
    names = [str(spec["name"]) for spec in specs]
    zeros = {name: 0.0 for name in names}
    points = []
    for role, name_suffix, transforms, common_payload in (
        ("curriculum", family, relative, list(IDENTITY_SIX)),
        ("gauge_control", f"{family}_plus_common", control, list(common)),
    ):
        native = native_values_from_station_transforms(transforms)
        values = dict(zeros)
        values.update({name: native[name] for name in names})
        point = _point(
            name=f"{prefix}_{name_suffix}",
            values=values,
            specs=specs,
            base_transforms={str(station): list(IDENTITY_SIX) for station in STATION_IDS},
            role=role,
            direction_trial=f"{prefix}_{name_suffix}",
        )
        point.update(
            {
                "condition_value": 1.0,
                "condition_magnitude": 1.0,
                "relative_family": family,
                "gauge_role": "s0_sampling_chart" if role == "curriculum" else "left_se3_control",
                "sampling_chart": GAUGE_REFERENCE_STATION,
                "sampling_reference_station": 0,
                "relative_native_values": {key: float(value) for key, value in relative_native.items()},
                "common_left_se3": [float(value) for value in common_payload],
                "relative_alignment_delta_t_ij": {key: list(value) for key, value in relatives.items()},
                "relative_curriculum": RELATIVE_CURRICULUM_KIND,
            }
        )
        points.append(point)
    return points


def compile_four_station_relative_curriculum(
    template: Mapping[str, object],
    *,
    iteration: int,
    current_values: Mapping[str, float],
) -> tuple[dict[str, object], dict[str, object]]:
    raw_scan = template.get("physical_refit_capture_scan", template)
    if not isinstance(raw_scan, Mapping):
        raise ValueError("physical_refit_capture_scan must be a mapping")
    scan = dict(raw_scan)
    if scan.get("scan_mode") != "station_rigid_multidof":
        raise ValueError("relative curriculum requires station_rigid_multidof")
    if str(scan.get("alignment_formulation", "")) != FORMULATION:
        raise ValueError("relative curriculum requires alignment_formulation: four_station_v1")
    if str(scan.get("gauge", "")) != GAUGE_UNCONSTRAINED_FULL:
        raise ValueError("relative curriculum writes unconstrained_full payloads; S0-gauge is sampling-only")
    if str(scan.get("relative_curriculum", "")) != RELATIVE_CURRICULUM_KIND:
        raise ValueError(f"relative curriculum requires relative_curriculum: {RELATIVE_CURRICULUM_KIND}")
    if scan.get("require_central_finite_difference_probes") is not False:
        raise ValueError("relative curriculum must disable central finite-difference probes")
    if int(scan.get("q_over_p_mode", -1)) != 0:
        raise ValueError("relative curriculum requires q_over_p_mode: 0")
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    specs = _parameter_specs(scan)
    names = [str(spec["name"]) for spec in specs]
    if set(current_values) != set(names):
        raise ValueError("current_values must specify every four-station parameter")
    if any(not math.isclose(float(current_values[name]), 0.0, rel_tol=0.0, abs_tol=1.0e-15) for name in names):
        raise ValueError("relative curriculum linearizes production around the all-zero payload")
    settings = scan.get("relative_sampling")
    if not isinstance(settings, Mapping):
        raise ValueError("relative curriculum lacks relative_sampling")
    seed = int(settings["seed"])
    n_random = int(settings.get("n_random_families", 0))
    if n_random < 0:
        raise ValueError("n_random_families must be non-negative")
    relative_translation, relative_rotation = _bounds(dict(settings["relative_bounds"]), label="relative")
    common_translation, common_rotation = _bounds(dict(settings["common_bounds"]), label="common")
    if relative_translation > 0.5 + 1.0e-12 or relative_rotation > 5.0 + 1.0e-12:
        raise ValueError("relative sampling exceeds the pre-registered local region (<=0.5 mm, <=5 mrad)")
    rng = np.random.default_rng(seed)
    prefix = f"iteration_{iteration:02d}"
    points: list[dict[str, object]] = []
    zeros = {name: 0.0 for name in names}
    reference = _point(
        name=f"{prefix}_reference",
        values=zeros,
        specs=specs,
        base_transforms={str(station): list(IDENTITY_SIX) for station in STATION_IDS},
        role="nominal",
        direction_trial=f"{prefix}_reference",
    )
    reference.update(
        {
            "condition_value": 0.0,
            "condition_magnitude": 0.0,
            "relative_family": "nominal",
            "gauge_role": "identity",
            "sampling_chart": GAUGE_REFERENCE_STATION,
            "sampling_reference_station": 0,
            "relative_native_values": {
                name: 0.0
                for name in relative_free_parameter_names(gauge=GAUGE_REFERENCE_STATION, reference_station=0)
            },
            "common_left_se3": list(IDENTITY_SIX),
            "relative_curriculum": RELATIVE_CURRICULUM_KIND,
        }
    )
    points.append(reference)
    hard_families = settings.get("hard_relative_families", [])
    if not isinstance(hard_families, list):
        raise ValueError("hard_relative_families must be a list")
    families: list[tuple[str, dict[str, float]]] = []
    for item in hard_families:
        if not isinstance(item, Mapping):
            raise ValueError("hard relative family must be a mapping")
        families.append(
            (_safe_name(str(item["name"]), label="relative family"), _complete_relative(dict(item.get("relative_native", {}))))
        )
    for index in range(n_random):
        families.append(
            (
                f"draw_{index:02d}",
                draw_relative_native(
                    rng,
                    translation_mm=relative_translation,
                    rotation_mrad=relative_rotation,
                ),
            )
        )
    if not families:
        raise ValueError("relative curriculum needs at least one non-nominal family")
    for family, relative_native in families:
        common = draw_common_left_se3(
            rng,
            translation_mm=common_translation,
            rotation_mrad=common_rotation,
        )
        points.extend(
            _family_points(
                prefix=prefix,
                family=family,
                relative_native=relative_native,
                common=common,
                specs=specs,
            )
        )
    scan["rigid_points"] = points
    scan["run_alignment_closure"] = False
    scan["reference_station_ids"] = []
    scan["movable_station_ids"] = list(STATION_IDS)
    scan.pop("held_out_closure_points", None)
    contract = formulation_contract(gauge=GAUGE_UNCONSTRAINED_FULL, include_survey_dz=True)
    contract.update(
        {
            "method": "physical_four_station_relative_association_curriculum",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "iteration": iteration,
            "physical_geometry_repropagation": True,
            "coordinate_surrogate": False,
            "q_over_p_mode": 0,
            "relative_curriculum": RELATIVE_CURRICULUM_KIND,
            "sampling_chart": GAUGE_REFERENCE_STATION,
            "sampling_reference_station": 0,
            "require_central_finite_difference_probes": False,
            "reference_point": f"{prefix}_reference",
            "anchor_point": f"{prefix}_reference",
            "n_points": len(points),
            "n_relative_families": len(families),
            "update_semantics": (
                "Association curriculum only: sample 15-DoF relatives in an S0 chart, "
                "then left-multiply a common SE(3) gauge control.  Do not use workbook-52 "
                "held-outs for threshold, unmatched-penalty, or calibration selection."
            ),
        }
    )
    return {"alignment_iteration": contract, "physical_refit_capture_scan": scan}, contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--iteration-template", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--nevents", type=int, default=50)
    parser.add_argument("--source-id", action="append", default=None)
    args = parser.parse_args()
    template = _read_template(Path(args.iteration_template).expanduser().resolve())
    raw_scan = template.get("physical_refit_capture_scan", template)
    if not isinstance(raw_scan, Mapping):
        raise SystemExit("iteration template has no physical_refit_capture_scan")
    specs = raw_scan.get("alignment_parameter_specs") or default_parameter_specs(include_survey_dz=True)
    current = zero_parameter_values([dict(item) for item in specs if isinstance(item, Mapping)])
    manifest = prepare_iteration(
        source_config_path=Path(args.source_config).expanduser().resolve(),
        iteration_template_path=Path(args.iteration_template).expanduser().resolve(),
        output_root=Path(args.output_root).expanduser().resolve(),
        iteration=int(args.iteration),
        current_values=current,
        nevents=int(args.nevents),
        source_ids=args.source_id,
    )
    print(
        json.dumps(
            {
                "iteration_manifest": str(
                    Path(args.output_root).expanduser().resolve() / "iteration_manifest.json"
                ),
                "sources": [item["source_id"] for item in manifest["sources"]],
                "points": len(manifest["common_scan_plan"]["points"]),
                "alignment_formulation": FORMULATION,
                "relative_curriculum": RELATIVE_CURRICULUM_KIND,
                "test_data_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
