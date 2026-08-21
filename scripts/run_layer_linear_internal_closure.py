#!/usr/bin/env python3
"""Truth-selected IFT layer-internal closure from a reused linear Jacobian.

The finite-difference bank stays frozen (layer dx/rx/ry only).  The observed
payloads are a separate held-out-only physical scan: outer-antisymmetric
relative dx or relative rx.  Station 5-DoF, mode-0, and sealed test stay
closed.  Admission is in the zero-common-mode family
(``outer_contrast``, equal-weight ``sum_to_zero``).  Frozen-station
``reference_layer`` is solved and reported as a negative control of a
different physical constraint, not as a gauge cross-check.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from alignment.gauge_equivalence import (
    CANONICAL_INTERNAL_BASIS,
    FROZEN_STATION_NEGATIVE_CONTROL,
    ZERO_COMMON_MODE_CHOICES,
)
from alignment.layer_hierarchy import (
    FREE_COMPONENTS,
    HIERARCHY_FIT_CHOICES,
    IFT_LAYER_IDS,
    expand_gauged_parameters,
    reduce_gauge,
    spec_scope,
    split_common_and_internal,
)
from alignment.physical_jacobian import parameter_values_from_payload
from scripts.audit_6dof_identifiability import _fit_summary, _source_entries
from scripts.audit_layer_identifiability import (
    _aligned_rows,
    _bank_for_target,
    _evaluation,
    _expected_delta,
    _fit,
    _json_ready,
    _layer_weights_from_hit_pattern,
    _ordered_fd_points,
    _payload_for_point,
    _point_map,
    _pool_layer_banks,
    _read_json,
    _recovery_from_fit,
)


SCHEMA_VERSION = "faser-ift-layer-linear-internal-closure-v3"
GAUGE_CHOICES = HIERARCHY_FIT_CHOICES
COMPONENT_THRESHOLDS = {
    "dx_mm": {
        "internal_abs": 0.03,
        "gauge_agreement_abs": 0.02,
        "station_abs": 0.03,
        "source_spread_abs": 0.03,
        "relative_fraction": 0.25,
    },
    "rx_mrad": {
        "internal_abs": 0.15,
        "gauge_agreement_abs": 0.10,
        "station_abs": 0.15,
        "source_spread_abs": 0.25,
        "relative_fraction": 0.25,
    },
    "ry_mrad": {
        "internal_abs": 0.35,
        "gauge_agreement_abs": 0.25,
        "station_abs": 0.35,
        "source_spread_abs": 0.35,
        "relative_fraction": 0.50,
    },
}


def _require_sealed_manifest(manifest: Mapping[str, object], *, label: str) -> None:
    if manifest.get("test_data_accessed") is not False:
        raise ValueError(f"{label} has an invalid test-access declaration")
    if int(manifest.get("q_over_p_mode", -1)) != 0:
        raise ValueError(f"{label} is not a mode-0 physical scan")


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(array))))


def _layer_specs_for_component(specs: Sequence[Mapping[str, object]], component: str) -> tuple[str, ...]:
    names = []
    for spec in specs:
        if spec_scope(spec) == "layer" and str(spec.get("component")) == component:
            names.append(str(spec["name"]))
    if len(names) != len(IFT_LAYER_IDS):
        raise ValueError(f"expected three IFT layer parameters for '{component}', found {names}")
    return tuple(names)


def _station_specs(specs: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    names = tuple(str(spec["name"]) for spec in specs if spec_scope(spec) == "station")
    if len(names) != 5:
        raise ValueError(f"expected five frozen station parameters, found {names}")
    return names


def injected_linear_component(
    values: Mapping[str, float],
    specs: Sequence[Mapping[str, object]],
) -> str:
    active: set[str] = set()
    for spec in specs:
        name = str(spec["name"])
        if math.isclose(float(values.get(name, 0.0)), 0.0, rel_tol=0.0, abs_tol=1.0e-15):
            continue
        if spec_scope(spec) == "station":
            raise ValueError(f"held-out injects station parameter '{name}'")
        component = str(spec.get("component"))
        if component not in {"dx_mm", "rx_mrad", "ry_mrad"}:
            raise ValueError(f"held-out injects non-linear or excluded component '{component}'")
        active.add(component)
    if len(active) != 1:
        raise ValueError(f"held-out must inject exactly one of layer dx, rx, or ry, found {sorted(active)}")
    return next(iter(active))


def internals_by_layer(split: Mapping[str, object], component: str) -> dict[str, float]:
    internal = split["layer_internal"]
    return {f"layer_{layer}": float(internal[f"layer_{layer}"][component]) for layer in IFT_LAYER_IDS}


def _partition_representations(choices: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split zero-common-mode admission charts from the frozen-station negative control."""
    available = tuple(str(choice) for choice in choices)
    admission = tuple(choice for choice in ZERO_COMMON_MODE_CHOICES if choice in available)
    if not admission:
        admission = tuple(choice for choice in available if choice != FROZEN_STATION_NEGATIVE_CONTROL)
    if not admission:
        admission = available
    negative = tuple(choice for choice in available if choice == FROZEN_STATION_NEGATIVE_CONTROL)
    return admission, negative


def physics_verdict(
    *,
    component: str,
    expected_internals: Mapping[str, float],
    recovered_by_gauge: Mapping[str, Mapping[str, float]],
    source_internals: Mapping[str, Mapping[str, Mapping[str, float]]],
    station_leakage_by_gauge: Mapping[str, Mapping[str, float]],
    ranks: Mapping[str, Mapping[str, object]],
    residual_ratios: Mapping[str, float],
    max_condition_number: float,
) -> dict[str, Any]:
    """Admit the zero-common-mode family; treat frozen-station reference_layer as a control."""
    thresholds = COMPONENT_THRESHOLDS[component]
    expected = {key: float(value) for key, value in expected_internals.items()}
    choices = tuple(recovered_by_gauge)
    if not choices:
        raise ValueError("physics verdict requires at least one hierarchy representation")
    admission_choices, negative_choices = _partition_representations(choices)
    gauge_tables = {
        choice: {key: float(value) for key, value in recovered_by_gauge[choice].items()} for choice in choices
    }
    outer_expected = expected["layer_0"] - expected["layer_2"]
    peak = abs(outer_expected) or 1.0
    internal_tol = max(float(thresholds["internal_abs"]), float(thresholds["relative_fraction"]) * (max(abs(value) for value in expected.values()) or 1.0))
    outer_tol = max(float(thresholds["internal_abs"]), float(thresholds["relative_fraction"]) * peak)
    agreement_tol = max(float(thresholds["gauge_agreement_abs"]), 0.5 * outer_tol)

    outer_recovered = {
        choice: gauge_tables[choice]["layer_0"] - gauge_tables[choice]["layer_2"] for choice in choices
    }
    recovered = all(abs(outer_recovered[choice] - outer_expected) <= outer_tol for choice in admission_choices)
    gauge_agreement = True
    for index, left in enumerate(admission_choices):
        for right in admission_choices[index + 1 :]:
            if abs(outer_recovered[left] - outer_recovered[right]) > agreement_tol:
                gauge_agreement = False
    per_layer: dict[str, Any] = {}
    for layer_key, truth in expected.items():
        values = {choice: gauge_tables[choice][layer_key] for choice in choices}
        errors = {choice: abs(value - truth) for choice, value in values.items()}
        per_layer[layer_key] = {
            "expected": truth,
            "recovered": values,
            "abs_error": errors,
            "informational": layer_key == "layer_1",
        }

    sample_gauges = next(iter(source_internals.values()), {})
    if "outer_contrast" in sample_gauges:
        primary = "outer_contrast"
    elif "sum_to_zero" in sample_gauges:
        primary = "sum_to_zero"
    else:
        primary = admission_choices[0] if admission_choices else choices[0]
    outer_by_source = {
        source_id: float(gauges[primary]["layer_0"] - gauges[primary]["layer_2"])
        for source_id, gauges in source_internals.items()
    }
    outer_spread = (
        float(max(outer_by_source.values()) - min(outer_by_source.values())) if outer_by_source else float("nan")
    )
    source_stable = bool(outer_by_source) and outer_spread <= float(thresholds["source_spread_abs"])
    direction_consistent = True
    per_source_direction: dict[str, Any] = {}
    for source_id, gauges in source_internals.items():
        source_outers = {
            choice: float(table["layer_0"] - table["layer_2"]) for choice, table in gauges.items() if choice in admission_choices
        }
        signs_ok = all(
            value == 0.0 or outer_expected == 0.0 or ((value > 0.0) == (outer_expected > 0.0))
            for value in source_outers.values()
        )
        direction_consistent = direction_consistent and signs_ok
        per_source_direction[source_id] = {
            "outer_relative": {
                choice: float(table["layer_0"] - table["layer_2"]) for choice, table in gauges.items()
            },
            "admission_outer_relative": source_outers,
            "direction_pass": signs_ok,
        }
    source_spread = {
        "outer_relative_layer0_minus_layer2": {
            "expected": outer_expected,
            "recovered": outer_recovered,
            "admission_recovered": {choice: outer_recovered[choice] for choice in admission_choices},
            "per_source": outer_by_source,
            "abs_spread": outer_spread,
            "pass": source_stable,
            "direction_consistent": direction_consistent,
            "per_source_all_representations": per_source_direction,
        },
        "per_layer_informational": {},
    }
    for layer_key, truth in expected.items():
        by_source = {
            source_id: float(gauges[primary][layer_key]) for source_id, gauges in source_internals.items()
        }
        spread = float(max(by_source.values()) - min(by_source.values())) if by_source else float("nan")
        source_spread["per_layer_informational"][layer_key] = {
            "per_source": by_source,
            "abs_spread": spread,
        }

    no_leakage = True
    leakage = {}
    for choice in admission_choices:
        station = station_leakage_by_gauge.get(choice, {})
        dx_or_rot = abs(float(station.get(component, 0.0)))
        leak_pass = dx_or_rot <= float(thresholds["station_abs"])
        no_leakage = no_leakage and leak_pass
        leakage[choice] = {"station_common": dict(station), "pass": leak_pass}

    rank_pass = True
    condition_pass = True
    rank_table = {}
    for choice in admission_choices:
        payload = ranks[choice]
        full_rank = bool(payload["full_rank"])
        condition = payload["normal_matrix_condition_number"]
        well = condition is not None and float(condition) <= float(max_condition_number)
        rank_pass = rank_pass and full_rank
        condition_pass = condition_pass and well
        rank_table[choice] = {
            "rank": payload["normal_matrix_rank"],
            "dimension": payload["dimension"],
            "full_rank": full_rank,
            "condition_number": condition,
            "condition_pass": well,
            "role": "admission",
        }
    for choice in negative_choices:
        if choice not in ranks:
            continue
        payload = ranks[choice]
        rank_table[choice] = {
            "rank": payload["normal_matrix_rank"],
            "dimension": payload["dimension"],
            "full_rank": bool(payload["full_rank"]),
            "condition_number": payload["normal_matrix_condition_number"],
            "condition_pass": None,
            "role": "frozen_station_negative_control",
        }

    residual_pass = True
    residual_table = {}
    for choice in admission_choices:
        ratio = residual_ratios[choice]
        ok = math.isfinite(ratio) and ratio <= 0.5
        residual_pass = residual_pass and ok
        residual_table[choice] = {"post_over_pre_rms": ratio, "pass": ok, "role": "admission"}
    for choice in negative_choices:
        if choice not in residual_ratios:
            continue
        residual_table[choice] = {
            "post_over_pre_rms": residual_ratios[choice],
            "pass": None,
            "role": "frozen_station_negative_control",
        }

    negative_control = None
    if negative_choices:
        choice = negative_choices[0]
        negative_control = {
            "choice": choice,
            "role": "frozen_station_different_physical_family",
            "not_a_gauge_cross_check": True,
            "outer_relative": outer_recovered.get(choice),
            "outer_relative_error": (
                None if choice not in outer_recovered else abs(outer_recovered[choice] - outer_expected)
            ),
            "agrees_with_zero_common_mode_family": (
                choice in outer_recovered
                and all(abs(outer_recovered[choice] - outer_recovered[item]) <= agreement_tol for item in admission_choices)
            ),
            "post_over_pre_rms": residual_ratios.get(choice),
        }

    passed = bool(
        recovered
        and gauge_agreement
        and source_stable
        and direction_consistent
        and no_leakage
        and rank_pass
        and condition_pass
        and residual_pass
    )
    return {
        "component": component,
        "canonical_internal_basis": CANONICAL_INTERNAL_BASIS,
        "admission_representations": list(admission_choices),
        "negative_control_representations": list(negative_choices),
        "internal_tolerance": internal_tol,
        "outer_relative_tolerance": outer_tol,
        "gauge_agreement_tolerance": agreement_tol,
        "layers": per_layer,
        "source_stability": source_spread,
        "station_leakage": leakage,
        "rank_condition": rank_table,
        "post_fit_residual": residual_table,
        "negative_control": negative_control,
        "physics_recovered": recovered,
        "gauges_agree": gauge_agreement,
        "source_stable": source_stable,
        "source_direction_consistent": direction_consistent,
        "no_station_common_mode_leakage": no_leakage,
        "full_rank": rank_pass,
        "well_conditioned": condition_pass,
        "residual_improved": residual_pass,
        "passed": passed,
    }


def _combined_source_bank(
    jacobian_entry: Mapping[str, object],
    observed_entry: Mapping[str, object],
    *,
    anchor_point: str,
    target_point: str,
    only_parameters: Sequence[str],
    min_truth_match_fraction: float,
) -> dict[str, Any]:
    source_id = str(observed_entry["source_id"])
    if str(jacobian_entry["source_id"]) != source_id:
        raise ValueError("jacobian and observed banks must be paired by source_id")
    jacobian_root = Path(str(jacobian_entry["physical_scan_root"])).expanduser().resolve()
    observed_root = Path(str(observed_entry["physical_scan_root"])).expanduser().resolve()
    jacobian_plan = _read_json(jacobian_root / "scan_plan.json")
    observed_plan = _read_json(observed_root / "scan_plan.json")
    if jacobian_plan.get("held_out_only"):
        raise ValueError(f"jacobian bank for '{source_id}' must contain finite-difference probes")
    specs, names, scales, movable, ordered, _ = _ordered_fd_points(
        jacobian_plan,
        anchor_point=anchor_point,
        target_points=(),
        only_parameters=only_parameters,
        require_targets=False,
    )
    target = _point_map(observed_plan).get(target_point)
    if target is None:
        raise ValueError(f"observed scan for '{source_id}' lacks target '{target_point}'")

    payloads = []
    evaluations = []
    name_to_index: dict[str, int] = {}
    for point, root in [(item, jacobian_root) for item in ordered] + [(target, observed_root)]:
        payload, tracklets, propagations = _payload_for_point(root, point)
        name_to_index[str(point["name"])] = len(evaluations)
        payloads.append(payload)
        evaluations.append(_evaluation(tracklets, propagations, min_truth_match_fraction))
    keys, indexes, overlap = _aligned_rows(evaluations, movable)
    rows = [np.asarray([index[key] for key in keys], dtype=np.intp) for index in indexes]
    parameters = len(names)
    nested_layers = [
        {
            str(station): {
                str(layer): list(payload.transform_for_layer(int(station), int(layer)))
                for layer in IFT_LAYER_IDS
            }
            for station in (0,)
        }
        for payload in payloads
    ]
    anchor_values = parameter_values_from_payload(specs, payloads[0].transforms, nested_layers[0])
    positive_values = np.asarray(
        [
            parameter_values_from_payload(specs, payloads[1 + 2 * index].transforms, nested_layers[1 + 2 * index])[name]
            for index, name in enumerate(names)
        ],
        dtype=np.float64,
    )
    negative_values = np.asarray(
        [
            parameter_values_from_payload(specs, payloads[2 + 2 * index].transforms, nested_layers[2 + 2 * index])[name]
            for index, name in enumerate(names)
        ],
        dtype=np.float64,
    )
    target_index = name_to_index[target_point]
    target_values = parameter_values_from_payload(specs, payloads[target_index].transforms, nested_layers[target_index])
    target_residual = np.asarray(evaluations[target_index].residual[rows[target_index]], dtype=np.float64)
    anchor_evaluation = evaluations[0]
    anchor_rows = rows[0]
    return {
        "source_id": source_id,
        "split": str(observed_entry.get("split", "")),
        "specs": specs,
        "names": names,
        "scales": scales,
        "anchor_values": np.asarray([anchor_values[name] for name in names], dtype=np.float64),
        "reference_values": np.asarray([target_values[name] for name in names], dtype=np.float64),
        "target_names": (target_point,),
        "target_values": {target_point: np.asarray([target_values[name] for name in names], dtype=np.float64)},
        "target_residuals": {target_point: target_residual},
        "positive_values": positive_values,
        "negative_values": negative_values,
        "anchor_residual": np.asarray(anchor_evaluation.residual[anchor_rows], dtype=np.float64),
        "positive_residual": np.asarray(
            [evaluations[1 + 2 * index].residual[rows[1 + 2 * index]] for index in range(parameters)],
            dtype=np.float64,
        ),
        "negative_residual": np.asarray(
            [evaluations[2 + 2 * index].residual[rows[2 + 2 * index]] for index in range(parameters)],
            dtype=np.float64,
        ),
        "reference_residual": target_residual,
        "covariance": np.asarray(anchor_evaluation.combined_covariance[anchor_rows], dtype=np.float64),
        "run_id": np.asarray(anchor_evaluation.run_id[anchor_rows], dtype=np.int64),
        "event_id": np.asarray(anchor_evaluation.event_id[anchor_rows], dtype=np.int64),
        "truth_particle_id": np.asarray(anchor_evaluation.truth_particle_id[anchor_rows], dtype=np.int64),
        "source_station_id": np.asarray(anchor_evaluation.source_station_id[anchor_rows], dtype=np.int64),
        "target_station_id": np.asarray(anchor_evaluation.target_station_id[anchor_rows], dtype=np.int64),
        "overlap": overlap,
        "layer_weights": _layer_weights_from_hit_pattern(anchor_evaluation, anchor_rows),
    }


def _gauge_recovery(bank: Mapping[str, Any], *, choice: str, rcond: float) -> dict[str, Any]:
    specs = bank["specs"]
    weights = bank["layer_weights"]
    reduction = reduce_gauge(
        specs,
        choice=choice,
        layer_weights=weights,
        reference_layer=0,
        dropped_layer=2,
    )
    reduced_scales = np.asarray([float(bank["scales"][index]) for index in reduction.keep_indices], dtype=np.float64)
    fit = _fit(
        bank,
        rcond=rcond,
        names=reduction.names,
        scales=reduced_scales,
        column_transform=reduction.column_transform,
    )
    recovery = _recovery_from_fit(bank, reduction=reduction, fit=fit)
    pre_rms = _rms(fit.response)
    post_rms = _rms(fit.residual_response)
    return {
        "choice": choice,
        "constraint": reduction.constraint,
        "reduced_parameter_names": list(reduction.names),
        "dropped_parameter_names": list(reduction.dropped_names),
        "layer_weights": [float(value) for value in weights],
        **recovery,
        "fit": _fit_summary(fit, fit.parameter_names),
        "prefit_residual_rms": pre_rms,
        "postfit_residual_rms": post_rms,
        "post_over_pre_rms": (post_rms / pre_rms) if pre_rms > 0.0 and math.isfinite(pre_rms) else float("nan"),
    }


def _station_only_fit(
    jacobian_entry: Mapping[str, object],
    observed_entry: Mapping[str, object],
    *,
    anchor_point: str,
    target_point: str,
    station_parameters: Sequence[str],
    min_truth_match_fraction: float,
    rcond: float,
) -> dict[str, Any]:
    bank = _combined_source_bank(
        jacobian_entry,
        observed_entry,
        anchor_point=anchor_point,
        target_point=target_point,
        only_parameters=station_parameters,
        min_truth_match_fraction=min_truth_match_fraction,
    )
    fit = _fit(bank, rcond=rcond)
    recovered = {
        name: float(fit.recovered_parameters[index]) for index, name in enumerate(fit.parameter_names)
    }
    return {
        "source_id": bank["source_id"],
        "recovered": recovered,
        "expected": _expected_delta(bank),
        "fit": _fit_summary(fit, fit.parameter_names),
        "prefit_residual_rms": _rms(fit.response),
        "postfit_residual_rms": _rms(fit.residual_response),
    }


def _pair_sources(
    jacobian_manifest: Mapping[str, object],
    observed_manifest: Mapping[str, object],
) -> list[tuple[dict[str, object], dict[str, object]]]:
    jacobian = {str(entry["source_id"]): entry for entry in _source_entries(jacobian_manifest)}
    observed = [entry for entry in _source_entries(observed_manifest)]
    missing = [str(entry["source_id"]) for entry in observed if str(entry["source_id"]) not in jacobian]
    if missing:
        raise ValueError("observed sources missing from jacobian bank: " + ", ".join(missing))
    return [(jacobian[str(entry["source_id"])], entry) for entry in observed]


def _close_target(
    pairs: Sequence[tuple[Mapping[str, object], Mapping[str, object]]],
    *,
    target_point: str,
    anchor_point: str,
    full_specs: Sequence[Mapping[str, object]],
    min_truth_match_fraction: float,
    rcond: float,
    max_condition_number: float,
) -> dict[str, Any]:
    sample_values = None
    for _jacobian_entry, observed_entry in pairs:
        observed_root = Path(str(observed_entry["physical_scan_root"])).expanduser().resolve()
        plan = _read_json(observed_root / "scan_plan.json")
        point = _point_map(plan)[target_point]
        values = {str(key): float(value) for key, value in dict(point["alignment_parameter_values"]).items()}
        if sample_values is None:
            sample_values = values
        break
    if sample_values is None:
        raise ValueError(f"no observed source available for '{target_point}'")
    component = injected_linear_component(sample_values, full_specs)
    layer_parameters = _layer_specs_for_component(full_specs, component)
    station_parameters = _station_specs(full_specs)

    layer_banks = [
        _combined_source_bank(
            jacobian_entry,
            observed_entry,
            anchor_point=anchor_point,
            target_point=target_point,
            only_parameters=layer_parameters,
            min_truth_match_fraction=min_truth_match_fraction,
        )
        for jacobian_entry, observed_entry in pairs
    ]
    pooled = _pool_layer_banks(layer_banks)
    gauges = {
        choice: _gauge_recovery(_bank_for_target(pooled, target_point), choice=choice, rcond=rcond)
        for choice in GAUGE_CHOICES
    }
    source_gauges = {
        bank["source_id"]: {
            choice: _gauge_recovery(_bank_for_target(bank, target_point), choice=choice, rcond=rcond)
            for choice in GAUGE_CHOICES
        }
        for bank in layer_banks
    }

    expected_split = gauges["sum_to_zero"]["expected_split"]
    expected_internals = internals_by_layer(expected_split, component)
    recovered_by_gauge = {
        choice: internals_by_layer(payload["recovered_split"], component) for choice, payload in gauges.items()
    }
    source_internals = {
        source_id: {
            choice: internals_by_layer(payload["recovered_split"], component) for choice, payload in by_choice.items()
        }
        for source_id, by_choice in source_gauges.items()
    }
    station_leakage = {
        choice: {
            item: float(gauges[choice]["recovered_split"]["station_common"][item])
            for item in FREE_COMPONENTS
        }
        for choice in GAUGE_CHOICES
    }
    ranks = {
        choice: {
            "full_rank": bool(payload["fit"]["full_rank"]),
            "normal_matrix_rank": payload["fit"]["normal_matrix_rank"],
            "dimension": len(payload["reduced_parameter_names"]),
            "normal_matrix_condition_number": payload["fit"]["normal_matrix_condition_number"],
        }
        for choice, payload in gauges.items()
    }
    residual_ratios = {choice: float(payload["post_over_pre_rms"]) for choice, payload in gauges.items()}

    station_fits = [
        _station_only_fit(
            jacobian_entry,
            observed_entry,
            anchor_point=anchor_point,
            target_point=target_point,
            station_parameters=station_parameters,
            min_truth_match_fraction=min_truth_match_fraction,
            rcond=rcond,
        )
        for jacobian_entry, observed_entry in pairs
    ]
    pooled_station = _pool_layer_banks(
        [
            _combined_source_bank(
                jacobian_entry,
                observed_entry,
                anchor_point=anchor_point,
                target_point=target_point,
                only_parameters=station_parameters,
                min_truth_match_fraction=min_truth_match_fraction,
            )
            for jacobian_entry, observed_entry in pairs
        ]
    )
    station_pooled_fit = _fit(_bank_for_target(pooled_station, target_point), rcond=rcond)
    station_recovered = {
        name: float(station_pooled_fit.recovered_parameters[index])
        for index, name in enumerate(station_pooled_fit.parameter_names)
    }
    station_component_name = {
        "dx_mm": "ift_dx_mm",
        "rx_mrad": "ift_rx_mrad",
        "ry_mrad": "ift_ry_mrad",
    }[component]
    station_value = abs(float(station_recovered.get(station_component_name, 0.0)))
    layer_post = min(float(payload["postfit_residual_rms"]) for payload in gauges.values())
    station_post = _rms(station_pooled_fit.residual_response)
    station_explains = math.isfinite(layer_post) and math.isfinite(station_post) and station_post <= 1.5 * max(layer_post, 1.0e-12)
    station_moved = station_value > float(COMPONENT_THRESHOLDS[component]["station_abs"])
    station_only_leak = bool(station_moved and station_explains)
    if station_only_leak:
        for choice in station_leakage:
            station_leakage[choice][component] = float(station_recovered[station_component_name])

    verdict = physics_verdict(
        component=component,
        expected_internals=expected_internals,
        recovered_by_gauge=recovered_by_gauge,
        source_internals=source_internals,
        station_leakage_by_gauge=station_leakage,
        ranks=ranks,
        residual_ratios=residual_ratios,
        max_condition_number=max_condition_number,
    )
    if station_only_leak:
        verdict["no_station_common_mode_leakage"] = False
        verdict["passed"] = False
        verdict["station_only_diagnostic_failed"] = True
    else:
        verdict["station_only_diagnostic_failed"] = False

    overlap = pooled["overlap"]
    return {
        "target_point": target_point,
        "component": component,
        "layer_parameters": list(layer_parameters),
        "injected_values": {name: float(sample_values[name]) for name in layer_parameters},
        "n_pairs": int(gauges["sum_to_zero"]["fit"]["observations"]),
        "overlap": overlap,
        "gauges": gauges,
        "per_source_gauges": source_gauges,
        "gauge_invariant_internals": {
            "expected": expected_internals,
            "recovered": recovered_by_gauge,
        },
        "station_only_leakage_diagnostic": {
            "recovered": station_recovered,
            "fit": _fit_summary(station_pooled_fit, station_pooled_fit.parameter_names),
            "per_source": station_fits,
            "leaked": station_only_leak,
        },
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jacobian-manifest", required=True)
    parser.add_argument("--observed-manifest", required=True)
    parser.add_argument("--anchor-point", default="iteration_00_reference")
    parser.add_argument("--target-point", action="append", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-truth-match-fraction", type=float, default=0.99)
    parser.add_argument("--rcond", type=float, default=1.0e-10)
    parser.add_argument("--max-condition-number", type=float, default=1.0e4)
    args = parser.parse_args()

    jacobian_path = Path(args.jacobian_manifest).expanduser().resolve()
    observed_path = Path(args.observed_manifest).expanduser().resolve()
    jacobian_manifest = _read_json(jacobian_path)
    observed_manifest = _read_json(observed_path)
    _require_sealed_manifest(jacobian_manifest, label="jacobian manifest")
    _require_sealed_manifest(observed_manifest, label="observed manifest")
    if jacobian_manifest.get("physical_geometry_repropagation") is not True:
        raise ValueError("jacobian manifest is not a physical repropagation bank")
    if observed_manifest.get("physical_geometry_repropagation") is not True:
        raise ValueError("observed manifest is not a physical repropagation bank")

    pairs = _pair_sources(jacobian_manifest, observed_manifest)
    sample_root = Path(str(pairs[0][0]["physical_scan_root"])).expanduser().resolve()
    full_specs = tuple(_read_json(sample_root / "scan_plan.json")["alignment_parameter_specs"])
    observed_iteration = observed_manifest.get("alignment_iteration", {})
    default_targets = observed_iteration.get("held_out_closure_points") if isinstance(observed_iteration, Mapping) else None
    target_points = tuple(args.target_point) if args.target_point else tuple(default_targets or ())
    if not target_points:
        raise ValueError("no held-out target points were provided")

    targets = {
        name: _close_target(
            pairs,
            target_point=name,
            anchor_point=str(args.anchor_point),
            full_specs=full_specs,
            min_truth_match_fraction=float(args.min_truth_match_fraction),
            rcond=float(args.rcond),
            max_condition_number=float(args.max_condition_number),
        )
        for name in target_points
    }
    dx_targets = [name for name, payload in targets.items() if payload["component"] == "dx_mm"]
    rx_targets = [name for name, payload in targets.items() if payload["component"] == "rx_mrad"]
    ry_targets = [name for name, payload in targets.items() if payload["component"] == "ry_mrad"]
    dx_passed = bool(dx_targets) and all(targets[name]["verdict"]["passed"] for name in dx_targets)
    rx_passed = bool(rx_targets) and all(targets[name]["verdict"]["passed"] for name in rx_targets)
    ry_passed = all(targets[name]["verdict"]["passed"] for name in ry_targets) if ry_targets else None
    ready = rx_passed if rx_targets else dx_passed
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "jacobian_manifest": str(jacobian_path),
        "observed_manifest": str(observed_path),
        "anchor_point": str(args.anchor_point),
        "target_points": list(target_points),
        "observation_semantics": "truth_selected_physical_edge",
        "q_over_p_mode": 0,
        "test_data_accessed": False,
        "station_5dof_frozen": True,
        "reused_existing_fd_jacobian": True,
        "fit_representations": list(GAUGE_CHOICES),
        "canonical_internal_basis": CANONICAL_INTERNAL_BASIS,
        "admission_representations": list(ZERO_COMMON_MODE_CHOICES),
        "negative_control_representations": [FROZEN_STATION_NEGATIVE_CONTROL],
        "forbidden_directions": ["layer_dy_mm", "layer_rz_mrad", "layer1_rx_mrad"],
        "sources": [str(observed["source_id"]) for _jacobian, observed in pairs],
        "targets": targets,
        "dx_passed": dx_passed if dx_targets else None,
        "rx_passed": rx_passed if rx_targets else None,
        "ry_passed": ry_passed,
        "ready_for_route_selected": ready,
        "answer": "yes" if ready else "no",
        "question": (
            "Without station rigid motion and without dy/rz nonlinear contamination, "
            "does real FASER reconstruction stably recover a gauge-defined IFT internal relative deformation?"
        ),
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "layer_linear_internal_closure.json"
    output_path.write_text(json.dumps(_json_ready(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"dx_passed: {report['dx_passed']}")
    print(f"rx_passed: {report['rx_passed']}")
    print(f"ry_passed: {report['ry_passed']}")
    print(f"ready_for_route_selected: {ready}")
    for name, payload in targets.items():
        verdict = payload["verdict"]
        print(
            f"{name} {payload['component']}: passed={verdict['passed']} "
            f"pairs={payload['n_pairs']} recovered={verdict['physics_recovered']} "
            f"gauges_agree={verdict['gauges_agree']} leakage_free={verdict['no_station_common_mode_leakage']}"
        )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
