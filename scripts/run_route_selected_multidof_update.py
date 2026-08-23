#!/usr/bin/env python3
"""Solve one truth-free, route-selected physical multi-DoF alignment update.

Inputs are outputs of the frozen V1/V2 association backbone for an anchor
payload, its real central finite-difference probes, and a separately refitted
reference target.  The selected routes are intersected only by original
tracklet provenance.  By default the update consumes exact field-aware mode-0
ACTS residuals carried by selected route edges.  The older straight-line
leave-one-out table remains available only as an explicit diagnostic option;
it is not the default alignment objective in a magnetic field.

The default target is intended for an MC closure scan whose nominal payload is
known.  It is deliberately not labelled as a real-data correction: a data
alignment loop needs a collaboration-validated residual target and conditions
sign convention before the proposed payload may be deployed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from alignment.capture_criteria import attach_capture_and_prior, load_capture_criteria
from alignment.calibration_modes import (
    C_DX as MODE_C_DX,
    DEFAULT_CONTRACT_RELATIVE,
    IFT_INTERNAL_MODE,
    STATION_MODE,
    evaluate_mode_validity,
    load_mode_validity_contract,
    normalize_mode,
    read_station_framework_capture_success,
)
from alignment.layer_hierarchy import (
    HIERARCHY_FIT_CHOICES,
    IFT_LAYER_IDS,
    expand_gauged_parameters,
    reduce_gauge,
    spec_scope,
    split_common_and_internal,
)
from alignment.physical_jacobian import (
    parameter_values_from_payload,
    parameter_values_from_station_transforms,
    payload_transforms_with_parameter_values,
    solve_physical_finite_difference,
    station_transforms_with_parameter_values,
)
from alignment.route_selected_update import (
    align_route_selected_observations,
    apply_observation_statistics,
    read_anchor_selected_field_edge_observations,
    read_route_selected_field_edge_observations,
    read_route_selected_observations,
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact is not a mapping: {path}")
    return dict(payload)


def _points(plan: Mapping[str, object]) -> dict[str, dict[str, object]]:
    raw = plan.get("points")
    if not isinstance(raw, list) or not raw:
        raise ValueError("physical scan plan has no points")
    result: dict[str, dict[str, object]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("physical scan point is not a mapping")
        name = str(item.get("name", ""))
        if not name or name in result:
            raise ValueError("physical scan has invalid/duplicate point names")
        result[name] = dict(item)
    return result


def _specs(plan: Mapping[str, object]) -> list[dict[str, object]]:
    raw = plan.get("alignment_parameter_specs")
    if not isinstance(raw, list) or not raw:
        raise ValueError("physical scan lacks alignment_parameter_specs")
    result = [dict(item) for item in raw if isinstance(item, Mapping)]
    if len(result) != len(raw):
        raise ValueError("alignment_parameter_specs entries must be mappings")
    names = [str(item.get("name", "")) for item in result]
    if not all(names) or len(set(names)) != len(names):
        raise ValueError("alignment_parameter_specs has invalid names")
    return result


def _parameter_paths(values: Sequence[str] | None, names: Sequence[str], *, label: str) -> dict[str, Path]:
    if not values:
        raise ValueError(f"{label} must specify exactly one PATH for every parameter")
    result: dict[str, Path] = {}
    for raw in values:
        name, separator, raw_path = str(raw).partition(":")
        if not separator or not name or not raw_path:
            raise ValueError(f"{label} entries must have form PARAMETER:PATH")
        if name in result:
            raise ValueError(f"{label} repeats parameter '{name}'")
        result[name] = Path(raw_path).expanduser().resolve()
    if set(result) != set(names):
        raise ValueError(f"{label} must specify exactly: " + ", ".join(names))
    return result


def _name_values(
    values: Sequence[str] | None,
    names: Sequence[str],
    *,
    label: str,
    require_all: bool = True,
) -> np.ndarray | None:
    if not values:
        return None
    parsed: dict[str, float] = {}
    for raw in values:
        name, separator, raw_value = str(raw).partition(":")
        if not separator or not name:
            raise ValueError(f"{label} entries must have form PARAMETER:VALUE")
        if name in parsed:
            raise ValueError(f"{label} repeats parameter '{name}'")
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{label} parameter '{name}' must be finite and non-negative")
        parsed[name] = value
    unknown = set(parsed) - set(names)
    if unknown:
        raise ValueError(f"{label} has unknown parameter(s): " + ", ".join(sorted(unknown)))
    if require_all and set(parsed) != set(names):
        raise ValueError(f"{label} must specify exactly: " + ", ".join(names))
    return np.asarray([parsed[name] if name in parsed else math.nan for name in names], dtype=np.float64)


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = sorted({str(key) for row in rows for key in row}) or ["empty"]
    data = list(rows) if rows else [{"empty": ""}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def _validate_frozen_output(
    path: Path,
    *,
    expected_point: str,
    movable: Sequence[int],
    observation_kind: str,
    q_over_p_mode: int = 0,
):
    summary_path = path / "association_summary.json"
    observation_file = {
        "field_edge": "selected_route_field_edge_residuals.csv",
        "leave_one_out": "selected_route_leave_one_out_residuals.csv",
    }.get(observation_kind)
    if observation_file is None:  # pragma: no cover - argparse constrains this
        raise ValueError(f"unsupported observation kind '{observation_kind}'")
    observation_path = path / observation_file
    if not summary_path.is_file() or not observation_path.is_file():
        raise FileNotFoundError(f"frozen association output is incomplete: {path}")
    summary = _read_json(summary_path)
    if summary.get("test_opened") is not False or summary.get("architecture_or_threshold_tuning") is not False:
        raise ValueError("route-selected update requires a frozen association output sealed from test/tuning")
    if summary.get("physical_geometry_repropagation") is not True or int(
        summary.get("q_over_p_mode", -1)
    ) != int(q_over_p_mode):
        raise ValueError(
            f"route-selected update requires mode-{q_over_p_mode} physical association output"
        )
    # This scan's synthetic materialization has exactly one payload per
    # backbone run.  Check the table directly so a point cannot accidentally
    # be paired with association results from another real conditions payload.
    with observation_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        payload_ids = {str(row.get("payload_id", "")) for row in reader if row.get("payload_id")}
    if payload_ids and payload_ids != {expected_point}:
        raise ValueError(
            f"frozen association output {path} has payload IDs {sorted(payload_ids)}, expected '{expected_point}'"
        )
    reader = (
        read_route_selected_field_edge_observations
        if observation_kind == "field_edge"
        else read_route_selected_observations
    )
    return summary, reader(observation_path, movable_station_ids=movable)


def _load_anchor_selected_payload_bank(
    path: Path,
    *,
    expected_point: str,
    movable: Sequence[int],
    anchor_table: Path,
    covariance_calibration: Any = None,
    q_over_p_mode: int = 0,
    require_mc_labels: bool = True,
):
    """Load one payload's candidate graph and re-measure the anchor's route set in it."""
    config_path = path / "resolved_config.json"
    synthetic_path = path / "synthetic_tracklets.root"
    candidates_path = path / "field_candidates.root"
    if not config_path.is_file() or not synthetic_path.is_file() or not candidates_path.is_file():
        raise FileNotFoundError(f"materialized payload sample is incomplete: {path}")
    config = _read_json(config_path)
    if str(config.get("payload_id")) != expected_point:
        raise ValueError(
            f"payload sample {path} has payload_id '{config.get('payload_id')}', expected '{expected_point}'"
        )
    if config.get("physical_geometry_repropagation") is not True or int(
        config.get("q_over_p_mode", -1)
    ) != int(q_over_p_mode):
        raise ValueError(
            f"anchor-selected update requires a mode-{q_over_p_mode} physical payload sample"
        )
    bank, audit = read_anchor_selected_field_edge_observations(
        anchor_table,
        synthetic_path,
        candidates_path,
        movable_station_ids=movable,
        covariance_calibration=covariance_calibration,
        q_over_p_mode=q_over_p_mode,
        require_mc_labels=require_mc_labels,
    )
    summary = {
        "payload_sample": str(path),
        "payload_id": expected_point,
        "observation_kind": "anchor_selected_field_edge",
        "anchor_edge_audit": audit,
        "covariance_calibration": (
            None if covariance_calibration is None else dict(covariance_calibration.to_json()["provenance"])
        ),
        "test_opened": False,
        "architecture_or_threshold_tuning": False,
        "physical_geometry_repropagation": True,
        "q_over_p_mode": int(q_over_p_mode),
    }
    return summary, bank


def _probe_for_parameter(points: Mapping[str, Mapping[str, object]], name: str, sign: str, anchor: str) -> str:
    matches = [
        point_name
        for point_name, point in points.items()
        if point.get("finite_difference_for") == name
        and point.get("probe_sign") == sign
        and point.get("finite_difference_anchor") == anchor
    ]
    if len(matches) != 1:
        raise ValueError(
            f"scan plan needs exactly one {sign} finite-difference probe for '{name}' around '{anchor}'"
        )
    return matches[0]


def _role_counts(bank) -> dict[str, int]:
    result: dict[str, int] = {}
    for observation in bank.values():
        key = str(observation.target_synthetic_role)
        result[key] = result.get(key, 0) + 1
    return result


def _select_parameter_specs(
    specs: Sequence[Mapping[str, object]],
    only_parameters: Sequence[str] | None,
) -> list[dict[str, object]]:
    if only_parameters is None:
        return [dict(spec) for spec in specs]
    requested = tuple(str(name) for name in only_parameters)
    names = [str(spec["name"]) for spec in specs]
    unknown = set(requested) - set(names)
    if len(set(requested)) != len(requested):
        raise ValueError("duplicated --only-parameters")
    if unknown:
        raise ValueError("unknown --only-parameters: " + ", ".join(sorted(unknown)))
    by_name = {str(spec["name"]): dict(spec) for spec in specs}
    return [by_name[name] for name in requested]


def _point_parameter_values(
    specs: Sequence[Mapping[str, object]],
    point: Mapping[str, object],
) -> dict[str, float]:
    stations = point.get("injected_station_transforms")
    if not isinstance(stations, Mapping):
        raise ValueError(f"physical point '{point.get('name')}' lacks injected_station_transforms")
    if any(spec_scope(spec) in {"layer", "contrast"} for spec in specs):
        layers = point.get("injected_layer_transforms")
        if not isinstance(layers, Mapping):
            raise ValueError(f"physical point '{point.get('name')}' lacks injected_layer_transforms")
        return parameter_values_from_payload(specs, stations, layers)
    return parameter_values_from_station_transforms(specs, stations)


def _gauged_finite_difference_fit(
    *,
    anchor_residual,
    positive_residual,
    negative_residual,
    target_residual,
    covariance,
    names: Sequence[str],
    positive_values: Sequence[float],
    negative_values: Sequence[float],
    scales,
    prior,
    rcond: float,
    reduction,
):
    unconstrained = solve_physical_finite_difference(
        anchor_residual,
        positive_residual,
        negative_residual,
        target_residual,
        covariance,
        parameter_names=tuple(names),
        positive_values=positive_values,
        negative_values=negative_values,
        parameter_scales=scales,
        prior_sigma_native=prior,
        rcond=rcond,
    )
    derivative = unconstrained.derivative_native @ reduction.column_transform
    step = 1.0
    reduced_positive = anchor_residual[None, :, :] + step * np.moveaxis(derivative, 2, 0)
    reduced_negative = anchor_residual[None, :, :] - step * np.moveaxis(derivative, 2, 0)
    reduced_scales = np.asarray(
        [float(scales[index]) for index in reduction.keep_indices],
        dtype=np.float64,
    )
    reduced_prior = None
    if prior is not None:
        reduced_prior = np.asarray(
            [float(prior[index]) for index in reduction.keep_indices],
            dtype=np.float64,
        )
    gauged = solve_physical_finite_difference(
        anchor_residual,
        reduced_positive,
        reduced_negative,
        target_residual,
        covariance,
        parameter_names=reduction.names,
        positive_values=np.full(len(reduction.names), step),
        negative_values=np.full(len(reduction.names), -step),
        parameter_scales=reduced_scales,
        prior_sigma_native=reduced_prior,
        rcond=rcond,
    )
    return unconstrained, gauged


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(array))))


def _layer_component(specs: Sequence[Mapping[str, object]]) -> str:
    components = {str(spec["component"]) for spec in specs if spec_scope(spec) == "layer"}
    if len(components) != 1:
        raise ValueError("layer-internal route-selected updates require exactly one layer component")
    return next(iter(components))


def _component_internals(
    values: Mapping[str, float],
    specs: Sequence[Mapping[str, object]],
    component: str,
) -> dict[str, object]:
    split = split_common_and_internal(values, specs)
    internals = {
        f"layer_{layer}": float(split["layer_internal"][f"layer_{layer}"][component]) for layer in IFT_LAYER_IDS
    }
    return {
        "layer_internal": internals,
        "outer_relative_layer0_minus_layer2": float(internals["layer_0"] - internals["layer_2"]),
        "station_common": {key: float(value) for key, value in split["station_common"].items()},
        "layer_weighted_mean": {key: float(value) for key, value in split["layer_weighted_mean"].items()},
    }


def _six_vectors_close(left: Mapping[str, object], right: Mapping[str, object], *, atol: float = 1.0e-12) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        a = np.asarray(left[key], dtype=np.float64)
        b = np.asarray(right[key], dtype=np.float64)
        if a.shape != (6,) or b.shape != (6,) or not np.allclose(a, b, rtol=0.0, atol=atol):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", required=True)
    parser.add_argument(
        "--target-scan-root",
        default=None,
        help="Optional second physical scan that contains only the held-out target point.",
    )
    parser.add_argument("--anchor-point", required=True)
    parser.add_argument("--anchor-association-output", required=True)
    parser.add_argument("--target-point", default=None)
    parser.add_argument("--target-association-output", default=None)
    parser.add_argument(
        "--self-nulling-update",
        action="store_true",
        help=(
            "Real-data Gauss-Newton: drive current residuals toward zero.  Forbids a "
            "known target payload and does not interpret capture vs an injected truth."
        ),
    )
    parser.add_argument(
        "--allow-real-data",
        action="store_true",
        help="Load identity real-data samples without MC truth labels.",
    )
    parser.add_argument("--positive-association-output", action="append", default=None, metavar="PARAMETER:PATH")
    parser.add_argument("--negative-association-output", action="append", default=None, metavar="PARAMETER:PATH")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--only-parameters",
        nargs="+",
        default=None,
        help="Restrict the update to these named parameters (required for layer-internal hierarchy scans).",
    )
    parser.add_argument(
        "--gauge",
        choices=HIERARCHY_FIT_CHOICES,
        default=None,
        help="Explicit IFT layer gauge or outer-contrast basis. Required when floating layer internals.",
    )
    parser.add_argument(
        "--allow-nominal-anchor",
        action="store_true",
        help="Permit a nominal/reference payload as the FD anchor for an MC held-out closure.",
    )
    parser.add_argument(
        "--observation-kind",
        choices=("field_edge", "leave_one_out", "anchor_selected_field_edge"),
        default="field_edge",
        help=(
            "Use exact selected mode-0 ACTS edges (default), the legacy straight-line leave-one-out "
            "diagnostic table, or anchor-selected edges recomputed in every payload's candidate graph. "
            "The anchor-selected mode keeps the anchor's truth-free route set fixed and looks the same "
            "origins up in each payload; target/probe arguments then point at the materialized per-payload "
            "sample directories (synthetic_tracklets.root + field_candidates.root) instead of per-payload "
            "backbone outputs."
        ),
    )
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument(
        "--q-over-p-mode",
        type=int,
        default=0,
        choices=(0, 3),
        help=(
            "Propagation record variant of every association input. Mode 0 is the canonical "
            "production baseline; mode 3 is the fixed-q/p-seed covariance-suppression "
            "diagnostic variant. All inputs must carry the same mode."
        ),
    )
    parser.add_argument("--prior-sigma", action="append", default=None, metavar="PARAMETER:VALUE")
    parser.add_argument("--capture-tolerance", action="append", default=None, metavar="PARAMETER:VALUE")
    parser.add_argument(
        "--capture-criteria",
        default=None,
        help="Pre-registered dual engineering+pull capture JSON. Validation must not have selected it.",
    )
    parser.add_argument(
        "--calibration-mode",
        choices=(STATION_MODE, IFT_INTERNAL_MODE, "C_dx"),
        default=None,
        help="Exclusive production mode: station or ift_internal. Required to write geometry under the V1 protocol.",
    )
    parser.add_argument(
        "--mode-validity-contract",
        default=None,
        help="Frozen mode-validity JSON from workbook 45. Required when --calibration-mode is set.",
    )
    parser.add_argument(
        "--declared-unmodeled-cdx-um",
        type=float,
        default=None,
        help="Potential unmodeled |C_dx| in micrometres for Station Mode (survey/dedicated-cal bound).",
    )
    parser.add_argument(
        "--cdx-fixed-by",
        choices=("external_geometry", "dedicated_calibration", "isolation_zero"),
        default=None,
        help="How IFT internal C_dx was closed before Station Mode.",
    )
    parser.add_argument(
        "--station-capture-artifact",
        default=None,
        help="Prior station-mode JSON whose framework capture must be true before C_dx Mode.",
    )
    parser.add_argument(
        "--station-framework-capture-success",
        choices=("true", "false"),
        default=None,
        help="Explicit workbook-36 framework capture result for C_dx Mode (not the same data stage).",
    )
    parser.add_argument(
        "--same-data-stage-as-other-mode",
        action="store_true",
        help="Mark a forbidden same-stage station/C_dx iteration. Production must not pass this.",
    )
    parser.add_argument(
        "--require-mode-valid",
        action="store_true",
        help="Refuse to finish if the mode-validity contract rejects geometry write.",
    )
    parser.add_argument("--rcond", type=float, default=1.0e-10)
    parser.add_argument("--require-full-rank", action="store_true")
    parser.add_argument(
        "--covariance-calibration",
        default=None,
        help=(
            "frozen train-only diagonal covariance calibration JSON applied to the "
            "propagated covariances of every payload before candidate lookup; only "
            "valid with --observation-kind anchor_selected_field_edge"
        ),
    )
    parser.add_argument(
        "--huber-k",
        type=float,
        default=None,
        help=(
            "a priori fixed Huber threshold on the anchor edge Mahalanobis chi2; "
            "edges above k^2 are down-weighted as w=k/sqrt(chi2) in the WLS solve"
        ),
    )
    parser.add_argument(
        "--anchor-payload-sample",
        default=None,
        help=(
            "materialized anchor payload sample; with anchor_selected_field_edge the "
            "anchor bank is then also re-measured from the candidate graph (required "
            "for a covariance calibration to reach the WLS weight, which is the "
            "anchor covariance) instead of read from the frozen backbone CSV"
        ),
    )
    parser.add_argument(
        "--observation-statistics",
        choices=("replica_weighted", "physical_edge_deduplicated", "physical_edge_inverse_multiplicity_weighted"),
        default="replica_weighted",
        help=(
            "statistical semantics for overlay-reused physical edges; "
            "replica_weighted is the legacy control, the other two normalize each "
            "physical edge's total weight to one and should be numerically close"
        ),
    )
    args = parser.parse_args()
    if args.self_nulling_update:
        if args.target_point or args.target_association_output:
            parser.error("--self-nulling-update forbids --target-point / --target-association-output")
        args.allow_nominal_anchor = True
        args.allow_real_data = True
    elif not args.target_point or not args.target_association_output:
        parser.error("provide --target-point and --target-association-output, or --self-nulling-update")
    if args.self_nulling_update and args.observation_kind != "anchor_selected_field_edge":
        parser.error("--self-nulling-update requires --observation-kind anchor_selected_field_edge")
    if args.capture_criteria is not None and args.capture_tolerance is not None:
        parser.error("provide only one of --capture-criteria or --capture-tolerance")
    if not math.isfinite(args.damping) or not 0.0 < args.damping <= 1.0:
        parser.error("--damping must lie in (0, 1]")
    if not math.isfinite(args.rcond) or not 0.0 < args.rcond < 1.0:
        parser.error("--rcond must lie in (0, 1)")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    scan_root = Path(args.scan_root).expanduser().resolve()
    plan = _read_json(scan_root / "scan_plan.json")
    scan_mode = str(plan.get("scan_mode", ""))
    if int(plan.get("q_over_p_mode", -1)) != 0:
        raise ValueError("route-selected update requires a mode-0 physical scan")
    if scan_mode not in {"station_rigid_multidof", "ift_layer_hierarchy"}:
        raise ValueError("route-selected update requires a station_rigid_multidof or ift_layer_hierarchy scan")
    if scan_mode == "ift_layer_hierarchy" and not args.only_parameters:
        if any(spec_scope(spec) != "contrast" for spec in _specs(plan)):
            raise ValueError("layer-internal route-selected updates require --only-parameters")
    if scan_mode == "station_rigid_multidof" and args.gauge is not None:
        raise ValueError("station-only route-selected updates do not take --gauge")
    specs = _select_parameter_specs(_specs(plan), args.only_parameters)
    fit_basis = str(plan.get("fit_basis", ""))
    from alignment.hierarchical_v1 import is_hierarchical_v1_specs

    hierarchical_v1 = fit_basis == "hierarchical_v1" or is_hierarchical_v1_specs(_specs(plan))
    if hierarchical_v1:
        floated_names = [str(spec["name"]) for spec in specs]
        has_cdx = "C_dx" in floated_names
        has_station = any(spec_scope(spec) == "station" for spec in specs)
        if has_cdx and has_station:
            raise ValueError("hierarchical V1 forbids a joint station+C_dx Newton step")
        if has_cdx and len(floated_names) != 1:
            raise ValueError("hierarchical V1 C_dx step floats only C_dx")
    if any(spec_scope(spec) == "layer" and str(spec["component"]) in {"dy_mm", "rz_mrad"} for spec in specs):
        raise ValueError("layer dy/rz stay frozen; omit them from --only-parameters")
    if any(spec_scope(spec) == "contrast" and str(spec["component"]) not in {"dx_mm", "rx_mrad"} for spec in specs):
        raise ValueError("2D contrast updates admit only C_dx and C_rx")
    has_layer = any(spec_scope(spec) == "layer" for spec in specs)
    has_contrast = any(spec_scope(spec) == "contrast" for spec in specs)
    if has_layer and has_contrast:
        raise ValueError("do not mix outer-contrast coordinates with layer labels")
    if has_layer and args.gauge is None:
        raise ValueError("layer-internal route-selected updates require --gauge")
    if args.gauge is not None and not has_layer:
        raise ValueError("--gauge is only valid when floating IFT layer internals")
    if has_contrast and args.gauge is not None:
        raise ValueError("C_dx/C_rx are already outer-contrast coordinates; do not pass --gauge")
    has_hierarchy = has_layer or has_contrast
    names = [str(spec["name"]) for spec in specs]
    scales = np.asarray([float(spec["severity_scale"]) for spec in specs], dtype=np.float64)
    movable = [int(value) for value in plan.get("movable_station_ids", ())]
    if not movable:
        raise ValueError("physical scan has no movable station")
    points = _points(plan)
    if args.target_scan_root is not None:
        target_plan = _read_json(Path(args.target_scan_root).expanduser().resolve() / "scan_plan.json")
        if int(target_plan.get("q_over_p_mode", -1)) != 0:
            raise ValueError("target scan is not mode-0")
        if str(target_plan.get("scan_mode", "")) not in {"station_rigid_multidof", "ift_layer_hierarchy"}:
            raise ValueError("target scan is not a physical hierarchy or station-rigid scan")
        points.update(_points(target_plan))
    anchor_point = points.get(str(args.anchor_point))
    if args.self_nulling_update:
        target_point = dict(anchor_point) if isinstance(anchor_point, Mapping) else None
        if target_point is not None:
            target_point["name"] = str(args.anchor_point)
            target_point["point_role"] = "self_nulling_zero_residual"
    else:
        target_point = points.get(str(args.target_point))
    if anchor_point is None or target_point is None:
        raise ValueError("anchor-point and target-point must be present in the frozen scan plan")
    calibration_mode = None if args.calibration_mode is None else normalize_mode(args.calibration_mode)
    mode_contract = None
    mode_contract_path = None
    if calibration_mode is not None:
        from alignment.calibration_modes import assert_exclusive_parameters

        mode_contract_path = (
            Path(args.mode_validity_contract).expanduser().resolve()
            if args.mode_validity_contract
            else Path(__file__).resolve().parents[1] / DEFAULT_CONTRACT_RELATIVE
        )
        mode_contract = load_mode_validity_contract(mode_contract_path)
        assert_exclusive_parameters(calibration_mode, names)
    if (
        anchor_point.get("point_role") == "nominal"
        and not args.allow_nominal_anchor
        and target_point.get("point_role") != "held_out_closure"
    ):
        raise ValueError("anchor-point must be a non-reference physical payload")
    positive_paths = _parameter_paths(args.positive_association_output, names, label="--positive-association-output")
    negative_paths = _parameter_paths(args.negative_association_output, names, label="--negative-association-output")
    positive_points = {
        name: _probe_for_parameter(points, name, "positive", str(args.anchor_point)) for name in names
    }
    negative_points = {
        name: _probe_for_parameter(points, name, "negative", str(args.anchor_point)) for name in names
    }
    anchor_summary, anchor_bank = _validate_frozen_output(
        Path(args.anchor_association_output).expanduser().resolve(),
        expected_point=str(args.anchor_point),
        movable=movable,
        observation_kind="field_edge" if args.observation_kind == "anchor_selected_field_edge" else args.observation_kind,
        q_over_p_mode=int(args.q_over_p_mode),
    )
    if args.observation_kind == "anchor_selected_field_edge":
        anchor_table = (
            Path(args.anchor_association_output).expanduser().resolve()
            / "selected_route_field_edge_residuals.csv"
        )
        covariance_calibration = None
        if args.covariance_calibration is not None:
            from alignment.covariance_calibration import load_covariance_calibration

            covariance_calibration = load_covariance_calibration(args.covariance_calibration)
    elif args.covariance_calibration is not None:
        raise ValueError(
            "--covariance-calibration is only supported with "
            "--observation-kind anchor_selected_field_edge"
        )

    if args.observation_kind == "anchor_selected_field_edge":

        def _payload_bank(path: Path, expected_point: str):
            return _load_anchor_selected_payload_bank(
                path,
                expected_point=expected_point,
                movable=movable,
                anchor_table=anchor_table,
                covariance_calibration=covariance_calibration,
                q_over_p_mode=int(args.q_over_p_mode),
                require_mc_labels=not args.allow_real_data,
            )

        if args.anchor_payload_sample is not None:
            # The WLS weight is the anchor covariance; re-measuring the anchor
            # bank from its own candidate graph is what lets a frozen covariance
            # calibration reach the solve.  The route set stays exactly the
            # frozen backbone's anchor selection.
            _anchor_sample_summary, anchor_bank = _payload_bank(
                Path(args.anchor_payload_sample).expanduser().resolve(), str(args.anchor_point)
            )
        if args.self_nulling_update:
            target_summary, target_bank = None, None
        else:
            target_summary, target_bank = _payload_bank(
                Path(args.target_association_output).expanduser().resolve(), str(args.target_point)
            )
        positive_summaries = {}
        negative_summaries = {}
        positive_banks = {}
        negative_banks = {}
        for name in names:
            positive_summaries[name], positive_banks[name] = _payload_bank(
                positive_paths[name], positive_points[name]
            )
            negative_summaries[name], negative_banks[name] = _payload_bank(
                negative_paths[name], negative_points[name]
            )
    else:
        target_summary, target_bank = _validate_frozen_output(
            Path(args.target_association_output).expanduser().resolve(),
            expected_point=str(args.target_point),
            movable=movable,
            observation_kind=args.observation_kind,
            q_over_p_mode=int(args.q_over_p_mode),
        )
        positive_summaries = {}
        negative_summaries = {}
        positive_banks = {}
        negative_banks = {}
        for name in names:
            positive_summaries[name], positive_banks[name] = _validate_frozen_output(
                positive_paths[name],
                expected_point=positive_points[name],
                movable=movable,
                observation_kind=args.observation_kind,
                q_over_p_mode=int(args.q_over_p_mode),
            )
            negative_summaries[name], negative_banks[name] = _validate_frozen_output(
                negative_paths[name],
                expected_point=negative_points[name],
                movable=movable,
                observation_kind=args.observation_kind,
                q_over_p_mode=int(args.q_over_p_mode),
            )
    # The order is fixed: anchor, p/m for each parameter, then target.  The
    # same provenance intersection is used for every derivative and response.
    banks = [anchor_bank]
    for name in names:
        banks.extend((positive_banks[name], negative_banks[name]))
    if not args.self_nulling_update:
        banks.append(target_bank)
    keys, residuals, covariances, overlap = align_route_selected_observations(banks)
    physical_edge_keys = [anchor_bank[key].physical_edge_key for key in keys]
    keys, residuals, covariances, observation_statistics_audit = apply_observation_statistics(
        keys, residuals, covariances, physical_edge_keys, str(args.observation_statistics)
    )
    anchor_residual = residuals[0]
    anchor_covariance = covariances[0]
    huber_summary = None
    if args.huber_k is not None:
        huber_k = float(args.huber_k)
        if not np.isfinite(huber_k) or huber_k <= 0.0:
            raise ValueError("--huber-k must be positive")
        # One-step M-estimation: down-weight anchor edges whose Mahalanobis
        # chi2 exceeds k^2 by inflating their WLS covariance as C/w.  The
        # threshold k is fixed a priori, never tuned on the split under study.
        try:
            inverse = np.linalg.inv(anchor_covariance)
        except np.linalg.LinAlgError:
            inverse = np.asarray(
                [np.linalg.pinv(block) for block in anchor_covariance], dtype=np.float64
            )
        chi2 = np.einsum("ni,nij,nj->n", anchor_residual, inverse, anchor_residual)
        weights = np.minimum(1.0, huber_k / np.sqrt(np.maximum(chi2, 1.0e-300)))
        anchor_covariance = anchor_covariance / weights[:, None, None]
        huber_summary = {
            "k": huber_k,
            "downweighted_edges": int(np.sum(weights < 1.0)),
            "total_edges": int(weights.size),
            "min_weight": float(np.min(weights)),
            "median_weight": float(np.median(weights)),
        }
    positive_residual = np.asarray([residuals[1 + 2 * index] for index in range(len(names))])
    negative_residual = np.asarray([residuals[2 + 2 * index] for index in range(len(names))])
    anchor_values = _point_parameter_values(specs, anchor_point)
    if args.self_nulling_update:
        target_residual = np.zeros_like(anchor_residual)
        target_bank = anchor_bank
        target_values = dict(anchor_values)
    else:
        target_residual = residuals[-1]
        target_values = _point_parameter_values(specs, target_point)
    positive_values = [
        _point_parameter_values(specs, points[positive_points[name]])[name] for name in names
    ]
    negative_values = [
        _point_parameter_values(specs, points[negative_points[name]])[name] for name in names
    ]
    prior = _name_values(args.prior_sigma, names, label="--prior-sigma", require_all=False)
    tolerance = _name_values(args.capture_tolerance, names, label="--capture-tolerance")
    criteria = None if args.capture_criteria is None else load_capture_criteria(Path(args.capture_criteria).expanduser().resolve())
    unconstrained_fit = None
    reduction = None
    if args.gauge is None:
        fit = solve_physical_finite_difference(
            anchor_residual,
            positive_residual,
            negative_residual,
            target_residual,
            anchor_covariance,
            parameter_names=names,
            positive_values=positive_values,
            negative_values=negative_values,
            parameter_scales=scales,
            prior_sigma_native=prior,
            rcond=float(args.rcond),
        )
        recovered_delta = fit.recovered_parameters
        covariance_native = fit.covariance_native
        correlation_native = fit.correlation_native
        derivative_native = fit.derivative_native
    else:
        reduction = reduce_gauge(specs, choice=str(args.gauge))
        unconstrained_fit, fit = _gauged_finite_difference_fit(
            anchor_residual=anchor_residual,
            positive_residual=positive_residual,
            negative_residual=negative_residual,
            target_residual=target_residual,
            covariance=anchor_covariance,
            names=names,
            positive_values=positive_values,
            negative_values=negative_values,
            scales=scales,
            prior=prior,
            rcond=float(args.rcond),
            reduction=reduction,
        )
        recovered_reduced = {
            name: float(fit.recovered_parameters[index]) for index, name in enumerate(fit.parameter_names)
        }
        recovered_full = expand_gauged_parameters(recovered_reduced, reduction, specs)
        recovered_delta = np.asarray([recovered_full[name] for name in names], dtype=np.float64)
        covariance_native = reduction.column_transform @ fit.covariance_native @ reduction.column_transform.T
        covariance_native = 0.5 * (covariance_native + covariance_native.T)
        std = np.sqrt(np.clip(np.diag(covariance_native), 0.0, None))
        correlation_native = np.full(covariance_native.shape, np.nan, dtype=np.float64)
        valid = std > 0.0
        if np.any(valid):
            correlation_native[np.ix_(valid, valid)] = covariance_native[np.ix_(valid, valid)] / np.outer(
                std[valid], std[valid]
            )
        derivative_native = unconstrained_fit.derivative_native
    if args.require_full_rank and not fit.full_rank:
        raise RuntimeError("route-selected physical update normal matrix is rank deficient")
    current = np.asarray([anchor_values[name] for name in names], dtype=np.float64)
    target = np.asarray([target_values[name] for name in names], dtype=np.float64)
    expected_delta = target - current
    proposed = current + float(args.damping) * recovered_delta
    proposal_values = {name: float(proposed[index]) for index, name in enumerate(names)}
    anchor_transforms = anchor_point.get("injected_station_transforms")
    if not isinstance(anchor_transforms, Mapping):
        raise ValueError("anchor point lacks injected_station_transforms")
    proposed_layer_transforms = None
    if has_hierarchy:
        anchor_layers = anchor_point.get("injected_layer_transforms")
        if not isinstance(anchor_layers, Mapping):
            raise ValueError("anchor point lacks injected_layer_transforms")
        proposed_transforms, proposed_layer_transforms = payload_transforms_with_parameter_values(
            specs, anchor_transforms, anchor_layers, proposal_values
        )
    else:
        proposed_transforms = station_transforms_with_parameter_values(specs, anchor_transforms, proposal_values)
    delta_error = recovered_delta - expected_delta
    parameter_rows: list[dict[str, object]] = []
    observability = (
        unconstrained_fit.parameter_observability_fraction if unconstrained_fit is not None else fit.parameter_observability_fraction
    )
    for index, (name, spec) in enumerate(zip(names, specs)):
        variance = float(covariance_native[index, index])
        sigma = math.sqrt(variance) if math.isfinite(variance) and variance >= 0.0 else None
        parameter_rows.append(
            {
                "name": name,
                "station_id": int(spec["station_id"]),
                "layer_id": None if "layer_id" not in spec else int(spec["layer_id"]),
                "scope": spec_scope(spec),
                "component": str(spec["component"]),
                "unit": str(spec["unit"]),
                "anchor_value": float(current[index]),
                "target_value": float(target[index]),
                "expected_delta_to_target": float(expected_delta[index]),
                "recovered_local_delta": float(recovered_delta[index]),
                "local_delta_error": float(delta_error[index]),
                "proposed_next_value": float(proposed[index]),
                "recovered_sigma": sigma,
                "observability_fraction": float(observability[index]),
                "capture_tolerance": None if tolerance is None else float(tolerance[index]),
                "capture_success": (
                    None if tolerance is None else bool(abs(delta_error[index]) <= tolerance[index])
                ),
            }
        )
    sigmas = [row["recovered_sigma"] for row in parameter_rows]
    if args.self_nulling_update:
        capture_aggregate = {
            "applicable": False,
            "reason": "real_data_self_nulling_has_no_injected_target",
            "residual_reduction_is_not_alignment_success": True,
        }
        prior_rows = []
        for row in parameter_rows:
            row["capture_success"] = None
            row["expected_delta_to_target"] = None
            row["local_delta_error"] = None
            row["target_value"] = None
    else:
        parameter_rows, capture_aggregate, prior_rows = attach_capture_and_prior(
            parameter_rows,
            names=names,
            errors=[float(value) for value in delta_error],
            fit_sigmas=sigmas,
            criteria=criteria,
            normal_matrix_native=(
                unconstrained_fit.normal_matrix_native if unconstrained_fit is not None else fit.normal_matrix_native
            ),
            covariance_native=covariance_native,
            prior_sigma_native=prior,
        )
    observation_rows: list[dict[str, object]] = []
    for row_index, key in enumerate(keys):
        sample, run_id, event_id, signature, *_ = key
        anchor_observation = anchor_bank[key]
        target_observation = target_bank[key]
        entry: dict[str, object] = {
            "sample_id": sample,
            "run_id": run_id,
            "event_id": event_id,
            "route_origin_signature": signature,
            "observation_kind": anchor_observation.observation_kind,
            "source_station_id": anchor_observation.source_station_id,
            "target_station_id": anchor_observation.target_station_id,
            "anchor_source_synthetic_role": anchor_observation.source_synthetic_role,
            "anchor_target_synthetic_role": anchor_observation.target_synthetic_role,
            "target_source_synthetic_role": target_observation.source_synthetic_role,
            "target_target_synthetic_role": target_observation.target_synthetic_role,
        }
        for residual_index, label in enumerate(("x_mm", "y_mm", "tx", "ty")):
            entry[f"anchor_residual_{label}"] = float(anchor_residual[row_index, residual_index])
            entry[f"target_residual_{label}"] = float(target_residual[row_index, residual_index])
            entry[f"response_{label}"] = float(fit.response[row_index, residual_index])
            entry[f"post_update_response_{label}"] = float(fit.residual_response[row_index, residual_index])
            for parameter_index, name in enumerate(names):
                entry[f"derivative_{name}_{label}"] = float(
                    derivative_native[row_index, residual_index, parameter_index]
                )
        observation_rows.append(entry)
    _write_csv(output / "route_selected_update_observations.csv", observation_rows)
    np.savez_compressed(
        output / "route_selected_update_arrays.npz",
        parameter_names=np.asarray(names),
        observation_keys=np.asarray([json.dumps(key) for key in keys]),
        anchor_residual=anchor_residual,
        target_residual=target_residual,
        covariance=anchor_covariance,
        derivative_native=derivative_native,
        response=fit.response,
        predicted_response=fit.predicted_response,
        residual_response=fit.residual_response,
        normal_matrix_native=fit.normal_matrix_native,
        normal_matrix_scaled=fit.normal_matrix_scaled,
        covariance_native=covariance_native,
        correlation_native=correlation_native,
        recovered_delta=recovered_delta,
        expected_delta=expected_delta,
    )
    expected_values = {name: float(expected_delta[index]) for index, name in enumerate(names)}
    recovered_values = {name: float(recovered_delta[index]) for index, name in enumerate(names)}
    hierarchy_audit = None
    hierarchy_capture_success = None
    if hierarchical_v1:
        from alignment.five_dof_sampling import five_dof_severity
        from alignment.hierarchical_v1 import (
            C_DX,
            LAYER_LEVEL,
            STATION_LEVEL,
            complete_hierarchical_values,
            layer_transforms_for_cdx,
            remaining_after_level_step,
            station_absorption_of_cdx,
            station_payload_stability,
        )

        floated_level = LAYER_LEVEL if names == [C_DX] else STATION_LEVEL
        injected_hierarchical = complete_hierarchical_values(
            _point_parameter_values(_specs(plan), target_point)
        )
        remaining_hierarchical = remaining_after_level_step(
            injected=injected_hierarchical,
            recovered=recovered_values,
            floated_level=floated_level,
        )
        remaining_layers = {"0": layer_transforms_for_cdx(remaining_hierarchical[C_DX])}
        rank_ok = bool(fit.full_rank) and int(fit.normal_matrix_rank) == len(names)
        pre_rms = _rms(fit.response)
        post_rms = _rms(fit.residual_response)
        residual_ok = math.isfinite(pre_rms) and math.isfinite(post_rms) and post_rms <= 0.5 * max(pre_rms, 1.0e-12)
        leakage_station_absorbs_cdx = None
        leakage_layer_moves_station = None
        if floated_level == STATION_LEVEL:
            leakage_station_absorbs_cdx = station_absorption_of_cdx(
                injected=injected_hierarchical, recovered_station=recovered_values
            )
        else:
            leakage_layer_moves_station = station_payload_stability(
                injected_hierarchical, remaining_hierarchical
            )
        station_fixed = (
            floated_level == LAYER_LEVEL
            and leakage_layer_moves_station is not None
            and abs(float(leakage_layer_moves_station["max_abs_station_delta"])) <= 1.0e-15
        )
        hierarchy_capture_success = bool(rank_ok and residual_ok)
        if capture_aggregate is not None:
            hierarchy_capture_success = bool(capture_aggregate["capture_success"] and rank_ok)
        hierarchy_audit = {
            "gauge": "outer_contrast",
            "fit_basis": "hierarchical_v1",
            "canonical_internal_basis": "outer_contrast",
            "constraint": (
                "float station 5-DoF with survey dz prior; C_dx stays at the current geometry"
                if floated_level == STATION_LEVEL
                else "float only C_dx; station 5-DoF stays at the current geometry; L0=+C_dx, L1=0, L2=-C_dx"
            ),
            "joint_station_cdx_newton": False,
            "block_coordinate": True,
            "floated_level": floated_level,
            "fixed_level": LAYER_LEVEL if floated_level == STATION_LEVEL else STATION_LEVEL,
            "reduced_parameter_names": list(names),
            "proposed_next_is_not_remaining": True,
            "injected_hierarchical_v1": injected_hierarchical,
            "remaining_hierarchical_v1": remaining_hierarchical,
            "remaining_layer_transforms": remaining_layers,
            "recovered": recovered_values,
            "expected": expected_values,
            "station_five_dof_severity_injected": five_dof_severity(injected_hierarchical),
            "station_five_dof_severity_remaining": five_dof_severity(remaining_hierarchical),
            "leakage_station_absorbs_C_dx": leakage_station_absorbs_cdx,
            "leakage_layer_moves_station": leakage_layer_moves_station,
            "station_transforms_unchanged": station_fixed,
            "full_rank": rank_ok,
            "normal_matrix_rank": int(fit.normal_matrix_rank),
            "normal_matrix_condition_number": (
                None if fit.normal_matrix_condition_number is None else float(fit.normal_matrix_condition_number)
            ),
            "prefit_residual_rms": pre_rms,
            "postfit_residual_rms": post_rms,
            "post_over_pre_rms": (post_rms / pre_rms) if pre_rms > 0.0 and math.isfinite(pre_rms) else float("nan"),
            "capture_success": hierarchy_capture_success,
        }
    elif reduction is not None:
        component = _layer_component(specs)
        expected_split = _component_internals(expected_values, specs, component)
        recovered_split = _component_internals(recovered_values, specs, component)
        outer_expected = float(expected_split["outer_relative_layer0_minus_layer2"])
        outer_recovered = float(recovered_split["outer_relative_layer0_minus_layer2"])
        outer_error = abs(outer_recovered - outer_expected)
        station_fixed = _six_vectors_close(proposed_transforms, dict(anchor_transforms))
        outer_tol = 0.05 if component == "dx_mm" else 0.20
        internals_tol = outer_tol
        internals = expected_split["layer_internal"]
        recovered_internals = recovered_split["layer_internal"]
        internals_ok = all(
            abs(float(recovered_internals[key]) - float(internals[key])) <= internals_tol
            for key in internals
            if key != "layer_1"
        )
        rank_ok = bool(fit.full_rank) and int(fit.normal_matrix_rank) == len(reduction.names)
        pre_rms = _rms(fit.response)
        post_rms = _rms(fit.residual_response)
        residual_ok = math.isfinite(pre_rms) and math.isfinite(post_rms) and post_rms <= 0.5 * max(pre_rms, 1.0e-12)
        hierarchy_capture_success = bool(outer_error <= outer_tol and station_fixed and rank_ok and residual_ok)
        hierarchy_audit = {
            "gauge": str(args.gauge),
            "constraint": reduction.constraint,
            "component": component,
            "reduced_parameter_names": list(reduction.names),
            "dropped_parameter_names": list(reduction.dropped_names),
            "compare_gauge_invariant_outer_relative_not_labels": True,
            "expected": expected_split,
            "recovered": recovered_split,
            "outer_relative_error": outer_error,
            "outer_relative_tolerance": outer_tol,
            "internal_tolerance": internals_tol,
            "per_layer_internal_within_tolerance": internals_ok,
            "station_transforms_unchanged": station_fixed,
            "full_rank": rank_ok,
            "prefit_residual_rms": pre_rms,
            "postfit_residual_rms": post_rms,
            "post_over_pre_rms": (post_rms / pre_rms) if pre_rms > 0.0 and math.isfinite(pre_rms) else float("nan"),
            "capture_success": hierarchy_capture_success,
        }
    elif has_contrast:
        from alignment.contrast_sampling import CONTRAST_PARAMETERS
        from alignment.layer_hierarchy import contrast_layer_six_vectors
        from alignment.sequential_contrast import remaining_after_block_step

        station_fixed = _six_vectors_close(proposed_transforms, dict(anchor_transforms))
        layers = None
        if isinstance(proposed_layer_transforms, Mapping):
            raw_layers = proposed_layer_transforms.get("0", proposed_layer_transforms.get(0))
            if isinstance(raw_layers, Mapping):
                layers = {str(key): list(value) for key, value in raw_layers.items()}
        expected_layers = contrast_layer_six_vectors(
            float(recovered_values.get("C_dx", 0.0)),
            float(recovered_values.get("C_rx", 0.0)),
        )
        payload_ok = layers is not None and all(
            np.allclose(
                np.asarray(layers.get(str(layer), ()), dtype=np.float64),
                np.asarray(expected_layers[str(layer)], dtype=np.float64),
                rtol=0.0,
                atol=1.0e-12,
            )
            for layer in (0, 1, 2)
        )
        layer1_zero = layers is not None and np.allclose(
            np.asarray(layers.get("1", [1.0]), dtype=np.float64), 0.0, rtol=0.0, atol=1.0e-12
        )
        rank_ok = bool(fit.full_rank) and int(fit.normal_matrix_rank) == len(names)
        pre_rms = _rms(fit.response)
        post_rms = _rms(fit.residual_response)
        residual_ok = math.isfinite(pre_rms) and math.isfinite(post_rms) and post_rms <= 0.5 * max(pre_rms, 1.0e-12)
        correlation = None
        if (
            correlation_native.shape == (len(names), len(names))
            and "C_dx" in names
            and "C_rx" in names
        ):
            correlation = float(correlation_native[names.index("C_dx"), names.index("C_rx")])
        sequential = len(names) == 1
        floated = names[0] if sequential else None
        remaining_contrast = None
        remaining_layers = None
        injected_contrast = None
        if sequential:
            all_contrast = [spec for spec in _specs(plan) if spec_scope(spec) == "contrast"]
            injected_contrast = dict(_point_parameter_values(all_contrast, target_point))
            for name in CONTRAST_PARAMETERS:
                injected_contrast.setdefault(name, 0.0)
            remaining_contrast = remaining_after_block_step(
                injected=injected_contrast,
                recovered_floated=float(recovered_values[floated]),
                floated=floated,
            )
            remaining_layers = contrast_layer_six_vectors(
                remaining_contrast["C_dx"], remaining_contrast["C_rx"]
            )
        hierarchy_capture_success = bool(station_fixed and layer1_zero and payload_ok and rank_ok and residual_ok)
        hierarchy_audit = {
            "gauge": "outer_contrast",
            "canonical_internal_basis": "outer_contrast",
            "constraint": (
                "station rigid transform frozen; float exactly one of C_dx or C_rx; "
                "the other stays at the current geometry; L0=+C, L1=0, L2=-C"
                if sequential
                else "station rigid transform frozen; fit C_dx and C_rx; L0=+C, L1=0, L2=-C"
            ),
            "reduced_parameter_names": list(names),
            "joint_2d_newton": not sequential,
            "block_coordinate": sequential,
            "floated": floated,
            "fixed": [] if floated is None else [name for name in CONTRAST_PARAMETERS if name != floated],
            "station_transforms_unchanged": station_fixed,
            "layer1_frozen": layer1_zero,
            "payload_expansion_ok": payload_ok,
            "proposed_next_is_not_remaining": sequential,
            "injected_contrast": injected_contrast,
            "remaining_contrast": remaining_contrast,
            "remaining_layer_transforms": remaining_layers,
            "posterior_correlation_C_dx_C_rx": correlation,
            "near_degenerate_abs_correlation_ge_0.9": (
                None if correlation is None else bool(abs(correlation) >= 0.9)
            ),
            "full_rank": rank_ok,
            "normal_matrix_rank": int(fit.normal_matrix_rank),
            "normal_matrix_condition_number": (
                None if fit.normal_matrix_condition_number is None else float(fit.normal_matrix_condition_number)
            ),
            "prefit_residual_rms": pre_rms,
            "postfit_residual_rms": post_rms,
            "post_over_pre_rms": (post_rms / pre_rms) if pre_rms > 0.0 and math.isfinite(pre_rms) else float("nan"),
            "expected": expected_values,
            "recovered": recovered_values,
            "capture_success": hierarchy_capture_success,
        }
    if args.self_nulling_update:
        capture_success = None
    elif capture_aggregate is not None and "capture_success" in capture_aggregate:
        capture_success = bool(capture_aggregate["capture_success"])
    elif hierarchy_capture_success is not None:
        capture_success = hierarchy_capture_success
    else:
        capture_success = None if tolerance is None else bool(all(row["capture_success"] for row in parameter_rows))
    mode_validity = None
    if mode_contract is not None and calibration_mode is not None:
        target_values = target_point.get("alignment_parameter_values")
        if not isinstance(target_values, Mapping):
            target_values = {}
        if args.declared_unmodeled_cdx_um is not None:
            unmodeled_um = float(args.declared_unmodeled_cdx_um)
        elif MODE_C_DX in target_values:
            unmodeled_um = abs(float(target_values[MODE_C_DX])) * 1.0e3
        elif scan_mode == "station_rigid_multidof":
            unmodeled_um = 0.0
        else:
            unmodeled_um = None
        cdx_fixed_by = args.cdx_fixed_by
        if cdx_fixed_by is None and unmodeled_um == 0.0:
            cdx_fixed_by = "isolation_zero"
        station_capture = None
        if args.station_framework_capture_success is not None:
            station_capture = args.station_framework_capture_success == "true"
        elif args.station_capture_artifact is not None:
            station_capture = read_station_framework_capture_success(
                Path(args.station_capture_artifact).expanduser().resolve()
            )
        mode_validity = evaluate_mode_validity(
            mode_contract,
            mode=calibration_mode,
            floated_parameters=names,
            unmodeled_abs_C_dx_um=unmodeled_um,
            cdx_fixed_by=cdx_fixed_by,
            station_framework_capture_success=station_capture,
            same_data_stage_as_other_mode=bool(args.same_data_stage_as_other_mode),
        )
        mode_validity = {
            **mode_validity,
            "contract": str(mode_contract_path),
        }
    summary: dict[str, object] = {
        "method": "truth_free_route_selected_physical_multidof_local_update",
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "q_over_p_mode": 0,
        "test_opened": False,
        "architecture_or_threshold_tuning": False,
        "scan_root": str(scan_root),
        "target_scan_root": (
            None if args.target_scan_root is None else str(Path(args.target_scan_root).expanduser().resolve())
        ),
        "scan_mode": scan_mode,
        "fit_basis": fit_basis or None,
        "joint_station_cdx_newton": False if hierarchical_v1 else None,
        "calibration_mode": calibration_mode,
        "mode_validity": mode_validity,
        "geometry_write_allowed": False if args.self_nulling_update else (
            None if mode_validity is None else bool(mode_validity["geometry_write_allowed"])
        ),
        "residual_reduction_is_not_alignment_success": True,
        "self_nulling_update": bool(args.self_nulling_update),
        "official_conditions_db_write": False,
        "real_data": bool(args.allow_real_data or args.self_nulling_update),
        "only_parameters": list(names),
        "gauge": None if args.gauge is None else str(args.gauge),
        "allow_nominal_anchor": bool(
            args.allow_nominal_anchor or target_point.get("point_role") == "held_out_closure"
        ),
        "anchor_point": str(args.anchor_point),
        "target_point": str(args.target_point),
        "probe_points": {name: {"positive": positive_points[name], "negative": negative_points[name]} for name in names},
        "damping": float(args.damping),
        "observation_kind": str(args.observation_kind),
        "covariance_calibration": (
            None
            if args.covariance_calibration is None
            else str(Path(args.covariance_calibration).expanduser().resolve())
        ),
        "anchor_bank_remeasured_from_candidate_graph": bool(args.anchor_payload_sample),
        "observation_statistics": observation_statistics_audit,
        "huber_weighting": huber_summary,
        "observation_contract": (
            "Exact selected mode-0 ACTS edge residual/covariance from the physical candidate graph."
            if args.observation_kind == "field_edge"
            else (
                "Anchor-selected truth-free route set held fixed; exact mode-0 ACTS edge "
                "residual/covariance recomputed from every payload's physical candidate graph "
                "by original tracklet provenance."
                if args.observation_kind == "anchor_selected_field_edge"
                else "Legacy straight-line leave-one-out residual used only as an explicit diagnostic comparison."
            )
        ),
        "update_semantics": (
            "A local physical Newton/Gauss-Newton step in payload coordinates from the anchor residual toward "
            "the independently refitted target residual.  This MC closure update has no assumed manual sign flip."
        ),
        "association_contract": {
            "anchor": anchor_summary,
            "target": target_summary,
            "positive": positive_summaries,
            "negative": negative_summaries,
            "selected_route_overlap": overlap,
            "target_synthetic_role_counts_by_bank": {
                "anchor": _role_counts(anchor_bank),
                "target": _role_counts(target_bank),
                **{f"positive_{name}": _role_counts(positive_banks[name]) for name in names},
                **{f"negative_{name}": _role_counts(negative_banks[name]) for name in names},
            },
        },
        "normal_matrix_rank": int(fit.normal_matrix_rank),
        "fit_matrix_rank": int(fit.fit_matrix_rank),
        "normal_matrix_condition_number": fit.normal_matrix_condition_number,
        "identifiable_subspace_condition_number": fit.identifiable_subspace_condition_number,
        "data_singular_values": fit.data_singular_values,
        "response_chi2": float(fit.response_chi2),
        "response_ndof": int(fit.response_ndof),
        "parameter_covariance": covariance_native,
        "parameter_correlation": correlation_native,
        "parameters": parameter_rows,
        "prior_audit": prior_rows,
        "capture_aggregate": capture_aggregate,
        "capture_criteria": None if args.capture_criteria is None else str(Path(args.capture_criteria).expanduser().resolve()),
        "hierarchy_internal_audit": hierarchy_audit,
        "proposed_next_parameter_values": proposal_values,
        "proposed_next_station_transforms": proposed_transforms,
        "proposed_next_layer_transforms": proposed_layer_transforms,
        "remaining_contrast": None if hierarchy_audit is None else hierarchy_audit.get("remaining_contrast"),
        "remaining_hierarchical_v1": (
            None if hierarchy_audit is None else hierarchy_audit.get("remaining_hierarchical_v1")
        ),
        "proposed_next_is_not_remaining": bool(
            hierarchy_audit is not None and hierarchy_audit.get("proposed_next_is_not_remaining")
        ),
        "capture_success": capture_success,
    }
    (output / "route_selected_update.json").write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    printed: dict[str, object] = {
        "output_dir": str(output),
        "used_selected_observations": len(keys),
        "normal_matrix_rank": fit.normal_matrix_rank,
        "normal_matrix_condition_number": fit.normal_matrix_condition_number,
        "proposed_next_parameter_values": proposal_values,
        "capture_success": capture_success,
        "calibration_mode": calibration_mode,
        "geometry_write_allowed": None if mode_validity is None else bool(mode_validity["geometry_write_allowed"]),
        "cross_level_contaminated": None if mode_validity is None else bool(mode_validity["cross_level_contaminated"]),
    }
    if hierarchy_audit is not None:
        expected_audit = hierarchy_audit.get("expected")
        printed_hierarchy: dict[str, object] = {
            "gauge": hierarchy_audit.get("gauge"),
            "station_transforms_unchanged": hierarchy_audit.get("station_transforms_unchanged"),
            "capture_success": hierarchy_audit.get("capture_success"),
            "floated_level": hierarchy_audit.get("floated_level"),
            "remaining_hierarchical_v1": hierarchy_audit.get("remaining_hierarchical_v1"),
            "post_over_pre_rms": hierarchy_audit.get("post_over_pre_rms"),
        }
        if isinstance(expected_audit, Mapping) and "outer_relative_layer0_minus_layer2" in expected_audit:
            printed_hierarchy["expected_outer_relative"] = expected_audit["outer_relative_layer0_minus_layer2"]
            printed_hierarchy["recovered_outer_relative"] = hierarchy_audit["recovered"][
                "outer_relative_layer0_minus_layer2"
            ]
        else:
            printed_hierarchy["expected"] = expected_audit
            printed_hierarchy["recovered"] = hierarchy_audit.get("recovered")
            printed_hierarchy["posterior_correlation_C_dx_C_rx"] = hierarchy_audit.get(
                "posterior_correlation_C_dx_C_rx"
            )
            printed_hierarchy["near_degenerate_abs_correlation_ge_0.9"] = hierarchy_audit.get(
                "near_degenerate_abs_correlation_ge_0.9"
            )
            printed_hierarchy["post_over_pre_rms"] = hierarchy_audit.get("post_over_pre_rms")
            printed_hierarchy["joint_2d_newton"] = hierarchy_audit.get("joint_2d_newton")
            printed_hierarchy["floated"] = hierarchy_audit.get("floated")
            printed_hierarchy["remaining_contrast"] = hierarchy_audit.get("remaining_contrast")
        printed["hierarchy_internal_audit"] = printed_hierarchy
    fit_for_arrays = unconstrained_fit if unconstrained_fit is not None else fit
    prior_native = fit_for_arrays.prior_sigma_native
    np.savez_compressed(
        output / "fit_arrays.npz",
        parameter_names=np.asarray(list(names)),
        recovered_parameters=np.asarray(recovered_delta, dtype=np.float64),
        normal_matrix_native=np.asarray(fit_for_arrays.normal_matrix_native, dtype=np.float64),
        right_hand_side_native=np.asarray(fit_for_arrays.right_hand_side_native, dtype=np.float64),
        prior_sigma_native=np.asarray(
            np.full(len(names), np.nan, dtype=np.float64) if prior_native is None else prior_native,
            dtype=np.float64,
        ),
    )
    print(json.dumps(_json_ready(printed), indent=2))
    if args.require_mode_valid and mode_validity is not None and not bool(mode_validity["geometry_write_allowed"]):
        raise ValueError(
            "mode-validity contract forbids writing this geometry: "
            + str(mode_validity.get("status"))
            + " indicators="
            + ",".join(mode_validity.get("reject_indicators") or ())
        )


if __name__ == "__main__":
    main()
