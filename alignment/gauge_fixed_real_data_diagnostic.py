"""Workbook-78 gauge-fixed real-data alignment diagnostic V1.

Scientific question (the only one): without claiming to recover a unique
mechanical station pose, can a pre-frozen reconstruction gauge representative
be stably determined from a real-data calibration subset and improve
gauge-invariant tracker observables on completely held-out real data?

This is a diagnostic, not a geometry correction deployment:
``geometry_write_allowed = false`` and ``official_conditions_write_allowed =
false`` for the whole workbook.

Frozen inheritance:

* 7D parameter space, severity scales, rank tolerance, identifiable/null
  basis and the amended projector-Frobenius regression contract from
  workbooks 68/77 (verified by SHA and regression before any solve).
* Real-data population from the frozen Operating Protocol V1 / Frozen-V2
  selected-route contract (12 runs; residual-blind role map:
  calibration = 14973+14974, held-out = 14975+14976 plus 7 expansion
  monitoring runs, 14977 report-only).
* Gauge contract from workbook 77: primary ``minimum_norm_scaled_gauge``
  (severity-scaled minimum-norm representative, inheriting the
  ``solver_restricted_to_identifiable_subspace`` doctrine - chosen by
  contract, not by condition number), secondary control
  ``named_parameter_zero_gauge``, report-only ``minimum_norm_native_gauge``.

Observable model transfer: per-pair Jacobians on real data would require FD
probes, which remain forbidden.  The only legal transfer is the frozen MC
pooled-bank station-pair mean native Jacobian; its model error is quantified
by the pre-registered MC control (injection closure + null ensemble) before
any real-data solve.  The real-data pair bank is rebuilt from the frozen
identity files with the frozen ``evaluate_field_propagation`` chain and must
reproduce the frozen DQ adjacent-edge CSV rows bit-exactly.

All output parameters are named ``reconstruction gauge representative`` or
``gauge-fixed candidate diagnostic``; they are never mechanical station
measurements.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.cad_survey_nov22 import git_head_sha
from alignment.gauge_constraint_feasibility import regression_against_frozen_basis
from alignment.identifiable_subspace import IdentifiableSubspace
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment.tracker_only_identifiable_subspace import (
    _pool_campaign_banks,
    load_physical_banks,
    subspace_from_physical_bank,
)
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events
from evaluation.field_propagation import evaluate_field_propagation

SCHEMA_VERSION = "faser-gauge-fixed-real-data-alignment-diagnostic-v1"
DEFAULT_CONFIG_RELATIVE = Path("configs/gauge_fixed_real_data_alignment_diagnostic_v1.yaml")

PARAMETER_NAMES = (
    "ift_dx_mm",
    "ift_dy_mm",
    "ift_dz_mm",
    "ift_rx_mrad",
    "ift_ry_mrad",
    "ift_rz_mrad",
    "C_dx",
)
OBSERVABLE_NAMES = ("x_mm", "y_mm", "tx", "ty")
STATION_PAIRS = ((0, 1), (0, 2), (0, 3))

DECISION_PASS = "gauge_fixed_real_data_candidate_diagnostic_pass"
DECISION_TRACKER_REGRESSION_FAILED = "tracker_information_regression_failed"
DECISION_POPULATION_MISMATCH = "real_data_population_provenance_mismatch"
DECISION_TRANSFER_FAILED = "real_data_observable_model_transfer_failed"
DECISION_INCONCLUSIVE = "real_data_diagnostic_inconclusive"
DECISION_RANK_ZERO = "real_data_identifiable_information_rank_zero"
DECISION_OUT_OF_SUPPORT = "real_data_linearized_model_out_of_support"
DECISION_NOT_STABLE = "real_data_gauge_fixed_candidate_not_source_stable"
DECISION_GAUGE_INVARIANCE_FAILED = "real_data_gauge_invariance_failed"
DECISION_NOT_IMPROVED = "real_data_heldout_observable_not_improved"

CONFIG_MUST_BE_TRUE = (
    "do_not_switch_primary_gauge_after_real_data",
    "do_not_repick_split",
    "do_not_select_events_by_residual",
    "do_not_tune_on_held_out",
    "do_not_generate_fd_probes",
    "do_not_run_newton",
    "do_not_write_payload",
    "do_not_restack_2024_r0022_collision_like",
    "do_not_use_external_evidence_as_prior",
    "do_not_reopen_identifiability_rescue",
    "do_not_redefine_identifiable_basis_on_real_data",
)
CONFIG_MUST_BE_FALSE = (
    "geometry_write_allowed",
    "official_conditions_write_allowed",
    "real_data_candidate_alignment_authorized",
    "external_constraint_ingest_authorized",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    text = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG_RELATIVE))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    for key in CONFIG_MUST_BE_TRUE:
        if config.get(key) is not True:
            raise ValueError(f"config must set {key}: true")
    for key in CONFIG_MUST_BE_FALSE:
        if config.get(key) is not False:
            raise ValueError(f"config must set {key}: false")

    corpus = config["tracker_information"]["jacobian_corpus"]
    if tuple(str(n) for n in corpus["parameter_names"]) != PARAMETER_NAMES:
        raise ValueError("tracker information must keep the frozen 7D parameter order")
    scales = np.asarray(corpus["parameter_scales"], dtype=np.float64)
    if not np.array_equal(scales, np.array([5.0, 5.0, 5.0, 60.0, 60.0, 60.0, 0.12])):
        raise ValueError("severity scales must stay (5,5,5,60,60,60,0.12)")
    if float(config["tracker_information"]["rank_tolerance"]) != 0.01:
        raise ValueError("rank_tolerance must remain the frozen 0.01")
    if str(corpus["kind"]) != "physical_central_finite_difference":
        raise ValueError("tracker information must stay the frozen physical FD corpus")
    if float(corpus["min_truth_match_fraction"]) != 0.99:
        raise ValueError("min_truth_match_fraction must stay 0.99")
    if int(corpus["q_over_p_mode"]) != 0:
        raise ValueError("q_over_p_mode must stay 0")
    if str(corpus["anchor_point"]) != "iteration_00_reference":
        raise ValueError("anchor_point must stay iteration_00_reference")

    gauges = config["gauges"]
    if str(gauges["primary"]) != "minimum_norm_scaled_gauge":
        raise ValueError("primary gauge is frozen to minimum_norm_scaled_gauge")
    if str(gauges["secondary_control"]) != "named_parameter_zero_gauge":
        raise ValueError("secondary control is frozen to named_parameter_zero_gauge")
    if str(gauges["report_only"]) != "minimum_norm_native_gauge":
        raise ValueError("report-only gauge is frozen to minimum_norm_native_gauge")

    inheritance = config["inheritance"]
    wb77_config = resolve_under_root(project_root(), str(inheritance["workbook_77_config"]))
    if sha256_file(wb77_config) != str(inheritance["workbook_77_config_sha256"]):
        raise ValueError("workbook-77 config SHA256 mismatch")
    wb77_root = resolve_under_root(project_root(), str(inheritance["workbook_77_output_root"]))
    for name, expected in inheritance["workbook_77_artifact_sha256"].items():
        actual = sha256_file(wb77_root / name)
        if actual != str(expected):
            raise ValueError(f"workbook-77 artifact SHA256 mismatch for {name}")
    basis = resolve_under_root(project_root(), str(inheritance["workbook_68_identifiable_basis"]))
    if sha256_file(basis) != str(inheritance["workbook_68_identifiable_basis_sha256"]):
        raise ValueError("workbook-68 identifiable basis SHA256 mismatch")
    checkpoint = resolve_under_root(
        project_root(),
        "outputs/mc24_v3_expanded_trainval_v2_bce_control_v1/route_aware_transformer_v2.pt",
    )
    if sha256_file(checkpoint) != str(inheritance["frozen_v2_checkpoint_sha256"]):
        raise ValueError("frozen V2 checkpoint SHA256 mismatch")

    population = config["real_data_population"]
    roles = {str(src["role"]) for src in population["sources"]}
    split = config["split"]
    expected_roles = set(split["calibration_roles"]) | set(split["held_out_roles"]) | set(
        split["report_only_roles"]
    )
    if roles != expected_roles:
        raise ValueError("population roles must exactly cover the frozen split roles")
    if len({str(src["source_id"]) for src in population["sources"]}) != len(
        population["sources"]
    ):
        raise ValueError("population source_ids must be unique")

    if str(config["output_parameter_naming"]) != "reconstruction_gauge_representative":
        raise ValueError("output parameters must be named reconstruction_gauge_representative")
    if not list(config["eligible_external_physical_constraints"]) == []:
        raise ValueError("external constraint branch remains closed: eligibility list must be empty")

    config["config_path"] = str(config_path)
    return config


# ---------------------------------------------------------------------------
# Tracker information (frozen WB68/77 contract) + amended regression
# ---------------------------------------------------------------------------


def load_tracker_information(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], IdentifiableSubspace, dict[str, Any], dict[str, Any]]:
    """Rebuild the pooled WB68 tracker information and regress it (SHA-gated)."""
    corpus = dict(config["tracker_information"]["jacobian_corpus"])
    banks = load_physical_banks({"jacobian_corpus": corpus})
    pooled = _pool_campaign_banks(list(banks))
    pooled["source_id"] = "pooled_workbook68_corpus"
    subspace, extras = subspace_from_physical_bank(
        pooled,
        rank_tolerance=float(config["tracker_information"]["rank_tolerance"]),
        rcond=float(config["tracker_information"]["rcond"]),
    )
    frozen = _read_json(
        resolve_under_root(
            project_root(), str(config["inheritance"]["workbook_68_identifiable_basis"])
        )
    )["pooled"]
    regression = regression_against_frozen_basis(
        subspace,
        frozen,
        singular_value_rtol=1.0e-10,
        max_projector_frobenius=float(
            config["inheritance"]["regression_max_projector_frobenius"]
        ),
        expected_identifiable_rank=int(frozen["identifiable_rank"]),
        expected_null_dimension=int(frozen["null_dimension"]),
    )
    regression.update(
        {
            "kind": "tracker_information_regression",
            "amended_contract": "projector_frobenius_distance_not_arccos_principal_angles",
            "n_sources": len(banks),
            "n_pairs": int(extras["n_pairs"]),
        }
    )
    return banks, pooled, subspace, extras, regression


# ---------------------------------------------------------------------------
# Gauge candidates (WB77 contract, native G matrices)
# ---------------------------------------------------------------------------


def build_gauge_candidates(
    config: Mapping[str, Any], subspace: IdentifiableSubspace
) -> dict[str, np.ndarray]:
    """Return {gauge_id: G_native} for the three frozen gauges."""
    scales = np.asarray(subspace.parameter_scales, dtype=np.float64)
    s_mat = np.diag(scales)
    v_null = np.asarray(subspace.v_null, dtype=np.float64)
    names = tuple(subspace.parameter_names)
    gauges: dict[str, np.ndarray] = {}
    for spec in config["gauges"]["candidates"]:
        name = str(spec["name"])
        kind = str(spec["kind"])
        if kind == "fixed_reference_rigid_mode":
            g_native = np.zeros((len(spec["fixed_parameters"]), len(names)))
            for row, parameter in enumerate(spec["fixed_parameters"]):
                g_native[row, names.index(str(parameter))] = 1.0
        elif kind == "minimum_norm_representative":
            metric = str(spec["metric"])
            if metric == "severity_scaled":
                g_native = v_null.T @ np.linalg.inv(s_mat)
            elif metric == "native":
                g_native = (s_mat @ v_null).T
            else:
                raise ValueError(f"unknown minimum-norm metric {metric}")
        else:
            raise ValueError(f"unknown gauge kind {kind}")
        intersection = g_native @ s_mat @ v_null
        if abs(float(np.linalg.det(intersection))) <= 1.0e-12:
            raise ValueError(f"gauge {name} does not complement the frozen null space")
        gauges[name] = g_native
    expected = {
        str(config["gauges"]["primary"]),
        str(config["gauges"]["secondary_control"]),
        str(config["gauges"]["report_only"]),
    }
    if set(gauges) != expected:
        raise ValueError("gauge candidate set must be exactly the three frozen gauges")
    return gauges


# ---------------------------------------------------------------------------
# Real-data population inventory and frozen split (residual-blind metadata)
# ---------------------------------------------------------------------------


def _source_roots(config: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Path]:
    source_id = str(source["source_id"])
    core = resolve_under_root(project_root(), str(config["real_data_population"]["core_root"]))
    expansion = resolve_under_root(
        project_root(), str(config["real_data_population"]["expansion_root"])
    )
    if (core / "association" / source_id).is_dir():
        return {
            "association": core / "association" / source_id,
            "identity": core / "identity" / "samples" / source_id / "iteration_00_current",
        }
    return {
        "association": expansion / "athena" / "association" / source_id,
        "identity": expansion
        / "athena"
        / "identity"
        / "samples"
        / source_id
        / "iteration_00_current",
    }


def _load_routes(association_dir: Path) -> list[dict[str, Any]]:
    path = association_dir / "selected_routes.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"frozen route record missing: {path}")
    routes = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                routes.append(json.loads(line))
    return routes


def freeze_split(config: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze the residual-blind calibration/held-out split from route metadata.

    Only route-record metadata (run/event/source IDs, station coverage) is
    read; no residual value is touched.  Any mismatch against the frozen
    population expectations is a hard provenance failure.
    """
    entries = []
    for source in config["real_data_population"]["sources"]:
        source_id = str(source["source_id"])
        routes = _load_routes(_source_roots(config, source)["association"])
        if len(routes) != int(source["expected_selected_routes"]):
            raise ValueError(
                f"frozen population mismatch for {source_id}: "
                f"{len(routes)} routes != expected {source['expected_selected_routes']}"
            )
        event_ids = sorted({int(route["event_id"]) for route in routes})
        station_coverage = sorted(
            {int(station) for route in routes for station in route["endpoint_stations"]}
        )
        anchor_pairs = _anchor_pair_keys(routes)
        anchor_composition = {
            f"0->{station}": sum(
                1 for pair in anchor_pairs if pair["target_station_id"] == station
            )
            for station in (1, 2, 3)
        }
        entries.append(
            {
                "source_id": source_id,
                "run_id": int(source["run_id"]),
                "role": str(source["role"]),
                "n_selected_routes": len(routes),
                "event_ids": event_ids,
                "n_events": len(event_ids),
                "station_coverage": station_coverage,
                "anchor_pair_composition": anchor_composition,
            }
        )
    split_payload = {
        "kind": "calibration_held_out_split_freeze",
        "rule": str(config["split"]["rule"]),
        "residual_blind": True,
        "entries": entries,
    }
    split_payload["split_sha256"] = _canonical_sha256(entries)
    # The pre-registered MC-control compositions must match the frozen split
    # exactly; any drift is a hard provenance failure before any solve.
    control = config["mc_control"]
    for role_set, composition_key in (
        (set(config["split"]["calibration_roles"]), "calibration_composition"),
        (
            set(config["split"]["held_out_roles"]) | set(config["split"]["report_only_roles"]),
            "held_out_composition",
        ),
    ):
        actual = {f"0->{station}": 0 for station in (1, 2, 3)}
        for entry in entries:
            if entry["role"] in role_set:
                for label, count in entry["anchor_pair_composition"].items():
                    actual[label] += count
        expected = {str(k): int(v) for k, v in control[composition_key].items()}
        if actual != expected:
            raise ValueError(
                f"mc-control {composition_key} {expected} does not match the frozen "
                f"split composition {actual}"
            )
    return split_payload


# ---------------------------------------------------------------------------
# Real-data bank builder (frozen evaluation chain; CSV-exact reproduction)
# ---------------------------------------------------------------------------


def _anchor_pair_keys(routes: Sequence[Mapping[str, Any]]) -> list[dict[str, int]]:
    """Anchor pairs (station 0 x station j>=1) of frozen selected routes."""
    pairs = []
    for route in routes:
        provenance = route["endpoint_provenance"]
        by_station = {int(item["station_id"]): item for item in provenance}
        if 0 not in by_station:
            continue
        anchor = by_station[0]
        for station in sorted(by_station):
            if station == 0:
                continue
            target = by_station[station]
            pairs.append(
                {
                    "run_id": int(route["run_id"]),
                    "event_id": int(route["event_id"]),
                    "route_index": int(route["route_index"]),
                    "source_tracklet_id": int(anchor["synthetic_tracklet_id"]),
                    "target_tracklet_id": int(target["synthetic_tracklet_id"]),
                    "target_station_id": int(station),
                }
            )
    return pairs


def build_real_data_bank(
    config: Mapping[str, Any],
    *,
    roles: Sequence[str],
) -> dict[str, Any]:
    """Rebuild the anchor-pair residual bank for the frozen sources in ``roles``.

    Uses the frozen identity files and the frozen ``evaluate_field_propagation``
    chain (truth-free, q_over_p_mode=0).  No event selection: every frozen
    selected route contributes all of its anchor pairs whose propagation
    record survived the frozen acceptance chain.  Also verifies that the
    rebuilt evaluation reproduces the frozen DQ adjacent-edge CSV rows
    bit-exactly (implementation regression).
    """
    role_set = {str(role) for role in roles}
    banks = []
    reproduction_max_abs_diff = 0.0
    reproduction_rows = 0
    for source in config["real_data_population"]["sources"]:
        if str(source["role"]) not in role_set:
            continue
        source_id = str(source["source_id"])
        roots = _source_roots(config, source)
        events = load_events(roots["identity"] / "synthetic_tracklets.root", require_mc_labels=False)
        records = load_propagation_records(roots["identity"] / "field_candidates.root")
        evaluation = evaluate_field_propagation(
            events, records, require_truth_match=False, q_over_p_mode=0
        )
        index = {
            (
                int(evaluation.run_id[row]),
                int(evaluation.event_id[row]),
                int(evaluation.source_tracklet_id[row]),
                int(evaluation.target_tracklet_id[row]),
            ): row
            for row in range(evaluation.size)
        }
        if len(index) != evaluation.size:
            raise ValueError(f"duplicate evaluation pair identity in {source_id}")

        # Regression: reproduce every frozen DQ adjacent-edge CSV row exactly.
        csv_path = roots["association"] / "selected_route_field_edge_residuals.csv"
        with csv_path.open(encoding="utf-8") as handle:
            csv_rows = list(csv.DictReader(handle))
        for csv_row in csv_rows:
            key = (
                int(csv_row["run_id"]),
                int(csv_row["event_id"]),
                int(csv_row["source_synthetic_tracklet_id"]),
                int(csv_row["target_synthetic_tracklet_id"]),
            )
            row = index.get(key)
            if row is None:
                raise ValueError(f"frozen CSV edge missing from rebuilt evaluation: {key}")
            residual = np.array(
                [float(csv_row[f"residual_{name}"]) for name in ("x_mm", "y_mm", "tx", "ty")]
            )
            reproduction_max_abs_diff = max(
                reproduction_max_abs_diff,
                float(np.max(np.abs(evaluation.residual[row] - residual))),
            )
            reproduction_rows += 1

        routes = _load_routes(roots["association"])
        pair_keys = _anchor_pair_keys(routes)
        rows = []
        missing = 0
        for pair in pair_keys:
            key = (
                pair["run_id"],
                pair["event_id"],
                pair["source_tracklet_id"],
                pair["target_tracklet_id"],
            )
            row = index.get(key)
            if row is None:
                missing += 1
                continue
            rows.append((pair, row))
        if not rows:
            raise ValueError(f"no anchor pairs rebuilt for {source_id}")
        banks.append(
            {
                "source_id": source_id,
                "run_id": int(source["run_id"]),
                "role": str(source["role"]),
                "n_routes": len(routes),
                "n_anchor_pair_keys": len(pair_keys),
                "n_pairs_missing_propagation": missing,
                "residual": np.asarray([evaluation.residual[row] for _, row in rows]),
                "covariance": np.asarray(
                    [evaluation.combined_covariance[row] for _, row in rows]
                ),
                "run_id_arr": np.asarray([pair["run_id"] for pair, _ in rows], dtype=np.int64),
                "event_id": np.asarray([pair["event_id"] for pair, _ in rows], dtype=np.int64),
                "route_index": np.asarray([pair["route_index"] for pair, _ in rows], dtype=np.int64),
                "source_tracklet_id": np.asarray(
                    [pair["source_tracklet_id"] for pair, _ in rows], dtype=np.int64
                ),
                "target_tracklet_id": np.asarray(
                    [pair["target_tracklet_id"] for pair, _ in rows], dtype=np.int64
                ),
                "target_station_id": np.asarray(
                    [pair["target_station_id"] for pair, _ in rows], dtype=np.int64
                ),
            }
        )
    if not banks:
        raise ValueError(f"no real-data sources matched roles {sorted(role_set)}")
    return {
        "kind": "real_data_anchor_pair_bank",
        "roles": sorted(role_set),
        "banks": banks,
        "csv_reproduction": {
            "rows_checked": reproduction_rows,
            "max_abs_diff": reproduction_max_abs_diff,
            "bit_exact": reproduction_max_abs_diff == 0.0,
        },
    }


def bank_sha256(bank: Mapping[str, Any]) -> str:
    payload = {
        "roles": bank["roles"],
        "banks": [
            {
                "source_id": item["source_id"],
                "residual": item["residual"],
                "covariance": item["covariance"],
                "event_id": item["event_id"],
                "target_station_id": item["target_station_id"],
            }
            for item in bank["banks"]
        ],
    }
    return _canonical_sha256(payload)


def concatenate_banks(bank: Mapping[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = {}
    for item in bank["banks"]:
        for key, value in item.items():
            if isinstance(value, np.ndarray):
                arrays.setdefault(key, []).append(value)
    return {key: np.concatenate(values) for key, values in arrays.items()}


# ---------------------------------------------------------------------------
# Observable-model transfer (frozen MC station-pair mean native Jacobian)
# ---------------------------------------------------------------------------


def build_transfer_model(
    pooled: Mapping[str, Any], extras: Mapping[str, Any]
) -> dict[str, Any]:
    """Station-pair mean of the frozen per-pair native Jacobian (MC only)."""
    jacobian = np.asarray(extras["fit"].derivative_native, dtype=np.float64)
    source_station = np.asarray(pooled["source_station_id"], dtype=np.int64)
    target_station = np.asarray(pooled["target_station_id"], dtype=np.int64)
    model: dict[str, np.ndarray] = {}
    diagnostics = {}
    for pair in STATION_PAIRS:
        mask = (source_station == pair[0]) & (target_station == pair[1])
        if not np.any(mask):
            raise ValueError(f"frozen MC bank lacks station pair {pair}")
        block = jacobian[mask]
        mean = block.mean(axis=0)
        model[f"{pair[0]}->{pair[1]}"] = mean
        row_norm = np.linalg.norm(mean, axis=-1)
        spread = block.std(axis=0)
        diagnostics[f"{pair[0]}->{pair[1]}"] = {
            "n_mc_pairs": int(mask.sum()),
            "mean_row_norm": [float(v) for v in row_norm],
            "max_row_relative_spread": float(
                np.max(spread / np.maximum(row_norm[:, None], 1.0e-300))
            ),
        }
    return {
        "kind": "station_pair_mean_native_jacobian",
        "mean_jacobian": model,
        "diagnostics": diagnostics,
        "per_pair_real_data_jacobian_would_require_forbidden_fd_probes": True,
    }


def _pair_design_rows(
    transfer: Mapping[str, Any],
    target_station_id: np.ndarray,
    subspace: IdentifiableSubspace,
) -> np.ndarray:
    """Per-pair identifiable design rows ``J_mean S V_id`` (n_pairs x 4 x 5)."""
    scales = np.asarray(subspace.parameter_scales, dtype=np.float64)
    s_mat = np.diag(scales)
    v_id = np.asarray(subspace.v_id, dtype=np.float64)
    rows = np.zeros((target_station_id.size, 4, v_id.shape[1]), dtype=np.float64)
    for pair in STATION_PAIRS:
        mask = target_station_id == pair[1]
        if np.any(mask):
            mean = transfer["mean_jacobian"][f"{pair[0]}->{pair[1]}"]
            rows[mask] = mean @ s_mat @ v_id
    return rows


def _information_and_rhs(
    design_rows: np.ndarray,
    covariance: np.ndarray,
    residual: np.ndarray,
    *,
    covariance_scale: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    scale = np.asarray(covariance_scale, dtype=np.float64)
    information = np.zeros((design_rows.shape[2], design_rows.shape[2]))
    rhs = np.zeros(design_rows.shape[2])
    for row in range(design_rows.shape[0]):
        cov = covariance[row] * np.diag(scale)
        weight = np.linalg.inv(cov)
        a = design_rows[row]
        information += a.T @ weight @ a
        rhs += a.T @ weight @ residual[row]
    return information, rhs


def _informed_basis(
    information: np.ndarray, *, rank_tolerance: float
) -> tuple[np.ndarray, np.ndarray, int]:
    eigvals, eigvecs = np.linalg.eigh(information)
    if eigvals[-1] <= 0.0:
        return eigvals, np.zeros((information.shape[0], 0)), 0
    rank = int(np.count_nonzero(eigvals > float(rank_tolerance) * eigvals[-1]))
    return eigvals, eigvecs[:, -rank:] if rank else np.zeros((information.shape[0], 0)), rank


def _solve_informed(
    information: np.ndarray,
    rhs: np.ndarray,
    *,
    rank_tolerance: float,
) -> dict[str, Any]:
    eigvals, basis, rank = _informed_basis(information, rank_tolerance=rank_tolerance)
    if rank == 0:
        return {
            "rank": 0,
            "eigenvalues": eigvals,
            "basis": basis,
            "gamma": np.zeros(0),
            "beta": np.zeros(information.shape[0]),
            "restricted_condition": math.inf,
        }
    restricted = basis.T @ information @ basis
    condition = float(np.linalg.cond(restricted))
    try:
        gamma = np.linalg.solve(restricted, basis.T @ rhs)
    except np.linalg.LinAlgError:
        return {
            "rank": 0,
            "eigenvalues": eigvals,
            "basis": np.zeros((information.shape[0], 0)),
            "gamma": np.zeros(0),
            "beta": np.zeros(information.shape[0]),
            "restricted_condition": math.inf,
        }
    return {
        "rank": rank,
        "eigenvalues": eigvals,
        "basis": basis,
        "gamma": gamma,
        "beta": basis @ gamma,
        "restricted_condition": condition,
    }


# ---------------------------------------------------------------------------
# Gauge representatives (null-space formulation of the WB77 KKT contract)
# ---------------------------------------------------------------------------


def gauge_representatives(
    beta: np.ndarray,
    subspace: IdentifiableSubspace,
    gauges: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """Gauge-fixed representatives ``theta(g)`` of an identifiable solution.

    ``theta(g) = S (V_id beta + V_null mu(g))`` with
    ``mu(g) = -(G S V_null)^{-1} G S V_id beta`` so that ``G theta(g) = 0``.
    The identifiable projection ``P_id theta`` and every observable
    prediction are gauge-independent by construction; absolute gauge-dependent
    components differ and are never compared as physics.
    """
    scales = np.asarray(subspace.parameter_scales, dtype=np.float64)
    s_mat = np.diag(scales)
    v_id = np.asarray(subspace.v_id, dtype=np.float64)
    v_null = np.asarray(subspace.v_null, dtype=np.float64)
    representatives = {}
    for gauge_id, g_native in gauges.items():
        intersection = g_native @ s_mat @ v_null
        mu = -np.linalg.solve(intersection, g_native @ s_mat @ v_id @ beta)
        theta = s_mat @ (v_id @ beta + v_null @ mu)
        representatives[gauge_id] = {
            "theta_native": theta,
            "null_coordinates_mu": mu,
            "gauge_residual_max_abs": float(np.max(np.abs(g_native @ theta))),
            "identifiable_projection_beta": beta.copy(),
        }
    return representatives


def predicted_observable_change(
    transfer: Mapping[str, Any],
    target_station_id: np.ndarray,
    subspace: IdentifiableSubspace,
    beta: np.ndarray,
) -> np.ndarray:
    """Gauge-invariant predicted observable change ``J_mean S V_id beta``."""
    scales = np.asarray(subspace.parameter_scales, dtype=np.float64)
    v_id = np.asarray(subspace.v_id, dtype=np.float64)
    direction = np.diag(scales) @ v_id @ beta
    delta = np.zeros((target_station_id.size, 4), dtype=np.float64)
    for pair in STATION_PAIRS:
        mask = target_station_id == pair[1]
        if np.any(mask):
            delta[mask] = transfer["mean_jacobian"][f"{pair[0]}->{pair[1]}"] @ direction
    return delta


# ---------------------------------------------------------------------------
# MC control (before any real-data solve): transfer closure + null ensemble
# ---------------------------------------------------------------------------


def _sample_pairs(
    rng: np.random.Generator,
    pooled_index: Mapping[str, np.ndarray],
    composition: Mapping[str, int],
) -> np.ndarray:
    rows = []
    for pair_label, count in composition.items():
        available = pooled_index[pair_label]
        rows.append(rng.choice(available, size=int(count), replace=True))
    return np.concatenate(rows)


def mc_control(
    config: Mapping[str, Any],
    pooled: Mapping[str, Any],
    subspace: IdentifiableSubspace,
    extras: Mapping[str, Any],
    transfer: Mapping[str, Any],
    gauges: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Pre-registered MC control for the transfer + solve + gate machinery.

    Mirrors the real-data pipeline on the frozen MC bank with the frozen real
    calibration/held-out anchor-pair composition.  Runs before any real
    calibration residual is used for a solve.  Produces the null-improvement
    floor consumed by held-out gate A.
    """
    control = config["mc_control"]
    rank_tolerance = float(config["solver"]["rank_tolerance"])
    jacobian = np.asarray(extras["fit"].derivative_native, dtype=np.float64)
    covariance = np.asarray(pooled["covariance"], dtype=np.float64)
    anchor_residual = np.asarray(pooled["anchor_residual"], dtype=np.float64)
    source_station = np.asarray(pooled["source_station_id"], dtype=np.int64)
    target_station = np.asarray(pooled["target_station_id"], dtype=np.int64)
    scales = np.asarray(subspace.parameter_scales, dtype=np.float64)
    s_mat = np.diag(scales)
    v_id = np.asarray(subspace.v_id, dtype=np.float64)

    pooled_index = {
        f"{pair[0]}->{pair[1]}": np.flatnonzero(
            (source_station == pair[0]) & (target_station == pair[1])
        )
        for pair in STATION_PAIRS
    }
    # Target-station array for a synthetic composition (for predictions).
    def _composition_targets(composition: Mapping[str, int]) -> np.ndarray:
        values = []
        for pair_label, count in composition.items():
            values.extend([int(pair_label.split("->")[1])] * int(count))
        return np.asarray(values, dtype=np.int64)

    rng = np.random.default_rng(int(control["seed"]))
    amplitude = float(control["injection_amplitude_scaled"])
    recovery_gate = float(control["injection_recovery_max_relative"])
    leakage_gate = float(control["leakage_max_relative"])
    invariance_gate = float(control["gauge_invariance_max_relative_diff"])
    n_null = int(control["null_ensemble_replicates"])
    quantile = float(control["null_improvement_quantile"])

    scenario_reports = []
    null_improvements: list[float] = []
    for scenario in control["scenarios"]:
        scale_map = scenario["covariance_scale"]
        covariance_scale = (
            float(scale_map["x"]),
            float(scale_map["y"]),
            float(scale_map["tx"]),
            float(scale_map["ty"]),
        )
        calib_rows = _sample_pairs(rng, pooled_index, control["calibration_composition"])
        calib_targets = target_station[calib_rows]
        design = _pair_design_rows(transfer, calib_targets, subspace)
        information, _rhs = _information_and_rhs(
            design,
            covariance[calib_rows],
            anchor_residual[calib_rows],
            covariance_scale=covariance_scale,
        )
        eigvals, basis, rank = _informed_basis(information, rank_tolerance=rank_tolerance)

        # Injection closure: recover known injections through the exact
        # per-pair MC Jacobian while solving with the transfer model.
        injection_report = []
        directions = [("informed", basis[:, k]) for k in range(rank)]
        if 0 < rank < information.shape[0]:
            complement = np.linalg.qr(
                np.concatenate([basis, np.eye(information.shape[0])], axis=1)
            )[0][:, rank : information.shape[0]]
            directions.append(("uninformed", complement[:, 0]))
        for label, direction in directions:
            direction = direction / np.linalg.norm(direction)
            beta_inj = amplitude * direction
            theta_inj = s_mat @ v_id @ beta_inj
            injected = anchor_residual[calib_rows] + np.einsum(
                "nij,j->ni", jacobian[calib_rows], theta_inj
            )
            _info, rhs = _information_and_rhs(
                design,
                covariance[calib_rows],
                injected,
                covariance_scale=covariance_scale,
            )
            solved = _solve_informed(information, rhs, rank_tolerance=rank_tolerance)
            proj_inj = basis.T @ beta_inj
            proj_hat = basis.T @ solved["beta"]
            abs_err = np.abs(proj_hat - proj_inj)
            rel = float(np.max(abs_err) / amplitude)
            injection_report.append(
                {
                    "direction": label,
                    "injected_projection": [float(v) for v in proj_inj],
                    "recovered_projection": [float(v) for v in proj_hat],
                    "max_relative_deviation": rel,
                    "within_gate": rel
                    <= (recovery_gate if label == "informed" else leakage_gate),
                }
            )

        # Null ensemble: spurious held-out improvement under zero truth.
        held_out_targets = _composition_targets(control["held_out_composition"])
        held_out_design = _pair_design_rows(transfer, held_out_targets, subspace)
        scenario_nulls = []
        for _replicate in range(n_null):
            null_calib = _sample_pairs(rng, pooled_index, control["calibration_composition"])
            null_held = _sample_pairs(rng, pooled_index, control["held_out_composition"])
            null_design = _pair_design_rows(transfer, target_station[null_calib], subspace)
            n_info, n_rhs = _information_and_rhs(
                null_design,
                covariance[null_calib],
                anchor_residual[null_calib],
                covariance_scale=covariance_scale,
            )
            solved = _solve_informed(n_info, n_rhs, rank_tolerance=rank_tolerance)
            if solved["rank"] == 0:
                continue
            beta = v_id @ solved["beta"]
            direction = s_mat @ beta
            delta = np.zeros((null_held.size, 4))
            for pair in STATION_PAIRS:
                mask = target_station[null_held] == pair[1]
                if np.any(mask):
                    delta[mask] = transfer["mean_jacobian"][f"{pair[0]}->{pair[1]}"] @ direction
            chi2_before = 0.0
            chi2_after = 0.0
            for i, row in enumerate(null_held):
                cov = covariance[row] * np.diag(np.asarray(covariance_scale))
                weight = np.linalg.inv(cov)
                r = anchor_residual[row]
                chi2_before += float(r @ weight @ r)
                chi2_after += float((r - delta[i]) @ weight @ (r - delta[i]))
            scenario_nulls.append(chi2_after - chi2_before)
        improvements = [-value for value in scenario_nulls]
        null_improvements.extend(improvements)
        floor = (
            float(np.quantile(np.asarray(improvements), quantile)) if improvements else 0.0
        )

        # Gauge invariance on this scenario's information.
        _info, rhs = _information_and_rhs(
            design,
            covariance[calib_rows],
            anchor_residual[calib_rows],
            covariance_scale=covariance_scale,
        )
        solved = _solve_informed(information, rhs, rank_tolerance=rank_tolerance)
        representatives = gauge_representatives(solved["beta"], subspace, gauges)
        predictions = {
            gauge_id: predicted_observable_change(
                transfer, calib_targets, subspace, rep["identifiable_projection_beta"]
            )
            for gauge_id, rep in representatives.items()
        }
        max_invariance = 0.0
        ids = sorted(predictions)
        for left in range(len(ids)):
            for right in range(left + 1, len(ids)):
                diff = float(np.max(np.abs(predictions[ids[left]] - predictions[ids[right]])))
                norm = float(np.max(np.abs(predictions[ids[left]])))
                max_invariance = max(max_invariance, diff / max(norm, 1.0e-300))
        max_gauge_residual = max(
            rep["gauge_residual_max_abs"] for rep in representatives.values()
        )

        scenario_reports.append(
            {
                "name": str(scenario["name"]),
                "covariance_scale": scenario["covariance_scale"],
                "informed_rank": int(rank),
                "information_eigenvalues": [float(v) for v in eigvals],
                "injection": injection_report,
                "null_replicates": len(scenario_nulls),
                "null_improvement_floor_q95": floor,
                "gauge_invariance_max_relative_diff": max_invariance,
                "gauge_residual_max_abs": max_gauge_residual,
                "pass": (
                    rank > 0
                    and all(item["within_gate"] for item in injection_report)
                    and max_invariance <= invariance_gate
                    and max_gauge_residual
                    <= float(config["gauges"]["gauge_residual_max_abs"])
                ),
            }
        )

    overall_floor = max(
        (report["null_improvement_floor_q95"] for report in scenario_reports), default=0.0
    )
    return {
        "kind": "mc_control",
        "seed": int(control["seed"]),
        "scenarios": scenario_reports,
        "null_improvement_floor": max(overall_floor, 0.0),
        "injection_recovery_max_relative": recovery_gate,
        "leakage_max_relative": leakage_gate,
        "pass": all(report["pass"] for report in scenario_reports),
    }


# ---------------------------------------------------------------------------
# Calibration: applicability audit + one-shot solve + bootstrap + freeze
# ---------------------------------------------------------------------------


def applicability_audit(
    config: Mapping[str, Any],
    bank: Mapping[str, Any],
    transfer: Mapping[str, Any],
    pooled: Mapping[str, Any],
) -> dict[str, Any]:
    """Pre-fit applicability / linearity audit on the calibration subset.

    Report-first: nothing here selects events.  Hard gates are only
    non-finite/corrupt data and the bank/CSV bit-exact reproduction.
    """
    arrays = concatenate_banks(bank)
    residual = arrays["residual"]
    covariance = arrays["covariance"]
    finite_ok = bool(np.isfinite(residual).all() and np.isfinite(covariance).all())
    diag = np.diagonal(covariance, axis1=1, axis2=2)
    covariance_ok = bool((diag > 0.0).all()) and bool(
        np.allclose(covariance, np.swapaxes(covariance, 1, 2), rtol=1.0e-7, atol=1.0e-12)
    )
    chi2_zero = 0.0
    for row in range(residual.shape[0]):
        weight = np.linalg.inv(covariance[row])
        chi2_zero += float(residual[row] @ weight @ residual[row])
    per_observable = {}
    for index, name in enumerate(OBSERVABLE_NAMES):
        values = residual[:, index]
        per_observable[name] = {
            "median": float(np.median(values)),
            "rms": float(np.sqrt(np.mean(np.square(values)))),
            "p05": float(np.percentile(values, 5.0)),
            "p95": float(np.percentile(values, 95.0)),
        }
    per_run = {}
    for item in bank["banks"]:
        r = item["residual"]
        c = item["covariance"]
        chi2 = 0.0
        for row in range(r.shape[0]):
            weight = np.linalg.inv(c[row])
            chi2 += float(r[row] @ weight @ r[row])
        per_run[str(item["run_id"])] = {
            "n_pairs": int(r.shape[0]),
            "chi2_zero_candidate": chi2,
            "median_residual_y_mm": float(np.median(r[:, 1])),
            "median_residual_x_mm": float(np.median(r[:, 0])),
        }
    mc_covariance = np.asarray(pooled["covariance"], dtype=np.float64)
    w_applicability = {
        "real_median_diag_covariance": [float(v) for v in np.median(diag, axis=0)],
        "mc_median_diag_covariance": [
            float(v) for v in np.median(np.diagonal(mc_covariance, axis1=1, axis2=2), axis=0)
        ],
        "note": "real x/tx rows carry ~300x/~800x larger covariance than MC; "
        "this is a frozen property of the collision-data population, not a tuning knob",
    }
    hard_fail = not (finite_ok and covariance_ok)
    return {
        "kind": "pre_fit_applicability_audit",
        "n_pairs": int(residual.shape[0]),
        "finite_ok": finite_ok,
        "covariance_positive_symmetric_ok": covariance_ok,
        "hard_fail": hard_fail,
        "weighted_residual_norm_zero_candidate": chi2_zero,
        "per_observable": per_observable,
        "per_run": per_run,
        "w_applicability": w_applicability,
        "csv_reproduction": bank["csv_reproduction"],
        "no_event_selection_applied": True,
    }


def solve_candidate(
    config: Mapping[str, Any],
    subspace: IdentifiableSubspace,
    transfer: Mapping[str, Any],
    bank: Mapping[str, Any],
    gauges: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """One-shot gauge-fixed candidate from the calibration subset only."""
    arrays = concatenate_banks(bank)
    design = _pair_design_rows(transfer, arrays["target_station_id"], subspace)
    information, rhs = _information_and_rhs(design, arrays["covariance"], arrays["residual"])
    solved = _solve_informed(
        information, rhs, rank_tolerance=float(config["solver"]["rank_tolerance"])
    )
    result: dict[str, Any] = {
        "kind": "one_shot_gauge_fixed_candidate",
        "informed_rank": int(solved["rank"]),
        "information_eigenvalues": [float(v) for v in solved["eigenvalues"]],
        "restricted_condition": float(solved["restricted_condition"]),
        "solver_method": str(config["solver"]["method"]),
        "one_shot_no_iteration": True,
    }
    if solved["rank"] == 0:
        result["status"] = DECISION_RANK_ZERO
        return result
    gamma = solved["gamma"]
    beta = solved["beta"]
    max_amplitude = float(np.max(np.abs(gamma)))
    envelope = float(config["applicability"]["max_scaled_mode_amplitude"])
    representatives = gauge_representatives(beta, subspace, gauges)
    max_gauge_residual = max(
        rep["gauge_residual_max_abs"] for rep in representatives.values()
    )
    predicted_calibration = predicted_observable_change(
        transfer, arrays["target_station_id"], subspace, beta
    )
    result.update(
        {
            "gamma_hat_informed_coordinates": [float(v) for v in gamma],
            "beta_hat_identifiable_projection": [float(v) for v in beta],
            "informed_basis_columns": [
                [float(v) for v in solved["basis"][:, k]] for k in range(solved["rank"])
            ],
            "max_scaled_mode_amplitude": max_amplitude,
            "linear_envelope": envelope,
            "within_linear_envelope": bool(max_amplitude <= envelope),
            "gauge_residual_max_abs": max_gauge_residual,
            "solver_within_condition_gate": bool(
                solved["restricted_condition"] <= float(config["solver"]["max_informed_condition"])
            ),
            "predicted_calibration_observable_change_norm": float(
                np.linalg.norm(predicted_calibration)
            ),
            "reconstruction_gauge_representative": {
                gauge_id: {
                    "theta_native": [float(v) for v in rep["theta_native"]],
                    "null_coordinates_mu": [float(v) for v in rep["null_coordinates_mu"]],
                    "gauge_residual_max_abs": rep["gauge_residual_max_abs"],
                }
                for gauge_id, rep in representatives.items()
            },
            "uninformed_identifiable_modes_zero_update_not_informed_by_real_data": True,
            "parameters_are_reconstruction_gauge_representatives_not_measurements": True,
        }
    )

    # Gate E: event-level bootstrap stability of the identifiable projection.
    rng = np.random.default_rng(int(config["solver"]["bootstrap_seed"]))
    n_bootstrap = int(config["solver"]["bootstrap_replicates"])
    target_station = arrays["target_station_id"]
    n_pairs = target_station.size
    angles = []
    rank_changes = 0
    for _replicate in range(n_bootstrap):
        rows = []
        for pair in STATION_PAIRS:
            available = np.flatnonzero(target_station == pair[1])
            rows.append(rng.choice(available, size=available.size, replace=True))
        sample = np.concatenate(rows)
        b_design = _pair_design_rows(transfer, target_station[sample], subspace)
        b_info, b_rhs = _information_and_rhs(
            b_design, arrays["covariance"][sample], arrays["residual"][sample]
        )
        b_solved = _solve_informed(
            b_info, b_rhs, rank_tolerance=float(config["solver"]["rank_tolerance"])
        )
        if b_solved["rank"] != solved["rank"]:
            rank_changes += 1
            continue
        cosine = float(
            np.dot(b_solved["beta"], beta)
            / max(np.linalg.norm(b_solved["beta"]) * np.linalg.norm(beta), 1.0e-300)
        )
        angles.append(math.degrees(math.acos(min(max(cosine, -1.0), 1.0))))
    max_angle = max(angles) if angles else math.inf
    result["bootstrap"] = {
        "replicates": n_bootstrap,
        "rank_changes": rank_changes,
        "max_direction_angle_deg": max_angle,
        "gate_deg": float(config["solver"]["bootstrap_max_direction_angle_deg"]),
        "stable": bool(
            rank_changes == 0
            and max_angle <= float(config["solver"]["bootstrap_max_direction_angle_deg"])
        ),
    }
    result["status"] = "solved"
    return result


def freeze_candidate_artifact(
    config: Mapping[str, Any],
    *,
    split: Mapping[str, Any],
    bank: Mapping[str, Any],
    audit: Mapping[str, Any],
    candidate: Mapping[str, Any],
    subspace: IdentifiableSubspace,
    gauges: Mapping[str, np.ndarray],
    mc_control_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the frozen candidate artifact (SHA computed by the caller)."""
    return {
        "kind": "gauge_fixed_candidate_diagnostic",
        "schema_version": SCHEMA_VERSION,
        "git_head_sha": git_head_sha(),
        "config_path": str(config.get("config_path", "")),
        "config_sha256": sha256_file(Path(str(config["config_path"]))),
        "calibration_provenance": {
            "sources": [
                {
                    "source_id": item["source_id"],
                    "run_id": item["run_id"],
                    "n_routes": item["n_routes"],
                    "n_anchor_pairs": int(item["residual"].shape[0]),
                    "n_anchor_pair_keys": item["n_anchor_pair_keys"],
                    "n_pairs_missing_propagation": item["n_pairs_missing_propagation"],
                }
                for item in bank["banks"]
            ],
            "split_sha256": split["split_sha256"],
            "input_bank_sha256": bank_sha256(bank),
        },
        "tracker_information_provenance": {
            "workbook_68_identifiable_basis_sha256": str(
                config["inheritance"]["workbook_68_identifiable_basis_sha256"]
            ),
            "parameter_names": list(PARAMETER_NAMES),
            "parameter_units": ["mm", "mm", "mm", "mrad", "mrad", "mrad", "mm"],
            "rank_tolerance": float(config["tracker_information"]["rank_tolerance"]),
        },
        "gauge_matrices_native": {
            gauge_id: [[float(v) for v in row] for row in g]
            for gauge_id, g in gauges.items()
        },
        "primary_gauge": str(config["gauges"]["primary"]),
        "applicability_audit": audit,
        "mc_control_null_improvement_floor": float(mc_control_report["null_improvement_floor"]),
        "candidate": candidate,
        "frozen_before_held_out_access": True,
    }


# ---------------------------------------------------------------------------
# Held-out evaluation (only after the candidate artifact is frozen)
# ---------------------------------------------------------------------------


def predict_delta_for_pairs(
    transfer: Mapping[str, Any],
    source_station_id: np.ndarray,
    target_station_id: np.ndarray,
    subspace: IdentifiableSubspace,
    beta: np.ndarray,
) -> np.ndarray:
    """Predicted observable change for arbitrary station pairs.

    The frozen 7D parameterization moves only IFT (station 0); pairs not
    involving station 0 have exactly zero predicted correction.  This is the
    null control for the held-out evaluation.
    """
    delta = np.zeros((source_station_id.size, 4), dtype=np.float64)
    scales = np.asarray(subspace.parameter_scales, dtype=np.float64)
    v_id = np.asarray(subspace.v_id, dtype=np.float64)
    direction = np.diag(scales) @ v_id @ beta
    for pair in STATION_PAIRS:
        mask = (source_station_id == pair[0]) & (target_station_id == pair[1])
        if np.any(mask):
            delta[mask] = transfer["mean_jacobian"][f"{pair[0]}->{pair[1]}"] @ direction
    return delta


def load_frozen_csv_edges(
    config: Mapping[str, Any], *, roles: Sequence[str]
) -> dict[str, np.ndarray]:
    """Frozen DQ adjacent-edge CSV rows for the sources in ``roles``."""
    role_set = {str(role) for role in roles}
    residual = []
    covariance = []
    source_station = []
    target_station = []
    run_ids = []
    for source in config["real_data_population"]["sources"]:
        if str(source["role"]) not in role_set:
            continue
        roots = _source_roots(config, source)
        with (roots["association"] / "selected_route_field_edge_residuals.csv").open(
            encoding="utf-8"
        ) as handle:
            for row in csv.DictReader(handle):
                residual.append(
                    [float(row[f"residual_{name}"]) for name in ("x_mm", "y_mm", "tx", "ty")]
                )
                covariance.append(
                    [
                        [
                            float(row["combined_cov_xx_mm2"]),
                            float(row["combined_cov_xy_mm2"]),
                            float(row["combined_cov_xtx_mm"]),
                            float(row["combined_cov_xty_mm"]),
                        ],
                        [
                            float(row["combined_cov_xy_mm2"]),
                            float(row["combined_cov_yy_mm2"]),
                            float(row["combined_cov_ytx_mm"]),
                            float(row["combined_cov_yty_mm"]),
                        ],
                        [
                            float(row["combined_cov_xtx_mm"]),
                            float(row["combined_cov_ytx_mm"]),
                            float(row["combined_cov_txtx"]),
                            float(row["combined_cov_txty"]),
                        ],
                        [
                            float(row["combined_cov_xty_mm"]),
                            float(row["combined_cov_yty_mm"]),
                            float(row["combined_cov_txty"]),
                            float(row["combined_cov_tyty"]),
                        ],
                    ]
                )
                source_station.append(int(row["source_station_id"]))
                target_station.append(int(row["target_station_id"]))
                run_ids.append(int(row["run_id"]))
    return {
        "residual": np.asarray(residual, dtype=np.float64),
        "covariance": np.asarray(covariance, dtype=np.float64),
        "source_station_id": np.asarray(source_station, dtype=np.int64),
        "target_station_id": np.asarray(target_station, dtype=np.int64),
        "run_id": np.asarray(run_ids, dtype=np.int64),
    }


def evaluate_held_out(
    config: Mapping[str, Any],
    subspace: IdentifiableSubspace,
    transfer: Mapping[str, Any],
    candidate_artifact: Mapping[str, Any],
    held_out_bank: Mapping[str, Any],
    held_out_csv_edges: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Gauge-invariant held-out evaluation of the frozen candidate."""
    candidate = candidate_artifact["candidate"]
    beta = np.asarray(candidate["beta_hat_identifiable_projection"], dtype=np.float64)
    arrays = concatenate_banks(held_out_bank)
    residual = arrays["residual"]
    covariance = arrays["covariance"]
    target_station = arrays["target_station_id"]
    run_ids = arrays["run_id_arr"]
    delta = predicted_observable_change(transfer, target_station, subspace, beta)

    def _chi2(rows: np.ndarray, subtract: bool) -> float:
        total = 0.0
        for i in rows:
            weight = np.linalg.inv(covariance[i])
            r = residual[i] - delta[i] if subtract else residual[i]
            total += float(r @ weight @ r)
        return total

    all_rows = np.arange(residual.shape[0])
    chi2_before = _chi2(all_rows, subtract=False)
    chi2_after = _chi2(all_rows, subtract=True)
    delta_chi2 = chi2_after - chi2_before

    # Gate A: improvement beyond the pre-registered MC null floor.
    floor = float(candidate_artifact["mc_control_null_improvement_floor"])
    gate_a = bool(delta_chi2 < 0.0 and -delta_chi2 > floor)

    # Gate B: run-level consistency.
    per_run = {}
    sufficient_runs = []
    min_pairs = int(config["split"]["min_anchor_pairs_for_run_gates"])
    for run in sorted(set(int(v) for v in run_ids)):
        rows = np.flatnonzero(run_ids == run)
        before = _chi2(rows, subtract=False)
        after = _chi2(rows, subtract=True)
        per_run[str(run)] = {
            "n_pairs": int(rows.size),
            "chi2_before": before,
            "chi2_after": after,
            "delta_chi2": after - before,
            "sufficient": bool(rows.size >= min_pairs),
        }
        if rows.size >= min_pairs:
            sufficient_runs.append(run)
    primary_runs = [int(run) for run in config["held_out_gates"]["primary_held_out_runs"]]
    primary_improve = all(per_run[str(run)]["delta_chi2"] < 0.0 for run in primary_runs)
    leave_one_out_ok = True
    leave_one_out = {}
    for excluded in sufficient_runs:
        rows = np.flatnonzero(run_ids != excluded)
        d = _chi2(rows, subtract=True) - _chi2(rows, subtract=False)
        leave_one_out[str(excluded)] = d
        leave_one_out_ok = leave_one_out_ok and d < 0.0
    gate_b = bool(primary_improve and leave_one_out_ok)

    # Gate C: per station-pair x observable cell catastrophic-degradation bound.
    cells = {}
    cell_ok = True
    for pair in STATION_PAIRS:
        pair_rows = np.flatnonzero(target_station == pair[1])
        if pair_rows.size == 0:
            continue
        for obs_index, obs_name in enumerate(OBSERVABLE_NAMES):
            w = 1.0 / np.asarray(
                [covariance[i][obs_index, obs_index] for i in pair_rows]
            )
            before = float(
                np.sum(residual[pair_rows, obs_index] ** 2 * w)
            )
            after = float(
                np.sum((residual[pair_rows, obs_index] - delta[pair_rows, obs_index]) ** 2 * w)
            )
            bound = 2.0 * math.sqrt(2.0 * pair_rows.size)
            worsening = after - before
            cells[f"{pair[0]}->{pair[1]}:{obs_name}"] = {
                "n": int(pair_rows.size),
                "delta_chi2_diagonal": worsening,
                "bound": bound,
                "within_bound": bool(worsening <= bound),
            }
            cell_ok = cell_ok and worsening <= bound
    # Null control: frozen CSV adjacent edges that do not involve station 0
    # ((1,2), (2,3)) must see exactly zero predicted correction, hence exactly
    # zero chi2 change; (0,1) CSV edges must agree with the bank evaluation.
    csv_residual = held_out_csv_edges["residual"]
    csv_covariance = held_out_csv_edges["covariance"]
    csv_delta = predict_delta_for_pairs(
        transfer,
        held_out_csv_edges["source_station_id"],
        held_out_csv_edges["target_station_id"],
        subspace,
        beta,
    )
    non_ift = held_out_csv_edges["source_station_id"] != 0
    null_control_max_abs_prediction = (
        float(np.max(np.abs(csv_delta[non_ift]))) if np.any(non_ift) else 0.0
    )
    null_control_chi2_change = 0.0
    for i in np.flatnonzero(non_ift):
        weight = np.linalg.inv(csv_covariance[i])
        r = csv_residual[i]
        null_control_chi2_change += float(
            (r - csv_delta[i]) @ weight @ (r - csv_delta[i]) - r @ weight @ r
        )
    null_control_ok = bool(
        null_control_max_abs_prediction == 0.0 and null_control_chi2_change == 0.0
    )
    gate_c = bool(cell_ok and null_control_ok)

    # Gate D: gauge invariance of held-out predictions.
    representatives = candidate["reconstruction_gauge_representative"]
    v_id = np.asarray(subspace.v_id, dtype=np.float64)
    predictions = {}
    for gauge_id, rep in representatives.items():
        theta = np.asarray(rep["theta_native"], dtype=np.float64)
        scales = np.asarray(subspace.parameter_scales, dtype=np.float64)
        u = theta / scales  # u = S^{-1} theta (scaled coordinates)
        beta_g = v_id.T @ u  # identifiable projection (gauge-invariant part)
        predictions[gauge_id] = predicted_observable_change(
            transfer, target_station, subspace, beta_g
        )
    max_invariance = 0.0
    ids = sorted(predictions)
    for left in range(len(ids)):
        for right in range(left + 1, len(ids)):
            diff = float(np.max(np.abs(predictions[ids[left]] - predictions[ids[right]])))
            norm = float(np.max(np.abs(predictions[ids[left]])))
            max_invariance = max(max_invariance, diff / max(norm, 1.0e-300))
    gate_d = bool(
        max_invariance <= float(config["held_out_gates"]["gauge_invariance_max_relative_diff"])
    )

    # Gate E: bootstrap stability (from the frozen candidate artifact).
    gate_e = bool(candidate["bootstrap"]["stable"])

    # Gate F: DQ slices on isolation channels (y_mm, ty) per sufficient run.
    gates_f = config["held_out_gates"]
    ref_dy = float(gates_f["frozen_reference_dy_median_mm"])
    scale_dy = float(gates_f["frozen_reference_dy_robust_scale_mm"])
    ref_rx = float(gates_f["frozen_reference_rx_median"])
    scale_rx = float(gates_f["frozen_reference_rx_robust_scale"])
    alarm = float(gates_f["robust_z_detector_condition"])
    dq_slices = {}
    gate_f = True
    for run in sufficient_runs:
        rows = np.flatnonzero(run_ids == run)
        entry = {}
        for obs_index, channel, ref, scale in (
            (1, "dy", ref_dy, scale_dy),
            (3, "rx", ref_rx, scale_rx),
        ):
            pre = float(np.median(residual[rows, obs_index]))
            post = float(np.median((residual - delta)[rows, obs_index]))
            z_pre = (pre - ref) / scale
            z_post = (post - ref) / scale
            entry[channel] = {
                "median_pre": pre,
                "median_post": post,
                "robust_z_pre": z_pre,
                "robust_z_post": z_post,
                "not_moved_away_from_zero": abs(post) <= abs(pre),
                "new_alarm": bool(abs(z_post) > alarm and abs(z_pre) <= alarm),
            }
            if not entry[channel]["not_moved_away_from_zero"] or entry[channel]["new_alarm"]:
                gate_f = False
        dq_slices[str(run)] = entry

    # Gate G: amplitude within the frozen linear envelope (frozen candidate).
    gate_g = bool(candidate["within_linear_envelope"])

    gates = {
        "A_held_out_weighted_residual_improvement_beyond_null_floor": gate_a,
        "B_run_level_consistency": gate_b,
        "C_no_catastrophic_cell_degradation": gate_c,
        "D_gauge_invariant_held_out_prediction": gate_d,
        "E_identifiable_projection_bootstrap_stable": gate_e,
        "F_no_new_systematic_shift_in_dq_slices": gate_f,
        "G_correction_within_linear_envelope": gate_g,
    }
    return {
        "kind": "held_out_gauge_invariant_evaluation",
        "n_pairs": int(residual.shape[0]),
        "chi2_before": chi2_before,
        "chi2_after": chi2_after,
        "delta_chi2": delta_chi2,
        "null_improvement_floor": floor,
        "per_run": per_run,
        "leave_one_run_out_delta_chi2": leave_one_out,
        "per_cell": cells,
        "null_control_non_ift_edges": {
            "n_edges": int(np.count_nonzero(non_ift)),
            "max_abs_predicted_correction": null_control_max_abs_prediction,
            "chi2_change": null_control_chi2_change,
            "exactly_zero_as_required": null_control_ok,
        },
        "gauge_invariance_max_relative_diff": max_invariance,
        "dq_slices": dq_slices,
        "gates": gates,
        "pass": all(gates.values()),
        "no_post_hoc_metric_selection": True,
    }


# ---------------------------------------------------------------------------
# Decision tree
# ---------------------------------------------------------------------------


def decide_campaign(
    *,
    regression: Mapping[str, Any],
    mc_control_report: Mapping[str, Any] | None,
    audit: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None,
    held_out: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not bool(regression["pass"]):
        decision = DECISION_TRACKER_REGRESSION_FAILED
    elif mc_control_report is None or not bool(mc_control_report["pass"]):
        decision = DECISION_TRANSFER_FAILED
    elif audit is None or bool(audit["hard_fail"]):
        decision = DECISION_INCONCLUSIVE
    elif candidate is None:
        decision = DECISION_INCONCLUSIVE
    elif candidate.get("status") == DECISION_RANK_ZERO:
        decision = DECISION_RANK_ZERO
    elif not bool(candidate["within_linear_envelope"]):
        decision = DECISION_OUT_OF_SUPPORT
    elif not bool(candidate["solver_within_condition_gate"]) or not bool(
        candidate["bootstrap"]["stable"]
    ):
        decision = DECISION_NOT_STABLE
    elif held_out is None:
        decision = DECISION_INCONCLUSIVE
    elif not bool(held_out["gates"]["D_gauge_invariant_held_out_prediction"]):
        decision = DECISION_GAUGE_INVARIANCE_FAILED
    elif not bool(held_out["pass"]):
        decision = DECISION_NOT_IMPROVED
    else:
        decision = DECISION_PASS

    return {
        "kind": "next_stage_decision",
        "decision": decision,
        "diagnostic_pass": decision == DECISION_PASS,
        "geometry_write_allowed": False,
        "official_conditions_write_allowed": False,
        "real_data_candidate_alignment_authorized": False,
        "external_constraint_ingest_authorized": False,
        "eligible_external_physical_constraints": [],
        "output_parameters_are_reconstruction_gauge_representatives": True,
        "next_stage_if_pass": (
            "gauge_fixed_temporary_reconstruction_candidate_validation_v1_"
            "requires_separate_preregistration"
        ),
        "if_fail_continue": "residual_dq_monitoring_only",
        "official_cool_pool_write_remains_closed": True,
    }
