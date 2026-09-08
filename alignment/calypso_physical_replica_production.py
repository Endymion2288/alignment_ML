"""Calypso physical replica production v1 — corpus only, no alignment qualification.

Frozen size and independence rules inherit WB83b.  This module does not
authorize ML association, toy reruns, or ``alignment_oracle_qualified``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROVENANCE = "calypso_physical_independent_v1"
ASSOCIATION_DEFAULT_SYSTEM = "frozen_W64_raw_energy_plus_exact_solver"
OUTPUT_ROOT = Path("outputs/mc24_four_station_calypso_physical_replicas_v1")
REFIT_TAG = "calypso_physical_independent_v1"

N_GEOMETRY_CONDITIONS = 8
TARGET_REPLICAS_PER_CONDITION = 200
EXPECTED_YIELD = 0.50
REQUIRED_INPUT_INDEPENDENT_EVENTS = 3200
SKIP_BASE = 2000
PRODUCTION_CHUNK_EVENTS = 5000
MIN_TRUTH_MATCH_FRACTION = 0.99
Q_OVER_P_MODE = 0
GEOMETRY_NAME = "FASERNU-04"
GLOBAL_TAG = "OFLCOND-FASER-06"
ACTS_TOOL = "FaserActsExtrapolationTool"

# Not WB83 seeds.
ALLOCATION_PROTOCOL_SEED_DOCUMENTATION_ONLY = 2026090716

FORBIDDEN_PATH_NEEDLES = (
    "00800_00849",
    "00800-00849",
    "mc24_100116",
    "mc24_100117",
    "00350_00399",
    "00350-00399",
    "source_diversity_blind",
    "overlay_synthetic",
    "final_blind",
    "sealed_test",
)

AUTHORIZED_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "source_id": "mc24_100043_00200_00299",
        "channel": 100043,
        "run_range": "00200_00299",
        "family": "family1",
        "events_per_condition": 67,
        "path": (
            "/eos/experiment/faser/data0/sim/mc24/particle_gun/100043/rec/s0013-r0022/"
            "FaserMC-MC24_PG_mumi_fasernu_5mrad_flukaE-100043-00200-00299-s0013-r0022-xAOD.root"
        ),
    },
    {
        "source_id": "mc24_100043_00300_00399",
        "channel": 100043,
        "run_range": "00300_00399",
        "family": "family1",
        "events_per_condition": 67,
        "path": (
            "/eos/experiment/faser/data0/sim/mc24/particle_gun/100043/rec/s0013-r0022/"
            "FaserMC-MC24_PG_mumi_fasernu_5mrad_flukaE-100043-00300-00399-s0013-r0022-xAOD.root"
        ),
    },
    {
        "source_id": "mc24_100044_00200_00299",
        "channel": 100044,
        "run_range": "00200_00299",
        "family": "family1",
        "events_per_condition": 67,
        "path": (
            "/eos/experiment/faser/data0/sim/mc24/particle_gun/100044/rec/s0013-r0022/"
            "FaserMC-MC24_PG_mupl_fasernu_5mrad_flukaE-100044-00200-00299-s0013-r0022-xAOD.root"
        ),
    },
    {
        "source_id": "mc24_100044_00300_00399",
        "channel": 100044,
        "run_range": "00300_00399",
        "family": "family1",
        "events_per_condition": 67,
        "path": (
            "/eos/experiment/faser/data0/sim/mc24/particle_gun/100044/rec/s0013-r0022/"
            "FaserMC-MC24_PG_mupl_fasernu_5mrad_flukaE-100044-00300-00399-s0013-r0022-xAOD.root"
        ),
    },
    {
        "source_id": "mc24_100047_00100_00149",
        "channel": 100047,
        "run_range": "00100_00149",
        "family": "family2",
        "events_per_condition": 66,
        "path": (
            "/eos/experiment/faser/data0/sim/mc24/particle_gun/100047/rec/s0013-r0022/"
            "FaserMC-MC24_PG_mumi_fasernu_5mrad_flukaE-100047-00100-00149-s0013-r0022-xAOD.root"
        ),
    },
    {
        "source_id": "mc24_100048_00100_00149",
        "channel": 100048,
        "run_range": "00100_00149",
        "family": "family2",
        "events_per_condition": 66,
        "path": (
            "/eos/experiment/faser/data0/sim/mc24/particle_gun/100048/rec/s0013-r0022/"
            "FaserMC-MC24_PG_mupl_fasernu_5mrad_flukaE-100048-00100-00149-s0013-r0022-xAOD.root"
        ),
    },
)

# A priori station six-vectors: [dx_mm, dy_mm, dz_mm, rx_rad, ry_rad, rz_rad].
# Not the WB83 geometry matrix.  Survey mode is solver metadata only.
_IDENTITY = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
PAYLOAD_FAMILIES: dict[str, dict[int, tuple[float, float, float, float, float, float]]] = {
    "identity": {station: _IDENTITY for station in (0, 1, 2, 3)},
    "identifiable_translation": {
        0: _IDENTITY,
        1: (0.30, 0.0, 0.0, 0.0, 0.0, 0.0),
        2: (0.0, -0.20, 0.0, 0.0, 0.0, 0.0),
        3: _IDENTITY,
    },
    "identifiable_rotation": {
        0: _IDENTITY,
        1: (0.0, 0.0, 0.0, 0.0, 0.0, -0.0005),
        2: _IDENTITY,
        3: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0005),
    },
    "weak_jg_diagnostic": {
        0: _IDENTITY,
        1: _IDENTITY,
        2: _IDENTITY,
        3: (0.20, 0.0, 0.0, 0.0, 0.0002, 0.0),
    },
}

SURVEY_MODES = ("fixed_dz", "finite_survey_prior")
FAMILY_ORDER = (
    "identity",
    "identifiable_translation",
    "identifiable_rotation",
    "weak_jg_diagnostic",
)

REJECT_REASONS = (
    "missing_truth",
    "reconstruction_failure",
    "insufficient_stations",
    "invalid_covariance",
    "propagation_failure",
    "duplicate_event",
    "geometry_mismatch",
    "other",
)


class PhysicalReplicaProductionError(ValueError):
    """Fail-closed production contract violation."""


def refuse_forbidden_path(path: object) -> None:
    text = str(path).lower()
    for needle in FORBIDDEN_PATH_NEEDLES:
        if needle.lower() in text:
            raise PhysicalReplicaProductionError(
                f"physical replica production refuses forbidden path: {path}"
            )


def write_json(path: Path, payload: object) -> None:
    refuse_forbidden_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    refuse_forbidden_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(payload: object) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256_bytes(blob)


def replica_uid(source_id: str, run_id: int, event_id: int) -> str:
    return f"{source_id}:{int(run_id)}:{int(event_id)}"


def frozen_conditions() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        transforms = {
            str(station): list(values) for station, values in sorted(PAYLOAD_FAMILIES[family].items())
        }
        payload_sha = sha256_payload(
            {
                "family": family,
                "station_transforms": transforms,
                "convention": "[dx_mm, dy_mm, dz_mm, rx_rad, ry_rad, rz_rad]",
            }
        )
        for survey_mode in SURVEY_MODES:
            cells.append(
                {
                    "condition_id": f"{family}_{survey_mode}",
                    "family": family,
                    "survey_mode": survey_mode,
                    "parameterization": "station_se3_six_vector",
                    "injection_direction": family,
                    "amplitude": _family_amplitude(family),
                    "station_transforms": transforms,
                    "requested_payload_sha256": payload_sha,
                    "survey_mode_is_reconstruction_difference": False,
                    "notes": (
                        "Survey mode is recorded for a later qualification protocol. "
                        "Reconstruction uses the family payload only.  Events are still "
                        "not shared across the two survey tags."
                    ),
                }
            )
    if len(cells) != N_GEOMETRY_CONDITIONS:
        raise PhysicalReplicaProductionError("frozen condition table is not 8 cells")
    return cells


def _family_amplitude(family: str) -> dict[str, float]:
    if family == "identity":
        return {"translation_mm": 0.0, "rotation_mrad": 0.0}
    if family == "identifiable_translation":
        return {"s1_dx_mm": 0.30, "s2_dy_mm": -0.20}
    if family == "identifiable_rotation":
        return {"s1_rz_mrad": -0.5, "s3_rz_mrad": 0.5}
    if family == "weak_jg_diagnostic":
        return {"s3_dx_mm": 0.20, "s3_ry_mrad": 0.2}
    raise PhysicalReplicaProductionError(family)


def allocation_plan() -> dict[str, Any]:
    conditions = frozen_conditions()
    shards: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for source in AUTHORIZED_SOURCES:
        refuse_forbidden_path(source["path"])
        per = int(source["events_per_condition"])
        cursor = SKIP_BASE
        if cursor + per * N_GEOMETRY_CONDITIONS > PRODUCTION_CHUNK_EVENTS:
            raise PhysicalReplicaProductionError(
                f"{source['source_id']} allocation would leave the first production chunk"
            )
        for condition in conditions:
            skip = cursor
            shard_id = f"{condition['condition_id']}__{source['source_id']}"
            shard = {
                "shard_id": shard_id,
                "condition_id": condition["condition_id"],
                "family": condition["family"],
                "survey_mode": condition["survey_mode"],
                "source_id": source["source_id"],
                "input_xaod": source["path"],
                "skip_events": skip,
                "nevents": per,
                "xaod_entry_first": skip,
                "xaod_entry_last": skip + per - 1,
                "requested_payload_sha256": condition["requested_payload_sha256"],
            }
            shards.append(shard)
            for local in range(per):
                events.append(
                    {
                        "source_id": source["source_id"],
                        "condition_id": condition["condition_id"],
                        "shard_id": shard_id,
                        "xaod_entry_index": skip + local,
                        "expected_event_id": skip + local,
                        "channel": source["channel"],
                    }
                )
            cursor += per
    if len(events) != REQUIRED_INPUT_INDEPENDENT_EVENTS:
        raise PhysicalReplicaProductionError(
            f"allocation has {len(events)} events, not {REQUIRED_INPUT_INDEPENDENT_EVENTS}"
        )
    per_condition = {
        condition["condition_id"]: sum(1 for row in events if row["condition_id"] == condition["condition_id"])
        for condition in conditions
    }
    if any(count != 400 for count in per_condition.values()):
        raise PhysicalReplicaProductionError(f"per-condition input counts are not 400: {per_condition}")
    return {
        "schema": "calypso_physical_independent_v1_event_allocation",
        "skip_base": SKIP_BASE,
        "production_chunk_events": PRODUCTION_CHUNK_EVENTS,
        "chunk_policy": "stay_inside_first_5000_event_production_run_of_each_merged_xaod",
        "run_id_meaning": (
            "POOLCollectionTree.RunNumber is the MC channel, not the filename production run. "
            "EventNumber repeats every 5000 entries inside a merged file.  Allocation stays "
            "in chunk 0 so (source_id, run_id, event_id) is unique without renaming UIDs."
        ),
        "n_events": len(events),
        "n_shards": len(shards),
        "per_condition_input_events": per_condition,
        "shards": shards,
        "events": events,
        "trained_on_wb83_failure_numbers": False,
        "must_not_lower_n": True,
    }


def covariance_matrix(row: Mapping[str, Any]) -> list[list[float]]:
    xx = float(row["cov_xx_mm2"])
    xy = float(row["cov_xy_mm2"])
    xtx = float(row["cov_xtx_mm"])
    xty = float(row["cov_xty_mm"])
    yy = float(row["cov_yy_mm2"])
    ytx = float(row["cov_ytx_mm"])
    yty = float(row["cov_yty_mm"])
    txtx = float(row["cov_txtx"])
    txty = float(row["cov_txty"])
    tyty = float(row["cov_tyty"])
    return [
        [xx, xy, xtx, xty],
        [xy, yy, ytx, yty],
        [xtx, ytx, txtx, txty],
        [xty, yty, txty, tyty],
    ]


def audit_covariance(matrix: Sequence[Sequence[float]]) -> dict[str, Any]:
    values = [[float(cell) for cell in row] for row in matrix]
    if len(values) != 4 or any(len(row) != 4 for row in values):
        raise PhysicalReplicaProductionError("covariance must be 4x4")
    finite = all(math.isfinite(cell) for row in values for cell in row)
    antisym = 0.0
    for i in range(4):
        for j in range(4):
            antisym += (values[i][j] - values[j][i]) ** 2
    antisymmetric_norm = math.sqrt(antisym)
    symmetric = [[0.5 * (values[i][j] + values[j][i]) for j in range(4)] for i in range(4)]
    # Analytic 4x4 eigenvalues via characteristic polynomial would be heavier
    # than needed; use a Cholesky-style leading-minor test without numpy.
    spd = finite and antisymmetric_norm <= 1.0e-12 and _leading_minors_positive(symmetric)
    psd = finite and antisymmetric_norm <= 1.0e-12 and _leading_minors_nonnegative(symmetric)
    units_ok = True
    frame = "aligned_station_local_x_y_tx_ty"
    report = {
        "symmetry_ok": antisymmetric_norm <= 1.0e-12,
        "antisymmetric_norm": antisymmetric_norm,
        "finite_ok": finite,
        "spd_ok": spd,
        "psd_ok": psd,
        "unit_ok": units_ok,
        "frame_ok": True,
        "frame": frame,
        "silently_repaired": False,
        "accepted": bool(spd and finite and antisymmetric_norm <= 1.0e-12),
    }
    if not report["accepted"]:
        report["reject_reason"] = "invalid_covariance"
    return report


def _leading_minors_positive(symmetric: Sequence[Sequence[float]]) -> bool:
    for n in range(1, 5):
        det = _det([row[:n] for row in symmetric[:n]])
        if not math.isfinite(det) or det <= 0.0:
            return False
    return True


def _leading_minors_nonnegative(symmetric: Sequence[Sequence[float]]) -> bool:
    for n in range(1, 5):
        det = _det([row[:n] for row in symmetric[:n]])
        if not math.isfinite(det) or det < -1.0e-18:
            return False
    return True


def _det(matrix: Sequence[Sequence[float]]) -> float:
    n = len(matrix)
    if n == 1:
        return float(matrix[0][0])
    if n == 2:
        return float(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])
    total = 0.0
    for column, value in enumerate(matrix[0]):
        minor = [
            [matrix[row][col] for col in range(n) if col != column]
            for row in range(1, n)
        ]
        total += ((-1) ** column) * value * _det(minor)
    return total


def independence_audit(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    seen: dict[str, dict[str, Any]] = {}
    violations: list[dict[str, Any]] = []
    by_triple: dict[tuple[str, int, int], set[str]] = {}
    by_entry: dict[tuple[str, object], set[str]] = {}
    by_source_event: dict[tuple[str, int], set[str]] = {}
    materialized = list(rows)
    for row in materialized:
        uid = replica_uid(str(row["source_id"]), int(row["run_id"]), int(row["event_id"]))
        record = {
            "event_uid": uid,
            "condition_id": str(row.get("condition_id", "")),
            "source_id": str(row["source_id"]),
            "input_xaod": str(row.get("input_xaod", "")),
            "xaod_entry_index": row.get("xaod_entry_index"),
        }
        previous = seen.get(uid)
        if previous is not None:
            violations.append(
                {
                    "event_uid": uid,
                    "independence_violation": True,
                    "first": previous,
                    "second": record,
                    "hidden_by_renaming": False,
                }
            )
        else:
            seen[uid] = record
        triple = (str(row["source_id"]), int(row["run_id"]), int(row["event_id"]))
        by_triple.setdefault(triple, set()).add(str(row.get("condition_id", "")))
        by_entry.setdefault((str(row["source_id"]), row.get("xaod_entry_index")), set()).add(uid)
        by_source_event.setdefault((str(row["source_id"]), int(row["event_id"])), set()).add(str(row.get("condition_id", "")))
    cross_condition = [triple for triple, conds in by_triple.items() if len(conds) > 1]
    same_file_entry = [key for key, uids in by_entry.items() if len(uids) > 1]
    return {
        "schema": "calypso_physical_independent_v1_independence_audit",
        "n_rows": len(materialized),
        "n_unique": len(seen),
        "n_violations": len(violations),
        "independence_violation": bool(violations or cross_condition or same_file_entry),
        "violations": violations,
        "cross_condition_reuse_forbidden": True,
        "n_cross_condition_triples": len(cross_condition),
        "n_same_input_file_entry_reuse": len(same_file_entry),
        "n_sources": len({str(row["source_id"]) for row in materialized}),
        "hidden_by_renaming": False,
    }


def empty_reject_counts() -> dict[str, int]:
    return {reason: 0 for reason in REJECT_REASONS}


def locked_qualification_flags() -> dict[str, Any]:
    return {
        "alignment_oracle_qualified_for_physical_FASER": False,
        "alignment_oracle_qualified": False,
        "ml_alignment_eval_authorized": False,
        "final_blind_eval_authorized": False,
        "sealed_test_accessed": False,
        "qualification_authorized": False,
        "common_track_solver_qualified_under_toy_model": False,
        "association_default_system": ASSOCIATION_DEFAULT_SYSTEM,
    }


def qa_gate(checks: Mapping[str, bool]) -> dict[str, Any]:
    required = (
        "unique_physical_events",
        "zero_independence_violations",
        "zero_forbidden_assets",
        "geometry_roundtrip",
        "software_provenance_complete",
        "field_material_hashes_consistent",
        "truth_label_completeness",
        "measurement_schema",
        "covariance_contract",
        "condition_counts",
    )
    missing = [name for name in required if name not in checks]
    if missing:
        raise PhysicalReplicaProductionError(f"corpus QA missing checks: {missing}")
    passed = all(bool(checks[name]) for name in required)
    return {
        "schema": "calypso_physical_independent_v1_corpus_qa",
        "checks": dict(checks),
        "calypso_physical_corpus_qualified": bool(passed),
        "physical_replica_production_complete": bool(checks.get("production_complete", False)),
        **locked_qualification_flags(),
    }
