"""Workbook-78 helpers for a read-only run-range / data-contract audit.

These functions do not train, do not open Final Blind / sealed test, and do
not change the W64 / V5A / WB75 checkpoints.  They exist so the audit script
and its unit tests share one fail-closed comparison contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from alignment.four_station import (
    GAUGE_REFERENCE_STATION,
    IDENTITY_SIX,
    draw_common_left_se3,
    draw_relative_native,
    relative_family_payloads,
    relative_free_parameter_names,
)
from training.source_diversity_audit import UNUSED_RESERVE_SOURCES, is_sealed_source


QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
HARD_S3_RY_NATIVE = {
    "s1_dx_mm": 0.30,
    "s2_dy_mm": -0.20,
    "s3_ry_mrad": 5.0,
}
RELATIVE_TRANSLATION_MM = 0.50
RELATIVE_ROTATION_MRAD = 5.0
COMMON_TRANSLATION_MM = 0.30
COMMON_ROTATION_MRAD = 3.0
FORBIDDEN_SOURCE_PREFIXES = tuple(UNUSED_RESERVE_SOURCES) + ("mc24_100116_", "mc24_100117_")
KS_LARGE = 0.20
KS_SMALL = 0.05
OCCUPANCY_LARGE = 0.10


def feature_origin(name: str) -> dict[str, str]:
    """Map a reported feature name back to the data-chain layer that produces it."""
    text = str(name)
    if text.startswith("L") and text[1:].isdigit():
        return {
            "name": text,
            "layer": "route_head",
            "code": "head",
            "note": "frozen W64 adjacent logit; not a raw physical coordinate",
        }
    if "delta_z" in text or text.endswith("_z_mm"):
        return {
            "name": text,
            "layer": "A_geometry_station_plane",
            "code": "A",
            "note": "z_target - z_source or station-plane z from the payload / geometry",
        }
    if any(token in text for token in ("residual_", "pull_", "log1p_chi2", "combined_covariance_logdet")):
        return {
            "name": text,
            "layer": "C_propagation",
            "code": "C",
            "note": "Acts residual / pull / chi2 / covariance of a field candidate",
        }
    if any(token in text for token in ("x_mm", "y_mm", "_tx", "_ty", "chi2", "n_hit", "log_sigma")):
        return {
            "name": text,
            "layer": "B_reconstruction_or_event_content",
            "code": "B",
            "note": "tracklet state or local fit quality after SegmentFitRefit",
        }
    return {
        "name": text,
        "layer": "unassigned",
        "code": "?",
        "note": "no pre-registered origin rule",
    }


def decide_shift_sources(
    *,
    matched_payload_max_l2: float | None,
    draw_00_min_l2: float | None,
    same_software_tag: bool,
    station_z_identical: bool | None,
    identity_delta_z_max_ks: float | None,
    identity_residual_max_ks: float | None,
    identity_state_max_ks: float | None,
    overlay_recipe_identical: bool,
    occupancy_rate_max_abs_diff: float | None,
) -> dict[str, Any]:
    """Pre-registered A–E decision.  Thresholds are not tuned on the audit run."""
    matched_ok = matched_payload_max_l2 is not None and float(matched_payload_max_l2) <= 1.0e-9
    draw_differs = draw_00_min_l2 is not None and float(draw_00_min_l2) > 0.1
    residual_shift = identity_residual_max_ks is not None and float(identity_residual_max_ks) >= KS_LARGE
    state_shift = identity_state_max_ks is not None and float(identity_state_max_ks) >= KS_LARGE
    delta_z_quiet = identity_delta_z_max_ks is not None and float(identity_delta_z_max_ks) < KS_SMALL
    occupancy_shift = (
        occupancy_rate_max_abs_diff is not None and float(occupancy_rate_max_abs_diff) >= OCCUPANCY_LARGE
    )
    return {
        "thresholds": {
            "ks_large": KS_LARGE,
            "ks_small": KS_SMALL,
            "occupancy_large": OCCUPANCY_LARGE,
            "matched_payload_l2": 1.0e-9,
            "draw_payload_l2": 0.1,
        },
        "A_geometry_payload": {
            "supported": bool(draw_differs and matched_ok),
            "role": "confound_for_name_matched_nonidentical_payloads",
            "matched_payload_max_l2": matched_payload_max_l2,
            "draw_00_min_l2": draw_00_min_l2,
            "identity_delta_z_quiet": delta_z_quiet,
            "station_z_identical": station_z_identical,
            "note": (
                "Name-matched draw_* payloads are different random draws when the "
                "curriculum seeds differ.  They are not a run-range test.  "
                "iteration_00_reference / hard_s3_ry are the matched-geometry tests."
            ),
        },
        "B_reconstruction_policy": {
            "supported": bool(same_software_tag and state_shift),
            "same_software_tag": same_software_tag,
            "identity_state_max_ks": identity_state_max_ks,
            "note": (
                "A shared s0013-r0022 tag means the reconstruction policy matches. "
                "A large identity-payload state KS is therefore event content, not a rec-tag change."
            ),
        },
        "C_propagation": {
            "supported": bool(residual_shift),
            "identity_residual_max_ks": identity_residual_max_ks,
            "identity_delta_z_max_ks": identity_delta_z_max_ks,
            "station_z_identical": station_z_identical,
            "note": (
                "If station z is identical, raw delta_z cannot be a run-range plane shift. "
                "A large identity residual KS is then propagation given different tracklets, "
                "or a true field/material change."
            ),
        },
        "D_overlay": {
            "supported": bool(not overlay_recipe_identical),
            "recipe_fields_match": overlay_recipe_identical,
            "note": (
                "The overlay recipe (q_over_p, repropagation, synthetic_multitrack) is "
                "checked here.  A matching recipe cannot be the origin of a physical-file shift."
            ),
        },
        "E_selection_filtering": {
            "supported": bool(occupancy_shift),
            "occupancy_rate_max_abs_diff": occupancy_rate_max_abs_diff,
            "note": (
                "Occupancy / accepted-event rate differences are selection. "
                "UID-count differences that follow from a two-source vs six-source pool are not."
            ),
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def refuse_forbidden_source_id(source_id: str) -> None:
    text = str(source_id)
    if is_sealed_source(text) or any(text.startswith(prefix) for prefix in FORBIDDEN_SOURCE_PREFIXES):
        raise ValueError(f"refusing sealed or final-blind source {source_id!r}")
    if "00800_00849" in text:
        raise ValueError(f"refusing unused final-blind source {source_id!r}")


def refuse_forbidden_path(path: str | Path) -> None:
    text = str(path)
    if "00800_00849" in text or "mc24_100116" in text or "mc24_100117" in text:
        raise ValueError(f"refusing forbidden asset path {text}")


def finite_vector(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def support_overlap(train: np.ndarray, development: np.ndarray) -> float | None:
    """Fraction of development samples inside the train 5–95% interval."""
    left = finite_vector(train)
    right = finite_vector(development)
    if left.size < 2 or right.size == 0:
        return None
    lo, hi = np.quantile(left, [0.05, 0.95])
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)):
        return None
    if float(hi) <= float(lo) + 1.0e-15:
        return float(np.mean(np.isclose(right, lo, rtol=0.0, atol=1.0e-9)))
    return float(np.mean((right >= lo) & (right <= hi)))


def compare_univariate(train: Sequence[float], development: Sequence[float], name: str) -> dict[str, Any]:
    """KS / Wasserstein / mean-std / quantile / overlap for one feature."""
    left = finite_vector(train)
    right = finite_vector(development)
    payload: dict[str, Any] = {
        "name": name,
        "n_train": int(left.size),
        "n_development": int(right.size),
    }
    if left.size == 0 or right.size == 0:
        payload["status"] = "empty"
        return payload
    ks_stat, ks_p = ks_2samp(left, right)
    train_std = float(left.std())
    payload.update(
        {
            "status": "ok",
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_p),
            "wasserstein": float(wasserstein_distance(left, right)),
            "train_mean": float(left.mean()),
            "development_mean": float(right.mean()),
            "train_std": train_std,
            "development_std": float(right.std()),
            "mean_shift": float(right.mean() - left.mean()),
            "mean_shift_over_train_std": float((right.mean() - left.mean()) / max(train_std, 1.0e-12)),
            "train_quantiles": {f"q{int(q * 100):02d}": float(np.quantile(left, q)) for q in QUANTILES},
            "development_quantiles": {f"q{int(q * 100):02d}": float(np.quantile(right, q)) for q in QUANTILES},
            "development_in_train_q05_q95": support_overlap(left, right),
        }
    )
    return payload


def payload_parameter_l2(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = sorted(set(left) | set(right))
    return float(math.sqrt(sum((float(left.get(key, 0.0)) - float(right.get(key, 0.0))) ** 2 for key in keys)))


def parse_xaod_identity(path: str) -> dict[str, str | None]:
    """Extract DSID / run-range / rec-tag from a FASER MC xAOD filename."""
    refuse_forbidden_path(path)
    name = Path(path).name
    rec_tag = None
    for token in name.replace(".root", "").split("-"):
        if token.startswith("s") and "-r" not in token and token[1:].isdigit():
            rec_tag = token
        if token.startswith("s") and "r" in token:
            rec_tag = token
    # Typical: FaserMC-MC24_PG_...-100047-00100-00149-s0013-r0022-xAOD.root
    parts = name.replace(".root", "").split("-")
    dsid = None
    run_lo = None
    run_hi = None
    software = None
    for index, part in enumerate(parts):
        if part.isdigit() and len(part) == 6 and index + 2 < len(parts):
            dsid = part
            run_lo = parts[index + 1]
            run_hi = parts[index + 2]
        if part.startswith("s") and part[1:].isdigit() and index + 1 < len(parts) and parts[index + 1].startswith("r"):
            software = f"{part}-{parts[index + 1]}"
    return {
        "path": path,
        "filename": name,
        "dsid": dsid,
        "run_lo": run_lo,
        "run_hi": run_hi,
        "software_tag": software,
        "rec_tag": rec_tag or software,
    }


def generate_geometry_payload_table(
    *,
    seed: int,
    n_random_families: int = 2,
    include_reference: bool = True,
    include_hard_s3_ry: bool = False,
    table_role: str,
) -> dict[str, Any]:
    """Sample a payload table.  Does not refit, overlay, or open ROOT."""
    if int(n_random_families) < 1:
        raise ValueError("n_random_families must be at least 1")
    rng = np.random.default_rng(int(seed))
    families: list[tuple[str, dict[str, float]]] = []
    if include_hard_s3_ry:
        names = relative_free_parameter_names(gauge=GAUGE_REFERENCE_STATION, reference_station=0)
        hard = {name: 0.0 for name in names}
        hard.update(HARD_S3_RY_NATIVE)
        families.append(("hard_s3_ry", hard))
    for index in range(int(n_random_families)):
        families.append(
            (
                f"draw_{index:02d}",
                draw_relative_native(
                    rng,
                    translation_mm=RELATIVE_TRANSLATION_MM,
                    rotation_mrad=RELATIVE_ROTATION_MRAD,
                ),
            )
        )
    points: list[dict[str, Any]] = []
    if include_reference:
        points.append(
            {
                "name": "iteration_00_reference",
                "role": "nominal",
                "relative_family": "nominal",
                "gauge_role": "identity",
                "relative_native_values": {
                    name: 0.0
                    for name in relative_free_parameter_names(
                        gauge=GAUGE_REFERENCE_STATION, reference_station=0
                    )
                },
                "common_left_se3": list(IDENTITY_SIX),
                "station_transforms": {str(station): list(IDENTITY_SIX) for station in range(4)},
                "holdout": False,
            }
        )
    for family, native in families:
        common = draw_common_left_se3(
            rng,
            translation_mm=COMMON_TRANSLATION_MM,
            rotation_mrad=COMMON_ROTATION_MRAD,
        )
        relative, control = relative_family_payloads(native, common)
        points.append(
            {
                "name": f"iteration_00_{family}",
                "role": "s0_sampling_chart",
                "relative_family": family,
                "gauge_role": "s0_sampling_chart",
                "relative_native_values": {key: float(value) for key, value in native.items()},
                "common_left_se3": [0.0] * 6,
                "station_transforms": {key: [float(v) for v in values] for key, values in relative.items()},
                "holdout": table_role == "held_out_geometry",
            }
        )
        points.append(
            {
                "name": f"iteration_00_{family}_plus_common",
                "role": "left_se3_control",
                "relative_family": family,
                "gauge_role": "left_se3_control",
                "relative_native_values": {key: float(value) for key, value in native.items()},
                "common_left_se3": [float(value) for value in common],
                "station_transforms": {key: [float(v) for v in values] for key, values in control.items()},
                "holdout": table_role == "held_out_geometry",
            }
        )
    return {
        "table_role": table_role,
        "seed": int(seed),
        "n_random_families": int(n_random_families),
        "include_reference": bool(include_reference),
        "include_hard_s3_ry": bool(include_hard_s3_ry),
        "relative_bounds": {
            "translation_mm": RELATIVE_TRANSLATION_MM,
            "rotation_mrad": RELATIVE_ROTATION_MRAD,
        },
        "common_bounds": {
            "translation_mm": COMMON_TRANSLATION_MM,
            "rotation_mrad": COMMON_ROTATION_MRAD,
        },
        "n_points": len(points),
        "points": points,
        "final_blind_authorized": False,
        "sealed_test_accessed": False,
        "training_authorized": False,
    }


def _family_signature(point: Mapping[str, Any]) -> tuple[object, ...]:
    native = point.get("relative_native_values") or {}
    common = point.get("common_left_se3") or []
    return (
        str(point.get("relative_family")),
        str(point.get("gauge_role")),
        tuple(sorted((str(key), float(value)) for key, value in dict(native).items())),
        tuple(float(value) for value in common),
    )


def assert_geometry_tables_disjoint(train: Mapping[str, Any], heldout: Mapping[str, Any]) -> dict[str, Any]:
    """Random-family geometries must not collide; identity may be shared."""
    if int(train["seed"]) == int(heldout["seed"]):
        raise ValueError("train and held-out geometry tables must use different seeds")
    train_sigs = {
        _family_signature(point)
        for point in train["points"]
        if str(point.get("relative_family")) not in {"nominal", "hard_s3_ry"}
    }
    held_sigs = {
        _family_signature(point)
        for point in heldout["points"]
        if str(point.get("relative_family")) not in {"nominal", "hard_s3_ry"}
    }
    overlap = sorted(train_sigs.intersection(held_sigs))
    if overlap:
        raise ValueError(f"geometry holdout collides with train table: {overlap}")
    return {
        "disjoint": True,
        "train_seed": int(train["seed"]),
        "heldout_seed": int(heldout["seed"]),
        "n_train_random_points": int(len(train_sigs)),
        "n_heldout_random_points": int(len(held_sigs)),
        "shared_nominal_allowed": True,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
