"""Workbook-82: Wide-ty Real-Support-Matched MC Coverage Feasibility V1.

This campaign is **residual-blind**.  It answers ONE question:

    Does any existing non-sealed MC/control sample provide enough track
    kinematic coverage to cover the WB80/WB81 real calibration population's
    (0,1)/(0,2) ``(pred_tx, pred_ty)`` applicability domain, so that a future
    conditional-J model could be trained/validated WITHOUT extrapolating to
    real tracks?

This is NOT alignment.  It reads NO alignment residual, NO Jacobian/FD
derivative, NO singular value/rank, and NO final-correction performance.
Candidate admission uses ONLY physics metadata and kinematics (generator
metadata, particle species, momentum/q-over-p, origin, tx/ty, position,
station/pair coverage, truth-match availability, file/source provenance).

Frozen permissions for the whole campaign:
    held_out_accessed = false
    real_data_alignment_authorized = false
    geometry_write_allowed = false
    official_conditions_write_allowed = false
    external_constraint_ingest_authorized = false

Kinematic frame
---------------
The primary kinematic is the **source-tracklet local slope ``(tx, ty)``** at
the source station (station 0), because it is available uniformly for the real
calibration pairs and for EVERY MC candidate -- including the floor-origin muon
sample (100120) and the K-short export, which ship ``tracklets.root`` but no
``propagations.root``.  The WB80 applicability domain was defined in the
propagation frame ``(pred_tx, pred_ty)``; for the high-momentum muon-like
populations at issue the propagated slope is a deterministic, near-identity
map of the source slope (per-event field bending << the width of the support
gap), so the source-slope frame is the same physical support statement.  The
propagation-frame ``(pred_tx, pred_ty)`` coverage is ALSO computed for every
candidate that has propagations, as a WB80-consistency cross-check.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events
from alignment import gauge_fixed_real_data_diagnostic as wb78
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)

SCHEMA_VERSION = "faser-wide-ty-real-support-matched-mc-coverage-feasibility-v1"
DEFAULT_CONFIG = "configs/wide_ty_real_support_matched_mc_coverage_feasibility_v1.yaml"

# Pair types under test.  (0,1) and (0,2) are gated; (0,3) has only 2 real
# pairs and remains report-only (inherited from WB80).
STATION_PAIRS = ((0, 1), (0, 2), (0, 3))
GATED_TARGET_STATIONS = (1, 2)
REPORT_ONLY_TARGET_STATIONS = (3,)

HELD_OUT_RUN_IDS = frozenset({14975, 14976})

# Flags that must be exactly False / True in the frozen config.
CONFIG_MUST_BE_FALSE = (
    "geometry_write_allowed",
    "official_conditions_write_allowed",
    "real_data_candidate_alignment_authorized",
    "external_constraint_ingest_authorized",
    "held_out_accessed",
)
CONFIG_MUST_BE_TRUE = (
    "do_not_open_held_out",
    "do_not_open_sealed_test",
    "do_not_read_alignment_residual",
    "do_not_use_jacobian_or_fd_derivative_for_admission",
    "do_not_select_candidate_by_coverage_result",
    "residual_blind",
)


# ---------------------------------------------------------------------------
# Config loading + freeze verification
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    for key in CONFIG_MUST_BE_FALSE:
        if config.get(key) is not False:
            raise ValueError(f"config must set {key}: false")
    for key in CONFIG_MUST_BE_TRUE:
        if config.get(key) is not True:
            raise ValueError(f"config must set {key}: true")

    inheritance = config["inheritance"]
    wb81_config_path = resolve_under_root(project_root(), str(inheritance["workbook_81_config"]))
    if sha256_file(wb81_config_path) != str(inheritance["workbook_81_config_sha256"]):
        raise ValueError("workbook-81 config SHA256 mismatch")
    wb81_root = resolve_under_root(project_root(), str(inheritance["workbook_81_output_root"]))
    for name, expected in inheritance["workbook_81_artifact_sha256"].items():
        actual = sha256_file(wb81_root / name)
        if actual != str(expected):
            raise ValueError(f"workbook-81 artifact SHA256 mismatch for {name}")

    # Candidate definitions must be complete and sealed sources excluded.
    sealed = {str(s) for s in config["sealed_test"]["forbidden_source_ids"]}
    for cand in config["candidates"]:
        for sid in cand["source_ids"]:
            if str(sid) in sealed:
                raise ValueError(f"candidate {cand['name']} contains sealed source {sid}")
    config["config_path"] = str(config_path)
    return config


def assert_no_held_out_access(run_ids: np.ndarray) -> None:
    seen = {int(v) for v in np.atleast_1d(run_ids)}
    bad = seen & HELD_OUT_RUN_IDS
    if bad:
        raise ValueError(f"held-out run access refused: {sorted(bad)}")


def load_wb80_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Load the frozen WB80 config via the WB81 inheritance chain (SHA-pinned).

    WB82 -> WB81 (verified in ``load_config``) -> WB80.  The WB80 config carries
    the ``wb78_config`` (real-data roots) needed to rebuild the calibration
    bank and read the residual-blind pair kinematics.
    """
    from alignment import real_data_measurement_model_validation as wb80

    wb81_config_path = resolve_under_root(
        project_root(), str(config["inheritance"]["workbook_81_config"])
    )
    wb81_cfg = yaml.safe_load(wb81_config_path.read_text(encoding="utf-8"))
    wb80_config_path = wb81_cfg["inheritance"]["workbook_80_config"]
    return wb80.load_config(wb80_config_path)


def load_real_target(config: dict[str, Any]) -> dict[str, np.ndarray]:
    """Rebuild the frozen WB80 calibration bank and read the residual-blind
    real support target kinematics (both source-slope and propagated frames)."""
    from alignment import real_data_measurement_model_validation as wb80

    wb80_config = load_wb80_config(config)
    baseline = wb80.reproduce_baseline(wb80_config)
    if not baseline["report"]["pass"]:
        raise ValueError("WB80 baseline reproduction failed; WB82 refused")
    config["wb78_config"] = wb80_config["wb78_config"]
    return load_real_target_kinematics(config, baseline["bank"])


# ---------------------------------------------------------------------------
# Real support target (residual-blind; inherits the frozen WB80 calibration
# population 14973+14974 and ONLY reads residual-independent track/pair
# metadata: source-tracklet slope, propagated slope, lever arm, station ids).
# ---------------------------------------------------------------------------


def load_real_target_kinematics(
    config: Mapping[str, Any], bank: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Return the real calibration (0,1)/(0,2)/(0,3) pair kinematics.

    Reads the frozen identity files (``synthetic_tracklets.root`` +
    ``field_candidates.root``) for the calibration sources and extracts, per
    frozen bank pair, the source-tracklet local slope ``(tx, ty)``, the
    propagated slope ``(pred_tx, pred_ty)``, the lever arm and the target
    station.  NO residual value is read.  The pair set is exactly the frozen
    WB80 calibration bank (route-selected), so the target domain is identical
    to the WB80 applicability domain.
    """
    wb78_config = config["wb78_config"]
    # Per-calibration-source lookup: pair identity -> kinematic record.
    records_by_key: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for source in wb78_config["real_data_population"]["sources"]:
        if str(source["role"]) != "calibration":
            continue
        roots = wb78._source_roots(wb78_config, source)
        events = load_events(roots["identity"] / "synthetic_tracklets.root", require_mc_labels=False)
        records = load_propagation_records(roots["identity"] / "field_candidates.root")
        tracklet_index: dict[tuple[int, int, int], tuple[Any, int]] = {}
        for event in events:
            for row, tid in enumerate(event.tracklet_id):
                tracklet_index[(event.run_id, event.event_id, int(tid))] = (event, row)
        for row in range(records.size):
            if int(records.q_over_p_mode[row]) != 0:
                continue
            if not records.success[row] or not records.has_covariance[row]:
                continue
            skey = (int(records.run_id[row]), int(records.event_id[row]), int(records.source_tracklet_id[row]))
            tkey = (int(records.run_id[row]), int(records.event_id[row]), int(records.target_tracklet_id[row]))
            if skey not in tracklet_index or tkey not in tracklet_index:
                continue
            sevent, srow = tracklet_index[skey]
            pair_key = (
                int(records.run_id[row]),
                int(records.event_id[row]),
                int(records.source_tracklet_id[row]),
                int(records.target_tracklet_id[row]),
            )
            records_by_key[pair_key] = {
                "source_tx": float(sevent.state[srow, 2]),
                "source_ty": float(sevent.state[srow, 3]),
                "pred_tx": float(records.prediction[row, 2]),
                "pred_ty": float(records.prediction[row, 3]),
                "lever_arm_mm": float(records.target_z_mm[row] - sevent.z_mm[srow]),
            }

    arrays = wb78.concatenate_banks(bank)
    run_id = np.asarray(arrays["run_id_arr"], dtype=np.int64)
    assert_no_held_out_access(run_id)
    n = run_id.shape[0]
    source_tx = np.full(n, np.nan)
    source_ty = np.full(n, np.nan)
    pred_tx = np.full(n, np.nan)
    pred_ty = np.full(n, np.nan)
    lever = np.full(n, np.nan)
    for i in range(n):
        key = (
            int(arrays["run_id_arr"][i]),
            int(arrays["event_id"][i]),
            int(arrays["source_tracklet_id"][i]),
            int(arrays["target_tracklet_id"][i]),
        )
        rec = records_by_key.get(key)
        if rec is None:
            raise ValueError(f"bank pair missing kinematic record: {key}")
        source_tx[i] = rec["source_tx"]
        source_ty[i] = rec["source_ty"]
        pred_tx[i] = rec["pred_tx"]
        pred_ty[i] = rec["pred_ty"]
        lever[i] = rec["lever_arm_mm"]
    return {
        "source_tx": source_tx,
        "source_ty": source_ty,
        "pred_tx": pred_tx,
        "pred_ty": pred_ty,
        "lever_arm_mm": lever,
        "target_station_id": np.asarray(arrays["target_station_id"], dtype=np.int64),
        "run_id": run_id,
        "residual_blind": True,
    }


# ---------------------------------------------------------------------------
# Candidate MC cloud loading (kinematics + physics metadata only).
# ---------------------------------------------------------------------------


def _refit_dir_for_source(root_template: str, source_id: str) -> Path | None:
    """Resolve the nominal-field refit directory for a curriculum source.

    ``root_template`` contains ``{source}`` and may contain a ``*`` glob for the
    point directory (e.g. ``mag_0_*_00``).  Returns the first match, or None.
    """
    import glob as _glob

    pattern = str(resolve_under_root(project_root(), root_template)).format(source=source_id)
    matches = sorted(p for p in _glob.glob(pattern) if Path(p).is_dir())
    return Path(matches[0]) if matches else None


def load_candidate_pair_cloud(
    config: Mapping[str, Any], candidate: Mapping[str, Any], target_station: int
) -> dict[str, Any]:
    """Load one candidate's source-tracklet ``(tx, ty)`` cloud for one pair type.

    Two loading kinds are supported (both residual-blind):

    * ``propagation``: read ``propagations.root`` (mode 0, success,
      source_station 0 -> ``target_station``) and join each record to its source
      tracklet for the local slope, truth-match fraction, truth pdg and truth
      q/p.  Also captures the propagated slope ``(pred_tx, pred_ty)`` and lever
      arm for the WB80-consistency cross-check.
    * ``tracklet_truth_matched``: for samples with no propagations (e.g. the
      100120 floor-muon export), form (0, target) pairs by matching a station-0
      tracklet to a station-``target`` tracklet sharing the same truth particle
      (``truth_particle_id``), then take the source tracklet slope.

    The truth-match filter ``min_truth_match_fraction`` (inherited from the
    WB80 FD-bank chain) is applied to the source tracklet.  Returns the cloud
    plus physics metadata for the particle-domain audit.
    """
    kind = str(candidate["kind"])
    min_tmf = float(config["coverage"]["min_truth_match_fraction"])
    tx: list[float] = []
    ty: list[float] = []
    pred_tx: list[float] = []
    pred_ty: list[float] = []
    tmf: list[float] = []
    pdg: list[int] = []
    qop: list[float] = []
    lever: list[float] = []
    src_z: list[float] = []
    n_files = 0
    n_sources_used = 0
    for source_id in candidate["source_ids"]:
        if kind == "propagation":
            import uproot

            refit = _refit_dir_for_source(str(candidate["refit_template"]), str(source_id))
            if refit is None:
                continue
            prop_path = refit / "propagations.root"
            trk_path = refit / "tracklets.root"
            if not prop_path.is_file() or not trk_path.is_file():
                continue
            recs = load_propagation_records(prop_path)
            # Read source tracklets directly.  Keyed by (run, event, station,
            # tracklet_id); some productions reuse tracklet ids across stations
            # or rows, so the first occurrence is kept (robust, deterministic).
            ta = uproot.open(trk_path)["tracklets"].arrays(
                [
                    "run_id",
                    "event_id",
                    "station_id",
                    "tracklet_id",
                    "tx",
                    "ty",
                    "z_mm",
                    "truth_match_fraction",
                    "truth_pdg",
                    "truth_q_over_p_per_mev",
                ],
                library="np",
            )
            tindex: dict[tuple[int, int, int, int], int] = {}
            for i in range(len(ta["run_id"])):
                key = (
                    int(ta["run_id"][i]),
                    int(ta["event_id"][i]),
                    int(ta["station_id"][i]),
                    int(ta["tracklet_id"][i]),
                )
                tindex.setdefault(key, i)
            sel = (
                (recs.q_over_p_mode == 0)
                & recs.success
                & (recs.source_station_id == 0)
                & (recs.target_station_id == target_station)
            )
            used = False
            for row in np.flatnonzero(sel):
                skey = (
                    int(recs.run_id[row]),
                    int(recs.event_id[row]),
                    0,
                    int(recs.source_tracklet_id[row]),
                )
                srow = tindex.get(skey)
                if srow is None:
                    continue
                sm = float(ta["truth_match_fraction"][srow])
                if not (sm >= min_tmf):
                    continue
                tx.append(float(ta["tx"][srow]))
                ty.append(float(ta["ty"][srow]))
                pred_tx.append(float(recs.prediction[row, 2]))
                pred_ty.append(float(recs.prediction[row, 3]))
                tmf.append(sm)
                pdg.append(int(ta["truth_pdg"][srow]))
                qop.append(float(ta["truth_q_over_p_per_mev"][srow]))
                lever.append(float(recs.target_z_mm[row] - ta["z_mm"][srow]))
                src_z.append(float(ta["z_mm"][srow]))
                used = True
            n_files += 1
            n_sources_used += int(used)
        elif kind == "tracklet_truth_matched":
            import uproot

            trk_path = resolve_under_root(
                project_root(), str(candidate["tracklets_template"]).format(source=source_id)
            )
            if not trk_path.is_file():
                continue
            # Read tracklets directly: the export uses per-station tracklet ids
            # (not unique within an event), so load_events cannot be used.
            ta = uproot.open(trk_path)["tracklets"].arrays(
                [
                    "run_id",
                    "event_id",
                    "station_id",
                    "tx",
                    "ty",
                    "z_mm",
                    "truth_particle_id",
                    "truth_match_fraction",
                    "truth_pdg",
                    "truth_q_over_p_per_mev",
                ],
                library="np",
            )
            # station-0 / station-target truth-matched tracklets keyed by
            # (run, event, truth particle).  Residual-blind: only kinematics.
            s0: dict[tuple[int, int, int], int] = {}
            strow: dict[tuple[int, int, int], int] = {}
            n_rows = len(ta["run_id"])
            for i in range(n_rows):
                if float(ta["truth_match_fraction"][i]) < min_tmf:
                    continue
                key = (
                    int(ta["run_id"][i]),
                    int(ta["event_id"][i]),
                    int(ta["truth_particle_id"][i]),
                )
                station = int(ta["station_id"][i])
                if station == 0:
                    s0.setdefault(key, i)
                elif station == target_station:
                    strow.setdefault(key, i)
            used = False
            for key, i in s0.items():
                j = strow.get(key)
                if j is None:
                    continue
                tx.append(float(ta["tx"][i]))
                ty.append(float(ta["ty"][i]))
                pred_tx.append(np.nan)  # no propagations available for this kind
                pred_ty.append(np.nan)
                tmf.append(float(ta["truth_match_fraction"][i]))
                pdg.append(int(ta["truth_pdg"][i]))
                qop.append(float(ta["truth_q_over_p_per_mev"][i]))
                lever.append(float(ta["z_mm"][j] - ta["z_mm"][i]))
                src_z.append(float(ta["z_mm"][i]))
                used = True
            n_files += 1
            n_sources_used += int(used)
        else:
            raise ValueError(f"unknown candidate kind {kind}")
    return {
        "candidate": str(candidate["name"]),
        "target_station": int(target_station),
        "kind": kind,
        "tx": np.asarray(tx, dtype=np.float64),
        "ty": np.asarray(ty, dtype=np.float64),
        "pred_tx": np.asarray(pred_tx, dtype=np.float64),
        "pred_ty": np.asarray(pred_ty, dtype=np.float64),
        "truth_match_fraction": np.asarray(tmf, dtype=np.float64),
        "truth_pdg": np.asarray(pdg, dtype=np.int64),
        "truth_q_over_p_per_mev": np.asarray(qop, dtype=np.float64),
        "lever_arm_mm": np.asarray(lever, dtype=np.float64),
        "source_z_mm": np.asarray(src_z, dtype=np.float64),
        "n_files": int(n_files),
        "n_sources_with_pairs": int(n_sources_used),
        "n_sources_declared": int(len(candidate["source_ids"])),
    }


# ---------------------------------------------------------------------------
# Coverage metrics (residual-blind).
# ---------------------------------------------------------------------------


def _mahalanobis_envelope_fraction(
    cloud: np.ndarray, real: np.ndarray, maha2_gate: float
) -> float | None:
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    real = real[np.isfinite(real).all(axis=1)]
    if cloud.shape[0] < 10 or real.shape[0] == 0:
        return None
    center = cloud.mean(axis=0)
    cov = np.cov(cloud, rowvar=False)
    cov += np.eye(2) * (1.0e-6 * float(np.mean(np.diagonal(cov))) + 1.0e-12)
    inv = np.linalg.inv(cov)
    diff = real - center
    maha2 = np.einsum("ni,ij,nj->n", diff, inv, diff)
    return float(np.mean(maha2 <= maha2_gate))


def _bin_occupancy_fraction(cloud: np.ndarray, real: np.ndarray, bin_width: float) -> float | None:
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    real = real[np.isfinite(real).all(axis=1)]
    if cloud.shape[0] < 1 or real.shape[0] == 0:
        return None
    occupied = {tuple(int(v) for v in np.floor(row / bin_width)) for row in cloud}
    covered = sum(
        1 for row in real if tuple(int(v) for v in np.floor(row / bin_width)) in occupied
    )
    return float(covered / real.shape[0])


def _quadrant_coverage(cloud: np.ndarray, real: np.ndarray) -> dict[str, Any]:
    """Fraction of the real-populated (sign(tx), sign(ty)) quadrants that the
    candidate populates, and the per-quadrant real coverage."""
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    real = real[np.isfinite(real).all(axis=1)]
    if cloud.shape[0] == 0 or real.shape[0] == 0:
        return {"real_quadrants_populated": 0, "quadrants_covered": 0, "fraction": None}
    def _quad(row: np.ndarray) -> tuple[int, int]:
        return (int(row[0] >= 0.0), int(row[1] >= 0.0))
    cloud_quads = {_quad(row) for row in cloud}
    real_quads = [_quad(row) for row in real]
    populated = sorted(set(real_quads))
    covered = sum(1 for q in populated if q in cloud_quads)
    return {
        "real_quadrants_populated": int(len(populated)),
        "quadrants_covered": int(covered),
        "fraction": float(covered / len(populated)) if populated else None,
    }


def compute_support_coverage(
    real_xy: np.ndarray, candidate_xy: np.ndarray, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Compute the residual-blind support-coverage metrics of a candidate cloud
    against the real target cloud (both ``(n, 2)`` arrays of the kinematic)."""
    spec = config["coverage"]
    maha2_gate = float(spec["mahalanobis2_99_2dof"])
    bin_widths = [float(v) for v in spec["density_bin_widths"]]
    acc = float(spec["physical_acceptance_abs_tx_ty_max"])
    real_xy = np.asarray(real_xy, dtype=np.float64)
    candidate_xy = np.asarray(candidate_xy, dtype=np.float64)
    real_xy = real_xy[np.isfinite(real_xy).all(axis=1)]
    candidate_xy = candidate_xy[np.isfinite(candidate_xy).all(axis=1)]
    # Pre-registered physical acceptance: drop tracklets outside the FASER
    # tracker acceptance (mis-reconstructed / unphysical) from BOTH clouds so
    # the Mahalanobis envelope is not inflated by unphysical outliers.
    real_xy = real_xy[(np.abs(real_xy) <= acc).all(axis=1)]
    candidate_xy = candidate_xy[(np.abs(candidate_xy) <= acc).all(axis=1)]

    out: dict[str, Any] = {
        "n_real": int(real_xy.shape[0]),
        "n_candidate": int(candidate_xy.shape[0]),
        "mahalanobis2_gate": maha2_gate,
    }
    if real_xy.shape[0] == 0:
        out["sufficient"] = False
        return out
    out["sufficient"] = candidate_xy.shape[0] >= int(spec["min_candidate_pairs_absolute"])
    out["fraction_within_mahalanobis_99"] = _mahalanobis_envelope_fraction(
        candidate_xy, real_xy, maha2_gate
    )
    out["density_bin_occupancy_fraction"] = {
        f"{w:g}": _bin_occupancy_fraction(candidate_xy, real_xy, w) for w in bin_widths
    }
    out["quadrant_coverage"] = _quadrant_coverage(candidate_xy, real_xy)
    # Quantile comparison (report-only): does the candidate reach the real tail?
    for axis, name in ((0, "tx"), (1, "ty")):
        out[f"real_{name}_abs_quantiles"] = {
            f"p{q:g}": float(np.percentile(np.abs(real_xy[:, axis]), q))
            for q in spec["quantile_levels"]
        }
        out[f"candidate_{name}_abs_quantiles"] = {
            f"p{q:g}": float(np.percentile(np.abs(candidate_xy[:, axis]), q))
            for q in spec["quantile_levels"]
        }
    return out


# ---------------------------------------------------------------------------
# Particle-species / momentum / origin audit (report-only).
# ---------------------------------------------------------------------------


def audit_particle_domain(
    cloud: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Report-only audit of the candidate's particle domain vs the muon-like
    real target.  A candidate can cover ``(tx, ty)`` angularly yet be the wrong
    species/momentum for a muon conditional-J model (e.g. K-short pion
    daughters); such a candidate is flagged
    ``angular_support_only_particle_domain_mismatch`` and is NOT authorised for
    J-model training."""
    spec = config["particle_domain_audit"]
    pdg = np.asarray(cloud["truth_pdg"], dtype=np.int64)
    qop = np.asarray(cloud["truth_q_over_p_per_mev"], dtype=np.float64)
    qop = qop[np.isfinite(qop)]
    # Momentum magnitude in GeV from |q/p| [1/MeV] (|q|=1 for these tracks).
    p_gev = 1.0e-3 / np.maximum(np.abs(qop), 1.0e-300)
    pdg_vals, pdg_counts = np.unique(np.abs(pdg), return_counts=True)
    order = np.argsort(-pdg_counts)
    dominant_pdg = int(pdg_vals[order[0]]) if pdg_vals.size else 0
    dominant_fraction = float(pdg_counts[order[0]] / pdg.size) if pdg_vals.size and pdg.size else 0.0
    expected = {int(v) for v in spec["expected_abs_pdg"]}
    species_compatible = dominant_pdg in expected
    p_lo = float(spec["momentum_gev_min"])
    p_hi = float(spec["momentum_gev_max"])
    p_in = p_gev[(p_gev >= p_lo) & (p_gev <= p_hi)]
    momentum_fraction_in_range = float(p_in.size / p_gev.size) if p_gev.size else 0.0
    momentum_compatible = momentum_fraction_in_range >= float(spec["min_momentum_fraction_in_range"])
    return {
        "n_tracks": int(pdg.size),
        "dominant_abs_pdg": dominant_pdg,
        "dominant_abs_pdg_fraction": dominant_fraction,
        "abs_pdg_counts": {str(int(v)): int(c) for v, c in zip(pdg_vals, pdg_counts)},
        "expected_abs_pdg": sorted(expected),
        "species_compatible": bool(species_compatible),
        "momentum_gev_median": float(np.median(p_gev)) if p_gev.size else None,
        "momentum_gev_p05": float(np.percentile(p_gev, 5)) if p_gev.size else None,
        "momentum_gev_p95": float(np.percentile(p_gev, 95)) if p_gev.size else None,
        "momentum_range_gev": [p_lo, p_hi],
        "momentum_fraction_in_range": momentum_fraction_in_range,
        "momentum_compatible": bool(momentum_compatible),
        "particle_domain_compatible": bool(species_compatible and momentum_compatible),
        "source_z_mm_median": float(np.median(cloud["source_z_mm"])) if cloud["source_z_mm"].size else None,
        "lever_arm_mm_median": float(np.median(cloud["lever_arm_mm"])) if cloud["lever_arm_mm"].size else None,
        "report_only": True,
    }


# ---------------------------------------------------------------------------
# Gate evaluation + decision tree (pre-registered).
# ---------------------------------------------------------------------------


def evaluate_candidate(
    candidate: Mapping[str, Any],
    coverage_by_pair: Mapping[str, dict[str, Any]],
    audit_by_pair: Mapping[str, dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen coverage + statistics gates to one candidate.

    Primary gate (per gated pair type): ``fraction_within_mahalanobis_99 >=
    min_fraction_within_support`` AND the pre-registered density bin-occupancy
    gate at the primary bin width.  Statistics gates: minimum candidate pairs
    and minimum independent sources.  The particle-domain audit is report-only
    but determines the ``angular_support_only_particle_domain_mismatch`` class.
    """
    cov_spec = config["coverage"]
    frac_gate = float(cov_spec["min_fraction_within_support"])
    density_width = f"{float(cov_spec['primary_density_bin_width']):g}"
    density_gate = float(cov_spec["min_density_bin_occupancy_fraction"])
    min_pairs = int(cov_spec["min_candidate_pairs_per_pair_type"])
    min_sources = int(cov_spec["min_independent_sources"])

    per_pair: dict[str, Any] = {}
    all_gated_pass = True
    any_stats_fail = False
    for tgt in GATED_TARGET_STATIONS:
        label = f"0->{tgt}"
        cov = coverage_by_pair.get(label)
        if cov is None or not cov.get("sufficient", False):
            per_pair[label] = {"gated": True, "within_gate": False, "reason": "insufficient_data"}
            all_gated_pass = False
            any_stats_fail = True
            continue
        maha = cov["fraction_within_mahalanobis_99"]
        density = cov["density_bin_occupancy_fraction"].get(density_width)
        n_cand = cov["n_candidate"]
        n_src = cov.get("n_sources_with_pairs")
        stats_ok = (n_cand >= min_pairs) and (n_src is None or n_src >= min_sources)
        maha_ok = maha is not None and maha >= frac_gate
        density_ok = density is not None and density >= density_gate
        within = bool(maha_ok and density_ok and stats_ok)
        # Preserve the full coverage metric block and add the gate evaluation.
        per_pair[label] = {
            **cov,
            "gated": True,
            "density_bin_occupancy_at_primary_width": density,
            "mahalanobis_ok": bool(maha_ok),
            "density_ok": bool(density_ok),
            "statistics_ok": bool(stats_ok),
            "within_gate": within,
        }
        all_gated_pass = all_gated_pass and within
        any_stats_fail = any_stats_fail or (not stats_ok)

    # Report-only (0,3).
    for tgt in REPORT_ONLY_TARGET_STATIONS:
        label = f"0->{tgt}"
        cov = coverage_by_pair.get(label) or {}
        per_pair[label] = {
            **cov,
            "gated": False,
            "report_only": True,
        }

    # Particle-domain compatibility is taken from the gated-pair audits (they
    # share the same source population).
    audits = [audit_by_pair[f"0->{t}"] for t in GATED_TARGET_STATIONS if f"0->{t}" in audit_by_pair]
    particle_compatible = bool(audits) and all(a["particle_domain_compatible"] for a in audits)

    return {
        "candidate": str(candidate["name"]),
        "per_pair": per_pair,
        "coverage_gates_pass": bool(all_gated_pass),
        "any_statistics_fail": bool(any_stats_fail),
        "particle_domain_compatible": particle_compatible,
        "particle_domain_audit": audit_by_pair,
    }


def decide(candidate_results: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    """Pre-registered mutually-exclusive decision tree.

    * ``existing_mc_real_wide_ty_support_validated``: >=1 candidate passes the
      coverage gates on BOTH gated pair types AND is particle-domain compatible.
    * ``existing_mc_angular_support_only_particle_domain_mismatch``: >=1
      candidate passes coverage gates but NONE is particle-domain compatible.
    * ``existing_mc_insufficient_real_wide_ty_support``: no candidate passes the
      coverage gates.
    * ``wide_ty_mc_coverage_validation_inconclusive``: every candidate is
      statistically insufficient (cannot even evaluate coverage).
    """
    tree = config["decision_tree"]
    coverage_passing = [r for r in candidate_results if r["coverage_gates_pass"]]
    compatible_passing = [r for r in coverage_passing if r["particle_domain_compatible"]]
    all_insufficient = all(r["any_statistics_fail"] and not r["coverage_gates_pass"] for r in candidate_results)

    if compatible_passing:
        decision = tree["validated"]
        validated = [r["candidate"] for r in compatible_passing]
    elif coverage_passing:
        decision = tree["angular_only_particle_mismatch"]
        validated = []
    elif all_insufficient and candidate_results:
        decision = tree["inconclusive"]
        validated = []
    else:
        decision = tree["insufficient"]
        validated = []

    success = decision == tree["validated"]
    return {
        "kind": "wide_ty_mc_support_feasibility_decision",
        "decision": decision,
        "validated_candidates": validated,
        "coverage_passing_candidates": [r["candidate"] for r in coverage_passing],
        # Terminal permissions: WB82 NEVER authorises alignment, and only the
        # `validated` state permits a FUTURE conditional-J retrain (itself a
        # separate pre-registered campaign, not this one).
        "existing_mc_real_wide_ty_support_validated": bool(success),
        "real_kinematic_jacobian_support_validated": bool(success),
        "conditional_j_retrain_permitted": bool(success),
        "new_mc_generation_required": bool(decision == tree["insufficient"]),
        "held_out_accessed": False,
        "real_data_alignment_authorized": False,
        "geometry_write_allowed": False,
        "official_conditions_write_allowed": False,
        "external_constraint_ingest_authorized": False,
        "measurement_model_validated": False,
        "real_data_alignment_v2_preregistration_allowed": False,
        "residual_blind": True,
        "diagnostic_only": True,
    }
