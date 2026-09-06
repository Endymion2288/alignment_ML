#!/usr/bin/env python3
"""Workbook 78: read-only run-range / data-contract audit.

Compare Family-2 train-range 00100-00149 against already-opened development
00350-00399, per DSID, along the physical data chain.  Does not train, does
not change architecture / loss / B, and never opens Final Blind or sealed
test.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.field_chi2_matching import build_field_candidates, candidate_labels
from datasets.propagation_loader import load_propagation_records
from datasets.root_loader import load_events
from evaluation.pairwise_metrics import probability_to_logit
from training.curriculum_mlp import CandidateSet
from training.geometry_aware_transformer import (
    ADJACENT_STATION_PAIRS,
    ALL_STATION_PAIRS,
    EDGE_FEATURE_NAMES,
    build_transformer_graph_bundle,
    geometric_edge_features,
)
from training.run_range_data_contract import (
    KS_LARGE,
    KS_SMALL,
    assert_geometry_tables_disjoint,
    compare_univariate,
    decide_shift_sources,
    feature_origin,
    generate_geometry_payload_table,
    parse_xaod_identity,
    payload_parameter_l2,
    refuse_forbidden_path,
    refuse_forbidden_source_id,
    sha256_file,
    write_json,
)
from training.source_diversity_audit import (
    AUTHORIZED_SIX_TRAIN_SOURCES,
    RESERVED_BLIND_SOURCES,
    UNUSED_RESERVE_SOURCES,
)
from datasets.physical_curriculum import CurriculumSample


TRAIN_PHYSICAL = Path("outputs/mc24_four_station_source_diversity_train_v1/physical_corpus_manifest.json")
DEV_PHYSICAL = Path("outputs/mc24_four_station_source_diversity_blind_v1/physical_corpus_manifest.json")
TRAIN_OVERLAY = Path(
    "outputs/mc24_four_station_source_diversity_train_v1/overlay_synthetic_v1/synthetic_corpus_manifest.json"
)
DEV_OVERLAY = Path(
    "outputs/mc24_four_station_source_diversity_blind_v1/overlay_synthetic_v1/synthetic_corpus_manifest.json"
)
W64_ROOT = Path("outputs/mc24_four_station_source_diversity_v1/checkpoint")
W64_SHA = "0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236"
OUT = Path("outputs/mc24_four_station_run_range_data_contract_v1")

SOURCE_PAIRS = (
    ("mc24_100047_00100_00149", "mc24_100047_00350_00399"),
    ("mc24_100048_00100_00149", "mc24_100048_00350_00399"),
)
MATCHED_PAYLOADS = ("iteration_00_reference", "iteration_00_hard_s3_ry")
NAME_ONLY_PAYLOADS = ("iteration_00_draw_00",)
TRACKLET_FEATURES = ("x_mm", "y_mm", "tx", "ty", "z_mm", "chi2", "n_hit", "log_sigma_x", "log_sigma_y", "log_sigma_tx", "log_sigma_ty")
SCAN_POLICY_KEYS = (
    "scan_mode",
    "alignment_formulation",
    "gauge",
    "relative_curriculum",
    "nevents",
    "q_over_p_mode",
    "min_truth_match_fraction",
    "chi2_gate",
    "refinement_iterations",
)


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_json(path: Path) -> dict[str, Any]:
    refuse_forbidden_path(path)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _annotate(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    payload["origin"] = feature_origin(str(row["name"]))
    return payload


def _source_entry(manifest: Mapping[str, Any], source_id: str) -> dict[str, Any]:
    refuse_forbidden_source_id(source_id)
    for item in manifest.get("sources", ()):
        if str(item.get("source_id")) == source_id:
            return dict(item)
    raise SystemExit(f"{source_id} is absent from physical corpus manifest")


def _point(entry: Mapping[str, Any], payload_id: str) -> dict[str, Any]:
    for item in entry.get("points", ()):
        if str(item.get("payload_id") or item.get("name")) == payload_id:
            return dict(item)
    raise SystemExit(f"{entry.get('source_id')} has no point {payload_id}")


def _dummy_sample(source_id: str, payload_id: str, point: Mapping[str, Any]) -> CurriculumSample:
    return CurriculumSample(
        source_id=source_id,
        source_ids=(source_id,),
        split="train" if "00100_00149" in source_id else "validation",
        payload_id=payload_id,
        magnitude_mm=float(point.get("condition_magnitude") or 0.0),
        direction_trial=str(point.get("direction_trial") or payload_id),
        injected_offsets_xy_mm={},
        source_event_uids=(),
        physical_event_uids=(),
        physical_tracklets=Path(point["physical_tracklets"]),
        physical_propagations=Path(point["physical_propagations"]),
        physical_payload_manifest=Path(point["physical_payload_manifest"]),
        synthetic_tracklets=Path(point["physical_tracklets"]),
        field_candidates=Path(point["physical_propagations"]),
        condition_axis=str(point.get("condition_axis") or "four_station_relative_l2"),
        condition_value=point.get("condition_value"),
        condition_magnitude=point.get("condition_magnitude"),
        injected_station_transforms=dict(point.get("injected_station_transforms") or {}),
        alignment_parameter_values=dict(point.get("alignment_parameter_values") or {}),
    )


def _tracklet_columns(events) -> dict[str, np.ndarray]:
    buckets: dict[str, list[float]] = {name: [] for name in TRACKLET_FEATURES}
    buckets["station"] = []
    for event in events:
        for index in range(event.size):
            station = int(event.station_id[index])
            diagonal = np.diagonal(np.asarray(event.covariance[index], dtype=np.float64))
            buckets["station"].append(float(station))
            buckets["x_mm"].append(float(event.state[index, 0]))
            buckets["y_mm"].append(float(event.state[index, 1]))
            buckets["tx"].append(float(event.state[index, 2]))
            buckets["ty"].append(float(event.state[index, 3]))
            buckets["z_mm"].append(float(event.z_mm[index]))
            buckets["chi2"].append(float(event.chi2[index]))
            buckets["n_hit"].append(float(event.n_hit[index]))
            buckets["log_sigma_x"].append(float(np.log(diagonal[0])) if diagonal[0] > 0.0 else np.nan)
            buckets["log_sigma_y"].append(float(np.log(diagonal[1])) if diagonal[1] > 0.0 else np.nan)
            buckets["log_sigma_tx"].append(float(np.log(diagonal[2])) if diagonal[2] > 0.0 else np.nan)
            buckets["log_sigma_ty"].append(float(np.log(diagonal[3])) if diagonal[3] > 0.0 else np.nan)
    return {name: np.asarray(values, dtype=np.float64) for name, values in buckets.items()}


def _truth_adjacent_features(events, records) -> dict[str, np.ndarray]:
    collected: dict[str, list[float]] = defaultdict(list)
    for event in events:
        if event.truth_particle_id is None:
            continue
        for pair in ADJACENT_STATION_PAIRS:
            candidates = build_field_candidates(event, records, pair[0], pair[1], chi2_gate=None, q_over_p_mode=0)
            if not candidates:
                continue
            labels = candidate_labels(event, candidates)
            for candidate, label in zip(candidates, labels):
                if not bool(label):
                    continue
                features = geometric_edge_features(event, candidate)
                prefix = f"e{pair[0]}{pair[1]}_"
                for name, value in zip(EDGE_FEATURE_NAMES, features):
                    collected[prefix + name].append(float(value))
    return {name: np.asarray(values, dtype=np.float64) for name, values in collected.items()}


def _compare_maps(left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    names = sorted(set(left) | set(right))
    rows = [
        _annotate(compare_univariate(left.get(name, np.empty(0)), right.get(name, np.empty(0)), name))
        for name in names
    ]
    rows.sort(key=lambda row: -float(row["ks_statistic"]) if row.get("status") == "ok" else -1.0)
    return rows


def _max_ks(rows: list[dict[str, Any]], substring: str | None = None) -> float | None:
    values = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        if substring is not None and substring not in str(row["name"]):
            continue
        values.append(float(row["ks_statistic"]))
    return None if not values else float(max(values))


def _station_z_identical(train_cols: Mapping[str, np.ndarray], dev_cols: Mapping[str, np.ndarray]) -> bool:
    for station in (0, 1, 2, 3):
        left = train_cols["z_mm"][train_cols["station"] == float(station)]
        right = dev_cols["z_mm"][dev_cols["station"] == float(station)]
        if left.size == 0 or right.size == 0:
            return False
        if abs(float(np.median(left)) - float(np.median(right))) > 1.0e-6:
            return False
        if float(np.ptp(left)) > 1.0e-6 or float(np.ptp(right)) > 1.0e-6:
            return False
    return True


def _w64_truth_logits(events, records, sample, loaded, device: str, batch_size: int) -> dict[str, Any]:
    from training.route_aware_transformer import predict_route_aware_scores

    sets: list[CandidateSet] = []
    n_events = 0
    n_skipped_incomplete = 0
    for event in events:
        if event.truth_particle_id is None:
            n_skipped_incomplete += 1
            continue
        event_sets: list[CandidateSet] = []
        complete = True
        for pair in ALL_STATION_PAIRS:
            candidates = build_field_candidates(event, records, pair[0], pair[1], chi2_gate=None, q_over_p_mode=0)
            if not candidates:
                complete = False
                break
            event_sets.append(
                CandidateSet(
                    sample=sample,
                    event=event,
                    station_pair=pair,
                    candidates=tuple(candidates),
                    features=np.zeros((len(candidates), 1), dtype=np.float64),
                    labels=candidate_labels(event, candidates),
                )
            )
        if not complete:
            n_skipped_incomplete += 1
            continue
        sets.extend(event_sets)
        n_events += 1
    if not sets:
        return {
            "n_complete_events": 0,
            "n_skipped_incomplete": n_skipped_incomplete,
            "features": [],
        }
    bundle = build_transformer_graph_bundle(sets, context_mode="full_event")
    prediction = predict_route_aware_scores(
        loaded["model"],
        bundle,
        loaded["artifact"].node_standardizer,
        loaded["artifact"].edge_standardizer,
        device=device,
        batch_size=batch_size,
    )
    logits = [probability_to_logit(np.asarray(block, dtype=np.float64)) for block in prediction.base_edge_scores]
    collected: dict[str, list[float]] = defaultdict(list)
    for candidate_set, block in zip(bundle.adjacent_sets, logits):
        pair = candidate_set.station_pair
        if pair not in ADJACENT_STATION_PAIRS:
            continue
        for score, label in zip(block.tolist(), np.asarray(candidate_set.labels).tolist()):
            if not bool(label):
                continue
            collected[f"L{pair[0]}{pair[1]}"].append(float(score))
    return {
        "n_complete_events": n_events,
        "n_skipped_incomplete": n_skipped_incomplete,
        "logits": {name: np.asarray(values, dtype=np.float64) for name, values in collected.items()},
    }


def _overlay_metadata(manifest_path: Path) -> dict[str, Any]:
    refuse_forbidden_path(manifest_path)
    raw = _load_json(manifest_path)
    counts: dict[str, int] = defaultdict(int)
    payloads = []
    constituent: set[str] = set()
    for sample in raw.get("samples", ()):
        source_id = str(sample.get("source_id"))
        refuse_forbidden_source_id(source_id)
        for source in sample.get("source_ids") or (source_id,):
            refuse_forbidden_source_id(str(source))
            constituent.add(str(source))
        for uid in sample.get("source_event_uids") or ():
            refuse_forbidden_source_id(str(uid).split(":")[0])
            counts[str(uid).split(":")[0]] += 1
        payloads.append(
            {
                "payload_id": sample.get("payload_id"),
                "source_id": source_id,
                "n_source_event_uids": int(len(sample.get("source_event_uids") or ())),
                "alignment_parameter_l2_vs_zero": payload_parameter_l2(
                    {str(key): float(value) for key, value in dict(sample.get("alignment_parameter_values") or {}).items()},
                    {},
                ),
            }
        )
    recipe = dict(raw.get("synthetic_multitrack") or {})
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "created_utc": raw.get("created_utc"),
        "q_over_p_mode": raw.get("q_over_p_mode"),
        "physical_geometry_repropagation": raw.get("physical_geometry_repropagation"),
        "sources_by_split": raw.get("sources_by_split"),
        "uid_counts_by_source": dict(counts),
        "constituent_sources": sorted(constituent),
        "payloads": payloads,
        "synthetic_multitrack": recipe,
        "overlay_seed": recipe.get("seed"),
        "events_per_payload": recipe.get("events_per_payload"),
        "scale_events_by_pool_sources": recipe.get("scale_events_by_pool_sources"),
    }


def _scan_policy(path: str | Path) -> dict[str, Any]:
    refuse_forbidden_path(path)
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    block = dict(raw.get("physical_refit_capture_scan") or {})
    sampling = dict(block.get("relative_sampling") or {})
    return {
        "path": str(path),
        "input_xaod": block.get("input_xaod"),
        "policy": {key: block.get(key) for key in SCAN_POLICY_KEYS},
        "relative_sampling_seed": sampling.get("seed"),
        "relative_bounds": sampling.get("relative_bounds"),
        "common_bounds": sampling.get("common_bounds"),
        "hard_relative_families": sampling.get("hard_relative_families"),
        "n_random_families": sampling.get("n_random_families"),
    }


def _content_selection(path: str | Path) -> dict[str, Any]:
    refuse_forbidden_path(path)
    audit = _load_json(Path(path))
    n_events = float(audit.get("events") or 0)
    occupancy = {}
    for station, count in dict(audit.get("events_with_station") or {}).items():
        occupancy[str(station)] = None if n_events <= 0 else float(count) / n_events
    return {
        "path": str(path),
        "n_events": int(audit.get("events") or 0),
        "n_tracklets": int(audit.get("tracklets") or 0),
        "station_counts": dict(audit.get("station_counts") or {}),
        "events_with_station": dict(audit.get("events_with_station") or {}),
        "occupancy_rate": occupancy,
        "z_mm_by_station": dict(audit.get("z_mm_by_station") or {}),
        "truth_pdg_counts": dict((audit.get("truth") or {}).get("truth_pdg_counts") or {}),
    }


def _layer_a(train_entry, dev_entry, payloads: tuple[str, ...]) -> dict[str, Any]:
    rows = []
    for payload_id in payloads:
        train_point = _point(train_entry, payload_id)
        dev_point = _point(dev_entry, payload_id)
        left = {str(key): float(value) for key, value in dict(train_point.get("alignment_parameter_values") or {}).items()}
        right = {str(key): float(value) for key, value in dict(dev_point.get("alignment_parameter_values") or {}).items()}
        train_manifest = Path(train_point["physical_payload_manifest"])
        dev_manifest = Path(dev_point["physical_payload_manifest"])
        refuse_forbidden_path(train_manifest)
        refuse_forbidden_path(dev_manifest)
        rows.append(
            {
                "payload_id": payload_id,
                "parameter_l2": payload_parameter_l2(left, right),
                "train_parameters": left,
                "development_parameters": right,
                "geometry_matched": bool(payload_parameter_l2(left, right) <= 1.0e-9),
                "train_payload_sha256": sha256_file(train_manifest),
                "development_payload_sha256": sha256_file(dev_manifest),
            }
        )
    return {"comparisons": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--with-w64", action="store_true", default=True)
    parser.add_argument("--no-w64", action="store_false", dest="with_w64")
    parser.add_argument("--output-root", default=str(OUT))
    args = parser.parse_args()
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)

    for source in (*sum(SOURCE_PAIRS, ()),):
        refuse_forbidden_source_id(source)
    for path in (TRAIN_PHYSICAL, DEV_PHYSICAL, TRAIN_OVERLAY, DEV_OVERLAY):
        refuse_forbidden_path(path)

    train_physical = _load_json(TRAIN_PHYSICAL)
    dev_physical = _load_json(DEV_PHYSICAL)
    if set(UNUSED_RESERVE_SOURCES) & {str(item.get("source_id")) for item in dev_physical.get("sources", ())}:
        raise SystemExit("development physical corpus leaked unused final-blind sources")

    layer_a: dict[str, Any] = {
        "historical_seeds": {"train_curriculum": 20260821, "development_curriculum": 20260824}
    }
    layer_b: dict[str, Any] = {}
    layer_c: dict[str, Any] = {}
    layer_e: dict[str, Any] = {}
    xaod_rows = []
    policy_rows = []
    shift_tables: list[dict[str, Any]] = []
    w64_loaded = None
    if args.with_w64:
        from scripts.run_frozen_association_backbone import _load_v2, _sha256

        observed = _sha256(W64_ROOT / "route_aware_transformer_v2.pt")
        if observed != W64_SHA:
            raise SystemExit(f"W64 checkpoint sha {observed} != frozen {W64_SHA}")
        w64_loaded = _load_v2(W64_ROOT, args.device)
        w64_loaded["checkpoint_sha256"] = observed

    for train_source, dev_source in SOURCE_PAIRS:
        train_entry = _source_entry(train_physical, train_source)
        dev_entry = _source_entry(dev_physical, dev_source)
        train_xaod = parse_xaod_identity(str(train_entry["input_xaod"]))
        dev_xaod = parse_xaod_identity(str(dev_entry["input_xaod"]))
        xaod_rows.append(
            {
                "train_source": train_source,
                "development_source": dev_source,
                "train_xaod": train_xaod,
                "development_xaod": dev_xaod,
                "same_software_tag": train_xaod["software_tag"] == dev_xaod["software_tag"],
                "train_refit_chain": train_physical.get("refit_chain"),
                "development_refit_chain": dev_physical.get("refit_chain"),
                "same_refit_chain": train_physical.get("refit_chain") == dev_physical.get("refit_chain"),
            }
        )
        train_policy = _scan_policy(train_entry["physical_scan_config"])
        dev_policy = _scan_policy(dev_entry["physical_scan_config"])
        policy_rows.append(
            {
                "train_source": train_source,
                "development_source": dev_source,
                "train": train_policy,
                "development": dev_policy,
                "same_reconstruction_policy": train_policy["policy"] == dev_policy["policy"],
                "same_sampling_bounds": train_policy["relative_bounds"] == dev_policy["relative_bounds"]
                and train_policy["common_bounds"] == dev_policy["common_bounds"]
                and train_policy["hard_relative_families"] == dev_policy["hard_relative_families"],
                "same_sampling_seed": train_policy["relative_sampling_seed"] == dev_policy["relative_sampling_seed"],
            }
        )
        pair_key = f"{train_source}__vs__{dev_source}"
        layer_a[pair_key] = _layer_a(train_entry, dev_entry, MATCHED_PAYLOADS + NAME_ONLY_PAYLOADS)
        layer_b[pair_key] = {}
        layer_c[pair_key] = {}
        layer_e[pair_key] = {}
        for payload_id in MATCHED_PAYLOADS + NAME_ONLY_PAYLOADS:
            train_point = _point(train_entry, payload_id)
            dev_point = _point(dev_entry, payload_id)
            for path in (
                train_point["physical_tracklets"],
                train_point["physical_propagations"],
                dev_point["physical_tracklets"],
                dev_point["physical_propagations"],
            ):
                refuse_forbidden_path(path)
            train_events = load_events(train_point["physical_tracklets"], require_mc_labels=True)
            dev_events = load_events(dev_point["physical_tracklets"], require_mc_labels=True)
            train_cols = _tracklet_columns(train_events)
            dev_cols = _tracklet_columns(dev_events)
            tracklet_rows = []
            for station in (0, 1, 2, 3):
                train_mask = train_cols["station"] == float(station)
                dev_mask = dev_cols["station"] == float(station)
                for name in TRACKLET_FEATURES:
                    tracklet_rows.append(
                        _annotate(
                            compare_univariate(
                                train_cols[name][train_mask],
                                dev_cols[name][dev_mask],
                                f"s{station}_{name}",
                            )
                        )
                    )
            tracklet_rows.sort(key=lambda row: -float(row["ks_statistic"]) if row.get("status") == "ok" else -1.0)
            train_prop = load_propagation_records(train_point["physical_propagations"])
            dev_prop = load_propagation_records(dev_point["physical_propagations"])
            edge_rows = _compare_maps(
                _truth_adjacent_features(train_events, train_prop),
                _truth_adjacent_features(dev_events, dev_prop),
            )
            selection = {
                "train": _content_selection(train_point["physical_content_audit"]),
                "development": _content_selection(dev_point["physical_content_audit"]),
            }
            occ_diffs = []
            for station in ("0", "1", "2", "3"):
                left = selection["train"]["occupancy_rate"].get(station)
                right = selection["development"]["occupancy_rate"].get(station)
                if left is not None and right is not None:
                    occ_diffs.append(abs(float(left) - float(right)))
            layer_b[pair_key][payload_id] = {
                "n_train_events": int(len(train_events)),
                "n_development_events": int(len(dev_events)),
                "n_train_tracklets": int(train_cols["z_mm"].size),
                "n_development_tracklets": int(dev_cols["z_mm"].size),
                "station_z_identical": _station_z_identical(train_cols, dev_cols),
                "features": tracklet_rows,
                "content_audit": selection,
                "occupancy_rate_max_abs_diff": None if not occ_diffs else float(max(occ_diffs)),
            }
            layer_c[pair_key][payload_id] = {
                "n_train_propagations": int(train_prop.size),
                "n_development_propagations": int(dev_prop.size),
                "q_over_p_mode": 0,
                "features": edge_rows,
            }
            w64_block: dict[str, Any] = {"skipped": True, "reason": "with_w64=false"}
            if w64_loaded is not None:
                train_w64 = _w64_truth_logits(
                    train_events,
                    train_prop,
                    _dummy_sample(train_source, payload_id, train_point),
                    w64_loaded,
                    args.device,
                    args.batch_size,
                )
                dev_w64 = _w64_truth_logits(
                    dev_events,
                    dev_prop,
                    _dummy_sample(dev_source, payload_id, dev_point),
                    w64_loaded,
                    args.device,
                    args.batch_size,
                )
                w64_block = {
                    "w64_sha256": W64_SHA,
                    "n_train_complete_events": train_w64["n_complete_events"],
                    "n_development_complete_events": dev_w64["n_complete_events"],
                    "n_train_skipped_incomplete": train_w64["n_skipped_incomplete"],
                    "n_development_skipped_incomplete": dev_w64["n_skipped_incomplete"],
                    "features": _compare_maps(train_w64["logits"], dev_w64["logits"]),
                }
            layer_e[pair_key][payload_id] = w64_block
            shift_tables.append(
                {
                    "pair": pair_key,
                    "payload_id": payload_id,
                    "geometry_role": "matched" if payload_id in MATCHED_PAYLOADS else "name_only_confound",
                    "tracklet_features": tracklet_rows,
                    "edge_features": edge_rows,
                    "w64_features": w64_block.get("features") or [],
                }
            )

    overlay = {"train": _overlay_metadata(TRAIN_OVERLAY), "development": _overlay_metadata(DEV_OVERLAY)}
    overlay_recipe_same = (
        overlay["train"]["q_over_p_mode"] == overlay["development"]["q_over_p_mode"]
        and overlay["train"]["physical_geometry_repropagation"]
        == overlay["development"]["physical_geometry_repropagation"]
        and overlay["train"]["synthetic_multitrack"] == overlay["development"]["synthetic_multitrack"]
    )

    train_table = generate_geometry_payload_table(
        seed=271828, table_role="train_domain", include_hard_s3_ry=True
    )
    held_table = generate_geometry_payload_table(
        seed=314159, table_role="held_out_geometry", include_hard_s3_ry=False
    )
    disjoint = assert_geometry_tables_disjoint(train_table, held_table)
    write_json(output / "geometry_holdout" / "train_payloads.json", train_table)
    write_json(output / "geometry_holdout" / "heldout_payloads.json", held_table)
    write_json(output / "geometry_holdout" / "disjointness.json", disjoint)

    evidence_rows = []
    for pair_key, payload_block in layer_a.items():
        if pair_key == "historical_seeds":
            continue
        for item in payload_block["comparisons"]:
            evidence_rows.append({"pair": pair_key, **item})
    matched_l2 = [row["parameter_l2"] for row in evidence_rows if row["payload_id"] in MATCHED_PAYLOADS]
    draw_l2 = [row["parameter_l2"] for row in evidence_rows if row["payload_id"] in NAME_ONLY_PAYLOADS]
    ref_delta_ks = []
    ref_residual_ks = []
    ref_state_ks = []
    ref_l23_ks = []
    z_identical = []
    occupancy_diffs = []
    for pair_key in layer_c:
        ref_c = layer_c[pair_key]["iteration_00_reference"]["features"]
        ref_b = layer_b[pair_key]["iteration_00_reference"]["features"]
        z_identical.append(bool(layer_b[pair_key]["iteration_00_reference"]["station_z_identical"]))
        occ = layer_b[pair_key]["iteration_00_reference"].get("occupancy_rate_max_abs_diff")
        if occ is not None:
            occupancy_diffs.append(float(occ))
        delta = _max_ks(ref_c, "delta_z_mm")
        if delta is not None:
            ref_delta_ks.append(delta)
        residual = _max_ks(ref_c, "residual_")
        if residual is not None:
            ref_residual_ks.append(residual)
        state = max(
            value
            for value in (
                _max_ks(ref_b, "_tx"),
                _max_ks(ref_b, "_ty"),
                _max_ks(ref_b, "_x_mm"),
                _max_ks(ref_b, "_y_mm"),
            )
            if value is not None
        ) if any(
            value is not None
            for value in (
                _max_ks(ref_b, "_tx"),
                _max_ks(ref_b, "_ty"),
                _max_ks(ref_b, "_x_mm"),
                _max_ks(ref_b, "_y_mm"),
            )
        ) else None
        if state is not None:
            ref_state_ks.append(state)
        if pair_key in layer_e and "iteration_00_reference" in layer_e[pair_key]:
            l23 = _max_ks(layer_e[pair_key]["iteration_00_reference"].get("features") or [], "L23")
            if l23 is not None:
                ref_l23_ks.append(l23)

    same_software = all(row["same_software_tag"] for row in xaod_rows)
    same_policy = all(row["same_reconstruction_policy"] for row in policy_rows)
    decision = decide_shift_sources(
        matched_payload_max_l2=None if not matched_l2 else float(max(matched_l2)),
        draw_00_min_l2=None if not draw_l2 else float(min(draw_l2)),
        same_software_tag=bool(same_software and same_policy),
        station_z_identical=all(z_identical) if z_identical else None,
        identity_delta_z_max_ks=None if not ref_delta_ks else float(max(ref_delta_ks)),
        identity_residual_max_ks=None if not ref_residual_ks else float(max(ref_residual_ks)),
        identity_state_max_ks=None if not ref_state_ks else float(max(ref_state_ks)),
        overlay_recipe_identical=overlay_recipe_same,
        occupancy_rate_max_abs_diff=None if not occupancy_diffs else float(max(occupancy_diffs)),
    )
    decision.update(
        {
            "artifacts": {
                "A": "layer_A_geometry.json",
                "B": "layer_B_reconstruction.json",
                "C": "layer_C_propagation.json",
                "D": "layer_D_overlay.json",
                "E": "layer_E_rphys.json",
            },
            "L23_identity_payload": {
                "identity_L23_max_ks": None if not ref_l23_ks else float(max(ref_l23_ks)),
                "artifact": "layer_E_rphys.json",
            },
            "why_00350_is_not_an_exchange_test": [
                "00350_00399 is already-opened reserved-blind development, not a held-out geometry.",
                "Historical train/development geometry banks used seeds 20260821 vs 20260824.",
                "Name-matched draw_* payloads are different numeric transforms.",
                "Family-2 train range is 00100-00149; development is a different run range.",
            ],
            "wb76_not_overturned": True,
            "representation_assumption_in_train_range": "OK",
            "training_authorized": False,
            "final_blind_eval_authorized": False,
            "sealed_test_accessed": False,
            "continue_to_v5a_frozen_head": False,
            "continue_to_15d_relative_wls": False,
            "ks_thresholds_used": {"ks_large": KS_LARGE, "ks_small": KS_SMALL},
        }
    )

    dataset_manifest_v2 = {
        "schema": "faser-dataset-contract-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "training_authorized": False,
        "final_blind_eval_authorized": False,
        "sealed_test_accessed": False,
        "continue_to_v5a_frozen_head": False,
        "continue_to_15d_relative_wls": False,
        "train_domain": {
            "sources": list(AUTHORIZED_SIX_TRAIN_SOURCES),
            "run_ranges": {
                "family1": ["00200_00299", "00300_00399"],
                "family2": ["00100_00149"],
            },
            "geometry_table": "geometry_holdout/train_payloads.json",
            "geometry_seed": 271828,
            "reconstruction_version": "s0013-r0022",
            "propagation_configuration": {
                "tool": "FaserActsExtrapolationTool",
                "q_over_p_mode": 0,
                "physical_geometry_repropagation": True,
            },
            "event_uid_convention": "source_id:run_id:event_id",
            "overlay_seed_historical": 20260813,
            "historical_physical_corpus": str(TRAIN_PHYSICAL),
        },
        "validation_domain": {
            "sources": list(RESERVED_BLIND_SOURCES),
            "run_range": "00350_00399",
            "status": "already_opened_development_diagnostic",
            "model_selection": "forbidden",
            "not_a_geometry_holdout": True,
            "reconstruction_version": "s0013-r0022",
            "historical_physical_corpus": str(DEV_PHYSICAL),
        },
        "heldout_geometry": {
            "table": "geometry_holdout/heldout_payloads.json",
            "seed": 314159,
            "generated": True,
            "physical_refit_authorized": False,
            "include_hard_s3_ry": False,
        },
        "heldout_run_range": {
            "status": "not_authorized",
            "note": "A new run-range holdout must be chosen after this audit; 00350 remains development-only.",
        },
        "prohibited_blind_assets": {
            "final_blind": list(UNUSED_RESERVE_SOURCES),
            "sealed": ["mc24_100116_*", "mc24_100117_*"],
        },
        "frozen_contracts_for_later_arms": {
            "utility_contract": "raw_energy_v1",
            "metric_version": "route_accounting_v2",
            "solver": "exact_unit_capacity",
            "candidate_builder": "truth_free_physical_enumeration",
            "arms": {
                "A": "W64 canonical energy baseline",
                "B": "raw physical route scorer",
                "C": "trainable physical route representation",
            },
            "arms_authorized": False,
        },
        "representation_study": {
            "authorized": False,
            "blocked_until": "dataset_contract_v2 accepted and geometry/run-range domains stop being confounded",
        },
    }

    environment = {
        "hostname": socket.gethostname(),
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_sha": _git_sha(),
    }
    metadata = {
        "workbook": 78,
        "git_sha": _git_sha(),
        "resolved_config": {
            "config": "configs/research_review/wp78_run_range_data_contract.yaml",
            "train_physical": str(TRAIN_PHYSICAL),
            "development_physical": str(DEV_PHYSICAL),
            "train_overlay": str(TRAIN_OVERLAY),
            "development_overlay": str(DEV_OVERLAY),
            "w64_root": str(W64_ROOT),
            "w64_sha256": W64_SHA,
            "with_w64": bool(args.with_w64),
        },
        "dataset_manifest": "dataset_manifest_v2.json",
        "environment": "environment.json",
        "training_authorized": False,
        "final_blind_eval_authorized": False,
        "sealed_test_accessed": False,
        "continue_to_v5a_frozen_head": False,
        "continue_to_15d_relative_wls": False,
    }

    write_json(output / "layer_A_geometry.json", _json_ready(layer_a))
    write_json(
        output / "layer_B_reconstruction.json",
        _json_ready({"xaod": xaod_rows, "scan_policy": policy_rows, "tracklets": layer_b}),
    )
    write_json(output / "layer_C_propagation.json", _json_ready(layer_c))
    write_json(
        output / "layer_D_overlay.json",
        _json_ready({"recipe_fields_match": overlay_recipe_same, **overlay}),
    )
    write_json(output / "layer_E_rphys.json", _json_ready(layer_e))
    write_json(output / "feature_shift_tables.json", _json_ready({"comparisons": shift_tables}))
    write_json(output / "shift_source_decision.json", _json_ready(decision))
    write_json(output / "dataset_manifest_v2.json", _json_ready(dataset_manifest_v2))
    write_json(output / "environment.json", _json_ready(environment))
    write_json(output / "run_metadata.json", _json_ready(metadata))
    print(json.dumps({"output_root": str(output), "git_sha": _git_sha(), "decision": decision}, indent=2, default=str))


if __name__ == "__main__":
    main()
