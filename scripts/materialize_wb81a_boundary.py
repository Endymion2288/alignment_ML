#!/usr/bin/env python3
"""Workbook 81a: materialize Arm A boundary sets and prove ΔU=0 identity.

Does not train.  Does not load WB79 Arm B or WB80 Arm C.  Does not open
00350, Final Blind, or sealed test.  Does not submit Condor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.physical_curriculum import load_synthetic_curriculum_manifest
from evaluation.route_counterfactuals import inclusion_gap
from models.route_energy import CANONICAL_ENERGY_VERSION
from training.curriculum_mlp import build_candidate_sets
from training.explicit_route_energy import (
    account_event,
    energy_table_from_utilities,
    enumerate_contiguous_physical_routes,
    w64_route_utilities,
)
from training.geometry_aware_transformer import ALL_STATION_PAIRS, build_transformer_graph_bundle
from training.route_aware_transformer import (
    _forward_route_batch,
    _make_route_batch,
    load_route_aware_transformer_artifact,
    materialize_route_candidate_tables,
)
from training.source_diversity_audit import RESERVED_BLIND_SOURCES, UNUSED_RESERVE_SOURCES
from training.source_transfer_cv import FAMILY1_DSID_PAIR, FAMILY2_DSID_PAIR, V5A_SOURCE_FAMILIES
from training.wb81_calibration_contract import (
    DELTA_MAX,
    EXPECTED_HOLDOUT_COUNTS,
    UNMATCHED_PENALTY,
    calibrated_energy_table,
    checksum_holdout,
    compare_identity,
    contract_source_guards,
    margin_bin,
    refuse_wb81a_path,
    selected_mask,
    zero_theta_delta,
)


W64_SHA = "0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236"
W64_ROOT = Path("outputs/mc24_four_station_source_diversity_v1/checkpoint")
FAMILY_CORPUS = {
    FAMILY1_DSID_PAIR: Path(
        "outputs/mc24_four_station_relative_route_v5a_source_transfer_v1/corpora/family1/overlay_synthetic_v1/synthetic_corpus_manifest.json"
    ),
    FAMILY2_DSID_PAIR: Path(
        "outputs/mc24_four_station_relative_route_v5a_source_transfer_v1/corpora/family2/overlay_synthetic_v1/synthetic_corpus_manifest.json"
    ),
}
VAL_PAYLOAD_IDS = frozenset({"iteration_00_draw_01", "iteration_00_draw_01_plus_common"})
OUTPUT_ROOT = Path("outputs/mc24_four_station_wb81a_calibration_foundation_v1")
IDENTITY_FAIL_MESSAGE = "WB81a identity fail => stop.  WB81b remains unauthorized."


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_ready(row), sort_keys=True, allow_nan=False) + "\n")


def _load_family_bundle(family: str):
    manifest = FAMILY_CORPUS[family]
    refuse_wb81a_path(manifest)
    expected = V5A_SOURCE_FAMILIES[family]
    _, samples, raw = load_synthetic_curriculum_manifest(
        manifest, require_all_splits=False, allowed_splits=("train",)
    )
    constituents = {str(source) for sample in samples for source in sample.source_ids}
    if constituents & (set(RESERVED_BLIND_SOURCES) | set(UNUSED_RESERVE_SOURCES)):
        raise SystemExit(f"{family} corpus leaked reserved/blind sources")
    if constituents != set(expected):
        raise SystemExit(f"{family} corpus sources {sorted(constituents)} != {list(expected)}")
    for sample in samples:
        refuse_wb81a_path(sample.synthetic_tracklets)
        refuse_wb81a_path(sample.field_candidates)
    candidate_sets = build_candidate_sets(
        samples, ALL_STATION_PAIRS, chi2_gate=None, feature_set="residual_v1", q_over_p_mode=0
    )
    bundle = build_transformer_graph_bundle(candidate_sets, context_mode="full_event")
    return samples, bundle, raw


def _w64_base_logits(bundle, model, artifact, device: torch.device, batch_size: int) -> dict[tuple[int, int], float]:
    tables = materialize_route_candidate_tables(bundle.graphs)
    logits: dict[tuple[int, int], float] = {}
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        for start in range(0, len(bundle.graphs), batch_size):
            graphs = bundle.graphs[start : start + batch_size]
            batch = _make_route_batch(graphs, artifact.node_standardizer, artifact.edge_standardizer, device, tables)
            output = _forward_route_batch(model, batch)
            values = output.base_edge_logits.detach().cpu().numpy().astype(np.float64)
            for value, owner, row in zip(values, batch.base.score_owner, batch.base.score_row):
                key = (int(owner), int(row))
                if key in logits:
                    raise RuntimeError("W64 base-logit reconstruction duplicated a candidate row")
                logits[key] = float(value)
    return logits


def _event_key(graph, family: str, event_index: int) -> dict[str, object]:
    payload_id = str(graph.sample.payload_id)
    return {
        "family": family,
        "event_index": int(event_index),
        "payload_id": payload_id,
        "in_family_split": "val_payload" if payload_id in VAL_PAYLOAD_IDS else "train_payload",
        "source_ids": list(graph.sample.source_ids),
        "source_event_uids": list(graph.sample.source_event_uids),
        "physical_event_uids": list(graph.sample.physical_event_uids),
        "run_id": int(graph.event.run_id),
        "event_id": int(graph.event.event_id),
    }


def materialize_family(family: str, graphs, adjacent_sets, logits, output: Path) -> dict[str, Any]:
    for name in (
        "complete_truths.jsonl",
        "a_selected_fakes.jsonl",
        "competing_routes.jsonl",
        "selected_sets.jsonl",
    ):
        path = output / name
        if path.exists():
            path.unlink()

    counts = {key: 0 for key in ("t_neg", "t_near", "t_comf", "f_a", "n_competing_rows", "flippable_neg_under_delta_max")}
    accounting = None
    n_events = 0
    n_routes = 0
    for event_index, graph in enumerate(graphs):
        routes = enumerate_contiguous_physical_routes(graph)
        if not routes:
            continue
        u_w64 = w64_route_utilities(graph, routes, logits, UNMATCHED_PENALTY)
        delta = zero_theta_delta(len(routes))
        if not np.array_equal(u_w64, u_w64 + delta):
            raise SystemExit(f"{IDENTITY_FAIL_MESSAGE} utilities changed at event {event_index}")
        table_a = energy_table_from_utilities(routes, u_w64, unmatched_penalty=UNMATCHED_PENALTY)
        table_new = calibrated_energy_table(routes, u_w64, delta, UNMATCHED_PENALTY)
        selected_a = selected_mask(table_a)
        selected_new = selected_mask(table_new)
        if selected_a != selected_new:
            raise SystemExit(f"{IDENTITY_FAIL_MESSAGE} selected set changed at event {event_index}")
        if not np.array_equal(u_w64, np.asarray([record.energy for record in table_new.records], dtype=np.float64)):
            raise SystemExit(f"{IDENTITY_FAIL_MESSAGE} table energies drifted at event {event_index}")
        key = _event_key(graph, family, event_index)
        block_a = account_event(graph, routes, table_a, adjacent_sets)
        metrics_new = account_event(graph, routes, table_new, adjacent_sets).as_dict()
        event_identity = compare_identity(selected_a, selected_new, block_a.as_dict(), metrics_new)
        if not event_identity["passed"]:
            raise SystemExit(f"{IDENTITY_FAIL_MESSAGE} metrics drifted at event {event_index}: {event_identity}")
        if accounting is None:
            accounting = block_a
        else:
            accounting.add(block_a)
        n_events += 1
        n_routes += len(routes)
        _append_jsonl(
            output / "selected_sets.jsonl",
            {
                **key,
                "selected_route_ids": list(selected_a),
                "n_routes": len(routes),
            },
        )
        selected_set = set(selected_a)
        for route_id, route in enumerate(routes):
            if bool(route.truth_consistent and len(route.stations) == 4):
                gap = inclusion_gap(table_a, route_id)
                bin_name = margin_bin(float(gap.inclusion_gap))
                counts[{"negative": "t_neg", "near_zero": "t_near", "comfortable": "t_comf"}[bin_name]] += 1
                flippable = bool(bin_name == "negative" and abs(float(gap.inclusion_gap)) < DELTA_MAX)
                counts["flippable_neg_under_delta_max"] += int(flippable)
                _append_jsonl(
                    output / "complete_truths.jsonl",
                    {
                        **key,
                        "route_id": int(route_id),
                        "stations": list(route.stations),
                        "node_indices": list(route.node_indices),
                        "n_stations": 4,
                        "U_W64": float(u_w64[route_id]),
                        "inclusion_gap_A": float(gap.inclusion_gap),
                        "single_rival_margin_A": gap.single_rival_margin,
                        "selected_A": route_id in selected_set,
                        "bin": bin_name,
                        "flippable_under_delta_max": flippable,
                    },
                )
                for competitor_id in gap.forced_out_selected:
                    competitor = routes[int(competitor_id)]
                    _append_jsonl(
                        output / "competing_routes.jsonl",
                        {
                            **key,
                            "truth_route_id": int(route_id),
                            "competitor_route_id": int(competitor_id),
                            "U_W64": float(u_w64[int(competitor_id)]),
                            "n_stations": int(len(competitor.stations)),
                            "truth_consistent": bool(competitor.truth_consistent),
                            "in_forced_out": True,
                        },
                    )
                    counts["n_competing_rows"] += 1
            if route_id in selected_set and len(route.stations) == 4 and not route.truth_consistent:
                counts["f_a"] += 1
                _append_jsonl(
                    output / "a_selected_fakes.jsonl",
                    {
                        **key,
                        "route_id": int(route_id),
                        "stations": list(route.stations),
                        "node_indices": list(route.node_indices),
                        "n_stations": 4,
                        "U_W64": float(u_w64[route_id]),
                        "selected_A": True,
                        "truth_consistent": False,
                    },
                )
        if (event_index + 1) % 200 == 0:
            print(f"{family} events {event_index + 1}/{len(graphs)}", flush=True)
    if accounting is None:
        raise SystemExit("Arm A produced no evaluable events")
    evaluation = accounting.as_dict()
    evaluation.update(
        {
            "arm": "A_sum_raw_w64_edge_logits",
            "utility_contract": CANONICAL_ENERGY_VERSION,
            "n_scored_events": n_events,
            "n_enumerated_routes": n_routes,
        }
    )
    return {"counts": counts, "evaluation": evaluation}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(FAMILY_CORPUS), action="append")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args()
    if args.device != "cpu":
        raise SystemExit("WB81a runs on CPU only")
    device = torch.device("cpu")
    families = tuple(args.family) if args.family else (FAMILY1_DSID_PAIR, FAMILY2_DSID_PAIR)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    refuse_wb81a_path(output_root)

    guards = contract_source_guards()
    if not all(guards.values()):
        raise SystemExit(f"WB81a contract guards failed: {guards}")

    w64_checkpoint = W64_ROOT / "route_aware_transformer_v2.pt"
    refuse_wb81a_path(w64_checkpoint)
    w64_sha = _sha256(w64_checkpoint)
    if w64_sha != W64_SHA:
        raise SystemExit(f"W64 checkpoint sha {w64_sha} != frozen {W64_SHA}")

    summaries = {}
    for family in families:
        output = output_root / family
        output.mkdir(parents=True, exist_ok=True)
        print(f"loading {family}", flush=True)
        _, bundle, raw = _load_family_bundle(family)
        print(f"extracting frozen W64 base-edge logits on {family}", flush=True)
        model_w64, artifact_w64 = load_route_aware_transformer_artifact(w64_checkpoint, device=str(device))
        logits = _w64_base_logits(bundle, model_w64, artifact_w64, device, args.batch_size)
        print(f"evaluating Arm A, identity, and boundary rows for {family}", flush=True)
        materialized = materialize_family(family, bundle.graphs, bundle.adjacent_sets, logits, output)
        evaluation = materialized["evaluation"]
        counts = materialized["counts"]
        checksum = checksum_holdout(
            family,
            {
                "n_scored_events": int(evaluation["n_scored_events"]),
                "complete_truth_chains": int(evaluation["complete_truth_chains"]),
                "t_neg": counts["t_neg"],
                "t_near": counts["t_near"],
                "t_comf": counts["t_comf"],
                "f_a": counts["f_a"],
                "correct_complete_routes": int(evaluation["correct_complete_routes"]),
                "fake_complete_routes": int(evaluation["fake_complete_routes"]),
                "selected_complete_routes": int(evaluation["selected_complete_routes"]),
                "selected_fragment_routes": int(evaluation["selected_fragment_routes"]),
            },
        )
        identity = {
            "passed": True,
            "theta": 0,
            "delta_u": 0.0,
            "u_new_equals_u_w64": True,
            "selected_identical": True,
            "eff_identical": True,
            "fake_identical": True,
            "purity_identical": True,
            "n_events": int(evaluation["n_scored_events"]),
            "stop_on_fail": IDENTITY_FAIL_MESSAGE,
        }
        if not checksum["passed"]:
            raise SystemExit(f"WB81a holdout checksum failed for {family}: {checksum['mismatches']}")
        dataset_manifest = {
            "schema": "faser-dataset-contract-v2",
            "experiment": "wb81a_calibration_foundation_v1",
            "family": family,
            "sources": list(V5A_SOURCE_FAMILIES[family]),
            "corpus_manifest": str(FAMILY_CORPUS[family]),
            "energy": "U_W64_only",
            "wb79_arm_b_used": False,
            "wb80_arm_c_used": False,
            "w64_latent_used": False,
            "development_00350_used": False,
            "final_blind_accessed": False,
            "sealed_test_accessed": False,
            "overlay_seed": (raw.get("synthetic_multitrack") or {}).get("seed"),
        }
        resolved_config = {
            "workbook": "81a",
            "training_authorized": False,
            "utility_contract": CANONICAL_ENERGY_VERSION,
            "metric_version": "route_accounting_v2",
            "solver": "exact_unit_capacity",
            "delta_max": DELTA_MAX,
            "device": "cpu",
            "identity_required": True,
        }
        _write_json(output / "evaluation_arm_a.json", evaluation)
        _write_json(output / "identity.json", identity)
        _write_json(output / "checksums.json", {**checksum, "boundary_counts": counts})
        _write_json(output / "dataset_manifest.json", dataset_manifest)
        _write_json(output / "resolved_config.json", resolved_config)
        _write_json(output / "contract_guards.json", guards)
        _write_json(
            output / "environment.json",
            {
                "hostname": socket.gethostname(),
                "utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version,
                "cuda": torch.cuda.is_available(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "git_sha": _git_sha(),
            },
        )
        _write_json(
            output / "run_metadata.json",
            {
                "workbook": "81a",
                "git_sha": _git_sha(),
                "family": family,
                "training_authorized": False,
                "wb81b_authorized": False,
                "w64_sha256": w64_sha,
                "n_graphs": len(bundle.graphs),
            },
        )
        summaries[family] = {
            "identity_passed": True,
            "checksum_passed": True,
            "evaluation": {
                key: evaluation.get(key)
                for key in (
                    "complete_track_efficiency",
                    "complete_fake_rate",
                    "all_route_purity",
                    "n_scored_events",
                    "complete_truth_chains",
                )
            },
            "boundary_counts": counts,
            "expected": EXPECTED_HOLDOUT_COUNTS[family],
        }
        del bundle, logits, model_w64
        print(json.dumps({"family": family, "summary": summaries[family]}, indent=2, default=str), flush=True)

    _write_json(
        output_root / "summary.json",
        {
            "workbook": "81a",
            "training_authorized": False,
            "wb81b_authorized": False,
            "identity_passed": True,
            "families": summaries,
            "contract_guards": guards,
        },
    )
    print("=== Workbook 81a completed successfully ===", flush=True)


if __name__ == "__main__":
    main()
