#!/usr/bin/env python3
"""Workbook 71: RelativeRoute V4 train->development generalization failure audit.

Read-only mechanism diagnosis.  Loads the frozen Arm 0/1/2 checkpoints, reruns
inference and the frozen unit-capacity solver on the six-source TRAIN overlay
and the already-opened reserved-blind DEVELOPMENT overlay, and decomposes why
the complete-route correction head amplified Mechanism C on development.

Never trains, never retunes, never opens ``00800_00849`` or sealed test.
``continue_to_15d_relative_wls`` stays false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.physical_curriculum import load_synthetic_curriculum_manifest
from evaluation.route_metrics import _unique_truth_by_station, _complete_truth_chains
from scripts.audit_four_station_blind_failure_localization import PAYLOADS, _namespace_map
from scripts.evaluate_relative_route_v4_development import (
    _filter_route_maps,
    _filter_sets,
    _load_arm,
    _score_arm,
    cd_counts,
    frozen_packing_config,
)
from scripts.run_refit_multidof_closure import _json_ready
from training.curriculum_mlp import build_candidate_sets
from training.dustbin_aware_route_margin import PACKING_MARGIN
from training.gauge_consistent_route import (
    GaugeConsistentAuxConfig,
    _aux_losses_for_batch,
    _iter_training_batches,
    pair_gauge_twin_graphs,
    payload_gauge_role,
)
from training.geometry_aware_transformer import (
    ALL_STATION_PAIRS,
    build_transformer_graph_bundle,
)
from training.route_assignment import (
    assign_adjacent_route_sets,
    prepare_route_assignment_context,
)
from training.route_aware_transformer import (
    RouteAwareTrainingConfig,
    _edge_positive_weight,
    _forward_route_batch,
    _make_route_batch,
    _route_loss_components,
    materialize_route_candidate_tables,
    predict_route_aware_scores,
)
from training.route_operating_audit import (
    STATION_PATH,
    audit_event_truth_chains,
    pair_tables_from_sets,
    solver_log_odds,
)
from training.route_reduction_audit import attach_reduction_audit
from training.solver_hard_negative_audit import production_hypotheses
from training.source_diversity_audit import (
    RESERVED_BLIND_SOURCES,
    UNUSED_RESERVE_SOURCES,
    assert_sources_allowed,
)

LOGIT_CLIP = math.log((1.0 - 1.0e-6) / 1.0e-6)  # solver probability-floor clip
WORKBOOK64_SHA256 = "0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236"
ARM1_SHA256 = "e8a6d6c6e26545de91d9ded59c7b4f9b7840de2e9c94b88d56e1857d6aebd3c8"
ARM2_SHA256 = "a52037ead07555ef937a7e02b65adc9526b509e765341e8af31992552f1fdd3a"

TRAIN_MANIFEST = "outputs/mc24_four_station_source_diversity_train_v1/overlay_synthetic_v1/synthetic_corpus_manifest.json"
DEV_MANIFEST = "outputs/mc24_four_station_source_diversity_blind_v1/overlay_synthetic_v1/synthetic_corpus_manifest.json"
W64_ROOT = "outputs/mc24_four_station_source_diversity_v1/checkpoint"
ARM1_CKPT = "outputs/mc24_four_station_relative_route_v4_head_only/absolute_control/checkpoint_last.pt"
ARM2_CKPT = "outputs/mc24_four_station_relative_route_v4_head_only/relative_primary/checkpoint_last.pt"

LOSS_WEIGHTS = {
    "route_consistency": 1.0,
    "one_to_one_competition": 0.25,
    "fake_route_penalty": 0.25,
    "packing_route_competition": 0.07061055340401011,
    "dustbin_aware_route_margin": 0.05,
    "gauge_twin": 1.0,
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_split(manifest_path: str, split: str, max_events_per_payload: int | None = None):
    _, samples, manifest = load_synthetic_curriculum_manifest(
        manifest_path, require_all_splits=False, allowed_splits=(split,)
    )
    if {sample.split for sample in samples} != {split}:
        raise SystemExit(f"expected only '{split}' samples in {manifest_path}")
    constituents = {source for sample in samples for source in sample.source_ids}
    if split == "validation":
        assert_sources_allowed(constituents, allow_reserved_blind=True)
        if constituents != set(RESERVED_BLIND_SOURCES):
            raise SystemExit("development split must be the reserved blind pair")
        if constituents & set(UNUSED_RESERVE_SOURCES):
            raise SystemExit("unused final-blind sources were loaded")
    if split == "train" and constituents & (set(RESERVED_BLIND_SOURCES) | set(UNUSED_RESERVE_SOURCES)):
        raise SystemExit("train split leaked reserved/final-blind sources")
    candidate_sets = build_candidate_sets(
        samples, ALL_STATION_PAIRS, chi2_gate=None, feature_set="residual_v1", q_over_p_mode=0
    )
    if max_events_per_payload:
        # Keep every station-pair candidate set of the first N events per payload,
        # so gauge twins (distinct payloads) are both retained and the rebuilt
        # bundle keeps score_owner -> adjacent_sets index alignment.
        seen: dict[str, list] = defaultdict(list)
        for cs in candidate_sets:
            pid = str(cs.sample.payload_id)
            key = (int(cs.event.run_id), int(cs.event.event_id))
            if key not in seen[pid]:
                seen[pid].append(key)
        keep = {pid: set(keys[:max_events_per_payload]) for pid, keys in seen.items()}
        candidate_sets = [
            cs for cs in candidate_sets
            if (int(cs.event.run_id), int(cs.event.event_id)) in keep[str(cs.sample.payload_id)]
        ]
    bundle = build_transformer_graph_bundle(candidate_sets, context_mode="full_event")
    tables = materialize_route_candidate_tables(bundle.graphs)
    namespace = _namespace_map(samples)
    return samples, bundle, tables, namespace


def _route_share_classes(graph, table) -> np.ndarray:
    """Max number of endpoints sharing one valid truth particle, per route."""
    tids = np.asarray(graph.event.truth_particle_id, dtype=np.int64)[table.node_indices]
    valid = tids >= 0
    same = (tids[:, None, :] == tids[:, :, None]) & valid[:, None, :] & valid[:, :, None]
    counts = same.sum(axis=2)
    max_share = np.where(valid, counts, 0).max(axis=1)
    return max_share.astype(np.int64)


def _share_class_name(label: bool, max_share: int) -> str:
    if label:
        return "A_truth"
    return {3: "B_fake_share3", 2: "C_fake_share2", 1: "D_fake_share1", 0: "E_fake_fully_unrelated"}[int(max_share)]


def _label_key_audit(bundle, tables, prediction_route_sets, *, split: str, max_events: int):
    """Recompute route labels from event truth; cross-check score-map keys."""
    label_mismatch = 0
    fake_mismatch = 0
    truth_missing = 0
    truth_wrong_label = 0
    routes_checked = 0
    map_key_mismatch = 0
    events_done = 0
    score_map_by_event = {
        (str(s.sample.source_id), str(s.sample.payload_id), int(s.event.run_id), int(s.event.event_id)): s
        for s in prediction_route_sets
    }
    for graph in bundle.graphs:
        if events_done >= max_events:
            break
        table = tables[id(graph)]
        if not table.size:
            continue
        event = graph.event
        tids = np.asarray(event.truth_particle_id, dtype=np.int64)[table.node_indices]
        valid = tids >= 0
        recomputed_label = valid.all(axis=1) & (tids == tids[:, :1]).all(axis=1)
        recomputed_fake = (~valid).any(axis=1)
        label_mismatch += int(np.count_nonzero(recomputed_label != table.labels))
        fake_mismatch += int(np.count_nonzero(recomputed_fake != table.fake_endpoint))
        routes_checked += int(table.size)
        _, unique_by_index = _unique_truth_by_station(event, STATION_PATH)
        complete = _complete_truth_chains(unique_by_index, STATION_PATH)
        table_label_by_key = {tuple(int(v) for v in row): bool(lab) for row, lab in zip(table.node_indices, table.labels)}
        for truth_id, endpoints in complete.items():
            key = tuple(int(index) for _, index in endpoints)
            if key not in table_label_by_key:
                truth_missing += 1
            elif not table_label_by_key[key]:
                truth_wrong_label += 1
        event_key = (str(graph.sample.source_id), str(graph.sample.payload_id), int(event.run_id), int(event.event_id))
        score_set = score_map_by_event.get(event_key)
        if score_set is not None:
            map_keys = {tuple(int(v) for v in row) for row in score_set.endpoint_indices}
            if map_keys != set(table_label_by_key):
                map_key_mismatch += 1
        events_done += 1
    ok = label_mismatch == 0 and fake_mismatch == 0 and truth_missing == 0 and truth_wrong_label == 0 and map_key_mismatch == 0
    return {
        "split": split,
        "events_audited": int(events_done),
        "routes_checked": int(routes_checked),
        "label_mismatch_count": int(label_mismatch),
        "fake_endpoint_mismatch_count": int(fake_mismatch),
        "complete_truth_chain_missing_from_candidates": int(truth_missing),
        "complete_truth_chain_wrong_label": int(truth_wrong_label),
        "score_map_key_mismatch_events": int(map_key_mismatch),
        "ok": bool(ok),
    }


def _predict_route_details(model, bundle, tables, node_standardizer, edge_standardizer, device, batch_size):
    model.eval()
    details: dict[int, dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, len(bundle.graphs), batch_size):
            graphs = bundle.graphs[start : start + batch_size]
            batch = _make_route_batch(graphs, node_standardizer, edge_standardizer, device, tables)
            output = _forward_route_batch(model, batch)
            edge_logits = output.edge_logits.detach().cpu().numpy().astype(np.float64)
            route_logits = output.route_logits.detach().cpu().numpy().astype(np.float64)
            delta = output.delta_route_logits
            delta_arr = None if delta is None else delta.detach().cpu().numpy().astype(np.float64)
            eoff = 0
            per_graph_edges = {}
            for g in graphs:
                w = int(g.score_labels.size)
                per_graph_edges[id(g)] = edge_logits[eoff : eoff + w]
                eoff += w
            roff = 0
            for g in graphs:
                table = tables[id(g)]
                stop = roff + table.size
                details[id(g)] = {
                    "edge_logits": per_graph_edges[id(g)],
                    "route_logits": route_logits[roff:stop],
                    "delta": None if delta_arr is None else delta_arr[roff:stop],
                }
                roff = stop
    return details


def _produce_truth_rows(bundle, calibrated_scores, route_maps, config):
    rows_all: list[dict[str, object]] = []
    for payload_id in PAYLOADS:
        sets, values = _filter_sets(bundle.adjacent_sets, calibrated_scores, payload_id)
        maps = _filter_route_maps(route_maps, payload_id)
        context = prepare_route_assignment_context(sets, values, 15, config.station_path)
        assigned = assign_adjacent_route_sets(
            sets, values, config, 15, context=context, complete_route_scores_by_event=maps
        )
        tables = pair_tables_from_sets(sets, values, values)
        assigned_by_key = {item.key: item for item in assigned}
        pending = []
        event_truth_counts: Counter = Counter()
        for group in context.groups:
            event = group.event
            item = assigned_by_key[group.key]
            pair_tables = tables[group.key]
            _, unique_by_index = _unique_truth_by_station(event, STATION_PATH)
            query_map = None if maps is None else maps[group.key]
            hypotheses = production_hypotheses(event, group.station_matrices, config, complete_route_scores=query_map)
            truth_rows = audit_event_truth_chains(
                event, group.station_matrices, pair_tables, config, item.result, complete_route_scores=query_map
            )
            event_truth_counts[group.key] += len(truth_rows)
            for row in truth_rows:
                row["sample_id"] = group.key[0]
                row["payload_id"] = group.key[1]
                row["run_id"] = group.key[2]
                row["event_id"] = group.key[3]
                pending.append((group.key, row, event, unique_by_index, pair_tables, hypotheses, item.result.routes))
        for event_key, row, event, unique_by_index, pair_tables, hypotheses, selected_routes in pending:
            attached = attach_reduction_audit(
                row,
                event=event,
                unique_by_index=unique_by_index,
                pair_tables=pair_tables,
                hypotheses=hypotheses,
                selected_routes=selected_routes,
                n_complete_truth_in_event=int(event_truth_counts[event_key]),
                margin=float(PACKING_MARGIN),
            )
            rows_all.append(attached)
    return rows_all


def _gradient_audit(model, bundle, tables, node_standardizer, edge_standardizer, device, *, n_batches: int, seed: int):
    graphs = list(bundle.graphs)
    pairs, leftovers = pair_gauge_twin_graphs(graphs)
    edge_pos_weight, _, _ = _edge_positive_weight(graphs, 1.0, 30.0)
    training_config = RouteAwareTrainingConfig(
        batch_size=32, learning_rate=2.0e-4, weight_decay=1.0e-4, seed=seed, device="cuda",
        focal_gamma=1.5, hard_negative_weight=2.0, edge_loss_weight=1.0,
        route_consistency_weight=1.0, one_to_one_competition_weight=0.25, fake_route_penalty_weight=0.25,
    )
    aux_config = GaugeConsistentAuxConfig(
        packing_margin=1.0, pair_threshold=0.001, unmatched_penalty=-1.0,
        enable_dustbin_aware_route_margin=True, route_competition_reduction="max",
    )
    rng = np.random.default_rng(seed + 30)  # epoch-30 stream
    torch.manual_seed(seed)
    model.train()
    accum: dict[str, dict[str, list]] = {name: {"truth": [], "fake": []} for name in LOSS_WEIGHTS}
    n_done = 0
    for batch_graphs in _iter_training_batches(pairs, leftovers, 32, rng):
        if n_done >= n_batches:
            break
        batch = _make_route_batch(batch_graphs, node_standardizer, edge_standardizer, device, tables)
        output = _forward_route_batch(model, batch)
        body = _route_loss_components(output, batch, training_config, edge_pos_weight)
        aux = _aux_losses_for_batch(
            batch_graphs, output.edge_logits, pairs, aux_config,
            route_logits=output.route_logits, route_tables=tables,
        )
        delta = output.delta_route_logits
        if delta is None or not delta.numel():
            continue
        labels = (batch.route_labels.detach().cpu().numpy() > 0.5)
        losses = {
            "route_consistency": body["route_consistency"],
            "one_to_one_competition": body["one_to_one_competition"],
            "fake_route_penalty": body["fake_route_penalty"],
            "packing_route_competition": aux["packing_route_competition"],
            "dustbin_aware_route_margin": aux["dustbin_aware_route_margin"],
            "gauge_twin": aux["gauge_twin"],
        }
        for name, loss in losses.items():
            grad = torch.autograd.grad(loss, delta, retain_graph=True, allow_unused=True)[0]
            if grad is None:
                continue
            g = grad.detach().cpu().numpy().astype(np.float64)
            accum[name]["truth"].append(g[labels])
            accum[name]["fake"].append(g[~labels])
        n_done += 1
    summary: dict[str, Any] = {"n_batches": n_done, "losses": {}}
    for name, parts in accum.items():
        weight = LOSS_WEIGHTS[name]
        entry: dict[str, Any] = {"weight": weight}
        for cls in ("truth", "fake"):
            g = np.concatenate(parts[cls]) if parts[cls] else np.empty(0)
            nonzero = g[np.abs(g) > 0]
            entry[cls] = {
                "n_routes": int(g.size),
                "n_nonzero_grad": int(nonzero.size),
                "mean_grad": None if not g.size else float(g.mean()),
                "median_grad": None if not g.size else float(np.median(g)),
                "frac_positive": None if not g.size else float(np.mean(g > 0)),
                "frac_negative": None if not g.size else float(np.mean(g < 0)),
                "abs_grad_q50": None if not nonzero.size else float(np.quantile(np.abs(nonzero), 0.50)),
                "abs_grad_q90": None if not nonzero.size else float(np.quantile(np.abs(nonzero), 0.90)),
                "abs_grad_q99": None if not nonzero.size else float(np.quantile(np.abs(nonzero), 0.99)),
                "weighted_net_force": None if not g.size else float(weight * g.sum()),
                "weighted_mean_grad": None if not g.size else float(weight * g.mean()),
            }
        summary["losses"][name] = entry
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/mc24_four_station_relative_route_v4_generalization_audit_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-batches", type=int, default=24)
    parser.add_argument("--max-events-per-payload", type=int, default=None)
    parser.add_argument("--skip-solver", action="store_true")
    parser.add_argument("--skip-gradient", action="store_true")
    parser.add_argument("--label-audit-events", type=int, default=3)
    args = parser.parse_args()

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = args.device

    print("=== Workbook 71 generalization audit ===", flush=True)
    print(f"hostname {socket.gethostname()}", flush=True)
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    git_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True)
    print(f"git_commit {git_commit} dirty={bool(git_status.strip())}", flush=True)

    config = frozen_packing_config()

    print("Loading TRAIN split ...", flush=True)
    train_samples, train_bundle, train_tables, train_namespace = _load_split(
        TRAIN_MANIFEST, "train", args.max_events_per_payload
    )
    print(f"  train graphs={len(train_bundle.graphs)} routes={sum(t.size for t in train_tables.values())}", flush=True)
    print("Loading DEVELOPMENT split ...", flush=True)
    dev_samples, dev_bundle, dev_tables, dev_namespace = _load_split(
        DEV_MANIFEST, "validation", args.max_events_per_payload
    )
    print(f"  dev graphs={len(dev_bundle.graphs)} routes={sum(t.size for t in dev_tables.values())}", flush=True)

    arms = {}
    for arm in ("arm0", "arm1", "arm2"):
        arms[arm] = _load_arm(
            arm, w64_root=Path(W64_ROOT), arm1_checkpoint=Path(ARM1_CKPT), arm2_checkpoint=Path(ARM2_CKPT), device=device
        )
        print(f"  loaded {arm} sha={arms[arm]['checkpoint_sha256'][:12]} inject={arms[arm]['inject_complete_route_scores']}", flush=True)

    splits = {"train": (train_bundle, train_tables, train_namespace), "development": (dev_bundle, dev_tables, dev_namespace)}
    run: dict[str, dict[str, Any]] = {}
    for split_name, (bundle, tables, _) in splits.items():
        run[split_name] = {"bundle": bundle, "tables": tables}
        for arm in ("arm0", "arm1", "arm2"):
            loaded = arms[arm]
            raw, calibrated, route_maps = _score_arm(loaded, bundle, device, args.batch_size)
            details = _predict_route_details(
                loaded["model"], bundle, tables,
                loaded["artifact"].node_standardizer, loaded["artifact"].edge_standardizer,
                device, args.batch_size,
            )
            run[split_name][arm] = {"raw": raw, "calibrated": calibrated, "route_maps": route_maps, "details": details}
            print(f"  scored {split_name}/{arm}", flush=True)

    # Label / key semantics audit (arm1 route score sets for the key cross-check).
    label_audit = {}
    for split_name, (bundle, tables, namespace) in splits.items():
        prediction = predict_route_aware_scores(
            arms["arm1"]["model"], bundle,
            arms["arm1"]["artifact"].node_standardizer, arms["arm1"]["artifact"].edge_standardizer,
            device=device, batch_size=args.batch_size, route_tables=tables,
        )
        label_audit[split_name] = _label_key_audit(
            bundle, tables, prediction.route_score_sets, split=split_name, max_events=int(args.label_audit_events)
        )
        print(f"  label audit {split_name}: ok={label_audit[split_name]['ok']}", flush=True)

    # Route class balance.
    class_balance: dict[str, Any] = {}
    for split_name, (bundle, tables, _) in splits.items():
        total = n_truth = n_fake_endpoint = n_hard = 0
        per_event_counts = []
        share_counts: Counter = Counter()
        for g in bundle.graphs:
            table = tables[id(g)]
            total += table.size
            n_truth += int(np.count_nonzero(table.labels))
            n_fake_endpoint += int(np.count_nonzero(table.fake_endpoint))
            n_hard += int(np.count_nonzero(table.hard_negative))
            per_event_counts.append(int(table.size))
            shares = _route_share_classes(g, table)
            for lab, sh in zip(table.labels, shares):
                share_counts[_share_class_name(bool(lab), int(sh))] += 1
        per_event = np.asarray(per_event_counts, dtype=np.float64)
        class_balance[split_name] = {
            "complete_route_candidates": int(total),
            "truth_routes": int(n_truth),
            "fake_routes": int(total - n_truth),
            "truth_fraction": float(n_truth / max(total, 1)),
            "truth_to_fake_ratio": float(n_truth / max(total - n_truth, 1)),
            "fake_endpoint_routes": int(n_fake_endpoint),
            "hard_negative_routes": int(n_hard),
            "events": int(len(bundle.graphs)),
            "routes_per_event": {
                "mean": float(per_event.mean()), "median": float(np.median(per_event)),
                "min": int(per_event.min()), "max": int(per_event.max()),
                "q95": float(np.quantile(per_event, 0.95)),
            },
            "share_class_counts": dict(share_counts),
            "bce_route_positive_weight_global": float(min((total - n_truth) / max(n_truth, 1), 80.0)),
        }
    _write_json(output / "route_class_balance.json", class_balance)

    # Solver C/D rows.
    solver_rows: dict[str, dict[str, list]] = {}
    if not args.skip_solver:
        for split_name in splits:
            solver_rows[split_name] = {}
            for arm in ("arm0", "arm1", "arm2"):
                rows = _produce_truth_rows(
                    run[split_name]["bundle"], run[split_name][arm]["calibrated"], run[split_name][arm]["route_maps"], config
                )
                solver_rows[split_name][arm] = rows
                cd = cd_counts(rows)
                print(f"  solver {split_name}/{arm}: C={cd['C']} D={cd['D']} selected={cd['selected']}", flush=True)

    # Join route details with solver rows -> route_rows.jsonl
    route_rows_path = output / "route_rows.jsonl"
    n_written = 0
    with route_rows_path.open("w", encoding="utf-8") as handle:
        for split_name, (bundle, tables, namespace) in splits.items():
            truth_index: dict[str, dict[tuple, dict]] = {}
            if not args.skip_solver:
                for arm in ("arm0", "arm1", "arm2"):
                    truth_index[arm] = {}
                    for row in solver_rows[split_name][arm]:
                        key = (
                            str(row["payload_id"]), int(row["run_id"]), int(row["event_id"]),
                            tuple(int(e["index"]) for e in row["endpoints"]),
                        )
                        truth_index[arm][key] = row
            for g in bundle.graphs:
                table = tables[id(g)]
                if not table.size:
                    continue
                event = g.event
                edge_logits = run[split_name]["arm0"]["details"][id(g)]["edge_logits"]
                d1 = run[split_name]["arm1"]["details"][id(g)]["delta"]
                d2 = run[split_name]["arm2"]["details"][id(g)]["delta"]
                shares = _route_share_classes(g, table)
                family = str(g.sample.payload_id)
                role = payload_gauge_role(family)
                source_id = namespace.get(int(event.run_id))
                for i in range(table.size):
                    nodes = tuple(int(v) for v in table.node_indices[i])
                    e01 = float(edge_logits[int(table.score_edge_indices[i][0])])
                    e12 = float(edge_logits[int(table.score_edge_indices[i][1])])
                    e23 = float(edge_logits[int(table.score_edge_indices[i][2])])
                    l_edge = e01 + e12 + e23
                    label = bool(table.labels[i])
                    rec: dict[str, Any] = {
                        "split": split_name,
                        "source_id": source_id,
                        "payload_id": family,
                        "payload_family": family[: -len("_plus_common")] if family.endswith("_plus_common") else family,
                        "gauge_role": None if role is None else role[1],
                        "run_id": int(event.run_id),
                        "event_id": int(event.event_id),
                        "route_key": nodes,
                        "truth_share_class": _share_class_name(label, int(shares[i])),
                        "label": label,
                        "fake_endpoint": bool(table.fake_endpoint[i]),
                        "hard_negative": bool(table.hard_negative[i]),
                        "L01": e01, "L12": e12, "L23": e23, "L_edge": l_edge,
                        "p23": float(1.0 / (1.0 + math.exp(-e23))),
                        "delta_arm1": None if d1 is None else float(d1[i]),
                        "delta_arm2": None if d2 is None else float(d2[i]),
                    }
                    u0 = (
                        solver_log_odds(float(1.0 / (1.0 + math.exp(-e01))))
                        + solver_log_odds(float(1.0 / (1.0 + math.exp(-e12))))
                        + solver_log_odds(float(1.0 / (1.0 + math.exp(-e23))))
                        - 4.0
                    )
                    rec["U_arm0"] = u0
                    for arm, d in (("arm1", d1), ("arm2", d2)):
                        rec[f"U_{arm}"] = u0 if d is None else float(max(min(l_edge + float(d[i]), LOGIT_CLIP), -LOGIT_CLIP) - 4.0)
                    if label and not args.skip_solver:
                        tkey = (family, int(event.run_id), int(event.event_id), nodes)
                        for arm in ("arm0", "arm1", "arm2"):
                            trow = truth_index[arm].get(tkey)
                            if trow is not None:
                                rec[f"decision_class_{arm}"] = str(trow["loss_stage"])
                                rec[f"selected_{arm}"] = bool(trow["selected"])
                                rec[f"U_complete_{arm}"] = trow.get("complete_truth_route_utility")
                                rec[f"U_best_fragment_{arm}"] = trow.get("u_best_solver_fragment")
                                rec[f"margin_{arm}"] = trow.get("production_margin")
                        trow0 = truth_index["arm0"].get(tkey)
                        if trow0 is not None:
                            rec["truth_id"] = trow0.get("truth_id")
                            rec["origin_signature"] = trow0.get("origin_signature")
                            edges = trow0.get("edges") or []
                            if len(edges) == 3:
                                rec["chi2_23"] = edges[2].get("chi2")
                                rec["L23_calibrated"] = edges[2].get("calibrated_logit")
                    handle.write(json.dumps(_json_ready(rec), allow_nan=False) + "\n")
                    n_written += 1
    print(f"  wrote {n_written} route rows", flush=True)

    for split_name in splits:
        summary: dict[str, Any] = {"split": split_name}
        if not args.skip_solver:
            summary["cd"] = {arm: cd_counts(solver_rows[split_name][arm]) for arm in ("arm0", "arm1", "arm2")}
        _write_json(output / f"{split_name}_summary.json", summary)

    if not args.skip_gradient:
        grad_summary = {}
        for arm in ("arm1", "arm2"):
            grad_summary[arm] = _gradient_audit(
                arms[arm]["model"], train_bundle, train_tables,
                arms[arm]["artifact"].node_standardizer, arms[arm]["artifact"].edge_standardizer,
                device, n_batches=int(args.gradient_batches), seed=20260822,
            )
            print(f"  gradient audit {arm} done", flush=True)
        _write_json(output / "loss_gradient_summary.json", grad_summary)

    _write_json(
        output / "audit_contract.json",
        {
            "workbook": 71,
            "git_commit": git_commit,
            "git_dirty": bool(git_status.strip()),
            "hostname": socket.gethostname(),
            "checkpoints": {"arm0": WORKBOOK64_SHA256, "arm1": ARM1_SHA256, "arm2": ARM2_SHA256},
            "continue_to_15d_relative_wls": False,
            "training_authorized": False,
            "final_blind_eval_authorized": False,
            "development_accessed": True,
            "new_final_blind_content_accessed": False,
            "sealed_test_accessed": False,
            "label_audit": label_audit,
        },
    )
    print("=== audit done ===", flush=True)


if __name__ == "__main__":
    main()
