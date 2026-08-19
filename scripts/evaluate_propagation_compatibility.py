#!/usr/bin/env python3
"""Fit and evaluate train-frozen propagation-compatibility models.

Fits every model of ``alignment.propagation_compatibility`` on TRAIN physical
truth edges only (unique physical edges, no overlay duplication), then
evaluates truth/fake separation on the multi-track synthetic overlays of both
splits — validation exactly once as a frozen transfer.  Reports per station
pair and payload: ROC/AUC of chi2 and of each model score, tail fractions at
the frozen train 99.5% truth-retention veto, the known mis-associated edge's
score decomposition, per-source dependence, and anchor-vs-reference
misalignment dependence.  Read-only; no candidate set, V2 score, or WLS
covariance is modified.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from alignment import propagation_compatibility as pc

STATE_LABELS = ("x_mm", "y_mm", "tx", "ty")
COV_LABELS = (
    "xx_mm2", "xy_mm2", "xtx_mm", "xty_mm",
    "yy_mm2", "ytx_mm", "yty_mm", "txtx", "txty", "tyty",
)
# Known mis-associated 0->1 edge of mc24_100047_00150_00199: synthetic-event
# identities of its six overlay route copies (run 996000), from the frozen
# validation backbone anchor table.
BAD_EDGE_IDENTITIES = {
    (996000, 5, 4, 10, "0->1"),
    (996000, 165, 3, 0, "0->1"),
    (996000, 297, 3, 6, "0->1"),
    (996000, 397, 3, 4, "0->1"),
    (996000, 670, 2, 3, "0->1"),
    (996000, 763, 10, 7, "0->1"),
}


def _load_edges(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = {
                "split": row["split"],
                "sample_kind": row["sample_kind"],
                "origin_id": row["origin_id"],
                "payload": row["payload"],
                "station_pair": row["station_pair"],
                "is_truth": row["is_truth"] == "1",
                "route_selected": row["route_selected"] == "1",
                "chi2": float(row["chi2"]),
                "residual": np.asarray([float(row[f"residual_{l}"]) for l in STATE_LABELS]),
                "pull": np.asarray([float(row[f"pull_{l}"]) for l in STATE_LABELS]),
                "source_state": np.asarray(
                    [float(row[f"source_state_{l}"]) for l in STATE_LABELS]
                ),
                "identity": (
                    int(row["run_id"]), int(row["event_id"]),
                    int(row["source_tracklet_id"]), int(row["target_tracklet_id"]),
                    row["station_pair"],
                ),
            }
            flat = [float(row[f"combined_cov_{l}"]) for l in COV_LABELS]
            cov = np.zeros((4, 4))
            index = 0
            for i in range(4):
                for j in range(i, 4):
                    cov[i, j] = cov[j, i] = flat[index]
                    index += 1
            parsed["covariance"] = cov
            rows.append(parsed)
    return rows


def _auc(scores_truth: np.ndarray, scores_fake: np.ndarray) -> float | None:
    """P(score_fake > score_truth); higher score = less compatible."""
    if scores_truth.size == 0 or scores_fake.size == 0:
        return None
    all_scores = np.concatenate([scores_truth, scores_fake])
    order = np.argsort(all_scores, kind="mergesort")
    ranks = np.empty(all_scores.size)
    ranks[order] = np.arange(1, all_scores.size + 1)
    # average ties
    sorted_scores = all_scores[order]
    unique, inverse, counts = np.unique(sorted_scores, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    mean_ranks = cumulative - (counts - 1) / 2.0
    ranks = mean_ranks[inverse][np.argsort(order, kind="mergesort")]
    rank_sum_truth = float(np.sum(ranks[: scores_truth.size]))
    n_t, n_f = scores_truth.size, scores_fake.size
    return float(1.0 - (rank_sum_truth - n_t * (n_t + 1) / 2.0) / (n_t * n_f))


def _scores(model: str, pair_models: pc.PairModels, edges: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [pair_models.score(model, edge["residual"], edge["covariance"]) for edge in edges]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-edges", required=True)
    parser.add_argument("--validation-edges", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    train_edges = _load_edges(Path(args.train_edges).expanduser().resolve())
    validation_edges = _load_edges(Path(args.validation_edges).expanduser().resolve())

    # ---- Fit on train PHYSICAL truth edges only (unique physical edges).
    fit_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in train_edges:
        if edge["sample_kind"] == "physical" and edge["is_truth"]:
            fit_rows[edge["station_pair"]].append(edge)
    models: dict[str, pc.PairModels] = {}
    for pair, edges in sorted(fit_rows.items()):
        models[pair] = pc.fit_pair_models(
            pair,
            np.asarray([edge["residual"] for edge in edges]),
            np.asarray([edge["covariance"] for edge in edges]),
        )
    pc.save(models, output / "compatibility_models.json")

    # ---- Mechanism decomposition of the known bad edge (validation overlay).
    bad_edge_rows = [
        edge for edge in validation_edges if edge["identity"] in BAD_EDGE_IDENTITIES
    ]
    bad_decomposition: dict[str, Any] = {"copies_found": len(bad_edge_rows), "copies": []}
    pair_models = models.get("0->1")
    train_truth_01 = [
        edge
        for edge in train_edges
        if edge["sample_kind"] == "physical" and edge["is_truth"]
        and edge["station_pair"] == "0->1"
    ]
    truth_chi2_01 = np.asarray([edge["chi2"] for edge in train_truth_01])
    for edge in bad_edge_rows[:1]:
        residual, covariance = edge["residual"], edge["covariance"]
        inverse = np.linalg.inv(covariance)
        chi2 = float(residual @ inverse @ residual)
        per_component = {
            label: {
                "residual": float(residual[i]),
                "combined_sigma": float(np.sqrt(covariance[i, i])),
                "marginal_pull": float(residual[i] / np.sqrt(covariance[i, i])),
                "chi2_share": float(residual[i] * (inverse @ residual)[i]),
            }
            for i, label in enumerate(STATE_LABELS)
        }
        scores = (
            {model: pair_models.score(model, residual, covariance) for model in pc.MODELS}
            if pair_models is not None
            else {}
        )
        bad_decomposition["copies"].append(
            {
                "identity": edge["identity"],
                "payload": edge["payload"],
                "route_selected": edge["route_selected"],
                "chi2": chi2,
                "train_truth_chi2_quantile": float(np.mean(truth_chi2_01 <= chi2)),
                "per_component": per_component,
                "model_scores": scores,
                "veto_thresholds": (
                    dict(pair_models.veto_thresholds) if pair_models is not None else {}
                ),
                "vetoed_by": (
                    [
                        model
                        for model, score in scores.items()
                        if score > pair_models.veto_thresholds[model]
                    ]
                    if pair_models is not None
                    else []
                ),
            }
        )

    # ---- Separation tables per pair/payload/split on synthetic overlays.
    separation_rows: list[dict[str, Any]] = []
    for split, edges in (("train", train_edges), ("validation", validation_edges)):
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for edge in edges:
            if edge["sample_kind"] == "synthetic_overlay":
                grouped[(edge["station_pair"], edge["payload"])].append(edge)
        for (pair, payload), group in sorted(grouped.items()):
            truth = [edge for edge in group if edge["is_truth"]]
            fake = [edge for edge in group if not edge["is_truth"]]
            pair_models = models.get(pair)
            row: dict[str, Any] = {
                "split": split,
                "station_pair": pair,
                "payload": payload,
                "truth_edges": len(truth),
                "fake_edges": len(fake),
                "truth_chi2_median": float(np.median([e["chi2"] for e in truth])),
                "truth_chi2_q99": float(np.quantile([e["chi2"] for e in truth], 0.99)),
                "fake_chi2_median": float(np.median([e["chi2"] for e in fake])),
                "fake_chi2_q01": float(np.quantile([e["chi2"] for e in fake], 0.01)),
                "auc_chi2": _auc(
                    np.asarray([e["chi2"] for e in truth]),
                    np.asarray([e["chi2"] for e in fake]),
                ),
            }
            if pair_models is not None:
                truth_scores = {
                    model: _scores(model, pair_models, truth) for model in pc.MODELS
                }
                fake_scores = {model: _scores(model, pair_models, fake) for model in pc.MODELS}
                for model in pc.MODELS:
                    row[f"auc_{model}"] = _auc(truth_scores[model], fake_scores[model])
                    threshold = pair_models.veto_thresholds[model]
                    row[f"truth_retention_{model}"] = float(
                        np.mean(truth_scores[model] <= threshold)
                    )
                    row[f"fake_rejection_{model}"] = float(
                        np.mean(fake_scores[model] > threshold)
                    )
            separation_rows.append(row)

    # ---- Per-source dependence (synthetic overlay, anchor payload).
    source_rows: list[dict[str, Any]] = []
    for split, edges in (("train", train_edges), ("validation", validation_edges)):
        per_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in edges:
            if edge["sample_kind"] == "synthetic_overlay" and edge["payload"].endswith("anchor"):
                # origin run id encodes the source namespace
                per_source[str(edge["identity"][0])].append(edge)
        for namespace, group in sorted(per_source.items()):
            truth = [e for e in group if e["is_truth"]]
            fake = [e for e in group if not e["is_truth"]]
            if not truth or not fake:
                continue
            source_rows.append(
                {
                    "split": split,
                    "origin_run_namespace": namespace,
                    "truth_edges": len(truth),
                    "fake_edges": len(fake),
                    "truth_chi2_median": float(np.median([e["chi2"] for e in truth])),
                    "truth_chi2_q99": float(np.quantile([e["chi2"] for e in truth], 0.99)),
                    "fake_chi2_median": float(np.median([e["chi2"] for e in fake])),
                    "auc_chi2": _auc(
                        np.asarray([e["chi2"] for e in truth]),
                        np.asarray([e["chi2"] for e in fake]),
                    ),
                }
            )

    def _write_csv(name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with (output / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    _write_csv("separation_by_pair_payload.csv", separation_rows)
    _write_csv("separation_by_source.csv", source_rows)
    (output / "bad_edge_decomposition.json").write_text(
        json.dumps(bad_decomposition, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "models_fitted_on": "train physical truth edges",
        "pairs": sorted(models),
        "bad_edge_copies_found": bad_decomposition["copies_found"],
        "test_data_accessed": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), **summary}, indent=2, default=str))


if __name__ == "__main__":
    main()
