"""Workbook 73: extract per-route representation levels R0-R7 / R_phys.

Diagnostic-only extraction (no production training).  For each complete-route
candidate in a Workbook-72 source-pure family corpus we dump, keyed by route:

  R0  raw physical pair-relative edge observables (3 edges x 11)   [residual_v1]
  R1  frozen W64 standardized edge features (3 x 11)
  R2  frozen W64 edge latent (3 x 128) + production edge logits L01/L12/L23 (3)
  R3  raw node input features (4 nodes x 17), absolute + quality split
  R4  frozen W64 node latent states s0..s3 (4 x 128)
  R5  absolute route head input tensor (route_encoder input, 1075)
  R6  route encoder hidden state (route_score input, 128)
  R7  raw_delta, bounded_delta, L_edge, L_corrected  (scalars)
  R_phys = [R0 (33), L01/L12/L23 (3)]  physical pair-relative route (36)

R0/R1/R2/R3/R4/R_phys are head-independent (frozen W64 backbone + physical
observables).  R5/R6/R7 come from a single fixed reference absolute head
(default: Workbook-72 holdout_family1 primary, the bounded head whose fold-1
failures define the catastrophic cohort).  Truth/fake share-class follows
Workbook-71 semantics (A_truth / B_share3 / C_share2 / D_share1 / E_unrelated).

Storage is two-tier to keep the full pooled distribution for the low-dim
physical features while keeping high-dim latents for an all-truth + subsampled
fake set:
  {family}_lowdim.npz   all routes: keys, share_class, R0,R1,R2logits,R3,R7,R_phys
  {family}_highdim.npz  subset:     route_uid, R2latent,R4,R5,R6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.audit_relative_route_v4_generalization import (
    _load_split,
    _route_share_classes,
)
from training.geometry_aware_transformer import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES
from training.route_aware_transformer import (
    _make_route_batch,
    load_relative_route_v4_head_only_artifact,
)

FAMILY_CORPUS = {
    "family1": "outputs/mc24_four_station_relative_route_v5a_source_transfer_v1/corpora/family1/overlay_synthetic_v1/synthetic_corpus_manifest.json",
    "family2": "outputs/mc24_four_station_relative_route_v5a_source_transfer_v1/corpora/family2/overlay_synthetic_v1/synthetic_corpus_manifest.json",
}
FAMILY_HEAD = {
    # Fixed reference absolute head per family for the head-pipeline levels
    # R5/R6/R7.  fold_k primary is the bounded (B=4) absolute head whose
    # held-out family is the one being evaluated in that fold.
    "family1": "outputs/mc24_four_station_relative_route_v5a_source_transfer_v1/training/holdout_family1/primary/relative_route_v4_head_only.pt",
    "family2": "outputs/mc24_four_station_relative_route_v5a_source_transfer_v1/training/holdout_family2/primary/relative_route_v4_head_only.pt",
}
SHARE_CLASS_NAMES = ("A_truth", "B_fake_share3", "C_fake_share2", "D_fake_share1", "E_fake_fully_unrelated")

# Low-dim per-route scalar/vector block widths (for documentation / manifest).
EDGE_W = len(EDGE_FEATURE_NAMES)          # 11
NODE_W = len(NODE_FEATURE_NAMES)          # 17
R0_W = 3 * EDGE_W                          # 33
R1_W = 3 * EDGE_W                          # 33
R2LOG_W = 3                                # L01,L12,L23
R3_W = 4 * NODE_W                          # 68
RPHYS_W = 3 * EDGE_W + 3                   # 36
R7_W = 4                                   # raw_delta, bounded_delta, L_edge, L_corrected
R2LAT_W = 3 * 128                          # 384
R4_W = 4 * 128                             # 512
R5_W = 1075
R6_W = 128

# Node-feature index splits (absolute/global vs quality/covariance/hit-pattern).
NODE_ABS_IDX = tuple(range(0, 5))          # x_mm,y_mm,tx,ty,z_mm
NODE_QUAL_IDX = tuple(range(5, 17))        # log_sigma_*, chi2, n_hit, hit bits


def _share_class_index(label: bool, max_share: int) -> int:
    if label:
        return 0  # A_truth
    return {3: 1, 2: 2, 1: 3, 0: 4}[int(max_share)]


def extract_family(
    family: str,
    *,
    head_checkpoint: str,
    output_dir: Path,
    max_events_per_payload: int | None,
    device: str,
    fake_keep_per_mille: int,
) -> dict[str, object]:
    samples, bundle, tables, _ns = _load_split(
        FAMILY_CORPUS[family], "train", max_events_per_payload
    )
    wrapper, artifact = load_relative_route_v4_head_only_artifact(head_checkpoint, device=device)
    tr = wrapper.trainable
    fr = wrapper.frozen_workbook64
    bound = tr.config.route_correction_bound

    cap: dict[str, np.ndarray] = {}
    hooks = [
        fr.backbone.edge_score[2].register_forward_pre_hook(
            lambda m, i: cap.__setitem__("edge_lat", i[0].detach().cpu().numpy())
        ),
        tr.backbone.blocks[-1].register_forward_hook(
            lambda m, i, o: cap.__setitem__("node_lat", (o[0] if isinstance(o, tuple) else o).detach().cpu().numpy())
        ),
        tr.route_encoder.register_forward_pre_hook(
            lambda m, i: cap.__setitem__("route_in", i[0].detach().cpu().numpy())
        ),
        tr.route_score.register_forward_pre_hook(
            lambda m, i: cap.__setitem__("route_hid", i[0].detach().cpu().numpy())
        ),
        tr.route_score.register_forward_hook(
            lambda m, i, o: cap.__setitem__("raw_delta", o.detach().cpu().numpy().reshape(-1))
        ),
    ]

    low: dict[str, list[np.ndarray]] = {k: [] for k in (
        "uid", "payload_idx", "run_id", "event_id", "n0", "n1", "n2", "n3",
        "share_class", "label", "R0", "R1", "R2logits", "R3", "R7", "R_phys",
    )}
    high: dict[str, list[np.ndarray]] = {k: [] for k in ("uid", "R2latent", "R4", "R5", "R6")}
    uid = 0
    n_events = 0
    rng = np.random.default_rng(20260905)
    batch_size = 16

    graphs = [g for g in bundle.graphs if tables[id(g)].size]
    try:
        for start in range(0, len(graphs), batch_size):
            batch_graphs = graphs[start:start + batch_size]
            n_events += len(batch_graphs)
            cap.clear()
            batch = _make_route_batch(
                batch_graphs, artifact.node_standardizer, artifact.edge_standardizer,
                torch.device(device), tables,
            )
            with torch.no_grad():
                out = wrapper(
                    batch.base.node_features, batch.base.station_ids,
                    batch.base.message_edge_source, batch.base.message_edge_destination,
                    batch.base.message_edge_features, batch.base.message_edge_chi2,
                    batch.base.message_edge_station_pair, batch.base.message_edge_direction,
                    batch.base.score_edge_source, batch.base.score_edge_destination,
                    batch.base.score_edge_features, batch.base.score_edge_station_pair,
                    batch.route_node_indices, batch.route_score_edge_indices,
                )
            edge_lat = cap["edge_lat"]          # [E_total, 128]
            node_lat = cap["node_lat"]          # [N_total, 128]
            route_in = cap["route_in"]          # [R_total, 1075]
            route_hid = cap["route_hid"]        # [R_total, 128]
            raw_delta_all = cap["raw_delta"].reshape(-1)  # [R_total]
            edge_logits_all = out.edge_logits.detach().cpu().numpy()       # [E_total]
            bounded_all = out.delta_route_logits.detach().cpu().numpy().reshape(-1)
            lcorr_all = out.route_logits.detach().cpu().numpy().reshape(-1)
            std_edge_feats = batch.base.score_edge_features.cpu().numpy()  # [E_total, 11]

            node_offset = 0
            score_offset = 0
            route_offset = 0
            for graph in batch_graphs:
                table = tables[id(graph)]
                R = table.size
                rni = table.node_indices                     # local [R,4]
                rsei = table.score_edge_indices              # local [R,3]
                rni_g = rni + node_offset
                rsei_g = rsei + score_offset
                rslice = slice(route_offset, route_offset + R)
                sc = _route_share_classes(graph, table)
                label = np.asarray(table.labels, dtype=bool)
                sc_idx = np.array([_share_class_index(bool(l), int(s)) for l, s in zip(label, sc)], dtype=np.int64)

                R0 = graph.score_edge_features[rsei].reshape(R, -1).astype(np.float32)       # raw 33
                R1 = std_edge_feats[rsei_g].reshape(R, -1).astype(np.float32)                # std 33
                R2log = edge_logits_all[rsei_g].astype(np.float32)                           # [R,3]
                R3 = graph.node_features[rni].reshape(R, -1).astype(np.float32)              # raw 68
                L_edge = R2log.sum(axis=1)
                bounded = bounded_all[rslice].astype(np.float32)
                raw_delta = raw_delta_all[rslice].astype(np.float32)
                L_corr = lcorr_all[rslice].astype(np.float32)
                R7 = np.stack([raw_delta, bounded, L_edge, L_corr], axis=1)
                R_phys = np.concatenate([R0, R2log], axis=1)

                R2lat = edge_lat[rsei_g].reshape(R, -1).astype(np.float32)
                R4 = node_lat[rni_g].reshape(R, -1).astype(np.float32)
                R5 = route_in[rslice].astype(np.float32)
                R6 = route_hid[rslice].astype(np.float32)

                run_id = int(graph.event.run_id)
                event_id = int(graph.event.event_id)
                try:
                    payload_idx = int(graph.sample.payload_id)
                except (TypeError, ValueError):
                    payload_idx = -1

                uids = np.arange(uid, uid + R, dtype=np.int64)
                uid += R
                low["uid"].append(uids)
                low["payload_idx"].append(np.full(R, payload_idx, np.int64))
                low["run_id"].append(np.full(R, run_id, np.int64))
                low["event_id"].append(np.full(R, event_id, np.int64))
                for k in range(4):
                    low[f"n{k}"].append(rni[:, k].astype(np.int64))
                low["share_class"].append(sc_idx)
                low["label"].append(label.astype(np.int64))
                low["R0"].append(R0); low["R1"].append(R1); low["R2logits"].append(R2log)
                low["R3"].append(R3); low["R7"].append(R7); low["R_phys"].append(R_phys)
                keep_fake = (rng.random(R) < (fake_keep_per_mille / 1000.0))
                high_mask = label | keep_fake
                if high_mask.any():
                    high["uid"].append(uids[high_mask])
                    high["R2latent"].append(R2lat[high_mask])
                    high["R4"].append(R4[high_mask])
                    high["R5"].append(R5[high_mask])
                    high["R6"].append(R6[high_mask])

                node_offset += graph.node_features.shape[0]
                score_offset += graph.score_labels.size
                route_offset += R
    finally:
        for h in hooks:
            h.remove()

    output_dir.mkdir(parents=True, exist_ok=True)

    def _cat(d: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
        return {k: (np.concatenate(v, axis=0) if v else np.empty(0)) for k, v in d.items()}

    low_arr = _cat(low)
    high_arr = _cat(high)
    np.savez(output_dir / f"{family}_lowdim.npz", **low_arr)
    np.savez_compressed(output_dir / f"{family}_highdim.npz", **high_arr)

    summary = {
        "family": family,
        "head_checkpoint": head_checkpoint,
        "route_correction_bound": bound,
        "n_events": n_events,
        "n_routes_total": int(low_arr["uid"].size),
        "n_routes_highdim": int(high_arr["uid"].size),
        "share_class_counts_total": {
            SHARE_CLASS_NAMES[i]: int((low_arr["share_class"] == i).sum()) for i in range(5)
        },
        "widths": {
            "R0": R0_W, "R1": R1_W, "R2logits": R2LOG_W, "R3": R3_W, "R_phys": RPHYS_W,
            "R7": R7_W, "R2latent": R2LAT_W, "R4": R4_W, "R5": R5_W, "R6": R6_W,
        },
        "node_abs_idx": NODE_ABS_IDX,
        "node_qual_idx": NODE_QUAL_IDX,
        "edge_feature_names": list(EDGE_FEATURE_NAMES),
        "node_feature_names": list(NODE_FEATURE_NAMES),
        "share_class_names": list(SHARE_CLASS_NAMES),
    }
    (output_dir / f"{family}_extract_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(FAMILY_CORPUS), required=True)
    parser.add_argument("--head-checkpoint", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-events-per-payload", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fake-keep-per-mille", type=int, default=120)
    args = parser.parse_args()
    head = args.head_checkpoint or FAMILY_HEAD[args.family]
    summary = extract_family(
        args.family,
        head_checkpoint=head,
        output_dir=Path(args.output_dir),
        max_events_per_payload=args.max_events_per_payload,
        device=args.device,
        fake_keep_per_mille=args.fake_keep_per_mille,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
