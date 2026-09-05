#!/usr/bin/env python3
"""Workbook 75 pre-training audit (before Condor six-source training).

Checks, without opening development / Final Blind / sealed data:

1. git state (Workbook 74 commits retained; no reset)
2. architecture freeze: route_representation_mode = physical_pair_relative
3. parameter audit: trainable / frozen counts
4. data provenance: six authorized sources only; 00350_00399 and 00800_00849 absent
5. W64 parent SHA
6. training contract inherited from Workbook 74 (no retune)

Writes outputs/mc24_four_station_physical_pair_relative_route_v1_six_source/pretraining_audit.json
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config_loader import load_yaml_with_base
from scripts.run_refit_multidof_closure import _json_ready
from training.source_diversity_audit import (
    AUTHORIZED_SIX_TRAIN_SOURCES,
    RESERVED_BLIND_SOURCES,
    UNUSED_RESERVE_SOURCES,
    is_sealed_source,
)

CONFIG = PROJECT_ROOT / "configs/wb75_physical_pair_relative_bounded_primary_six_source.yaml"
WB74_CONFIG = PROJECT_ROOT / "configs/wb74_physical_pair_relative_bounded_primary_source_transfer.yaml"
TRAIN_MANIFEST = PROJECT_ROOT / (
    "outputs/mc24_four_station_source_diversity_train_v1/"
    "overlay_synthetic_v1/synthetic_corpus_manifest.json"
)
W64_CKPT = PROJECT_ROOT / (
    "outputs/mc24_four_station_source_diversity_v1/checkpoint/route_aware_transformer_v2.pt"
)
W64_SHA = "0c3a28704cc01151fab7ac943e41338e859b5dde1502c0b10e2f12ada04e6236"
WB74_HEAD = "7a8c894970a7cd9c08df08f6a35800b2db3fc55d"
ORIGIN_REF = "cc8d22036be5f0381fa7ad9441477dae155d5a66"
OUT = PROJECT_ROOT / "outputs/mc24_four_station_physical_pair_relative_route_v1_six_source"
FORBIDDEN_SUBSTRINGS = ("00350_00399", "00800_00849", "mc24_100116_", "mc24_100117_")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def main() -> None:
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    origin = subprocess.check_output(["git", "rev-parse", "origin/4station"], cwd=PROJECT_ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT, text=True
    )
    log = subprocess.check_output(["git", "log", "--oneline", "-8"], cwd=PROJECT_ROOT, text=True)
    wb74_present = bool(
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", WB74_HEAD, "HEAD"],
            cwd=PROJECT_ROOT,
        ).returncode
        == 0
    )
    git_ok = (
        wb74_present
        and origin == ORIGIN_REF
        and not dirty.strip()
    )

    cfg = load_yaml_with_base(CONFIG)["wb75_physical_pair_relative_bounded_primary"]
    wb74 = load_yaml_with_base(WB74_CONFIG)["wb74_physical_pair_relative_bounded_primary"]
    arch = cfg["architecture"]
    mode_ok = arch.get("route_representation_mode") == "physical_pair_relative"
    bound_ok = float(arch.get("route_correction_bound")) == 4.0
    additive_ok = arch.get("use_additive_route_correction") is True
    no_relative_latent = arch.get("use_relative_route_representation") is False

    # Contract inheritance: architecture + training + aux + curriculum identical
    # except workbook-level naming / description / forbidden_sources annotation.
    fc, fp = _flatten(wb74), _flatten(cfg)
    allowed_diff = {
        "arm_name",
        "description",
        "workbook",
        "input_contract.forbidden_sources",
    }
    unexpected = {
        k: {"wb74": fc.get(k), "wb75": fp.get(k)}
        for k in sorted(set(fc) | set(fp))
        if fc.get(k) != fp.get(k) and k not in allowed_diff
    }
    contract_inherited = not unexpected

    from models.route_transformer import (
        RouteAwareSparseTransformer,
        RouteAwareTransformerConfig,
        freeze_backbone_and_edge_scorer,
    )

    model = RouteAwareSparseTransformer(
        RouteAwareTransformerConfig(node_feature_dim=17, edge_feature_dim=11, **arch)
    )
    freeze = freeze_backbone_and_edge_scorer(model)
    trainable = int(freeze["n_trainable_parameters"])
    frozen = int(freeze["n_frozen_parameters"])
    param_ok = trainable == 21377 and frozen == 614947
    no_node_latent_modules = all(
        not hasattr(model, name)
        for name in ("route_query", "route_node_key", "route_edge_projection", "route_pair_embedding")
    )
    input_width_ok = model.route_encoder[0].in_features == 36

    manifest = json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8"))
    srcs = Counter()
    splits = Counter()
    for sample in manifest["samples"]:
        splits[sample.get("split")] += 1
        for sid in sample.get("source_ids") or []:
            srcs[str(sid)] += 1
    constituents = set(srcs)
    leaking = {
        s
        for s in constituents
        if s in set(RESERVED_BLIND_SOURCES)
        or s in set(UNUSED_RESERVE_SOURCES)
        or is_sealed_source(s)
        or any(n in s for n in FORBIDDEN_SUBSTRINGS)
    }
    six_ok = constituents == set(AUTHORIZED_SIX_TRAIN_SOURCES)
    split_ok = set(splits) == {"train"}
    provenance_ok = six_ok and split_ok and not leaking
    parent_sha = _sha256(W64_CKPT)
    parent_ok = parent_sha == W64_SHA

    training_authorized = bool(
        git_ok
        and mode_ok
        and bound_ok
        and additive_ok
        and no_relative_latent
        and contract_inherited
        and param_ok
        and no_node_latent_modules
        and input_width_ok
        and provenance_ok
        and parent_ok
    )
    audit = {
        "workbook": 75,
        "git": {
            "local_head": git_commit,
            "origin_4station": origin,
            "wb74_head_is_ancestor": wb74_present,
            "tracked_dirty": bool(dirty.strip()),
            "recent_log": log.strip().splitlines(),
            "ok": git_ok,
        },
        "architecture": {
            "route_representation_mode": arch.get("route_representation_mode"),
            "use_relative_route_representation": arch.get("use_relative_route_representation"),
            "use_additive_route_correction": arch.get("use_additive_route_correction"),
            "route_correction_bound": arch.get("route_correction_bound"),
            "no_absolute_node_latent_modules": no_node_latent_modules,
            "route_input_width": int(model.route_encoder[0].in_features),
            "ok": bool(mode_ok and bound_ok and additive_ok and no_relative_latent and no_node_latent_modules and input_width_ok),
        },
        "parameter_audit": {
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "expected_trainable": 21377,
            "expected_frozen": 614947,
            "ok": param_ok,
        },
        "data_provenance": {
            "train_manifest": str(TRAIN_MANIFEST),
            "train_manifest_sha256": _sha256(TRAIN_MANIFEST),
            "n_samples": int(len(manifest["samples"])),
            "splits": dict(splits),
            "sources": dict(sorted(srcs.items())),
            "authorized_six": list(AUTHORIZED_SIX_TRAIN_SOURCES),
            "development_00350_00399_absent": not any("00350_00399" in s for s in constituents),
            "final_blind_00800_00849_absent": not any("00800_00849" in s for s in constituents),
            "sealed_absent": not any(is_sealed_source(s) for s in constituents),
            "leaking": sorted(leaking),
            "ok": provenance_ok,
        },
        "frozen_w64_parent": {
            "path": str(W64_CKPT),
            "sha256": parent_sha,
            "expected": W64_SHA,
            "ok": parent_ok,
        },
        "training_contract_inherited_from_wb74": {
            "unexpected_differences": unexpected,
            "ok": contract_inherited,
            "seed": cfg["training"]["seed"],
            "learning_rate": cfg["training"]["learning_rate"],
            "weight_decay": cfg["training"]["weight_decay"],
            "batch_size": cfg["training"]["batch_size"],
            "early_stopping": cfg["training"]["early_stopping"],
            "selection_policy": cfg["training"]["selection_policy"],
            "curriculum_stages": cfg["curriculum_stages"],
        },
        "continue_to_15d_relative_wls": False,
        "new_final_blind_content_accessed": False,
        "sealed_test_accessed": False,
        "training_authorized": training_authorized,
    }
    _write_json(OUT / "pretraining_audit.json", audit)
    print(json.dumps(_json_ready({
        "training_authorized": training_authorized,
        "git_ok": git_ok,
        "architecture_ok": audit["architecture"]["ok"],
        "parameter_ok": param_ok,
        "provenance_ok": provenance_ok,
        "parent_ok": parent_ok,
        "contract_inherited": contract_inherited,
        "trainable": trainable,
        "frozen": frozen,
    }), indent=2, sort_keys=True))
    if not training_authorized:
        raise SystemExit("Workbook 75 pre-training audit FAILED; training_authorized = false")


if __name__ == "__main__":
    main()
