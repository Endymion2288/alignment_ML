#!/usr/bin/env python3
"""Prepare a source-disjoint, anchor-centred physical alignment iteration.

The command writes one immutable scan configuration per original xAOD source.
Every configuration carries the same explicit alignment anchor, nominal target,
and central finite-difference probes, but each is later run through the real
``/Tracker/Align -> SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit
-> NtupleDumper -> FaserActsExtrapolationTool(mode 0)`` chain on its own
source file.  It deliberately never resolves a sealed test source.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from scripts.build_physical_curriculum_corpus import _allowed_splits, _load_config, _validate_sources
from scripts.config_loader import load_yaml_with_base
from scripts.prepare_multidof_alignment_iteration import compile_iteration
from scripts.run_physical_refit_capture_scan import _build_plan


SCHEMA_VERSION = "faser-multisource-physical-alignment-iteration-v1"
ALLOWED_SPLITS = ("train", "validation")


def _read_template(path: Path) -> dict[str, Any]:
    payload = load_yaml_with_base(path)
    if not isinstance(payload, Mapping):
        raise ValueError("iteration template must be a YAML mapping")
    return dict(payload)


def _parse_named_values(values: Sequence[str], names: Sequence[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for raw in values:
        name, separator, raw_value = str(raw).partition(":")
        if not separator or not name:
            raise ValueError("--current entries must have form PARAMETER:VALUE")
        if name in parsed:
            raise ValueError(f"--current repeats parameter '{name}'")
        try:
            value = float(raw_value)
        except ValueError as error:
            raise ValueError(f"--current has a non-numeric value for '{name}'") from error
        if not math.isfinite(value):
            raise ValueError(f"--current value for '{name}' must be finite")
        parsed[name] = value
    if set(parsed) != set(names):
        raise ValueError("--current must specify exactly: " + ", ".join(names))
    return {name: float(parsed[name]) for name in names}


def _values_from_update(path: Path, names: Sequence[str], *, allow_unverified: bool) -> dict[str, float]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("alignment update JSON must be a mapping")
    if payload.get("capture_success") is not True and not allow_unverified:
        raise ValueError(
            "refusing to advance from an update without capture_success=true; "
            "use --allow-unverified-update only for an explicitly documented diagnostic iteration"
        )
    values = payload.get("proposed_next_parameter_values")
    if not isinstance(values, Mapping):
        raise ValueError("alignment update JSON lacks proposed_next_parameter_values")
    return _parse_named_values(
        [f"{name}:{values[name]}" for name in names] if set(values) == set(names) else [],
        names,
    )


def _plan_signature(plan: Mapping[str, object]) -> dict[str, object]:
    """Return the source-independent physics contract used for comparisons."""
    fields = (
        "scan_mode",
        "q_over_p_mode",
        "station_ids",
        "reference_station_ids",
        "movable_station_ids",
        "movable_layer_ids",
        "condition_axis",
        "alignment_parameter_specs",
        "points",
    )
    return {field: copy.deepcopy(plan.get(field)) for field in fields}


def _compile_scan(
    template: Mapping[str, object],
    *,
    iteration: int,
    current_values: Mapping[str, float],
) -> tuple[dict[str, object], dict[str, object]]:
    raw_scan = template.get("physical_refit_capture_scan", template)
    if not isinstance(raw_scan, Mapping):
        raise ValueError("physical_refit_capture_scan must be a mapping")
    mode = str(raw_scan.get("scan_mode", ""))
    if mode == "station_rigid_multidof":
        return compile_iteration(template, iteration=iteration, current_values=current_values)
    if mode == "ift_layer_hierarchy":
        if str(raw_scan.get("fit_basis", "")) == "hierarchical_v1":
            from scripts.prepare_hierarchical_v1_iteration import compile_hierarchical_v1

            return compile_hierarchical_v1(
                template,
                iteration=iteration,
                current_values=current_values,
                include_finite_differences=not bool(raw_scan.get("held_out_only", False)),
            )
        from scripts.prepare_layer_identifiability_pilot import compile_layer_identifiability_pilot

        return compile_layer_identifiability_pilot(
            template,
            iteration=iteration,
            current_values=current_values,
            include_finite_differences=not bool(raw_scan.get("held_out_only", False)),
        )
    raise ValueError(f"unsupported iteration scan_mode '{mode}'")


def prepare_iteration(
    *,
    source_config_path: Path,
    iteration_template_path: Path,
    output_root: Path,
    iteration: int,
    current_values: Mapping[str, float],
    nevents: int,
    source_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Write an immutable source-specific physical scan bank and manifest."""
    if nevents < 1:
        raise ValueError("nevents must be positive")
    corpus_config = _load_config(source_config_path)
    allowed = tuple(_allowed_splits(corpus_config))
    if allowed != ALLOWED_SPLITS:
        raise ValueError("multi-source alignment iteration requires exactly train and validation")
    forbidden = {str(value) for value in corpus_config.get("forbidden_splits", ())}
    if "test" not in forbidden:
        raise ValueError("multi-source alignment iteration must explicitly forbid test")
    sources = _validate_sources(corpus_config, allowed)
    known_sources = {source["source_id"] for source in sources}
    if source_ids:
        requested = {str(value) for value in source_ids}
        unknown = requested - known_sources
        if unknown:
            raise ValueError("unknown configured source ID(s): " + ", ".join(sorted(unknown)))
        sources = [source for source in sources if source["source_id"] in requested]
    if not sources:
        raise ValueError("no train/validation source selected")
    represented = {source["split"] for source in sources}
    if source_ids is None and represented != set(ALLOWED_SPLITS):
        raise ValueError("source configuration must provide both train and validation")

    template = _read_template(iteration_template_path)
    compiled, contract = _compile_scan(template, iteration=iteration, current_values=current_values)
    raw_scan = compiled.get("physical_refit_capture_scan")
    if not isinstance(raw_scan, Mapping):
        raise ValueError("compiled iteration has no physical_refit_capture_scan")
    scan = dict(raw_scan)
    if int(scan.get("q_over_p_mode", -1)) != 0:
        raise ValueError("multi-source iteration requires field-aware mode-0 propagation")
    common_plan = _build_plan(scan)
    signature = _plan_signature(common_plan)

    if output_root.exists():
        if any(output_root.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty iteration root: {output_root}")
    else:
        output_root.mkdir(parents=True, exist_ok=False)

    source_records: list[dict[str, object]] = []
    for source in sources:
        source_scan = copy.deepcopy(scan)
        source_scan["input_xaod"] = str(source["input_xaod"])
        source_scan["nevents"] = int(nevents)
        source_scan["run_alignment_closure"] = False
        path = output_root / "sources" / str(source["source_id"]) / "physical_scan_config.yaml"
        path.parent.mkdir(parents=True, exist_ok=False)
        path.write_text(
            yaml.safe_dump({"physical_refit_capture_scan": source_scan}, sort_keys=False),
            encoding="utf-8",
        )
        source_records.append(
            {
                "source_id": str(source["source_id"]),
                "split": str(source["split"]),
                "input_xaod": str(source["input_xaod"]),
                "physical_scan_config": str(path),
                "physical_scan_root": str(path.parent / "physical_scan"),
            }
        )

    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_config": str(source_config_path),
        "iteration_template": str(iteration_template_path),
        "iteration": int(iteration),
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "q_over_p_mode": 0,
        "refit_chain": (
            "persisted SCT_ClusterContainer -> SegmentFitRefit -> SegmentsRefit -> "
            "NtupleDumper -> FaserActsExtrapolationTool(mode 0)"
        ),
        "source_split_unit": "original_xAOD_file",
        "allowed_splits": list(ALLOWED_SPLITS),
        "forbidden_splits": ["test"],
        "test_data_accessed": False,
        "nevents_per_source": int(nevents),
        "alignment_iteration": contract,
        "common_scan_plan": signature,
        "sources": source_records,
    }
    manifest_path = output_root / "iteration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--iteration-template", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--iteration", required=True, type=int)
    current_group = parser.add_mutually_exclusive_group(required=True)
    current_group.add_argument("--current", action="append", metavar="PARAMETER:VALUE")
    current_group.add_argument("--update-json", default=None)
    parser.add_argument(
        "--allow-unverified-update",
        action="store_true",
        help="Permit --update-json without capture_success=true for a documented diagnostic scan only.",
    )
    parser.add_argument("--nevents", type=int, default=100)
    parser.add_argument("--source-id", action="append", default=None)
    args = parser.parse_args()
    if args.iteration < 0:
        parser.error("--iteration must be non-negative")
    template_path = Path(args.iteration_template).expanduser().resolve()
    template = _read_template(template_path)
    raw_scan = template.get("physical_refit_capture_scan", template)
    if not isinstance(raw_scan, Mapping):
        parser.error("iteration template has no physical_refit_capture_scan")
    raw_specs = raw_scan.get("alignment_parameter_specs")
    if not isinstance(raw_specs, list) or not raw_specs:
        parser.error("iteration template has no alignment_parameter_specs")
    names = [str(spec.get("name", "")) for spec in raw_specs if isinstance(spec, Mapping)]
    if len(names) != len(raw_specs) or not all(names) or len(set(names)) != len(names):
        parser.error("iteration template has invalid parameter names")
    current = (
        _parse_named_values(args.current, names)
        if args.current is not None
        else _values_from_update(
            Path(str(args.update_json)).expanduser().resolve(),
            names,
            allow_unverified=bool(args.allow_unverified_update),
        )
    )
    manifest = prepare_iteration(
        source_config_path=Path(args.source_config).expanduser().resolve(),
        iteration_template_path=template_path,
        output_root=Path(args.output_root).expanduser().resolve(),
        iteration=int(args.iteration),
        current_values=current,
        nevents=int(args.nevents),
        source_ids=args.source_id,
    )
    print(
        json.dumps(
            {
                "iteration_manifest": str(Path(args.output_root).expanduser().resolve() / "iteration_manifest.json"),
                "sources_by_split": {
                    split: sum(1 for item in manifest["sources"] if item["split"] == split)
                    for split in ALLOWED_SPLITS
                },
                "points": len(manifest["common_scan_plan"]["points"]),
                "test_data_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
