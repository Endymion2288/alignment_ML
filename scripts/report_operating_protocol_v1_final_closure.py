#!/usr/bin/env python3
"""Write the Operating Protocol V1 final evidence package.

Reads frozen 47-53 artifacts only.  Does not generate FD probes, run
Newton, write a payload, retrain V2, or reopen Station / C_dx Mode.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    RESIDUAL_DECREASE_LABEL,
    SCHEMA_VERSION,
    assert_no_alignment_payload,
    build_chart_data,
    build_evidence_matrix,
    build_final_report,
    build_reproducibility_manifest,
    evaluate_unlock,
    extract_frozen_A,
    extract_reference_scale,
    file_record,
    frozen_operating_state,
    load_closure_config,
    occupancy_provenance,
    project_root,
    resolve_under_root,
    sha256_file,
    unlock_criteria,
    walk_forbidden,
)


PROJECT_ROOT = project_root()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON mapping: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _required(root: Path, relative: str) -> Path:
    path = resolve_under_root(root, relative)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/operating_protocol_v1_final_real_data_closure_v1.yaml"),
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "outputs/operating_protocol_v1_final_real_data_closure_v1"),
    )
    args = parser.parse_args()

    config = load_closure_config(args.config)
    root = PROJECT_ROOT
    created = datetime.now(timezone.utc).isoformat()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    chart_root = output_root / "chart_data"

    checkpoint = _required(root, str(config["frozen_v2_checkpoint"]))
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != FROZEN_V2_CHECKPOINT_SHA256:
        raise ValueError(
            f"frozen V2 checkpoint SHA256 mismatch: {checkpoint_sha256} != "
            f"{FROZEN_V2_CHECKPOINT_SHA256}"
        )

    artifacts = {
        "mode_validity_contract": _read_json(_required(root, str(config["mode_validity_contract"]))),
        "transfer_report": _read_json(
            _required(root, f"{config['mc_transfer_root']}/transfer_report.json")
        ),
        "scaling_report": _read_json(
            _required(root, f"{config['scaling_root']}/route_acceptance_scaling_report.json")
        ),
        "fullscale_decision": _read_json(
            _required(
                root,
                f"{config['self_nulling_root']}/operating_protocol_next_decision.json",
            )
        ),
        "failure_classification": _read_json(
            _required(
                root,
                f"{config['failure_audit_root']}/operating_protocol_failure_classification.json",
            )
        ),
        "identifiability_audit": _read_json(
            _required(
                root,
                f"{config['failure_audit_root']}/station_identifiability_real_data_audit.json",
            )
        ),
        "reduced_decision": _read_json(
            _required(
                root,
                f"{config['reduced_mode_root']}/operating_protocol_next_decision.json",
            )
        ),
        "reduced_identifiability": _read_json(
            _required(
                root,
                f"{config['reduced_mode_root']}/reduced_mode_identifiability_audit.json",
            )
        ),
        "reduced_transfer": _read_json(
            _required(
                root,
                f"{config['reduced_mode_root']}/reduced_mode_transfer_dq_report.json",
            )
        ),
        "reduced_feasibility": _read_json(
            _required(
                root,
                f"{config['reduced_mode_root']}/reduced_station_mode_feasibility_report.json",
            )
        ),
        "monitoring_decision": _read_json(
            _required(
                root,
                f"{config['monitoring_root']}/operating_protocol_monitoring_decision.json",
            )
        ),
        "parent_run_level": _read_json(_required(root, str(config["parent_run_level_report"]))),
        "expansion_decision": _read_json(
            _required(
                root,
                f"{config['expansion_root']}/operating_protocol_monitoring_decision.json",
            )
        ),
        "expansion_run_level": _read_json(
            _required(root, f"{config['expansion_root']}/run_level_dq_report.json")
        ),
        "time_stability": _read_json(
            _required(root, f"{config['expansion_root']}/time_stability_report.json")
        ),
        "parent_windows": _read_json(
            _required(root, f"{config['occupancy_preflight_root']}/frozen_windows.json")
        ),
        "expansion_provenance": _read_json(
            _required(root, f"{config['expansion_root']}/expansion_provenance.json")
        ),
    }

    if artifacts["transfer_report"].get("both_modes_independent_closure") is not True:
        raise ValueError("entry 47 MC transfer is not an independent-closure PASS")
    if artifacts["failure_classification"].get("decision") != "physical_nonidentifiability":
        raise ValueError("entry 50 unique class is not physical_nonidentifiability")
    if artifacts["fullscale_decision"].get("decision") != "cross_level_contaminated":
        raise ValueError("entry 49 full-segment Station decision is not cross_level_contaminated")
    if artifacts["reduced_decision"].get("unique_class") != "real_data_residual_dq_monitoring_only":
        raise ValueError("entry 51 did not close to residual_dq_monitoring_only")
    if artifacts["expansion_decision"].get("alignment_drift_candidate") is not False:
        raise ValueError("entry 53 must remain alignment_drift_candidate=false")

    frozen_a = extract_frozen_A(artifacts["mode_validity_contract"])
    reference_scale = extract_reference_scale(artifacts["parent_run_level"])
    occupancy_rows = occupancy_provenance(artifacts["parent_windows"]) + occupancy_provenance(
        artifacts["expansion_provenance"]
    )

    file_records = [
        file_record(Path(config["path"]), role="closure_config"),
        file_record(_required(root, str(config["window_rule"])), role="occupancy_selection_rule"),
        file_record(checkpoint, role="frozen_v2_checkpoint"),
        file_record(
            _required(root, str(config["mode_validity_contract"])),
            role="mode_validity_contract",
        ),
        file_record(
            _required(root, f"{config['mc_transfer_root']}/transfer_report.json"),
            role="entry_47_mc_transfer",
        ),
        file_record(
            _required(root, f"{config['occupancy_preflight_root']}/frozen_windows.json"),
            role="entry_48_occupancy_windows",
        ),
        file_record(
            _required(root, f"{config['scaling_root']}/route_acceptance_scaling_report.json"),
            role="entry_49_route_acceptance_scaling",
        ),
        file_record(
            _required(
                root,
                f"{config['self_nulling_root']}/operating_protocol_next_decision.json",
            ),
            role="entry_49_fullscale_station_decision",
        ),
        file_record(
            _required(
                root,
                f"{config['failure_audit_root']}/operating_protocol_failure_classification.json",
            ),
            role="entry_50_station_failure_class",
        ),
        file_record(
            _required(
                root,
                f"{config['reduced_mode_root']}/operating_protocol_next_decision.json",
            ),
            role="entry_51_reduced_mode_decision",
        ),
        file_record(
            _required(
                root,
                f"{config['monitoring_root']}/operating_protocol_monitoring_decision.json",
            ),
            role="entry_52_monitoring_decision",
        ),
        file_record(
            _required(root, str(config["parent_run_level_report"])),
            role="entry_52_reference_residual_scale",
        ),
        file_record(
            _required(
                root,
                f"{config['expansion_root']}/operating_protocol_monitoring_decision.json",
            ),
            role="entry_53_expansion_decision",
        ),
    ]

    evidence_matrix = build_evidence_matrix(artifacts)
    chart_data = build_chart_data(artifacts)
    manifest = build_reproducibility_manifest(
        config=config,
        file_records=file_records,
        occupancy_rows=occupancy_rows,
        reference_scale=reference_scale,
        frozen_a=frozen_a,
        checkpoint_sha256=checkpoint_sha256,
    )
    final_report = build_final_report(
        config=config,
        evidence_matrix=evidence_matrix,
        manifest=manifest,
        expansion_decision=artifacts["expansion_decision"],
        created_utc=created,
    )
    unlock = unlock_criteria()
    unlock_eval = evaluate_unlock()

    named = {
        "operating_protocol_v1_final_report.json": final_report,
        "real_data_evidence_matrix.json": {
            **evidence_matrix,
            "created_utc": created,
            "unlock_criteria": unlock,
        },
        "reproducibility_manifest.json": {
            **manifest,
            "created_utc": created,
        },
        "operating_protocol_v2_unlock_criteria.json": {
            **unlock,
            "evaluation": unlock_eval,
            "created_utc": created,
            "residual_decrease_label": RESIDUAL_DECREASE_LABEL,
        },
        "chart_data/chart_data.json": {
            **chart_data,
            "created_utc": created,
        },
    }
    for name, payload in named.items():
        clean = dict(payload)
        walk_forbidden(clean, where=name)
        assert_no_alignment_payload(clean if "geometry_write_allowed" in clean else final_report)
        _write_json(output_root / name, clean)

    scaling_rows = list(chart_data["selected_route_scaling"]["rows"])
    _write_csv(
        chart_root / "selected_route_scaling.csv",
        scaling_rows,
        [
            "campaign",
            "run",
            "role",
            "scale",
            "n_events",
            "n_tracklets",
            "n_all_pairs_candidates",
            "selected_routes",
            "complete_four_station_routes",
        ],
    )
    _write_csv(
        chart_root / "route_composition.csv",
        list(chart_data["route_composition"]["rows"]),
        [
            "run",
            "lhc_fill",
            "role",
            "selected_routes",
            "n_2_station",
            "n_3_station",
            "n_4_station",
            "status",
        ],
    )
    _write_csv(
        chart_root / "dy_rx_robust_z_timeseries.csv",
        list(chart_data["robust_z_timeseries"]["rows"]),
        [
            "lhc_fill",
            "run",
            "skip_events",
            "role",
            "selected_routes",
            "status",
            "dy_robust_z",
            "rx_robust_z",
            "dy_median_mm",
            "rx_median",
        ],
    )
    spectrum_rows = []
    for row in chart_data["station_jacobian_singular_spectrum"]["rows"]:
        for index, value in enumerate(row.get("six_dof_singular_values") or [], start=1):
            spectrum_rows.append(
                {
                    "run": row["run"],
                    "dof": "six_dof",
                    "index": index,
                    "singular_value": value,
                    "condition_number": row.get("six_dof_condition_number"),
                    "residual_decrease_label": RESIDUAL_DECREASE_LABEL,
                }
            )
        for index, value in enumerate(row.get("five_dof_singular_values") or [], start=1):
            spectrum_rows.append(
                {
                    "run": row["run"],
                    "dof": "five_dof_excluding_dz",
                    "index": index,
                    "singular_value": value,
                    "condition_number": row.get("five_dof_condition_number"),
                    "residual_decrease_label": RESIDUAL_DECREASE_LABEL,
                }
            )
    _write_csv(
        chart_root / "station_jacobian_singular_spectrum.csv",
        spectrum_rows,
        ["run", "dof", "index", "singular_value", "condition_number", "residual_decrease_label"],
    )
    reduced_rows = []
    for row in chart_data["reduced_mode_cross_run_inconsistency"]["per_run"]:
        delta = row.get("delta") or {}
        reduced_rows.append(
            {
                "run": row["run"],
                "condition_number": row.get("condition_number"),
                "ift_dy_mm": delta.get("ift_dy_mm"),
                "ift_rx_mrad": delta.get("ift_rx_mrad"),
                "ift_rz_mrad": delta.get("ift_rz_mrad"),
                "residual_decrease_label": RESIDUAL_DECREASE_LABEL,
            }
        )
    _write_csv(
        chart_root / "reduced_mode_cross_run_inconsistency.csv",
        reduced_rows,
        [
            "run",
            "condition_number",
            "ift_dy_mm",
            "ift_rx_mrad",
            "ift_rz_mrad",
            "residual_decrease_label",
        ],
    )
    a_rows = []
    native = chart_data["frozen_A_versus_station_weak_direction"]["frozen_A_native_per_mm_C_dx"]
    for name, value in native.items():
        a_rows.append({"kind": "frozen_A", "run": None, "parameter": name, "value": value})
    for row in chart_data["frozen_A_versus_station_weak_direction"]["per_run"]:
        a_rows.append(
            {
                "kind": "five_dof_cosine_with_A",
                "run": row["run"],
                "parameter": "cosine",
                "value": row.get("five_dof_weak_direction_cosine_with_A"),
            }
        )
        for name, value in (row.get("five_dof_smallest_direction") or {}).items():
            a_rows.append(
                {
                    "kind": "five_dof_weak_direction",
                    "run": row["run"],
                    "parameter": name,
                    "value": value,
                }
            )
    _write_csv(
        chart_root / "frozen_A_versus_station_weak_direction.csv",
        a_rows,
        ["kind", "run", "parameter", "value"],
    )

    state = frozen_operating_state()
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "schema_version": SCHEMA_VERSION,
                **{key: state[key] for key in (
                    "real_data_operating_mode",
                    "geometry_write_allowed",
                    "station_calibration_mode_available",
                    "cdx_mode_allowed",
                    "alignment_drift_candidate",
                )},
                "frozen_v2_checkpoint_sha256": checkpoint_sha256,
                "unlock_currently_met": False,
                "residual_decrease_label": RESIDUAL_DECREASE_LABEL,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
