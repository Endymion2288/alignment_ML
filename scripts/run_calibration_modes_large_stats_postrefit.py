#!/usr/bin/env python3
"""Post-refit stages for exclusive-mode large-statistics MC transfer.

Physical Condor jobs must finish first.  This driver never retrains V2,
never re-registers capture or A, never opens the sealed test, and never
treats residual reduction as alignment success.

Stages:
  complete  audit the three isolated physical banks
  corpus    write physical_corpus_manifest.json once every point is accepted
  commands  print the frozen synthetic / V2 / closure / A / matrix commands
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
SOURCE_CONFIG = ROOT / "configs" / "physical_curriculum_calibration_modes_large_stats_sources.yaml"
FROZEN_V2 = ROOT / "outputs" / "mc24_v3_expanded_trainval_v2_bce_control_v1"
CONTRACT = ROOT / "outputs" / "mc24_ift_calibration_mode_validity_contract_v1" / "mode_validity_contract.json"
STATION_CAPTURE = ROOT / "outputs" / "mc24_ift_5dof_survey_dz_capture_criteria_train_v1" / "capture_criteria.json"
CDX_CAPTURE = ROOT / "outputs" / "mc24_ift_layer_contrast_2d_capture_criteria_train_v1" / "capture_criteria.json"
CURRENT_REPORT = ROOT / "outputs" / "mc24_ift_calibration_mode_transfer_current_corpus_v1" / "transfer_report.json"

BANKS = {
    "station": ROOT / "outputs" / "mc24_ift_station_mode_large_stats_transfer_physical_v1",
    "ift_internal": ROOT / "outputs" / "mc24_ift_internal_cdx_large_stats_transfer_physical_v1",
    "leakage": ROOT / "outputs" / "mc24_ift_leakage_operator_large_stats_transfer_physical_v1",
}


def _run(argv: list[str]) -> None:
    completed = subprocess.run(argv, cwd=str(ROOT), check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def _commands() -> list[str]:
    station = BANKS["station"]
    cdx = BANKS["ift_internal"]
    leakage = BANKS["leakage"]
    lines = [
        "# Frozen overlay seed: alignment_iteration_shared_across_payloads",
        "# Frozen V2: outputs/mc24_v3_expanded_trainval_v2_bce_control_v1 (ungated)",
        "# Do not retrain, retune capture/A/V2, add DoF, or enter real data.",
        "",
        "# 1. Physical corpus manifests (after --stage corpus)",
        f"python scripts/build_6dof_pilot_physical_corpus.py --iteration-manifest {station}/iteration_manifest.json",
        f"python scripts/build_6dof_pilot_physical_corpus.py --iteration-manifest {cdx}/iteration_manifest.json",
        f"python scripts/build_6dof_pilot_physical_corpus.py --iteration-manifest {leakage}/iteration_manifest.json",
        "",
        "# 2. Synthetics (shared overlay seed; train then validation --resume)",
        "python scripts/materialize_pooled_curriculum_synthetics.py \\",
        f"  --physical-manifest {station}/physical_corpus_manifest.json \\",
        f"  --config {SOURCE_CONFIG} \\",
        "  --output-dir outputs/mc24_ift_station_mode_large_stats_transfer_synthetic_v1 \\",
        "  --split train",
        "python scripts/materialize_pooled_curriculum_synthetics.py \\",
        f"  --physical-manifest {station}/physical_corpus_manifest.json \\",
        f"  --config {SOURCE_CONFIG} \\",
        "  --output-dir outputs/mc24_ift_station_mode_large_stats_transfer_synthetic_v1 \\",
        "  --split validation --resume",
        "python scripts/materialize_pooled_curriculum_synthetics.py \\",
        f"  --physical-manifest {cdx}/physical_corpus_manifest.json \\",
        f"  --config {SOURCE_CONFIG} \\",
        "  --output-dir outputs/mc24_ift_internal_cdx_large_stats_transfer_synthetic_v1 \\",
        "  --split train",
        "python scripts/materialize_pooled_curriculum_synthetics.py \\",
        f"  --physical-manifest {cdx}/physical_corpus_manifest.json \\",
        f"  --config {SOURCE_CONFIG} \\",
        "  --output-dir outputs/mc24_ift_internal_cdx_large_stats_transfer_synthetic_v1 \\",
        "  --split validation --resume",
        "python scripts/materialize_pooled_curriculum_synthetics.py \\",
        f"  --physical-manifest {leakage}/physical_corpus_manifest.json \\",
        f"  --config {SOURCE_CONFIG} \\",
        "  --output-dir outputs/mc24_ift_leakage_operator_large_stats_transfer_synthetic_v1 \\",
        "  --split train",
        "python scripts/materialize_pooled_curriculum_synthetics.py \\",
        f"  --physical-manifest {leakage}/physical_corpus_manifest.json \\",
        f"  --config {SOURCE_CONFIG} \\",
        "  --output-dir outputs/mc24_ift_leakage_operator_large_stats_transfer_synthetic_v1 \\",
        "  --split validation --resume",
        "",
        "# 3. Frozen V2 inference only (Station: iteration_00_anchor; C_dx/A: iteration_00_reference)",
        "python scripts/run_frozen_association_backbone.py --backbone v2 --q-over-p-mode 0 \\",
        "  --frozen-output outputs/mc24_v3_expanded_trainval_v2_bce_control_v1 \\",
        "  --synthetic-manifest outputs/mc24_ift_station_mode_large_stats_transfer_synthetic_v1/synthetic_corpus_manifest.json \\",
        "  --split train --payload-id iteration_00_anchor \\",
        "  --output-dir outputs/mc24_ift_station_mode_large_stats_transfer_v2_backbone_train_v1",
        "python scripts/run_frozen_association_backbone.py --backbone v2 --q-over-p-mode 0 \\",
        "  --frozen-output outputs/mc24_v3_expanded_trainval_v2_bce_control_v1 \\",
        "  --synthetic-manifest outputs/mc24_ift_station_mode_large_stats_transfer_synthetic_v1/synthetic_corpus_manifest.json \\",
        "  --split validation --payload-id iteration_00_anchor \\",
        "  --output-dir outputs/mc24_ift_station_mode_large_stats_transfer_v2_backbone_validation_v1",
        "python scripts/run_frozen_association_backbone.py --backbone v2 --q-over-p-mode 0 \\",
        "  --frozen-output outputs/mc24_v3_expanded_trainval_v2_bce_control_v1 \\",
        "  --synthetic-manifest outputs/mc24_ift_internal_cdx_large_stats_transfer_synthetic_v1/synthetic_corpus_manifest.json \\",
        "  --split train --payload-id iteration_00_reference \\",
        "  --output-dir outputs/mc24_ift_internal_cdx_large_stats_transfer_v2_backbone_train_v1",
        "python scripts/run_frozen_association_backbone.py --backbone v2 --q-over-p-mode 0 \\",
        "  --frozen-output outputs/mc24_v3_expanded_trainval_v2_bce_control_v1 \\",
        "  --synthetic-manifest outputs/mc24_ift_internal_cdx_large_stats_transfer_synthetic_v1/synthetic_corpus_manifest.json \\",
        "  --split validation --payload-id iteration_00_reference \\",
        "  --output-dir outputs/mc24_ift_internal_cdx_large_stats_transfer_v2_backbone_validation_v1",
        "python scripts/run_frozen_association_backbone.py --backbone v2 --q-over-p-mode 0 \\",
        "  --frozen-output outputs/mc24_v3_expanded_trainval_v2_bce_control_v1 \\",
        "  --synthetic-manifest outputs/mc24_ift_leakage_operator_large_stats_transfer_synthetic_v1/synthetic_corpus_manifest.json \\",
        "  --split train --payload-id iteration_00_reference \\",
        "  --output-dir outputs/mc24_ift_leakage_operator_large_stats_transfer_v2_backbone_train_v1",
        "python scripts/run_frozen_association_backbone.py --backbone v2 --q-over-p-mode 0 \\",
        "  --frozen-output outputs/mc24_v3_expanded_trainval_v2_bce_control_v1 \\",
        "  --synthetic-manifest outputs/mc24_ift_leakage_operator_large_stats_transfer_synthetic_v1/synthetic_corpus_manifest.json \\",
        "  --split validation --payload-id iteration_00_reference \\",
        "  --output-dir outputs/mc24_ift_leakage_operator_large_stats_transfer_v2_backbone_validation_v1",
        "",
        "# 4. Station Mode closures: truth-selected + frozen-V2 route-selected",
        "#    --calibration-mode station --cdx-fixed-by isolation_zero --declared-unmodeled-cdx-um 0",
        "#    --prior-sigma ift_dz_mm:5.0 --observation-kind anchor_selected_field_edge",
        "#    --observation-statistics physical_edge_deduplicated --require-mode-valid",
        "#    --capture-criteria outputs/mc24_ift_5dof_survey_dz_capture_criteria_train_v1/capture_criteria.json",
        "#    target iteration_00_reference from anchor iteration_00_anchor",
        "",
        "# 5. C_dx Mode closures: --only-parameters C_dx --calibration-mode ift_internal",
        "#    --require-mode-valid; quote statistical sigma separately from frozen B systematic",
        "#    evaluate C_dx row of the frozen 2-D capture JSON; do not retune it",
        "",
        "# 6. Remeasure A on the leakage Jacobian bank; score 10% vs freeze; never update A",
        "python scripts/audit_hierarchical_v1_leakage.py ...  # train+validation, truth and route",
        "python scripts/audit_leakage_operator_stability.py --do-not-retune ...",
        "",
        "# 7. Score transfer and emit current-corpus vs large-stats matrix",
        "python scripts/score_calibration_mode_transfer.py ...",
        f"python scripts/report_calibration_mode_transfer_matrix.py --current-report {CURRENT_REPORT} \\",
        "  --transfer-report outputs/mc24_ift_calibration_mode_transfer_large_stats_v1/transfer_report.json \\",
        "  --a-stability outputs/mc24_ift_calibration_mode_transfer_large_stats_v1/A_stability.json \\",
        "  --output outputs/mc24_ift_calibration_mode_transfer_large_stats_v1/transfer_matrix.json",
        "",
        "# If both modes pass frozen criteria on new train AND independent validation",
        "# with no new source-dependent failure: MC transfer gate passed.",
        "# Otherwise localize the failing mode; do not retune V2/capture/A or add DoF.",
        "# Real FASER data Operating Protocol V1 dry-run is forbidden until the gate passes.",
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("complete", "corpus", "commands"), default="complete")
    args = parser.parse_args()
    if args.stage == "commands":
        print("\n".join(_commands()))
        return
    audit_argv = [
        PYTHON,
        str(ROOT / "scripts" / "audit_calibration_mode_transfer_completion.py"),
        "--output",
        str(ROOT / "outputs" / "mc24_ift_calibration_mode_transfer_large_stats_v1" / "physical_completion.json"),
    ]
    if args.stage == "corpus":
        audit_argv.append("--require-complete")
    (ROOT / "outputs" / "mc24_ift_calibration_mode_transfer_large_stats_v1").mkdir(parents=True, exist_ok=True)
    _run(audit_argv)
    if args.stage == "complete":
        return
    for bank in BANKS.values():
        _run(
            [
                PYTHON,
                str(ROOT / "scripts" / "build_6dof_pilot_physical_corpus.py"),
                "--iteration-manifest",
                str(bank / "iteration_manifest.json"),
            ]
        )


if __name__ == "__main__":
    main()
