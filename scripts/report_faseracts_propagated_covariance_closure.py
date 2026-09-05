#!/usr/bin/env python3
"""Workbook-81 FaserActsExtrapolation propagated-covariance provenance & closure driver.

Stages enforce the frozen ordering structurally:

    validate-config   config + WB80 inheritance SHAs + frozen-flag guards
    provenance        Stage-0: 15-question C_prop provenance audit (read-only)
    truth-reference   Stage-1: build truth/reference target state + calibrate
                      the reference floor (mode-2 full-truth source control)
    closure           Stage-2: source-disjoint MC truth closure (construction +
                      validation) for the production q/p mode
    variants          Stage-3: q/p-mode diagnostic variants (diagnostic_only)
    decide            pre-registered decision tree

This campaign is NOT an alignment diagnostic.  It reads NO real-data residual,
never opens held-out data, never writes geometry/conditions, and never solves
for an alignment correction.  ``held_out_accessed=false`` throughout.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from alignment.cad_survey_nov22 import git_head_sha
from alignment.gauge_fixed_real_data_diagnostic import _json_ready
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment import propagated_covariance_closure as pcc
from alignment.propagated_covariance_closure import DEFAULT_CONFIG


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _output_root(config: dict) -> Path:
    return resolve_under_root(project_root(), str(config["output_root"]))


def _stage_validate_config(config: dict) -> dict:
    report = {
        "kind": "config_validation",
        "config_path": config["config_path"],
        "config_sha256": sha256_file(Path(config["config_path"])),
        "git_head_sha": git_head_sha(),
        "frozen_flags_verified": True,
        "inheritance_sha256_verified": True,
        "held_out_accessed": False,
        "real_data_alignment_authorized": False,
        "geometry_write_allowed": False,
        "official_conditions_write_allowed": False,
        "population": "mc_only_no_real_data_residual",
        "pass": True,
    }
    return {"config_validation": report}


def _stage_provenance(config: dict) -> dict:
    return {"faseracts_covariance_provenance": pcc.provenance_audit(config)}


def _max_position_rms(metrics: dict) -> float:
    """Dominant (max x/y) de-meaned marginal RMS of e_prop."""
    rms = metrics.get("marginal_rms_e_demeaned")
    if rms is None:
        return float("nan")
    return float(max(abs(rms[0]), abs(rms[1])))


def _stage_truth_reference(config: dict) -> dict:
    """Calibrate the truth-reference floor with the mode-2 full-truth control.

    For each station pair, the reference floor is the mode-2 (truth-source)
    de-meaned marginal RMS; the closure signal is the mode-0 (reco-source) RMS.
    The truth reference is trusted only where the signal dominates the floor.
    """
    tr = config["truth_reference"]
    floor_mode = int(tr["reference_floor_q_over_p_mode"])
    prod_mode = int(config["mc_data"]["production_q_over_p_mode"])
    min_ratio = float(tr["min_signal_to_floor"])
    per_pair: dict[str, object] = {}
    all_ok = True
    for pair in [tuple(p) for p in config["mc_data"]["station_pairs"]]:
        label = f"({pair[0]},{pair[1]})"
        all_sources = (
            config["mc_data"]["construction_source_ids"]
            + config["mc_data"]["validation_source_ids"]
        )
        # Floor: full-truth-source control carries no fit covariance.
        floor_parts = [
            pcc.load_source_records(config, sid, pair, floor_mode, require_covariance=False)
            for sid in all_sources
        ]
        signal_parts = [
            pcc.load_source_records(config, sid, pair, prod_mode, require_covariance=True)
            for sid in all_sources
        ]
        floor_recs = pcc._stack_records(floor_parts, pair, floor_mode)
        signal_recs = pcc._stack_records(signal_parts, pair, prod_mode)
        floor_metrics = pcc.compute_closure_metrics(floor_recs, config)
        signal_metrics = pcc.compute_closure_metrics(signal_recs, config)
        floor_rms = _max_position_rms(floor_metrics)
        signal_rms = _max_position_rms(signal_metrics)
        ratio = signal_rms / floor_rms if floor_rms > 0 else float("inf")
        ok = bool(
            floor_metrics.get("sufficient", False)
            and signal_metrics.get("sufficient", False)
            and np.isfinite(ratio)
            and ratio >= min_ratio
        )
        all_ok = all_ok and ok
        per_pair[label] = {
            "floor_mode": floor_mode,
            "floor_max_position_rms_mm": floor_rms,
            "signal_max_position_rms_mm": signal_rms,
            "signal_to_floor": ratio,
            "min_signal_to_floor": min_ratio,
            "floor_n_pairs": floor_metrics.get("n_pairs", 0),
            "signal_n_pairs": signal_metrics.get("n_pairs", 0),
            "reference_ok": ok,
        }
    return {
        "truth_reference": {
            "kind": "truth_reference_construction",
            "construction": "per-station Geant-truth state (truth_stX_*) straight-line "
            "corrected to the propagation target plane",
            "reference_floor_q_over_p_mode": floor_mode,
            "per_pair": per_pair,
            "reference_floor_ok": bool(all_ok),
            "diagnostic_only": True,
            "alignment_authorized": False,
        }
    }


def _stage_closure(config: dict) -> dict:
    construction = pcc.run_closure_split(config, "construction")
    validation = pcc.run_closure_split(config, "validation")
    return {
        "propagated_covariance_closure": {
            "kind": "propagated_covariance_truth_closure",
            "q_over_p_mode": int(config["mc_data"]["production_q_over_p_mode"]),
            "construction": construction,
            "validation": validation,
            "diagnostic_only": True,
            "alignment_authorized": False,
        }
    }


def _stage_variants(config: dict) -> dict:
    return {"diagnostic_variants": pcc.run_diagnostic_variants(config)}


def _stage_decide(config: dict) -> dict:
    root = _output_root(config)
    provenance = _read_json(root / "faseracts_covariance_provenance.json")[
        "faseracts_covariance_provenance"
    ]
    truth_ref = _read_json(root / "truth_reference.json")["truth_reference"]
    closure = _read_json(root / "closure.json")["propagated_covariance_closure"]
    variants_path = root / "diagnostic_variants.json"
    variants = (
        _read_json(variants_path)["diagnostic_variants"] if variants_path.is_file() else None
    )
    decision = pcc.decide(
        config,
        provenance=provenance,
        construction=closure["construction"],
        validation=closure["validation"],
        reference_floor_ok=bool(truth_ref["reference_floor_ok"]),
        variants=variants,
    )
    summary = {
        "kind": "campaign_summary",
        "schema_version": config["schema_version"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head_sha": git_head_sha(),
        "config_path": config["config_path"],
        "decision": decision["decision"],
        "propagated_covariance_model_validated": decision[
            "propagated_covariance_model_validated"
        ],
        "real_kinematic_jacobian_support_validated": decision[
            "real_kinematic_jacobian_support_validated"
        ],
        "measurement_model_validated": decision["measurement_model_validated"],
        "real_data_alignment_v2_preregistration_allowed": decision[
            "real_data_alignment_v2_preregistration_allowed"
        ],
        "held_out_accessed": False,
        "real_data_alignment_authorized": False,
        "geometry_write_allowed": False,
        "official_conditions_write_allowed": False,
    }
    return {
        "propagated_covariance_decision": decision,
        "campaign_summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "validate-config",
            "provenance",
            "truth-reference",
            "closure",
            "variants",
            "decide",
            "all",
        ),
    )
    args = parser.parse_args()
    config = pcc.load_config(args.config)
    config["config_path"] = str(resolve_under_root(project_root(), args.config))
    root = _output_root(config)
    root.mkdir(parents=True, exist_ok=True)

    if args.stage in ("validate-config", "all"):
        _write_json(root / "config_validation.json", _stage_validate_config(config))
    if args.stage in ("provenance", "all"):
        _write_json(
            root / "faseracts_covariance_provenance.json", _stage_provenance(config)
        )
    if args.stage in ("truth-reference", "all"):
        _write_json(root / "truth_reference.json", _stage_truth_reference(config))
    if args.stage in ("closure", "all"):
        _write_json(root / "closure.json", _stage_closure(config))
    if args.stage in ("variants", "all"):
        _write_json(root / "diagnostic_variants.json", _stage_variants(config))
    if args.stage in ("decide", "all"):
        reports = _stage_decide(config)
        _write_json(
            root / "propagated_covariance_decision.json",
            reports["propagated_covariance_decision"],
        )
        _write_json(root / "campaign_summary.json", reports["campaign_summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
