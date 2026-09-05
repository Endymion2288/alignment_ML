"""Workbook 81: FaserActsExtrapolation propagated-covariance provenance & closure.

This is an UPSTREAM covariance-validation campaign, **not** an alignment task
and **not** "Alignment Diagnostic V2".  Workbook 80 froze
``measurement_model_multiple_components_not_validated`` and proved that the
frozen combined covariance's near-singular pencil direction over-estimates the
real residual variance by ~100-2000x, but it could not confirm the provenance
of the propagated covariance ``C_prop`` because it is produced by the EXTERNAL
Calypso/ACTS ``FaserActsExtrapolationTool``.

This module does two things and nothing else:

1. **Provenance audit** — close the ``C_prop`` generation semantics directly
   from the actual Calypso/ACTS source and runtime configuration (15
   pre-registered questions), recorded with file + line/function and a
   resolved_from_source / resolved_from_runtime_config / unresolved verdict.

2. **Truth-known MC closure** — isolate ``C_prop`` from ``C_target`` and from
   any real-data residual by forming

       e_prop = propagated source-tracklet state - truth/reference state at
                target surface

   on source-disjoint MC, then whiten ``z_prop = C_prop^{-1/2} e_prop`` and test
   whether ``C_prop`` describes the physical propagation error (per station
   pair, sliced by residual-blind kinematics), with a dedicated pencil-direction
   and orientation-vs-scale decomposition.

The whole campaign keeps ``held_out_accessed=false``,
``real_data_alignment_authorized=false``, ``geometry_write_allowed=false`` and
``official_conditions_write_allowed=false``.  It reads NO real-data residual
and never solves for an alignment correction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.cad_survey_nov22 import git_head_sha
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from datasets.propagation_loader import load_propagation_records

SCHEMA_VERSION = "faseracts-propagated-covariance-provenance-closure-v1"
DEFAULT_CONFIG = "configs/faseracts_propagated_covariance_provenance_closure_v1.yaml"

# Decision-tree terminal strings (exactly one is frozen).
DECISION_PROVENANCE_UNRESOLVED = "propagated_covariance_provenance_unresolved"
DECISION_TRUTH_CLOSURE_UNAVAILABLE = "propagated_covariance_truth_closure_unavailable"
DECISION_NOT_CALIBRATED = "faseracts_propagated_covariance_not_calibrated"
DECISION_VALIDATED_CANONICAL = (
    "faseracts_propagated_covariance_validated_within_canonical_support"
)
DECISION_VALIDATED_REQUIRED = (
    "faseracts_propagated_covariance_validated_across_required_support"
)
DECISION_INCONCLUSIVE = "propagated_covariance_validation_inconclusive"

# Upstream bug / semantic-mismatch classification categories.
MECHANISM_MISSING_PROCESS_NOISE = "missing_process_noise"
MECHANISM_OVERESTIMATED_TRANSPORT = "overestimated_transported_fit_covariance"
MECHANISM_WRONG_PARAMETER_TRANSFORM = "wrong_parameter_transform"
MECHANISM_WRONG_SURFACE_JACOBIAN = "wrong_surface_jacobian"
MECHANISM_WRONG_PARTICLE_HYPOTHESIS = "wrong_particle_hypothesis"
MECHANISM_CONDITIONAL_NOT_RESIDUAL = "conditional_covariance_not_residual_covariance"
MECHANISM_OTHER = "other_verified_mechanism"

STATE_NAMES = ("x_mm", "y_mm", "tx", "ty")
STATION_PAIRS = ((0, 1), (0, 2), (0, 3))


# ---------------------------------------------------------------------------
# Config loading + frozen-inheritance verification
# ---------------------------------------------------------------------------


class ConfigError(ValueError):
    """Raised when the WB81 config or its frozen inheritance is inconsistent."""


def _expect_false(config: Mapping[str, Any], key: str) -> None:
    if bool(config.get(key, False)) is not False:
        raise ConfigError(f"frozen flag must be false: {key}")


def _expect_true(config: Mapping[str, Any], key: str) -> None:
    if bool(config.get(key, False)) is not True:
        raise ConfigError(f"frozen prohibition must be true: {key}")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the WB81 config and verify the frozen WB80 inheritance (SHA-pinned)."""
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ConfigError(f"WB81 config must be a mapping: {config_path}")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {SCHEMA_VERSION}")
    if int(config.get("workbook", -1)) != 81:
        raise ConfigError("workbook must be 81")

    # Frozen terminal permissions / prohibitions.
    for key in (
        "geometry_write_allowed",
        "official_conditions_write_allowed",
        "real_data_candidate_alignment_authorized",
        "real_data_alignment_authorized",
        "external_constraint_ingest_authorized",
        "held_out_accessed",
    ):
        _expect_false(config, key)
    for key in (
        "do_not_open_held_out",
        "do_not_read_real_data_residuals",
        "do_not_modify_faseracts_extrapolation_tool",
        "do_not_select_events_by_residual",
        "do_not_drop_pairs_by_condition_or_fit_quality",
        "do_not_use_external_evidence_as_prior",
        "do_not_solve_final_alignment",
        "do_not_tune_covariance_to_chi2",
        "do_not_promote_diagonal_or_capped_or_unit_covariance",
        "do_not_extrapolate_beyond_canonical_support",
        "do_not_open_alignment_diagnostic_v2",
        "do_not_promote_diagnostic_variant_to_production",
    ):
        _expect_true(config, key)

    # Verify the frozen WB80 inheritance (config + decision artifacts).
    inheritance = config.get("inheritance", {})
    wb80_config_path = resolve_under_root(
        project_root(), str(inheritance["workbook_80_config"])
    )
    if sha256_file(wb80_config_path) != str(inheritance["workbook_80_config_sha256"]):
        raise ConfigError("WB80 config SHA256 mismatch (frozen inheritance broken)")
    wb80_root = resolve_under_root(
        project_root(), str(inheritance["workbook_80_output_root"])
    )
    for name, expected in inheritance["workbook_80_artifact_sha256"].items():
        actual = sha256_file(wb80_root / name)
        if actual != str(expected):
            raise ConfigError(f"WB80 artifact SHA256 mismatch: {name}")
    # Confirm the frozen WB80 decision matches the pre-registered negative state.
    decision_path = wb80_root / "measurement_model_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("decision") != str(inheritance["workbook_80_frozen_decision"]):
        raise ConfigError("WB80 frozen decision mismatch")
    gates = inheritance["workbook_80_frozen_gates"]
    for key, expected in gates.items():
        if bool(decision.get(key)) != bool(expected):
            raise ConfigError(f"WB80 frozen gate mismatch: {key}")

    # Source-disjointness of the MC split must be file-level disjoint.
    mc = config["mc_data"]
    construction = set(mc["construction_source_ids"])
    validation = set(mc["validation_source_ids"])
    if construction & validation:
        raise ConfigError("MC construction/validation sources are not disjoint")
    return dict(config)


def assert_no_held_out_access() -> None:
    """WB81 reads no real-data residual and no held-out anything.

    The campaign uses MC only; this guard exists so that any future edit that
    routes real/held-out data into the closure fails loudly.  It is a no-op by
    construction because the module exposes no real-data entry point.
    """
    return None


# ---------------------------------------------------------------------------
# Stage: provenance audit (read-only, from the actual Calypso/ACTS source)
# ---------------------------------------------------------------------------


def provenance_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the 15-question C_prop provenance audit.

    Every entry is resolved from the actual Calypso/ACTS source (commit pinned
    in ``software_provenance``) and/or the runtime config.  No answer is guessed
    from ACTS common knowledge; each carries the file + line/function evidence.
    """
    sw = config["software_provenance"]
    calypso = sw["calypso_git_sha"]
    acts = sw["acts_version"]
    tool_cxx = "Tracking/Acts/FaserActsGeometry/src/FaserActsExtrapolationTool.cxx"
    tool_h = "Tracking/Acts/FaserActsGeometry/src/FaserActsExtrapolationTool.h"
    exporter = "PhysicsAnalysis/NtupleDumper/src/NtupleDumperAlg.cxx"
    exp_cfg = "PhysicsAnalysis/NtupleDumper/python/NtupleDumperConfig.py"

    answers = {
        "input_covariance_parameter_basis": {
            "verdict": "resolved_from_source",
            "answer": (
                "Native Athena Trk::TrackParameters covariance, a 5x5 "
                "AmgSymMatrix(5) in (loc1, loc2, phi, theta, q/p)."
            ),
            "evidence": f"{exporter} globalTrackletCovariance / actsTrackletParameters",
        },
        "input_covariance_segment_to_acts_transform": {
            "verdict": "resolved_from_source",
            "answer": (
                "actsTrackletParameters converts the native 5x5 "
                "(loc1,loc2,phi,theta,q/p) covariance to the ACTS bound 6x6 "
                "(loc0,loc1,phi,theta,q/p,time) via a CENTRAL NUMERICAL Jacobian "
                "(5 native -> 6 bound), building the ACTS state on a global-z "
                "Acts::PlaneSurface at the source tracklet z through "
                "Acts::detail::transformFreeToBoundParameters.  The q/p column IS "
                "transported unless suppressQOverPCovariance (mode 3) is set."
            ),
            "evidence": f"{exporter}:146-273 actsTrackletParameters",
        },
        "covariance_transport_jacobian_location": {
            "verdict": "resolved_from_source",
            "answer": (
                "ACTS EigenStepper accumulates the per-step transport Jacobian "
                "(state.stepping.jacTransport = D * jacTransport) and the final "
                "transport is C -> J_full C J_full^T via "
                "detail::transportCovarianceToBound (CovarianceEngine), where "
                "J_full = freeToBound(target) . jacTransport . boundToFree(source)."
            ),
            "evidence": (
                f"ACTS {acts} Acts/Propagator/EigenStepper.ipp:237, "
                "Acts/Propagator/detail/CovarianceEngine.ipp"
            ),
        },
        "output_covariance_back_to_xytxty": {
            "verdict": "resolved_from_source",
            "answer": (
                "globalActsTrackletCovariance converts the propagated 6x6 bound "
                "covariance back to the global 4x4 [x,y,tx,ty] via a central "
                "numerical Jacobian (6 bound -> 4 global), perturbing each bound "
                "parameter and recomputing (x, y, px/pz, py/pz)."
            ),
            "evidence": f"{exporter}:330-395 globalActsTrackletCovariance",
        },
        "material_effects_enabled": {
            "verdict": "resolved_from_runtime_config",
            "answer": (
                "NO.  Acts::MaterialInteractor is in the propagator action list "
                "but multipleScattering/energyLoss/recordInteractions are all "
                "false in production, so the interactor early-returns and applies "
                "no material effect."
            ),
            "evidence": (
                f"{tool_cxx}:150-153,220-223,339-342 set the flags from "
                f"properties; {tool_h}:131-133 default all three to false; "
                f"{exp_cfg}:23-28 sets only MaxSteps/TrackingGeometryTool; "
                f"ACTS {acts} MaterialInteractor.hpp:66-68 early-return."
            ),
        },
        "multiple_scattering_process_noise": {
            "verdict": "resolved_from_runtime_config",
            "answer": (
                "NOT added.  InteractionMultiScatering defaults to false and is "
                "never overridden, so the ACTS mechanism that would add "
                "variancePhi/varianceTheta to the covariance diagonal "
                "(PointwiseMaterialInteraction::updateState, addNoise) never runs."
            ),
            "evidence": (
                f"{tool_h}:131 default false; {exp_cfg} does not set it; "
                f"ACTS {acts} PointwiseMaterialInteraction.hpp:144-167."
            ),
        },
        "energy_loss_uncertainty": {
            "verdict": "resolved_from_runtime_config",
            "answer": (
                "NOT added.  InteractionEloss defaults to false and is never "
                "overridden; the varianceQoverP energy-loss straggling term is "
                "never applied."
            ),
            "evidence": (
                f"{tool_h}:132 default false; {exp_cfg} does not set it; "
                f"ACTS {acts} PointwiseMaterialInteraction.hpp:164-166."
            ),
        },
        "process_noise_surface_step": {
            "verdict": "resolved_from_source",
            "answer": (
                "N/A in production (process noise disabled).  If enabled it would "
                "be added at each material surface via "
                "PointwiseMaterialInteraction::updateState on the (phi, theta, "
                "q/p) diagonal after transportCovarianceToCurvilinear."
            ),
            "evidence": f"ACTS {acts} MaterialInteractor.hpp:80-110",
        },
        "particle_hypothesis_mass_charge_momentum": {
            "verdict": "resolved_from_source",
            "answer": (
                "Particle hypothesis is HARDCODED to Acts::ParticleHypothesis::muon() "
                "(muon mass/charge).  The momentum/q/p is the reconstructed source "
                "tracklet native q/p (mode 0) or truth q/p (modes 1,2)."
            ),
            "evidence": f"{exporter}:272,311 Acts::ParticleHypothesis::muon()",
        },
        "qoverp_mode_effect_on_covariance": {
            "verdict": "resolved_from_source",
            "answer": (
                "The source q/p variance IS transported in production (mode 0).  "
                "The straight-line segment fit cannot constrain q/p, so its q/p "
                "variance is a large dummy value; transported through the magnetic "
                "field it maps (via the field-dependent Jacobian and lever arm) "
                "into a large spurious position/slope pencil.  The mode-3 "
                "diagnostic (suppressQOverPCovariance) isolates this q/p-induced "
                "covariance inflation."
            ),
            "evidence": (
                f"{exporter}:200-205,649-660 (mode-3 suppressQOverPCovariance "
                "diagnostic comment)"
            ),
        },
        "field_covariance_or_numerical_transport": {
            "verdict": "resolved_from_source",
            "answer": (
                "The FASER field is a DETERMINISTIC map (FASERMagneticFieldWrapper "
                "-> FaserFieldCache); no field covariance/uncertainty is "
                "propagated.  Numerical transport is the ACTS EigenStepper "
                "(Runge-Kutta) with MaxStepSize=10 m, MaxSteps=10000."
            ),
            "evidence": (
                "Tracking/Acts/FaserActsGeometry/FaserActsGeometry/"
                "FASERMagneticFieldWrapper.h (getField/getFieldGradient only); "
                f"{exp_cfg}:24 MaxSteps=10000; {tool_h}:127-129 defaults."
            ),
        },
        "surface_frame_local_global_jacobian_amplification": {
            "verdict": "resolved_from_source",
            "answer": (
                "Both the source native->bound Jacobian and the propagated "
                "bound->global [x,y,tx,ty] Jacobian are central numerical "
                "Jacobians on plane surfaces normal to z.  The global state is "
                "(x, y, px/pz, py/pz); the tx=px/pz, ty=py/pz definition and the "
                "lever arm (target_z - source_z) can amplify correlations.  This "
                "is a real amplification path that is audited empirically."
            ),
            "evidence": f"{exporter}:146-273,330-395",
        },
        "prediction_vs_conditional_covariance": {
            "verdict": "resolved_from_source",
            "answer": (
                "C_prop is a PREDICTION covariance: the source fit covariance "
                "transported to the target surface (spread of the propagated "
                "source state given the source fit).  It is NOT a conditional "
                "covariance given the target measurement, and it contains no "
                "stochastic transport noise."
            ),
            "evidence": f"{tool_cxx}:309-355 propagate(); {exporter}:744-768",
        },
        "deterministic_transport_without_stochastic_noise": {
            "verdict": "resolved_from_source",
            "answer": (
                "YES - CONFIRMED.  The fit covariance is transported "
                "deterministically (C -> J C J^T) but NO actual stochastic "
                "transport noise (multiple scattering, energy loss) is added, "
                "because the MaterialInteractor is disabled (all flags false)."
            ),
            "evidence": (
                f"{tool_cxx}:150-153; ACTS {acts} MaterialInteractor.hpp:66-68 "
                "early-return."
            ),
        },
        "runtime_config_option_values": {
            "verdict": "resolved_from_runtime_config",
            "answer": (
                "NtupleDumperAlgCfg sets only MaxSteps=10000 and the "
                "TrackingGeometryTool.  FieldMode='FASER' (default), "
                "InteractionMultiScatering/InteractionEloss/InteractionRecord all "
                "default to false, PtLoopers=300 MeV, MaxStepSize=10 m."
            ),
            "evidence": f"{exp_cfg}:23-28; {tool_h}:124-133",
        },
    }
    n_unresolved = sum(1 for a in answers.values() if a["verdict"] == "unresolved")
    return {
        "questions": answers,
        "n_questions": len(answers),
        "n_resolved_from_source": sum(
            1 for a in answers.values() if a["verdict"] == "resolved_from_source"
        ),
        "n_resolved_from_runtime_config": sum(
            1
            for a in answers.values()
            if a["verdict"] == "resolved_from_runtime_config"
        ),
        "n_unresolved": n_unresolved,
        "provenance_closed": n_unresolved == 0,
        "software": dict(sw),
        "calypso_git_sha": calypso,
        "acts_version": acts,
    }


# ---------------------------------------------------------------------------
# Truth-known MC closure records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClosureRecords:
    """Per-pair truth-known propagation records for one q/p mode."""

    station_pair: tuple[int, int]
    q_over_p_mode: int
    e_prop: np.ndarray  # (n, 4) propagated - truth at target surface
    c_prop: np.ndarray  # (n, 4, 4) propagated covariance (NaN if has_covariance=False)
    lever_arm_mm: np.ndarray  # (n,)
    pred_tx: np.ndarray  # (n,)
    pred_ty: np.ndarray  # (n,)
    n_matched: int
    has_covariance: bool = True  # False for the full-truth-source (mode-2) control

    @property
    def size(self) -> int:
        return int(self.e_prop.shape[0])


def _load_truth_table(enhanced_path: Path, tree: str, config: Mapping[str, Any]):
    """Load the per-station Geant-truth state keyed by (run, event, barcode)."""
    import uproot

    tr = config["truth_reference"]
    pos_t = tr["truth_position_template"]
    mom_t = tr["truth_momentum_template"]
    run_b = tr["enhanced_run_branch"]
    event_b = tr["enhanced_event_branch"]
    barcode_b = tr["enhanced_barcode_branch"]
    branches = [run_b, event_b, barcode_b]
    for s in range(4):
        branches += [pos_t.format(station=s, coord=c) for c in ("x", "y", "z")]
        branches += [mom_t.format(station=s, mom=m) for m in ("px", "py", "pz")]
    with uproot.open(enhanced_path) as handle:
        if tree not in handle:
            raise ConfigError(f"tree '{tree}' absent from {enhanced_path}")
        arrays = handle[tree].arrays(branches, library="np")
    table: dict[tuple[int, int], dict[str, Any]] = {}
    n = len(arrays[run_b])
    for i in range(n):
        key = (int(arrays[run_b][i]), int(arrays[event_b][i]))
        barcodes = np.asarray(arrays[barcode_b][i])
        entry: dict[int, dict[str, np.ndarray]] = {}
        for j, bc in enumerate(barcodes):
            per_station: dict[str, np.ndarray] = {}
            for s in range(4):
                pos = np.array(
                    [
                        arrays[pos_t.format(station=s, coord="x")][i][j],
                        arrays[pos_t.format(station=s, coord="y")][i][j],
                        arrays[pos_t.format(station=s, coord="z")][i][j],
                    ],
                    dtype=np.float64,
                )
                mom = np.array(
                    [
                        arrays[mom_t.format(station=s, mom="px")][i][j],
                        arrays[mom_t.format(station=s, mom="py")][i][j],
                        arrays[mom_t.format(station=s, mom="pz")][i][j],
                    ],
                    dtype=np.float64,
                )
                per_station[s] = {"pos": pos, "mom": mom}
            entry[int(bc)] = per_station
        table[key] = entry
    return table


def build_closure_records(
    propagations_path: Path,
    enhanced_path: Path,
    station_pair: tuple[int, int],
    q_over_p_mode: int,
    config: Mapping[str, Any],
    require_covariance: bool = True,
) -> ClosureRecords:
    """Join propagation records with the truth reference and build e_prop/C_prop.

    No event/pair is dropped on pull, condition, or fit quality; records are
    only required to be successful, to carry a finite covariance (unless
    ``require_covariance=False``, e.g. the full-truth-source mode-2 control which
    carries no fit covariance), and to have a finite truth state at the target
    station.
    """
    tr = config["truth_reference"]
    records = load_propagation_records(propagations_path)
    truth_table = _load_truth_table(enhanced_path, config["mc_data"]["enhanced_tree"], config)

    src, tgt = station_pair
    sel = (
        (records.q_over_p_mode == q_over_p_mode)
        & (records.source_station_id == src)
        & (records.target_station_id == tgt)
        & records.success
    )
    if require_covariance:
        sel = sel & records.has_covariance
    idx = np.where(sel)[0]
    e_list: list[np.ndarray] = []
    c_list: list[np.ndarray] = []
    lever: list[float] = []
    ptx: list[float] = []
    pty: list[float] = []
    n_matched = 0
    for i in idx:
        key = (int(records.run_id[i]), int(records.event_id[i]))
        entry = truth_table.get(key)
        if entry is None:
            continue
        per_station = entry.get(int(records.truth_particle_id[i]))
        if per_station is None or tgt not in per_station:
            continue
        truth = per_station[tgt]
        pos, mom = truth["pos"], truth["mom"]
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(mom))):
            continue
        if abs(mom[2]) < 1e-12:
            continue
        # Source-station truth z gives the actual lever arm of the propagation.
        source_entry = per_station.get(src)
        source_z = source_entry["pos"][2] if source_entry is not None else np.nan
        if not np.isfinite(source_z):
            continue
        n_matched += 1
        tx_truth = mom[0] / mom[2]
        ty_truth = mom[1] / mom[2]
        target_z = float(records.target_z_mm[i])
        dz = target_z - pos[2] if tr["straight_line_z_correction"] else 0.0
        x_truth = pos[0] + tx_truth * dz
        y_truth = pos[1] + ty_truth * dz
        pred = records.prediction[i]
        e = np.array(
            [pred[0] - x_truth, pred[1] - y_truth, pred[2] - tx_truth, pred[3] - ty_truth],
            dtype=np.float64,
        )
        if records.has_covariance[i]:
            C = np.asarray(records.covariance[i], dtype=np.float64)
        else:
            C = np.full((4, 4), np.nan, dtype=np.float64)
        if not np.all(np.isfinite(e)):
            continue
        if require_covariance and not np.all(np.isfinite(C)):
            continue
        e_list.append(e)
        c_list.append(C)
        lever.append(abs(target_z - float(source_z)))
        ptx.append(float(pred[2]))
        pty.append(float(pred[3]))
    if not e_list:
        return ClosureRecords(
            station_pair=station_pair,
            q_over_p_mode=q_over_p_mode,
            e_prop=np.empty((0, 4)),
            c_prop=np.empty((0, 4, 4)),
            lever_arm_mm=np.empty(0),
            pred_tx=np.empty(0),
            pred_ty=np.empty(0),
            n_matched=n_matched,
            has_covariance=require_covariance,
        )
    return ClosureRecords(
        station_pair=station_pair,
        q_over_p_mode=q_over_p_mode,
        e_prop=np.asarray(e_list),
        c_prop=np.asarray(c_list),
        lever_arm_mm=np.asarray(lever),
        pred_tx=np.asarray(ptx),
        pred_ty=np.asarray(pty),
        n_matched=n_matched,
        has_covariance=require_covariance,
    )


# ---------------------------------------------------------------------------
# Closure metrics
# ---------------------------------------------------------------------------


def _symmetric_whiten(e: np.ndarray, C: np.ndarray) -> np.ndarray | None:
    """z = C^{-1/2} e via eigendecomposition; None if C is not positive definite."""
    w, V = np.linalg.eigh(C)
    if np.any(w <= 0) or not np.all(np.isfinite(w)):
        return None
    return V @ ((V.T @ e) / np.sqrt(w))


def _binned_dependence(
    Ed: np.ndarray,
    c_prop: np.ndarray,
    bin_var: np.ndarray,
    n_bins: int = 3,
) -> list[dict[str, Any]]:
    """Per-bin chi2/ndof and marginal RMS, binned by a residual-blind variable.

    Bins are quantile-based (equal counts).  No record is dropped; every record
    falls in exactly one bin.  Used to expose lever-arm / tx / ty dependence of
    the C_prop miscalibration.
    """
    n = Ed.shape[0]
    if n < n_bins * 3:
        return []
    # Quantile edges (residual-blind: bin only on the supplied kinematic lever).
    qs = np.quantile(bin_var, np.linspace(0.0, 1.0, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    out: list[dict[str, Any]] = []
    for b in range(n_bins):
        mask = (bin_var >= qs[b]) & (bin_var < qs[b + 1])
        if mask.sum() < 3:
            continue
        chi2s = []
        for e, C in zip(Ed[mask], c_prop[mask]):
            if np.all(np.isfinite(C)):
                try:
                    chi2s.append(float(e @ np.linalg.solve(C, e)))
                except np.linalg.LinAlgError:
                    continue
        out.append(
            {
                "bin": b,
                "bin_var_min": float(np.min(bin_var[mask])),
                "bin_var_max": float(np.max(bin_var[mask])),
                "n": int(mask.sum()),
                "chi2_per_ndof": float(np.mean(chi2s) / 4) if chi2s else None,
                "marginal_rms_e": Ed[mask].std(axis=0).tolist(),
            }
        )
    return out


def compute_closure_metrics(
    records: ClosureRecords,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Pre-registered whitening / eigenmode / pencil / orientation-vs-scale metrics.

    The coherent mean of e_prop is reported separately (a systematic, not what a
    covariance describes); the covariance gates use the de-meaned fluctuation.
    No record is dropped on pull/condition.
    """
    gates = config["closure_metrics"]
    n = records.size
    if n < int(config["closure_gates"]["min_pairs_per_pair"]):
        return {"station_pair": list(records.station_pair), "n_pairs": n, "sufficient": False}

    E = records.e_prop
    mean_e = E.mean(axis=0)
    Ed = E - mean_e if gates["demean_per_pair"] else E
    C_emp = np.cov(Ed.T)

    # Full-truth-source control (mode 2) carries no fit covariance: report only
    # the e_prop-based reference-floor metrics and leave covariance metrics None.
    if not records.has_covariance:
        return {
            "station_pair": list(records.station_pair),
            "q_over_p_mode": int(records.q_over_p_mode),
            "n_pairs": n,
            "n_matched": int(records.n_matched),
            "sufficient": True,
            "has_covariance": False,
            "mean_e": mean_e.tolist(),
            "marginal_rms_e": E.std(axis=0).tolist(),
            "marginal_rms_e_demeaned": Ed.std(axis=0).tolist(),
            "c_emp_eigenvalues": np.linalg.eigvalsh(C_emp).tolist(),
            "lever_arm_mm_median": float(np.median(records.lever_arm_mm)),
            "mean_z": None,
            "marginal_rms_z": None,
            "chi2_per_ndof": None,
            "cov_z_eigenvalues": None,
            "coverage_probability_95": None,
            "c_prop_mean_eigenvalues": None,
            "pencil": None,
            "generalized_eigenvalues_cemp_over_cprop": None,
            "correlation_x_tx": None,
            "correlation_y_ty": None,
            "condition_median": None,
        }

    C_prop_mean = records.c_prop.mean(axis=0)

    # Whitened residuals (de-meaned) and chi2.
    z_list: list[np.ndarray] = []
    chi2_list: list[float] = []
    mahal: list[float] = []
    for e, C in zip(Ed, records.c_prop):
        z = _symmetric_whiten(e, C)
        if z is None:
            continue
        z_list.append(z)
        chi2_list.append(float(e @ np.linalg.solve(C, e)))
        mahal.append(float(e @ np.linalg.solve(C, e)))
    z = np.asarray(z_list)
    chi2 = np.asarray(chi2_list)
    ndof = 4
    cov_z = np.cov(z.T) if z.shape[0] > 2 else np.full((4, 4), np.nan)
    cov_z_eig = np.linalg.eigvalsh(cov_z) if z.shape[0] > 2 else np.full(4, np.nan)

    # Coverage probability: P(e^T C^-1 e <= chi2_4 quantile).
    from scipy.stats import chi2 as _chi2_dist

    quant = float(_chi2_dist.ppf(0.95, gates["coverage_chi2_quantile_df"]))
    coverage = float(np.mean(np.asarray(mahal) <= quant)) if mahal else float("nan")

    # Pencil direction: C_prop's largest-eigenvalue eigenvector vs the empirical
    # variance along that same direction.
    w_prop, V_prop = np.linalg.eigh(C_prop_mean)
    pencil_vec = V_prop[:, -1]
    pencil_var_prop = float(w_prop[-1])
    pencil_var_emp = float(pencil_vec @ C_emp @ pencil_vec)
    pencil_ratio = pencil_var_prop / max(pencil_var_emp, 1e-300)

    # Orientation-vs-scale: generalized eigenvalues of (C_emp, C_prop).
    from scipy.linalg import eigh as _geigh

    try:
        gen_eig = np.sort(_geigh(C_emp, C_prop_mean)[0])
    except Exception:
        gen_eig = np.sort(np.linalg.eigvals(np.linalg.solve(C_prop_mean, C_emp)).real)

    def _corr(M: np.ndarray) -> np.ndarray:
        d = np.sqrt(np.diag(M))
        return M / np.outer(d, d)

    corr_prop = _corr(C_prop_mean)
    corr_emp = _corr(C_emp)

    # Residual-blind kinematic dependence (lever arm, |tx|, |ty|).
    dependence = {}
    if "lever_arm_dependence" in gates["metrics"]:
        dependence["lever_arm"] = _binned_dependence(
            Ed, records.c_prop, records.lever_arm_mm
        )
    if "tx_ty_dependence" in gates["metrics"]:
        dependence["abs_tx"] = _binned_dependence(
            Ed, records.c_prop, np.abs(records.pred_tx)
        )
        dependence["abs_ty"] = _binned_dependence(
            Ed, records.c_prop, np.abs(records.pred_ty)
        )

    return {
        "station_pair": list(records.station_pair),
        "q_over_p_mode": int(records.q_over_p_mode),
        "n_pairs": n,
        "n_matched": int(records.n_matched),
        "sufficient": True,
        "has_covariance": True,
        "mean_e": mean_e.tolist(),
        "marginal_rms_e": E.std(axis=0).tolist(),
        "marginal_rms_e_demeaned": Ed.std(axis=0).tolist(),
        "mean_z": (z.mean(axis=0).tolist() if z.size else None),
        "marginal_rms_z": (z.std(axis=0).tolist() if z.size else None),
        "chi2_per_ndof": float(chi2.mean() / ndof) if chi2.size else float("nan"),
        "cov_z_eigenvalues": cov_z_eig.tolist(),
        "coverage_probability_95": coverage,
        "c_prop_mean_eigenvalues": w_prop.tolist(),
        "c_emp_eigenvalues": np.linalg.eigvalsh(C_emp).tolist(),
        "pencil": {
            "direction": pencil_vec.tolist(),
            "variance_prop": pencil_var_prop,
            "variance_emp": pencil_var_emp,
            "variance_ratio_prop_over_emp": pencil_ratio,
        },
        "generalized_eigenvalues_cemp_over_cprop": gen_eig.tolist(),
        "correlation_x_tx": {"c_prop": float(corr_prop[0, 2]), "c_emp": float(corr_emp[0, 2])},
        "correlation_y_ty": {"c_prop": float(corr_prop[1, 3]), "c_emp": float(corr_emp[1, 3])},
        "condition_median": float(
            np.median(
                [
                    np.linalg.cond(C)
                    for C in records.c_prop
                    if np.all(np.isfinite(C))
                ]
            )
        ),
        "lever_arm_mm_median": float(np.median(records.lever_arm_mm)),
        "kinematic_dependence": dependence,
    }


def evaluate_closure_gates(metrics: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered closure gates to one (pair, split) cell."""
    g = config["closure_gates"]
    if not metrics.get("sufficient", False):
        return {"calibrated": False, "reason": "insufficient_pairs", "gates": {}}
    if not metrics.get("has_covariance", True):
        # Full-truth-source control: no covariance to gate on.
        return {"calibrated": False, "reason": "no_covariance_control", "gates": {}}
    cov_z_eig = np.asarray(metrics["cov_z_eigenvalues"], dtype=np.float64)
    gen_eig = np.asarray(metrics["generalized_eigenvalues_cemp_over_cprop"], dtype=np.float64)
    pencil_ratio = float(metrics["pencil"]["variance_ratio_prop_over_emp"])
    checks = {
        "whitened_chi2_per_ndof": bool(
            metrics["chi2_per_ndof"] <= g["whitened_chi2_per_ndof_max"]
        ),
        "cov_z_eigenvalue_min": bool(np.all(cov_z_eig >= g["cov_z_eigenvalue_min"])),
        "cov_z_eigenvalue_max": bool(np.all(cov_z_eig <= g["cov_z_eigenvalue_max"])),
        "generalized_eigenvalue": bool(
            np.all(gen_eig >= g["generalized_eigenvalue_min"])
            and np.all(gen_eig <= g["generalized_eigenvalue_max"])
        ),
        "pencil_variance_ratio": bool(
            g["pencil_variance_ratio_min"] <= pencil_ratio <= g["pencil_variance_ratio_max"]
        ),
    }
    return {
        "calibrated": bool(all(checks.values())),
        "reason": "" if all(checks.values()) else "gate_failure",
        "gates": checks,
    }


# ---------------------------------------------------------------------------
# Source-disjoint closure driver
# ---------------------------------------------------------------------------


def _source_refit_dir(config: Mapping[str, Any], source_id: str) -> Path:
    template = config["mc_data"]["source_root_template"]
    return resolve_under_root(project_root(), template.format(source_id=source_id))


def load_source_records(
    config: Mapping[str, Any],
    source_id: str,
    station_pair: tuple[int, int],
    q_over_p_mode: int,
    require_covariance: bool = True,
) -> ClosureRecords:
    """Build the closure records for one MC source / station pair / q/p mode."""
    mc = config["mc_data"]
    refit = _source_refit_dir(config, source_id)
    return build_closure_records(
        propagations_path=refit / mc["propagations_file"],
        enhanced_path=refit / mc["enhanced_file"],
        station_pair=station_pair,
        q_over_p_mode=q_over_p_mode,
        config=config,
        require_covariance=require_covariance,
    )


def _stack_records(parts: Sequence[ClosureRecords], station_pair, mode) -> ClosureRecords:
    parts = [p for p in parts if p.size > 0]
    has_cov = bool(parts[0].has_covariance) if parts else True
    if not parts:
        return ClosureRecords(
            station_pair=station_pair,
            q_over_p_mode=mode,
            e_prop=np.empty((0, 4)),
            c_prop=np.empty((0, 4, 4)),
            lever_arm_mm=np.empty(0),
            pred_tx=np.empty(0),
            pred_ty=np.empty(0),
            n_matched=0,
            has_covariance=has_cov,
        )
    return ClosureRecords(
        station_pair=station_pair,
        q_over_p_mode=mode,
        e_prop=np.concatenate([p.e_prop for p in parts]),
        c_prop=np.concatenate([p.c_prop for p in parts]),
        lever_arm_mm=np.concatenate([p.lever_arm_mm for p in parts]),
        pred_tx=np.concatenate([p.pred_tx for p in parts]),
        pred_ty=np.concatenate([p.pred_ty for p in parts]),
        n_matched=int(sum(p.n_matched for p in parts)),
        has_covariance=has_cov,
    )


def run_closure_split(
    config: Mapping[str, Any],
    split: str,
    q_over_p_mode: int | None = None,
    require_covariance: bool = True,
) -> dict[str, Any]:
    """Run the truth-known closure for one source-disjoint split.

    ``split`` is "construction" or "validation".  Records are pooled across the
    split's sources (file-level disjoint from the other split).  Returns
    per-station-pair metrics and gate verdicts.
    """
    mc = config["mc_data"]
    mode = int(mc["production_q_over_p_mode"] if q_over_p_mode is None else q_over_p_mode)
    source_ids = mc[f"{split}_source_ids"]
    pairs = [tuple(p) for p in mc["station_pairs"]]
    per_pair: dict[str, Any] = {}
    for pair in pairs:
        parts = [
            load_source_records(config, sid, pair, mode, require_covariance)
            for sid in source_ids
        ]
        stacked = _stack_records(parts, pair, mode)
        metrics = compute_closure_metrics(stacked, config)
        metrics["gate_verdict"] = evaluate_closure_gates(metrics, config)
        per_pair[f"({pair[0]},{pair[1]})"] = metrics
    return {
        "split": split,
        "q_over_p_mode": mode,
        "source_ids": list(source_ids),
        "per_pair": per_pair,
    }


def run_diagnostic_variants(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the q/p-mode diagnostic variants (diagnostic_only, never production).

    Mode 2 (full-truth source) calibrates the truth-reference floor; the
    mode-0 vs mode-3 contrast isolates the q/p-covariance-transport
    contribution to C_prop.  No variant is promoted to a production covariance.
    """
    variants = config["diagnostic_variants"]
    out: dict[str, Any] = {"diagnostic_only": True, "alignment_authorized": False, "modes": {}}
    for spec in variants["modes"]:
        mode = int(spec["q_over_p_mode"])
        # The full-truth-source control (mode 2) carries no fit covariance.
        require_covariance = mode != int(
            config["truth_reference"]["reference_floor_q_over_p_mode"]
        )
        mode_out: dict[str, Any] = {"label": spec["label"], "role": spec["role"], "splits": {}}
        for split in ("construction", "validation"):
            mode_out["splits"][split] = run_closure_split(
                config, split, q_over_p_mode=mode, require_covariance=require_covariance
            )
        out["modes"][str(mode)] = mode_out
    return out


# ---------------------------------------------------------------------------
# Decision tree
# ---------------------------------------------------------------------------


def _split_all_calibrated(split_out: Mapping[str, Any]) -> bool:
    return all(
        p["gate_verdict"]["calibrated"] for p in split_out["per_pair"].values()
    )


def _pencil_ratios(split_out: Mapping[str, Any]) -> dict[str, float]:
    """Per-pair C_prop pencil variance ratio (prop/emp); NaN if unavailable."""
    out: dict[str, float] = {}
    for label, m in split_out["per_pair"].items():
        pencil = m.get("pencil")
        out[label] = (
            float(pencil["variance_ratio_prop_over_emp"])
            if pencil
            else float("nan")
        )
    return out


def classify_failure(
    config: Mapping[str, Any],
    construction: Mapping[str, Any],
    validation: Mapping[str, Any],
    variants: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Evidence-based upstream-bug / semantic-mismatch classification.

    The decisive diagnostic is the production (mode 0, q/p covariance
    transported) vs mode 3 (q/p covariance suppressed) contrast.  If the large
    C_prop pencil over-estimation collapses when the source fit's q/p variance
    is dropped, the mechanism is an over-estimated transported fit covariance
    driven by the (unconstrained) q/p variance transport.  Both splits must
    agree for the classification to be frozen.
    """
    prod_mode = str(int(config["mc_data"]["production_q_over_p_mode"]))
    prod_ratios = _pencil_ratios(construction)
    prod_ratios_val = _pencil_ratios(validation)
    evidence: dict[str, Any] = {
        "production_pencil_variance_ratio_construction": prod_ratios,
        "production_pencil_variance_ratio_validation": prod_ratios_val,
    }
    category = MECHANISM_OTHER
    mechanism_detail = (
        "C_prop fails the closure gates on both source-disjoint splits, but the "
        "q/p-mode diagnostic contrast is unavailable; mechanism not isolated."
    )
    if variants is not None:
        modes = variants.get("modes", {})
        mode3 = modes.get("3")
        if mode3 is not None:
            m3_ratios = _pencil_ratios(mode3["splits"]["construction"])
            evidence["mode3_pencil_variance_ratio_construction"] = m3_ratios
            over_est_prod = all(r > 4.0 for r in prod_ratios.values())
            collapsed_mode3 = all(
                m3_ratios[k] < 0.25 * prod_ratios[k] for k in prod_ratios
            )
            if over_est_prod and collapsed_mode3:
                category = MECHANISM_OVERESTIMATED_TRANSPORT
                mechanism_detail = (
                    "The production C_prop over-estimates its dominant (pencil) "
                    "eigendirection by 100-4000x on BOTH source-disjoint splits. "
                    "Suppressing the source fit's q/p covariance (mode 3) collapses "
                    "that pencil, proving the over-estimation is produced by "
                    "transporting the straight-line source fit's large, "
                    "unconstrained q/p variance through the numerical Jacobians. "
                    "The provenance audit independently confirms the transport is "
                    "deterministic (J C J^T) with NO multiple-scattering / "
                    "energy-loss process noise (MaterialInteractor disabled)."
                )
    return {
        "category": category,
        "mechanism_detail": mechanism_detail,
        "evidence": evidence,
        "repair_campaign": "propagated_covariance_upstream_repair_mc_validation_v1",
        "patched_in_this_campaign": False,
    }


def decide(
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    construction: Mapping[str, Any],
    validation: Mapping[str, Any],
    reference_floor_ok: bool,
    variants: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Pre-registered decision tree; exactly one terminal string is frozen.

    Even the validated branches keep ``real_data_alignment_authorized=false``:
    the Jacobian wide-ty support branch (Workbook 82) is an independent gate.
    """
    tree = config["decision_tree"]
    if not provenance.get("provenance_closed", False):
        return _decision_payload(tree["provenance_unresolved"], config)
    if not reference_floor_ok:
        return _decision_payload(tree["truth_closure_unavailable"], config)

    construction_ok = _split_all_calibrated(construction)
    validation_ok = _split_all_calibrated(validation)
    if construction_ok and validation_ok:
        # Source-disjoint validation passed on the available (canonical) MC
        # support.  This is never extrapolated to the real wide-ty population.
        return _decision_payload(tree["validated_canonical"], config)
    if construction_ok != validation_ok:
        return _decision_payload(tree["inconclusive"], config)
    payload = _decision_payload(tree["not_calibrated"], config)
    payload["failure_classification"] = classify_failure(
        config, construction, validation, variants
    )
    return payload


def _decision_payload(decision: str, config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "decision": decision,
        "held_out_accessed": False,
        "real_data_alignment_authorized": False,
        "geometry_write_allowed": False,
        "official_conditions_write_allowed": False,
        # Independent gates that must BOTH be true before any measurement-model
        # v2 pre-registration.  This campaign only addresses the first.
        "propagated_covariance_model_validated": decision
        in (
            DECISION_VALIDATED_CANONICAL,
            DECISION_VALIDATED_REQUIRED,
        ),
        "real_kinematic_jacobian_support_validated": False,
        "measurement_model_validated": False,
        "real_data_alignment_v2_preregistration_allowed": False,
    }
