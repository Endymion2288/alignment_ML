# FaserActsExtrapolation Propagated-Covariance Provenance & Closure V1

Workbook 81. **Status: completed and frozen** — executed in freeze order within a
single interactive session; every validation gate was written into the config and
frozen before any confirmatory computation, and no gate was modified after
execution.

**Final decision: `faseracts_propagated_covariance_not_calibrated`.**
**Mechanism classification: `overestimated_transported_fit_covariance`.**

> **Naming:** this Workbook 81 is **not** the "Gauge-Fixed Real-Data Alignment
> Diagnostic V2". Workbook 80 failed
> (`measurement_model_multiple_components_not_validated`), so that conditional
> success branch was never triggered. The real Alignment Diagnostic V2 gets a new
> number only after the measurement model is fully validated.

This campaign is **upstream covariance validation, not alignment**. For the whole
workbook: `held_out_accessed=false`, `real_data_alignment_authorized=false`,
`geometry_write_allowed=false`, `official_conditions_write_allowed=false`,
`external_constraint_ingest_authorized=false`, and the
`FaserActsExtrapolationTool` is **not** modified in the first stage.

## Scientific question (the only one)

What does the propagated covariance `C_prop` produced by the
`FaserActsExtrapolationTool` actually represent, and does its covariance
transport / process noise correctly describe the source-tracklet →
downstream-surface prediction error on truth-known MC?

Workbook 80 showed the huge χ² of the frozen combined covariance is not a
numerical bug and that `C_combined = C_prop + C_target` is the correct code
semantics, but `C_prop` comes from the external Calypso/ACTS tool and its
pencil-direction covariance is clearly too strong (over-estimating the real
variance by ~100–2000×). This campaign isolates `C_prop` from `C_target` and from
real-data residuals, and validates `C_prop` alone on truth-known MC.

## Stage 0 — provenance audit (`faseracts_covariance_provenance.json`)

Direct audit of the local Calypso source (git `40892527e9c6…`, Athena `24.0.41`,
ACTS `32.0.2`) and runtime config. **All 15 pre-registered questions resolved (11
from source, 4 from runtime config), 0 unresolved, `provenance_closed=true`.**
Key findings (each with file:line evidence in the artifact):

- Input is the native Athena `Trk::TrackParameters` 5×5 `(loc1,loc2,phi,theta,q/p)`.
- `actsTrackletParameters` converts to the ACTS bound 6×6 via a central numerical
  Jacobian; the q/p column **is transported** unless suppressed (mode 3).
- Transport is `C → J C J^T` accumulated by the ACTS `EigenStepper` and applied by
  `CovarianceEngine::transportCovarianceToBound`.
- **Material effects, multiple-scattering process noise, and energy-loss
  uncertainty are all disabled** (the `MaterialInteractor` early-returns).
- Particle hypothesis is hardcoded to `Acts::ParticleHypothesis::muon()`.
- `C_prop` is a **prediction covariance** (transported source-fit covariance), not
  a conditional covariance, and contains **no stochastic transport noise**.

## Stage 1 — truth/reference target state (`truth_reference.json`)

The reference target state reuses the existing per-station Geant-truth state in
`enhanced_tracklets.root` (`truth_stX_*`), joined on `(run, eventID,
truth_barcode)`, straight-line corrected to the propagation target plane. No new
exporter, no invented branch, no change to reconstruction selection. The mode-2
full-truth-source control calibrates the irreducible reference floor; the
pre-registered `signal/floor ≥ 3` gate passes for all pairs (28×, 101×, 118× for
(0,1)/(0,2)/(0,3)), so `reference_floor_ok=true`.

## Stage 2 — source-disjoint MC truth closure (`closure.json`, production mode 0)

Observable: `e_prop = propagated source-tracklet state − truth state at target`
(no `C_target`). Whitened `z_prop = C_prop^{-1/2} e_prop` is analysed per station
pair and per residual-blind kinematic slice, on two source-disjoint MC splits
(construction: 5 sources; validation: 4 sources, file-level disjoint).

**All 6 cells (3 pairs × 2 splits) fail the gates (`calibrated=false`), and the
two source-disjoint splits agree.** Whitened χ²/ndof ranges 1142–149577 (gate ≤
4); the pencil-direction variance ratio (C_prop / empirical) is **310–3894×**,
reproducing and amplifying the WB80 finding; generalized eigenvalues of
`(C_empirical, C_prop)` are simultaneously ≪1 (three directions, over-estimated)
and >1 (one direction, under-estimated) — **both orientation and scale are
wrong**. The miscalibration is pervasive across lever-arm / |tx| / |ty| bins.

## Stage 3 — q/p-mode diagnostic variants (`diagnostic_variants.json`)

Four pre-registered q/p-handling modes (all `diagnostic_only=true`,
`alignment_authorized=false`). The decisive contrast is mode 0 (q/p covariance
transported) vs mode 3 (q/p covariance suppressed):

- **Mode 0 (production):** pencil over-estimated by 310–3894×.
- **Mode 3 (q/p covariance suppressed):** the spurious pencil collapses
  (~4000× drop to ~0.1), proving the WB80 pencil over-estimation is produced by
  transporting the straight-line source fit's large, unconstrained q/p variance.
- **But** modes 3 and 1 then **under**-estimate the dominant remaining direction
  by 10–100× (pencil ratio ~0.01–0.10) with even larger χ² — the underlying
  source-tracklet fit covariance itself under-estimates the real fit error.

So `C_prop` is wrong in both scale and orientation, in both directions: the q/p
covariance transport creates an over-estimated spurious pencil, while the
underlying source-fit covariance under-estimates the bending-plane error.

## Decision and follow-up

- Frozen decision: `faseracts_propagated_covariance_not_calibrated`;
  classification `overestimated_transported_fit_covariance`.
- `propagated_covariance_model_validated=false`,
  `real_kinematic_jacobian_support_validated=false`,
  `measurement_model_validated=false`,
  `real_data_alignment_v2_preregistration_allowed=false`,
  `held_out_accessed=false` — all preserved.
- Per the pre-registered rule, the upstream bug is **not** patched inside this
  campaign. A separate **Propagated-Covariance Upstream Repair & MC Validation
  V1** campaign is required; only after the upstream `C_prop` closure passes may
  the combined residual covariance be redefined.
- The independent **Workbook 82 (Wide-ty Real-Support-Matched MC Coverage
  Feasibility V1)** branch is still required: WB80's real (0,1)/(0,2) `pred_ty`
  support shortfall is an independent blocker. Both gates
  (`propagated_covariance_model_validated` and
  `real_kinematic_jacobian_support_validated`) must be true before any
  measurement-model v2 pre-registration; the two branches must not rescue each
  other.

## Artifacts

Live on EOS under `outputs/faseracts_propagated_covariance_provenance_closure_v1/`
(not git-tracked); SHA256 digests are recorded in the Workbook 81 workbook.
Engineering files:
`configs/faseracts_propagated_covariance_provenance_closure_v1.yaml`,
`alignment/propagated_covariance_closure.py`,
`scripts/report_faseracts_propagated_covariance_closure.py`,
`tests/test_propagated_covariance_closure.py` (40 regression tests).
