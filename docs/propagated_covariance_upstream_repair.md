# Propagated-Covariance Upstream Repair & MC Validation V1

Workbook 83, **Stage A: source-tracklet covariance truth closure**.
**Status: completed and frozen** — executed in freeze order within a single
interactive session; every gate was written into the config and frozen before
any confirmatory computation, and no gate was modified after execution.

**Final decision: `source_tracklet_fit_covariance_not_calibratable`.**
**Mechanism classification: `position_xy_swap_with_slope_miscalibration`.**
**Points to upstream repair campaign: `segment_fit_hit_error_model_covariance_audit`
(no patch applied in this campaign).**

> **Naming:** this Workbook 83 is the covariance branch (following WB81), **not**
> an alignment task and **not** the WB82 kinematic-support branch. WB82
> independently froze `existing_mc_real_wide_ty_support_validated`. This campaign
> addresses only problem **(B)** that WB81 identified — the source-tracklet fit
> covariance itself does not cover the real source-state error — and it does
> **Stage A first** (source-surface truth closure, **no propagation**). Stage B
> (propagation covariance construction) is pre-registered in the config but is
> only entered if Stage A passes. Stage A did **not** pass, so Stage B was not
> entered and the FaserActs propagated covariance was **not** touched.

For the whole workbook: `held_out_accessed=false`,
`real_data_alignment_authorized=false`, `geometry_write_allowed=false`,
`official_conditions_write_allowed=false`,
`external_constraint_ingest_authorized=false`,
`propagated_covariance_model_validated=false`, `measurement_model_validated=false`,
`real_kinematic_jacobian_support_validated=false`. It reads **no** real-data
residual and never solves for an alignment correction.

## Scientific question (the only one)

Does the source-tracklet fit covariance `C_source` describe the real source-state
error `e_source = fitted source state − truth state at the source surface` (in the
measured `[x,y,tx,ty]` basis, with **no propagation**, validated on source-disjoint
MC)? If not, can a pre-registered limited physically-interpretable repair yield a
portable calibrated source covariance?

WB81 froze `faseracts_propagated_covariance_not_calibrated` (mechanism
`overestimated_transported_fit_covariance`): the production `C_prop` is a
deterministic `J C J^T` transport of the source-fit covariance with **no**
material/process noise, and the unconstrained dummy q/p covariance forms a
dominant spurious pencil (over-estimated 310–3894×); mode 3 (suppress q/p
covariance transport) removes the pencil but then **under**-estimates the real
propagation error 10–100×. WB81 identified two independent problems: **(A)** the
dummy q/p covariance should not enter the propagated covariance; **(B)** the
source-tracklet fit covariance itself does not cover the real source-state /
propagation stochastic error. This campaign validates only problem (B), and it
validates the **source surface first** — propagation cannot repair a wrong input
uncertainty model.

q/p is not a measured quantity of a straight tracklet, so the Stage-A primary
observable does **not** require the dummy q/p to be "calibrated".

## Initial-state audit (STEP 1, done)

`load_config` SHA-verifies the frozen WB81/WB82 inheritance (any mismatch raises
`ConfigError`):

- WB81 config `5a13b0cc…`, `propagated_covariance_decision.json` `fd605f0e…`,
  `campaign_summary.json` `97a92006…`, `closure.json` `27ac8820…`; frozen decision
  `faseracts_propagated_covariance_not_calibrated`, mechanism
  `overestimated_transported_fit_covariance`,
  `propagated_covariance_model_validated=false` — all verified verbatim.
- WB82 config `b4417b4d…`, `wide_ty_mc_support_decision.json` `e5386b61…`,
  `campaign_summary.json` `0ba7ed82…`; frozen decision
  `existing_mc_real_wide_ty_support_validated` — verified verbatim.
- Software provenance (read-only): Calypso
  `40892527e9c65409afd2378a2abfc25ddbddac03`, Athena `24.0.41`, ACTS `32.0.2`,
  exporter `PhysicsAnalysis/NtupleDumper/src/NtupleDumperAlg.cxx`.

## Frozen data roles and prohibitions

- **MC data (source-disjoint, file-level):** the same hierarchical-V1 sources and
  the same construction/validation split as WB81 (construction: 100043×3 +
  100044×2; validation: 100047×2 + 100048×2), at the nominal-geometry
  `iteration_00_reference` point (tracklet fit and Geant truth share one
  geometry). Source tracklet state + fit covariance from `tracklets.root`; truth
  from `enhanced_tracklets.root`.
- **Stage A tests every station (0–3):** source and target tracklets share the
  same exporter transform, and the miscalibration is station-universal. Station 0
  is the primary alignment-source case.
- **Truth reference:** per-station Geant truth (`truth_stX_*`),
  `tx_truth=px/pz`, `ty_truth=py/pz`, straight-line z-correction for
  `x/y_truth(z_mm)`. `e_source = [x_mm,y_mm,tx,ty]_fitted − [x,y,tx,ty]_truth`.
- **Filters (residual-blind):** truth-match ≥ 0.99; physical acceptance
  `|tx|,|ty| ≤ 0.2` (drops only unphysical near-vertical outliers so they cannot
  inflate covariance envelopes). **No tracklet is dropped on pull / condition /
  fit quality.**
- Prohibited: opening held-out/sealed data, reading real-data residuals, solving
  an alignment correction, modifying FaserActsExtrapolationTool, promoting mode 3
  to a production covariance, using truth q/p as a real-data solution,
  reverse-optimizing the source covariance from propagated-target whitening,
  changing the central propagated state in a covariance repair, tuning the
  covariance to chi2 ≈ 1.

## Stage A method (`source_closure.json`)

For each (station, split): whiten `z = C_source^{-1/2} e_source` (symmetric
eigendecomposition) and compute the pre-registered metrics — `chi2/ndof` (mean +
robust median + tail fraction), `Cov(z)` eigenvalues, generalized eigenvalues
`eig(C_emp, C_source)`, coverage probability, marginal variance ratios
`C_ii/emp_ii`, correlation orientation, and a **position-swap diagnostic**
(report-only: cross-swap ratios `C_xx/emp_yy`, `C_yy/emp_xx`, and the chi2/ndof
before vs after a position-only x↔y permutation).

**Pre-registered gates (confirmatory, frozen, unchanged):** `chi2/ndof ≤ 4`,
`Cov(z)` eigenvalues ∈ [0.25, 4], generalized eigenvalues ∈ [0.25, 4], ≥ 30
tracklets per station; and the construction and validation source-disjoint splits
must **agree** (portability).

## Stage A findings (core result)

**The as-is source covariance fails on all 4 stations and both source-disjoint
splits.** Station 0 construction (the rest are identical in pattern):

| component | empirical RMS `e_source` | fit covariance marginal RMS | variance ratio `C_ii/emp_ii` |
| --- | --- | --- | --- |
| x | 0.524 mm (along-strip, imprecise) | 0.011 mm (claims precise) | **≈ 0.0004 (under-estimated ~2500×)** |
| y | 0.0099 mm (precision) | 0.530 mm (claims imprecise) | **≈ 2873 (over-estimated ~2900×)** |
| tx | 0.020 | 0.020 | **≈ 1.03 (calibrated)** |
| ty | 0.00040 | 0.0049 | **≈ 154 (over-estimated ~150–300×)** |

- **Position x↔y swap (structural):** cross-swap ratio `C_yy/emp_xx ≈ 0.93–1.18`
  (clean ≈ 1), `C_xx/emp_yy ≈ 0.97–5.9`. The written covariance has the x/y
  position precision **exchanged**: it claims x precise and y imprecise, but the
  truth is the opposite (y is the precision/bending coordinate, x is along-strip).
  This is consistent with WB81's q/p pencil lying in y, the magnet bending in y,
  and y being the empirical precision coordinate.
- **tx calibrated, ty over-estimated** by ~150–300× (variance).
- **Whitened chi2/ndof (median):** as-is ≈ **360–437** (all stations/splits), far
  above the ≤ 4 gate.

### Position-swap diagnostic: the swap fixes the position diagonal but not the correlation structure

After a position-only x↔y permutation of the covariance (report-only diagnostic),
the **median** whitened chi2/ndof drops from ~400 to **~1.2–1.8** (near 1) — the
permutation repairs the position **diagonal** for the typical tracklet. But the
repair is only **partial**: the fraction of tracklets with chi2/ndof > 4 stays at
**34–46%** (vs ~0.4% expected for a calibrated 4-dof covariance). The residual
tail comes from a **spurious near-null direction** (a wrong y–ty correlation that
drives one linear combination's variance to near zero; outliers have pull ≈ 10.8
along it vs ≈ 0.7 for typical tracklets), which the permutation does not fix.

### Repair ceiling test (report-only)

- as-is: median chi2 ≈ 409, frac>4 ≈ 0.95.
- **position swap only:** median ≈ 1.24, frac>4 ≈ 0.34, generalized eigenvalues
  `[0.006, 0.80, 0.96, 1.01]` (three directions ~1, but the ty direction still
  ~0.006, over-estimated).
- **swap + diagonal rescale:** generalized eigenvalues all ~1, but the median chi2
  **worsens to ~26** and frac>4 to ~0.80 — a diagonal rescale amplifies the
  spurious near-null direction.
- **C = C_emp (ceiling, not a repair):** median ≈ 0.83, frac>4 ≈ 0.021.

**Conclusion:** the source covariance is in the **wrong frame/basis** — the
position is swapped **and** the correlation structure does not match the empirical
`e_source` covariance. No scale/permutation repair yields a portable calibration.

## Repair evaluation (`stage_a_repairs.json`)

The 4 pre-registered repair arms (derived on construction, confirmed on
validation, **not** chi2-tuned) all **fail** the source-disjoint gates:

| repair | parameters | validated_source_disjoint |
| --- | --- | --- |
| `global_scale` | scale ≈ 0.94 | **no** |
| `xy_block_scale` | scale_x ≈ 2289, scale_y ≈ 0.00035 | **no** |
| `position_permutation_xy_swap` | (parameter-free structural) | **no** |
| `station_dependent_scale` | per-station ≈ 0.85–1.07 | **no** |

`position_permutation_xy_swap` and `xy_block_scale` fix the position diagonal
(median ~1.2–1.8) but still fail the mean-chi2 and generalized-eigenvalue gates
because of the spurious correlation / ty over-estimation. **No pre-registered
repair yields a portable source covariance calibration.**

## Decision (`propagated_covariance_upstream_repair_decision.json`)

**`source_tracklet_fit_covariance_not_calibratable`**

- Mechanism classification: `position_xy_swap_with_slope_miscalibration`.
- `position_swap_fixes_diagonal_median_chi2 ≈ 1.51` (median chi2 after the swap
  repairs the position diagonal).
- `residual_tail_fraction_after_swap ≈ 0.43` (remaining chi2>4 tail after the swap).
- Points to upstream repair campaign: `segment_fit_hit_error_model_covariance_audit`;
  **no** patch applied in this campaign (`patched_in_this_campaign = false`).
- All stations and both source-disjoint splits fail consistently
  (`swap_consistent_across_stations = true`).
- `propagated_covariance_model_validated = false`,
  `measurement_model_validated = false`,
  `real_kinematic_jacobian_support_validated = false`,
  `real_data_alignment_v2_preregistration_allowed = false`.

**Physical interpretation:** the mismatch between the written covariance and the
empirical `e_source` covariance is at the **frame/basis level** (position swap +
wrong correlation structure + ty over-estimation), not a simple global/block scale
error. This is consistent with the exporter
`NtupleDumperAlg.globalTrackletCovariance` transforming the Athena native 5×5
`(loc1,loc2,phi,theta,q/p)` SegmentFit covariance to the global 4×4 `[x,y,tx,ty]`
via a central numerical Jacobian — the root cause is upstream in the SegmentFit /
hit-error model / exporter covariance transform, and **propagation cannot repair a
wrong input uncertainty model**.

## Next steps (frozen recommendation)

1. **Do not enter Stage B** (Stage A failed); do not touch the FaserActs
   propagated covariance; do not promote mode 3 to a production covariance.
2. Open the upstream repair campaign
   **`segment_fit_hit_error_model_covariance_audit`**: audit the SegmentFit native
   `(loc1,loc2,phi,theta)` covariance and the exporter numerical Jacobian for
   basis/frame consistency, and locate the root cause of the position x↔y swap and
   the spurious y–ty correlation.
3. Only after the source covariance passes all Stage-A gates on source-disjoint MC
   may Stage B be re-entered (q/p covariance semantic repair + physical process
   noise), and finally combined with the WB82 J-support conclusion in a
   Measurement Model V2.

## Engineering artifacts

- Config: `configs/propagated_covariance_upstream_repair_mc_validation_v1.yaml`
- Module: `alignment/source_tracklet_covariance_closure.py`
- Driver: `scripts/report_propagated_covariance_upstream_repair.py`
- Tests: `tests/test_source_tracklet_covariance_closure.py` (45 tests)
- Artifacts (EOS, not in git):
  `outputs/propagated_covariance_upstream_repair_mc_validation_v1/`
  (`config_validation.json`, `source_closure.json`, `stage_a_repairs.json`,
  `propagated_covariance_upstream_repair_decision.json`, `campaign_summary.json`)
