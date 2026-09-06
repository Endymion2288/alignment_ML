# SegmentFit Covariance Coordinate Contract Audit V1

Workbook 84, **SegmentFit covariance coordinate contract audit**.
**Status: completed and frozen** — executed in freeze order within a single
interactive session; the audit is based on a code audit and a synthetic
covariance closure (reimplementing the transform in Python and injecting known
covariances), **not** on alignment residuals or real data.

**Final decision: `deterministic_segmentfit_get_state_transform_bug`.**
**Root cause: `coordinate_convention_mismatch_and_jacobian_sign_error`.**
**Location: `SegmentFitAlg::GetState`** (the `NtupleDumperAlg` exporter transform
is **correct**; the SegmentFit hit-error model itself is **calibrated**).

> **Naming:** this Workbook 84 is the upstream-source audit of the WB83 covariance
> branch. It does **not** reopen any closed alignment / identifiability branch
> (WB81 `faseracts_propagated_covariance_not_calibrated`, WB82
> `existing_mc_real_wide_ty_support_validated`, WB83
> `source_tracklet_fit_covariance_not_calibratable` are all inherited frozen and
> SHA-verified). It does **not** repair the covariance; it only **locates** the
> WB83 mechanism.

For the whole workbook: `held_out_accessed=false`,
`real_data_alignment_authorized=false`, `geometry_write_allowed=false`,
`official_conditions_write_allowed=false`, `measurement_model_validated=false`.
It reads **no** real-data residual, never opens held-out data, never writes
geometry/conditions, never modifies the covariance, never adds scale factors, and
never tunes parameters to chi2.

## Scientific question (the only one)

WB83 froze `source_tracklet_fit_covariance_not_calibratable` with mechanism
`position_xy_swap_with_slope_miscalibration`: the exported source-tracklet
covariance has a structural position x<->y swap (not a scale error), and an XY
swap recovers the position diagonal but cannot repair the remaining spurious
y-ty correlation and tail. WB84 asks: **where does this mechanism actually
originate?**

The audit chain is

    SegmentFit native state: (loc1, loc2, phi, theta, q/p)
        -> NtupleDumperAlg covariance transformation
        -> exported covariance: [x, y, tx, ty]

and the success criterion is to answer whether the WB83 mechanism occurs in
**(1)** SegmentFit covariance generation, **(2)** the covariance exporter
transform, or **(3)** a coordinate convention mismatch.

## Initial-state audit (STEP 1, done)

`load_config` SHA-verifies the frozen WB81/WB82/WB83 inheritance (any mismatch
raises `ConfigError`):

- WB81 config `5a13b0cc…`, frozen decision
  `faseracts_propagated_covariance_not_calibrated`, mechanism
  `overestimated_transported_fit_covariance` — verified verbatim.
- WB82 config `b4417b4d…`, frozen decision
  `existing_mc_real_wide_ty_support_validated` — verified verbatim.
- WB83 config `398e78be…`, frozen decision
  `source_tracklet_fit_covariance_not_calibratable`, mechanism
  `position_xy_swap_with_slope_miscalibration` — verified verbatim.
- Software provenance (read-only): Calypso
  `40892527e9c65409afd2378a2abfc25ddbddac03`, Athena `24.0.41`, ACTS `32.0.2`,
  exporter `PhysicsAnalysis/NtupleDumper/src/NtupleDumperAlg.cxx`, SegmentFit
  `Tracker/TrackerRecAlgs/TrackerSegmentFit/src/SegmentFitAlg.cxx`, Curvilinear
  frame `Tracking/TrkEvent/TrkEventPrimitives/TrkEventPrimitives/CurvilinearUVT.icc`.

## Method

The audit reimplements, in Python
(`alignment/segmentfit_covariance_coordinate_contract.py`):

1. **the Athena `CurvilinearUVT` frame** (`curvilinear_uvt`) — the local
   `loc1`/`loc2` axes on the curvilinear surface (perpendicular to the track
   direction);
2. **the `NtupleDumperAlg::globalTrackletCovariance` numerical Jacobian**
   (`exporter_jacobian`) — `d(x, y, tx, ty) / d(loc1, loc2, phi, theta)`;
3. **the `SegmentFitAlg::GetState` analytic Jacobian** (`segment_fit_jacobian`) —
   `d(x, y, phi, theta) / d(x, y, tx, ty)`, with a `sign_error` switch that
   reproduces the code as written (`True`) or the mathematically corrected form
   (`False`).

It then runs the **synthetic covariance injection test (Cases A-E)** (inject a
unit variance in one native parameter, check which global component receives it)
and a **closure test** (invert the actual transform to recover the SegmentFit
`(x, y, tx, ty)` fit covariance from the exported one, then re-propagate it with
the corrected Jacobian).

## loc1/loc2 are not detector local x/y

This is the first required audit item.  Two different `loc1`/`loc2` contracts
exist in `SegmentFitAlg::GetState` and they must not be conflated:

- **Track-parameter `loc1`/`loc2`** live on `Trk::CurvilinearParameters`
  (`SegmentFitAlg.cxx:722`).  They are coordinates on the curvilinear surface
  perpendicular to the track.  For FASER beam tracks (`|t · z| ≥ 0.99`) the
  Athena `CurvilinearUVT` frame sets `loc1` (`curvU`) ~ `-global y` and `loc2`
  (`curvV`) ~ `+global x`.
- **Detector-local `loc1`/`loc2`** live on the separate
  `FaserSCT_ClusterOnTrack` measurement (`SegmentFitAlg.cxx:711-712`,
  `fitCluster->localPosition()`).  That object is the SCT wafer measurement; it
  does **not** enter the 5×5 track-parameter covariance that the exporter
  transforms.

`GetState` writes the **global** `(x, y)` fit covariance into the **curvilinear**
`(loc1, loc2)` slots.  That is the coordinate-convention mismatch.  It is not a
detector-local ordering error.

## Findings

### Finding 1 — position x<->y swap: coordinate convention mismatch

`SegmentFitAlg::GetState` fits `(x, y, tx, ty)` in **global** coordinates,
converts the covariance to `(x, y, phi, theta)` via the analytic Jacobian, and
places it into the `Trk::CurvilinearParameters` `(loc1, loc2, phi, theta, q/p)`
slots **assuming `(loc1, loc2) = (global x, global y)`**.

But the Athena `CurvilinearUVT` frame for FASER beam tracks (`|t . z| >= 0.99`)
defines **`loc1` (`curvU`) ~ `-global y`** and **`loc2` (`curvV`) ~ `+global x`**.
The synthetic injection test confirms the exporter's coordinate contract:

- **Case A (inject `loc1`)** lands on **`y`** (Jacobian column `y = -1.0`);
- **Case B (inject `loc2`)** lands on **`x`** (Jacobian column `x = +1.0`);
- **Case E (inject `q/p`)** is dropped (the exporter skips the `q/p` column).

So the exporter faithfully maps `loc1 -> -y` and `loc2 -> +x`, and because
SegmentFit put `var(x)` into `loc1` and `var(y)` into `loc2`, the exported
`cov_xx` receives `var(y)` and `cov_yy` receives `var(x)`: a **deterministic
position x<->y swap**. This is exactly the WB83 observation (exported `cov_xx`
median RMS 0.0099 = the precise empirical y; exported `cov_yy` median RMS 0.4966
= the imprecise empirical x).

### Finding 2 — slope ty miscalibration: Jacobian sign error

`SegmentFitAlg::GetState`'s analytic Jacobian has **`d phi/d tx = +ty/r^2`**,
but the mathematically correct derivative for `phi = atan2(ty, tx)` is
**`-ty/r^2`**. The closure test shows:

- **Inverting the ACTUAL transform** (the code-as-written Jacobian) recovers a
  **calibrated, direction-independent** SegmentFit fit covariance:
  `var(x) = 0.247 >> var(y) = 9.9e-5` (correct position assignment),
  `var(ty)/var(tx) ~ 0.0004 = 1/alpha^2` for **all** `|tx|` bins (the correct
  stereo geometry), and `cov(tx, ty) ~ 0`. So the **SegmentFit hit-error model is
  calibrated**.
- **Re-propagating that fit covariance with the CORRECTED Jacobian** yields a
  **direction-independent, calibrated** `cov_tyty` (`~1.5e-7`, matching the
  empirical ty resolution) for all `|tx|` bins — whereas the code-as-written
  Jacobian reproduces the actual direction-dependent `cov_tyty` exactly
  (`sign_error_is_sole_cause_of_slope_miscalibration = true`).

So the direction-dependent `cov_tyty` over-estimation (the WB83 "slope
miscalibration") is **solely** the `d phi/d tx` sign error, which couples
`var(tx)` into `var(ty)` in a direction-dependent way.

### Conclusion — failure location

The WB83 `position_xy_swap_with_slope_miscalibration` mechanism is **two
deterministic bugs in `SegmentFitAlg::GetState`'s covariance transform**:

1. a **coordinate convention mismatch** — the global `(x, y)` covariance is
   written into the Curvilinear `(loc1, loc2)` slots assuming
   `(loc1, loc2) = (x, y)`, but the frame is `(loc1, loc2) = (-y, +x)`;
2. a **Jacobian sign error** — `d phi/d tx = +ty/r^2` (should be `-ty/r^2`).

Answering the WB84 success criterion: the mechanism occurs in **(1) SegmentFit
covariance generation** via **(3) a coordinate convention mismatch** (plus a
Jacobian sign error). The **(2) covariance exporter transform is CORRECT**, and
the SegmentFit hit-error model itself is **calibrated**.

WB83 had pointed the next campaign at
`segment_fit_hit_error_model_covariance_audit`.  That pointer is **superseded**:
inverting the actual GetState+exporter transform recovers a calibrated,
direction-independent fit covariance (`var(x) ≫ var(y)`,
`var(ty)/var(tx) = 1/α²`, `cov(tx,ty) ≈ 0`).  The hit-error model is not the
failure location.  A future campaign, if pre-registered, would validate a
**GetState transform repair**, not a hit-error-model rewrite.
`hit_error_model_audit_authorized=false`.

## Outputs

- `coordinate_contract_audit.json` — the coordinate contract audit.
- `covariance_transform_matrix.json` — the reconstructed transform matrices.
- `synthetic_basis_test_report.json` — the synthetic injection test (Cases A-E).
- `failure_location_report.json` — the failure location + closure tests.
- `segmentfit_covariance_coordinate_contract_decision.json` — the decision.
- `campaign_summary.json` — the campaign summary.

## Next step (not authorized here)

A new **covariance repair validation campaign** may be pre-registered separately
to repair the two deterministic bugs in `SegmentFitAlg::GetState` and validate
the repaired covariance. This campaign does **not** repair the covariance. The
Frozen-V2 alignment loop remains **not authorized**
(`geometry_write_allowed=false`, `real_data_alignment_authorized=false`,
`measurement_model_validated=false`).
