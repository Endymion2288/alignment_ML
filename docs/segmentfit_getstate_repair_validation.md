# GetState Covariance Transform Repair Validation V1

Workbook 85. **Status: completed and frozen.**
**Final decision: `segmentfit_getstate_repair_insufficient`.**

This campaign validates whether the two deterministic `SegmentFitAlg::GetState`
repairs located by WB84 are a **sufficient** cause of the WB83 source-covariance
failure.  It re-runs the frozen WB83 Stage A on the same source-disjoint
construction/validation split with the **same gates**.  It is **not** an
alignment task.

**Inherited frozen state (SHA-verified, not reopened):**

- WB81 `faseracts_propagated_covariance_not_calibrated`
- WB82 `existing_mc_real_wide_ty_support_validated` (`floor_muon_100120`)
- WB83 `source_tracklet_fit_covariance_not_calibratable` /
  `position_xy_swap_with_slope_miscalibration`
- WB84 `deterministic_segmentfit_get_state_transform_bug` /
  `coordinate_convention_mismatch_and_jacobian_sign_error`

`geometry_write_allowed=false`, `real_data_alignment_authorized=false`,
`measurement_model_validated=false`.  No real residual, no geometry/conditions
write, no FaserActs / Stage B, no Frozen-V2.

## Repair scope (closed)

1. Curvilinear slot mapping: `loc1 = -y`, `loc2 = +x` (Athena beam-track frame).
2. Jacobian sign: `dφ/dtx = -ty/r²`.

Forbidden: scale factors, empirical rescaling, chi2 tuning, residual-based
correction, station-dependent calibration, central-prediction changes.

The repair is applied offline as the deterministic transform
`C_rep = J_rep J_buggy^{-1} C_exp J_buggy^{-T} J_rep^T`.  The fitted
`[x,y,tx,ty]` state is unchanged.

## Gates (identical to WB83, not modified)

`chi2/ndof ≤ 4`, `Cov(z)` eigenvalues ∈ [0.25, 4], generalized eigenvalues
∈ [0.25, 4], source-disjoint construction/validation consistency.

## Result

Baseline reproduces WB83: all 8 cells fail (median chi2/ndof ~ 360–437).

After the two GetState repairs (3230 tracklets, 0 inversion failures):

| cell | chi2/ndof | median | frac>4 | calibrated |
| --- | --- | --- | --- | --- |
| construction 0 | 1.00 | 0.838 | 0.026 | **yes** |
| construction 1 | 3.68 | 0.851 | 0.026 | **no** (mean C_yy) |
| construction 2 | 0.987 | 0.849 | 0.015 | **yes** |
| construction 3 | 0.973 | 0.824 | 0.022 | **yes** |
| validation 0 | 1.11 | 0.826 | 0.031 | **no** (mean C_yy) |
| validation 1 | 0.932 | 0.784 | 0.022 | **yes** |
| validation 2 | 0.910 | 0.820 | 0.008 | **yes** |
| validation 3 | 1.02 | 0.844 | 0.025 | **yes** |

The typical-tracklet contract is restored: median chi2/ndof = 0.83, a
position-only XY swap now **worsens** chi2 to ~300, and `var(ty)` is calibrated.
The two failing cells fail only the **mean**-covariance eigen-gates because
MEAN `C_yy` is inflated by 1–2 large-angle tracklets (`|t·z|<0.99`) for which
the beam-track slot map is not the Athena Curvilinear frame.  Median `C_yy`
on those cells remains calibrated.  This is **not** a residual XY swap and is
**not** a justification for a scale factor.

## Decision

`segmentfit_getstate_repair_insufficient`.

WB84's two bugs are the **dominant** cause of the WB83 failure (6/8 cells pass;
typical-tracklet whitening is recovered).  They are **not yet sufficient** for
every station/split under the frozen mean-covariance gates.

`measurement_model_v2_discussion_allowed=false`.  Measurement Model V2,
conditional Jacobian, and alignment closure remain closed.

## Next step (not authorized here)

A later campaign may pre-register a **full CurvilinearUVT slot map** (both
`|t·z|` branches), still deterministic, still without scale factors, and
re-run the same Stage A.  This campaign does not apply that extension.
