# Full CurvilinearUVT Branch Covariance Contract Repair Validation V1

Workbook 86. **Status: completed and frozen.**
**Final decision: `segmentfit_full_curvilinear_covariance_contract_validated`.**

This campaign asks whether the WB85 remaining structure
`mean_cyy_inflation_on_large_angle_subset` is the missing general
(`|t·z|<0.99`) Athena `CurvilinearUVT` branch of `SegmentFitAlg::GetState`.
It re-runs the frozen WB83 Stage A on the same source-disjoint
construction/validation split with the **same gates**, comparing baseline /
WB85 beam-approximation repair / WB86 full CurvilinearUVT repair.  It is
**not** an alignment task.

**Inherited frozen state (SHA-verified, not reopened):**

- WB81 `faseracts_propagated_covariance_not_calibrated` /
  `overestimated_transported_fit_covariance`
- WB82 `existing_mc_real_wide_ty_support_validated` (`floor_muon_100120`)
- WB83 `source_tracklet_fit_covariance_not_calibratable` /
  `position_xy_swap_with_slope_miscalibration`
- WB84 `deterministic_segmentfit_get_state_transform_bug` /
  `coordinate_convention_mismatch_and_jacobian_sign_error`
- WB85 `segmentfit_getstate_repair_insufficient` /
  remaining structure `mean_cyy_inflation_on_large_angle_subset`

`geometry_write_allowed=false`, `real_data_alignment_authorized=false`,
`measurement_model_validated=false`.  No real residual, no geometry/conditions
write, no FaserActs / Stage B entry, no Frozen-V2.  No scale factor, chi2
tuning, outlier drop, or angle-based rejection.

## Repair scope (closed)

Reconstruct the GetState map from the Athena `CurvilinearUVT` definition on
**both** branches:

1. Full slot map: solve `loc1·curvU_xy + loc2·curvV_xy = (x, y)` from
   `(curvU, curvV)`.  On `|t·z|≥0.99` this is only approximately
   `loc1≈−y`, `loc2≈+x` (`O(tx)` corrections).  On `|t·z|<0.99`,
   `curvU=−(t×ẑ)` is **not** −ŷ.
2. Jacobian sign: `dφ/dtx = −ty/r²` (same as WB85).

Forbidden: the hardcoded beam map as the only repair, scale factors, chi2
tuning, outlier / angle rejection, residual-based correction,
station-dependent calibration, central-prediction changes.

The repair is applied offline as
`C_rep = J_full J_buggy⁻¹ C_exp J_buggy⁻ᵀ J_fullᵀ`.  The fitted
`[x,y,tx,ty]` state is unchanged.

## Gates (identical to WB83, not modified)

`chi2/ndof ≤ 4`, `Cov(z)` eigenvalues ∈ [0.25, 4], generalized eigenvalues
∈ [0.25, 4], source-disjoint construction/validation consistency.  Campaign
gates are evaluated on the **full** station population.  `|t·z|` subsets are
reported separately; the 3 large-angle tracks are **not** dropped.

## Result

3230 tracklets (3227 beam, 3 large-angle), 0 inversion failures on either
repair.

### Full population (WB83 gates)

| cell | baseline med χ² | WB85 χ² / med / cal | WB86 χ² / med / cal |
| --- | --- | --- | --- |
| construction 0 | 409 | 1.00 / 0.838 / yes | 0.999 / 0.833 / **yes** |
| construction 1 | 397 | 3.68 / 0.851 / **no** | 1.02 / 0.851 / **yes** |
| construction 2 | 415 | 0.987 / 0.849 / yes | 0.987 / 0.849 / **yes** |
| construction 3 | 377 | 0.973 / 0.824 / yes | 0.973 / 0.824 / **yes** |
| validation 0 | 391 | 1.11 / 0.826 / **no** | 0.988 / 0.826 / **yes** |
| validation 1 | 360 | 0.932 / 0.784 / yes | 0.932 / 0.783 / **yes** |
| validation 2 | 361 | 0.910 / 0.820 / yes | 0.909 / 0.820 / **yes** |
| validation 3 | 437 | 1.02 / 0.844 / yes | 1.02 / 0.844 / **yes** |

Baseline still reproduces WB83 (8/8 fail, median χ² ~360–437).  WB85 still
fails the same 2 cells (mean-covariance eigen-gates).  WB86 passes **8/8**
on both source-disjoint splits.

The two previously failing cells recover because MEAN `C_yy` is no longer
inflated:

- construction station 1: `Cov(z)` max 11.6 → 1.21; gen-eig min 0.206 → 0.828
- validation station 0: gen-eig min 0.172 → 0.833

Typical-tracklet median χ² remains ~0.83.  `frac(χ²>4)` stays ~0.01–0.03.

### Beam subset (`|t·z|≥0.99`)

All 8 beam-only cells pass under both WB85 and WB86.  The two repairs agree
to the reported precision wherever `n_large=0`.  WB85 was already a valid
beam-branch contract.

### Large-angle subset (`|t·z|<0.99`)

Three tracks total (construction stations 0 and 1; validation station 0).
`n=1` is below the frozen `min_tracklets=30` gate, so WB83 population gates
are **not** applied to this subset (`wb83_population_gates_applicable=false`).
Per-track diagnostics (not a rejection cut):

| track | `|t·z|` | `C_yy` WB85 → WB86 | χ² WB85 → WB86 |
| --- | --- | --- | --- |
| construction 0 | 0.9820 | 3.28×10⁻³ → 7.97×10⁻⁵ | 9.54 → 2.26 |
| construction 1 | 0.9887 | 0.182 → 8.29×10⁻⁵ | 4190 → 3.64 |
| validation 0 | 0.9890 | 0.150 → 1.03×10⁻⁴ | 227 → 3.17 |

On construction station 1 the general-branch `curvU=−(t×ẑ)` is mostly −x̂,
so the hardcoded beam map `loc1=−y` reintroduces a swap-like `C_yy`
inflation.  That single track accounts for the WB85 MEAN `C_yy` failure.
WB86 restores `C_yy` to the calibrated ~10⁻⁴ mm² scale.  No scale factor,
outlier cut, or angle rejection is used.

## Decision

`segmentfit_full_curvilinear_covariance_contract_validated`.

SegmentFit exported covariance satisfies source-level covariance closure
under the full Athena Curvilinear coordinate contract on both `|t·z|`
branches.  This is **not** an alignment improvement.

`measurement_model_v2_discussion_allowed=true`.
`propagated_covariance_validation_authorized=true`.

Still frozen here:

- `measurement_model_validated=false`
- `real_data_alignment_authorized=false`
- `geometry_write_allowed=false`
- `stage_b_entered=false`
- `faseracts_propagation_entered=false`
- `frozen_v2_alignment_authorized=false`

A later campaign may now **discuss / enter** FaserActs propagated-covariance
validation (Stage B), Measurement Model V2, or a conditional Jacobian
campaign.  This campaign does not enter them.
