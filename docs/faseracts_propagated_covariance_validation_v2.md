# FaserActs Propagated Covariance Validation V2

Workbook 87. **Status: completed and frozen.**
**Final decision: `faseracts_transport_covariance_not_validated`.**
**Mechanism: `q_over_p_uncertainty_semantics`.**

This is the authorized Stage B after WB86.  It asks whether

    C_target = J_transport C_source_WB86 J_transport^T + process_noise

describes `e_target = propagated fitted state − truth target state` on the
frozen WB83/86 source-disjoint MC.  It is **not** an alignment task.  WB81
mode 3 is **not** inherited as the answer.

**Inherited frozen state (SHA-verified, not reopened):**

- WB81 `faseracts_propagated_covariance_not_calibrated` /
  `overestimated_transported_fit_covariance`
- WB82 `existing_mc_real_wide_ty_support_validated` (`floor_muon_100120`, reserved only)
- WB83 `source_tracklet_fit_covariance_not_calibratable`
- WB84 `deterministic_segmentfit_get_state_transform_bug`
- WB85 `segmentfit_getstate_repair_insufficient`
- WB86 `segmentfit_full_curvilinear_covariance_contract_validated`

`geometry_write_allowed=false`, `real_data_alignment_authorized=false`,
`measurement_model_validated=false`.  No real residual, no geometry/conditions
write, no Frozen-V2, no truth q/p as a real-data solution, no chi2 tuning, no
deleting the q/p column as the final scheme.

## Part A — deterministic transport

`C = J_geom C_src_WB86 J_geom^T` with no process noise.  `J_geom` is the
field-free 4D lever-arm Jacobian.  Production ACTS has no material process
noise (WB81 provenance).  Empirically `J_geom C_exported J_geom^T` matches
ntuple mode-3 `C_prop`.

**Jacobian self-consistency** (`J e_source` vs `J C J^T`): **6/6 cells pass**
on both splits (χ²/ndof ≈ 1.00, median ≈ 0.85, `frac(χ²>4)` ≈ 0.025).  The
validated source covariance is transported correctly.  The 4D Jacobian is
**not** the failure.

**`e_target` vs the same `C`:** all 6 cells fail.  Leftover
`e_target − J e_source` lives in `y/ty` and scales with `1/p` (magnetic
transport with unconstrained momentum), not with a 100 GeV Highland term.

## Four modes (all fail `e_target` closure; splits agree)

| mode | model | typical result |
| --- | --- | --- |
| 0 | existing production `C_prop` | χ² ~ 800–1.5×10⁵; pencil **727–43890×** (WB81 reproduced) |
| 1 | WB86 source + existing q/p ΔC | median χ² ~ 0.7–1.0, but pencil still **727–43890×** |
| 2 | WB86 source, dummy q/p **not** transported | pencil **~1.06–1.22** (fixed); χ² **4×10⁴–1.4×10⁶** (y/ty under-estimate) |
| 3 | mode 2 + Highland 5% X0, not chi2-tuned | χ² drops ~10–50×, still **3.8×10³–5.4×10⁴** |

## q/p audit

- **With** dummy q/p (mode 0): artificial pencil, same mechanism as WB81.
- **Without** dummy q/p (mode 2): pencil collapses to ~1; precision-plane
  χ² explodes.
- Dummy SegmentFit `q/p` covariance is an **unconstrained straight-tracklet
  input**, not a physical momentum uncertainty.
- A **physical** `q/p` is required because IFT→downstream transport crosses
  magnetic field.  Deleting the q/p column is **not** the final scheme.
- Truth q/p is diagnostic-only and is not a real-data solution.

## Decision

`faseracts_transport_covariance_not_validated` /
`q_over_p_uncertainty_semantics`.

Source covariance is no longer the limiter.  FaserActs 4D deterministic
transport of that source is validated against `J e_source`.  Production
`C_prop` and every pre-registered reconstruction of
`J C_src J^T (+ Highland)` fail `e_target` closure.  Concurrent facts:
ACTS process noise is disabled; the first-principles Highland term is
insufficient and is not promoted.

`measurement_model_v2_discussion_allowed=false`.
`measurement_model_v2_entered=false`.
`stage_b_entered=true` (this campaign only).

A later campaign may pre-register a **physical momentum prior** for magnetic
covariance transport (not dummy variance, not column deletion, not truth q/p
on real data) and/or an ACTS material process-noise contract.  This campaign
does not enter Measurement Model V2 or alignment.
