# Focus-86 Common-Grid Shadow Segment Reference (Stage B / Task B14Z)

Workbook 125.  Incoming freeze is **WB124**, not WB123.  WB123 repaired
the field-gradient tangent and passed `100043/0,1,37`.  WB124's
adaptive DOPRI5 independent FD failed because `+δ/−δ` took different
accepted-step sequences, and its vacuum mean still differed from the
production segment by up to 0.044 mm.  This task does **not** retune
that DOPRI5, does **not** add or shrink a track-state FD rung, and
does **not** modify `FieldGradientDefaultExtension`.  It only builds a
common-grid shadow of the **same physical production mean** for
`100048/86` required hops `1/6, 2/11, 3/11`.

```
WB124:
adaptive independent FD failed
(step-sequence noise + vacuum mean gap)
        ↓
WB125:
frozen common-grid shadow
same physical mean contract
(field + surface energy-loss + supporting plane;
no process noise)
        ↓
ONLY IF C itself converges, is grid-stable,
mean contract holds, and A≈C at 5%
on 1/6, 2/11, 3/11 AND controls still PASS
        ↓
jacobian_contract_established
        ↓
reopen B14M smoke only
```

Required hops stay frozen.  Do not substitute the already-converged
earlier hit 6 on targets 2/3:

```
target 1: hit 6
target 2: hit 11
target 3: hit 11
```

Official `h_i(θ)` is unchanged.  Production `stepTolerance` stays
`1e-4`.  The frozen FD ladder stays `h, h/2, h/4, h/8`.  The 5% gate
is not relaxed.  No extra rung, no smaller track-state FD step, no
WB124 DOPRI5 retune, no prior, ridge, truth q/p, Q / Cin, B14M, B15,
V4, Measurement Model V2, or 1989.  Analytic self-certification is
forbidden.  A vacuum ODE must not be called a production-map
reference.  Target 3 production loc1 3.53% PASS must not alone
authorize the hop.

```
theta = (loc0, loc1, phi, theta, q/p)
alpha = (loc0, theta)
nu    = (loc1, phi, q/p)
chi2(theta) = Σ r_i(theta)^T R_i^{-1} r_i(theta)
R_i = (0.08 mm)² / 12
r_i = m_loc0 - predicted_loc0_on_supporting_plane
```

## Allowed decisions

- `focus_common_grid_reference_established`
- `shadow_mean_contract_not_established`
- `field_map_interpolation_nonsmoothness`
- `material_map_discontinuity`
- `supporting_plane_terminal_event_sensitivity`
- `shadow_integrator_resolution_unresolved`
- `repaired_variational_focus_inconsistent`
- `mixed_or_inconclusive`

Certification is not `A≈C` alone.  `C` itself must converge on the
frozen track FD ladder, be stable under pre-registered shadow-grid
refinement, close the mean contract, and agree with `A` at 5% on
`loc1/phi/q/p` of all three required hops.  Production `B` may keep
oscillating.

If the nominal shadow mean does not close, the decision is
`shadow_mean_contract_not_established` and derivatives are not
certified.

## Shadow integrator (diagnostic only)

`alignment/leave_target_out_dump/CommonGridShadowIntegrator.hpp`

```
method: classical_rk4_common_grid
NOT EigenStepper / GenericDefaultExtension /
    FieldGradientDefaultExtension / WB124 DOPRI5
field:  official FASERMagneticFieldWrapper::getField
eloss:  Acts::computeEnergyLossMean on frozen surface slabs
MS / process noise / covariance: not applied
```

The mesh is built from the **nominal unperturbed hop only**.  Every
`±h…±h/8` arm reuses that frozen node sequence, the same material
partition, and the same RK4 stages.  Grid sizes and mean-closure
gates were registered **before** any 86 Jacobian comparison:

```
coarse_max_step_mm   20
nominal_max_step_mm  10    # one FASER magnet-cell scale
fine_max_step_mm      5
mean_loc0_abs_mm      1e-3
mean_path_abs_mm      1e-2
mean_pos_abs_mm       1e-2
mean_dir_abs          1e-6
mean_qop_rel          1e-4
grid_deriv_rel_max    0.05
grid_mean_loc0_abs_mm 1e-3
```

Grid refinement acts only on the shadow mesh, never on the track
FD ladder.  Coarse / nominal / fine were all pre-registered.

## Numerical freeze (WB125)

Official run: `sbb14z_focus86_common_grid_shadow_20260909T130344Z_2aa503bd`

```
decision = shadow_mean_contract_not_established
control_0_pass = true
control_1_pass = true
control_37_pass = true
mean_path_unchanged = true
target_exclusion_holds = true       # 12 rows, 0 leaked
focus_independent_reference_established = false
jacobian_contract_established = false
b14m_reopen_authorized = false
next_step = keep_86_common_grid_shadow_mean_contract
```

Production mean on the required hops contains magnetic field,
supporting-plane termination, and deterministic surface energy loss
(`n_surface = 5, 4, 1`).  Volume material is recorded and not applied
to the ACTS mean.  Process noise is not in the mean.  Path length
matches production exactly.  Common-grid `branch_identity_same` is
true on every arm, so WB124 accepted-step branch noise is gone.

The nominal shadow does **not** close the pre-registered mean
contract: T1/6 loc0 residual 0.015 mm, T2/11 0.00131 mm, T3/11
direction and q/p residuals miss the dir / qop gates.  Because the
function values are not closed, derivatives are not certified.  Do
not retune the mean gates or the common grid from Jacobian
agreement.  Do not go back to retuning track FD or WB124 DOPRI5.

Config SHA `547d727517659b2f5e8c20a14f282982fdfb71e0d4d34483b6ce640eba952dcc`.
Helper SHA `3c1f1d41a79b2ff179173088d82057bcc31362822481f63c1e81610699763844`.
Shadow integrator SHA `be1e086040adc4e2c1b45732ceb21a0b49422731843655a322ffe3d049d8dce5`.
Decision SHA `15fbffbd06ef6747d3adecec9cdb6e64a0be9c9986bd08703e8e9c553f49d5c2`.
Field-gradient extension SHA unchanged
`ca5e4f0ef1a7a09edd1c24099e7ce689c4f0511e2d4afab3a160ebd2d6ff03d8`.
