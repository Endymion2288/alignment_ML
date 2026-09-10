# Focus-86 Independent Segment Derivative Reference (Stage B / Task B14Y)

Workbook 124.  WB123 repaired the field-gradient tangent and passed
`100043/0,1,37` at the frozen 5% gate with an unchanged official mean.
This task does **not** touch that implementation.  It only builds a
numerical derivative reference for `100048/86` required first-unstable
long magnet segments that is independent of the repaired analytic
tangent.

```
WB123:
field-gradient tangent repaired
controls 0/1/37 PASS
official mean unchanged
86 required-hop reference missing
        ↓
WB124:
true hop-start production FD on 1/6, 2/11, 3/11
+
independent DOPRI5 mean ODE (same B(x), same plane)
+
independent variational (includes proven ∂B/∂x)
dual-closed against that integrator's frozen FD ladder
        ↓
triple: repaired tangent / production FD / independent reference
        ↓
ONLY IF independent reference is stable AND agrees at 5%
        ↓
jacobian_contract_established
        ↓
reopen B14M smoke
```

Required hops are frozen.  Do not substitute the already-converged
earlier hit 6 on targets 2/3:

```
target 1: hit 6
target 2: hit 11
target 3: hit 11
```

Official `h_i(θ)` is unchanged.  Production `stepTolerance` stays
`1e-4`.  The frozen FD ladder stays `h, h/2, h/4, h/8`.  The 5% gate
is not relaxed.  No extra rung, no smaller track-state FD step, no
prior, ridge, truth q/p, Q / Cin tuning, B14M, B15, V4, Measurement
Model V2, or 1989 campaign.  Analytic self-certification is
forbidden.  Independent-integrator tolerances were registered before
any 86 Jacobian comparison and are not retuned.

```
theta = (loc0, loc1, phi, theta, q/p)
alpha = (loc0, theta)
nu    = (loc1, phi, q/p)
chi2(theta) = Σ r_i(theta)^T R_i^{-1} r_i(theta)
R_i = (0.08 mm)² / 12
r_i = m_loc0 - predicted_loc0_on_supporting_plane
```

A small residual impact is not a contract pass.

## Allowed decisions

- `production_fd_not_certifying_but_independent_reference_established`
- `independent_segment_reference_contracted`
- `focus_segment_reference_not_established`
- `repaired_variational_focus_inconsistent`
- `mixed_or_inconclusive`

`jacobian_contract_established` and `b14m_reopen_authorized` require
all three required hops to have an established independent reference
that agrees with the repaired tangent at the frozen 5% gate, plus
control 0/1/37 still PASS and official mean unchanged.  Even that
only authorizes a B14M **smoke** with restart invariance.  It does
not submit 1989.

## Independent integrator (diagnostic only)

The ACTS stack has no second production-ready mean integrator that
can replace EigenStepper.  The offline reference is
`IndependentMeanOdeIntegrator.hpp`: Dormand–Prince 5(4) of the pinned
mean ODE

```
dr/ds = T
dT/ds = (q/p) T × B(x)
d(q/p)/ds = 0
```

`B` and `∂B/∂x` come from `FASERMagneticFieldWrapper` (same units as
production).  The code path is not EigenStepper, not
`GenericDefaultExtension`, and not `FieldGradientDefaultExtension`.
It never replaces official `h_i`.  Vacuum / field-only: no navigator,
no material actor.

Tolerances registered **before** any 86 Jacobian comparison:

```
method            dormand_prince_5_4
abs_tol_pos_mm    1e-6     # loc0 invariance floor
abs_tol_dir       1e-10
abs_tol_qop       1e-14
rel_tol           1e-9
initial_step_mm   1.0
min_step_mm       1e-6
max_step_mm       50.0
max_path_mm       20000
max_steps         100000
plane_hit_abs_mm  1e-6
```

The independent derivative reference still uses the frozen four-rung
track-state FD.  It is established only if that FD sequence itself
converges, and — when the independent variational is present — dual-
closes against it.  Variational-only agreement with the repaired
tangent is not a reference.

## Numerical freeze (WB124)

Official run: `sbb14y_focus86_segment_reference_20260909T092655Z_18ab2ab0`

```
decision = focus_segment_reference_not_established
control_0_pass = true
control_1_pass = true
control_37_pass = true
mean_path_unchanged = true          # official repair Δloc0 = 0
target_exclusion_holds = true       # 12 rows, 0 leaked
focus_independent_reference_established = false
jacobian_contract_established = false
b14m_reopen_authorized = false
next_step = keep_86_independent_segment_reference
```

All three required hops now have hop-start state, production
EigenStepper segment FD, independent mean, independent variational,
and independent four-rung FD.  No branch switch.  Official mean is
unchanged.  Independent FD on `loc1/phi/q/p` does not converge, so
the independent reference is not established.  Do not retune the
pre-registered DOPRI5 tolerances.  Do not self-certify from
independent variational ≈ repaired tangent (those two agree at
≪ 5%; that is not the gate).

Config SHA `46ab8197bdbf40b6fa77c955ca5d5696f0fbbb7692213c2fa39edbe539162119`.
Helper SHA `3558bc6bdedb326a5dc48bba2dc4d57833574b82eb66d1a14fa8106d45d3edf1`.
Decision SHA `f4ba49cffe187929c9eb79ae1e6e604a9852f29b8ef2ed7e8da3f7dcbbd471e7`.
Field-gradient extension SHA unchanged
`ca5e4f0ef1a7a09edd1c24099e7ce689c4f0511e2d4afab3a160ebd2d6ff03d8`.
