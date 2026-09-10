# Official-Path Derivative Residual Contract (Stage B / Task B14V)

Workbook 121.  WB120 already built the same-path `rk_free` Jacobian of
official Mode-B `h_i(theta)` and proved that function values match
official predicted loc0.  This task only classifies the two remaining
mismatches against the frozen `h, h/2, h/4, h/8` ladder.

```
source bound
  → unbounded navigator/stepper (nullopt cov, BoundaryCheck false)
  → final free state
  → supporting-plane intersection
  → local loc0
  → residual = m_loc0 - loc0
```

It does not change the Jacobian implementation, retune Gauss–Newton,
change production `stepTolerance`, pick a best FD step or tolerance,
relax the frozen 5% gate, switch the official sequential likelihood to
direct-from-source, replace the official likelihood, introduce a prior
or ridge, delete 37/86, use truth q/p, repair 5D Cin, or enter B14M /
B15 / Measurement Model V2.  Dummy-cov bounded `transportJacobian`
remains forbidden.  Athena is not rerun; the frozen WB120 dumps are
the only smoke input.

```
theta = (loc0, loc1, phi, theta, q/p)
alpha = (loc0, theta)
nu    = (loc1, phi, q/p)
chi2(theta) = Σ r_i(theta)^T R_i^{-1} r_i(theta)
R_i = (0.08 mm)² / 12
r_i = m_loc0 - predicted_loc0_on_supporting_plane
```

The pre-registered consistency envelope is `||J_h/4 - J_h/8||`.  It is
only a diagnostic of whether analytic-vs-FD sits inside the FD self-
difference.  It does not change the 5% gate and does not select a rung.

## Allowed decisions

- `official_path_derivative_contract_established`
- `fd_reference_not_precise_enough_for_derivative_certification`
- `small_column_relative_metric_pathology`
- `official_path_derivative_still_inconsistent`
- `mixed_or_inconclusive`

`jacobian_contract_established` and `b14m_reopen_authorized` require
the first case **and** control 0/1/37 PASS **and** 86 PASS.  Pathology
does not auto-relax the gate.  Even a PASS only reopens B14M smoke
restart invariance, not the 1989-row campaign.

Official run `sbb14v_official_residual_20260908T194821Z_c5b463f5`:

```
decision = mixed_or_inconclusive
smoke_gate_passed = true
jacobian_contract_established = false
b14m_reopen_authorized = false
free_state_jacobian_available = true
official_loc0_matches_diagnostic = true
control_fd_converged = true
control_five_percent_pass = false
focus_fd_converged = false
segment_composition_closed = true
step_level_D_persisted_in_acts = false
five_percent_gate_unchanged = true
active_residual_kinds =
  official_path_derivative_still_inconsistent,
  fd_reference_not_precise_enough_for_derivative_certification
next_step = keep_residual_diagnosis_without_shrinking_fd
```

## What this run showed

Control `100043/0` and `100043/37` still pass the frozen 5% column
contract on all five parameters and all three targets.

Control `100043/1` `loc1` on targets 1 and 2 is **not** a small-column
relative-gate pathology.  The FD ladder has already converged (last-pair
rel ≈ 7e-4).  Analytic is exactly the stereo projection `±0.02` on every
hit; official FD keeps that on the first six hits and then develops an
extra 0.001–0.002 transport coupling.  `|J_a-J_FD|` is about 90× the
last-pair envelope, so the 6.16% / 5.64% relative errors are a real
local derivative discrepancy.  The residual effect at the official
0.01 mm loc1 step is only 0.2% of the strip sigma, but that does not
relax the gate.

Focus `100048/86` is `fd_reference_not_precise_enough`.  The FD four
rungs do not converge.  Analytic is the same-order, same-path `rk_free`
column already certified on the controls.  Analytic-vs-FD differences
are the same order as the FD self-differences, so shrinking the FD step
is forbidden.  The next reference must be independent of this ladder.

Hop-level composition on events 1 and 86 closes at 0 relative error:
`jRkFreeAcc * boundToFree(start) = bound_to_free_rk_product`, and
`jacTransport * jacToGlobal * jacobian = acts_composed`.  Official
minus diagnostic loc0 remains 0.  ACTS 32.0.2 EigenStepper computes
the per-step `D` in `GenericDefaultExtension::transportMatrix` and
multiplies it into `jacTransport` in place; it does not persist
step-level D or a higher-precision variational state.  The allowed
readout is the hop-level product already dumped in WB120.

This is not a B14M PASS and does not authorize restart invariance or
the 1989-row campaign.
