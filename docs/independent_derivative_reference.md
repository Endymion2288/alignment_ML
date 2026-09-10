# Independent Derivative Reference and Missing Transport Coupling (Stage B / Task B14W)

Workbook 122.  WB121 froze two different residuals: control `100043/1`
`loc1` is a real analytic-vs-converged-FD mismatch, and `100048/86` has
no certifiable FD reference.  This task splits those problems.  It
locates the extra 0.001–0.002 loc1 coupling on control 1 and tries to
build an independent 86 reference from the already-frozen
`h, h/2, h/4, h/8` rungs.

```
source bound
  → unbound-to-free
  → RK variational transport
  → material interaction / curvilinear reset (mean Jacobian only)
  → surface crossing
  → supporting-plane intersection
  → free-to-bound continuation
  → next-hop source
```

It does not change the official `rk_free` Jacobian, retune Gauss–Newton,
change production `stepTolerance`, add an FD rung, shrink the FD step,
pick a best FD step or tolerance, relax the frozen 5% gate, switch the
official sequential likelihood to direct-from-source, replace the
official likelihood, introduce a prior or ridge, delete 37/86, use
truth q/p, repair 5D Cin, or enter B14M / B15 / Measurement Model V2.
Dummy-cov bounded `transportJacobian` remains forbidden.  Athena is
not rerun; the frozen WB120 dumps are the only smoke input.

```
theta = (loc0, loc1, phi, theta, q/p)
alpha = (loc0, theta)
nu    = (loc1, phi, q/p)
chi2(theta) = Σ r_i(theta)^T R_i^{-1} r_i(theta)
R_i = (0.08 mm)² / 12
r_i = m_loc0 - predicted_loc0_on_supporting_plane
```

A small residual impact is not a contract pass.  Control 1's extra
loc1 coupling is still only ~0.2% of the strip sigma at the official
0.01 mm step; the frozen 5% gate still fails.

## Allowed decisions

- `missing_deterministic_transport_coupling_identified`
- `segment_variational_derivative_inconsistent`
- `derivative_composition_or_reparameterization_incomplete`
- `independent_reference_established_for_focus`
- `independent_reference_not_established`
- `mixed_or_inconclusive`

`jacobian_contract_established` and `b14m_reopen_authorized` require
control 0/1/37 PASS, an independent 86 reference, official-path
analytic agreement with that reference, no branch switching, zero
target leakage, and an unchanged 5% gate.  Identifying the missing
coupling authorizes a later derivative-implementation repair, not a
repair inside this task.

Official run `sbb14w_independent_reference_20260908T204204Z_d305d4b7`:

```
decision = missing_deterministic_transport_coupling_identified
smoke_gate_passed = true
jacobian_contract_established = false
b14m_reopen_authorized = false
control1_station0_loc1_agrees = true
rk_magnet_hop_pos_to_dir_is_zero = true
material_reset_is_highest_priority_mechanism = false
focus_independent_reference_established = false
five_percent_gate_unchanged = true
small_physical_effect_does_not_pass_contract = true
next_step = repair_missing_pos_to_dir_variational_coupling_then_recontract
```

## What this run showed

On `100043/1`, station-0 loc1 is stereo `±0.02` for both FD and
`rk_free` (ΔJ ~ 1e-13).  The first `|ΔJ| > 5e-5` is always measurement
6, the first long magnet hop.  That hop's free-transport product has
`∂dir/∂pos = 0`, the last-reset `jacTransport` has the same zero, and
the continuation loc1 column is `(-0.040, 0.9992, 0, 0, 0, 0)`.  Later
0-step hits inherit the extra 0.001–0.002 FD coupling.  The extra
sensitivity therefore never enters the `rk_free` chain because loc1
never couples into direction through B.

Material resets occur on that hop, but they are not the primary leak:
destination SCT planes have no surface material, and `∂dir/∂pos`
is already zero in the full RK product.  Covariance transport and
process noise were not mixed into the mean-state derivative.

Station-0 segments close against the frozen full-chain FD.  No new
hop-start Athena FD was computed.  Zero-step inheritance localizes the
leak to the magnet-hop variational omission, not to dumped-matrix
composition (already closed at 0 relative error in WB121).

Focus `100048/86` still has no independent reference.  Richardson on
the frozen four rungs is `not_applicable` (oscillatory / sign-changing).
The local odd polynomial is diagnostic only and does not replace
official FD.  ACTS variational self-reference is forbidden until
control 1 is repaired.  Per-hit, station 0 is stable; the first
unstable hit is the first remaining long hop, then downstream hits
drift coherently.  Do not shrink the global FD step.

This is not a B14M PASS and does not authorize restart invariance or
the 1989-row campaign.
