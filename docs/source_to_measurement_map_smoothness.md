# Source-to-Measurement Map Smoothness Root-Cause Audit (Stage B / Task B14K)

Workbook 118.  WB117 established that the official sequential
supporting-plane map `h_i(theta)` is not a certifiable Jacobian on
`100048/86` for `loc1 / phi / q/p`.  That decision is **not** a
physical claim that the map is nondifferentiable.  This task splits
`h_i` into stages and asks which continuous numerical operation first
fails to form a stable derivative.

It does not change the WB114 statistical model, retune Gauss–Newton,
pick a best FD step or `stepTolerance`, switch the official sequential
likelihood to direct-from-source, introduce a prior or ridge, delete
37/86, replace q/p, repair 5D Cin, or enter B14M / B15 / Measurement
Model V2.

```
theta = (loc0, loc1, phi, theta, q/p)
alpha = (loc0, theta)
nu    = (loc1, phi, q/p)
chi2(theta) = Σ r_i(theta)^T R_i^{-1} r_i(theta)
R_i = (0.08 mm)² / 12
```

Official `h_i` remains one deterministic sequential supporting-plane
trajectory.  Production `PropagatorPlainOptions::stepTolerance`
remains the ACTS 32.0.2 default `1e-4`.

## Allowed decisions

- `transport_integration_resolution_insufficient`
- `supporting_plane_projection_ill_conditioned`
- `parameterization_scale_not_resolved`
- `acts_transport_jacobian_inconsistent_with_fd`
- `source_to_measurement_map_genuinely_nonsmooth`
- `mixed_or_inconclusive`

`source_to_measurement_map_genuinely_nonsmooth` is allowed only after
determinism PASS, integration FAIL to improve, projection checked,
FD signal above numerical noise, parameter scaling checked, and no
branch switching.  Presence of an unused ACTS transport Jacobian is
not an inconsistency.

Official run `sbb14k_map_smoothness_20260908T163845Z_2cbd5d0e`:

```
decision = mixed_or_inconclusive
smoke_gate_passed = true
jacobian_contract_established = false
b14m_reopen_authorized = false
restart_invariance_authorized = false
acts_transport_jacobian_available = true
analytic_vs_fd_not_yet_contracted = true
next_step = open_wb119_analytic_vs_fd_derivative_contract
```

The earlier classifier run `...T163048Z_6e7351eb` that printed
`acts_transport_jacobian_inconsistent_with_fd` is superseded.  ACTS
`Result::transportJacobian` is available, but no contracted
analytic-versus-FD comparison was performed in B14K.

## What this run showed

Transport on identical `theta` is bitwise/numerically deterministic
on both `100048/86` and the pre-registered same-source control
`100048/44`.  Repeat noise in predicted loc0 and χ² is exactly 0.

The frozen-plane projection map is smooth.  `|n·d| ≥ 0.99986` on
every hop; incidence is a few × 10⁻² rad; not near-parallel.

Official FD signal is above that zero noise floor.  The official
`h(q/p) = 1e-6 /GeV` is `8.4e-5` of `|q/p|` on 86, not a huge
nonlinear kick.

On 86, hop 0 `loc1 / phi` stage Jacobians converge at the 1e-9–1e-12
level.  The first real failure is a later sequential magnetic hop:

- target 1: hop 6 `free_near_plane` (`z ≈ -1822 → 1203` mm, 3.0 m, 129 steps)
- targets 2/3: hop 11 (`phi` at `free_near_plane`; `loc1` at `predicted_loc0`)

Hop-0 `q/p` `plane_intersection` slightly misses the 0.05 last-pair
cut on both 86 and 44.  That is a tiny first-hop loc0 response, not
the official residual Jacobian (44 still PASSes).

Tighter pre-registered `stepTolerance` rungs `1e-4 / 1e-5 / 1e-6`
improve 86 target 1 and 3 last-pair means (`2.91 → 0.94 → 0.25` and
`1.21 → 0.61 → 0.21`) and make `loc1` converge at `1e-6`, but `phi`
and `q/p` still fail.  Target 2 is not monotone.  This is not
`transport_integration_resolution_insufficient` (last rung does not
fully converge) and must not be used to pick a production tolerance.

Control 44 converges at every official rung.  86 is kinematically
isolated in the source (nearest WB109 score still has relative q/p
~ 1).

ACTS diagnostic `transportJacobian` is present when a dummy
covariance is attached.  Official production still uses `nullopt`
covariance and finite differences.  ACTS end `loc0` matches the
official supporting-plane `loc0` on every diagnostic hop.  A fair
column-to-column contract is WB119, not a production replacement.

This is not a B14M PASS and does not authorize restart invariance or
the 1989-row campaign.
