# Supporting-Plane Jacobian Continuity Contract (Stage B / Task B14J)

Workbook 117.  WB116 made `h_i(theta)` evaluable on every surviving
measurement for smoke events 0/1/37/86 by targeting the supporting
plane.  The remaining blocker was the 86 Jacobian spot-check.  This
task audits whether that map is locally continuous and has a
certifiable Jacobian.  It does not change the WB114 statistical
model, retune Gauss–Newton, pick a best FD step, switch the official
sequential likelihood to direct-from-source, introduce a prior or
ridge, delete 37/86, replace q/p, repair 5D Cin, or enter B14M / B15
/ Measurement Model V2.

```
theta = (loc0, loc1, phi, theta, q/p)
alpha = (loc0, theta)
nu    = (loc1, phi, q/p)
chi2(theta) = Σ r_i(theta)^T R_i^{-1} r_i(theta)
R_i = (0.08 mm)² / 12
```

Official `h_i` remains one deterministic sequential supporting-plane
trajectory.  Direct-from-source is a pre-registered comparison only.

## Fixed FD ladder

Every official column is differentiated with the pre-registered
central steps

```
h(loc0) = 0.01 mm
h(loc1) = 0.01 mm
h(phi)  = 1e-5
h(theta)= 1e-5
h(q/p)  = 1e-6 /GeV
```

and the fixed rung set `h, h/2, h/4, h/8`.  The audit records all
four rungs.  It does not select the rung with the smallest error as
the official Jacobian.

Each perturbation records predicted loc0/loc1, χ², propagation
steps, path length, geometry/surface sequence, inside-bounds,
continuation-state construction, free-to-bound fallback, and final
direction.

## Allowed decisions

- `supporting_plane_jacobian_continuity_established`
- `finite_difference_step_not_in_asymptotic_region`
- `acts_navigation_material_branch_switching`
- `supporting_plane_continuation_state_discontinuous`
- `source_to_measurement_transport_not_smooth`
- `mixed_or_inconclusive`

If direct PASS and sequential FAIL, the root cause is preferentially
the post-supporting-plane continuation / sequential hop.  If both
FAIL, the source→measurement map itself remains the object.  A
numerically better direct Jacobian does not replace the official
likelihood.

Official run `sbb14j_jacobian_continuity_20260908T150238Z_a30b79e2`:

```
decision = source_to_measurement_transport_not_smooth
smoke_gate_passed = false
jacobian_contract_established = false
b14m_reopen_authorized = false
restart_invariance_authorized = false
```

Events 0/1/37 converge on the full ladder with sign and surface/path
consistency.  Event 86 fails for both sequential and direct on
`loc1 / phi / q/p`; smaller steps make the relative error worse, not
better.  No geometry/continuation branch switch and no free-to-bound
fallback.  The official sequential likelihood is unchanged.

This is not a B14M PASS and does not authorize restart invariance or
the 1989-row campaign.
