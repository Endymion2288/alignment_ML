# Standalone Measurement Transport Contract (Stage B / Task B14T)

Workbook 116.  WB115 established that the frozen measurement
likelihood can be evaluated, differentiated, scaled, and restarted
on events 0/1, but the standalone evaluator could not define
`h_i(theta)` on every WB109 surviving measurement.  The failures
are not one "propagation failed" class:

```
Class M  100043/37   IFT → first downstream hit
                     long magnet-crossing hop
                     maxSteps abort

Class S  100048/86   magnet hop succeeds
                     1234.97 → 1235.86 mm stereo pair
                     globalToLocal / not on surface
```

This task repairs the **transport contract** only.  It does not
change the WB114 statistical model, retune Gauss–Newton, introduce
a prior or ridge, delete 37/86, replace q/p with truth, or enter
B14M / B15 / Measurement Model V2.

```
theta = (loc0, loc1, phi, theta, q/p)
alpha = (loc0, theta)
nu    = (loc1, phi, q/p)
chi2(theta) = Σ r_i(theta)^T R_i^{-1} r_i(theta)
R_i = (0.08 mm)² / 12
```

## What ACTS Kalman actually does

Pinned ACTS 32.0.2 `KalmanFitter` is navigator-driven.  Measurement
surfaces are inserted as **external surfaces**.  The source comment
is explicit: the fitter tries to hit those surfaces **ignoring
boundary checks**.  `Navigator` uses `BoundaryCheck(false)` for
external surfaces and for target-volume initialization.
`SurfaceReached` itself defaults to `BoundaryCheck(true)`.
`s_onSurfaceTolerance = 1e-4` mm.

Kalman **does** apply a measurement update and then continues from
the filtered state.  The standalone likelihood **must not**.  Doing
so would replace the frozen `chi2(theta)` with a stateful filter
objective.

## Official Mode B

The evaluator still hops measurement-to-measurement on one
deterministic trajectory.  The hop target is the **supporting
plane** (`SurfaceReached.boundaryCheck = false`), matching Kalman
external-surface targeting.  `loc0` is the plane-chart coordinate.
`insideBounds` is recorded and is **not** required for `chi2`.

Mode A (WB115 bounded `SurfaceReached`) remains a comparison row.
A 20000-step overlay is a **diagnostic** row only.  Official
`maxSteps` stays 4000.  "It worked with more steps" is not PASS.

## Allowed decisions

- `standalone_measurement_transport_contract_established`
- `reconstructed_state_not_transportable_to_surviving_measurements`
- `measurement_surface_projection_contract_not_established`
- `acts_navigation_transport_contract_not_established`
- `mixed_or_inconclusive`

`physical_nonidentifiability` is not a B14T decision.

Official run `sbb14t_profile_transport_20260907T212851Z_f00dad92`:

```
decision = mixed_or_inconclusive
smoke_gate_passed = false
b14m_reopen_authorized = false
jacobian_contract_established = false
```

Mode B makes `h_i` evaluable on 0/1/37/86.  Event 0/1 matches
WB115.  Event 37 is monotonic magnet crossing onto the supporting
plane (misses the active wafer).  Event 86 is on the stereo plane
but ~19 μm outside loc0 bounds.  The 86 Jacobian spot check fails,
so the optimizer stays closed.

This is not a B14M PASS and does not authorize the 1989-row campaign.
