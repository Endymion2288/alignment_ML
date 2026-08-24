# Module-Level Residual & Identifiability Proof-of-Concept V1

## Scope

This stage does not train a new association model, does not change the
frozen V2 checkpoint, route policy, threshold, or unmatched penalty, and
does not reopen Station Mode, reduced Station Mode, or `C_dx` Mode.  It
does not write official geometry and does not solve an alignment
correction.

The only question is whether the station-level `dx/ry` versus internal
`C_dx` degeneracy is mainly measurement compression.

## Inputs

Frozen V2 selected routes on real run 14973
(`data24_r14973_00007_skip49500_n84988`) together with the already-produced
enhanced ntuple `TrackletHit_*` identifiers.

## Mapping and residuals

Each selected route is joined to SCT cluster / module identifiers:

```text
V2 selected route → station tracklet → SCT cluster → layer → module
```

Calypso/Acts unbiased cluster residuals and cluster global positions are
not in the current export.  The PoC therefore uses the tracklet intercept
at the nominal layer `z` and a leave-one-station-out straight-line
prediction.  That quantity is a **module-proxy unbiased residual**.  It
is not a complete Acts cluster residual and must not be labelled as one.
In-sample biased residuals are not used as alignment observables.  Any
residual width or decrease is a **DQ observable** only.

## Finite-difference smoke test

One representative station (IFT, station 0), one layer (layer 0), and
two to four modules are perturbed in software (`dx/dy/dz/rx/ry/rz` and
layer-antisymmetric `C_dx`).  The probe checks that the payload moves
only the target module, that `dx` and `C_dx` derivative signs are
correct, and that two step sizes stay in the linear region.  This is not
a Calypso conditions write.

## Identifiability and leakage

`J_module = ∂r_x/∂θ` is analysed for rank, singular spectrum, condition
number, Fisher information, parameter correlation, and weak/null modes.
No correction is computed.

The quantitative test compares

```text
|cos(J_station_dx, J_C_dx)|
|cos(J_station_ry, J_C_dx)|
```

in the station-recompressed observation (compressed tracklet propagated
to the other-station `z`, the field-edge equivalent of
`cluster → tracklet` compression) versus the module-proxy residual
space.  It also reports the rank of the `{station dx, station ry, C_dx}`
leakage subspace.

## Go / No-Go

The only question:

> Does the module-level observable actually lift the current station
> `dx/ry` versus internal `C_dx` degeneracy, so that a full-detector
> module-level alignment basis study is warranted?

**Yes** (`module_level_observable_restores_identifiability_candidate`):
both absolute cosines drop significantly and the leakage subspace stays
resolvable (rank 3).  The next allowed step is a full-detector
module-level identifiability map.  Still no new network.

**No, mixed**: only `dx↔C_dx` drops.  Measurement compression explains
the translational degeneracy, but `ry/C_dx` remains limited by
near-straight three-plane track topology.  Overall Phase 1 is No-Go.

**No, `track_topology_limited`**: both pairs stay collinear, leakage
rank `< 2`, or the module Jacobian is more singular.  Stop sinking into
more complex ML.  Prefer external survey or a new track topology.

Implied `|C_dx|` is not a measurement.
