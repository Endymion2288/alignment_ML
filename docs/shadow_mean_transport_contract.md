# Production-vs-Shadow Mean Transport Contract (Stage B / Task B14ZA)

Workbook 126. Incoming freeze is **WB125**, not WB124. WB125 removed
common-grid FD branch noise and closed path length, but the independent
shadow mean still did not reproduce the production EigenStepper mean.
This task **stops all derivative / Jacobian certification** and asks
only: why the same field, surface energy loss, supporting plane, and
nominal path still fail to close the function values.

```
WB125:
common-grid removes FD branch noise
BUT shadow mean != production mean
        ↓
WB126:
STOP DERIVATIVES
reconstruct the deterministic production mean
close shadow function values first
        ↓
shadow_mean_contract_established
        ↓
NEXT workbook only:
independent common-grid derivative reference
```

Required hops stay frozen:

```
target 1: hit 6
target 2: hit 11
target 3: hit 11
```

Do not evaluate Jacobian agreement. Do not retune the WB125 20/10/5 mm
recipe from leftover FD files. A new mean-only sequence `10, 5, 2.5,
1.25 mm` is pre-registered from RKN4 order and the FASER magnet-cell
scale.

## Source-backed production mean

Pinned ACTS 32.0.2 / AthenaExternals 24.0.41:

- Field update is **RKN4 / Nyström** (`EigenStepper.ipp` +
  `GenericDefaultExtension.hpp`), not classical RK4.
- After each accepted step the direction is explicitly `normalize()`d.
- `q/p` is constant on a field interval (`kQoP = 0`).
- `MaterialInteractor` runs after arrival: `evaluateMaterialSlab`
  (start=`PostUpdate`, target=`PreUpdate`, else `FullUpdate`) then
  `pathCorrection` scales the slab, then
  `evaluatePointwiseMaterialInteraction`, then `updateState` changes
  **q/p only**.
- Volume material is recorded, not applied to the ACTS mean.

## Official result

```
decision = surface_energy_loss_semantics_mismatch
verdict  = FAIL
```

Shadow P (official RKN4 on the production accepted-step sequence)
matches production position and direction **until the first surface
material node**. The first physical divergence is always
`surface_material`, with `Δpos ≈ 0` and `Δdir ≈ 0`. The split is
`q/p`.

Production `ΔE` on the first T1 slab sits **between**
`computeEnergyLossMean` and `computeEnergyLossMode` applied to the
recorded (already path-corrected) slab at the recorded `q/p`. Units
are internally consistent (`q/p` in 1/GeV, Eloss in GeV, thickness in
mm, muon mass `0.1057 GeV`). Some later recorded surfaces have
production `Δq/p = 0` while `computeEnergyLossMean` is finite.

Normalization is source-supported and does not explain the residual.
The mean-only `10/5/2.5/1.25 mm` sequence plateaus; this is not an
integrator-resolution problem.

Frozen flags stay:

```
control 0/1/37 PASS = true
mean_path_unchanged = true
focus_independent_reference_established = false
jacobian_contract_established = false
b14m_reopen_authorized = false
```
