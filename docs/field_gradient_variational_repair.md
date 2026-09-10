# Field-Gradient Variational Coupling Repair (Stage B / Task B14X)

Workbook 123.  WB122 located the extra 0.001–0.002 `loc1` coupling on
`100043/1`: the first long magnetic hop has `rk_free ∂dir/∂pos ≡ 0`,
while official mean FD sees a spatially varying field.  This task
proves that ACTS 32.0.2 `GenericDefaultExtension::transportMatrix`
omits `∂B/∂x`, completes that same mean RKN4 map on a
**diagnostic-only** stepper, and recontracts `100043/0,1,37` plus
`100048/86`.

```
WB122:
missing coupling identified
magnetic long hop: ∂dir/∂pos incorrectly zero
        ↓
WB123:
prove field-gradient tangent term
+
repair derivative-only variational propagation
+
segment-start FD falsification
        ↓
control 0/1/37 derivative contract
        ↓
86 independent segment reference
```

Official `h_i(θ)` is unchanged.  Only the diagnostic tangent is
repaired.  Production `stepTolerance` stays `1e-4`.  The frozen FD
ladder stays `h, h/2, h/4, h/8`.  The 5% gate is not relaxed.  No
prior, ridge, truth q/p, Q / Cin tuning, extra FD rung, smaller
track-state FD step, B14M, B15, V4, Measurement Model V2, or 1989
campaign.

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

- `field_gradient_variational_coupling_repaired_and_contracted`
- `field_gradient_coupling_repaired_focus_reference_unresolved`
- `field_gradient_hypothesis_not_supported`
- `field_gradient_variational_implementation_inconsistent`
- `mixed_or_inconclusive`

`jacobian_contract_established` and `b14m_reopen_authorized` require
Case A only: missing `∂dir/∂pos` source-proven, field-gradient
contract established, official mean path unchanged, control 0/1/37
PASS at the frozen 5% gate, and 86 segment independent reference
established by repaired tangent ≈ hop-start segment FD.  Even Case A
does not submit 1989.

## Source audit (must precede any patch)

Pinned stack:

```
ACTS 32.0.2
AthenaExternals 24.0.41
EigenStepper
GenericDefaultExtension::transportMatrix
FASERMagneticFieldWrapper
```

Mean ODE in `EigenStepper.hpp`:

```
dr/ds = T
dT/ds = (q/p) T × B(x)
```

RKN4 stages in `EigenStepper.ipp` sample `B` at

```
pos0 = x
pos1 = x + (h/2) T + (h²/8) k1
pos2 = x + h T + (h²/2) k3
```

with `k = (q/p) T_stage × B(x_stage)` and `kQoP = {0,0,0,0}` in the
default extension (no energy-loss in `k`).

`GenericDefaultExtension::transportMatrix` fills only `dFdT`, `dFdL`,
`dGdT`, `dGdL`, and `D(3,7)`.  The header says the terms of
ATL-SOFT-PUB-2009-002 eq. 18 are currently 0, and `dGdx` is left as
the Identity-initialized zero block.  It never calls
`getFieldGradient`.  That is why accumulated `rk_free` has
`∂dir/∂pos ≡ 0`.

Distinguish and do not mix:

```
uniform-field tangent     ← already in ACTS D
field-gradient tangent    ← missing; this task
energy-loss deterministic
multiple-scattering / process noise
covariance transport      ← dummy cov is only the variational switch
```

This task repairs only the deterministic mean-state variational
derivative.

## Same-unit tangent

The diagnostic extension keeps the ACTS mean `k_i` and differentiates
that map:

```
dk = qop * (dT_stage)×B + qop * [T_stage]_× * G * dx_stage
```

`G = ∂B/∂x` comes from the official
`FASERMagneticFieldWrapper::getFieldGradient` at the same three stage
positions.  Wrapper units: native kT → ACTS Tesla, same factor on
`G` (Tesla / mm).  If every `G = 0`, `D` reduces to ACTS
`GenericDefaultExtension::transportMatrix`.  Coefficients are not
hand-written from a schematic ODE; they are the RKN4 derivative of
the pinned mean step.  Energy-loss, process noise, and a
direction-normalization Jacobian are not added.

## Field-gradient contract

Official gradient API exists.  A 1 mm field-query FD is a
**diagnostic of official G only**, registered before any track
Jacobian comparison from the magnet-region mesh of O(10 mm).  It is
not the official `G` and is not tuned from track-Jacobian agreement.

## Diagnostic-only repair

```
using GradientStepper =
    Acts::EigenStepper<Acts::StepperExtensionList<FieldGradientDefaultExtension>>;
```

Official `EigenStepper<>` still evaluates `h_i(θ)`.  Dummy covariance
still only flips `covTransport` on the diagnostic hop.  Required
invariance:

```
predicted_loc0 official == predicted_loc0 gradient-stepper mean
(|Δloc0| ≤ 1e-6)
chi2 from official mean unchanged
```

Hop-start segment FD uses the official mean stepper on the real
hop-start state with the frozen `±h … ±h/8` ladder.  Only the first
long hop per arm is segmented.  Analytic self-reference is forbidden:
86 is certified only if that **required first-unstable** segment FD
converges and agrees with the repaired tangent under the frozen 5%
gate.

## Numerical freeze (WB123)

Official run: `sbb14x_field_gradient_repair_20260908T235722Z_f86bd896`

```
decision = field_gradient_coupling_repaired_focus_reference_unresolved
missing_term_source_proven = true
mean_path_unchanged = true          # Δloc0 = 0 on all 12 rows
control_0_pass = true
control_1_pass = true              # loc1 6.16%/5.64% → 0.016%/0.035%
control_37_pass = true
focus_independent_reference_established = false
jacobian_contract_established = false
b14m_reopen_authorized = false
next_step = keep_86_independent_segment_reference
```

`100043/1` hit 6: old `∂dir/∂pos = 0`, new `2.46e-6` (target 1).
The inherited 0.001–0.002 loc1 coupling is gone.  Hop-start segment
FD on every control first-long hop converges and agrees with the
repaired hop.  `100048/86` target 1 hit 6 segment FD still oscillates
(`last_pair_rel = 0.56`); targets 2/3 required hit 11 have no
hop-start FD.  Global 86 loc1 improved (98% → 6–27%) but the frozen
ladder still changes sign / does not converge.  Do not self-certify
the repaired analytic.

Config SHA `c290594f834233ace2c41d2d339edaa6fd9a4b6b88dc781ece67e014a012906b`.
Helper SHA `ff74be09335bd6c63cd1357570d4623563cef7a3bca921909a456c0f5208bbf2`.
Decision SHA `4632686ec5e68f4d30470438d257c50ec7056edf430ea2460453052fd0b995aa`.
