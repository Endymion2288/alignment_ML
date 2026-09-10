# ACTS Surface Energy-Loss Mean Semantics Contract (Stage B / Task B14ZB)

Workbook 127. Incoming freeze is **WB126**, not WB125. WB126 proved
that official accepted-step RKN4 matches production position and
direction until the first surface material node, and that production
`ΔE` sits between `computeEnergyLossMean` and `computeEnergyLossMode`.
This task **does not reopen derivatives**. It asks only: which
deterministic energy-loss quantity does production ACTS 32.0.2
`MaterialInteractor` use to update mean `q/p`, and which recorded
material surfaces actually call `updateState`.

```
WB126:
surface energy-loss semantics mismatch
        ↓
WB127:
copy the real production material mean update
close per-node q/p, then the mean function
        ↓
shadow_mean_contract_established
        ↓
NEXT workbook only:
reuse this certified mean shadow for common-grid independent FD
```

Required hops stay frozen:

```
target 1: hit 6
target 2: hit 11
target 3: hit 11
```

Do not evaluate Jacobian agreement. Do not fit an Eloss to the
endpoint. Do not infer material loss from final `loc0`. Field replay
reuses the WB126 official accepted-step RKN4 path.

## Compiled production path

Pinned ACTS 32.0.2 / AthenaExternals 24.0.41 `libActsCore.so`:

```
MaterialInteractor
  → evaluateMaterialSlab
  → evaluatePointwiseMaterialInteraction
  → PointwiseMaterialInteraction::updateState
```

`evaluatePointwiseMaterialInteraction` is **not** in the installed
headers. Its compiled body at `0x2afce0` calls
`Acts::computeEnergyLossBethe` and stores that float as `Eloss`.
It does **not** call `computeEnergyLossMean` or
`computeEnergyLossMode`.

The other compiled identities, used only as offline controls:

```
Mean = Bethe + Radiative
Mode = 0.9 * Landau + 0.15 * Radiative
```

That is why WB126 production `ΔE` sits between Mean and Mode on a
thick muon slab: production is ionization-mean Bethe, not the
combined Mean and not the Landau-like Mode.

`updateState` conversion, copied by the shadow:

```
nextE = hypot(mass, p) - Eloss * navDir
nextP = max(10 MeV, sqrt(nextE^2 - mass^2) if nextE > mass else 0)
q/p   = copysign(|q| / nextP, q/p)
```

## Gating is not 1-1 with a material pointer

`evaluateMaterialSlab` overrides the stage:

```
start surface  → PostUpdate
target surface → PreUpdate
otherwise      → FullUpdate
```

then `ISurfaceMaterial::factor(direction, stage)` scales the slab.
Default `splitFactor = 1`. Forward + `PreUpdate` therefore has
factor `0`, so the **target** surface does not call `updateState`.
`MaterialInteractor` records a surface interaction only when the
slab is valid, which is exactly when `updateState` runs. Volume
material is recorded and is not applied to the ACTS mean.

WB126 emitted every `hasSurfaceMaterial` actor cycle and then
consumed the next recorded slab. Gated surfaces therefore showed
production `Δq/p = 0` while a forced Mean on the attached slab was
nonzero. That is a recorder/updater pairing bug, not a second Eloss
formula.

## Official result

Official run: `sbb14zb_surface_eloss_semantics_20260909T161125Z_be4e054a`

```
decision = surface_energy_loss_semantics_established
verdict  = PASS
shadow_mean_contract_established = true
production_eloss_quantity = computeEnergyLossBethe
```

```
config SHA      6b89a65070355d0143561d3017dbd451870ab74a7cbcf5fac77e9e1bd1ba27fa
decision SHA    aa80751fb532cf36350a8cf78b25f9d74da5bc707b9739ba7b683ef294fe04c4
helper SHA      b95e2416296feb241addd75a01284fb60e310932706781168d3f9bd11e83869d
mean header SHA 80c21c62d17ce27fb306465b84102ad93f7a266e7ba368554f613728ef16f516
dump SHA        7a454ee298d6e9612afb6868b19a454086f199f7171b69d504df59205aa86537
```

Dump: `outputs/leave_target_out_dump_v1/b14zb_smoke/mc24_100048_00000_00049/ckf_leave_target_out_surface_eloss.jsonl`

T1 first slab (the WB126 smoking gun):

```
production ΔE = 0.460932 GeV
Bethe = evaluatePointwise Eloss = 0.460934 GeV
Mean  = 0.624323 GeV
Mode  = 0.338027 GeV
```

Production matches Bethe, not Mean/Mode. Gated surfaces are
`FullUpdate` vacuum / zero-thickness bins: `updateState` does not
run and shadow does not apply Mean. The WB126 `Δq/p=0` vs forced
Mean split was a recorder/updater pairing bug.

Required-hop endpoints after copying Bethe + `updateState` gating
and fixing the shadow path double-count of the last field step
(the shared −6.113 mm):

```
T1/6:  Δloc0=-3.07e-7 mm, Δpath=0, Δqop_rel=1.60e-7, nodes 7/7
T2/11: Δloc0=+1.70e-8 mm, Δpath=0, Δqop_rel=3.19e-8, nodes 5/5
T3/11: Δloc0= 3.55e-14 mm, Δpath=0, Δqop_rel=1.14e-8, nodes 1/1
```

`shadow_mean_contract_established` is true because T1/6, T2/11 and
T3/11 all close per-node `q/p` **and** the pre-registered mean
endpoint gates. Even then:

```
focus_independent_reference_established = false
jacobian_contract_established = false
b14m_reopen_authorized = false
```

The next workbook may reuse this certified mean shadow for
common-grid independent FD. This run did not evaluate a derivative.
