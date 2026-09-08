# Profile Likelihood Numerical Stabilization (Stage B / Task B14N)

Workbook 115.  WB114 established the measurement-level profile
likelihood mathematically and on a synthetic linear problem.  The
same likelihood was **numerically unstable** on real ACTS geometry
and field.  This task does not change that likelihood.  It asks
whether the identical objective can be evaluated and profiled in a
stable, reproducible, initialization-independent way.

```
theta = (loc0, loc1, phi, theta, q/p)
alpha = (loc0, theta)
nu    = (loc1, phi, q/p)
chi2_prof(alpha) = min_nu chi2(alpha, nu)
```

`h_i` remains field-aware ACTS transport plus the production loc0
strip projection.  `R_i = (0.08 mm)² / 12`.  Target-station hits
stay out.  No prior.  No ridge as information.  No invented χ²
penalty for a failed hop.  No 5D Cin.  No B15.  No Measurement
Model V2.

## What is allowed

- Sequential, z-ordered measurement hops instead of independent
  source→surface jumps
- Kalman-matched adaptive `maxStepSize` (no 10 m cap)
- Per-hit propagation audit
- Fixed numerical unit scales `z_i = (θ_i − θ_ref_i) / s_i`
  with `s = (1 mm, 1 mm, 1 mrad, 1 mrad, 1e-3 /GeV)`
- Gauss–Newton plus backtracking line search
- Pre-registered restart invariance on the same four WB114 inits

Physical `χ²(θ(z))` is unchanged by scaling.  The scales are not a
prior covariance and are not tuned from truth, Hessian, or a seed
campaign.

## What is not a physical diagnosis

Propagation failure is not nonidentifiability.  Hessian rank 3/5
is allowed: a profile likelihood may have a null direction in
`ν`.  The objects that must be well-defined are the profile
objective, the target observable, and the observable uncertainty.

## Cases

- A: `profile_numerical_contract_established` — then re-run B14M
  physics / uncertainty on the stable solver
- B: `profile_transport_contract_broken` — fix the standalone
  evaluator; do not assume “both use ACTS” means the same contract
- C: `profile_derivative_contract_broken` — fix `∂r/∂θ` before
  interpreting optimizer failure
- D: `profile_optimizer_contract_broken` — evaluator and Jacobian
  are trusted, globalization is not
- E: `physical_nonidentifiability_exposed` — only after A-class
  numerical contracts
- F: mixed or no smoke yet

## Smoke gate

Login-node smoke is events `0,1,37` on `100043` and `86` on
`100048`.  The 1989-row campaign is **not** submitted unless every
gate check passes: restart invariance including `loc1+1 mm`,
Jacobian, nominal propagation, 37/86 no longer step-limit-only,
no hidden ridge/prior, and zero leaked target measurements.

## Official tokens

Official run `sbb14n_profile_numerics_20260907T150627Z_d77ddddc`:

```
decision = profile_transport_contract_broken
smoke_gate_passed = false
full_sample_authorized = false
synthetic_profile_passed = true
target_exclusion_holds = true
b15_authorized = false
do_not_force_5d_lto_covariance = true
```

Event 0/1 now have a stable profile solver and `loc1+1 mm`
restart invariance.  `100043/37` still cannot evaluate the
magnet-crossing hop.  `100048/86` crosses the magnet but misses
one stereo-partner surface.  Frozen permissions stay false.
WB109 and WB114 dumps are not overwritten.
