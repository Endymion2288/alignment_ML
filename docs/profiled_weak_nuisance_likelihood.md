# Profiled Weak-Nuisance Measurement Likelihood (Stage B / Task B14M)

Workbook 114.  WB113 found no legal target-independent prior, so this
task stops asking for a seed-independent 5×5 LTO Cin.  It builds a
measurement likelihood from surviving LTO hits and **profiles** the
weak parameters.

Profiling is the contract:

```
chi2_prof(alpha) = min_nu chi2(alpha, nu)
```

Marginalization is **not** executed.  Without a legal prior or an
independently defined measure, `∫ L dν` is not a legal marginal
likelihood.

This is **not** Measurement Model V2.  It does **not** enter B15 or
V4 C/D.

## Likelihood

Do not use WB109 Cin or WB107 Cin.

```
r_i(theta) = m_i - h_i(theta)
chi2(theta) = sum_i r_i^T R_i^{-1} r_i
```

`theta` is the source-plane native chart
`(loc0, loc1, phi, theta, q/p)`.  `h_i` is field-aware ACTS transport
plus the production loc0 strip projection.  `R_i = (0.08 mm)² / 12`,
the same CKF pitch/√12 used by the LTO helper.

Target-station measurements stay out of the fit.

## Partition

WB112 classes stay frozen.  They are **not** a license to truncate or
seed-fix the state.

```
alpha = (loc0, theta)   ~ (x, ty)
nu    = (loc1, phi, q/p) ~ (y, tx, q/p)
```

Coupling is kept.  Rank deficiency of `H_nn` is a diagnosis, not a
reason to add ridge.

## Numerical contract

B14M pre-registers `pinv_relative = 1e-8`.  That is not alignment
`rank_tolerance=0.01`.  Joint NLS, explicit nuisance minimization, and
the Schur complement must agree on a controlled linear problem.

## Cases

- A: `profiled_measurement_likelihood_validated` — exclusion, optimizer,
  seed-invariant observables, explicit null space, uncertainty
  semantics, and frozen construction/validation calibration.  Then
  decide whether Gaussian-state V4 is still the right abstraction.
- B: nuisance unresolved, but the target prediction observable is
  identified → next is an observable-space model, not a 5D Cin.
- C: the prediction itself depends on the seed or has no finite
  uncertainty → stop this LTO estimator.
- D: numerics unstable → fix the implementation, no ridge/prior cover.
- E: mixed

## Official tokens

The official B14M run landed on Case D:

```
decision = profile_likelihood_numerically_unstable
profiling_executed = true
marginalization_executed = false
prior_introduced = false
lto_cin_contract_established = false
b15_authorized = false
do_not_force_5d_lto_covariance = true
```

The synthetic joint / profile / Schur contract passed.  The ACTS
measurement-only helper is not yet stable: `100043/37` and
`100048/86` fail during field-aware `h_i(theta)`, and a `loc1+1mm`
restart does not return to the same χ².  That is under-convergence,
not a license to pick the best seed or to resume 5D Cin repair.
