# Physically Supported LTO Track-State Contract (Stage B / Task B14S)

Workbook 112.  WB111 showed that the exported 5×5 is a Kalman fitted
state at the source plane, but the covariance remains seed-sensitive
and a few catastrophic fits dominate the y-pull RMS.

This task asks a more basic question: after the target station is
removed, which track-state degrees of freedom do the remaining
measurements actually support?

It does **not** hunt for a low-dimensional subspace that passes a
covariance gate.  It does **not** introduce a prior.  It does **not**
enter B14P, B15, V4 C/D, or Measurement Model V2.

## Questions

1. For each excluded target, which surviving stations constrain
   `x, y, tx, ty, q/p`?
2. Which directions still move when only one seed-variance axis is
   scaled (`0.1 / 1 / 10`)?
3. Which directions gain information relative to the uninformative
   seed on the same source-plane bound chart?
4. Which directions must remain explicit weak nuisances rather than a
   fake seed-independent 5D posterior?
5. Are independent physical priors even available, without using them?
6. Do `100043/37` and `100048/86` coincide with prior-dominated
   directions?

## Classes

Per direction, from pre-registered sensitivity / information-gain
thresholds only — not from truth, residual, χ², or Transport V3/V4:

```
measurement_dominated
weakly_measured
prior_dominated
unconstrained
fit_failed
```

`weakly_measured` means the direction gains information versus the
uninformative seed, but the fitted width still follows the seed.
That is a weak nuisance, not a license to delete the parameter.

## Cases

- A: all five directions are measurement-dominated → redo B14
- B: some directions are supported, some are weak / prior-dominated
  → next is B14P, not B15; do not delete `q/p`
- C: no stable information gain on the key kinematic directions →
  stop this LTO estimator as a propagation seed
- D: the body contract exists, but the catastrophic-fit mechanism
  is still unexplained
- E: mixed

Directional seed scales are a falsification smoke.  The production
scale stays 1.  `tx`/`ty` smoke axes scale bound `φ`/`θ` on the
source plane; they are proxies, not exact derived slopes.

## Official tokens

The official B14S run landed on Case B:

```
decision = reduced_measurement_supported_state_with_weak_nuisance
lto_cin_contract_established = false
b15_authorized = false
b14p_authorized = true
prior_introduced = false
```

Typical classes: `x` and `ty` measurement-dominated; `y` and `q/p`
weakly measured; `tx`/`φ` prior-dominated.  `q/p` stays an explicit
nuisance.  This workbook does not introduce a prior.
