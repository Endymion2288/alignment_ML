# Target-Independent Weak-Parameter Prior Contract (Stage B / Task B14P)

Workbook 113.  WB112 showed that the 5D LTO state is not
measurement-supported: `x` and `ty` are, while `y`, `tx/φ`, and
`q/p` are weak or prior-dominated.

This task asks whether those weak directions have a prior that is
target-independent, usable on real data, and semantically defined.

It does **not** assume the answer is yes.  It does **not** introduce a
prior into the fitter or into WB109 Cin.  It does **not** enter B14M,
B15, V4 C/D, or Measurement Model V2.

## Questions

1. Why do `y`, `tx/φ`, and `q/p` need a prior?
2. Which candidate sources are actually target-independent?
3. Is a candidate event-level or only a population spectrum?
4. If a prior existed, what would `L_meas × π_independent` be?
5. If no prior is admissible, stop forcing a 5D Gaussian seed.

## Admissibility

A candidate is admissible only if it is all of:

- independent of the excluded target measurements
- applicable to real data, not MC truth / particle-gun energy
- disjoint in measurement identity from the surviving LTO likelihood
- independent of the geometry being estimated
- equipped with defined uncertainty semantics

Forbidden: full-track CKF q/p, WB107 Cin, truth residuals, χ²
closure, `0.1/1/10` seed scales, arbitrary Gaussians.

`source-only` IFT can constrain local `x,y,tx,ty` but **cannot**
measure `q/p` without the magnet.  Those hits are already inside
`L_surviving_measurements`, so they are not an independent prior.

## Cases

- A: admissible prior for all weak directions → write the likelihood
  contract, then redo B14.  `prior_contract_established` is not
  `lto_cin_contract_established`.
- B: no admissible prior, especially not for `q/p` → FAIL, next is
  B14M (profile / marginalize the weak nuisances).  Do not keep
  repairing a 5D Cin.
- C: admissible prior for a subset only
- D: a candidate looks usable but independence or bias is not
  established
- E: mixed

## Official tokens

The official B14P run landed on Case B:

```
decision = target_independent_prior_not_available
prior_contract_established = false
prior_introduced = false
lto_cin_contract_established = false
b15_authorized = false
b14m_authorized = true
```

No admissible real-data prior exists for `y`, `tx/φ`, or `q/p`.
This workbook does not enter B14M.
