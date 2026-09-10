# 3ST q/p Population-Mean / Reconstruction-Bias Decomposition (Yasu-S2F)

Workbook 122.  WB119 froze official WithoutIFT \(q/p\) as **not** a
trusted reconstructed momentum.  WB121 split the high-\(p\) information
limit from covariance miscalibration.  This stage does not repair either
failure and does not open Stage 3.

It asks only how a non-zero **population** mean of fitted \(q/p\) should
be read in MC: physical charge/momentum composition, reconstruction
bias, or a handful of extreme tails.

## Official run

Smoke: `yasu_s2f_three_st_qp_mean_decomposition_smoke_20260908T222355Z_63018db8`

```
decision = three_st_qp_mean_decomposition_contract_established
diagnosis_verdict = INCONCLUSIVE
three_st_qp_trusted_observable = false
residual_conditional_authorized = false
official_mean_trimmed = false
```

Batch: `yasu_s2f_three_st_qp_mean_decomposition_batch_20260908T224209Z_06d45778`
(same 3 398 774 primary tracks and 243 758 sign-flips)

```
decision = three_st_qp_mean_decomposition_recorded
construction = population_mean_dominates
validation   = reconstruction_bias_dominates
pooled       = tail_dominated_mean   (not a stable split conclusion)
official_mechanism = mixed/inconclusive
train_validation_direction_consistent = false
three_st_qp_trusted_observable = false
residual_conditional_authorized = false
official_mean_trimmed = false
```

## Frozen definitions

On the untrimmed primary sample:

\[
\mu_{\rm fit}=E[(q/p)_{\rm fit}],\quad
\mu_{\rm truth}=E[(q/p)_{\rm truth}],\quad
\mu_{\rm bias}=E[\Delta(q/p)].
\]

The identity \(\mu_{\rm fit}=\mu_{\rm truth}+\mu_{\rm bias}\) is checked
to \(10^{-18}\).  The official mean is never replaced by a median,
trimmed mean, or reweighted mean.

Charge-balanced (\(0.5\mu_++0.5\mu_-\)) and kinematics-matched
(both charges reweighted to the pooled \((p,t_x,t_y)\) mix) numbers are
**diagnostic counterfactuals only**.  They are not a new calibration
population.

Top 0.1% and 0.01% tails in \(\lvert\Delta\rvert\) and
\(\lvert q/p_{\rm fit}\rvert\) are reported as contributions to the
official mean.  Those tracks stay in the sample.

## Mechanisms (frozen before batch)

Priority: tail → population → reconstruction, else `mixed/inconclusive`.

- `tail_dominated_mean`: top 0.1% \(\lvert\Delta\rvert\) holds
  \(\ge 50\%\) of \(\sum(q/p)_{\rm fit}\)
- `population_mean_dominates`: \(\mu_{\rm truth}\) has the same sign as
  \(\mu_{\rm fit}\) and at least 60% of its magnitude; \(\lvert\mu_{\rm bias}\rvert\)
  is below 50%
- `reconstruction_bias_dominates`: \(\mu_{\rm truth}\) is below 20% of
  \(\lvert\mu_{\rm fit}\rvert\) and \(\mu_{\rm bias}\) has the same sign
  and at least 60% of the magnitude

Construction and validation must name the same mechanism and agree in
direction, or the official mechanism is `mixed/inconclusive`.

## Batch conclusion

Construction is 60% \(\mu^-\).  Truth already has a same-sign mean
(\(\mu_{\rm truth}=-1.95\times 10^{-6}/\mathrm{MeV}\),
\(\mu_{\rm fit}=-1.83\times 10^{-6}\)).  Charge-balancing sends
\(\mu_{\rm truth}\) to \(\sim 0\).

Validation is 50/50.  \(\mu_{\rm truth}\approx 0\) and 94% of
\(\mu_{\rm fit}=-6.87\times 10^{-6}\) is \(\Delta\).  Fitted \(\mu^+\)
even has the opposite sign from truth.

The pooled official mean \(-3.28\times 10^{-6}\) is six times the
median.  The top 0.1% in \(\lvert\Delta\rvert\) (3399 tracks) owns
52.6% of \(\sum(q/p)_{\rm fit}\).  Those tracks were not removed.

Because the two source-disjoint splits name different mechanisms, the
official verdict is `mixed/inconclusive`.  A non-zero \(\mu_{\rm fit}\)
still does **not** authorize Stage 3.

## What this stage must not claim

A non-zero \(\mu_{\rm fit}\) does **not** authorize
\(E[r_{\rm IFT}\mid q/p]\), IFT alignment, or an \(R_y/d_x\) weak mode.
It only says whether Yasu’s “distribution not centered at zero” is a
population effect, a reconstruction bias, or a tail artifact in this MC.
