# 3ST q/p Extreme-Tail and Split/Source Instability Audit (Yasu-S2G)

Workbook 123.  WB119 froze official WithoutIFT \(q/p\) as **not** a
trusted reconstructed momentum.  WB122 showed that construction is a
population-mean effect while validation is a reconstruction bias, and
that the pooled official mean is tail-dominated.  This stage does not
repair those findings and does not open Stage 3.

It asks only two remaining questions:

1. Why does validation show a strong reconstruction bias
   (\(\mu_{\rm bias}=-6.48\times 10^{-6}/\mathrm{MeV}\))?
2. Why can a handful of tracks dominate the official \(q/p\) mean and
   the pull RMS?

## Official run

Smoke: `yasu_s2g_three_st_qp_tail_source_smoke_20260909T075818Z_45694506`

```
decision = three_st_qp_tail_source_contract_established
diagnosis_verdict = INCONCLUSIVE
three_st_qp_trusted_observable = false
residual_conditional_authorized = false
official_mean_trimmed = false
new_reconstruction_dump_authorized = false
```

Batch: `yasu_s2g_three_st_qp_tail_source_batch_20260909T081928Z_89657e0f`
(same 3 398 774 primary tracks and 243 758 sign-flips)

```
decision = three_st_qp_tail_source_recorded
tail_from_sparse_catastrophic_fits     = rejected
broad_reconstruction_shift             = unresolved
source_specific_reconstruction_response = supported
population/composition_explains_split  = rejected
refit_provenance_suspected             = supported
official_mechanism = mixed/inconclusive
repeatable_reconstruction_bias = false
three_st_qp_trusted_observable = false
residual_conditional_authorized = false
official_mean_trimmed = false
new_reconstruction_dump_authorized = true   # contract only; this stage did not dump
```

## Frozen definitions

On the untrimmed primary sample:

\[
\mu_{\rm fit}=E[(q/p)_{\rm fit}],\quad
\mu_{\rm truth}=E[(q/p)_{\rm truth}],\quad
\mu_{\rm bias}=E[\Delta(q/p)].
\]

Class contributions are reported for \(\sum\Delta\), \(\sum(q/p)_{\rm fit}\),
\(\sum z^2\), sign-flip, and coverage failure (\(\lvert z\rvert\ge 1\)).
Tracks are never deleted from the official mean.

Tail definitions frozen before inspection:

- \(\lvert z\rvert\ge 10\) and \(\lvert z\rvert\ge 100\)
- top \(0.1\%\) and \(0.01\%\) in \(\lvert\Delta(q/p)\rvert\)

“Mean after removing class \(X\)” is an **influence diagnostic only**.
Same-charge, same \((p,t_x,t_y)\), same-topology matched comparisons are
**not** a new calibration population.

## Mechanisms (frozen before batch)

Evaluated primarily on validation for A/B, on pooled sources for C,
on the construction/validation match for D, and on validation front-\(z\)
for E.  Official mechanism is the unique supported label, else
`mixed/inconclusive`.

- `tail_from_sparse_catastrophic_fits`: after removing \(\lvert z\rvert\ge 10\)
  **and** after removing the top \(0.1\%\) \(\lvert\Delta\rvert\), the
  remaining \(\lvert\mu_{\rm bias}\rvert\) falls below 20% of the original
  (or below \(10^{-7}/\mathrm{MeV}\))
- `broad_reconstruction_shift`: both removals leave a same-sign residual
  of at least 60%
- `source_specific_reconstruction_response`: same-charge sources have
  relative \(\mu_{\rm bias}\) range \(\ge 0.50\) or opposite signs
- `population/composition_explains_split`: matched-cell gap
  \(\le 30\%\) of the raw split gap
- `refit_provenance_suspected`: a non-`front_near_s1` front-\(z\) proxy
  holds \(\ge 50\%\) of validation \(\sum\Delta\), \(\sum z^2\), or the
  top \(0.1\%\) tail

A subsequent \(\le 20\)-event diagnostic dump is authorized only if that
proxy could own the validation bias or tail **and** existing fields
cannot tag `CKF-only fallback` versus `KalmanFitter refit success`.
Allowed fields: `kf_refit_succeeded`, pre/post-refit \(q/p\) and
covariance, front surface type/\(z\), track-state provenance.  The
fitter, seed, hits, geometry, and covariance scale stay frozen.  Truth
does not enter the fit.

## Batch conclusion

Validation \(\mu_{\rm bias}=-6.48\times 10^{-6}/\mathrm{MeV}\) is **not**
a persistent shift of the clean three-station majority.  The clean 58%
has \(\mu_{\rm bias}=-3.6\times 10^{-8}\approx 0\) and 0.3% of
\(\sum\Delta\).  The dirty remainder owns 99.7% of \(\sum\Delta\) and
essentially all of \(\sum z^2\).

The concentration is topological and provenance-like:

- missing \(\ge 2\) stations (2.5% of tracks) holds **103.9%** of
  \(\sum\Delta\) and 99.7% of \(\sum z^2\)
- `front()` at S2 (3.9%) holds **75.4%** of \(\sum\Delta\), with
  \(\mu_{\rm bias}=-1.25\times 10^{-4}\)
- `n_mot<12` holds essentially all of the signed sum

The top 0.1% in \(\lvert\Delta\rvert\) (981 tracks) does **not** own the
signed validation mean: its share of \(\sum\Delta\) is \(-38.6\%\).
Removing it makes \(\mu_{\rm bias}\) more negative.  Removing
\(\lvert z\rvert\ge 10\) (3.8%) leaves 24% of the original mean — below
the 60% broad line and above the 20% catastrophic line — so A is
rejected and B is unresolved.

The same S2-front class in construction has the **opposite** sign
(\(+8.4\times 10^{-5}\)).  Same-charge sources disagree, including sign
flips among \(\mu^-\) files.  Matched cells (same charge, \(p\), \(t_x\),
\(t_y\), topology) retain 74.5% of the raw split gap, so composition
does not explain the split.  Train and validation \(\mu_{\rm bias}\)
have opposite signs: the reconstruction bias is **not repeatable**.

Pull RMS is a different sentence from the official mean.  On validation,
\(\lvert z\rvert\ge 10\) owns 99.9997% of \(\sum z^2\); \(\lvert z\rvert\ge 100\)
(3226 tracks) still owns 99.998%.  The pooled top 0.1% in \(\lvert\Delta\rvert\)
still owns 52.6% of \(\sum(q/p)_{\rm fit}\), as in WB122.  Those tracks
were not removed.

Because C and E are both supported, the official mechanism is
`mixed/inconclusive`.  The unusual-front proxy holds 75.3% of validation
\(\sum\Delta\) and existing dumps have no `kf_refit_succeeded` tag, so
the minimum diagnostic-export contract is **authorized**.  This stage
did not submit that dump.

## Stage-3 status

`three_st_qp_trusted_observable = false`.
`residual_conditional_authorized = false`.
WB119 remains `not_established`.

## What this stage must not claim

A non-zero validation \(\mu_{\rm bias}\) does **not** authorize
\(E[r_{\mathrm{IFT}}\mid q/p]\), IFT alignment, or an \(R_y/d_x\) weak
mode.  Influence remaining means and matched/reweighted numbers are not
a new official mean or calibration population.  The S2-front proxy is
not yet a proof that those tracks are CKF-only fallbacks.  WB124
executed that dump: S2-front is mixed Hole-99 fallback and legal
first-MOT, not a one-to-one fallback tag.
