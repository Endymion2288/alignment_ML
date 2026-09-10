# 3ST q/p Failure-Mechanism Diagnosis (Yasu-S2D)

Workbook 120.  WB119 froze the conclusion that official
`CKFTrackCollectionWithoutIFT` \(q/p\) is not a trusted reconstructed
momentum observable.  This stage asks **why** — it does not reopen S2
as PASS and does not enter Stage 3 residual conditionals, alignment, or
a weak-mode fit.

Truth remains a calibration reference only.  The WB119 source-disjoint
denominator, matching, binning, and failure sample are reused unchanged.

## Official run

Smoke: `yasu_s2d_three_st_qp_failure_diagnosis_smoke_20260908T182759Z_d4782b4a`

```
decision = three_st_qp_failure_diagnosis_contract_established
diagnosis_verdict = INCONCLUSIVE
three_st_qp_trusted_observable = false
residual_conditional_authorized = false
```

Batch: `yasu_s2d_three_st_qp_failure_diagnosis_batch_20260908T190106Z_b5bf3e74`
(same 3 398 774 primary tracks and 243 758 sign-flips as WB119)

```
decision = three_st_qp_failure_diagnosis_recorded
mechanism_verdict = mixed/inconclusive
supported = curvature_information_loss, covariance_semantics_failure
excluded  = reference_definition_error, charge_asymmetric_reconstruction
three_st_qp_trusted_observable = false
residual_conditional_authorized = false
new_reconstruction_dump_authorized = false
```

## Mechanism verdict

High-\(p\) sign flips follow a curvature-information pattern: flip rate
18.2% at \(p\ge 2\,\mathrm{TeV}\) versus 4.2% at \(p<200\,\mathrm{GeV}\);
40.1% when \(\lvert q/p_{\rm truth}\rvert/\sigma<0.5\) versus 1.66% when
that significance is \(\ge 10\).  Mean high-\(p\) significance is 1.67.

That does **not** explain pull RMS 955 or 68%/95% coverage 0.415/0.659.
The native 5×5 has \(\langle\log\kappa\rangle=30.4\) and 2038
non-positive eigenvalues.  Mean-wrong and \(\sigma\)-wrong both apply.

Reference identities pass (S1 recompute, \(q\cdot p\), `front()` not at
IFT).  Swapping to production \(q/p\) or a GeV unit does not rescue
coverage.  Charge-odd \(\Delta\) is larger than charge-even, but
\(\mu^+/\mu^-\) flip rates differ by only 0.86 percentage points, so
charge-asymmetric reconstruction is excluded and is **not** a magnetic
weak mode.

Short / incomplete tracks concentrate flips (`n_mot<6` or two stations
missing: ~45%).  This is a reported conditional, not a cut.  Seed
\(q/p\) and leave-one-station increments are absent from the WB119 dump;
a new reconstruction dump is **not** authorized because ① and ② are
already decided from existing fields.

Stage 3 stays closed.
