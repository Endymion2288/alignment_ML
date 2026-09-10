# 3ST q/p Covariance / Curvature-Information Root-Cause Audit (Yasu-S2E)

Workbook 121.  WB119 froze official `CKFTrackCollectionWithoutIFT`
\(q/p\) as **not** a trusted reconstructed momentum observable.  WB120
supported two independent problems and stopped at
`mixed/inconclusive`.  This stage splits them:

1. How unidentifiable is \(q/p\) itself (curvature information)?
2. Why the reported \(\sigma(q/p)\) / native 5×5 cannot represent that
   unidentifiability?

It does **not** reopen S2 as PASS and does not enter Stage 3 residual
conditionals, alignment, \(R_y/d_x\) weak-mode, ACTS boundary scan,
B14M/B15, or Measurement Model V2.  No new reconstruction dump is
submitted.  Truth remains a calibration reference only.

## Official run

Smoke: `yasu_s2e_three_st_qp_root_cause_smoke_20260908T203419Z_42264e70`

```
decision = three_st_qp_root_cause_contract_established
diagnosis_verdict = INCONCLUSIVE
A/B/D = unresolved
C = rejected
three_st_qp_trusted_observable = false
residual_conditional_authorized = false
new_reconstruction_dump_authorized = false
```

Batch: `yasu_s2e_three_st_qp_root_cause_batch_20260908T210534Z_f1fa87c2`
(same 3 398 774 primary tracks and 243 758 sign-flips as WB119)

```
decision = three_st_qp_root_cause_recorded
A = supported   intrinsic_curvature_information_limit
B = unresolved  topology_amplified_information_loss
C = rejected    covariance_basis_unit_provenance_bug
D = supported   covariance_model_calibration_failure
information_limit_remains_if_covariance_fixed = true
three_st_qp_trusted_observable = false
residual_conditional_authorized = false
new_reconstruction_dump_authorized = false
```

## Structured A/B/C/D tests (frozen before batch)

Clean subsample (conditional only; not a new physics sample): complete
S1+S2+S3, `n_mot==18`, `ndof>0`, no outliers, \(\chi^2/\mathrm{ndof}<5\),
SPD, minimum eigenvalue \(>0\).

| ID | Mechanism | Support | Reject |
| --- | --- | --- | --- |
| A | intrinsic curvature-information limit | In the clean subsample: high−low \(p\) flip gap \(\ge 0.05\), flip vs significance monotonically decreasing, low−high significance flip gap \(\ge 0.10\) | Applicable and high−low \(p\) gap \(< 0\) |
| B | topology-amplified information loss | dirty−clean flip gap \(\ge 0.10\) (\(n\ge 50\) each) | Applicable and gap \(< 0.05\) |
| C | covariance basis/unit/provenance bug | Rescaling pulls by \(10^{\pm 3}\) or \(\sqrt{10^3}\) yields RMS \(\in[0.60,1.40]\) **and** 68% coverage \(\in[0.50,0.80]\) (Gaussian-like, not fat-tail overcoverage) | Otherwise reject a *global* unit/Jacobian cause |
| D | covariance model/calibration failure without conversion bug | C did not rescue; clean still miscalibrated; \(\lvert z\rvert\ge 10\) holds \(\ge 50\%\) of clean \(\sum z^2\) | Clean is calibrated |

Non-SPD tracks and extreme \(\lvert z\rvert\) tails stay in the sample.
Their contribution to pull RMS, coverage failure, and sign-flips is
reported.  They are not deleted to manufacture a “clean PASS”.

Measurement-level curvature significance is
\(\lvert q/p_{\rm fit}\rvert/\sigma(q/p)\).
\(\lvert q/p_{\rm truth}\rvert/\sigma\) is an MC diagnostic only.
A large condition number alone does **not** prove the covariance is
wrong.

## Conversion chain (source-locked)

Pinned Calypso SHA `40892527e9c65409afd2378a2abfc25ddbddac03`.

`CreateTrkTrackTool::ConvertActsTrackParameterToATLAS` takes the Acts
bound 6×6, drops time, multiplies the \(q/p\) row/column by
`1_MeV=1e-3` (so \(C_{44}\times 10^{-6}\), a consistent Jacobian), and
stuffs the 5×5 into `Trk::CurvilinearParameters` **without** a
bound→curvilinear Jacobian.  The source comment says “to GeV” but the
arithmetic converts Acts \(1/\mathrm{GeV}\) to Trk \(1/\mathrm{MeV}\).
\(q/p\) is basis-invariant, so \(C_{44}\) should not change under that
relabeling.

`CKF2` may prepend fitted parameters at the CKF target plane as a Hole
`front()`, then `KalmanFitterTool::fit` refits.  A successful refit
rewrites the track without those fitted params (`front()` = first MOT
smoothed state).  A failed refit keeps the CKF track.  The dump has no
`kf_refit_succeeded` flag.  The refit inflates the input covariance
\(\times 10\) and seeds loc from the surface centre, not the track loc.

A missing global \(10^3/10^6\) would scale **all** pulls.  WB119 already
has median \(\bar z\approx -0.07\) and 41.5% within \(1\sigma\), which
argues against that.  Bound-as-curvilinear does not explain pull RMS
955.

## Diagnostic-export contract

`authorized = false`.  Existing fields plus the pinned conversion chain
already decide A / B / C-unit / D.  Acts-native 6×6 would only label
CKF-versus-refit and would not change those decisions.

A later dump is allowed only if a written contract shows that existing
fields are insufficient **and** the new fields distinguish two concrete
mechanisms.  It must not change the seed, fitter, hit selection,
covariance scale, or geometry, and must not put truth into the fit.

## Batch conclusion

In the clean 18-hit / complete-3ST / SPD subsample (\(n=1\,973\,802\)):
high-\(p\) flip is 14.2% versus 0.59% at low \(p\) (gap +13.6 pt) and
still falls monotonically with curvature significance (34.1% at
\(\lvert q/p_{\rm truth}\rvert/\sigma<0.5\) versus 0.065% at
\(\ge 10\)).  That is an **intrinsic information limit**, not a
topology defect.  52.5% of all sign-flips already sit in the
measurement-level region \(\lvert q/p_{\rm fit}\rvert/\sigma<1\).

A global \(10^3/10^6\) unit/Jacobian bug is **rejected**: scaling
pulls by \(10^3\) yields RMS 0.955 but 68% coverage 0.999
(overcoverage, not a Gaussian rescue).  The conversion chain’s
\(C_{44}\times 10^{-6}\) Jacobian is internally consistent, and
\(q/p\) is basis-invariant.

The reported 5×5 is still miscalibrated **without** that conversion
bug: clean RMS 21.9, coverage 0.440 / 0.700, and \(\lvert z\rvert\ge 10\)
owns 98.9% of clean \(\sum z^2\).  The 2038 non-positive-eigenvalue
tracks contribute only 0.18% of \(\sum z^2\) and are not deleted.
The 7947 tracks with \(\lvert z\rvert\ge 100\) own 99.995% of the
pooled sum of squares.

Topology amplifies flips (dirty 11.4% vs clean 4.1%, gap +7.2 pt) but
misses the +10 pt support line, so B stays unresolved.  It does not
explain the clean high-\(p\) flips.

Even if the covariance is later corrected, the high-\(p\) physical
information limit remains.

## Stage-3 status

```
three_st_qp_trusted_observable = false
residual_conditional_authorized = false
```

Stage 3 may be rediscussed only after a **future independent** stage
shows that both the reconstructed \(q/p\) sign/mean and its uncertainty
are usable inside a preregistered window.
