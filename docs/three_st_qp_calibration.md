# 3ST CKF q/p Calibration (Yasu Stage 2)

Workbook 119.  After Stage 0 locked `CKFTrackCollectionWithoutIFT` and
Stage 1 materialized an independent 3ST→IFT prediction, this stage asks
only whether the three-station CKF signed \(q/p\) is unbiased, sign
reliable, and uncertainty-calibrated.

It does **not** enter IFT residual weak-mode, alignment, ACTS boundary
scan, B14M, B15, or Measurement Model V2.  Truth is a **calibration
reference only**.  It must not enter the fit, seed, prior, prediction,
or geometry solve.

## Official observable and reference

```
q/p_fit     = native Trk::CurvilinearParameters q/p of WithoutIFT front()
              (S1, unit 1/MeV, signed)
σ(q/p)      = sqrt(C_44) of the persisted native 5×5
q/p_truth   = truthParticle.charge() / |p_S1|
              p_S1 = FiducialParticleTool::getTruthMomenta(barcode)[1]
```

Matching is `TrackTruthMatchingTool` (majority barcode from
`SCT_SDO_Map` deposits on measurements-on-track).  The formula is the
NtupleDumper tracklet path, applied at station 1 because that is the
WithoutIFT `front()` station.  Production \(p\) and stations 0/2/3 are
exported as diagnostics and are **not** the official reference.

Slopes are \(t_x = p_x/p_z\), \(t_y = p_y/p_z\).

## Primary quantities (frozen)

- \(\Delta(q/p) = q/p_{\rm fit} - q/p_{\rm truth}\)
- pull \(z_{q/p} = \Delta(q/p)/\sigma(q/p)\)
- sign agreement
- 68% / 95% coverage (\(|z|<1\), \(|z|<1.96\))
- pull mean / width
- charge-even bias \(0.5(\langle\Delta\rangle_{\mu^+} + \langle\Delta\rangle_{\mu^-})\)
- charge-odd bias \(0.5(\langle\Delta\rangle_{\mu^+} - \langle\Delta\rangle_{\mu^-})\)

Do **not** look only at \(E[q/p]\).  The official conditionals are
\(E[\Delta\mid q/p_{\rm truth}]\), \(E[\Delta\mid t_x]\),
\(E[\Delta\mid t_y]\).  Charge-even / charge-odd exist to separate a
later magnetic-curvature bias from a pure geometry effect.  This stage
does not interpret that split as a weak mode.

## Frozen sample rules

Keep sign-flip tracks, high-momentum tails, and large pulls.  Unmatched
tracks and NaN S1 truth momenta are excluded from \(\Delta\)/pull and
kept in the selection denominator.  Match fraction is diagnostic, not a
cut.  Do not pick a favourable momentum range, rescale covariance,
replace fit \(q/p\) with truth, or add a prior.

High-momentum / \(q/p\to 0\) is the explicit
`qp_near_zero_p_ge_5tev` and `p_ge_2000gev` pair.

## Failure classes (kept, not deleted)

- `mean_bias`
- `charge_sign_failure`
- `kinematic_dependent_bias`
- `covariance_miscalibration`
- `candidate_selection_loss`
- `fit_failure`

A FAIL is a negative result.  It is not reversed by redefining the
sample.

## Allowed decisions

- `three_st_qp_calibration_contract_established` — smoke/contract only;
  authorizes HTCondor batch, **not** Stage 3
- `three_st_qp_calibration_established` — physics PASS; 3ST \(q/p\) may
  be used as a reconstructed momentum observable; authorizes Stage 3
  preregistration of \(E[r_{\rm IFT}\mid q/p]\) and \(E[r_{\rm IFT}\mid t_y]\)
- `three_st_qp_calibration_not_established` — contract or physics FAIL

Smoke with \(n < 200\) matched tracks is physics-INCONCLUSIVE by the
frozen gate.  Only the batch campaign can declare the observable
trusted.

## Official run

Smoke: `yasu_s2_three_st_qp_calibration_smoke_20260908T150024Z_63aec9de`

```
decision = three_st_qp_calibration_contract_established
contract_verdict = PASS
physics_verdict = INCONCLUSIVE
batch_htcondor_authorized = true
residual_conditional_authorized = false
three_st_qp_trusted_observable = false
```

20-event construction / validation smoke.  20/20 and 19/19 WithoutIFT
tracks have a finite S1 truth reference.  IFT leak = 0.  One validation
event has no WithoutIFT candidate (selection loss, already seen in
Stages 0–1).  Two validation sign flips and the high-momentum tail are
kept.  \(n=39<200\), so physics is INCONCLUSIVE by the frozen gate.
Smoke cannot authorize Stage 3.

Batch: `yasu_s2_three_st_qp_calibration_batch_20260908T175453Z_c6f5561a`
(HTCondor cluster `1112283`, 9/9 return 0; 3 500 000 events,
3 398 774 truth-matched primary tracks)

```
decision = three_st_qp_calibration_not_established
contract_verdict = PASS
physics_verdict = FAIL
mechanism = charge_sign_failure
failure_classes = charge_sign_failure, mean_bias,
                  kinematic_dependent_bias, covariance_miscalibration
residual_conditional_authorized = false
three_st_qp_trusted_observable = false
```

**Ruling:** three-station CKF \(q/p\) is **not** a trusted reconstructed
momentum observable for later Yasu weak-mode work.  The worst regions
are \(p\ge 2\,\mathrm{TeV}\) (sign agreement 0.818) and
\(|t_x|\ge 0.010\), \(|t_y|\ge 0.010\).  Pooled sign agreement is
0.928; pull RMS is 955; 68%/95% coverage is 0.415/0.659;
\(\bar z=-0.5002\).  Selection and fit-failure gates pass.  Sign-flip
and high-\(p\) tracks are kept.  Stage 3 stays closed.

See workbook 119 for the full denominator, charge-even/odd split, and
\(E[\Delta\mid q/p_{\rm truth}]\), \(E[\Delta\mid t_x]\),
\(E[\Delta\mid t_y]\).
