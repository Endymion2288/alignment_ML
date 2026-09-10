# 3ST q/p CKF / KalmanFitter-refit provenance (Yasu-S2H)

Workbook 124.  Executes the WB123-authorized \(\le 20\)-identity
diagnostic dump.  It does not flip WB119, does not open Stage 3, and
does not change the fitter, seed, hits, geometry, field, covariance
scale, outlier policy, or collection definition.

The only scientific question is whether WB123’s `front_near_s2` class
is KalmanFitter-refit failure / CKF-only fallback, and whether the
abnormal \(q/p\) and covariance are created in the original CKF, the
KF refit, or the persisted `front()` selection.

## Official run

Batch: `yasu_s2h_three_st_qp_refit_provenance_batch_20260909T150728Z_aebef4ed`

```
decision = three_st_qp_refit_provenance_recorded
ckf_only_fallback_drives_s2_front          = unresolved
kf_refit_introduces_qp_shift               = unresolved
persisted_front_state_selection_artifact   = rejected
covariance_changed_in_refit                = unresolved
official_mechanism = mixed/inconclusive
three_st_qp_trusted_observable = false
residual_conditional_authorized = false
s2_flipped_to_pass = false
focus_replaced_after_results = false
large_dump_submitted = false
```

20/20 frozen identities were written.  Offline WB119 join recovered
20/20 rows; truth did not rewrite any reconstructed state.

## Source flow (pinned Calypso `40892527e9c65409afd2378a2abfc25ddbddac03`)

File hashes match `configs/three_st_qp_refit_provenance_v1.yaml`
`pinned_calypso_sources` (same pins as WB121).

- `CKF2.cxx:268-297`: after `createTrack(..., fittedParams)` the job
  always calls `KalmanFitterTool::fit`.  Success persists `trk2`;
  failure persists `trk` (CKF-only).
- `CKF2Config.py:137`: `addFittedParamsToTrack=True`.
- `CreateTrkTrackTool.cxx:25-78`: reversed Acts states are inserted at
  `begin` when not backward.  `:82-93`: a Hole TSOS with
  `FitQualityOnSurface(-99, 0)` is prepended when `fittedParams` is
  present.  That Hole is the CKF-only `front()`.
- `KalmanFitterTool.cxx:348-351`: fail if MOT `< MinMeasurements`
  (default 12).  `:360`: origin = `front()->z()-10`.  `:372-374`:
  inflate input covariance \(\times 10\).  `:423`: success calls
  `createTrack` **without** `fittedParams`, so `front()` is the first
  MOT smoothed state.  `:424-426`: failure returns `nullptr`.
- `CircleFitTrackSeedTool.cxx:176-186`: forward target
  \(z=\min Z-10\).  Complete three-station seeds sit just upstream of
  S1; missing-S1 seeds sit just upstream of S2.
- `faser_reco.py:317-321`: `CKF_woIFT`,
  `maskedLayers=[0,1,2,3,4,5]`,
  `OutputCollection=CKFTrackCollectionWithoutIFT`,
  `BackwardPropagation=False`.

CKF2 keeps only one of `trk` / `trk2`.  The complementary original
state is not in the xAOD.  This stage inspects the persisted
collection only.  A second live KF was not compiled in: the private
`KalmanFitterTool` header is not usable from the standalone plugin,
and a re-fit would start at `front()->z()-10`, not at the original
CKF target plane.

## Frozen focus list

Frozen before dump inspection; not replaced afterwards.
`skip_index` is Athena order in the single source xAOD
(`event_id` repeats across chained copies).  In the first 180 events,
`skip_index == event_id`.

Six WB119/123 identities, the first five validation and first three
construction dirty `front_near_s2` tracks (missing S1 or `n_mot<12`)
in skip order, and 3+3 complete `n_mot=18` S1-front controls.

## Mechanisms (frozen before dump)

- `ckf_only_fallback_drives_s2_front`: Hole-99 fraction among
  S2-front \(\ge 0.80\) (\(n_{\rm S2}\ge 4\)).  Rejected if that
  fraction \(\le 0.20\), or if every S2-front is first-MOT and none
  is Hole-99.
- `kf_refit_introduces_qp_shift`: needs a true original pre/post
  \(q/p\) pair.
- `persisted_front_state_selection_artifact`: persisted `front()`
  matches neither known terminal state.
- `covariance_changed_in_refit`: needs a true original pre/post
  covariance pair.

Official mechanism is the unique supported label, else
`mixed/inconclusive`.

## Batch conclusion

`front_near_s2` is **not** a one-to-one, and not a \(\ge 80\%\),
proxy for CKF-only fallback.

When S1 is missing the seed `minZ` is at S2, so **both** outcomes can
place `front()` at S2:

| S2-front subset | \(n\) | persisted front | original KF | `n_mot` |
| --- | ---: | --- | --- | --- |
| Hole \(\chi^2=-99\), ndof \(=0\) | 5 | CKF target plane (pre-refit) | failed | all \(<12\) |
| first MOT | 3 | first MOT smoothed (post-refit) | succeeded | all \(=12\) |

Fallback fraction \(=5/8=0.625\) → unresolved.  The three
`n_mot=12` rows are legal successful-refit first-MOT states; the
fallback-only hypothesis is therefore **not** accepted.  It is also
**not** rejected, because five S2-front rows really are Hole-99
CKF-only tracks.  `MinMeasurements=12` matches this split exactly.

All 20 persisted fronts are either Hole-99 or first-MOT, so a
serialization / state-ordering artifact is rejected.  Original
pre→post \(q/p\) and covariance shifts remain unresolved: the xAOD
never stored both states.

S1-front complete controls (\(n=6\)) are all successful first-MOT
with \(\tilde\sigma\approx 2.3\times 10^{-7}/\mathrm{MeV}\).  The
eight S2-front dirty tracks have \(\tilde\sigma\approx 2.8\times
10^{-6}/\mathrm{MeV}\) (about \(12\times\) larger).  That is a
comparison of persisted terminal states, not a pre→post pair.

Extra facts that a \(z\)-only proxy cannot see:

- WB119 catastrophic identity 100048 / skip 15 is Hole-99 at
  \(z=-0.5\) (S1 window), `n_mot=5`, \(q/p=3.95\times 10^{-4}\),
  pull \(=125\).  CKF-only fallback, **not** S2-front.
- WB119 sign-flips (100048 / 6, 9) and the other complete-station
  large-pulls are successful-refit first-MOT at S1.

## Stage-3 status

`three_st_qp_trusted_observable = false`.
`residual_conditional_authorized = false`.
WB119 remains `not_established`.

## What this stage must not claim

Explaining S2-front provenance does **not** calibrate 3ST \(q/p\),
does not flip WB119, and does not authorize
\(E[r_{\mathrm{IFT}}\mid q/p]\), alignment, an \(R_y/d_x\) weak
mode, B14M/B15, Measurement Model V2, a covariance-repair study, or
a change to `MinMeasurements` / the fitter.  `front_near_s2` is not
synonymous with CKF-only fallback.  A second diagnostic KF must not
be treated as the original pre/post pair.

The follow-on stage is Yasu-S2I / WB125: a fit-independent
measurement-level `bending_raw`.  Stage 3 stays closed.
