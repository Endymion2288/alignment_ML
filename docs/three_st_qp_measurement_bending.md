# 3ST measurement-level bending / fit-independent curvature proxy (Yasu-S2I)

Workbook 125.  Constructs a signed, calibratable three-station
bending observable from S1/S2/S3 measurement geometry only.  It does
not repair `CKFTrackCollectionWithoutIFT` \(q/p\), does not reopen
covariance or refit work, does not chase `front()` provenance, does
not flip WB119, and does not open Stage 3.

## Official run

Contract: `yasu_s2i_three_st_qp_measurement_bending_contract_20260909T170948Z_9164bf14`

```
decision = three_st_qp_measurement_bending_contract_established
raw_bending_information_present              = supported
raw_bending_information_limited_at_high_p    = rejected
fit_additional_sign_failure                  = supported
raw_measurement_bias/source_dependence       = supported
official_mechanism = mixed/inconclusive
three_st_qp_trusted_observable = false
residual_conditional_authorized = false
official_qp_like_jacobian_authorized = false
measurement_uncertainty_propagated = false
s2_flipped_to_pass = false
htcondor_submitted = false
```

20 events on each of the two frozen xAODs.  39 WithoutIFT tracks;
26 clean 18-hit; 37 complete three-station; 13 dirty (kept).
37/39 tracks have constructable centroids.

## Physical definition (frozen before dump inspection)

Bending plane is **YZ**, not XZ.  Locked from pinned Calypso
`40892527e9c65409afd2378a2abfc25ddbddac03`:

- `CircleFit.cxx:6-8` feeds space-point \((z,y)\) to the Taubin circle
- `CircleFitTrackSeedTool.cxx:310-314` fits \((z,x)\) as the
  non-bending line
- `CircleFitTrackSeedTool.cxx:335,345`: `charge = (cy < 0) ? +1 : -1`
- Lorentz: \(v\sim\hat z\), dipole \(B\sim\hat x\) \(\Rightarrow\)
  \(v\times B\) in \(y\).  \(\mu^+\) \(\Rightarrow\) \(+y\) sagitta

Primary official observable (radian):

\[
\texttt{bending\_raw}
=\arctan\frac{y_2-y_1}{z_2-z_1}
-\arctan\frac{y_3-y_2}{z_3-z_2}
\]

Companion sagitta
\(s_y=y_2-[y_1+(y_3-y_1)(z_2-z_1)/(z_3-z_1)]\).
\(+\texttt{bending\_raw}\Leftrightarrow +s_y\Leftrightarrow\)
CircleFit charge \(+1\).  Orthogonal control `bending_x` uses \(x\).

Inputs are MOT-matched `SCT_SpacePoint` global positions, averaged
per station.  Fallback is `SCT_DetectorManager` `localToGlobal` of
the MOT cluster.  Persisted `front()` \(xyz\), fitted \(q/p\), the
5×5, truth \(q/p\), and the seed circle are forbidden inputs.

The seed constants \(0.55\,\mathrm{T}\) and \(0.3\)
(`CircleFitTrackSeedTool.cxx:334-345`) are **not** an official
\(\int B_\perp dl\).  No \(q/p\)-like conversion is authorized.
\(\sigma_b\) is not invented: cluster `local_cov00` is a 1-D strip
variance without a validated stereo Jacobian.  This stage is
point-estimator calibration only.

Station \(z\) (mm, inherited, not re-derived): IFT \(-1860.15\),
S1 \(47.4\), S2 \(1237.4\), S3 \(2427.4\).

## Geometry / field provenance

Tags: FASERNU-04, OFLCOND-FASER-06, OFLP200,
`GLOBAL-BField-Maps-03/FaserFieldTable_v2.root`,
`GLOBAL-BField-Scale-03`, reco `s0013-r0022`.
The field-map line integral was **not** machine-confirmed here.

A first `dumps/smoke/` write found every ROT `globalPosition()`
unset after POOL read (`FaserSCT_ClusterOnTrack` has no detector
element until `setValues`).  Those files are kept as forensics and
are **not** the official contract.  Official dumps are
`dumps/contract/`, which match MOT clusters to space points
(CircleFit’s measurement).

## Frozen denominator and SHAs

Construction `mc24_100043_00400_00499` (\(\mu^-\)),
validation `mc24_100048_00000_00049` (\(\mu^+\)).
`SkipEvents=0`, 20 events, local only.

| item | SHA-256 / SHA |
| --- | --- |
| config | `b03976103cf8df63ebc10a6db6bc709a077e6b61bc0b8d7df2078f199fe2b4a4` |
| contract | `d11537b731846b4fc4b7e5cd71d95b62258a32bea24c79a9286166c6edac3f97` |
| git HEAD | `55cf982302a3c62c57b74f368d5e3ba7723fd33a` |
| dump 100043 | `94fa6f95cefd07a650b06ed19db5b6fe7f3e674e467c7dbd10fdb108ca72b1de` |
| dump 100048 | `aa2f1429f5ea73b48114c3eaca7fbedb71c7f3a11b7802d85a99a54e827819b1` |

WB117–124 inheritance hashes match the S2H config.  CircleFit
sources are pinned and verified.  Truth is a source-disjoint MC
calibration reference only.

## Calibration and source stability (clean 18-hit, \(n=26\))

| quantity | value |
| --- | ---: |
| sign(`bending_raw`) vs truth charge | 0.962 (25/26) |
| sign(fitted \(q/p\)) vs truth | 0.923 (24/26) |
| Spearman(`bending_raw`, \(q/p_{\rm truth}\)) | 0.908 |
| Spearman(`bending_x`, \(q/p_{\rm truth}\)) | 0.010 |
| charge-odd mean | \(+1.03\times10^{-3}\) |
| high-\(p\) (\(p\ge 100\,\mathrm{GeV}\), \(n=24\)) sign agree | 0.958 |
| low-\(p\) (\(n=2\)) sign agree | 1.0 |
| construction charge-odd mean | \(+1.68\times10^{-3}\) (\(n=14\)) |
| validation charge-odd mean | \(+2.64\times10^{-4}\) (\(n=12\)) |
| source rel-range | 0.842 (same sign) |

Dirty tracks that remain constructable (\(n=11\)) have sign
agreement 1.0 and are not deleted.  Complete three-station
(\(n=37\)) sign agreement is 0.973.

The source-rel-range gate fires.  Both charge-odd means are
positive; the factor-of-six gap can still be the two particle-gun
spectra.  It is **not** a demonstrated charge-even measurement bias.

High-\(p\) degradation is **rejected** on this contract: the sample
median is \(879\,\mathrm{GeV}\) and the 100 GeV split drops by only
0.042.  That does **not** consolidate WB121’s intrinsic
curvature-information limit.

## Same-track comparison with fitted \(q/p\)

| class | \(n\) (clean) | identities |
| --- | ---: | --- |
| A truth & bending same, fit flipped | 2 | 100048 / skip 6 and 9 (WB119 sign-flips) |
| B \(\lvert\mathrm{bending}\rvert<10^{-5}\) | 0 | — |
| C bending flipped vs truth | 1 | 100043 / skip 10 (fit agrees with truth) |
| D \(\lvert\mathrm{pull}\rvert\ge 10\) | 0 | — |

Both class-A rows sit in the lowest \(\lvert\mathrm{bending\_raw}\rvert\)
quartile (2/7).  The other three quartiles have zero fitted
sign-flips.  WB119’s 243 758 sign-flips are **not** quantified here.

On this window the two frozen WB119 sign-flips already have
measurement-level charge information.  Reconstruction mapping is a
justified *later* preregistration; it is not opened now.

## What this stage must not claim

3ST \(q/p\) is not calibrated.  WB119 is not flipped.
\(E[r_{\mathrm{IFT}}\mid q/p]\) and
\(E[r_{\mathrm{IFT}}\mid\mathrm{bending}]\) are not authorized.
`bending_raw` is not an authorized residual-conditioning
observable.  0.55 T is not \(\int B_\perp dl\).  \(\sigma_b\) was
not invented.  The 243 758 fraction is unknown *at this stage*
(quantified in WB126).  Raw bending is not
shown to die at high \(p\).  Source-mean tension is not proven
measurement bias.  The first empty-centroid smoke dump is not a
bending input.

## Next step

Stage 3 stays closed.  Any future
\(E[r_{\mathrm{IFT}}\mid\mathrm{bending}]\) must be separately
preregistered.  A field-map
\(\int B_x\,dl\) remains required before any official \(q/p\)-like
conversion.

Yasu-S2J / WB126 is **recorded** (`mixed/inconclusive`): cluster
`1113523` finished; the 243 758 WB119 flips are split (46.4%
fit-additional, 11.2% already flip at measurement, 41.5% no
constructable 3ST bending).  High-\(p\) raw degradation and
matched source dependence are rejected.  Trusted / S3 stay false.
