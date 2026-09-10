# 3ST measurement-level bending batch calibration / CKF mapping (Yasu-S2J)

Workbook 126.  Batch-calibrates the frozen WB125 YZ `bending_raw`
observable and splits the WB119 fitted-\(q/p\) sign-flip sample into
measurement-level loss versus reconstruction mapping.  It does not
flip WB119, does not open Stage 3, and does not change the bending
definition, fitter, seed, hits, geometry, field, or collection.

## Official run

`yasu_s2j_three_st_qp_measurement_bending_batch_batch_20260910T082512Z_f61fa2a8`

```
decision = three_st_qp_measurement_bending_batch_recorded
raw_bending_information_present     = supported
raw_bending_high_p_limit            = rejected
fit_additional_sign_failure         = rejected
raw_measurement_source_dependence   = rejected
mapping_nontransferability          = supported
official_mechanism = mixed/inconclusive
three_st_qp_trusted_observable = false
residual_conditional_authorized = false
official_qp_like_jacobian_authorized = false
s2_flipped_to_pass = false
```

HTCondor cluster `1113523` (9/9 normal completion).  Truth-reference
count \(3\,398\,774\) and fitted sign-flips **243 758** match WB119
exactly.

## Answers

**Of the 243 758 fitted sign-flips**

| class | \(n\) | share of 243 758 |
| --- | ---: | ---: |
| A raw bending already correct | 113 144 | 46.4% |
| C raw bending already flipped | 27 206 | 11.2% |
| \(\lvert b\rvert<10^{-5}\) | 2 135 | 0.88% |
| no constructable 3ST bending | 101 273 | 41.5% |

Among the 142 485 classifiable flips, A is 79.4% and C is 19.1%.
`mapping_nontransferability` is therefore **supported**.

Official conditionals on clean 18-hit (\(n=2\,007\,715\)):

- \(\Pr(\mathrm{fit\ flip}\mid\mathrm{raw\ bending\ correct})=0.0404\)
- \(\Pr(\mathrm{raw\ bending\ flip})=0.0329\)

**Source dependence** does not survive matching.  Unmatched
charge-odd rel-range is 0.397.  After matching charge, \(p\),
\(t_x\), \(t_y\), and topology (165 cells, matched fraction 0.999)
the rel-range is 0.038.  WB125’s unmatched gate was a
population-spectrum effect.  Official:
`raw_measurement_source_dependence=rejected`.

**High \(p\)** does not meet the frozen degradation gate.  Clean
\(\ge2\,\mathrm{TeV}\) (\(n=176\,492\)) raw sign agreement is 0.923
versus 0.983 below 200 GeV (drop 0.060 < 0.15).  1–2 TeV is 0.951.
Do **not** reuse WB125’s \(n=26\) / 100 GeV rejection.
`raw_bending_high_p_limit=rejected`.  Fitted \(q/p\) at
\(\ge2\,\mathrm{TeV}\) is 0.853; the raw-minus-fit gap 0.070 is
below the frozen 0.15 extra-failure gate, so
`fit_additional_sign_failure=rejected` even though raw stays
clearly better than the fit in that bin.  Dirty
\(\ge2\,\mathrm{TeV}\) raw agreement is only 0.724
(\(n=81\,208\)) and is not used for the official gate.

## Frozen observable and sample

YZ `bending_raw` is unchanged from WB125.  Centroids are MOT-matched
space points, else `localToGlobal`.  Matching slopes are the S1–S3
measurement chord.  Truth is a source-disjoint calibration
reference only.  The same nine WB119 xAODs are used.  Clean is
`n_mot=18` with stations \(\{1,2,3\}\) and no IFT leak; dirty is
kept.

## Clean 18-hit calibration

| quantity | value |
| --- | ---: |
| sign(`bending_raw`) vs truth charge | 0.967 |
| sign(fitted \(q/p\)) vs truth | 0.955 |
| Spearman(`bending_raw`, \(q/p_{\rm truth}\)) | 0.954 |
| Spearman(`bending_x`) | 0.0005 |
| charge-odd mean | \(+8.74\times10^{-4}\) |

Complete three-station (\(n=3\,075\,701\)) sign agreement is 0.942.
Dirty constructable tracks (\(n=1\,067\,986\)) are 0.896 and are
not deleted.

## Field contract (read-only)

`FaserFieldTable_v2.root` is at
`/cvmfs/faser.cern.ch/repo/sw/software/22.0/faser/offline/ReleaseData/v20/MagneticFieldMaps/FaserFieldTable_v2.root`.
One zone, millimetre bounds, `bscale=1e-7`.  On \(x=y=0\) the
stored \(B_x\) is about \(-0.56\,\mathrm{T}\) if the cache comment
(kT) is used.  A straight-line \(\int B_x\,dz\) from S1 to S3 is
unofficially \(\sim 1.13\,\mathrm{T\cdot m}\).

Path integral, units, and Jacobian are **not** locked.  Do not
convert `bending_raw` to momentum.

## Provenance

| item | SHA / id |
| --- | --- |
| config | `7871a77ec5b7698ce2bb1518868d4f97f2dc3ff03db485461d36236575e8de55` |
| batch contract | `9358ee1442426a0db97d069ab525bee218f39e55ec4315aaa935d60fefb12c5b` |
| smoke contract | `c882510e2cf8f879457281cc614103ced99bcf44658db5df10efed089ff2caee` |
| git HEAD | `55cf982302a3c62c57b74f368d5e3ba7723fd33a` |
| field contract | `d6879d5c020558a3d4ad72fcc1c5beaa0790d3c67fa570261a85408e79a38fac` |
| Calypso | `40892527e9c65409afd2378a2abfc25ddbddac03` |
| Athena / ACTS | 24.0.41 / 32.0.2 |
| HTCondor | cluster `1113523` |

WB117–125 hashes verified.  WB119 stays
`three_st_qp_calibration_not_established`.

Dump SHA-256 values are listed in
`workbook/2026-09-09_126_3ST_qp_measurement_bending_batch.md`.

## What this stage must not claim

3ST \(q/p\) is not trusted.  Stage 3 is closed.
`bending_raw` is not an authorized residual-conditioning
observable.  The seed 0.55 T and the unofficial line integral are
not an official Jacobian.  \(\sigma_b\) is not invented.  Raw
bending is not shown to die at \(\ge2\,\mathrm{TeV}\) under the
frozen gate.  The 101 273 flips without constructable centroids
are not attributed by this observable.

## Next step

Stage 3 stays closed.  Any later
\(E[r_{\mathrm{IFT}}\mid\mathrm{bending}]\) must be separately
preregistered.  The 41.5% of WB119 flips with no 3ST centroid
would need a new, separately frozen study.  A \(q/p\)-like
conversion stays unauthorized until the field Jacobian is locked.
