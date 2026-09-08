# CKF Reconstruction Contract Implementation Validation (Stage B / Task B7)

Workbook 103.  After WB102 showed that transport validation consumed the
raw `CKFTrackCollection`, this task applies an explicit
`eligible_for_transport_validation` predicate and replays the frozen
WB98 closure on the same dumps, C0 / C1 / Q, and gates.

The predicate is independent of closure.  It does not use χ², q/p pull,
residuals, or truth.  It does not retune C/Q or enter Measurement
Model V2 / alignment.

## Eligibility (B7.1)

A dump row is eligible when all of the following hold:

- ntuple long-track equivalent (`longTracks > 0`)
- station completeness: four distinct reconstruction tracklet stations
  (not `Track_InStation0`, which is 0 for almost every long track)
- segment completeness: `TrackSegments >= 4`
- finite 5×5 input covariance with positive diagonal
- finite signed q/p, not the dummy sentinel
- valid state surface (finite z, stations in 0–3, finite 5-vector)

These requirements come from the WB102 contract document.  They were
not tuned on WB98 χ².

## Samples (B7.2)

| Sample | Dump rows | Unique CKF | Official pairs | χ² mean | χ² median | q/p pull RMS |
| --- | --- | --- | --- | --- | --- | --- |
| A `raw_CKF` | 2680 | 895 | 2352 | 33.33 | 0.150 | 4.82 |
| B `contract_eligible` | 1989 | 663 | 1974 | 25.76 | 0.115 | 0.84 |
| C `reference_longTrack` | 2337 | 779 | 2094 | 24.64 | 0.116 | 1.15 |

Exclusive ineligible reasons (dump rows): 343 not long-track, 348
incomplete stations.  Sample B is a subset of C: 116 long tracks lack
four tracklet stations.  Covariance eigenvalues stay on the same Cin
scale.  Residuals are diagnostic only.

## Closure replay (B7.3)

Same WB98 dump, same C0 / C1 / Q, same frozen gates.  Only the input
scope changes.

Official-pair event χ² mean drops 22.7% (33.33 → 25.76).  This is
**not** “filter then PASS”:

- `closure_under_contracted_input_scope = false`
- `covariance_model_fixed = false`
- construction MODEL1 still fails every pair; event `100043/37` keeps
  pair (0,3) mean χ² at 111
- validation χ² means fall to 1.1–3.3 and the whitened-χ² gate passes
  there, but pencil ratios move from ~5–11 to ~11–21 (more overcover)
- every contracted pair still fails pencil / generalized-eigenvalue
  gates

The contracted sample is a physically consistent reconstruction input.
It does not repair the covariance model.

## Remaining tails (B7.4)

Frozen WB99 1% list is not rescreened or deleted.  Nine frozen rows
remain after the contract, seven identities.  The only remaining
`|q/p pull| ≥ 5` identity is `100043/37`:

- long track, four stations, four segments, 21 measurements
- CKF χ²/ndof = 1.42
- reco 414 GeV vs truth 2.0 TeV, q/p pull ≈ −13.8
- transport χ² 759 / 2971 / 37120 on pairs (0,2) / (0,1) / (0,3)

That leftover is CKF fitting / high-momentum q/p, not an input-scope
defect.

## Case classes

| Case | Meaning | This run |
| --- | --- | --- |
| A | Contract improves closure; input scope was the mismatch | **Primary** (22.7% mean drop) |
| B | A few q/p tails remain on the contracted sample | **Secondary** (`100043/37`) |
| C | Contracted sample still fails as a whole | No (median χ² 0.11) |

Official run `sbb7_ckf_contract_validation_20260906T193419Z_6d906fda`.
Next step: Transport covariance V3, using this contract as the input
scope.  Do not rescale C or Q to absorb event 37.
