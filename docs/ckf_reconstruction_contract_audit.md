# CKF Reconstruction Input Contract Audit (Stage B / Task B6)

Workbook 102.  After WB101 showed that the frozen χ² tail is dominated
by CKF states that fail the long-track reconstruction contract, this
task asks which CKF states are currently allowed into transport
covariance validation and alignment likelihood.

It audits the current pipeline.  It does not design cuts, implement a
filter, drop tails, retune C/Q, or enter Measurement Model V2 /
alignment.

## What is audited

- **B6.1** `CKFTrackCollection` versus ntuple `longTracks` on all WB87
  sources: unique CKF tracks, accepted long tracks, descriptive
  rejection reasons.
- **B6.2** Contract definition: `raw_CKF_collection` versus
  `eligible_for_transport_validation`.  The latter is documented, not
  implemented.
- **B6.3** Frozen WB101 tails, not rescreened.  Diagnostic contrast of
  the official-pair sample against its long-track-equivalent subset.
- **B6.4** The frozen 15/24 not-a-long-track tails: station pattern,
  segments, missing stations, length, q/p uncertainty.

Flags and floors are descriptive.  They are not selection cuts.

## Case classes

| Case | Meaning | Next step |
| --- | --- | --- |
| A | Raw CKF contains many incomplete states; input contract missing | Transport covariance V3 after a defined input scope |
| B | Long-track sample still has wrong q/p or frozen tails | CKF fitting quality audit |
| C | Quality contract complete but tails remain | ACTS material / physics diagnosis |

Official run `sbb6_ckf_reconstruction_contract_20260906T191538Z_afbc27fc`
is **Case A**, with secondary **Case B**.  Closure is not claimed.
The long-track subset is a diagnostic contrast only.

## B6.1 Input scope

All nine WB87 sources.  Unique CKF identity is
`(source_id, run_id, event_id, track_index)`.

| Quantity | Count |
| --- | --- |
| Unique CKF tracks | 895 |
| Accepted ntuple long tracks | 779 |
| CKF tracks with `longTracks = 0` | 116 (13.0%) |
| Ntuple long-track events without a CKF dump | 0 |

Exclusive descriptive reasons for the 116 rejected CKF tracks:

| Reason | N |
| --- | --- |
| `missing_station` | 59 |
| `other` | 46 |
| `no_truth_match` | 9 |
| `fit_failure` | 2 |
| `short_segment` | 0 (all also miss a station; counted above) |
| `low_measurement_count` | 0 exclusive |

Independent flags: `short_segment = 59` (same set as missing station),
`low_measurement_count = 17`, `no_truth_match = 17`.  `other` means
the event has four tracklet stations and enough hits but still has no
ntuple long track.  The production long-track filter is stricter than
those floors.  B6 does not invent the missing cut.

## B6.2 Contract

Current production / transport validation input is
`raw_CKF_collection` = all `CKFTrackCollection` states that carry a
5×5 covariance and a finite signed q/p.  No station, measurement, or
fit-status requirement is enforced.

`eligible_for_transport_validation` is **undefined in production**.
The documented reconstruction contract, not applied as a filter, is:

- four stations / four segments
- state and 5×5 covariance available
- finite signed q/p
- fit status available
- long-track equivalent

The ntuple `Track_*` block is already the long-track filtered subset.
Alignment and transport validation currently use the raw collection
instead.

## B6.3 Tail reproduction (diagnostic only)

Frozen official-pair top 1% = 24 rows.  Not rescreened.  Not deleted.

| Sample | N | χ² mean | χ² median | Frozen 1% remaining | q/p pull RMS |
| --- | --- | --- | --- | --- | --- |
| Raw official pairs | 2352 | 33.33 | 0.150 | 24 (1.02%) | 4.82 |
| Long-track equivalent | 2094 | 24.64 | 0.116 | 9 (0.43%) | 1.15 |

Mean χ² falls because 15/24 frozen tails leave with the incomplete
CKF states.  It does **not** become a closure PASS: the maximum is
still \(3.7 \times 10^4\), and event `100043/37` remains.

Remaining frozen long-track identities: `100043/50`, `100043/13`,
`100043/37` (three pairs), `100044/3`, `100044/2`, `100047/80`,
`100047/68`.  The three remaining `|q/p pull| ≥ 5` rows are all
`100043/37`.

## B6.4 Short-track / missing-station tails

The frozen 15/24 not-a-long-track rows:

| Exclusive reason | N |
| --- | --- |
| `missing_station` | 5 |
| `no_truth_match` | 4 |
| `other` | 6 |

Typical missing-station pattern: stations 2 and 3 absent, two
segments, ~10 hits, large q/p pull.  The six `other` rows already
have four stations and ≥19 hits; they are still absent from
`longTracks`.  q/p uncertainties stay at the dumped Cin scale
(\(10^{-7}\)–\(10^{-6}\) / MeV) and do not explain the pulls.

## Verdict

Case A: the reconstruction input contract is missing.  Thirteen
percent of unique CKF tracks, and 15/24 frozen tails, are states the
long-track selection never accepted.

Case B is secondary, not resolved by scope: after the long-track
contrast, frozen tails and the `100043/37` q/p failure remain.  That
is CKF fitting / leftover high-momentum physics, not a licence to
rescale C or Q.

Do not write “delete the incomplete events and closure PASSes.”
