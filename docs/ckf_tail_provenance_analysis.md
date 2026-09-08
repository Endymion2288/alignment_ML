# CKF Tail Provenance / Association Audit (Stage B / Task B5)

Workbook 101.  After WB100 showed the closure mean is dominated by
wrong track states, this task asks where those states come from.

It uses the frozen WB99/WB100 top 1% and 0.1% lists.  It does not
rescreen, drop events, design quality cuts, retune C/Q, or enter
Measurement Model V2 / alignment.

Truth q/p is diagnostic only and never enters reconstruction.

## What is audited

- **B5.1** Track identity: source, run/event, CKF `track_index`,
  collection, station hits, n measurements, seed station, dump vs
  ntuple long-track presence.
- **B5.2** Fitted q/p vs truth q/p (diagnostic): residual, pull,
  momentum ratio, charge, fit quality.
- **B5.3** Association: multiple CKF candidates, conflicting tracklet
  truth IDs, unmatched tracklets, same state on multiple station pairs.
- **B5.4** Quality variables if present: χ², ndof, n measurements,
  n layers, station pattern.  Smoothing / outliers / material
  interactions are recorded as null when the ntuple does not store them.

Flags are descriptive.  They are not selection cuts.

## Case classes

| Case | Meaning | Next step |
| --- | --- | --- |
| A | Wrong association / candidate / identity | Association QC |
| B | CKF not a long track, missing stations, or fit failure | Tracking reconstruction audit |
| C | Remaining physics / material / propagation | ACTS material diagnosis |

Official run `sbb5_ckf_tail_provenance_20260906T183700Z_f95fb941` is
**Case B**: 14/24 frozen 1% rows are CKF reconstruction failures,
mostly `CKFTrackCollection` tracks that are not ntuple long tracks.
Association issues are secondary (7/24).  No C/Q retune.
