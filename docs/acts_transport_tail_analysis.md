# ACTS Transport Tail / Uncertainty Analysis (Stage B / Task B4)

Workbook 100.  After WB99 diagnosed the closure failure as
`high_chi2_tail_dominated`, this task asks whether the mean χ² is driven
by a few wrong track states or by a missing physical uncertainty model.

It uses the frozen WB99 top 1% and 0.1% lists.  It does not rescreen,
drop, clip, inflate C, rescale Q, replace ACTS Q with Highland, or enter
Measurement Model V2 / alignment.

## What is analyzed

- **B4.1** Classify each frozen tail row as reconstruction (A),
  transport (B), or geometry/material (C).  Original residual, C, and Q
  stay untouched.
- **B4.2** Reco vs truth momentum, repeated identities across station
  pairs, construction/validation fractions.  Events stay in the sample.
- **B4.3** Pull distributions on the full official sample, plus a
  tail-removed **diagnostic only** view of the bulk.
- **B4.4** Eigenvalues of `Q = C1 − C0`.  Diagnostic only; Q is not
  retuned.

## Case classes

| Case | Meaning | Next step |
| --- | --- | --- |
| 1 | Tail is mostly a wrong track state | Reconstruction / association QC |
| 2 | Tail is enriched in material or slope | ACTS material diagnosis |
| 3 | Real uncertainty-model mismatch | Reassess MM V2 **input** model |

B4 does not enter V2 even if the case is 3.

Wrong momentum is locked at `|log10(p_reco/p_truth)| ≥ 1` (one decade),
the scale already reported in WB99.  `|q/p|` bins stay frozen.

Official run `sbb4_acts_transport_tail_analysis_20260906T182102Z_a19d9ada`
is **Case 1**: 18/24 frozen 1% rows are reconstruction anomalies
(mostly `|q/p pull| ≥ 5`, plus decade-wrong p and repeated identities).
The next step is reconstruction / association QC, not covariance retuning.
