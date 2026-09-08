# Transport Covariance Validation V3 (Stage B / Task B8)

Workbook 104.  After WB103 froze `eligible_for_transport_validation`,
this task asks whether the current ACTS propagated covariance describes
target-surface prediction error on that contracted input.

The official input is the WB103 contracted CKF state plus the WB96
physical 5×5 and the WB98 C0 / C1 / Q dump.  Gates stay frozen.
The task does not retune C or Q, drop `100043/37`, reopen raw CKF, or
enter Measurement Model V2 / alignment.

## Official input (B8.1)

Eligibility is copied from WB103 and is independent of residual, χ²,
pull, truth, event 37, and momentum discrepancy:

- ntuple long-track equivalent
- four distinct reconstruction tracklet stations
- `TrackSegments >= 4`
- finite 5×5 Cin with positive diagonal
- finite signed q/p, not the dummy sentinel
- valid state surface

| Quantity | Count |
| --- | --- |
| Raw dump rows seen | 2680 |
| Ineligible excluded | 691 |
| Contracted rows | 1989 |
| Official pairs | 1974 |
| Construction / validation rows | 1116 / 873 |

Raw CKF is not the official input.  Event `100043/37` remains.

## Official MODEL1 metrics (B8.2)

Same WB98 dump, same C0 / C1 / Q, same WB81/WB87 gates.

Event-level official pairs: mean χ²/ndof = 25.76, median = 0.115,
q/p pull RMS = 0.84.  Pooled mean χ² is not the decision.

| Split | Pair | N | Mean χ² | Median χ² | Pencil | λ(C_emp, C_prop) | 95% cov | χ² gate | Shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| construction | (0,1) | 370 | 13.13 | 0.359 | 21.50 | 0.006–0.070 | 0.732 | fail | fail |
| construction | (0,2) | 370 | 8.24 | 0.261 | 20.99 | 0.018–0.155 | 0.781 | fail | fail |
| construction | (0,3) | 370 | 111.39 | 0.295 | 20.17 | 0.030–0.144 | 0.768 | fail | fail |
| validation | (0,1) | 288 | 3.34 | 0.044 | 12.67 | 0.007–0.111 | 0.920 | pass | fail |
| validation | (0,2) | 288 | 1.59 | 0.046 | 12.06 | 0.009–0.094 | 0.931 | pass | fail |
| validation | (0,3) | 288 | 1.10 | 0.048 | 10.85 | 0.027–0.181 | 0.931 | pass | fail |

Typical-event layer: median χ² ≪ 1; validation whitened-χ² passes.
Tail layer: `100043/37` owns 80.3% of the pooled χ² sum.
Shape layer: every official pair fails pencil and generalized
eigenvalue.  C_prop is systematically wider than C_emp.

Source-level official-pair means stay low on validation
(0.99–2.96) and on construction sources other than
`mc24_100043_00400_00499` (3.3–11.8).  That one construction source
has mean 181.6 because it contains event 37; its median is still 0.33.

## Frozen diagnostic: 100043/37 (B8.3)

The event is a legal contracted long track: four stations, four
segments, 21 measurements, CKF χ²/ndof = 1.42, reco ≈ 414 GeV,
truth diagnostic ≈ 2.0 TeV, `|q/p pull| ≈ 13.8`.  Transport χ² is
2971 / 760 / 37120 on (0,1) / (0,2) / (0,3).  It is retained.

Leave-one-focus-out on construction is diagnostic only, not a cut:

| Pair | Official mean χ² | LOO mean χ² | LOO pencil |
| --- | --- | --- | --- |
| (0,1) | 13.13 | 5.11 | 21.84 |
| (0,2) | 8.24 | 6.20 | 21.39 |
| (0,3) | 111.39 | 11.10 | 20.53 |

Removing 37 lowers the construction mean, especially (0,3).  Pencil
and generalized eigenvalues still fail.  V3 is therefore a
typical-event overcoverage mixture that is also polluted by a real
CKF fitting tail.  It is not “bulk calibrated, only 37 blocks the
gate.”

## Covariance shape (B8.4)

Pencil directions are almost pure `x`.  Generalized modes of
λ(C_emp, C_prop) are overwide in every direction; the smallest
λ values sit on `tx` / `ty`.  Validation 95% coverage is 0.92–0.93
while mean χ² is already near 1–3, which is the same overcoverage
pattern: C_prop is too large, not too small.  Scalar rescale is
forbidden.

## Material / Q diagnostic (B8.5)

`Q = C1 − C0` is the increment between two possibly re-linearized
propagations.  63.8% of contracted rows have a non-PSD Q.  That does
not fail V3, and Q is not PSD-projected.

Median Frobenius: C0 = 27.4, C1 = 18.3, Q = 5.7.  Material-on is
not a simple additive inflation of material-off.  Q grows toward
low momentum (median 16.5 vs 2.1 in the high-momentum bin) and is
trajectory-dependent.  This is diagnostic only.

## Case classes (B8.6)

| Case | Meaning | This run |
| --- | --- | --- |
| A | Construction and validation both pass frozen gates | No |
| B | Typical calibration holds; legal CKF tails block the aggregate | **Secondary** (event 37, 80% of χ²) |
| C | Generalized eigenvalue / pencil fail even on typical events | **Primary** |
| D | Sample too small to tell | No (N = 1974) |

Official run `sbb8_transport_covariance_v3_20260906T195338Z_e6c68984`.

```
verdict = NOT_ESTABLISHED
decision = transport_covariance_shape_not_validated
next_step = transport_material_uncertainty_diagnosis
measurement_model_v2_authorized = false
measurement_model_v2_entered = false
```

Do not tighten the reconstruction contract to force a PASS.  Do not
rescale C or Q.  The next allowed step is transport / material /
state-uncertainty diagnosis on this same contracted input.
