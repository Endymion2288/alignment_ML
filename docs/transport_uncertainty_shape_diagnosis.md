# Transport / Material / State-Uncertainty Shape Diagnosis (Stage B / Task B9)

Workbook 105.  WB104 left Transport Covariance V3 as
`transport_covariance_shape_not_validated`: typical-event χ² can be
reasonable while every official pair fails pencil and generalized
eigenvalue.  This task asks why ACTS `C_prop` is systematically
wider than the empirical target residual covariance on the frozen
WB103 contracted input.

The task does not retune C or Q, drop `100043/37`, invent
`Cov(pred,target)`, reopen raw CKF, tighten the reconstruction
contract, or enter Measurement Model V2 / alignment.

## Official residual contract

The official WB81 / WB104 residual is versus truth, not versus a
target measurement:

```
e = x_prop − x_truth
C = C_prop = MODEL1 4×4
```

The official gate does **not** add `C_target`.  The independence
formula `C_residual = C_pred + C_target` is therefore **not** what
V3 used.  If a later measurement residual were used, the required
formula would be

```
C_residual = C_pred + C_target − 2 Cov(pred,target)
```

This task does not invent that cross-covariance.

## Official input

Eligibility is copied from WB103 and is independent of residual,
χ², pull, and truth.  Raw CKF is not the official input.

| Quantity | Count |
| --- | --- |
| Contracted rows | 1989 |
| Official pairs | 1974 |
| Unique CKF tracks | 663 |

Event `100043/37` remains.

## B9.1 Covariance budget

Per split and pair the audit reports Cin (source-surface Frobenius
only), `F Cin F^T`, material-off `C0`, material-on `C1`, and
target-surface `C_emp`.  Cin is not compared to target `C_emp`
here; that comparison is invalid across surfaces.

Official `C_pred` is transported CKF Cin (`C1` for MODEL1).
Official `C_target` is absent from the gate.  There is no
double-counting of `C_target` in the official truth residual.
Shared-fit still matters because Cin is not a leave-target-out
prediction.

Construction (0,1) Frobenius: `C_emp` = 4.13, Cin 4×4 = 2.81,
`F Cin F^T` = 88.0, `C0` = 85.3, `C1` = 85.2.  Transported Cin
already exceeds `C_emp` before material-on is interpreted as
process noise.

## B9.2 Shared-measurement dependency

Every official pair (fraction 1.0) has source and target on the
same reconstructed CKF long track.  Every contracted track has
four tracklet stations, so the target station participated in the
fit.  Source state origin is `Trk::Track.trackParameters().front()`.

Filter versus smoother, per-cluster identities, and
`Cov(pred,target)` are **unavailable** in the current dump/ntuple
and are not estimated.

```
prediction_and_target_measurement_independent = false
cov_pred_target_estimated = false
```

`Track_InStation0` is still typically 0; station membership uses
tracklets, not that flag.

## B9.3 Cin shape (before propagation)

663 unique contracted CKF tracks, all matched to source-station
truth for diagnosis only.

| Cin diagonal | Median | Mean |
| --- | --- | --- |
| x | 2.96 | 2.84 |
| y | 0.0135 | 0.0188 |
| tx | 1.94e-6 | 1.63e-5 |
| ty | 1.49e-6 | 7.95e-6 |
| q/p | 4.90e-13 | 1.39e-11 |

Cin 5×5 condition-number median is 5.8e12 because the q/p
eigenvalue is ~1e-12 against an x eigenvalue ~3.  Median q/p
correlations with x/y/tx/ty stay near 0.  Cin Frobenius is almost
flat across the frozen |q/p| bins (median 2.93–3.00).

Cin 4×4 versus empirical source-state error:

| λ(C_emp, Cin) | Dominant | Overwide Cin |
| --- | --- | --- |
| 0.027 | ty | yes |
| 0.054 | tx | yes |
| 0.153 | tx | yes |
| 11.35 | ty | no |

Pencil = 6.80 on `x`.  Overcoverage is already present **before**
propagation.  One ty combination is under-covered; Cin is not a
uniform scalar inflate.

## B9.4 Material-on versus material-off

`Q = C1 − C0` is not treated as a PSD process-noise matrix and is
not projected.

| Split | Pair | C0 pencil | C0 λ | C1 pencil | C1 λ | C1 further overwide |
| --- | --- | --- | --- | --- | --- | --- |
| construction | (0,1) | 22.39 | 0.036–0.191 | 21.50 | 0.006–0.070 | yes |
| construction | (0,2) | 23.16 | 0.018–0.197 | 20.99 | 0.018–0.155 | yes |
| construction | (0,3) | 22.07 | 0.030–0.191 | 20.17 | 0.030–0.144 | yes |
| validation | (0,1) | 12.89 | 0.021–0.629 | 12.67 | 0.007–0.111 | yes |
| validation | (0,2) | 12.96 | 0.014–0.628 | 12.06 | 0.009–0.094 | yes |
| validation | (0,3) | 11.94 | 0.030–0.628 | 10.85 | 0.027–0.181 | yes |

`C0` is already overwide.  Material-on does not repair shape; it
further shrinks the smallest generalized eigenvalues.  Missing
material is not the primary mismatch.

## B9.5 Transport linearization

Contracted `‖C0 − F Cin F^T‖_F / ‖C0‖_F`: median 1.63e-4, mean
1.00, p95 0.155, fraction above 0.05 = 6.7%.  The ~7% numerical
Jacobian tail is **not** upgraded to a systematic transport
failure: the median is far below 0.05 and the tail fraction is
far below 30%.

Event `100043/37` Jacobian relative error is ~1.6e-4.  That focus
tail is not a linearization failure.

## B9.6 Focus 100043/37

Retained.  Not deleted, downweighted, covariance-inflated, or
replaced by truth q/p.  It still owns 80.3% of the pooled
official χ².  Construction leave-one-out (diagnostic only):

| Pair | LOO pencil | LOO λ |
| --- | --- | --- |
| (0,1) | 21.84 | 0.006–0.070 |
| (0,2) | 21.39 | 0.018–0.155 |
| (0,3) | 20.53 | 0.029–0.144 |

```
ckf_tail != covariance-shape root cause
```

## Case classes

| Case | Token | This run |
| --- | --- | --- |
| A | `shared_measurement_covariance_semantics_missing` | **Active** (shared-fit fraction 1.0; Cin is not leave-target-out) |
| B | `ckf_input_covariance_overcovered` | **Active** (Cin already fails vs source empirical error) |
| C | `material_transport_shape_mismatch` | No (`C0` already overwide) |
| D | `transport_linearization_mismatch` | No (Jacobian tail is numerical, not systematic) |
| E | `mixed_or_inconclusive` | **Primary** because A and B both fire |

Official run `sbb9_transport_uncertainty_shape_20260906T201529Z_793f63dd`.

```
verdict = DIAGNOSED
decision = mixed_or_inconclusive
next_step = no_forced_single_root_cause
measurement_model_v2_authorized = false
measurement_model_v2_entered = false
conservative_covariance_declared_acceptable = false
```

Do not force a single root to advance.  Do not treat overcoverage
as acceptable conservative covariance.  Do not invent an empirical
cross-covariance.

Allowed next work, still on this contracted input and still not
Measurement Model V2:

1. leave-target-out prediction contract (mechanism A)
2. CKF fit-covariance semantics (mechanism B)

Then redo Transport Covariance V3.  Only a V3 PASS authorizes
Measurement Model V2.
