# ACTS Transport Covariance Diagnosis (Stage B / Task B3)

Workbook 99.  After WB98 materialized `C0`, `F`, `C1`, and `Q_ACTS` and
still failed the frozen closure gates, this task only diagnoses why

    C1 = F Cin F^T + Q_ACTS

does not cover

    e = x_prop - x_target.

It does not repair closure, enter Measurement Model V2, enter alignment,
tune Q, rescale covariance, or drop high-χ² events.

## What is audited

- **B3.1** State and surface contract: native Athena
  `(loc1, loc2, phi, theta, q/p)` → Acts bound → export
  `(x, y, tx, ty, q/p)`, target `PlaneSurface((0,0,z), n=ẑ)`,
  Forward/Backward, q/p sign, and the numerical Jacobians.
- **B3.2** Jacobian consistency: compare dumped `C0` to `F Cin F^T`
  directly.  `F` is never inferred from the covariances.
- **B3.3** Residual decomposition for `x, y, tx, ty, q/p` on the frozen
  construction/validation split and the frozen `|q/p|` bins.
- **B3.4** Top 1% and 0.1% χ² provenance.  Every tail event is kept.

## Case classes

| Case | Meaning | Next step |
| --- | --- | --- |
| A | State, surface, or Jacobian contract is wrong | Fix the transport contract |
| B | Material / scattering / process-noise model mismatch | ACTS material diagnosis |
| C | A few high-χ² events dominate the mean | Uncertainty-model analysis, no covariance retune |
| D | Transport contract is correct and closure still fails | Reassess Measurement Model V2 **input** model |

B3 does not enter V2 even if the case is D.

Official run `sbb3_acts_transport_diagnosis_20260906T180100Z_574d6429`
is **Case C** (high-χ² tail dominated), with secondary Case B
(`Q=C1−C0` often not PSD) and Case D (bulk x overcoverage).
The state/surface contract holds; typical `C0 ≈ F Cin F^T`.

## Frozen inheritance

WB87–WB98 tokens are read-only.  WB98 remains
`acts_q_materialized_closure_failed`.  Dumps stay under
`outputs/acts_transport_dump_v1/dumps/`.
