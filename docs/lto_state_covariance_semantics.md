# LTO State / Covariance Semantics (Stage B / Task B14)

Workbook 110.  WB109 materialized independent leave-target-out
states.  This task asks whether those states and their 5×5
covariances have correct uncertainty semantics **before**
propagation.

The comparison is official full-track Cin (WB107) versus the new
LTO Cin (WB109/B13) on the frozen WB103 contracted identities and
the construction/validation split.  Truth is diagnostic only.

B14 does **not** validate transport covariance and does **not**
enter V4 arms C/D.  Those remain Task B15.

## What is compared

```
5×5 eigen spectrum
condition number
q/p uncertainty
cross correlations
source-state empirical error covariance
generalized eigenvalues
pencil
per-component pull
source / target dependence
```

## Questions

1. Does the WB105 pre-propagation Cin overcoverage disappear or
   shrink under LTO?
2. Is official full-track smoothing / shared-measurement leakage
   the shape-mismatch source?
3. Is the LTO covariance still overwide or undercovered?
4. Does the `100043/37` wrong-momentum state survive LTO refit?

## Forbidden

Rescale, eigenvalue clipping, PSD projection, empirical
calibration, truth-tuning the fitter, dropping ugly events,
redefining WB103 eligibility, and Measurement Model V2.

## Official tokens

PASS, and only PASS, may set:

```
decision = lto_state_covariance_semantics_established
lto_cin_contract_established = true
b15_authorized = true
```

This still keeps:

```
transport_covariance_validated = false
measurement_model_v2_authorized = false
```
