# 3ST field-normalized bending — ISOLATED (Yasu-S2K)

Workbook 127.  Independent review (P0) showed that `bending_raw` is a
two-chord angle difference whose physical response is the
triangle-weighted curvature kernel \(W(z)\), not the unweighted path
integral \(\int(B_x\,dz-B_z\,dx)\).  On a uniform-field analytic
circular orbit the existing `qp_bending_proxy` is \(\sim 0.5\times\)
truth.  Existing smoke/batch artifacts are in-progress implementation
output, not a completed calibration.

```
decision = three_st_qp_field_normalized_bending_isolated
qp_bending_proxy_isolated = true
physical_interpretation_authorized = false
times_two_patch_authorized = false
s2k_batch_authorized = false
fit_free_scale_from_truth = false
three_st_qp_trusted_observable = false
residual_conditional_authorized = false
next_authorized_stage = YASU-S3A
```

Do **not** multiply the kernel by two, fit a free truth scale, continue
the S2K batch, or use `qp_bending_proxy` for physics or calibration.
The next authorized stage is Yasu-S3A (Workbook 128): a
fixed-measurement curvature/geometry response Jacobian that does not
require a trusted reconstructed momentum.

An independent analytic-circle unit test expects proxy/truth \(\approx 0.5\).
The formula is retained in code only for provenance.

P1 side-audits (plurality matcher, S1 truth-momentum surface,
bound→curvilinear semantics, CircleFit static map, bending space-point
fallback, WB118 4ST association) are recorded in Yasu-S3A.  They do not
reopen ten diagnostic stages here.
