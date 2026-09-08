# ACTS Transport Dump Helper (Stage B / Task B0)

Workbook 98.  After WB97 showed that Python cannot safely call
`FaserActsExtrapolationTool::propagate`, this task adds an independent C++
helper inside `alignment_ML` only.

It does not edit Calypso/FaserActs source, does not construct Acts objects in
PyAthena, and does not enter alignment or Measurement Model V2.

## Models

- Model 0: `C0 = F Cin F^T` with MS/Eloss off
- Model 1: `C1 = C0 + Q_ACTS` with MS/Eloss on (the pass model)
- Model 2: Highland diagnostic only; it cannot replace ACTS

`Cin` is the persisted CKF 5x5 from WB96, converted to
`(x, y, tx, ty, q/p)` with signed `q/p` in `1/MeV`.  `F` is a numerical
Jacobian of the no-material propagate.  `Q_ACTS := C1 - C0` is the honest ACTS
increment, not a tuned scale.

## Situation classes

- A: `Q` still not materialized — keep fixing the transport export
- B: `Q` is visible but closure fails — then physics diagnosis / MM V2
  discussion is allowed
- PASS: `Q != 0` (not a scale), construction+validation pass the frozen
  WB81/WB87 gates

WB98 official run `sbb0_acts_transport_dump_20260906T172303Z_9825e122` is
situation B: Q is visible, closure fails.  Measurement Model V2 was not entered.
