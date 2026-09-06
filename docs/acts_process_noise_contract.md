# ACTS Process-Noise Contract (Stage B / Task B)

Workbook 97.  After WB96 established the persisted CKF 5x5, this task asks
whether

    C_out = F C_in F^T + Q

with ACTS `MaterialInteractor` process noise describes the target residual on
the frozen WB87 construction/validation split.

It is not alignment and does not enter Measurement Model V2.

Model 0: no process noise.  Model 1: ACTS Q (the pass model).  Model 2:
Highland diagnostic only; it cannot replace ACTS and cannot be tuned to chi2.

Production extrapolation keeps `InteractionMultiScatering/Eloss/Record` false.
The validation job may enable those Gaudi properties without editing Calypso
C++.  Gates stay the frozen WB81/WB87/WB93 set.
