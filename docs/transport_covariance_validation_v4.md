# Transport Covariance Validation V4 (Stage B / Task B12)

Workbook 108.  Falsification after B10 and B11.  This is not a
rerun of V3.  Four arms were pre-registered on the frozen WB103
input and the frozen WB81/WB87/WB98 gates.

| Arm | State | Covariance | This run |
| --- | --- | --- | --- |
| A | full-track CKF | C0 | evaluated from frozen WB104 metrics |
| B | full-track CKF | C1 | evaluated from frozen WB104 metrics |
| C | leave-target-out | C0 | **blocked** |
| D | leave-target-out | C1 | **blocked** |

`100043/37` is retained.  Cin is not repaired.  Gates are not
retuned.  Measurement Model V2 is not entered.

## Official result

```
verdict = NOT_ESTABLISHED
decision = leave_target_out_prediction_contract_not_established
primary_case = leave_target_out_arms_unavailable
measurement_model_v2_authorized = false
shared_measurement_leakage_isolated = false
cin_semantics_isolated = false
material_on_isolated = false
```

Official run `sbb12_transport_covariance_v4_20260906T212151Z_2112e1b4`.

V4 cannot yet tell whether shared-measurement leakage or Cin
semantics is the root of the shape mismatch, because the two
leave-target-out arms do not exist.  That is not an excuse to
force a single root or to treat overcoverage as acceptable.

## Full-track arms (A / B)

Same contracted input, same dump, same gates as WB104.  Construction
(0,1):

| Arm | N | Mean χ² | Median χ² | Pencil | λ | Shape |
| --- | --- | --- | --- | --- | --- | --- |
| A C0 | 370 | 7.38 | 0.048 | 22.40 | 0.036–0.197 | fail |
| B C1 | 370 | 13.13 | 0.359 | 21.50 | 0.006–0.070 | fail |

C0 is already overwide.  C1 is not a repair.  Pre-propagation Cin
from WB105 remains `λ = 0.027 / 0.054 / 0.153 / 11.35`, pencil 6.80.

## Classification table

| Outcome | Token | This run |
| --- | --- | --- |
| LTO + certified Cin pass frozen gates | `transport_covariance_validated` | no |
| LTO repairs shape, Cin still fails before prop | `ckf_covariance_contract_repair` | no (LTO absent) |
| Cin holds, material-on breaks shape | `acts_material_process_noise_diagnosis` | no (LTO absent) |
| both remain | `mixed_or_inconclusive` | not chosen; LTO arms unavailable |
| LTO contract missing | `leave_target_out_prediction_contract_not_established` | **yes** |

Next allowed step is an independent leave-target-out helper that
does **not** wrap `KalmanFitterTool.fit`.  Only a later V4 PASS
authorizes Measurement Model V2.
