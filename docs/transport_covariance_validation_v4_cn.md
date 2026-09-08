# Transport Covariance Validation V4（Stage B / Task B12）

Workbook 108。B10 与 B11 之后的证伪，不是重跑 V3。
在冻结的 WB103 输入和冻结的 WB81/WB87/WB98 闸门上
预注册四条臂。

| 臂 | 状态 | 协方差 | 本 run |
| --- | --- | --- | --- |
| A | full-track CKF | C0 | 用冻结 WB104 指标评估 |
| B | full-track CKF | C1 | 用冻结 WB104 指标评估 |
| C | leave-target-out | C0 | **封锁** |
| D | leave-target-out | C1 | **封锁** |

`100043/37` 保留。不修 Cin。不改闸门。不进 Measurement Model V2。

## 正式结果

```
verdict = NOT_ESTABLISHED
decision = leave_target_out_prediction_contract_not_established
primary_case = leave_target_out_arms_unavailable
measurement_model_v2_authorized = false
shared_measurement_leakage_isolated = false
cin_semantics_isolated = false
material_on_isolated = false
```

正式 run `sbb12_transport_covariance_v4_20260906T212151Z_2112e1b4`。

因为两条 leave-target-out 臂不存在，V4 还不能区分
共享测量泄漏和 Cin 语义谁是形状失配的根因。
这不是强行选单一根因、或把过覆盖说成可接受的理由。

## Full-track 臂（A / B）

与 WB104 同一契约输入、同一 dump、同一闸门。construction (0,1)：

| 臂 | N | χ² 均值 | χ² 中位 | pencil | λ | 形状 |
| --- | --- | --- | --- | --- | --- | --- |
| A C0 | 370 | 7.38 | 0.048 | 22.40 | 0.036–0.197 | 失败 |
| B C1 | 370 | 13.13 | 0.359 | 21.50 | 0.006–0.070 | 失败 |

C0 已经过宽。C1 不是修复。WB105 的传播前 Cin 仍是
`λ = 0.027 / 0.054 / 0.153 / 11.35`，pencil 6.80。

## 分类表

| 结果 | token | 本 run |
| --- | --- | --- |
| LTO + 经认证 Cin 通过冻结闸门 | `transport_covariance_validated` | 否 |
| LTO 修复形状，Cin 传播前仍失败 | `ckf_covariance_contract_repair` | 否（无 LTO） |
| Cin 成立，material-on 破坏形状 | `acts_material_process_noise_diagnosis` | 否（无 LTO） |
| 两者仍在 | `mixed_or_inconclusive` | 不选用；LTO 臂不可用 |
| LTO 契约缺失 | `leave_target_out_prediction_contract_not_established` | **是** |

下一步允许的是不包装 `KalmanFitterTool.fit` 的独立
leave-target-out helper。只有以后 V4 PASS 才授权
Measurement Model V2。
