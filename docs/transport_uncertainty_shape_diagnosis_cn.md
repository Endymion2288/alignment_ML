# Transport / Material / State-Uncertainty 形状诊断（Stage B / Task B9）

Workbook 105。WB104 把 Transport Covariance V3 留在
`transport_covariance_shape_not_validated`：主体事件 χ² 可以合理，
但每个 official pair 都失败 pencil 与 generalized eigenvalue。
本任务问：在冻结的 WB103 契约输入下，为什么 ACTS `C_prop`
相对经验 target residual covariance 系统性过宽。

不调 C/Q，不删 `100043/37`，不编造 `Cov(pred,target)`，
不重开 raw CKF，不收紧 reconstruction contract，
不进入 Measurement Model V2 / alignment。

## 正式残差契约

正式 WB81 / WB104 残差相对 truth，不是相对 target measurement：

```
e = x_prop − x_truth
C = C_prop = MODEL1 4×4
```

正式闸门**不加** `C_target`。因此 V3 **没有**使用
`C_residual = C_pred + C_target`。若以后改用测量残差，
正确公式应是

```
C_residual = C_pred + C_target − 2 Cov(pred,target)
```

本任务不编造该交叉协方差。

## 正式输入

资格谓词与 WB103 相同，不用残差、χ²、pull、truth。
raw CKF 不是正式输入。

| 量 | 计数 |
| --- | --- |
| 契约行 | 1989 |
| official pair | 1974 |
| 唯一 CKF track | 663 |

`100043/37` 保留。

## B9.1 协方差预算

按 split / pair 报告 Cin（只给 source-surface Frobenius）、
`F Cin F^T`、material-off `C0`、material-on `C1`、
target-surface `C_emp`。这里不把 Cin 与 target `C_emp` 比形状；
那是跨表面的无效比较。

正式 `C_pred` 是传播后的 CKF Cin（MODEL1 用 `C1`）。
正式闸门没有 `C_target`。官方 truth 残差不存在
`C_target` 重复计数。共享拟合仍然重要，因为 Cin
不是 leave-target-out 预测。

construction (0,1) Frobenius：`C_emp` = 4.13，Cin 4×4 = 2.81，
`F Cin F^T` = 88.0，`C0` = 85.3，`C1` = 85.2。
传播后的 Cin 在把 material-on 解释成 process noise 之前
就已经大于 `C_emp`。

## B9.2 共享测量依赖

每个 official pair（比例 1.0）的 source 与 target 来自
同一条重建 CKF long track。每条契约 track 都有四个
tracklet 站，因此目标站参与了拟合。
source 状态来自 `Trk::Track.trackParameters().front()`。

滤波 / 平滑、逐 cluster 身份、`Cov(pred,target)`
在当前 dump / ntuple 中**不可用**，也未估计。

```
prediction_and_target_measurement_independent = false
cov_pred_target_estimated = false
```

`Track_InStation0` 仍几乎总是 0；站完备性用 tracklet，不用该旗标。

## B9.3 Cin 形状（传播前）

663 条唯一契约 CKF track，全部匹配到源站 truth，只作诊断。

| Cin 对角 | 中位数 | 均值 |
| --- | --- | --- |
| x | 2.96 | 2.84 |
| y | 0.0135 | 0.0188 |
| tx | 1.94e-6 | 1.63e-5 |
| ty | 1.49e-6 | 7.95e-6 |
| q/p | 4.90e-13 | 1.39e-11 |

Cin 5×5 条件数中位数 5.8e12，因为 q/p 本征值约 1e-12，
而 x 本征值约 3。q/p 与 x/y/tx/ty 的中位相关接近 0。
Cin Frobenius 在冻结 |q/p| 箱几乎平坦（中位数 2.93–3.00）。

Cin 4×4 相对经验源站误差：

| λ(C_emp, Cin) | 主轴 | Cin 过宽 |
| --- | --- | --- |
| 0.027 | ty | 是 |
| 0.054 | tx | 是 |
| 0.153 | tx | 是 |
| 11.35 | ty | 否 |

pencil = 6.80，主轴 `x`。过覆盖在**传播前**已经存在。
有一个 ty 组合方向覆盖不足；Cin 不是均匀标量膨胀。

## B9.4 material-on 与 material-off

`Q = C1 − C0` 不按 PSD process-noise 解释，也不做投影。

| split | pair | C0 pencil | C0 λ | C1 pencil | C1 λ | C1 进一步过宽 |
| --- | --- | --- | --- | --- | --- | --- |
| construction | (0,1) | 22.39 | 0.036–0.191 | 21.50 | 0.006–0.070 | 是 |
| construction | (0,2) | 23.16 | 0.018–0.197 | 20.99 | 0.018–0.155 | 是 |
| construction | (0,3) | 22.07 | 0.030–0.191 | 20.17 | 0.030–0.144 | 是 |
| validation | (0,1) | 12.89 | 0.021–0.629 | 12.67 | 0.007–0.111 | 是 |
| validation | (0,2) | 12.96 | 0.014–0.628 | 12.06 | 0.009–0.094 | 是 |
| validation | (0,3) | 11.94 | 0.030–0.628 | 10.85 | 0.027–0.181 | 是 |

`C0` 已经过宽。material-on 不修复形状，反而把最小
广义本征值压得更小。物质不足不是主因。

## B9.5 Transport 线性化

契约样本 `‖C0 − F Cin F^T‖_F / ‖C0‖_F`：中位数 1.63e-4，
均值 1.00，p95 0.155，高于 0.05 的比例 6.7%。
约 7% 的数值 Jacobian 尾巴**不**升级为系统性 transport 失败：
中位数远小于 0.05，尾巴比例远小于 30%。

`100043/37` 的 Jacobian 相对误差约 1.6e-4。该焦点尾巴
不是线性化失败。

## B9.6 焦点 100043/37

保留。未删除、未降权、未膨胀协方差、未用 truth q/p 替换。
仍占混合正式 χ² 的 80.3%。construction leave-one-out
（只诊断，不是 cut）：

| pair | LOO pencil | LOO λ |
| --- | --- | --- |
| (0,1) | 21.84 | 0.006–0.070 |
| (0,2) | 21.39 | 0.018–0.155 |
| (0,3) | 20.53 | 0.029–0.144 |

```
ckf_tail != covariance-shape root cause
```

## 分类

| 情况 | token | 本 run |
| --- | --- | --- |
| A | `shared_measurement_covariance_semantics_missing` | **成立**（共享拟合比例 1.0；Cin 不是 leave-target-out） |
| B | `ckf_input_covariance_overcovered` | **成立**（Cin 相对源站经验误差已经失败） |
| C | `material_transport_shape_mismatch` | 否（`C0` 已经过宽） |
| D | `transport_linearization_mismatch` | 否（Jacobian 尾巴是数值的，不是系统性的） |
| E | `mixed_or_inconclusive` | **主因**（A 与 B 同时成立） |

正式 run `sbb9_transport_uncertainty_shape_20260906T201529Z_793f63dd`。

```
verdict = DIAGNOSED
decision = mixed_or_inconclusive
next_step = no_forced_single_root_cause
measurement_model_v2_authorized = false
measurement_model_v2_entered = false
conservative_covariance_declared_acceptable = false
```

不得为了推进而强行选单一根因。不得把过覆盖说成可接受的
保守协方差。不得编造经验交叉协方差。

允许的下一步，仍在本契约输入上，仍不是 Measurement Model V2：

1. leave-target-out 预测契约（机制 A）
2. CKF 拟合协方差语义（机制 B）

然后重做 Transport Covariance V3。只有 V3 PASS 才授权
Measurement Model V2。
