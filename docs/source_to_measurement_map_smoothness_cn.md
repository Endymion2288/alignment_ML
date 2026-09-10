# Source→Measurement 映射光滑性根因审计（Stage B / Task B14K）

Workbook 118。WB117 已经证明正式 sequential 支撑平面映射 `h_i(θ)`
在 `100048/86` 的 `loc1 / phi / q/p` 上没有可认证 Jacobian。那不是
“物理上不可微”的最终结论。本任务把 `h_i` 拆成阶段，定位第一个
不再收敛的连续数值操作。

不改 WB114 统计模型，不重调 Gauss–Newton，不按结果挑最好 FD 步长
或 `stepTolerance`，不把正式 sequential likelihood 换成
direct-from-source，不加 prior / ridge，不删 37/86，不用 truth q/p，
不以修 5D Cin 为目标，不进 B14M / B15 / Measurement Model V2。

```
theta = (loc0, loc1, phi, theta, q/p)
alpha = (loc0, theta)
nu    = (loc1, phi, q/p)
chi2(θ) = Σ r_i(θ)^T R_i^{-1} r_i(θ)
R_i = (0.08 mm)² / 12
```

正式 `h_i` 仍是一条确定性支撑平面顺序轨迹。生产
`PropagatorPlainOptions::stepTolerance` 仍是 ACTS 32.0.2 默认 `1e-4`。

## 允许的判决

- `transport_integration_resolution_insufficient`
- `supporting_plane_projection_ill_conditioned`
- `parameterization_scale_not_resolved`
- `acts_transport_jacobian_inconsistent_with_fd`
- `source_to_measurement_map_genuinely_nonsmooth`
- `mixed_or_inconclusive`

`source_to_measurement_map_genuinely_nonsmooth` 只能在确定性通过、
积分收紧不再改善、投影已检查、FD 信号高于数值噪声、参数尺度已检查、
且无分支切换之后使用。ACTS transport Jacobian 存在本身不是 inconsistency。

正式 run `sbb14k_map_smoothness_20260908T163845Z_2cbd5d0e`：

```
decision = mixed_or_inconclusive
smoke_gate_passed = true
jacobian_contract_established = false
b14m_reopen_authorized = false
restart_invariance_authorized = false
acts_transport_jacobian_available = true
analytic_vs_fd_not_yet_contracted = true
next_step = open_wb119_analytic_vs_fd_derivative_contract
```

更早的 `...T163048Z_6e7351eb` 把“ACTS Jacobian 存在”误写成
`acts_transport_jacobian_inconsistent_with_fd`，已作废。B14K 没有做
契约化的解析 vs 有限差分比较。

## 本 run 结论

同一 `θ` 的 transport 在 86 与预注册同 source 对照 `100048/44` 上都
可精确复现，predicted loc0 / χ² 重复差分为 0。

冻结 free state 后的平面投影光滑。`|n·d| ≥ 0.99986`，不是近平行相交。

官方 FD 信号高于该零噪声底。86 上 `h(q/p)/|q/p| ≈ 8.4e-5`，不是大非线性扰动。

86 的 hop 0 `loc1 / phi` 各级 Jacobian 在 1e-9–1e-12 收敛。真正失稳发生在
后续长磁传播 hop：target 1 为 hop 6（`z ≈ -1822 → 1203` mm，约 3.0 m，
129 步）；target 2/3 为 hop 11。hop 0 的 `q/p` `plane_intersection`
在 86 与 44 都略超 0.05 阈值，那是第一跳 loc0 响应对 `q/p` 极小，
不是 44 正式残差 Jacobian 失败（44 仍 PASS）。

预注册 `stepTolerance` 三档 `1e-4 / 1e-5 / 1e-6` 让 86 的 target 1/3
平均 last-pair 下降，且 `loc1` 在 `1e-6` 收敛，但 `phi / q/p` 仍失败；
target 2 非单调。因此不是
`transport_integration_resolution_insufficient`，也不得据此挑选生产
tolerance。对照 44 在官方阶梯上全部收敛。86 在该 source 内运动学孤立。

ACTS 在附加 dummy covariance 时可给出 `transportJacobian`。正式路径
仍用 `nullopt` 协方差和有限差分。ACTS 终点 `loc0` 与官方支撑平面
`loc0` 在每个诊断 hop 上一致。公平的列对列契约是 WB119，不是替换
正式 likelihood。

这不是 B14M PASS，也不授权 restart invariance 或 1989 行全样本。
