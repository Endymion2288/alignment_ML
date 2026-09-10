# 磁场梯度变分耦合修复（Stage B / Task B14X）

Workbook 123。WB122 已经把 `100043/1` 多出来的 0.001–0.002 `loc1`
耦合钉在第一条长磁 hop：`rk_free ∂dir/∂pos ≡ 0`，而正式 mean FD
看到了场图的空间依赖。本任务从源码证明 ACTS 32.0.2
`GenericDefaultExtension::transportMatrix` 省略了 `∂B/∂x`，在**只修
diagnostic tangent** 的前提下补齐同一条 mean RKN4 map，并重新契约
`100043/0,1,37` 与 `100048/86`。

```
WB122:
缺失耦合已定位
磁长 hop: ∂dir/∂pos 被错误地写成 0
        ↓
WB123:
证明场梯度 tangent 项
+
只修变分传播
+
hop-start segment FD 证伪
        ↓
control 0/1/37 derivative contract
        ↓
86 独立 segment reference
```

正式 `h_i(θ)` 一字不动。生产 `stepTolerance` 保持 `1e-4`。冻结 FD
ladder 保持 `h, h/2, h/4, h/8`。5% gate 不放宽。不加 prior / ridge，
不用 truth q/p，不调 Q / Cin，不增加 FD rung，不缩小 track-state FD
步长，不重开 B14M，不进 B15 / V4 / Measurement Model V2，不提交
1989。

```
theta = (loc0, loc1, phi, theta, q/p)
alpha = (loc0, theta)
nu    = (loc1, phi, q/p)
chi2(θ) = Σ r_i(θ)^T R_i^{-1} r_i(θ)
R_i = (0.08 mm)² / 12
r_i = m_loc0 - predicted_loc0_on_supporting_plane
```

残差效应小不等于契约通过。

## 允许的判决

- `field_gradient_variational_coupling_repaired_and_contracted`
- `field_gradient_coupling_repaired_focus_reference_unresolved`
- `field_gradient_hypothesis_not_supported`
- `field_gradient_variational_implementation_inconsistent`
- `mixed_or_inconclusive`

只有 Case A 才能把 `jacobian_contract_established` 和
`b14m_reopen_authorized` 设为 true：缺失 `∂dir/∂pos` 已被源码证明、
场梯度契约成立、正式 mean 路径不变、control 0/1/37 过冻结 5%
gate、并且 86 的独立 reference 由修复后 tangent ≈ hop-start
segment FD 建立。即便 Case A 也不提交 1989。

## 源码审计（必须先于任何 patch）

钉死的栈：

```
ACTS 32.0.2
AthenaExternals 24.0.41
EigenStepper
GenericDefaultExtension::transportMatrix
FASERMagneticFieldWrapper
```

`EigenStepper.hpp` 的 mean ODE：

```
dr/ds = T
dT/ds = (q/p) T × B(x)
```

`EigenStepper.ipp` 的 RKN4 在

```
pos0 = x
pos1 = x + (h/2) T + (h²/8) k1
pos2 = x + h T + (h²/2) k3
```

处采样 `B`，`k = (q/p) T_stage × B(x_stage)`，默认 extension 的
`kQoP = {0,0,0,0}`（`k` 里没有能量损失）。

`transportMatrix` 只填 `dFdT`、`dFdL`、`dGdT`、`dGdL` 和 `D(3,7)`。
头文件写明 ATL-SOFT-PUB-2009-002 eq. 18 目前为 0，`dGdx` 保持
Identity 初值下的零块，从不调用 `getFieldGradient`。这就是累积
`rk_free` 出现 `∂dir/∂pos ≡ 0` 的原因。

必须分开、不得混写：

```
均匀场 tangent     ← ACTS D 里已有
场梯度 tangent     ← 缺失；本任务只修这一项
能量损失确定项
多次散射 / process noise
协方差传输         ← dummy cov 只是变分开关
```

本任务只修 deterministic mean-state variational derivative。

## 同单位 tangent

诊断 extension 保持 ACTS 的 mean `k_i`，并对同一 map 求导：

```
dk = qop * (dT_stage)×B + qop * [T_stage]_× * G * dx_stage
```

`G = ∂B/∂x` 来自官方
`FASERMagneticFieldWrapper::getFieldGradient`，查询点与三个 RKN4
stage 位置相同。Wrapper 单位：原生 kT → ACTS Tesla，`G` 乘同一因子
（Tesla / mm）。若所有 `G = 0`，`D` 退回 ACTS
`GenericDefaultExtension::transportMatrix`。系数不是按示意 ODE
手写，而是钉死 mean step 的 RKN4 导数。不加入能量损失、process
noise、方向归一化 Jacobian。

## 场梯度契约

官方 gradient API 已经存在。1 mm 的 field-query FD 只是官方 `G`
的诊断，按磁区 O(10 mm) 网格在比较任何 track Jacobian **之前**
预注册。它不是官方 `G`，也不得按 track Jacobian 符合度回调步长。

## 只修诊断路径

```
using GradientStepper =
    Acts::EigenStepper<Acts::StepperExtensionList<FieldGradientDefaultExtension>>;
```

正式 `EigenStepper<>` 仍然计算 `h_i(θ)`。dummy covariance 仍然只
打开诊断 hop 的 `covTransport`。必须保持：

```
predicted_loc0 official == predicted_loc0 gradient-stepper mean
(|Δloc0| ≤ 1e-6)
chi2 来自正式 mean，前后一致
```

Hop-start segment FD 从真实 hop 起点出发，用正式 mean stepper 和
冻结的 `±h … ±h/8`。每个臂只对第一条长 hop 做 segment。禁止
analytic 自我认证：86 只有在 **required first-unstable** segment
FD 收敛并且与修复后 tangent 在冻结 5% gate 下一致时，才能建立
独立 reference。

## 数值冻结（WB123）

Official run：`sbb14x_field_gradient_repair_20260908T235722Z_f86bd896`

```
decision = field_gradient_coupling_repaired_focus_reference_unresolved
missing_term_source_proven = true
mean_path_unchanged = true          # 12 行 Δloc0 = 0
control_0_pass = true
control_1_pass = true              # loc1 6.16%/5.64% → 0.016%/0.035%
control_37_pass = true
focus_independent_reference_established = false
jacobian_contract_established = false
b14m_reopen_authorized = false
next_step = keep_86_independent_segment_reference
```

`100043/1` hit 6：旧 `∂dir/∂pos = 0`，新 `2.46e-6`（target 1）。
后续 0.001–0.002 loc1 coupling 已被解释。所有 control 第一条长
hop 的 hop-start FD 收敛且与 repaired hop 一致。`100048/86`
target 1 hit 6 的 segment FD 仍振荡（`last_pair_rel = 0.56`）；
target 2/3 的 required hit 11 没有 hop-start FD。86 全局 loc1
从 98% 降到 6–27%，但冻结四档仍不收敛。禁止用修复后 analytic
自我认证。

Config SHA `c290594f834233ace2c41d2d339edaa6fd9a4b6b88dc781ece67e014a012906b`。
Helper SHA `ff74be09335bd6c63cd1357570d4623563cef7a3bca921909a456c0f5208bbf2`。
Decision SHA `4632686ec5e68f4d30470438d257c50ec7056edf430ea2460453052fd0b995aa`。
