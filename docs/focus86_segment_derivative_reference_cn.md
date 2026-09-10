# 86 独立段导数参考（Stage B / Task B14Y）

Workbook 124。WB123 已经修好场梯度 tangent，并让 `100043/0,1,37`
在冻结 5% gate 下通过，正式 mean 完全不变。本任务**不再修改**
那份实现。唯一目标是给 `100048/86` 的 required first-unstable
长磁 hop 建立真正独立于 repaired analytic tangent 的 numerical
derivative reference。

```
WB123:
场梯度 tangent 已修
control 0/1/37 PASS
正式 mean 不变
86 required hop 参考缺失
        ↓
WB124:
1/6、2/11、3/11 的真实 hop-start production FD
+
独立 DOPRI5 mean ODE（同一 B(x)、同一 supporting plane）
+
独立 variational（含已证明的 ∂B/∂x）
与该积分器自己的冻结 FD ladder 双重闭合
        ↓
三者比较：repaired tangent / production FD / independent reference
        ↓
只有独立参考自身稳定且 5% 吻合
        ↓
jacobian_contract_established
        ↓
重开 B14M smoke
```

Required hops 冻结。禁止把 target 2/3 已经收敛的 earlier hit 6
偷换成 required reference：

```
target 1: hit 6
target 2: hit 11
target 3: hit 11
```

正式 `h_i(θ)` 一字不动。生产 `stepTolerance` 保持 `1e-4`。冻结
FD ladder 保持 `h, h/2, h/4, h/8`。5% gate 不放宽。不增加 rung，
不缩小 track-state FD 步长，不加 prior / ridge，不用 truth q/p，
不调 Q / Cin，不重开 B14M，不进 B15 / V4 / Measurement Model V2，
不提交 1989。禁止 analytic 自我认证。独立积分器容差在看到任何
86 Jacobian 符合度之前预注册，不得回调。

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

- `production_fd_not_certifying_but_independent_reference_established`
- `independent_segment_reference_contracted`
- `focus_segment_reference_not_established`
- `repaired_variational_focus_inconsistent`
- `mixed_or_inconclusive`

只有三个 required hop 的独立参考都建立、并且与 repaired tangent
在冻结 5% gate 下吻合，同时 control 0/1/37 继续 PASS、正式 mean
不变，才能把 `jacobian_contract_established` 和
`b14m_reopen_authorized` 设为 true。即便如此也只授权下一步
B14M **smoke** 与 restart invariance，不能直接提交 1989。

## 独立积分器（只作诊断）

仓库 / ACTS 栈没有第二条可替换 EigenStepper 的生产积分器。离线
参考是 `IndependentMeanOdeIntegrator.hpp`：对钉死的 mean ODE 做
Dormand–Prince 5(4)

```
dr/ds = T
dT/ds = (q/p) T × B(x)
d(q/p)/ds = 0
```

`B` 与 `∂B/∂x` 来自正式 `FASERMagneticFieldWrapper`（与生产同一
单位）。代码路径不是 EigenStepper，不是
`GenericDefaultExtension`，也不是 `FieldGradientDefaultExtension`。
永不替换正式 `h_i`。真空 / 只含场：无 navigator，无 material
actor。

在任何 86 Jacobian 比较之前预注册的容差：

```
method            dormand_prince_5_4
abs_tol_pos_mm    1e-6     # loc0 不变性地板
abs_tol_dir       1e-10
abs_tol_qop       1e-14
rel_tol           1e-9
initial_step_mm   1.0
min_step_mm       1e-6
max_step_mm       50.0
max_path_mm       20000
max_steps         100000
plane_hit_abs_mm  1e-6
```

独立导数参考仍使用冻结四档 track-state FD。只有该 FD 序列自身
收敛，并且——当独立 variational 存在时——与它双重闭合，才能
算参考建立。只拿 variational 去对 repaired tangent 不算参考。

## 数值冻结（WB124）

Official run：`sbb14y_focus86_segment_reference_20260909T092655Z_18ab2ab0`

```
decision = focus_segment_reference_not_established
control_0_pass = true
control_1_pass = true
control_37_pass = true
mean_path_unchanged = true          # 正式 repair Δloc0 = 0
target_exclusion_holds = true       # 12 行，0 leaked
focus_independent_reference_established = false
jacobian_contract_established = false
b14m_reopen_authorized = false
next_step = keep_86_independent_segment_reference
```

三个 required hop 现在都有 hop-start state、正式 EigenStepper
segment FD、独立 mean、独立 variational、以及独立四档 FD。无
branch switching。正式 mean 不变。独立 FD 在 `loc1/phi/q/p` 上
不收敛，因此独立参考未建立。不得回调预注册的 DOPRI5 容差。
不得因为独立 variational ≈ repaired tangent（两者 ≪ 5% 吻合）
而自我认证。

Config SHA `46ab8197bdbf40b6fa77c955ca5d5696f0fbbb7692213c2fa39edbe539162119`。
Helper SHA `3558bc6bdedb326a5dc48bba2dc4d57833574b82eb66d1a14fa8106d45d3edf1`。
Decision SHA `f4ba49cffe187929c9eb79ae1e6e604a9852f29b8ef2ed7e8da3f7dcbbd471e7`。
场梯度扩展 SHA 未改
`ca5e4f0ef1a7a09edd1c24099e7ce689c4f0511e2d4afab3a160ebd2d6ff03d8`。
