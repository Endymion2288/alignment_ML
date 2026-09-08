# 独立 Measurement Transport 合同（Stage B / Task B14T）

Workbook 116。WB115 已经证明：冻结的 measurement likelihood 在
event 0/1 上可以求值、求 Jacobian、做数值缩放，并且 restart
不变。但独立 evaluator 无法对每一个 WB109 surviving measurement
定义 `h_i(θ)`。失败不是一类笼统的 “propagation failed”：

```
Class M  100043/37   IFT → 第一张下游 hit
                     跨越 spectrometer magnet 的长 hop
                     maxSteps 中止

Class S  100048/86   magnet hop 成功
                     1234.97 → 1235.86 mm stereo 对
                     globalToLocal / position not on surface
```

本任务只修 **transport 合同**。不改 WB114 统计模型，不重调
Gauss–Newton，不加 prior / ridge，不删除 37/86，不用 truth q/p，
不进 B14M / B15 / Measurement Model V2。

```
theta = (loc0, loc1, phi, theta, q/p)
alpha = (loc0, theta)
nu    = (loc1, phi, q/p)
chi2(θ) = Σ r_i(θ)^T R_i^{-1} r_i(θ)
R_i = (0.08 mm)² / 12
```

## ACTS Kalman 实际在做什么

钉死的 ACTS 32.0.2 `KalmanFitter` 由 navigator 驱动。测量面被插入
为 **external surfaces**。源码注释写明：尝试打到这些面时
**忽略 boundary check**。`Navigator` 对 external surface 和
target volume 初始化使用 `BoundaryCheck(false)`。
`SurfaceReached` 默认却是 `BoundaryCheck(true)`。
`s_onSurfaceTolerance = 1e-4` mm。

Kalman **会**做 measurement update，然后从 filtered state 继续。
独立 likelihood **不能**这样做，否则冻结的 `chi2(θ)` 会变成有状态
的 filter 目标。

## 正式 Mode B

evaluator 仍然在一条确定性轨迹上逐 measurement hop。hop 目标是
**支撑平面**（`SurfaceReached.boundaryCheck = false`），与 Kalman
external-surface 瞄准一致。`loc0` 是平面坐标。`insideBounds`
只记录，**不**作为 `chi2` 的必要条件。

Mode A（WB115 有界面 `SurfaceReached`）只作对照。20000 步 overlay
只是诊断行。正式 `maxSteps` 仍是 4000。“步数加大就成功”不是 PASS。

## 允许的判决

- `standalone_measurement_transport_contract_established`
- `reconstructed_state_not_transportable_to_surviving_measurements`
- `measurement_surface_projection_contract_not_established`
- `acts_navigation_transport_contract_not_established`
- `mixed_or_inconclusive`

`physical_nonidentifiability` 不是 B14T 判决。

正式 run `sbb14t_profile_transport_20260907T212851Z_f00dad92`：

```
decision = mixed_or_inconclusive
smoke_gate_passed = false
b14m_reopen_authorized = false
jacobian_contract_established = false
```

Mode B 已让 `h_i` 在 0/1/37/86 上可求值。0/1 与 WB115 一致。
37 是单调磁铁跨越到支撑平面（未打中 active wafer）。86 已在
stereo 平面上，但 loc0 越出 bounds 约 19 μm。86 的 Jacobian
spot check 失败，因此不重开 optimizer。

这不是 B14M PASS，也不授权 1989 行全样本。
