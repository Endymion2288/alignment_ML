# SegmentFit 协方差坐标契约审计 V1

Workbook 84，**SegmentFit 协方差坐标契约审计**。
**状态：已完成并冻结** —— 在单个交互会话内按冻结顺序执行；审计基于代码审计和
合成协方差闭合（在 Python 中重新实现坐标变换并注入已知协方差），**不**基于
alignment 残差或真实数据。

**最终决定：`deterministic_segmentfit_get_state_transform_bug`。**
**根本原因：`coordinate_convention_mismatch_and_jacobian_sign_error`。**
**位置：`SegmentFitAlg::GetState`**（`NtupleDumperAlg` 导出器的变换是**正确**的；
SegmentFit 的 hit-error 模型本身是**已校准**的）。

> **命名：** 本 Workbook 84 是 WB83 协方差支线的上游来源审计。它**不**重新打开
> 任何已关闭的 alignment / identifiability 支线（WB81
> `faseracts_propagated_covariance_not_calibrated`、WB82
> `existing_mc_real_wide_ty_support_validated`、WB83
> `source_tracklet_fit_covariance_not_calibratable` 均以冻结方式继承并做了 SHA 校验）。
> 它**不**修复协方差；它只**定位** WB83 机制。

整个 workbook 期间：`held_out_accessed=false`、
`real_data_alignment_authorized=false`、`geometry_write_allowed=false`、
`official_conditions_write_allowed=false`、`measurement_model_validated=false`。
它**不**读取真实数据残差，从不打开 held-out 数据，从不写 geometry/conditions，
从不修改协方差，从不添加 scale factor，从不根据 chi2 调参数。

## 科学问题（唯一的问题）

WB83 冻结了 `source_tracklet_fit_covariance_not_calibratable`，机制为
`position_xy_swap_with_slope_miscalibration`：导出的 source-tracklet 协方差存在
结构性的位置 x<->y 交换（不是 scale 错误），仅 XY swap 可以恢复位置 diagonal，
但无法修复剩余的 y-ty 虚假相关和尾部。WB84 问：**这个机制到底起源于哪里？**

审计链为

    SegmentFit native state: (loc1, loc2, phi, theta, q/p)
        -> NtupleDumperAlg 协方差变换
        -> 导出协方差: [x, y, tx, ty]

成功标准是回答 WB83 机制发生在 **(1)** SegmentFit 协方差生成、**(2)** 协方差
导出器变换，还是 **(3)** 坐标约定不匹配。

## 初始状态审计（STEP 1，已完成）

`load_config` 对冻结的 WB81/WB82/WB83 继承做 SHA 校验（任何不匹配都会抛出
`ConfigError`）：

- WB81 config `5a13b0cc…`，冻结决定
  `faseracts_propagated_covariance_not_calibrated`，机制
  `overestimated_transported_fit_covariance` —— 逐字验证。
- WB82 config `b4417b4d…`，冻结决定
  `existing_mc_real_wide_ty_support_validated` —— 逐字验证。
- WB83 config `398e78be…`，冻结决定
  `source_tracklet_fit_covariance_not_calibratable`，机制
  `position_xy_swap_with_slope_miscalibration` —— 逐字验证。
- 软件来源（只读）：Calypso
  `40892527e9c65409afd2378a2abfc25ddbddac03`，Athena `24.0.41`，ACTS `32.0.2`，
  导出器 `PhysicsAnalysis/NtupleDumper/src/NtupleDumperAlg.cxx`，SegmentFit
  `Tracker/TrackerRecAlgs/TrackerSegmentFit/src/SegmentFitAlg.cxx`，Curvilinear
  坐标架 `Tracking/TrkEvent/TrkEventPrimitives/TrkEventPrimitives/CurvilinearUVT.icc`。

## 方法

审计在 Python 中重新实现
（`alignment/segmentfit_covariance_coordinate_contract.py`）：

1. **Athena `CurvilinearUVT` 坐标架**（`curvilinear_uvt`）—— 曲率面上
   （垂直于径迹方向）的局部 `loc1`/`loc2` 轴；
2. **`NtupleDumperAlg::globalTrackletCovariance` 数值 Jacobian**
   （`exporter_jacobian`）—— `d(x, y, tx, ty) / d(loc1, loc2, phi, theta)`；
3. **`SegmentFitAlg::GetState` 解析 Jacobian**（`segment_fit_jacobian`）——
   `d(x, y, phi, theta) / d(x, y, tx, ty)`，带一个 `sign_error` 开关，可复现
   代码原样（`True`）或数学修正形式（`False`）。

随后运行**合成协方差注入测试（Case A-E）**（在单个 native 参数中注入单位方差，
检查它落到哪个 global 分量）和**闭合测试**（反演实际变换，从导出协方差恢复
SegmentFit `(x, y, tx, ty)` 拟合协方差，再用修正后的 Jacobian 重新传播）。

## loc1/loc2 不是探测器局部 x/y

这是审计的第一项。`SegmentFitAlg::GetState` 里存在两套不能混用的 `loc1`/`loc2`
契约：

- **径迹参数 `loc1`/`loc2`** 属于 `Trk::CurvilinearParameters`
  （`SegmentFitAlg.cxx:722`）。它们是垂直于径迹的曲率面上的坐标。对 FASER
  束流径迹（`|t · z| ≥ 0.99`），Athena `CurvilinearUVT` 定义 `loc1`（`curvU`）~
  `-全局 y`，`loc2`（`curvV`）~ `+全局 x`。
- **探测器局部 `loc1`/`loc2`** 属于单独的 `FaserSCT_ClusterOnTrack` 测量
  （`SegmentFitAlg.cxx:711-712`，`fitCluster->localPosition()`）。那是 SCT
  wafer 测量，**不**进入导出器变换的 5×5 径迹参数协方差。

`GetState` 把**全局** `(x, y)` 拟合协方差写进了**曲率** `(loc1, loc2)` 槽位。
这就是坐标约定不匹配，不是探测器局部坐标排序错误。

## 发现

### 发现 1 —— 位置 x<->y 交换：坐标约定不匹配

`SegmentFitAlg::GetState` 在**全局**坐标下拟合 `(x, y, tx, ty)`，通过解析
Jacobian 把协方差转换为 `(x, y, phi, theta)`，并把它放进
`Trk::CurvilinearParameters` 的 `(loc1, loc2, phi, theta, q/p)` 槽位，**假设
`(loc1, loc2) = (全局 x, 全局 y)`**。

但 Athena `CurvilinearUVT` 坐标架对 FASER 束流径迹（`|t . z| >= 0.99`）定义
**`loc1`（`curvU`）~ `-全局 y`**，**`loc2`（`curvV`）~ `+全局 x`**。
合成注入测试确认了导出器的坐标契约：

- **Case A（注入 `loc1`）** 落到 **`y`**（Jacobian 列 `y = -1.0`）；
- **Case B（注入 `loc2`）** 落到 **`x`**（Jacobian 列 `x = +1.0`）；
- **Case E（注入 `q/p`）** 被丢弃（导出器跳过 `q/p` 列）。

因此导出器忠实地把 `loc1 -> -y`、`loc2 -> +x` 映射，而 SegmentFit 把 `var(x)`
放进了 `loc1`、`var(y)` 放进了 `loc2`，所以导出的 `cov_xx` 收到 `var(y)`、
`cov_yy` 收到 `var(x)`：一个**确定性的位置 x<->y 交换**。这正是 WB83 的观测
（导出的 `cov_xx` 中位 RMS 0.0099 = 精确的经验 y；导出的 `cov_yy` 中位 RMS
0.4966 = 不精确的经验 x）。

### 发现 2 —— 斜率 ty 失校准：Jacobian 符号错误

`SegmentFitAlg::GetState` 的解析 Jacobian 用了 **`d phi/d tx = +ty/r^2`**，
但对 `phi = atan2(ty, tx)` 数学上正确的导数是 **`-ty/r^2`**。闭合测试显示：

- **反演实际变换**（代码原样的 Jacobian）恢复出一个**已校准、方向无关**的
  SegmentFit 拟合协方差：`var(x) = 0.247 >> var(y) = 9.9e-5`（位置归属正确），
  `var(ty)/var(tx) ~ 0.0004 = 1/alpha^2` 对**所有** `|tx|` 区间成立（正确的
  stereo 几何），且 `cov(tx, ty) ~ 0`。所以 **SegmentFit 的 hit-error 模型是
  已校准的**。
- **用修正后的 Jacobian 重新传播该拟合协方差**，得到**方向无关、已校准**的
  `cov_tyty`（`~1.5e-7`，与经验 ty 分辨率相符），对所有 `|tx|` 区间成立 ——
  而代码原样的 Jacobian 则精确复现了实际的方向相关 `cov_tyty`
  （`sign_error_is_sole_cause_of_slope_miscalibration = true`）。

因此，方向相关的 `cov_tyty` 高估（即 WB83 的"slope miscalibration"）**完全**
由 `d phi/d tx` 符号错误造成，它以方向相关的方式把 `var(tx)` 耦合进 `var(ty)`。

### 结论 —— 失败定位

WB83 的 `position_xy_swap_with_slope_miscalibration` 机制是
**`SegmentFitAlg::GetState` 协方差变换中的两个确定性 bug**：

1. **坐标约定不匹配** —— 全局 `(x, y)` 协方差被写进 Curvilinear
   `(loc1, loc2)` 槽位时假设 `(loc1, loc2) = (x, y)`，但该坐标架实际是
   `(loc1, loc2) = (-y, +x)`；
2. **Jacobian 符号错误** —— `d phi/d tx = +ty/r^2`（应为 `-ty/r^2`）。

回答 WB84 的成功标准：该机制发生在 **(1) SegmentFit 协方差生成**，通过
**(3) 坐标约定不匹配**（外加一个 Jacobian 符号错误）。**(2) 协方差导出器变换
是正确的**，且 SegmentFit 的 hit-error 模型本身是**已校准的**。

WB83 曾把下一战役指向 `segment_fit_hit_error_model_covariance_audit`。该指针
**已被取代**：反演实际的 GetState+导出器变换可以恢复出已校准、方向无关的拟合
协方差（`var(x) ≫ var(y)`，`var(ty)/var(tx) = 1/α²`，`cov(tx,ty) ≈ 0`）。
失败位置不是 hit-error 模型。若单独预注册下一战役，应验证 **GetState 变换修复**，
而不是改写 hit-error 模型。`hit_error_model_audit_authorized=false`。

## 输出

- `coordinate_contract_audit.json` —— 坐标契约审计。
- `covariance_transform_matrix.json` —— 重建的变换矩阵。
- `synthetic_basis_test_report.json` —— 合成注入测试（Case A-E）。
- `failure_location_report.json` —— 失败定位 + 闭合测试。
- `segmentfit_covariance_coordinate_contract_decision.json` —— 决定。
- `campaign_summary.json` —— 战役总结。

## 下一步（此处未授权）

可以单独预注册一个新的**协方差修复验证战役**，修复 `SegmentFitAlg::GetState`
中的两个确定性 bug 并验证修复后的协方差。本战役**不**修复协方差。
Frozen-V2 alignment loop 仍未授权（`geometry_write_allowed=false`、
`real_data_alignment_authorized=false`、`measurement_model_validated=false`）。
