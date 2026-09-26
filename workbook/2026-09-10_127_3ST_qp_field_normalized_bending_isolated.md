# Workbook 127: 3ST field-normalized bending — **ISOLATED**（Yasu-S2K）

日期：2026-09-10
状态：**隔离 / 不解释** —— 独立 reviewer P0 已证明现有 `qp_bending_proxy` 的物理 kernel 错误。本 workbook **不是**完成的校准阶段。已有 smoke/batch 产物只是未完成的实现输出，不得用于物理解释或 calibration。未进入 S3、alignment、B14M/B15、MM V2。未改 fitter / seed / hit / geometry / covariance。未把 WB119 翻成 PASS。

**最终判定：`ISOLATED`**

- `decision = three_st_qp_field_normalized_bending_isolated`
- `qp_bending_proxy_isolated = true`
- `physical_interpretation_authorized = false`
- `times_two_patch_authorized = false`
- `s2k_batch_authorized = false`
- `fit_free_scale_from_truth = false`
- `three_st_qp_trusted_observable = false`
- `residual_conditional_authorized = false`
- `official_qp_like_jacobian_authorized = false`
- `next_authorized_stage = YASU-S3A`

已有 in-progress 产物（**不是**完成的 WB127）：

- smoke：`outputs/three_st_qp_field_normalized_bending_v1/yasu_s2k_three_st_qp_field_normalized_bending_smoke_20260910T121532Z_59fd69d6`
- batch：`outputs/three_st_qp_field_normalized_bending_v1/yasu_s2k_three_st_qp_field_normalized_bending_batch_20260910T125021Z_7fd3e13f`

## P0（必须先读）

`bending_raw = atan(t12) − atan(t23)` 是两段 chord angle 的差。它对曲率的物理 response 是带三角形权重 \(W(z)\) 的 kernel，**不是**整段无权 \(\int(B_x\,dz - B_z\,dx)\)。

在均匀场解析圆轨道上，现有公式给出约 \(0.5\times\) truth。S2K 测试曾用同一 proxy 公式生成“truth”，斜率 1 是循环论证。

因此：

- **不要**乘 2 修补
- **不要**用 truth 拟合自由 scale
- **不要**继续跑 S2K batch
- **不要**用 `qp_bending_proxy` 做任何物理解释或 calibration
- 下一阶段直接新开 **Yasu-S3A**（WB128）：fixed-measurement curvature/geometry response Jacobian

独立解析圆轨道控制已写入 `tests/test_three_st_qp_field_normalized_bending.py`，期望 proxy/truth \(\approx 0.5\)。

## 本阶段明确不做

- 不把孤立 proxy 当成 \(q/p\)-like Jacobian
- 不进入 \(E[r_{\rm IFT}|q/p]\) 或 residual-conditional
- 不声称自然数据中的 \(q/p\)-\(R_y\) weak mode
- 不改 CircleFit / CKF / geometry payload

P1 侧审计（plurality matcher、S1 truth momentum 表面、bound→curvilinear 语义、`s_spacePointMap` stale pointer、bending space-point fallback、WB118 4ST association）登记在 WB128，不在此重开十个诊断阶段。
