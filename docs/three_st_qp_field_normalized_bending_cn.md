# 3ST 场归一化弯曲 — 已隔离（Yasu-S2K）

Workbook 127。独立审查（P0）表明 `bending_raw` 是两段 chord angle 的差，
物理 response 是带三角形权重 \(W(z)\) 的曲率 kernel，而不是无权路径积分
\(\int(B_x\,dz-B_z\,dx)\)。在均匀场解析圆轨道上，现有 `qp_bending_proxy`
约为 \(0.5\times\) truth。已有 smoke/batch 产物只是未完成的实现输出，
不是完成的校准。

```
decision = three_st_qp_field_normalized_bending_isolated
qp_bending_proxy_isolated = true
physical_interpretation_authorized = false
times_two_patch_authorized = false
s2k_batch_authorized = false
fit_free_scale_from_truth = false
three_st_qp_trusted_observable = false
residual_conditional_authorized = false
next_authorized_stage = YASU-S3A
```

**不要**乘 2 修补、用 truth 拟合自由 scale、继续跑 S2K batch，或把
`qp_bending_proxy` 用于物理或 calibration。下一授权阶段是 Yasu-S3A
（Workbook 128）：不要求可信 reconstructed momentum 的
fixed-measurement curvature/geometry response Jacobian。

独立解析圆轨道单元测试期望 proxy/truth \(\approx 0.5\)。公式仅作 provenance
保留。P1 侧审计登记在 Yasu-S3A，不在此重开十个诊断阶段。
