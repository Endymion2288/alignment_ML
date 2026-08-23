# 2026-08-23 (51) 真实数据 reduced Station calibration mode 可行性审计

## 任务

条目 50 已确认：原来的 6-DoF Station Mode 不能同时满足可辨识性与
cross-level isolation，也不应再修补。本条目只做 **analysis**，重新定义
“真实数据本身能安全约束哪些 station 自由度”。

冻结不变：不改 V2 / route / threshold / unmatched penalty；不写
geometry；不启动 C_dx Mode；不打开 sealed test；不新增 DoF。
`dz` 从 track-driven solve 中彻底移除，固定 survey 值 0。

本步使用已有 14973/14974 的 12 个 station FD Jacobian 与冻结 MC `A`，
无需 Condor。

## 本步已落地

配置：
`configs/operating_protocol_v1_real_data_reduced_station_mode_feasibility_v1.yaml`

Provenance：
`outputs/operating_protocol_v1_real_data_reduced_station_mode_feasibility_v1/`

- `reduced_mode_identifiability_audit.json`
- `reduced_mode_leakage_audit.json`
- `reduced_mode_transfer_dq_report.json`
- `reduced_station_mode_feasibility_report.json`
- `operating_protocol_next_decision.json`

测试 `tests/test_real_data_reduced_station_mode.py` 10 项通过。

## 预声明 reduced mode

`dz` 一律不浮动。含 `dx` 或 `ry` 的 mode，若沿冻结 `A` 的等效
`|C_dx|` 超过 1.5–1.7 µm，立即拒绝，不允许用更多事件或更松阈值挽救。

| mode | 浮动 | 14973 cond | 14974 cond | 与 A 投影 | 等效 \|C_dx\| | 跨 run | 准入 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `{dy,rx,rz}` | 3 | 73 | 86 | 0.040 | 不含 dx/ry | 不一致（最大 258σ） | 否 |
| `{dy,rx,ry,rz}`（固定 dx） | 4 | 91 | 1624 | 0.476 | 1126 / 479 µm | 不一致 | 硬拒绝 |
| `{dx,dy,rx,rz}`（固定 ry） | 4 | 89 | 90 | 0.881 | 60 / 268 µm | 不一致 | 硬拒绝 |
| `{dx,dy,rx,ry,rz}` | 5 | 114 | 1.64e4 | 1.00 | 20–1100 µm | 不一致 | 硬拒绝 |

`{dy,rx,rz}` 在两个标定 run 上满秩、条件数稳定且与 `A` 明显正交，
但这只是 Fisher 空间的可辨识性。self-nulling 点估计跨 run 完全不能
搬运（`dy` +1.08 vs −8.93 mm，`rz` −4.3 vs +95 mrad），14974→14973
的线性 transfer 使 χ² 恶化约 16 倍。因此它 **不是** V2 candidate。

更小的 isolation-safe 子集（`{dy}` / `{rx}` / `{rz}` 及两两组合）同样
满秩、与 `A` 正交，但无一跨 run 一致。

## Common identifiable subspace

按轴与冻结 `A` 的余弦：

- `dx` 0.880，`ry` 0.474 → 泄漏方向，不能安全浮动
- `dy` 0.025，`rx` 0.030，`rz` 0.005 → isolation-safe

推荐的 Fisher 子空间是 `{dy,rx,rz}`，其余 `dx/ry/dz` 交给
survey/external。该子空间 **没有** 通过“跨 run 一致 + transfer 不恶化”
合同，所以不能升级为可写/可更新的 calibration mode。

## 只读 transfer DQ

14975/14976/14977 没有 FD Jacobian，不能施加候选几何。
当前几何下 residual IQR：14975/14976 与标定兼容；14977 只有 2 条
selected route，不足判断。这是 DQ，不是 alignment 成功。
14977 不参与判定。

## 唯一结论

**`real_data_residual_dq_monitoring_only`**

- 不存在 Real-Data Station Calibration Mode V2 candidate
- 全部 station DoF（含原来认为 isolation-safe 的 `dy/rx/rz`）在当前
  真实数据上都不能产生可搬运的 geometry update，交给
  survey/external alignment
- `geometry_write_allowed=false`
- 不写任何 self-nulling correction
- 不启动 C_dx Mode
- residual 下降不是成功

当前真实数据只能做 dedicated residual / DQ monitoring，不能产生
station geometry update。
