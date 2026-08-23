# Operating Protocol V1 reduced Station calibration mode 可行性

条目 51 / 2026-08-23。本步 **不再修补** 原来的 Station Mode，只问：
真实数据本身能安全约束哪些 station 自由度。只做分析。

`geometry_write_allowed` 保持 false。不启动 C_dx Mode。冻结 V2、
route policy、阈值 `0.001` 与 `unmatched_penalty=-1.0` 未改。`dz`
从 track-driven solve 中移除，固定为 survey 0。

复用 14973/14974 上已经捕获的 12 个 station FD probe。无需 Condor。

## 冻结不变量

- 传播：mode-0
- association：冻结 V2（`0c85a001…766a27`）
- 观测：`anchor_selected_field_edge` + `physical_edge_deduplicated`
- 泄漏算子 `A` 与 1.5–1.7 µm isolation budget
- 判定只用 14973/14974。14977 永不参与选择

禁止：写官方 conditions；写入任何 self-nulling 修正；启动 C_dx Mode；
联合 Newton；新增 layer/module DoF；打开 sealed test；把 residual
下降当成成功；用更多事件或更松阈值挽救 `dx`/`ry` 的 isolation 失败。

## DQ 与 alignment 正确性必须分开

| 产物 | 角色 |
| --- | --- |
| reduced mode 的 rank、condition、σ | 可辨识性诊断，不是 closure |
| 由浮动 `dx`/`ry` 推出的 implied \|C_dx\| | 拒绝诊断，不是测量 |
| 线性化 χ² 变化 | 只是 transfer DQ；下降永远不是成功 |
| 盲块在当前几何上的 residual IQR | 只是数据质量 |

## 预声明 mode

`dz` 一律不浮动。

| mode | 浮动 | 14973 cond. | 14974 cond. | A 投影 | implied \|C_dx\| | 跨 run | 准入 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `{dy,rx,rz}` | 3 | 73 | 86 | 0.040 | 不含 `dx`/`ry` | 不一致（最大 258σ） | 否 |
| `{dy,rx,ry,rz}`（固定 `dx`） | 4 | 91 | 1624 | 0.476 | 1126 / 479 µm | 不一致 | 硬拒绝 |
| `{dx,dy,rx,rz}`（固定 `ry`） | 4 | 89 | 90 | 0.881 | 60 / 268 µm | 不一致 | 硬拒绝 |
| `{dx,dy,rx,ry,rz}` | 5 | 114 | 1.64e4 | 1.00 | 20–1100 µm | 不一致 | 硬拒绝 |

任何仍浮动 `dx` 或 `ry`、并且超过 1.5–1.7 µm 带的 mode，立即拒绝。
不能靠更多事件或更松阈值挽救。

`{dy,rx,rz}` 在两个标定 run 上满秩、条件数稳定、与 `A` 明显正交。
这只是 Fisher 空间结果。self-nulling 点估计不能搬运（`dy` +1.08 vs
−8.93 mm，`rz` −4.3 vs +95 mrad）。把 14974 的修正应用到 14973，
线性 χ² 恶化约 16 倍。它 **不是** V2 candidate。

更小的 isolation-safe 子集（`{dy}` / `{rx}` / `{rz}` 及两两组合）
同样满秩且与 `A` 正交，但没有一个跨 run 一致。

## Common identifiable subspace

各轴与冻结 `A` 的余弦：

- `dx` 0.880，`ry` 0.474 — 泄漏方向
- `dy` 0.025，`rx` 0.030，`rz` 0.005 — isolation-safe

推荐的 Fisher 子空间是 `{dy,rx,rz}`。若它再通过跨 run 一致与
transfer，则 `dx`/`ry`/`dz` 交给 survey/external。它没有通过，
因此不定义可写的 reduced mode。

## 只读 transfer DQ

14975/14976/14977 没有 FD `J_s`，不施加任何候选几何。当前几何下，
14975/14976 的 residual IQR 与标定块重叠。14977 只有两条 selected
route，不足判断。14977 不参与判定。residual 下降不是 alignment
成功。

## 唯一判定

**`real_data_residual_dq_monitoring_only`**

- 不存在 Real-Data Station Calibration Mode V2 candidate
- 全部 station DoF，包括 Fisher 意义上 isolation-safe 的
  `{dy,rx,rz}`，都交给 survey/external alignment
- `geometry_write_allowed=false`
- 不写入任何 self-nulling 修正
- C_dx Mode 保持关闭

当前真实数据只能做 dedicated residual / DQ monitoring，不能产生
station geometry update。

## 产物

目录 `outputs/operating_protocol_v1_real_data_reduced_station_mode_feasibility_v1/`：

- `reduced_mode_identifiability_audit.json`
- `reduced_mode_leakage_audit.json`
- `reduced_mode_transfer_dq_report.json`
- `reduced_station_mode_feasibility_report.json`
- `operating_protocol_next_decision.json`
