# Operating Protocol V1 真实数据最终闭环与可复现冻结

Workbook 54 / 2026-08-23。条目 47–53 被整理成不可再调参的证据包。本阶段
**不再**打开 Station Mode、reduced Station Mode 或 `C_dx` Mode，不重训
V2，不改 occupancy / 报警 / `A` / 14973–14974 residual scale，也不从
self-nulling residual 提取 alignment payload。

## 证据链

`MC transfer PASS → real-data frozen-V2 acceptance statistics-limited but recovered at scale → full-segment Station calibration REJECTED by physical_nonidentifiability/cross_level_contamination → reduced-mode calibration REJECTED by cross-run non-transferability → current-geometry residual/DQ monitoring PASS on independent runs`

| 步 | 条目 | 结论 |
| ---: | --- | --- |
| 1 | 47 | MC transfer **PASS**。源不相交的 Station 与 IFT-Internal 均独立闭合。`A` 未更新。 |
| 2 | 48–49 | 冻结 V2 在 100-event 上 statistics-limited，在 full-segment 上恢复（121 / 109 / 156 / 63 / 2 条 selected route）。 |
| 3 | 49–50 | 全段 Station calibration **否决**（`cross_level_contaminated`，唯一类 `physical_nonidentifiability`）。 |
| 4 | 51 | reduced Station calibration 因跨 run 不可搬运 **否决**。`{dy,rx,rz}` Fisher 可识别且几乎与 `A` 正交，但 self-nulling correction 不能转移（最大约 258σ；14974→14973 线性化 χ² 约 ×16）。 |
| 5 | 52–53 | 当前 geometry residual/DQ monitoring **通过**。7 个独立 r0022 run 全部 `nominal_monitoring`。`alignment_drift_candidate=false`。 |

本包中任何 residual 或 χ² 下降都只是 **DQ observable**，不是 alignment
成功。

## 冻结不变量

以下内容不再调参：

- 传播：mode-0
- 关联：冻结 V2（SHA256
  `0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27`）
- 阈值 `0.001`，`unmatched_penalty=-1.0`
- 观测：`anchor_selected_field_edge` + `physical_edge_deduplicated`
- Occupancy：第一个满足 16/16/32/8 的 100-event 窗口；不够则下一 r0022
  segment。Occupancy 只找起点，正式监测用 full remaining segment
- Geometry / conditions / reconstruction：`FASERNU-04` /
  `OFLCOND-FASER-06` / `r0022`
- 真实数据 Athena：`--NoTrackFilt --no_stable`
- 泄漏算子 `A`：`A_dx=-59.213`，`A_ry=-31.908` µm / µm `C_dx`
- Isolation budget：1.5–1.7 µm
- 参考 residual scale（14973/14974 field-edge）：`dy` 中位数 −3.810 mm，
  robust scale 6.712 mm；`rx` 中位数 −0.00178，scale 0.00524。禁止重估
- 报警（条目 52，不重调）：
  1. `selected==0` 且 `all_pairs>=50` → 关联 / 重建退化
  2. `0<selected<10` → 统计不足（不是 alignment 异常）
  3. selected ≥ 10 且 event share > 0.50 → 关联 / 重建退化
  4. 相对 14973/14974 的 isolation `|robust z|>5` → 探测器工况变化
  5. 至少两个非参考、统计充分的 run 上同号 `|z|≥3` → 只标记
     `alignment_drift_candidate`，绝不写成改正

禁止：FD probe、Newton、alignment payload、重训 V2、改阈值 / penalty、
按 residual 重选窗口、把 drift candidate 转成 geometry、重开
self-nulling、打开密封测试。

## 最终机器可读状态

```
real_data_operating_mode=residual_dq_monitoring_only
geometry_write_allowed=false
station_calibration_mode_available=false
cdx_mode_allowed=false
alignment_drift_candidate=false
```

唯一允许的代码路径：

`当前官方 geometry → 冻结 V2 → residual/DQ monitoring`。

禁止从真实数据 self-nulling residual 生成任何 alignment payload。

## 解除 monitoring-only 的条件（Operating Protocol V2）

机器可读文件：
`outputs/operating_protocol_v1_final_real_data_closure_v1/operating_protocol_v2_unlock_criteria.json`

当前 `currently_met=false`。满足以下任一条件才允许另开 V2：

1. 独立 survey / 外部约束固定 `dx/ry/dz`，并且独立测量证明真实
   `|C_dx|` 落在 1.5–1.7 µm isolation budget 内
2. 新的独立真实 track topology 实证得到跨 run 可搬运的 calibration
   subspace

条件满足后也只允许打开 V2 协议。本证据包仍然不写 geometry。

## 论文 / 组会图表数据

目录：`outputs/operating_protocol_v1_final_real_data_closure_v1/chart_data/`

| 文件 | 内容 |
| --- | --- |
| `selected_route_scaling.csv` | selected-route 数随 event 统计的标度（含 expansion 全段） |
| `route_composition.csv` | 12 个 run 的 2/3/4 站 route 组成 |
| `dy_rx_robust_z_timeseries.csv` | `dy/rx` robust-z 相对 LHC fill 的时间序列 |
| `station_jacobian_singular_spectrum.csv` | 14973/14974 的 6-DoF / 5-DoF 奇异谱 |
| `reduced_mode_cross_run_inconsistency.csv` | `{dy,rx,rz}` 在 14973 与 14974 上的 self-nulling 差 |
| `frozen_A_versus_station_weak_direction.csv` | 冻结 `A` 与 station 弱方向（14974 五自由度余弦 0.994） |

图中凡 residual / χ² 下降，一律标 **DQ observable**。

## 输出

`outputs/operating_protocol_v1_final_real_data_closure_v1/`

- `operating_protocol_v1_final_report.json`
- `real_data_evidence_matrix.json`
- `reproducibility_manifest.json`
- `operating_protocol_v2_unlock_criteria.json`
- `chart_data/`
