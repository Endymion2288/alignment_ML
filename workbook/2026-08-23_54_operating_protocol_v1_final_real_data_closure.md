# 2026-08-23 (54) Operating Protocol V1 真实数据闭环与可复现冻结

## 任务

条目 53 已经完成当前真实数据阶段最重要的闭环：冻结 V2 在
full-segment 统计下稳定工作，新增 7 个独立 r0022 run 全部通过同一套
未调参 monitoring protocol；station geometry calibration 因
cross-level leakage、physical non-identifiability 和跨 run 不可搬运性
被明确否决。

本条目不再扩展任何 alignment solve，也不重新打开 Station Mode、
reduced Station Mode 或 `C_dx` Mode。只把条目 47–53 整理成不可再调参的
**Operating Protocol V1 evidence package**。

## 冻结对象

- 配置：`configs/operating_protocol_v1_final_real_data_closure_v1.yaml`
- V2 checkpoint SHA256
  `0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27`
- occupancy 规则：first 100-event window，minima 16/16/32/8
- 14973/14974 reference residual scale（`dy` median −3.810 mm，robust
  scale 6.712 mm；`rx` median −0.00178，scale 0.00524）
- 报警门限：selected=0 且 all-pairs≥50；0<selected<10；event share>0.50；
  isolation `|z|>5`；drift `|z|≥3` 且至少两个非参考 run
- mode-validity contract 与冻结 `A`（`A_dx=-59.213`，`A_ry=-31.908`）
- 全部输入 run provenance（xAOD / segment / skip / LHC fill）

代码：

- `alignment/operating_protocol_v1_final_closure.py`
- `scripts/report_operating_protocol_v1_final_closure.py`
- `tests/test_operating_protocol_v1_final_closure.py`（7 项通过）

报告脚本只读取 47–53 已有产物，不生成 FD probe，不求 Newton，不写
payload，不重训 V2。

## 证据链

`MC transfer PASS → real-data frozen-V2 acceptance statistics-limited but recovered at scale → full-segment Station calibration REJECTED by physical_nonidentifiability/cross_level_contamination → reduced-mode calibration REJECTED by cross-run non-transferability → current-geometry residual/DQ monitoring PASS on independent runs`

| 步 | 条目 | 结论 |
| ---: | --- | --- |
| 1 | 47 | MC transfer PASS；A 未更新 |
| 2 | 48–49 | 100-event 空图是 statistics-limited；full-segment 恢复 121/109/156/63/2 |
| 3 | 49–50 | 全段 Station **REJECTED**：`cross_level_contaminated` / `physical_nonidentifiability` |
| 4 | 51 | reduced mode **REJECTED**：`{dy,rx,rz}` 可识别但跨 run 不可搬运（最大 ~258σ） |
| 5 | 52–53 | 当前 geometry residual/DQ monitoring **PASS**；7 个独立 run 全 nominal |

任何 residual decrease 只能标为 **DQ observable**，不能当作 alignment
成功。

## 最终机器可读状态

- `real_data_operating_mode=residual_dq_monitoring_only`
- `geometry_write_allowed=false`
- `station_calibration_mode_available=false`
- `cdx_mode_allowed=false`
- `alignment_drift_candidate=false`
- 代码路径：`current official geometry → frozen V2 → residual/DQ monitoring`
- 禁止从真实数据 self-nulling residual 生成任何 alignment payload

## Unlock criteria（Operating Protocol V2）

当前 `currently_met=false`。只有满足以下任一条件才允许另开 V2：

1. 独立 survey / 外部约束固定 `dx/ry/dz`，并且证明真实 `|C_dx|` 落在
   1.5–1.7 µm isolation budget 内
2. 新的独立真实 track topology 实证得到跨 run transferable calibration
   subspace

满足后也只是允许打开 V2 协议，本证据包仍然不写 geometry。

## 组会 / 论文图表数据

输出根：
`outputs/operating_protocol_v1_final_real_data_closure_v1/`

- `operating_protocol_v1_final_report.json`
- `real_data_evidence_matrix.json`
- `reproducibility_manifest.json`
- `operating_protocol_v2_unlock_criteria.json`
- `chart_data/selected_route_scaling.csv`
- `chart_data/route_composition.csv`
- `chart_data/dy_rx_robust_z_timeseries.csv`
- `chart_data/station_jacobian_singular_spectrum.csv`
- `chart_data/reduced_mode_cross_run_inconsistency.csv`
- `chart_data/frozen_A_versus_station_weak_direction.csv`

图中若出现 residual / χ² 下降，一律标 **DQ observable**。
