# Operating Protocol V1 真实数据 residual/DQ monitoring expansion

条目 53 / 2026-08-23。条目 52 的
`real_data_residual_dq_monitoring_only` 保持冻结。本步 **不** 打开
Station 或 C_dx solve。把已经冻结的 current-geometry monitoring chain
应用到预先列出的独立 2024 r0022 run：14971、14972、14980、14981、
14985、14989、15007。

## 冻结不变量

- 传播：mode-0
- association：冻结 V2（`0c85a001…766a27`）
- 阈值 `0.001`，`unmatched_penalty=-1.0`
- 观测：`anchor_selected_field_edge` + `physical_edge_deduplicated`
- occupancy 规则：第一个满足 16/16/32/8 的 100-event 窗口；不够则下一
  r0022 segment。occupancy 只找起点。正式监测用 full remaining segment
- calibration reference：条目 52 的 14973/14974 scale，不重估
- 报警：条目 52 的优先级，不回改
- `geometry_write_allowed=false`
- `station_calibration_mode_available=false`
- `cdx_mode_allowed=false`

禁止：FD probe；Newton；alignment payload；重训 V2；改 threshold 或
penalty；按 residual 重选窗口；把 `alignment_drift_candidate` 转成
geometry update；重新打开 self-nulling calibration。

## Occupancy

Condor **1001193** 扫了七个 run 的 `00000–00007`。14972/`00003` 撞上
EOS mkdir 竞态，与 14985 的 `00008–00015` 一起在 **1001194** 补扫。
七个窗口全部冻结，阈值未降。

| run | LHC fill | segment | skip | remaining events |
| ---: | ---: | --- | ---: | ---: |
| 14971 | 9564 | 00003 | 20900 | 112357 |
| 14972 | 9565 | 00004 | 23300 | 110071 |
| 14980 | 9573 | 00005 | 57300 | 77621 |
| 14981 | 9574 | 00006 | 95600 | 40646 |
| 14985 | 9575 | 00012 | 95300 | 40837 |
| 14989 | 9579 | 00005 | 71900 | 63559 |
| 15007 | 9585 | 00005 | 111800 | 24905 |

## Current-geometry Athena 与冻结 V2

Condor **1001195** 在官方几何上重建了七个 full remaining segment
（`--NoTrackFilt --no_stable`，只有 current 一点，无 FD）。冻结 V2
association 在本地 GPU 完成。

| run | events | tracklets | all-pairs | selected | 2/3/4-st | `dy` z | `rx` z | 状态 |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| 14971 | 112357 | 88562 | 99021 | 233 | 78/151/4 | −0.055 | −0.008 | nominal |
| 14972 | 110071 | 84697 | 94951 | 193 | 72/119/2 | 0.048 | 0.034 | nominal |
| 14980 | 77621 | 55978 | 62395 | 114 | 43/69/2 | −0.175 | −0.046 | nominal |
| 14981 | 40646 | 28678 | 31787 | 77 | 31/46/0 | −0.049 | 0.056 | nominal |
| 14985 | 40837 | 29762 | 33246 | 74 | 20/53/1 | −0.055 | 0.035 | nominal |
| 14989 | 63559 | 44454 | 49757 | 91 | 20/66/5 | 0.089 | 0.046 | nominal |
| 15007 | 24905 | 18387 | 20570 | 32 | 13/19/0 | 0.050 | 0.014 | nominal |

父语料 14973–14976 仍是 nominal。14977 仍是
`insufficient_statistics_for_alignment_dq`。时间顺序是 LHC fill → run
→ skip，不是 run-number proxy。本语料里 fill 与 run 同序。相邻且统计
充分的 run 没有同方向 isolation `|z| ≥ 3`。
`alignment_drift_candidate=false`。`dx`/`ry` 仍只是
cross-level-sensitive 辅助观测。`dz` 不是 track-driven observable。
`rz` 未伪造。

## 唯一结论

**`real_data_residual_dq_monitoring_only`** 保持冻结。

七个新增 run 全部 `nominal_monitoring`。当前 official geometry + 冻结
V2 可以承担长期 DQ monitoring。没有连续 detector-condition-change，也
没有 alignment-drift candidate，因此这不是 survey / external alignment
触发，也不是 self-nulling 重开。

`geometry_write_allowed=false`，
`station_calibration_mode_available=false`，
`cdx_mode_allowed=false`。

## 输出

`outputs/operating_protocol_v1_real_data_residual_dq_monitoring_expansion_v1/`

- `expansion_provenance.json`
- `run_level_dq_report.json`
- `time_stability_report.json`
- `alarm_summary.json`
- `operating_protocol_monitoring_decision.json`
