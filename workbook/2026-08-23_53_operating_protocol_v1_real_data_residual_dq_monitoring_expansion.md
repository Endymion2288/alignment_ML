# 2026-08-23 (53) 真实数据 residual/DQ monitoring expansion V1

## 任务

条目 52 的 `real_data_residual_dq_monitoring_only` 保持冻结。不再构造
Station / C_dx solve，不生成 FD probe，不求 Newton，不写 alignment
payload，不重训 V2，不改 threshold / unmatched penalty，也不根据 residual
重选 occupancy 窗口。

对预先列出的独立 2024 r0022 run
`14971/14972/14980/14981/14985/14989/15007` 执行与条目 48/52 完全相同的
current-geometry monitoring chain：residual-blind occupancy 只找有数据的
起点，正式监测使用该 segment 从冻结 skip 起的 full remaining segment。

## Occupancy

规则未改：`configs/operating_protocol_v1_real_data_occupancy_window_rule.yaml`
（first 100-event window，minima 16/16/32/8，不够则下一 r0022 segment）。

Condor：

| 波 | cluster | 作业 | 结果 |
| --- | ---: | ---: | --- |
| 00000–00007 七 run | **1001193** | 56 | 55 完成；14972/00003 因 EOS mkdir 竞态失败 |
| 补扫 14972/00003 + 14985/00008–00015 | **1001194** | 18 | 完成；七窗全部冻结 |

| run | fill | segment | skip | remaining | 100-event 四站/tracklets |
| ---: | ---: | --- | ---: | ---: | --- |
| 14971 | 9564 | 00003 | 20900 | 112357 | 9 / 58 |
| 14972 | 9565 | 00004 | 23300 | 110071 | 13 / 56 |
| 14980 | 9573 | 00005 | 57300 | 77621 | 10 / 55 |
| 14981 | 9574 | 00006 | 95600 | 40646 | 9 / 58 |
| 14985 | 9575 | 00012 | 95300 | 40837 | 8 / 58 |
| 14989 | 9579 | 00005 | 71900 | 63559 | 10 / 71 |
| 15007 | 9585 | 00005 | 111800 | 24905 | 8 / 47 |

## Athena + 冻结 V2

Current official geometry only。`--NoTrackFilt --no_stable`。一点 current，
无 FD。Condor **1001195** 七作业全部完成。本地 GPU 冻结 V2 association
全部完成。reference scale 与报警门限未重估（`dy` robust scale 仍为
6.712 mm）。

| run | n_events | n_tracklets | all-pairs | selected | 2/3/4-st | dy z | rx z | 状态 |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| 14971 | 112357 | 88562 | 99021 | 233 | 78/151/4 | −0.055 | −0.008 | nominal |
| 14972 | 110071 | 84697 | 94951 | 193 | 72/119/2 | 0.048 | 0.034 | nominal |
| 14980 | 77621 | 55978 | 62395 | 114 | 43/69/2 | −0.175 | −0.046 | nominal |
| 14981 | 40646 | 28678 | 31787 | 77 | 31/46/0 | −0.049 | 0.056 | nominal |
| 14985 | 40837 | 29762 | 33246 | 74 | 20/53/1 | −0.055 | 0.035 | nominal |
| 14989 | 63559 | 44454 | 49757 | 91 | 20/66/5 | 0.089 | 0.046 | nominal |
| 15007 | 24905 | 18387 | 20570 | 32 | 13/19/0 | 0.050 | 0.014 | nominal |

父语料 14973–14976 仍为 nominal；14977 仍为
`insufficient_statistics_for_alignment_dq`。时间顺序按 LHC fill → run →
skip，不是 run-number proxy。本语料 fill 与 run 同序。相邻且统计充分的
run 上没有同方向 `|z|≥3` 的 `dy/rx` 漂移。
`alignment_drift_candidate=false`。`dx/ry` 只作 cross-level-sensitive
辅助观测。`dz` 不进入 track-driven DQ。`rz` 未伪造。

## 唯一结论

**`real_data_residual_dq_monitoring_only`** 保持冻结。

- 七个新增 run 全部 `nominal_monitoring`
- 正式证明当前 official geometry + 冻结 V2 可以承担长期 DQ monitoring
- 没有连续 detector-condition-change，也没有 alignment-drift candidate
- `geometry_write_allowed=false`
- `station_calibration_mode_available=false`
- `cdx_mode_allowed=false`
- 不重新打开 self-nulling calibration

输出：
`outputs/operating_protocol_v1_real_data_residual_dq_monitoring_expansion_v1/`

- `expansion_provenance.json`
- `run_level_dq_report.json`
- `time_stability_report.json`
- `alarm_summary.json`
- `operating_protocol_monitoring_decision.json`

测试 `tests/test_real_data_residual_dq_monitoring_expansion.py` 7 项通过。
