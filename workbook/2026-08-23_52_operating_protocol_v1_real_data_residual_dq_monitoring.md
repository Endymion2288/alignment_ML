# 2026-08-23 (52) 真实数据 residual/DQ monitoring-only 协议冻结

## 任务

条目 51 已经给出足够强的结论：`{dy,rx,rz}` 在 Fisher 空间上满秩、
条件数良好且与冻结 `A` 基本正交，但 self-nulling correction 在
14973/14974 之间严重不一致，不能跨 run transfer。当前真实数据不能
支持可搬运的 station geometry update。

本条目把该结论冻结成 production-like DQ protocol。不再构造任何新的
Station calibration mode，不启动 `C_dx` Mode，不生成 FD probe，不求
Newton，不写 payload。继续使用完全冻结的 mode-0、V2 checkpoint、
route policy、threshold、unmatched penalty、
`physical_edge_deduplicated` 和当前官方 geometry，对已有 2024 r0022
full-segment association 做 **current-geometry only** 长期稳定性扫描。

## 本步已落地

配置：
`configs/operating_protocol_v1_real_data_residual_dq_monitoring_v1.yaml`

代码：
`alignment/real_data_residual_dq_monitoring.py`
`scripts/report_real_data_residual_dq_monitoring.py`
`tests/test_real_data_residual_dq_monitoring.py`（9 项通过）

Provenance：
`outputs/operating_protocol_v1_real_data_residual_dq_monitoring_v1/`

- `run_level_dq_report.json`
- `time_stability_report.json`
- `alignment_drift_candidate_report.json`
- `operating_protocol_monitoring_decision.json`

本步复用条目 48–49 的 current-geometry frozen-V2 association，无需
Condor。邻近 r0022 run（14971/14972/14980/14981/14985/14989/15007）
只列为 expansion candidates：必须先走 residual-blind occupancy +
current-only Athena + 冻结 V2，才能进入同一套 monitoring 报告。

## 冻结报警规则

优先级固定，禁止事后改阈值：

1. `selected_routes==0` 且 all-pairs ≥ 50 →
   `association_or_reconstruction_degradation`
2. `0 < selected_routes < 10` →
   `insufficient_statistics_for_alignment_dq`，**不是** alignment 异常
3. selected ≥ 10 且单事件占比 > 0.50 →
   `association_or_reconstruction_degradation`
4. isolation 通道相对 14973/14974 的 |robust z| > 5 →
   `detector_condition_change`
5. 否则 `nominal_monitoring`

`dy/rx` 才是 isolation residual observables。`dx/ry` 额外标记
`cross_level_sensitive`。`dz` 完全不是 track-driven observable。
`rz` 没有独立 4-vector residual 通道，不伪造。即使 `dy/rx` 出现可重复
长期漂移，也只能标 `alignment_drift_candidate`，禁止转成 geometry
correction。

## 五个 full-segment run

| run | 角色 | selected | 2/3/4-st | 最大事件占比 | dy median [mm] | dy z | rx z | 状态 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 14973 | calibration_reference | 121 | 38/83/0 | 0.008 | −3.83 | −0.00 | −0.02 | nominal |
| 14974 | calibration_reference | 109 | 34/73/2 | 0.009 | −3.81 | 0.00 | 0.03 | nominal |
| 14975 | monitoring | 156 | 60/95/1 | 0.006 | −4.66 | −0.13 | 0.01 | nominal |
| 14976 | monitoring | 63 | 18/45/0 | 0.016 | −4.27 | −0.07 | 0.03 | nominal |
| 14977 | monitoring | 2 | 0/2/0 | 0.50 | −4.24 | −0.06 | −0.14 | insufficient |

14975/14976 的 isolation |z| 远低于 3，更低于 detector-condition 阈值 5。
Theil–Sen 斜率（run 号作时间代理）`dy ≈ −0.12 mm/run`、
`rx ≈ −1.9×10⁻⁵ /run`，相对 reference robust scale
（`dy` 6.71 mm，`rx` 0.00524）不是可重复长期漂移。
`alignment_drift_candidate=false`。14977 自动归为
`insufficient_statistics_for_alignment_dq`。

`dx/ry` residual 继续报告，但一律 `cross_level_sensitive`，不参与
drift candidate，也不触发 detector-condition。

## 唯一结论

**`real_data_residual_dq_monitoring_only`**

- `geometry_write_allowed=false`
- `station_calibration_mode_available=false`
- `cdx_mode_allowed=false`
- 不从当前 self-nulling residual 提取任何 alignment payload

未来重新开启 geometry calibration 的外部前提（满足任一即可，当前均未满足）：

1. 独立 survey / external station constraints 能固定 `dx/ry/dz`，并且
   证明 `C_dx` 落在当前 1.5–1.7 µm isolation budget 内；或
2. 提供新的独立真实数据 topology / track sample，使跨 run transferable
   subspace 得到实证。

在这些条件满足前，只做 current-geometry residual/DQ monitoring。
