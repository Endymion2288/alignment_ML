# 2026-08-22 (49) 真实数据 full-segment Station Mode self-nulling dry-run

## 任务

条目 48 的 100-event dry-run 因 selected_routes=0 停在 candidate-graph DQ。
条目 48 之后的 scaling study 证明：这主要是统计量不足，不是冻结 V2
association 失败。本条目在 **不改 V2 / threshold / unmatched penalty /
occupancy window** 的前提下，用 full remaining segment 的 selected routes
进入仅标定的 Station Mode self-nulling dry-run。

仍禁止写官方 conditions、禁止 C_dx Mode、禁止联合 Newton、禁止新 DoF。
residual 下降只是 DQ，不是 alignment 成功。

## 由 scaling 更新的结论

- 所有真实数据窗口 all-pairs candidate graph 非空
- n1000 / n10000 / full 上 selected routes 稳定出现
- 战役标签：`real_data_route_acceptance_statistics_limited`
- 完整四站路线仍稀（14973 full=0，14974 full=2），记为 DQ，不改 V2

## 本步已落地

冻结配置：
`configs/operating_protocol_v1_real_data_station_mode_self_nulling_fullscale_v1.yaml`

Provenance：
`outputs/operating_protocol_v1_real_data_station_mode_self_nulling_fullscale_v1/`

四份 JSON：

- `route_dq_fullscale_report.json`：路线完整率、边复用、事件集中度、
  run 间稳定性；冻结 candidate-graph DQ gate **通过**
- `cdx_leakage_diagnostic_report.json`：冻结 A 与 10% 门（MC 注册已过，
  真实数据无法重测 A）；Newton 前 implied `|C_dx|` 不可用
- `station_mode_self_nulling_report.json`
- `operating_protocol_next_decision.json`：最终
  `cross_level_contaminated`，`geometry_write_allowed=false`

Full-segment selected-route 摘要（判定只用 14973/14974）：

| run | 角色 | selected | 完整四站 | all-pairs | 单事件占比 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 14973 | 标定 | 121 | 0 | 70560 | 0.008 |
| 14974 | 标定 | 109 | 2 | 52626 | 0.009 |
| 14975 | holdout | 156 | 1 | 95288 | 0.006 |
| 14976 | holdout | 63 | 0 | 34786 | 0.016 |
| 14977 | held-out DQ | 2 | 0 | 1329 | 0.50 |

真实数据没有 MC 纯度。JSON 里用 `truth_free_complete_route_fraction`，
禁止 `purity` / `efficiency` / `auc` 键。

Selected-route residual 与 leave-one-out 只作 DQ observable。未因
residual 大小判定成功或失败。

## Station Mode 物理计划

只对标定 run 做 current + 12 个轴向 FD probe。current 点软链接复用
scaling study 的 full-segment 重建。Holdout / 14977 保持 current-only，
不提交 FD Athena。

Condor（eossubmit / `nextweek` / 12000 MB）：

| 内容 | cluster | 作业 | 结果 |
| --- | ---: | ---: | --- |
| 标定 full-segment FD | **1000840** | 2（14973, 14974） | 全部 return 0；各 13/13 点（current 复用 + 12 FD） |

14974 于 15:58 结束，14973 于 16:37 结束。current 点软链接复用 scaling
重建（63151 / 47014 tracklets）。标定 FD 点已物化 identity /
field-candidate，并用已有 current V2 selected routes 做 self-nulling
Newton（未重训 V2）。

Self-nulling 提议（只作估计器输出，不是正确性证据）：

| run | dx mm | dy mm | dz mm | ry mrad | implied `C_dx`(dx) µm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 14973 | −1.30 | −0.77 | 658 | 20.0 | −22 |
| 14974 | +6.57 | −3.21 | 1250 | 33.5 | +111 |

run 间 dx 差 7.87 mm。implied `|C_dx|` 远超 1.5–1.7 µm 工作带，因此战役
判定为 `cross_level_contaminated`。`dz` 不物理，说明短 route 主导的
selected graph 不能辨识 station 6-DoF。这不是把 residual 当成功，也不是
`C_dx` 测量。冻结 A 10% 门仍用 MC 注册结果（7.1% < 10%），真实数据未重测
A。C_dx Mode 不启动。`geometry_write_allowed=false`。官方 conditions 未写。
