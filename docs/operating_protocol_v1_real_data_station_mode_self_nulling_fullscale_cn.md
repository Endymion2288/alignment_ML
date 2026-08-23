# Operating Protocol V1 真实数据 Station Mode self-nulling dry-run（full segment）

条目 49 / 2026-08-22。本步在 full remaining occupancy segment 上启动
**仅标定 run 的 Station Mode self-nulling**。仍是 transfer / 数据质量验证，
**不写入**官方 conditions 数据库。

## 由 scaling study 更新的结论

冻结 100-event 窗口上的空 selected graph 是 **统计量不足**，不是冻结 V2
association 失败。所有真实数据窗口的 all-pairs candidate graph 都非空。
在 `n1000`、`n10000` 和 full remaining segment 上，selected routes 已稳定出现。

冻结 V2、route policy、分数阈值 `0.001` 与 `unmatched_penalty=-1.0` 均未改。

## 冻结不变量

- 传播：mode-0
- association：冻结 V2
  （`outputs/mc24_v3_expanded_trainval_v2_bce_control_v1`，SHA256
  `0c85a001…766a27`）
- 观测：`anchor_selected_field_edge` + `physical_edge_deduplicated`
- Station Mode：5-DoF + `dz=5 mm` survey prior，写入 `dz=0`
- 全部 capture、mode-validity contract、冻结泄漏算子 `A` 与 10% 稳定性门
- 盲角色：`14973/14974` 标定，`14975/14976` holdout，`14977` held-out DQ
  （永不参与选择）

禁止：重训 V2；改阈值或 unmatched penalty；联合 station+`C_dx` Newton；
新增 layer/module DoF；启动 C_dx Mode；把 residual 下降当成正确
alignment；写官方 conditions。

## DQ 与 alignment 正确性必须分开

| 产物 | 角色 |
| --- | --- |
| 路线完整率、边复用、事件集中度、run 间重叠 | 只是数据质量 |
| selected route 的 station residual 与 leave-one-out closure | 只是数据质量 |
| 由冻结 `A` 与未来 Newton `dx/ry` 推出的 implied `|C_dx|` | 污染诊断，**不是** `C_dx` 测量 |
| self-nulling 后 residual 下降 | 永远不是 alignment 成功 |

在 mode-validity contract、冻结 A 10% 门和 Station Mode capture 全部通过、
**并且**存在独立 `C_dx` 证据之前，`geometry_write_allowed` 保持 false。
本阶段真实数据还不具备最后一条。

## Full-segment selected-route DQ

标定 selected graph 非空，且不是单事件主导：

| run | 角色 | selected routes | 完整四站 | all-pairs |
| ---: | --- | ---: | ---: | ---: |
| 14973 | calibration | 121 | 0 | 70560 |
| 14974 | calibration | 109 | 2 | 52626 |
| 14975 | holdout | 156 | 1 | 95288 |
| 14976 | holdout | 63 | 0 | 34786 |
| 14977 | held-out DQ | 2 | 0 | 1329 |

真实数据没有 MC 标签，因此用 `truth_free_complete_route_fraction` 作为
所请求的路线质量审计。完整四站路线仍然很少。这记为 DQ，不是改 V2 的理由。

## Station Mode capture 计划

只有标定 run 做 current + 12 个轴向有限差分 probe。current 几何重建复用
scaling study。Holdout 与 14977 保持 current-only，不估计几何。

Condor 集群 **1000840**（2 个作业，`nextweek`，12000 MB，eossubmit）。

## 输出

目录：`outputs/operating_protocol_v1_real_data_station_mode_self_nulling_fullscale_v1/`

- `route_dq_fullscale_report.json`
- `cdx_leakage_diagnostic_report.json`
- `station_mode_self_nulling_report.json`
- `operating_protocol_next_decision.json`

## Self-nulling Newton（仅标定）

Condor **1000840** 两个作业均 return 0（14974 15:58，14973 16:37）。
12 个 FD probe 已物化 identity / field-candidate。Self-nulling Gauss-Newton
使用 **已有** full-segment current V2 selected routes；V2 未重跑、未改阈值。

| run | dx mm | ry mrad | 由 dx 推出的 `C_dx` (µm) | 由 ry 推出的 `C_dx` (µm) |
| ---: | ---: | ---: | ---: | ---: |
| 14973 | −1.30 | 20.0 | −22 | 628 |
| 14974 | +6.57 | 33.5 | +111 | 1049 |

run 间 `dx` 差 7.87 mm。两个 implied `|C_dx|` 都超过冻结的 1.5–1.7 µm
工作带。`dz` 提议（658 mm / 1250 mm）不物理；selected graph 仍几乎全是
短 route，6-DoF station 解不可辨识。这些数字是 **DQ / 污染诊断**，
不是可写入几何，也不是 `C_dx` 测量。

冻结 MC `A` 的 10% 门在注册时已过（相对散度 7.1% < 10%）。真实数据
没有独立 `C_dx` 变化，不能重测 `A`。

## 唯一下一步判定

`cross_level_contaminated`

- `geometry_write_allowed=false`
- C_dx Mode blocked
- 官方 conditions 未改
- residual 下降不是 alignment 成功
