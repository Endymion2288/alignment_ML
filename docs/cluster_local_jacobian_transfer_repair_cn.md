# Cluster-local Jacobian transfer exact-join 修复 V1

Workbook 70 / 2026-09-03。条目 68 跳过 cluster-local 诊断，原因
是 `cluster_local_reference_or_transfer_unavailable`：transfer
run 14974 的 Frozen-V2 selected routes 被 join 到条目 57 的
r14973 可行性 dump 上。该 dump 只有 run 14973，exact join 测量
数为 0。

这是纯 provenance 修复。它不是 identifiability 战役，也 **不是**
条目 69 的 stable-core 门。只回答：能否在不放宽 provenance 的
前提下恢复 cluster-local transfer join。

不重新训练 V2/V3/Transformer，不改冻结的 pairwise/route policy，
不继续堆 2024 r0022 collision-like tracks，不发明 cosine cut，不
Newton，不写 geometry，不产生 alignment payload。真实数据仍是
`residual_dq_monitoring_only`。恢复 join **不** 重开条目 58 的
`cluster_local_identifiability_not_transferable`。

## 冻结合同

- 真实数据仍是 `residual_dq_monitoring_only`。
- `geometry_write_allowed=false`。
- 冻结 V2 checkpoint SHA256
  `0c85a001677678163512c59bd5ceb6d08bdefaab79c0f4628659a4a8ad766a27`。
- join key 保持
  `run + event + TrackletHit_cluster_identifier =
  FaserSCT_Cluster.identify().get_compact()`。
- 严禁模糊匹配、按 residual / position 最近邻 join、或重选
  population 来强迫非零匹配。
- 接线已有
  `outputs/true_cluster_local_ry_cdx_stability_transfer_v1/dumps/cluster_local_r14974.root`，
  不发明新 key。
- 保留条目 68 失败模式作为负控制：r14974 routes × r14973 dump
  必须仍为 0。
- 恢复 exact join 不授权新的 cluster-local observable
  identifiability 战役，也不得并入 stable-core 门。

## 判定

`cluster_local_transfer_exact_join_restored_not_an_identifiability_campaign`

| 门 | 值 |
| --- | --- |
| exact join restored | **true** |
| fuzzy / nearest-neighbour join | **false** |
| wrong-dump control 仍为 0 | **true** |
| 条目 58 identifiability 重开 | **false** |
| 授权 cluster-local identifiability 战役 | **false** |
| 并入 stable-core gate | **false** |
| Frozen-V2 unknown-association | **未授权** |
| 真实数据 alignment correction | **未授权** |
| geometry write | **false** |

条目 68 的零匹配被复现为 run-id 不一致，不是 join key 坏了。把
已有 r14974 dump 接上后，exact join 完整非零。这只恢复了
provenance。

## Provenance 分层

配置
`configs/cluster_local_jacobian_transfer_repair_v1.yaml`，SHA256
`c761af1fa3cba38ee88c8c5e80cf819557483b65fe9d397d21be0d340e433e97`。

| 层 | reference r14973 | transfer r14974 |
| --- | --- | --- |
| role | calibration_reference | independent_calibration_transfer |
| source_id | `data24_r14973_00007_skip49500_n84988` | `data24_r14974_00005_skip74400_n60886` |
| reconstruction | r0022 | r0022 |
| cluster dump | `true_cluster_local_residual_feasibility_v1/cluster_local.root` | `true_cluster_local_ry_cdx_stability_transfer_v1/dumps/cluster_local_r14974.root` |
| dump runs | `{14973: 4073}` | `{14974: 3843}` |
| selected routes | 121 | 109 |
| wanted cluster keys | 2473 | 2123 |
| exact joined keys | **2473 / 2473** | **2123 / 2123** |
| measurements | **454** | **403** |
| missing keys | 0 | 0 |

负控制（条目 68 失败模式）：r14974 routes × r14973 dump 的 run
overlap 为 0，joined keys 0，measurements 0，
`mismatch_reason = run_id_disjoint_dump_does_not_contain_route_run`。

Join key 未改。`fuzzy_join_used=false`，
`nearest_neighbour_join_used=false`，
`population_reselected=false`。Population 仍是条目 57–58 的
Frozen-V2 selected-route 集合。

## 命令

```bash
source scripts/setup_environment.sh ml
pytest -q tests/test_cluster_local_jacobian_transfer_repair.py
python scripts/report_cluster_local_jacobian_transfer_repair.py
```

CUDA `True`（Tesla T4）。相关测试包含在 26 passed 中。没有
Condor。

报告：`outputs/cluster_local_jacobian_transfer_repair_v1/`。
记录 HEAD `a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1`。

## 下一阶段

保持 residual DQ monitoring。Exact join 可以在不放宽 provenance
的前提下恢复。这不是 identifiability 成功，也不是 stable-core
成功。未来若研究 cluster-local observable identifiability，必须
新开预注册 config 和 workbook，并继续只用 exact join。
