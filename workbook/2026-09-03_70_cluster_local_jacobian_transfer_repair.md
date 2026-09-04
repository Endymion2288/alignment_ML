# 2026-09-03 Cluster-local Jacobian transfer exact-join 修复 V1

## 任务

条目 68 跳过了 cluster-local 诊断：
`cluster_local_reference_or_transfer_unavailable`。原因是
transfer run 14974 的 Frozen-V2 selected routes 被 join 到条目
57 的 r14973 可行性 dump 上，dump 里只有 run 14973，exact join
测量数为 0。

本阶段是并行的 **纯 provenance 支线**，不是 identifiability
战役，也 **不是** 条目 69 stable-core gate。只回答：能否在不
放宽 provenance 的前提下恢复 cluster-local transfer exact join。

合同：

- join key 保持
  `run + event + TrackletHit_cluster_identifier =
  FaserSCT_Cluster.identify().get_compact()`
- 严禁模糊匹配、按 residual / position 最近邻强行 join、或
  重选 population 来得到非零匹配
- 使用条目 58 已经产出的
  `outputs/true_cluster_local_ry_cdx_stability_transfer_v1/dumps/cluster_local_r14974.root`，
  不发明新 key
- 保留 workbook-68 失败模式作为负控制：r14974 routes ×
  r14973 dump 必须仍为 0
- 恢复 exact join **不** 重开条目 58 的
  `cluster_local_identifiability_not_transferable`
- 不授权新的 cluster-local observable identifiability 战役

不训练、不改 pairwise/route、不 Newton、不写 geometry。真实
数据仍是 `residual_dq_monitoring_only`。

## 仓库状态

```text
branch: master
HEAD:   a1fd01b Add survey/metrology IOV provenance and prior-interface feasibility chain (60-65).
ancestor check: a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1 ⊂ HEAD
host:   lxplus909.cern.ch
CUDA:   True (Tesla T4)
```

Population 仍是条目 57–58 的 Frozen-V2 selected-route 集合，
未重选。Reconstruction 仍是 2024 r0022。未提交 Condor。

## 答案

冻结
`cluster_local_transfer_exact_join_restored_not_an_identifiability_campaign`。

| 门 | 冻结值 |
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

Workbook-68 的零匹配被复现为 run-id 不一致，不是 join key
坏了。把 transfer dump 接到已有的 r14974 文件后，exact join
非零且完整。这只恢复了 provenance，不改变条目 58 的
identifiability 否定结论，也不构成条目 69 的独立验证。

## Provenance 分层

配置
`configs/cluster_local_jacobian_transfer_repair_v1.yaml`，
SHA256
`c761af1fa3cba38ee88c8c5e80cf819557483b65fe9d397d21be0d340e433e97`。

| 层 | reference r14973 | transfer r14974 |
| --- | --- | --- |
| role | calibration_reference | independent_calibration_transfer |
| source_id | `data24_r14973_00007_skip49500_n84988` | `data24_r14974_00005_skip74400_n60886` |
| reconstruction | r0022 | r0022 |
| input xAOD | `/eos/experiment/faser/rec/2024/r0022/014973/...00007...` | `/eos/experiment/faser/rec/2024/r0022/014974/...00005...` |
| cluster dump | `true_cluster_local_residual_feasibility_v1/cluster_local.root` | `true_cluster_local_ry_cdx_stability_transfer_v1/dumps/cluster_local_r14974.root` |
| dump runs | `{14973: 4073}` | `{14974: 3843}` |
| selected routes | 121 | 109 |
| wanted cluster keys | 2473 | 2123 |
| exact joined keys | **2473 / 2473** | **2123 / 2123** |
| measurements | **454** | **403** |
| missing keys | 0 | 0 |
| event overlap | 121 | 109 |

负控制（workbook-68 失败模式）：r14974 routes × r14973 dump

- run overlap 0，event overlap 0
- wanted 2123 keys，joined 0
- measurements 0
- `mismatch_reason = run_id_disjoint_dump_does_not_contain_route_run`

Join key 未改。`fuzzy_join_used=false`，
`nearest_neighbour_join_used=false`，
`population_reselected=false`。

## 输入 / 命令

```bash
source scripts/setup_environment.sh ml
pytest -q tests/test_cluster_local_jacobian_transfer_repair.py
python scripts/report_cluster_local_jacobian_transfer_repair.py
```

CUDA `True`（Tesla T4）。相关测试包含在 26 passed 中。没有
Condor。

## 数值结果（artifact）

| 量 | 值 | 用途 |
| --- | ---: | --- |
| reference measurements | 454 | exact join 完整 |
| transfer measurements | 403 | exact join 完整 |
| wrong-dump measurements | 0 | 复现条目 68 失败 |
| identifiability 战役授权 | false | 合同 |
| stable-core gate | false | 合同 |
| geometry write | false | 合同 |

报告：`outputs/cluster_local_jacobian_transfer_repair_v1/`。
配置 SHA256
`c761af1fa3cba38ee88c8c5e80cf819557483b65fe9d397d21be0d340e433e97`。
记录 HEAD `a1fd01bb5c2c5daf432c13db3b021feb42bdf6d1`。

## 未解除 / 明确不做

- 条目 58
  `cluster_local_identifiability_not_transferable` 继续冻结。
  恢复 join 只证明 dump 接对了，不证明 ry / `C_dx` 可跨 run
  搬运。
- 不得把 454 / 403 条测量偷偷并进条目 69 的独立验证。
- 不得用模糊匹配或最近邻“救”未来任何零 join。
- 未来若开 cluster-local observable identifiability，必须新
  config、新 workbook，并继续 exact join。
- 不对真实数据求 correction，不写 geometry。

## 冻结结论

- Frozen V2 / association policy 不变。
- 真实数据继续 `residual_dq_monitoring_only`。
- `geometry_write_allowed=false`；无 alignment payload。
- Cluster-local transfer exact join **可以在不放宽
  provenance 的前提下恢复**。Workbook-68 的零匹配是 dump
  接错 run，不是 key 失效。
- 这不是 identifiability 成功，也不是 stable-core 成功。
- 下一步：保持 DQ monitoring。只有另开预注册战役才允许把
  该 join 用于 cluster-local observable 研究。
