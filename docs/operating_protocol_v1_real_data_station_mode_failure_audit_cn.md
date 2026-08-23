# Operating Protocol V1 真实数据 Station Mode 失败审计

条目 50 / 2026-08-22。本步刻画已经观测到的
`cross_level_contaminated` 结果。**不写入**官方 conditions，**不启动**
C_dx Mode。

100-event 空 selected graph 是统计量不足。full-segment selected
routes 已经存在。剩下的失败是：当前 selected graph 无法同时满足
station 可辨识性与 cross-level isolation。

冻结 V2、route policy、分数阈值 `0.001` 与 `unmatched_penalty=-1.0`
均未改。

## 冻结不变量

- 传播：mode-0
- association：冻结 V2
  （`outputs/mc24_v3_expanded_trainval_v2_bce_control_v1`，SHA256
  `0c85a001…766a27`）
- 观测：`anchor_selected_field_edge` + `physical_edge_deduplicated`
- 泄漏算子 `A` 与 1.5–1.7 µm isolation budget 仍是已注册的 MC 合同
- 盲角色：判定只用 `14973/14974` 标定

禁止：重训 V2；改阈值或 unmatched penalty；联合 station+`C_dx`
Newton；新增 layer/module DoF；启动 C_dx Mode；使用 Schur 生产估计器；
把 residual 下降当成 closure；把 implied `|C_dx|` 反演成新 payload；
写官方 conditions；打开 sealed test。

本步是对已捕获 FD probe 的本地分析，无需 Condor。

## DQ 与 alignment 正确性必须分开

| 产物 | 角色 |
| --- | --- |
| selected-route 数量、完整四站比例、边复用、事件集中度 | 只是数据质量 |
| `J_s` 的 rank、condition、奇异谱 | 可辨识性诊断，**不是** closure |
| 由冻结 `A` 与 self-nulling `dx/ry` 推出的 implied `|C_dx|` | 拒绝诊断，**不是** `C_dx` 测量 |
| self-nulling 后 residual 下降 | 永远不是 alignment 成功 |

`geometry_write_allowed` 保持 false。

## 1. Identifiability 审计

`J_s` 由已经捕获的 12 个轴向 station FD probe 在冻结 selected-route
边上重建。不把 residual 改善当成分数。

| run | 边数 | 6-DoF rank | 6-DoF cond. | 近零模 | 5-DoF cond.（去掉 survey `dz`） |
| ---: | ---: | ---: | ---: | --- | ---: |
| 14973 | 118 | 6 | 4.90e5 | `dz`（\|v\|=1.000） | 114 |
| 14974 | 104 | 6 | 2.75e6 | `dz`（\|v\|=1.000） | 1.64e4 |

两个标定 run 的 6-DoF 近零模都是 survey `dz`。`dx`、`dy`、`ry` 不是
6-DoF 最小方向。去掉 `dz` 后，14973 已低于 MC 准入门 `1e4`。14974
剩下的 5-DoF 弱方向几乎是纯 `dx`（\|v\|=0.993），与冻结 `A` 的余弦为
0.994。

2-station / 3-station 子集同样以 `dz` 为近零模。14974 仅有的两条完整
四站观测在 `dz` 上比混合图更退化（6-DoF condition ~3.4e9）。

self-nulling 的 `dz` 提议（658 mm / 1250 mm）只说明该方向未被
track 约束，不是可写几何。

## 2. 冻结 A 反投影

把已注册的 MC 算子（`A_dx=-59.21`，`A_ry=-31.91`，子空间
`r2=0.99966`）作用到 self-nulling 的 station 修正上。不重测 `A`。
不发出新的 `C_dx` payload（`new_cdx_payload=null`）。

| run | dx mm | ry mrad | implied `C_dx`（由 dx，µm） | implied `C_dx`（由 ry，µm） |
| ---: | ---: | ---: | ---: | ---: |
| 14973 | −1.30 | 20.0 | −22 | +628 |
| 14974 | +6.57 | 33.5 | +111 | +1049 |

两者都超过 1.5–1.7 µm isolation budget。dx 与 ry 的投影不一致
（14973 甚至异号），因此只是拒绝诊断，不是物理 `C_dx`。

## 3. Selected-route 统计

保留当前 V2 association。

| run | selected | 完整四站 | 比例 | 边复用 | 单事件占比 |
| ---: | ---: | ---: | ---: | --- | ---: |
| 14973 | 121 | 0 | 0 | 无 | 0.008 |
| 14974 | 109 | 2 | 0.018 | 无 | 0.009 |

完整四站极少。station solve 被局部的 2-station / 3-station field
topology 主导。这是并发的统计限制，不是 V2 association 失败：
selected graph 非空，all-pairs graph 也已经非空。

## 4. 仅分析的统计量控制

不写几何。不使用 Schur 生产估计器。

- 同一 topology 整体放大：condition number 不变
- 把观测到的完整四站法块按倍数加入：
  - 6-DoF：**不能**恢复。观测到的完整块在 `dz` 上比混合图更退化
  - 5-DoF：14973 已低于 `1e4`；14974 需要约 758 倍该块（约 1515
    条完整路线，按现有出现率约 4.6e7 事件）。不实用，且剩余弱方向
    仍与 `A` 对齐
- `station_covariance_can_recover_by_more_complete_tracks=false`
- `practical_path_to_geometry_write=false`

再多采同样 V2 / occupancy 的真实数据，也不能同时恢复 station
6-DoF 协方差与 cross-level isolation。

## 唯一判定

**B. `physical_nonidentifiability`**

- 并发：A. `reconstruction_statistics_limitation`（完整四站占用也不足）
- 排除：C. `v2_association_failure`

`geometry_write_allowed=false`。C_dx Mode 保持关闭。sealed test
保持关闭。不新增 DoF。不重训 V2。若要进入 production conditions，
必须先重新定义 calibration mode，而不是采用这次 Newton 步长。

## 产物

目录 `outputs/operating_protocol_v1_real_data_station_mode_failure_audit_v1/`：

- `station_identifiability_real_data_audit.json`
- `cdx_cross_level_contamination_audit.json`
- `route_statistics_limitation_audit.json`
- `operating_protocol_failure_classification.json`
