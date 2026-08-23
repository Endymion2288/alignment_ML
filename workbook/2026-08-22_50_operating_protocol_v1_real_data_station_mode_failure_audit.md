# 2026-08-22 (50) 真实数据 Station Mode 失败刻画（冻结后）

## 任务

条目 49 的 full-segment self-nulling 已确认：失败不是 V2 route
acceptance，而是当前统计条件下 selected graph 无法同时满足 station
可辨识性与 cross-level isolation。本条目在 **不改 V2 / route policy /
threshold / unmatched penalty、不启动 C_dx Mode、不写 geometry** 的前提下，
对已观测的 `cross_level_contaminated` 做冻结后的 failure
characterization。

禁止：官方 conditions、sealed test、新增 layer/module DoF、Schur 生产估计器、
把 residual 下降当成 closure、由 implied `|C_dx|` 反推新的 C_dx payload。

本步是本地分析（已有 12 个 station FD probe + 冻结 V2 selected
routes），无需 Condor。

## 本步已落地

冻结配置：
`configs/operating_protocol_v1_real_data_station_mode_failure_audit_v1.yaml`

Provenance：
`outputs/operating_protocol_v1_real_data_station_mode_failure_audit_v1/`

四份 JSON：

- `station_identifiability_real_data_audit.json`：由 12 个 FD probe 重建
  真实数据 `J_s`，报告 rank、condition、奇异谱、近退化方向
- `cdx_cross_level_contamination_audit.json`：冻结 MC `A` 的反投影，只作
  拒绝诊断
- `route_statistics_limitation_audit.json`：14973/14974 selected-route
  统计来源
- `operating_protocol_failure_classification.json`：唯一标签
  **`physical_nonidentifiability`**（B）

单元测试 `tests/test_real_data_station_mode_failure_audit.py` 13 项通过。

## 1. Identifiability（不是 closure）

`J_s` 来自条目 49 已捕获的 12 个轴向 station FD probe，作用在冻结
selected-route 边上。未重跑 Athena，未重训 V2。residual 改善不记入本报告。

| run | 观测边 | 6-DoF rank | 6-DoF cond | 近退化 | 5-DoF cond（去掉 survey `dz`） |
| ---: | ---: | ---: | ---: | --- | ---: |
| 14973 | 118 | 6 | 4.90e5 | `dz`（\|v\|=1.000） | 114 |
| 14974 | 104 | 6 | 2.75e6 | `dz`（\|v\|=1.000） | 1.64e4 |

近退化方向几乎是纯 `dz`。`dx/dy/ry` 不是 6-DoF 最小奇异方向。
去掉 survey `dz` 后，14973 的 5-DoF 已低于 MC 门 1e4；14974 的 5-DoF
弱方向几乎是纯 `dx`（\|v\|=0.993），与冻结 `A` 的夹角余弦 0.994。

2-station / 3-station 子集的 6-DoF 同样以 `dz` 为近零模。14974 仅有的
2 条完整四站观测本身在 `dz` 上更退化（6-DoF cond ~ 3.4e9）。

Self-nulling 的 `dz` 提议（658 / 1250 mm）只说明该方向未被 tracks
约束，不是 alignment 正确性，也不是可写几何。

## 2. 冻结 A 反投影（拒绝诊断，不是 C_dx 测量）

沿用 MC 注册的 `A`（`A_dx=-59.21`，`A_ry=-31.91`，子空间
`r2=0.99966`）。未重测 A，未生成新的 C_dx payload。

| run | 提议 dx mm | 提议 ry mrad | implied `C_dx`(dx) µm | implied `C_dx`(ry) µm |
| ---: | ---: | ---: | ---: | ---: |
| 14973 | −1.30 | 20.0 | −22 | +628 |
| 14974 | +6.57 | 33.5 | +111 | +1049 |

两者都超过 1.5–1.7 µm isolation budget。dx 与 ry 推出的 implied
`C_dx` 互不一致（14973 甚至异号），因此不能当成一个真实 C_dx 值，
只能作为拒绝诊断。`new_cdx_payload=null`。

## 3. Selected-route 统计来源（保留当前 V2）

| run | selected | 完整四站 | 分数 | 边复用 | 单事件占比 |
| ---: | ---: | ---: | ---: | --- | ---: |
| 14973 | 121 | 0 | 0 | 无复用 | 0.008 |
| 14974 | 109 | 2 | 0.018 | 无复用 | 0.009 |

完整四站极少，2-station / 3-station 主导。station solve 被局部 field
topology 主导。这是并发的统计限制，**不是** V2 association 失败：
selected graph 非空，all-pairs 非空，V2 / threshold / penalty 未改。

## 4. 仅分析的统计量控制（不用 Schur，不写几何）

- 同一 topology 整体放大：condition number 不变，6-DoF 不能恢复
- 把观测到的完整四站法块按倍数加入：
  - 6-DoF：**不能**恢复。观测到的完整四站块在 `dz` 上比混合图更退化
  - 5-DoF：14973 已低于 1e4；14974 需要约 758 倍该完整块
    （约 1515 条完整四站，按现有出现率约 4.6e7 事件），且剩余弱方向
    仍与 `A` 对齐，不实用
- `station_covariance_can_recover_by_more_complete_tracks=false`
- `practical_path_to_geometry_write=false`

因此“再多采一些同样 V2 / occupancy 的真实数据”不能同时恢复
station 6-DoF 与 cross-level isolation。

## 唯一判定

**B. `physical_nonidentifiability`**

- 并发：A. 完整四站统计量也不足（当前图被短 route 主导）
- 排除：C. V2 association failure

`geometry_write_allowed=false`。不启动 C_dx Mode。不打开 sealed
test。不新增 DoF。不重训 V2。下一步若要进入 production conditions，
必须先重新定义 calibration mode，而不是写这次 Newton 步长。
