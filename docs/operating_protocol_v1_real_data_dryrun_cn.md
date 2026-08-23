# Operating Protocol V1 真实数据 dry-run（transfer / DQ）

条目 48 / 2026-08-21。本步仍是 transfer 与数据质量验证，**不写入**官方
conditions 数据库。

## 冻结不变量

与 Operating Protocol V1 / large-statistics MC transfer 相同：

- 传播：mode-0
- association：冻结 V2 checkpoint / calibration / route policy
- 观测：`anchor_selected_field_edge` + `physical_edge_deduplicated`
- Station Mode：5-DoF + `dz=5 mm` survey prior，写入 `dz=0`
- IFT-Internal Mode：只浮动 1-D `C_dx`，payload `L0=+C_dx`，`L1=0`，`L2=-C_dx`
- 全部 capture、mode-validity contract、冻结泄漏算子 `A` 与 10% 稳定性门
- sealed test：永不打开
- 真实链：`/Tracker/Align → SCT_ClusterContainer → SegmentFitRefit →
  SegmentsRefit → NtupleDumper → Acts(mode 0)`

禁止重训、重注册 capture/`A`/阈值、联合 station+`C_dx` Newton、同阶段互迭代、
新的 layer/module DoF、把 residual 下降当成正确 alignment。

## 条目 03 provenance 门

2022 data0 IFT PHYS/xAOD 重导出仍被拒绝：event ID 变成 `1...100`，与已有
PHYS 序列不符，且 local segment 为空。

本 dry-run 只准入 **2024 r0022 重建 xAOD**（持久化 `SCT_ClusterContainer`、
`SegmentFit`、`Segments`）。Geometry tag `FASERNU-04`，conditions tag
`OFLCOND-FASER-06`，重建 tag `r0022`。旧 PHYS 只作 provenance 辅助，不是
alignment 输入。

选定时间块（每个 run 的首个 `00000` xAOD segment，100 events）：

| run | 角色 | 是否产生候选几何 |
| ---: | --- | --- |
| 14973 | calibration | 是 |
| 14974 | calibration | 是 |
| 14975 | holdout | 否；只用冻结候选评价 |
| 14976 | holdout | 否 |
| 14977 | held_out_dq | 永不参与几何估计 |

该分割在
`configs/operating_protocol_v1_real_data_dryrun.yaml` 中、在看 residual
**之前**冻结。禁止 MC truth，也禁止把 MC calibration label 混进真实数据。

每个 run 的 `00000` 开头 100 events 在官方 current geometry 下 SCT cluster
为空。不要在这个空窗上重跑 13 点 alignment scan。residual-blind occupancy
preflight 已按文件名顺序扫 r0022 segment（官方几何、无 overlay、无 V2、
无 residual 分支），并按
`configs/operating_protocol_v1_real_data_occupancy_window_rule.yaml`
冻结窗口。五个 run 均已过线（未降阈值、未改 V2）：

| run | 角色 | segment | skip_events |
| ---: | --- | --- | ---: |
| 14973 | calibration | 00007 | 49500 |
| 14974 | calibration | 00005 | 74400 |
| 14975 | holdout | 00005 | 21600 |
| 14976 | holdout | 00004 | 95500 |
| 14977 | held_out_dq | 00005 | 135900 |

Provenance：
`outputs/operating_protocol_v1_real_data_occupancy_preflight_v1/frozen_windows.json`。
`n_tracklets=0` 的空 ROOT 不能当成功。

## Station Mode dry-run（第一阶段）

metadata/contract 声明 `C_dx` 已由**当前官方几何 / 外部 alignment** 固定
（`cdx_fixed_by=external_geometry`）。真实数据不能直接知道真实 `C_dx`，因此
每个 block 还必须报告 cross-level contamination diagnostic，来源为：

- unbiased residual pattern
- run-to-run 参数稳定性
- 冻结 `A_dx` / `A_ry`

若 implied station `dx` 图案足以对应 `|C_dx| > 1.5–1.7 µm`，或合同前提无法
被证明，则 `geometry_write_allowed=false`，只保留诊断。

估计器是官方附加 payload 原点上的 **self-nulling** Gauss-Newton（当前几何
+ 轴向有限差分）。没有已知 MC residual 目标。residual 下降只记为 DQ。

## Dedicated `C_dx` Mode（第二阶段，独立）

只对已经通过 Station Mode contract、station 参数稳定、并达到冻结
framework capture/DQ 的数据块另开阶段。station 六矢固定为该阶段已接受的
几何，只浮动 `C_dx`。统计误差与冻结 station→`C_dx` 系统项（`B` RSS
0.715 µm）分开报告。禁止把该 `C_dx` 回灌同一阶段再求 station。

## 盲 transfer 检查

- 只有 calibration block 可以产生候选几何
- holdout 只用冻结几何评价 unbiased residual、route 稳定性、参数漂移
- run 14977 完全不参与几何估计，作为 held-out DQ

## Station Mode 物理波 1（2026-08-21）

holdout / held-out DQ 扫描为 `current_geometry_only`（仅官方几何一点）。
calibration 保留 current + 12 个轴向有限差分。NtupleDumper 已带
`ExportTrackletPropagationAllPairs` 重建。

Condor **1000432** 因 `Calypso_SET_UP` 泄漏失败。**1000433** 写出 payload 后，
因 sqlite 只有 `OFLP200`、data job 读 `CONDBR3` 失败。重提 **1000435** 会复制
两个 COOL instance，但用的是 `00000` 开头空窗。occupancy 窗口冻结后重生 wave-1：
`outputs/operating_protocol_v1_real_data_station_mode_physical_frozen_windows_v2/`，
Condor **1000607**。**1000604** 因 NtupleDumper 默认 CKF long-track 过滤器与
`StableOnly` 写出空 ntuple。真实数据 alignment 导出现用 `--NoTrackFilt
--no_stable`，CKF handle 指向本 job 的 `SegmentFitRefit`。官方 conditions 不写入。

## 产物

`outputs/operating_protocol_v1_real_data_dryrun_v1/`：

- `real_data_station_mode_report.json`
- `real_data_cdx_mode_report.json`
- `operating_protocol_decision.json`

每个 block 恰好一个标签：`valid_for_station_mode`、`valid_for_cdx_mode`、
`cross_level_contaminated`、`dq_failed`、`geometry_write_candidate`、
`insufficient_real_data_occupancy`。

只有两个 mode 在各自前提下都稳定、held-out DQ 不恶化、参数无异常漂移、
合同无拒绝项，才允许进入下一阶段的 conditions-writing rehearsal /
physics-production validation。本条不修改官方 conditions DB。
