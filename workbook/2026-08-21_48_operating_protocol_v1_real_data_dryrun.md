# 2026-08-21 (48) Operating Protocol V1 真实 FASER data dry-run

## 任务

条目 47 的 large-statistics MC transfer 已通过。本条目正式进入真实 FASER
data 的 Operating Protocol V1 dry-run，但仍是 **transfer / DQ 验证**，不是
写入 production conditions。

冻结：mode-0；V2 checkpoint/calibration/route policy；
`physical_edge_deduplicated`；Station Mode `dx/dy/rx/ry/rz` + `dz=5 mm`
survey prior；IFT-Internal Mode 1-D `C_dx`；全部 capture；mode-validity
contract；冻结泄漏算子 `A` 与 10% stability gate。禁止重训、重注册、联合
Newton、同阶段互迭代、新 layer/module DoF。residual 下降只是 DQ。

## Provenance

条目 03 继续拒绝 2022 data0 IFT 重导出。本步准入 2024 `r0022` xAOD：

- geometry `FASERNU-04`，conditions `OFLCOND-FASER-06`
- 每个 run 的 `00000` segment 作为时间块，100 events
- 盲分割（看 residual 之前冻结）：calibration `14973/14974`，holdout
  `14975/14976`，held-out DQ `14977`

Station Mode 声明 `cdx_fixed_by=external_geometry`。真实 `C_dx` 未知，必须
同时报告由 unbiased residual、run-to-run 稳定性与冻结 `A` 推出的
cross-level contamination diagnostic。implied `|C_dx|` 超 1.5–1.7 µm 或
前提无法证明时 `geometry_write_allowed=false`。

## 本步已落地

- NtupleDumper 真实数据 mode-0 全对 pairing（`ExportTrackletPropagationAllPairs`），
  不发射 truth mode 1/2
- 物理 scan 可在 `is_mc=false` 下跑，禁止 `--include-truth`
- identity sample（origin 戳记、无 MC label）
- self-nulling Station Mode 更新（无已知 residual 目标）
- 配置、provenance 目录、Condor 提交器、三份 JSON 报告与 EN/CN 文档

官方 conditions DB **未修改**。Dedicated `C_dx` Mode 在 Station Mode 合同
通过之前不启动。

## Station Mode 物理波 1

卡住点已修复：holdout / held-out DQ 的 `current_geometry_only` 扫描只有
官方几何一点，`_build_plan` 不再要求 12 个 FD probe。NtupleDumper 已带
`ExportTrackletPropagationAllPairs` 重建（`.so` + Configurable + confdb）。

Prepare 根目录：
`outputs/operating_protocol_v1_real_data_station_mode_physical_v1/`
（calibration 13 点，holdout / held-out DQ 各 1 点）。xAOD 经 resolve 落到
`/eos/experiment/faser/data0/rec/2024/r0022/`（仍满足条目 03 的 2024 r0022
准入；2022 data0 IFT 重导出继续拒绝）。

Condor（eossubmit / `tomorrow` / 6000 MB，每源一作业）：

| bank | cluster | 作业 | 结果 |
| --- | ---: | ---: | --- |
| Station Mode 真实数据 wave-1 | **1000432** | 5 | payload：`Calypso_SET_UP` 泄漏 |
| 同上 `--resume` retry | **1000433** | 5 | payload 已写出；refit 因 sqlite 只有 `OFLP200`、data job 读 `CONDBR3` 失败 |
| Occupancy preflight 00000 全文件 | **1000439** | 5 | 完成（return 0）；无 100-event 窗口过冻结阈值 |
| Occupancy preflight 00001 全文件 | **1000443** | 5 | 完成（return 0）；仍无窗口过冻结阈值 |
| Occupancy preflight 00002 全文件 | **1000444** | 5 | 完成（return 0）；仍无窗口过冻结阈值 |
| Occupancy preflight 00003 全文件 | **1000446** | 5 | 完成（return 0）；仍无窗口过冻结阈值 |
| Occupancy preflight 00004 全文件 | **1000593** | 5 | 完成；**14976 holdout** 冻结 `skip_events=95500` |
| Occupancy preflight 00005（其余 4 run） | **1000594** | 4 | 完成；14974/14975/14977 冻结窗口 |
| Occupancy preflight 00006（仅 14973） | **1000595** | 1 | 完成；14973 仍未过线 |
| Occupancy preflight 00007（仅 14973） | **1000603** | 1 | 完成；**14973 calibration** 冻结 `skip_events=49500` |
| Station Mode wave-1 v1（冻结窗口） | **1000604** | 5 | holdout 因 NtupleDumper 默认 CKF/`StableOnly` 空 ntuple 失败 |
| Station Mode wave-1 v2（`--NoTrackFilt --no_stable`） | **1000607** | 5 | **全部完成** return 0；`n_tracklets` 与 occupancy 一致 |

**1000435 完成结论：** Athena 链跑通（payload + CONDBR3 + ntuple maker），但每个
block 的冻结窗口 `nevents=100`（xAOD 开头）`SegmentFit` 报告 400 stations
low occupancy、0 clusters、NtupleDumper 0 events。物理文件存在却是空样本，
`_physical_point_completion` 仍判 `content_audit_empty`，不能做 identity/V2/Newton。
scan driver 现已禁止把 `n_tracklets=0` 的空 ROOT 打成 `[complete]`。

Occupancy 选窗规则已在看全文件 occupancy **之前**冻结：
`configs/operating_protocol_v1_real_data_occupancy_window_rule.yaml`
（`skip_events=0,100,200,...` 取第一个满足 16/16/32/8 的 100-event 窗口；
不够则按 r0022 segment 编号取下一个文件；角色不变）。2-event smoke：
cluster 容器可读但 event 1–2 仍为 0 cluster / 0 tracklet，无 residual 分支。
全文件扫描 Condor **1000439** flavour `testmatch` 已完成。五个 `00000`
segment（各约 13.8 万 event）有 cluster（每 run 约 5.2 万 event 有
cluster），但 current-geometry SegmentFit tracklet 极稀（整文件约 150–178
个，100-event 窗口最多 2–4 个，四站 event 几乎为 0），全部不满足冻结阈值
16/16/32/8。未降阈值、未改 V2。按规则改扫 `00001`（cluster **1000443**）。
五个 run 都冻结窗口前不重生 wave-1，也不跑 identity/V2/Newton。官方
conditions 仍未写。

根因 1：提交机 `getenv=True` 把本会话的 `Calypso_SET_UP=1` 带进 worker。
`_calypso_command` 现已 unset Calypso/Athena setup 标记。

根因 2：`WriteAlignment` 写入 `OFLP200`，真实数据 `faser_ntuple_maker` 用
`CONDBR3`。payload writer 现用 `AtlCoolCopy -create` 复制实例；已有 29 个
sqlite 已补 CONDBR3。官方 conditions 仍未写。

五个 run 的 occupancy 窗口均已冻结（阈值未降、V2 未改、角色未改）：

| run | 角色 | segment | skip_events | 100-event occupancy（cluster events / tracklet events / tracklets / 四站） |
| ---: | --- | --- | ---: | --- |
| 14973 | calibration | 00007 | 49500 | 52 / 19 / 53 / 10 |
| 14974 | calibration | 00005 | 74400 | 53 / 26 / 66 / 10 |
| 14975 | holdout | 00005 | 21600 | 57 / 19 / 57 / 9 |
| 14976 | holdout | 00004 | 95500 | 48 / 18 / 51 / 8 |
| 14977 | held_out_dq | 00005 | 135900 | 57 / 17 / 47 / 8 |

Provenance：`outputs/operating_protocol_v1_real_data_occupancy_preflight_v1/frozen_windows.json`
（`all_windows_frozen=true`）。**1000604** 在冻结窗口上仍写出 0-event ntuple：
NtupleDumper 默认 `DoTrackFilter` 要 CKF long track，且 `StableOnly` 要
`FaserLHCData.stableBeams()`；这些 r0022 xAOD 的 `stableBeams()` 全为 false，
`--useIFT` 还去读不存在的 `CKFTrackCollection`。occupancy 阈值未降、V2 未改。
真实数据 alignment 导出现为 `--NoTrackFilt --no_stable`，并把 CKF handle 指到
本 job 写出的 `SegmentFitRefit`。本地 14975 smoke：100 ntuple events / 19 有
tracklet / 57 tracklets，与 occupancy 一致。wave-1 v2 目录：
`outputs/operating_protocol_v1_real_data_station_mode_physical_frozen_windows_v2/`。
Condor **1000607** 五个作业全部 return 0。physical `n_tracklets` 与 occupancy 一致
（14973:53，14974:66，14975:57，14976:51，14977:47）。identity（无 MC overlay）
已写入
`outputs/operating_protocol_v1_real_data_identity_frozen_windows_v1/`。
三份 JSON 报告在
`outputs/operating_protocol_v1_real_data_dryrun_v1/`：`physical_wave1_complete=true`，
`geometry_write_allowed=false`，C_dx Mode `not_started`。下一步：冻结 V2
`--allow-real-data` all-pairs → candidate-graph DQ → 仅 calibration 的
self-nulling Station Mode。官方 conditions 仍未写。
